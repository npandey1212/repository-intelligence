"""Request-budget preflight at the model-call boundary (H1 + H2).

Covers the two halves of the fix against the real failure class (a
request that reaches the provider over its context window):

H1 — every outgoing request is measured COMPLETELY (messages, tool
     arguments, tool definitions, and the structured-output schema) at
     the one seam every model call passes through, producing a
     structured, privacy-safe diagnostic record on the trace
     iteration — and a provider ``context_length_exceeded`` failure is
     correlated back to the preflight record.

H2 — the budget is enforced BEFORE any provider call: over-budget tool
     observations are deterministically clipped to valid-JSON excerpts;
     protected content that cannot fit is rejected outright — never
     sent, never retried unchanged — with a typed terminal failure.

Everything is scripted and LLM-free (queued responses, recording
dispatcher, one question): no network, no API key.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import (
    SYSTEM_PROMPT,
    Agent,
    AgentError,
    AgentExecutionConfig,
    estimate_context_characters,
)
from repo_intelligence.agent.request_budget import (
    CHARS_PER_TOKEN,
    DISPOSITION_ACCEPTED,
    DISPOSITION_REDUCED,
    DISPOSITION_REJECTED,
    DEFAULT_CONTEXT_CAPACITY_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    ESTIMATION_METHOD,
    MIN_TOOL_EXCERPT_CHARS,
    RequestBudgetPolicy,
    build_request_budget_record,
    enforce_request_budget,
    estimate_tokens,
    note_provider_context_overflow,
)
from repo_intelligence.agent.tool_definitions import AGENT_TOOL_DEFINITIONS
from repo_intelligence.llm import (
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    ToolCall,
    ToolDefinition,
)
from repo_intelligence.tools import ToolResult

QUESTION = "How do the agents, tools, and MCP servers interact?"
SECRET = "SECRET-REPOSITORY-BODY-CONTENT-DO-NOT-COPIE"
SENTINEL = "SENTINEL-VALUE-8f3a"

# Four tool definitions as shipped with the Agent (~2.7k characters) —
# the fixed part of every request that the message-only estimator
# historically ignored.
_SCHEMA_CHARACTERS = sum(
    len(tool.name)
    + len(tool.description or "")
    + len(json.dumps(tool.input_schema or {}))
    for tool in AGENT_TOOL_DEFINITIONS
)


def _big_tools():
    """Tool definitions whose SCHEMA alone dwarfs any message."""
    return [
        ToolDefinition(
            name="giant",
            description=SENTINEL + " description",
            input_schema={
                "type": "object",
                "properties": {
                    "blob": {"type": "string", "description": "x" * 4000}
                },
            },
        ),
    ]


def _spec():
    return StructuredOutputSpec(
        name="BigContract",
        schema={
            "type": "object",
            "properties": {
                "field_{}".format(i): {"type": "string", "description": "y" * 300}
                for i in range(6)
            },
        },
    )


def _request(
    *,
    tool_content=None,
    tools=None,
    structured_output=None,
    system=SENTINEL + " system contract",
):
    if tool_content is None:
        # Shaped like a real tool observation (the agent serializes
        # ToolResult.to_dict() into the message), body carries SECRET.
        tool_content = json.dumps(
            {
                "tool": "read_file",
                "success": True,
                "data": {"path": "a.py", "content": SECRET + "a" * 5000},
            }
        )
    return LLMRequest(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=QUESTION),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})
                ],
            ),
            Message(role="tool", content=tool_content, tool_call_id="c1"),
        ],
        tools=list(AGENT_TOOL_DEFINITIONS) if tools is None else tools,
        structured_output=structured_output,
    )


def _records(payload):
    return json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Estimator + policy (the documented numbers)
# ---------------------------------------------------------------------------


def test_estimation_is_an_explicit_four_char_ratio():
    assert CHARS_PER_TOKEN == 4
    assert ESTIMATION_METHOD == "chars_per_token_estimate"
    assert estimate_tokens(0) == 0
    assert estimate_tokens(1) == 1
    assert estimate_tokens(4) == 1
    assert estimate_tokens(5) == 2  # ceil, never truncates to zero


def test_default_budget_derives_from_the_observed_window_with_reserves():
    # The window the real overflow hit — minus output and margin, so
    # the default input target never approaches the window itself.
    assert DEFAULT_CONTEXT_CAPACITY_TOKENS == 128_000
    policy = RequestBudgetPolicy(
        context_capacity_tokens=DEFAULT_CONTEXT_CAPACITY_TOKENS
    )
    assert policy.effective_input_budget_tokens == (
        DEFAULT_CONTEXT_CAPACITY_TOKENS
        - DEFAULT_RESERVED_OUTPUT_TOKENS
        - DEFAULT_SAFETY_MARGIN_TOKENS
    )
    assert policy.effective_input_budget_characters == (
        policy.effective_input_budget_tokens * CHARS_PER_TOKEN
    )


def test_explicit_input_budget_wins_over_capacity_and_characters():
    policy = RequestBudgetPolicy(
        max_input_tokens=1000,
        context_capacity_tokens=128_000,
        max_input_characters=9_999_999,
    )
    assert policy.effective_input_budget_tokens == 1000
    # The character ceiling (when present) can only TIGHTEN the
    # effective character budget, never widen it.
    assert policy.effective_input_budget_characters == 4000

    characters_only = RequestBudgetPolicy(
        max_input_characters=4444, context_capacity_tokens=None
    )
    assert characters_only.effective_input_budget_tokens == 1111
    assert characters_only.effective_input_budget_characters == 4444


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_input_tokens": True},
        {"max_input_tokens": 0},
        {"max_input_tokens": -1},
        {"max_input_tokens": "1000"},
        {"max_input_characters": 0},
        {"context_capacity_tokens": -5},
        {"reserved_output_tokens": -1},
        {"safety_margin_tokens": -1},
        {"chars_per_token": 0},
    ],
)
def test_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RequestBudgetPolicy(**kwargs)


# ---------------------------------------------------------------------------
# H1 — the diagnostic record
# ---------------------------------------------------------------------------


def test_record_measures_the_whole_request_not_only_messages():
    request = _request(tools=_big_tools(), structured_output=_spec())
    record = build_request_budget_record(request, RequestBudgetPolicy())

    totals = record["totals"]
    schema_chars = sum(
        len(tool.name)
        + len(tool.description or "")
        + len(json.dumps(tool.input_schema or {}))
        for tool in _big_tools()
    )
    spec_chars = len("BigContract") + len(
        json.dumps(request.structured_output.schema)
    )
    # The complete request is measured: tool definitions and the
    # structured-output schema are first-class, not afterthoughts —
    # exactly the part a message-only estimate misses.
    assert record["tool_definitions"]["characters"] == schema_chars
    assert record["structured_output"]["characters"] == spec_chars
    assert record["structured_output"]["name"] == "BigContract"
    assert totals["total_characters"] == (
        totals["messages_characters"]
        + record["tool_definitions"]["characters"]
        + record["structured_output"]["characters"]
    )
    assert totals["total_tokens_est"] == estimate_tokens(
        totals["total_characters"]
    )


def test_record_captures_roles_largest_messages_and_budget():
    record = build_request_budget_record(_request(), RequestBudgetPolicy())
    assert record["role_counts"] == {
        "assistant": 1,
        "system": 1,
        "tool": 1,
        "user": 1,
    }
    assert record["message_count"] == 4
    assert record["tool_call_count"] == 1
    assert record["largest_messages"][0]["role"] in ("system", "tool")
    assert record["largest_messages"][0]["characters"] > 0
    assert record["estimation"]["is_estimate"] is True
    assert record["estimation"]["chars_per_token"] == CHARS_PER_TOKEN
    assert "effective_input_budget_tokens" in record["budget"]
    assert record["within_budget"] is True
    assert record["disposition"] == DISPOSITION_ACCEPTED


def test_record_carries_no_message_content():
    request = _request(system=SENTINEL + " " + "s" * 500)
    record = build_request_budget_record(request, RequestBudgetPolicy())
    serialized = _records(record)
    # No content at any depth: never the observation body, never the
    # system contract, never the question — sizes and labels only.
    assert SECRET not in serialized
    assert SENTINEL not in serialized
    assert QUESTION not in serialized
    # The tool NAME is the only label kept (a safe short identifier).
    labels = [
        entry.get("tool")
        for entry in record["messages"]
        if entry["role"] == "tool"
    ]
    assert labels == ["read_file"]


def test_accepted_request_is_returned_untouched():
    request = _request()
    final, record = enforce_request_budget(request, RequestBudgetPolicy())
    assert final is request
    assert record["disposition"] == DISPOSITION_ACCEPTED
    assert "reduction" not in record


# ---------------------------------------------------------------------------
# H2 — reduction and rejection
# ---------------------------------------------------------------------------


def test_over_budget_tool_observations_clip_to_valid_envelopes():
    request = _request()
    original_content = request.messages[3].content
    original_chars = len(original_content)
    policy = RequestBudgetPolicy(max_input_tokens=1800)  # 7200 chars
    final, record = enforce_request_budget(request, policy)

    assert record["disposition"] == DISPOSITION_REDUCED
    assert record["within_budget"] is True
    assert record["totals"]["total_characters"] <= (
        policy.effective_input_budget_characters
    )
    assert record["reduction"]["tool_messages_clipped"] >= 1
    assert record["reduction"]["before_characters"] > record["reduction"][
        "after_characters"
    ]

    clipped = final.messages[3]
    envelope = json.loads(clipped.content)  # valid JSON — never a raw cut
    assert envelope["truncated"] is True
    assert envelope["tool"] == "read_file"
    assert envelope["original_characters"] == original_chars
    assert len(envelope["excerpt"]) >= MIN_TOOL_EXCERPT_CHARS
    assert original_content.startswith(envelope["excerpt"])  # head slice
    assert "clipped" in envelope["note"]
    # Pairing survives: the provider's tool message order stays valid.
    assert clipped.tool_call_id == "c1"
    # Protected messages are byte-identical — only observations shrink.
    assert final.messages[0].content == request.messages[0].content
    assert final.messages[1].content == request.messages[1].content
    assert final.messages[2].content == request.messages[2].content
    # The record itself still carries no observation content.
    assert SECRET not in _records(record)


def test_reduction_is_deterministic():
    policy = RequestBudgetPolicy(max_input_tokens=2000)
    first = enforce_request_budget(_request(), policy)
    second = enforce_request_budget(_request(), policy)
    assert json.dumps(first[0].messages, default=str) == json.dumps(
        second[0].messages, default=str
    )
    assert first[1] == second[1]


def test_escaped_content_clips_without_breaking_json():
    quoted = json.dumps(
        {"tool": "read_file", "content": '"' * 4000}
    )
    request = _request(tool_content=quoted)
    policy = RequestBudgetPolicy(max_input_tokens=1500)  # 6000 chars
    final, record = enforce_request_budget(request, policy)

    if record["disposition"] == DISPOSITION_REDUCED:
        envelope = json.loads(final.messages[3].content)
        assert envelope["truncated"] is True
        assert record["totals"]["total_characters"] <= (
            policy.effective_input_budget_characters
        )
    else:
        # Rejection is also legitimate — but then it must be the
        # ORIGINAL request, honestly reported as over budget.
        assert record["disposition"] == DISPOSITION_REJECTED
        assert final is request


def test_protected_content_over_budget_rejects_and_returns_original():
    # No tool definitions in this request: the REJECTION reason can
    # only come from protected conversation content alone.
    request = LLMRequest(
        messages=[
            Message(role="system", content="s" * 3000),
            Message(role="user", content=QUESTION),
        ],
        tools=[],
    )
    policy = RequestBudgetPolicy(max_input_tokens=100)  # 400 chars
    final, record = enforce_request_budget(request, policy)

    assert record["disposition"] == DISPOSITION_REJECTED
    assert record["within_budget"] is False
    assert record["rejection_reason"] == "protected_request_exceeds_budget"
    assert final is request  # NOTHING may be sent
    # Reduction was attempted honestly: nothing to clip here.
    assert record["reduction"]["tool_messages_clipped"] == 0


def test_tool_schema_contract_over_budget_is_named():
    request = _request(tools=_big_tools())
    policy = RequestBudgetPolicy(max_input_tokens=100)  # 400 chars
    final, record = enforce_request_budget(request, policy)
    assert record["disposition"] == DISPOSITION_REJECTED
    assert (
        record["rejection_reason"]
        == "tool_or_response_contract_exceeds_budget"
    )
    assert final is request


def test_tiny_observations_are_never_wrapped_into_bigger_envelopes():
    tiny = json.dumps({"tool": "read_file", "content": "tiny"})
    request = _request(tool_content=tiny, tools=_big_tools())
    policy = RequestBudgetPolicy(max_input_tokens=100)
    final, record = enforce_request_budget(request, policy)
    assert record["disposition"] == DISPOSITION_REJECTED
    # The small observation was left alone — clipping it would only
    # have ADDED envelope words to an already-rejected request.
    assert final.messages[3].content == tiny
    assert record["reduction"]["tool_messages_clipped"] == 0


def test_provider_overflow_annotates_the_preflight_record():
    record = build_request_budget_record(_request(), RequestBudgetPolicy())
    annotated = note_provider_context_overflow(
        record, LLMError("context_length_exceeded: max context 128k")
    )
    assert annotated["provider_error"]["category"] == "context_length_exceeded"
    assert "max context" in annotated["provider_error"]["message"]
    assert (
        annotated["provider_error"]["preflight_total_tokens_est"]
        == record["totals"]["total_tokens_est"]
    )
    assert annotated["provider_error"]["preflight_within_budget"] is True
    # The original record is not mutated, and None stays None.
    assert "provider_error" not in record
    assert note_provider_context_overflow(None, LLMError("x")) is None


# ---------------------------------------------------------------------------
# H1/H2 in the Agent — the gate on every call site
# ---------------------------------------------------------------------------


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _call(name, arguments, call_id):
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def _text(text):
    return LLMResponse(
        message=Message(role="assistant", content=text),
        finish_reason="stop",
    )


def _observation(artifact, marker):
    return ToolResult(
        success=True,
        tool="read_file",
        data={"path": artifact, "content": marker * 5000},
        source={"artifact": artifact},
    )


def _operations(trace):
    return [event["operation"] for event in trace.lifecycle_events]


def _scripted_run(execution_config, observation=None):
    """One read + answer (or answer-only) with a recording client."""
    observation = observation or _observation("src/a.py", "A")
    responses = (
        [
            _tool_calls(_call("read_file", {"path": "src/a.py"}, "c1")),
            _text("the run answered from the evidence it had"),
        ]
    )
    fake = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(observation)
    agent = Agent(
        fake,
        tool_dispatcher=dispatcher,
        execution_config=execution_config,
    )
    return agent, fake, dispatcher


def _request_budget_tokens_for(iteration_total_characters, extra=1500):
    """A budget that admits the FIRST request but not the second."""
    return (iteration_total_characters + extra + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def test_default_run_records_accepted_budget_and_changes_nothing():
    agent, fake, _ = _scripted_run(AgentExecutionConfig())
    answer = agent.run(QUESTION)

    assert answer == "the run answered from the evidence it had"
    trace = agent.last_trace
    assert len(trace.iterations) == 2
    for iteration in trace.iterations:
        budget = iteration.request_budget
        assert budget is not None
        assert budget["disposition"] == DISPOSITION_ACCEPTED
        assert budget["within_budget"] is True
        assert budget["run_id"] == trace.run_id
        assert budget["model"] == "test-model-1"
        assert budget["totals"]["total_tokens_est"] <= budget["budget"][
            "effective_input_budget_tokens"
        ]
    # Accepted requests record NO new lifecycle events: the default
    # path is observably unchanged.
    assert "context.request_reduced" not in _operations(trace)
    assert "context.budget_exceeded" not in _operations(trace)
    # The provider received the original request, byte for byte.
    assert [len(r.messages) for r in fake.requests] == [2, 4]


def test_rejected_request_is_never_sent_and_fails_typed():
    agent, fake, dispatcher = _scripted_run(
        AgentExecutionConfig(
            max_request_input_tokens=100,  # 400 chars < the system prompt
            finalize_on_boundary=False,
        )
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)

    assert excinfo.value.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    # H2 hard guarantee: the provider was never invoked at all.
    assert fake.requests == []
    assert dispatcher.calls == []

    trace = agent.last_trace
    assert trace.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    # Same failure class as a provider overflow — predicted locally.
    assert trace.stop_reason == "CONTEXT_LENGTH_EXCEEDED"
    assert "CONTEXT_BUDGET_EXCEEDED" in trace.summary.termination_reason

    iteration = trace.iterations[0]
    assert iteration.response_type == "ERROR"
    assert iteration.request_budget["disposition"] == DISPOSITION_REJECTED
    assert "request budget" in (iteration.error or "")
    assert _operations(trace) == ["context.budget_exceeded"]
    event = trace.lifecycle_events[0]
    # At 400 characters even the fixed tool-definition contract cannot
    # fit — that is the BINDING constraint (the conversation cannot be
    # shrunk to fit a contract that is itself over budget), so the
    # record names it.
    assert (
        event["reason"] == "tool_or_response_contract_exceeds_budget"
    )
    assert event["effective_input_budget_tokens"] == 100


def test_rejection_on_the_boundary_finalizes_from_collected_state():
    # Budget math (all measured, not assumed): the first request (the
    # preamble + tool definitions) fits; the second (a full tool
    # observation inside) cannot fit even at the excerpt floor; the
    # tool-free digest request fits comfortably.
    head_chars = estimate_context_characters(
        [Message(role="system", content=SYSTEM_PROMPT),
         Message(role="user", content=QUESTION)]
    )
    schema_chars = _SCHEMA_CHARACTERS
    iteration_one = head_chars + schema_chars
    budget_chars = iteration_one + 100
    # The un-clippable remainder of iteration two must exceed it even
    # at the minimum excerpt (envelope > 400 chars once serialized).
    without_observation = (
        head_chars
        + len("read_file")
        + len(json.dumps({"path": "src/a.py"}))
        + schema_chars
    )
    assert without_observation + MIN_TOOL_EXCERPT_CHARS > budget_chars

    agent, fake, dispatcher = _scripted_run(
        AgentExecutionConfig(
            max_request_input_tokens=(budget_chars + CHARS_PER_TOKEN - 1)
            // CHARS_PER_TOKEN,
            finalize_on_boundary=True,
        )
    )
    answer = agent.run(QUESTION)
    assert answer == "the run answered from the evidence it had"

    # Three requests were allowed: iteration 1, the tool-free
    # finalization pass — the rejected iteration-2 candidate was
    # NEVER among them.
    assert len(fake.requests) == 2
    assert fake.requests[0].tools
    assert fake.requests[1].tools == []
    assert len(dispatcher.calls) == 1

    trace = agent.last_trace
    assert trace.termination_reason == "FINAL_ANSWER"
    assert _operations(trace) == [
        "context.budget_exceeded",
        "finalization.started",
        "tools.disabled",
        "finalization.completed",
    ]
    assert trace.stop_reason == "CONTEXT_LENGTH_EXCEEDED"
    rejected = trace.iterations[1]
    assert rejected.request_budget["disposition"] == DISPOSITION_REJECTED
    assert rejected.response_type == "ERROR"
    finalization = trace.iterations[2]
    assert finalization.request_budget["disposition"] == DISPOSITION_ACCEPTED
    # The evidence the observation carried is still on the trace —
    # the rejection reduced NOTHING durable.
    assert len(trace.iterations[0].tool_results) == 1


def test_rejection_even_at_finalization_fails_explicitly():
    agent, fake, _ = _scripted_run(
        AgentExecutionConfig(
            max_request_input_tokens=100,
            finalize_on_boundary=True,
        )
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)

    assert excinfo.value.termination_reason == "FINALIZATION_FAILED"
    assert fake.requests == []  # not the finalization call either
    trace = agent.last_trace
    assert trace.termination_reason == "FINALIZATION_FAILED"
    operations = _operations(trace)
    assert operations.count("context.budget_exceeded") == 2
    assert "finalization.failed" in operations


def test_over_budget_observation_is_clipped_before_the_provider():
    head_chars = estimate_context_characters(
        [Message(role="system", content=SYSTEM_PROMPT),
         Message(role="user", content=QUESTION)]
    )
    observation = _observation("src/a.py", "A")
    observation_chars = len(json.dumps(observation.to_dict()))
    iteration_one = head_chars + _SCHEMA_CHARACTERS
    policy_tokens = _request_budget_tokens_for(iteration_one, extra=2000)
    # The full observation must NOT fit the budget (it gets clipped).
    assert (
        iteration_one
        + observation_chars
        > policy_tokens * CHARS_PER_TOKEN
    )

    agent, fake, dispatcher = _scripted_run(
        AgentExecutionConfig(max_request_input_tokens=policy_tokens),
        observation=observation,
    )
    answer = agent.run(QUESTION)
    assert answer == "the run answered from the evidence it had"

    # The provider's SECOND request carries the envelope, not the body.
    clipped_message = [
        message
        for message in fake.requests[1].messages
        if message.role == "tool"
    ]
    assert len(clipped_message) == 1
    envelope = json.loads(clipped_message[0].content)
    assert envelope["truncated"] is True
    assert envelope["original_characters"] == observation_chars

    trace = agent.last_trace
    dispositions = [
        iteration.request_budget["disposition"]
        for iteration in trace.iterations
    ]
    # The pre-observation request passes untouched; every later request
    # carrying the observation is reduced (the envelope fills its room,
    # so the answer turn needs one more reduction — all deterministic).
    assert dispositions[0] == DISPOSITION_ACCEPTED
    assert DISPOSITION_REDUCED in dispositions[1:]
    assert len(dispositions) == len(fake.requests)
    reductions = [
        event
        for event in trace.lifecycle_events
        if event["operation"] == "context.request_reduced"
    ]
    assert reductions  # at least the observation's own reduction
    assert reductions[0]["tool_messages_clipped"] >= 1
    assert reductions[0]["before_tokens_est"] > reductions[0]["after_tokens_est"]
    # Every request the provider actually received fits the budget.
    for request in fake.requests:
        record = build_request_budget_record(request, RequestBudgetPolicy())
        assert record["totals"]["total_tokens_est"] <= policy_tokens
    # The full observation is still on the durable trace.
    assert len(dispatcher.calls) == 1
    assert "A" * 100 in json.dumps(
        trace.iterations[0].tool_results[0]["data"]
    )


def test_planner_request_is_gated_before_any_provider_call():
    fake = FakeLLMClient()  # must stay empty: nothing may be sent
    agent = Agent(
        fake,
        tool_dispatcher=FakeDispatcher(),
        execution_config=AgentExecutionConfig(max_request_input_tokens=50),
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run_planned(QUESTION)

    assert excinfo.value.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    assert fake.requests == []
    trace = agent.last_trace
    assert trace.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    assert trace.plan is None  # the plan never even started
    planning_iteration = trace.iterations[0]
    assert planning_iteration.request_budget["disposition"] == (
        DISPOSITION_REJECTED
    )
    assert planning_iteration.response_type == "ERROR"


def test_provider_context_failure_is_correlated_with_preflight():
    fake = FakeLLMClient(
        LLMError("context_length_exceeded: maximum context length exceeded")
    )
    agent = Agent(fake, tool_dispatcher=FakeDispatcher())
    with pytest.raises(LLMError):
        agent.run(QUESTION)

    trace = agent.last_trace
    budget = trace.iterations[0].request_budget
    assert budget is not None
    assert budget["provider_error"]["category"] == "context_length_exceeded"
    assert (
        budget["provider_error"]["preflight_total_tokens_est"]
        == budget["totals"]["total_tokens_est"]
    )
    assert budget["disposition"] == DISPOSITION_ACCEPTED  # preflight saw fit
    assert trace.summary.termination_reason == "LLM_ERROR"


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, "1000"],
)
def test_execution_config_validates_the_token_budget(value):
    with pytest.raises(ValueError, match="max_request_input_tokens"):
        AgentExecutionConfig(max_request_input_tokens=value)


def test_default_execution_config_has_no_explicit_token_budget():
    assert AgentExecutionConfig().max_request_input_tokens is None
    # ...and the derived default still comfortably contains a normal
    # request (messages + the fixed tool-definition contract).
    normal = _request()
    record = build_request_budget_record(normal, RequestBudgetPolicy())
    assert record["within_budget"] is True
    assert record["totals"]["total_tokens_est"] < record["budget"][
        "effective_input_budget_tokens"
    ]
    # Guard the blind spot explicitly: total > messages alone.
    assert record["totals"]["total_characters"] > record["totals"][
        "messages_characters"
    ]
    assert record["tool_definitions"]["characters"] >= 1000
    assert _SCHEMA_CHARACTERS >= 1000
