"""H4 — end-to-end request-budget lifecycle regression (run-wide).

``test_request_budget.py`` pins the gate mechanics per call site; this
file pins the RUN-WIDE invariants promised by the H1/H2 work:

1. every model call of a full planned run (planner + every plan step)
   carries a within-budget preflight record on the DEFAULT derived
   budget — the record rides real runs, not just isolated gates;
2. an incident-sized protected request (the 128,882-token shape, here
   a giant question) is rejected LOCALLY on the default budget —
   the provider is never invoked and no typed failure is softened;
3. the token budget and the pre-existing character ceiling are
   independent controls: a tight ceiling never surfaces as a token
   rejection, and an armed token budget reports itself even when the
   ceiling is armed too (attribution, both directions);
4. a provider-reported context overflow correlates with a preflight
   record that never embeds request content (privacy of
   ``provider_error`` on a real run);
5. an armed-but-generous token budget sends byte-identical requests
   to the default configuration — the control is a no-op until it
   binds (non-intrusiveness across a whole run).

Everything is scripted and LLM-free (queued responses, recording
dispatchers): no network, no API key.
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
    DISPOSITION_ACCEPTED,
    DISPOSITION_REJECTED,
    DEFAULT_CONTEXT_CAPACITY_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    RequestBudgetPolicy,
    build_request_budget_record,
)
from repo_intelligence.llm import (
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
)
from repo_intelligence.tools import ToolResult

QUESTION = "How do the agents, tools, and MCP servers interact?"
SENTINEL = "SENTINEL-VALUE-8f3a"
CONTEXT = (
    "# Architecture Context\n\n- Contract: `architecture_context.v1`\n"
)
CONTEXT_RECORD = {
    "artifact": "artifacts/context/architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "d" * 40,
    "snapshot_id": "pinned_snapshot",
}

DEFAULT_EFFECTIVE_BUDGET = (
    DEFAULT_CONTEXT_CAPACITY_TOKENS
    - DEFAULT_RESERVED_OUTPUT_TOKENS
    - DEFAULT_SAFETY_MARGIN_TOKENS
)

HEAD = (
    Message(role="system", content=SYSTEM_PROMPT),
    Message(role="user", content=QUESTION),
)

BUDGET_OPERATIONS = frozenset(
    {"context.request_reduced", "context.budget_exceeded"}
)


# -- per-file test doubles (no conftest in this repo) ----------------------


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
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


def _call(call_id, name="search_repository", arguments=None):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else dict(arguments),
    )


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def _text(content):
    return LLMResponse(
        message=Message(role="assistant", content=content),
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


def _build(responses, results=(), **config_kwargs):
    client = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(*results)
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(**config_kwargs),
    )
    return agent, client, dispatcher


def _two_step_script():
    plan = {
        "objective": QUESTION,
        "steps": [
            {"id": "P1", "goal": "Identify relevant agent definitions"},
            {"id": "P2", "goal": "Produce an evidence-backed explanation"},
        ],
    }
    return (
        LLMResponse(
            message=Message(role="assistant", content=json.dumps(plan)),
            finish_reason="stop",
            structured_output=plan,
        ),
        _tool_calls(_call("c1", "search_repository", {"query": "agent"})),
        _text("identified 3 agent definition files"),
        _tool_calls(_call("c2", "read_file", {"path": "agents/dag.md"})),
        _text("final evidence-backed answer"),
    )


def _read_answer_script():
    return (
        _tool_calls(_call("c1", "read_file", {"path": "src/a.py"})),
        _text("the run answered from the evidence it had"),
    )


def _transcript(client):
    return [
        [
            (
                message.role,
                message.content,
                tuple(
                    call.name
                    for call in (message.tool_calls or [])
                ),
            )
            for message in request.messages
        ]
        for request in client.requests
    ]


# ---------------------------------------------------------------------------
# 1 — every call site of a REAL planned run is measured (H1 on the wire)
# ---------------------------------------------------------------------------


def test_every_model_call_of_a_planned_run_carries_a_record():
    agent, client, _ = _build(_two_step_script(), max_tool_calls=8)
    result = agent.run_planned(QUESTION, CONTEXT, CONTEXT_RECORD)

    assert result.answer == "final evidence-backed answer"
    assert len(client.requests) == 5  # planner + P1 (2) + P2 (2)
    trace = agent.last_trace
    assert len(trace.iterations) == len(client.requests)

    for iteration in trace.iterations:
        record = iteration.request_budget
        assert record is not None
        assert record["disposition"] == DISPOSITION_ACCEPTED
        assert record["within_budget"] is True
        assert (
            record["budget"]["effective_input_budget_tokens"]
            == DEFAULT_EFFECTIVE_BUDGET
        )
        assert record["totals"]["total_tokens_est"] <= (
            DEFAULT_EFFECTIVE_BUDGET
        )
        assert record["run_id"] == trace.run_id
        assert record["model"] == "test-model-1"

    assert not (BUDGET_OPERATIONS & set(_operations(trace)))
    for request in client.requests:
        probe = build_request_budget_record(request, RequestBudgetPolicy())
        assert probe["within_budget"] is True


# ---------------------------------------------------------------------------
# 2 — the incident shape: giant protected request, default budget
# ---------------------------------------------------------------------------


def test_incident_sized_protected_request_never_leaves_the_process():
    agent, client, dispatcher = _build(
        _read_answer_script(), finalize_on_boundary=False
    )
    incident_question = "x" * 400_000  # > 380,928 default char budget

    with pytest.raises(AgentError) as excinfo:
        agent.run(incident_question)

    assert excinfo.value.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    # The hard guarantee: nothing was sent, nothing was dispatched.
    assert client.requests == []
    assert dispatcher.calls == []

    trace = agent.last_trace
    assert trace.stop_reason == "CONTEXT_LENGTH_EXCEEDED"
    record = trace.iterations[0].request_budget
    assert record["disposition"] == DISPOSITION_REJECTED
    assert (
        record["rejection_reason"] == "protected_request_exceeds_budget"
    )
    assert (
        record["budget"]["effective_input_budget_tokens"]
        == DEFAULT_EFFECTIVE_BUDGET
    )
    assert record["totals"]["total_tokens_est"] > DEFAULT_EFFECTIVE_BUDGET
    # The character ceiling was armed wide open: the TOKEN control is
    # the one that acted (and no history restriction ran).
    assert _operations(trace) == ["context.budget_exceeded"]
    assert trace.termination_reason == "CONTEXT_BUDGET_EXCEEDED"


# ---------------------------------------------------------------------------
# 3 — the two context controls never steer each other
# ---------------------------------------------------------------------------


def _ceiling_run(ceiling, **config_kwargs):
    """Read, read, answer — the working-context ceiling script."""
    responses = (
        _tool_calls(_call("c1", "read_file", {"path": "src/a.py"})),
        _tool_calls(_call("c2", "read_file", {"path": "src/b.py"})),
        _text("the run answered from the evidence it had"),
    )
    return _build(
        responses,
        results=(
            _observation("src/a.py", "A"),
            _observation("src/b.py", "B"),
        ),
        max_context_characters=ceiling,
        **config_kwargs,
    )


def _ceiling_operations(trace):
    return {
        op
        for op in _operations(trace)
        if op.startswith("context.compaction")
        or op.startswith("context.restricted")
    }


def test_a_tight_character_ceiling_is_not_a_token_rejection():
    ceiling = estimate_context_characters(list(HEAD)) + 6000
    agent, client, _ = _ceiling_run(ceiling)
    answer = agent.run(QUESTION)

    assert answer == "the run answered from the evidence it had"
    trace = agent.last_trace
    operations = set(_operations(trace))
    # The ceiling acted through ITS OWN machinery...
    assert _ceiling_operations(trace)
    # ...and the request budget never bound: no budget events, every
    # record accepted under the default window budget.
    assert not (BUDGET_OPERATIONS & operations)
    for iteration in trace.iterations:
        record = iteration.request_budget
        assert record["disposition"] == DISPOSITION_ACCEPTED
        assert record["within_budget"] is True
    for request in client.requests:
        probe = build_request_budget_record(request, RequestBudgetPolicy())
        assert probe["within_budget"] is True


def test_an_armed_token_budget_reports_itself_even_with_a_ceiling():
    # Both controls armed: the ceiling would admit the first request,
    # the token budget (below the fixed contract) rejects it — and the
    # rejection is attributed to the TOKEN control, not the ceiling.
    ceiling = estimate_context_characters(list(HEAD)) + 6000
    agent, client, _ = _ceiling_run(
        ceiling, max_request_input_tokens=100
    )

    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)

    assert excinfo.value.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    assert client.requests == []
    trace = agent.last_trace
    operations = _operations(trace)
    assert "context.budget_exceeded" in operations
    # The ceiling never fired: request one fits under it, so no
    # restriction/compaction can claim this failure.
    assert not _ceiling_operations(trace)
    event = [
        item
        for item in trace.lifecycle_events
        if item["operation"] == "context.budget_exceeded"
    ][0]
    assert event["effective_input_budget_tokens"] == 100
    assert (
        trace.iterations[0].request_budget["disposition"]
        == DISPOSITION_REJECTED
    )


# ---------------------------------------------------------------------------
# 4 — provider overflow diagnostics stay content-free
# ---------------------------------------------------------------------------


def test_provider_overflow_diagnostics_never_embed_request_content():
    question = "{} — {}".format(QUESTION, SENTINEL)
    observation = _observation("src/a.py", SENTINEL[:8])
    provider_message = (
        "context_length_exceeded: maximum context length exceeded; "
        + "z" * 600
        + " " + SENTINEL
    )
    fake = FakeLLMClient(LLMError(provider_message))
    dispatcher = FakeDispatcher(observation)
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(),
    )

    with pytest.raises(LLMError):
        agent.run(question)

    trace = agent.last_trace
    record = trace.iterations[0].request_budget
    assert record is not None
    assert record["disposition"] == DISPOSITION_ACCEPTED  # preflight fit
    error_block = record["provider_error"]
    assert error_block["category"] == "context_length_exceeded"
    assert error_block["preflight_within_budget"] is True
    assert error_block["preflight_total_tokens_est"] == record["totals"][
        "total_tokens_est"
    ]
    # Privacy: neither the question nor the observation content may
    # leak into the diagnostic record — including the provider's own
    # message, which is clipped rather than copied whole.
    serialized = json.dumps(record, sort_keys=True)
    assert SENTINEL not in serialized
    assert "SENTINEL"[:8] not in serialized
    assert len(error_block["message"]) <= 500
    assert len(provider_message) > 500


# ---------------------------------------------------------------------------
# 5 — an armed-but-generous budget is a whole-run no-op
# ---------------------------------------------------------------------------


def test_an_armed_generous_budget_sends_the_identical_requests():
    default_agent, default_client, _ = _build(_read_answer_script())
    default_answer = default_agent.run(QUESTION)

    armed_agent, armed_client, _ = _build(
        _read_answer_script(), max_request_input_tokens=2_000_000
    )
    armed_answer = armed_agent.run(QUESTION)

    assert armed_answer == default_answer
    assert _transcript(armed_client) == _transcript(default_client)
    assert len(armed_client.requests) == len(default_client.requests)

    for iteration in default_agent.last_trace.iterations:
        record = iteration.request_budget
        assert record["disposition"] == DISPOSITION_ACCEPTED
        assert (
            record["budget"]["effective_input_budget_tokens"]
            == DEFAULT_EFFECTIVE_BUDGET
        )
    for iteration in armed_agent.last_trace.iterations:
        record = iteration.request_budget
        assert record["disposition"] == DISPOSITION_ACCEPTED
        assert record["budget"]["effective_input_budget_tokens"] == 2_000_000
    for agent in (default_agent, armed_agent):
        assert not (BUDGET_OPERATIONS & set(_operations(agent.last_trace)))
