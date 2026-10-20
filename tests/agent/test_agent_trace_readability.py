"""M8.8 follow-up — agent trace readability (spec Tests 1–8).

The trace must read as the conversation loop it documents. Since
M8.7.1 each iteration reads as the ordered interaction stages
**LLM Request → LLM Response → Actions → Tool Executions → Tool
Results → State Update → Continuation → Answer**, with the observable
inputs (messages actually sent) and outputs (the explicit
``llm_response`` record) recorded — plus the run-level structured
claims / M8.8 validation section after the iterations of structured
runs. Agent behavior is unchanged; these tests pin the observability
only.

Test 9 is running the full suites (reported separately).
"""

from __future__ import annotations

import json

from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
    render_markdown,
)
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolError, ToolResult

QUESTION = "How does the data onboarding workflow work?"
ANSWER = "The workflow has four deterministic stages."

PATH_READ = ".claude/agents/dag-agent.md"
STATEMENT_A = "The repository defines a DAG Agent."
STRUCTURED_FINAL = json.dumps(
    {
        "answer": ANSWER,
        "claims": [
            {
                "statement": STATEMENT_A,
                "evidence_refs": [
                    {"path": PATH_READ, "line_start": 1, "line_end": 20}
                ],
            }
        ],
    }
)

# The recorded conversation may only carry these message keys.
MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id"}
# … and the explicit LLM-response record only these.
LLM_RESPONSE_KEYS = {
    "type",
    "tool_call_count",
    "has_text",
    "tool_calls",
    "id",
    "name",
    "arguments",
}


# -- per-file test doubles (no conftest in this repo) ----------------------

class FakeLLMClient:
    """Queued responses (or queued exceptions); records requests."""

    def __init__(self, *responses, model=None) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = model

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    """Records dispatches; canned results first, else generic success."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _text_response(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_calls_response(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name, arguments=None, call_id="call_1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _read_result(path=PATH_READ, start=1, end=20):
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": path,
            "line_start": start,
            "line_end": end,
            "content": "# {} (excerpt)".format(path),
        },
        source={"repository": "sample-repo", "commit_sha": "a" * 40},
    )


def _failed_read(path="tool-registry/README.md"):
    return ToolResult(
        success=False,
        tool="read_file",
        error=ToolError(code="FILE_NOT_FOUND", message="no such file"),
    )


def _run(responses, results=None, config=None, question=QUESTION):
    """One scripted run through the real Agent (run_result)."""
    fake = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(*(results or ()))
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=config or AgentExecutionConfig(),
    )
    result = agent.run_result(question)
    return result, agent, dispatcher


def _walk_keys(value):
    """Yield every mapping key anywhere inside a JSON-ready value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, list):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


# ---------------------------------------------------------------------------
# Test 1 — the LLM request records the messages actually sent
# ---------------------------------------------------------------------------


def test_request_records_the_sent_conversation():
    big_content = "x" * 5000
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("search_repository", {"query": "x"})),
            _text_response(),
        ],
        results=[
            ToolResult(success=True, tool="search_repository", data={"hits": big_content}),
        ],
    )
    assert result.answer == ANSWER
    first, second = agent.last_trace.iterations

    # iteration 1: system + user, counts match the recorded list
    messages = first.request["messages"]
    assert first.request["message_count"] == len(messages) == 2
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == QUESTION
    assert messages[0]["content"].startswith(
        "You are a repository intelligence assistant"
    )
    assert first.request["tool_definition_count"] == 4

    # iteration 2: the tool observation rode back in the next request
    messages = second.request["messages"]
    assert second.request["message_count"] == len(messages) == 4
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[3]["tool_call_id"] == "call_1"
    # the observation content is a clipped *preview* of the tool JSON
    recorded = messages[3]["content"]
    assert recorded.startswith('{"success": true')
    assert "[trace clipped:" in recorded
    assert len(recorded) < len(big_content)

    # the human projection renders the request as a readable transcript
    markdown = render_markdown(agent.last_trace)
    assert "### LLM Request" in markdown
    assert "1. **system**" in markdown
    assert "2. **user**" in markdown
    assert QUESTION in markdown
    assert "4. **tool** (`call_1`)" in markdown


def test_hand_built_iteration_keeps_its_exact_request_shape():
    # a hand-built trace never records messages — canonical shape
    # of pre-existing serialized traces is unchanged
    from repo_intelligence.agent import AgentIteration

    payload = AgentIteration(
        iteration=1, request={"message_count": 2}
    ).to_dict()
    assert payload["request"] == {"message_count": 2}
    assert "llm_response" not in payload


# ---------------------------------------------------------------------------
# Test 2 — the LLM response is explicitly recorded
# ---------------------------------------------------------------------------


def test_llm_response_records_observable_output():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("search_repository", {"query": "x"})),
            _text_response(),
        ],
    )
    first, second = agent.last_trace.iterations

    assert first.llm_response["type"] == "TOOL_CALL"
    assert first.llm_response["tool_call_count"] == 1
    assert first.llm_response["has_text"] is False
    assert first.llm_response["tool_calls"] == [
        {"id": "call_1", "name": "search_repository", "arguments": {"query": "x"}}
    ]

    assert second.llm_response["type"] == "FINAL_ANSWER"
    assert second.llm_response["tool_call_count"] == 0
    assert second.llm_response["has_text"] is True
    assert "tool_calls" not in second.llm_response

    payload = agent.last_trace.to_dict()
    assert payload["iterations"][0]["llm_response"]["type"] == "TOOL_CALL"
    assert payload["iterations"][1]["llm_response"]["type"] == "FINAL_ANSWER"

    markdown = render_markdown(agent.last_trace)
    assert "### LLM Response" in markdown
    assert "Type: `TOOL_CALL`" in markdown
    assert "Finish reason: `tool_calls`" in markdown
    # the tool-call response is complete without prose — it says so —
    # and carries the correlation id the action/result stages reuse
    assert "Text: <none \u2014 response requested tool execution>" in markdown
    assert "Tool calls requested: `call_1`" in markdown
    # the final response states its type AND its text, together
    assert "Type: `FINAL_ANSWER`" in markdown
    assert "Text:\n{}".format(ANSWER) in markdown


def test_invalid_llm_response_is_recorded_as_invalid():
    fake = FakeLLMClient(
        LLMResponse(
            message=Message(role="assistant", content=None),
            finish_reason="stop",
        )
    )
    agent = Agent(llm_client=fake, tool_dispatcher=FakeDispatcher())
    try:
        agent.run(QUESTION)
    except AgentError:
        pass
    iteration = agent.last_trace.iterations[0]
    assert iteration.llm_response["type"] == "INVALID_RESPONSE"
    assert iteration.llm_response["has_text"] is False
    assert iteration.response_type == "INVALID_RESPONSE"


# ---------------------------------------------------------------------------
# Test 3 — requested vs executed stay visibly distinct
# ---------------------------------------------------------------------------


def test_requested_calls_and_executed_results_stay_distinct():
    result, agent, dispatcher = _run(
        responses=[
            _tool_calls_response(
                _call("read_file", {"path": PATH_READ}, "c_read")
            ),
            _text_response(),
        ],
        results=[_read_result()],
    )
    first = agent.last_trace.iterations[0]

    requested = first.llm_response["tool_calls"]
    assert requested == [
        {"id": "c_read", "name": "read_file", "arguments": {"path": PATH_READ}}
    ]
    # same ids, but separate records: requested (verbatim) vs executed
    assert [call["id"] for call in first.tool_calls] == ["c_read"]
    assert [entry["tool_call_id"] for entry in first.tool_results] == ["c_read"]
    assert first.tool_results[0]["success"] is True
    # what the dispatcher actually received matches the request
    assert dispatcher.calls[0].arguments == {"path": PATH_READ}


def test_budget_blocked_call_is_requested_but_never_executed():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "x"}))
    )
    dispatcher = FakeDispatcher()
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=0),
    )
    try:
        agent.run(QUESTION)
    except AgentError as exc:
        assert exc.termination_reason == "EXECUTION_LIMIT_REACHED"
    iteration = agent.last_trace.iterations[0]

    # requested and visible …
    assert iteration.llm_response["type"] == "TOOL_CALL"
    assert len(iteration.llm_response["tool_calls"]) == 1
    assert len(iteration.tool_calls) == 1
    # … but never executed, never a ToolResult
    assert iteration.tool_results == []
    assert len(iteration.blocked_tool_calls) == 1
    assert dispatcher.calls == []
    assert agent.last_trace.summary.tool_calls_blocked == 1

    markdown = render_markdown(agent.last_trace)
    assert "**Result:** NOT EXECUTED (execution budget exhausted)" in markdown


# ---------------------------------------------------------------------------
# Test 4 — a failed tool reads as requested vs failed
# ---------------------------------------------------------------------------


def test_failed_tool_reads_as_requested_then_failed_result():
    missing = "tool-registry/README.md"
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("read_file", {"path": missing})),
            _text_response("Could not find the registry README."),
        ],
        results=[_failed_read(missing)],
    )
    first = agent.last_trace.iterations[0]

    assert first.llm_response["tool_calls"][0]["arguments"] == {"path": missing}
    assert first.tool_results[0]["success"] is False
    assert first.tool_results[0]["error"]["code"] == "FILE_NOT_FOUND"
    assert first.observations[0]["success"] is False

    markdown = render_markdown(agent.last_trace)
    assert '"path": "tool-registry/README.md"' in markdown
    assert "**Result:** FAILURE (FILE_NOT_FOUND)" in markdown
    assert (
        "**Observation:** read_file failure (FILE_NOT_FOUND) "
        "returned to the Agent."
    ) in markdown
    # the failure is an observation like any other — visible, not hidden
    assert "* `read_file` (`call_1`) → FAILURE (FILE_NOT_FOUND)" in markdown


# ---------------------------------------------------------------------------
# Test 5 — iterations read as one growing conversation
# ---------------------------------------------------------------------------


def test_iterations_read_as_a_growing_conversation():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(
                _call("search_repository", {"query": "a"}, "c1")
            ),
            _tool_calls_response(_call("read_file", {"path": "b"}, "c2")),
            _text_response(),
        ],
        results=[
            ToolResult(
                success=True,
                tool="search_repository",
                data={"query": "a"},
            ),
            _read_result("b", 1, 2),
        ],
    )
    trace = agent.last_trace
    payload = trace.to_dict()

    assert [item["response_type"] for item in payload["iterations"]] == [
        "TOOL_CALL",
        "TOOL_CALL",
        "FINAL_ANSWER",
    ]
    counts = [item["request"]["message_count"] for item in payload["iterations"]]
    assert counts == [2, 4, 6]
    roles = [
        [message["role"] for message in item["request"]["messages"]]
        for item in payload["iterations"]
    ]
    assert roles[0] == ["system", "user"]
    assert roles[1] == ["system", "user", "assistant", "tool"]
    assert roles[2] == ["system", "user", "assistant", "tool", "assistant", "tool"]

    markdown = render_markdown(trace)
    # the ordered interaction stages per tool iteration (M8.7.1), in
    # loop order: request → response → actions → executions → results
    # → state update → continuation
    request_at = markdown.index("### LLM Request")
    response_at = markdown.index("### LLM Response")
    actions_at = markdown.index("### Actions")
    executions_at = markdown.index("### Tool Executions")
    results_at = markdown.index("### Tool Results")
    state_at = markdown.index("### State Update")
    continuation_at = markdown.index("### Continuation")
    assert request_at < response_at < actions_at < executions_at
    assert executions_at < results_at < state_at < continuation_at
    assert "## Iteration 1" in markdown
    assert "## Iteration 2" in markdown
    assert "## Iteration 3" in markdown
    assert "## Run Summary" in markdown
    # the second request visibly carries the first observation back in
    assert "**Includes 1 previous observation:**" in markdown
    assert "* `search_repository` (`c1`) → SUCCESS" in markdown
    # State Update names the next request the loop will make
    assert "Next LLM request: iteration 2 (4 messages)" in markdown


# ---------------------------------------------------------------------------
# Test 6 — structured claims are visible in the trace
# ---------------------------------------------------------------------------


def test_structured_claims_are_visible_in_the_trace():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(
                _call("read_file", {"path": PATH_READ}, "c_read")
            ),
            _text_response(STRUCTURED_FINAL),
        ],
        results=[_read_result()],
        config=AgentExecutionConfig(structured_claims=True),
    )
    assert [claim.statement for claim in result.claims] == [STATEMENT_A]

    payload = agent.last_trace.to_dict()
    block = payload["evidence_validation"]
    assert block["claims"] == [
        {
            "statement": STATEMENT_A,
            "evidence_refs": [
                {"path": PATH_READ, "line_start": 1, "line_end": 20}
            ],
        }
    ]

    markdown = render_markdown(agent.last_trace)
    assert "### Structured Claims" in markdown
    assert STATEMENT_A in markdown
    assert "evidence: `{}`".format("{}:1-20".format(PATH_READ)) in markdown


def test_free_form_run_has_no_claims_section():
    result, agent, _ = _run(
        responses=[_text_response()],
    )
    assert result.claims == ()
    assert "evidence_validation" not in agent.last_trace.to_dict()
    markdown = render_markdown(agent.last_trace)
    assert "### Structured Claims" not in markdown
    assert "### M8.8 Evidence Validation" not in markdown


# ---------------------------------------------------------------------------
# Test 7 — validation results are bound to their claims
# ---------------------------------------------------------------------------


def test_validation_results_are_bound_to_claims_after_the_iterations():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(
                _call("read_file", {"path": PATH_READ}, "c_read")
            ),
            _text_response(STRUCTURED_FINAL),
        ],
        results=[_read_result()],
        config=AgentExecutionConfig(structured_claims=True),
    )
    assert len(result.evidence_validation) == 1
    entry = result.evidence_validation[0]
    assert entry.claim.statement == result.claims[0].statement
    assert entry.status == "SUPPORTED"
    assert entry.reason.startswith("matched gathered evidence")

    payload = agent.last_trace.to_dict()
    recorded = payload["evidence_validation"]["results"][0]
    assert recorded["claim"]["statement"] == STATEMENT_A
    assert recorded["status"] == "SUPPORTED"

    markdown = render_markdown(agent.last_trace)
    answer_at = markdown.index("### Answer")
    claims_at = markdown.index("### Structured Claims")
    validation_at = markdown.index("### M8.8 Evidence Validation")
    summary_at = markdown.index("## Run Summary")
    assert answer_at < claims_at < validation_at < summary_at
    assert "`SUPPORTED` — {}".format(STATEMENT_A) in markdown


# ---------------------------------------------------------------------------
# Test 8 — no hidden reasoning anywhere in the readable trace
# ---------------------------------------------------------------------------


def test_no_hidden_reasoning_anywhere_in_the_readable_trace():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("search_repository", {"query": "x"})),
            _text_response(STRUCTURED_FINAL),
        ],
        results=[
            ToolResult(success=True, tool="search_repository", data={"hits": 1})
        ],
        config=AgentExecutionConfig(structured_claims=True),
    )
    payload = agent.last_trace.to_dict()

    for key in _walk_keys(payload):
        lowered = key.lower()
        for banned in BANNED_REASONING_FIELDS:
            assert banned not in lowered, key
    serialized = json.dumps(payload).lower()
    for banned in BANNED_REASONING_FIELDS:
        assert banned not in serialized
    assert "thought" not in serialized

    # the new blocks carry only their documented keys
    first = payload["iterations"][0]
    assert set(first["llm_response"]) <= LLM_RESPONSE_KEYS
    for call in first["llm_response"].get("tool_calls") or []:
        assert set(call) <= {"id", "name", "arguments"}
    for message in first["request"]["messages"]:
        assert set(message) <= MESSAGE_KEYS
        for call in message.get("tool_calls") or []:
            assert set(call) <= {"id", "name", "arguments"}

    markdown = render_markdown(agent.last_trace)
    assert "thought" not in markdown.lower()
    for banned in BANNED_REASONING_FIELDS:
        assert banned not in markdown.lower()
