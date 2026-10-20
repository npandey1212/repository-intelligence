"""M8.8 integration tests — structured claims validated after the run.

End-to-end through the REAL Agent (M8.4 loop, single M8.5 tool-call
limit), a scripted offline model, and a canned dispatcher — then the
EXISTING deterministic validator (``repo_intelligence.evidence``)
over the run's own trace. No API key, no network, no second loop, no
second agent: validation runs inside ``Agent.run_result`` strictly
AFTER the investigation, reads only the finished trace, and must
never execute a tool. The model is never the authority for "was this
evidence gathered?" — only the validator is.

Required integration cases:

    1 supported claim        successful read + matching ref → SUPPORTED
    2 failed evidence        failed read + matching ref → UNSUPPORTED
                              (FILE_NOT_FOUND is never positive)
    3 multiple claims        independent results, input order preserved
    4 trace contains validation  claims + deterministic results
                              recorded in the runtime trace
    5 bounded behavior       validation dispatches no tool
                              (call counts before == after)
    6 no hidden reasoning    structured output + trace field names
    7 existing regressions   tests/agent (loop/bounded/trace/react) and
                              tests/evidence stay green — run with this
                              file (see the documented command)

The loop/budget mechanics themselves are covered by
``tests/agent/``; the validator's rules by
``tests/evidence/test_evidence_traceability.py``.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import (
    Agent,
    AgentExecutionConfig,
    AgentRunResult,
    JsonFileTraceSink,
    render_markdown,
)
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.evidence import (
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    gather_evidence,
    validate_claims,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolError, ToolResult

OBJECTIVE = "Investigate which agents and MCP connections this repository defines."
ANSWER = (
    "The repository defines a DAG Agent "
    "(.claude/agents/dag-agent.md) and MCP servers in "
    "tool-registry/servers.yaml."
)

PATH_READ = ".claude/agents/dag-agent.md"
PATH_OTHER = "tool-registry/servers.yaml"
PATH_MISSING = "tool-registry/README.md"

STATEMENT_A = "The repository defines a DAG Agent."
STATEMENT_B = "MCP servers are listed in tool-registry/servers.yaml."
STATEMENT_C = "tool-registry/README.md documents the tool registry."

# Spec §9 on top of the trace module's own ban list (same pattern as
# the M8.6 tests): plain "reasoning" and "reasoning_trace" too.
EXTRA_BANNED_FIELDS = ("reasoning", "reasoning_trace")

# Deterministic reason vocabulary produced by the validator (never
# model-authored text).
REASON_PREFIXES = (
    "matched gathered evidence",
    "no gathered evidence matches",
    "no evidence references supplied",
    "evidence metadata insufficient",
)


class FakeLLMClient:
    """Queued scripted responses; records every request."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


class FakeDispatcher:
    """Canned ToolResults in order; records every dispatch (the count
    is the bound Test 5 asserts never grows during validation)."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if not self._queue:
            raise AssertionError("no scripted ToolResult for {}".format(tool_call.name))
        return self._queue.pop(0)


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name: str, arguments, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _claim_json(statement: str, path, start=None, end=None):
    ref = {"path": path}
    if start is not None:
        ref["line_start"] = start
    if end is not None:
        ref["line_end"] = end
    return {"statement": statement, "evidence_refs": [ref]}


def _final(answer: str, claims) -> LLMResponse:
    """The model's explicit structured final response (the contract)."""
    return _text_response(json.dumps({"answer": answer, "claims": claims}))


def _read(path: str, start: int, end: int) -> ToolResult:
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": path,
            "line_start": start,
            "line_end": end,
            "content": "# {} (excerpt)".format(path),
        },
        source={"repository": "repository-intelligence", "commit_sha": "a" * 40},
    )


def _read_failed(path: str) -> ToolResult:
    return ToolResult(
        success=False,
        tool="read_file",
        error=ToolError(code="FILE_NOT_FOUND", message="no such file"),
    )


def _read_call(path: str, start: int, end: int, call_id: str) -> ToolCall:
    return _call(
        "read_file",
        {"path": path, "line_start": start, "line_end": end},
        call_id,
    )


def _run(responses, results, sink=None, structured=True, max_tool_calls=8):
    """One scripted run through the real Agent (single run_result)."""
    fake = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(*results)
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        trace_sink=sink,
        execution_config=AgentExecutionConfig(
            max_tool_calls=max_tool_calls,
            structured_claims=structured,
        ),
    )
    result = agent.run_result(OBJECTIVE)
    return result, agent, dispatcher


def _single_read_run(sink=None, statement=STATEMENT_A, path=PATH_READ):
    """Iteration 1: read PATH_READ (success) → iteration 2: structured
    final claiming that read. The baseline for tests 1, 4, 5, 6."""
    return _run(
        responses=[
            _tool_response(_read_call(path, 1, 20, "c_read")),
            _final(
                ANSWER,
                [_claim_json(statement, path, 1, 20)],
            ),
        ],
        results=[_read(path, 1, 20)],
        sink=sink,
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


# ---------------------------------------------------------------------------
# 1 — supported claim
# ---------------------------------------------------------------------------


def test_supported_claim_references_successful_read():
    result, agent, dispatcher = _single_read_run()

    assert isinstance(result, AgentRunResult)
    assert result.answer == ANSWER  # readable text, never the raw JSON
    assert not result.answer.strip().startswith("{")
    assert len(result.claims) == 1

    assert len(result.evidence_validation) == 1
    only = result.evidence_validation[0]
    assert only.status == STATUS_SUPPORTED
    assert only.claim.statement == STATEMENT_A
    assert only.matched_evidence, "must record which gathered item matched"
    assert only.matched_evidence[0].path == PATH_READ
    assert only.matched_evidence[0].tool == "read_file"
    assert only.reason.startswith("matched gathered evidence")

    # the evidence really is the run's successful ToolResult
    index = gather_evidence(agent.last_trace)
    assert [item.path for item in index.gathered] == [PATH_READ]


# ---------------------------------------------------------------------------
# 2 — failed evidence
# ---------------------------------------------------------------------------


def test_failed_evidence_is_never_supported():
    result, agent, dispatcher = _run(
        responses=[
            _tool_response(_read_call(PATH_MISSING, 1, 5, "c_missing")),
            _final(
                ANSWER,
                [_claim_json(STATEMENT_C, PATH_MISSING, 1, 5)],
            ),
        ],
        results=[_read_failed(PATH_MISSING)],
    )

    assert len(result.evidence_validation) == 1
    only = result.evidence_validation[0]
    assert only.status == STATUS_UNSUPPORTED
    assert only.matched_evidence == ()
    assert "FILE_NOT_FOUND" in only.reason

    # the failed attempt is recorded as failed — never as evidence
    index = gather_evidence(agent.last_trace)
    assert index.gathered == ()
    assert len(index.failed) == 1
    assert index.failed[0].path == PATH_MISSING


# ---------------------------------------------------------------------------
# 3 — multiple claims
# ---------------------------------------------------------------------------


def test_multiple_claims_validate_independently():
    result, agent, dispatcher = _run(
        responses=[
            _tool_response(
                _read_call(PATH_READ, 1, 20, "c_a"),
                _read_call(PATH_OTHER, 1, 30, "c_b"),
                _read_call(PATH_MISSING, 1, 5, "c_c"),
            ),
            _final(
                ANSWER,
                [
                    _claim_json(STATEMENT_A, PATH_READ, 1, 20),
                    _claim_json(STATEMENT_B, PATH_OTHER, 1, 30),
                    _claim_json(STATEMENT_C, PATH_MISSING, 1, 5),
                ],
            ),
        ],
        results=[
            _read(PATH_READ, 1, 20),
            _read(PATH_OTHER, 1, 30),
            _read_failed(PATH_MISSING),
        ],
    )

    statuses = [entry.status for entry in result.evidence_validation]
    assert statuses == [STATUS_SUPPORTED, STATUS_SUPPORTED, STATUS_UNSUPPORTED]
    # results are independent and in input order
    statements = [entry.claim.statement for entry in result.evidence_validation]
    assert statements == [STATEMENT_A, STATEMENT_B, STATEMENT_C]
    # each supported claim matched only its own file
    assert result.evidence_validation[0].matched_evidence[0].path == PATH_READ
    assert result.evidence_validation[1].matched_evidence[0].path == PATH_OTHER
    assert result.evidence_validation[2].matched_evidence == ()


# ---------------------------------------------------------------------------
# 4 — trace contains the structured claims + deterministic validation
# ---------------------------------------------------------------------------


def test_trace_records_structured_claims_and_validation(tmp_path):
    sink = JsonFileTraceSink(root=str(tmp_path / "runs"))
    result, agent, dispatcher = _single_read_run(sink=sink)

    trace_dict = agent.last_trace.to_dict()
    block = trace_dict["evidence_validation"]
    assert [claim["statement"] for claim in block["claims"]] == [STATEMENT_A]
    assert len(block["results"]) == 1
    entry = block["results"][0]
    assert entry["status"] == STATUS_SUPPORTED
    assert entry["claim"]["evidence_refs"] == [
        {"path": PATH_READ, "line_start": 1, "line_end": 20}
    ]
    assert entry["matched_evidence"][0]["path"] == PATH_READ
    assert entry["reason"].startswith("matched gathered evidence")

    # canonical JSON trace stays serializable and carries the block
    json.dumps(trace_dict)

    # recorded BEFORE persistence: the written runtime trace has it
    written = json.loads(
        (tmp_path / "runs" / agent.last_trace.run_id / "trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["evidence_validation"]["results"][0]["status"] == (
        STATUS_SUPPORTED
    )

    # the human projection shows it as the run-level section after
    # the iterations and before the run summary (M8.7.1 stage order)
    markdown = render_markdown(agent.last_trace)
    assert "### Structured Claims" in markdown
    assert "### M8.8 Evidence Validation" in markdown
    assert STATEMENT_A in markdown
    assert (
        markdown.index("### M8.8 Evidence Validation")
        < markdown.index("## Run Summary")
    )
    assert markdown.index("### Answer") < markdown.index(
        "### Structured Claims"
    )

    # ... and a run WITHOUT structured claims gains no new field
    plain_result, plain_agent, _ = _run(
        responses=[
            _tool_response(_read_call(PATH_READ, 1, 20, "c_read")),
            _text_response(ANSWER),
        ],
        results=[_read(PATH_READ, 1, 20)],
        structured=False,
    )
    assert plain_result.claims == ()
    assert plain_result.evidence_validation == ()
    assert "evidence_validation" not in plain_agent.last_trace.to_dict()


# ---------------------------------------------------------------------------
# 5 — Agent behavior stays bounded (validation dispatches no tool)
# ---------------------------------------------------------------------------


def test_validation_never_executes_a_tool():
    result, agent, dispatcher = _single_read_run()

    # exactly the scripted investigation ran — no extra dispatch
    assert len(dispatcher.calls) == 1
    assert agent.last_trace.summary.tool_calls_executed == 1
    tool_calls_before_validation = agent.last_trace.summary.total_tool_calls
    dispatches_before_validation = len(dispatcher.calls)

    # replaying the deterministic validation changes nothing:
    replay = validate_claims(result.claims, gather_evidence(agent.last_trace))
    assert len(dispatcher.calls) == dispatches_before_validation
    assert agent.last_trace.summary.total_tool_calls == (
        tool_calls_before_validation
    )
    assert [entry.status for entry in replay] == [
        entry.status for entry in result.evidence_validation
    ]


# ---------------------------------------------------------------------------
# 6 — no hidden reasoning anywhere in the structured output or trace
# ---------------------------------------------------------------------------


def test_structured_output_and_trace_have_no_hidden_reasoning():
    result, agent, dispatcher = _single_read_run()

    structured_payload = {
        "answer": result.answer,
        "claims": [claim.to_dict() for claim in result.claims],
        "results": [entry.to_dict() for entry in result.evidence_validation],
        "trace": agent.last_trace.to_dict(),
    }
    keys = list(_walk_keys(structured_payload))
    assert keys, "payload must not be empty"
    for banned in tuple(BANNED_REASONING_FIELDS) + EXTRA_BANNED_FIELDS:
        assert banned not in keys, banned

    # validation reasons are deterministic match descriptions only
    for entry in result.evidence_validation:
        assert entry.reason.startswith(REASON_PREFIXES), entry.reason


# ---------------------------------------------------------------------------
# Extras — config contract, tolerant parsing, run() delegation
# ---------------------------------------------------------------------------


def test_structured_claims_config_must_be_boolean():
    with pytest.raises(ValueError):
        AgentExecutionConfig(structured_claims="yes")


def test_malformed_structured_response_degrades_without_failing_the_run():
    """The model answered with prose instead of the JSON contract: the
    answer stands, there is simply nothing to validate — recorded as
    an empty (never inventing) claim set."""
    result, agent, dispatcher = _run(
        responses=[
            _tool_response(_read_call(PATH_READ, 1, 20, "c_read")),
            _text_response("The DAG Agent is defined in dag-agent.md."),  # not JSON
        ],
        results=[_read(PATH_READ, 1, 20)],
    )

    assert result.answer == "The DAG Agent is defined in dag-agent.md."
    assert result.claims == ()
    assert result.evidence_validation == ()
    block = agent.last_trace.to_dict()["evidence_validation"]
    assert block == {"claims": [], "results": []}


def test_claims_survive_an_unusable_answer_field_and_are_validated():
    """Closeout regression: the structured payload DID contain claims
    but the ``answer`` field was not a usable string. The answer falls
    back to the raw text, yet the claims must still reach the
    deterministic validator — validation is never silently skipped."""
    raw = json.dumps(
        {
            "answer": {"not": "a string"},
            "claims": [_claim_json(STATEMENT_A, PATH_READ, 1, 20)],
        }
    )
    result, agent, dispatcher = _run(
        responses=[
            _tool_response(_read_call(PATH_READ, 1, 20, "c_read")),
            _text_response(raw),
        ],
        results=[_read(PATH_READ, 1, 20)],
    )

    assert result.answer == raw  # readable fallback, never a run failure
    assert [claim.statement for claim in result.claims] == [STATEMENT_A]
    assert [entry.status for entry in result.evidence_validation] == [
        STATUS_SUPPORTED
    ]
    block = agent.last_trace.to_dict()["evidence_validation"]
    assert len(block["claims"]) == 1
    assert block["claims"][0]["statement"] == STATEMENT_A
    assert block["results"][0]["status"] == STATUS_SUPPORTED
    # still only the one investigation dispatch — validation runs no tool
    assert [call.id for call in dispatcher.calls] == ["c_read"]


def test_early_closed_payload_recovers_the_intact_claims_array():
    """Closeout regression (real-model shape): gpt-4o-mini closed the
    top-level object right after ``answer`` and trailed the INTACT
    claims array outside it. Recovery is verbatim and deterministic:
    the answer prefix parses, the array parses, validation runs —
    nothing is repaired or invented."""
    answer_text = "The repository defines a DAG Agent."
    raw = (
        '{"answer": "%s"},'
        '"claims":[{"statement":"%s","evidence_refs":'
        '[{"path":"%s","line_start":1,"line_end":20}]}]}'
    ) % (answer_text, STATEMENT_A, PATH_READ)
    with pytest.raises(ValueError):
        json.loads(raw)  # strict whole-text parse must fail (extra data)
    result, agent, dispatcher = _run(
        responses=[
            _tool_response(_read_call(PATH_READ, 1, 20, "c_read")),
            _text_response(raw),
        ],
        results=[_read(PATH_READ, 1, 20)],
    )

    assert result.answer == answer_text
    assert [claim.statement for claim in result.claims] == [STATEMENT_A]
    assert [entry.status for entry in result.evidence_validation] == [
        STATUS_SUPPORTED
    ]
    block = agent.last_trace.to_dict()["evidence_validation"]
    assert len(block["claims"]) == 1
    assert block["results"][0]["status"] == STATUS_SUPPORTED


def test_broken_claims_array_still_degrades_to_an_empty_block():
    """The other real-model slip — a stray ``}`` INSIDE the claims
    array — leaves nothing parseable: recovery must NOT invent
    structure. The raw answer stands, the validation block is empty."""
    raw = (
        '{"answer": "Some answer.",'
        '"claims":[{"statement":"%s","evidence_refs":'
        '[{"path":"%s","line_start":1,"line_end":20}]}},'
        '{"statement":"second","evidence_refs":[]}]}'
    ) % (STATEMENT_A, PATH_READ)
    result, agent, dispatcher = _run(
        responses=[
            _tool_response(_read_call(PATH_READ, 1, 20, "c_read")),
            _text_response(raw),
        ],
        results=[_read(PATH_READ, 1, 20)],
    )

    assert result.claims == ()
    assert result.evidence_validation == ()
    assert agent.last_trace.to_dict()["evidence_validation"] == {
        "claims": [],
        "results": [],
    }


def test_run_delegates_to_run_result_and_returns_the_readable_answer():
    fake = FakeLLMClient(
        _tool_response(_read_call(PATH_READ, 1, 20, "c_read")),
        _final(ANSWER, [_claim_json(STATEMENT_A, PATH_READ, 1, 20)]),
    )
    dispatcher = FakeDispatcher(_read(PATH_READ, 1, 20))
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(structured_claims=True),
    )

    answer = agent.run(OBJECTIVE)
    assert answer == ANSWER
    # the run's outcome is still fully observable on the agent/trace
    assert agent.last_trace.evidence_validation is not None
    assert agent.last_trace.evidence_validation["results"][0]["status"] == (
        STATUS_SUPPORTED
    )
