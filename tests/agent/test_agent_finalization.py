"""Execution-lifecycle boundary tests — stop tools, then finalize.

The minimal M17 fix: a run that reaches its execution boundary stops
taking tools and closes from the state it already collected, instead
of failing the whole workflow. Covered here:

1. tool-call boundary   → no further tool executes, the finalization
                           pass is issued with ``tools=[]``, and the
                           structured result returns through the
                           ordinary contract
2. evidence preservation → references gathered before the boundary are
                           carried into the finalization request (and
                           counted on the trace)
3. context-length failure → the oversized request is never replayed;
                           the run finalizes from a bounded digest
4. transient provider failure → unchanged ``LLMError`` (existing retry
                           semantics untouched)
5. a model that asks for tools DURING finalization is refused, never
                           executed
6. the default config is unchanged: budget exhaustion still raises
                           ``EXECUTION_LIMIT_REACHED``

Everything is scripted and LLM-free: queued responses, a recording
dispatcher, one question — no network, no API key.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
)
from repo_intelligence.agent.agent import FINALIZATION_DIGEST_CHARS
from repo_intelligence.llm import LLMError, LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

QUESTION = "How do the agents and tools interact?"
READ_A = "src/alpha.py"
READ_B = "src/beta.py"


class TinyReport(BaseModel):
    """Minimal typed contract (M8.9) — the result schema under test."""

    question: str
    findings: list = []
    evidence: list = []
    status: str = "complete"


class FakeLLMClient:
    """Queued responses (or queued exceptions); records every request."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than queued responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    """Records every dispatch; returns queued results in order."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _structured(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name, arguments, call_id="call_1"):
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _read(path, call_id="call_1"):
    return _call("read_file", {"path": path}, call_id)


def _read_result(path):
    return ToolResult(
        success=True,
        tool="read_file",
        data={"path": path, "content": "file content of {}".format(path)},
        source={"artifact": path},
    )


def _payload(**overrides):
    data = {
        "question": QUESTION,
        "findings": ["alpha depends on beta"],
        "evidence": [READ_A, READ_B],
        "status": "complete",
    }
    data.update(overrides)
    return data


def _agent(responses, dispatcher, max_tool_calls):
    return Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_tool_calls=max_tool_calls,
            finalize_on_boundary=True,
        ),
        structured_output_model=TinyReport,
    )


def _operations(trace):
    return [event["operation"] for event in trace.lifecycle_events]


def _event(trace, operation):
    matches = [
        event for event in trace.lifecycle_events
        if event["operation"] == operation
    ]
    assert matches, operation
    return matches[0]


# ---------------------------------------------------------------------------
# 1 — tool-call boundary → finalization, no further tool execution
# ---------------------------------------------------------------------------


def test_tool_call_boundary_finalizes_without_executing_another_tool():
    dispatcher = FakeDispatcher(_read_result(READ_A))
    agent = _agent(
        [
            _tool_calls(_read(READ_A)),
            _tool_calls(_read(READ_B, "call_2")),
            _structured(_payload()),
        ],
        dispatcher,
        max_tool_calls=1,
    )

    result = agent.run_result(QUESTION)

    # The boundary blocked the second tool: exactly one dispatch total.
    assert len(dispatcher.calls) == 1
    assert isinstance(result.structured_output, TinyReport)
    assert result.structured_output.question == QUESTION

    # The finalization pass is the request AFTER the blocked tool call,
    # and it is tool-free.
    assert len(agent._llm_client.requests) == 3
    finalization = agent._llm_client.requests[-1]
    assert finalization.tools == []
    assert finalization.structured_output is not None

    trace = agent.last_trace
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    assert _operations(trace) == [
        "execution.boundary_reached",
        "finalization.started",
        "tools.disabled",
        "finalization.completed",
    ]
    summary = trace.summary
    assert summary.tool_calls_executed == 1
    assert summary.tool_calls_blocked == 1
    # The SECOND iteration is the one that hit the boundary.
    boundary_iteration = trace.iterations[1]
    blocked = boundary_iteration.blocked_tool_calls
    assert blocked and blocked[0]["reason"] == "EXECUTION_LIMIT_REACHED"
    # The boundary iteration continues INTO finalization (it does not
    # terminate the run by itself).
    assert boundary_iteration.continuation["decision"] == "CONTINUE"
    assert "finalization" in boundary_iteration.continuation["reason"]


# ---------------------------------------------------------------------------
# 2 — evidence gathered before the boundary survives finalization
# ---------------------------------------------------------------------------


def test_evidence_collected_before_the_boundary_survives_finalization():
    dispatcher = FakeDispatcher(_read_result(READ_A), _read_result(READ_B))
    agent = _agent(
        [
            _tool_calls(_read(READ_A)),
            _tool_calls(_read(READ_B, "call_2")),
            _tool_calls(_read(READ_B, "call_3")),
            _structured(_payload()),
        ],
        dispatcher,
        max_tool_calls=2,
    )

    result = agent.run_result(QUESTION)

    assert len(dispatcher.calls) == 2
    finalization_message = agent._llm_client.requests[-1].messages[-1]
    # Both references reach the finalization prompt, with their
    # inspected/discovered status intact.
    assert "Evidence references (2):" in finalization_message.content
    assert READ_A in finalization_message.content
    assert READ_B in finalization_message.content
    assert "inspected" in finalization_message.content
    assert finalization_message.content.count("* " + READ_A) == 1
    # The typed result still carries both references through the
    # ordinary contract.
    assert result.structured_output.evidence == [READ_A, READ_B]
    started = _event(agent.last_trace, "finalization.started")
    assert started["evidence_count"] == 2
    assert started["observation_count"] == 2


# ---------------------------------------------------------------------------
# 3 — context exhaustion: never the same oversized request again
# ---------------------------------------------------------------------------


def test_context_length_failure_finalizes_without_blind_retry():
    dispatcher = FakeDispatcher(_read_result(READ_A))
    client = FakeLLMClient(
        _tool_calls(_read(READ_A)),
        LLMError(
            "LLM request failed: Error code: 400 - {'error': {'message': "
            "\"This model's maximum context length is 128000 tokens. "
            "However, your messages resulted in 129592 tokens.\", "
            "'type': 'invalid_request_error', "
            "'code': 'context_length_exceeded'}}"
        ),
        _structured(_payload()),
    )
    agent = Agent(
        client,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_tool_calls=8, finalize_on_boundary=True
        ),
        structured_output_model=TinyReport,
    )

    result = agent.run_result(QUESTION)

    # The oversized request was NOT replayed: one loop request, then
    # one bounded finalization request (fewer messages, no tools).
    assert len(client.requests) == 3
    oversized = client.requests[1]
    finalization = client.requests[2]
    assert oversized.tools
    assert finalization.tools == []
    assert len(finalization.messages) < len(oversized.messages)

    trace = agent.last_trace
    assert isinstance(result.structured_output, TinyReport)
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    operations = _operations(trace)
    assert operations[0] == "context.length_exceeded"
    assert "finalization.started" in operations
    assert operations[-1] == "finalization.completed"
    # The failing iteration is closed and continues into finalization.
    failed_iteration = trace.iterations[1]
    assert failed_iteration.response_type == "ERROR"
    assert failed_iteration.continuation["decision"] == "CONTINUE"
    # Exactly one tool executed — the boundary was never re-entered.
    assert len(dispatcher.calls) == 1


def test_context_length_error_is_classified_not_just_any_llm_error():
    from repo_intelligence.llm import is_context_length_error

    context = LLMError("... code': 'context_length_exceeded'")
    transient = LLMError("LLM request failed: rate limit exceeded")
    assert is_context_length_error(context) is True
    assert is_context_length_error(transient) is False
    assert is_context_length_error(ValueError("context_length_exceeded")) is False


# ---------------------------------------------------------------------------
# 4 — transient provider failure keeps the existing semantics
# ---------------------------------------------------------------------------


def test_transient_llm_error_is_still_an_llm_error_with_no_finalization():
    dispatcher = FakeDispatcher()
    client = FakeLLMClient(LLMError("LLM request failed: rate limit"))
    agent = Agent(
        client,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_tool_calls=8, finalize_on_boundary=True
        ),
        structured_output_model=TinyReport,
    )

    with pytest.raises(LLMError):
        agent.run_result(QUESTION)

    assert len(client.requests) == 1
    trace = agent.last_trace
    assert trace.termination_reason == "LLM_ERROR"
    assert trace.lifecycle_events == []


# ---------------------------------------------------------------------------
# 5 — a tool request DURING finalization is refused, never executed
# ---------------------------------------------------------------------------


def test_finalization_never_executes_a_requested_tool():
    dispatcher = FakeDispatcher(_read_result(READ_A))
    agent = _agent(
        [
            _tool_calls(_read(READ_A)),
            _tool_calls(_read(READ_B, "call_2")),
            _tool_calls(_read(READ_B, "call_3")),
        ],
        dispatcher,
        max_tool_calls=1,
    )

    with pytest.raises(AgentError) as excinfo:
        agent.run_result(QUESTION)

    assert excinfo.value.termination_reason == "FINALIZATION_FAILED"
    assert len(dispatcher.calls) == 1  # still exactly one tool, ever
    trace = agent.last_trace
    assert trace.termination_reason == "FINALIZATION_FAILED"
    assert _operations(trace)[-1] == "finalization.failed"


# ---------------------------------------------------------------------------
# 6 — bounded digest (the finalization request stays finite)
# ---------------------------------------------------------------------------


def test_finalization_digest_is_bounded_even_for_a_long_investigation():
    # 40 executions, then a 41st tool request that hits the boundary.
    responses = [
        _tool_calls(_read("file{}.py".format(index), "call_{}".format(index)))
        for index in range(41)
    ]
    responses.append(_structured(_payload()))
    dispatcher = FakeDispatcher(
        *[
            ToolResult(
                success=True,
                tool="read_file",
                data={
                    "path": "file{}.py".format(index),
                    "content": "x" * 3000,
                },
            )
            for index in range(40)
        ]
    )
    agent = _agent(responses, dispatcher, max_tool_calls=40)

    agent.run_result(QUESTION)

    assert len(dispatcher.calls) == 40
    finalization_message = agent._llm_client.requests[-1].messages[-1]
    content = finalization_message.content
    assert len(content) < FINALIZATION_DIGEST_CHARS + 8000
    # Every evidence reference is still listed — only the clipped
    # observations are truncated.
    assert "Evidence references (40):" in content
    observations = content.split("Tool observations (", 1)[1]
    rendered = [
        line for line in observations.splitlines()
        if line.startswith("* {")
    ]
    assert rendered and len(rendered) < 40
    assert "omitted: digest limit reached" in observations


# ---------------------------------------------------------------------------
# 7 — the default configuration is unchanged
# ---------------------------------------------------------------------------


def test_default_config_keeps_the_established_budget_termination():
    config = AgentExecutionConfig()
    assert config.finalize_on_boundary is False
    assert config.max_tool_calls == 8

    dispatcher = FakeDispatcher(_read_result(READ_A))
    agent = Agent(
        FakeLLMClient(
            *[
                _tool_calls(_read(READ_B, "call_{}".format(index)))
                for index in range(9)
            ]
        ),
        tool_dispatcher=dispatcher,
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run_result(QUESTION)

    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert agent.last_trace.lifecycle_events == []
