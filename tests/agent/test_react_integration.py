"""Integration of the shared ReAct safety controls with the REAL
application execution path: ``agent.py::_execute_loop`` — the one
production ReAct loop every specialist, the router, and the Analyst
run through (via ``Agent.run_result`` / ``Agent.run``), including the
LangGraph task path (the graph executes tasks through the same
``AgentTaskExecutor`` boundary).

Two seams are pinned here, with fake clients and fake tools only:

* observations are bounded by ``react_safety.bound_observation``
  BEFORE they join the working conversation, so the NEXT outgoing
  model request carries a bounded observation — identity for
  tool-protocol-compliant results (the tool layer already caps the
  serialized result), a line-aligned excerpt + continuation note for
  anything else; the TRACE's evidence copy is untouched;
* EVERY outgoing request — the initial one and each repeated request
  over the grown history — carries its own independent H1/H2 budget
  record (the loop re-runs the gate per iteration).

The oversized-complete-request rejection and the tool-call exhaustion
terminal already have real-path coverage in
``tests/agent/test_request_budget.py``
(``test_rejected_request_is_never_sent_and_fails_typed``) and
``tests/agent/test_agent_bounded.py`` /
``test_agent_trace.py::test_execution_budget_records_execution_limit_
reached_termination``; the graph-workflow coverage is
``test_repository_query_langgraph.py`` (including its workflow-level
budget-record test). No duplicate versions of those are written here.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import Agent, AgentError, AgentExecutionConfig
from repo_intelligence.agent.request_budget import DISPOSITION_ACCEPTED
from repo_intelligence.llm.models import (
    LLMResponse,
    Message,
    ToolCall,
)
from repo_intelligence.react_safety import (
    MAX_TOOL_RESPONSE_CHARS,
)
from repo_intelligence.tools import ToolResult

QUESTION = "What evidence backs this architecture?"
MARKER_NOTE = "[truncated: showing first"


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
        if not self._queue:
            return ToolResult(success=True, tool=tool_call.name, data={})
        return self._queue.pop(0)


def _call(call_id, name="read_file", arguments=None):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=dict(arguments or {"path": "src/a.py"}),
    )


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


def _tool_messages(request):
    return [m for m in request.messages if m.role == "tool"]


def test_oversized_observation_is_bounded_before_the_next_request():
    """The real loop bounds an observation BEFORE it joins history:
    the next outgoing request carries ≤ 24,000 chars with the
    truncation marker, while the trace keeps the evidence fields."""
    oversized = ToolResult(
        success=True,
        tool="read_file",
        data={"path": "src/big.py", "content": "x" * 50_000},
        source={"artifact": "src/big.py"},
    )
    fake = FakeLLMClient(
        _tool_calls(_call("c1")),
        _text("answered from what the tool returned"),
    )
    agent = Agent(fake, tool_dispatcher=FakeDispatcher(oversized))
    answer = agent.run(QUESTION)

    assert answer == "answered from what the tool returned"
    # exactly two requests: the initial one, and the one AFTER the
    # observation joined history
    assert len(fake.requests) == 2
    initial_tools = _tool_messages(fake.requests[0])
    assert initial_tools == []
    follow_up_tools = _tool_messages(fake.requests[1])
    assert len(follow_up_tools) == 1
    content = follow_up_tools[0].content or ""
    assert len(content) <= MAX_TOOL_RESPONSE_CHARS
    assert MARKER_NOTE in content
    assert "re-read with a targeted query" in content
    assert len(content) < 50_000 + 100  # genuinely bounded
    # the bounded observation keeps the request well inside the
    # budget — the follow-up iteration's preflight accepted it
    trace = agent.last_trace
    assert len(trace.iterations) == 2
    follow_up_budget = trace.iterations[1].request_budget
    assert follow_up_budget is not None
    assert follow_up_budget["disposition"] == DISPOSITION_ACCEPTED
    assert follow_up_budget["within_budget"] is True
    # evidence preservation: the TRACE's own copy of the executed
    # result keeps success + source provenance intact (its data is
    # separately trace-capped — established TRACE_DATA_CHAR_LIMIT
    # behavior, not the conversation bound)
    recorded = trace.iterations[0].tool_results
    assert len(recorded) == 1
    assert recorded[0]["success"] is True
    assert recorded[0]["source"] == {"artifact": "src/big.py"}
    assert recorded[0]["data"]["path"] == "src/big.py"
    assert recorded[0]["data"]["content"].startswith("x")


def test_every_repeated_request_carries_an_independent_record():
    """Budget checks are per-request, not one-shot: three outgoing
    requests over a growing history produce three independent H1/H2
    records, each measured against the request actually sent."""
    fake = FakeLLMClient(
        _tool_calls(_call("c1")),
        _tool_calls(_call("c2")),
        _text("done after two observations"),
    )
    dispatcher = FakeDispatcher(
        ToolResult(
            success=True,
            tool="read_file",
            data={"path": "src/a.py", "content": "first evidence"},
            source={"artifact": "src/a.py"},
        ),
        ToolResult(
            success=True,
            tool="search_repository",
            data={"hits": ["second evidence"]},
        ),
    )
    agent = Agent(fake, tool_dispatcher=dispatcher)
    agent.run(QUESTION)

    assert len(fake.requests) == 3
    trace = agent.last_trace
    assert len(trace.iterations) == 3
    counts = []
    for iteration in trace.iterations:
        budget = iteration.request_budget
        assert budget is not None, "every model call is gated"
        assert budget["disposition"] == DISPOSITION_ACCEPTED
        assert budget["within_budget"] is True
        assert budget["run_id"] == trace.run_id
        assert budget["budget"]["effective_input_budget_tokens"] == 95_232
        counts.append(budget["message_count"])
    # independently measured over the GROWN history each time
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_compliant_observation_passes_the_seam_untouched():
    """Identity guarantee: a tool-protocol-compliant result (≤ the
    ceiling) reaches the conversation byte-identical — the seam
    changes nothing for existing well-behaved tools."""
    compliant = ToolResult(
        success=True,
        tool="read_file",
        data={"path": "src/a.py", "content": "line1\nline2"},
        source={"artifact": "src/a.py"},
    )
    expected = json.dumps(compliant.to_dict())
    assert len(expected) <= MAX_TOOL_RESPONSE_CHARS
    fake = FakeLLMClient(
        _tool_calls(_call("c1")),
        _text("done"),
    )
    agent = Agent(fake, tool_dispatcher=FakeDispatcher(compliant))
    agent.run(QUESTION)

    tools = _tool_messages(fake.requests[1])
    assert len(tools) == 1
    assert tools[0].content == expected, "byte-identical pass-through"
    assert MARKER_NOTE not in (tools[0].content or "")


def test_tool_call_exhaustion_still_terminates_after_seam_change():
    """The existing terminal for execution-limit exhaustion is
    unchanged by the observation integration: the run stops with the
    established EXECUTION_LIMIT_REACHED policy, never silently."""
    responses = [_tool_calls(_call("c1")), _tool_calls(_call("c2"))]
    client = FakeLLMClient(*responses)
    agent = Agent(
        client,
        tool_dispatcher=FakeDispatcher(),
        execution_config=AgentExecutionConfig(max_tool_calls=1),
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)
    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    trace = agent.last_trace
    assert trace.summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    # the one executed observation still went through the seam and
    # reached the follow-up request before the budget blocked it
    follow_up_tools = _tool_messages(client.requests[1])
    assert len(follow_up_tools) == 1
    assert len(follow_up_tools[0].content or "") <= (
        MAX_TOOL_RESPONSE_CHARS
    )
