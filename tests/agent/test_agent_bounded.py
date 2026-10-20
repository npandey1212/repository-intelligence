"""M8.5 bounded-agent tests — explicit execution budget + termination.

The M8.4 loop under an explicit ``AgentExecutionConfig.max_tool_calls``
budget: the check happens BEFORE every tool execution, blocked calls
are recorded (never executed, never a fabricated ToolResult), and
termination is machine-readable. Covers the twelve required M8.5
cases: final answer before exhaustion / one tool then final / multiple
tools within budget / budget exhaustion / ``max_tool_calls=0`` /
no off-by-one / tool failure consuming budget / multiple tool calls in
one response / LLM failure without retry / final answer terminal /
trace budget validation / regression (the existing M8.1–M8.4 files,
unchanged and still passing).

Loop mechanics live in ``test_agent_loop.py``; trace serialization in
``test_agent_trace.py``; the CLI flag in ``test_agent_cli.py``.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
)
from repo_intelligence.agent.trace import TERMINATION_REASONS
from repo_intelligence.llm import LLMError, LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolError, ToolResult

QUESTION = "How does the data onboarding workflow work?"
ANSWER = "The workflow has four deterministic stages."


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
    """Records dispatches; returns queued results in order (else a
    generic success for the called tool)."""

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


def _run(fake, dispatcher, max_tool_calls):
    """Run with an explicit budget; returns (agent, outcome-or-None)."""
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=max_tool_calls),
    )
    outcome = agent.run(QUESTION)
    return agent, outcome


def _blocked_tool(*names):
    """One tool-requesting response with distinct call ids."""
    return _tool_calls_response(
        *[
            _call(name, {"query": "a"}, "call_{}".format(index + 1))
            for index, name in enumerate(names)
        ]
    )


FAILED_TOOL = ToolResult(
    success=False,
    tool="search_repository",
    error=ToolError(code="NO_MATCH", message="nothing found"),
)


# ---------------------------------------------------------------------------
# 1 — final answer before budget exhaustion
# ---------------------------------------------------------------------------


def test_final_answer_before_budget_ignores_the_budget():
    fake = FakeLLMClient(_text_response())
    dispatcher = FakeDispatcher()

    agent, outcome = _run(fake, dispatcher, max_tool_calls=5)

    assert outcome == ANSWER
    assert dispatcher.calls == []
    assert len(fake.requests) == 1
    summary = agent.last_trace.summary
    assert summary.total_tool_calls == 0
    assert summary.tool_calls_executed == 0
    assert summary.tool_calls_blocked == 0
    assert summary.termination_reason == "FINAL_ANSWER"
    assert summary.max_tool_calls == 5


# ---------------------------------------------------------------------------
# 2 — one tool call then final answer (successful completion)
# ---------------------------------------------------------------------------


def test_one_tool_call_then_final_answer_completes():
    fake = FakeLLMClient(_blocked_tool("search_repository"), _text_response())
    dispatcher = FakeDispatcher()

    agent, outcome = _run(fake, dispatcher, max_tool_calls=5)

    assert outcome == ANSWER
    assert [call.name for call in dispatcher.calls] == ["search_repository"]
    summary = agent.last_trace.summary
    assert summary.termination_reason == "FINAL_ANSWER"
    assert summary.tool_calls_executed == 1
    assert summary.tool_calls_blocked == 0


# ---------------------------------------------------------------------------
# 3 — multiple tool calls within budget (all execute)
# ---------------------------------------------------------------------------


def test_three_tool_calls_execute_within_budget():
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("read_file"),
        _blocked_tool("get_architecture_context"),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    agent, outcome = _run(fake, dispatcher, max_tool_calls=3)

    assert outcome == ANSWER
    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
        "get_architecture_context",
    ]
    summary = agent.last_trace.summary
    assert summary.termination_reason == "FINAL_ANSWER"
    assert summary.tool_calls_executed == 3
    assert summary.tool_calls_blocked == 0


# ---------------------------------------------------------------------------
# 4 — budget exhaustion (the most important case)
# ---------------------------------------------------------------------------


def test_budget_exhaustion_blocks_the_third_tool_call():
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("read_file"),
        _blocked_tool("get_architecture_context"),
    )
    dispatcher = FakeDispatcher()

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=2),
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)

    # structured, machine-detectable failure — no fabricated answer
    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert "execution budget exhausted" in str(excinfo.value)
    # tool A and B executed; tool C blocked BEFORE dispatch
    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
    ]
    assert len(fake.requests) == 3

    trace = agent.last_trace
    summary = trace.summary
    assert summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert summary.termination_reason in TERMINATION_REASONS
    assert summary.max_tool_calls == 2
    assert summary.tool_calls_executed == 2
    assert summary.tool_calls_blocked == 1
    assert summary.total_tool_calls == 3  # requested = 2 + 1 blocked
    assert summary.final_answer is None  # nothing was fabricated

    blocked_iteration = trace.iterations[-1]
    assert blocked_iteration.decision["type"] == "LIMIT_REACHED"
    assert blocked_iteration.tool_results == []  # no fake ToolResult
    assert blocked_iteration.observations == []
    assert len(blocked_iteration.blocked_tool_calls) == 1
    assert (
        blocked_iteration.blocked_tool_calls[0]["tool"]
        == "get_architecture_context"
    )


# ---------------------------------------------------------------------------
# 5 — boundary value: max_tool_calls = 0
# ---------------------------------------------------------------------------


def test_zero_budget_executes_zero_tools():
    fake = FakeLLMClient(_blocked_tool("search_repository"))
    dispatcher = FakeDispatcher()

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=0),
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)

    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert dispatcher.calls == []  # zero dispatches, ever
    assert len(fake.requests) == 1  # the blocked request ends the run
    summary = agent.last_trace.summary
    assert summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert summary.tool_calls_executed == 0
    assert summary.tool_calls_blocked == 1
    assert summary.max_tool_calls == 0


# ---------------------------------------------------------------------------
# 6 — no off-by-one: max_tool_calls = 1 executes exactly one
# ---------------------------------------------------------------------------


def test_budget_of_one_executes_exactly_one_tool():
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("read_file"),
    )
    dispatcher = FakeDispatcher()

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=1),
    )
    with pytest.raises(AgentError):
        agent.run(QUESTION)

    assert [call.name for call in dispatcher.calls] == ["search_repository"]
    assert agent.last_trace.summary.tool_calls_executed == 1
    assert agent.last_trace.summary.tool_calls_blocked == 1
    assert agent.last_trace.summary.termination_reason == (
        "EXECUTION_LIMIT_REACHED"
    )


# ---------------------------------------------------------------------------
# 7 — a tool FAILURE still consumes budget
# ---------------------------------------------------------------------------


def test_failed_tool_result_consumes_one_budget_unit():
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("search_repository"),
        _blocked_tool("read_file"),
    )
    dispatcher = FakeDispatcher(
        FAILED_TOOL,  # attempt 1 fails (consumes budget)
        ToolResult(success=True, tool="search_repository", data={}),
    )

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=2),
    )
    with pytest.raises(AgentError):
        agent.run(QUESTION)

    # two attempts consumed both units; the third never dispatched
    assert len(dispatcher.calls) == 2
    # the failure was observed by the model (M8.4 behavior preserved)
    first_follow_up = fake.requests[1]
    tool_messages = [m for m in first_follow_up.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert '"success": false' in tool_messages[0].content

    summary = agent.last_trace.summary
    assert summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert summary.tool_calls_executed == 2
    assert summary.tool_calls_blocked == 1


# ---------------------------------------------------------------------------
# 8 — multiple tool calls in one LLM response
# ---------------------------------------------------------------------------


def test_multi_tool_response_executes_until_the_budget_is_exhausted():
    # A, B, C requested at once with max_tool_calls = 2: A and B
    # execute (in returned order), C is blocked — and the run ends
    # immediately (no further LLM call with unanswered tool calls).
    fake = FakeLLMClient(
        _tool_calls_response(
            _call("search_repository", {"query": "a"}, "c1"),
            _call("read_file", {"path": "README.md"}, "c2"),
            _call("get_architecture_context", {}, "c3"),
        )
    )
    dispatcher = FakeDispatcher()

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=2),
    )
    with pytest.raises(AgentError):
        agent.run(QUESTION)

    assert [call.id for call in dispatcher.calls] == ["c1", "c2"]
    assert len(fake.requests) == 1  # blocked mid-response: no next turn

    iteration = agent.last_trace.iterations[0]
    assert len(iteration.tool_calls) == 3  # all requested
    assert len(iteration.tool_results) == 2  # two executed
    assert len(iteration.observations) == 2  # only executed ones observed
    assert len(iteration.blocked_tool_calls) == 1  # C: requested, blocked
    assert iteration.blocked_tool_calls[0]["tool_call_id"] == "c3"
    assert iteration.decision["type"] == "LIMIT_REACHED"
    summary = agent.last_trace.summary
    assert summary.tool_calls_executed == 2
    assert summary.tool_calls_blocked == 1
    assert summary.total_tool_calls == 3


# ---------------------------------------------------------------------------
# 9 — LLM failure: terminate, observable, no hidden retry
# ---------------------------------------------------------------------------


def test_llm_failure_terminates_without_retry_or_extra_execution():
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        LLMError("provider exploded"),
    )
    dispatcher = FakeDispatcher()

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=5),
    )
    with pytest.raises(LLMError, match="provider exploded"):
        agent.run(QUESTION)

    # no retry: the failing call is the last one; the executed tool
    # was not re-run either
    assert len(fake.requests) == 2
    assert len(dispatcher.calls) == 1

    trace = agent.last_trace
    assert trace.summary.termination_reason == "LLM_ERROR"
    assert trace.summary.termination_reason in TERMINATION_REASONS
    error_iteration = trace.iterations[-1]
    assert error_iteration.response_type == "ERROR"
    assert "provider exploded" in error_iteration.error


# ---------------------------------------------------------------------------
# 10 — the final answer is terminal
# ---------------------------------------------------------------------------


def test_final_answer_is_terminal():
    fake = FakeLLMClient(_text_response())
    dispatcher = FakeDispatcher()

    agent, outcome = _run(fake, dispatcher, max_tool_calls=5)

    assert outcome == ANSWER
    # no additional LLM call, no tool call, no dispatcher invocation
    assert len(fake.requests) == 1
    assert dispatcher.calls == []
    assert agent.last_trace.summary.total_iterations == 1
    assert agent.last_trace.summary.termination_reason == "FINAL_ANSWER"


# ---------------------------------------------------------------------------
# 11 — trace validation: budget vs executed vs blocked
# ---------------------------------------------------------------------------


def test_trace_distinguishes_executed_from_budget_blocked():
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("read_file"),
    )
    dispatcher = FakeDispatcher()

    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=1),
    )
    with pytest.raises(AgentError):
        agent.run(QUESTION)

    trace = agent.last_trace
    payload = json.dumps(trace.to_dict())

    # the four required summary fields
    summary = trace.summary
    assert summary.max_tool_calls == 1
    assert summary.total_tool_calls == 2
    assert summary.total_iterations == 2
    assert summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert summary.tool_calls_executed == 1
    assert summary.tool_calls_blocked == 1

    # executed tool → full ToolResult + observation
    executed = trace.iterations[0]
    assert len(executed.tool_results) == 1
    assert executed.tool_results[0]["success"] is True
    assert executed.observations[0]["type"] == "TOOL_RESULT"
    assert executed.blocked_tool_calls == []

    # blocked tool → explicit blocked entry, NO fabricated ToolResult,
    # no success flag (a skipped tool is not a failed tool)
    blocked = trace.iterations[1]
    assert blocked.tool_results == []
    assert blocked.observations == []
    assert len(blocked.blocked_tool_calls) == 1
    entry = blocked.blocked_tool_calls[0]
    assert entry["tool"] == "read_file"
    assert entry["reason"] == "EXECUTION_LIMIT_REACHED"
    assert entry["tool_calls_executed"] == 1
    assert entry["max_tool_calls"] == 1
    assert "success" not in entry

    # the budget is serialized; no hidden reasoning anywhere
    assert '"max_tool_calls": 1' in payload
    for banned in ("thought", "chain_of_thought", "internal_monologue"):
        assert banned not in payload


# ---------------------------------------------------------------------------
# Configuration contract (dependency-injected, single limit)
# ---------------------------------------------------------------------------


def test_execution_config_default_is_finite_and_safe():
    config = AgentExecutionConfig()
    assert config.max_tool_calls == 8
    assert config.max_tool_calls > 0


def test_execution_config_rejects_a_negative_budget():
    with pytest.raises(ValueError, match="must be >= 0"):
        AgentExecutionConfig(max_tool_calls=-1)
    with pytest.raises(ValueError, match="must be an integer"):
        AgentExecutionConfig(max_tool_calls=-1.5)


def test_agent_uses_the_injected_budget_and_records_it():
    # A budget different from the default (8) takes effect and is
    # recorded — the limit is the injected config, not a constant.
    fake = FakeLLMClient(_blocked_tool("search_repository"), _text_response())
    dispatcher = FakeDispatcher()
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=1),
    )
    outcome = agent.run(QUESTION)

    assert outcome == ANSWER
    assert agent.last_trace.summary.max_tool_calls == 1
    assert agent.last_trace.summary.tool_calls_executed == 1
    assert agent.last_trace.summary.tool_calls_blocked == 0


def test_default_agent_records_the_default_budget_in_the_trace():
    fake = FakeLLMClient(_text_response())
    agent = Agent(llm_client=fake, tool_dispatcher=FakeDispatcher())
    agent.run(QUESTION)

    assert agent.last_trace.summary.max_tool_calls == (
        AgentExecutionConfig().max_tool_calls
    )


# ---------------------------------------------------------------------------
# M20 — per-call max_tool_calls override (enforced, isolated, no leak)
# ---------------------------------------------------------------------------


def test_per_call_override_blocks_before_dispatch_and_does_not_leak():
    # Configured budget is 8; the per-call override is 0 — the tool
    # is blocked BEFORE the dispatcher ever sees it, the run raises
    # the established EXECUTION_LIMIT_REACHED path, and the
    # agent's configured budget is untouched afterwards.
    fake = FakeLLMClient(
        _blocked_tool("search_repository"), _text_response()
    )
    dispatcher = FakeDispatcher()
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=8),
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run_result(QUESTION, max_tool_calls=0)

    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert dispatcher.calls == []  # never reached the tool boundary
    assert agent.last_trace.summary.max_tool_calls == 0
    assert agent.last_trace.summary.tool_calls_executed == 0
    assert agent.last_trace.summary.tool_calls_blocked == 1
    # No shared-state leak: the configured budget is unchanged.
    assert agent._execution_config.max_tool_calls == 8


def test_per_call_override_allows_exactly_n_executions():
    # Override of 1: the first requested tool executes, the second is
    # blocked before dispatch (no off-by-one, same guard as the
    # configured budget).
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("search_repository"),
        _text_response(),
    )
    dispatcher = FakeDispatcher()
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=8),
    )
    with pytest.raises(AgentError):
        agent.run_result(QUESTION, max_tool_calls=1)

    assert len(dispatcher.calls) == 1
    assert agent.last_trace.summary.tool_calls_executed == 1
    assert agent.last_trace.summary.tool_calls_blocked == 1
    assert agent._execution_config.max_tool_calls == 8


def test_configured_budget_applies_again_after_a_per_call_override():
    # The override is per-call: the NEXT call without one reuses the
    # configured budget (no residue from the previous override).
    fake = FakeLLMClient(
        _blocked_tool("search_repository"),
        _blocked_tool("search_repository"),
        _text_response(),
    )
    dispatcher = FakeDispatcher()
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=8),
    )
    with pytest.raises(AgentError):
        agent.run_result(QUESTION, max_tool_calls=0)
    outcome = agent.run_result(QUESTION)

    assert outcome.answer == ANSWER
    assert len(dispatcher.calls) == 1  # only the second call's tool
    assert agent.last_trace.summary.max_tool_calls == 8


@pytest.mark.parametrize("bad", [-1, True, 1.5, "2"])
def test_per_call_override_rejects_invalid_values(bad):
    agent = Agent(llm_client=FakeLLMClient(), tool_dispatcher=FakeDispatcher())
    with pytest.raises(ValueError):
        agent.run_result(QUESTION, max_tool_calls=bad)
