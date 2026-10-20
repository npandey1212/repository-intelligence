"""M20 task budgets — ``AgentTaskRequest.max_tool_calls`` enforced at
the agent's pre-tool guard, per-attempt semantics, and the honest
run-level between-attempt gate.

Enforcement chain (one guard, no second dispatch loop):

    OrchestrationConfig.max_task_tool_calls
      → AgentTaskRequest.max_tool_calls (per attempt)
      → AgentTaskExecutor passes it to run_result(...)
      → Agent._execute_loop's pre-dispatch check blocks BEFORE the
        tool executes (a blocked call is recorded, never counted)

Semantics pinned here: the task limit is PER ATTEMPT (a retry gets a
fresh allowance; cumulative usage still sums every attempt); ``0``
forbids every tool; invalid values fail at construction; the shared
participant's configured budget is never mutated; blocked calls are
never counted as executed; and the RUN-level tool budget stops new
attempts between attempts — it never preempts a running one.
"""

from __future__ import annotations

import pytest

from repo_intelligence.agent import Agent, AgentExecutionConfig
from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.orchestration import (
    BUDGET_EXCEEDED,
    COMPLETED,
    FAILURE_AGENT_BUDGET,
    TASK_BLOCKED,
    TASK_FAILED,
    AgentTaskExecutor,
    AgentTaskRequest,
    OrchestrationConfig,
    Orchestrator,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.tools import ToolResult

from m17_doubles import (
    COMMIT,
    REPOSITORY,
    FakeLLMClient,
    call,
    investigator_bindings,
    report_payload,
    structured,
    task,
    tool_calls,
    two_step_plan,
    build,
)


def _counting_bindings(counter):
    base = investigator_bindings()

    def wrap(fn):
        def inner(arguments):
            counter["executed"] += 1
            return fn(arguments)

        return inner

    return {name: wrap(fn) for name, fn in base.items()}


def _agent(responses, counter):
    return Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=_counting_bindings(counter)
        ),
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
        # Configured budget deliberately LARGER than any per-call
        # override under test — so any blocking proves the REQUEST
        # limit, not the agent config.
        execution_config=AgentExecutionConfig(max_tool_calls=8),
    )


def _request(**overrides) -> AgentTaskRequest:
    data = {
        "run_id": "r1",
        "task_id": "P1",
        "attempt": 1,
        "role": "INVESTIGATOR",
        "assignment_reason": "M20 task budget check",
        "question": "how does it work?",
        "goal": "Identify the boundary",
        "prompt": "Investigate README.md and report",
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }
    data.update(overrides)
    return AgentTaskRequest(**data)


# -- the request limit reaches the real pre-tool guard ----------------------


def test_request_limit_blocks_the_tool_before_it_executes():
    counter = {"executed": 0}
    agent = _agent(
        [tool_calls(call("read_file", {"path": "README.md"})),
         structured(report_payload())],
        counter,
    )
    executor = AgentTaskExecutor.for_agents(agent, agent)

    result = executor.execute(_request(max_tool_calls=0))

    assert result.status == TASK_FAILED
    assert result.failure is not None
    assert result.failure.category == FAILURE_AGENT_BUDGET
    assert result.failure.termination_reason == "EXECUTION_LIMIT_REACHED"
    # Blocked call: never dispatched, never counted.
    assert counter["executed"] == 0
    assert result.usage.tool_calls == 0
    assert agent.last_trace.summary.tool_calls_blocked == 1
    # The request limit reached the trace, not the agent's config.
    assert agent.last_trace.summary.max_tool_calls == 0
    assert agent._execution_config.max_tool_calls == 8


def test_request_limit_allows_exactly_n_executions():
    counter = {"executed": 0}
    agent = _agent(
        [tool_calls(call("read_file", {"path": "README.md"})),
         tool_calls(call("read_file", {"path": "README.md"})),
         structured(report_payload())],
        counter,
    )
    executor = AgentTaskExecutor.for_agents(agent, agent)

    result = executor.execute(_request(max_tool_calls=1))

    assert result.status == TASK_FAILED
    assert result.failure.category == FAILURE_AGENT_BUDGET
    assert counter["executed"] == 1  # first tool ran, second blocked
    assert result.usage.tool_calls == 1


def test_request_without_a_limit_keeps_the_configured_budget():
    counter = {"executed": 0}
    agent = _agent(
        [tool_calls(call("read_file", {"path": "README.md"})),
         structured(report_payload())],
        counter,
    )
    executor = AgentTaskExecutor.for_agents(agent, agent)

    result = executor.execute(_request())  # max_tool_calls=None

    assert result.status == "completed"
    assert counter["executed"] == 1
    assert result.usage.tool_calls == 1
    assert agent.last_trace.summary.max_tool_calls == 8


def test_executor_passes_the_override_kwarg_only_when_set():
    class _Recording:
        def __init__(self):
            self.calls = []
            self.last_trace = None

        def run_result(self, question, **kwargs):
            self.calls.append((question, kwargs))
            return type("R", (), {"structured_output": None})()

    participant = _Recording()
    executor = AgentTaskExecutor.for_agents(participant, participant)

    executor.execute(_request())  # None → exact M19 one-arg call
    executor.execute(_request(max_tool_calls=3))

    assert participant.calls == [
        ("Investigate README.md and report", {}),
        ("Investigate README.md and report", {"max_tool_calls": 3}),
    ]


# -- zero and invalid values -----------------------------------------------


def test_zero_forbids_every_tool_through_the_orchestrator():
    orch, investigator, _ = build(
        inv_responses=[
            tool_calls(call("read_file", {"path": "README.md"})),
            structured(report_payload()),
        ],
        config=OrchestrationConfig(max_task_tool_calls=0),
    )
    run = orch.execute(task(), two_step_plan())

    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_AGENT_BUDGET
    assert p1.usage is not None and p1.usage.tool_calls == 0
    assert p1.usage_total is not None and p1.usage_total.tool_calls == 0
    assert run.tool_calls_total == 0
    assert p2.status == TASK_BLOCKED
    assert investigator.last_trace.summary.tool_calls_blocked == 1
    budget = [
        event
        for event in run.events
        if event["operation"] == "orchestration.budget.exceeded"
    ]
    assert budget and budget[0]["budget"] == "agent_max_tool_calls"
    assert budget[0]["limit"] == 0
    assert budget[0]["observed"] == 0


def test_invalid_request_and_config_limits_fail_early():
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            _request(max_tool_calls=bad)
    for bad in (-1, True):
        with pytest.raises(ValueError):
            OrchestrationConfig(max_task_tool_calls=bad)
    # Valid values construct fine.
    assert _request(max_tool_calls=0).max_tool_calls == 0
    assert (
        OrchestrationConfig(max_task_tool_calls=4).max_task_tool_calls
        == 4
    )


# -- per-attempt semantics --------------------------------------------------


def test_retry_gets_a_fresh_allowance_but_usage_sums_both_attempts():
    # Attempt 1: one tool runs, then a retryable structured failure.
    # Attempt 2: its OWN allowance lets one more tool run, then the
    # report validates. Cumulative task/run usage sums both attempts.
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", {"path": "README.md"})),
            structured({"findings": []}),  # broken → retryable
            tool_calls(call("read_file", {"path": "README.md"})),
            structured(report_payload()),
        ],
        config=OrchestrationConfig(max_task_tool_calls=1),
    )
    run = orch.execute(task(), two_step_plan())

    assert run.status == COMPLETED
    p1, _ = run.tasks
    assert p1.attempts == 2
    assert run.retries == 1
    assert p1.usage is not None
    assert p1.usage.tool_calls == 1  # most recent attempt alone
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 2  # both attempts
    assert run.tool_calls_total == 2


def test_task_budget_does_not_leak_into_the_shared_agents():
    orch, investigator, analyst = build(
        config=OrchestrationConfig(max_task_tool_calls=1),
    )
    run = orch.execute(task(), two_step_plan())

    assert run.status == COMPLETED
    # The participants are REUSED across both tasks and every attempt
    # — their configured budgets must be untouched.
    assert investigator._execution_config.max_tool_calls == (
        AgentExecutionConfig().max_tool_calls
    )
    assert analyst._execution_config.max_tool_calls == (
        AgentExecutionConfig().max_tool_calls
    )
    assert investigator.last_trace.summary.max_tool_calls == 1
    assert analyst.last_trace.summary.max_tool_calls == 1


# -- run-level budget: between attempts, counted once ----------------------


def test_run_tool_budget_blocks_a_retry_attempt():
    # Attempt 1 executes one tool (the whole run budget) and fails
    # retryably. The M20 per-attempt gate must refuse to START the
    # retry: no new attempt when no run budget remains. The measured
    # tool call from the failed attempt is committed exactly once.
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", {"path": "README.md"})),
            structured({"findings": []}),  # broken → retryable
        ],
        config=OrchestrationConfig(max_total_tool_calls=1),
    )
    run = orch.execute(task(), two_step_plan())

    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    assert run.retries == 0  # the retry never started
    assert "orchestration.retry" not in run.event_operations()
    assert run.tool_calls_total == 1  # counted once, never twice
    p1, p2 = run.tasks
    assert p1.status == TASK_BLOCKED
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 1
    assert p2.status == TASK_BLOCKED
    budget = [
        event
        for event in run.events
        if event["operation"] == "orchestration.budget.exceeded"
    ]
    assert budget and budget[0]["budget"] == "max_total_tool_calls"
    assert budget[0]["limit"] == 1
    assert budget[0]["observed"] == 1
