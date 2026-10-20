"""M19.1 usage semantics — most-recent attempt vs cumulative totals
and honest token sums (M20 adds per-attempt budget enforcement on top:
see ``test_m20_task_budget.py``).

``TaskRecord.usage`` stays the MOST RECENT attempt's measurement;
``usage_total`` (task and run level) sums every committed attempt via
``combine_usage`` — token fields sum only reported values and stay
``None`` when no provider reported any (never fabricated ``0``).
Contract violations commit nothing, so they contribute no usage.
``AgentTaskRequest.max_tool_calls`` (M20) is the PER-ATTEMPT limit:
``None`` (the default the runner dispatches) leaves each
participant's own configured budget in force — the runner never
invents a task-level budget.
"""

from __future__ import annotations

from repo_intelligence.orchestration import (
    COMPLETED,
    FAILURE_STRUCTURED_OUTPUT,
    TASK_COMPLETED,
    TASK_FAILED,
    AgentTaskExecutor,
    AgentTaskRequest,
    Orchestrator,
    TaskUsage,
    combine_usage,
)

from m17_doubles import (
    COMMIT,
    REPOSITORY,
    build,
    call,
    report_payload,
    structured,
    task,
    tool_calls,
    two_step_plan,
)

from repo_intelligence.multiagent import InvestigationReport


# -- combine_usage: the one summation rule --------------------------------


def test_combine_usage_sums_counters_and_reported_tokens():
    a = TaskUsage(
        tool_calls=2,
        model_iterations=3,
        elapsed_seconds=1.5,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )
    b = TaskUsage(
        tool_calls=1,
        model_iterations=1,
        elapsed_seconds=0.5,
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
    )
    total = combine_usage(a, b)
    assert total.tool_calls == 3
    assert total.model_iterations == 4
    assert total.elapsed_seconds == 2.0
    assert total.input_tokens == 120
    assert total.output_tokens == 60
    assert total.total_tokens == 180


def test_combine_usage_never_fabricates_tokens_from_missing_data():
    reported = TaskUsage(tool_calls=2, input_tokens=100)
    unreported = TaskUsage(tool_calls=1)
    # partial reporting sums the reported values only
    assert combine_usage(reported, unreported).input_tokens == 100
    # no reporting at all stays None — not a fabricated 0
    neither = combine_usage(unreported, TaskUsage(model_iterations=1))
    assert neither.input_tokens is None
    assert neither.output_tokens is None
    assert neither.total_tokens is None
    assert neither.tool_calls == 1
    assert neither.model_iterations == 1
    # empty / None inputs are legal
    assert combine_usage() == TaskUsage()
    assert combine_usage(None, None) == TaskUsage()
    assert combine_usage(None, reported) == reported


# -- runner: per-task and run-level accumulation ---------------------------


def test_retry_sums_usage_while_usage_stays_most_recent_attempt():
    broken = report_payload()
    del broken["findings"]
    responses = [
        tool_calls(call("read_file", {"path": "README.md"})),
        structured(broken),
        structured(broken),
    ]
    orch, _, _ = build(inv_responses=responses)
    run = orch.execute(task(), two_step_plan())

    p1, _ = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_STRUCTURED_OUTPUT
    assert p1.attempts == 2
    # most recent attempt alone (attempt 2 ran no tools)
    assert p1.usage is not None
    assert p1.usage.tool_calls == 0
    # cumulative across both attempts (attempt 1 read a file)
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 1
    assert p1.usage_total.model_iterations == 3
    # run-level totals cover exactly the committed attempts
    assert run.tool_calls_total == 1
    assert run.usage_total == p1.usage_total


def test_run_usage_total_sums_across_tasks_on_the_happy_path():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())

    p1, p2 = run.tasks
    assert run.status == COMPLETED
    assert p1.usage is not None and p2.usage is not None
    assert p1.usage_total == p1.usage
    assert p2.usage_total == p2.usage
    assert p1.usage_total.tool_calls == 1  # the scripted read_file
    assert p2.usage_total.tool_calls == 0
    assert run.usage_total is not None
    assert run.usage_total.tool_calls == 1
    assert run.usage_total.model_iterations == (
        p1.usage_total.model_iterations + p2.usage_total.model_iterations
    )
    assert run.tool_calls_total == 1


def test_runner_leaves_the_request_budget_unset_by_default():
    orch, investigator, analyst = build()
    inner = AgentTaskExecutor.for_agents(investigator, analyst)
    seen: list = []

    class _Recorder:
        def execute(self, request):
            seen.append(request)
            return inner.execute(request)

    recorder = Orchestrator(investigator, analyst, executor=_Recorder())
    run = recorder.execute(task(), two_step_plan())

    assert run.status == COMPLETED
    assert seen, "no requests were dispatched"
    # With the default OrchestrationConfig (max_task_tool_calls=None)
    # the runner dispatches no task-level override — each
    # participant's own configured budget applies (M19 behavior).
    assert all(
        request.max_tool_calls is None for request in seen
    )  # the runner never INVENTS a task-level budget


def test_request_max_tool_calls_none_keeps_the_agent_configured_budget(
):
    _, investigator, analyst = build()
    request = AgentTaskRequest(
        run_id="r1",
        task_id="P1",
        attempt=1,
        role="INVESTIGATOR",
        assignment_reason="usage semantics check",
        question="how does it work?",
        goal="Identify the boundary",
        prompt="Investigate README.md and report",
        repository_scope=REPOSITORY,
        commit_scope=COMMIT,
        # None (default): no task-level override — the scripted
        # read_file executes under the participant's own configured
        # budget (AgentExecutionConfig default of 8).
    )
    executor = AgentTaskExecutor.for_agents(investigator, analyst)
    result = executor.execute(request)

    assert request.max_tool_calls is None
    assert result.status == TASK_COMPLETED
    assert isinstance(result.output, InvestigationReport)
    assert result.usage.tool_calls == 1
