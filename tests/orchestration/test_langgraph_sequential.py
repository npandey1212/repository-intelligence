"""LangGraph sequential orchestration — behavioral proofs (M21 milestone).

The graph in ``orchestration/langgraph_runner.py`` owns TASK-LEVEL
control flow (selection, attempts, retries, budget gates, blocking,
termination, assembly) over the unchanged M19/M20 contracts. These
tests assert BEHAVIOR — order, records, budgets, validation outcomes,
captured executor requests — not node names or internal wiring.

Deterministic throughout: scripted ``FakeLLMClient`` participants or
replayed boundary results; no network.

``Orchestrator.execute(...)`` must NEVER be reached from the graph
task path: ``test_graph_path_never_calls_monolithic_execute`` fails
if the graph delegates the complete run to the monolithic
coordinator.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

langgraph_graph = pytest.importorskip(
    "langgraph.graph", reason="langgraph optional extra not installed"
)

from repo_intelligence.agent import Agent
from repo_intelligence.llm import LLMError
from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import (
    BUDGET_EXCEEDED,
    COMPLETED,
    FAILURE_CONTEXT_EXHAUSTED,
    FAILURE_CONTRACT_VIOLATION,
    FAILURE_STRUCTURED_OUTPUT,
    INCOMPLETE,
    AGENT_FAILED,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_PENDING,
    TASK_RUNNING,
    ANALYST,
    INVESTIGATOR,
    AgentTaskRequest,
    AgentTaskResult,
    OrchestrationConfig,
    Orchestrator,
    RoleAssignment,
    TaskFailure,
    TaskUsage,
)
from repo_intelligence.orchestration import langgraph_runner as lg
from repo_intelligence.orchestration.langgraph_runner import (
    build_langgraph_orchestration_graph,
    run_with_langgraph,
)
from repo_intelligence.orchestration.scheduling import ready_tasks
from repo_intelligence.planning.models import InvestigationPlan, PlanStep
from repo_intelligence.repository_query import QueryRouter

from m17_doubles import (
    COMMIT,
    README_REF,
    REPOSITORY,
    FakeLLMClient,
    build,
    call,
    finding_payload,
    investigate_read_then_report,
    investigator_bindings,
    report_payload,
    structured,
    task,
    tool_calls,
    two_step_plan,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ops(run):
    return [event["operation"] for event in run.events]


def _events(run, operation):
    return [event for event in run.events if event["operation"] == operation]


def _roles(*pairs):
    return [
        RoleAssignment(step_id=step_id, role=role, reason="graph test")
        for step_id, role in pairs
    ]


def _plan(*steps):
    """steps: (id, goal) tuples."""
    return two_step_plan(
        steps=[PlanStep(id=step_id, goal=goal) for step_id, goal in steps]
    )


class _Scripted:
    """Fake boundary: records every request, replays scripted results.

    The run id is per-run (unknowable when scripting), so each result
    echoes ``request.run_id``; scripted task_id/attempt/role must
    match the request or the contract validator rejects the result —
    exactly the production rule.
    """

    def __init__(self, *results):
        self.requests = []
        self.results = list(results)

    def execute(self, request: AgentTaskRequest) -> AgentTaskResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("executor invoked more times than scripted")
        return replace(self.results.pop(0), run_id=request.run_id)


class _Recording:
    """Wraps the REAL executor: captures requests/results in dispatch
    order plus a request→result timeline (proves sequential dispatch)."""

    def __init__(self, inner):
        self._inner = inner
        self.requests = []
        self.results = []
        self.timeline = []

    def execute(self, request):
        self.requests.append(request)
        self.timeline.append(("request", request.task_id, request.attempt))
        result = self._inner.execute(request)
        self.results.append(result)
        self.timeline.append(("result", request.task_id, request.attempt))
        return result


def _ok(task_id, role, attempt, output, *, usage=None, agent_run_id=None):
    return AgentTaskResult(
        run_id="scripted-run",
        task_id=task_id,
        attempt=attempt,
        role=role,
        status=TASK_COMPLETED,
        output=output,
        evidence_refs=tuple(
            getattr(output, "evidence_refs", ()) or ()
        ),
        usage=usage if usage is not None else TaskUsage(),
        agent_run_id=agent_run_id,
    )


def _failed(
    task_id,
    role,
    attempt,
    *,
    category,
    error_type,
    error_message,
    retryable,
    usage=None,
):
    return AgentTaskResult(
        run_id="scripted-run",
        task_id=task_id,
        attempt=attempt,
        role=role,
        status=TASK_FAILED,
        output=None,
        failure=TaskFailure(
            category=category,
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
        ),
        usage=usage if usage is not None else TaskUsage(),
    )


def _report_typed():
    """A TYPED boundary output (dicts are contract violations)."""
    return InvestigationReport.model_validate(report_payload())


def _graph_runtime(*, inv_responses=None, ana_responses=None, config=None,
                   executor=None, **build_kwargs):
    orch, investigator, analyst = build(
        inv_responses=inv_responses,
        ana_responses=ana_responses,
        config=config,
        **build_kwargs,
    )
    if executor is not None:
        orch = Orchestrator(
            investigator,
            analyst,
            config=config,
            executor=executor,
        )
    return orch, investigator, analyst


def _report(investigation_id, statement):
    return report_payload(
        investigation_id=investigation_id,
        findings=[
            {
                "statement": statement,
                "evidence_refs": [dict(README_REF)],
                "support": "SUPPORTED",
            }
        ],
    )


# ---------------------------------------------------------------------------
# 1/6/21 — order, roles, terminal state (real execution)
# ---------------------------------------------------------------------------


def test_two_investigators_then_analyst_execute_in_plan_order():
    orch, _, _ = build(
        inv_responses=investigate_read_then_report()
        + investigate_read_then_report(),
        ana_responses=[structured(finding_payload())],
    )
    plan = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    starts = [
        event["step_id"]
        for event in run.events
        if event["operation"] == "orchestration.step.start"
    ]
    assert starts == ["P1", "P2", "P3"]
    assert [item["step_id"] for item in run.agent_runs] == [
        "P1",
        "P2",
        "P3",
    ]
    assert [record.task_id for record in run.tasks] == ["P1", "P2", "P3"]
    assert all(
        record.status == TASK_COMPLETED for record in run.tasks
    )
    assert run.completed_steps == ["P1", "P2", "P3"]
    assert run.validation_status == "passed"


def test_preassigned_roles_are_respected_verbatim():
    orch, _, _ = build()
    plan = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
    )
    roles = [
        RoleAssignment("P1", INVESTIGATOR, "custom-reason-inv"),
        RoleAssignment("P2", ANALYST, "custom-reason-ana"),
    ]
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    assert [
        (record.task_id, record.role, record.assignment_reason)
        for record in run.tasks
    ] == [
        ("P1", INVESTIGATOR, "custom-reason-inv"),
        ("P2", ANALYST, "custom-reason-ana"),
    ]
    assigns = _events(run, "orchestration.step.assign")
    assert [event["assignment_reason"] for event in assigns] == [
        "custom-reason-inv",
        "custom-reason-ana",
    ]
    assert [event["agent_role"] for event in assigns] == [
        INVESTIGATOR,
        ANALYST,
    ]


def test_final_status_termination_and_records_follow_rules():
    orch, _, _ = build()
    run = run_with_langgraph(orch, task(), two_step_plan(), _roles(
        ("P1", INVESTIGATOR), ("P2", ANALYST),
    ))

    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert run.final is True
    assert run.plan.status == "completed"
    assert run.current_step is None
    assert run.completed_at != ""
    assert run.validation_status == "passed"
    assert run.validation is not None
    assert _ops(run)[-1] == "orchestration.complete"
    for record in run.tasks:
        assert record.status == TASK_COMPLETED
        assert record.completed_at != ""
        assert record.failure_category is None
        assert record.error is None


# ---------------------------------------------------------------------------
# 2 — one topology, dynamic plan sizes
# ---------------------------------------------------------------------------


def test_same_compiled_graph_handles_different_plan_sizes():
    orch, _, _ = build(
        inv_responses=(
            # invoke 1 runs P1 once; invoke 2 runs P1..P3 once each.
            investigate_read_then_report()
            + investigate_read_then_report()
            + investigate_read_then_report()
            + investigate_read_then_report()
        ),
        ana_responses=[
            structured(finding_payload()),
            structured(finding_payload()),
        ],
    )
    # ONE compiled graph, no rebuild between invocations.
    graph = build_langgraph_orchestration_graph(orch)

    small = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
    )
    small_roles = _roles(("P1", INVESTIGATOR), ("P2", ANALYST))
    first = graph.invoke({
        "task": task(),
        "plan": small,
        "role_assignments": small_roles,
    })

    large = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
        ("P3", "Locate the dispatcher responsibilities"),
        ("P4", "Analyze the merged evidence"),
    )
    large_roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", INVESTIGATOR),
        ("P4", ANALYST),
    )
    second = graph.invoke({
        "task": task(),
        "plan": large,
        "role_assignments": large_roles,
    })

    assert first["run"].status == COMPLETED
    assert second["run"].status == COMPLETED
    assert len(first["run"].completed_steps) == 2
    assert len(second["run"].completed_steps) == 4
    assert len(second["run"].tasks) == 4
    # Same compiled instance ran both — topology is not per-plan.
    assert first["run"].run_id != second["run"].run_id


# ---------------------------------------------------------------------------
# 3/4 — QueryRouter dispatch is real
# ---------------------------------------------------------------------------


def _specialist(responses):
    return Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=investigator_bindings()
        ),
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
    )


def _router_runtime(router, config=None):
    _, _, analyst = build()
    return Orchestrator(router, analyst, config=config)


def test_router_repeats_one_specialist_for_multiple_tasks():
    architect = _specialist(
        investigate_read_then_report()
        + investigate_read_then_report()
    )
    flow = _specialist([])  # must never run
    router = QueryRouter(architecture=architect, flow=flow)
    orch = _router_runtime(router)

    plan = _plan(
        ("P1", "Identify the architecture components"),
        ("P2", "Explain the architecture relationships"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    # Both investigator steps routed to (and executed on) the SAME
    # specialist; the other specialist's scripted client is untouched.
    assert router.calls == ["P1", "P2"]
    assert router.labels["P1"] == "architecture"
    assert router.labels["P2"] == "architecture"
    assert architect._llm_client.requests  # consumed
    assert len(architect._llm_client.requests) == 4  # 2 runs × 2 calls
    assert flow._llm_client.requests == []


def test_router_selected_specialist_is_the_one_invoked():
    architect = _specialist(investigate_read_then_report())
    flow = _specialist(investigate_read_then_report())
    router = QueryRouter(architecture=architect, flow=flow)
    orch = _router_runtime(router)

    plan = _plan(
        ("P1", "Identify the architecture components"),
        ("P2", "Trace the dependency paths"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    assert router.labels["P1"] == "architecture"
    assert router.labels["P2"] == "flow"
    assert router.calls == ["P1", "P2"]
    # The routed specialist is the one that consumed responses.
    assert len(architect._llm_client.requests) == 2
    assert len(flow._llm_client.requests) == 2
    # And its execution produced the committed evidence.
    by_step = {r.step_id: r for r in run.step_results}
    assert by_step["P1"].agent_run_id == architect.last_trace.run_id
    assert by_step["P2"].agent_run_id == flow.last_trace.run_id


# ---------------------------------------------------------------------------
# 5 — the analyst consumes the assembled investigator output
# ---------------------------------------------------------------------------


def test_analyst_receives_the_assembled_investigator_output():
    base, investigator, analyst = build(
        inv_responses=investigate_read_then_report()
        + investigate_read_then_report(),
        ana_responses=[structured(finding_payload())],
    )
    recorder = _Recording(base.executor)
    orch = Orchestrator(investigator, analyst, executor=recorder)
    plan = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    analyst_requests = [
        request
        for request in recorder.requests
        if request.role == ANALYST
    ]
    assert len(analyst_requests) == 1
    request = analyst_requests[0]
    # Dependency payload: the merged report of BOTH investigators.
    assert len(request.dependency_results) == 1
    merged = request.dependency_results[0]
    assert isinstance(merged, InvestigationReport)
    assert merged.investigation_id == "inv-1"  # first report keeps id
    assert len(merged.findings) == 2  # plan-order concatenation
    # The typed handoff the analyst received IS the merged report.
    handoffs = [
        message
        for message in run.messages
        if message.message_type == "INVESTIGATION_RESULT"
    ]
    assert len(handoffs) == 1
    assert handoffs[0].payload is run.report
    # And the prompt carries the merged report (existing runner helper).
    assert "Analyze the Investigator's typed report" in request.prompt
    assert run.report.investigation_id == "inv-1"


# ---------------------------------------------------------------------------
# 7 — invalid role input follows existing validation (nothing runs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "roles,expected",
    [
        (
            _roles(("P1", INVESTIGATOR)),
            "absent from the plan|P2",
        ),
        (
            _roles(("P1", "WIZARD"), ("P2", ANALYST)),
            "unsupported role",
        ),
        (
            [
                RoleAssignment("P1", INVESTIGATOR, "dup"),
                RoleAssignment("P1", INVESTIGATOR, "dup"),
                RoleAssignment("P2", ANALYST, "dup"),
            ],
            "duplicate step id",
        ),
        (
            _roles(("P1", ANALYST), ("P2", INVESTIGATOR)),
            "INVESTIGATOR",
        ),
        (
            ["P1", "P2"],
            "must be a RoleAssignment",
        ),
    ],
)
def test_invalid_role_assignments_raise_before_any_execution(
    roles, expected
):
    fake = _Scripted()  # any dispatch raises AssertionError
    orch, _, _ = _graph_runtime(executor=fake)
    plan = two_step_plan()
    with pytest.raises(ValueError, match=expected):
        run_with_langgraph(orch, task(), plan, roles)
    assert fake.requests == []


# ---------------------------------------------------------------------------
# 8 — dependencies: chain semantics and readiness
# ---------------------------------------------------------------------------


def test_dependencies_prevent_premature_execution():
    base, investigator, analyst = build(
        inv_responses=investigate_read_then_report()
        + investigate_read_then_report(),
        ana_responses=[structured(finding_payload())],
    )
    recorder = _Recording(base.executor)
    orch = Orchestrator(investigator, analyst, executor=recorder)
    plan = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    # Sequential chain dependencies on the records.
    assert [record.depends_on for record in run.tasks] == [
        [],
        ["P1"],
        ["P2"],
    ]
    # Dispatch timeline: strictly request→result pairs in plan order —
    # no second task starts while the first is still executing, and
    # every task's request follows its predecessor's RESULT.
    kinds = [entry[0] for entry in recorder.timeline]
    assert kinds == [
        "request", "result",
        "request", "result",
        "request", "result",
    ]
    assert [entry[1] for entry in recorder.timeline] == [
        "P1", "P1", "P2", "P2", "P3", "P3",
    ]
    # The readiness helper agrees: before P1 completes, only P1.
    from repo_intelligence.orchestration.models import TaskRecord

    pending = [
        TaskRecord(task_id="P1", role=INVESTIGATOR, status=TASK_PENDING),
        TaskRecord(
            task_id="P2",
            role=INVESTIGATOR,
            status=TASK_PENDING,
            depends_on=["P1"],
        ),
    ]
    assert [item.task_id for item in ready_tasks(pending)] == ["P1"]
    pending[0].status = TASK_COMPLETED
    assert [item.task_id for item in ready_tasks(pending)] == ["P2"]


def test_failed_prerequisite_blocks_dependent_tasks():
    fake = _Scripted(
        _failed(
            "P1",
            INVESTIGATOR,
            1,
            category=FAILURE_CONTEXT_EXHAUSTED,
            error_type="ContextExhausted",
            error_message="ContextExhausted: context window gone",
            retryable=False,
        ),
    )
    orch, _, _ = _graph_runtime(executor=fake)
    plan = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    # Only P1 was ever dispatched.
    assert [request.task_id for request in fake.requests] == ["P1"]
    records = {record.task_id: record for record in run.tasks}
    assert records["P1"].status == TASK_FAILED
    assert records["P2"].status == TASK_BLOCKED
    assert records["P3"].status == TASK_BLOCKED
    assert run.blocked_steps == ["P2", "P3"]
    assert run.failed_steps == ["P1"]
    # Blocked step results carry the existing reason vocabulary.
    blocked = [
        result
        for result in run.step_results
        if result.status == "blocked"
    ]
    assert [result.step_id for result in blocked] == ["P2", "P3"]
    assert all(result.error == "P1_FAILED" for result in blocked)


# ---------------------------------------------------------------------------
# 10/12/17 — retries, exhaustion, cumulative usage
# ---------------------------------------------------------------------------


def test_retryable_failure_retries_within_the_limit_with_cumulative_usage():
    # Established M17 pattern: attempt 1 reads evidence, then returns a
    # BROKEN structured response (retryable); attempt 2 succeeds.
    broken = report_payload()
    del broken["findings"]
    base, investigator, analyst = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            structured(broken),
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ],
        config=OrchestrationConfig(max_retries=1),
    )
    recorder = _Recording(base.executor)
    orch = Orchestrator(
        investigator,
        analyst,
        executor=recorder,
        config=OrchestrationConfig(max_retries=1),
    )
    plan = two_step_plan()
    roles = _roles(("P1", INVESTIGATOR), ("P2", ANALYST))
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == COMPLETED
    assert run.retries == 1
    p1 = run.tasks[0]
    assert p1.status == TASK_COMPLETED
    assert p1.attempts == 2
    # Recent attempt vs cumulative (M19/M20 semantics): each attempt
    # executed exactly ONE tool call — recent 1, cumulative 2.
    assert p1.usage.tool_calls == 1
    assert p1.usage_total.tool_calls == 2
    assert run.usage_total.tool_calls == 2
    assert run.tool_calls_total == 2
    assert [item["attempt"] for item in run.agent_runs] == [1, 2, 1]
    retries = _events(run, "orchestration.retry")
    assert len(retries) == 1
    assert retries[0]["attempt"] == 2
    # Distinct attempt identity on the boundary, same run.
    assert recorder.requests[0].attempt == 1
    assert recorder.requests[1].attempt == 2
    assert recorder.requests[0].run_id == recorder.requests[1].run_id


def test_non_retryable_failure_receives_no_retry():
    fake = _Scripted(
        _failed(
            "P1",
            INVESTIGATOR,
            1,
            category=FAILURE_CONTEXT_EXHAUSTED,
            error_type="ContextExhausted",
            error_message="ContextExhausted: gone",
            retryable=False,
            usage=TaskUsage(tool_calls=1),
        ),
    )
    orch, _, _ = _graph_runtime(
        executor=fake, config=OrchestrationConfig(max_retries=3)
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    assert run.retries == 0
    assert _events(run, "orchestration.retry") == []
    assert len(fake.requests) == 1
    p1 = run.tasks[0]
    assert p1.status == TASK_FAILED
    assert p1.attempts == 1
    assert p1.failure_category == FAILURE_CONTEXT_EXHAUSTED
    # Runner-parity events for a terminal failure.
    assert "orchestration.step.failed" in _ops(run)
    completes = _events(run, "orchestration.agent.complete")
    assert completes[0]["status"] == "failed"
    assert completes[0]["reason"] == "ContextExhausted"
    assert run.finding is None


def test_retry_exhaustion_produces_the_terminal_outcome():
    fake = _Scripted(
        _failed(
            "P1",
            INVESTIGATOR,
            1,
            category=FAILURE_STRUCTURED_OUTPUT,
            error_type="StructuredOutputError",
            error_message="missing once",
            retryable=True,
        ),
        _failed(
            "P1",
            INVESTIGATOR,
            2,
            category=FAILURE_STRUCTURED_OUTPUT,
            error_type="StructuredOutputError",
            error_message="missing twice",
            retryable=True,
        ),
    )
    orch, _, _ = _graph_runtime(
        executor=fake, config=OrchestrationConfig(max_retries=1)
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    assert run.retries == 1
    assert len(_events(run, "orchestration.retry")) == 1
    assert len(fake.requests) == 2
    p1 = run.tasks[0]
    assert p1.status == TASK_FAILED
    assert p1.attempts == 2
    assert p1.error == "missing twice"
    assert len(run.agent_runs) == 2
    assert run.tasks[1].status == TASK_BLOCKED
    assert run.finding is None


# ---------------------------------------------------------------------------
# 13/14 — correlation, malformed/missing output
# ---------------------------------------------------------------------------


def test_correlation_mismatch_is_rejected_and_never_committed():
    fake = _Scripted(
        replace(
            _ok("WRONG", INVESTIGATOR, 1, _report_typed()),
            task_id="WRONG",  # mismatch vs request task_id "P1"
        ),
    )
    orch, _, _ = _graph_runtime(executor=fake)
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    p1 = run.tasks[0]
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    # NOTHING from the untrusted result committed.
    assert p1.output is None
    assert p1.usage is None
    assert p1.usage_total is None
    assert p1.agent_run_id is None
    assert run.agent_runs == []
    assert run.tool_calls_total == 0
    assert run.usage_total is None
    # No retry, successors blocked.
    assert run.retries == 0
    assert _events(run, "orchestration.retry") == []
    assert run.tasks[1].status == TASK_BLOCKED
    assert run.finding is None


def test_missing_structured_output_fails_by_existing_policy():
    fake = _Scripted(
        _ok("P1", INVESTIGATOR, 1, None),  # completed, no payload
    )
    orch, _, _ = _graph_runtime(
        executor=fake, config=OrchestrationConfig(max_retries=0)
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    p1 = run.tasks[0]
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_STRUCTURED_OUTPUT
    assert p1.output is None
    assert "MISSING_STRUCTURED_OUTPUT" in p1.error
    assert _events(run, "orchestration.retry") == []


def test_free_form_dict_output_is_a_contract_violation():
    fake = _Scripted(
        _ok("P1", INVESTIGATOR, 1, {"findings": []}),
    )
    orch, _, _ = _graph_runtime(
        executor=fake, config=OrchestrationConfig(max_retries=0)
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    p1 = run.tasks[0]
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert p1.output is None
    assert run.retries == 0


# ---------------------------------------------------------------------------
# 15/16/17 — budgets
# ---------------------------------------------------------------------------


def test_per_attempt_task_budget_is_enforced_at_the_agent_guard():
    orch, _, _ = build(
        config=OrchestrationConfig(max_task_tool_calls=0),
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    budget = _events(run, "orchestration.budget.exceeded")
    agent_budget = [
        event
        for event in budget
        if event.get("budget") == "agent_max_tool_calls"
    ]
    assert len(agent_budget) == 1
    assert agent_budget[0]["limit"] == 0
    # Blocked calls never counted as executed.
    assert agent_budget[0]["observed"] == 0
    assert run.tool_calls_total == 0
    assert run.tasks[1].status == TASK_BLOCKED


def test_run_level_budget_blocks_an_attempt_before_it_starts():
    fake = _Scripted(
        _failed(
            "P1",
            INVESTIGATOR,
            1,
            category="provider",
            error_type="LLMError",
            error_message="LLMError: transient",
            retryable=True,
            usage=TaskUsage(tool_calls=1),
        ),
    )
    orch, _, _ = _graph_runtime(
        executor=fake, config=OrchestrationConfig(
            max_retries=3, max_total_tool_calls=1
        )
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    # The failed attempt consumed the run budget: the retry never
    # starts and is never counted as a consumed retry (M20 rule).
    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    assert run.retries == 0
    assert len(fake.requests) == 1
    assert _events(run, "orchestration.retry") == []
    assert run.tool_calls_total == 1
    assert run.tasks[0].attempts == 1
    assert run.tasks[0].usage_total.tool_calls == 1
    # Blocked, not failed: the gate stops the run before validation.
    assert run.tasks[0].status == TASK_BLOCKED
    assert run.validation is None
    assert run.validation_status is None
    budget = _events(run, "orchestration.budget.exceeded")
    run_budget = [
        event
        for event in budget
        if event.get("budget") == "max_total_tool_calls"
    ]
    assert len(run_budget) == 1
    assert run_budget[0]["limit"] == 1
    assert run_budget[0]["observed"] == 1


# ---------------------------------------------------------------------------
# 18 — trace identity
# ---------------------------------------------------------------------------


def test_trace_ids_belong_to_the_actual_attempts():
    base, investigator, analyst = build(
        inv_responses=investigate_read_then_report()
        + investigate_read_then_report(),
        ana_responses=[structured(finding_payload())],
    )
    recorder = _Recording(base.executor)
    orch = Orchestrator(investigator, analyst, executor=recorder)
    plan = _plan(
        ("P1", "Identify the application boundary"),
        ("P2", "Explain how the boundary is enforced"),
        ("P3", "Analyze the merged evidence"),
    )
    roles = _roles(
        ("P1", INVESTIGATOR),
        ("P2", INVESTIGATOR),
        ("P3", ANALYST),
    )
    graph = build_langgraph_orchestration_graph(orch)
    state = graph.invoke({
        "task": task(), "plan": plan, "role_assignments": roles,
    })
    run = state["run"]

    assert run.status == COMPLETED
    # Every attempt's result carried ITS OWN trace, and the agent
    # runs record exactly those ids.
    assert len(recorder.results) == 3
    result_ids = [result.agent_run_id for result in recorder.results]
    assert all(result_ids)
    assert len(set(result_ids)) == 3  # distinct per attempt
    assert [item["run_id"] for item in run.agent_runs] == result_ids
    assert run.tasks[0].agent_run_id == result_ids[0]
    assert run.tasks[1].agent_run_id == result_ids[1]
    assert run.tasks[2].agent_run_id == result_ids[2]
    # The graph state carries the traces assembly needs.
    assert len(state["investigator_traces"]) == 2
    assert state["analyst_trace"] is not None
    assert state["analyst_trace"].run_id == result_ids[2]
    # Step windows mirror onto the result's own trace (no dups).
    events = state["analyst_trace"].orchestration_events
    sequences = [event["sequence"] for event in events]
    assert len(sequences) == len(set(sequences))
    assert "orchestration.step.complete" in [
        event["operation"] for event in events
    ]


# ---------------------------------------------------------------------------
# 19/20 — missing analyst, declared-insufficient evidence
# ---------------------------------------------------------------------------


def test_missing_analyst_output_never_fabricates_a_finding():
    fake = _Scripted(
        _ok("P1", INVESTIGATOR, 1, _report_typed()),
        _ok("P2", ANALYST, 1, None),  # analyst produced nothing
    )
    orch, _, _ = _graph_runtime(
        executor=fake, config=OrchestrationConfig(max_retries=0)
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == AGENT_FAILED
    assert run.termination_reason == "AGENT_FAILED"
    assert run.finding is None  # nothing fabricated
    assert run.report is not None  # evidence package preserved
    assert run.tasks[1].status == TASK_FAILED
    assert run.tasks[1].failure_category == FAILURE_STRUCTURED_OUTPUT


def test_plan_without_analyst_ends_no_analysis_not_success():
    orch, _, _ = build(
        inv_responses=investigate_read_then_report(),
    )
    plan = _plan(("P1", "Identify the application boundary"))
    roles = _roles(("P1", INVESTIGATOR))
    run = run_with_langgraph(orch, task(), plan, roles)

    assert run.status == INCOMPLETE
    assert run.termination_reason == "NO_ANALYSIS"
    assert run.finding is None
    assert run.report is not None
    assert run.tasks[0].status == TASK_COMPLETED
    assert ANALYST not in [
        event.get("agent_role") for event in run.events
    ]


def test_declared_insufficient_evidence_is_distinguishable_from_execution():
    orch, _, _ = build(
        inv_responses=investigate_read_then_report(
            report_payload(status="incomplete")
        ),
        ana_responses=[structured(finding_payload())],
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    # Execution succeeded; the EVIDENCE declared insufficiency.
    assert run.status == INCOMPLETE
    assert run.termination_reason == "DECLARED_INCOMPLETE"
    assert all(
        record.status == TASK_COMPLETED for record in run.tasks
    )
    assert run.validation_status == "passed"
    assert run.report.status == "incomplete"
    assert run.finding is not None


# ---------------------------------------------------------------------------
# 22 — the graph never delegates the run to Orchestrator.execute
# ---------------------------------------------------------------------------


def test_graph_path_never_calls_monolithic_execute():
    orch, _, _ = build()
    calls = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "graph path invoked Orchestrator.execute — the graph must "
            "own task-level control flow itself"
        )

    orch.execute = _boom
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )
    assert run.status == COMPLETED
    assert calls == []


def test_langgraph_runner_source_has_no_monolithic_dispatch_call():
    source = Path(lg.__file__).read_text(encoding="utf-8")
    # Every `.execute(` call in the graph module must be the executor
    # boundary — never `runtime.execute(` / `Orchestrator.execute(`.
    offenders = [
        match.group(0)
        for match in re.finditer(r".{24}\.execute\(", source)
        if "executor.execute" not in match.group(0)
    ]
    assert offenders == []
    assert not re.search(r"runtime\.execute\s*\(", source)
    assert not re.search(r"Orchestrator\.execute\s*\(", source)


# ---------------------------------------------------------------------------
# contract prerequisites — boundary behavior reused unchanged
# ---------------------------------------------------------------------------


def test_requests_carry_scope_budget_and_attempt_identity():
    base, investigator, analyst = build()
    recorder = _Recording(base.executor)
    orch = Orchestrator(
        investigator,
        analyst,
        executor=recorder,
        config=OrchestrationConfig(max_task_tool_calls=7),
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )
    assert run.status == COMPLETED
    first, second = recorder.requests
    assert first.role == INVESTIGATOR
    assert second.role == ANALYST
    for request in recorder.requests:
        assert request.run_id == run.run_id
        assert request.repository_scope == REPOSITORY
        assert request.commit_scope == COMMIT
        assert request.max_tool_calls == 7
        assert request.attempt == 1
        assert request.assignment_reason == "graph test"
    assert first.depends_on == ()
    assert second.dependency_results == (run.report,)


# ---------------------------------------------------------------------------
# M21 boundary regression: budgeted participant wrappers (the observed
# notebook failure) + trace attribution on every execution shape
# ---------------------------------------------------------------------------


def test_budgeted_query_router_executes_through_the_graph():
    # Notebook-equivalent scenario: the M20 per-attempt task budget
    # (`max_task_tool_calls`) with QueryRouter as the INVESTIGATOR
    # participant. Before QueryRouter/ObservedAgent forwarded the
    # override, P1 failed twice with
    # "TypeError: QueryRouter.run_result() got an unexpected keyword
    # argument 'max_tool_calls'" — zero tool calls, no trace, every
    # downstream step blocked — identically on BOTH orchestration paths.
    base, architect, analyst = build(
        inv_responses=investigate_read_then_report(),
        ana_responses=[structured(finding_payload())],
    )
    router = QueryRouter(architecture=architect, flow=_specialist([]))
    orch = Orchestrator(
        router,
        analyst,
        config=OrchestrationConfig(max_task_tool_calls=12, max_retries=1),
    )
    run = run_with_langgraph(
        orch,
        task(),
        _plan(
            ("P1", "Identify the architecture components"),
            ("P2", "Explain how the boundary is enforced"),
        ),
        _roles(("P1", INVESTIGATOR), ("P2", ANALYST)),
    )

    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    p1, p2 = run.tasks
    assert p1.status == TASK_COMPLETED
    assert p1.attempts == 1
    assert p1.error is None
    # Trace attribution restored now that execution actually happens.
    assert p1.agent_run_id is not None
    assert p1.usage is not None
    assert p1.usage.tool_calls == 1
    assert p2.status == TASK_COMPLETED
    assert p2.agent_run_id is not None
    assert run.tool_calls_total == 1
    assert run.validation_status == "passed"
    assert router.calls == ["P1"]
    assert router.labels["P1"] == "architecture"


def test_budgeted_router_scenario_is_compatible_across_both_paths():
    # The SAME deterministic budgeted scenario through the LangGraph
    # path and the monolithic Orchestrator.execute path must produce
    # compatible task outcomes (the defect was shared-layer, so the
    # fix had to repair both).
    roles = _roles(("P1", INVESTIGATOR), ("P2", ANALYST))
    cfg = OrchestrationConfig(max_task_tool_calls=12, max_retries=1)
    scripts = dict(
        inv_responses=investigate_read_then_report(),
        ana_responses=[structured(finding_payload())],
    )

    g_base, g_inv, g_ana = build(**scripts)
    g_run = run_with_langgraph(
        Orchestrator(QueryRouter(g_inv, g_inv), g_ana, config=cfg),
        task(),
        two_step_plan(),
        roles,
    )

    r_base, r_inv, r_ana = build(**scripts)
    r_run = Orchestrator(
        QueryRouter(r_inv, r_inv), r_ana, config=cfg
    ).execute(task(), two_step_plan(), role_assignments=roles)

    assert g_run.status == r_run.status == COMPLETED
    assert g_run.termination_reason == r_run.termination_reason
    assert g_run.termination_reason == "SUCCESS"
    assert g_run.validation_status == r_run.validation_status
    assert g_run.validation_status == "passed"
    assert len(g_run.tasks) == len(r_run.tasks)
    for g_record, r_record in zip(g_run.tasks, r_run.tasks):
        assert g_record.task_id == r_record.task_id
        assert g_record.status == r_record.status
        assert g_record.attempts == r_record.attempts
        # Trace presence must agree; the ids themselves are
        # per-run objects and never equal across runs.
        assert (g_record.agent_run_id is not None) == (
            r_record.agent_run_id is not None
        )
        assert g_record.usage_total is not None
        assert r_record.usage_total is not None
        assert g_record.usage_total.tool_calls == (
            r_record.usage_total.tool_calls
        )
        assert g_record.usage_total.model_iterations == (
            r_record.usage_total.model_iterations
        )
    assert g_run.tool_calls_total == r_run.tool_calls_total


def test_terminal_failure_after_execution_start_keeps_trace():
    # The participant STARTED (one executed tool call) and then the
    # provider failed terminally (retries disabled). The failed
    # attempt must keep ITS trace id, measured usage, and the actual
    # error message — not collapse into an unexplained AGENT_FAILED.
    orch, investigator, analyst = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            LLMError("LLM request failed: upstream 502"),
        ],
        config=OrchestrationConfig(max_retries=0),
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    p1 = run.tasks[0]
    assert p1.status == TASK_FAILED
    assert p1.attempts == 1
    # Trace existed (execution began) → attribution is kept.
    assert p1.agent_run_id is not None
    assert run.agent_runs[0]["run_id"] == p1.agent_run_id
    assert run.agent_runs[0]["attempt"] == 1
    # Measured usage from the executed attempt survives.
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 1
    assert run.tool_calls_total == 1
    # Actionable diagnostics preserved verbatim.
    assert p1.error is not None
    assert "LLM request failed: upstream 502" in p1.error
    # Runner-parity failure events + downstream blocking.
    assert "orchestration.step.failed" in _ops(run)
    completes = _events(run, "orchestration.agent.complete")
    assert completes[0]["status"] == "failed"
    assert completes[0]["run_id"] == p1.agent_run_id
    assert run.tasks[1].status == TASK_BLOCKED
    assert run.finding is None


def test_retry_preserves_earlier_attempt_trace_attribution():
    # Attempt 1 reads evidence, then the provider fails (its trace
    # must remain visible); attempt 2 succeeds with its OWN trace.
    # The record holds the most recent attempt; run.agent_runs keeps
    # BOTH attributions.
    orch, investigator, analyst = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            LLMError("LLM request failed: transient upstream error"),
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ],
        config=OrchestrationConfig(max_retries=1),
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == COMPLETED
    assert run.retries == 1
    p1 = run.tasks[0]
    assert p1.attempts == 2
    first, second = (
        item for item in run.agent_runs if item["step_id"] == "P1"
    )
    assert [first["attempt"], second["attempt"]] == [1, 2]
    # Earlier attribution NOT lost: both attempt ids exist and differ.
    assert first["run_id"] is not None
    assert second["run_id"] is not None
    assert first["run_id"] != second["run_id"]
    # The record reports the most recent attempt's trace.
    assert p1.agent_run_id == second["run_id"]
    # Usage is cumulative across attempts (one tool call each).
    assert p1.usage is not None
    assert p1.usage.tool_calls == 1
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 2
    assert run.tool_calls_total == 2


def test_failure_before_trace_creation_has_no_fabricated_id():
    # A participant that blows up BEFORE any execution exists (the
    # shape of the original notebook failure: TypeError at the call
    # site) must record an honest None trace id, its real error, and
    # zero usage — never a fabricated or inherited trace id.
    class _ExplodingParticipant:
        last_trace = None

        def run_result(self, question):
            raise RuntimeError("boundary exploded before execution")

    _, _, analyst = build()
    orch = Orchestrator(
        _ExplodingParticipant(),
        analyst,
        config=OrchestrationConfig(max_retries=0),
    )
    run = run_with_langgraph(
        orch, task(), two_step_plan(), _roles(
            ("P1", INVESTIGATOR), ("P2", ANALYST),
        )
    )

    assert run.status == INCOMPLETE
    p1 = run.tasks[0]
    assert p1.status == TASK_FAILED
    assert p1.attempts == 1
    assert p1.agent_run_id is None  # honest: no trace ever existed
    assert p1.error is not None
    assert "boundary exploded before execution" in p1.error
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 0
    assert run.tool_calls_total == 0
    assert run.agent_runs == [
        {
            "step_id": "P1",
            "agent_role": INVESTIGATOR,
            "attempt": 1,
            "run_id": None,
        }
    ]
    assert run.tasks[1].status == TASK_BLOCKED
