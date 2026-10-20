"""M19 runner — TaskRecord lifecycle committed by the coordinator.

One record per plan step, mutated ONLY by the orchestrator from
``AgentTaskResult``s; the participants (scripted agents or an injected
executor) never touch run state. Failure categories come from the one
taxonomy, and blocked successors carry their blocking reason.
"""

from __future__ import annotations

from dataclasses import replace

from repo_intelligence.agent import AgentExecutionConfig
from repo_intelligence.multiagent import (
    ArchitectureFinding,
    InvestigationReport,
)
from repo_intelligence.orchestration import (
    BUDGET_EXCEEDED,
    COMPLETED,
    FAILURE_AGENT_BUDGET,
    FAILURE_CONTEXT_EXHAUSTED,
    FAILURE_SCOPE_MISMATCH,
    FAILURE_STRUCTURED_OUTPUT,
    INCOMPLETE,
    PLAN_FAILED,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    VALIDATION_FAILED,
    AgentTaskRequest,
    AgentTaskResult,
    Orchestrator,
    TaskFailure,
)

from m17_doubles import (
    COMMIT,
    REPOSITORY,
    build,
    call,
    finding_payload,
    report_payload,
    rule,
    structured,
    task,
    tool_calls,
    two_step_plan,
)

from repo_intelligence.planning.models import InvestigationPlan, PlanStep


def test_happy_path_commits_a_completed_record_per_plan_step():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    assert [record.task_id for record in run.tasks] == ["P1", "P2"]
    p1, p2 = run.tasks

    assert p1.role == "INVESTIGATOR"
    assert p1.status == TASK_COMPLETED
    assert p1.depends_on == []
    assert p1.assignment_reason
    assert p1.attempts == 1
    assert p1.failure_category is None
    assert p1.error is None
    assert isinstance(p1.output, InvestigationReport)
    assert p1.output.investigation_id == "inv-1"
    assert p1.evidence_refs
    assert p1.agent_run_id
    assert p1.started_at and p1.completed_at
    assert p1.started_at <= p1.completed_at
    assert p1.usage is not None
    assert p1.usage.tool_calls == 1
    assert p1.usage.model_iterations == 2

    assert p2.role == "ANALYST"
    assert p2.status == TASK_COMPLETED
    assert p2.depends_on == ["P1"]
    assert isinstance(p2.output, ArchitectureFinding)
    assert p2.usage is not None
    assert p2.usage.tool_calls == 0
    assert p2.usage.model_iterations == 1


def test_failed_task_carry_their_classification_and_block_successors():
    broken = report_payload()
    del broken["findings"]
    orch, _, analyst = build(
        inv_responses=[structured(broken), structured(broken)]
    )
    run = orch.execute(task(), two_step_plan())
    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_STRUCTURED_OUTPUT
    assert p1.error.startswith("StructuredOutputValidationError")
    assert p1.attempts == 2
    assert p1.completed_at
    assert p2.status == TASK_BLOCKED
    assert p2.error == "P1_FAILED"
    assert p2.output is None
    assert analyst._llm_client.requests == []


def test_agent_budget_failure_is_committed_with_its_own_category():
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", {"path": "README.md"})),
            structured(report_payload()),
        ],
        inv_kwargs={
            "execution_config": AgentExecutionConfig(max_tool_calls=0)
        },
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == BUDGET_EXCEEDED
    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_AGENT_BUDGET
    assert p1.attempts == 1
    assert p2.status == TASK_BLOCKED
    assert p2.error == "P1_FAILED"


def test_scope_mismatch_is_committed_as_its_own_failure_category():
    wrong = report_payload(commit_scope="d" * 40)
    orch, _, _ = build(
        inv_responses=[structured(wrong), structured(wrong)]
    )
    run = orch.execute(task(), two_step_plan())
    p1, _ = run.tasks
    assert run.status == INCOMPLETE
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_SCOPE_MISMATCH
    assert p1.error.startswith("SCOPE_MISMATCH:")
    assert p1.attempts == 2
    assert run.retries == 1


def test_rejected_plan_creates_no_task_records():
    orch, investigator, _ = build()
    run = orch.execute(
        task(),
        InvestigationPlan(objective="objective", steps=[]),
    )
    assert run.status == PLAN_FAILED
    assert run.tasks == []
    assert investigator._llm_client.requests == []


def test_single_step_plan_commits_a_completed_evidence_task():
    orch, _, _ = build()
    run = orch.execute(
        task(),
        InvestigationPlan(
            objective=task().question,
            steps=[PlanStep(id="P1", goal="Identify the boundary")],
        ),
    )
    assert run.status == INCOMPLETE
    assert run.termination_reason == "NO_ANALYSIS"
    assert run.report is not None
    assert run.finding is None
    (record,) = run.tasks
    assert record.status == TASK_COMPLETED
    assert record.output is not None


# -- injected executor ------------------------------------------------------


class ScriptedExecutor:
    """Deterministic fake boundary: records every request, replays a
    queued script of results (never touches the injected agents).

    The run id is generated per run (unknowable when the script was
    built), so each replayed result echoes the request's ``run_id``;
    the scripted ``task_id``/``attempt``/``role`` stay as written and
    are asserted against the request by the contract validator."""

    def __init__(self, *results):
        self.requests = []
        self.results = list(results)

    def execute(self, request: AgentTaskRequest) -> AgentTaskResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("executor invoked more times than scripted")
        return replace(self.results.pop(0), run_id=request.run_id)


def _ok(task_id, role, output, trace=None):
    return AgentTaskResult(
        run_id="run-1",
        task_id=task_id,
        attempt=1,
        role=role,
        status=TASK_COMPLETED,
        output=output,
        evidence_refs=tuple(
            getattr(output, "evidence_refs", ()) or ()
        ),
        agent_run_id="agent-{}".format(task_id),
        trace=trace,
    )


def test_injected_executor_receives_typed_requests_and_agents_stay_idle():
    report = InvestigationReport.model_validate(report_payload())
    finding = ArchitectureFinding.model_validate(finding_payload())
    executor = ScriptedExecutor(
        _ok("P1", "INVESTIGATOR", report),
        _ok("P2", "ANALYST", finding),
    )
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert investigator._llm_client.requests == []
    assert analyst._llm_client.requests == []
    assert len(executor.requests) == 2

    first, second = executor.requests
    assert isinstance(first, AgentTaskRequest)
    assert (first.run_id, first.task_id, first.attempt) == (
        run.run_id,
        "P1",
        1,
    )
    assert first.role == "INVESTIGATOR"
    assert first.assignment_reason
    assert first.question == task().question
    assert first.repository_scope == REPOSITORY
    assert first.commit_scope == COMMIT
    assert first.prompt != first.question  # task-scoped prompt
    assert REPOSITORY in first.prompt
    assert first.depends_on == ()
    assert first.dependency_results == ()

    assert second.role == "ANALYST"
    assert second.depends_on == ("P1",)
    assert len(second.dependency_results) == 1
    assert isinstance(second.dependency_results[0], InvestigationReport)

    # records committed from the RESULTS, not from participant state
    assert [record.status for record in run.tasks] == [
        TASK_COMPLETED,
        TASK_COMPLETED,
    ]
    assert run.tasks[0].output is report
    assert run.tasks[1].agent_run_id == "agent-P2"
    assert run.report is not None
    assert run.finding is not None
    # scripted traces carry no inspected evidence, so the deterministic
    # validator fails the run — records keep their committed state.
    assert run.status == VALIDATION_FAILED
    assert run.tasks[0].status == TASK_COMPLETED
    assert rule(run.validation, "scope_match").ok


def test_injected_executor_failure_without_retry_never_reburns_budget():
    failure = TaskFailure(
        category=FAILURE_CONTEXT_EXHAUSTED,
        error_type="LLMError",
        error_message="LLMError: context_length_exceeded",
        retryable=False,
    )
    executor = ScriptedExecutor(
        AgentTaskResult(
            run_id="run-1",
            task_id="P1",
            attempt=1,
            role="INVESTIGATOR",
            status="failed",
            failure=failure,
        )
    )
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    assert run.retries == 0
    assert len(executor.requests) == 1  # never retried
    assert investigator._llm_client.requests == []
    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_CONTEXT_EXHAUSTED
    assert p1.attempts == 1
    assert p1.error == "LLMError: context_length_exceeded"
    assert p2.status == TASK_BLOCKED
    assert [event["operation"] for event in run.events].count(
        "orchestration.retry"
    ) == 0
