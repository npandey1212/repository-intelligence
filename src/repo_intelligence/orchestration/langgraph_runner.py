"""LangGraph-owned sequential orchestration over the existing M19/M20 contracts.

This module deliberately does not call ``Orchestrator.execute``.  The existing
Orchestrator instance is used only as a compatibility runtime for its injected
executor/configuration and small event/status/budget helpers; the graph owns
plan gating, task selection, attempts, retry routing, task commits, assembly,
validation, and terminal routing.

LangGraph is an application dependency for this module. Importing the rest of
``repo_intelligence`` does not require LangGraph; importing this module gives a
clear dependency error when it is absent.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise ImportError(
        "LangGraph orchestration requires the 'langgraph' package. "
        "Install project dependencies before importing this module."
    ) from exc

from ..multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
    ValidationRequest,
)
from ..multiagent.validation import validate_collaboration
from ..planning.models import InvestigationPlan, PlanStep
from ..planning.validation import PlanValidationError, validate_plan
from .assembly import RunAssembler
from .contracts import (
    FAILURE_AGENT_BUDGET,
    FAILURE_CONTRACT_VIOLATION,
    FAILURE_SCOPE_MISMATCH,
    FAILURE_STRUCTURED_OUTPUT,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_PENDING,
    TASK_READY,
    TASK_RUNNING,
    AgentTaskRequest,
    TaskContractError,
    TaskFailure,
    combine_usage,
    validate_task_result,
)
from .models import (
    STEP_BLOCKED,
    STEP_COMPLETED,
    STEP_FAILED,
    OrchestrationRun,
    StepResult,
    TaskRecord,
    utcnow,
)
from .roles import ANALYST, INVESTIGATOR, RoleAssignment
from .scheduling import ready_tasks, sequential_dependencies
from .status import (
    AGENT_FAILED,
    BUDGET_EXCEEDED,
    COMPLETED,
    EXECUTING,
    INCOMPLETE,
    PLAN_FAILED,
    PLAN_VALIDATED,
    PLANNING,
    TIMEOUT,
    VALIDATING,
    VALIDATION_FAILED,
    advance,
)
from .runner import (
    _analyst_question,
    _investigator_question,
    _preassigned_role_assignments,
)


class LangGraphOrchestrationState(TypedDict, total=False):
    """Graph state. ``run`` is the authoritative domain record.

    Mutable domain objects are retained for compatibility with the existing
    orchestration contracts. Graph routing fields are explicit and small.
    """

    task: TaskPayload
    plan: InvestigationPlan
    role_assignments: list[RoleAssignment]
    run: OrchestrationRun
    started: float
    task_index: int
    attempt: int
    current_step_id: str
    current_request: AgentTaskRequest
    current_result: Any
    current_output: Any
    current_failure: Any
    current_error: str
    current_retryable: bool
    current_failure_category: str | None
    current_trace: Any
    investigator_traces: list[Any]
    analyst_trace: Any
    has_investigation_request: bool
    terminal: bool
    routing: str
    current_window: list[dict[str, Any]]
    expected_type: Any
    contract_error: str | None


def build_langgraph_orchestration_graph(runtime: Any):
    """Build a compiled graph using an injected, configured Orchestrator.

    ``runtime.execute`` is intentionally never referenced. The runtime provides
    the already-configured executor, config, clock, trace sink, and small
    deterministic lifecycle helpers so the graph can own the task loop.
    """

    def record(state: LangGraphOrchestrationState, operation: str, window=None, **fields):
        return runtime.record_event(state["run"], window, operation, **fields)

    def mark(run: OrchestrationRun, target: str) -> None:
        run.status = advance(run.status, target)
        if run.final:
            run.completed_at = utcnow()

    def terminate(state: LangGraphOrchestrationState, status: str, termination_reason: str, operation: str, **fields):
        # 3rd param is ``termination_reason`` (the runner's _terminate
        # signature): a caller may still pass a ``reason=`` EVENT field
        # (plan/validation failure detail) through ``fields``.
        run = state["run"]
        record(state, operation, status=status, termination_reason=termination_reason, **fields)
        run.termination_reason = termination_reason
        run.plan.status = "completed" if status == COMPLETED else "failed"
        run.current_step = None
        mark(run, status)
        return {"terminal": True, "routing": "end"}

    def block_from(state: LangGraphOrchestrationState, start_index: int, reason: str) -> None:
        run = state["run"]
        assignments = state["role_assignments"]
        steps_by_id = {step.id: step for step in run.plan.steps}
        for assignment in assignments[start_index:]:
            if assignment.step_id in run.completed_steps or any(
                result.step_id == assignment.step_id for result in run.step_results
            ):
                continue
            record(state, "orchestration.step.blocked", step_id=assignment.step_id, reason=reason)
            run.blocked_steps.append(assignment.step_id)
            run.step_results.append(StepResult(
                step_id=assignment.step_id, agent_role=assignment.role,
                status=STEP_BLOCKED, error=reason,
            ))
            step = steps_by_id.get(assignment.step_id)
            if step is not None:
                step.status = "blocked"
                step.result = reason
            task_record = next((item for item in run.tasks if item.task_id == assignment.step_id), None)
            if task_record is not None and task_record.status in (TASK_PENDING, TASK_READY, TASK_RUNNING):
                task_record.status = TASK_BLOCKED
                task_record.error = reason
                task_record.completed_at = utcnow()

    def initialize(state: LangGraphOrchestrationState):
        task, plan = state["task"], state["plan"]
        run = OrchestrationRun(run_id=uuid.uuid4().hex, task=task, plan=plan, started_at=utcnow())
        initial = {**state, "run": run, "started": runtime.clock(), "task_index": 0,
                   "attempt": 0, "investigator_traces": [], "analyst_trace": None,
                   "has_investigation_request": False, "terminal": False}
        record(initial, "orchestration.start", run_id=run.run_id, question=task.question,
               repository_scope=task.repository_scope, commit_scope=task.commit_scope)
        mark(run, PLANNING)
        record(initial, "orchestration.plan.created", objective=plan.objective,
               steps=[{"id": step.id, "goal": step.goal} for step in plan.steps])
        try:
            validate_plan(plan)
            if len(plan.steps) > runtime.config.max_plan_steps:
                raise PlanValidationError(
                    "plan has {} steps; max_plan_steps is {}".format(len(plan.steps), runtime.config.max_plan_steps)
                )
        except Exception as exc:
            record(initial, "orchestration.plan.validated", status="FAILED", errors=str(exc))
            return {**initial, **terminate(initial, PLAN_FAILED, "PLAN_FAILED", "orchestration.failed",
                                           run_id=run.run_id, reason=str(exc))}
        record(initial, "orchestration.plan.validated", status="SUCCESS")
        mark(run, PLAN_VALIDATED)
        mark(run, EXECUTING)
        plan.status = "in_progress"
        assignments = _preassigned_role_assignments(plan, state["role_assignments"])
        dependencies = sequential_dependencies([step.id for step in plan.steps])
        run.tasks = [TaskRecord(task_id=step.id, role=assignment.role, status=TASK_PENDING,
                                depends_on=list(dependencies[step.id]), assignment_reason=assignment.reason)
                     for step, assignment in zip(plan.steps, assignments)]
        return {**initial, "role_assignments": assignments, "routing": "select"}

    def select_task(state: LangGraphOrchestrationState):
        run = state["run"]
        index = state["task_index"]
        if index >= len(run.plan.steps):
            return {"routing": "assemble"}
        step = run.plan.steps[index]
        assignment = state["role_assignments"][index]
        needs_messages = 1 if assignment.role == INVESTIGATOR and not state["has_investigation_request"] else (3 if assignment.role == ANALYST else 0)
        gate = runtime.pre_task_gate(run, needs_messages, state["started"], step.id)
        if gate is not None:
            block_from(state, index, "ORCHESTRATION_STOPPED")
            record(state, gate["operation"], **gate["fields"])
            assembler = RunAssembler()
            reports = assembler.collect_investigator_reports(run.step_results)
            if reports and run.report is None:
                run.report = assembler.merge_reports(reports, run.task)
            terminate(state, gate["status"], gate["reason"], "orchestration.terminated", run_id=run.run_id)
            return {"routing": "end", "terminal": True}
        run.current_step = step.id
        record(state, "orchestration.step.start", step_id=step.id, goal=step.goal)
        window: list[dict[str, Any]] = []
        record(state, "orchestration.step.assign", window, step_id=step.id,
               agent_role=assignment.role, assignment_reason=assignment.reason)
        step.status = "in_progress"
        if assignment.role == INVESTIGATOR and not state["has_investigation_request"]:
            runtime.append_handoff(run, window, "INVESTIGATION_REQUEST", "task", "investigator", run.task, step.id)
            state["has_investigation_request"] = True
        if assignment.role == ANALYST:
            reports = RunAssembler().collect_investigator_reports(run.step_results)
            if run.report is None and reports:
                run.report = RunAssembler().merge_reports(reports, run.task)
            if run.report is None:
                raise RuntimeError("no investigation report available for analysis")
            runtime.append_handoff(run, window, "INVESTIGATION_RESULT", "investigator", "analyst", run.report, step.id)
            runtime.append_handoff(run, window, "ANALYSIS_REQUEST", "task", "analyst", run.report, step.id)
        expected_type = InvestigationReport if assignment.role == INVESTIGATOR else ArchitectureFinding
        prompt = _investigator_question(run.task, step) if assignment.role == INVESTIGATOR else _analyst_question(run.task, run.report)
        task_record = next(item for item in run.tasks if item.task_id == step.id)
        if step.id not in {item.task_id for item in ready_tasks(run.tasks)}:
            raise RuntimeError("task {!r} is not ready: dependencies incomplete".format(step.id))
        task_record.status = TASK_READY
        task_record.status = TASK_RUNNING
        task_record.started_at = task_record.started_at or utcnow()
        attempt = 1
        record(state, "orchestration.agent.start", window, step_id=step.id, agent_role=assignment.role, attempt=attempt)
        request = AgentTaskRequest(
            run_id=run.run_id, task_id=step.id, attempt=attempt, role=assignment.role,
            assignment_reason=assignment.reason, question=run.task.question, goal=step.goal,
            prompt=prompt, repository_scope=run.task.repository_scope, commit_scope=run.task.commit_scope,
            depends_on=tuple(task_record.depends_on), dependency_results=(run.report,) if assignment.role == ANALYST and run.report is not None else (),
            max_tool_calls=runtime.config.max_task_tool_calls,
        )
        return {"current_step_id": step.id, "current_request": request, "attempt": attempt,
                "current_window": window, "expected_type": expected_type,
                "has_investigation_request": state["has_investigation_request"],
                "routing": "execute"}

    def execute_task(state: LangGraphOrchestrationState):
        result = runtime.executor.execute(state["current_request"])
        request = state["current_request"]
        error = None
        try:
            validate_task_result(request, result)
        except Exception as exc:
            error = str(exc)
        return {"current_result": result, "contract_error": error, "routing": "commit"}

    def commit_result(state: LangGraphOrchestrationState):
        run = state["run"]
        step_id = state["current_step_id"]
        index = state["task_index"]
        step = run.plan.steps[index]
        assignment = state["role_assignments"][index]
        task_record = next(item for item in run.tasks if item.task_id == step_id)
        result = state["current_result"]
        request = state["current_request"]
        window = state.get("current_window", [])
        contract_error = state.get("contract_error")
        output = None
        failure = None
        trace = None
        agent_run_id = None
        if contract_error is None:
            output, failure, trace, agent_run_id = result.output, result.failure, result.trace, result.agent_run_id
            task_record.agent_run_id = agent_run_id
            task_record.usage = result.usage
            task_record.usage_total = combine_usage(task_record.usage_total, result.usage)
            run.agent_runs.append({"step_id": step_id, "agent_role": assignment.role, "attempt": request.attempt, "run_id": agent_run_id})
            run.tool_calls_total += result.usage.tool_calls
            run.usage_total = combine_usage(run.usage_total, result.usage)
            if assignment.role == INVESTIGATOR and trace is not None:
                state["investigator_traces"].append(trace)
        task_record.attempts = request.attempt
        category = failure.category if failure is not None else None
        retryable = failure.retryable if failure is not None else False
        if contract_error is not None:
            failure = TaskFailure(category=FAILURE_CONTRACT_VIOLATION, error_type="TaskContractError",
                                  error_message=contract_error, retryable=False)
            category, error, retryable = FAILURE_CONTRACT_VIOLATION, contract_error, False
        elif failure is None and isinstance(output, state["expected_type"]):
            scope_ok = not isinstance(output, InvestigationReport) or (
                output.repository_scope == run.task.repository_scope and output.commit_scope == run.task.commit_scope)
            if scope_ok:
                record(state, "orchestration.agent.complete", window, step_id=step_id, agent_role=assignment.role,
                       attempt=request.attempt, run_id=agent_run_id, status="completed")
                runtime.close_trace_window(window, trace)
                task_record.status = TASK_COMPLETED
                task_record.completed_at = utcnow()
                task_record.output = output
                task_record.evidence_refs = list(result.evidence_refs)
                task_record.failure_category = None
                task_record.error = None
                run.completed_steps.append(step_id)
                run.step_results.append(StepResult(step_id=step_id, agent_role=assignment.role, status=STEP_COMPLETED,
                                                   output=output, evidence_refs=list(result.evidence_refs),
                                                   agent_run_id=agent_run_id, attempts=request.attempt))
                if assignment.role == ANALYST:
                    run.finding = output
                    state["analyst_trace"] = trace
                    runtime.append_handoff(run, window, "ANALYSIS_RESULT", "analyst", "task", output, step_id)
                record(state, "orchestration.step.complete", window, step_id=step_id, status=STEP_COMPLETED,
                       artifact=getattr(output, "investigation_id", None))
                runtime.close_trace_window(window, trace)
                step.status = "completed"
                step.result = "{} {}".format(type(output).__name__, getattr(output, "investigation_id", "") or "").strip()
                updates = {"task_index": index + 1, "attempt": 0,
                           "has_investigation_request": state["has_investigation_request"],
                           "routing": "select"}
                if assignment.role == ANALYST:
                    # LangGraph discards in-place key writes (only
                    # returned keys become channels): the analyst trace
                    # must travel in the UPDATE or assembly would run
                    # validation with analyst_trace=None.
                    updates["analyst_trace"] = trace
                return updates
            category = FAILURE_SCOPE_MISMATCH
            error = "SCOPE_MISMATCH: report scope {}@{} != task scope {}@{}".format(
                output.repository_scope, output.commit_scope, run.task.repository_scope, run.task.commit_scope)
            retryable = True
        elif failure is None:
            error = "MISSING_STRUCTURED_OUTPUT: expected {}".format(state["expected_type"].__name__)
            category, retryable = FAILURE_STRUCTURED_OUTPUT, True
        else:
            error = failure.error_message
        # Runner parity: the failed attempt's reason is the failure's
        # ERROR TYPE when a classified failure exists (the message is
        # on the record/result), the error text otherwise.
        record(state, "orchestration.agent.complete", window, step_id=step_id, agent_role=assignment.role,
               attempt=request.attempt, run_id=agent_run_id, status="failed",
               reason=(failure.error_type if failure is not None else error))
        runtime.close_trace_window(window, trace)
        task_record.attempts = request.attempt
        if category == FAILURE_AGENT_BUDGET:
            retryable = False
        if retryable and request.attempt <= runtime.config.max_retries:
            gate = runtime.pre_task_gate(run, 0, state["started"], step_id)
            if gate is None:
                run.retries += 1
                # The retry event stays in THIS step's window (as the
                # runner does) so the next attempt's close mirrors it
                # onto the attempt trace.
                record(state, "orchestration.retry", window, step_id=step_id,
                       attempt=request.attempt + 1, reason=error)
                return {"attempt": request.attempt + 1, "routing": "retry"}
            block_from(state, index, "ORCHESTRATION_STOPPED")
            record(state, gate["operation"], **gate["fields"])
            reports = RunAssembler().collect_investigator_reports(run.step_results)
            if reports and run.report is None:
                run.report = RunAssembler().merge_reports(reports, run.task)
            terminate(state, gate["status"], gate["reason"], "orchestration.terminated", run_id=run.run_id)
            return {"routing": "end", "terminal": True}
        # Terminal failure — runner fail_step parity: record the step
        # failure INTO the window and close it (the trace gains
        # orchestration.step.failed; the run log is authoritative).
        record(state, "orchestration.step.failed", window, step_id=step_id,
               status="AGENT_FAILED", error=error)
        runtime.close_trace_window(window, trace)
        task_record.status = TASK_FAILED
        task_record.failure_category = category
        task_record.error = error
        task_record.completed_at = utcnow()
        run.failed_steps.append(step_id)
        run.failures.append("{}: {}".format(step_id, error))
        run.step_results.append(StepResult(step_id=step_id, agent_role=assignment.role, status=STEP_FAILED,
                                           agent_run_id=agent_run_id, attempts=request.attempt, error=error))
        step.status = "failed"
        step.result = error
        block_from(state, index + 1, "{}_FAILED".format(step_id))
        reports = RunAssembler().collect_investigator_reports(run.step_results)
        if reports and run.report is None:
            run.report = RunAssembler().merge_reports(reports, run.task)
        if category == FAILURE_AGENT_BUDGET:
            record(state, "orchestration.budget.exceeded", budget="agent_max_tool_calls",
                   limit=request.max_tool_calls, observed=getattr(task_record.usage, "tool_calls", None), step_id=step_id)
            terminate(state, BUDGET_EXCEEDED, "BUDGET_EXCEEDED", "orchestration.terminated", run_id=run.run_id)
        else:
            status = INCOMPLETE if assignment.role == INVESTIGATOR else AGENT_FAILED
            terminate(state, status, "AGENT_FAILED", "orchestration.terminated", run_id=run.run_id)
        return {"routing": "end", "terminal": True}

    def route_after_commit(state: LangGraphOrchestrationState):
        return state.get("routing", "select")

    def retry_task(state: LangGraphOrchestrationState):
        request = state["current_request"]
        # Fresh immutable request for this attempt; all identity/scope fields
        # stay constant while attempt number advances.
        from dataclasses import replace
        next_request = replace(request, attempt=state["attempt"])
        # Reuse THIS step's window (the runner keeps one window per
        # step across attempts, so the pending orchestration.retry
        # event and the new attempt's events close together).
        window = state.get("current_window")
        if window is None:
            window = []
        record(state, "orchestration.agent.start", window, step_id=next_request.task_id,
               agent_role=next_request.role, attempt=next_request.attempt)
        task_record = next(item for item in state["run"].tasks if item.task_id == next_request.task_id)
        task_record.status = TASK_RUNNING
        task_record.started_at = task_record.started_at or utcnow()
        return {"current_request": next_request, "current_window": window, "routing": "execute"}

    def assemble_and_validate(state: LangGraphOrchestrationState):
        run, task = state["run"], state["task"]
        assembler = RunAssembler()
        outcome = assembler.assemble(run.step_results, task, run.finding)
        if run.report is None:
            run.report = outcome.report
        if outcome.finding is None:
            return terminate(state, INCOMPLETE, "NO_ANALYSIS", "orchestration.terminated", run_id=run.run_id)
        gate = runtime.pre_task_gate(run, 2, state["started"], None)
        if gate is not None:
            record(state, gate["operation"], **gate["fields"])
            return terminate(state, gate["status"], gate["reason"], "orchestration.terminated", run_id=run.run_id)
        mark(run, VALIDATING)
        record(state, "orchestration.validation.start")
        runtime.append_handoff(run, None, "VALIDATION_REQUEST", "task", "validator",
                                ValidationRequest(investigation_id=run.report.investigation_id,
                                                  repository_scope=task.repository_scope, commit_scope=task.commit_scope), None)
        merged_trace = {"iterations": [iteration.to_dict() for trace in state["investigator_traces"]
                                        for iteration in trace.iterations]}
        verdict = validate_collaboration(task=task, report=run.report, finding=run.finding,
                                         investigator_trace=merged_trace, messages=run.messages,
                                         analyst_trace=state["analyst_trace"])
        run.validation, run.validation_status = verdict, verdict.status
        runtime.append_handoff(run, None, "VALIDATION_RESULT", "validator", "task", verdict, None)
        record(state, "orchestration.validation.complete", status=verdict.status,
               checks=[{"rule": check.rule, "ok": check.ok} for check in verdict.checks])
        if verdict.status != "passed":
            return terminate(state, VALIDATION_FAILED, "VALIDATION_FAILED", "orchestration.failed",
                             run_id=run.run_id, reason="deterministic validation rejected the result")
        if run.report.status == "incomplete" or run.finding.status == "incomplete":
            return terminate(state, INCOMPLETE, "DECLARED_INCOMPLETE", "orchestration.terminated", run_id=run.run_id)
        return terminate(state, COMPLETED, "SUCCESS", "orchestration.complete", run_id=run.run_id,
                         completed_steps=len(run.completed_steps))

    graph = StateGraph(LangGraphOrchestrationState)
    graph.add_node("initialize", initialize)
    graph.add_node("select_task", select_task)
    graph.add_node("execute_task", execute_task)
    graph.add_node("commit_result", commit_result)
    graph.add_node("retry_task", retry_task)
    graph.add_node("assemble_and_validate", assemble_and_validate)
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges("initialize", lambda s: s.get("routing", "select"),
                                {"select": "select_task", "end": END})
    graph.add_conditional_edges("select_task", lambda s: s.get("routing", "execute"),
                                {"execute": "execute_task", "assemble": "assemble_and_validate", "end": END})
    graph.add_edge("execute_task", "commit_result")
    graph.add_conditional_edges("commit_result", route_after_commit,
                                {"select": "select_task", "retry": "retry_task", "end": END})
    graph.add_edge("retry_task", "execute_task")
    graph.add_edge("assemble_and_validate", END)
    return graph.compile()


def run_with_langgraph(runtime: Any, task: TaskPayload, plan: InvestigationPlan,
                       role_assignments: list[RoleAssignment]):
    """Execute a preplanned run with LangGraph as task-loop owner."""
    graph = build_langgraph_orchestration_graph(runtime)
    result = graph.invoke({"task": task, "plan": plan, "role_assignments": role_assignments})
    return result["run"]
