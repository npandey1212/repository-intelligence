"""M17 orchestrator — plan execution, role assignment, bounds (spec §4–§5, §7, §11–§27).

The orchestrator answers "How, when, and under what constraints should
the planned work be executed?" over the EXISTING M10 plan and the
EXISTING M16 agents — no second planning framework, no agent framework:

    plan (validated by planning.validation)
        → sequential step loop (application-owned, no while-True)
            step.start → step.assign → [typed handoffs]
              → agent.start → Agent.run_result → agent.complete
              → step.complete | step.failed (+ bounded retry)
            → blocked downstream steps on failure
        → deterministic validation (M16 validate_collaboration)
        → terminal status

Design rules baked into this module:

* **Roles/contracts, never repository knowledge** — step → role comes
  from ``orchestration.roles`` (deterministic, application-owned) or
  is supplied preassigned by the caller through ``execute``; a
  preassigned set is validated against the same invariants and used
  verbatim (never silently reassigned); there is no repository-
  specific branch anywhere here (spec §34/§35).
* **Typed handoffs reuse M16** — the message list is the same
  six-message sequence the M16 validator checks (``make_message``);
  per-step inputs are step-scoped questions built here, and each
  step's typed output lands in a ``StepResult`` (spec §31).
* **One validation, one evidence authority** — the final verdict IS
  the M16 ``validate_collaboration`` over the MERGED investigator
  reports and the final finding, with every investigator trace merged
  into one evidence mapping (the M13 ``gather_evidence`` mechanism
  accepts a trace-mapping, so no second evidence model exists).
* **Bounded at every dimension that is real here** — plan steps,
  agent runs, typed messages, total tool calls, retries, wall-clock
  runtime (injectable ``clock``); the agent's own tool budget remains
  the agent's (``AgentExecutionConfig``). Budgets are checked BEFORE
  the next unit of work; nothing is ever preempted mid-run.
* **Retries are for TRANSIENT failures only** — a timeout, rate limit
  or 5xx is retried by the bounded retry above; an EXECUTION BOUNDARY
  (the agent's own budget, provider context exhaustion, a failed
  tool-free finalization pass) is not, because replaying it would
  reproduce the same failure. Such a step fails with its own explicit
  status (``INCOMPLETE`` for the investigator) instead of burning the
  retry budget.
* **Explicit failure semantics** — invalid plan → PLAN_FAILED (no
  agent ever runs); Investigator step failure → INCOMPLETE +
  ``AGENT_FAILED`` with downstream BLOCKED (spec §42); Analyst step
  failure → AGENT_FAILED (no fabricated architectural answer);
  validation failure → VALIDATION_FAILED (never converted to success);
  terminal states are sealed — no further agent execution.
* **Observable decisions** — sequence-numbered events on the run
  (authoritative), mirrored onto each task result's OWN execution
  trace for its step windows (``trace.record_orchestration_event``)
  — the ONE trace system, no second framework, no hidden reasoning
  (spec §16–§21).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..agent import Agent
from ..agent.trace_sink import TraceSink
from ..multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
    ValidationRequest,
    make_message,
    report_json,
)
from ..multiagent.validation import validate_collaboration
from ..planning.models import InvestigationPlan, PlanStep
from ..planning.validation import PlanValidationError, validate_plan
from .assembly import RunAssembler, merge_investigation_reports
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
from .executor import AgentExecutor, AgentTaskExecutor
from .models import (
    STEP_BLOCKED,
    STEP_COMPLETED,
    STEP_FAILED,
    OrchestrationRun,
    StepResult,
    TaskRecord,
    utcnow,
)
from .roles import (
    AGENT_ROLES,
    ANALYST,
    INVESTIGATOR,
    RoleAssignment,
    assign_roles,
)
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

__all__ = [
    "DEFAULT_MAX_AGENT_RUNS",
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MAX_PLAN_STEPS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOTAL_TOOL_CALLS",
    "OrchestrationConfig",
    "Orchestrator",
    "merge_investigation_reports",
]

DEFAULT_MAX_PLAN_STEPS = 8
DEFAULT_MAX_AGENT_RUNS = 16
DEFAULT_MAX_MESSAGES = 8
DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_TOTAL_TOOL_CALLS = 32

# PlanStep status vocabulary (planning.models.PlanStatus values used
# by this executor).
_PLAN_IN_PROGRESS = "in_progress"
_PLAN_COMPLETED = "completed"
_PLAN_FAILED = "failed"
_PLAN_BLOCKED = "blocked"


@dataclass
class OrchestrationConfig:
    """Bounds for the whole orchestrated run (all explicit).

    Only dimensions orchestration newly introduces are bounded here;
    per-agent tool execution stays bounded by each agent's own config
    (no overlapping budget systems), and M20 adds ONE explicit
    orchestration-owned task default (``max_task_tool_calls``) that
    is enforced through that same agent guard — never a second
    tool-dispatch loop.
    """

    # Plan gate: too many steps is a rejected plan, not a longer run.
    max_plan_steps: int = DEFAULT_MAX_PLAN_STEPS
    # Agent invocations across ALL steps (every attempt counts).
    max_agent_runs: int = DEFAULT_MAX_AGENT_RUNS
    # Typed handoffs (the M16 six-message sequence needs 6).
    max_messages: int = DEFAULT_MAX_MESSAGES
    # Bounded retry per step (0 disables retry entirely).
    max_retries: int = DEFAULT_MAX_RETRIES
    # Executed tool calls summed across every agent run. Checked
    # BEFORE a step starts AND before each retry attempt — it is a
    # between-attempts gate over committed measurements, never a
    # live shared counter injected into a running attempt (a started
    # attempt may still execute past the remaining budget; the next
    # gate stops new work).
    max_total_tool_calls: int = DEFAULT_MAX_TOTAL_TOOL_CALLS
    # M20: the PER-ATTEMPT tool-call limit placed on every dispatched
    # AgentTaskRequest. None (default) = no task-level override —
    # each participant's own AgentExecutionConfig.max_tool_calls
    # applies, exactly the M19 behavior. An int (>= 0, bool rejected)
    # is enforced by the default adapter at the agent's pre-dispatch
    # guard BEFORE any tool executes; each attempt/retry receives its
    # own fresh allowance (cumulative usage still sums every
    # attempt). 0 = no tool call may execute on any attempt.
    max_task_tool_calls: Optional[int] = None
    # Whole-run wall clock; checked between steps (never preempts).
    max_runtime_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "max_plan_steps",
            "max_agent_runs",
            "max_messages",
            "max_total_tool_calls",
        ):
            if getattr(self, name) < 1:
                raise ValueError("{} must be >= 1".format(name))
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.max_task_tool_calls is not None:
            if not isinstance(self.max_task_tool_calls, int) or isinstance(
                self.max_task_tool_calls, bool
            ):
                raise ValueError(
                    "max_task_tool_calls must be an integer or None, "
                    "got {!r}".format(self.max_task_tool_calls)
                )
            if self.max_task_tool_calls < 0:
                raise ValueError(
                    "max_task_tool_calls must be >= 0, got {}".format(
                        self.max_task_tool_calls
                    )
                )
        if self.max_runtime_seconds is not None:
            if self.max_runtime_seconds <= 0:
                raise ValueError(
                    "max_runtime_seconds must be > 0 when set"
                )


def _investigator_question(task: TaskPayload, step: PlanStep) -> str:
    """The step-scoped investigation prompt (application-built).

    The three deterministic metadata labels (``question`` /
    ``repository_scope`` / ``commit_scope``) are part of the contract
    of this text: the runtime reads them back to anchor the typed
    report to this task, so they are stated here, not left for the
    model to echo from memory.
    """
    return (
        "Investigate this plan step and produce your structured "
        "InvestigationReport.\n"
        "question: {question}\n"
        "step: {step_id} \u2014 {goal}\n"
        "repository_scope: {repository}\n"
        "commit_scope: {commit}\n"
        "Report findings backed only by evidence you inspected; carry "
        "open questions and uncertainties in the typed fields.".format(
            question=task.question,
            step_id=step.id,
            goal=step.goal,
            repository=task.repository_scope,
            commit=task.commit_scope,
        )
    )


def _analyst_question(task: TaskPayload, report: InvestigationReport) -> str:
    """The analysis prompt over the merged evidence package (same
    contract shape the M16 runner uses — typed report embedded)."""
    return (
        "Analyze the Investigator's typed report and produce your "
        "structured ArchitectureFinding.\n"
        "question: {question}\n"
        "repository_scope: {repository}\n"
        "commit_scope: {commit}\n"
        "investigation_id: {investigation_id}\n"
        "Use ONLY this report's evidence (no independent "
        "investigation):\n{report}".format(
            question=task.question,
            repository=task.repository_scope,
            commit=task.commit_scope,
            investigation_id=report.investigation_id,
            report=report_json(report),
        )
    )


def _preassigned_role_assignments(
    plan: InvestigationPlan,
    role_assignments: Sequence[RoleAssignment],
) -> List[RoleAssignment]:
    """Validate caller-preassigned step roles (the additive input).

    The preassigned path bypasses ``role_assigner`` entirely, so the
    guarantees ``assign_roles`` would normally establish are re-
    applied here — deterministically, BEFORE any agent runs, and
    always as an explicit ``ValueError`` (never a silent fallback to
    defaults, never a mutation of what the caller decided):

    * every item is a ``RoleAssignment`` carrying a supported role
      (``AGENT_ROLES`` — rejected, never coerced);
    * step ids are unique and cover the plan EXACTLY (an id absent
      from the plan, or a plan step without an assignment, names the
      offending id in the error);
    * workflow shape: an ANALYST step needs a preceding INVESTIGATOR
      step in plan order (the one runtime-shape invariant the step
      loop itself depends on — analysis with no report would be an
      execution crash, not an outcome);
    * the result is normalized to plan order, so the caller may list
      assignments in any order and ``zip(plan.steps, assignments)``
      still pairs each step with its own role.
    """
    try:
        provided = list(role_assignments)
    except TypeError:
        raise ValueError(
            "role_assignments must be a sequence of RoleAssignment, "
            "got {}".format(type(role_assignments).__name__)
        )
    for index, item in enumerate(provided):
        if not isinstance(item, RoleAssignment):
            raise ValueError(
                "role_assignments[{}] must be a RoleAssignment, got "
                "{}".format(index, type(item).__name__)
            )

    seen = set()
    for item in provided:
        if item.step_id in seen:
            raise ValueError(
                "duplicate step id {!r} in role_assignments".format(
                    item.step_id
                )
            )
        seen.add(item.step_id)
        if item.role not in AGENT_ROLES:
            raise ValueError(
                "unsupported role {!r} for step {!r} (expected one of "
                "{})".format(
                    item.role, item.step_id, ", ".join(AGENT_ROLES)
                )
            )

    known = {step.id for step in plan.steps}
    unknown = sorted(seen - known)
    if unknown:
        raise ValueError(
            "role_assignments name step id(s) absent from the plan: "
            "{}".format(", ".join(unknown))
        )
    missing = [step.id for step in plan.steps if step.id not in seen]
    if missing:
        raise ValueError(
            "role_assignments are missing step id(s) present in the "
            "plan: {}".format(", ".join(missing))
        )

    by_id = {item.step_id: item for item in provided}
    ordered = [by_id[step.id] for step in plan.steps]

    seen_investigator = False
    for item in ordered:
        if item.role == INVESTIGATOR:
            seen_investigator = True
        elif item.role == ANALYST and not seen_investigator:
            raise ValueError(
                "step {!r} is assigned ANALYST before any INVESTIGATOR "
                "step in plan order; analysis requires a preceding "
                "evidence step".format(item.step_id)
            )
    return ordered


class Orchestrator:
    """Coordinates one validated plan over the injected participants.

    The participants are injected (already constructed with their
    role permissions, memory/context, and structured-output contracts
    — this class never builds a client or a tool). Each task is
    dispatched through the ``AgentExecutor`` boundary (M19): the
    executor returns an ``AgentTaskResult`` and only this coordinator
    mutates authoritative run state — participant execution never
    touches the ``OrchestrationRun``.
    """

    def __init__(
        self,
        investigator: Agent,
        analyst: Agent,
        trace_sink: Optional[TraceSink] = None,
        config: Optional[OrchestrationConfig] = None,
        clock: Callable[[], float] = time.monotonic,
        role_assigner: Callable[..., List[RoleAssignment]] = assign_roles,
        executor: Optional[AgentExecutor] = None,
    ) -> None:
        self._investigator = investigator
        self._analyst = analyst
        self._trace_sink = trace_sink
        self._config = config if config is not None else OrchestrationConfig()
        self._clock = clock
        self._role_assigner = role_assigner
        # M19 dispatch boundary: the default adapter runs the two
        # injected participants; callers may inject any executor
        # (fakes in tests, a specialist registry in the pipeline).
        self._executor: AgentExecutor = (
            executor
            if executor is not None
            else AgentTaskExecutor.for_agents(investigator, analyst)
        )

    @property
    def executor(self) -> AgentExecutor:
        """The task-execution boundary tasks are dispatched through."""
        return self._executor

    @property
    def config(self) -> OrchestrationConfig:
        """Read-only access to the run bounds for alternate graph drivers."""
        return self._config

    @property
    def clock(self) -> Callable[[], float]:
        """The injected monotonic clock shared by every orchestration driver."""
        return self._clock

    # Public lifecycle primitives for framework adapters. These methods do
    # not run a plan or own task-loop control flow; a driver chooses when to
    # call them and remains responsible for its own transitions.
    def record_event(self, run, window, operation, **fields):
        return self._record(run, window, operation, **fields)

    def close_trace_window(self, window, trace):
        return self._close_window(window, trace)

    def append_handoff(self, run, window, message_type, sender, recipient, payload, step_id):
        return self._append_message(run, window, message_type, sender, recipient, payload, step_id)

    def pre_task_gate(self, run, needs_messages, started, step_id):
        return self._pre_step_gate(run, needs_messages, started, step_id)

    # -- event / window plumbing ----------------------------------------

    def _record(
        self,
        run: OrchestrationRun,
        window: Optional[List[Dict[str, Any]]],
        operation: str,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Append one event to the authoritative run log (and to the
        current step window, which is mirrored onto the task's own
        execution trace)."""
        run._sequence += 1
        event: Dict[str, Any] = {
            "operation": operation,
            "sequence": run._sequence,
            "timestamp": utcnow(),
        }
        event.update(fields)
        run.events.append(event)
        if window is not None:
            window.append(dict(event))
        return event

    def _close_window(
        self, window: List[Dict[str, Any]], trace: Optional[Any]
    ) -> None:
        """Mirror this window's events onto the RESULT's own execution
        trace (``AgentTaskResult.trace`` — the boundary's attribution;
        never a participant's ``last_trace``, which an injected
        executor may not even own) and re-write it through the sink
        (the ONE trace system — the orchestrator never opens a second
        one). ``trace is None`` (no exposed trace, or a contract
        violation whose result was never trusted) simply closes the
        window: the run's event list stays authoritative."""
        if not window:
            return
        if trace is not None and hasattr(
            trace, "record_orchestration_event"
        ):
            for event in window:
                trace.record_orchestration_event(event)
            if self._trace_sink is not None:
                try:
                    self._trace_sink.write(trace)
                except Exception:
                    # Observability only (same policy as a failed sink
                    # write inside the Agent): never a run failure.
                    pass
        window.clear()

    def _append_message(
        self,
        run: OrchestrationRun,
        window: Optional[List[Dict[str, Any]]],
        message_type: str,
        sender: str,
        recipient: str,
        payload: Any,
        step_id: Optional[str],
    ) -> None:
        """Append one typed handoff (budget pre-checked) and record the
        ``orchestration.handoff.send`` event."""
        message = make_message(
            message_type,
            sender,
            recipient,
            payload,
            run.task.repository_scope,
            run.task.commit_scope,
        )
        run.messages.append(message)
        evidence_count = len(getattr(payload, "evidence_refs", ()) or ())
        fields: Dict[str, Any] = {
            "sender": sender,
            "recipient": recipient,
            "message_type": message_type,
            "message_id": message.message_id,
            "evidence_count": evidence_count,
        }
        if step_id is not None:
            fields["step_id"] = step_id
        self._record(
            run, window, "orchestration.handoff.send", **fields
        )

    def _mark(self, run: OrchestrationRun, target: str) -> None:
        """Move the run through the status machine (single gate) and
        stamp the completion time on terminal moves."""
        run.status = advance(run.status, target)
        if run.final:
            run.completed_at = utcnow()

    def _terminate(
        self,
        run: OrchestrationRun,
        status: str,
        termination_reason: str,
        operation: str,
        **fields: Any,
    ) -> None:
        """Record the terminal event, seal the plan lifecycle, and
        move to a terminal status — every exit goes through here."""
        self._record(
            run,
            None,
            operation,
            status=status,
            termination_reason=termination_reason,
            **fields,
        )
        run.termination_reason = termination_reason
        run.plan.status = (
            _PLAN_COMPLETED if status == COMPLETED else _PLAN_FAILED
        )
        run.current_step = None
        self._mark(run, status)

    def _block_remaining(
        self,
        run: OrchestrationRun,
        assignments: Sequence[RoleAssignment],
        start_index: int,
        reason: str,
        plan_steps: Dict[str, PlanStep],
    ) -> None:
        """Mark every not-yet-executed step BLOCKED (spec §27 — no
        dependency inference; downstream steps do not run)."""
        for assignment in assignments[start_index:]:
            if assignment.step_id in run.completed_steps:
                continue
            if any(
                result.step_id == assignment.step_id
                for result in run.step_results
            ):
                continue
            self._record(
                run,
                None,
                "orchestration.step.blocked",
                step_id=assignment.step_id,
                reason=reason,
            )
            run.blocked_steps.append(assignment.step_id)
            run.step_results.append(
                StepResult(
                    step_id=assignment.step_id,
                    agent_role=assignment.role,
                    status=STEP_BLOCKED,
                    error=reason,
                )
            )
            plan_step = plan_steps.get(assignment.step_id)
            if plan_step is not None:
                plan_step.status = _PLAN_BLOCKED
                plan_step.result = reason
            record = next(
                (
                    item
                    for item in run.tasks
                    if item.task_id == assignment.step_id
                ),
                None,
            )
            if record is not None and record.status in (
                TASK_PENDING,
                TASK_READY,
                TASK_RUNNING,
            ):
                record.status = TASK_BLOCKED
                record.error = reason
                record.completed_at = utcnow()

    # -- budgets ----------------------------------------------------------

    def _pre_step_gate(
        self,
        run: OrchestrationRun,
        needs_messages: int,
        started: float,
        step_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Check every orchestration bound BEFORE the next unit of
        work. Returns the terminal descriptor (status, reason,
        event/fields) or ``None`` when work may proceed."""
        config = self._config
        if len(run.agent_runs) >= config.max_agent_runs:
            return {
                "status": BUDGET_EXCEEDED,
                "reason": "BUDGET_EXCEEDED",
                "operation": "orchestration.budget.exceeded",
                "fields": {
                    "budget": "max_agent_runs",
                    "limit": config.max_agent_runs,
                    "observed": len(run.agent_runs),
                    "step_id": step_id,
                },
            }
        if run.tool_calls_total >= config.max_total_tool_calls:
            return {
                "status": BUDGET_EXCEEDED,
                "reason": "BUDGET_EXCEEDED",
                "operation": "orchestration.budget.exceeded",
                "fields": {
                    "budget": "max_total_tool_calls",
                    "limit": config.max_total_tool_calls,
                    "observed": run.tool_calls_total,
                    "step_id": step_id,
                },
            }
        if config.max_runtime_seconds is not None:
            elapsed = self._clock() - started
            if elapsed > config.max_runtime_seconds:
                return {
                    "status": TIMEOUT,
                    "reason": "TIMEOUT",
                    "operation": "orchestration.timeout",
                    "fields": {
                        "limit": config.max_runtime_seconds,
                        "observed": elapsed,
                        "step_id": step_id,
                    },
                }
        if len(run.messages) + needs_messages > config.max_messages:
            return {
                "status": BUDGET_EXCEEDED,
                "reason": "BUDGET_EXCEEDED",
                "operation": "orchestration.budget.exceeded",
                "fields": {
                    "budget": "max_messages",
                    "limit": config.max_messages,
                    "observed": len(run.messages) + needs_messages,
                    "step_id": step_id,
                },
            }
        return None

    # -- task scheduling + step execution --------------------------------

    def execute(
        self,
        task: TaskPayload,
        plan: InvestigationPlan,
        role_overrides: Optional[Dict[str, str]] = None,
        role_assignments: Optional[Sequence[RoleAssignment]] = None,
    ) -> OrchestrationRun:
        """Run one plan end to end; ALWAYS returns a terminal-state
        record (raises ``ContractError`` only if the orchestrator
        itself would emit a malformed message — a bug, not an outcome;
        planner LLM failures happen before this method is called).

        Role input is EITHER ``role_overrides`` (the assigner applies
        its classifier and post-conditions on top) OR a fully
        preassigned ``role_assignments`` sequence (validated and used
        verbatim, the assigner never invoked) — supplying both raises
        ``ValueError``. Invalid role input of either kind raises
        ``ValueError`` before any agent runs; it is never repaired or
        defaulted silently."""
        if role_assignments is not None and role_overrides is not None:
            raise ValueError(
                "pass either role_overrides or role_assignments, "
                "not both"
            )
        run = OrchestrationRun(
            run_id=uuid.uuid4().hex,
            task=task,
            plan=plan,
            started_at=utcnow(),
        )
        started = self._clock()
        self._record(
            run,
            None,
            "orchestration.start",
            run_id=run.run_id,
            question=task.question,
            repository_scope=task.repository_scope,
            commit_scope=task.commit_scope,
        )
        self._mark(run, PLANNING)
        self._record(
            run,
            None,
            "orchestration.plan.created",
            objective=plan.objective,
            steps=[
                {"id": step.id, "goal": step.goal} for step in plan.steps
            ],
        )

        # -- plan gate: deterministic validation, then the step budget ---
        plan_error: Optional[str] = None
        try:
            validate_plan(plan)
        except PlanValidationError as exc:
            plan_error = str(exc)
        if plan_error is None and len(plan.steps) > self._config.max_plan_steps:
            plan_error = (
                "plan has {} steps; max_plan_steps is {}".format(
                    len(plan.steps), self._config.max_plan_steps
                )
            )
        if plan_error is not None:
            self._record(
                run,
                None,
                "orchestration.plan.validated",
                status="FAILED",
                errors=plan_error,
            )
            self._terminate(
                run,
                PLAN_FAILED,
                "PLAN_FAILED",
                "orchestration.failed",
                run_id=run.run_id,
                reason=plan_error,
            )
            return run
        self._record(
            run, None, "orchestration.plan.validated", status="SUCCESS"
        )
        self._mark(run, PLAN_VALIDATED)
        self._mark(run, EXECUTING)
        plan.status = _PLAN_IN_PROGRESS

        if role_assignments is not None:
            assignments = _preassigned_role_assignments(
                plan, role_assignments
            )
        else:
            assignments = self._role_assigner(plan.steps, role_overrides)
        plan_steps = {step.id: step for step in plan.steps}
        # M19: one authoritative task record per plan step, with the
        # deterministic sequential-chain dependencies; readiness is
        # evaluated by the pure scheduler (scheduling.py), dispatch
        # policy stays application-owned.
        dependencies = sequential_dependencies(
            [step.id for step in plan.steps]
        )
        run.tasks = [
            TaskRecord(
                task_id=step.id,
                role=assignment.role,
                status=TASK_PENDING,
                depends_on=list(dependencies[step.id]),
                assignment_reason=assignment.reason,
            )
            for step, assignment in zip(plan.steps, assignments)
        ]
        tasks_by_id = {
            record.task_id: record for record in run.tasks
        }
        assembler = RunAssembler()
        investigator_traces: List[Any] = []
        analyst_trace: Any = None
        has_investigation_request = False

        def ensure_merged() -> None:
            if run.report is not None:
                return
            reports = assembler.collect_investigator_reports(
                run.step_results
            )
            if reports:
                run.report = assembler.merge_reports(reports, task)

        def block_from(index: int, reason: str) -> None:
            self._block_remaining(
                run, assignments, index, reason, plan_steps
            )

        def fail_step(
            step: PlanStep,
            assignment: RoleAssignment,
            index: int,
            window: List[Dict[str, Any]],
            trace: Optional[Any],
            attempts: int,
            error: str,
            agent_run_id: Optional[str],
            failure_category: Optional[str] = None,
        ) -> None:
            self._record(
                run,
                window,
                "orchestration.step.failed",
                step_id=step.id,
                status="AGENT_FAILED",
                error=error,
            )
            self._close_window(window, trace)
            run.failed_steps.append(step.id)
            run.failures.append("{}: {}".format(step.id, error))
            run.step_results.append(
                StepResult(
                    step_id=step.id,
                    agent_role=assignment.role,
                    status=STEP_FAILED,
                    agent_run_id=agent_run_id,
                    attempts=attempts,
                    error=error,
                )
            )
            step.status = _PLAN_FAILED
            step.result = error
            record = tasks_by_id.get(step.id)
            if record is not None:
                record.status = TASK_FAILED
                record.attempts = attempts
                record.agent_run_id = agent_run_id
                record.failure_category = failure_category
                record.error = error
                record.completed_at = utcnow()
            ensure_merged()
            block_from(index + 1, "{}_FAILED".format(step.id))

        # -- the sequential step loop (application-owned, bounded) --------
        for index, (step, assignment) in enumerate(zip(plan.steps, assignments)):
            needs_messages = 0
            if assignment.role == INVESTIGATOR and not has_investigation_request:
                needs_messages = 1
            elif assignment.role == ANALYST:
                needs_messages = 3  # INV_RES + AREQ now, ARES on success

            gate = self._pre_step_gate(
                run, needs_messages, started, step.id
            )
            if gate is not None:
                block_from(index, "ORCHESTRATION_STOPPED")
                self._record(
                    run, None, gate["operation"], **gate["fields"]
                )
                ensure_merged()
                self._terminate(
                    run,
                    gate["status"],
                    gate["reason"],
                    "orchestration.terminated",
                    run_id=run.run_id,
                )
                return run

            run.current_step = step.id
            self._record(
                run, None, "orchestration.step.start",
                step_id=step.id, goal=step.goal,
            )
            # Step-level events go to this step's window from here on.
            window: List[Dict[str, Any]] = []
            self._record(
                run,
                window,
                "orchestration.step.assign",
                step_id=step.id,
                agent_role=assignment.role,
                assignment_reason=assignment.reason,
            )
            step.status = _PLAN_IN_PROGRESS

            if assignment.role == INVESTIGATOR and not has_investigation_request:
                self._append_message(
                    run, window, "INVESTIGATION_REQUEST",
                    "task", "investigator", task, step.id,
                )
                has_investigation_request = True

            if assignment.role == ANALYST:
                ensure_merged()
                if run.report is None:
                    # Unreachable: post-condition A guarantees an
                    # Investigator step, and a failed one terminates
                    # before analysis. Fail loudly rather than fabricate.
                    raise RuntimeError(
                        "no investigation report available for analysis"
                    )
                self._append_message(
                    run, window, "INVESTIGATION_RESULT",
                    "investigator", "analyst", run.report, step.id,
                )
                self._append_message(
                    run, window, "ANALYSIS_REQUEST",
                    "task", "analyst", run.report, step.id,
                )

            if assignment.role == INVESTIGATOR:
                question = _investigator_question(task, step)
                expected_type = InvestigationReport
            else:
                question = _analyst_question(task, run.report)
                expected_type = ArchitectureFinding

            # M19 readiness gate: the pure scheduler says whether this
            # task's dependencies completed (the sequential chain
            # guarantees it for plan-order dispatch; failing loudly
            # here rather than running a not-ready task).
            ready_ids = {
                record.task_id for record in ready_tasks(run.tasks)
            }
            if step.id not in ready_ids:
                raise RuntimeError(
                    "task {!r} is not ready: dependencies incomplete"
                    .format(step.id)
                )
            task_record = tasks_by_id[step.id]
            task_record.status = TASK_READY

            attempts = 0
            agent_run_id: Optional[str] = None
            while True:
                attempts += 1
                self._record(
                    run,
                    window,
                    "orchestration.agent.start",
                    step_id=step.id,
                    agent_role=assignment.role,
                    attempt=attempts,
                )
                task_record.status = TASK_RUNNING
                task_record.started_at = (
                    task_record.started_at or utcnow()
                )
                request = AgentTaskRequest(
                    run_id=run.run_id,
                    task_id=step.id,
                    attempt=attempts,
                    role=assignment.role,
                    assignment_reason=assignment.reason,
                    question=task.question,
                    goal=step.goal,
                    prompt=question,
                    repository_scope=task.repository_scope,
                    commit_scope=task.commit_scope,
                    depends_on=tuple(task_record.depends_on),
                    dependency_results=(
                        (run.report,)
                        if assignment.role == ANALYST
                        and run.report is not None
                        else ()
                    ),
                    # M20: the configured per-attempt tool limit (None
                    # = no task-level override — the participant's own
                    # configured budget applies, exactly M19).
                    max_tool_calls=self._config.max_task_tool_calls,
                )
                # Dispatch boundary: the executor owns participant
                # execution and returns an immutable result; the
                # orchestrator alone mutates run state below. The
                # result is first validated against ITS request — an
                # invalid result (contract_violation) is never
                # retried and contributes NO state (no output, usage,
                # trace, or correlation id).
                result = self._executor.execute(request)
                contract_error: Optional[str] = None
                try:
                    validate_task_result(request, result)
                except TaskContractError as exc:
                    contract_error = str(exc)
                except Exception as exc:
                    # Defensive: the validator is expected to raise
                    # only TaskContractError; ANY unexpected error
                    # here means the result cannot be trusted — the
                    # established contract promises execute() returns
                    # a terminal run result, so it becomes the same
                    # non-retryable contract_violation path (never an
                    # incidental exception escaping the run).
                    contract_error = (
                        "task result validation failed: {}: {}".format(
                            type(exc).__name__, exc
                        )
                    )
                output: Any = None
                failure: Optional[TaskFailure] = None
                agent_run_id: Optional[str] = None
                # The trace THIS attempt's step events mirror onto: the
                # result's own trace when the result is trusted; never
                # a participant's ``last_trace`` (an injected executor
                # may not own one), and nothing at all when the result
                # failed contract validation (its trace is untrusted).
                result_trace: Optional[Any] = None
                task_record.attempts = attempts
                if contract_error is None:
                    output = result.output
                    failure = result.failure
                    agent_run_id = result.agent_run_id
                    result_trace = result.trace
                    task_record.agent_run_id = agent_run_id
                    task_record.usage = result.usage
                    task_record.usage_total = combine_usage(
                        task_record.usage_total, result.usage
                    )
                    run.agent_runs.append(
                        {
                            "step_id": step.id,
                            "agent_role": assignment.role,
                            "attempt": attempts,
                            "run_id": agent_run_id,
                        }
                    )
                    run.tool_calls_total += result.usage.tool_calls
                    run.usage_total = combine_usage(
                        run.usage_total, result.usage
                    )
                    if assignment.role == INVESTIGATOR and (
                        result_trace is not None
                    ):
                        # EVERY attempt's evidence counts: a tool that
                        # ran on a failed first attempt was still
                        # inspected, so the merged evidence authority
                        # includes it (the successful report still
                        # decides WHICH refs the package claims).
                        investigator_traces.append(result_trace)

                failure_category: Optional[str] = (
                    failure.category if failure is not None else None
                )
                if contract_error is not None:
                    failure = TaskFailure(
                        category=FAILURE_CONTRACT_VIOLATION,
                        error_type="TaskContractError",
                        error_message=contract_error,
                        retryable=False,
                    )
                    failure_category = FAILURE_CONTRACT_VIOLATION
                    error = contract_error
                    retryable = False
                elif failure is None and isinstance(output, expected_type):
                    scope_ok = True
                    if isinstance(output, InvestigationReport):
                        scope_ok = (
                            output.repository_scope
                            == task.repository_scope
                            and output.commit_scope == task.commit_scope
                        )
                    if scope_ok:
                        self._record(
                            run,
                            window,
                            "orchestration.agent.complete",
                            step_id=step.id,
                            agent_role=assignment.role,
                            attempt=attempts,
                            run_id=agent_run_id,
                            status="completed",
                        )
                        self._close_window(window, result_trace)
                        break

                    failure = None
                    error = (
                        "SCOPE_MISMATCH: report scope {}@{} != task "
                        "scope {}@{}".format(
                            output.repository_scope,
                            output.commit_scope,
                            task.repository_scope,
                            task.commit_scope,
                        )
                    )
                    failure_category = FAILURE_SCOPE_MISMATCH
                    retryable = True
                elif failure is None:
                    error = "MISSING_STRUCTURED_OUTPUT: expected {}".format(
                        expected_type.__name__
                    )
                    failure_category = FAILURE_STRUCTURED_OUTPUT
                    retryable = True
                else:
                    error = failure.error_message
                    retryable = failure.retryable

                self._record(
                    run,
                    window,
                    "orchestration.agent.complete",
                    step_id=step.id,
                    agent_role=assignment.role,
                    attempt=attempts,
                    run_id=agent_run_id,
                    status="failed",
                    reason=(
                        failure.error_type if failure is not None else error
                    ),
                )
                self._close_window(window, result_trace)

                if failure_category == FAILURE_AGENT_BUDGET:
                    # The agent hit its OWN tool budget: retrying would
                    # only re-inflate a budget it already exhausted.
                    fail_step(
                        step, assignment, index, window, result_trace,
                        attempts, error, agent_run_id,
                        failure_category,
                    )
                    self._record(
                        run,
                        None,
                        "orchestration.budget.exceeded",
                        budget="agent_max_tool_calls",
                        # M20: report the task-level limit when one was
                        # dispatched (None = the participant's own
                        # configured budget, which the runner does not
                        # read) and the TRUSTED attempt's measured
                        # tool calls (blocked calls never executed and
                        # are never counted).
                        limit=request.max_tool_calls,
                        observed=(
                            task_record.usage.tool_calls
                            if task_record.usage is not None
                            else None
                        ),
                        step_id=step.id,
                    )
                    ensure_merged()
                    self._terminate(
                        run,
                        BUDGET_EXCEEDED,
                        "BUDGET_EXCEEDED",
                        "orchestration.terminated",
                        run_id=run.run_id,
                    )
                    return run

                if retryable and attempts <= self._config.max_retries:
                    # M20 per-attempt run-budget gate: a failed attempt
                    # may have consumed the remaining run budget (tool
                    # calls, agent runs, wall clock). Re-check the
                    # between-attempt bounds BEFORE deciding to retry:
                    # when none remains, the retry never starts, is
                    # never counted (``run.retries``), and the run
                    # terminates BUDGET_EXCEEDED with this step
                    # blocked. needs_messages=0: this step's handoffs
                    # were already appended (and pre-checked) above.
                    gate = self._pre_step_gate(run, 0, started, step.id)
                    if gate is not None:
                        block_from(index, "ORCHESTRATION_STOPPED")
                        self._record(
                            run,
                            None,
                            gate["operation"],
                            **gate["fields"],
                        )
                        ensure_merged()
                        self._terminate(
                            run,
                            gate["status"],
                            gate["reason"],
                            "orchestration.terminated",
                            run_id=run.run_id,
                        )
                        return run
                    run.retries += 1
                    self._record(
                        run,
                        window,
                        "orchestration.retry",
                        step_id=step.id,
                        attempt=attempts + 1,
                        reason=error,
                    )
                    continue

                fail_step(
                    step, assignment, index, window, result_trace,
                    attempts, error, agent_run_id, failure_category,
                )
                status = (
                    INCOMPLETE
                    if assignment.role == INVESTIGATOR
                    else AGENT_FAILED
                )
                self._terminate(
                    run,
                    status,
                    "AGENT_FAILED",
                    "orchestration.terminated",
                    run_id=run.run_id,
                )
                return run

            # -- step succeeded ------------------------------------------
            run.completed_steps.append(step.id)
            task_record.status = TASK_COMPLETED
            task_record.completed_at = utcnow()
            task_record.output = output
            task_record.evidence_refs = list(result.evidence_refs)
            task_record.failure_category = None
            task_record.error = None
            run.step_results.append(
                StepResult(
                    step_id=step.id,
                    agent_role=assignment.role,
                    status=STEP_COMPLETED,
                    output=output,
                    evidence_refs=list(result.evidence_refs),
                    agent_run_id=agent_run_id,
                    attempts=attempts,
                )
            )
            if assignment.role == INVESTIGATOR:
                has_investigation_request = True
            else:
                run.finding = output
                analyst_trace = result.trace

            if assignment.role == ANALYST:
                self._append_message(
                    run, window, "ANALYSIS_RESULT",
                    "analyst", "task", output, step.id,
                )
            artifact = getattr(output, "investigation_id", None)
            self._record(
                run,
                window,
                "orchestration.step.complete",
                step_id=step.id,
                status=STEP_COMPLETED,
                artifact=artifact,
            )
            self._close_window(window, result_trace)
            step.status = _PLAN_COMPLETED
            step.result = "{} {}".format(
                type(output).__name__, artifact or ""
            ).strip()

        # -- final assembly (explicit, deterministic — assembly.py) -------
        outcome = assembler.assemble(run.step_results, task, run.finding)
        if run.report is None:
            run.report = outcome.report
        if outcome.finding is None:
            # Single-step plan: only evidence collection ran — an
            # evidence package is preserved, never a fabricated
            # architectural answer (documented limitation).
            self._terminate(
                run,
                INCOMPLETE,
                "NO_ANALYSIS",
                "orchestration.terminated",
                run_id=run.run_id,
            )
            return run

        # -- validation gates ---------------------------------------------
        gate = self._pre_step_gate(run, 2, started, None)
        if gate is not None:
            self._record(run, None, gate["operation"], **gate["fields"])
            self._terminate(
                run,
                gate["status"],
                gate["reason"],
                "orchestration.terminated",
                run_id=run.run_id,
            )
            return run

        # -- deterministic validation (the M16 verdict, reused) -----------
        self._mark(run, VALIDATING)
        self._record(run, None, "orchestration.validation.start")
        self._append_message(
            run, None, "VALIDATION_REQUEST", "task", "validator",
            ValidationRequest(
                investigation_id=run.report.investigation_id,
                repository_scope=task.repository_scope,
                commit_scope=task.commit_scope,
            ),
            None,
        )
        merged_trace = {
            "iterations": [
                iteration.to_dict()
                for trace in investigator_traces
                for iteration in trace.iterations
            ]
        }
        verdict = validate_collaboration(
            task=task,
            report=run.report,
            finding=run.finding,
            investigator_trace=merged_trace,
            messages=run.messages,
            analyst_trace=analyst_trace,
        )
        run.validation = verdict
        run.validation_status = verdict.status
        self._append_message(
            run, None, "VALIDATION_RESULT", "validator", "task",
            verdict, None,
        )
        self._record(
            run,
            None,
            "orchestration.validation.complete",
            status=verdict.status,
            checks=[
                {"rule": check.rule, "ok": check.ok}
                for check in verdict.checks
            ],
        )
        if verdict.status != "passed":
            self._terminate(
                run,
                VALIDATION_FAILED,
                "VALIDATION_FAILED",
                "orchestration.failed",
                run_id=run.run_id,
                reason="deterministic validation rejected the result",
            )
            return run
        if run.report.status == "incomplete" or run.finding.status == "incomplete":
            self._terminate(
                run,
                INCOMPLETE,
                "DECLARED_INCOMPLETE",
                "orchestration.terminated",
                run_id=run.run_id,
            )
            return run
        self._terminate(
            run,
            COMPLETED,
            "SUCCESS",
            "orchestration.complete",
            run_id=run.run_id,
            completed_steps=len(run.completed_steps),
        )
        return run
