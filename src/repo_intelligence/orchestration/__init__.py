"""M17 Multi-Agent Orchestration — the plan-driven execution layer.

The orchestrator answers "How, when, and under what constraints should
the planned work be executed?" It takes the EXISTING M10 plan and the
EXISTING M16 agents — no second planner, no second agent framework:

    plan (planning.models.InvestigationPlan)
        → plan gate (planning.validation.validate_plan + step bound)
        → role assignment (deterministic, observable, application-owned)
        → sequential step loop (the application IS the loop — no
          while-True, no repository-specific branch)
            typed handoffs (the M16 six-message sequence)
            Agent.run_result (per-step structured output)
            bounded retry (never on validation failures, never endless)
            blocked downstream steps on failure
        → deterministic validation (the M16 validate_collaboration —
          one verdict, reused, never re-implemented)
        → one sealed terminal state (status.py — no restart, no
          further agent execution)

What this package is (spec §4–§27, extended M19):

* **status** — the 12-state run status machine with an explicit,
  sealed terminal set; every transition is checked.
* **roles** — deterministic INVESTIGATOR/ANALYST assignment with a
  human-readable reason per step; no repository knowledge.
* **contracts** — the M19 task boundary: immutable
  ``AgentTaskRequest``/``AgentTaskResult``, the ``TaskFailure``
  classification taxonomy (one place decides retryability), and
  ``TaskUsage`` accounting (app-owned, never invented).
* **executor** — the ``AgentExecutor`` Protocol plus the default
  ``AgentTaskExecutor`` adapter that runs an injected participant
  (M16 agent or specialist) for ONE task and classifies failures.
* **scheduling** — pure, deterministic readiness: the sequential-
  chain dependency mapping (the one place plan order becomes a
  dependency) and ``ready_tasks``.
* **models** — the single authoritative ``OrchestrationRun`` (status,
  current step, task records, step results, messages,
  report/finding/validation, events, retries, budgets) plus each
  step's ``StepResult`` and each task's ``TaskRecord``.
* **assembly** — explicit result assembly: investigator-report merge
  (moved from the runner) and ``RunAssembler`` pairing the merged
  report with the finding.
* **runner** — the application-owned ``Orchestrator``: plan gate,
  budgets (steps/agent runs/messages/tool calls/retries/runtime),
  the sequential step loop dispatching through the executor
  boundary, failure semantics, and event emission (run events
  mirrored onto each agent's own trace).

The domain core remains framework-neutral. The optional LangGraph adapter
is isolated in ``langgraph_runner`` and owns task-loop control flow while
reusing these contracts and lifecycle primitives. Parallel agents,
checkpoint/recovery, and a second evidence model remain out of scope.
"""

from __future__ import annotations

from .assembly import (
    AssemblyOutcome,
    RunAssembler,
    merge_investigation_reports,
)
from .contracts import (
    FAILURE_AGENT_BUDGET,
    FAILURE_CONTEXT_EXHAUSTED,
    FAILURE_CONTRACT_VIOLATION,
    FAILURE_DISPATCH_ERROR,
    FAILURE_FINALIZATION,
    FAILURE_INVALID_RESPONSE,
    FAILURE_SCOPE_MISMATCH,
    FAILURE_STRUCTURED_OUTPUT,
    FAILURE_UNEXPECTED,
    FAILURE_PROVIDER,
    TASK_BLOCKED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_INCOMPLETE,
    TASK_PENDING,
    TASK_READY,
    TASK_RUNNING,
    TASK_STATUSES,
    AgentTaskRequest,
    AgentTaskResult,
    TaskContractError,
    TaskFailure,
    TaskUsage,
    classify_exception,
    combine_usage,
    validate_task_result,
)
from .executor import (
    AgentExecutor,
    AgentTaskExecutor,
    ParticipantRegistry,
)
from .models import (
    STEP_BLOCKED,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_STATUSES,
    OrchestrationRun,
    StepResult,
    TaskRecord,
    utcnow,
)
from .roles import (
    AGENT_ROLES,
    ANALYST,
    INVESTIGATOR,
    AgentRole,
    RoleAssignment,
    assign_roles,
)
from .runner import (
    DEFAULT_MAX_AGENT_RUNS,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_PLAN_STEPS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOTAL_TOOL_CALLS,
    Orchestrator,
    OrchestrationConfig,
)
from .scheduling import ready_tasks, sequential_dependencies
from .status import (
    AGENT_FAILED,
    ALLOWED_TRANSITIONS,
    BUDGET_EXCEEDED,
    COMPLETED,
    EXECUTING,
    INCOMPLETE,
    PENDING,
    PLAN_FAILED,
    PLAN_VALIDATED,
    PLANNING,
    RUN_STATUSES,
    TERMINATION_REASONS,
    TIMEOUT,
    VALIDATING,
    VALIDATION_FAILED,
    OrchestrationStatus,
    OrchestrationTransitionError,
    advance,
    can_advance,
    is_terminal,
)

__all__ = [
    "AGENT_FAILED",
    "AGENT_ROLES",
    "ALLOWED_TRANSITIONS",
    "ANALYST",
    "BUDGET_EXCEEDED",
    "COMPLETED",
    "DEFAULT_MAX_AGENT_RUNS",
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MAX_PLAN_STEPS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOTAL_TOOL_CALLS",
    "EXECUTING",
    "FAILURE_AGENT_BUDGET",
    "FAILURE_CONTEXT_EXHAUSTED",
    "FAILURE_CONTRACT_VIOLATION",
    "FAILURE_DISPATCH_ERROR",
    "FAILURE_FINALIZATION",
    "FAILURE_INVALID_RESPONSE",
    "FAILURE_PROVIDER",
    "FAILURE_SCOPE_MISMATCH",
    "FAILURE_STRUCTURED_OUTPUT",
    "FAILURE_UNEXPECTED",
    "INCOMPLETE",
    "INVESTIGATOR",
    "PENDING",
    "PLAN_FAILED",
    "PLAN_VALIDATED",
    "PLANNING",
    "RUN_STATUSES",
    "STEP_BLOCKED",
    "STEP_COMPLETED",
    "STEP_FAILED",
    "STEP_STATUSES",
    "TASK_BLOCKED",
    "TASK_CANCELLED",
    "TASK_COMPLETED",
    "TASK_FAILED",
    "TASK_INCOMPLETE",
    "TASK_PENDING",
    "TASK_READY",
    "TASK_RUNNING",
    "TASK_STATUSES",
    "TERMINATION_REASONS",
    "TIMEOUT",
    "VALIDATING",
    "VALIDATION_FAILED",
    "AgentExecutor",
    "AgentRole",
    "AgentTaskExecutor",
    "AgentTaskRequest",
    "AgentTaskResult",
    "AssemblyOutcome",
    "OrchestrationConfig",
    "OrchestrationRun",
    "OrchestrationStatus",
    "OrchestrationTransitionError",
    "Orchestrator",
    "RoleAssignment",
    "RunAssembler",
    "StepResult",
    "TaskContractError",
    "TaskFailure",
    "TaskRecord",
    "TaskUsage",
    "advance",
    "assign_roles",
    "can_advance",
    "classify_exception",
    "combine_usage",
    "is_terminal",
    "merge_investigation_reports",
    "ready_tasks",
    "sequential_dependencies",
    "utcnow",
    "validate_task_result",
]
