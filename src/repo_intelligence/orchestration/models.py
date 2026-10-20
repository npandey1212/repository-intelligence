"""M17 orchestration state (spec §12/§13/§31 — smallest explicit run).

``OrchestrationRun`` is the authoritative record of ONE orchestrated
workflow — distinct from M16's ``MultiAgentTaskState`` (one
collaboration) and from the ``InvestigationPlan`` (the goal-level
plan this run executes):

    OrchestrationRun
    ├── run_id / task / plan
    ├── status + termination_reason      (status.advance-gated)
    ├── current/completed/failed/blocked step ids
    ├── tasks                            (M19 authoritative task state)
    ├── step_results                     (typed per-step artifacts)
    ├── agent_runs                       (attempt-level correlation)
    ├── messages                         (typed handoffs)
    ├── report / finding / validation    (the final M16 contracts)
    └── events                           (sequence-numbered orchestration log)

Hierarchy (spec §13): a step's ``StepResult.output`` holds an M16
typed contract (``InvestigationReport``/``ArchitectureFinding``); the
run-level ``validation`` IS the M16 collaboration verdict over the
merged report and the final finding. Nothing here duplicates the
agents' ``AgentTrace`` — ``agent_runs``/``StepResult.agent_run_id``
carry the correlation ids, and the traces stay on disk.

No field is model-authored: the orchestrator alone moves ``status``
and appends ``events`` (monotonic ``sequence``; timestamps are
informational).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    ReportEvidenceRef,
    TaskPayload,
    ValidationResult,
)
from ..planning.models import InvestigationPlan
from .contracts import (
    TASK_BLOCKED,
    TASK_PENDING,
    TaskUsage,
)
from .status import PENDING

__all__ = [
    "STEP_STATUSES",
    "STEP_BLOCKED",
    "STEP_COMPLETED",
    "STEP_FAILED",
    "OrchestrationRun",
    "StepResult",
    "TaskRecord",
    "utcnow",
]

STEP_COMPLETED = "completed"
STEP_FAILED = "failed"
STEP_BLOCKED = "blocked"

STEP_STATUSES = (STEP_COMPLETED, STEP_FAILED, STEP_BLOCKED)


def utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass
class StepResult:
    """One completed/failed/blocked plan step's structured result
    (spec §31) — the orchestrator's typed per-step artifact record."""

    step_id: str
    agent_role: str
    status: str
    output: Optional[Union[InvestigationReport, ArchitectureFinding]] = None
    evidence_refs: List[ReportEvidenceRef] = field(default_factory=list)
    agent_run_id: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None

    @property
    def investigation_id(self) -> Optional[str]:
        output = self.output
        if isinstance(output, (InvestigationReport, ArchitectureFinding)):
            return output.investigation_id
        return None


@dataclass
class TaskRecord:
    """Authoritative orchestration state of ONE scheduled task (M19).

    One record per plan step, created by the coordinator after plan
    validation and role resolution; ONLY the coordinator mutates it
    (agent executions return ``AgentTaskResult``s and never touch run
    state). Status uses the ``TASK_*`` vocabulary from
    ``orchestration.contracts`` (default ``pending``).

    Single-source-of-truth rule: ``TaskRecord`` is authoritative for
    orchestration task state (status, attempts, failure,
    usage/usage_total, trace id); ``StepResult`` stays the M17
    typed-result projection and ``PlanStep.status`` the plan-lifecycle
    projection — both written by the same coordinator commit path, so
    they can never disagree with the record that produced them.
    ``AgentTrace`` stays the detailed execution record; only its id
    and aggregate usage live here.
    """

    task_id: str
    role: str
    status: str = TASK_PENDING
    depends_on: List[str] = field(default_factory=list)
    assignment_reason: str = ""
    attempts: int = 0
    agent_run_id: Optional[str] = None
    # The task's typed output payload: an M16 InvestigationReport /
    # ArchitectureFinding, or a specialist contract
    # (ArchitectureReport / FlowDependencyReport / DiagramSpec) when a
    # specialist participant runs through the same task boundary —
    # typed, never a free-form dictionary.
    output: Optional[Any] = None
    evidence_refs: List[ReportEvidenceRef] = field(default_factory=list)
    # Most recent attempt's measured usage (None until the first
    # trusted result commits).
    usage: Optional[TaskUsage] = None
    # Cumulative measured usage across ALL attempts of this task
    # (contracts.combine_usage — retries sum honestly; None while no
    # trusted result has committed).
    usage_total: Optional[TaskUsage] = None
    failure_category: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class OrchestrationRun:
    """The authoritative orchestration record for one run.

    ``events`` carries monotonic ``sequence`` numbers (timestamps are
    informational, generated by the orchestrator's clock); ``status``
    only moves through ``orchestration.status.advance``.
    """

    run_id: str
    task: TaskPayload
    plan: InvestigationPlan
    status: str = PENDING
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    blocked_steps: List[str] = field(default_factory=list)
    # M19: authoritative per-task state table (empty for runs created
    # before task scheduling existed — additive field).
    tasks: List[TaskRecord] = field(default_factory=list)
    step_results: List[StepResult] = field(default_factory=list)
    agent_runs: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    messages: List[Any] = field(default_factory=list)
    report: Optional[InvestigationReport] = None
    finding: Optional[ArchitectureFinding] = None
    validation: Optional[ValidationResult] = None
    validation_status: Optional[str] = None
    termination_reason: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    tool_calls_total: int = 0
    # M19.1: cumulative measured usage across every committed attempt
    # of the run (contracts.combine_usage; None while no trusted
    # result has committed). Budgets keep using tool_calls_total.
    usage_total: Optional[TaskUsage] = None
    started_at: str = ""
    completed_at: str = ""
    _sequence: int = field(default=0, repr=False)

    @property
    def final(self) -> bool:
        """True once the run reached a terminal status."""
        from .status import is_terminal

        return is_terminal(self.status)

    def step_result(self, step_id: str) -> Optional[StepResult]:
        for result in self.step_results:
            if result.step_id == step_id:
                return result
        return None

    def event_operations(self) -> List[str]:
        """The event timeline as operation names, in order."""
        return [event.get("operation", "") for event in self.events]
