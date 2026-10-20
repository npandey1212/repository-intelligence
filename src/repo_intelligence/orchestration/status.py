"""M17 orchestration status model (spec §25/§39 — explicit, minimal).

One lifecycle for the whole orchestrated run — distinct from the M16
task status (one collaboration) and from the M10 plan status (the
plan's own application-managed step lifecycle):

    PENDING → PLANNING → PLAN_VALIDATED → EXECUTING → VALIDATING
                   │             │             │            │
                   ▼             ▼             ▼            ▼
             PLAN_FAILED        …        AGENT_FAILED    COMPLETED
                                        INCOMPLETE       VALIDATION_FAILED
                                        TIMEOUT          INCOMPLETE
                                        BUDGET_EXCEEDED  BUDGET_EXCEEDED

Every state has a concrete behavioral meaning in ``runner.py``:

* ``PENDING``           — run constructed, no plan gate yet.
* ``PLANNING``          — the plan is at the deterministic gate
                          (``planning.validation.validate_plan``).
* ``PLAN_VALIDATED``    — the plan may execute; no step has started.
* ``EXECUTING``         — steps are running sequentially (or retrying).
* ``VALIDATING``        — every step completed; the deterministic
                          validator is deciding.
* ``COMPLETED``         — validation passed and nothing declared the
                          work incomplete (``termination SUCCESS``).
* ``PLAN_FAILED``       — invalid/over-long plan; NO agent executed.
* ``AGENT_FAILED``      — a step's Analyst produced no conclusion; no
                          fabricated architectural answer exists.
* ``INCOMPLETE``        — work stopped early but evidence may exist:
                          a failed Investigator step (downstream
                          blocked, §42), declared-incomplete outputs,
                          or a plan with no analysis step.
* ``TIMEOUT``           — the orchestration wall-clock budget was hit
                          between steps (never preempted mid-run).
* ``BUDGET_EXCEEDED``   — an orchestration bound was hit (plan steps,
                          agent runs, messages, total tool calls, an
                          agent's own execution limit).
* ``VALIDATION_FAILED`` — outputs exist but deterministic validation
                          rejected them; never converted to success.

Terminal states have no outgoing transitions: once terminal, no
further agent execution is permitted. Only the orchestrator moves the
status, and every move goes through ``advance`` so illegal transitions
are impossible rather than merely unlikely.
"""

from __future__ import annotations

from typing import FrozenSet

from typing_extensions import Literal

__all__ = [
    "AGENT_FAILED",
    "ALLOWED_TRANSITIONS",
    "BUDGET_EXCEEDED",
    "COMPLETED",
    "EXECUTING",
    "INCOMPLETE",
    "PENDING",
    "PLAN_FAILED",
    "PLAN_VALIDATED",
    "PLANNING",
    "RUN_STATUSES",
    "TIMEOUT",
    "VALIDATING",
    "VALIDATION_FAILED",
    "OrchestrationStatus",
    "OrchestrationTransitionError",
    "advance",
    "can_advance",
    "is_terminal",
]

OrchestrationStatus = Literal[
    "PENDING",
    "PLANNING",
    "PLAN_VALIDATED",
    "EXECUTING",
    "VALIDATING",
    "COMPLETED",
    "VALIDATION_FAILED",
    "PLAN_FAILED",
    "AGENT_FAILED",
    "TIMEOUT",
    "BUDGET_EXCEEDED",
    "INCOMPLETE",
]

PENDING: OrchestrationStatus = "PENDING"
PLANNING: OrchestrationStatus = "PLANNING"
PLAN_VALIDATED: OrchestrationStatus = "PLAN_VALIDATED"
EXECUTING: OrchestrationStatus = "EXECUTING"
VALIDATING: OrchestrationStatus = "VALIDATING"
COMPLETED: OrchestrationStatus = "COMPLETED"
VALIDATION_FAILED: OrchestrationStatus = "VALIDATION_FAILED"
PLAN_FAILED: OrchestrationStatus = "PLAN_FAILED"
AGENT_FAILED: OrchestrationStatus = "AGENT_FAILED"
TIMEOUT: OrchestrationStatus = "TIMEOUT"
BUDGET_EXCEEDED: OrchestrationStatus = "BUDGET_EXCEEDED"
INCOMPLETE: OrchestrationStatus = "INCOMPLETE"

RUN_STATUSES: FrozenSet[str] = frozenset(
    {
        PENDING,
        PLANNING,
        PLAN_VALIDATED,
        EXECUTING,
        VALIDATING,
        COMPLETED,
        VALIDATION_FAILED,
        PLAN_FAILED,
        AGENT_FAILED,
        TIMEOUT,
        BUDGET_EXCEEDED,
        INCOMPLETE,
    }
)

# The M17 lifecycle (spec §39): the happy path plus the explicit
# failure transitions. VALIDATING also permits INCOMPLETE (outputs
# declared incomplete after a passed verdict) and BUDGET_EXCEEDED
# (the message budget gate before the final handoff) — both are real
# execution outcomes of this runner, documented in the module above.
ALLOWED_TRANSITIONS = {
    PENDING: frozenset({PLANNING}),
    PLANNING: frozenset({PLAN_VALIDATED, PLAN_FAILED}),
    PLAN_VALIDATED: frozenset({EXECUTING}),
    EXECUTING: frozenset(
        {VALIDATING, AGENT_FAILED, INCOMPLETE, TIMEOUT, BUDGET_EXCEEDED}
    ),
    VALIDATING: frozenset(
        {COMPLETED, VALIDATION_FAILED, INCOMPLETE, BUDGET_EXCEEDED}
    ),
    COMPLETED: frozenset(),
    VALIDATION_FAILED: frozenset(),
    PLAN_FAILED: frozenset(),
    AGENT_FAILED: frozenset(),
    TIMEOUT: frozenset(),
    BUDGET_EXCEEDED: frozenset(),
    INCOMPLETE: frozenset(),
}

# Why the workflow stopped — a small explicit vocabulary (spec §41:
# termination_reason is always one of these, never free-form prose).
TERMINATION_REASONS = (
    "SUCCESS",
    "PLAN_FAILED",
    "AGENT_FAILED",
    "VALIDATION_FAILED",
    "TIMEOUT",
    "BUDGET_EXCEEDED",
    "DECLARED_INCOMPLETE",
    "NO_ANALYSIS",
)


class OrchestrationTransitionError(Exception):
    """An illegal orchestration status transition was attempted."""


def is_terminal(status: str) -> bool:
    return status in RUN_STATUSES and not ALLOWED_TRANSITIONS[status]


def can_advance(current: str, target: str) -> bool:
    return (
        current in ALLOWED_TRANSITIONS
        and target in ALLOWED_TRANSITIONS[current]
    )


def advance(current: str, target: str) -> str:
    """Return *target* if the transition is legal, else raise.

    The orchestrator's single status gate — illegal moves (double
    finish, restart after terminal, unknown states) fail loudly at the
    call site instead of corrupting the run record.
    """
    if current not in ALLOWED_TRANSITIONS:
        raise OrchestrationTransitionError(
            "unknown orchestration status {!r}".format(current)
        )
    if not can_advance(current, target):
        raise OrchestrationTransitionError(
            "illegal orchestration status transition {!r} -> {!r} "
            "(allowed from {!r}: {})".format(
                current,
                target,
                current,
                ", ".join(sorted(ALLOWED_TRANSITIONS[current]))
                or "(terminal)",
            )
        )
    return target
