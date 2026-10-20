"""M16 multi-agent task status model (spec §21 — explicit, minimal).

One lifecycle for the whole sequential collaboration:

    PENDING → RUNNING → COMPLETED | INCOMPLETE | FAILED |
                        VALIDATION_FAILED | BUDGET_EXCEEDED | TIMEOUT

Every state has a concrete behavioral meaning in ``runner.py``:

* ``PENDING``          — task state constructed, no agent started.
* ``RUNNING``          — a stage (investigator / analyst / validator)
                         is executing.
* ``COMPLETED``        — validator passed; report, analysis, and
                         validation result all present.
* ``INCOMPLETE``       — the workflow produced its outputs but a
                         participant declared the work incomplete
                         (``report.status``/``finding.status`` or an
                         empty evidence package) — never presented as
                         a full success.
* ``FAILED``           — an agent raised; no fabricated result.
* ``VALIDATION_FAILED``— outputs exist but deterministic collaboration
                         validation rejected them.
* ``BUDGET_EXCEEDED``  — a configured bound was hit (tool budget,
                         message/handoff cap).
* ``TIMEOUT``          — a stage exceeded its wall-clock deadline.

Terminal states have no outgoing transitions. No retries exist in
M16 (the project has no retry infrastructure), so no retry states
exist either. Nothing here is model-authored: only the runner moves
the status, and every move goes through ``advance`` so illegal
transitions are impossible rather than merely unlikely.
"""

from __future__ import annotations

from typing import FrozenSet

from typing_extensions import Literal

__all__ = [
    "ALLOWED_TRANSITIONS",
    "BUDGET_EXCEEDED",
    "COMPLETED",
    "FAILED",
    "INCOMPLETE",
    "PENDING",
    "RUNNING",
    "TASK_STATUSES",
    "TIMEOUT",
    "VALIDATION_FAILED",
    "StatusTransitionError",
    "TaskStatus",
    "advance",
    "can_advance",
    "is_terminal",
]

TaskStatus = Literal[
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "INCOMPLETE",
    "FAILED",
    "VALIDATION_FAILED",
    "BUDGET_EXCEEDED",
    "TIMEOUT",
]

PENDING: TaskStatus = "PENDING"
RUNNING: TaskStatus = "RUNNING"
COMPLETED: TaskStatus = "COMPLETED"
INCOMPLETE: TaskStatus = "INCOMPLETE"
FAILED: TaskStatus = "FAILED"
VALIDATION_FAILED: TaskStatus = "VALIDATION_FAILED"
BUDGET_EXCEEDED: TaskStatus = "BUDGET_EXCEEDED"
TIMEOUT: TaskStatus = "TIMEOUT"

TASK_STATUSES: FrozenSet[str] = frozenset(
    {
        PENDING,
        RUNNING,
        COMPLETED,
        INCOMPLETE,
        FAILED,
        VALIDATION_FAILED,
        BUDGET_EXCEEDED,
        TIMEOUT,
    }
)

# Only the running stage may end in a terminal state; only PENDING
# may start. Everything else is refused.
ALLOWED_TRANSITIONS = {
    PENDING: frozenset({RUNNING}),
    RUNNING: frozenset(
        {COMPLETED, INCOMPLETE, FAILED, VALIDATION_FAILED,
         BUDGET_EXCEEDED, TIMEOUT}
    ),
    COMPLETED: frozenset(),
    INCOMPLETE: frozenset(),
    FAILED: frozenset(),
    VALIDATION_FAILED: frozenset(),
    BUDGET_EXCEEDED: frozenset(),
    TIMEOUT: frozenset(),
}


class StatusTransitionError(Exception):
    """An illegal multi-agent status transition was attempted."""


def is_terminal(status: str) -> bool:
    return status in TASK_STATUSES and not ALLOWED_TRANSITIONS[status]


def can_advance(current: str, target: str) -> bool:
    return (
        current in ALLOWED_TRANSITIONS
        and target in ALLOWED_TRANSITIONS[current]
    )


def advance(current: str, target: str) -> str:
    """Return *target* if the transition is legal, else raise.

    The runner's single status gate — illegal moves (double finish,
    restart after terminal, unknown states) fail loudly at the call
    site instead of corrupting the task record.
    """
    if current not in ALLOWED_TRANSITIONS:
        raise StatusTransitionError(
            "unknown task status {!r}".format(current)
        )
    if not can_advance(current, target):
        raise StatusTransitionError(
            "illegal task status transition {!r} -> {!r} "
            "(allowed from {!r}: {})".format(
                current,
                target,
                current,
                ", ".join(sorted(ALLOWED_TRANSITIONS[current]))
                or "(terminal)",
            )
        )
    return target
