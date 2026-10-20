"""``ContextEngine`` — the one selection authority (M15 §5).

The engine is deliberately thin: it validates the operation's scope,
hands the request to the pure policy (``policy.select_context``),
and — when asked — turns a selection into a persisted, inspectable
snapshot. It owns no database, no model call, no file format of its
own beyond one JSON projection per snapshot (M15 §15/§63):

    Engine ── select(task, candidates, budget) ──► ContextSelection
    Engine ── snapshot(selection, run, step)   ──► ContextSnapshot
    Engine ── persist(snapshot)                ──► runtime/context/
                                                   <run_id>/
                                                   context_<step>.json

Scope guard (M15 §4): an engine BOUND to a repository (the run it
was created for) refuses to select for another repository's task and
refuses to persist another repository's snapshot —
``CONTEXT_SCOPE_MISMATCH``, raised at the operation, never silently
ignored. Candidates of other repositories still pass through the
policy so their exclusion is OBSERVABLE (``OUT_OF_SCOPE`` decisions);
the binding guard is about the operation itself crossing scopes.

Determinism (M15 §30): same task/candidates/budget → same items,
same order, same reasons, same decision list. Only ids and
timestamps vary between invocations.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union

from ..artifacts import PROJECT_ROOT
from .errors import (
    CONTEXT_CONTRACT_VIOLATION,
    CONTEXT_SCOPE_MISMATCH,
    CONTEXT_SNAPSHOT_FAILURE,
    ContextOperationError,
)
from .models import (
    ContextBudget,
    ContextCandidate,
    ContextSelection,
    ContextSnapshot,
    TaskContext,
    new_snapshot_id,
)
from .policy import select_context

__all__ = [
    "ContextEngine",
    "default_context_root",
]

RUNTIME_DIR_NAME = "runtime"
CONTEXT_DIR_NAME = "context"

# Path-safe run/step identifiers: no separators, no leading dot —
# the persisted path is built from caller-supplied ids, so anything
# that could escape the runtime tree is rejected as a contract
# violation (never sanitized silently).
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def default_context_root() -> Path:
    """Default snapshot tree: ``<project-root>/runtime/context``.

    Runtime state, never ``artifacts/`` (M15 §7 — same rule as the
    memory database and the trace root): regenerable operational
    output, git-ignored."""
    return PROJECT_ROOT / RUNTIME_DIR_NAME / CONTEXT_DIR_NAME


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "{} must be a non-empty string, got {!r}".format(label, value),
        )
    if not _SAFE_COMPONENT.match(value):
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "{} must be a path-safe identifier ([A-Za-z0-9._-], no "
            "separators), got {!r}".format(label, value),
        )
    return value


def _utcnow() -> str:
    """UTC timestamp, ISO-8601 with millisecond precision — the same
    representation as the trace and the memory store."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


class ContextEngine:
    """Deterministic context selection for one repository/run scope.

    Constructor takes only policy configuration — a default ``budget``
    (overridable per call), the optional ``repository`` this engine is
    BOUND to (the isolation guard above), and an optional
    ``snapshot_root`` for ``persist`` (None → the default runtime
    tree). No LLM client, no dispatcher, no store: composition roots
    inject whatever produced the candidates (M15 §5).
    """

    def __init__(
        self,
        budget: Optional[ContextBudget] = None,
        repository: Optional[str] = None,
        snapshot_root: Optional[Union[str, Path]] = None,
    ) -> None:
        if budget is not None and not isinstance(budget, ContextBudget):
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "budget must be a ContextBudget, got {!r}".format(
                    type(budget).__name__
                ),
            )
        if repository is not None and (
            not isinstance(repository, str) or not repository.strip()
        ):
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "repository must be None or a non-empty string, got "
                "{!r}".format(repository),
            )
        self._budget = budget if budget is not None else ContextBudget()
        self._repository = repository
        self._snapshot_root = (
            Path(snapshot_root) if snapshot_root is not None else None
        )

    @property
    def budget(self) -> ContextBudget:
        """The default budget used when a call passes none."""
        return self._budget

    @property
    def repository(self) -> Optional[str]:
        """The repository this engine is bound to (None = any)."""
        return self._repository

    @property
    def snapshot_root(self) -> Path:
        """Where ``persist`` writes (explicit — never guessed)."""
        if self._snapshot_root is not None:
            return self._snapshot_root
        return default_context_root()

    def select(
        self,
        task: TaskContext,
        candidates: Sequence[ContextCandidate],
        budget: Optional[ContextBudget] = None,
        step_id: str = "001",
    ) -> ContextSelection:
        """Run the documented selection pipeline (M15 §7) for one
        reasoning step and return the package + every decision.

        *step_id* identifies WHICH reasoning step this package serves
        (the default "001" = the run's initial context); it becomes
        part of the package so a later snapshot can never mislabel
        which step its contents belonged to.
        """
        if not isinstance(task, TaskContext):
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "task must be a TaskContext, got {!r}".format(
                    type(task).__name__
                ),
            )
        if self._repository is not None and task.repository is not None:
            if task.repository != self._repository:
                raise ContextOperationError(
                    CONTEXT_SCOPE_MISMATCH,
                    "engine is bound to repository {!r} but the task "
                    "is scoped to {!r} — refusing to select a "
                    "cross-repository context (M15 isolation "
                    "invariant)".format(self._repository, task.repository),
                )
        effective = budget if budget is not None else self._budget
        return select_context(task, candidates, effective, step_id)

    def snapshot(
        self,
        selection: ContextSelection,
        run_id: str,
        step_id: str,
    ) -> ContextSnapshot:
        """Bind a selection to its run/step identity (M15 §11).

        The snapshot's step MUST be the step the package was selected
        for — a package cannot be relabeled into another step's
        record."""
        if not isinstance(selection, ContextSelection):
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "selection must be a ContextSelection, got {!r}".format(
                    type(selection).__name__
                ),
            )
        _safe_component(run_id, "run_id")
        _safe_component(step_id, "step_id")
        if step_id != selection.package.step_id:
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "snapshot step_id {!r} does not match the package's "
                "step {!r} — a snapshot may only record the step its "
                "package was selected for".format(
                    step_id, selection.package.step_id
                ),
            )
        if self._repository is not None and (
            selection.package.repository_scope is not None
            and selection.package.repository_scope != self._repository
        ):
            raise ContextOperationError(
                CONTEXT_SCOPE_MISMATCH,
                "engine is bound to repository {!r} but the selection "
                "is scoped to {!r} — refusing to snapshot a "
                "cross-repository context (M15 isolation "
                "invariant)".format(
                    self._repository, selection.package.repository_scope
                ),
            )
        return ContextSnapshot(
            snapshot_id=new_snapshot_id(),
            run_id=run_id,
            step_id=step_id,
            created_at=_utcnow(),
            package=selection.package,
            selection_decisions=selection.decisions,
            excluded_candidates=selection.excluded_candidate_ids,
        )

    def select_snapshot(
        self,
        task: TaskContext,
        candidates: Sequence[ContextCandidate],
        run_id: str,
        step_id: str,
        budget: Optional[ContextBudget] = None,
    ) -> ContextSnapshot:
        """Convenience: select + snapshot in one call (the demo's
        per-step flow — same pipeline, no extra state)."""
        selection = self.select(task, candidates, budget=budget,
                                step_id=step_id)
        return self.snapshot(selection, run_id, step_id)

    def persist(self, snapshot: ContextSnapshot) -> Path:
        """Write one snapshot as JSON (M15 §11/§42)::

            <snapshot_root>/<run_id>/context_<step_id>.json

        The file IS ``snapshot.to_dict()`` (canonical projection) —
        what a reader inspects is byte-for-byte the contract, not a
        view of it. Returns the written path; an I/O failure raises
        ``CONTEXT_SNAPSHOT_FAILURE`` (never a silently skipped
        artifact)."""
        if not isinstance(snapshot, ContextSnapshot):
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "snapshot must be a ContextSnapshot, got {!r}".format(
                    type(snapshot).__name__
                ),
            )
        run_id = _safe_component(snapshot.run_id, "run_id")
        step_id = _safe_component(snapshot.step_id, "step_id")
        if self._repository is not None and (
            snapshot.repository_scope is not None
            and snapshot.repository_scope != self._repository
        ):
            raise ContextOperationError(
                CONTEXT_SCOPE_MISMATCH,
                "engine is bound to repository {!r} but the snapshot "
                "is scoped to {!r} — refusing to persist a "
                "cross-repository context (M15 isolation "
                "invariant)".format(self._repository,
                                    snapshot.repository_scope),
            )
        target_dir = self.snapshot_root / run_id
        target = target_dir / "context_{}.json".format(step_id)
        payload = snapshot.to_dict()
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ContextOperationError(
                CONTEXT_SNAPSHOT_FAILURE,
                "snapshot could not be written to {}: {}".format(
                    target, exc
                ),
            )
        return target
