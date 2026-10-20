"""Memory Manager — the one application API over persistent memory.

Owns (M14 §19): contract validation, memory-ID generation,
persistence orchestration, deterministic retrieval, lifecycle
handling, provenance/scope preservation, structured errors.

Does NOT own: LLM reasoning, planning, ReAct, repository analysis,
truth determination, orchestration (M14 §19/§20). The Agent talks to
THIS, never to SQLite:

    Agent ── candidate / retrieval request ──► MemoryManager
                                                │ validate
                                                │ generate id
                                                │ query / persist
                                                │ lifecycle
                                                ▼
                                             SQLite

API (deterministic, provider-neutral — no LLM anywhere in this
package):

    initialize()                    create schema (idempotent)
    store(candidate) -> Memory      validate + generate mem_ id + persist
    get(memory_id) -> Memory        known-id lookup (MEMORY_NOT_FOUND)
    search(...) -> Tuple[Memory]    bounded deterministic discovery
    verify(memory_id, current)      freshness check (M14 §26)
        -> MemoryVerification
    supersede(memory_id, replace)   explicit replacement (M14 §28)
        -> Memory
    invalidate(memory_id) -> Memory explicit invalidation

search() is the NORMAL discovery mechanism — the agent asks "what is
relevant here" (repository/subject/owner/type/status), never "give me
mem-abc" (M14 §13); get() exists for known ids (inspection,
lifecycle).

verify() semantics (M14 §26, locked): scope comparison ONLY — the
result is TRI-STATE: ``fresh`` (both commits known and equal),
``stale`` (both known and differ), or ``unknown`` (a commit is
missing on either side, so the scopes cannot be compared). Unknown is
never reported as fresh — freshness must be positively established.
A fresh (scope-matching) verification RECORDS the check in
``last_verified_at`` ("last time this memory's scope was confirmed
against the current repository state" — not a content re-proof, which
would require re-running the investigation). Stale and unknown change
nothing: retention, superseding, and invalidation remain explicit
caller decisions — stale memory is never silently upgraded to current
and never silently rewritten.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

from .errors import (
    MEMORY_INVALID,
    MEMORY_NOT_FOUND,
    MEMORY_SCOPE_MISMATCH,
    MemoryOperationError,
)
from .models import (
    DEFAULT_SEARCH_LIMIT,
    MEMORY_STATUS_INVALID,
    MEMORY_STATUS_SUPERSEDED,
    MEMORY_STATUS_VERIFIED,
    MEMORY_TYPES,
    MEMORY_VERIFICATION_FRESH,
    MEMORY_VERIFICATION_STALE,
    MEMORY_VERIFICATION_UNKNOWN,
    Memory,
    MemoryCandidate,
    MemoryOwner,
    MemoryVerification,
    validate_candidate,
)
from .store import MemoryStore, new_memory_id

__all__ = [
    "MemoryManager",
    "is_stale",
    "verification_status",
]


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def verification_status(
    memory: Memory, current_commit: Optional[str]
) -> str:
    """Pure TRI-STATE freshness classification (M14 §26):

    * ``fresh``   — both commits are known and equal;
    * ``stale``   — both commits are known and differ;
    * ``unknown`` — a commit is missing on either side, so the scopes
      cannot be compared. Unknown is NEVER classified fresh — absence
      of a comparison is not evidence of freshness.

    The pinned commit is the scope anchor: verification compares the
    memory's repository scope against the caller's current repository
    state through it. It says nothing about the statement's semantic
    truth (that would require re-running the investigation).
    """
    scope_commit = memory.scope.commit
    if not isinstance(scope_commit, str) or not scope_commit:
        return MEMORY_VERIFICATION_UNKNOWN
    if not isinstance(current_commit, str) or not current_commit:
        return MEMORY_VERIFICATION_UNKNOWN
    if scope_commit == current_commit:
        return MEMORY_VERIFICATION_FRESH
    return MEMORY_VERIFICATION_STALE


def is_stale(memory: Memory, current_commit: Optional[str]) -> bool:
    """Convenience staleness predicate (M14 §26): True only when the
    memory's scope commit and the caller's current commit are BOTH
    known and differ.

    Compatibility note: ``is_stale(...) is False`` means "not proven
    stale" — it covers BOTH ``fresh`` and ``unknown``. Use
    ``verification_status(...)`` wherever the distinction matters;
    unknown must never be read as fresh.
    """
    return verification_status(memory, current_commit) == (
        MEMORY_VERIFICATION_STALE
    )


class MemoryManager:
    """Contract + lifecycle above the SQLite store (M14 §19–§20).

    Constructor takes only a database path (``None`` → the default
    ``db_store/memory/agent_memory.db``) — storage detail stays
    HERE, below the manager; the Agent holds this object and never
    imports ``sqlite3``.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self._store = MemoryStore(db_path)

    @property
    def db_path(self) -> Path:
        return self._store.db_path

    def initialize(self) -> None:
        """Deterministic, idempotent schema creation (M14 §30)."""
        self._store.initialize()

    # -- formation ---------------------------------------------------------

    def store(self, candidate: MemoryCandidate) -> Memory:
        """Validate the contract, generate the id, persist, return.

        Flow (M14 §12): candidate → validate → generate ``mem_<uuid>``
        → SQLite → return the stored Memory (the agent receives its
        ``memory_id`` from HERE — it never invents one). The stored
        status is always ``verified``: a candidate reaches this method
        only through the deterministic memory formation policy (M14
        §18 — SUPPORTED claims, bounded selection; never an LLM
        judgment about what deserves memory).
        """
        validate_candidate(candidate)
        memory = Memory(
            memory_id=new_memory_id(),
            owner=candidate.owner,
            memory_type=candidate.memory_type,
            statement=candidate.statement,
            scope=candidate.scope,
            evidence=candidate.evidence,
            status=MEMORY_STATUS_VERIFIED,
            created_at=_utcnow(),
            last_verified_at=_utcnow(),
            subject=candidate.subject,
            source_run_id=candidate.source_run_id,
            supersedes=None,
            superseded_by=None,
        )
        self._store.insert(memory)
        return memory

    # -- lookup ------------------------------------------------------------

    def get(self, memory_id: str) -> Memory:
        """Known-id lookup (M14 §13 — NOT the discovery mechanism).
        Raises ``MEMORY_NOT_FOUND``; never returns a default."""
        if not isinstance(memory_id, str) or not memory_id:
            raise MemoryOperationError(
                MEMORY_NOT_FOUND,
                "memory_id must be a non-empty string, got {!r}".format(
                    memory_id
                ),
            )
        memory = self._store.fetch(memory_id)
        if memory is None:
            raise MemoryOperationError(
                MEMORY_NOT_FOUND,
                "memory {!r} not found".format(memory_id),
            )
        return memory

    def search(
        self,
        repository: Optional[str] = None,
        commit: Optional[str] = None,
        owner: Optional[MemoryOwner] = None,
        memory_type: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> Tuple[Memory, ...]:
        """Deterministic discovery over exact/controlled dimensions
        (M14 §24): repository, commit, owner, memory_type, subject,
        status. ``None`` = dimension not filtered; results are ordered
        ``(created_at, memory_id)`` and bounded by *limit* — no
        embeddings, no ranking, no LLM, no "best match" pretense."""
        if memory_type is not None and memory_type not in MEMORY_TYPES:
            raise ValueError(
                "memory_type must be one of {}, got {!r}".format(
                    ", ".join(MEMORY_TYPES), memory_type
                )
            )
        if status is not None and status not in (
            MEMORY_STATUS_VERIFIED,
            MEMORY_STATUS_SUPERSEDED,
            MEMORY_STATUS_INVALID,
        ):
            raise ValueError(
                "status must be verified|superseded|invalid, got {!r}".format(
                    status
                )
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError(
                "limit must be an integer >= 1, got {!r}".format(limit)
            )
        return tuple(
            self._store.query(
                repository=repository,
                commit=commit,
                owner=owner,
                memory_type=memory_type,
                subject=subject,
                status=status,
                limit=limit,
            )
        )

    # -- lifecycle ---------------------------------------------------------

    def verify(
        self, memory_id: str, current_commit: Optional[str]
    ) -> MemoryVerification:
        """Check one memory's repository scope against the commit the
        caller works from — TRI-STATE (M14 §26):

        * ``fresh``   — both commits known and equal: ``stale=False``
          and ``last_verified_at`` is bumped (the scope check was
          positively established and recorded);
        * ``stale``   — both known and differ: ``stale=True``; NOTHING
          is written (staleness is reported, never resolved silently);
        * ``unknown`` — *current_commit* missing (``None``/``""``) or
          the memory carries no scope commit: freshness could not be
          determined; ``stale=False`` but NEVER reported as fresh, and
          no timestamp is invented for a check that never happened.

        The caller decides what stale means (retain / supersede /
        invalidate — M14 §26); this method only establishes and
        records scope freshness. It never re-proves the statement.
        """
        if current_commit is not None and not isinstance(
            current_commit, str
        ):
            raise ValueError(
                "current_commit must be None or a string, got {!r}".format(
                    current_commit
                )
            )
        memory = self.get(memory_id)
        outcome = verification_status(memory, current_commit)
        if outcome == MEMORY_VERIFICATION_FRESH:
            # Scope-matching verification: record the check (freshness
            # was positively established — both commits known+equal).
            verified_at = _utcnow()
            self._store.mark_verified(memory_id, verified_at)
            memory = Memory(
                memory_id=memory.memory_id,
                owner=memory.owner,
                memory_type=memory.memory_type,
                statement=memory.statement,
                scope=memory.scope,
                evidence=memory.evidence,
                status=memory.status,
                created_at=memory.created_at,
                last_verified_at=verified_at,
                subject=memory.subject,
                source_run_id=memory.source_run_id,
                supersedes=memory.supersedes,
                superseded_by=memory.superseded_by,
            )
        return MemoryVerification(
            memory_id=memory.memory_id,
            status=memory.status,
            scope_commit=memory.scope.commit,
            current_commit=current_commit,
            verification_status=outcome,
            stale=outcome == MEMORY_VERIFICATION_STALE,
        )

    def supersede(
        self, memory_id: str, replacement: MemoryCandidate
    ) -> Memory:
        """Replace *memory_id* with a NEW memory (M14 §28).

        The old row is kept and flipped to ``superseded`` with
        ``superseded_by = <new id>``; the new row carries
        ``supersedes = <old id>`` — one inspectable pointer each way,
        no version graph. The replacement must belong to the SAME
        repository scope (otherwise ``MEMORY_SCOPE_MISMATCH``); its
        commit may differ — that difference is the usual reason for
        superseding. Superseding an already superseded or invalid
        memory raises ``MEMORY_INVALID`` (lifecycle stays simple and
        legible). Returns the stored replacement.
        """
        validate_candidate(replacement)
        old = self.get(memory_id)
        if old.status in (MEMORY_STATUS_SUPERSEDED, MEMORY_STATUS_INVALID):
            raise MemoryOperationError(
                MEMORY_INVALID,
                "cannot supersede memory {} in status {!r}".format(
                    memory_id, old.status
                ),
            )
        old_repository = old.scope.repository
        new_repository = replacement.scope.repository
        if (
            old_repository is not None
            and new_repository is not None
            and old_repository != new_repository
        ):
            raise MemoryOperationError(
                MEMORY_SCOPE_MISMATCH,
                "replacement scope repository {!r} does not match memory "
                "{!r} scope repository {!r}".format(
                    new_repository, memory_id, old_repository
                ),
            )
        now = _utcnow()
        stored = Memory(
            memory_id=new_memory_id(),
            owner=replacement.owner,
            memory_type=replacement.memory_type,
            statement=replacement.statement,
            scope=replacement.scope,
            evidence=replacement.evidence,
            status=MEMORY_STATUS_VERIFIED,
            created_at=now,
            last_verified_at=now,
            subject=replacement.subject,
            source_run_id=replacement.source_run_id,
            supersedes=memory_id,
            superseded_by=None,
        )
        self._store.insert(stored)
        self._store.mark_superseded(memory_id, stored.memory_id)
        return stored

    def invalidate(self, memory_id: str) -> Memory:
        """Mark a memory ``invalid`` — retained and inspectable, never
        deleted (M14 §27). Idempotent on an already-invalid memory;
        a superseded memory can also be invalidated (both relationships
        preserved)."""
        memory = self.get(memory_id)
        if memory.status == MEMORY_STATUS_INVALID:
            return memory
        self._store.set_status(memory_id, MEMORY_STATUS_INVALID)
        return Memory(
            memory_id=memory.memory_id,
            owner=memory.owner,
            memory_type=memory.memory_type,
            statement=memory.statement,
            scope=memory.scope,
            evidence=memory.evidence,
            status=MEMORY_STATUS_INVALID,
            created_at=memory.created_at,
            last_verified_at=memory.last_verified_at,
            subject=memory.subject,
            source_run_id=memory.source_run_id,
            supersedes=memory.supersedes,
            superseded_by=memory.superseded_by,
        )

    # -- introspection -----------------------------------------------------

    def is_initialized(self) -> bool:
        """Whether the database schema exists (test/demo helper —
        never used as a silent fallback gate)."""
        return self._store.initialized()
