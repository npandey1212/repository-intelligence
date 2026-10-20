"""The M14 memory contract — small, explicit, repository-agnostic.

A memory is deliberately selected information that survives ONE agent
execution (M14 §11, refined against this codebase):

    Memory
    ├── memory_id            mem_<opaque uuid> — application-generated
    ├── owner
    │   ├── type             "agent" today (free-form, non-empty)
    │   └── id               e.g. "repo_intelligence.agent"
    ├── memory_type          "semantic" | "episodic"
    ├── content.statement    the remembered fact/investigation
    ├── scope
    │   ├── repository       repository NAME (data — never logic)
    │   └── commit           pinned commit the statement describes
    ├── evidence[]           EvidenceRef {path, line_start, line_end}
    │                        (reused from M13 — no second evidence model)
    ├── subject              optional retrieval key (e.g. topic) — the
    │                        §24 deterministic search dimension
    ├── source_run_id        the run that created it (trace linkage)
    ├── status               "verified" | "superseded" | "invalid"
    ├── created_at           ISO-8601 UTC (store-generated)
    ├── last_verified_at     ISO-8601 UTC (store-generated/updated)
    ├── supersedes           memory_id this record replaced (nullable)
    └── superseded_by        memory_id that replaced this record

Refinements over the §11 sketch (documented, not silent):

* ``subject`` — §24 lists subject as a first-class retrieval
  dimension; a filterable dimension must exist on the record.
* ``source_run_id`` — §34 provenance: memory → the run that formed it
  (the reverse link lives in the trace's ``memory.*`` events).
* ``supersedes`` / ``superseded_by`` — §28 requires a simple,
  inspectable superseded-by relationship; two nullable columns are the
  simplest inspectable representation.
* evidence reuses ``EvidenceRef`` (M13) — evidence REFERENCES only;
  the evidence itself stays in the run's trace.

Contract rules (enforced by ``validate_candidate``, never by the LLM):

* the agent NEVER invents ``memory_id`` — the manager generates it at
  store time from an opaque uuid (identity independent of content);
* ``memory_type`` ∈ {semantic, episodic}; NO persistent "working
  memory" type (working memory is execution state, M14 §15);
* ``statement`` non-empty; owner id/type non-empty; scope values are
  strings when present; evidence paths non-empty with consistent
  1-based line bounds;
* a candidate is TRANSIENT until validated and stored (M14 §27).

No secrets, no raw tool dumps, no hidden-reasoning fields — a memory
holds deliberately selected information only (M14 §67).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from ..evidence import EvidenceRef
from .errors import MEMORY_CONTRACT_VIOLATION, MemoryOperationError

__all__ = [
    "MEMORY_STATUSES",
    "MEMORY_STATUS_INVALID",
    "MEMORY_STATUS_SUPERSEDED",
    "MEMORY_STATUS_VERIFIED",
    "MEMORY_TYPES",
    "MEMORY_TYPE_EPISODIC",
    "MEMORY_TYPE_SEMANTIC",
    "MEMORY_VERIFICATION_FRESH",
    "MEMORY_VERIFICATION_STALE",
    "MEMORY_VERIFICATION_STATUSES",
    "MEMORY_VERIFICATION_UNKNOWN",
    "DEFAULT_SEARCH_LIMIT",
    "Memory",
    "MemoryCandidate",
    "MemoryContext",
    "MemoryOwner",
    "MemoryScope",
    "MemoryVerification",
    "candidate_from",
    "validate_candidate",
]

MEMORY_TYPE_SEMANTIC = "semantic"
MEMORY_TYPE_EPISODIC = "episodic"
MEMORY_TYPES = (MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_EPISODIC)

MEMORY_STATUS_VERIFIED = "verified"
MEMORY_STATUS_SUPERSEDED = "superseded"
MEMORY_STATUS_INVALID = "invalid"
MEMORY_STATUSES = (
    MEMORY_STATUS_VERIFIED,
    MEMORY_STATUS_SUPERSEDED,
    MEMORY_STATUS_INVALID,
)

# Tri-state scope verification (M14 §26): freshness must be positively
# established — absence of a comparison is NEVER reported as fresh.
MEMORY_VERIFICATION_FRESH = "fresh"
MEMORY_VERIFICATION_STALE = "stale"
MEMORY_VERIFICATION_UNKNOWN = "unknown"
MEMORY_VERIFICATION_STATUSES = (
    MEMORY_VERIFICATION_FRESH,
    MEMORY_VERIFICATION_STALE,
    MEMORY_VERIFICATION_UNKNOWN,
)

DEFAULT_SEARCH_LIMIT = 50


def _violation(message: str) -> MemoryOperationError:
    return MemoryOperationError(MEMORY_CONTRACT_VIOLATION, message)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _violation(
            "{} must be a non-empty string, got {!r}".format(label, value)
        )
    return value


def _optional_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _violation(
            "{} must be None or a non-empty string, got {!r}".format(
                label, value
            )
        )
    return value


@dataclass(frozen=True)
class MemoryOwner:
    """Who the memory belongs to (M14 §14 — ownership ≠ accessibility).

    ``type``/``id`` are free-form non-empty strings: today's owner is
    ``{"type": "agent", "id": "repo_intelligence.agent"}``, but the
    contract does not hard-code single-agent ownership — future owners
    fit without a schema change, and a future agent may READ a memory
    it does not own (no permission system in M14).
    """

    type: str
    id: str


@dataclass(frozen=True)
class MemoryScope:
    """The repository state the statement was established for
    (M14 §25): repository NAME + pinned commit. Both are data — the
    memory package never interprets repository names. ``None`` marks
    memory not derived from a repository state (the agent always fills
    what it knows; it never invents a commit).

    The repository value is also the ISOLATION BOUNDARY: repository-
    derived retrieval is always constrained to it, so memory stored
    for one repository can never be implicitly retrieved during an
    execution against another."""

    repository: Optional[str] = None
    commit: Optional[str] = None


@dataclass(frozen=True)
class MemoryCandidate:
    """What an agent offers to remember — transient until the manager
    validates and stores it (M14 §18/§27). Contains NO ``memory_id``:
    identity is the manager's to generate (M14 §12)."""

    owner: MemoryOwner
    memory_type: str
    statement: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    evidence: Tuple[EvidenceRef, ...] = ()
    subject: Optional[str] = None
    source_run_id: Optional[str] = None


@dataclass(frozen=True)
class Memory:
    """A stored, contract-valid memory record."""

    memory_id: str
    owner: MemoryOwner
    memory_type: str
    statement: str
    scope: MemoryScope
    evidence: Tuple[EvidenceRef, ...]
    status: str
    created_at: str
    last_verified_at: str
    subject: Optional[str] = None
    source_run_id: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record (canonical contract projection)."""
        return {
            "memory_id": self.memory_id,
            "owner": {"type": self.owner.type, "id": self.owner.id},
            "memory_type": self.memory_type,
            "content": {"statement": self.statement},
            "scope": {
                "repository": self.scope.repository,
                "commit": self.scope.commit,
            },
            "evidence": [ref.to_dict() for ref in self.evidence],
            "subject": self.subject,
            "source_run_id": self.source_run_id,
            "status": self.status,
            "created_at": self.created_at,
            "last_verified_at": self.last_verified_at,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class MemoryVerification:
    """Outcome of ``MemoryManager.verify`` — TRI-STATE scope check
    against the repository state the caller is working from NOW
    (M14 §26):

    * ``fresh``    — both commits are known and match: freshness was
      POSITIVELY established (``last_verified_at`` is bumped).
    * ``stale``    — both commits are known and differ: the repository
      scope changed (nothing is written — the caller decides).
    * ``unknown``  — the commits cannot be compared (commit missing on
      either side): freshness could NOT be determined. Unknown is
      NEVER reported as fresh.

    ``verification_status`` carries the tri-state; ``stale`` is kept as
    the compatibility convenience boolean
    (``verification_status == "stale"``). Verification is a SCOPE
    check only — it never re-proves the statement's semantic truth.
    Retention/superseding/invalidation stay explicit caller decisions
    (M14 §26/§27).
    """

    memory_id: str
    status: str
    scope_commit: Optional[str]
    current_commit: Optional[str]
    verification_status: str
    stale: bool


@dataclass(frozen=True)
class MemoryContext:
    """One execution's memory configuration (injected with the
    manager — M14 §22).

    * ``scope`` — repository/commit this execution works from: the
      retrieval filter, the stored candidates' scope, and the commit
      ``verify`` checks against (fresh/stale/unknown). The repository
      is the ISOLATION ANCHOR: a repository-scoped execution only
      ever retrieves its own repository's memory (M14 §23), and an
      execution with no repository scope performs NO repository-memory
      retrieval at all — never an unrestricted cross-repository
      search.
    * ``subject`` — optional retrieval key + stored subject.
    * ``owner`` — whose memory this execution reads/forms; ``None``
      means "the caller resolves its own default" (the Agent fills in
      its actual identity — this package never hard-codes an agent).
    * ``memory_type`` — type assigned to candidates THIS execution
      forms; retrieval always searches every type.
    * ``search_limit`` — deterministic bound on retrieved memories
      (§23: bounded, never "all memories").
    """

    scope: MemoryScope = field(default_factory=MemoryScope)
    subject: Optional[str] = None
    owner: Optional[MemoryOwner] = None
    memory_type: str = MEMORY_TYPE_EPISODIC
    search_limit: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise _violation(
                "MemoryContext.scope must be a MemoryScope, got {!r}".format(
                    type(self.scope).__name__
                )
            )
        _optional_text(self.subject, "subject")
        if self.owner is not None and not isinstance(self.owner, MemoryOwner):
            raise _violation(
                "MemoryContext.owner must be None or a MemoryOwner, got "
                "{!r}".format(type(self.owner).__name__)
            )
        if self.memory_type not in MEMORY_TYPES:
            raise _violation(
                "memory_type must be one of {}, got {!r}".format(
                    ", ".join(MEMORY_TYPES), self.memory_type
                )
            )
        if (
            isinstance(self.search_limit, bool)
            or not isinstance(self.search_limit, int)
            or self.search_limit < 1
        ):
            raise _violation(
                "search_limit must be an integer >= 1, got {!r}".format(
                    self.search_limit
                )
            )


def _validate_evidence(evidence: Any) -> Tuple[EvidenceRef, ...]:
    if isinstance(evidence, (str, bytes)) or not isinstance(
        evidence, (tuple, list)
    ):
        raise _violation(
            "evidence must be a sequence of EvidenceRef, got {!r}".format(
                type(evidence).__name__
            )
        )
    refs: Tuple[EvidenceRef, ...] = ()
    for index, ref in enumerate(evidence):
        if not isinstance(ref, EvidenceRef):
            raise _violation(
                "evidence[{}] must be an EvidenceRef, got {!r}".format(
                    index, type(ref).__name__
                )
            )
        _require_text(ref.path, "evidence[{}].path".format(index))
        for bound in ("line_start", "line_end"):
            value = getattr(ref, bound)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise _violation(
                    "evidence[{}].{} must be None or a positive integer, "
                    "got {!r}".format(index, bound, value)
                )
        if (
            ref.line_start is not None
            and ref.line_end is not None
            and ref.line_start > ref.line_end
        ):
            raise _violation(
                "evidence[{}] line_start {} > line_end {}".format(
                    index, ref.line_start, ref.line_end
                )
            )
        refs = refs + (ref,)
    return refs


def validate_candidate(candidate: Any) -> MemoryCandidate:
    """Enforce the memory contract on one candidate (M14 §11).

    Raises ``MemoryOperationError(MEMORY_CONTRACT_VIOLATION)`` for any
    violation — the application side (manager) is the authority; the
    model never validates its own memory. Returns the candidate
    unchanged for fluent use.
    """
    if not isinstance(candidate, MemoryCandidate):
        raise _violation(
            "candidate must be a MemoryCandidate, got {!r}".format(
                type(candidate).__name__
            )
        )
    if not isinstance(candidate.owner, MemoryOwner):
        raise _violation("owner must be a MemoryOwner")
    _require_text(candidate.owner.type, "owner.type")
    _require_text(candidate.owner.id, "owner.id")
    if candidate.memory_type not in MEMORY_TYPES:
        raise _violation(
            "memory_type must be one of {}, got {!r}".format(
                ", ".join(MEMORY_TYPES), candidate.memory_type
            )
        )
    _require_text(candidate.statement, "statement")
    if not isinstance(candidate.scope, MemoryScope):
        raise _violation("scope must be a MemoryScope")
    _optional_text(candidate.scope.repository, "scope.repository")
    _optional_text(candidate.scope.commit, "scope.commit")
    _optional_text(candidate.subject, "subject")
    _optional_text(candidate.source_run_id, "source_run_id")
    _validate_evidence(candidate.evidence)
    return candidate


def candidate_from(data: Mapping[str, Any]) -> MemoryCandidate:
    """Build a candidate from the §11-style mapping (docs/tests
    convenience — same contract, no persistence)."""
    owner = data.get("owner") or {}
    scope = data.get("scope") or {}
    content = data.get("content") or {}
    evidence = tuple(
        EvidenceRef(
            path=ref.get("path", ""),
            line_start=ref.get("line_start"),
            line_end=ref.get("line_end"),
        )
        for ref in data.get("evidence") or ()
    )
    candidate = MemoryCandidate(
        owner=MemoryOwner(
            type=owner.get("type", ""), id=owner.get("id", "")
        ),
        memory_type=data.get("memory_type", ""),
        statement=content.get("statement", ""),
        scope=MemoryScope(
            repository=scope.get("repository"), commit=scope.get("commit")
        ),
        evidence=evidence,
        subject=data.get("subject"),
        source_run_id=data.get("source_run_id"),
    )
    return validate_candidate(candidate)
