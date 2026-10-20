"""Memory formation — the DETERMINISTIC formation policy after M13
validation.

The M14 §18 pipeline, deliberately narrow:

    Tool Result → Observation → Evidence → Claim
      → deterministic validation (M13, unchanged)
      → [THIS MODULE] deterministic memory formation policy
      → MemoryCandidate → MemoryManager.store() → SQLite

Two hard rules this module embodies:

* NOT every claim becomes memory. Only claims the deterministic M13
  validator marked ``SUPPORTED`` are even eligible — an UNSUPPORTED or
  UNKNOWN claim never survives the execution. Eligibility is still not
  obligation: the agent-side gate (a manager must be injected) plus
  this named, bounded selection is the "deterministic memory
  formation policy", never a side effect of validation.
* NOT every observation becomes memory. Raw tool output, intermediate
  reasoning, and transient state are absent from this API entirely —
  candidates come from validated claims and nothing else.

There is NO intelligent memory selection here: no LLM call, no
learned policy, no importance scoring, no autonomous judgment about
what deserves memory. The policy is intentionally simple —
``SUPPORTED`` claim + valid provenance + within the formation limit →
eligible — because M14 teaches persistence, provenance, lifecycle,
and retrieval, not autonomous memory selection.

The selection is deterministic (same validation → same candidates),
bounded (``limit``), and carries M13 provenance forward unchanged:
statement + evidence_refs + the run id of the run that validated them
(M14 §17/§34).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..evidence import STATUS_SUPPORTED, ClaimValidation
from .models import (
    MEMORY_TYPE_EPISODIC,
    MemoryCandidate,
    MemoryOwner,
    MemoryScope,
)

__all__ = [
    "MAX_MEMORY_CANDIDATES",
    "select_memory_candidates",
]

# Most observations disappear with the execution (M14 §18); the few
# validated findings worth remembering are still explicitly bounded so
# one run can never dump an unbounded batch of memory.
MAX_MEMORY_CANDIDATES = 5


def select_memory_candidates(
    validation: Sequence[ClaimValidation],
    owner: MemoryOwner,
    scope: MemoryScope,
    subject: Optional[str] = None,
    memory_type: str = MEMORY_TYPE_EPISODIC,
    run_id: Optional[str] = None,
    limit: int = MAX_MEMORY_CANDIDATES,
) -> Tuple[MemoryCandidate, ...]:
    """Validated claims → memory candidates (the deterministic
    formation policy).

    Keeps, IN ORDER, at most *limit* validation results whose status
    is ``SUPPORTED`` (traceability established by the M13 validator
    against this run's inspected evidence). Each candidate carries the
    claim's own statement and evidence_refs — provenance, not copies
    of the evidence itself (evidence stays in the run's trace).

    Pure and deterministic: same ``validation`` in → same candidates
    out. It never retrieves, never scores, never calls a model.
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 0
    ):
        raise ValueError(
            "limit must be an integer >= 0, got {!r}".format(limit)
        )
    candidates: Tuple[MemoryCandidate, ...] = ()
    for result in validation:
        if len(candidates) >= limit:
            break
        if result.status != STATUS_SUPPORTED:
            continue
        claim = result.claim
        candidates = candidates + (
            MemoryCandidate(
                owner=owner,
                memory_type=memory_type,
                statement=claim.statement,
                scope=scope,
                evidence=tuple(claim.evidence_refs),
                subject=subject,
                source_run_id=run_id,
            ),
        )
    return candidates
