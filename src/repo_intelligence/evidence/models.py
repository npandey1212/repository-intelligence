"""M8.8/M13 evidence-traceability data model (deterministic, offline).

The smallest useful representation for one question:

    Given a structured claim, did this Agent run actually gather the
    evidence the claim points to?

    Claim
     ├── statement
     └── evidence_refs[] ──► EvidenceRef { path, line_start, line_end }

This package stores claims, evidence references, gathered-evidence
records, and deterministic validation results — nothing else. No
hidden reasoning fields (`thought`, `chain_of_thought`,
`private_reasoning`, `internal_monologue`, `reasoning_trace`) exist
anywhere in this model (asserted by tests, same vocabulary the
execution trace rejects). No file contents are duplicated into a
claim: evidence is *referenced* (path + lines + which ToolResult
observed it + its provenance), never copied.

**Gathered vs. inspected (M13).** Every gathered item records whether
the run *inspected* the file's content (``read_file``-style single-
file results → ``inspected=True``) or merely *discovered* it (search
matches → ``inspected=False`` — a candidate file, not evidence).
``EvidenceIndex.gathered`` holds both; ``EvidenceIndex.inspected``
is the eligibility subset the M13 claim contract validates against.
Discovery is never silently upgraded to inspection.

**Traceability, not truth.** Nothing here decides whether
``statement`` is semantically true — only whether its references
resolve to evidence actually gathered (successful tool results)
during the run — and, under inspected-only validation, to evidence
the run actually read. No LLM, no critic, no claim extraction.

Naming note: ``repo_intelligence.facts.evidence.EvidenceRef`` is the
M2 fact-provenance pointer (path + artifact type + commit + symbol,
back to observed source material). This ``EvidenceRef`` is the M8.8
claim-to-run pointer (path + optional lines) matched against
gathered tool evidence — a different concern sharing one word.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from typing_extensions import Literal

__all__ = [
    "CLAIM_EVIDENCE_INVALID",
    "STATUS_SUPPORTED",
    "STATUS_UNSUPPORTED",
    "STATUS_UNKNOWN",
    "VALIDATION_STATUSES",
    "Claim",
    "ClaimValidation",
    "EvidenceIndex",
    "EvidenceRef",
    "FailedEvidenceAttempt",
    "GatheredEvidence",
    "ValidationStatus",
]

ValidationStatus = Literal[
    "SUPPORTED", "UNSUPPORTED", "UNKNOWN"
]

# Structured failure code (M13 §6, existing error-convention style —
# the same shape as a ToolResult error ``code``): a claim's evidence
# reference failed the deterministic evidence contract (the referenced
# file was never inspected during the run). Recorded on the
# UNSUPPORTED ClaimValidation result, on the trace, and consumable by
# the M9 evaluator — never raised as a second error framework.
CLAIM_EVIDENCE_INVALID = "CLAIM_EVIDENCE_INVALID"

# Controlled vocabulary — small and stable (the smallest vocabulary
# that keeps "not gathered" distinct from "cannot be evaluated").
STATUS_SUPPORTED: ValidationStatus = "SUPPORTED"
STATUS_UNSUPPORTED: ValidationStatus = "UNSUPPORTED"
STATUS_UNKNOWN: ValidationStatus = "UNKNOWN"
VALIDATION_STATUSES = frozenset(
    {STATUS_SUPPORTED, STATUS_UNSUPPORTED, STATUS_UNKNOWN}
)


def _opt_int_pair(
    value: Optional[int], key: str, out: Dict[str, Any]
) -> None:
    if value is not None:
        out[key] = value


@dataclass(frozen=True)
class EvidenceRef:
    """A claim's pointer into gathered evidence (repository-relative).

    ``line_start``/``line_end`` are optional and, when present, must
    form a valid inclusive range (both bounds or neither — the
    validator treats a single bound as insufficient metadata).
    """

    path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path}
        _opt_int_pair(self.line_start, "line_start", out)
        _opt_int_pair(self.line_end, "line_end", out)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRef":
        return cls(
            path=data.get("path", ""),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
        )

    def render(self) -> str:
        """Compact human form: ``path`` or ``path:start-end``."""
        if self.line_start is None or self.line_end is None:
            return self.path
        return "{}:{}-{}".format(self.path, self.line_start, self.line_end)


@dataclass(frozen=True)
class Claim:
    """One structured statement plus the evidence it points to.

    Claims are *inputs* to validation — M8.8 never extracts them from
    free-form text (that is a future, separate problem).
    """

    statement: str
    evidence_refs: Tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Claim":
        refs = tuple(
            EvidenceRef.from_dict(item)
            for item in (data.get("evidence_refs") or [])
        )
        return cls(statement=data.get("statement", ""), evidence_refs=refs)


@dataclass(frozen=True)
class GatheredEvidence:
    """One piece of file-addressable evidence observed during a run.

    Produced by ``gather_evidence`` from a *successful* tool result —
    the record preserves the provenance chain:

        Claim → EvidenceRef → GatheredEvidence → ToolResult → source

    ``inspected`` (M13) distinguishes content inspection from mere
    discovery: a ``read_file``-style result returned the file's
    content (``inspected=True``); a search match only names a
    candidate file (``inspected=False``). Discovery never becomes
    eligible claim evidence on its own.
    """

    path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    tool: str = ""
    tool_call_id: Optional[str] = None
    source: Optional[Mapping[str, Any]] = None
    inspected: bool = True

    @property
    def ref(self) -> EvidenceRef:
        """The EvidenceRef form of this gathered evidence."""
        return EvidenceRef(
            path=self.path,
            line_start=self.line_start,
            line_end=self.line_end,
        )

    def render(self) -> str:
        tool = self.tool or "tool"
        return "{} {}".format(tool, self.ref.render())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path}
        _opt_int_pair(self.line_start, "line_start", out)
        _opt_int_pair(self.line_end, "line_end", out)
        out["tool"] = self.tool
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.source is not None:
            out["source"] = dict(self.source)
        out["inspected"] = self.inspected
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GatheredEvidence":
        source = data.get("source")
        inspected = data.get("inspected")
        return cls(
            path=data.get("path", ""),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            tool=data.get("tool", ""),
            tool_call_id=data.get("tool_call_id"),
            source=dict(source) if isinstance(source, Mapping) else None,
            # legacy payloads (pre-M13) carry no flag — they were
            # produced under gathered-only semantics; never downgrade
            inspected=True if inspected is None else bool(inspected),
        )


@dataclass(frozen=True)
class FailedEvidenceAttempt:
    """A tool result that FAILED for a referenced path (never evidence).

    Kept only so validation reasons can say *why* a reference is not
    positive evidence (e.g. ``read_file tool-registry/README.md →
    FILE_NOT_FOUND``). A failed result must NEVER count as gathered
    evidence.
    """

    path: Optional[str]
    tool: str
    tool_call_id: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path, "tool": self.tool}
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.error_code is not None:
            out["error_code"] = self.error_code
        return out

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "FailedEvidenceAttempt":
        return cls(
            path=data.get("path"),
            tool=data.get("tool", ""),
            tool_call_id=data.get("tool_call_id"),
            error_code=data.get("error_code"),
        )


@dataclass(frozen=True)
class EvidenceIndex:
    """Everything a run's execution data contributes to validation.

    ``gathered`` holds ONLY successful, file-addressable tool evidence
    (inspected reads *and* search discoveries); ``failed`` holds
    failed attempts (for reasons only — never matched as positive
    evidence). ``inspected`` (M13) is the eligibility subset —
    gathered items whose file content the run actually read — which
    inspected-only claim validation matches against.
    """

    gathered: Tuple[GatheredEvidence, ...] = ()
    failed: Tuple[FailedEvidenceAttempt, ...] = ()

    @property
    def inspected(self) -> Tuple[GatheredEvidence, ...]:
        """Gathered items the run actually read (M13 eligibility)."""
        return tuple(
            item for item in self.gathered if item.inspected
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceIndex":
        return cls(
            gathered=tuple(
                GatheredEvidence.from_dict(item)
                for item in (data.get("gathered") or [])
            ),
            failed=tuple(
                FailedEvidenceAttempt.from_dict(item)
                for item in (data.get("failed") or [])
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gathered": [item.to_dict() for item in self.gathered],
            "failed": [item.to_dict() for item in self.failed],
        }


@dataclass(frozen=True)
class ClaimValidation:
    """The deterministic validation result for one claim.

    Semantics:

        SUPPORTED   at least one reference resolved to successfully
                    gathered evidence (under inspected-only
                    validation: to evidence the run actually read)
        UNSUPPORTED references exist (and are structurally valid) but
                    none correspond to eligible evidence — recorded
                    with ``code = CLAIM_EVIDENCE_INVALID`` (M13 §6,
                    existing error-convention shape)
        UNKNOWN     the claim cannot be evaluated deterministically
                    (no references, or insufficient metadata)

    UNKNOWN is never silently converted to SUPPORTED, and an empty
    evidence list never means "false" — this checks traceability.
    """

    claim: Claim
    status: ValidationStatus
    matched_evidence: Tuple[GatheredEvidence, ...] = ()
    reason: str = ""
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "claim": self.claim.to_dict(),
            "status": self.status,
            "matched_evidence": [
                item.to_dict() for item in self.matched_evidence
            ],
            "reason": self.reason,
        }
        if self.code is not None:
            out["code"] = self.code
        return out
