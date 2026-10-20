"""M16 typed inter-agent contracts (spec §6/§7 — smallest protocol).

Two specialized agents collaborate through explicit, validated
typed results — never through raw dictionaries or free-form chat:

    InvestigationReport            ArchitectureFinding
    (Investigator output)    →     (Analyst output, typed)
             ↑                              │
             └──── AgentMessage ────────────┘
                   (typed payload per message_type)

Design rules baked into this module:

* **Same evidence semantics, no parallel model** — ``ReportEvidenceRef``
  is the Pydantic-shaped twin of ``evidence.EvidenceRef`` (identical
  fields and bounds rules) because structured-output contracts must be
  Pydantic models; ``to_evidence_ref()`` converts to the ONE existing
  evidence reference type used by ``gather_evidence``/``validate_claims``.
* **No arbitrary dicts at runtime** — ``AgentMessage.payload`` is
  validated against the exact model its ``message_type`` declares and
  stored as that typed instance (a mapping is parsed, anything else
  wrong is rejected). ``ContractError`` is the deterministic reject.
* **Support/classification reuse** — ``support`` reuses the M13
  ``ValidationStatus`` vocabulary (SUPPORTED/UNSUPPORTED/UNKNOWN);
  ``classification`` reuses the artifact-model ``CLASSIFICATIONS``
  ("observed"/"derived"/"inferred"/"conflicting") plus "unknown".
* **Scope travels on every contract** — repository/commit fields are
  required and non-empty; matching is the validator's job, carrying is
  this module's.
* **No hidden reasoning** — no thought/scratchpad fields; conclusions
  are statement + classification + evidence refs only.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError
from typing_extensions import Literal

from repo_intelligence.evidence.models import (
    STATUS_UNKNOWN,
    VALIDATION_STATUSES,
    EvidenceRef,
    ValidationStatus,
)

__all__ = [
    "CLASSIFICATIONS",
    "MESSAGE_TYPES",
    "REPORT_STATUSES",
    "AgentMessage",
    "ArchitectureConclusion",
    "ArchitectureFinding",
    "ContractError",
    "InvestigationFinding",
    "InvestigationReport",
    "MessageType",
    "ReportEvidenceRef",
    "RECIPIENTS",
    "SENDERS",
    "TaskPayload",
    "ToolFailure",
    "ValidationCheck",
    "ValidationRequest",
    "ValidationResult",
    "make_message",
    "message_from_dict",
    "message_to_dict",
    "report_json",
]

# The project's artifact-model classifications (architecture/models.py:
# observed/derived/inferred/conflicting) + "unknown" from the evidence
# vocabulary — a conclusion without valid evidence is UNKNOWN, never
# fabricated certainty (spec §9).
CLASSIFICATIONS = (
    "observed",
    "derived",
    "inferred",
    "conflicting",
    "unknown",
)

Classification = Literal[
    "observed", "derived", "inferred", "conflicting", "unknown"
]

REPORT_STATUSES = ("complete", "incomplete")
ReportStatus = Literal["complete", "incomplete"]

# Who may appear as a message participant (spec §7 — sender/recipient
# are data, validated, never free-form).
SENDERS = ("task", "investigator", "analyst", "validator")
RECIPIENTS = SENDERS
SenderAgent = Literal["task", "investigator", "analyst", "validator"]
RecipientAgent = Literal["task", "investigator", "analyst", "validator"]

MessageType = Literal[
    "INVESTIGATION_REQUEST",
    "INVESTIGATION_RESULT",
    "ANALYSIS_REQUEST",
    "ANALYSIS_RESULT",
    "VALIDATION_REQUEST",
    "VALIDATION_RESULT",
]

MESSAGE_TYPES: Tuple[str, ...] = (
    "INVESTIGATION_REQUEST",
    "INVESTIGATION_RESULT",
    "ANALYSIS_REQUEST",
    "ANALYSIS_RESULT",
    "VALIDATION_REQUEST",
    "VALIDATION_RESULT",
)


class ContractError(Exception):
    """A typed contract was malformed — deterministic reject.

    Raised at message construction/parsing (spec §22 contract
    validity). The workflow validator reports rule failures as a
    failed ``ValidationResult``; only malformed *protocol objects*
    raise this.
    """


def _utcnow() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ref_key(ref: "ReportEvidenceRef") -> Tuple[str, Optional[int], Optional[int]]:
    if isinstance(ref, Mapping):
        return (ref["path"], ref.get("line_start"), ref.get("line_end"))
    return (ref.path, ref.line_start, ref.line_end)


class ReportEvidenceRef(BaseModel):
    """Pydantic twin of ``evidence.EvidenceRef`` (same fields, same
    bounds rule: both line bounds or neither, start >= 1, start <= end).

    It exists only because the structured-output contract needs a
    Pydantic type; ``to_evidence_ref()`` produces the ONE evidence
    reference the rest of the system speaks.
    """

    path: str = Field(..., min_length=1)
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    @field_validator("line_start", "line_end")
    @classmethod
    def _bound_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError("line bounds must be >= 1")
        return value

    @model_validator(mode="after")
    def _range_coherent(self) -> "ReportEvidenceRef":
        start = self.line_start
        end = self.line_end
        if (start is None) != (end is None):
            raise ValueError(
                "line_start and line_end must be provided together "
                "(evidence.EvidenceRef bounds rule)"
            )
        if start is not None and end is not None and start > end:
            raise ValueError(
                "line_start ({}) must be <= line_end ({})".format(start, end)
            )
        return self

    def to_evidence_ref(self) -> EvidenceRef:
        return EvidenceRef(
            path=self.path,
            line_start=self.line_start,
            line_end=self.line_end,
        )

    @classmethod
    def from_evidence_ref(cls, ref: EvidenceRef) -> "ReportEvidenceRef":
        return cls(
            path=ref.path,
            line_start=ref.line_start,
            line_end=ref.line_end,
        )


class InvestigationFinding(BaseModel):
    """One Investigator finding: what was found, what backs it, how
    strongly (the M13 ``ValidationStatus`` vocabulary), and what
    remains uncertain (spec §6.1)."""

    statement: str = Field(..., min_length=1)
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    support: ValidationStatus = STATUS_UNKNOWN
    uncertainty: Optional[str] = None

    @field_validator("support")
    @classmethod
    def _support_known(cls, value: str) -> str:
        if value not in VALIDATION_STATUSES:
            raise ValueError(
                "support must be one of {}".format(
                    ", ".join(sorted(VALIDATION_STATUSES))
                )
            )
        return value


class InvestigationReport(BaseModel):
    """The Investigator's structured output — an evidence package, not
    an architecture interpretation (spec §4.1/§6.1).

    ``investigation_id``/scopes are required: a report that cannot say
    which pinned repository state it describes is malformed, not
    defaulted. Aggregate ``evidence_refs`` must contain every finding's
    refs (handoff integrity — checked here so a partial aggregate can
    never cross the boundary).
    """

    investigation_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)
    findings: List[InvestigationFinding]
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    investigated_sources: List[str] = Field(default_factory=list)
    status: ReportStatus = "complete"

    @model_validator(mode="before")
    @classmethod
    def _derive_evidence_aggregate(cls, data: Any) -> Any:
        """Canonicalize the redundant aggregate from finding-level refs.

        Structured-output models sometimes return a correct finding-level
        reference (for example SKILLS.md:1-2) but omit it from the aggregate,
        or include a broader overlapping range (SKILLS.md:1-40). The aggregate
        is a derived index, not independent evidence, so union the exact refs
        deterministically before normal validation rather than spending agent
        tool calls asking the model to repair a mechanically derivable field.
        """
        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        aggregate = list(normalized.get("evidence_refs") or [])
        seen = {_ref_key(ref) for ref in aggregate}
        findings = normalized.get("findings") or []
        for finding in findings:
            refs = (
                finding.get("evidence_refs", [])
                if isinstance(finding, Mapping)
                else getattr(finding, "evidence_refs", [])
            )
            for ref in refs or []:
                key = _ref_key(ref)
                if key not in seen:
                    aggregate.append(ref)
                    seen.add(key)
        normalized["evidence_refs"] = aggregate
        return normalized

    @model_validator(mode="after")
    def _finding_refs_in_aggregate(self) -> "InvestigationReport":
        aggregate = {_ref_key(ref) for ref in self.evidence_refs}
        for finding in self.findings:
            for ref in finding.evidence_refs:
                if _ref_key(ref) not in aggregate:
                    raise ValueError(
                        "finding evidence ref {!r} is missing from the "
                        "report's evidence_refs aggregate".format(ref)
                    )
        return self


class ArchitectureConclusion(BaseModel):
    """One Analyst conclusion about what the evidence means
    architecturally (spec §4.2). Classification must be a known value;
    ``observed`` without evidence is rejected by the validator (never
    silently rewritten)."""

    statement: str = Field(..., min_length=1)
    classification: Classification = "unknown"
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)

    @field_validator("classification")
    @classmethod
    def _known_classification(cls, value: str) -> str:
        if value not in CLASSIFICATIONS:
            raise ValueError(
                "classification must be one of {}".format(
                    ", ".join(CLASSIFICATIONS)
                )
            )
        return value


class ArchitectureFinding(BaseModel):
    """The Analyst's structured output (spec §5).

    ``investigation_id`` is REQUIRED and must equal the
    InvestigationReport's id — that equality is the machine-checkable
    proof the Analyst consumed the Investigator's contract instead of
    running an independent investigation (spec §22 collaboration).
    """

    investigation_id: str = Field(..., min_length=1)
    # Ordered IDs of the P1-P3 reports used for P4 synthesis. Legacy
    # producers may omit this; the P4 deterministic validator requires the
    # exact expected ordered list.
    source_investigation_ids: List[str] = Field(default_factory=list)
    question: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)
    conclusions: List[ArchitectureConclusion]
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    summary: str = Field(..., min_length=1)
    status: ReportStatus = "complete"

    @model_validator(mode="after")
    def _conclusion_refs_in_aggregate(self) -> "ArchitectureFinding":
        aggregate = {_ref_key(ref) for ref in self.evidence_refs}
        for conclusion in self.conclusions:
            for ref in conclusion.evidence_refs:
                if _ref_key(ref) not in aggregate:
                    raise ValueError(
                        "conclusion evidence ref {!r} is missing from the "
                        "finding's evidence_refs aggregate".format(ref)
                    )
        return self


class TaskPayload(BaseModel):
    """INVESTIGATION_REQUEST payload — the shared task handed to the
    Investigator (question + the scope that must survive every hop)."""

    question: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)


class ValidationRequest(BaseModel):
    """VALIDATION_REQUEST payload — the deterministic validator's
    input pointers (it reads the full state directly; the typed message
    makes the handoff explicit and traceable)."""

    investigation_id: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)


class ValidationCheck(BaseModel):
    """One deterministic validation rule outcome (spec §22)."""

    rule: str = Field(..., min_length=1)
    ok: bool
    detail: str = ""


class ToolFailure(BaseModel):
    """A tool failure surfaced on the validated result (spec §20 —
    visible in trace AND result)."""

    agent: str = Field(..., min_length=1)
    tool: str
    error_code: str = Field(..., min_length=1)


class ValidationResult(BaseModel):
    """VALIDATION_RESULT payload — the deterministic verdict (no LLM)."""

    investigation_id: str = Field(..., min_length=1)
    status: Literal["passed", "failed"]
    checks: List[ValidationCheck] = Field(default_factory=list)
    tool_failures: List[ToolFailure] = Field(default_factory=list)


# message_type → the exact payload model (spec §7 — one payload type
# per message type, enforced at construction).
_PAYLOAD_TYPES: Dict[str, Any] = {
    "INVESTIGATION_REQUEST": TaskPayload,
    "INVESTIGATION_RESULT": InvestigationReport,
    "ANALYSIS_REQUEST": InvestigationReport,
    "ANALYSIS_RESULT": ArchitectureFinding,
    "VALIDATION_REQUEST": ValidationRequest,
    "VALIDATION_RESULT": ValidationResult,
}


class AgentMessage(BaseModel):
    """One typed handoff between participants (spec §7).

    ``payload`` is declared ``Any`` so pydantic's Union coercion can
    never mistype it; the root validator parses/keeps the payload as
    EXACTLY the model this ``message_type`` declares, so at runtime the
    attribute is always the typed instance — never a dict.
    """

    message_id: str = Field(..., min_length=1)
    sender_agent: SenderAgent
    recipient_agent: RecipientAgent
    message_type: MessageType
    payload: Any
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)
    created_at: str = Field(default_factory=_utcnow)

    @model_validator(mode="before")
    @classmethod
    def _typed_payload_and_ref_consistency(
        cls, data: Any
    ) -> Any:
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        message_type = data.get("message_type")
        if message_type not in _PAYLOAD_TYPES:
            return data
        expected = _PAYLOAD_TYPES[message_type]
        payload = data.get("payload")
        if isinstance(payload, Mapping):
            payload = expected.model_validate(payload)
        elif not isinstance(payload, expected):
            raise ValueError(
                "payload for {} must be {}, got {!r}".format(
                    message_type, expected.__name__, type(payload).__name__
                )
            )
        data["payload"] = payload

        # evidence_refs on the envelope must match the payload's own
        # refs exactly — the envelope can neither invent nor drop
        # provenance (spec §9: provenance never weakens at the hop).
        payload_refs: Tuple[ReportEvidenceRef, ...] = ()
        if isinstance(payload, (InvestigationReport, ArchitectureFinding)):
            payload_refs = tuple(payload.evidence_refs)
        envelope = tuple(data.get("evidence_refs") or ())
        if set(_ref_key(r) for r in envelope) != set(
            _ref_key(r) for r in payload_refs
        ):
            raise ValueError(
                "message evidence_refs must equal the payload's "
                "evidence_refs ({} envelope vs {} payload)".format(
                    len(envelope), len(payload_refs)
                )
            )
        return data

    def typed_payload(self) -> Union[
        TaskPayload,
        InvestigationReport,
        ArchitectureFinding,
        ValidationRequest,
        ValidationResult,
    ]:
        return self.payload


def make_message(
    message_type: str,
    sender_agent: str,
    recipient_agent: str,
    payload: Any,
    repository_scope: str,
    commit_scope: str,
    message_id: Optional[str] = None,
) -> AgentMessage:
    """Build one valid message (ids/timestamps generated here so the
    runner and tests share the exact construction path).

    Envelope ``evidence_refs`` are taken from the payload — a message
    can never disagree with what it carries.
    """
    if message_type not in _PAYLOAD_TYPES:
        raise ContractError(
            "unknown message_type {!r} (expected one of {})".format(
                message_type, ", ".join(MESSAGE_TYPES)
            )
        )
    if isinstance(payload, Mapping):
        try:
            payload = _PAYLOAD_TYPES[message_type].model_validate(payload)
        except ValidationError as exc:
            raise ContractError(
                "malformed {} payload: {}".format(message_type, exc)
            )
    envelope_refs: List[ReportEvidenceRef] = []
    if isinstance(payload, (InvestigationReport, ArchitectureFinding)):
        envelope_refs = list(payload.evidence_refs)
    try:
        return AgentMessage(
            message_id=message_id or uuid.uuid4().hex,
            sender_agent=sender_agent,
            recipient_agent=recipient_agent,
            message_type=message_type,
            payload=payload,
            evidence_refs=envelope_refs,
            repository_scope=repository_scope,
            commit_scope=commit_scope,
        )
    except ValidationError as exc:
        raise ContractError(
            "malformed {}: {}".format(message_type, exc)
        )


def message_from_dict(data: Any) -> AgentMessage:
    """Parse a message dict — the boundary where malformed input is
    deterministically rejected (spec §22 contract validity)."""
    try:
        return AgentMessage.model_validate(data)
    except ValidationError as exc:
        raise ContractError("malformed AgentMessage: {}".format(exc))


def message_to_dict(message: AgentMessage) -> Dict[str, Any]:
    return message.model_dump()


def report_json(report: InvestigationReport) -> str:
    """Deterministic JSON projection of a report — what the Analyst's
    request embeds and what tests match against."""
    return json.dumps(report.model_dump(), sort_keys=True, default=str)
