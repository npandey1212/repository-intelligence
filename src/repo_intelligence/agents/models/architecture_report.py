"""ArchitectureReport — the Architecture Investigator's typed output.

What this module is: the structured result of one architecture
investigation (spec §20-§26), composed from the EXISTING M16/M13
evidence vocabulary — this package never forks ``EvidenceRef`` or the
classification/support languages:

    ArchitectureReport
    ├── question · repository_scope · commit_scope
    ├── summary                      (human-readable explanation)
    ├── components[]                 (§21: id/name/type/responsibility)
    ├── relationships[]              (§22: source/target/type/class/confidence)
    ├── findings[]                   (M16 InvestigationFinding — reused)
    ├── evidence_refs[]              (M16 ReportEvidenceRef — reused)
    ├── uncertainties[] · status

What it is NOT: a second evidence model, a second graph semantics, or
a runtime record.  The evidence chain stays the project's one chain
(claim → EvidenceRef → repository artifact → pinned commit), and the
relationship vocabulary is the canonical engineering-graph
``RELATIONSHIP_KINDS`` (M11/M12) — reuse, not parallel semantics.

The handoff boundary is explicit: ``to_investigation_report()``
projects this rich report onto the M16 ``InvestigationReport`` envelope
so components/relationships cross the typed hop as evidence-backed
finding statements instead of being dropped (spec §27).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal

from ...engineering.models import RELATIONSHIP_KINDS
from ...evidence.models import STATUS_SUPPORTED, STATUS_UNKNOWN
from ...multiagent.contracts import (
    CLASSIFICATIONS,
    Classification,
    InvestigationFinding,
    InvestigationReport,
    ReportEvidenceRef,
    ReportStatus,
)

# Component vocabulary (spec §21): a short closed list on purpose —
# "unknown" is the honest entry when evidence does not identify a type.
COMPONENT_TYPES = (
    "agent",
    "tool",
    "mcp_server",
    "workflow",
    "module",
    "service",
    "configuration",
    "data_store",
    "unknown",
)
ComponentType = Literal[COMPONENT_TYPES]

# Relationship vocabulary: the ONE canonical graph vocabulary (M11/M12
# RELATIONSHIP_KINDS) — this module never invents parallel graph
# semantics (spec §22).
RELATIONSHIP_TYPES = RELATIONSHIP_KINDS
RelationshipType = Literal[RELATIONSHIP_KINDS]

CONFIDENCE_LEVELS = ("high", "medium", "low", "unknown")
Confidence = Literal[CONFIDENCE_LEVELS]

__all__ = [
    "CLASSIFICATIONS",
    "COMPONENT_TYPES",
    "CONFIDENCE_LEVELS",
    "RELATIONSHIP_TYPES",
    "ArchitectureComponent",
    "ArchitectureRelationship",
    "ArchitectureReport",
    "Classification",
    "ComponentType",
    "Confidence",
    "RelationshipType",
    "component_statement",
    "relationship_statement",
]


def _ref_key(ref: Any) -> Any:
    """The identity of an evidence reference (path + line bounds) —
    the same triple equality the M16 contracts use."""
    return (ref.path, ref.line_start, ref.line_end)


class ArchitectureComponent(BaseModel):
    """One architectural component (spec §21 — useful fields only)."""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    type: ComponentType = "unknown"
    responsibility: str = ""
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    classification: Classification = "unknown"


class ArchitectureRelationship(BaseModel):
    """One architectural relationship between components (§22).

    ``relationship_type`` is validated against the canonical graph
    vocabulary; ``classification``/``confidence`` keep observed vs
    inferred explicit (§23) and never manufactured (§24).
    """

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relationship_type: RelationshipType
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    classification: Classification = "unknown"
    confidence: Confidence = "unknown"


def component_statement(component: ArchitectureComponent) -> str:
    """The evidence-backed statement form of a component — used by the
    M16 handoff projection and by the direct-mode evidence check so a
    component never means two different things."""
    text = "Component {} ({})".format(component.id, component.type)
    if component.responsibility:
        text = "{}: {}".format(text, component.responsibility)
    return "{} [{}]".format(text, component.classification)


def relationship_statement(relationship: ArchitectureRelationship) -> str:
    """The evidence-backed statement form of a relationship."""
    return "{} {} {} [{}]".format(
        relationship.source,
        relationship.relationship_type,
        relationship.target,
        relationship.classification,
    )


class ArchitectureReport(BaseModel):
    """The specialist's structured output (spec §20).

    Cross-field rule (mirrors the M16 report contract): every evidence
    reference used by a component, relationship, or finding must appear
    in the report's ``evidence_refs`` aggregate — a partial package can
    never cross a typed boundary.
    """

    question: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)
    summary: str = ""
    components: List[ArchitectureComponent] = Field(default_factory=list)
    relationships: List[ArchitectureRelationship] = Field(
        default_factory=list
    )
    findings: List[InvestigationFinding] = Field(default_factory=list)
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    status: ReportStatus = "complete"

    @model_validator(mode="after")
    def _refs_within_aggregate(self) -> "ArchitectureReport":
        aggregate = {
            _ref_key(ref) for ref in self.evidence_refs or []
        }

        def check(items: Any, label: str) -> None:
            for item in items or []:
                for ref in item.evidence_refs:
                    if _ref_key(ref) not in aggregate:
                        raise ValueError(
                            "{} evidence ref {!r} is missing from the "
                            "report's evidence_refs aggregate".format(
                                label, _ref_key(ref)
                            )
                        )

        check(self.components, "component")
        check(self.relationships, "relationship")
        check(self.findings, "finding")
        return self

    def to_investigation_report(self) -> InvestigationReport:
        """Project onto the M16 handoff envelope (spec §27).

        The Analyst's contract carries findings, not components: every
        component and relationship therefore becomes one
        ``InvestigationFinding`` statement (with its classification and
        evidence refs) ahead of the report's own findings, so nothing
        that was established is lost at the typed hop.  The
        ``investigation_id`` is derived deterministically from the
        question + scopes — an application correlation id, never
        model-authored prose.
        """
        findings: List[InvestigationFinding] = []
        evidence: List[ReportEvidenceRef] = list(self.evidence_refs)
        seen = {_ref_key(ref) for ref in evidence}

        def add(statement: str, refs: List[ReportEvidenceRef],
                support: str, uncertainty: Optional[str] = None) -> None:
            for ref in refs:
                if _ref_key(ref) not in seen:
                    seen.add(_ref_key(ref))
                    evidence.append(ref)
            findings.append(
                InvestigationFinding(
                    statement=statement,
                    evidence_refs=list(refs),
                    support=support,
                    uncertainty=uncertainty,
                )
            )

        for component in self.components:
            add(
                component_statement(component),
                component.evidence_refs,
                (
                    STATUS_SUPPORTED
                    if component.evidence_refs
                    else STATUS_UNKNOWN
                ),
            )
        for relationship in self.relationships:
            add(
                relationship_statement(relationship),
                relationship.evidence_refs,
                (
                    STATUS_SUPPORTED
                    if relationship.evidence_refs
                    else STATUS_UNKNOWN
                ),
            )
        for finding in self.findings:
            add(
                finding.statement,
                finding.evidence_refs,
                finding.support,
                finding.uncertainty,
            )

        return InvestigationReport(
            investigation_id=self.derived_investigation_id(),
            question=self.question,
            repository_scope=self.repository_scope,
            commit_scope=self.commit_scope,
            findings=findings,
            evidence_refs=evidence,
            uncertainties=list(self.uncertainties),
            status=self.status,
        )

    def derived_investigation_id(self) -> str:
        """A deterministic correlation id for the handoff envelope."""
        payload = "\u001f".join(
            (self.question, self.repository_scope, self.commit_scope)
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return "inv-arch-{}".format(digest[:32])
