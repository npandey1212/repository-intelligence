"""FlowDependencyReport — the Flow / Dependency Investigator's typed output.

What this module is: the structured result of one flow/dependency
investigation (spec §8-§9), composed from the EXISTING relationship,
finding, and evidence vocabulary — this package never forks
``EvidenceRef``, ``InvestigationFinding``, the classification language,
or the canonical graph semantics:

    FlowDependencyReport
    ├── question · repository_scope · commit_scope
    ├── source · target              (the investigated flow's endpoints)
    ├── path_status                  (FOUND | NOT_FOUND | UNKNOWN | CONFLICTING)
    ├── path[]                       (ordered chain — ArchitectureRelationship reused)
    ├── alternative_paths[][]        (competing candidates under CONFLICTING)
    ├── relationships[]              (every in-scope relationship established)
    ├── findings[]                   (M16 InvestigationFinding — reused)
    ├── evidence_refs[]              (M16 ReportEvidenceRef — reused)
    ├── uncertainties[] · summary · status

There is deliberately NO ``FlowStep`` model: one path step IS an
``ArchitectureRelationship`` (source → relationship_type → target,
plus classification, confidence, evidence refs) — the repository
already had the reusable relationship contract, and a flow is an
ordered list of them (spec §8: do not copy the conceptual structure
when a reusable contract exists).

What it is NOT: a second evidence model, a second graph vocabulary,
a pathfinding framework, or a runtime record.  Relationship types
stay the canonical engineering-graph ``RELATIONSHIP_KINDS`` (M11/M12);
classifications stay the shared ``Classification`` language; the
evidence chain stays the project's one chain.

The handoff boundary is explicit: ``to_investigation_report()``
projects this rich report onto the M16 ``InvestigationReport`` envelope
so every path step, alternative, and relationship crosses the typed
hop as an evidence-backed finding statement instead of being dropped
(spec §17).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal

from ...evidence.models import STATUS_SUPPORTED, STATUS_UNKNOWN
from ...multiagent.contracts import (
    Classification,
    InvestigationFinding,
    InvestigationReport,
    ReportEvidenceRef,
    ReportStatus,
)
from .architecture_report import (
    ArchitectureRelationship,
    Confidence,
    RELATIONSHIP_TYPES,
    _ref_key,
    relationship_statement,
)

# Path status vocabulary (spec §9): no existing project enum covers
# "did the investigation establish a path?", so this is the one new
# (tiny, closed) vocabulary this capability owns — and it is NOT a
# status machine: each report states one terminal answer.
#
#   FOUND       a supported path exists (path non-empty)
#   NOT_FOUND   evidence was sufficient to establish no path exists
#   UNKNOWN     available evidence is insufficient either way
#   CONFLICTING evidence supports contradictory interpretations
PATH_STATUSES = ("FOUND", "NOT_FOUND", "UNKNOWN", "CONFLICTING")
PathStatus = Literal[PATH_STATUSES]

# Classifications under which a FOUND path's steps may stand — an
# established path is never carried by "unknown" or "conflicting"
# steps (those belong to UNKNOWN / CONFLICTING reports instead).
_ESTABLISHED = ("observed", "derived", "inferred")

__all__ = [
    "PATH_STATUSES",
    "ArchitectureRelationship",
    "Classification",
    "Confidence",
    "FlowDependencyReport",
    "PathStatus",
    "RELATIONSHIP_TYPES",
    "relationship_statement",
]


class FlowDependencyReport(BaseModel):
    """The specialist's structured output (spec §8).

    Cross-field rules (all deterministic, all enforced here so a
    malformed flow never crosses a typed boundary):

    * every evidence reference used by a path step, alternative step,
      relationship, or finding must appear in the ``evidence_refs``
      aggregate;
    * every path step must carry evidence — a path is never returned
      unevidenced (§10);
    * a path (or alternative) is a connected chain whose first source
      and last target match the report's declared ``source``/``target``
      (§21 #6);
    * ``FOUND`` requires a non-empty path whose steps are established
      (never unknown/conflicting), and no competing candidate;
    * ``NOT_FOUND`` / ``UNKNOWN`` never carry a path or candidates —
      an unestablished path is never presented as found (§21 #7);
    * competing candidate paths force ``CONFLICTING`` — contradictions
      are preserved, never silently chosen (§21 #8);
    * an ``observed`` relationship without evidence is rejected —
      unsupported edges are never presented as observed (§21 #5).
    """

    question: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)
    source: Optional[str] = None
    target: Optional[str] = None
    path_status: PathStatus = "UNKNOWN"
    summary: str = ""
    path: List[ArchitectureRelationship] = Field(default_factory=list)
    alternative_paths: List[List[ArchitectureRelationship]] = Field(
        default_factory=list
    )
    relationships: List[ArchitectureRelationship] = Field(
        default_factory=list
    )
    findings: List[InvestigationFinding] = Field(default_factory=list)
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    status: ReportStatus = "complete"

    @model_validator(mode="after")
    def _refs_within_aggregate(self) -> "FlowDependencyReport":
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

        check(self.path, "path step")
        for index, alternative in enumerate(self.alternative_paths):
            check(alternative, "alternative path step {}".format(index))
        check(self.relationships, "relationship")
        check(self.findings, "finding")
        return self

    @model_validator(mode="after")
    def _endpoints_present(self) -> "FlowDependencyReport":
        for name in ("source", "target"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(
                    "{} must be a non-empty component id or omitted "
                    "(null) for open-ended queries".format(name)
                )
        return self

    @model_validator(mode="after")
    def _path_rules(self) -> "FlowDependencyReport":
        path = self.path or []
        alternatives = self.alternative_paths or []
        relationships = self.relationships or []
        source = self.source
        target = self.target
        path_status = self.path_status

        def check_chain(steps: Any, label: str) -> None:
            if not steps:
                raise ValueError(
                    "{} contains an empty candidate path".format(label)
                )
            for position, step in enumerate(steps):
                if not step.evidence_refs:
                    raise ValueError(
                        "{} step {} must carry evidence refs — a flow "
                        "step is never returned unevidenced".format(
                            label, position
                        )
                    )
            for before, after in zip(steps, steps[1:]):
                if before.target != after.source:
                    raise ValueError(
                        "{} is not a connected chain: {!r} does not "
                        "connect to {!r}".format(
                            label, before.target, after.source
                        )
                    )
            if source is None or target is None:
                raise ValueError(
                    "{} requires declared source and target "
                    "endpoints".format(label)
                )
            if steps[0].source != source or steps[-1].target != target:
                raise ValueError(
                    "{} does not run from source {!r} to target "
                    "{!r}".format(label, source, target)
                )

        if path:
            check_chain(path, "path")
        for index, alternative in enumerate(alternatives):
            check_chain(alternative, "alternative path {}".format(index))

        for relationship in relationships:
            if (
                relationship.classification == "observed"
                and not relationship.evidence_refs
            ):
                raise ValueError(
                    "observed relationship {!r} \u2192 {!r} has no "
                    "evidence — unsupported edges are never presented "
                    "as observed".format(
                        relationship.source, relationship.target
                    )
                )

        if path_status == "FOUND":
            if not path:
                raise ValueError(
                    "path_status FOUND requires a non-empty path"
                )
            if alternatives:
                raise ValueError(
                    "path_status FOUND cannot carry competing "
                    "candidate paths — use CONFLICTING"
                )
            for position, step in enumerate(path):
                if step.classification not in _ESTABLISHED:
                    raise ValueError(
                        "FOUND path step {} is classified {!r} — only "
                        "observed/derived/inferred steps establish a "
                        "path".format(position, step.classification)
                    )
        elif path_status in ("NOT_FOUND", "UNKNOWN"):
            if path or alternatives:
                raise ValueError(
                    "path_status {} never carries a path or "
                    "candidates — an unestablished path is never "
                    "presented as found".format(path_status)
                )
        return self

    def to_investigation_report(self) -> InvestigationReport:
        """Project onto the M16 handoff envelope (spec §17).

        Every path step, alternative step, and relationship becomes one
        ``InvestigationFinding`` statement (its canonical relationship
        statement, with classification) ahead of the report's own
        findings — nothing established at a flow step is lost at the
        typed hop, and duplicate statements (an edge that appears both
        in ``path`` and ``relationships``) collapse to one.  The
        ``investigation_id`` is derived deterministically from the
        question + scopes, never model-authored prose.
        """
        findings: List[InvestigationFinding] = []
        evidence: List[ReportEvidenceRef] = list(self.evidence_refs)
        seen_refs = {_ref_key(ref) for ref in evidence}
        seen_statements = set()

        def add(
            statement: str,
            refs: List[ReportEvidenceRef],
            support: str,
            uncertainty: Optional[str] = None,
        ) -> None:
            for ref in refs:
                if _ref_key(ref) not in seen_refs:
                    seen_refs.add(_ref_key(ref))
                    evidence.append(ref)
            findings.append(
                InvestigationFinding(
                    statement=statement,
                    evidence_refs=list(refs),
                    support=support,
                    uncertainty=uncertainty,
                )
            )

        def add_relationship(
            relationship: ArchitectureRelationship,
        ) -> None:
            statement = relationship_statement(relationship)
            if statement in seen_statements:
                return
            seen_statements.add(statement)
            add(
                statement,
                relationship.evidence_refs,
                (
                    STATUS_SUPPORTED
                    if relationship.evidence_refs
                    else STATUS_UNKNOWN
                ),
            )

        for step in self.path:
            add_relationship(step)
        for alternative in self.alternative_paths:
            for step in alternative:
                add_relationship(step)
        for relationship in self.relationships:
            add_relationship(relationship)
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
        return "inv-flow-{}".format(digest[:32])
