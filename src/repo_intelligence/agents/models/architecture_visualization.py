"""DiagramSpec / VisualizationRequest — the Architecture Visualization
Agent's typed contracts (spec §4, §6).

What this module is: the structured *representation* contract this
capability adds.  The two investigators already own knowledge
(``ArchitectureReport``, ``FlowDependencyReport``); this module owns
how that knowledge is arranged for presentation.  Every vocabulary it
speaks belongs to an existing home:

* relationship types — the canonical engineering graph
  ``RELATIONSHIP_KINDS`` (M11/M12), never a visualization dialect;
* node types — ``COMPONENT_TYPES`` (the Architecture Investigator's
  closed component list);
* classifications — the shared M16 ``CLASSIFICATIONS`` language;
* confidence — ``CONFIDENCE_LEVELS``;
* evidence — the M16 ``ReportEvidenceRef`` twin of ``EvidenceRef``;
* status — ``ReportStatus``.

What it is NOT: a second graph model, a second evidence model, or a
diagram planning system.  ``DiagramSpec`` is a *projection*: nodes and
edges are selected from the supplied reports, never invented (the
deterministic validator in ``agents.diagram_validation`` proves that).

Two projections keep it inside the existing contracts:

* ``to_diagram_projection()`` — the M5 ``DiagramProjection`` the ONE
  existing renderer (``graph.exporters``) already serializes, so
  Mermaid/DOT/SVG stay deterministic renderer output;
* ``to_finding()`` — the M16 ``ArchitectureFinding`` the M17 analyst
  slot is validated against, so evidence scope survives the handoff.

``DiagramSpec`` deliberately has NO ``Graph``/``Node``/``Edge`` graph
semantics of its own: the diagram reuses the engineering model's
vocabulary and adds only presentation fields (label, legend, groups).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Literal

from ...engineering.models import RELATIONSHIP_KINDS
from ...graph.models import (
    DIAGRAM_PROJECTION_SCHEMA_VERSION,
    DiagramProjection,
    diagram_node_id,
)
from ...multiagent.contracts import (
    CLASSIFICATIONS,
    REPORT_STATUSES,
    ArchitectureConclusion,
    ArchitectureFinding,
    Classification,
    ReportEvidenceRef,
    ReportStatus,
)
from .architecture_report import (
    CONFIDENCE_LEVELS,
    COMPONENT_TYPES,
    ArchitectureComponent,
    ArchitectureRelationship,
    ArchitectureReport,
    ComponentType,
    Confidence,
    RelationshipType,
    component_statement,
    relationship_statement,
)
from .flow_dependency_report import FlowDependencyReport

DIAGRAM_SPEC_SCHEMA_VERSION = "diagram_spec.v1"

# A deliberately small diagram taxonomy (spec §5): the classification
# problem the model solves is bounded, not a planning system.
# DATA_FLOW is absent on purpose — the canonical relationship
# vocabulary has no data-flow kind, and inventing one (PRODUCES,
# CONSUMES) would violate spec §7.
DIAGRAM_TYPES = (
    "ARCHITECTURE",
    "FLOW",
    "COMPONENT",
    "AGENT_TOOL_MCP",
)
DiagramType = Literal[DIAGRAM_TYPES]

# Legend entries are machine-checked against the SAME vocabularies, so
# the legend can never smuggle in new relationship semantics (spec §7).
LEGEND_KINDS = (
    "classification",
    "confidence",
    "relationship_type",
    "node_type",
    "status",
)
LegendKind = Literal[LEGEND_KINDS]

_LEGEND_VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "classification": tuple(CLASSIFICATIONS),
    "confidence": tuple(CONFIDENCE_LEVELS),
    "relationship_type": tuple(RELATIONSHIP_KINDS),
    "node_type": tuple(COMPONENT_TYPES),
    "status": tuple(REPORT_STATUSES),
}

__all__ = [
    "DIAGRAM_SPEC_SCHEMA_VERSION",
    "DIAGRAM_TYPES",
    "LEGEND_KINDS",
    "DiagramEdge",
    "DiagramGroup",
    "DiagramLegendEntry",
    "DiagramNode",
    "DiagramSpec",
    "DiagramType",
    "LegendKind",
    "VisualizationRequest",
]


def _ref_key(ref: Any) -> Any:
    """Evidence-reference identity: path + line bounds — the same
    triple equality the M16 contracts and both investigators use."""
    return (ref.path, ref.line_start, ref.line_end)


class DiagramNode(BaseModel):
    """One node of the diagram (spec §6) — a component the
    investigation already established, carrying its own provenance."""

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    node_type: ComponentType = "unknown"
    classification: Classification = "unknown"
    confidence: Confidence = "unknown"
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)


class DiagramEdge(BaseModel):
    """One directed relationship of the diagram (spec §6).

    ``relationship_type`` is required and closed over
    ``RELATIONSHIP_KINDS``: a diagram edge is the engineering model's
    relationship, never a presentation-only link.
    """

    id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    relationship_type: RelationshipType
    classification: Classification = "unknown"
    confidence: Confidence = "unknown"
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)


class DiagramGroup(BaseModel):
    """An optional visual grouping of existing nodes (a subgraph).

    Groups only *arrange* nodes: membership is validated against the
    node ids, so a group can never introduce an element the
    investigation did not establish.
    """

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    node_ids: List[str] = Field(default_factory=list)

    @field_validator("node_ids")
    @classmethod
    def _members_unique(cls, value: List[str]) -> List[str]:
        if len(set(value)) != len(value):
            raise ValueError("group node_ids contain duplicates")
        return value


class DiagramLegendEntry(BaseModel):
    """One legend row: a canonical vocabulary term and its meaning.

    ``term`` must be a member of the vocabulary ``kind`` names, so the
    legend documents existing semantics instead of adding new ones.
    """

    kind: LegendKind
    term: str = Field(..., min_length=1)
    meaning: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _term_is_canonical(self) -> "DiagramLegendEntry":
        vocabulary = _LEGEND_VOCABULARY.get(self.kind or "")
        if vocabulary is not None and self.term not in vocabulary:
            raise ValueError(
                "term {!r} is not a canonical {} value (expected one "
                "of: {})".format(self.term, self.kind, ", ".join(vocabulary))
            )
        return self


class VisualizationRequest(BaseModel):
    """The typed input contract for one visualization (spec §4).

    The agent consumes the investigators' structured outputs instead
    of raw repository content: at least one report is required, and
    every supplied report must carry the SAME repository/commit scope
    — evidence scope cannot be split across a typed boundary (§20).
    """

    question: str = Field(..., min_length=1)
    diagram_type: DiagramType = "ARCHITECTURE"
    architecture_report: Optional[ArchitectureReport] = None
    flow_report: Optional[FlowDependencyReport] = None
    additional_flow_reports: List[FlowDependencyReport] = Field(
        default_factory=list
    )
    evidence_scope: List[ReportEvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _at_least_one_report(self) -> "VisualizationRequest":
        if (
            self.architecture_report is None
            and self.flow_report is None
            and not self.additional_flow_reports
        ):
            raise ValueError(
                "a visualization request needs at least one "
                "investigation report (ArchitectureReport or "
                "FlowDependencyReport) — the agent never rediscover"
                "s the repository to fill the gap"
            )
        return self

    @model_validator(mode="after")
    def _scopes_agree(self) -> "VisualizationRequest":
        reports = [
            report
            for report in (
                self.architecture_report,
                self.flow_report,
            )
            if report is not None
        ]
        reports.extend(self.additional_flow_reports or [])
        scopes = {
            (report.repository_scope, report.commit_scope)
            for report in reports
        }
        if len(scopes) > 1:
            raise ValueError(
                "supplied reports disagree on scope: {} — one "
                "visualization describes one pinned repository "
                "state".format(sorted(scopes))
            )
        return self

    @model_validator(mode="after")
    def _diagram_type_has_its_investigation(
        self,
    ) -> "VisualizationRequest":
        """Failure propagation at the typed boundary (spec §22): a
        missing investigation can never be papered over by the
        visualizer — the request that would need it is rejected
        instead of producing a diagram from nothing."""
        if self.diagram_type == "FLOW" and not (
            self.flow_report is not None
            or self.additional_flow_reports
        ):
            raise ValueError(
                "a FLOW diagram needs the Flow / Dependency "
                "Investigator's report — the flow cannot be "
                "rediscovered here"
            )
        if (
            self.diagram_type == "ARCHITECTURE"
            and self.architecture_report is None
        ):
            raise ValueError(
                "an ARCHITECTURE diagram needs the Architecture "
                "Investigator's report — the structure cannot be "
                "rediscovered here"
            )
        return self

    def flow_reports(self) -> List[FlowDependencyReport]:
        """Every flow report in request order (primary first)."""
        reports: List[FlowDependencyReport] = []
        if self.flow_report is not None:
            reports.append(self.flow_report)
        reports.extend(self.additional_flow_reports)
        return reports

    def scope_label(self) -> Tuple[str, str]:
        """The one repository/commit scope every supplied report shares."""
        reports = [
            report
            for report in (
                self.architecture_report,
                self.flow_report,
                *self.additional_flow_reports,
            )
            if report is not None
        ]
        if not reports:  # unreachable through the validator, kept explicit
            raise ValueError("no reports supplied")
        return (reports[0].repository_scope, reports[0].commit_scope)


class DiagramSpec(BaseModel):
    """The Visualization Agent's typed output (spec §6).

    Cross-field rules (all deterministic, enforced here so a malformed
    diagram never crosses a typed boundary):

    * node ids are unique and edge ids are unique — an id names one
      element;
    * every edge endpoint refers to an existing node — no dangling
      references;
    * every element's evidence refs appear in the report aggregate —
      a partial package never crosses the boundary;
    * an ``observed`` element without evidence is rejected — the
      diagram never claims to have seen what was not evidenced;
    * every group member is an existing node id — a group arranges
      the investigation's elements, it cannot invent one.

    What it is NOT: an unvalidated model response.  Rendering,
    scope/evidence validation, and investigation all stay outside
    this class (spec §3).
    """

    diagram_type: DiagramType = "ARCHITECTURE"
    title: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    repository_scope: str = Field(..., min_length=1)
    commit_scope: str = Field(..., min_length=1)
    nodes: List[DiagramNode] = Field(default_factory=list)
    edges: List[DiagramEdge] = Field(default_factory=list)
    groups: List[DiagramGroup] = Field(default_factory=list)
    legend: List[DiagramLegendEntry] = Field(default_factory=list)
    evidence_refs: List[ReportEvidenceRef] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    status: ReportStatus = "complete"

    @model_validator(mode="after")
    def _ids_unique(self) -> "DiagramSpec":
        node_ids = [node.id for node in self.nodes or []]
        if len(set(node_ids)) != len(node_ids):
            duplicates = sorted(
                node_id
                for node_id in set(node_ids)
                if node_ids.count(node_id) > 1
            )
            raise ValueError(
                "duplicate node id(s): {}".format(", ".join(duplicates))
            )
        edge_ids = [edge.id for edge in self.edges or []]
        if len(set(edge_ids)) != len(edge_ids):
            duplicates = sorted(
                edge_id
                for edge_id in set(edge_ids)
                if edge_ids.count(edge_id) > 1
            )
            raise ValueError(
                "duplicate edge id(s): {}".format(", ".join(duplicates))
            )
        group_ids = [group.id for group in self.groups or []]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("duplicate group id")
        return self

    @model_validator(mode="after")
    def _edge_endpoints_exist(self) -> "DiagramSpec":
        node_ids = {node.id for node in self.nodes or []}
        for edge in self.edges or []:
            for endpoint in (edge.source, edge.target):
                if endpoint not in node_ids:
                    raise ValueError(
                        "edge {!r} endpoint {!r} has no node — a "
                        "diagram never references a component the "
                        "investigation did not establish".format(
                            edge.id, endpoint
                        )
                    )
        for group in self.groups or []:
            for member in group.node_ids:
                if member not in node_ids:
                    raise ValueError(
                        "group {!r} member {!r} has no node".format(
                            group.id, member
                        )
                    )
        return self

    @model_validator(mode="before")
    @classmethod
    def _aggregate_element_evidence(cls, values: Any) -> Any:
        """Build the top-level evidence aggregate from element references.

        The aggregate is a denormalized index, not an independent source of
        evidence. Models occasionally omit a node/edge reference from that
        repeated list even while citing it on the element itself. Merge those
        exact references deterministically before typed validation: this does
        not invent evidence or change any element-level citation, and it keeps
        the invariant that every element reference is present in the aggregate.
        """
        if not isinstance(values, dict):
            return values

        aggregate = list(values.get("evidence_refs") or [])
        seen = set()
        for ref in aggregate:
            if isinstance(ref, dict):
                key = (ref.get("path"), ref.get("line_start"), ref.get("line_end"))
            else:
                key = _ref_key(ref)
            seen.add(key)

        for collection_name in ("nodes", "edges"):
            for item in values.get(collection_name) or []:
                if not isinstance(item, dict):
                    continue
                for ref in item.get("evidence_refs") or []:
                    if isinstance(ref, dict):
                        key = (ref.get("path"), ref.get("line_start"), ref.get("line_end"))
                    else:
                        key = _ref_key(ref)
                    if key not in seen:
                        aggregate.append(ref)
                        seen.add(key)

        values["evidence_refs"] = aggregate
        return values

    @model_validator(mode="after")
    def _refs_within_aggregate(self) -> "DiagramSpec":
        # Defensive postcondition: pre-validation aggregation above should make
        # this impossible to violate for well-formed element references.
        aggregate = {
            _ref_key(ref) for ref in self.evidence_refs or []
        }

        def check(items: Any, label: str) -> None:
            for item in items or []:
                for ref in item.evidence_refs:
                    if _ref_key(ref) not in aggregate:
                        raise ValueError(
                            "{} evidence ref {!r} is missing from the "
                            "diagram's evidence_refs aggregate".format(
                                label, _ref_key(ref)
                            )
                        )

        check(self.nodes, "node")
        check(self.edges, "edge")
        return self

    @model_validator(mode="after")
    def _observed_needs_evidence(self) -> "DiagramSpec":
        for node in self.nodes or []:
            if node.classification == "observed" and not node.evidence_refs:
                raise ValueError(
                    "observed node {!r} has no evidence — unsupported "
                    "elements are never presented as observed".format(
                        node.id
                    )
                )
        for edge in self.edges or []:
            if edge.classification == "observed" and not edge.evidence_refs:
                raise ValueError(
                    "observed edge {!r} has no evidence — unsupported "
                    "elements are never presented as observed".format(
                        edge.id
                    )
                )
        return self

    # -- projections onto the existing contracts -------------------------

    def to_diagram_projection(self) -> DiagramProjection:
        """Project onto the M5 ``DiagramProjection`` the existing
        renderer serializes (spec §16 — adapt to the renderer's
        expected input instead of writing a second renderer).

        Node order follows the spec (the agent's curation is the
        diagram), so the same spec always yields byte-identical
        Mermaid/DOT (spec §19).
        """
        diagram_ids: Dict[str, str] = {
            node.id: diagram_node_id(index)
            for index, node in enumerate(self.nodes)
        }
        nodes = [
            {
                "architecture_node_id": node.id,
                "diagram_id": diagram_ids[node.id],
                "label": node.label,
                "kind": node.node_type,
                "subkind": None,
                "classification": node.classification,
            }
            for node in self.nodes
        ]
        relationships = [
            {
                "architecture_relationship_id": edge.id,
                "source_architecture_node_id": edge.source,
                "target_architecture_node_id": edge.target,
                "source_diagram_id": diagram_ids[edge.source],
                "target_diagram_id": diagram_ids[edge.target],
                "label": edge.relationship_type,
            }
            for edge in self.edges
        ]
        return DiagramProjection(
            nodes=nodes,
            relationships=relationships,
            metadata={
                "schema_version": DIAGRAM_PROJECTION_SCHEMA_VERSION,
                "input": "diagram_spec",
                "diagram_type": self.diagram_type,
                "title": self.title,
            },
        )

    def to_finding(
        self, investigation_id: Optional[str] = None
    ) -> ArchitectureFinding:
        """Project onto the M16 ``ArchitectureFinding`` envelope the
        M17 analyst slot is validated against (spec §20).

        Only evidence-backed elements become conclusions — a
        ref-less element is carried in ``open_questions`` instead, so
        an unevidenced element can never read as an established
        conclusion.  Node/edge statements reuse the investigators'
        own statement forms (``component_statement`` /
        ``relationship_statement``), so one fact never means two
        things at the handoff.
        """
        conclusions: List[ArchitectureConclusion] = []
        open_questions: List[str] = list(self.uncertainties)
        seen_statements = set()

        def add(statement: str, classification: str, refs: List[Any]) -> None:
            if statement in seen_statements:
                return
            seen_statements.add(statement)
            conclusions.append(
                ArchitectureConclusion(
                    statement=statement,
                    classification=classification,
                    evidence_refs=list(refs),
                )
            )

        for node in self.nodes:
            if not node.evidence_refs:
                open_questions.append(
                    "diagram node {!r} carries no evidence reference".format(
                        node.id
                    )
                )
                continue
            statement = component_statement(
                ArchitectureComponent(
                    id=node.id,
                    name=node.label,
                    type=node.node_type,
                    classification=node.classification,
                )
            )
            add(statement, node.classification, node.evidence_refs)

        for edge in self.edges:
            if not edge.evidence_refs:
                open_questions.append(
                    "diagram edge {!r} carries no evidence reference".format(
                        edge.id
                    )
                )
                continue
            statement = relationship_statement(
                ArchitectureRelationship(
                    source=edge.source,
                    target=edge.target,
                    relationship_type=edge.relationship_type,
                    classification=edge.classification,
                    confidence=edge.confidence,
                    evidence_refs=list(edge.evidence_refs),
                )
            )
            add(statement, edge.classification, edge.evidence_refs)

        return ArchitectureFinding(
            investigation_id=(
                investigation_id or self.derived_investigation_id()
            ),
            question=self.question,
            repository_scope=self.repository_scope,
            commit_scope=self.commit_scope,
            conclusions=conclusions,
            evidence_refs=list(self.evidence_refs),
            open_questions=open_questions,
            summary=(
                "{} diagram {!r}: {} node(s), {} edge(s).".format(
                    self.diagram_type,
                    self.title,
                    len(self.nodes),
                    len(self.edges),
                )
            ),
            status=self.status,
        )

    def with_context(
        self,
        question: str,
        repository_scope: str,
        commit_scope: str,
    ) -> DiagramSpec:
        """A copy whose envelope fields come from the application's
        typed input (task/request), never model-authored prose — the
        same provenance discipline the derived ``investigation_id``
        follows."""
        return self.model_copy(
            update={
                "question": question,
                "repository_scope": repository_scope,
                "commit_scope": commit_scope,
            }
        )

    def derived_investigation_id(self) -> str:
        """A deterministic correlation id for the handoff envelope."""
        payload = "\u001f".join(
            (self.question, self.repository_scope, self.commit_scope)
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return "inv-viz-{}".format(digest[:32])

    # -- artifact document -------------------------------------------------

    def to_document(self) -> Dict[str, Any]:
        """The persisted ``diagram_spec.json`` shape: schema version
        first, then the spec fields in declaration order (the
        repository's artifact convention)."""
        document: Dict[str, Any] = {
            "schema_version": DIAGRAM_SPEC_SCHEMA_VERSION
        }
        document.update(self.model_dump())
        return document

    @classmethod
    def from_document(cls, document: Dict[str, Any]) -> "DiagramSpec":
        payload = dict(document)
        payload.pop("schema_version", None)
        return cls(**payload)

    def to_json(self) -> str:
        """Pretty, deterministic JSON for the persisted artifact
        (``indent=2`` / ``ensure_ascii=False`` — the repository's
        artifact convention; element order is the spec's)."""
        return (
            json.dumps(
                self.to_document(),
                indent=2,
                ensure_ascii=False,
                sort_keys=False,
            )
            + "\n"
        )
