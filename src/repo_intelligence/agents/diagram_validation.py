"""Deterministic DiagramSpec validation — the trust boundary (spec §14).

The Visualization Agent proposes a representation; THIS module judges
it.  No model is involved anywhere in the path, and the verdict uses
the existing M16 ``ValidationResult`` / ``ValidationCheck`` contract so
a diagram verdict looks exactly like every other deterministic verdict
in the system (one ``status`` + named rules).

Six rules (spec §14), each answering one question:

======================  ==================================================
``node_validity``       unique ids, every edge/group endpoint exists
``relationship_validity`` every edge type is a canonical kind
``evidence_validity``   every reference is real (and inspected, when the
                        M13 evidence index is supplied)
``classification_validity`` canonical classification, never observed
                        without evidence
``status_validity``     a valid terminal report status
``scope_validity``      nothing outside the investigation scope
======================  ==================================================

The *scope* is the investigation's own knowledge, supplied by the
caller as a :class:`DiagramScope`:

* direct mode       — built from the ``VisualizationRequest``'s typed
  reports (``DiagramScope.from_request``);
* orchestrated mode — built from the M16 ``InvestigationReport`` the
  analyst received (``DiagramScope.from_investigation_report``), by
  reading back the investigators' deterministic statement forms.

A missing scope is reported honestly (structural rules only) instead
of being silently treated as "everything is in scope".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Optional, Sequence, Tuple

from ..evidence.models import (
    STATUS_SUPPORTED,
    Claim,
    EvidenceIndex,
)
from ..evidence.validator import validate_claims
from ..engineering.models import RELATIONSHIP_KINDS
from ..multiagent.contracts import (
    CLASSIFICATIONS,
    REPORT_STATUSES,
    InvestigationReport,
    ValidationCheck,
    ValidationResult,
)
from .models.architecture_visualization import DiagramSpec, VisualizationRequest

DIAGRAM_CHECKS: Tuple[str, ...] = (
    "node_validity",
    "relationship_validity",
    "evidence_validity",
    "classification_validity",
    "status_validity",
    "scope_validity",
)

NO_SCOPE_DETAIL = (
    "no investigation scope supplied — structural rules only"
)

# The investigators' deterministic statement forms read back as scope:
#   "Component <id> (<type>): <responsibility> [<classification>]"
#   "<source> <KIND> <target> [<classification>]"
# (see agents.models.architecture_report.component_statement /
# relationship_statement).  Statements that do not match simply
# contribute no scope — a parser, never a second evidence system.
_COMPONENT_STATEMENT = re.compile(
    r"^Component (\S+) \(([^)]+)\)(?:: (.*))? \[([a-z]+)\]$"
)
_RELATIONSHIP_STATEMENT = re.compile(
    r"^(\S+) ([A-Z_]+) (\S+) \[([a-z]+)\]$"
)

EdgeKey = Tuple[str, str, str]
RefKey = Tuple[str, Optional[int], Optional[int]]

__all__ = [
    "DIAGRAM_CHECKS",
    "DiagramScope",
    "validate_diagram_spec",
]


def _ref_key(ref: Any) -> RefKey:
    return (ref.path, ref.line_start, ref.line_end)


@dataclass(frozen=True)
class DiagramScope:
    """What the investigation established, in three closed sets.

    ``node_ids``       component ids the reports/statement forms name
    ``edges``          (source, target, relationship_type) triples
    ``evidence_keys``  reference identities the investigation carries

    ``requested_diagram_type`` is carried when the request fixed the
    bounded classification (spec §5), so the validator can catch a
    diagram that answers a different question than the one asked.
    """

    node_ids: FrozenSet[str] = field(default_factory=frozenset)
    edges: FrozenSet[EdgeKey] = field(default_factory=frozenset)
    evidence_keys: FrozenSet[RefKey] = field(default_factory=frozenset)
    requested_diagram_type: Optional[str] = None

    # -- builders --------------------------------------------------------

    @classmethod
    def from_request(
        cls, request: VisualizationRequest
    ) -> "DiagramScope":
        """Scope from the typed request (direct mode)."""
        return cls.from_reports(
            architecture_report=request.architecture_report,
            flow_reports=request.flow_reports(),
            evidence_scope=list(request.evidence_scope),
            requested_diagram_type=request.diagram_type,
        )

    @classmethod
    def from_reports(
        cls,
        architecture_report: Any = None,
        flow_reports: Sequence[Any] = (),
        evidence_scope: Sequence[Any] = (),
        requested_diagram_type: Optional[str] = None,
    ) -> "DiagramScope":
        """Scope from the investigators' typed reports.

        A non-empty ``evidence_scope`` narrows the reference set (the
        caller said which evidence this visualization may cite);
        otherwise the union of the reports' own aggregates applies.
        """
        node_ids = set()
        edges: set = set()
        evidence_keys = set()

        if architecture_report is not None:
            for component in architecture_report.components:
                node_ids.add(component.id)
            for relationship in architecture_report.relationships:
                node_ids.add(relationship.source)
                node_ids.add(relationship.target)
                edges.add(
                    (
                        relationship.source,
                        relationship.target,
                        relationship.relationship_type,
                    )
                )
            evidence_keys.update(
                _ref_key(ref) for ref in architecture_report.evidence_refs
            )

        for report in flow_reports:
            if report.source:
                node_ids.add(report.source)
            if report.target:
                node_ids.add(report.target)
            steps = list(report.path)
            for alternative in report.alternative_paths:
                steps.extend(alternative)
            steps.extend(report.relationships)
            for step in steps:
                node_ids.add(step.source)
                node_ids.add(step.target)
                edges.add(
                    (step.source, step.target, step.relationship_type)
                )
            evidence_keys.update(
                _ref_key(ref) for ref in report.evidence_refs
            )

        if evidence_scope:
            evidence_keys = {_ref_key(ref) for ref in evidence_scope}

        return cls(
            node_ids=frozenset(node_ids),
            edges=frozenset(edges),
            evidence_keys=frozenset(evidence_keys),
            requested_diagram_type=requested_diagram_type,
        )

    @classmethod
    def from_investigation_report(
        cls, report: InvestigationReport
    ) -> "DiagramScope":
        """Scope from the M16 envelope the analyst received
        (orchestrated mode): component/relationship statements are
        read back, and the report aggregate defines the evidence."""
        node_ids = set()
        edges: set = set()
        for finding in report.findings:
            component = _COMPONENT_STATEMENT.match(finding.statement)
            if component is not None:
                node_ids.add(component.group(1))
            relationship = _RELATIONSHIP_STATEMENT.match(finding.statement)
            if relationship is not None:
                kind = relationship.group(2)
                if kind in RELATIONSHIP_KINDS:
                    source = relationship.group(1)
                    target = relationship.group(3)
                    node_ids.add(source)
                    node_ids.add(target)
                    edges.add((source, target, kind))
        return cls(
            node_ids=frozenset(node_ids),
            edges=frozenset(edges),
            evidence_keys=frozenset(
                _ref_key(ref) for ref in report.evidence_refs
            ),
            requested_diagram_type=None,
        )


def _claims_of(spec: DiagramSpec) -> List[Claim]:
    """The claims the diagram asserts — the SAME statements its M16
    handoff projection carries, so the validator judges exactly what
    would cross the boundary."""
    finding = spec.to_finding()
    return [
        Claim(
            statement=conclusion.statement,
            evidence_refs=tuple(
                ref.to_evidence_ref() for ref in conclusion.evidence_refs
            ),
        )
        for conclusion in finding.conclusions
    ]


def _spec_ref_keys(spec: DiagramSpec) -> List[Tuple[str, RefKey]]:
    """Every reference the diagram carries, labelled by owner."""
    keys: List[Tuple[str, RefKey]] = []
    for ref in spec.evidence_refs:
        keys.append(("evidence_refs", _ref_key(ref)))
    for node in spec.nodes:
        for ref in node.evidence_refs:
            keys.append(("node {!r}".format(node.id), _ref_key(ref)))
    for edge in spec.edges:
        for ref in edge.evidence_refs:
            keys.append(("edge {!r}".format(edge.id), _ref_key(ref)))
    return keys


def _node_supported(spec: DiagramSpec, node: Any, scope: DiagramScope) -> bool:
    """A node is in scope when the investigation named it, or when
    every reference it carries resolves inside the investigation's
    evidence package (the honest fallback for a component the envelope
    carried only as an endpoint)."""
    if node.id in scope.node_ids:
        return True
    if not node.evidence_refs:
        return False
    return all(
        _ref_key(ref) in scope.evidence_keys for ref in node.evidence_refs
    )


def _check(
    rule: str, ok: bool, detail: str = ""
) -> ValidationCheck:
    return ValidationCheck(rule=rule, ok=ok, detail=detail)


def _node_validity(spec: DiagramSpec) -> ValidationCheck:
    problems: List[str] = []
    node_ids = [node.id for node in spec.nodes]
    if len(set(node_ids)) != len(node_ids):
        problems.append("duplicate node ids")
    edge_ids = [edge.id for edge in spec.edges]
    if len(set(edge_ids)) != len(edge_ids):
        problems.append("duplicate edge ids")
    known = set(node_ids)
    for edge in spec.edges:
        for endpoint in (edge.source, edge.target):
            if endpoint not in known:
                problems.append(
                    "edge {!r} endpoint {!r} has no node".format(
                        edge.id, endpoint
                    )
                )
    for group in spec.groups:
        for member in group.node_ids:
            if member not in known:
                problems.append(
                    "group {!r} member {!r} has no node".format(
                        group.id, member
                    )
                )
    return _check(
        "node_validity",
        not problems,
        "; ".join(problems),
    )


def _relationship_validity(spec: DiagramSpec) -> ValidationCheck:
    problems: List[str] = []
    for edge in spec.edges:
        if edge.relationship_type not in RELATIONSHIP_KINDS:
            problems.append(
                "edge {!r} uses non-canonical relationship {!r}".format(
                    edge.id, edge.relationship_type
                )
            )
        if not edge.source or not edge.target:
            problems.append("edge {!r} has an empty endpoint".format(edge.id))
    kinds = sorted({edge.relationship_type for edge in spec.edges})
    return _check(
        "relationship_validity",
        not problems,
        "; ".join(problems)
        or "canonical kinds present: {}".format(
            ", ".join(kinds) if kinds else "none"
        ),
    )


def _evidence_validity(
    spec: DiagramSpec,
    scope: Optional[DiagramScope],
    evidence: Optional[EvidenceIndex],
) -> ValidationCheck:
    problems: List[str] = []
    if not spec.evidence_refs:
        problems.append("diagram carries no evidence_refs")

    if scope is not None:
        outside = sorted(
            {
                "{} {}".format(where, key)
                for where, key in _spec_ref_keys(spec)
                if key not in scope.evidence_keys
            }
        )
        if outside:
            problems.append(
                "{} reference(s) not carried by the investigation: "
                "{}".format(len(outside), "; ".join(outside[:4]))
            )

    if evidence is not None:
        claims = _claims_of(spec)
        validations = validate_claims(
            claims, evidence, require_inspected=True
        )
        failures = [
            "{} ({}) -> {}".format(
                validation.claim.statement,
                ", ".join(
                    ref.path for ref in validation.claim.evidence_refs
                ),
                validation.status,
            )
            for validation in validations
            if validation.status != STATUS_SUPPORTED
        ]
        if failures:
            problems.append(
                "{} of {} element claim(s) not supported by inspected "
                "evidence: {}".format(
                    len(failures), len(validations), "; ".join(failures)
                )
            )

    return _check("evidence_validity", not problems, "; ".join(problems))


def _classification_validity(spec: DiagramSpec) -> ValidationCheck:
    problems: List[str] = []
    for node in spec.nodes:
        if node.classification not in CLASSIFICATIONS:
            problems.append(
                "node {!r} classification {!r} is not canonical".format(
                    node.id, node.classification
                )
            )
        if node.classification == "observed" and not node.evidence_refs:
            problems.append(
                "observed node {!r} has no evidence".format(node.id)
            )
    for edge in spec.edges:
        if edge.classification not in CLASSIFICATIONS:
            problems.append(
                "edge {!r} classification {!r} is not canonical".format(
                    edge.id, edge.classification
                )
            )
        if edge.classification == "observed" and not edge.evidence_refs:
            problems.append(
                "observed edge {!r} has no evidence".format(edge.id)
            )
    used = sorted(
        {item.classification for item in list(spec.nodes) + list(spec.edges)}
    )
    return _check(
        "classification_validity",
        not problems,
        "; ".join(problems)
        or "classifications present: {}".format(
            ", ".join(used) if used else "none"
        ),
    )


def _status_validity(spec: DiagramSpec) -> ValidationCheck:
    ok = spec.status in REPORT_STATUSES
    return _check(
        "status_validity",
        ok,
        "status {!r}".format(spec.status)
        if ok
        else "status {!r} is not a terminal report status".format(
            spec.status
        ),
    )


def _scope_validity(
    spec: DiagramSpec, scope: Optional[DiagramScope]
) -> ValidationCheck:
    if scope is None:
        return _check("scope_validity", True, NO_SCOPE_DETAIL)

    problems: List[str] = []
    if (
        scope.requested_diagram_type is not None
        and spec.diagram_type != scope.requested_diagram_type
    ):
        problems.append(
            "diagram_type {!r} does not answer the requested {!r}".format(
                spec.diagram_type, scope.requested_diagram_type
            )
        )
    for node in spec.nodes:
        if not _node_supported(spec, node, scope):
            problems.append(
                "node {!r} is outside the investigation scope".format(
                    node.id
                )
            )
    for edge in spec.edges:
        key = (edge.source, edge.target, edge.relationship_type)
        if key not in scope.edges:
            problems.append(
                "edge {!r} ({} -{}-> {}) is not established by the "
                "investigation".format(edge.id, edge.source, edge.relationship_type, edge.target)
            )
    return _check("scope_validity", not problems, "; ".join(problems))


def validate_diagram_spec(
    spec: DiagramSpec,
    scope: Optional[DiagramScope] = None,
    evidence: Optional[EvidenceIndex] = None,
) -> ValidationResult:
    """Run all six deterministic rules; never raises for a rule
    outcome — an invalid diagram is a ``ValidationResult(status=
    "failed")``, not an exception (the M16 verdict contract, reused).

    ``evidence`` (an M13 ``EvidenceIndex``, usually
    ``gather_evidence(investigator_trace)``) upgrades
    ``evidence_validity`` from structural to *inspected* — the
    investigators' traces are the evidence authority, never the
    visualization agent's own (it performs no repository reads).
    """
    checks = [
        _node_validity(spec),
        _relationship_validity(spec),
        _evidence_validity(spec, scope, evidence),
        _classification_validity(spec),
        _status_validity(spec),
        _scope_validity(spec, scope),
    ]
    return ValidationResult(
        investigation_id=spec.derived_investigation_id(),
        status="passed"
        if all(check.ok for check in checks)
        else "failed",
        checks=checks,
        tool_failures=[],
    )
