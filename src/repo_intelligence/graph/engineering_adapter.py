"""M11.1/M11.2 — NetworkX adapter: EngineeringModel (or the persisted
engineering graph artifact) → networkx.MultiDiGraph.

NetworkX is a runtime representation only. The canonical source of
truth remains ``engineering_model.json``; the persisted
``engineering_graph.json`` is a deterministic transformation of it
(see ``engineering_graph_persist``). Both entries below are
projections, never analyzers: they never mutate the input, never
invent nodes or relationships, and never resolve unresolved
claims — unresolved information is retained only as graph metadata
(count), never as a graph edge.

Why MultiDiGraph (and not the DiGraph used for the architecture
graph): the EngineeringModel may contain several canonical
relationships between the SAME pair of nodes with different kinds
(e.g. one CONTAINS and one IMPORTS edge with distinct canonical
relationship ids). A plain DiGraph would silently drop all but one of
them; a MultiDiGraph keyed by the canonical relationship id keeps the
projection faithful.

Conventions shared with the M5 architecture adapter (graph/adapter.py):

* node ids are the canonical EngineeringModel node ids;
* edge endpoints are canonical source/target node ids;
* edge keys are the canonical relationship ids (unique per model);
* edge/node attributes retain id, kind, subkind, classification,
  attributes and evidence as the model recorded them.

Invariant checking lives in ``engineering_graph_invariant_problems``
below, in the same style as ``graph_invariant_problems``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import networkx as nx

from ..engineering.models import RELATIONSHIP_KINDS
from ..engineering.validator import EngineeringModelError
from .engineering_graph_persist import ENGINEERING_GRAPH_SCHEMA_VERSION


def _engineering_section(
    source: Union[Dict[str, Any], Any]
) -> Dict[str, Any]:
    """Accept a full EngineeringModel document, a bare ``engineering``
    section, or an ``EngineeringModel`` object (``.engineering``)."""
    if isinstance(source, dict):
        engineering = source.get("engineering")
        if engineering is None and "nodes" in source:
            engineering = source
        if engineering is None:
            raise EngineeringModelError(
                "Expected EngineeringModel document with an engineering section"
            )
        return engineering
    engineering = getattr(source, "engineering", None)
    if not isinstance(engineering, dict):
        raise EngineeringModelError(
            "Expected EngineeringModel document with an engineering section"
        )
    return engineering


def _node_attributes(node: Dict[str, Any]) -> Dict[str, Any]:
    attributes = node.get("attributes")
    return {
        "id": node.get("id"),
        "kind": node.get("kind"),
        "subkind": node.get("subkind"),
        "classification": node.get("classification"),
        "attributes": dict(attributes or {}),
        "evidence": [dict(e) for e in node.get("evidence") or []],
    }


def _edge_attributes(relationship: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": relationship.get("id"),
        "kind": relationship.get("kind"),
        "source": relationship.get("source"),
        "target": relationship.get("target"),
        "classification": relationship.get("classification"),
        "evidence": [dict(e) for e in relationship.get("evidence") or []],
    }


def build_engineering_graph(
    engineering_model: Union[Dict[str, Any], Any]
) -> nx.MultiDiGraph:
    """Convert a canonical EngineeringModel into a networkx.MultiDiGraph.

    Node ids and edge endpoints use canonical engineering node ids;
    edge keys are the canonical relationship ids. Only
    ``engineering.relationships`` become edges — unresolved claims
    never do (they remain available through the source model and are
    counted in ``graph.graph["unresolved_count"]``).
    """
    engineering = _engineering_section(engineering_model)
    nodes: List[Dict[str, Any]] = list(engineering.get("nodes") or [])
    relationships: List[Dict[str, Any]] = list(
        engineering.get("relationships") or []
    )

    graph = nx.MultiDiGraph()
    seen_node_ids = set()
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            raise EngineeringModelError("Engineering node missing id")
        if node_id in seen_node_ids:
            raise EngineeringModelError(
                "Engineering node id duplicated: {!r}".format(node_id)
            )
        seen_node_ids.add(node_id)
        graph.add_node(node_id, **_node_attributes(node))

    seen_relationship_ids = set()
    for relationship in relationships:
        relationship_id = relationship.get("id")
        if not relationship_id:
            raise EngineeringModelError(
                "Engineering relationship missing id: {!r}".format(
                    relationship.get("kind")
                )
            )
        if relationship_id in seen_relationship_ids:
            raise EngineeringModelError(
                "Engineering relationship id duplicated: {!r}".format(
                    relationship_id
                )
            )
        seen_relationship_ids.add(relationship_id)
        source = relationship.get("source")
        target = relationship.get("target")
        if not source or not target:
            raise EngineeringModelError(
                "Engineering relationship missing endpoint: {!r}".format(
                    relationship_id
                )
            )
        if source not in graph or target not in graph:
            raise EngineeringModelError(
                "Engineering relationship endpoint not in nodes: "
                "{!r} -> {!r}".format(source, target)
            )
        graph.add_edge(
            source, target, key=relationship_id, **_edge_attributes(relationship)
        )

    # Unresolved claims stay OUT of the edge set; only their count is
    # retained as graph metadata (the source model keeps the details).
    unresolved = engineering.get("unresolved") or []
    graph.graph["unresolved_count"] = len(unresolved)
    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["relationship_count"] = graph.number_of_edges()

    return graph


def project_engineering_graph(graph_document: Any) -> nx.MultiDiGraph:
    """Persisted ``engineering_graph.json`` document → runtime
    networkx.MultiDiGraph.

    The persisted graph document is the canonical graph artifact; this
    is its runtime projection (never canonical itself). The document
    schema is validated first — a stray model document or corrupted
    artifact fails loudly — then the exact same projection as
    ``build_engineering_graph`` runs over the document's own
    nodes/relationships/unresolved sections.
    """
    if not isinstance(graph_document, dict):
        raise EngineeringModelError(
            "Expected an engineering graph document (JSON object)"
        )
    schema_version = graph_document.get("schema_version")
    if schema_version != ENGINEERING_GRAPH_SCHEMA_VERSION:
        raise EngineeringModelError(
            "Expected {!r} graph document, got {!r}".format(
                ENGINEERING_GRAPH_SCHEMA_VERSION, schema_version
            )
        )
    return build_engineering_graph(graph_document)


def engineering_graph_invariant_problems(
    graph: nx.MultiDiGraph, engineering: Union[Dict[str, Any], Any]
) -> List[str]:
    """Return human-readable problems when graph and EngineeringModel diverge.

    Verifies the M11 projection invariants: every graph node is a
    canonical model node, every graph edge is a canonical model
    relationship, every edge references existing nodes, no unresolved
    claim became an edge, and every edge kind stays inside the
    EngineeringModel relationship vocabulary.
    """
    section = _engineering_section(engineering)
    problems: List[str] = []
    nodes = list(section.get("nodes") or [])
    relationships = list(section.get("relationships") or [])
    unresolved = list(section.get("unresolved") or [])

    if graph.number_of_nodes() != len(nodes):
        problems.append(
            "graph_nodes_mismatch: graph={} engineering={}".format(
                graph.number_of_nodes(), len(nodes)
            )
        )
    if graph.number_of_edges() != len(relationships):
        problems.append(
            "graph_edges_mismatch: graph={} engineering={}".format(
                graph.number_of_edges(), len(relationships)
            )
        )

    eng_node_ids = {n.get("id") for n in nodes}
    graph_node_ids = set(graph.nodes)
    for node_id in sorted(graph_node_ids - eng_node_ids):
        problems.append("invented_graph_node: {!r}".format(node_id))
    for node_id in sorted(eng_node_ids - graph_node_ids):
        problems.append("missing_graph_node: {!r}".format(node_id))

    eng_rel_ids = {r.get("id") for r in relationships}
    graph_rel_ids = {
        data.get("id") for _, _, _, data in graph.edges(keys=True, data=True)
    }
    for rel_id in sorted(x for x in graph_rel_ids if x is not None):
        if rel_id not in eng_rel_ids:
            problems.append("invented_graph_relationship: {!r}".format(rel_id))
    for rel_id in sorted(x for x in eng_rel_ids if x is not None):
        if rel_id not in graph_rel_ids:
            problems.append("missing_graph_relationship: {!r}".format(rel_id))

    for source, target, key, data in graph.edges(keys=True, data=True):
        if source not in graph or target not in graph:
            problems.append(
                "invalid_edge_endpoint: {!r} -> {!r}".format(source, target)
            )
        if data.get("id") not in eng_rel_ids:
            problems.append(
                "unresolved_or_unknown_edge: {!r}".format(data.get("id"))
            )
        if key != data.get("id"):
            problems.append(
                "edge_key_mismatch: key={!r} id={!r}".format(
                    key, data.get("id")
                )
            )
        if data.get("kind") not in RELATIONSHIP_KINDS:
            problems.append(
                "relationship_kind_not_in_vocabulary: {!r} ({!r})".format(
                    data.get("kind"), data.get("id")
                )
            )

    # An unresolved claim must never surface as a graph edge: the
    # canonical id derived from its triple must not be an edge id.
    unresolved_triples = {
        "rel:{}:{}:{}".format(
            claim.get("kind"), claim.get("source"), claim.get("target")
        )
        for claim in unresolved
    }
    for rel_id in sorted(unresolved_triples & set(graph_rel_ids)):
        problems.append(
            "unresolved_relationship_edge: {!r}".format(rel_id)
        )

    return problems


__all__ = [
    "build_engineering_graph",
    "engineering_graph_invariant_problems",
    "project_engineering_graph",
]
