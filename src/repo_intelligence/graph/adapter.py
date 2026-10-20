"""M5.1 — NetworkX adapter: ArchitectureModel → networkx.DiGraph.

NetworkX is a runtime representation only. Canonical source of truth remains
architecture_model.json. The adapter never mutates the ArchitectureModel and
never invents nodes or relationships. Unresolved claims are not graph edges.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import networkx as nx

from ..architecture.validator import ArchitectureModelError


def _architecture_section(
    source: Union[Dict[str, Any], Any]
) -> Dict[str, Any]:
    if isinstance(source, dict):
        architecture = source.get("architecture")
        if architecture is None and "nodes" in source:
            architecture = source
        if architecture is None:
            raise ArchitectureModelError(
                "Expected ArchitectureModel document with an architecture section"
            )
        return architecture
    architecture = getattr(source, "architecture", None)
    if not isinstance(architecture, dict):
        raise ArchitectureModelError(
            "Expected ArchitectureModel document with an architecture section"
        )
    return architecture


def _node_attributes(node: Dict[str, Any]) -> Dict[str, Any]:
    attributes = node.get("attributes")
    return {
        "id": node.get("id"),
        "kind": node.get("kind"),
        "subkind": node.get("subkind"),
        "name": (attributes or {}).get("name") or node.get("id"),
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


def build_architecture_graph(
    architecture_model: Union[Dict[str, Any], Any]
) -> nx.DiGraph:
    """Convert a canonical ArchitectureModel into a networkx.DiGraph.

    Node IDs and edge endpoints use canonical architecture node IDs.
    Only architecture.relationships become edges (never unresolved claims).
    """
    architecture = _architecture_section(architecture_model)
    nodes: List[Dict[str, Any]] = list(architecture.get("nodes") or [])
    relationships: List[Dict[str, Any]] = list(
        architecture.get("relationships") or []
    )

    graph = nx.DiGraph()
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            raise ArchitectureModelError("Architecture node missing id")
        graph.add_node(node_id, **_node_attributes(node))

    for relationship in relationships:
        source = relationship.get("source")
        target = relationship.get("target")
        if not source or not target:
            raise ArchitectureModelError(
                "Architecture relationship missing endpoint: {!r}".format(
                    relationship.get("id")
                )
            )
        if source not in graph or target not in graph:
            raise ArchitectureModelError(
                "Architecture relationship endpoint not in nodes: "
                "{!r} -> {!r}".format(source, target)
            )
        graph.add_edge(source, target, **_edge_attributes(relationship))

    return graph


def graph_invariant_problems(
    graph: nx.DiGraph, architecture: Dict[str, Any]
) -> List[str]:
    """Return human-readable problems when graph and ArchitectureModel diverge."""
    problems: List[str] = []
    nodes = list(architecture.get("nodes") or [])
    relationships = list(architecture.get("relationships") or [])

    if graph.number_of_nodes() != len(nodes):
        problems.append(
            "graph_nodes_mismatch: graph={} architecture={}".format(
                graph.number_of_nodes(), len(nodes)
            )
        )
    if graph.number_of_edges() != len(relationships):
        problems.append(
            "graph_edges_mismatch: graph={} architecture={}".format(
                graph.number_of_edges(), len(relationships)
            )
        )

    arch_node_ids = {n.get("id") for n in nodes}
    graph_node_ids = set(graph.nodes)
    for node_id in sorted(graph_node_ids - arch_node_ids):
        problems.append(
            "invented_graph_node: {!r}".format(node_id)
        )
    for node_id in sorted(arch_node_ids - graph_node_ids):
        problems.append(
            "missing_graph_node: {!r}".format(node_id)
        )

    arch_rel_ids = {r.get("id") for r in relationships}
    graph_rel_ids = {
        data.get("id") for _, _, data in graph.edges(data=True)
    }
    for rel_id in sorted(x for x in graph_rel_ids if x is not None):
        if rel_id not in arch_rel_ids:
            problems.append(
                "invented_graph_relationship: {!r}".format(rel_id)
            )
    for rel_id in sorted(x for x in arch_rel_ids if x is not None):
        if rel_id not in graph_rel_ids:
            problems.append(
                "missing_graph_relationship: {!r}".format(rel_id)
            )

    for source, target, data in graph.edges(data=True):
        if source not in graph or target not in graph:
            problems.append(
                "invalid_edge_endpoint: {!r} -> {!r}".format(source, target)
            )
        if data.get("id") not in arch_rel_ids:
            problems.append(
                "unresolved_or_unknown_edge: {!r}".format(data.get("id"))
            )

    return problems


__all__ = [
    "build_architecture_graph",
    "graph_invariant_problems",
]
