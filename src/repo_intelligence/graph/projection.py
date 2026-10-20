"""M5.2/M5.3 — build DiagramProjection from a NetworkX architecture graph.

Diagram IDs are assigned from sorted canonical architecture node IDs so the
mapping is deterministic, collision-free, and independent of execution order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import networkx as nx

from ..architecture.validator import ArchitectureModelError
from .models import DIAGRAM_PROJECTION_SCHEMA_VERSION, DiagramProjection, diagram_node_id


def _graph_node_arch_id(graph: nx.DiGraph, key: Any) -> str:
    data = graph.nodes[key]
    return str(data.get("id") or key)


def build_diagram_projection(graph: nx.DiGraph) -> DiagramProjection:
    """Project a NetworkX architecture graph into a DiagramProjection.

    Does not mutate the graph.
    """
    if not isinstance(graph, nx.DiGraph):
        raise ArchitectureModelError(
            "Expected networkx.DiGraph, got {!r}".format(type(graph).__name__)
        )

    # Sort by canonical architecture node ID (not NetworkX iteration order).
    arch_ids = sorted(_graph_node_arch_id(graph, key) for key in graph.nodes)
    # Prefer the graph attribute id when present for the node record.
    key_by_arch_id: Dict[str, Any] = {}
    for key in graph.nodes:
        arch_id = _graph_node_arch_id(graph, key)
        key_by_arch_id[arch_id] = key

    diagram_id_by_arch: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    for index, arch_id in enumerate(arch_ids):
        key = key_by_arch_id[arch_id]
        data = graph.nodes[key]
        d_id = diagram_node_id(index)
        diagram_id_by_arch[arch_id] = d_id
        nodes.append(
            {
                "architecture_node_id": arch_id,
                "diagram_id": d_id,
                "label": data.get("name") or arch_id,
                "kind": data.get("kind"),
                "subkind": data.get("subkind"),
                "classification": data.get("classification"),
            }
        )

    edges: List[Dict[str, Any]] = []
    for source_key, target_key, data in graph.edges(data=True):
        source_arch = _graph_node_arch_id(graph, source_key)
        target_arch = _graph_node_arch_id(graph, target_key)
        rel_id = data.get("id") or "rel:{}:{}:{}".format(
            data.get("kind"), source_arch, target_arch
        )
        edges.append(
            {
                "architecture_relationship_id": rel_id,
                "source_architecture_node_id": source_arch,
                "target_architecture_node_id": target_arch,
                "source_diagram_id": diagram_id_by_arch[source_arch],
                "target_diagram_id": diagram_id_by_arch[target_arch],
                "label": data.get("kind") or "",
            }
        )

    # Deterministic edge order: source canonical, target canonical, kind, id.
    edges.sort(
        key=lambda e: (
            e["source_architecture_node_id"],
            e["target_architecture_node_id"],
            e["label"],
            e["architecture_relationship_id"],
        )
    )

    return DiagramProjection(
        nodes=nodes,
        relationships=edges,
        metadata={
            "schema_version": DIAGRAM_PROJECTION_SCHEMA_VERSION,
            "input": "networkx_digraph",
        },
    )


__all__ = ["build_diagram_projection"]
