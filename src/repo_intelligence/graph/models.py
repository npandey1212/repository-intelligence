"""DiagramProjection domain helpers (M5 intermediate diagram model).

    ArchitectureModel
           ↓
    NetworkX DiGraph (runtime)
           ↓
    DiagramProjection (validated intermediate)
           ↓
     ┌─────┴─────┐
     ▼           ▼
  Mermaid      Graphviz DOT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

DIAGRAM_PROJECTION_SCHEMA_VERSION = "diagram_projection.v1"
DIAGRAM_VALIDATION_SCHEMA_VERSION = "diagram_validation.v1"

MERMAID_FILENAME = "architecture.mmd"
GRAPHVIZ_FILENAME = "architecture.dot"
SVG_FILENAME = "architecture.svg"
DIAGRAM_VALIDATION_FILENAME = "validation.json"


def diagram_node_id(index: int) -> str:
    """Deterministic Graphviz/Mermaid-safe id for the index-th sorted node."""
    return "node_{:03d}".format(index + 1)


@dataclass
class DiagramProjection:
    """Faithful projection of ArchitectureModel nodes/relationships for export."""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": DIAGRAM_PROJECTION_SCHEMA_VERSION,
            "nodes": [dict(n) for n in self.nodes],
            "relationships": [dict(r) for r in self.relationships],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> DiagramProjection:
        return cls(
            nodes=[dict(n) for n in document.get("nodes") or []],
            relationships=[dict(r) for r in document.get("relationships") or []],
            metadata=dict(document.get("metadata") or {}),
        )

    def diagram_id_by_architecture_id(self) -> Dict[str, str]:
        return {
            n["architecture_node_id"]: n["diagram_id"] for n in self.nodes
        }


def projection_counts(projection: DiagramProjection) -> Dict[str, int]:
    return {
        "diagram_nodes": len(projection.nodes),
        "diagram_relationships": len(projection.relationships),
    }


__all__ = [
    "DIAGRAM_PROJECTION_SCHEMA_VERSION",
    "DIAGRAM_VALIDATION_SCHEMA_VERSION",
    "GRAPHVIZ_FILENAME",
    "MERMAID_FILENAME",
    "SVG_FILENAME",
    "DIAGRAM_VALIDATION_FILENAME",
    "DiagramProjection",
    "diagram_node_id",
    "projection_counts",
]
