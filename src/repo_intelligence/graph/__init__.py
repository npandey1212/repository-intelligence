"""M5 NetworkX graph runtime, diagram exporters, and the M11
engineering graph projection + query API.

Architecture pipeline (M5):

    M4 ArchitectureModel JSON (canonical)
        ↓
    networkx.DiGraph (runtime only — never canonical)
        ↓
    DiagramProjection (validated intermediate)
        ↓
     ┌─────┴─────┐
     ▼           ▼
  Mermaid      Graphviz DOT  (→ optional SVG presentation)
  .mmd           .dot

Engineering pipeline (M11):

    M3 EngineeringModel JSON (source of truth — never NetworkX)
        ↓ build_engineering_graph_document (deterministic, verbatim)
    engineering_graph.json (canonical persisted graph artifact)
        ↓ project_engineering_graph / build_engineering_graph
    networkx.MultiDiGraph (runtime only — never canonical;
                            multi-edges keyed by relationship id,
                            unresolved claims are never edges)
        ↓ deterministic query API (bounded traversal + filtering,
                             named per-operation façade; the
                             agent-facing tool sits on top)

Non-goals: UI, LLM inference, graph databases, making NetworkX/Mermaid/DOT
the architecture source of truth.
"""

from __future__ import annotations

from .adapter import build_architecture_graph, graph_invariant_problems
from .builder import (
    DiagramError,
    DiagramValidationError,
    build_validation_report,
    create_diagram_artifacts,
    load_architecture_model_document,
)
from .engineering_adapter import (
    build_engineering_graph,
    engineering_graph_invariant_problems,
    project_engineering_graph,
)
from .engineering_graph_persist import (
    ENGINEERING_GRAPH_FILENAME,
    ENGINEERING_GRAPH_SCHEMA_VERSION,
    build_engineering_graph_document,
    engineering_graph_document_problems,
    generate_engineering_graph_document,
    load_engineering_graph_document,
    validate_engineering_graph_document,
    write_engineering_graph_document,
)
from .engineering_query import (
    DEFAULT_DIRECTION,
    DEFAULT_QUERY_DEPTH,
    MAX_QUERY_DEPTH,
    MAX_QUERY_NODES,
    MAX_QUERY_RELATIONSHIPS,
    QUERY_DIRECTIONS,
    EngineeringGraphQueryResult,
    GraphQueryError,
    find_nodes,
    find_nodes_by_property,
    get_node,
    neighbors,
    query_engineering_graph,
    relationships,
    traverse,
)
from .exporters import (
    export_graphviz,
    export_mermaid,
    graphviz_available,
    render_graphviz,
    render_mermaid,
    render_svg,
)
from .models import (
    DIAGRAM_PROJECTION_SCHEMA_VERSION,
    DIAGRAM_VALIDATION_FILENAME,
    DIAGRAM_VALIDATION_SCHEMA_VERSION,
    GRAPHVIZ_FILENAME,
    MERMAID_FILENAME,
    SVG_FILENAME,
    DiagramProjection,
    diagram_node_id,
    projection_counts,
)
from .projection import build_diagram_projection
from .validator import validate_diagram_projection, validate_diagram_projection_or_raise

__all__ = [
    "DEFAULT_DIRECTION",
    "DEFAULT_QUERY_DEPTH",
    "DIAGRAM_PROJECTION_SCHEMA_VERSION",
    "DIAGRAM_VALIDATION_FILENAME",
    "DIAGRAM_VALIDATION_SCHEMA_VERSION",
    "ENGINEERING_GRAPH_FILENAME",
    "ENGINEERING_GRAPH_SCHEMA_VERSION",
    "GRAPHVIZ_FILENAME",
    "MAX_QUERY_DEPTH",
    "MAX_QUERY_NODES",
    "MAX_QUERY_RELATIONSHIPS",
    "MERMAID_FILENAME",
    "QUERY_DIRECTIONS",
    "SVG_FILENAME",
    "DiagramError",
    "DiagramProjection",
    "DiagramValidationError",
    "EngineeringGraphQueryResult",
    "GraphQueryError",
    "build_architecture_graph",
    "build_diagram_projection",
    "build_engineering_graph",
    "build_engineering_graph_document",
    "build_validation_report",
    "create_diagram_artifacts",
    "diagram_node_id",
    "engineering_graph_document_problems",
    "engineering_graph_invariant_problems",
    "export_graphviz",
    "export_mermaid",
    "find_nodes",
    "find_nodes_by_property",
    "generate_engineering_graph_document",
    "get_node",
    "graph_invariant_problems",
    "graphviz_available",
    "load_architecture_model_document",
    "load_engineering_graph_document",
    "neighbors",
    "project_engineering_graph",
    "projection_counts",
    "query_engineering_graph",
    "render_graphviz",
    "render_mermaid",
    "render_svg",
    "relationships",
    "traverse",
    "validate_diagram_projection",
    "validate_diagram_projection_or_raise",
    "validate_engineering_graph_document",
    "write_engineering_graph_document",
]
