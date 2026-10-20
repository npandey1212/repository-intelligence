"""M5 diagram pipeline: ArchitectureModel → NetworkX → Projection → exports.

Consumes the already-generated M4 architecture artifact. Does not re-run
repository analysis. Writes deterministic diagram artifacts under
artifacts/<snapshot_id>/diagram/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..architecture.validator import ArchitectureModelError, validate_knowledge_model_or_raise
from ..artifacts import (
    PRIMARY_SNAPSHOT_ID,
    architecture_model_path as artifact_architecture_model_path,
    default_artifacts_root,
    diagram_dir,
    resolve_output_root,
)
from .adapter import build_architecture_graph, graph_invariant_problems
from .exporters import export_graphviz, export_mermaid, graphviz_available, render_svg
from .models import (
    DIAGRAM_VALIDATION_FILENAME,
    DIAGRAM_VALIDATION_SCHEMA_VERSION,
    GRAPHVIZ_FILENAME,
    MERMAID_FILENAME,
    SVG_FILENAME,
    DiagramProjection,
)
from .projection import build_diagram_projection
from .validator import (
    DiagramError,
    DiagramValidationError,
    validate_diagram_projection,
    validate_diagram_projection_or_raise,
)


def load_architecture_model_document(path: Path) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ArchitectureModelError(
            "ArchitectureModel not found: {}".format(source)
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArchitectureModelError(
            "Unable to read ArchitectureModel {}: {}".format(source, exc)
        )
    if not isinstance(document, dict) or "architecture" not in document:
        raise ArchitectureModelError(
            "Expected repository_knowledge_model document with architecture "
            "section: {}".format(source)
        )
    return document


def build_validation_report(
    projection: DiagramProjection,
    document: Dict[str, Any],
    graph_problems: List[str],
    projection_problems: List[str],
    snapshot_id: Optional[str] = None,
) -> Dict[str, Any]:
    problems = list(graph_problems) + list(projection_problems)
    architecture = document.get("architecture") or {}
    metadata = document.get("metadata") or {}
    return {
        "schema_version": DIAGRAM_VALIDATION_SCHEMA_VERSION,
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "counts": {
            "architecture_nodes": len(architecture.get("nodes") or []),
            "architecture_relationships": len(
                architecture.get("relationships") or []
            ),
            "architecture_unresolved": len(
                architecture.get("unresolved") or []
            ),
            "diagram_nodes": len(projection.nodes),
            "diagram_relationships": len(projection.relationships),
        },
        "commit": (document.get("repository") or {}).get("commit_sha"),
        "snapshot_id": snapshot_id or metadata.get("snapshot_id"),
    }


def create_diagram_artifacts(
    architecture_model_path: Optional[Union[Path, str]] = None,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
    render_svg_output: bool = True,
) -> Tuple[DiagramProjection, Dict[str, Path]]:
    """Load M4 ArchitectureModel, export Mermaid/DOT, write validation.json.

    Returns (projection, paths) where paths includes mermaid, graphviz,
    validation, and svg when Graphviz is available and render_svg_output.
    """
    root = resolve_output_root(output_root)
    if snapshot_id is None:
        snapshot_id = PRIMARY_SNAPSHOT_ID

    if architecture_model_path is None:
        model_path = artifact_architecture_model_path(root, snapshot_id)
    else:
        model_path = Path(architecture_model_path)

    document = load_architecture_model_document(model_path)
    try:
        validate_knowledge_model_or_raise(
            document, require_architecture=True
        )
    except Exception as exc:
        raise DiagramError(str(exc))

    graph = build_architecture_graph(document)
    graph_problems = graph_invariant_problems(
        graph, document.get("architecture") or {}
    )

    projection = build_diagram_projection(graph)
    projection_problems = validate_diagram_projection(projection, document)

    report = build_validation_report(
        projection, document, graph_problems, projection_problems, snapshot_id
    )

    dest_dir = diagram_dir(root, snapshot_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    validation_path = dest_dir / DIAGRAM_VALIDATION_FILENAME
    validation_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    if report["status"] != "PASS":
        validate_diagram_projection_or_raise(projection, document)
        if graph_problems:
            raise DiagramValidationError(
                "Architecture graph failed validation:\n  - "
                + "\n  - ".join(graph_problems)
            )

    mermaid_path = dest_dir / MERMAID_FILENAME
    graphviz_path = dest_dir / GRAPHVIZ_FILENAME
    export_mermaid(projection, mermaid_path)
    export_graphviz(projection, graphviz_path)

    paths: Dict[str, Path] = {
        "mermaid": mermaid_path,
        "graphviz": graphviz_path,
        "validation": validation_path,
    }

    if render_svg_output and graphviz_available():
        svg_path = render_svg(graphviz_path, dest_dir / SVG_FILENAME)
        if svg_path is not None:
            paths["svg"] = svg_path

    return projection, paths


__all__ = [
    "DiagramError",
    "DiagramValidationError",
    "build_validation_report",
    "create_diagram_artifacts",
    "load_architecture_model_document",
]
