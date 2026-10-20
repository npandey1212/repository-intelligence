"""M11.1 — persisted engineering graph artifact (engineering_graph.json).

Transformation pipeline (deterministic, no re-analysis, no LLM):

    engineering_model.json                 (M3 — source of truth)
        |
        | build_engineering_graph_document  (this module: ordering
        |                                   canonicalized, content
        |                                   copied verbatim)
        v
    engineering_graph.json                 (canonical persisted graph
                                            artifact: nodes,
                                            relationships, unresolved,
                                            metadata, provenance)
        |
        | project_engineering_graph         (M11.2, runtime only)
        v
    networkx.MultiDiGraph                   (bounded queries; never
                                             canonical)

The persisted graph is a faithful transformation of the model — never a
second source of truth. Node ids, relationship ids/kinds/endpoints,
classifications, attributes, evidence and unresolved claims are copied
verbatim (deep copies); only ordering is canonicalized (nodes by id,
relationships by id; nested collections keep their model order).
Unresolved claims are carried as data and never become edges. Nothing
is invented, enriched, resolved or discarded.

Write/read conventions match every other artifact:
``json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\\n"``.

Usage:

    python -m repo_intelligence.graph.engineering_graph_persist \\
        [--model PATH] [--output PATH] [--output-root DIR]
        [--snapshot-id ID]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..artifacts import (
    ENGINEERING_GRAPH_FILENAME,
    ENGINEERING_MODEL_FILENAME,
    engineering_graph_path,
    engineering_model_path,
    resolve_output_root,
)
from ..context.models import ArchitectureContextError, discover_snapshot_id
from ..engineering.validator import EngineeringModelError

ENGINEERING_GRAPH_SCHEMA_VERSION = "engineering_graph.v1"

__all__ = [
    "ENGINEERING_GRAPH_FILENAME",
    "ENGINEERING_GRAPH_SCHEMA_VERSION",
    "build_engineering_graph_document",
    "engineering_graph_document_problems",
    "generate_engineering_graph_document",
    "load_engineering_graph_document",
    "validate_engineering_graph_document",
    "write_engineering_graph_document",
]


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------


def _engineering_section(
    source: Any
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Accept a full EngineeringModel document or a bare ``engineering``
    section; return ``(section, model_document_or_None)``."""
    if not isinstance(source, dict):
        raise EngineeringModelError(
            "Expected an EngineeringModel document (JSON object)"
        )
    engineering = source.get("engineering")
    model_document: Optional[Dict[str, Any]] = None
    if engineering is None and "nodes" in source:
        engineering = source
    else:
        model_document = source
    if not isinstance(engineering, dict) or not isinstance(
        engineering.get("nodes"), list
    ):
        raise EngineeringModelError(
            "Expected EngineeringModel document with an engineering "
            "section containing a nodes list"
        )
    return engineering, model_document


def _list_section(
    engineering: Dict[str, Any], key: str, required: bool
) -> List[Dict[str, Any]]:
    value = engineering.get(key)
    if value is None:
        if required:
            raise EngineeringModelError(
                "Engineering section missing {!r}".format(key)
            )
        return []
    if not isinstance(value, list):
        raise EngineeringModelError(
            "Engineering section {!r} must be a list".format(key)
        )
    return value


def _repository_block(model_document: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Provenance essentials only (name, URL, commit) — copied, not
    re-derived."""
    if model_document is None:
        return {}
    repository = model_document.get("repository")
    if not isinstance(repository, dict):
        return {}
    block: Dict[str, Any] = {}
    for key in ("name", "url", "commit_sha"):
        if key in repository:
            block[key] = copy.deepcopy(repository[key])
    return block


# ---------------------------------------------------------------------------
# Validation (shared by build and load)
# ---------------------------------------------------------------------------


def engineering_graph_document_problems(document: Any) -> List[str]:
    """Structural problems of a persisted graph document (empty = valid)."""
    if not isinstance(document, dict):
        return ["document root is not a JSON object"]
    problems: List[str] = []
    schema_version = document.get("schema_version")
    if schema_version != ENGINEERING_GRAPH_SCHEMA_VERSION:
        problems.append(
            "schema_version must be {!r}, got {!r}".format(
                ENGINEERING_GRAPH_SCHEMA_VERSION, schema_version
            )
        )
    for key in ("nodes", "relationships", "unresolved"):
        if not isinstance(document.get(key), list):
            problems.append("{!r} must be a list".format(key))
    if problems:
        return problems

    nodes = document["nodes"]
    relationships = document["relationships"]
    node_ids = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            problems.append("nodes[{}] is not an object".format(index))
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            problems.append(
                "nodes[{}] has no non-empty string id".format(index)
            )
            continue
        if node_id in node_ids:
            problems.append("duplicate node id: {!r}".format(node_id))
            continue
        node_ids.add(node_id)

    relationship_ids = set()
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            problems.append("relationships[{}] is not an object".format(index))
            continue
        relationship_id = relationship.get("id")
        if not isinstance(relationship_id, str) or not relationship_id:
            problems.append(
                "relationships[{}] has no non-empty string id".format(index)
            )
            continue
        if relationship_id in relationship_ids:
            problems.append(
                "duplicate relationship id: {!r}".format(relationship_id)
            )
            continue
        relationship_ids.add(relationship_id)
        for endpoint_key in ("source", "target"):
            endpoint = relationship.get(endpoint_key)
            if not isinstance(endpoint, str) or not endpoint:
                problems.append(
                    "relationship {!r} missing {}".format(
                        relationship_id, endpoint_key
                    )
                )
            elif endpoint not in node_ids:
                problems.append(
                    "relationship {!r} {} {!r} not among graph nodes".format(
                        relationship_id, endpoint_key, endpoint
                    )
                )
    for index, unresolved in enumerate(document["unresolved"]):
        if not isinstance(unresolved, dict):
            problems.append("unresolved[{}] is not an object".format(index))

    metadata = document.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            problems.append("'metadata' must be an object")
        else:
            counts = metadata.get("counts")
            if counts is not None:
                if not isinstance(counts, dict):
                    problems.append("'metadata.counts' must be an object")
                else:
                    expected = {
                        "nodes": len(nodes),
                        "relationships": len(relationships),
                        "unresolved": len(document["unresolved"]),
                    }
                    for key, value in expected.items():
                        declared = counts.get(key)
                        if declared is not None and declared != value:
                            problems.append(
                                "metadata.counts.{} is {} but the document "
                                "holds {}".format(key, declared, value)
                            )
    return problems


def validate_engineering_graph_document(document: Any) -> None:
    """Raise ``EngineeringModelError`` unless the document is a valid
    persisted engineering graph."""
    problems = engineering_graph_document_problems(document)
    if problems:
        raise EngineeringModelError(
            "Invalid engineering graph document: {}".format(
                "; ".join(problems)
            )
        )


# ---------------------------------------------------------------------------
# M11.1 — build (deterministic transformation)
# ---------------------------------------------------------------------------


def build_engineering_graph_document(
    model_document: Any, snapshot_id: Optional[str] = None
) -> Dict[str, Any]:
    """EngineeringModel document → persisted engineering graph document.

    Deterministic: the output depends only on the model content.
    Node/relationship ordering is canonical (by id); every entry is a
    deep copy of the model's own record (evidence and attributes stay
    exactly as the model recorded them).
    """
    engineering, model_document_dict = _engineering_section(model_document)
    nodes = _list_section(engineering, "nodes", required=True)
    relationships = _list_section(engineering, "relationships", required=True)
    unresolved = _list_section(engineering, "unresolved", required=False)

    document: Dict[str, Any] = {
        "schema_version": ENGINEERING_GRAPH_SCHEMA_VERSION,
        "provenance": {
            "source_artifact": ENGINEERING_MODEL_FILENAME,
            "source_schema_version": (
                model_document_dict.get("schema_version")
                if model_document_dict is not None
                else None
            ),
            "repository": _repository_block(model_document_dict),
            "snapshot_id": snapshot_id,
            "generator": "repo_intelligence.graph.engineering_graph_persist",
        },
        "metadata": {
            "counts": {
                "nodes": len(nodes),
                "relationships": len(relationships),
                "unresolved": len(unresolved),
            },
        },
        "nodes": sorted(
            (copy.deepcopy(node) for node in nodes),
            key=lambda node: str(node.get("id")) if isinstance(node, dict) else "",
        ),
        "relationships": sorted(
            (copy.deepcopy(relationship) for relationship in relationships),
            key=lambda relationship: (
                str(relationship.get("id"))
                if isinstance(relationship, dict)
                else ""
            ),
        ),
        "unresolved": [copy.deepcopy(claim) for claim in unresolved],
    }
    validate_engineering_graph_document(document)
    return document


# ---------------------------------------------------------------------------
# Write / load
# ---------------------------------------------------------------------------


def write_engineering_graph_document(
    document: Dict[str, Any], path: Path
) -> Path:
    """Validate and persist the graph document with the project-wide
    serialization convention."""
    validate_engineering_graph_document(document)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False)
    dest.write_text(text + "\n", encoding="utf-8")
    return dest


def load_engineering_graph_document(path: Path) -> Dict[str, Any]:
    """Read and structurally validate a persisted engineering graph."""
    artifact = Path(path)
    try:
        raw = artifact.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EngineeringModelError(
            "Cannot read engineering graph {}: {}".format(artifact, exc)
        )
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise EngineeringModelError(
            "Engineering graph {} is not valid JSON: {}".format(artifact, exc)
        )
    problems = engineering_graph_document_problems(document)
    if problems:
        raise EngineeringModelError(
            "Invalid engineering graph {}: {}".format(
                artifact, "; ".join(problems)
            )
        )
    return document


# ---------------------------------------------------------------------------
# Generation entry point (CLI below)
# ---------------------------------------------------------------------------


def generate_engineering_graph_document(
    model_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
    graph_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Load the M3 model artifact, transform it, and write the M11 graph
    artifact beside it.

    Default paths follow the project layout: the current run's snapshot
    (discovered when ``snapshot_id`` is omitted) under ``output_root``.
    Returns ``(document, written_path)``.
    """
    root = resolve_output_root(output_root)
    if model_path is None:
        snapshot = snapshot_id
        if snapshot is None:
            snapshot = discover_snapshot_id(root)
        model_file = engineering_model_path(root, snapshot)
        if graph_path is None:
            graph_path = engineering_graph_path(root, snapshot)
    else:
        model_file = Path(model_path)
        if graph_path is None:
            graph_path = model_file.parent / ENGINEERING_GRAPH_FILENAME
    if not model_file.is_file():
        raise EngineeringModelError(
            "EngineeringModel not found: {}".format(model_file)
        )
    try:
        model_document = json.loads(model_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EngineeringModelError(
            "Cannot read EngineeringModel {}: {}".format(model_file, exc)
        )
    provenance_snapshot = snapshot_id
    if provenance_snapshot is None and model_path is None:
        provenance_snapshot = discover_snapshot_id(root)
    document = build_engineering_graph_document(
        model_document, snapshot_id=provenance_snapshot
    )
    written = write_engineering_graph_document(document, Path(graph_path))
    return document, written


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: write engineering_graph.json (exit 0 on success, 6 on error)."""
    parser = argparse.ArgumentParser(
        prog=(
            "python -m repo_intelligence.graph.engineering_graph_persist"
        ),
        description=(
            "Deterministically transform engineering_model.json into the "
            "persisted engineering_graph.json artifact."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to engineering_model.json (default: the current run's "
            "artifact under --output-root)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination path override (default: "
            "<output>/<snapshot_id>/engineering/engineering_graph.json)"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts.",
    )
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="Snapshot id (default: discovered from the artifact root).",
    )
    args = parser.parse_args(argv)
    try:
        document, written = generate_engineering_graph_document(
            model_path=Path(args.model) if args.model else None,
            output_root=Path(args.output_root) if args.output_root else None,
            snapshot_id=args.snapshot_id,
            graph_path=Path(args.output) if args.output else None,
        )
    except (EngineeringModelError, ArchitectureContextError) as exc:
        print(str(exc), file=sys.stderr)
        return 6
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "path": str(written),
                "counts": document["metadata"]["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
