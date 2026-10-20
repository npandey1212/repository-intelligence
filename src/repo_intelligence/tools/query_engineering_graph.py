"""Tool 4 (read-only): query_engineering_graph.

Deterministic, bounded graph queries over the already-generated
EngineeringModel artifact — the NetworkX projection built by
``build_engineering_graph``. One question for a future agent runtime:

    "Given this node id, which artifacts/code entities does it
    contain / depend on / reference — and with what evidence?"

Two modes (mirroring the query layer):

    query_engineering_graph({})                     → node-kind summary
                                                      (discovery: counts +
                                                      bounded id samples)
    query_engineering_graph({"start_node_id": ...}) → bounded neighborhood
                                                      traversal

Execution path (no LLM, no graph database, no mutation):

    query_engineering_graph(arguments)
        → validate the input contract (allowed keys only)
        → resolve the run's persisted graph artifact (application
          context): engineering_graph.json — the canonical, persisted
          NetworkX projection of engineering_model.json (M11.1); it is
          the only artifact the run-context path reads
        → project it into a read-only MultiDiGraph (read-only
          projection; nothing is rebuilt, nothing is repaired)
        → run one bounded query (GraphQueryError → structured failure)
        → return ToolResult with the query result + provenance
          (which artifact was read, counts, commit, snapshot)

There is no silent path from engineering_model.json: when the graph
artifact is absent, the tool fails with ENGINEERING_GRAPH_NOT_FOUND
instead of projecting the model. An explicit ``engineering_model_path``
(testing/standalone only — agents never supply application context)
still projects that file directly, so lower-level tests keep
EngineeringModel → NetworkX construction without the agent-facing
path ever depending on it.

The tool never writes artifacts: a missing/corrupted graph artifact is
never repaired in place, only reported (ENGINEERING_GRAPH_NOT_FOUND /
MODEL_INVALID).

Standalone (JSON ToolResult on stdout):

    python -m repo_intelligence.tools.query_engineering_graph \
        --start-node-id "..." --direction out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..architecture.builder import (
    ArchitectureModelError,
    load_engineering_model_document,
)
from ..artifacts import (
    ENGINEERING_GRAPH_FILENAME,
    ENGINEERING_MODEL_FILENAME,
    engineering_graph_path,
    resolve_output_root,
)
from ..context.models import ArchitectureContextError, discover_snapshot_id
from ..engineering.validator import EngineeringModelError
from ..graph.engineering_query import GraphQueryError, query_engineering_graph
from ..graph.engineering_adapter import (
    build_engineering_graph,
    project_engineering_graph,
)
from ..graph.engineering_graph_persist import (
    load_engineering_graph_document,
)
from .protocol import ToolError, ToolResult

TOOL_NAME = "query_engineering_graph"

_ALLOWED_ARGUMENTS = (
    "start_node_id",
    "direction",
    "relationship_kinds",
    "node_kinds",
    "max_depth",
)


def _failure(code: str, message: str) -> ToolResult:
    return ToolResult(
        success=False, tool=TOOL_NAME, error=ToolError(code=code, message=message)
    )


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def _validate_arguments(
    arguments: Any,
) -> Optional[ToolResult]:
    """Structural validation; values are validated by the query layer
    (same deterministic codes: INVALID_ARGUMENT / NODE_NOT_FOUND)."""
    if arguments is None:
        return None
    if not isinstance(arguments, Mapping):
        return _failure(
            "INVALID_ARGUMENT",
            "arguments must be a mapping; omit start_node_id for the "
            "node-kind summary.",
        )
    unknown = [str(key) for key in arguments if key not in _ALLOWED_ARGUMENTS]
    if unknown:
        return _failure(
            "INVALID_ARGUMENT",
            "unexpected argument(s): {}.".format(", ".join(sorted(unknown))),
        )
    return None


# ---------------------------------------------------------------------------
# Artifact resolution (application context)
# ---------------------------------------------------------------------------


def _snapshot_id_for_path(path: Path) -> Optional[str]:
    """Best-effort snapshot id from the artifact layout
    <output>/<snapshot_id>/engineering/<artifact>.json."""
    parent = path.parent
    return parent.parent.name if parent.name == "engineering" else None


def _resolve_artifact_paths(
    output_root: Optional[Path],
    engineering_model_path_arg: Optional[Path],
    snapshot_id: Optional[str] = None,
) -> tuple:
    """Return ``(graph_path, model_path, snapshot_id, failure)``.

    * explicit ``engineering_model_path_arg`` (testing/standalone
      only, never agent-supplied): that file is projected directly —
      the sanctioned EngineeringModel → NetworkX construction path for
      tests;
    * run context: the persisted ``engineering_graph.json`` and nothing
      else. A missing graph artifact fails with
      ``ENGINEERING_GRAPH_NOT_FOUND``; ``engineering_model.json`` is
      never consulted, so the agent cannot silently bypass the
      persisted M11 artifact.
    At most one failure.
    """
    if engineering_model_path_arg is not None:
        path = Path(engineering_model_path_arg)
        if not path.is_file():
            return None, None, None, _failure(
                "MODEL_NOT_FOUND",
                "EngineeringModel artifact was not found: {}.".format(path),
            )
        return None, path, _snapshot_id_for_path(path), None

    root = resolve_output_root(output_root)
    if snapshot_id is None:
        try:
            snapshot_id = discover_snapshot_id(root)
        except ArchitectureContextError as exc:
            return None, None, None, _failure("MODEL_NOT_FOUND", str(exc))
    elif (
        not isinstance(snapshot_id, str)
        or not snapshot_id.strip()
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", ".."}
    ):
        return None, None, None, _failure(
            "INVALID_SNAPSHOT_ID",
            "snapshot_id must be a non-empty single path component.",
        )
    else:
        snapshot_id = snapshot_id.strip()
    graph_path = engineering_graph_path(root, snapshot_id)
    if graph_path.is_file():
        return graph_path, None, snapshot_id, None
    return None, None, snapshot_id, _failure(
        "ENGINEERING_GRAPH_NOT_FOUND",
        "{} was not found for run {!r} (expected {}). The agent-facing "
        "graph tool reads only the persisted graph artifact — it does "
        "not fall back to {}; generate the graph artifact first "
        "(python -m repo_intelligence.graph.engineering_graph_persist) "
        "and query again.".format(
            ENGINEERING_GRAPH_FILENAME,
            snapshot_id,
            graph_path,
            ENGINEERING_MODEL_FILENAME,
        ),
    )


def _graph_source(
    artifact: Path,
    derived_from: str,
    schema_version: Any,
    model_schema_version: Any,
    commit_sha: Any,
    snapshot_id: Optional[str],
    graph: Any,
) -> Dict[str, Any]:
    return {
        "artifact": str(artifact),
        "derived_from": derived_from,
        "schema_version": schema_version,
        "model_schema_version": model_schema_version,
        "commit_sha": commit_sha,
        "snapshot_id": snapshot_id,
        "counts": {
            "nodes": graph.graph.get("node_count"),
            "relationships": graph.graph.get("relationship_count"),
            "unresolved": graph.graph.get("unresolved_count"),
        },
    }


def _load_projected_graph(
    graph_path: Optional[Path],
    model_path: Optional[Path],
    snapshot_id: Optional[str],
) -> tuple:
    """Load the resolved artifact and project it read-only.

    Returns ``(graph, source, failure)`` — at most one failure. The
    run-context path always resolves ``graph_path`` (the persisted
    artifact); ``model_path`` is set only by the explicit
    testing/standalone path.
    """
    if graph_path is not None:
        try:
            document = load_engineering_graph_document(graph_path)
        except EngineeringModelError as exc:
            return None, None, _failure("MODEL_INVALID", str(exc))
        try:
            graph = project_engineering_graph(document)
        except EngineeringModelError as exc:
            return None, None, _failure("MODEL_INVALID", str(exc))
        provenance = document.get("provenance") or {}
        repository = provenance.get("repository") or {}
        source = _graph_source(
            artifact=graph_path,
            derived_from=ENGINEERING_GRAPH_FILENAME,
            schema_version=document.get("schema_version"),
            model_schema_version=provenance.get("source_schema_version"),
            commit_sha=repository.get("commit_sha"),
            snapshot_id=snapshot_id or provenance.get("snapshot_id"),
            graph=graph,
        )
        return graph, source, None

    assert model_path is not None  # resolution never leaves both unset
    try:
        document = load_engineering_model_document(model_path)
    except ArchitectureModelError as exc:
        return None, None, _failure("MODEL_INVALID", str(exc))
    try:
        graph = build_engineering_graph(document)
    except EngineeringModelError as exc:
        return None, None, _failure("MODEL_INVALID", str(exc))
    repository = document.get("repository") or {}
    source = _graph_source(
        artifact=model_path,
        derived_from=ENGINEERING_MODEL_FILENAME,
        schema_version=document.get("schema_version"),
        model_schema_version=document.get("schema_version"),
        commit_sha=repository.get("commit_sha")
        if isinstance(repository, dict)
        else None,
        snapshot_id=snapshot_id,
        graph=graph,
    )
    return graph, source, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def query_engineering_graph_tool(
    arguments: Optional[Mapping[str, Any]] = None,
    output_root: Optional[Path] = None,
    engineering_model_path_arg: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
) -> ToolResult:
    """Run one deterministic, bounded, read-only engineering graph query.

    Input contract::

        {
            # traversal mode (omit nothing — all keys optional):
            "start_node_id": "artifact:file:...",   # non-empty string
            "direction": "out",                     # out | in | both
            "relationship_kinds": ["CONTAINS"],     # vocabulary subset
            "node_kinds": ["artifact/file"],        # kind or kind/subkind
            "max_depth": 2                          # 0..6
        }

        {}  → node-kind summary (discovery: counts + bounded id samples)

    ``output_root``, ``engineering_model_path_arg``, and ``snapshot_id``
    are application context (artifact locations), not tool arguments;
    agents never supply them. A supplied snapshot ID pins the query to
    the already-verified repository run. Run-context calls read the persisted
    ``engineering_graph.json`` and nothing else: a missing graph
    artifact fails with ``ENGINEERING_GRAPH_NOT_FOUND`` — there is no
    silent projection of ``engineering_model.json``. An explicit
    ``engineering_model_path_arg`` (testing/standalone only) projects
    that file directly.
    """
    failure = _validate_arguments(arguments)
    if failure is not None:
        return failure
    query_args: Dict[str, Any] = dict(arguments or {})

    graph_path, model_path, resolved_snapshot_id, failure = _resolve_artifact_paths(
        output_root, engineering_model_path_arg, snapshot_id=snapshot_id
    )
    if failure is not None:
        return failure
    assert model_path is not None or graph_path is not None

    graph, source, failure = _load_projected_graph(
        graph_path, model_path, resolved_snapshot_id
    )
    if failure is not None:
        return failure
    assert graph is not None and source is not None

    try:
        result = query_engineering_graph(graph, **query_args)
    except GraphQueryError as exc:
        return _failure(exc.code, exc.message)

    return ToolResult(
        success=True,
        tool=TOOL_NAME,
        data=result.to_dict(),
        source=source,
        error=None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Standalone entry point: print the ToolResult as JSON on stdout."""
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.tools.query_engineering_graph",
        description=(
            "Read-only tool: deterministic bounded query over the current "
            "run's persisted engineering graph; prints the ToolResult as JSON."
        ),
    )
    parser.add_argument(
        "--start-node-id",
        dest="start_node_id",
        default=None,
        help="node id to traverse from (omit for the node-kind summary)",
    )
    parser.add_argument(
        "--direction",
        default=None,
        choices=("out", "in", "both"),
        help="edge direction for traversal (default: out)",
    )
    parser.add_argument(
        "--relationship-kinds",
        dest="relationship_kinds",
        default=None,
        help="comma-separated relationship kinds to follow "
        "(CONTAINS, IMPORTS, DEPENDS_ON, TRIGGERS, "
        "DECLARES, EXPOSED_BY, CONFIGURED_BY)",
    )
    parser.add_argument(
        "--node-kinds",
        dest="node_kinds",
        default=None,
        help="comma-separated node kinds ('kind' or 'kind/subkind') the "
        "walk may visit",
    )
    parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=None,
        help="traversal depth 0..6 (default: 1)",
    )
    parser.add_argument(
        "--engineering-model-path",
        dest="engineering_model_path",
        default=None,
        help="explicit EngineeringModel JSON path (testing and standalone "
        "use only — agents never provide this; default: the current run's "
        "engineering_graph.json under <output>/<snapshot>/engineering/, "
        "required — no engineering_model.json fallback)",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="application context: artifact output root "
        "(default: <project>/artifacts)",
    )
    args = parser.parse_args(argv)

    arguments: Dict[str, Any] = {}
    if args.start_node_id is not None:
        arguments["start_node_id"] = args.start_node_id
    if args.direction is not None:
        arguments["direction"] = args.direction
    if args.relationship_kinds is not None:
        arguments["relationship_kinds"] = [
            kind.strip()
            for kind in args.relationship_kinds.split(",")
            if kind.strip()
        ]
    if args.node_kinds is not None:
        arguments["node_kinds"] = [
            kind.strip() for kind in args.node_kinds.split(",") if kind.strip()
        ]
    if args.max_depth is not None:
        arguments["max_depth"] = args.max_depth

    result = query_engineering_graph_tool(
        arguments,
        output_root=Path(args.output_root) if args.output_root else None,
        engineering_model_path_arg=(
            Path(args.engineering_model_path) if args.engineering_model_path else None
        ),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 6


if __name__ == "__main__":
    sys.exit(main())
