"""Spec §33 J/K — graph regression and no fabricated relationships.

Building and querying the RAG index must be provably incapable of
touching the Engineering Graph: the persisted artifact stays
byte-identical, graph queries answer the same before and after
retrieval, and retrieval never contributes an edge.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import enable_search, install_pinned_run
from repo_intelligence.graph import (
    build_engineering_graph_document,
    write_engineering_graph_document,
)
from repo_intelligence.tools.query_engineering_graph import (
    query_engineering_graph_tool,
)
from repo_intelligence.tools.search_repository import search_repository

COMMIT = "9" * 40

FILES = {
    "pkg/mod.py": "def main():\n    return 42\n",
    "docs/guide.md": "# Guide\n\nThe checkout flow starts here.\n",
}

MODEL = {
    "schema_version": "repository_knowledge_model.v1",
    "repository": {
        "name": "graph-fixture",
        "url": "https://example.invalid/graph.git",
        "commit_sha": COMMIT,
    },
    "engineering": {
        "nodes": [
            {
                "id": "artifact:directory:pkg",
                "kind": "artifact",
                "subkind": "directory",
                "classification": "observed",
                "evidence": [],
                "attributes": {},
            },
            {
                "id": "artifact:file:pkg/mod.py",
                "kind": "artifact",
                "subkind": "file",
                "classification": "observed",
                "evidence": [
                    {
                        "path": "pkg/mod.py",
                        "artifact_type": "python",
                        "commit": COMMIT,
                    }
                ],
                "attributes": {},
            },
        ],
        "relationships": [
            {
                "id": (
                    "rel:CONTAINS:artifact:directory:pkg"
                    ":artifact:file:pkg/mod.py"
                ),
                "kind": "CONTAINS",
                "source": "artifact:directory:pkg",
                "target": "artifact:file:pkg/mod.py",
                "classification": "derived",
                "evidence": [
                    {
                        "path": "pkg/mod.py",
                        "artifact_type": "python",
                        "commit": COMMIT,
                    }
                ],
            }
        ],
        "unresolved": [],
        "metadata": {},
    },
}


def _fixture(tmp_path: Path, monkeypatch, index: bool = True):
    output_root = tmp_path / "artifacts"
    checkout = install_pinned_run(
        output_root,
        dict(FILES),
        run_id="run_graph",
        repository_name="graph-fixture",
        commit_sha=COMMIT,
    )
    run_dir = output_root / "run_graph"
    engineering = run_dir / "engineering"
    engineering.mkdir(parents=True, exist_ok=True)
    (engineering / "engineering_model.json").write_text(
        json.dumps(MODEL), encoding="utf-8"
    )
    graph_path = engineering / "engineering_graph.json"
    write_engineering_graph_document(
        build_engineering_graph_document(
            MODEL, snapshot_id="run_graph"
        ),
        graph_path,
    )
    if index:
        enable_search(checkout, monkeypatch, tmp_path)
    return checkout, output_root, graph_path


def _graph_census(output_root: Path) -> dict:
    result = query_engineering_graph_tool({}, output_root=output_root)
    assert result.success is True
    return dict(result.data)


def _graph_walk(output_root: Path, node_id: str) -> dict:
    result = query_engineering_graph_tool(
        {
            "start_node_id": node_id,
            "direction": "both",
            "max_depth": 3,
        },
        output_root=output_root,
    )
    assert result.success is True
    return dict(result.data)


def test_graph_queries_work_before_and_after_index_build(
    tmp_path, monkeypatch
):
    """Graph regression: the persisted graph answers identically
    before and after the retrieval index exists."""
    checkout, output_root, graph_path = _fixture(
        tmp_path, monkeypatch, index=False
    )
    before_bytes = graph_path.read_bytes()
    census_before = _graph_census(output_root)
    walk_before = _graph_walk(
        output_root, "artifact:directory:pkg"
    )

    # now build the retrieval index and query it — the graph
    # answers must be bit-for-bit the same
    enable_search(checkout, monkeypatch, tmp_path)
    result = search_repository(
        {"query": "checkout flow"},
        repository_root=checkout,
        output_root=output_root,
    )
    assert result.success is True
    result = search_repository(
        {"query": "def main"},
        repository_root=checkout,
        output_root=output_root,
    )
    assert result.success is True

    assert graph_path.read_bytes() == before_bytes
    assert _graph_census(output_root) == census_before
    assert _graph_walk(output_root, "artifact:directory:pkg") == (
        walk_before
    )


def test_retrieval_never_fabricates_graph_relationships(
    tmp_path, monkeypatch
):
    """No fabricated relationships: the graph still reports exactly
    the one derived CONTAINS edge after RAG queries."""
    checkout, output_root, graph_path = _fixture(
        tmp_path, monkeypatch
    )
    for query in (
        "checkout flow",
        "agent uses tool",
        "pkg mod",
        "completely unrelated phrasing",
    ):
        result = search_repository(
            {"query": query},
            repository_root=checkout,
            output_root=output_root,
        )
        assert result.success is True

    walk = _graph_walk(output_root, "artifact:directory:pkg")
    relationships = walk["relationships"]
    assert len(relationships) == 1
    rel = relationships[0]
    assert rel["kind"] == "CONTAINS"
    assert rel["source"] == "artifact:directory:pkg"
    assert rel["target"] == "artifact:file:pkg/mod.py"
    document = json.loads(graph_path.read_text(encoding="utf-8"))
    assert len(document["relationships"]) == 1


def test_index_build_never_writes_into_the_run(
    tmp_path, monkeypatch
):
    """Building or querying the index writes nothing under the run
    directory (facts/architecture/engineering stay untouched)."""
    checkout, output_root, _graph_path = _fixture(
        tmp_path, monkeypatch
    )
    run_dir = output_root / "run_graph"

    def snapshot() -> dict:
        state = {}
        for path in sorted(run_dir.rglob("*")):
            rel = path.relative_to(run_dir).as_posix()
            if path.is_file():
                state[rel] = path.read_bytes()
            else:
                state[rel] = b"<dir>"
        return state

    before = snapshot()
    result = search_repository(
        {"query": "checkout flow"},
        repository_root=checkout,
        output_root=output_root,
    )
    assert result.success is True
    assert snapshot() == before


def test_graph_tool_stays_read_only_for_queries(
    tmp_path, monkeypatch
):
    _checkout, output_root, graph_path = _fixture(
        tmp_path, monkeypatch
    )
    before = graph_path.read_bytes()
    _graph_census(output_root)
    _graph_walk(output_root, "artifact:file:pkg/mod.py")
    _graph_walk(output_root, "artifact:directory:pkg")
    assert graph_path.read_bytes() == before
