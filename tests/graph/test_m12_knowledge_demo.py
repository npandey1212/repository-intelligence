"""M12 demonstration script tests (offline — no network, no LLM).

Covers the demo's own contract: module imports without side effects,
the offline slice preflight (counts + complete forward path over the
persisted graph), the spec §21 structured result (finding /
relationships / evidence_paths), stage classification over a
hand-built trace-shaped object, the exit-code gate, and configuration
failures exiting 6 before any LLM client is constructed.

The live run (python agentrun_m12_knowledge_demo.py) is manual and
never executed by this suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = PROJECT_ROOT / "agentrun_m12_knowledge_demo.py"
ADOP_GRAPH = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_graph.json"
)


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "agentrun_m12_knowledge_demo", str(DEMO_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iteration(index, response_type, tools=(), results=(), answer=None):
    return SimpleNamespace(
        iteration=index,
        response_type=response_type,
        reason_summary="step {}".format(index),
        tool_calls=[
            {"id": "call-{}".format(position), "name": name, "arguments": {}}
            for position, name in enumerate(tools)
        ],
        tool_results=[
            {
                "tool": name,
                "tool_call_id": "call-0",
                "success": success,
                "source": source,
            }
            for name, success, source in results
        ],
        final_answer=answer,
    )


def _trace(*iterations):
    return SimpleNamespace(iterations=list(iterations))


@pytest.fixture(scope="module")
def demo_pinned():
    """Demo module + offline slice over the pinned graph (skipped when
    the artifact is absent)."""
    if not ADOP_GRAPH.is_file():
        pytest.skip("pinned engineering graph artifact absent")
    demo = _load_demo()
    document, path = demo.check_graph_artifact(None)
    from repo_intelligence.graph import project_engineering_graph

    graph = project_engineering_graph(document)
    return demo, document, path, graph


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    demo = _load_demo()
    for name in (
        "check_graph_artifact",
        "slice_counts",
        "slice_present",
        "find_slice_path",
        "structured_slice_result",
        "classify_stages",
        "exit_code_for",
        "main",
        "DEFAULT_OBJECTIVE",
        "PATH_KINDS",
    ):
        assert hasattr(demo, name)
    assert demo.PATH_KINDS == ["DECLARES", "EXPOSED_BY", "CONFIGURED_BY"]
    # the spec §20 objective directs the bounded path walk
    for kind in demo.PATH_KINDS:
        assert kind in demo.DEFAULT_OBJECTIVE
    assert "query_engineering_graph" in demo.DEFAULT_OBJECTIVE
    assert "evidence" in demo.DEFAULT_OBJECTIVE


# ---------------------------------------------------------------------------
# Offline slice preflight (persisted graph only — no LLM)
# ---------------------------------------------------------------------------


def test_check_graph_artifact_reads_the_persisted_counts(demo_pinned):
    demo, document, path, _graph = demo_pinned
    assert path.name == "engineering_graph.json"
    assert path.is_file()
    assert document["schema_version"] == "engineering_graph.v1"
    assert document["metadata"]["counts"] == {
        "nodes": 1113,
        "relationships": 1627,
        "unresolved": 449,
    }


def test_slice_counts_from_the_projected_graph(demo_pinned):
    demo, _document, _path, graph = demo_pinned
    counts = demo.slice_counts(graph)
    assert counts == {
        "nodes_tool": 72,
        "nodes_mcp_server": 13,
        "rel_declares": 37,
        "rel_exposed_by": 66,
        "rel_configured_by": 13,
    }
    assert demo.slice_present(counts) is True


def test_slice_present_false_without_the_slice():
    demo = _load_demo()
    empty = {
        key: 0 for key in (
            "nodes_tool",
            "nodes_mcp_server",
            "rel_declares",
            "rel_exposed_by",
            "rel_configured_by",
        )
    }
    assert demo.slice_present(empty) is False
    partially = dict(empty)
    partially["nodes_tool"] = 5
    assert demo.slice_present(partially) is False


def test_find_slice_path_walks_a_complete_forward_chain(demo_pinned):
    demo, _document, _path, graph = demo_pinned
    report = demo.find_slice_path(graph)
    assert report is not None
    hops = report["hops"]
    assert [rel["kind"] for rel in hops] == [
        "DECLARES",
        "EXPOSED_BY",
        "CONFIGURED_BY",
    ]
    # the chain links node-to-node without gaps
    assert hops[0]["target"] == hops[1]["source"]
    assert hops[1]["target"] == hops[2]["source"]
    for rel in hops:
        assert rel["evidence"], rel["kind"]
    # the bounded query it ran is the path query
    assert report["query"]["direction"] == "out"
    assert report["query"]["relationship_kinds"] == demo.PATH_KINDS
    assert report["query"]["max_depth"] == 3
    # deterministic: the same walk comes back every time
    assert demo.find_slice_path(graph) == report


def test_structured_slice_result_is_reviewer_answerable(demo_pinned):
    demo, _document, _path, graph = demo_pinned
    result = demo.structured_slice_result(demo.find_slice_path(graph))
    assert set(result) == {"finding", "relationships", "evidence_paths"}
    assert [rel["kind"] for rel in result["relationships"]] == [
        "DECLARES",
        "EXPOSED_BY",
        "CONFIGURED_BY",
    ]
    for kind in demo.PATH_KINDS:
        assert kind in result["finding"]
    # evidence paths are concrete repository paths (path[:line] labels)
    assert result["evidence_paths"]
    for label in result["evidence_paths"]:
        assert label
        assert not label.startswith("/")
        if ":" in label:
            path_part, _, line = label.rpartition(":")
            assert path_part
            assert line.isdigit() and int(line) >= 1


# ---------------------------------------------------------------------------
# Stage classification (spec §14)
# ---------------------------------------------------------------------------


def test_classify_stages_buckets_every_stage_from_the_trace():
    demo = _load_demo()
    trace = _trace(
        _iteration(1, "TOOL_CALL", tools=("query_engineering_graph",)),
        _iteration(
            2,
            "TOOL_CALL",
            tools=("search_repository",),
            results=(
                ("query_engineering_graph", True, {"artifact": "g.json"}),
                ("search_repository", False, {"artifact": "r"}),
            ),
        ),
        _iteration(
            3,
            "FINAL_ANSWER",
            answer="the agent declares a tool exposed by the server",
        ),
    )

    stages = demo.classify_stages(trace)

    assert [entry["iteration"] for entry in stages["llm_reasoning"]] == [
        1,
        2,
        3,
    ]
    assert len(stages["graph_retrieval"]) == 1
    assert stages["graph_retrieval"][0]["iteration"] == 2
    assert stages["graph_retrieval"][0]["success"] is True
    assert len(stages["source_file_retrieval"]) == 1
    assert stages["structured_answer"] == (
        "the agent declares a tool exposed by the server"
    )
    assert stages["counts"] == {
        "iterations": 3,
        "graph_queries": 1,
        "source_file_calls": 1,
        "answer_recorded": True,
    }


def test_classify_stages_never_mutates_the_trace():
    demo = _load_demo()
    iteration = _iteration(
        1,
        "TOOL_CALL",
        tools=("query_engineering_graph",),
        results=(("query_engineering_graph", True, {}),),
        answer=None,
    )
    trace = _trace(iteration)
    before = (
        list(iteration.tool_calls),
        list(iteration.tool_results),
        iteration.final_answer,
    )
    demo.classify_stages(trace)
    after = (
        list(iteration.tool_calls),
        list(iteration.tool_results),
        iteration.final_answer,
    )
    assert before == after


# ---------------------------------------------------------------------------
# Exit-code gate (0 only with slice path + answer + a graph query)
# ---------------------------------------------------------------------------


def test_exit_code_zero_requires_all_three_gates():
    demo = _load_demo()
    slice_path = {"agent": "a", "hops": [], "query": {}}
    answered_with_graph = {
        "counts": {
            "iterations": 3,
            "graph_queries": 1,
            "source_file_calls": 0,
            "answer_recorded": True,
        }
    }
    answered_without_graph = {
        "counts": {
            "iterations": 3,
            "graph_queries": 0,
            "source_file_calls": 2,
            "answer_recorded": True,
        }
    }
    graph_but_no_answer = {
        "counts": {
            "iterations": 3,
            "graph_queries": 1,
            "source_file_calls": 0,
            "answer_recorded": False,
        }
    }
    assert demo.exit_code_for(answered_with_graph, slice_path) == 0
    assert demo.exit_code_for(answered_without_graph, slice_path) == 6
    assert demo.exit_code_for(graph_but_no_answer, slice_path) == 6
    # offline slice walk failing is never success, even with a run
    assert demo.exit_code_for(answered_with_graph, None) == 6


# ---------------------------------------------------------------------------
# Configuration failures (exit 6, offline, no client constructed)
# ---------------------------------------------------------------------------


def test_main_without_api_key_fails_clearly(monkeypatch, capsys):
    demo = _load_demo()
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = demo.main([])

    captured = capsys.readouterr()
    assert exit_code == 6
    assert "OPENAI_API_KEY" in captured.err


def test_main_without_a_run_context_fails_clearly(monkeypatch, capsys):
    demo = _load_demo()
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")

    def _no_run(output_root=None):
        raise demo.ArchitectureContextError("no discoverable run")

    monkeypatch.setattr(demo, "discover_snapshot_id", _no_run)

    exit_code = demo.main([])

    captured = capsys.readouterr()
    assert exit_code == 6
    assert "M12 demonstration failed." in captured.err


def test_main_without_the_graph_artifact_fails_clearly(monkeypatch, capsys):
    demo = _load_demo()
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(
        demo,
        "discover_snapshot_id",
        lambda output_root=None: "run_without_graph",
    )
    monkeypatch.setattr(
        demo,
        "engineering_graph_path",
        lambda output_root, snapshot_id: Path("/nonexistent/engineering/"
                                              "engineering_graph.json"),
    )

    exit_code = demo.main([])

    captured = capsys.readouterr()
    assert exit_code == 6
    assert "engineering_graph.json" in captured.err
    assert "engineering_graph_persist" in captured.err  # how to fix it


def test_main_without_the_slice_fails_clearly(monkeypatch, capsys):
    import networkx as nx

    demo = _load_demo()
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")

    def _empty_graph(document):
        return nx.MultiDiGraph()

    monkeypatch.setattr(
        demo,
        "project_engineering_graph",
        _empty_graph,
    )

    exit_code = demo.main([])

    captured = capsys.readouterr()
    assert exit_code == 6
    assert "M12 knowledge slice" in captured.err
    assert "engineering_graph_persist" in captured.err  # how to fix it
