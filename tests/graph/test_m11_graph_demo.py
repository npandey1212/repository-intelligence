"""M11 demonstration script tests (offline — no network, no LLM).

Covers the demo's own contract: module imports without side effects,
stage classification over a hand-built trace-shaped object (spec §14:
LLM reasoning / graph retrieval / source-file retrieval / structured
answer), the graph-queried exit-code gate, and configuration failures
exiting 6 before any LLM client is constructed.

The live run (python agentrun_m11_graph_demo.py) is manual and never
executed by this suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = PROJECT_ROOT / "agentrun_m11_graph_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "agentrun_m11_graph_demo", str(DEMO_PATH)
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


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    demo = _load_demo()
    for name in (
        "check_graph_artifact",
        "classify_stages",
        "exit_code_for",
        "main",
        "DEFAULT_OBJECTIVE",
    ):
        assert hasattr(demo, name)
    # the spec §14 objective is the default question
    assert "MCP" in demo.DEFAULT_OBJECTIVE
    assert "query_engineering_graph" in demo.DEFAULT_OBJECTIVE


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
            answer="found the MCP artifacts",
        ),
    )

    stages = demo.classify_stages(trace)

    assert [entry["iteration"] for entry in stages["llm_reasoning"]] == [
        1,
        2,
        3,
    ]
    assert [entry["response_type"] for entry in stages["llm_reasoning"]] == [
        "TOOL_CALL",
        "TOOL_CALL",
        "FINAL_ANSWER",
    ]
    assert len(stages["graph_retrieval"]) == 1
    assert stages["graph_retrieval"][0]["iteration"] == 2
    assert stages["graph_retrieval"][0]["success"] is True
    assert len(stages["source_file_retrieval"]) == 1
    assert stages["source_file_retrieval"][0]["success"] is False
    assert stages["structured_answer"] == "found the MCP artifacts"
    assert stages["counts"] == {
        "iterations": 3,
        "graph_queries": 1,
        "source_file_calls": 1,
        "answer_recorded": True,
    }


def test_classify_stages_without_graph_or_answer_is_reported():
    demo = _load_demo()
    stages = demo.classify_stages(
        _trace(_iteration(1, "FINAL_ANSWER", answer="no tools used"))
    )
    assert stages["counts"]["graph_queries"] == 0
    assert stages["structured_answer"] == "no tools used"
    stages = demo.classify_stages(_trace())
    assert stages["counts"]["answer_recorded"] is False


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
# Exit-code gate (0 only with a recorded answer AND a graph query)
# ---------------------------------------------------------------------------


def test_exit_code_zero_requires_a_graph_query():
    demo = _load_demo()
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
    assert demo.exit_code_for(answered_with_graph) == 0
    assert demo.exit_code_for(answered_without_graph) == 6
    assert demo.exit_code_for(graph_but_no_answer) == 6


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
    assert "M11 demonstration failed." in captured.err


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


def test_check_graph_artifact_reads_the_persisted_counts():
    demo = _load_demo()
    document, path = demo.check_graph_artifact()
    assert path.name == "engineering_graph.json"
    assert path.is_file()
    assert document["schema_version"] == "engineering_graph.v1"
    assert document["metadata"]["counts"] == {
        "nodes": 1113,
        "relationships": 1627,
        "unresolved": 449,
    }
