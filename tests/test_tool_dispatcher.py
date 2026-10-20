"""M8.3 ToolDispatcher tests — application-owned execution boundary.

Registry resolution, safe unknown-tool handling, argument guards, the
structured failure protocol, and exactly-once execution. Tool-internal
behavior stays in ``tests/tools/`` (``search_repository``,
``read_file``, ``get_architecture_context``,
``query_engineering_graph`` have their own suites).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import repo_intelligence.artifacts as artifacts
from conftest import enable_search
from repo_intelligence.graph import (
    build_engineering_graph_document,
    write_engineering_graph_document,
)
from repo_intelligence.llm import ToolCall
from repo_intelligence.tools import ToolResult
from repo_intelligence.tool_dispatcher import (
    ToolDispatcher,
    default_tool_dispatcher,
)

REGISTRY = (
    "get_architecture_context",
    "query_engineering_graph",
    "read_file",
    "search_repository",
)

# Minimal synthetic EngineeringModel used by the graph-tool dispatch test
# (no repository-specific ids anywhere).
_DISPATCH_ENGINEERING_MODEL = {
    "schema_version": "repository_knowledge_model.v1",
    "repository": {
        "name": "sample-repo-alpha",
        "url": "https://example.invalid/sample.git",
        "commit_sha": "b" * 40,
    },
    "engineering": {
        "nodes": [
            {
                "id": "artifact:directory:pkg",
                "kind": "artifact",
                "subkind": "directory",
                "classification": "observed",
                "evidence": [
                    {"path": "pkg", "artifact_type": "directory", "commit": "b" * 40}
                ],
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
                        "commit": "b" * 40,
                    }
                ],
                "attributes": {},
            },
        ],
        "relationships": [
            {
                "id": "rel:CONTAINS:artifact:directory:pkg:artifact:file:pkg/mod.py",
                "kind": "CONTAINS",
                "source": "artifact:directory:pkg",
                "target": "artifact:file:pkg/mod.py",
                "classification": "derived",
                "evidence": [
                    {"path": "pkg", "artifact_type": "directory", "commit": "b" * 40}
                ],
            }
        ],
        "unresolved": [],
        "metadata": {},
    },
}


def _call(name="search_repository", arguments=None):
    return ToolCall(
        id="call_1",
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _controlled_pinned_context(tmp_path: Path, monkeypatch) -> Path:
    """Minimal deterministic pinned-run context for ``search_repository``.

    Builds the run layout the real tool requires in order to resolve the
    pinned snapshot (facts + architecture run markers, snapshot artifact,
    and the pinned ``repository/`` checkout) under *tmp_path*, then points
    the default artifacts root at it. Only the root location is
    redirected — the production resolution path (discover the run →
    repository dir → search) and all snapshot validation are unchanged,
    so the test neither depends on developer-machine artifacts nor
    bypasses the snapshot architecture. Cleanup is automatic (tmp_path).
    """
    root = tmp_path / "artifacts"
    run_dir = root / "run_fixture_alpha"
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "repository").mkdir(parents=True)
    (run_dir / "engineering").mkdir(parents=True)
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "engineering" / "engineering_model.json").write_text(
        json.dumps(_DISPATCH_ENGINEERING_MODEL), encoding="utf-8"
    )
    # the agent-facing graph tool reads only the persisted graph
    # artifact, so the fixture persists the derived projection (M11.1)
    write_engineering_graph_document(
        build_engineering_graph_document(
            _DISPATCH_ENGINEERING_MODEL, snapshot_id="run_fixture_alpha"
        ),
        run_dir / "engineering" / "engineering_graph.json",
    )
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {
                    "name": "sample-repo-alpha",
                    "url": "https://example.invalid/sample.git",
                },
                "snapshot": {"ref": "main", "commit_sha": "b" * 40},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "repository" / "README.md").write_text(
        "# Sample Project\n\nThe onboarding workflow lives here.\n",
        encoding="utf-8",
    )
    # M18: isolated store + deterministic index for search_repository.
    enable_search(run_dir / "repository", monkeypatch, tmp_path)
    monkeypatch.setattr(artifacts, "default_artifacts_root", lambda: root)
    return run_dir


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_dispatcher_registers_exactly_the_four_read_only_tools():
    dispatcher = default_tool_dispatcher()
    assert dispatcher.tool_names == REGISTRY


def test_empty_dispatcher_has_no_registered_tools():
    assert ToolDispatcher().tool_names == ()


def test_register_adds_one_tool_explicitly():
    dispatcher = ToolDispatcher()
    dispatcher.register("search_repository", lambda arguments: None)
    assert dispatcher.tool_names == ("search_repository",)


def test_register_rejects_invalid_tool_names():
    dispatcher = ToolDispatcher()
    with pytest.raises(ValueError):
        dispatcher.register("", lambda arguments: None)
    with pytest.raises(ValueError):
        dispatcher.register(None, lambda arguments: None)


# ---------------------------------------------------------------------------
# Successful execution (registered resolution + arguments)
# ---------------------------------------------------------------------------


def test_registered_tool_is_invoked_with_the_call_arguments():
    seen = []

    def fake_search(arguments):
        seen.append(arguments)
        return ToolResult(success=True, tool="search_repository", data={})

    dispatcher = ToolDispatcher({"search_repository": fake_search})
    arguments = {"query": "onboarding workflow", "max_results": 5}

    result = dispatcher.dispatch(_call(arguments=arguments))

    assert seen == [arguments]
    assert result.success is True
    assert result.tool == "search_repository"


def test_default_dispatcher_executes_get_architecture_context():
    result = default_tool_dispatcher().dispatch(
        _call(name="get_architecture_context", arguments={})
    )

    assert result.success is True
    assert result.tool == "get_architecture_context"
    assert "context" in (result.data or {})


def test_default_dispatcher_executes_search_repository(tmp_path, monkeypatch):
    # Deterministic test context: the pinned run lives under tmp_path,
    # so the real search tool resolves it through the normal production
    # path (default artifacts root → discover run → repository dir).
    _controlled_pinned_context(tmp_path, monkeypatch)

    result = default_tool_dispatcher().dispatch(
        _call(name="search_repository", arguments={"query": "onboarding"})
    )

    assert result.success is True
    assert result.tool == "search_repository"
    assert (result.data or {}).get("query") == "onboarding"


def test_default_dispatcher_executes_query_engineering_graph(tmp_path, monkeypatch):
    # Controlled pinned-run context: facts + architecture + engineering
    # artifacts under tmp_path; the real graph tool resolves them through
    # the normal production path (default artifacts root → discover run).
    _controlled_pinned_context(tmp_path, monkeypatch)

    result = default_tool_dispatcher().dispatch(
        _call(
            name="query_engineering_graph",
            arguments={
                "start_node_id": "artifact:directory:pkg",
                "direction": "out",
                "max_depth": 1,
            },
        )
    )

    assert result.success is True
    assert result.tool == "query_engineering_graph"
    data = result.data or {}
    assert data.get("mode") == "traversal"
    assert [n["id"] for n in data["nodes"]] == [
        "artifact:directory:pkg",
        "artifact:file:pkg/mod.py",
    ]
    assert [r["kind"] for r in data["relationships"]] == ["CONTAINS"]
    assert (result.source or {}).get("snapshot_id") == "run_fixture_alpha"


def test_default_dispatcher_query_engineering_graph_summary(tmp_path, monkeypatch):
    _controlled_pinned_context(tmp_path, monkeypatch)

    result = default_tool_dispatcher().dispatch(
        _call(name="query_engineering_graph", arguments={})
    )

    assert result.success is True
    data = result.data or {}
    assert data.get("mode") == "summary"
    assert data.get("counts", {}).get("nodes") == 2
    assert data.get("counts", {}).get("relationships") == 1
    assert data.get("counts", {}).get("unresolved") == 0


def test_dispatch_executes_the_registered_tool_exactly_once():
    invocations = []

    def counting_tool(arguments):
        invocations.append(arguments)
        return ToolResult(success=True, tool="read_file", data={})

    dispatcher = ToolDispatcher({"read_file": counting_tool})
    dispatcher.dispatch(_call(name="read_file", arguments={"path": "README.md"}))
    dispatcher.dispatch(_call(name="read_file", arguments={"path": "README.md"}))

    assert len(invocations) == 2  # one dispatch → exactly one execution


# ---------------------------------------------------------------------------
# Unknown tool (D) and arbitrary-execution protection (L)
# ---------------------------------------------------------------------------


def test_unknown_tool_fails_safely_without_executing_anything():
    invoked = []

    def spy(arguments):
        invoked.append(arguments)
        return ToolResult(success=True, tool="search_repository", data={})

    dispatcher = ToolDispatcher({"search_repository": spy})
    result = dispatcher.dispatch(_call(name="does_not_exist", arguments={}))

    assert result.success is False
    assert result.tool == "does_not_exist"
    assert result.error is not None
    assert result.error.code == "TOOL_NOT_FOUND"
    assert invoked == []  # no application tool executed


def test_unknown_tool_names_cannot_execute_arbitrary_python():
    invoked = []

    def spy(arguments):
        invoked.append(arguments)
        return ToolResult(success=True, tool="search_repository", data={})

    dispatcher = ToolDispatcher({"search_repository": spy})
    hostile_names = (
        "__import__",
        "eval",
        "exec",
        "os.system",
        "subprocess.run",
        "repo_intelligence.tools.search_repository",
        "builtins.eval",
    )

    for name in hostile_names:
        result = dispatcher.dispatch(_call(name=name, arguments={}))
        assert result.success is False
        assert result.error is not None
        assert result.error.code == "TOOL_NOT_FOUND"
        assert result.tool == name

    assert invoked == []
    assert dispatcher.tool_names == ("search_repository",)  # registry unchanged


# ---------------------------------------------------------------------------
# Invalid arguments (E)
# ---------------------------------------------------------------------------


def test_non_mapping_arguments_become_a_structured_failure():
    invoked = []

    def spy(arguments):
        invoked.append(arguments)
        return ToolResult(success=True, tool="search_repository", data={})

    dispatcher = ToolDispatcher({"search_repository": spy})
    result = dispatcher.dispatch(_call(arguments=["not", "a", "mapping"]))

    assert result.success is False
    assert result.tool == "search_repository"
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENT"
    assert invoked == []


def test_missing_tool_name_becomes_a_structured_failure():
    dispatcher = ToolDispatcher(
        {"search_repository": lambda arguments: ToolResult(success=True, tool="x")}
    )
    result = dispatcher.dispatch(ToolCall(id="call_1", name=""))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENT"


def test_invalid_arguments_are_rejected_by_the_tools_own_contract():
    # The dispatcher does not duplicate tool validation: the registered
    # tool applies its existing input contract and returns its own
    # structured failure.
    result = default_tool_dispatcher().dispatch(
        _call(name="search_repository", arguments={})
    )

    assert result.success is False
    assert result.tool == "search_repository"
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# Tool execution failure (F)
# ---------------------------------------------------------------------------


def test_tool_exception_becomes_a_structured_tool_result():
    def boom(arguments):
        raise RuntimeError("boom")

    dispatcher = ToolDispatcher({"search_repository": boom})
    result = dispatcher.dispatch(_call(arguments={}))

    assert result.success is False
    assert result.tool == "search_repository"
    assert result.error is not None
    assert result.error.code == "TOOL_FAILED"
    assert "boom" in result.error.message
    assert "Traceback" not in result.error.message


def test_non_toolresult_return_becomes_a_structured_failure():
    dispatcher = ToolDispatcher({"search_repository": lambda arguments: "oops"})
    result = dispatcher.dispatch(_call(arguments={}))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "TOOL_FAILED"
