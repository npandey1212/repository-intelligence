"""Level 1 — tool correctness for the Architecture Investigator (§39).

Does the tool return correct repository facts about the fixtures the
specialist will investigate?  Fast, tiny, deterministic: these checks
isolate tool failures from prompt/agent/orchestration failures.
"""

from __future__ import annotations

from pathlib import Path

from architecture_doubles import (
    FIXTURE_ROOT,
    MCP_FIXTURE_ROOT,
    MCP_CONFIG_PATH,
    RESEARCHER_PATH,
)

from conftest import enable_search
from repo_intelligence.agents import ARCHITECTURE_TOOLS
from repo_intelligence.multiagent.permissions import INVESTIGATOR_TOOLS
from repo_intelligence.tool_dispatcher import tool_registry
from repo_intelligence.tools.read_file import read_file
from repo_intelligence.tools.search_repository import search_repository


def test_search_repository_finds_agent_definitions(tmp_path, monkeypatch):
    enable_search(FIXTURE_ROOT, monkeypatch, tmp_path)
    result = search_repository(
        {"query": "Researcher"}, repository_root=FIXTURE_ROOT
    )

    assert result.success is True
    paths = [hit["path"] for hit in result.data["results"]]
    assert RESEARCHER_PATH in paths


def test_read_file_returns_agent_source():
    result = read_file(
        {"path": RESEARCHER_PATH}, repository_root=FIXTURE_ROOT
    )

    assert result.success is True
    content = result.data["content"]
    assert 'TOOLS = ("search_notes", "fetch_note")' in content
    assert result.data["path"] == RESEARCHER_PATH


def test_search_repository_finds_mcp_configuration(tmp_path, monkeypatch):
    enable_search(MCP_FIXTURE_ROOT, monkeypatch, tmp_path)
    result = search_repository(
        {"query": "mcpServers"}, repository_root=MCP_FIXTURE_ROOT
    )

    assert result.success is True
    paths = [hit["path"] for hit in result.data["results"]]
    assert MCP_CONFIG_PATH in paths


def test_read_file_rejects_paths_outside_the_fixture():
    result = read_file(
        {"path": "../outside.txt"}, repository_root=FIXTURE_ROOT
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "PATH_OUTSIDE_REPOSITORY"


def test_search_is_deterministic(tmp_path, monkeypatch):
    enable_search(FIXTURE_ROOT, monkeypatch, tmp_path)
    first = search_repository(
        {"query": "tools"}, repository_root=FIXTURE_ROOT
    )
    second = search_repository(
        {"query": "tools"}, repository_root=FIXTURE_ROOT
    )

    assert first.data["results"] == second.data["results"]


def test_architecture_tools_are_the_existing_read_only_inventory():
    assert ARCHITECTURE_TOOLS == tuple(sorted(tool_registry()))
    assert ARCHITECTURE_TOOLS == INVESTIGATOR_TOOLS


def test_fixtures_are_tiny_by_design():
    roots = [FIXTURE_ROOT, MCP_FIXTURE_ROOT]
    for root in roots:
        assert isinstance(root, Path)
        files = [p for p in root.rglob("*") if p.is_file()]
        assert 3 <= len(files) <= 6, root
