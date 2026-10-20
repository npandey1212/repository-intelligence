"""M12 knowledge-slice graph projection + bounded path query tests.

Levels:
- fixture: synthetic Agent → Tool → MCP Server → MCP Configuration slice
  projected into the engineering graph, walked with the existing bounded
  query API (no new traversal code)
- integration: pinned ADOP graph artifact (skipped when absent)

Offline — no network, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.construction import load_construction_rules
from repo_intelligence.engineering import build_engineering_model
from repo_intelligence.facts import run_analyzers
from repo_intelligence.facts.builder import _attach_evidence
from repo_intelligence.facts.normalize import build_entities
from repo_intelligence.facts.records import build_fact_records
from repo_intelligence.facts.tooling import build_tooling_facts
from repo_intelligence.graph import (
    build_engineering_graph,
    query_engineering_graph,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADOP_GRAPH = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_graph.json"
)

COMMIT = "d" * 40

PATH_KINDS = ["DECLARES", "EXPOSED_BY", "CONFIGURED_BY"]
AGENT = "artifact:agent_definition:.claude/agents/specialist-agent.md"
CONFIG = "artifact:mcp_config:.mcp.json"


@pytest.fixture(scope="module")
def slice_graph():
    """Synthetic slice: facts → model → graph (all offline)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        (root / ".claude" / "agents").mkdir(parents=True)
        (root / ".claude" / "agents" / "analyst-agent.md").write_text(
            "---\n"
            "name: analyst-agent\n"
            "description: Reads files.\n"
            "tools: Read\n"
            "---\n"
            "\n"
            "You are the Analyst Agent.\n",
            encoding="utf-8",
        )
        (root / ".claude" / "agents" / "specialist-agent.md").write_text(
            "---\n"
            "name: specialist-agent\n"
            "description: Talks to the data server.\n"
            "tools:\n"
            "  - mcp__alpha__fetch\n"
            "---\n"
            "\n"
            "You are the Specialist Agent.\n",
            encoding="utf-8",
        )
        (root / "registry.yaml").write_text(
            "servers:\n"
            "  - name: alpha\n"
            "    tools:\n"
            "      - fetch\n"
            "  - name: beta\n"
            "    tools:\n"
            "      - ping\n",
            encoding="utf-8",
        )
        (root / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"alpha": {"command": "x"}}}) + "\n",
            encoding="utf-8",
        )

        results = _attach_evidence(run_analyzers(root), COMMIT)
        entities = build_entities(results.get("python") or {})
        facts = build_fact_records(results, entities=entities, commit=COMMIT)
        facts = facts + build_tooling_facts(
            facts, results.get("tooling") or {}, commit=COMMIT
        )
        model = build_engineering_model(
            {
                "schema_version": "repository_facts.v2",
                "snapshot": {"commit_sha": COMMIT, "snapshot_id": "fx"},
                "repository": {
                    "name": "fixture",
                    "url": "https://example.local/fx",
                },
                "facts": facts,
            },
            load_construction_rules(),
        )
        return build_engineering_graph(model.to_dict())


@pytest.fixture(scope="module")
def adop_graph():
    if not ADOP_GRAPH.is_file():
        pytest.skip("ADOP engineering graph absent")
    return build_engineering_graph(
        json.loads(ADOP_GRAPH.read_text(encoding="utf-8"))
    )


# ---------------------------------------------------------------------------
# A — synthetic slice: projection + bounded path walk
# ---------------------------------------------------------------------------


def test_slice_projects_tool_and_server_nodes(slice_graph):
    kinds = set(slice_graph.nodes)
    assert "tool:mcp:alpha:fetch" in kinds
    assert "tool:builtin:Read" in kinds
    assert "mcp_server:alpha" in kinds
    assert "mcp_server:beta" in kinds
    assert CONFIG in kinds
    assert AGENT in kinds
    assert slice_graph.graph["unresolved_count"] >= 1


def test_slice_path_walk_reaches_configuration(slice_graph):
    result = query_engineering_graph(
        slice_graph,
        start_node_id=AGENT,
        direction="out",
        relationship_kinds=PATH_KINDS,
        max_depth=3,
    )
    node_ids = {n["id"] for n in result.nodes}
    assert "tool:mcp:alpha:fetch" in node_ids
    assert "mcp_server:alpha" in node_ids
    assert CONFIG in node_ids
    rel_kinds = {r["kind"] for r in result.relationships}
    assert rel_kinds == {"DECLARES", "EXPOSED_BY", "CONFIGURED_BY"}
    # unresolved claims contributed no edges
    edge_pairs = {(r["source"], r["target"]) for r in result.relationships}
    assert ("mcp_server:beta", CONFIG) not in edge_pairs


def test_slice_max_depth_bounds_the_path(slice_graph):
    shallow = query_engineering_graph(
        slice_graph,
        start_node_id=AGENT,
        direction="out",
        relationship_kinds=PATH_KINDS,
        max_depth=1,
    )
    shallow_ids = {n["id"] for n in shallow.nodes}
    assert "tool:mcp:alpha:fetch" in shallow_ids
    assert "mcp_server:alpha" not in shallow_ids
    assert CONFIG not in shallow_ids

    mid = query_engineering_graph(
        slice_graph,
        start_node_id=AGENT,
        direction="out",
        relationship_kinds=PATH_KINDS,
        max_depth=2,
    )
    mid_ids = {n["id"] for n in mid.nodes}
    assert "mcp_server:alpha" in mid_ids
    assert CONFIG not in mid_ids


def test_slice_relationship_kind_filter(slice_graph):
    declares_only = query_engineering_graph(
        slice_graph,
        start_node_id=AGENT,
        direction="out",
        relationship_kinds=["DECLARES"],
        max_depth=3,
    )
    rel_kinds = {r["kind"] for r in declares_only.relationships}
    assert rel_kinds == {"DECLARES"}
    node_ids = {n["id"] for n in declares_only.nodes}
    assert "mcp_server:alpha" not in node_ids


def test_slice_summary_lists_new_node_kinds(slice_graph):
    result = query_engineering_graph(slice_graph)
    assert result.mode == "summary"
    kinds = {(k["kind"], k.get("subkind")) for k in result.kinds}
    assert ("tool", "mcp") in kinds
    assert ("tool", "built_in") in kinds
    assert ("mcp_server", "registered") in kinds


def test_slice_query_is_deterministic(slice_graph):
    kwargs = dict(
        start_node_id=AGENT,
        direction="out",
        relationship_kinds=PATH_KINDS,
        max_depth=3,
    )
    first = query_engineering_graph(slice_graph, **kwargs).to_dict()
    second = query_engineering_graph(slice_graph, **kwargs).to_dict()
    assert first == second


# ---------------------------------------------------------------------------
# B — pinned ADOP graph (skipped when artifact absent)
# ---------------------------------------------------------------------------


def test_adop_graph_counts(adop_graph):
    assert adop_graph.number_of_nodes() == 1113
    assert adop_graph.number_of_edges() == 1627
    assert adop_graph.graph["unresolved_count"] == 449


def test_adop_forward_path_is_queryable(adop_graph):
    agent = "artifact:agent_definition:.claude/agents/ontology-agent.md"
    result = query_engineering_graph(
        adop_graph,
        start_node_id=agent,
        direction="out",
        relationship_kinds=PATH_KINDS,
        max_depth=3,
    )
    node_ids = {n["id"] for n in result.nodes}
    assert "tool:mcp:glue-athena:get_table" in node_ids
    assert "mcp_server:glue-athena" in node_ids
    assert "artifact:mcp_config:.mcp.json" in node_ids
    rel_kinds = {r["kind"] for r in result.relationships}
    assert rel_kinds == set(PATH_KINDS)


def test_adop_every_configured_server_edge_points_at_config(adop_graph):
    server_ids = [
        n["id"]
        for n in adop_graph.nodes.values()
        if isinstance(n, dict) and n.get("kind") == "mcp_server"
    ]
    assert len(server_ids) == 13
    for server_id in server_ids:
        result = query_engineering_graph(
            adop_graph,
            start_node_id=server_id,
            direction="out",
            relationship_kinds=["CONFIGURED_BY"],
            max_depth=1,
        )
        for rel in result.relationships:
            assert rel["kind"] == "CONFIGURED_BY"
            assert rel["target"] == "artifact:mcp_config:.mcp.json"
