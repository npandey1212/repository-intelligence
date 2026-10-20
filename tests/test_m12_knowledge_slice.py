"""M12 Agent → Tool → MCP Server → MCP Configuration knowledge slice tests.

Levels:
- fixture: synthetic explicit statements → M2 facts → EngineeringModel
  (nodes, forward relationships, unresolved, conflicting, counts)
- integration: pinned ADOP artifacts (skipped when absent)

Offline — no network, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.construction import load_construction_rules
from repo_intelligence.engineering import (
    RELATIONSHIP_KINDS,
    build_engineering_model,
)
from repo_intelligence.facts.builder import _attach_evidence
from repo_intelligence.facts.normalize import build_entities
from repo_intelligence.facts.records import build_fact_records
from repo_intelligence.facts.tooling import build_tooling_facts
from repo_intelligence.facts import run_analyzers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADOP_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_model.json"
)

COMMIT = "c" * 40

PATH_KINDS = {"DECLARES", "EXPOSED_BY", "CONFIGURED_BY"}


@pytest.fixture(scope="module")
def rules() -> Dict[str, Any]:
    return load_construction_rules()


@pytest.fixture
def tooling_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".claude" / "agents").mkdir(parents=True)
    (root / ".claude" / "agents" / "analyst-agent.md").write_text(
        "---\n"
        "name: analyst-agent\n"
        "description: Reads files.\n"
        "tools: Read, Write\n"
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
        json.dumps(
            {"mcpServers": {"alpha": {"command": "python"}}}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _facts_doc(root: Path) -> Dict[str, Any]:
    results = _attach_evidence(run_analyzers(root), COMMIT)
    entities = build_entities(results.get("python") or {})
    facts = build_fact_records(results, entities=entities, commit=COMMIT)
    facts = facts + build_tooling_facts(
        facts, results.get("tooling") or {}, commit=COMMIT
    )
    return {
        "schema_version": "repository_facts.v2",
        "snapshot": {"commit_sha": COMMIT, "snapshot_id": "fixture_m12"},
        "repository": {"name": "fixture", "url": "https://example.local/fx"},
        "facts": facts,
    }


def _by_id(model) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in model.engineering["nodes"]}


def _rel_kinds(model) -> List[str]:
    return [r["kind"] for r in model.engineering["relationships"]]


# ---------------------------------------------------------------------------
# A — forward path nodes + relationships from explicit statements
# ---------------------------------------------------------------------------


def test_slice_builds_tool_server_config_nodes(tooling_repo: Path, rules):
    model = build_engineering_model(_facts_doc(tooling_repo), rules)
    nodes = _by_id(model)

    assert nodes["tool:mcp:alpha:fetch"]["kind"] == "tool"
    assert nodes["tool:mcp:alpha:fetch"]["subkind"] == "mcp"
    assert nodes["tool:mcp:alpha:fetch"]["attributes"]["server"] == "alpha"
    assert nodes["tool:builtin:Read"]["kind"] == "tool"
    assert nodes["tool:builtin:Read"]["subkind"] == "built_in"
    assert "server" not in nodes["tool:builtin:Read"]["attributes"]
    assert nodes["mcp_server:alpha"]["kind"] == "mcp_server"
    assert nodes["mcp_server:alpha"]["subkind"] == "registered"
    assert nodes["mcp_server:beta"]["kind"] == "mcp_server"
    assert (
        "artifact:agent_definition:.claude/agents/analyst-agent.md" in nodes
    )
    assert "artifact:mcp_config:.mcp.json" in nodes
    for node in nodes.values():
        assert node["classification"] in ("observed", "derived", "conflicting")
        assert node["evidence"]


def test_slice_forward_relationships(tooling_repo: Path, rules):
    model = build_engineering_model(_facts_doc(tooling_repo), rules)
    rels = {
        (r["kind"], r["source"], r["target"])
        for r in model.engineering["relationships"]
    }

    assert (
        "DECLARES",
        "artifact:agent_definition:.claude/agents/analyst-agent.md",
        "tool:builtin:Read",
    ) in rels
    assert (
        "DECLARES",
        "artifact:agent_definition:.claude/agents/specialist-agent.md",
        "tool:mcp:alpha:fetch",
    ) in rels
    assert (
        "EXPOSED_BY",
        "tool:mcp:alpha:fetch",
        "mcp_server:alpha",
    ) in rels
    assert (
        "EXPOSED_BY",
        "tool:mcp:beta:ping",
        "mcp_server:beta",
    ) in rels
    assert (
        "CONFIGURED_BY",
        "mcp_server:alpha",
        "artifact:mcp_config:.mcp.json",
    ) in rels

    for rel in model.engineering["relationships"]:
        if rel["kind"] in PATH_KINDS:
            assert rel["classification"] == "observed"
            assert rel["evidence"]


def test_slice_full_forward_path_reaches_configuration(tooling_repo: Path, rules):
    """Agent --DECLARES--> Tool --EXPOSED_BY--> Server --CONFIGURED_BY--> Config."""
    model = build_engineering_model(_facts_doc(tooling_repo), rules)
    out: Dict[str, List[str]] = {}
    for rel in model.engineering["relationships"]:
        out.setdefault(rel["source"], []).append(
            "{}|{}".format(rel["kind"], rel["target"])
        )

    agent = "artifact:agent_definition:.claude/agents/specialist-agent.md"
    declares = out.get(agent) or []
    tool_edges = [e.split("|", 1)[1] for e in declares if e.startswith("DECLARES|")]
    assert "tool:mcp:alpha:fetch" in tool_edges

    server_edges = [
        e.split("|", 1)[1]
        for e in out.get("tool:mcp:alpha:fetch") or []
        if e.startswith("EXPOSED_BY|")
    ]
    assert server_edges == ["mcp_server:alpha"]

    config_edges = [
        e.split("|", 1)[1]
        for e in out.get("mcp_server:alpha") or []
        if e.startswith("CONFIGURED_BY|")
    ]
    assert config_edges == ["artifact:mcp_config:.mcp.json"]


def test_slice_unresolved_claims_stay_unresolved(tooling_repo: Path, rules):
    model = build_engineering_model(_facts_doc(tooling_repo), rules)
    claims = model.engineering["unresolved"]
    path_claims = [c for c in claims if c["kind"] in PATH_KINDS]
    assert path_claims
    for claim in path_claims:
        assert claim["kind"] in RELATIONSHIP_KINDS
        assert claim["reason"] in (
            "target_not_resolved",
            "ambiguous_identity",
            "missing_endpoint",
            "unsupported_relationship",
        )
        assert claim["evidence"]

    # registered server with no configuration entry stays a claim, not an edge
    beta_claims = [
        c
        for c in path_claims
        if c["kind"] == "CONFIGURED_BY" and c["source"] == "mcp_server:beta"
    ]
    assert len(beta_claims) == 1
    assert beta_claims[0]["reason"] == "target_not_resolved"
    rels = {
        (r["kind"], r["source"]) for r in model.engineering["relationships"]
    }
    assert ("CONFIGURED_BY", "mcp_server:beta") not in rels

    # the unresolved CONFIGURED_BY claim is never a canonical edge
    edge_ids = {r["id"] for r in model.engineering["relationships"]}
    for claim in path_claims:
        derived = "rel:{}:{}:{}".format(
            claim["kind"], claim["source"], claim["target"]
        )
        assert derived not in edge_ids


def test_slice_counts_expose_tooling_keys(tooling_repo: Path, rules):
    model = build_engineering_model(_facts_doc(tooling_repo), rules)
    counts = model.counts()
    assert counts["nodes_tool"] == 4  # alpha:fetch + builtin Read/Write
    assert counts["nodes_mcp_server"] == 2
    assert counts["rel_declares"] == 3  # Read, Write, mcp fetch
    assert counts["rel_exposed_by"] == 2  # alpha:fetch, beta:ping
    assert counts["rel_configured_by"] == 1  # alpha only (beta unresolved)
    assert counts["relationships"] == (
        counts["rel_contains"]
        + counts["rel_imports"]
        + counts["rel_depends_on"]
        + counts["rel_triggers"]
        + counts["rel_declares"]
        + counts["rel_exposed_by"]
        + counts["rel_configured_by"]
    )


def test_slice_conflicting_configuration_keeps_both_edges(tmp_path: Path, rules):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "registry.yaml").write_text(
        "servers:\n  - name: alpha\n    tools:\n      - fetch\n",
        encoding="utf-8",
    )
    body = json.dumps({"mcpServers": {"alpha": {"command": "x"}}}) + "\n"
    (root / ".mcp.json").write_text(body, encoding="utf-8")
    (root / "mcp.json").write_text(body, encoding="utf-8")

    model = build_engineering_model(_facts_doc(root), rules)
    config_edges = [
        r
        for r in model.engineering["relationships"]
        if r["kind"] == "CONFIGURED_BY"
    ]
    assert len(config_edges) == 2
    assert {r["target"] for r in config_edges} == {
        "artifact:mcp_config:.mcp.json",
        "artifact:mcp_config:mcp.json",
    }
    assert all(r["classification"] == "conflicting" for r in config_edges)
    # conflicting edges are still canonical (retained), not dropped
    assert model.counts()["rel_configured_by"] == 2


# ---------------------------------------------------------------------------
# B — pinned ADOP integration (skipped when artifacts absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ADOP_MODEL.is_file(), reason="ADOP model absent")
def test_adop_slice_counts():
    document = json.loads(ADOP_MODEL.read_text(encoding="utf-8"))
    counts = document["metadata"]["counts"]
    assert counts["nodes_tool"] == 72  # 66 MCP + 6 built-in
    assert counts["nodes_mcp_server"] == 13
    assert counts["rel_declares"] == 37
    assert counts["rel_exposed_by"] == 66
    assert counts["rel_configured_by"] == 13
    assert counts["unresolved"] == 449  # no new unresolved claims


@pytest.mark.skipif(not ADOP_MODEL.is_file(), reason="ADOP model absent")
def test_adop_slice_forward_path_edges():
    document = json.loads(ADOP_MODEL.read_text(encoding="utf-8"))
    rels = {
        (r["kind"], r["source"], r["target"])
        for r in document["engineering"]["relationships"]
    }
    agent = "artifact:agent_definition:.claude/agents/ontology-agent.md"
    assert (
        "DECLARES",
        agent,
        "tool:mcp:glue-athena:get_table",
    ) in rels
    assert (
        "EXPOSED_BY",
        "tool:mcp:glue-athena:get_table",
        "mcp_server:glue-athena",
    ) in rels
    assert (
        "CONFIGURED_BY",
        "mcp_server:glue-athena",
        "artifact:mcp_config:.mcp.json",
    ) in rels
