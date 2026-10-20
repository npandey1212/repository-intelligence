"""M12 tooling analyzer + tool/server/configuration fact tests.

Covers: explicit-statement-only extraction (agent frontmatter, registry
shape discovery, MCP configuration keys), identity/dedup rules (merged
evidence, ambiguous bare declarations, unresolved targets, conflicting
configuration), and the composed-facts integration path.

Offline — no network, no LLM.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from repo_intelligence.facts import (
    FACT_TYPE_ORDER,
    compose_facts,
    make_fact,
    run_analyzers,
    validate_facts_document,
    validate_facts_invariants,
)
from repo_intelligence.facts.builder import _attach_evidence
from repo_intelligence.facts.normalize import build_entities
from repo_intelligence.facts.records import build_fact_records
from repo_intelligence.facts.tooling import (
    analyze_tooling,
    build_tooling_facts,
)
from repo_intelligence.snapshot import create_snapshot

COMMIT = "a" * 40


@pytest.fixture
def tooling_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".claude" / "agents").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "tooling").mkdir(parents=True)

    (root / ".claude" / "agents" / "analyst-agent.md").write_text(
        "---\n"
        "name: analyst-agent\n"
        "description: Reads and edits files.\n"
        "tools: Read, Write, Glob\n"
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
        "  - Grep\n"
        "  - mcp__alpha__fetch\n"
        "---\n"
        "\n"
        "You are the Specialist Agent.\n",
        encoding="utf-8",
    )
    # Bare declaration colliding with a registry tool name: ambiguous.
    (root / ".claude" / "agents" / "collision-agent.md").write_text(
        "---\n"
        "name: collision-agent\n"
        "description: Uses a server-scoped tool by bare name.\n"
        "tools: Fetch\n"
        "---\n"
        "\n"
        "You are the Collision Agent.\n",
        encoding="utf-8",
    )
    # No frontmatter: a `tools:` line here is prose, never a declaration.
    (root / "docs" / "notes.md").write_text(
        "Tools: Read, Write\n",
        encoding="utf-8",
    )
    # Registry discovered by shape (servers: -> - name: + tools:), any path.
    (root / "tooling" / "registry.yaml").write_text(
        "servers:\n"
        "  - name: alpha\n"
        "    type: custom\n"
        "    tools:\n"
        "      - Fetch\n"
        "      - fetch\n"
        "  - name: beta\n"
        "    tools:\n"
        "      - Ping\n",
        encoding="utf-8",
    )
    # Sibling YAML with a `servers:` key that does not match the shape.
    (root / "docs" / "servers.yaml").write_text(
        "servers: alpha\n",
        encoding="utf-8",
    )
    (root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "alpha": {"command": "python", "args": ["-m", "alpha"]}
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _facts_from_repo(root: Path, commit: str = COMMIT) -> List[Dict[str, Any]]:
    results = _attach_evidence(run_analyzers(root), commit)
    entities = build_entities(results.get("python") or {})
    facts = build_fact_records(results, entities=entities, commit=commit)
    return facts + build_tooling_facts(
        facts, results.get("tooling") or {}, commit=commit
    )


# ---------------------------------------------------------------------------
# A — analyzer: explicit statements only
# ---------------------------------------------------------------------------


def test_analyze_tooling_statement_types_and_lines(tooling_repo: Path):
    result = analyze_tooling(tooling_repo)
    statements = result["statements"]
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for statement in statements:
        by_type.setdefault(statement["type"], []).append(statement)

    assert len(by_type["agent_tool"]) == 6
    assert len(by_type["registry_server"]) == 2
    assert len(by_type["server_tool"]) == 3
    assert len(by_type["config_server"]) == 1

    # every statement carries repository-relative path + 1-based line
    for statement in statements:
        assert statement["path"]
        assert statement["line"] >= 1

    namespaced = [
        s
        for s in by_type["agent_tool"]
        if s.get("server") == "alpha" and s.get("name") == "fetch"
    ]
    assert len(namespaced) == 1
    assert namespaced[0]["entry"] == "mcp__alpha__fetch"

    config = by_type["config_server"][0]
    assert config["path"] == ".mcp.json"
    assert config["server"] == "alpha"


def test_analyze_tooling_requires_frontmatter(tooling_repo: Path):
    result = analyze_tooling(tooling_repo)
    paths = {s["path"] for s in result["statements"]}
    assert "docs/notes.md" not in paths


def test_analyze_tooling_skips_non_matching_registry_shape(tooling_repo: Path):
    result = analyze_tooling(tooling_repo)
    paths = {s["path"] for s in result["statements"]}
    # `servers: alpha` (scalar) does not match servers -> - name: shape
    assert "docs/servers.yaml" not in paths


def test_analyze_tooling_deterministic(tooling_repo: Path):
    first = analyze_tooling(tooling_repo)["statements"]
    second = analyze_tooling(tooling_repo)["statements"]
    assert first == second
    assert first == sorted(
        first,
        key=lambda s: (
            s["type"],
            s["path"],
            int(s["line"]),
            str(s.get("server") or ""),
            str(s.get("name") or ""),
        ),
    )


# ---------------------------------------------------------------------------
# B — fact identity + composed facts
# ---------------------------------------------------------------------------


def test_compute_fact_id_tooling_identities():
    builtin = make_fact(
        fact_type="tool",
        kind="built_in_tool",
        source="tooling",
        attributes={"name": "Read"},
        evidence=[],
    )
    assert builtin["fact_id"] == "fact:tool:builtin:Read"

    mcp_tool = make_fact(
        fact_type="tool",
        kind="mcp_tool",
        source="tooling",
        attributes={"name": "fetch", "server": "alpha"},
        evidence=[],
    )
    assert mcp_tool["fact_id"] == "fact:tool:mcp:alpha:fetch"

    server = make_fact(
        fact_type="mcp_server",
        kind="registered",
        source="tooling",
        attributes={"name": "alpha"},
        evidence=[],
    )
    assert server["fact_id"] == "fact:mcp_server:server:alpha"

    declares = make_fact(
        fact_type="interaction",
        kind="declares",
        source="tooling",
        attributes={
            "source_fact_id": "fact:artifact:agents:agent_definition:a.md",
            "target_fact_id": builtin["fact_id"],
        },
        evidence=[],
    )
    assert declares["fact_id"] == (
        "fact:interaction:declares:"
        "fact:artifact:agents:agent_definition:a.md:"
        "fact:tool:builtin:Read"
    )

    # triggers keeps the same explicit-endpoint slot (M7 contract)
    triggers = make_fact(
        fact_type="interaction",
        kind="triggers",
        source="interactions",
        attributes={"source_fact_id": "s", "target_fact_id": "t"},
        evidence=[],
    )
    assert triggers["fact_id"] == "fact:interaction:triggers:s:t"


def test_fact_type_order_includes_tooling_types():
    assert FACT_TYPE_ORDER.index("tool") > FACT_TYPE_ORDER.index("interaction")
    assert FACT_TYPE_ORDER.index("mcp_server") > FACT_TYPE_ORDER.index(
        "interaction"
    )


def test_build_tooling_facts_node_identities(tooling_repo: Path):
    facts = _facts_from_repo(tooling_repo)
    by_id = {f["fact_id"]: f for f in facts}

    assert "fact:tool:mcp:alpha:fetch" in by_id
    assert "fact:tool:mcp:alpha:Fetch" in by_id
    assert "fact:tool:mcp:beta:Ping" in by_id
    for name in ("Read", "Write", "Glob", "Grep"):
        assert "fact:tool:builtin:{}".format(name) in by_id
    # ambiguous bare declaration never becomes a builtin node fact
    assert "fact:tool:builtin:Fetch" not in by_id
    assert "fact:mcp_server:server:alpha" in by_id
    assert "fact:mcp_server:server:beta" in by_id

    # merged evidence: same builtin declared by several agents
    read = by_id["fact:tool:builtin:Read"]
    assert read["fact_type"] == "tool" and read["kind"] == "built_in_tool"
    assert len(read["evidence"]) == 1  # only analyst declares Read

    # merged evidence: namespaced agent declaration + registry entry
    fetch = by_id["fact:tool:mcp:alpha:fetch"]
    assert fetch["kind"] == "mcp_tool"
    assert fetch["attributes"]["server"] == "alpha"
    assert {e["path"] for e in fetch["evidence"]} == {
        ".claude/agents/specialist-agent.md",
        "tooling/registry.yaml",
    }


def test_build_tooling_facts_relationships(tooling_repo: Path):
    facts = _facts_from_repo(tooling_repo)
    tooling = [f for f in facts if f.get("source") == "tooling"]
    assert {f["fact_type"] for f in tooling} == {
        "tool",
        "mcp_server",
        "interaction",
    }

    declares = [
        f for f in tooling if f["fact_type"] == "interaction" and f["kind"] == "declares"
    ]
    exposes = [
        f for f in tooling if f["fact_type"] == "interaction" and f["kind"] == "exposes"
    ]
    configures = [
        f
        for f in tooling
        if f["fact_type"] == "interaction" and f["kind"] == "configures"
    ]

    # analyst(3) + specialist(Grep + namespaced) + collision(ambiguous)
    assert len(declares) == 6
    assert len(exposes) == 3
    assert len(configures) == 2  # alpha resolved + beta unresolved

    ambiguous = [
        f for f in declares if (f["attributes"] or {}).get("reason")
    ]
    assert len(ambiguous) == 1
    assert ambiguous[0]["attributes"]["reason"] == "ambiguous_identity"
    assert ambiguous[0]["attributes"]["target_label"] == "Fetch"
    assert "target_fact_id" not in ambiguous[0]["attributes"]

    unresolved_config = [
        f for f in configures if (f["attributes"] or {}).get("reason")
    ]
    assert len(unresolved_config) == 1
    assert unresolved_config[0]["attributes"]["reason"] == "target_not_resolved"
    assert (
        unresolved_config[0]["attributes"]["source_fact_id"]
        == "fact:mcp_server:server:beta"
    )
    assert unresolved_config[0]["attributes"]["target_label"] == ".mcp.json"

    resolved = [f for f in configures if not (f["attributes"] or {}).get("reason")]
    assert len(resolved) == 1
    assert resolved[0]["attributes"]["target_fact_id"].startswith(
        "fact:artifact:mcp:mcp_config:"
    )
    assert not any(
        (f["attributes"] or {}).get("conflicting") for f in configures
    )


def test_build_tooling_facts_deterministic(tooling_repo: Path):
    first = _facts_from_repo(tooling_repo)
    second = _facts_from_repo(tooling_repo)
    assert [f["fact_id"] for f in first] == [f["fact_id"] for f in second]


# ---------------------------------------------------------------------------
# C — conflicting configuration (two config files, one server)
# ---------------------------------------------------------------------------


def test_two_config_files_mark_conflicting(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    registry = "servers:\n  - name: alpha\n    tools:\n      - fetch\n"
    (root / "registry.yaml").write_text(registry, encoding="utf-8")
    body = json.dumps(
        {"mcpServers": {"alpha": {"command": "python"}}}, indent=2
    ) + "\n"
    (root / ".mcp.json").write_text(body, encoding="utf-8")
    (root / "mcp.json").write_text(body, encoding="utf-8")

    facts = _facts_from_repo(root)
    configures = [
        f
        for f in facts
        if f["fact_type"] == "interaction" and f["kind"] == "configures"
    ]
    resolved = [f for f in configures if f["attributes"].get("target_fact_id")]
    assert len(resolved) == 2
    targets = {f["attributes"]["target_fact_id"] for f in resolved}
    assert len(targets) == 2  # two different configuration artifacts
    assert all(f["attributes"].get("conflicting") is True for f in resolved)


# ---------------------------------------------------------------------------
# D — composed facts integration (M1/M2 envelope, schema validation)
# ---------------------------------------------------------------------------


def test_compose_facts_includes_tooling(tooling_repo: Path):
    subprocess.run(["git", "init"], cwd=tooling_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.local"],
        cwd=tooling_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tooling_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=tooling_repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tooling_repo,
        check=True,
        capture_output=True,
    )

    snapshot, _ = create_snapshot(
        repository_url=str(tooling_repo),
        ref="HEAD",
        output_root=tooling_repo.parent / "out",
    )
    results = run_analyzers(tooling_repo)
    facts = compose_facts(snapshot, results, "fixture_tooling")

    validate_facts_invariants(facts)
    validate_facts_document(facts.to_dict())

    counts = facts.counts()
    assert counts["tools"] == len(facts.facts_of_type("tool"))
    assert counts["mcp_servers"] == len(facts.facts_of_type("mcp_server"))
    assert counts["tools"] == 7  # 3 mcp + 4 builtin (Fetch ambiguous)
    assert counts["mcp_servers"] == 2
    assert counts["interactions"] == len(facts.facts_of_type("interaction"))
    # tooling relationship facts only (no trigger statements in fixture)
    assert counts["interactions"] == 11
