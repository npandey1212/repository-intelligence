"""M2 explicit interaction fact tests (candidate TRIGGERS observations).

Covers: explicit-only detection, endpoint resolution to existing fact_ids,
and no-fabrication rules (mentions / unresolved peers produce nothing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from repo_intelligence.facts import (
    compose_facts,
    compute_fact_id,
    make_fact,
    run_analyzers,
    validate_facts_document,
    validate_facts_invariants,
)
from repo_intelligence.facts.interactions import (
    analyze_interactions,
    build_endpoint_registry,
    build_interaction_facts,
)
from repo_intelligence.facts.records import build_fact_records
from repo_intelligence.facts.normalize import build_entities
from repo_intelligence.snapshot import create_snapshot
from repo_intelligence.facts.builder import _attach_evidence

COMMIT = "a" * 40


@pytest.fixture
def interaction_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".claude" / "agents").mkdir(parents=True)
    (root / ".claude" / "commands").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)

    (root / ".claude" / "agents" / "worker-agent.md").write_text(
        "---\n"
        "name: worker-agent\n"
        "description: Does work. Spawned by the Controller Agent in phase 2.\n"
        "---\n"
        "\n"
        "You are the Worker Agent.\n",
        encoding="utf-8",
    )
    (root / ".claude" / "agents" / "idle-agent.md").write_text(
        "---\n"
        "name: idle-agent\n"
        "description: Never spawned. Mentions the Controller Agent only.\n"
        "---\n"
        "\n"
        "You are the Idle Agent.\n"
        "Related reading: Controller Agent notes in docs.\n",
        encoding="utf-8",
    )
    (root / ".claude" / "commands" / "controller.md").write_text(
        "---\n"
        "description: Orchestrates workers\n"
        "---\n"
        "\n"
        "You are the Controller Agent orchestrating work.\n"
        "\n"
        "Spawn Worker Agent to validate input.\n",
        encoding="utf-8",
    )
    (root / "docs" / "notes.md").write_text(
        "The Controller Agent exists.\n"
        "Nothing explicit here.\n",
        encoding="utf-8",
    )
    return root


def _facts_from_repo(root: Path, commit: str = COMMIT) -> List[Dict[str, Any]]:
    results = _attach_evidence(run_analyzers(root), commit)
    entities = build_entities(results.get("python") or {})
    facts = build_fact_records(results, entities=entities, commit=commit)
    return facts + build_interaction_facts(
        facts, results.get("interactions") or {}, commit=commit
    )


# ---------------------------------------------------------------------------
# A — M2 fixture: explicit statement → fact; mere mention → no fact
# ---------------------------------------------------------------------------


def test_explicit_spawned_by_produces_interaction_fact(interaction_repo: Path):
    facts = _facts_from_repo(interaction_repo)
    interactions = [f for f in facts if f.get("fact_type") == "interaction"]
    assert interactions, "expected at least one interaction fact"

    pairs = {
        (
            f["attributes"]["source_path"],
            f["attributes"]["target_path"],
        )
        for f in interactions
    }
    assert (
        ".claude/commands/controller.md",
        ".claude/agents/worker-agent.md",
    ) in pairs

    fact = next(
        f
        for f in interactions
        if f["attributes"]["source_path"].endswith("controller.md")
        and f["attributes"]["target_path"].endswith("worker-agent.md")
    )
    assert fact["kind"] == "triggers"
    assert fact["source"] == "interactions"
    assert fact["fact_id"].startswith("fact:interaction:triggers:")
    assert fact["attributes"]["source_fact_id"].startswith("fact:")
    assert fact["attributes"]["target_fact_id"].startswith("fact:")
    assert fact["evidence"]
    evidence = fact["evidence"][0]
    assert evidence["commit"] == COMMIT
    assert evidence["path"] in {
        ".claude/agents/worker-agent.md",
        ".claude/commands/controller.md",
    }
    assert evidence["line_start"] >= 1


def test_mere_mention_produces_no_interaction(interaction_repo: Path):
    facts = _facts_from_repo(interaction_repo)
    interactions = [f for f in facts if f.get("fact_type") == "interaction"]
    for fact in interactions:
        # idle-agent only mentions Controller Agent — must not be a peer edge
        assert not fact["attributes"]["target_path"].endswith("idle-agent.md")
        assert not (
            fact["attributes"]["source_path"].endswith("idle-agent.md")
        )
    # docs/notes.md mention-only must not appear
    for fact in interactions:
        assert fact["attributes"]["source_path"] != "docs/notes.md"
        assert fact["attributes"]["target_path"] != "docs/notes.md"


def test_explicit_spawn_target_in_source_file(interaction_repo: Path):
    facts = _facts_from_repo(interaction_repo)
    pairs = {
        (f["attributes"]["source_path"], f["attributes"]["target_path"])
        for f in facts
        if f.get("fact_type") == "interaction"
    }
    # controller.md contains "Spawn Worker Agent"
    assert (
        ".claude/commands/controller.md",
        ".claude/agents/worker-agent.md",
    ) in pairs


def test_endpoint_fact_ids_exist(interaction_repo: Path):
    facts = _facts_from_repo(interaction_repo)
    ids = {f["fact_id"] for f in facts}
    for fact in facts:
        if fact.get("fact_type") != "interaction":
            continue
        assert fact["attributes"]["source_fact_id"] in ids
        assert fact["attributes"]["target_fact_id"] in ids


def test_interaction_fact_id_deterministic():
    attrs = {
        "source_fact_id": "fact:artifact:agents:agent_command:a.md",
        "target_fact_id": "fact:artifact:agents:agent_definition:b.md",
    }
    a = compute_fact_id("interaction", "triggers", "interactions", attrs)
    b = compute_fact_id("interaction", "triggers", "interactions", dict(attrs))
    assert a == b
    assert a == (
        "fact:interaction:triggers:"
        "fact:artifact:agents:agent_command:a.md:"
        "fact:artifact:agents:agent_definition:b.md"
    )


def test_build_interaction_facts_skips_unresolved_peer():
    facts = [
        make_fact(
            fact_type="artifact",
            kind="agent_definition",
            source="agents",
            attributes={"path": ".claude/agents/w.md"},
            evidence=[
                {
                    "path": ".claude/agents/w.md",
                    "artifact_type": "agent_definition",
                    "commit": COMMIT,
                }
            ],
        )
    ]
    analysis = {
        "statements": [
            {
                "path": ".claude/agents/w.md",
                "line_start": 3,
                "line_end": 3,
                "direction": "spawned_by",
                "peer_label": "Unknown Phantom Agent",
            }
        ],
        "aliases": [],
    }
    assert build_interaction_facts(facts, analysis, COMMIT) == []


def test_analyze_interactions_alias_and_statements(interaction_repo: Path):
    result = analyze_interactions(interaction_repo)
    assert result["statements"]
    assert any(
        s["direction"] == "spawned_by"
        and s["path"].endswith("worker-agent.md")
        for s in result["statements"]
    )
    assert any(
        a["alias"] == "Controller Agent"
        and a["path"].endswith("controller.md")
        for a in result["aliases"]
    )


def test_compose_facts_includes_interactions(interaction_repo: Path):
    import json
    import subprocess

    # Initialize a tiny git repo so M1 snapshot can pin a commit.
    subprocess.run(["git", "init"], cwd=interaction_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.local"],
        cwd=interaction_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=interaction_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=interaction_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=interaction_repo,
        check=True,
        capture_output=True,
    )

    snapshot, _ = create_snapshot(
        repository_url=str(interaction_repo),
        ref="HEAD",
        output_root=interaction_repo.parent / "out",
    )
    results = run_analyzers(interaction_repo)
    facts = compose_facts(snapshot, results, "fixture_interactions")
    interactions = facts.facts_of_type("interaction")
    assert interactions
    validate_facts_invariants(facts)
    validate_facts_document(facts.to_dict())
    counts = facts.counts()
    assert counts["interactions"] == len(interactions)


def test_registry_first_wins_agent_definition_alias(interaction_repo: Path):
    facts = _facts_from_repo(interaction_repo)
    analysis = analyze_interactions(interaction_repo)
    registry = build_endpoint_registry(facts, analysis["aliases"])
    # Controller Agent resolves uniquely to the command fact
    from repo_intelligence.facts.interactions import _normalize_alias

    key = _normalize_alias("Controller Agent")
    assert key in registry
    assert len(registry[key]) == 1
    assert registry[key][0].startswith("fact:")
