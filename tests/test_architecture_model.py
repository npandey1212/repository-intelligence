"""M3.5 ArchitectureModel construction tests.

Levels:
- unit: identity, mapping, evidence, unsupported artifacts, validation
- fixture: synthetic EngineeringModel → architecture section
- integration: pinned ADOP EngineeringModel (skipped when absent)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from repo_intelligence.architecture import (
    ArchitectureModelError,
    ArchitectureModelValidationError,
    ArchitectureRulesSchemaError,
    agent_name_from_engineering_node,
    architecture_agent_node_id,
    architecture_relationship_identity,
    build_architecture_model,
    create_architecture_model,
    load_architecture_rules,
    mapping_identity,
    validate_architecture_rules_document,
    validate_knowledge_model_document,
    validate_knowledge_model_or_raise,
)
from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID, default_artifacts_root
from repo_intelligence.engineering import (
    build_engineering_model,
    load_facts_document,
)
from repo_intelligence.construction import load_construction_rules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FACTS = (
    PROJECT_ROOT / "tests" / "fixtures" / "construction" / "repository_facts.json"
)
ADOP_ENG_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_model.json"
)
ADOP_ARCH_RULES = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "architecture"
    / "architecture_construction_rules.v1.json"
)
ADOP_COMMIT = "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
FIXTURE_COMMIT = "a" * 40
DOA_ID = "concept:agent:data-onboarding-agent"
ORCHESTRATION_SOURCE_IDS = [
    "artifact:agent_command:.claude/commands/onboard-workflow.md",
    "artifact:file:SKILLS.md",
    "artifact:file:docs/workflow-diagrams.md",
    "artifact:file:runbooks/data-onboarding-agent/README.md",
]


@pytest.fixture(scope="module")
def arch_rules() -> Dict[str, Any]:
    return load_architecture_rules()


@pytest.fixture(scope="module")
def fixture_engineering_document() -> Dict[str, Any]:
    facts = load_facts_document(FIXTURE_FACTS)
    eng_rules = load_construction_rules()
    eng = build_engineering_model(facts, eng_rules)
    return eng.to_dict()


@pytest.fixture(scope="module")
def fixture_architecture_model(fixture_engineering_document, arch_rules):
    return build_architecture_model(fixture_engineering_document, arch_rules)


def _engineer_model_with_extra_agents(
    base_doc: Dict[str, Any], extra_paths
) -> Dict[str, Any]:
    """Clone engineering document and append synthetic agent_definition nodes."""
    doc = json.loads(json.dumps(base_doc))
    nodes = doc["engineering"]["nodes"]
    existing = {n["id"] for n in nodes}
    for path in extra_paths:
        node_id = "artifact:agent_definition:{}".format(path)
        if node_id in existing:
            continue
        nodes.append(
            {
                "id": node_id,
                "kind": "artifact",
                "subkind": "agent_definition",
                "classification": "observed",
                "attributes": {"path": path, "fact_type": "artifact", "fact_kind": "agent_definition"},
                "evidence": [
                    {
                        "path": path,
                        "artifact_type": "agent_definition",
                        "commit": base_doc["repository"]["commit_sha"],
                    }
                ],
            }
        )
    nodes.sort(key=lambda n: n["id"])
    return doc


# ---------------------------------------------------------------------------
# Rulebook
# ---------------------------------------------------------------------------


def test_architecture_rules_schema_and_invariants(arch_rules):
    assert arch_rules["schema_version"] == "architecture_construction_rules.v1"
    problems = validate_architecture_rules_document(arch_rules)
    assert problems == []
    node_ids = {r["rule_id"] for r in arch_rules["node_rules"]}
    map_ids = {r["rule_id"] for r in arch_rules["mapping_rules"]}
    rel_ids = {r["rule_id"] for r in arch_rules["relationship_rules"]}
    assert "AR-N001" in node_ids
    assert "AR-M001" in map_ids
    assert "AR-N002" in node_ids
    assert "AR-M002" in map_ids
    assert "AR-R001" in rel_ids
    assert arch_rules["relationship_rules"], "AR-R001 required"
    assert arch_rules["relationship_vocabulary"] == ["TRIGGERS"]
    assert arch_rules["mapping_vocabulary"] == ["CONTRIBUTES_TO"]
    r001 = next(r for r in arch_rules["relationship_rules"] if r["rule_id"] == "AR-R001")
    assert r001["relationship"]["kind"] == "TRIGGERS"
    assert r001["relationship"]["source_engineering_relationship"] == "TRIGGERS"
    assert r001["classification"] == "derived"
    assert r001["requires_evidence"] is True

    n002 = next(r for r in arch_rules["node_rules"] if r["rule_id"] == "AR-N002")
    assert n002["identity"]["template"] == "concept:agent:data-onboarding-agent"
    assert n002["attributes"]["name"] == "Data Onboarding Agent"
    ids = n002["match"]["engineering_node_ids"]
    assert "artifact:agent_command:.claude/commands/onboard-workflow.md" in ids
    assert "artifact:file:SKILLS.md" in ids
    assert "artifact:file:docs/workflow-diagrams.md" in ids
    assert (
        "artifact:file:runbooks/data-onboarding-agent/README.md" in ids
    )
    m002 = next(r for r in arch_rules["mapping_rules"] if r["rule_id"] == "AR-M002")
    assert m002["target"]["from_node_rule"] == "AR-N002"
    assert m002["match"]["engineering_node_ids"] == ids


def test_architecture_rules_rejects_bad_document(arch_rules):
    bad = json.loads(json.dumps(arch_rules))
    bad["schema_version"] = "nope"
    bad["relationship_rules"] = []
    problems = validate_architecture_rules_document(bad)
    assert any("schema_version" in p for p in problems)
    assert any(
        "relationship_rules must be a non-empty array" in p for p in problems
    )


def test_architecture_rules_rejects_missing_ar_n002(arch_rules):
    bad = json.loads(json.dumps(arch_rules))
    bad["node_rules"] = [r for r in bad["node_rules"] if r["rule_id"] != "AR-N002"]
    problems = validate_architecture_rules_document(bad)
    assert any("AR-N002" in p for p in problems)


# ---------------------------------------------------------------------------
# A. Agent definition mapping
# ---------------------------------------------------------------------------


def test_agent_definition_maps_to_concept_agent(fixture_architecture_model):
    arch = fixture_architecture_model.architecture
    nodes = arch["nodes"]
    assert nodes, "expected architecture nodes"
    for node in nodes:
        assert node["kind"] == "concept"
        assert node["subkind"] == "agent"
        assert node["id"].startswith("concept:agent:")
        assert "agent_definition" not in node["id"]
        assert ".md" not in node["id"]
        assert not node["id"].startswith("concept:agent:artifact:")

    mappings = fixture_architecture_model.mappings
    assert mappings
    eng_ids = {n["id"] for n in fixture_architecture_model.engineering["nodes"]}
    arch_ids = {n["id"] for n in nodes}
    agent_eng = [
        n
        for n in fixture_architecture_model.engineering["nodes"]
        if n.get("kind") == "artifact" and n.get("subkind") == "agent_definition"
    ]
    assert agent_eng
    for eng_node in agent_eng:
        related = [
            m
            for m in mappings
            if m["source"] == eng_node["id"] and m["kind"] == "CONTRIBUTES_TO"
        ]
        assert related, eng_node["id"]
        for m in related:
            assert m["target"] in arch_ids
            assert m["classification"] == "derived"
            assert m["evidence"]
            assert m["id"] == mapping_identity(
                "CONTRIBUTES_TO", m["source"], m["target"]
            )
            assert m["source"] in eng_ids


def test_fixture_agent_identity(fixture_architecture_model):
    names = {
        (n.get("attributes") or {}).get("name")
        for n in fixture_architecture_model.architecture["nodes"]
    }
    assert "fixture-agent" in names
    assert "concept:agent:fixture-agent" in {
        n["id"] for n in fixture_architecture_model.architecture["nodes"]
    }


def test_resolver_identity_helpers():
    node = {
        "id": "artifact:agent_definition:.claude/agents/dag-agent.md",
        "kind": "artifact",
        "subkind": "agent_definition",
        "attributes": {"path": ".claude/agents/dag-agent.md"},
    }
    assert agent_name_from_engineering_node(node) == "dag-agent"
    assert architecture_agent_node_id("dag-agent") == "concept:agent:dag-agent"
    # not a file path id
    assert architecture_agent_node_id("dag-agent") != "concept:agent:.claude/agents/dag-agent.md"
    # unsupported kind
    assert (
        agent_name_from_engineering_node(
            {"kind": "artifact", "subkind": "agent_command", "attributes": {}}
        )
        is None
    )


# ---------------------------------------------------------------------------
# B. Evidence
# ---------------------------------------------------------------------------


def test_architecture_node_and_mapping_have_evidence(fixture_architecture_model):
    commit = FIXTURE_COMMIT
    for node in fixture_architecture_model.architecture["nodes"]:
        assert node["evidence"]
        assert all(e["commit"] == commit for e in node["evidence"])
    for mapping in fixture_architecture_model.mappings:
        assert mapping["evidence"]
        assert all(e["commit"] == commit for e in mapping["evidence"])


def test_evidence_traceable_to_engineering_node(fixture_architecture_model):
    eng_by_id = {
        n["id"]: n for n in fixture_architecture_model.engineering["nodes"]
    }
    for mapping in fixture_architecture_model.mappings:
        eng = eng_by_id[mapping["source"]]
        eng_ev = eng["evidence"]
        map_ev = mapping["evidence"]
        assert map_ev
        for item in eng_ev:
            assert item in map_ev


def test_no_duplicate_evidence(fixture_architecture_model):
    for node in fixture_architecture_model.architecture["nodes"]:
        keys = [
            (
                e.get("path"),
                e.get("line_start"),
                e.get("symbol"),
                e.get("artifact_type"),
                e.get("commit"),
            )
            for e in node["evidence"]
        ]
        assert len(keys) == len(set(keys))
    for mapping in fixture_architecture_model.mappings:
        keys = [
            (
                e.get("path"),
                e.get("line_start"),
                e.get("symbol"),
                e.get("artifact_type"),
                e.get("commit"),
            )
            for e in mapping["evidence"]
        ]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# C. Architecture identity reuse
# ---------------------------------------------------------------------------


def test_multiple_engineering_nodes_map_to_one_architecture_node(
    fixture_engineering_document, arch_rules
):
    base = fixture_engineering_document
    # same stem → same concept:agent:shared-agent
    doc = _engineer_model_with_extra_agents(
        base,
        [
            ".claude/agents/fixture-agent.md",  # existing
            "runbooks/shared-agent.md",
            "docs/shared-agent.md",
        ],
    )
    model = build_architecture_model(doc, arch_rules)
    arch_ids = [n["id"] for n in model.architecture["nodes"]]
    assert arch_ids.count("concept:agent:shared-agent") == 1
    shared_maps = [
        m for m in model.mappings if m["target"] == "concept:agent:shared-agent"
    ]
    assert len(shared_maps) == 2
    # fixture-agent still present once
    assert arch_ids.count("concept:agent:fixture-agent") == 1
    # evidence on shared node is union without duplicates
    shared_node = next(
        n for n in model.architecture["nodes"] if n["id"] == "concept:agent:shared-agent"
    )
    paths = [e["path"] for e in shared_node["evidence"]]
    assert set(paths) == {"runbooks/shared-agent.md", "docs/shared-agent.md"}
    assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# D. No unsupported architecture relationships
# ---------------------------------------------------------------------------


def test_engineering_relationships_not_copied(fixture_architecture_model):
    eng_rels = fixture_architecture_model.engineering["relationships"]
    assert eng_rels, "fixture should have engineering relationships"
    kinds = {r["kind"] for r in eng_rels}
    assert "CONTAINS" in kinds or "IMPORTS" in kinds

    arch_rels = fixture_architecture_model.architecture["relationships"]
    # fixture TRIGGERS source is agent_command (unmapped) → no architecture edge
    assert arch_rels == []
    for mapping in fixture_architecture_model.mappings:
        assert mapping["kind"] == "CONTRIBUTES_TO"
        assert mapping["kind"] not in {
            "CONTAINS",
            "IMPORTS",
            "DEPENDS_ON",
            "TRIGGERS",
            "DEPENDS_ON",
            "EXPOSES",
        }


# ---------------------------------------------------------------------------
# D2. TRIGGERS elevation (AR-R001)
# ---------------------------------------------------------------------------


def _inject_agent_triggers(
    doc: Dict[str, Any], source_path: str, target_path: str
) -> Dict[str, Any]:
    """Clone doc and add engineering TRIGGERS between two agent_definition nodes."""
    clone = json.loads(json.dumps(doc))
    commit = clone["repository"]["commit_sha"]
    nodes = clone["engineering"]["nodes"]
    existing_ids = {n["id"] for n in nodes}
    for path in (source_path, target_path):
        node_id = "artifact:agent_definition:{}".format(path)
        if node_id not in existing_ids:
            nodes.append(
                {
                    "id": node_id,
                    "kind": "artifact",
                    "subkind": "agent_definition",
                    "classification": "observed",
                    "attributes": {
                        "path": path,
                        "fact_type": "artifact",
                        "fact_kind": "agent_definition",
                    },
                    "evidence": [
                        {
                            "path": path,
                            "artifact_type": "agent_definition",
                            "commit": commit,
                        }
                    ],
                }
            )
            existing_ids.add(node_id)
    nodes.sort(key=lambda n: n["id"])

    source_id = "artifact:agent_definition:{}".format(source_path)
    target_id = "artifact:agent_definition:{}".format(target_path)
    rel_id = "rel:TRIGGERS:{}:{}".format(source_id, target_id)
    rels = clone["engineering"]["relationships"]
    if not any(r.get("id") == rel_id for r in rels):
        rels.append(
            {
                "id": rel_id,
                "kind": "TRIGGERS",
                "source": source_id,
                "target": target_id,
                "classification": "observed",
                "evidence": [
                    {
                        "path": ".claude/commands/fixture-command.md",
                        "artifact_type": "file",
                        "commit": commit,
                    }
                ],
            }
        )
        rels.sort(
            key=lambda r: (
                r.get("kind") or "",
                r.get("source") or "",
                r.get("target") or "",
                r.get("id") or "",
            )
        )
    return clone


def test_triggers_relationship_created_for_resolved_mappings(
    fixture_engineering_document, arch_rules
):
    """Both endpoints agent_definition → mapped → architecture TRIGGERS."""
    doc = _inject_agent_triggers(
        fixture_engineering_document,
        ".claude/agents/fixture-agent.md",
        ".claude/agents/peer-agent.md",
    )
    model = build_architecture_model(doc, arch_rules)
    arch_rels = model.architecture["relationships"]
    triggers = [r for r in arch_rels if r["kind"] == "TRIGGERS"]
    assert triggers, "expected architecture TRIGGERS when both endpoints map"
    rel = triggers[0]
    assert rel["classification"] == "derived"
    assert rel["evidence"]
    assert rel["id"] == architecture_relationship_identity(
        "TRIGGERS", rel["source"], rel["target"]
    )
    assert rel["source"] == "concept:agent:fixture-agent"
    assert rel["target"] == "concept:agent:peer-agent"
    arch_ids = {n["id"] for n in model.architecture["nodes"]}
    assert rel["source"] in arch_ids
    assert rel["target"] in arch_ids
    # evidence includes M3 relationship evidence
    assert any(
        e["path"] == ".claude/commands/fixture-command.md" for e in rel["evidence"]
    )


def test_triggers_missing_source_architecture_mapping(
    fixture_engineering_document, arch_rules
):
    """Source agent_command (unmapped) → unresolved, not architecture edge."""
    # fixture already has: agent_command → agent_definition TRIGGERS
    model = build_architecture_model(fixture_engineering_document, arch_rules)
    arch_rels = model.architecture["relationships"]
    assert all(r["kind"] != "TRIGGERS" for r in arch_rels)

    unresolved = [
        u for u in model.architecture["unresolved"] if u["kind"] == "TRIGGERS"
    ]
    assert unresolved, "expected unresolved TRIGGERS candidate"
    claim = next(
        u
        for u in unresolved
        if "agent_command" in (u["source"] or "")
        and "fixture-agent" in (u["target"] or "")
    )
    assert claim["reason"] == "missing_endpoint"
    assert claim["evidence"]


def test_triggers_missing_target_architecture_mapping(
    fixture_engineering_document, arch_rules
):
    """Target agent_definition exists but source peer removed → only target maps;
    if we inject TRIGGERS from unmapped file to mapped agent, target maps only."""
    doc = json.loads(json.dumps(fixture_engineering_document))
    commit = doc["repository"]["commit_sha"]
    # ensure target agent exists (it does) and add file→agent TRIGGERS
    source_id = "artifact:file:docs/workflow.md"
    target_id = "artifact:agent_definition:.claude/agents/fixture-agent.md"
    nodes = doc["engineering"]["nodes"]
    if not any(n["id"] == source_id for n in nodes):
        nodes.append(
            {
                "id": source_id,
                "kind": "artifact",
                "subkind": "file",
                "classification": "observed",
                "attributes": {"path": "docs/workflow.md"},
                "evidence": [
                    {
                        "path": "docs/workflow.md",
                        "artifact_type": "file",
                        "commit": commit,
                    }
                ],
            }
        )
        nodes.sort(key=lambda n: n["id"])
    rel_id = "rel:TRIGGERS:{}:{}".format(source_id, target_id)
    rels = doc["engineering"]["relationships"]
    if not any(r["id"] == rel_id for r in rels):
        rels.append(
            {
                "id": rel_id,
                "kind": "TRIGGERS",
                "source": source_id,
                "target": target_id,
                "classification": "observed",
                "evidence": [
                    {
                        "path": "docs/workflow.md",
                        "artifact_type": "file",
                        "commit": commit,
                    }
                ],
            }
        )
        rels.sort(
            key=lambda r: (
                r.get("kind") or "",
                r.get("source") or "",
                r.get("target") or "",
                r.get("id") or "",
            )
        )
    model = build_architecture_model(doc, arch_rules)
    # no architecture edge from this pair
    for r in model.architecture["relationships"]:
        assert not (r["kind"] == "TRIGGERS" and r["source"] == source_id)
    claim = next(
        u
        for u in model.architecture["unresolved"]
        if u["kind"] == "TRIGGERS" and u["source"] == source_id
    )
    assert claim["reason"] == "missing_endpoint"
    assert claim["evidence"]


def test_triggers_architecture_relationship_determinism(
    fixture_engineering_document, arch_rules
):
    doc = _inject_agent_triggers(
        fixture_engineering_document,
        ".claude/agents/fixture-agent.md",
        ".claude/agents/peer-agent.md",
    )
    a = build_architecture_model(doc, arch_rules).to_dict()
    b = build_architecture_model(doc, arch_rules).to_dict()
    assert [r["id"] for r in a["architecture"]["relationships"]] == [
        r["id"] for r in b["architecture"]["relationships"]
    ]
    assert a["architecture"]["relationships"] == b["architecture"]["relationships"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_triggers_architecture_evidence_preserves_m3_relationship(
    fixture_engineering_document, arch_rules
):
    doc = _inject_agent_triggers(
        fixture_engineering_document,
        ".claude/agents/fixture-agent.md",
        ".claude/agents/peer-agent.md",
    )
    m3_path = ".claude/commands/fixture-command.md"
    model = build_architecture_model(doc, arch_rules)
    rel = next(
        r for r in model.architecture["relationships"] if r["kind"] == "TRIGGERS"
    )
    # M3 relationship evidence carried forward
    assert any(e["path"] == m3_path for e in rel["evidence"])
    # mapping evidence present (agent definition paths)
    assert any(
        "fixture-agent" in (e.get("path") or "") or "peer-agent" in (e.get("path") or "")
        for e in rel["evidence"]
    )
    assert all(e["commit"] == FIXTURE_COMMIT for e in rel["evidence"])
    assert len(rel["evidence"]) >= 1


# ---------------------------------------------------------------------------
# E. Unsupported classifications
# ---------------------------------------------------------------------------


def test_unsupported_artifacts_do_not_become_architecture(
    fixture_engineering_document, arch_rules
):
    doc = json.loads(json.dumps(fixture_engineering_document))
    commit = doc["repository"]["commit_sha"]
    extras = []
    for subkind, path in (
        ("agent_command", ".claude/commands/deploy.md"),
        ("agent_config", ".claude/agents/config.json"),
        ("mcp_config", ".mcp.json"),
        ("mcp_server_source", "src/mcp/server.py"),
    ):
        extras.append(
            {
                "id": "artifact:{}:{}".format(subkind, path),
                "kind": "artifact",
                "subkind": subkind,
                "classification": "observed",
                "attributes": {"path": path},
                "evidence": [
                    {
                        "path": path,
                        "artifact_type": subkind,
                        "commit": commit,
                    }
                ],
            }
        )
    # also a Python class named Agent-like via code entity — not architecture
    extras.append(
        {
            "id": "code_entity:src/service.py::Agent",
            "kind": "code_entity",
            "subkind": "class",
            "classification": "observed",
            "attributes": {"path": "src/service.py", "qualified_name": "src/service.py::Agent"},
            "evidence": [
                {
                    "path": "src/service.py",
                    "artifact_type": "python",
                    "commit": commit,
                    "symbol": "Agent",
                }
            ],
        }
    )
    doc["engineering"]["nodes"].extend(extras)
    doc["engineering"]["nodes"].sort(key=lambda n: n["id"])

    model = build_architecture_model(doc, arch_rules)
    for node in model.architecture["nodes"]:
        assert node["kind"] == "concept" and node["subkind"] == "agent"
        node_id = node["id"]
        assert "agent_command" not in node_id
        assert "agent_config" not in node_id
        assert "mcp_config" not in node_id
        assert "mcp_server" not in node_id
        assert not node_id.endswith("::Agent")
        assert "src/service.py" not in node_id

    # no mappings from unsupported engineering sources
    unsupported_ids = {e["id"] for e in extras}
    for mapping in model.mappings:
        assert mapping["source"] not in unsupported_ids
    # architecture relationships still empty
    assert model.architecture["relationships"] == []


# ---------------------------------------------------------------------------
# F. Determinism
# ---------------------------------------------------------------------------


def test_build_deterministic(fixture_engineering_document, arch_rules):
    a = build_architecture_model(fixture_engineering_document, arch_rules).to_dict()
    b = build_architecture_model(fixture_engineering_document, arch_rules).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_create_writes_artifact_byte_stable(fixture_engineering_document, arch_rules, tmp_path):
    eng_path = tmp_path / "eng.json"
    eng_path.write_text(
        json.dumps(fixture_engineering_document, indent=2) + "\n", encoding="utf-8"
    )
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(arch_rules, indent=2) + "\n", encoding="utf-8")

    m1, p1 = create_architecture_model(
        engineering_model_path=eng_path,
        rules_path=rules_path,
        output_root=tmp_path / "o1",
        snapshot_id="fixture_2026-09-22_aaaaaaaa",
    )
    m2, p2 = create_architecture_model(
        engineering_model_path=eng_path,
        rules_path=rules_path,
        output_root=tmp_path / "o2",
        snapshot_id="fixture_2026-09-22_aaaaaaaa",
    )
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.name == "architecture_model.json"
    assert p1.parent.name == "architecture"


# ---------------------------------------------------------------------------
# G. Validation
# ---------------------------------------------------------------------------


def test_validation_rejects_missing_mapping_evidence(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    doc["mappings"][0]["evidence"] = []
    problems = validate_knowledge_model_document(
        doc, expected_commit_sha=FIXTURE_COMMIT
    )
    assert any("mapping requires evidence" in p for p in problems)
    with pytest.raises(ArchitectureModelValidationError):
        validate_knowledge_model_or_raise(
            doc, expected_commit_sha=FIXTURE_COMMIT, require_architecture=True
        )


def test_validation_rejects_unknown_mapping_source(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    doc["mappings"][0]["source"] = "artifact:file:missing.py"
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any("existing engineering node" in p for p in problems)


def test_validation_rejects_unknown_mapping_target(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    doc["mappings"][0]["target"] = "concept:agent:nope"
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any("existing architecture node" in p for p in problems)


def test_validation_rejects_missing_architecture_evidence(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    doc["architecture"]["nodes"][0]["evidence"] = []
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any("architecture node requires evidence" in p for p in problems)


def test_validation_rejects_unsupported_architecture_relationship_kind(
    fixture_architecture_model,
):
    doc = fixture_architecture_model.to_dict()
    arch_ids = [n["id"] for n in doc["architecture"]["nodes"]]
    assert arch_ids
    doc["architecture"]["relationships"].append(
        {
            "id": "rel:DEPENDS_ON:{}:{}".format(arch_ids[0], arch_ids[0]),
            "kind": "DEPENDS_ON",
            "source": arch_ids[0],
            "target": arch_ids[0],
            "classification": "derived",
            "evidence": [
                {
                    "path": "x",
                    "artifact_type": "file",
                    "commit": FIXTURE_COMMIT,
                }
            ],
        }
    )
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any("invalid kind" in p for p in problems)


def test_validation_rejects_triggers_without_evidence(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    arch_ids = [n["id"] for n in doc["architecture"]["nodes"]]
    assert arch_ids
    doc["architecture"]["relationships"].append(
        {
            "id": "rel:TRIGGERS:{}:{}".format(arch_ids[0], arch_ids[0]),
            "kind": "TRIGGERS",
            "source": arch_ids[0],
            "target": arch_ids[0],
            "classification": "derived",
            "evidence": [],
        }
    )
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any(
        "architecture.relationships[0]" in p and "evidence" in p for p in problems
    )


def test_validation_rejects_triggers_unknown_endpoint(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    arch_ids = [n["id"] for n in doc["architecture"]["nodes"]]
    assert arch_ids
    doc["architecture"]["relationships"].append(
        {
            "id": "rel:TRIGGERS:{}:concept:agent:missing".format(arch_ids[0]),
            "kind": "TRIGGERS",
            "source": arch_ids[0],
            "target": "concept:agent:missing",
            "classification": "derived",
            "evidence": [
                {
                    "path": "x",
                    "artifact_type": "file",
                    "commit": FIXTURE_COMMIT,
                }
            ],
        }
    )
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any("target must exist" in p for p in problems)


def test_validation_rejects_duplicate_architecture_ids(fixture_architecture_model):
    doc = fixture_architecture_model.to_dict()
    doc["architecture"]["nodes"].append(dict(doc["architecture"]["nodes"][0]))
    problems = validate_knowledge_model_document(doc, expected_commit_sha=FIXTURE_COMMIT)
    assert any("unique_architecture_node_ids" in p for p in problems)


def test_require_architecture_flag(fixture_engineering_document, arch_rules):
    eng_only = fixture_engineering_document
    problems = validate_knowledge_model_document(
        eng_only, expected_commit_sha=FIXTURE_COMMIT, require_architecture=True
    )
    assert any("architecture section required" in p for p in problems)


# ---------------------------------------------------------------------------
# ADOP integration (§13) + Data Onboarding Agent mapping (AR-N002)
# ---------------------------------------------------------------------------


def _doa_node(model):
    nodes = [n for n in model.architecture["nodes"] if n["id"] == DOA_ID]
    assert nodes, "expected {}".format(DOA_ID)
    return nodes[0]


def _doa_mappings(model):
    return [m for m in model.mappings if m["target"] == DOA_ID]


def _arch_triggers(model):
    return [
        r for r in model.architecture["relationships"] if r["kind"] == "TRIGGERS"
    ]


def _unresolved_triggers(model):
    return [
        u for u in model.architecture["unresolved"] if u["kind"] == "TRIGGERS"
    ]


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_data_onboarding_agent_node_created():
    """Test 1 — AR-N002 creates concept:agent:data-onboarding-agent with evidence."""
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    model = build_architecture_model(eng, rules)
    node = _doa_node(model)
    assert node["kind"] == "concept"
    assert node["subkind"] == "agent"
    assert node["classification"] == "derived"
    assert node["attributes"]["name"] == "Data Onboarding Agent"
    assert node["evidence"], "canonical architecture node requires evidence"
    paths = {e["path"] for e in node["evidence"]}
    assert ".claude/commands/onboard-workflow.md" in paths
    assert "SKILLS.md" in paths
    assert "docs/workflow-diagrams.md" in paths
    assert "runbooks/data-onboarding-agent/README.md" in paths
    assert all(e["commit"] == ADOP_COMMIT for e in node["evidence"])
    # identity is fixed — not analyzer/filename-derived
    for forbidden in ("ast", "yaml", "filesystem", "uuid", ".md"):
        assert forbidden not in node["id"]


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_data_onboarding_agent_mappings_exist():
    """Test 2 — four orchestration sources CONTRIBUTES_TO the DOA concept."""
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    model = build_architecture_model(eng, rules)
    maps = _doa_mappings(model)
    assert len(maps) == 4
    sources = {m["source"] for m in maps}
    assert sources == set(ORCHESTRATION_SOURCE_IDS)
    for m in maps:
        assert m["kind"] == "CONTRIBUTES_TO"
        assert m["classification"] == "derived"
        assert m["evidence"]
        assert m["id"] == mapping_identity("CONTRIBUTES_TO", m["source"], m["target"])
        assert all(e["commit"] == ADOP_COMMIT for e in m["evidence"])


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_adop_triggers_resolve_via_data_onboarding_mapping():
    """Test 3 — AR-R001 elevates M3 TRIGGERS once DOA source mapping exists."""
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    m3_triggers = [
        r for r in eng["engineering"]["relationships"] if r.get("kind") == "TRIGGERS"
    ]
    assert m3_triggers, "expected M3 TRIGGERS on ADOP"

    rules = load_architecture_rules()
    model = build_architecture_model(eng, rules)
    triggers = _arch_triggers(model)
    unresolved = _unresolved_triggers(model)

    # All four orchestration sources map → every M3 TRIGGERS endpoint resolves
    assert unresolved == [], (
        "expected all TRIGGERS endpoints mapped after AR-N002; got {}".format(
            unresolved
        )
    )
    assert triggers, "expected architecture TRIGGERS"

    arch_ids = {n["id"] for n in model.architecture["nodes"]}
    eng_to_arch = {m["source"]: m["target"] for m in model.mappings}
    expected_pairs = set()
    for rel in m3_triggers:
        expected_pairs.add(
            (eng_to_arch[rel["source"]], eng_to_arch[rel["target"]])
        )

    actual_pairs = {(r["source"], r["target"]) for r in triggers}
    assert actual_pairs == expected_pairs

    for r in triggers:
        assert r["source"] == DOA_ID
        assert r["classification"] == "derived"
        assert r["evidence"]
        assert r["source"] in arch_ids and r["target"] in arch_ids
        assert r["id"] == architecture_relationship_identity(
            "TRIGGERS", r["source"], r["target"]
        )
        # evidence includes at least one M3 TRIGGERS relationship path
        assert any(e.get("path") for e in r["evidence"])

    # expected specialized agents (from actual M3 targets)
    targets = {r["target"] for r in triggers}
    assert "concept:agent:metadata-agent" in targets
    assert "concept:agent:transformation-agent" in targets
    assert "concept:agent:quality-agent" in targets
    assert "concept:agent:dag-agent" in targets
    assert "concept:agent:iac-agent" in targets
    assert "concept:agent:ontology-agent" in targets


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_adop_unsupported_claims_stay_unresolved_when_source_unmapped(
    fixture_engineering_document, arch_rules
):
    """Test 4 — sources outside AR-N002 remain unresolved, not canonical."""
    # fixture command → fixture-agent is NOT in AR-N002's engineering_node_ids
    model = build_architecture_model(fixture_engineering_document, arch_rules)
    assert all(
        r["source"] != "artifact:agent_command:.claude/commands/fixture-command.md"
        for r in model.architecture["relationships"]
        if r["kind"] == "TRIGGERS"
    )
    claim = next(
        u
        for u in model.architecture["unresolved"]
        if u["kind"] == "TRIGGERS"
        and "fixture-command" in (u.get("source") or "")
    )
    assert claim["reason"] == "missing_endpoint"
    assert claim["evidence"]
    # not also present as canonical
    for r in model.architecture["relationships"]:
        assert not (
            r["kind"] == "TRIGGERS"
            and r["source"] == claim["source"]
            and r["target"] == claim["target"]
        )


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_data_onboarding_evidence_invariant():
    """Test 5 — every canonical node/relationship/mapping has evidence."""
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    model = build_architecture_model(eng, rules)
    for node in model.architecture["nodes"]:
        assert len(node["evidence"]) > 0
    for rel in model.architecture["relationships"]:
        assert len(rel["evidence"]) > 0
    for mapping in model.mappings:
        assert len(mapping["evidence"]) > 0
    for claim in model.architecture["unresolved"]:
        assert len(claim["evidence"]) > 0


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_data_onboarding_determinism():
    """Test 6 — repeated builds serialize identically."""
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    a = build_architecture_model(eng, rules).to_dict()
    b = build_architecture_model(eng, rules).to_dict()
    assert [n["id"] for n in a["architecture"]["nodes"]] == [
        n["id"] for n in b["architecture"]["nodes"]
    ]
    assert [r["id"] for r in a["architecture"]["relationships"]] == [
        r["id"] for r in b["architecture"]["relationships"]
    ]
    assert [m["id"] for m in a["mappings"]] == [m["id"] for m in b["mappings"]]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_adop_architecture_model_builds():
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    model = build_architecture_model(eng, rules)
    document = model.to_dict()

    validate_knowledge_model_or_raise(
        document, expected_commit_sha=ADOP_COMMIT, require_architecture=True
    )

    arch_nodes = model.architecture["nodes"]
    mappings = model.mappings
    arch_rels = model.architecture["relationships"]
    unresolved = model.architecture["unresolved"]

    assert DOA_ID in {n["id"] for n in arch_nodes}
    assert len(arch_nodes) > 0
    assert len(mappings) > 0
    assert any(r["kind"] == "TRIGGERS" for r in arch_rels)

    for node in arch_nodes:
        assert node["kind"] == "concept"
        assert node["subkind"] == "agent"
        assert node["id"].startswith("concept:agent:")
        assert node["evidence"]
        assert all(e["commit"] == ADOP_COMMIT for e in node["evidence"])

    eng_ids = {n["id"] for n in model.engineering["nodes"]}
    arch_ids = {n["id"] for n in arch_nodes}
    for mapping in mappings:
        assert mapping["kind"] == "CONTRIBUTES_TO"
        assert mapping["source"] in eng_ids
        assert mapping["target"] in arch_ids
        assert mapping["evidence"]
        assert mapping["id"] == mapping_identity(
            "CONTRIBUTES_TO", mapping["source"], mapping["target"]
        )

    for rel in arch_rels:
        assert rel["kind"] == "TRIGGERS"
        assert rel["source"] in arch_ids
        assert rel["target"] in arch_ids
        assert rel["evidence"]

    # unique architecture ids, sorted
    ids = [n["id"] for n in arch_nodes]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)

    # AR-N001 agent_definitions (11) + AR-N002 sources (4) = 15 mappings
    agent_eng = [
        n
        for n in model.engineering["nodes"]
        if n.get("kind") == "artifact" and n.get("subkind") == "agent_definition"
    ]
    assert len(mappings) == len(agent_eng) + 4

    evidence_total = sum(len(n["evidence"]) for n in arch_nodes)
    evidence_total += sum(len(r["evidence"]) for r in arch_rels)
    evidence_total += sum(len(m["evidence"]) for m in mappings)
    evidence_total += sum(len(u["evidence"]) for u in unresolved)
    assert evidence_total > 0

    print("ADOP architecture nodes:", len(arch_nodes))
    print("ADOP mappings:", len(mappings))
    print("ADOP architecture relationships:", len(arch_rels))
    print("ADOP architecture unresolved:", len(unresolved))
    print("ADOP architecture evidence:", evidence_total)
    print("sample ids:", ids[:5])


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_adop_triggers_before_after_refinement():
    """Document actual before/after for the 15 M3 TRIGGERS claims."""
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    m3_triggers = [
        r for r in eng["engineering"]["relationships"] if r.get("kind") == "TRIGGERS"
    ]
    assert len(m3_triggers) == 15

    rules = load_architecture_rules()
    model = build_architecture_model(eng, rules)
    triggers = _arch_triggers(model)
    unresolved = _unresolved_triggers(model)
    doa = _doa_node(model)

    # Before: DOA absent, 0 arch TRIGGERS, 15 unresolved (prior milestone state)
    # After: DOA present, elevated pairs, 0 unresolved (all sources in AR-N002)
    assert doa["id"] == DOA_ID
    assert len(unresolved) == 0
    # 15 M3 claims collapse to unique (source,target) architecture pairs
    eng_to_arch = {m["source"]: m["target"] for m in model.mappings}
    expected = {
        (eng_to_arch[r["source"]], eng_to_arch[r["target"]]) for r in m3_triggers
    }
    actual = {(r["source"], r["target"]) for r in triggers}
    assert actual == expected
    assert len(triggers) < 15  # collapsed duplicates
    assert len(triggers) == 7

    print("BEFORE: DOA=0 arch_TRIGGERS=0 unresolved_TRIGGERS=15")
    print(
        "AFTER: DOA=1 arch_TRIGGERS={} unresolved_TRIGGERS={}".format(
            len(triggers), len(unresolved)
        )
    )
    print("canonical TRIGGERS:")
    for r in sorted(triggers, key=lambda x: x["id"]):
        print(" ", r["id"])


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_adop_architecture_determinism():
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    a = build_architecture_model(eng, rules).to_dict()
    b = build_architecture_model(eng, rules).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.skipif(not ADOP_ENG_MODEL.is_file(), reason="ADOP EngineeringModel absent")
def test_adop_architecture_determinism():
    eng = json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    rules = load_architecture_rules()
    a = build_architecture_model(eng, rules).to_dict()
    b = build_architecture_model(eng, rules).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.skipif(not ADOP_ARCH_RULES.is_file(), reason="ADOP architecture rules absent")
def test_adop_create_artifact(tmp_path):
    model, path = create_architecture_model(
        engineering_model_path=ADOP_ENG_MODEL,
        rules_path=ADOP_ARCH_RULES,
        output_root=tmp_path,
        snapshot_id=PRIMARY_SNAPSHOT_ID,
    )
    assert path.is_file()
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_knowledge_model_or_raise(
        document, expected_commit_sha=ADOP_COMMIT, require_architecture=True
    )
    assert document["mappings"]
    # AR-N002 + AR-R001: DOA node present; architecture TRIGGERS elevated
    arch_ids = {n["id"] for n in document["architecture"]["nodes"]}
    assert DOA_ID in arch_ids
    triggers = [
        r
        for r in document["architecture"]["relationships"]
        if r["kind"] == "TRIGGERS"
    ]
    assert triggers
    assert all(r["source"] == DOA_ID for r in triggers)
    assert all(r["evidence"] for r in triggers)
    assert not any(
        u["kind"] == "TRIGGERS" for u in document["architecture"]["unresolved"]
    )
    # no project-root canonical copy
    assert not (PROJECT_ROOT / "architecture_model.json").exists()


def test_bad_rules_path_raises(tmp_path):
    with pytest.raises((ArchitectureModelError, ArchitectureRulesSchemaError)):
        create_architecture_model(
            engineering_model_path=ADOP_ENG_MODEL if ADOP_ENG_MODEL.is_file() else None,
            rules_path=tmp_path / "missing.json",
            output_root=tmp_path / "out",
        )
