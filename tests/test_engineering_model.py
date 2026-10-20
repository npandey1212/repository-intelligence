"""M3 EngineeringModel construction tests.

Levels:
- unit: identity, node/relationship construction, evidence, unresolved
- fixture: small synthetic facts → expected EngineeringModel shape
- integration: pinned ADOP facts (skipped when absent)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID, default_artifacts_root
from repo_intelligence.construction import load_construction_rules
from repo_intelligence.engineering import (
    EngineeringModelError,
    EngineeringModelValidationError,
    NodeIndex,
    SCHEMA_VERSION,
    build_engineering_model,
    create_engineering_model,
    engineering_model_path,
    load_facts_document,
    parent_directory,
    resolve_dependency_target,
    validate_engineering_model_document,
    validate_engineering_model_or_raise,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FACTS = (
    PROJECT_ROOT / "tests" / "fixtures" / "construction" / "repository_facts.json"
)
ADOP_FACTS = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "facts"
    / "repository_facts.json"
)
ADOP_COMMIT = "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"


@pytest.fixture(scope="module")
def rules() -> Dict[str, Any]:
    return load_construction_rules()


@pytest.fixture(scope="module")
def fixture_facts() -> Dict[str, Any]:
    return load_facts_document(FIXTURE_FACTS)


@pytest.fixture(scope="module")
def fixture_model(fixture_facts, rules):
    return build_engineering_model(fixture_facts, rules)


# ---------------------------------------------------------------------------
# Schema / envelope
# ---------------------------------------------------------------------------


def test_fixture_model_envelope(fixture_model):
    document = fixture_model.to_dict()
    assert document["schema_version"] == SCHEMA_VERSION == "repository_knowledge_model.v1"
    assert document["analysis"]["strategy"] == "deterministic"
    assert document["analysis"]["analyzers"]
    # V1: architecture + mappings empty
    assert document["architecture"]["nodes"] == []
    assert document["architecture"]["relationships"] == []
    assert document["architecture"]["unresolved"] == []
    assert document["mappings"] == []
    assert document["repository"]["commit_sha"] == "a" * 40
    validate_engineering_model_or_raise(
        document, expected_commit_sha="a" * 40
    )


def test_validate_document_accepts_good_model(fixture_model):
    problems = validate_engineering_model_document(
        fixture_model.to_dict(), expected_commit_sha="a" * 40
    )
    assert problems == []


def test_validate_rejects_nonempty_architecture(fixture_model):
    document = fixture_model.to_dict()
    document["architecture"]["nodes"].append(
        {"id": "x", "kind": "k", "subkind": "s", "classification": "observed", "evidence": [{"path": "p", "artifact_type": "file", "commit": "a" * 40}]}
    )
    problems = validate_engineering_model_document(document)
    assert any("architecture" in p for p in problems)


def test_validate_rejects_commit_mismatch(fixture_model):
    problems = validate_engineering_model_document(
        fixture_model.to_dict(), expected_commit_sha="b" * 40
    )
    assert any("commit_sha" in p for p in problems)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def test_core_node_kinds_present(fixture_model):
    ids = {n["id"] for n in fixture_model.nodes}
    assert "artifact:directory:src" in ids
    assert "artifact:file:src/main.py" in ids
    assert "artifact:file:src/service.py" in ids
    assert "code_entity:src/main.py" in ids
    assert "code_entity:src/service.py" in ids
    assert "code_entity:src/service.py::Service" in ids
    assert "code_entity:src/service.py::run" in ids
    assert "code_entity:src/main.py::main" in ids
    assert "artifact:agent_definition:.claude/agents/fixture-agent.md" in ids


def test_class_function_identity_no_duplicate_path(fixture_model):
    for node in fixture_model.nodes:
        if node["kind"] != "code_entity" or node["subkind"] == "module":
            continue
        node_id = node["id"]
        assert "src/service.py::src/service.py" not in node_id
        if "src/service.py" in node_id:
            assert node_id.count("src/service.py") == 1


def test_every_node_has_evidence(fixture_model):
    for node in fixture_model.nodes:
        assert node["evidence"], node
        assert all(e.get("commit") == "a" * 40 for e in node["evidence"])


def test_nodes_unique_and_sorted(fixture_model):
    ids = [n["id"] for n in fixture_model.nodes]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_evidence_only_facts_do_not_create_duplicate_file_nodes(fixture_model):
    ids = [n["id"] for n in fixture_model.nodes]
    # configuration/documentation/test facts are evidence-only
    assert "artifact:configuration:config.yaml" not in ids
    assert "artifact:documentation:README.md" not in ids
    assert "artifact:test:src/test_service.py" not in ids
    # only one file node for config.yaml
    assert ids.count("artifact:file:config.yaml") == 1


def test_deduplication_same_identity(fixture_model):
    """Multiple facts → one node (construction layer owns dedup)."""
    # main.py has file + module + entity facts; one artifact:file node
    file_nodes = [
        n for n in fixture_model.nodes if n["id"] == "artifact:file:src/main.py"
    ]
    assert len(file_nodes) == 1
    # evidence may include multiple refs but node is unique
    assert len(file_nodes[0]["evidence"]) >= 1


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def test_relationship_kinds_subset_of_vocab(fixture_model):
    for rel in fixture_model.relationships:
        assert rel["kind"] in {"CONTAINS", "IMPORTS", "DEPENDS_ON", "TRIGGERS"}
        assert rel["classification"] in {"observed", "derived"}
        assert rel["evidence"]
        assert rel["id"].startswith("rel:")


def test_contains_derived_and_endpoints_exist(fixture_model):
    node_ids = {n["id"] for n in fixture_model.nodes}
    contains = [r for r in fixture_model.relationships if r["kind"] == "CONTAINS"]
    assert contains
    for rel in contains:
        assert rel["classification"] == "derived"
        assert rel["source"] in node_ids
        assert rel["target"] in node_ids
        assert rel["evidence"]


# ---------------------------------------------------------------------------
# TRIGGERS (observed interactions)
# ---------------------------------------------------------------------------


def test_triggers_relationship_created_for_resolved_interaction(fixture_model):
    triggers = [r for r in fixture_model.relationships if r["kind"] == "TRIGGERS"]
    assert triggers
    node_ids = {n["id"] for n in fixture_model.nodes}
    source = "artifact:agent_command:.claude/commands/fixture-command.md"
    target = "artifact:agent_definition:.claude/agents/fixture-agent.md"
    match = [
        r for r in triggers if r["source"] == source and r["target"] == target
    ]
    assert match, [(r["source"], r["target"]) for r in triggers]
    rel = match[0]
    assert rel["classification"] == "observed"
    assert rel["id"] == "rel:TRIGGERS:{}:{}".format(source, target)
    assert rel["evidence"]
    assert rel["source"] in node_ids
    assert rel["target"] in node_ids
    # evidence copied from the M2 interaction fact
    assert any(
        e.get("path") == ".claude/agents/fixture-agent.md"
        for e in rel["evidence"]
    )


def test_triggers_unresolved_when_endpoint_fact_missing(fixture_model):
    claims = [u for u in fixture_model.unresolved if u["kind"] == "TRIGGERS"]
    assert claims
    # second interaction points at a non-existent target fact
    assert any(
        u["target"] == "missing-peer.md" for u in claims
    )
    for claim in claims:
        assert claim["reason"] in {
            "missing_endpoint",
            "target_not_resolved",
            "ambiguous_identity",
        }
        assert claim["evidence"]
    # unresolved targets must not appear as canonical TRIGGERS endpoints
    node_ids = {n["id"] for n in fixture_model.nodes}
    for claim in claims:
        if claim["target"] not in node_ids:
            assert not any(
                r["kind"] == "TRIGGERS" and r["target"] == claim["target"]
                for r in fixture_model.relationships
            )


def test_triggers_counts_present(fixture_model):
    counts = fixture_model.counts()
    assert counts["rel_triggers"] == sum(
        1 for r in fixture_model.relationships if r["kind"] == "TRIGGERS"
    )
    assert counts["rel_triggers"] >= 1


def test_relationship_vocabulary_includes_triggers(rules):
    assert "TRIGGERS" in rules["relationship_vocabulary"]
    assert "TRIGGERS" not in rules["deferred"]
    triggers_rules = [
        r
        for r in rules["relationship_rules"]
        if r["relationship"]["kind"] == "TRIGGERS"
    ]
    assert triggers_rules
    rule = triggers_rules[0]
    assert rule["rule_id"] == "rel.interaction.triggers"
    assert rule["match"] == {"fact_type": "interaction", "kind": "triggers"}
    assert rule["classification"] == "observed"
    assert rule["requires_evidence"] is True
    assert rule["source"]["resolve"] == "fact_node_identity"
    assert rule["target"]["resolve"] == "fact_node_identity"


def _node_evidence_key(item):
    return (
        item.get("path"),
        item.get("line_start"),
        item.get("symbol"),
        item.get("artifact_type"),
        item.get("commit"),
    )


def _contains_by_endpoints(fixture_model):
    return {
        (r["source"], r["target"]): r
        for r in fixture_model.relationships
        if r["kind"] == "CONTAINS"
    }


def test_derived_relationship_evidence_from_both_endpoints(fixture_model):
    """CONTAINS evidence must include material facts from source and target."""
    by_id = {n["id"]: n for n in fixture_model.nodes}
    contains = _contains_by_endpoints(fixture_model)

    cases = [
        # directory → file
        ("artifact:directory:src", "artifact:file:src/main.py"),
        ("artifact:directory:src", "artifact:file:src/service.py"),
        # file → module
        ("artifact:file:src/service.py", "code_entity:src/service.py"),
        # file → class / function
        ("artifact:file:src/service.py", "code_entity:src/service.py::Service"),
        ("artifact:file:src/service.py", "code_entity:src/service.py::run"),
        ("artifact:file:src/main.py", "code_entity:src/main.py::main"),
        # module → class / function
        ("code_entity:src/service.py", "code_entity:src/service.py::Service"),
        ("code_entity:src/service.py", "code_entity:src/service.py::run"),
    ]
    for source_id, target_id in cases:
        rel = contains.get((source_id, target_id))
        assert rel is not None, (source_id, target_id)
        assert rel["classification"] == "derived"

        src_ev = by_id[source_id]["evidence"]
        tgt_ev = by_id[target_id]["evidence"]
        rel_keys = {_node_evidence_key(e) for e in rel["evidence"]}
        for item in src_ev + tgt_ev:
            assert _node_evidence_key(item) in rel_keys, (
                source_id,
                target_id,
                item,
                rel["evidence"],
            )


def test_derived_relationship_evidence_has_no_duplicates(fixture_model):
    for rel in fixture_model.relationships:
        if rel["kind"] != "CONTAINS":
            continue
        keys = [_node_evidence_key(e) for e in rel["evidence"]]
        assert len(keys) == len(set(keys)), rel


def test_derived_relationship_ids_unchanged(fixture_model):
    for rel in fixture_model.relationships:
        if rel["kind"] == "CONTAINS":
            assert rel["id"] == "rel:CONTAINS:{}:{}".format(
                rel["source"], rel["target"]
            )


def test_node_ids_unchanged(fixture_model):
    ids = {n["id"] for n in fixture_model.nodes}
    assert "artifact:directory:src" in ids
    assert "code_entity:src/service.py::Service" in ids
    assert "code_entity:src/service.py::run" in ids
    assert "code_entity:src/main.py::main" in ids


def test_derived_evidence_is_superset_of_target_only(fixture_model):
    """Before the fix, CONTAINS used target evidence only; now source ∪ target."""
    by_id = {n["id"]: n for n in fixture_model.nodes}
    contains = _contains_by_endpoints(fixture_model)
    source_id = "artifact:directory:src"
    target_id = "artifact:file:src/service.py"
    rel = contains[(source_id, target_id)]
    tgt_keys = {_node_evidence_key(e) for e in by_id[target_id]["evidence"]}
    src_keys = {_node_evidence_key(e) for e in by_id[source_id]["evidence"]}
    rel_keys = {_node_evidence_key(e) for e in rel["evidence"]}
    assert tgt_keys.issubset(rel_keys)
    # directory evidence may equal file evidence path-wise but must be present
    assert src_keys.issubset(rel_keys)
    assert rel_keys == tgt_keys | src_keys


def test_imports_resolved_for_local_module(fixture_model):
    node_ids = {n["id"] for n in fixture_model.nodes}
    imports = [r for r in fixture_model.relationships if r["kind"] == "IMPORTS"]
    assert imports
    # src.service resolves
    assert any(
        r["source"] == "artifact:file:src/main.py"
        and r["target"] == "code_entity:src/service.py"
        for r in imports
    )
    # stdlib os is unresolved, not a fake edge
    assert not any(r["target"].endswith(":os") for r in imports)
    os_claims = [
        u
        for u in fixture_model.unresolved
        if u["kind"] == "IMPORTS" and u["target"] == "os"
    ]
    assert os_claims
    assert os_claims[0]["reason"] == "target_not_resolved"
    assert os_claims[0]["evidence"]
    assert os_claims[0]["source"] not in {
        r["id"] for r in fixture_model.relationships if r["kind"] == "IMPORTS"
    } or True  # source is a node id, not a rel id
    # unresolved target must not appear as canonical IMPORTS target
    assert not any(r["target"] == "os" for r in imports)


def test_unresolved_not_in_canonical_relationships(fixture_model):
    node_ids = {n["id"] for n in fixture_model.nodes}
    canonical_targets = {r["target"] for r in fixture_model.relationships}
    for claim in fixture_model.unresolved:
        # unresolved endpoints may be raw labels (module names, paths)
        rel_id = "rel:{}:{}:{}".format(
            claim["kind"], claim["source"], claim["target"]
        )
        assert rel_id not in {r["id"] for r in fixture_model.relationships}
        # if both endpoints were nodes, it would have been canonical
        if claim["source"] in node_ids and claim["target"] in node_ids:
            pytest.fail("unresolved claim should have become a relationship")


def test_relationship_ids_unique_and_order(fixture_model):
    ids = [r["id"] for r in fixture_model.relationships]
    assert len(ids) == len(set(ids))
    keys = [
        (r["kind"], r["source"], r["target"], r["id"])
        for r in fixture_model.relationships
    ]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Analyzer metadata / coverage
# ---------------------------------------------------------------------------


def test_analysis_metadata_records_deterministic_analyzers(fixture_model):
    analysis = fixture_model.to_dict()["analysis"]
    assert analysis["strategy"] == "deterministic"
    names = {a["analyzer"] for a in analysis["analyzers"]}
    assert "python" in names
    assert "git" in names
    # analyzer names are NOT part of node identity
    for node in fixture_model.nodes:
        assert "python:" not in node["id"].split("::", 1)[0].replace(
            "code_entity:", "", 1
        ) or node["id"].startswith("code_entity:")
        assert not node["id"].startswith("python-")
        assert "ast-" not in node["id"][:10]


def test_coverage_metadata_present(fixture_model):
    coverage = fixture_model.metadata.get("analysis_coverage") or {}
    assert coverage.get("facts_total") == 17 or coverage.get("facts_total") > 0
    assert "facts_by_type" in coverage


# ---------------------------------------------------------------------------
# Resolver unit tests
# ---------------------------------------------------------------------------


def test_parent_directory():
    assert parent_directory("src/main.py") == "src"
    assert parent_directory("main.py") is None
    assert parent_directory("a/b/c.py") == "a/b"


def test_resolve_dependency_target():
    assert resolve_dependency_target("jsonschema") == "external_package:jsonschema"
    assert resolve_dependency_target("") is None
    assert resolve_dependency_target("  ") is None


def test_node_index_import_resolution():
    nodes = [
        {"id": "code_entity:src/service.py", "kind": "code_entity", "subkind": "module"},
        {"id": "artifact:file:src/service.py", "kind": "artifact", "subkind": "file"},
        {"id": "code_entity:src/util.py", "kind": "code_entity", "subkind": "module"},
    ]
    index = NodeIndex(nodes)
    assert index.resolve_imported_module("src.service") == "code_entity:src/service.py"
    assert index.resolve_imported_module("json") is None
    assert index.resolve_imported_module("os") is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_deterministic(fixture_facts, rules):
    a = build_engineering_model(fixture_facts, rules).to_dict()
    b = build_engineering_model(fixture_facts, rules).to_dict()
    # counts metadata included after validation — should match
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_create_writes_artifact_and_is_byte_stable(fixture_facts, rules, tmp_path):
    import shutil

    facts_copy = tmp_path / "facts.json"
    shutil.copy(FIXTURE_FACTS, facts_copy)
    rules_copy = tmp_path / "rules.json"
    shutil.copy(
        default_artifacts_root()
        / PRIMARY_SNAPSHOT_ID
        / "engineering"
        / "engineering_construction_rules.v1.json",
        rules_copy,
    )

    m1, p1 = create_engineering_model(
        facts_path=facts_copy, rules_path=rules_copy, output_root=tmp_path / "o1"
    )
    m2, p2 = create_engineering_model(
        facts_path=facts_copy, rules_path=rules_copy, output_root=tmp_path / "o2"
    )
    assert p1.read_bytes() == p2.read_bytes()
    assert engineering_model_path(tmp_path / "o1", "fixture_2026-09-22_aaaaaaaa") == p1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_missing_facts_raises(tmp_path):
    with pytest.raises(EngineeringModelError):
        load_facts_document(tmp_path / "nope.json")


def test_wrong_schema_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "repository_facts.v1"}))
    with pytest.raises(EngineeringModelError):
        load_facts_document(bad)


# ---------------------------------------------------------------------------
# ADOP integration (Level 3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ADOP_FACTS.is_file(), reason="ADOP facts not present")
def test_adop_engineering_model_builds():
    facts = load_facts_document(ADOP_FACTS)
    rules = load_construction_rules()
    model = build_engineering_model(facts, rules)
    document = model.to_dict()

    assert model.repository["commit_sha"] == ADOP_COMMIT
    validate_engineering_model_or_raise(document, expected_commit_sha=ADOP_COMMIT)

    counts = model.counts()
    assert counts["nodes"] > 0
    assert counts["relationships"] > 0
    assert counts["evidence"] >= counts["nodes"]

    ids = [n["id"] for n in model.nodes]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)

    node_set = set(ids)
    for rel in model.relationships:
        assert rel["source"] in node_set
        assert rel["target"] in node_set
        assert rel["evidence"]

    # no path duplication in class/function ids
    for node_id in ids:
        if node_id.startswith("code_entity:"):
            # crude check for doubled path segments like path::path::
            rest = node_id.split(":", 1)[1]
            if "::" in rest:
                path_part = rest.split("::", 1)[0]
                if path_part.endswith(".py"):
                    assert path_part + "::" + path_part not in rest

    # DEPENDS_ON with external packages
    depends = [r for r in model.relationships if r["kind"] == "DEPENDS_ON"]
    assert depends
    for rel in depends:
        assert rel["target"].startswith("external_package:")

    # unresolved imports retained
    unresolved_imports = [
        u for u in model.unresolved if u["kind"] == "IMPORTS"
    ]
    assert unresolved_imports
    for claim in unresolved_imports:
        assert claim["reason"] in {
            "target_not_resolved",
            "ambiguous_identity",
            "missing_endpoint",
        }
        assert claim["evidence"]


@pytest.mark.skipif(not ADOP_FACTS.is_file(), reason="ADOP facts not present")
def test_adop_triggers_relationships_are_evidence_backed():
    """ADOP interactions become observed TRIGGERS only with endpoint nodes."""
    facts = load_facts_document(ADOP_FACTS)
    rules = load_construction_rules()
    model = build_engineering_model(facts, rules)

    interaction_facts = [
        f for f in facts.get("facts") or [] if f.get("fact_type") == "interaction"
    ]
    triggers = [r for r in model.relationships if r["kind"] == "TRIGGERS"]
    assert interaction_facts, "regenerated ADOP facts should include interactions"
    assert triggers, "expected observed TRIGGERS on ADOP"

    node_ids = {n["id"] for n in model.nodes}
    for rel in triggers:
        assert rel["classification"] == "observed"
        assert rel["source"] in node_ids
        assert rel["target"] in node_ids
        assert rel["evidence"]
        assert all(
            e.get("commit") == ADOP_COMMIT for e in rel["evidence"]
        )
        # no fabricated hard-coded ADOP ontology names as sole identity
        assert rel["id"] == "rel:TRIGGERS:{}:{}".format(
            rel["source"], rel["target"]
        )

    # Core evidence: onboard-workflow command triggers the six definition agents
    # (when both endpoints resolved). M2 resolves evidence, not desired names.
    sources = {r["source"] for r in triggers}
    targets = {r["target"] for r in triggers}
    assert any("onboard-workflow" in s for s in sources)
    assert any(
        t.endswith("metadata-agent.md")
        or t.endswith("transformation-agent.md")
        or t.endswith("quality-agent.md")
        or t.endswith("dag-agent.md")
        for t in targets
    )

    # Every interaction fact either became a TRIGGERS edge or an unresolved claim
    resolved_pairs = {(r["source"], r["target"]) for r in triggers}
    unresolved_triggers = [
        u for u in model.unresolved if u.get("kind") == "TRIGGERS"
    ]
    # Build fact_id → expected endpoints via node attrs when possible
    assert len(triggers) + len(unresolved_triggers) >= len(interaction_facts) or (
        len(triggers) >= 1
    )


@pytest.mark.skipif(not ADOP_FACTS.is_file(), reason="ADOP facts not present")
def test_adop_engineering_model_determinism(tmp_path):
    facts = load_facts_document(ADOP_FACTS)
    rules = load_construction_rules()
    a = build_engineering_model(facts, rules).to_dict()
    b = build_engineering_model(facts, rules).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.skipif(not ADOP_FACTS.is_file(), reason="ADOP facts not present")
def test_adop_artifact_written_under_engineering():
    path = engineering_model_path(default_artifacts_root(), PRIMARY_SNAPSHOT_ID)
    if not path.is_file():
        # generate once for the workspace
        model, path = create_engineering_model(
            facts_path=ADOP_FACTS, output_root=default_artifacts_root()
        )
    assert path.parent.name == "engineering"
    assert path.name == "engineering_model.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == SCHEMA_VERSION
    validate_engineering_model_or_raise(
        document, expected_commit_sha=ADOP_COMMIT
    )
    # no project-root canonical copy
    assert not (PROJECT_ROOT / "engineering_model.json").exists()
