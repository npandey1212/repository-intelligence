"""M3 construction rulebook contract tests.

Covers schema/structure, node and relationship mappings, evidence rules,
identity determinism, unresolved handling, duplicate-artifact protection,
and dual-run determinism against small synthetic fixtures (not full ADOP).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from repo_intelligence.artifacts import (
    CONSTRUCTION_RULES_FILENAME,
    PRIMARY_SNAPSHOT_ID,
    construction_rules_path,
    default_artifacts_root,
)
from repo_intelligence.construction import (
    ConstructionRulesError,
    ConstructionRulesSchemaError,
    SCHEMA_VERSION,
    code_entity_identity,
    default_matrix_path,
    evidence_only_matches,
    format_identity,
    load_construction_rules,
    node_rules_by_action,
    normalize_entity_key,
    relationship_identity,
    validate_construction_rules_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FACTS = (
    PROJECT_ROOT / "tests" / "fixtures" / "construction" / "repository_facts.json"
)
FIXTURE_SRC = PROJECT_ROOT / "tests" / "fixtures" / "construction" / "src"
ADOP_FACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "main_2026-09-22_ef11dac"
    / "facts"
    / "repository_facts.json"
)

# Frozen M2 vocabulary from the regenerated ADOP artifact (spec §1).
EXPECTED_FACT_COMBOS = {
    ("file", "repository_file"),
    ("directory", "repository_directory"),
    ("entity", "module"),
    ("entity", "class"),
    ("entity", "function"),
    ("import", "import"),
    ("import", "import_from"),
    ("configuration", "file"),
    ("configuration", "json"),
    ("configuration", "toml"),
    ("configuration", "yaml"),
    ("dependency", "dependencies"),
    ("artifact", "agent_definition"),
    ("artifact", "agent_command"),
    ("artifact", "agent_config"),
    ("artifact", "mcp_config"),
    ("artifact", "mcp_documentation"),
    ("artifact", "mcp_server_source"),
    ("artifact", "markdown"),
    ("artifact", "python"),
    ("artifact", "shell"),
    ("documentation", "documentation"),
    ("documentation", "markdown"),
    ("documentation", "readme"),
    ("documentation", "runbook"),
    ("test", "test_directory"),
    ("test", "test_file"),
    ("interaction", "triggers"),
    # M12 agent/tool/MCP knowledge slice
    ("tool", "mcp_tool"),
    ("tool", "built_in_tool"),
    ("mcp_server", "registered"),
    ("interaction", "declares"),
    ("interaction", "exposes"),
    ("interaction", "configures"),
}

EXPECTED_EVIDENCE_ONLY = {
    ("artifact", "mcp_documentation"),
    ("artifact", "markdown"),
    ("artifact", "python"),
    ("artifact", "shell"),
    ("configuration", "file"),
    ("configuration", "json"),
    ("configuration", "toml"),
    ("configuration", "yaml"),
    ("documentation", "documentation"),
    ("documentation", "markdown"),
    ("documentation", "readme"),
    ("documentation", "runbook"),
    ("test", "test_directory"),
    ("test", "test_file"),
}


@pytest.fixture(scope="module")
def rules() -> Dict[str, Any]:
    return load_construction_rules()


@pytest.fixture(scope="module")
def fixture_facts() -> Dict[str, Any]:
    return json.loads(FIXTURE_FACTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Schema / structure
# ---------------------------------------------------------------------------


def test_schema_version_and_required_sections(rules):
    assert rules["schema_version"] == SCHEMA_VERSION
    for section in (
        "input",
        "node_rules",
        "relationship_rules",
        "identity_rules",
        "evidence_rules",
        "unresolved_rules",
        "validation_rules",
        "deferred",
    ):
        assert section in rules
    assert rules["input"]["fact_schema"] == "repository_facts.v2"


def test_matrix_schema_validation_clean():
    document = json.loads(default_matrix_path().read_text(encoding="utf-8"))
    problems = validate_construction_rules_document(document)
    assert problems == []


def test_canonical_matrix_artifact_location():
    """Matrix lives under artifacts/<snapshot_id>/engineering/, not project root."""
    expected = (
        default_artifacts_root()
        / PRIMARY_SNAPSHOT_ID
        / "engineering"
        / CONSTRUCTION_RULES_FILENAME
    )
    assert default_matrix_path() == expected
    assert default_matrix_path().is_file()
    # helper path matches default
    assert (
        construction_rules_path(default_artifacts_root(), PRIMARY_SNAPSHOT_ID)
        == expected
    )
    # no stale project-root copy
    assert not (PROJECT_ROOT / CONSTRUCTION_RULES_FILENAME).exists()
    # under artifacts/
    assert expected.relative_to(default_artifacts_root()).parts[0] == PRIMARY_SNAPSHOT_ID
    assert expected.parent.name == "engineering"


def test_rule_ids_unique(rules):
    ids = [r["rule_id"] for r in rules["node_rules"]]
    ids += [r["rule_id"] for r in rules["relationship_rules"]]
    ids += [r["rule_id"] for r in rules["identity_rules"]]
    assert len(ids) == len(set(ids))


def test_load_rejects_corrupt_document(rules):
    bad = copy.deepcopy(rules)
    bad["schema_version"] = "engineering_construction_rules.v999"
    with pytest.raises(ConstructionRulesSchemaError):
        # simulate load-time validation path
        problems = validate_construction_rules_document(bad)
        assert problems
        raise ConstructionRulesSchemaError("; ".join(problems))


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ConstructionRulesError):
        load_construction_rules(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Node mappings
# ---------------------------------------------------------------------------


def _node_rule(rules, fact_type: str, kind: str) -> Dict[str, Any]:
    for rule in rules["node_rules"]:
        if rule["match"] == {"fact_type": fact_type, "kind": kind}:
            return rule
    raise AssertionError("missing node rule for {}/{}".format(fact_type, kind))


@pytest.mark.parametrize(
    "fact_type,kind,expect_kind,expect_subkind",
    [
        ("file", "repository_file", "artifact", "file"),
        ("directory", "repository_directory", "artifact", "directory"),
        ("entity", "module", "code_entity", "module"),
        ("entity", "class", "code_entity", "class"),
        ("entity", "function", "code_entity", "function"),
    ],
)
def test_core_node_mappings(rules, fact_type, kind, expect_kind, expect_subkind):
    rule = _node_rule(rules, fact_type, kind)
    assert rule["action"] == "CREATE_NODE"
    assert rule["node"] == {"kind": expect_kind, "subkind": expect_subkind}
    assert rule["classification"] == "observed"
    assert rule["requires_evidence"] is True


@pytest.mark.parametrize(
    "kind",
    [
        "agent_definition",
        "agent_command",
        "agent_config",
        "mcp_config",
        "mcp_server_source",
    ],
)
def test_specialized_artifact_node_mappings(rules, kind):
    rule = _node_rule(rules, "artifact", kind)
    assert rule["action"] == "CREATE_NODE"
    assert rule["node"] == {"kind": "artifact", "subkind": kind}
    assert rule["identity"]["template"].startswith("artifact:{}:".format(kind))


def test_no_high_level_architecture_concepts(rules):
    forbidden_substrings = (
        "concept:agent",
        "concept:workflow",
        "concept:mcp-system",
        "service",
        "application",
    )
    for rule in rules["node_rules"]:
        template = rule["identity"]["template"]
        for bad in forbidden_substrings:
            assert bad not in template
        assert rule["node"]["kind"] in {
            "artifact",
            "code_entity",
            "external_package",
            "tool",
            "mcp_server",
        }


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_every_create_node_requires_evidence(rules):
    for rule in node_rules_by_action(rules, "CREATE_NODE"):
        assert rule["requires_evidence"] is True


def test_every_relationship_rule_requires_evidence(rules):
    for rule in rules["relationship_rules"]:
        assert rule["requires_evidence"] is True


def test_evidence_rules_flags(rules):
    evidence = rules["evidence_rules"]
    assert evidence["canonical_node_requires_evidence"] is True
    assert evidence["canonical_relationship_requires_evidence"] is True
    assert evidence["derived_relationship_requires_contributing_facts"] is True
    assert evidence["unresolved_retains_evidence"] is True
    assert evidence["minimum_evidence"] == 1
    assert evidence["propagation"] == "copy_m2_evidence_refs"


def test_evidence_only_pairs_cover_spec_list(rules):
    assert set(evidence_only_matches(rules)) == EXPECTED_EVIDENCE_ONLY


def test_evidence_only_node_rules_action(rules):
    for fact_type, kind in EXPECTED_EVIDENCE_ONLY:
        rule = _node_rule(rules, fact_type, kind)
        assert rule["action"] == "EVIDENCE_ONLY", (fact_type, kind)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def _rel_rule(rules, rule_id: str) -> Dict[str, Any]:
    for rule in rules["relationship_rules"]:
        if rule["rule_id"] == rule_id:
            return rule
    raise AssertionError("missing relationship rule {}".format(rule_id))


def test_import_and_dependency_relationship_mappings(rules):
    assert _rel_rule(rules, "rel.import.import")["relationship"]["kind"] == "IMPORTS"
    assert (
        _rel_rule(rules, "rel.import.import_from")["relationship"]["kind"]
        == "IMPORTS"
    )
    assert (
        _rel_rule(rules, "rel.dependency.dependencies")["relationship"]["kind"]
        == "DEPENDS_ON"
    )
    for rule_id in (
        "rel.import.import",
        "rel.import.import_from",
        "rel.dependency.dependencies",
    ):
        rule = _rel_rule(rules, rule_id)
        assert rule["action"] == "CREATE_RELATIONSHIP_IF_RESOLVED"
        assert rule["classification"] == "observed"
        assert rule["target"].get("on_failure") == "UNRESOLVED"


def test_contains_rules_are_derived(rules):
    contains = [
        r for r in rules["relationship_rules"] if r["relationship"]["kind"] == "CONTAINS"
    ]
    assert contains
    for rule in contains:
        assert rule["classification"] == "derived"
        assert rule["action"] == "CREATE_DERIVED_RELATIONSHIP"
        assert "derive" in rule
        assert rule["requires_evidence"] is True


def test_relationship_vocabulary_limits(rules):
    assert set(rules["relationship_vocabulary"]) == {
        "IMPORTS",
        "DEPENDS_ON",
        "CONTAINS",
        "TRIGGERS",
        # M12 forward path: Agent -> Tool -> MCP Server -> MCP Configuration
        "DECLARES",
        "EXPOSED_BY",
        "CONFIGURED_BY",
    }
    for rule in rules["relationship_rules"]:
        assert rule["relationship"]["kind"] in rules["relationship_vocabulary"]
        assert rule["classification"] in {"observed", "derived"}


# ---------------------------------------------------------------------------
# Unresolved
# ---------------------------------------------------------------------------


def test_unresolved_rules_contract(rules):
    unresolved = rules["unresolved_rules"]
    assert set(unresolved["reasons"]) == {
        "target_not_resolved",
        "ambiguous_identity",
        "missing_endpoint",
        "unsupported_relationship",
    }
    assert unresolved["excluded_from_canonical_relationships"] is True
    assert unresolved["container"] == "engineering.unresolved[]"
    assert set(unresolved["required_fields"]) == {
        "source",
        "target",
        "kind",
        "reason",
        "evidence",
    }


def test_unresolved_import_not_in_canonical_relationships():
    """Resolved local import creates IMPORTS; stdlib import becomes unresolved only."""
    rules = load_construction_rules()
    facts = json.loads(FIXTURE_FACTS.read_text(encoding="utf-8"))["facts"]
    graph = _evaluate_fixture(rules, facts)

    resolved_targets = {
        rel["target"] for rel in graph["relationships"] if rel["kind"] == "IMPORTS"
    }
    # src.service resolves to a module/code entity or file-backed module path
    assert any("service" in t for t in resolved_targets)

    # stdlib 'os' must not appear as a canonical IMPORTS target
    assert not any(t == "external_package:os" or t.endswith("::os") for t in resolved_targets)
    os_unresolved = [
        u
        for u in graph["unresolved"]
        if u["kind"] == "IMPORTS" and u["reason"] == "target_not_resolved"
    ]
    assert os_unresolved
    assert all(u.get("evidence") for u in os_unresolved)
    assert not any(
        u["target"] in {r["target"] for r in graph["relationships"]}
        for u in os_unresolved
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_templates_and_formatting(rules):
    file_rule = _node_rule(rules, "file", "repository_file")
    assert (
        format_identity(
            file_rule["identity"]["template"], {"path": "src/main.py"}
        )
        == "artifact:file:src/main.py"
    )
    module_rule = _node_rule(rules, "entity", "module")
    assert (
        format_identity(module_rule["identity"]["template"], {"path": "src/main.py"})
        == "code_entity:src/main.py"
    )


def test_class_identity_normalizes_path_without_duplication(rules):
    class_rule = _node_rule(rules, "entity", "class")
    # M2 qualified_name already includes path context
    assert (
        format_identity(
            class_rule["identity"]["template"],
            {"path": "src/service.py", "qualified_name": "src/service.py::Service"},
        )
        == "code_entity:src/service.py::Service"
    )
    # bare qualified_name still normalizes correctly
    assert (
        format_identity(
            class_rule["identity"]["template"],
            {"path": "src/service.py", "qualified_name": "Service"},
        )
        == "code_entity:src/service.py::Service"
    )
    assert code_entity_identity("src/service.py", "src/service.py::Service") == (
        "code_entity:src/service.py::Service"
    )


def test_function_identity_normalizes_path_without_duplication(rules):
    function_rule = _node_rule(rules, "entity", "function")
    assert (
        format_identity(
            function_rule["identity"]["template"],
            {"path": "src/service.py", "qualified_name": "src/service.py::run"},
        )
        == "code_entity:src/service.py::run"
    )
    assert (
        format_identity(
            function_rule["identity"]["template"],
            {"path": "src/service.py", "qualified_name": "run"},
        )
        == "code_entity:src/service.py::run"
    )
    # nested method-style qualified name
    assert (
        format_identity(
            function_rule["identity"]["template"],
            {"path": "src/service.py", "qualified_name": "src/service.py::Service.run"},
        )
        == "code_entity:src/service.py::Service.run"
    )


@pytest.mark.parametrize(
    "path,qualified_name,expected",
    [
        ("src/service.py", "src/service.py::Service", "src/service.py::Service"),
        ("src/service.py", "src/service.py::run", "src/service.py::run"),
        ("src/service.py", "Service", "src/service.py::Service"),
        ("src/service.py", "src/service.py::Service.run", "src/service.py::Service.run"),
        ("src/service.py", "src/service.py", "src/service.py"),
    ],
)
def test_normalize_entity_key_cases(path, qualified_name, expected):
    assert normalize_entity_key(path, qualified_name) == expected


def test_class_function_identity_never_duplicates_path(rules):
    cases = [
        ("entity", "class", "src/service.py", "src/service.py::Service"),
        ("entity", "function", "src/service.py", "src/service.py::run"),
        ("entity", "function", "src/service.py", "run"),
    ]
    for fact_type, kind, path, qn in cases:
        rule = _node_rule(rules, fact_type, kind)
        node_id = format_identity(
            rule["identity"]["template"], {"path": path, "qualified_name": qn}
        )
        # path segment must appear exactly once as the file portion
        assert node_id.count(path) == 1, node_id
        assert path + "::" + path not in node_id
        assert "::src/service.py::src/" not in node_id
        assert "src/service.py::src/service.py" not in node_id


def test_class_function_identity_deterministic():
    attrs = {"path": "src/service.py", "qualified_name": "src/service.py::Service"}
    first = code_entity_identity(attrs["path"], attrs["qualified_name"])
    second = code_entity_identity(attrs["path"], attrs["qualified_name"])
    assert first == second == "code_entity:src/service.py::Service"
    # independent of dict order / template fill
    assert (
        format_identity("code_entity:{entity_key}", dict(reversed(list(attrs.items()))))
        == first
    )


def test_identity_stable_and_paths_do_not_collide():
    rules = load_construction_rules()
    a = format_identity("artifact:file:{path}", {"path": "src/a.py"})
    b = format_identity("artifact:file:{path}", {"path": "src/a.py"})
    c = format_identity("artifact:file:{path}", {"path": "src/b.py"})
    assert a == b
    assert a != c
    assert relationship_identity("IMPORTS", a, c) == relationship_identity(
        "IMPORTS", a, c
    )
    assert relationship_identity("IMPORTS", a, c) != relationship_identity(
        "CONTAINS", a, c
    )


def test_identity_rules_forbid_analyzer_and_random(rules):
    for rule in rules["identity_rules"]:
        forbidden = set(rule["forbidden"])
        assert "analyzer" in forbidden
        assert "uuid" in forbidden
        assert "random" in forbidden
        assert "timestamp" in forbidden
        template = rule["template"]
        assert "analyzer" not in template
        assert "uuid" not in template.lower()


def test_module_identity_has_no_analyzer_prefix(rules):
    template = _node_rule(rules, "entity", "module")["identity"]["template"]
    assert template == "code_entity:{path}"
    assert "python-module" not in template
    assert "ast-module" not in template
    assert "tree-sitter" not in template


def test_no_analyzer_prefixes_in_class_function_ids(rules):
    for fact_type, kind in (("entity", "class"), ("entity", "function")):
        rule = _node_rule(rules, fact_type, kind)
        node_id = format_identity(
            rule["identity"]["template"],
            {"path": "src/service.py", "qualified_name": "src/service.py::Service"},
        )
        for bad in ("python:", "ast:", "tree-sitter:", "repository-intelligence:"):
            assert bad not in node_id


# ---------------------------------------------------------------------------
# Duplicate physical artifact protection
# ---------------------------------------------------------------------------


def test_multiple_classifications_do_not_duplicate_file_nodes():
    rules = load_construction_rules()
    facts = json.loads(FIXTURE_FACTS.read_text(encoding="utf-8"))["facts"]
    graph = _evaluate_fixture(rules, facts)
    file_ids = [n["id"] for n in graph["nodes"] if n["id"].startswith("artifact:file:")]
    assert len(file_ids) == len(set(file_ids))
    # config.yaml has file + configuration facts → exactly one file node
    assert file_ids.count("artifact:file:config.yaml") == 1
    # configuration evidence-only must not emit artifact:configuration node here
    assert "artifact:configuration:config.yaml" not in {
        n["id"] for n in graph["nodes"]
    }
    # documentation/test evidence-only must not emit their own nodes
    assert "artifact:documentation:README.md" not in {n["id"] for n in graph["nodes"]}
    assert "artifact:test:src/test_service.py" not in {n["id"] for n in graph["nodes"]}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_rule_evaluation_deterministic_across_runs():
    rules = load_construction_rules()
    facts = json.loads(FIXTURE_FACTS.read_text(encoding="utf-8"))["facts"]
    first = _evaluate_fixture(rules, facts)
    second = _evaluate_fixture(rules, copy.deepcopy(facts))
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_fixture_source_files_exist():
    assert (FIXTURE_SRC / "main.py").is_file()
    assert (FIXTURE_SRC / "service.py").is_file()


# ---------------------------------------------------------------------------
# Optional ADOP integration benchmark (skipped when artifact absent)
# ---------------------------------------------------------------------------


def test_adop_fact_combos_match_expected_matrix_coverage():
    if not ADOP_FACTS.is_file():
        pytest.skip("ADOP facts artifact not present")
    rules = load_construction_rules()
    document = json.loads(ADOP_FACTS.read_text(encoding="utf-8"))
    combos = {(f["fact_type"], f["kind"]) for f in document["facts"]}
    assert combos == EXPECTED_FACT_COMBOS

    # Every combo is either CREATE_NODE, EVIDENCE_ONLY, or relationship-driven.
    node_matches = {
        (r["match"]["fact_type"], r["match"]["kind"]): r["action"]
        for r in rules["node_rules"]
    }
    rel_matches = {
        (r["match"]["fact_type"], r["match"]["kind"])
        for r in rules["relationship_rules"]
    }
    evidence_only = set(evidence_only_matches(rules))
    for combo in combos:
        action = node_matches.get(combo)
        if action == "CREATE_NODE":
            continue
        if combo in rel_matches:
            # Relationship-driven (import, dependency): optional evidence-only node rule.
            if action == "EVIDENCE_ONLY":
                # May or may not also be listed under evidence_rules.evidence_only.
                pass
            continue
        assert action == "EVIDENCE_ONLY", combo
        assert combo in evidence_only, combo


# ---------------------------------------------------------------------------
# Minimal fixture evaluator (contract-only; not the EngineeringModel builder)
# ---------------------------------------------------------------------------


def _parent_dir(path: str) -> str:
    if "/" not in path:
        return ""
    return path.rsplit("/", 1)[0]


def _evaluate_fixture(
    rules: Dict[str, Any], facts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply matrix node/relationship rules to fixture facts.

    Intentionally small and deterministic — enough to test the rulebook
    contract. Not a production EngineeringModel builder.
    """
    create_node_rules = node_rules_by_action(rules, "CREATE_NODE")
    evidence_only_pairs = set(evidence_only_matches(rules))

    nodes: Dict[str, Dict[str, Any]] = {}
    evidence_by_identity: Dict[str, List[Dict[str, Any]]] = {}

    def add_evidence(node_id: str, evidence: List[Dict[str, Any]]) -> None:
        bucket = evidence_by_identity.setdefault(node_id, [])
        for item in evidence:
            if item not in bucket:
                bucket.append(item)

    for fact in facts:
        key = (fact["fact_type"], fact["kind"])
        if key in evidence_only_pairs:
            continue
        for rule in create_node_rules:
            if (rule["match"]["fact_type"], rule["match"]["kind"]) != key:
                continue
            attrs = fact.get("attributes") or {}
            identity = format_identity(rule["identity"]["template"], attrs)
            if identity not in nodes:
                nodes[identity] = {
                    "id": identity,
                    "kind": rule["node"]["kind"],
                    "subkind": rule["node"]["subkind"],
                    "classification": rule["classification"],
                    "evidence": [],
                }
            for item in fact.get("evidence") or []:
                if item not in nodes[identity]["evidence"]:
                    nodes[identity]["evidence"].append(item)
            add_evidence(identity, fact.get("evidence") or [])
            break

    # Index modules/files for import resolution
    module_paths = set()
    for node_id, node in nodes.items():
        if node["kind"] == "code_entity" and node["subkind"] == "module":
            # code_entity:{path}
            module_paths.add(node_id.split(":", 1)[1])
    file_paths = {
        node_id.split(":", 2)[2]
        for node_id, node in nodes.items()
        if node["kind"] == "artifact" and node["subkind"] == "file"
    }

    relationships: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    seen_rel = set()

    def resolve_import_target(module: str) -> Optional[str]:
        if not module:
            return None
        dotted_path = module.replace(".", "/") + ".py"
        if dotted_path in module_paths or dotted_path in file_paths:
            if dotted_path in module_paths:
                return "code_entity:{}".format(dotted_path)
            return "artifact:file:{}".format(dotted_path)
        if module in module_paths:
            return "code_entity:{}".format(module)
        # suffix match against known module paths
        matches = [
            p
            for p in module_paths
            if p == module or p.endswith("/" + module.replace(".", "/") + ".py")
        ]
        if len(matches) == 1:
            return "code_entity:{}".format(matches[0])
        if module in file_paths:
            return "artifact:file:{}".format(module)
        return None

    for fact in facts:
        fact_type, kind = fact["fact_type"], fact["kind"]
        attrs = fact.get("attributes") or {}
        evidence = fact.get("evidence") or []

        if fact_type == "import":
            rule = _rel_rule(
                rules,
                "rel.import.import" if kind == "import" else "rel.import.import_from",
            )
            source_path = attrs.get("path") or ""
            source_id = (
                "artifact:file:{}".format(source_path) if source_path in file_paths else None
            )
            target_id = resolve_import_target(attrs.get("module") or "")
            if source_id is None or target_id is None:
                unresolved.append(
                    {
                        "source": source_id or source_path,
                        "target": target_id or attrs.get("module") or "",
                        "kind": "IMPORTS",
                        "reason": "missing_endpoint"
                        if source_id is None
                        else "target_not_resolved",
                        "evidence": evidence,
                    }
                )
                continue
            rel_id = relationship_identity("IMPORTS", source_id, target_id)
            if rel_id in seen_rel:
                continue
            seen_rel.add(rel_id)
            relationships.append(
                {
                    "id": rel_id,
                    "kind": "IMPORTS",
                    "source": source_id,
                    "target": target_id,
                    "classification": "observed",
                    "evidence": evidence,
                }
            )

        elif fact_type == "dependency" and kind == "dependencies":
            rule = _rel_rule(rules, "rel.dependency.dependencies")
            source_path = attrs.get("source") or ""
            name = attrs.get("name") or ""
            source_id = (
                "artifact:file:{}".format(source_path) if source_path in file_paths else None
            )
            if source_id is None or not name:
                unresolved.append(
                    {
                        "source": source_id or source_path,
                        "target": name,
                        "kind": "DEPENDS_ON",
                        "reason": "missing_endpoint"
                        if source_id is None
                        else "target_not_resolved",
                        "evidence": evidence,
                    }
                )
                continue
            target_id = "external_package:{}".format(name)
            rel_id = relationship_identity("DEPENDS_ON", source_id, target_id)
            if rel_id in seen_rel:
                continue
            seen_rel.add(rel_id)
            if target_id not in nodes:
                nodes[target_id] = {
                    "id": target_id,
                    "kind": "external_package",
                    "subkind": "package",
                    "classification": "observed",
                    "evidence": list(evidence),
                }
            relationships.append(
                {
                    "id": rel_id,
                    "kind": "DEPENDS_ON",
                    "source": source_id,
                    "target": target_id,
                    "classification": "observed",
                    "evidence": evidence,
                }
            )

    # Derived CONTAINS: directory→file, file→module/class/function, module→class/function
    dir_ids = {
        n["id"]: n
        for n in nodes.values()
        if n["kind"] == "artifact" and n["subkind"] == "directory"
    }
    for file_id, node in list(nodes.items()):
        if not (node["kind"] == "artifact" and node["subkind"] == "file"):
            continue
        path = file_id.split(":", 2)[2]
        parent = _parent_dir(path)
        parent_id = "artifact:directory:{}".format(parent)
        if parent_id in dir_ids and parent != "":
            rel_id = relationship_identity("CONTAINS", parent_id, file_id)
            if rel_id not in seen_rel:
                seen_rel.add(rel_id)
                relationships.append(
                    {
                        "id": rel_id,
                        "kind": "CONTAINS",
                        "source": parent_id,
                        "target": file_id,
                        "classification": "derived",
                        "evidence": list(node["evidence"]),
                    }
                )
        # file → module / class / function via path equality
        for other_id, other in list(nodes.items()):
            if other_id == file_id:
                continue
            if other["kind"] != "code_entity":
                continue
            # derive path from identity
            if other["subkind"] == "module":
                other_path = other_id.split(":", 1)[1]
            else:
                # code_entity:{path}::{qualified_name} → path is before '::' after first segment
                rest = other_id.split(":", 1)[1]
                other_path = rest.split("::", 1)[0]
            if other_path == path:
                rel_id = relationship_identity("CONTAINS", file_id, other_id)
                if rel_id not in seen_rel:
                    seen_rel.add(rel_id)
                    relationships.append(
                        {
                            "id": rel_id,
                            "kind": "CONTAINS",
                            "source": file_id,
                            "target": other_id,
                            "classification": "derived",
                            "evidence": list(other["evidence"]),
                        }
                    )
            if other["subkind"] == "module":
                continue

    # module → class/function (same path)
    modules = {
        n["id"]: n["id"].split(":", 1)[1]
        for n in nodes.values()
        if n["kind"] == "code_entity" and n["subkind"] == "module"
    }
    for other_id, other in list(nodes.items()):
        if other["kind"] != "code_entity" or other["subkind"] == "module":
            continue
        rest = other_id.split(":", 1)[1]
        other_path = rest.split("::", 1)[0]
        for module_id, module_path in modules.items():
            if module_path == other_path:
                rel_id = relationship_identity("CONTAINS", module_id, other_id)
                if rel_id not in seen_rel:
                    seen_rel.add(rel_id)
                    relationships.append(
                        {
                            "id": rel_id,
                            "kind": "CONTAINS",
                            "source": module_id,
                            "target": other_id,
                            "classification": "derived",
                            "evidence": list(other["evidence"]),
                        }
                    )

    # Canonical nodes must have evidence
    for node in nodes.values():
        assert node["evidence"], node
    for rel in relationships:
        assert rel["evidence"], rel

    # Sort deterministically
    ordered_nodes = [nodes[k] for k in sorted(nodes)]
    ordered_rels = sorted(
        relationships, key=lambda r: (r["kind"], r["source"], r["target"], r["id"])
    )
    ordered_unresolved = sorted(
        unresolved, key=lambda u: (u["kind"], str(u["source"]), str(u["target"]), u["reason"])
    )
    return {
        "nodes": ordered_nodes,
        "relationships": ordered_rels,
        "unresolved": ordered_unresolved,
    }
