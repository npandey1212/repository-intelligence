"""ArchitectureContext (architecture_context.v1) tests.

Levels:
- unit: repo-level template contract + generic synthetic M1–M4-shaped
  documents (repository-agnostic)
- integration: pinned artifacts (skipped when absent)

Covers the template contract (existence, structure, schema version),
template-driven generation, M1–M4 projection into sections, Markdown
consistency, artifact layout, determinism, absence semantics (no
fabrication), and repository-agnostic generation.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from repo_intelligence.context import (
    ARCHITECTURE_CONTEXT_SCHEMA_VERSION,
    CONTEXT_SECTION_KEYS,
    CONTEXT_TEMPLATE_PATH,
    SECTION_HEADINGS,
    ArchitectureContextError,
    build_architecture_context,
    build_architecture_context_document,
    create_architecture_context,
    default_context_markdown_path,
    default_context_path,
    format_source_line,
    load_context_template,
    render_architecture_context_markdown,
    validate_architecture_context_document,
    validate_architecture_context_or_raise,
    validate_context_document,
    validate_context_document_shape,
    validate_context_or_raise,
    validate_markdown_context_consistency,
    validate_markdown_context_or_raise,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SRC_DIR = PROJECT_ROOT / "src" / "repo_intelligence" / "context"


def _discover_pinned_run_id() -> str:
    """Run id of the pinned run, discovered from the artifact layout."""
    root = PROJECT_ROOT / "artifacts"
    if not root.is_dir():
        return ""
    candidates = sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir()
        and (child / "facts" / "repository_facts.json").is_file()
        and (child / "architecture" / "architecture_model.json").is_file()
    )
    return candidates[0] if candidates else ""


PINNED_RUN_ID = _discover_pinned_run_id()
PINNED_ARCH_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / PINNED_RUN_ID
    / "architecture"
    / "architecture_model.json"
)
PINNED_FACTS = (
    PROJECT_ROOT
    / "artifacts"
    / PINNED_RUN_ID
    / "facts"
    / "repository_facts.json"
)

# The exact template contract supplied by the milestone.
EXPECTED_TEMPLATE = {
    "schema_version": "architecture_context.v1",
    "repository": {
        "name": "",
        "description": "",
        "purpose": "",
        "project_type": "",
        "primary_domains": [],
    },
    "structure": {
        "overview": "",
        "key_directories": [],
        "key_files": [],
        "important_artifacts": [],
    },
    "technology": {
        "languages": [],
        "frameworks": [],
        "libraries": [],
        "infrastructure": [],
        "external_services": [],
    },
    "architecture": {
        "overview": "",
        "components": [],
        "relationships": [],
        "boundaries": [],
    },
    "workflows": {"overview": "", "workflows": []},
    "interfaces": {
        "overview": "",
        "apis": [],
        "commands": [],
        "events": [],
        "external_interfaces": [],
    },
    "operations": {
        "build": [],
        "test": [],
        "run": [],
        "deploy": [],
    },
    "documentation": {"overview": "", "key_documents": []},
    "evidence": {
        "sources": [],
        "known_unknowns": [],
        "analysis_limitations": [],
    },
}

FORBIDDEN_SOURCE_TERMS = (
    "data-onboarding",
    "Data Onboarding",
    "ADOP",
    "adop",
    "metadata-agent",
    "dag-agent",
    "iac-agent",
    "ontology-agent",
    "quality-agent",
    "transformation-agent",
    "onboard-workflow",
    "agentcore",
    "sample-Agentic-Ai-Data-Operations",
    "aws-samples",
    "aws",
    "s3",
    "Airflow",
    "airflow",
    "Glue",
    "MCP",
    "mcp",
    ".claude",
    "claude",
    "Claude",
    "agents",
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_architecture_document():
    """Repository-A-shaped M4 document (ComponentA → WorkflowA)."""
    commit = "c" * 40
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {
            "name": "sample-repo-alpha",
            "url": "https://example.invalid/alpha.git",
            "commit_sha": commit,
        },
        "analysis": {"strategy": "deterministic", "analyzers": []},
        "engineering": {
            "nodes": [
                {
                    "id": "artifact:file:src/service.py",
                    "kind": "artifact",
                    "subkind": "file",
                    "attributes": {"path": "src/service.py"},
                    "evidence": [
                        {
                            "path": "src/service.py",
                            "artifact_type": "file",
                            "commit": commit,
                        }
                    ],
                },
                {
                    "id": "external_package:libalpha",
                    "kind": "external_package",
                    "subkind": "package",
                    "attributes": {"name": "libalpha"},
                    "evidence": [
                        {
                            "path": "pyproject.toml",
                            "artifact_type": "file",
                            "commit": commit,
                        }
                    ],
                },
            ],
            "relationships": [],
            "unresolved": [],
            "metadata": {},
        },
        "architecture": {
            "nodes": [
                {
                    "id": "concept:component:component-a",
                    "kind": "concept",
                    "subkind": "component",
                    "attributes": {"name": "ComponentA"},
                    "classification": "derived",
                    "evidence": [
                        {
                            "path": "docs/component_a.md",
                            "artifact_type": "file",
                            "commit": commit,
                        }
                    ],
                },
                {
                    "id": "concept:component:workflow-a",
                    "kind": "concept",
                    "subkind": "component",
                    "attributes": {"name": "WorkflowA"},
                    "classification": "derived",
                    "evidence": [
                        {
                            "path": "docs/workflow_a.md",
                            "artifact_type": "file",
                            "commit": commit,
                        }
                    ],
                },
            ],
            "relationships": [
                {
                    "id": "rel:TRIGGERS:concept:component:component-a:"
                    "concept:component:workflow-a",
                    "kind": "TRIGGERS",
                    "source": "concept:component:component-a",
                    "target": "concept:component:workflow-a",
                    "classification": "derived",
                    "evidence": [
                        {
                            "path": "docs/component_a.md",
                            "artifact_type": "file",
                            "commit": commit,
                        }
                    ],
                }
            ],
            "unresolved": [],
            "metadata": {},
        },
        "mappings": [
            {
                "id": "map:CONTRIBUTES_TO:artifact:file:src/service.py:"
                "concept:component:component-a",
                "kind": "CONTRIBUTES_TO",
                "source": "artifact:file:src/service.py",
                "target": "concept:component:component-a",
                "classification": "derived",
                "evidence": [],
            }
        ],
        "metadata": {"counts": {"unresolved": 0}},
    }


def _synthetic_facts_document():
    commit = "c" * 40
    return {
        "schema_version": "repository_facts.v2",
        "snapshot": {
            "snapshot_id": "synthetic_alpha",
            "commit_sha": commit,
            "repository_url": "https://example.invalid/alpha.git",
        },
        "repository": {
            "name": "sample-repo-alpha",
            "url": "https://example.invalid/alpha.git",
        },
        "facts": [
            {
                "fact_id": "fact:directory:packages",
                "fact_type": "directory",
                "kind": "repository_directory",
                "source": "filesystem",
                "attributes": {"path": "packages"},
                "evidence": [
                    {
                        "path": "packages",
                        "artifact_type": "directory",
                        "commit": commit,
                    }
                ],
            },
            {
                "fact_id": "fact:directory:packages/core",
                "fact_type": "directory",
                "kind": "repository_directory",
                "source": "filesystem",
                "attributes": {"path": "packages/core"},
                "evidence": [
                    {
                        "path": "packages/core",
                        "artifact_type": "directory",
                        "commit": commit,
                    }
                ],
            },
            {
                "fact_id": "fact:documentation:README.md",
                "fact_type": "documentation",
                "kind": "readme",
                "source": "documentation",
                "attributes": {"path": "README.md", "type": "readme"},
                "evidence": [
                    {
                        "path": "README.md",
                        "artifact_type": "documentation",
                        "commit": commit,
                    }
                ],
            },
            {
                "fact_id": "fact:dependency:pyproject.toml:dependencies:"
                "libalpha",
                "fact_type": "dependency",
                "kind": "dependency",
                "source": "dependencies",
                "attributes": {
                    "name": "libalpha",
                    "source": "pyproject.toml",
                    "section": "dependencies",
                },
                "evidence": [
                    {
                        "path": "pyproject.toml",
                        "artifact_type": "file",
                        "commit": commit,
                    }
                ],
            },
            {
                "fact_id": "fact:test:packages/core/test_service.py",
                "fact_type": "test",
                "kind": "test_file",
                "source": "tests",
                "attributes": {"path": "packages/core/test_service.py"},
                "evidence": [
                    {
                        "path": "packages/core/test_service.py",
                        "artifact_type": "test",
                        "commit": commit,
                    }
                ],
            },
        ],
        "analyzers": [],
    }


def _build_alpha_context(template=None):
    return build_architecture_context_document(
        _synthetic_architecture_document(),
        _synthetic_facts_document(),
        template=template,
    )


@pytest.fixture(scope="module")
def alpha_context():
    return _build_alpha_context()


@pytest.fixture(scope="module")
def alpha_markdown(alpha_context):
    return render_architecture_context_markdown(alpha_context)


# ---------------------------------------------------------------------------
# Contract: contract/architecture_context_template.json
# ---------------------------------------------------------------------------


def test_contract_template_exists_and_is_valid_json():
    assert CONTEXT_TEMPLATE_PATH.is_file()
    assert CONTEXT_TEMPLATE_PATH == (
        PROJECT_ROOT / "contract" / "architecture_context_template.json"
    )
    document = json.loads(CONTEXT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)


def test_contract_template_schema_version():
    document = _load(CONTEXT_TEMPLATE_PATH)
    assert document["schema_version"] == "architecture_context.v1"
    assert document["schema_version"] == ARCHITECTURE_CONTEXT_SCHEMA_VERSION


def test_contract_template_structure_matches_spec():
    document = _load(CONTEXT_TEMPLATE_PATH)
    assert list(document.keys()) == list(EXPECTED_TEMPLATE.keys())
    assert document == EXPECTED_TEMPLATE
    # no unexpected fields anywhere (the approved contract is exhaustive)
    assert "overview" not in document["technology"]
    assert sum(
        len(section)
        for key, section in document.items()
        if key != "schema_version"
    ) == 34
    for section, expected in EXPECTED_TEMPLATE.items():
        if not isinstance(expected, dict):
            continue
        assert list(document[section].keys()) == list(expected.keys()), section


def test_contract_template_contains_no_domain_values():
    document = _load(CONTEXT_TEMPLATE_PATH)
    for key, value in document.items():
        if key == "schema_version":
            continue
        for field, section_value in value.items():
            if isinstance(section_value, str):
                assert section_value == ""
            else:
                assert section_value == []
    text = CONTEXT_TEMPLATE_PATH.read_text(encoding="utf-8")
    for term in FORBIDDEN_SOURCE_TERMS:
        assert term not in text, term


def test_contract_template_loads_through_generator_contract():
    template = load_context_template()
    assert template == EXPECTED_TEMPLATE
    with pytest.raises(ArchitectureContextError):
        load_context_template(Path("/nonexistent/template.json"))


# ---------------------------------------------------------------------------
# Generator consumes the contract
# ---------------------------------------------------------------------------


def test_generator_consumes_contract_template():
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    template = load_context_template()
    context = build_architecture_context_document(
        arch, facts, template=template
    )
    # document identity is exactly the template identity
    assert list(context.keys()) == list(CONTEXT_SECTION_KEYS)
    assert list(context.keys()) == list(template.keys())
    for section in template:
        if section == "schema_version":
            continue
        assert list(context[section].keys()) == list(template[section].keys())
    # the template itself is never mutated
    assert template == load_context_template()
    # the template is an explicit evidence source
    sources = context["evidence"]["sources"]
    assert sources[-1]["path"] == "contract/architecture_context_template.json"
    assert (
        format_source_line(sources[-1])
        == "Architecture context template: "
        "contract/architecture_context_template.json"
    )
    # validation succeeds against the contract
    assert validate_context_document_shape(context, template) == []
    assert validate_context_document(context, arch, facts) == []


def test_custom_template_drives_output(tmp_path):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    template_document = _load(CONTEXT_TEMPLATE_PATH)
    template_document["structure"]["custom_notes"] = ""
    custom_path = tmp_path / "architecture_context_template.json"
    custom_path.write_text(
        json.dumps(template_document, indent=2) + "\n", encoding="utf-8"
    )
    custom_template = load_context_template(custom_path)

    context = build_architecture_context_document(
        arch, facts, template=custom_template
    )
    assert context["structure"]["custom_notes"] == ""
    markdown = render_architecture_context_markdown(context)
    assert "- Custom notes: _not established_" in markdown
    # matches the custom template, and only the custom template
    assert validate_context_document_shape(context, custom_template) == []
    assert validate_context_document_shape(context) != []
    assert validate_markdown_context_consistency(markdown, context) == []


def test_render_rejects_documents_from_other_contracts():
    with pytest.raises(ArchitectureContextError):
        render_architecture_context_markdown(
            _synthetic_architecture_document()
        )
    with pytest.raises(ArchitectureContextError):
        render_architecture_context_markdown({"schema_version": "x"})


# ---------------------------------------------------------------------------
# M1–M4 projection into template sections
# ---------------------------------------------------------------------------


def test_sections_project_m1m4(alpha_context):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    context = alpha_context

    repository = context["repository"]
    assert repository["name"] == "sample-repo-alpha"

    structure = context["structure"]
    assert structure["key_directories"] == ["packages"]
    assert structure["key_files"] == ["README.md"]
    assert structure["important_artifacts"] == ["file: 1 item(s)"]
    assert structure["overview"]

    technology = context["technology"]
    assert technology["libraries"] == ["libalpha"]
    assert technology["languages"] == []  # no code entities in the fixture

    node_ids = {node["id"] for node in arch["architecture"]["nodes"]}
    component_ids = {
        component["id"] for component in context["architecture"]["components"]
    }
    assert component_ids == node_ids
    rel_ids = {
        rel["id"] for rel in arch["architecture"]["relationships"]
    }
    assert {
        rel["id"] for rel in context["architecture"]["relationships"]
    } == rel_ids
    assert context["architecture"]["overview"]

    workflows = context["workflows"]["workflows"]
    assert len(workflows) == 1
    assert workflows[0]["source_component_id"] == (
        "concept:component:component-a"
    )
    assert workflows[0]["participant_names"] == ["WorkflowA"]
    assert workflows[0]["relationship_ids"] == sorted(rel_ids)

    assert context["documentation"]["key_documents"] == ["README.md"]
    sources = context["evidence"]["sources"]
    assert sources[0]["commit_sha"] == "c" * 40
    assert sources[0]["snapshot_id"] == "synthetic_alpha"
    assert sources[-1]["path"] == "contract/architecture_context_template.json"
    assert (
        "Test files observed as facts: 1."
        in context["evidence"]["known_unknowns"]
    )
    assert (
        "Engineering -> architecture mappings recorded in M4: 1."
        in context["evidence"]["analysis_limitations"]
    )
    assert validate_context_document(context, arch, facts) == []


def test_markdown_has_required_sections(alpha_context, alpha_markdown):
    assert alpha_markdown.startswith("# Architecture Context\n")
    assert (
        "- Contract: `architecture_context.v1`" in alpha_markdown
    )
    headings = [
        line[3:].strip()
        for line in alpha_markdown.splitlines()
        if line.startswith("## ")
    ]
    assert headings == list(SECTION_HEADINGS)
    for heading in SECTION_HEADINGS:
        assert alpha_markdown.count("\n## {}\n".format(heading)) == 1
    # names come from the inputs, not from the generator
    assert "ComponentA" in alpha_markdown
    assert "WorkflowA" in alpha_markdown
    assert "sample-repo-alpha" in alpha_markdown
    assert "- libalpha" in alpha_markdown
    assert "- packages" in alpha_markdown
    assert "- README.md" in alpha_markdown


def test_markdown_is_rendered_from_document_only(alpha_context, alpha_markdown):
    assert validate_markdown_context_consistency(alpha_markdown, alpha_context) == []
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    assert (
        validate_architecture_context_document(alpha_markdown, arch, facts)
        == []
    )


# ---------------------------------------------------------------------------
# Absence semantics: missing information is never fabricated
# ---------------------------------------------------------------------------


def test_missing_information_is_not_fabricated(alpha_context, alpha_markdown):
    repository = alpha_context["repository"]
    assert repository["description"] == ""
    assert repository["purpose"] == ""
    assert repository["project_type"] == ""
    assert repository["primary_domains"] == []
    for field in ("frameworks", "infrastructure", "external_services"):
        assert alpha_context["technology"][field] == []
    assert alpha_context["architecture"]["boundaries"] == []
    for field in ("apis", "commands", "events", "external_interfaces"):
        assert alpha_context["interfaces"][field] == []
    for field in ("build", "test", "run", "deploy"):
        assert alpha_context["operations"][field] == []

    for marker in (
        "- Description: _not established_",
        "- Purpose: _not established_",
        "- Project type: _not established_",
        "- Primary domains: _none recorded_",
        "- Frameworks: _none recorded_",
        "- Boundaries: _none recorded_",
        "- APIs: _none recorded_",
        "- Build: _none recorded_",
    ):
        assert marker in alpha_markdown, marker

    assert validate_context_document(
        alpha_context,
        _synthetic_architecture_document(),
        _synthetic_facts_document(),
    ) == []


def test_document_equality_checks_reject_fabrication(alpha_context):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()

    bad = copy.deepcopy(alpha_context)
    bad["interfaces"]["apis"].append("https://invented.example/api")
    problems = validate_context_document(bad, arch, facts)
    assert any("interfaces" in p for p in problems)

    bad2 = copy.deepcopy(alpha_context)
    bad2["repository"]["purpose"] = "invented purpose"
    problems = validate_context_document(bad2, arch, facts)
    assert any("purpose" in p for p in problems)

    bad3 = copy.deepcopy(alpha_context)
    bad3["technology"]["frameworks"] = ["invented-framework"]
    problems = validate_context_document(bad3, arch, facts)
    assert any("frameworks" in p for p in problems)


# ---------------------------------------------------------------------------
# Document vs M1–M4 consistency
# ---------------------------------------------------------------------------


def test_context_document_identity(alpha_context):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    context = alpha_context
    assert context["repository"]["name"] == "sample-repo-alpha"
    assert validate_context_document(context, arch, facts) == []

    bad = copy.deepcopy(context)
    bad["repository"]["name"] = "other-repo"
    problems = validate_context_document(bad, arch, facts)
    assert any("repository_name" in p for p in problems)

    bad2 = copy.deepcopy(context)
    bad2["evidence"]["sources"][0]["commit_sha"] = "d" * 40
    problems = validate_context_document(bad2, arch, facts)
    assert any("commit" in p for p in problems)


def test_context_document_evidence_integrity(alpha_context):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    assert validate_context_document(alpha_context, arch, facts) == []

    bad = copy.deepcopy(alpha_context)
    bad["architecture"]["components"][0]["evidence"][0]["path"] = (
        "docs/invented.md"
    )
    problems = validate_context_document(bad, arch, facts)
    assert any("evidence" in p for p in problems)


def test_context_document_contains_no_unsupported_relationships(alpha_context):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    assert validate_context_document(alpha_context, arch, facts) == []

    bad = copy.deepcopy(alpha_context)
    bad["architecture"]["relationships"].append(
        {
            "id": "rel:TRIGGERS:concept:component:workflow-a:"
            "concept:component:component-a",
            "kind": "TRIGGERS",
            "source": "concept:component:workflow-a",
            "target": "concept:component:component-a",
            "classification": "derived",
            "evidence": [],
        }
    )
    problems = validate_context_document(bad, arch, facts)
    assert any(
        "invented_architecture_relationship" in p
        or "relationship_set_mismatch" in p
        for p in problems
    )


def test_unresolved_claims_are_counted_explicitly():
    arch = copy.deepcopy(_synthetic_architecture_document())
    facts = _synthetic_facts_document()
    arch["architecture"]["unresolved"].append(
        {
            "kind": "TRIGGERS",
            "source": "concept:component:component-a",
            "target": "concept:component:unknown",
            "reason": "missing_target_evidence",
        }
    )
    context = build_architecture_context_document(arch, facts)
    known_unknowns = context["evidence"]["known_unknowns"]
    assert "Unresolved architecture claims reported by M4: 1." in known_unknowns
    assert any(
        entry.startswith(
            "Unresolved claim: TRIGGERS concept:component:component-a -> "
            "concept:component:unknown (missing_target_evidence)"
        )
        for entry in known_unknowns
    )
    assert validate_context_document(context, arch, facts) == []
    markdown = render_architecture_context_markdown(context)
    assert "- Unresolved architecture claims reported by M4: 1." in markdown
    assert validate_markdown_context_consistency(markdown, context) == []

    # the count line must not be silently dropped
    broken = copy.deepcopy(context)
    broken["evidence"]["known_unknowns"].remove(
        "Unresolved architecture claims reported by M4: 1."
    )
    problems = validate_context_document(broken, arch, facts)
    assert any("known_unknown_missing" in p for p in problems)

    # invented claim lines are rejected
    fabricated = copy.deepcopy(context)
    fabricated["evidence"]["known_unknowns"].append(
        "Unresolved claim: TRIGGERS a -> b (made_up)"
    )
    problems = validate_context_document(fabricated, arch, facts)
    assert any("invented_unresolved_claim" in p for p in problems)


# ---------------------------------------------------------------------------
# Markdown projection consistency (tamper detection)
# ---------------------------------------------------------------------------


def test_markdown_tampering_is_detected(alpha_context, alpha_markdown):
    bad = alpha_markdown.replace("c" * 40, "d" * 40, 1)
    problems = validate_markdown_context_consistency(bad, alpha_context)
    assert any("commit" in p for p in problems)

    bad2 = alpha_markdown.replace(
        "### Component: ComponentA\n\n- Component id: "
        "`concept:component:component-a`",
        "### Component: Renamed\n\n- Component id: "
        "`concept:component:component-a`",
        1,
    )
    problems = validate_markdown_context_consistency(bad2, alpha_context)
    assert any("component_label_mismatch" in p for p in problems)

    rel_id = alpha_context["architecture"]["relationships"][0]["id"]
    bad3 = alpha_markdown.replace("`{}`".format(rel_id), "`rel:fake`")
    problems = validate_markdown_context_consistency(bad3, alpha_context)
    assert any("relationship" in p for p in problems)

    bad4 = alpha_markdown.replace("- Test: _none recorded_", "- Test: pytest")
    problems = validate_markdown_context_consistency(bad4, alpha_context)
    assert any("operations" in p or "empty_list" in p for p in problems)

    validate_markdown_context_or_raise(alpha_markdown, alpha_context)


def test_markdown_vs_m1m4_validation_detects_tampering(
    alpha_context, alpha_markdown
):
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    assert (
        validate_architecture_context_document(alpha_markdown, arch, facts)
        == []
    )
    bad = alpha_markdown.replace(
        "- Evidence: `docs/component_a.md`",
        "- Evidence: `docs/invented.md`",
        1,
    )
    problems = validate_architecture_context_document(bad, arch, facts)
    assert any("evidence" in p for p in problems)

    bad2 = alpha_markdown.replace(
        "- Component id: `concept:component:component-a`",
        "- Component id: `concept:component:invented`",
        1,
    )
    problems = validate_architecture_context_document(bad2, arch, facts)
    assert any("component" in p or "invented" in p for p in problems)

    validate_context_or_raise(alpha_context, arch, facts)
    validate_architecture_context_or_raise(alpha_markdown, arch, facts)


def test_shape_validation_detects_contract_drift(alpha_context):
    bad = copy.deepcopy(alpha_context)
    bad["repository"]["surprise_field"] = "x"
    problems = validate_context_document_shape(bad)
    assert problems

    bad2 = copy.deepcopy(alpha_context)
    del bad2["evidence"]
    problems = validate_context_document_shape(bad2)
    assert problems

    bad3 = copy.deepcopy(alpha_context)
    bad3["schema_version"] = "architecture_context.v9"
    problems = validate_context_document_shape(bad3)
    assert any("schema_version" in p for p in problems)

    bad4 = copy.deepcopy(alpha_context)
    bad4["structure"]["key_files"] = [{"not": "a string"}]
    problems = validate_context_document_shape(bad4)
    assert problems

    bad5 = copy.deepcopy(alpha_context)
    bad5["structure"] = "not an object"
    problems = validate_context_document_shape(bad5)
    assert problems


def test_architecture_context_determinism():
    arch = _synthetic_architecture_document()
    facts = _synthetic_facts_document()
    arch_before = copy.deepcopy(arch)
    facts_before = copy.deepcopy(facts)
    first = build_architecture_context_document(arch, facts)
    second = build_architecture_context_document(arch, facts)
    assert first == second
    first_markdown = render_architecture_context_markdown(first)
    second_markdown = render_architecture_context_markdown(second)
    assert first_markdown == second_markdown
    assert first_markdown.encode("utf-8") == second_markdown.encode("utf-8")
    # inputs are never modified
    assert arch == arch_before
    assert facts == facts_before


def test_architecture_context_rejects_wrong_documents():
    with pytest.raises(ArchitectureContextError):
        build_architecture_context(
            {"repository": {}}, _synthetic_facts_document()
        )
    with pytest.raises(ArchitectureContextError):
        build_architecture_context(
            _synthetic_architecture_document(), {}
        )


# ---------------------------------------------------------------------------
# Repository-agnostic generation (Repo A vs Repo B)
# ---------------------------------------------------------------------------


def test_architecture_context_is_repository_agnostic():
    alpha_markdown = build_architecture_context(
        _synthetic_architecture_document(), _synthetic_facts_document()
    )
    assert "ComponentA" in alpha_markdown
    assert "WorkflowA" in alpha_markdown
    assert "sample-repo-alpha" in alpha_markdown

    arch_b = _synthetic_architecture_document()
    arch_b["repository"]["name"] = "sample-repo-beta"
    arch_b["architecture"]["nodes"][0]["attributes"]["name"] = "ServiceX"
    arch_b["architecture"]["nodes"][1]["attributes"]["name"] = "WorkflowX"
    facts_b = _synthetic_facts_document()
    facts_b["repository"]["name"] = "sample-repo-beta"
    facts_b["snapshot"]["snapshot_id"] = "synthetic_beta"
    beta_context = build_architecture_context_document(arch_b, facts_b)
    beta_markdown = render_architecture_context_markdown(beta_context)
    assert "ServiceX" in beta_markdown
    assert "WorkflowX" in beta_markdown
    assert "sample-repo-beta" in beta_markdown
    assert "ComponentA" not in beta_markdown
    assert "sample-repo-alpha" not in beta_markdown
    names = {
        component["name"]
        for component in beta_context["architecture"]["components"]
    }
    assert names == {"ServiceX", "WorkflowX"}
    assert beta_context["repository"]["name"] == "sample-repo-beta"
    assert validate_context_document(beta_context, arch_b, facts_b) == []
    assert validate_markdown_context_consistency(beta_markdown, beta_context) == []

    # implementation source must not hard-code reference-repo domain terms
    for path in sorted(CONTEXT_SRC_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_SOURCE_TERMS:
            assert term not in text, (
                "hard-coded reference-repository term {!r} in {}".format(
                    term, path
                )
            )


# ---------------------------------------------------------------------------
# Artifact layout: architecture_context.md under <output>/<run-id>/...
# ---------------------------------------------------------------------------


def test_create_writes_markdown_at_run_path(tmp_path):
    arch_path = tmp_path / "architecture_model.json"
    facts_path = tmp_path / "repository_facts.json"
    arch_path.write_text(
        json.dumps(_synthetic_architecture_document()), encoding="utf-8"
    )
    facts_path.write_text(
        json.dumps(_synthetic_facts_document()), encoding="utf-8"
    )
    result = create_architecture_context(
        architecture_model_path=arch_path,
        facts_path=facts_path,
        output_root=tmp_path,
        snapshot_id="run_fixture_001",
    )
    run_dir = tmp_path / "run_fixture_001" / "architecture_context"
    assert result.markdown_path == run_dir / "architecture_context.md"
    assert run_dir.is_dir()
    assert result.markdown_path.is_file()
    # Markdown is the only generated context artifact
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "architecture_context.md"
    ]
    on_disk = result.markdown_path.read_text(encoding="utf-8")
    assert on_disk == result.markdown
    assert "- Snapshot: `synthetic_alpha`" in result.markdown
    assert "- Contract: `architecture_context.v1`" in result.markdown
    assert (
        validate_markdown_context_consistency(
            result.markdown, result.document
        )
        == []
    )
    assert not (tmp_path / "artifact").exists()


def test_default_paths_use_run_directory(tmp_path):
    md_path = default_context_markdown_path(tmp_path, "alternate_run_id")
    run_dir = tmp_path / "alternate_run_id" / "architecture_context"
    assert md_path == run_dir / "architecture_context.md"
    assert default_context_path(tmp_path, "alternate_run_id") == md_path
    assert ("artifact" + "/architecture_context") not in str(md_path)
    assert md_path.parent.name == "architecture_context"

    # without an explicit run id, the run id comes from the run context
    discovered_run = tmp_path / "discovered_run"
    (discovered_run / "facts").mkdir(parents=True)
    (discovered_run / "architecture").mkdir(parents=True)
    (discovered_run / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (discovered_run / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    assert default_context_markdown_path(tmp_path) == (
        discovered_run / "architecture_context" / "architecture_context.md"
    )

    # no run context -> explicit error, never a repository-specific default
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    with pytest.raises(ArchitectureContextError):
        default_context_markdown_path(empty_root)

    # ambiguous run context -> explicit error too
    second_run = tmp_path / "second_run"
    (second_run / "facts").mkdir(parents=True)
    (second_run / "architecture").mkdir(parents=True)
    (second_run / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (second_run / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ArchitectureContextError):
        default_context_markdown_path(tmp_path)


def test_context_generation_does_not_depend_on_hardcoded_snapshot_id(
    tmp_path,
):
    # 1) implementation source must not reference a pinned snapshot id
    for path in sorted(CONTEXT_SRC_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "PRIMARY_SNAPSHOT_ID" not in text, path
        assert "main_2026-09-22" not in text, path
        assert "ef11dac" not in text, path

    # 2) run id comes from the supplied input artifacts (document identity)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    arch_path = inputs / "architecture_model.json"
    facts_file = inputs / "repository_facts.json"
    arch_path.write_text(
        json.dumps(_synthetic_architecture_document()), encoding="utf-8"
    )
    facts_file.write_text(
        json.dumps(_synthetic_facts_document()), encoding="utf-8"
    )
    result = create_architecture_context(
        architecture_model_path=arch_path,
        facts_path=facts_file,
        output_root=inputs,
    )
    assert result.markdown_path == (
        inputs
        / "synthetic_alpha"
        / "architecture_context"
        / "architecture_context.md"
    )
    assert "- Snapshot: `synthetic_alpha`" in result.markdown

    # 3) run id comes from the existing run context (no explicit inputs)
    runs = tmp_path / "runs"
    run_dir = runs / "custom_run_abc"
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture" / "architecture_model.json").write_text(
        json.dumps(_synthetic_architecture_document()), encoding="utf-8"
    )
    (run_dir / "facts" / "repository_facts.json").write_text(
        json.dumps(_synthetic_facts_document()), encoding="utf-8"
    )
    discovered = create_architecture_context(output_root=runs)
    assert discovered.markdown_path == (
        run_dir / "architecture_context" / "architecture_context.md"
    )

    # 4) no run context at all -> explicit error, not a hard-coded fallback
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()
    with pytest.raises(ArchitectureContextError):
        create_architecture_context(output_root=empty_root)


def test_no_stale_artifact_context_references():
    """No source or docs file references the removed top-level artifact/ path."""
    stale = "artifact" + "/architecture_context"
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "tests",
    ]
    files = [PROJECT_ROOT / "README.md", CONTEXT_TEMPLATE_PATH]
    for root in roots:
        files.extend(p for p in sorted(root.rglob("*")) if p.is_file())
    offenders = []
    for path in files:
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".json", ".txt", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if stale in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == [], "stale artifact/ references: {}".format(offenders)


def test_no_stale_json_contract_references():
    """No source, doc, or test file references the removed JSON contract."""
    stale_json = "architecture_context" + ".v1.json"
    stale_schema = "schemas" + "/architecture_context"
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "contract",
    ]
    files = [PROJECT_ROOT / "README.md"]
    for root in roots:
        files.extend(p for p in sorted(root.rglob("*")) if p.is_file())
    offenders = []
    for path in files:
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".json", ".txt", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if stale_json in text or stale_schema in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == [], "stale JSON contract references: {}".format(
        offenders
    )
    assert not (PROJECT_ROOT / "schemas" / "architecture_context.v1.schema.json").exists()


# ---------------------------------------------------------------------------
# Integration: pinned artifacts (skipped when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PINNED_ARCH_MODEL.is_file() and PINNED_FACTS.is_file()),
    reason="pinned architecture/facts artifacts absent",
)
def test_pinned_run_generation(tmp_path):
    arch = _load(PINNED_ARCH_MODEL)
    facts = _load(PINNED_FACTS)
    commit = str(arch["repository"]["commit_sha"])
    result = create_architecture_context(
        architecture_model_path=PINNED_ARCH_MODEL,
        facts_path=PINNED_FACTS,
        output_path=tmp_path / "architecture_context.md",
    )
    assert result.markdown_path.is_file()
    assert "- Contract: `architecture_context.v1`" in result.markdown
    assert "- Commit:" not in result.markdown.replace(
        "- Pinned commit:", ""
    )
    assert "- Pinned commit: `{}`".format(commit) in result.markdown
    validate_context_or_raise(result.document, arch, facts)
    validate_markdown_context_or_raise(result.markdown, result.document)
    validate_architecture_context_or_raise(
        result.markdown, arch, facts
    )
    component_ids = {
        component["id"]
        for component in result.document["architecture"]["components"]
    }
    for node in arch["architecture"]["nodes"]:
        assert "- Component id: `{}`".format(node["id"]) in result.markdown
        assert node["id"] in component_ids
    rel_ids = {
        rel["id"] for rel in arch["architecture"]["relationships"]
    }
    assert {
        rel["id"]
        for rel in result.document["architecture"]["relationships"]
    } == rel_ids


@pytest.mark.skipif(
    not (PINNED_ARCH_MODEL.is_file() and PINNED_FACTS.is_file()),
    reason="pinned architecture/facts artifacts absent",
)
def test_pinned_run_artifact_location(tmp_path):
    facts = _load(PINNED_FACTS)
    run_id = str(facts["snapshot"]["snapshot_id"])
    result = create_architecture_context(
        architecture_model_path=PINNED_ARCH_MODEL,
        facts_path=PINNED_FACTS,
        output_root=tmp_path,
    )
    run_dir = tmp_path / run_id / "architecture_context"
    assert result.markdown_path == run_dir / "architecture_context.md"
    assert result.markdown_path.is_file()
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "architecture_context.md"
    ]
    assert not (tmp_path / "artifact").exists()
    assert not (
        PROJECT_ROOT / "artifact" / "architecture_context.md"
    ).exists()
    assert run_id in result.markdown_path.parts
    assert default_context_markdown_path(tmp_path, run_id) == (
        result.markdown_path
    )
    assert default_context_path(tmp_path, run_id) == result.markdown_path


@pytest.mark.skipif(
    not (PINNED_ARCH_MODEL.is_file() and PINNED_FACTS.is_file()),
    reason="pinned architecture/facts artifacts absent",
)
def test_pinned_run_identity_and_evidence():
    arch = _load(PINNED_ARCH_MODEL)
    facts = _load(PINNED_FACTS)
    context = build_architecture_context_document(arch, facts)
    markdown = render_architecture_context_markdown(context)
    assert "- Name: `{}`".format(arch["repository"]["name"]) in markdown
    assert (
        "- Pinned commit: `{}`".format(arch["repository"]["commit_sha"])
        in markdown
    )
    assert context["repository"]["name"] == arch["repository"]["name"]
    doa = next(
        component
        for component in context["architecture"]["components"]
        if component["id"] == "concept:agent:data-onboarding-agent"
    )
    model_doa = next(
        node
        for node in arch["architecture"]["nodes"]
        if node["id"] == "concept:agent:data-onboarding-agent"
    )
    json_paths = {e["path"] for e in doa["evidence"]}
    model_paths = {e["path"] for e in model_doa["evidence"]}
    assert json_paths == model_paths
    for path in model_paths:
        assert "`{}`".format(path) in markdown
    validate_context_or_raise(context, arch, facts)
    validate_markdown_context_or_raise(markdown, context)
    validate_architecture_context_or_raise(markdown, arch, facts)
