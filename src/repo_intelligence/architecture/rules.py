"""Load and validate architecture_construction_rules.v1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

from ..artifacts import (
    PRIMARY_SNAPSHOT_ID,
    PROJECT_ROOT,
    architecture_rules_path,
    default_artifacts_root,
)
from .models import ARCHITECTURE_RULES_SCHEMA_VERSION

RULES_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "architecture_construction_rules.v1.json"
)
KNOWLEDGE_MODEL_SCHEMA = "repository_knowledge_model.v1"
_RULE_ID_RE = re.compile(r"^AR-[NMR][0-9]{3}$")


class ArchitectureRulesError(Exception):
    """Base error for architecture rulebook problems."""


class ArchitectureRulesSchemaError(ArchitectureRulesError):
    """Matrix document failed JSON Schema validation."""


class ArchitectureRulesInvariantError(ArchitectureRulesError):
    """Matrix document failed structural/semantic invariants."""


def default_architecture_rules_path() -> Path:
    return architecture_rules_path(default_artifacts_root(), PRIMARY_SNAPSHOT_ID)


def load_architecture_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and fully validate the architecture rulebook."""
    matrix_path = Path(path) if path is not None else default_architecture_rules_path()
    if not matrix_path.is_file():
        raise ArchitectureRulesError(
            "architecture construction rules not found: {}".format(matrix_path)
        )
    try:
        document = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ArchitectureRulesError(
            "cannot read architecture construction rules {}: {}".format(
                matrix_path, exc
            )
        )
    if not isinstance(document, dict):
        raise ArchitectureRulesSchemaError(
            "architecture construction rules root must be an object"
        )
    problems = validate_architecture_rules_document(document)
    if problems:
        raise ArchitectureRulesSchemaError(
            "architecture construction rules failed validation:\n  - "
            + "\n  - ".join(problems)
        )
    return document


def validate_architecture_rules_document(
    document: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> List[str]:
    """Return schema + invariant problems (empty list = valid)."""
    problems: List[str] = []
    rules_schema_path = Path(schema_path) if schema_path else RULES_SCHEMA_PATH
    if not rules_schema_path.is_file():
        problems.append("architecture rules schema missing: {}".format(rules_schema_path))
    else:
        try:
            schema = json.loads(rules_schema_path.read_text(encoding="utf-8"))
            validator = Draft202012Validator(schema)
            for error in sorted(
                validator.iter_errors(document), key=lambda e: list(e.path)
            ):
                location = ".".join(str(p) for p in error.path) or "<root>"
                problems.append("{}: {}".format(location, error.message))
        except (ValueError, OSError) as exc:
            problems.append("cannot load architecture rules schema: {}".format(exc))

    problems.extend(_invariant_problems(document))
    return problems


def validate_architecture_rules_document_or_raise(
    document: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> None:
    problems = validate_architecture_rules_document(document, schema_path=schema_path)
    if problems:
        raise ArchitectureRulesSchemaError(
            "architecture construction rules failed validation:\n  - "
            + "\n  - ".join(problems)
        )


def _invariant_problems(document: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if document.get("schema_version") != ARCHITECTURE_RULES_SCHEMA_VERSION:
        problems.append(
            "schema_version must be {!r}".format(ARCHITECTURE_RULES_SCHEMA_VERSION)
        )

    input_section = document.get("input")
    if isinstance(input_section, dict):
        if input_section.get("knowledge_model_schema") != KNOWLEDGE_MODEL_SCHEMA:
            problems.append(
                "input.knowledge_model_schema must be {!r}".format(KNOWLEDGE_MODEL_SCHEMA)
            )

    node_rules = document.get("node_rules")
    mapping_rules = document.get("mapping_rules")
    relationship_rules = document.get("relationship_rules")
    if not isinstance(node_rules, list) or not node_rules:
        problems.append("node_rules must be a non-empty array")
        node_rules = []
    if not isinstance(mapping_rules, list) or not mapping_rules:
        problems.append("mapping_rules must be a non-empty array")
        mapping_rules = []
    if not isinstance(relationship_rules, list) or not relationship_rules:
        problems.append(
            "relationship_rules must be a non-empty array for AR-R001 (TRIGGERS)"
        )
        relationship_rules = []

    rule_ids: List[str] = []
    for section, rules in (
        ("node_rules", node_rules),
        ("mapping_rules", mapping_rules),
        ("relationship_rules", relationship_rules),
    ):
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                problems.append("{}[{}] must be an object".format(section, index))
                continue
            rule_id = rule.get("rule_id")
            if not isinstance(rule_id, str) or not rule_id:
                problems.append(
                    "{}[{}].rule_id must be a non-empty string".format(section, index)
                )
            else:
                if not _RULE_ID_RE.match(rule_id):
                    problems.append(
                        "{}[{}].rule_id must match AR-N001/AR-M001/AR-R001 pattern".format(
                            section, index
                        )
                    )
                rule_ids.append(rule_id)
            if rule.get("requires_evidence") is not True:
                problems.append(
                    "{}[{}]: requires_evidence must be true".format(section, index)
                )
            if rule.get("classification") != "derived":
                problems.append(
                    "{}[{}]: classification must be derived".format(section, index)
                )

    for rule_id in rule_ids:
        # uniqueness checked below
        pass
    seen = set()
    for rule_id in rule_ids:
        if rule_id in seen:
            problems.append("duplicate rule_id: {}".format(rule_id))
        seen.add(rule_id)

    # frozen rule ids present
    node_ids = {r.get("rule_id") for r in node_rules if isinstance(r, dict)}
    map_ids = {r.get("rule_id") for r in mapping_rules if isinstance(r, dict)}
    rel_ids = {r.get("rule_id") for r in relationship_rules if isinstance(r, dict)}
    if "AR-N001" not in node_ids:
        problems.append("node_rules must include AR-N001 (agent_definition → concept/agent)")
    if "AR-M001" not in map_ids:
        problems.append("mapping_rules must include AR-M001 (CONTRIBUTES_TO)")
    if "AR-N002" not in node_ids:
        problems.append(
            "node_rules must include AR-N002 (orchestration boundary → "
            "concept:agent:data-onboarding-agent)"
        )
    if "AR-M002" not in map_ids:
        problems.append(
            "mapping_rules must include AR-M002 (orchestration sources → "
            "concept:agent:data-onboarding-agent)"
        )
    if "AR-R001" not in rel_ids:
        problems.append(
            "relationship_rules must include AR-R001 (M3 TRIGGERS → architecture TRIGGERS)"
        )

    for rule in node_rules:
        if not isinstance(rule, dict):
            continue
        match = rule.get("match") or {}
        if match.get("kind") == "artifact" and match.get("subkind") == "agent_definition":
            node = rule.get("node") or {}
            if node.get("kind") != "concept" or node.get("subkind") != "agent":
                problems.append(
                    "AR-N001 must map artifact/agent_definition → concept/agent"
                )
        if rule.get("rule_id") == "AR-N002":
            identity = rule.get("identity") or {}
            template = identity.get("template") or ""
            if template != "concept:agent:data-onboarding-agent":
                problems.append(
                    "AR-N002 identity.template must be "
                    "concept:agent:data-onboarding-agent"
                )
            if "{agent_name}" in template:
                problems.append(
                    "AR-N002 identity.template must be fixed (no {agent_name})"
                )
            node = rule.get("node") or {}
            if node.get("kind") != "concept" or node.get("subkind") != "agent":
                problems.append("AR-N002 must create concept/agent")
            attrs = rule.get("attributes") or {}
            if attrs.get("name") != "Data Onboarding Agent":
                problems.append(
                    "AR-N002 attributes.name must be 'Data Onboarding Agent'"
                )
            ids = match.get("engineering_node_ids") or []
            if not ids:
                problems.append(
                    "AR-N002 match.engineering_node_ids must be a non-empty list"
                )
            required_sources = {
                "artifact:agent_command:.claude/commands/onboard-workflow.md",
                "artifact:file:SKILLS.md",
                "artifact:file:docs/workflow-diagrams.md",
                "artifact:file:runbooks/data-onboarding-agent/README.md",
            }
            if not required_sources.issubset(set(ids)):
                problems.append(
                    "AR-N002 match.engineering_node_ids must include the four "
                    "ADOP orchestration evidence entities"
                )

    for rule in mapping_rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("rule_id") == "AR-M001":
            rel = rule.get("relationship") or {}
            if rel.get("kind") != "CONTRIBUTES_TO":
                problems.append("AR-M001 relationship.kind must be CONTRIBUTES_TO")
        if rule.get("rule_id") == "AR-M002":
            rel = rule.get("relationship") or {}
            if rel.get("kind") != "CONTRIBUTES_TO":
                problems.append("AR-M002 relationship.kind must be CONTRIBUTES_TO")
            target = rule.get("target") or {}
            if target.get("from_node_rule") != "AR-N002":
                problems.append(
                    "AR-M002 target.from_node_rule must be AR-N002"
                )
            ids = (rule.get("match") or {}).get("engineering_node_ids") or []
            node_rule = next(
                (
                    r
                    for r in node_rules
                    if isinstance(r, dict) and r.get("rule_id") == "AR-N002"
                ),
                None,
            )
            node_ids_list = (
                ((node_rule or {}).get("match") or {}).get("engineering_node_ids")
                or []
            )
            if ids != node_ids_list:
                problems.append(
                    "AR-M002 match.engineering_node_ids must match AR-N002"
                )

    for rule in relationship_rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("rule_id") == "AR-R001":
            if rule.get("action") != "CREATE_ARCHITECTURE_RELATIONSHIP":
                problems.append(
                    "AR-R001 action must be CREATE_ARCHITECTURE_RELATIONSHIP"
                )
            rel = rule.get("relationship") or {}
            if rel.get("kind") != "TRIGGERS":
                problems.append("AR-R001 relationship.kind must be TRIGGERS")
            if rel.get("source_engineering_relationship") != "TRIGGERS":
                problems.append(
                    "AR-R001 relationship.source_engineering_relationship must be TRIGGERS"
                )
            src = rule.get("source_resolution") or {}
            tgt = rule.get("target_resolution") or {}
            if src.get("from_engineering_node_via_mapping") is not True:
                problems.append(
                    "AR-R001 source_resolution.from_engineering_node_via_mapping must be true"
                )
            if tgt.get("from_engineering_node_via_mapping") is not True:
                problems.append(
                    "AR-R001 target_resolution.from_engineering_node_via_mapping must be true"
                )

    if (document.get("relationship_vocabulary") or []) != ["TRIGGERS"]:
        problems.append(
            "relationship_vocabulary must be exactly ['TRIGGERS'] for this refinement"
        )
    if (document.get("mapping_vocabulary") or []) != ["CONTRIBUTES_TO"]:
        problems.append("mapping_vocabulary must be exactly ['CONTRIBUTES_TO']")

    evidence = document.get("evidence_rules")
    if not isinstance(evidence, dict):
        problems.append("evidence_rules must be an object")
    else:
        for key in (
            "canonical_architecture_node_requires_evidence",
            "canonical_mapping_requires_evidence",
            "canonical_architecture_relationship_requires_evidence",
            "unresolved_retains_evidence",
        ):
            if evidence.get(key) is not True:
                problems.append("evidence_rules.{} must be true".format(key))
        if evidence.get("minimum_evidence") != 1:
            problems.append("evidence_rules.minimum_evidence must be 1")
        if evidence.get("propagation") != "copy_engineering_node_evidence_refs":
            problems.append(
                "evidence_rules.propagation must be copy_engineering_node_evidence_refs"
            )

    unresolved = document.get("unresolved_rules")
    if not isinstance(unresolved, dict):
        problems.append("unresolved_rules must be an object")
    else:
        if unresolved.get("container") != "architecture.unresolved[]":
            problems.append('unresolved_rules.container must be "architecture.unresolved[]"')
        if unresolved.get("excluded_from_canonical_architecture") is not True:
            problems.append(
                "unresolved_rules.excluded_from_canonical_architecture must be true"
            )
        required = set(unresolved.get("required_fields") or [])
        needed = {"source", "target", "kind", "reason", "evidence"}
        if not needed.issubset(required):
            problems.append(
                "unresolved_rules.required_fields must include {}".format(sorted(needed))
            )

    validation = document.get("validation_rules")
    if not isinstance(validation, dict):
        problems.append("validation_rules must be an object")
    else:
        expected_flags = (
            "unique_architecture_node_ids",
            "unique_architecture_relationship_ids",
            "mapping_source_must_exist",
            "mapping_target_must_exist",
            "relationship_source_must_exist",
            "relationship_target_must_exist",
            "canonical_architecture_node_requires_evidence",
            "canonical_mapping_requires_evidence",
            "canonical_architecture_relationship_requires_evidence",
            "deterministic_identity",
            "deterministic_output_order",
            "unsupported_claims_not_in_canonical_relationships",
        )
        for flag in expected_flags:
            if validation.get(flag) is not True:
                problems.append("validation_rules.{} must be true".format(flag))

    deferred = document.get("deferred")
    if not isinstance(deferred, list) or not deferred:
        problems.append("deferred must be a non-empty array")

    return problems


def node_rules_by_action(
    document: Dict[str, Any], action: str
) -> List[Dict[str, Any]]:
    return [
        rule
        for rule in document.get("node_rules") or []
        if isinstance(rule, dict) and rule.get("action") == action
    ]


def mapping_rules_by_action(
    document: Dict[str, Any], action: str
) -> List[Dict[str, Any]]:
    return [
        rule
        for rule in document.get("mapping_rules") or []
        if isinstance(rule, dict) and rule.get("action") == action
    ]


def relationship_rules_by_action(
    document: Dict[str, Any], action: str
) -> List[Dict[str, Any]]:
    return [
        rule
        for rule in document.get("relationship_rules") or []
        if isinstance(rule, dict) and rule.get("action") == action
    ]


def architecture_node_match_keys(
    document: Dict[str, Any],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for rule in document.get("node_rules") or []:
        match = (rule or {}).get("match") or {}
        pairs.append((match.get("kind") or "", match.get("subkind") or ""))
    return pairs


def architecture_mapping_match_keys(
    document: Dict[str, Any],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for rule in document.get("mapping_rules") or []:
        match = (rule or {}).get("match") or {}
        pairs.append((match.get("kind") or "", match.get("subkind") or ""))
    return pairs


__all__ = [
    "ArchitectureRulesError",
    "ArchitectureRulesInvariantError",
    "ArchitectureRulesSchemaError",
    "KNOWLEDGE_MODEL_SCHEMA",
    "RULES_SCHEMA_PATH",
    "architecture_mapping_match_keys",
    "architecture_node_match_keys",
    "default_architecture_rules_path",
    "load_architecture_rules",
    "mapping_rules_by_action",
    "node_rules_by_action",
    "relationship_rules_by_action",
    "validate_architecture_rules_document",
    "validate_architecture_rules_document_or_raise",
]
