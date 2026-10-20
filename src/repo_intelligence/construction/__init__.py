"""M3 construction rulebook (engineering_construction_rules.v1).

Loads and validates the deterministic construction specification that maps
frozen M2 ``repository_facts.v2`` observations toward a future
EngineeringModel. This package is **not** the EngineeringModel builder and
does not emit graph nodes/relationships for production use.

Contract-only helpers:

- load the matrix from ``artifacts/<snapshot_id>/engineering/`` (or an explicit path)
- schema-validate against ``schemas/engineering_construction_rules.v1.json``
- enforce matrix invariants (unique rule ids, required evidence flags, …)
- normalize class/function identities so path is never duplicated
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

from ..artifacts import (
    CONSTRUCTION_RULES_FILENAME,
    PRIMARY_SNAPSHOT_ID,
    PROJECT_ROOT,
    construction_rules_path,
    default_artifacts_root,
)

SCHEMA_VERSION = "engineering_construction_rules.v1"
MATRIX_SOURCE = construction_rules_path(default_artifacts_root(), PRIMARY_SNAPSHOT_ID)
RULES_SCHEMA_SOURCE = PROJECT_ROOT / "schemas" / "engineering_construction_rules.v1.json"

FACT_SCHEMA = "repository_facts.v2"


class ConstructionRulesError(Exception):
    """Base error for construction-rulebook problems."""


class ConstructionRulesSchemaError(ConstructionRulesError):
    """Matrix document failed JSON Schema validation."""


class ConstructionRulesInvariantError(ConstructionRulesError):
    """Matrix document failed structural/semantic invariants."""


def default_matrix_path() -> Path:
    return MATRIX_SOURCE


def default_rules_schema_path() -> Path:
    return RULES_SCHEMA_SOURCE


def load_construction_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and fully validate the construction rulebook."""
    matrix_path = Path(path) if path is not None else default_matrix_path()
    if not matrix_path.is_file():
        raise ConstructionRulesError(
            "construction rules file not found: {}".format(matrix_path)
        )
    try:
        document = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ConstructionRulesError(
            "cannot read construction rules {}: {}".format(matrix_path, exc)
        )
    if not isinstance(document, dict):
        raise ConstructionRulesSchemaError("construction rules root must be an object")
    problems = validate_construction_rules_document(document)
    if problems:
        raise ConstructionRulesSchemaError(
            "construction rules failed validation:\n  - " + "\n  - ".join(problems)
        )
    return document


def validate_construction_rules_document(
    document: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> List[str]:
    """Return a list of schema + invariant problems (empty list = valid)."""
    problems: List[str] = []
    rules_schema_path = (
        Path(schema_path) if schema_path is not None else default_rules_schema_path()
    )
    if not rules_schema_path.is_file():
        problems.append("rules schema missing: {}".format(rules_schema_path))
    else:
        try:
            schema = json.loads(rules_schema_path.read_text(encoding="utf-8"))
            validator = Draft202012Validator(schema)
            for error in sorted(validator.iter_errors(document), key=lambda e: list(e.path)):
                location = ".".join(str(p) for p in error.path) or "<root>"
                problems.append("{}: {}".format(location, error.message))
        except (ValueError, OSError) as exc:
            problems.append("cannot load rules schema: {}".format(exc))

    problems.extend(_invariant_problems(document))
    return problems


def validate_construction_rules_document_or_raise(
    document: Dict[str, Any],
    schema_path: Optional[Path] = None,
) -> None:
    problems = validate_construction_rules_document(document, schema_path=schema_path)
    if problems:
        raise ConstructionRulesSchemaError(
            "construction rules failed validation:\n  - " + "\n  - ".join(problems)
        )


def _invariant_problems(document: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version must be {!r}".format(SCHEMA_VERSION))

    input_section = document.get("input")
    if isinstance(input_section, dict):
        if input_section.get("fact_schema") != FACT_SCHEMA:
            problems.append("input.fact_schema must be {!r}".format(FACT_SCHEMA))

    for section in ("node_rules", "relationship_rules", "identity_rules"):
        value = document.get(section)
        if not isinstance(value, list) or not value:
            problems.append("{} must be a non-empty array".format(section))

    rule_ids: List[str] = []
    for section in ("node_rules", "relationship_rules", "identity_rules"):
        for index, rule in enumerate(document.get(section) or []):
            if not isinstance(rule, dict):
                problems.append("{}[{}] must be an object".format(section, index))
                continue
            rule_id = rule.get("rule_id")
            if not isinstance(rule_id, str) or not rule_id:
                problems.append(
                    "{}[{}].rule_id must be a non-empty string".format(section, index)
                )
            else:
                rule_ids.append(rule_id)
            if section == "node_rules":
                if rule.get("action") == "CREATE_NODE" and rule.get("requires_evidence") is not True:
                    problems.append(
                        "{}[{}]: CREATE_NODE rules must set requires_evidence=true".format(
                            section, index
                        )
                    )
                identity = rule.get("identity")
                if not isinstance(identity, dict) or identity.get("deterministic") is not True:
                    problems.append(
                        "{}[{}]: identity.deterministic must be true".format(section, index)
                    )
                template = identity.get("template", "") if isinstance(identity, dict) else ""
                if "analyzer" in template or "uuid" in template.lower():
                    problems.append(
                        "{}[{}]: identity template must not embed analyzer/uuid".format(
                            section, index
                        )
                    )
            if section == "relationship_rules":
                if rule.get("requires_evidence") is not True:
                    problems.append(
                        "{}[{}]: relationship rules must set requires_evidence=true".format(
                            section, index
                        )
                    )
                classification = rule.get("classification")
                if classification not in {"observed", "derived"}:
                    problems.append(
                        "{}[{}]: V1 relationship classification must be observed or derived".format(
                            section, index
                        )
                    )

    seen = set()
    for rule_id in rule_ids:
        if rule_id in seen:
            problems.append("duplicate rule_id: {}".format(rule_id))
        seen.add(rule_id)

    evidence = document.get("evidence_rules")
    if not isinstance(evidence, dict):
        problems.append("evidence_rules must be an object")
    else:
        for key in (
            "canonical_node_requires_evidence",
            "canonical_relationship_requires_evidence",
            "derived_relationship_requires_contributing_facts",
            "unresolved_retains_evidence",
        ):
            if evidence.get(key) is not True:
                problems.append("evidence_rules.{} must be true".format(key))
        if evidence.get("minimum_evidence") != 1:
            problems.append("evidence_rules.minimum_evidence must be 1")
        if evidence.get("propagation") != "copy_m2_evidence_refs":
            problems.append("evidence_rules.propagation must be copy_m2_evidence_refs")
        only = evidence.get("evidence_only")
        if not isinstance(only, list) or not only:
            problems.append("evidence_rules.evidence_only must be a non-empty array")
        else:
            for index, entry in enumerate(only):
                if not isinstance(entry, dict) or not entry.get("fact_type") or not entry.get("kind"):
                    problems.append(
                        "evidence_rules.evidence_only[{}] needs fact_type and kind".format(
                            index
                        )
                    )

    unresolved = document.get("unresolved_rules")
    if not isinstance(unresolved, dict):
        problems.append("unresolved_rules must be an object")
    else:
        if unresolved.get("excluded_from_canonical_relationships") is not True:
            problems.append(
                "unresolved_rules.excluded_from_canonical_relationships must be true"
            )
        if unresolved.get("container") != "engineering.unresolved[]":
            problems.append('unresolved_rules.container must be "engineering.unresolved[]"')
        required = set(unresolved.get("required_fields") or [])
        needed = {"source", "target", "kind", "reason", "evidence"}
        if not needed.issubset(required):
            problems.append(
                "unresolved_rules.required_fields must include {}".format(sorted(needed))
            )
        reasons = set(unresolved.get("reasons") or [])
        expected_reasons = {
            "target_not_resolved",
            "ambiguous_identity",
            "missing_endpoint",
            "unsupported_relationship",
        }
        if reasons != expected_reasons:
            problems.append(
                "unresolved_rules.reasons must be exactly {}".format(sorted(expected_reasons))
            )

    validation = document.get("validation_rules")
    if not isinstance(validation, dict):
        problems.append("validation_rules must be an object")
    else:
        expected_flags = (
            "unique_node_ids",
            "unique_relationship_ids",
            "relationship_source_must_exist",
            "relationship_target_must_exist",
            "canonical_node_requires_evidence",
            "canonical_relationship_requires_evidence",
            "deterministic_node_identity",
            "deterministic_relationship_identity",
            "deterministic_output_order",
            "unresolved_claims_excluded_from_canonical_relationships",
            "no_analyzer_name_in_entity_identity",
            "no_random_ids_for_canonical_entities",
            "no_duplicate_physical_artifact_nodes",
        )
        for flag in expected_flags:
            if validation.get(flag) is not True:
                problems.append("validation_rules.{} must be true".format(flag))

    deferred = document.get("deferred")
    if not isinstance(deferred, list) or not deferred:
        problems.append("deferred must be a non-empty array")

    return problems


def node_rules_by_action(document: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
    return [
        rule
        for rule in document.get("node_rules") or []
        if isinstance(rule, dict) and rule.get("action") == action
    ]


def evidence_only_matches(document: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (fact_type, kind) pairs documented as evidence-only."""
    pairs: List[Tuple[str, str]] = []
    for entry in (document.get("evidence_rules") or {}).get("evidence_only") or []:
        if isinstance(entry, dict):
            pairs.append((entry.get("fact_type", ""), entry.get("kind", "")))
    return pairs


def normalize_entity_key(path: str, qualified_name: str) -> str:
    """Canonical entity key from M2 path + qualified_name without duplicating path.

    M2 ``qualified_name`` already carries repository path context for classes
    and functions (e.g. ``src/service.py::Service``). Blindly formatting
    ``{path}::{qualified_name}`` would yield ``src/service.py::src/service.py::Service``.

    Normalization (deterministic):

    - ``qualified_name`` already equals or is prefixed by ``path`` / ``path::``
      → use ``qualified_name`` as-is
    - otherwise → ``{path}::{qualified_name}`` (bare symbol name)

    Analyzer names are never part of the identity.
    """
    normalized_path = (path or "").strip()
    normalized_qn = (qualified_name or "").strip()
    if not normalized_path:
        return normalized_qn
    if not normalized_qn:
        return normalized_path
    if normalized_qn == normalized_path:
        return normalized_qn
    if normalized_qn.startswith(normalized_path + "::"):
        return normalized_qn
    if normalized_qn.startswith(normalized_path):
        rest = normalized_qn[len(normalized_path) :]
        if rest == "" or rest.startswith("::"):
            return normalized_qn
    return "{}::{}".format(normalized_path, normalized_qn)


def format_identity(template: str, attributes: Dict[str, Any]) -> str:
    """Fill an identity template from M2 attributes (strict; missing key errors).

    When both ``path`` and ``qualified_name`` are present, ``entity_key`` is
    injected and ``qualified_name`` is normalized so class/function identities
    never duplicate the repository path.
    """
    attrs = dict(attributes)
    if "path" in attrs or "qualified_name" in attrs:
        entity_key = normalize_entity_key(
            str(attrs.get("path") or ""), str(attrs.get("qualified_name") or "")
        )
        attrs["entity_key"] = entity_key
        if "qualified_name" in attrs:
            # Templates that still reference qualified_name receive the
            # path-normalized form (without re-prefixing path in the template).
            attrs["qualified_name"] = entity_key
    try:
        return template.format(**attrs)
    except KeyError as exc:
        raise ConstructionRulesInvariantError(
            "identity template {!r} missing attribute {}".format(template, exc)
        )


def code_entity_identity(path: str, qualified_name: str) -> str:
    """Canonical code_entity node id: ``code_entity:<entity_key>``."""
    return "code_entity:{}".format(normalize_entity_key(path, qualified_name))


def relationship_identity(kind: str, source_id: str, target_id: str) -> str:
    """Canonical relationship identity: rel:<kind>:<source>:<target>."""
    return "rel:{}:{}:{}".format(kind, source_id, target_id)


__all__ = [
    "CONSTRUCTION_RULES_FILENAME",
    "ConstructionRulesError",
    "ConstructionRulesInvariantError",
    "ConstructionRulesSchemaError",
    "FACT_SCHEMA",
    "MATRIX_SOURCE",
    "PRIMARY_SNAPSHOT_ID",
    "RULES_SCHEMA_SOURCE",
    "SCHEMA_VERSION",
    "code_entity_identity",
    "default_matrix_path",
    "default_rules_schema_path",
    "evidence_only_matches",
    "format_identity",
    "load_construction_rules",
    "normalize_entity_key",
    "node_rules_by_action",
    "relationship_identity",
    "validate_construction_rules_document",
    "validate_construction_rules_document_or_raise",
]
