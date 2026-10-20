"""EngineeringModel validation (deterministic invariants)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator

from ..artifacts import PROJECT_ROOT
from .models import (
    CLASSIFICATIONS,
    RELATIONSHIP_KINDS,
    SCHEMA_VERSION,
    UNRESOLVED_REASONS,
    EngineeringModel,
)

MODEL_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "repository_knowledge_model.v1.json"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class EngineeringModelError(Exception):
    """Raised when the EngineeringModel cannot be built or is invalid."""


class EngineeringModelValidationError(EngineeringModelError):
    """Raised when schema or invariant validation fails."""


def load_model_schema() -> Dict[str, Any]:
    import json

    return json.loads(MODEL_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_engineering_model_document(
    document: Dict[str, Any],
    expected_commit_sha: Optional[str] = None,
) -> List[str]:
    """Return schema + invariant problems (empty list = valid)."""
    problems: List[str] = []
    if not isinstance(document, dict):
        return ["model root must be an object"]

    try:
        schema = load_model_schema()
        validator = Draft202012Validator(schema)
        for error in sorted(
            validator.iter_errors(document), key=lambda e: list(e.path)
        ):
            location = ".".join(str(p) for p in error.path) or "<root>"
            problems.append("{}: {}".format(location, error.message))
    except Exception as exc:  # noqa: BLE001 — surface schema load issues
        problems.append("cannot load model schema: {}".format(exc))

    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            "schema_version must be {!r}".format(SCHEMA_VERSION)
        )

    repository = document.get("repository") or {}
    commit = repository.get("commit_sha") or ""
    if expected_commit_sha is not None and commit != expected_commit_sha:
        problems.append(
            "repository.commit_sha {!r} != expected {!r}".format(
                commit, expected_commit_sha
            )
        )

    analysis = document.get("analysis") or {}
    if analysis.get("strategy") != "deterministic":
        problems.append('analysis.strategy must be "deterministic"')

    architecture = document.get("architecture") or {}
    if architecture.get("nodes") or architecture.get("relationships") or architecture.get(
        "unresolved"
    ):
        problems.append(
            "architecture must remain empty in V1 (nodes/relationships/unresolved)"
        )
    if document.get("mappings"):
        problems.append("mappings must be empty in V1")

    engineering = document.get("engineering") or {}
    nodes = engineering.get("nodes") or []
    relationships = engineering.get("relationships") or []
    unresolved = engineering.get("unresolved") or []

    node_ids = [n.get("id") for n in nodes]
    if any(not i for i in node_ids):
        problems.append("every engineering node requires a non-empty id")
    if len(node_ids) != len(set(node_ids)):
        problems.append("unique_node_ids violated")

    rel_ids = [r.get("id") for r in relationships]
    if any(not i for i in rel_ids):
        problems.append("every relationship requires a non-empty id")
    if len(rel_ids) != len(set(rel_ids)):
        problems.append("unique_relationship_ids violated")

    node_id_set = set(node_ids)
    for index, node in enumerate(nodes):
        classification = node.get("classification")
        if classification not in CLASSIFICATIONS:
            problems.append(
                "engineering.nodes[{}]: invalid classification".format(index)
            )
        evidence = node.get("evidence") or []
        if len(evidence) < 1:
            problems.append(
                "engineering.nodes[{}]: canonical node requires evidence".format(
                    index
                )
            )
        for e_index, item in enumerate(evidence):
            e_commit = item.get("commit")
            if expected_commit_sha and e_commit != expected_commit_sha:
                problems.append(
                    "engineering.nodes[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )

    for index, rel in enumerate(relationships):
        kind = rel.get("kind")
        if kind not in RELATIONSHIP_KINDS:
            problems.append(
                "engineering.relationships[{}]: invalid kind".format(index)
            )
        classification = rel.get("classification")
        if classification not in CLASSIFICATIONS:
            problems.append(
                "engineering.relationships[{}]: invalid classification".format(
                    index
                )
            )
        if classification == "inferred":
            problems.append(
                "engineering.relationships[{}]: inferred not allowed in V1".format(
                    index
                )
            )
        source = rel.get("source")
        target = rel.get("target")
        if source not in node_id_set:
            problems.append(
                "engineering.relationships[{}]: source must exist".format(index)
            )
        if target not in node_id_set:
            problems.append(
                "engineering.relationships[{}]: target must exist".format(index)
            )
        evidence = rel.get("evidence") or []
        if len(evidence) < 1:
            problems.append(
                "engineering.relationships[{}]: requires evidence".format(index)
            )
        for e_index, item in enumerate(evidence):
            e_commit = item.get("commit")
            if expected_commit_sha and e_commit != expected_commit_sha:
                problems.append(
                    "engineering.relationships[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )

    unresolved_ids = set()
    for index, claim in enumerate(unresolved):
        for field_name in ("source", "target", "kind", "reason", "evidence"):
            if field_name not in claim:
                problems.append(
                    "engineering.unresolved[{}]: missing {}".format(
                        index, field_name
                    )
                )
        if claim.get("kind") not in RELATIONSHIP_KINDS:
            problems.append(
                "engineering.unresolved[{}]: invalid kind".format(index)
            )
        if claim.get("reason") not in UNRESOLVED_REASONS:
            problems.append(
                "engineering.unresolved[{}]: invalid reason".format(index)
            )
        # unresolved must not also be a canonical relationship id
        source = claim.get("source") or ""
        target = claim.get("target") or ""
        kind = claim.get("kind") or ""
        rel_id = "rel:{}:{}:{}".format(kind, source, target)
        unresolved_ids.add(rel_id)

    for rel_id in rel_ids:
        if rel_id in unresolved_ids:
            problems.append(
                "unresolved claim leaked into canonical relationships: {}".format(
                    rel_id
                )
            )

    # deterministic ordering: nodes by id
    if node_ids != sorted(node_ids):
        problems.append("engineering.nodes must be sorted by id")
    if rel_ids != sorted(
        rel_ids,
        key=lambda i: i,  # primary check below uses structured key
    ):
        pass
    expected_rel_order = sorted(
        relationships,
        key=lambda r: (
            r.get("kind") or "",
            r.get("source") or "",
            r.get("target") or "",
            r.get("id") or "",
        ),
    )
    if relationships != expected_rel_order:
        problems.append(
            "engineering.relationships must be deterministically ordered"
        )

    # mappings empty already checked; if present would validate source/target ids

    return problems


def validate_engineering_model_or_raise(
    document: Dict[str, Any],
    expected_commit_sha: Optional[str] = None,
) -> None:
    problems = validate_engineering_model_document(
        document, expected_commit_sha=expected_commit_sha
    )
    if problems:
        raise EngineeringModelValidationError(
            "EngineeringModel failed validation:\n  - "
            + "\n  - ".join(problems)
        )


def validate_model(
    model: EngineeringModel,
    expected_commit_sha: Optional[str] = None,
) -> List[str]:
    return validate_engineering_model_document(
        model.to_dict(), expected_commit_sha=expected_commit_sha
    )


__all__ = [
    "EngineeringModelError",
    "EngineeringModelValidationError",
    "MODEL_SCHEMA_PATH",
    "load_model_schema",
    "validate_engineering_model_document",
    "validate_engineering_model_or_raise",
    "validate_model",
]
