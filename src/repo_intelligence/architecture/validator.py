"""ArchitectureModel / knowledge-model validation (deterministic invariants)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..engineering.models import CLASSIFICATIONS as ENG_CLASSIFICATIONS
from ..engineering.models import RELATIONSHIP_KINDS, SCHEMA_VERSION
from ..engineering.validator import load_model_schema
from .models import (
    ARCHITECTURE_RELATIONSHIP_KINDS,
    ARCHITECTURE_RULES_SCHEMA_VERSION,
    MAPPING_KINDS,
    UNRESOLVED_REASONS,
)

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover — jsonschema is a hard dep in practice
    Draft202012Validator = None  # type: ignore

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ArchitectureModelError(Exception):
    """Raised when ArchitectureModel construction fails."""


class ArchitectureModelValidationError(ArchitectureModelError):
    """Raised when schema or architecture invariant validation fails."""


def _evidence_commit_ok(item: Dict[str, Any], expected_commit: Optional[str]) -> bool:
    if expected_commit is None:
        return True
    return item.get("commit") == expected_commit


def validate_knowledge_model_document(
    document: Dict[str, Any],
    expected_commit_sha: Optional[str] = None,
    require_architecture: bool = False,
) -> List[str]:
    """Validate repository_knowledge_model.v1 including architecture + mappings.

    Engineering invariants remain enforced. Architecture may be empty
    (pre-M3.5 / engineering-only artifact) unless ``require_architecture``.
    When architecture or mappings are present, M3.5 invariants apply.
    """
    problems: List[str] = []
    if not isinstance(document, dict):
        return ["model root must be an object"]

    # --- JSON Schema (schema allows architecture/mappings content) ---
    if Draft202012Validator is not None:
        try:
            schema = load_model_schema()
            validator = Draft202012Validator(schema)
            for error in sorted(
                validator.iter_errors(document), key=lambda e: list(e.path)
            ):
                location = ".".join(str(p) for p in error.path) or "<root>"
                problems.append("{}: {}".format(location, error.message))
        except Exception as exc:  # noqa: BLE001
            problems.append("cannot load model schema: {}".format(exc))

    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version must be {!r}".format(SCHEMA_VERSION))

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

    # --- engineering (same core checks as EngineeringModel V1) ---
    engineering = document.get("engineering") or {}
    eng_nodes = engineering.get("nodes") or []
    eng_rels = engineering.get("relationships") or []
    eng_unresolved = engineering.get("unresolved") or []

    node_ids = [n.get("id") for n in eng_nodes]
    if any(not i for i in node_ids):
        problems.append("every engineering node requires a non-empty id")
    if len(node_ids) != len(set(node_ids)):
        problems.append("unique_node_ids violated")

    rel_ids = [r.get("id") for r in eng_rels]
    if any(not i for i in rel_ids):
        problems.append("every relationship requires a non-empty id")
    if len(rel_ids) != len(set(rel_ids)):
        problems.append("unique_relationship_ids violated")

    node_id_set = set(node_ids)
    for index, node in enumerate(eng_nodes):
        if node.get("classification") not in ENG_CLASSIFICATIONS:
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
            if expected_commit_sha and item.get("commit") != expected_commit_sha:
                problems.append(
                    "engineering.nodes[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )

    for index, rel in enumerate(eng_rels):
        if rel.get("kind") not in RELATIONSHIP_KINDS:
            problems.append(
                "engineering.relationships[{}]: invalid kind".format(index)
            )
        if rel.get("classification") not in ENG_CLASSIFICATIONS:
            problems.append(
                "engineering.relationships[{}]: invalid classification".format(index)
            )
        if rel.get("classification") == "inferred":
            problems.append(
                "engineering.relationships[{}]: inferred not allowed in V1".format(
                    index
                )
            )
        if rel.get("source") not in node_id_set:
            problems.append(
                "engineering.relationships[{}]: source must exist".format(index)
            )
        if rel.get("target") not in node_id_set:
            problems.append(
                "engineering.relationships[{}]: target must exist".format(index)
            )
        evidence = rel.get("evidence") or []
        if len(evidence) < 1:
            problems.append(
                "engineering.relationships[{}]: requires evidence".format(index)
            )
        for e_index, item in enumerate(evidence):
            if expected_commit_sha and item.get("commit") != expected_commit_sha:
                problems.append(
                    "engineering.relationships[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )

    for index, claim in enumerate(eng_unresolved):
        for field_name in ("source", "target", "kind", "reason", "evidence"):
            if field_name not in claim:
                problems.append(
                    "engineering.unresolved[{}]: missing {}".format(
                        index, field_name
                    )
                )

    # engineering relationship order
    expected_rel_order = sorted(
        eng_rels,
        key=lambda r: (
            r.get("kind") or "",
            r.get("source") or "",
            r.get("target") or "",
            r.get("id") or "",
        ),
    )
    if eng_rels != expected_rel_order:
        problems.append(
            "engineering.relationships must be deterministically ordered"
        )

    # --- architecture ---
    architecture = document.get("architecture") or {}
    arch_nodes = architecture.get("nodes") or []
    arch_rels = architecture.get("relationships") or []
    arch_unresolved = architecture.get("unresolved") or []
    mappings = document.get("mappings") or []

    has_arch_content = bool(arch_nodes or arch_rels or arch_unresolved or mappings)
    if require_architecture and not has_arch_content:
        problems.append(
            "architecture section required for ArchitectureModel validation"
        )

    arch_ids = [n.get("id") for n in arch_nodes]
    if any(not i for i in arch_ids):
        problems.append("every architecture node requires a non-empty id")
    if len(arch_ids) != len(set(arch_ids)):
        problems.append("unique_architecture_node_ids violated")

    arch_id_set = set(arch_ids)
    for index, node in enumerate(arch_nodes):
        if node.get("classification") not in ENG_CLASSIFICATIONS:
            problems.append(
                "architecture.nodes[{}]: invalid classification".format(index)
            )
        evidence = node.get("evidence") or []
        if len(evidence) < 1:
            problems.append(
                "architecture.nodes[{}]: canonical architecture node requires evidence".format(
                    index
                )
            )
        for e_index, item in enumerate(evidence):
            if expected_commit_sha and item.get("commit") != expected_commit_sha:
                problems.append(
                    "architecture.nodes[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )
        if node.get("kind") != "concept":
            problems.append(
                "architecture.nodes[{}]: V1 architecture kind must be concept".format(
                    index
                )
            )
        if node.get("subkind") != "agent":
            problems.append(
                "architecture.nodes[{}]: V1 architecture subkind must be agent".format(
                    index
                )
            )

    # architecture relationships: only validated AR-R001 TRIGGERS edges
    arch_rel_ids = [r.get("id") for r in arch_rels]
    if any(not i for i in arch_rel_ids):
        problems.append("every architecture relationship requires a non-empty id")
    if len(arch_rel_ids) != len(set(arch_rel_ids)):
        problems.append("unique_architecture_relationship_ids violated")

    arch_rel_id_set = set()
    for index, rel in enumerate(arch_rels):
        if rel.get("kind") not in ARCHITECTURE_RELATIONSHIP_KINDS:
            problems.append(
                "architecture.relationships[{}]: invalid kind".format(index)
            )
        if rel.get("classification") not in ENG_CLASSIFICATIONS:
            problems.append(
                "architecture.relationships[{}]: invalid classification".format(index)
            )
        if rel.get("classification") == "inferred":
            problems.append(
                "architecture.relationships[{}]: inferred not allowed".format(index)
            )
        source = rel.get("source")
        target = rel.get("target")
        if not source or source not in arch_id_set:
            problems.append(
                "architecture.relationships[{}]: source must exist".format(index)
            )
        if not target or target not in arch_id_set:
            problems.append(
                "architecture.relationships[{}]: target must exist".format(index)
            )
        evidence = rel.get("evidence") or []
        if len(evidence) < 1:
            problems.append(
                "architecture.relationships[{}]: requires evidence".format(index)
            )
        for e_index, item in enumerate(evidence):
            if expected_commit_sha and item.get("commit") != expected_commit_sha:
                problems.append(
                    "architecture.relationships[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )
        rid = rel.get("id")
        if rid:
            arch_rel_id_set.add(rid)

    # unresolved claims must not also appear as canonical architecture relationships
    canonical_pairs = {
        (r.get("kind"), r.get("source"), r.get("target")) for r in arch_rels
    }
    for index, claim in enumerate(arch_unresolved):
        pair = (claim.get("kind"), claim.get("source"), claim.get("target"))
        if pair in canonical_pairs:
            problems.append(
                "architecture.unresolved[{}]: also present as canonical relationship".format(
                    index
                )
            )

    expected_arch_rel_order = sorted(
        arch_rels,
        key=lambda r: (
            r.get("kind") or "",
            r.get("source") or "",
            r.get("target") or "",
            r.get("id") or "",
        ),
    )
    if arch_rels != expected_arch_rel_order:
        problems.append(
            "architecture.relationships must be deterministically ordered"
        )

    # architecture unresolved: use engineering-like claim shape but architecture kinds
    for index, claim in enumerate(arch_unresolved):
        for field_name in ("source", "target", "kind", "reason", "evidence"):
            if field_name not in claim:
                problems.append(
                    "architecture.unresolved[{}]: missing {}".format(
                        index, field_name
                    )
                )
        if claim.get("reason") not in UNRESOLVED_REASONS:
            problems.append(
                "architecture.unresolved[{}]: invalid reason".format(index)
            )
        if claim.get("kind") not in (
            tuple(ARCHITECTURE_RELATIONSHIP_KINDS) + tuple(MAPPING_KINDS)
        ):
            problems.append(
                "architecture.unresolved[{}]: invalid kind".format(index)
            )
        evidence = claim.get("evidence") or []
        if not evidence:
            problems.append(
                "architecture.unresolved[{}]: requires evidence".format(index)
            )

    # --- mappings ---
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            problems.append("mappings[{}] must be an object".format(index))
            continue
        source = mapping.get("source")
        target = mapping.get("target")
        kind = mapping.get("kind")
        if not source:
            problems.append("mappings[{}]: source required".format(index))
        elif source not in node_id_set:
            problems.append(
                "mappings[{}]: source must reference an existing engineering node".format(
                    index
                )
            )
        if not target:
            problems.append("mappings[{}]: target required".format(index))
        elif target not in arch_id_set:
            problems.append(
                "mappings[{}]: target must reference an existing architecture node".format(
                    index
                )
            )
        if kind not in MAPPING_KINDS:
            problems.append("mappings[{}]: invalid kind".format(index))
        if mapping.get("classification") not in ENG_CLASSIFICATIONS:
            problems.append(
                "mappings[{}]: invalid classification".format(index)
            )
        evidence = mapping.get("evidence") or []
        if len(evidence) < 1:
            problems.append(
                "mappings[{}]: canonical mapping requires evidence".format(index)
            )
        for e_index, item in enumerate(evidence):
            if expected_commit_sha and item.get("commit") != expected_commit_sha:
                problems.append(
                    "mappings[{}].evidence[{}]: commit mismatch".format(
                        index, e_index
                    )
                )

    # ordering
    expected_arch_node_order = sorted(
        arch_nodes, key=lambda n: n.get("id") or ""
    )
    if arch_nodes != expected_arch_node_order:
        problems.append("architecture.nodes must be sorted by id")

    expected_mapping_order = sorted(
        mappings,
        key=lambda m: (
            m.get("kind") or "",
            m.get("source") or "",
            m.get("target") or "",
            m.get("id") or "",
        ),
    )
    if mappings != expected_mapping_order:
        problems.append("mappings must be deterministically ordered")

    expected_unresolved_order = sorted(
        arch_unresolved,
        key=lambda u: (
            str(u.get("kind") or ""),
            str(u.get("source") or ""),
            str(u.get("target") or ""),
            str(u.get("reason") or ""),
        ),
    )
    if arch_unresolved != expected_unresolved_order:
        problems.append(
            "architecture.unresolved must be deterministically ordered"
        )

    return problems


def validate_knowledge_model_or_raise(
    document: Dict[str, Any],
    expected_commit_sha: Optional[str] = None,
    require_architecture: bool = False,
) -> None:
    problems = validate_knowledge_model_document(
        document,
        expected_commit_sha=expected_commit_sha,
        require_architecture=require_architecture,
    )
    if problems:
        raise ArchitectureModelValidationError(
            "Architecture/knowledge model failed validation:\n  - "
            + "\n  - ".join(problems)
        )


__all__ = [
    "ARCHITECTURE_RULES_SCHEMA_VERSION",
    "ArchitectureModelError",
    "ArchitectureModelValidationError",
    "validate_knowledge_model_document",
    "validate_knowledge_model_or_raise",
]
