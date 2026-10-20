"""ArchitectureModel construction pipeline.

    EngineeringModel (repository_knowledge_model.v1)
        + architecture_construction_rules.v1
        → rule evaluation
        → identity / dedup / evidence
        → validation
        → ArchitectureModel section + mappings on the same envelope

Consumes the existing EngineeringModel only — never re-parses the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..artifacts import (
    PRIMARY_SNAPSHOT_ID,
    architecture_model_path,
    architecture_rules_path,
    default_artifacts_root,
    engineering_model_path as artifact_engineering_model_path,
    resolve_output_root,
)
from ..engineering.models import EngineeringModel
from ..engineering.validator import (
    EngineeringModelError,
    validate_engineering_model_document,
)
from .models import (
    ARCHITECTURE_RELATIONSHIP_KINDS,
    ARCHITECTURE_RULES_SCHEMA_VERSION,
    architecture_counts,
    architecture_relationship_identity,
    mapping_identity,
)
from .resolver import agent_name_from_engineering_node, architecture_agent_node_id
from .rules import (
    ArchitectureRulesError,
    load_architecture_rules,
    mapping_rules_by_action,
    node_rules_by_action,
    relationship_rules_by_action,
)
from .validator import (
    ArchitectureModelError,
    ArchitectureModelValidationError,
    validate_knowledge_model_or_raise,
)

ARCHITECTURE_MODEL_FILENAME = "architecture_model.json"

# Fixed-identity concept boundary (AR-N002): multi-entity orchestration sources
# that collectively represent the Data Onboarding Agent. Matched only by
# explicit engineering_node_ids in the rulebook — never by filename scan.
DATA_ONBOARDING_AGENT_ID = "concept:agent:data-onboarding-agent"


def _evidence_sort_key(item: Dict[str, Any]) -> Tuple:
    return (
        str(item.get("path") or ""),
        int(item.get("line_start") or 0),
        str(item.get("symbol") or ""),
        str(item.get("artifact_type") or ""),
        str(item.get("commit") or ""),
    )


def _merge_evidence(
    target: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
) -> None:
    for item in incoming or []:
        if item not in target:
            target.append(dict(item))


def _match_rule(
    rules: List[Dict[str, Any]], eng_node: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Select the first rule that matches this engineering node.

    Explicit ``engineering_node_ids`` matches win over kind/subkind scans so a
    fixed-identity orchestration rule never accidentally claims every file.
    """
    eng_id = eng_node.get("id") or ""
    kind = eng_node.get("kind")
    subkind = eng_node.get("subkind")

    for rule in rules:
        match = rule.get("match") or {}
        ids = match.get("engineering_node_ids")
        if ids is not None and eng_id in set(ids):
            return rule

    for rule in rules:
        match = rule.get("match") or {}
        if match.get("engineering_node_ids") is not None:
            continue
        if match.get("kind") == kind and match.get("subkind") == subkind:
            return rule
    return None


def _resolve_architecture_identity(
    node_rule: Dict[str, Any], eng_node: Dict[str, Any]
) -> Tuple[str, str]:
    """Return (arch_id, display_name) for a matched engineering node.

    Fixed-identity templates (no ``{agent_name}``) are used as-is.
    Parameterized templates derive agent_name from the agent_definition node.
    """
    identity = node_rule.get("identity") or {}
    template = identity.get("template") or "concept:agent:{agent_name}"
    fixed_attrs = node_rule.get("attributes") or {}

    if "{agent_name}" not in template:
        arch_id = template
        display = fixed_attrs.get("name") or arch_id
        return arch_id, str(display)

    agent_name = agent_name_from_engineering_node(eng_node)
    if not agent_name:
        raise ArchitectureModelError(
            "canonical agent identity could not be established for {}".format(
                eng_node.get("id")
            )
        )
    try:
        arch_id = template.format(agent_name=agent_name)
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureModelError(
            "architecture identity failed for {}: {}".format(
                eng_node.get("id"), exc
            )
        )
    display = fixed_attrs.get("name") or agent_name
    return arch_id, str(display)


def load_engineering_model_document(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArchitectureModelError("EngineeringModel not found: {}".format(path))
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ArchitectureModelError(
            "Cannot read EngineeringModel {}: {}".format(path, exc)
        )
    if not isinstance(document, dict):
        raise ArchitectureModelError("EngineeringModel root must be an object")
    if document.get("schema_version") != "repository_knowledge_model.v1":
        raise ArchitectureModelError(
            "Expected repository_knowledge_model.v1, got {!r}".format(
                document.get("schema_version")
            )
        )
    return document


def build_architecture_model(
    engineering_document: Dict[str, Any],
    rules: Dict[str, Any],
) -> EngineeringModel:
    """Apply architecture construction rules to an EngineeringModel document.

    Returns a new EngineeringModel with ``architecture`` and ``mappings``
    populated. Engineering sections are preserved unchanged.
    """
    if not isinstance(engineering_document, dict):
        raise ArchitectureModelError("EngineeringModel must be an object")
    if engineering_document.get("schema_version") != "repository_knowledge_model.v1":
        raise ArchitectureModelError(
            "Expected repository_knowledge_model.v1, got {!r}".format(
                engineering_document.get("schema_version")
            )
        )
    if rules.get("schema_version") != ARCHITECTURE_RULES_SCHEMA_VERSION:
        raise ArchitectureModelError(
            "Expected architecture_construction_rules.v1, got {!r}".format(
                rules.get("schema_version")
            )
        )

    commit = (engineering_document.get("repository") or {}).get("commit_sha") or ""
    if len(commit) != 40:
        raise ArchitectureModelError(
            "EngineeringModel repository.commit_sha must be a 40-char SHA"
        )

    # Validate pure engineering input first (architecture may already be empty).
    eng_problems = validate_engineering_model_document(
        engineering_document, expected_commit_sha=commit
    )
    if eng_problems:
        raise ArchitectureModelError(
            "input EngineeringModel failed validation:\n  - "
            + "\n  - ".join(eng_problems)
        )

    engineering = engineering_document.get("engineering") or {}
    eng_nodes: List[Dict[str, Any]] = list(engineering.get("nodes") or [])
    eng_rels: List[Dict[str, Any]] = list(engineering.get("relationships") or [])
    eng_unresolved: List[Dict[str, Any]] = list(
        engineering.get("unresolved") or []
    )

    create_node_rules = node_rules_by_action(rules, "CREATE_ARCHITECTURE_NODE")
    create_mapping_rules = mapping_rules_by_action(rules, "CREATE_MAPPING")
    create_relationship_rules = relationship_rules_by_action(
        rules, "CREATE_ARCHITECTURE_RELATIONSHIP"
    )
    if not create_node_rules:
        raise ArchitectureModelError("rules missing CREATE_ARCHITECTURE_NODE")
    if not create_mapping_rules:
        raise ArchitectureModelError("rules missing CREATE_MAPPING")

    # Match rules: explicit engineering_node_ids first, then (kind, subkind).
    # Kind/subkind indexes retained only for deterministic rule ordering helpers.
    arch_nodes: Dict[str, Dict[str, Any]] = {}
    mappings: Dict[str, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []

    # Sort engineering nodes by id for deterministic traversal.
    ordered_eng = sorted(eng_nodes, key=lambda n: n.get("id") or "")

    for eng_node in ordered_eng:
        node_rule = _match_rule(create_node_rules, eng_node)
        map_rule = _match_rule(create_mapping_rules, eng_node)
        if node_rule is None and map_rule is None:
            # unsupported for this milestone — not automatically architecture
            continue

        evidence_in = list(eng_node.get("evidence") or [])
        if not evidence_in:
            # no evidence → no canonical architectural claim
            continue

        if node_rule is None:
            # mapping without a node rule — park as unsupported
            unresolved.append(
                {
                    "source": eng_node.get("id") or "",
                    "target": "concept:agent",
                    "kind": "CONTRIBUTES_TO",
                    "reason": "unsupported_relationship",
                    "evidence": [dict(e) for e in evidence_in],
                }
            )
            continue

        try:
            arch_id, display_name = _resolve_architecture_identity(
                node_rule, eng_node
            )
        except ArchitectureModelError:
            unresolved.append(
                {
                    "source": eng_node.get("id") or "",
                    "target": "concept:agent",
                    "kind": "CONTRIBUTES_TO",
                    "reason": "ambiguous_identity",
                    "evidence": [dict(e) for e in evidence_in],
                }
            )
            continue

        node_spec = node_rule.get("node") or {}
        # upsert architecture node (dedupe by identity; merge evidence)
        if arch_id not in arch_nodes:
            arch_nodes[arch_id] = {
                "id": arch_id,
                "kind": node_spec.get("kind") or "concept",
                "subkind": node_spec.get("subkind") or "agent",
                "attributes": {"name": display_name},
                "classification": node_rule.get("classification") or "derived",
                "evidence": [],
            }
        else:
            # keep first deterministic display name; merge evidence only
            pass
        _merge_evidence(arch_nodes[arch_id]["evidence"], evidence_in)

        # mapping rule for this engineering node
        if map_rule is None:
            continue
        if map_rule.get("action") != "CREATE_MAPPING":
            continue

        # target must be the architecture node created by AR-N001
        if arch_id not in arch_nodes:
            unresolved.append(
                {
                    "source": eng_node.get("id") or "",
                    "target": arch_id,
                    "kind": "CONTRIBUTES_TO",
                    "reason": "target_not_resolved",
                    "evidence": [dict(e) for e in evidence_in],
                }
            )
            continue

        map_kind = ((map_rule.get("relationship") or {}).get("kind")) or (
            "CONTRIBUTES_TO"
        )
        map_id = mapping_identity(map_kind, eng_node.get("id") or "", arch_id)
        if map_id not in mappings:
            mappings[map_id] = {
                "id": map_id,
                "source": eng_node.get("id") or "",
                "target": arch_id,
                "kind": map_kind,
                "classification": map_rule.get("classification") or "derived",
                "evidence": [],
            }
        # mapping retains evidence for its individual source→target claim
        _merge_evidence(mappings[map_id]["evidence"], evidence_in)

    # AR-R001: elevate M3 TRIGGERS → architecture TRIGGERS only when both
    # engineering endpoints resolve through existing mappings to architecture
    # nodes. Never copy engineering ids into architecture relationships.
    eng_to_arch: Dict[str, str] = {}
    for mapping in mappings.values():
        # first-wins for deterministic multi-mapping (current rules emit one)
        eng_to_arch.setdefault(mapping.get("source") or "", mapping.get("target") or "")

    eng_node_by_id = {n.get("id"): n for n in eng_nodes}
    mapping_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for mapping in mappings.values():
        mapping_by_source.setdefault(mapping.get("source") or "", []).append(mapping)

    arch_relationships: Dict[str, Dict[str, Any]] = {}

    for rel_rule in create_relationship_rules:
        eng_kind = (
            (rel_rule.get("relationship") or {}).get("source_engineering_relationship")
            or "TRIGGERS"
        )
        arch_kind = (rel_rule.get("relationship") or {}).get("kind") or "TRIGGERS"
        if arch_kind not in ARCHITECTURE_RELATIONSHIP_KINDS:
            raise ArchitectureModelError(
                "unsupported architecture relationship kind: {}".format(arch_kind)
            )

        ordered_eng_rels = sorted(
            eng_rels,
            key=lambda r: (
                r.get("kind") or "",
                r.get("source") or "",
                r.get("target") or "",
                r.get("id") or "",
            ),
        )
        for eng_rel in ordered_eng_rels:
            if (eng_rel.get("kind") or "") != eng_kind:
                continue

            eng_source = eng_rel.get("source") or ""
            eng_target = eng_rel.get("target") or ""
            rel_evidence = list(eng_rel.get("evidence") or [])

            source_map_evidence: List[Dict[str, Any]] = []
            target_map_evidence: List[Dict[str, Any]] = []
            if eng_source in mapping_by_source:
                for m in mapping_by_source[eng_source]:
                    source_map_evidence.extend(m.get("evidence") or [])
            if eng_target in mapping_by_source:
                for m in mapping_by_source[eng_target]:
                    target_map_evidence.extend(m.get("evidence") or [])

            missing_engineering = (
                eng_source not in eng_node_by_id or eng_target not in eng_node_by_id
            )
            arch_source = eng_to_arch.get(eng_source) or ""
            arch_target = eng_to_arch.get(eng_target) or ""

            if missing_engineering or not arch_source or not arch_target:
                # Do not invent architecture endpoints (e.g. concept:workflow:…).
                claim_evidence: List[Dict[str, Any]] = []
                _merge_evidence(claim_evidence, rel_evidence)
                _merge_evidence(claim_evidence, source_map_evidence)
                _merge_evidence(claim_evidence, target_map_evidence)
                unresolved.append(
                    {
                        "source": eng_source,
                        "target": eng_target,
                        "kind": arch_kind,
                        "reason": "missing_endpoint",
                        "evidence": claim_evidence,
                    }
                )
                continue

            if not rel_evidence:
                # canonical architecture relationship requires evidence
                claim_evidence = []
                _merge_evidence(claim_evidence, source_map_evidence)
                _merge_evidence(claim_evidence, target_map_evidence)
                unresolved.append(
                    {
                        "source": eng_source,
                        "target": eng_target,
                        "kind": arch_kind,
                        "reason": "unsupported_relationship",
                        "evidence": claim_evidence,
                    }
                )
                continue

            evidence: List[Dict[str, Any]] = []
            _merge_evidence(evidence, rel_evidence)
            _merge_evidence(evidence, source_map_evidence)
            _merge_evidence(evidence, target_map_evidence)

            rel_id = architecture_relationship_identity(
                arch_kind, arch_source, arch_target
            )
            if rel_id not in arch_relationships:
                arch_relationships[rel_id] = {
                    "id": rel_id,
                    "kind": arch_kind,
                    "source": arch_source,
                    "target": arch_target,
                    "classification": rel_rule.get("classification") or "derived",
                    "evidence": [],
                }
            _merge_evidence(arch_relationships[rel_id]["evidence"], evidence)

    ordered_arch_relationships = [
        arch_relationships[key] for key in sorted(arch_relationships)
    ]
    # deterministic order: kind, source, target, id (same as engineering)
    ordered_arch_relationships.sort(
        key=lambda r: (
            r.get("kind") or "",
            r.get("source") or "",
            r.get("target") or "",
            r.get("id") or "",
        )
    )
    for rel in ordered_arch_relationships:
        rel["evidence"] = sorted(rel.get("evidence") or [], key=_evidence_sort_key)

    ordered_arch_nodes = [
        arch_nodes[key] for key in sorted(arch_nodes)
    ]
    for node in ordered_arch_nodes:
        node["evidence"] = sorted(node.get("evidence") or [], key=_evidence_sort_key)

    ordered_mappings = sorted(
        mappings.values(),
        key=lambda m: (
            m.get("kind") or "",
            m.get("source") or "",
            m.get("target") or "",
            m.get("id") or "",
        ),
    )
    for mapping in ordered_mappings:
        mapping["evidence"] = sorted(
            mapping.get("evidence") or [], key=_evidence_sort_key
        )

    ordered_unresolved = sorted(
        unresolved,
        key=lambda u: (
            str(u.get("kind") or ""),
            str(u.get("source") or ""),
            str(u.get("target") or ""),
            str(u.get("reason") or ""),
        ),
    )
    # de-dupe unresolved
    seen_u = set()
    deduped_u: List[Dict[str, Any]] = []
    for claim in ordered_unresolved:
        key = (
            claim.get("kind"),
            claim.get("source"),
            claim.get("target"),
            claim.get("reason"),
        )
        if key in seen_u:
            for existing in deduped_u:
                if (
                    existing.get("kind"),
                    existing.get("source"),
                    existing.get("target"),
                    existing.get("reason"),
                ) == key:
                    _merge_evidence(existing["evidence"], claim.get("evidence") or [])
                    existing["evidence"] = sorted(
                        existing["evidence"], key=_evidence_sort_key
                    )
                    break
            continue
        seen_u.add(key)
        claim["evidence"] = sorted(
            claim.get("evidence") or [], key=_evidence_sort_key
        )
        deduped_u.append(claim)

    arch_metadata = {
        "construction_rules": rules.get("schema_version"),
        "input": "engineering_model",
        "node_rules_applied": sorted(
            r.get("rule_id") for r in create_node_rules if r.get("rule_id")
        ),
        "mapping_rules_applied": sorted(
            r.get("rule_id") for r in create_mapping_rules if r.get("rule_id")
        ),
        "relationship_rules_applied": sorted(
            r.get("rule_id") for r in create_relationship_rules if r.get("rule_id")
        ),
        "status": "constructed",
    }

    model = EngineeringModel(
        schema_version=engineering_document.get("schema_version")
        or "repository_knowledge_model.v1",
        repository=dict(engineering_document.get("repository") or {}),
        analysis=dict(engineering_document.get("analysis") or {}),
        engineering={
            "nodes": list(eng_nodes),
            "relationships": list(eng_rels),
            "unresolved": list(eng_unresolved),
            "metadata": dict(engineering.get("metadata") or {}),
        },
        architecture={
            "nodes": ordered_arch_nodes,
            "relationships": ordered_arch_relationships,
            "unresolved": deduped_u,
            "metadata": arch_metadata,
        },
        mappings=ordered_mappings,
        metadata=dict(engineering_document.get("metadata") or {}),
    )

    document = model.to_dict()
    validate_knowledge_model_or_raise(
        document, expected_commit_sha=commit, require_architecture=False
    )
    model.metadata = dict(model.metadata or {})
    model.metadata["architecture_counts"] = architecture_counts(
        model.architecture, model.mappings
    )
    document = model.to_dict()
    validate_knowledge_model_or_raise(
        document, expected_commit_sha=commit, require_architecture=False
    )
    return model


def write_architecture_model_artifact(
    model: EngineeringModel, output_root: Path, snapshot_id: str
) -> Path:
    dest = architecture_model_path(output_root, snapshot_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    document = model.to_dict()
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    dest.write_text(text, encoding="utf-8")
    return dest


def create_architecture_model(
    engineering_model_path: Optional[Path] = None,
    rules_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
) -> Tuple[EngineeringModel, Path]:
    """Build ArchitectureModel section from an EngineeringModel artifact."""
    root = resolve_output_root(output_root)

    if snapshot_id is None:
        snapshot_id = PRIMARY_SNAPSHOT_ID

    if engineering_model_path is None:
        eng_path = artifact_engineering_model_path(root, snapshot_id)
    else:
        eng_path = Path(engineering_model_path)

    engineering_document = load_engineering_model_document(eng_path)

    if rules_path is None:
        rules_path = architecture_rules_path(root, snapshot_id)
        if not Path(rules_path).is_file():
            rules_path = architecture_rules_path(
                default_artifacts_root(), PRIMARY_SNAPSHOT_ID
            )

    try:
        rules = load_architecture_rules(Path(rules_path))
    except ArchitectureRulesError as exc:
        raise ArchitectureModelError(str(exc))

    model = build_architecture_model(engineering_document, rules)
    path = write_architecture_model_artifact(model, root, snapshot_id)
    return model, path


__all__ = [
    "ARCHITECTURE_MODEL_FILENAME",
    "ArchitectureModelError",
    "ArchitectureModelValidationError",
    "build_architecture_model",
    "create_architecture_model",
    "load_engineering_model_document",
    "write_architecture_model_artifact",
]
