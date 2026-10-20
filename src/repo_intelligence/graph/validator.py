"""M5 diagram projection validation (semantic faithfulness to ArchitectureModel)."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from .models import DiagramProjection


class DiagramError(Exception):
    """Raised when M5 diagram runtime construction fails."""


class DiagramValidationError(DiagramError):
    """Raised when a DiagramProjection is not a faithful architecture projection."""


def _architecture_section(
    source: Union[Dict[str, Any], Any]
) -> Dict[str, Any]:
    if isinstance(source, dict):
        architecture = source.get("architecture")
        if architecture is None and "nodes" in source:
            architecture = source
        if architecture is None:
            raise DiagramError(
                "Expected ArchitectureModel document with an architecture section"
            )
        return architecture
    architecture = getattr(source, "architecture", None)
    if not isinstance(architecture, dict):
        raise DiagramError(
            "Expected ArchitectureModel document with an architecture section"
        )
    return architecture


def validate_diagram_projection(
    projection: DiagramProjection, architecture_model: Union[Dict[str, Any], Any]
) -> List[str]:
    """Return problems proving the projection mirrors ArchitectureModel exactly."""
    problems: List[str] = []
    architecture = _architecture_section(architecture_model)
    arch_nodes = list(architecture.get("nodes") or [])
    arch_rels = list(architecture.get("relationships") or [])
    arch_unresolved = list(architecture.get("unresolved") or [])

    arch_node_ids = [n.get("id") for n in arch_nodes]
    arch_node_id_set = set(arch_node_ids)
    arch_rel_ids = [r.get("id") for r in arch_rels]
    arch_rel_id_set = set(arch_rel_ids)

    proj_nodes = list(projection.nodes or [])
    proj_rels = list(projection.relationships or [])

    proj_arch_node_ids = [n.get("architecture_node_id") for n in proj_nodes]
    proj_diagram_ids = [n.get("diagram_id") for n in proj_nodes]
    proj_rel_arch_ids = [r.get("architecture_relationship_id") for r in proj_rels]

    # 1–2 / 7: every architecture node ↔ exactly one diagram node
    if len(proj_nodes) != len(arch_nodes):
        problems.append(
            "node_count_mismatch: projection={} architecture={}".format(
                len(proj_nodes), len(arch_nodes)
            )
        )
    for node_id in sorted(arch_node_id_set):
        if proj_arch_node_ids.count(node_id) != 1:
            problems.append(
                "architecture_node_not_exactly_once: {!r} count={}".format(
                    node_id, proj_arch_node_ids.count(node_id)
                )
            )
    for node_id in sorted({x for x in proj_arch_node_ids if x is not None}):
        if node_id not in arch_node_id_set:
            problems.append("invented_node_reference: {!r}".format(node_id))
    if len(set(proj_diagram_ids)) != len(proj_diagram_ids):
        problems.append("duplicate_diagram_ids")
    if any(not d for d in proj_diagram_ids):
        problems.append("missing_diagram_id")

    # labels must come from ArchitectureModel name (or id fallback)
    name_by_id = {
        n.get("id"): (n.get("attributes") or {}).get("name") or n.get("id")
        for n in arch_nodes
    }
    for node in proj_nodes:
        arch_id = node.get("architecture_node_id")
        if arch_id in name_by_id and node.get("label") != name_by_id[arch_id]:
            problems.append(
                "node_label_mismatch: {!r} expected={!r} got={!r}".format(
                    arch_id, name_by_id[arch_id], node.get("label")
                )
            )

    # 3–5 / 8: every architecture relationship ↔ exactly one diagram relationship
    if len(proj_rels) != len(arch_rels):
        problems.append(
            "relationship_count_mismatch: projection={} architecture={}".format(
                len(proj_rels), len(arch_rels)
            )
        )
    for rel_id in sorted(arch_rel_id_set):
        if proj_rel_arch_ids.count(rel_id) != 1:
            problems.append(
                "architecture_relationship_not_exactly_once: {!r} count={}".format(
                    rel_id, proj_rel_arch_ids.count(rel_id)
                )
            )
    for rel_id in sorted({x for x in proj_rel_arch_ids if x is not None}):
        if rel_id not in arch_rel_id_set:
            problems.append(
                "invented_relationship_reference: {!r}".format(rel_id)
            )

    diagram_id_by_arch = {
        n.get("architecture_node_id"): n.get("diagram_id") for n in proj_nodes
    }
    kind_by_rel = {r.get("id"): r.get("kind") for r in arch_rels}
    endpoints_by_rel = {
        r.get("id"): (r.get("source"), r.get("target")) for r in arch_rels
    }

    for rel in proj_rels:
        rel_id = rel.get("architecture_relationship_id")
        source_diag = rel.get("source_diagram_id")
        target_diag = rel.get("target_diagram_id")
        if source_diag not in set(proj_diagram_ids) or target_diag not in set(
            proj_diagram_ids
        ):
            problems.append(
                "invalid_diagram_endpoint: {!r} -> {!r}".format(
                    source_diag, target_diag
                )
            )
        if rel_id in kind_by_rel and rel.get("label") != kind_by_rel[rel_id]:
            problems.append(
                "relationship_label_mismatch: {!r} expected={!r} got={!r}".format(
                    rel_id, kind_by_rel[rel_id], rel.get("label")
                )
            )
        if rel_id in endpoints_by_rel:
            source_arch, target_arch = endpoints_by_rel[rel_id]
            if diagram_id_by_arch.get(source_arch) != source_diag:
                problems.append(
                    "relationship_source_endpoint_mismatch: {!r}".format(rel_id)
                )
            if diagram_id_by_arch.get(target_arch) != target_diag:
                problems.append(
                    "relationship_target_endpoint_mismatch: {!r}".format(rel_id)
                )

    # 6: never project unresolved claims
    unresolved_ids = {
        u.get("id")
        for u in arch_unresolved
        if u.get("id")
    }
    for rel_id in proj_rel_arch_ids:
        if rel_id in unresolved_ids:
            problems.append(
                "unresolved_claim_projected: {!r}".format(rel_id)
            )
    # unresolved claims typically lack ids — match on source/target/kind too
    unresolved_keys = {
        (u.get("kind"), u.get("source"), u.get("target"))
        for u in arch_unresolved
    }
    arch_key_by_rel = {
        r.get("id"): (r.get("kind"), r.get("source"), r.get("target"))
        for r in arch_rels
    }
    for rel in proj_rels:
        key = arch_key_by_rel.get(rel.get("architecture_relationship_id"))
        if key is not None and key in unresolved_keys:
            problems.append(
                "unresolved_claim_projected: {!r}".format(
                    rel.get("architecture_relationship_id")
                )
            )
    # any projected pair that is not a canonical relationship is invented
    for rel in proj_rels:
        if rel.get("architecture_relationship_id") not in arch_rel_id_set:
            # already reported as invented_relationship_reference
            continue

    return problems


def validate_diagram_projection_or_raise(
    projection: DiagramProjection, architecture_model: Union[Dict[str, Any], Any]
) -> None:
    problems = validate_diagram_projection(projection, architecture_model)
    if problems:
        raise DiagramValidationError(
            "DiagramProjection failed validation:\n  - "
            + "\n  - ".join(problems)
        )


__all__ = [
    "DiagramError",
    "DiagramValidationError",
    "validate_diagram_projection",
    "validate_diagram_projection_or_raise",
]
