"""ArchitectureModel domain helpers (repository_knowledge_model.v1 architecture section)."""

from __future__ import annotations

from typing import Any, Dict, List

ARCHITECTURE_RULES_SCHEMA_VERSION = "architecture_construction_rules.v1"
MAPPING_KINDS = ("CONTRIBUTES_TO",)
ARCHITECTURE_NODE_KINDS = ("concept",)
ARCHITECTURE_SUBKINDS = ("agent",)
ARCHITECTURE_RELATIONSHIP_KINDS = ("TRIGGERS",)
CLASSIFICATIONS = ("observed", "derived", "inferred", "conflicting")
UNRESOLVED_REASONS = (
    "target_not_resolved",
    "ambiguous_identity",
    "missing_endpoint",
    "unsupported_relationship",
)


def empty_architecture_section() -> Dict[str, Any]:
    return {
        "nodes": [],
        "relationships": [],
        "unresolved": [],
        "metadata": {},
    }


def mapping_identity(kind: str, source_id: str, target_id: str) -> str:
    """Canonical mapping identity: map:<kind>:<source>:<target>."""
    return "map:{}:{}:{}".format(kind, source_id, target_id)


def architecture_relationship_identity(
    kind: str, source_id: str, target_id: str
) -> str:
    """Canonical architecture relationship identity: rel:<kind>:<source>:<target>.

    Source/target are ArchitectureModel node ids (not engineering ids).
    """
    return "rel:{}:{}:{}".format(kind, source_id, target_id)


def architecture_counts(
    architecture: Dict[str, Any], mappings: List[Dict[str, Any]]
) -> Dict[str, int]:
    nodes = architecture.get("nodes") or []
    rels = architecture.get("relationships") or []
    unresolved = architecture.get("unresolved") or []
    evidence = sum(len(n.get("evidence") or []) for n in nodes)
    evidence += sum(len(r.get("evidence") or []) for r in rels)
    evidence += sum(len(m.get("evidence") or []) for m in mappings)
    evidence += sum(len(u.get("evidence") or []) for u in unresolved)
    return {
        "architecture_nodes": len(nodes),
        "architecture_relationships": len(rels),
        "architecture_rel_triggers": sum(
            1 for r in rels if r.get("kind") == "TRIGGERS"
        ),
        "architecture_unresolved": len(unresolved),
        "architecture_unresolved_triggers": sum(
            1 for u in unresolved if u.get("kind") == "TRIGGERS"
        ),
        "mappings": len(mappings),
        "architecture_evidence": evidence,
        "architecture_nodes_concept_agent": sum(
            1
            for n in nodes
            if n.get("kind") == "concept" and n.get("subkind") == "agent"
        ),
        "mappings_contributes_to": sum(
            1 for m in mappings if m.get("kind") == "CONTRIBUTES_TO"
        ),
    }


__all__ = [
    "ARCHITECTURE_NODE_KINDS",
    "ARCHITECTURE_RELATIONSHIP_KINDS",
    "ARCHITECTURE_RULES_SCHEMA_VERSION",
    "ARCHITECTURE_SUBKINDS",
    "CLASSIFICATIONS",
    "MAPPING_KINDS",
    "UNRESOLVED_REASONS",
    "architecture_counts",
    "architecture_relationship_identity",
    "empty_architecture_section",
    "mapping_identity",
]
