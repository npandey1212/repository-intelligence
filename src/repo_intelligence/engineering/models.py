"""EngineeringModel domain model (repository_knowledge_model.v1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

SCHEMA_VERSION = "repository_knowledge_model.v1"

# Canonical relationship vocabulary. The M12 knowledge slice adds the
# forward Agent -> Tool -> MCP Server -> MCP Configuration path kinds
# (DECLARES, EXPOSED_BY, CONFIGURED_BY); everything else is M1-M11.
RELATIONSHIP_KINDS = (
    "CONTAINS",
    "IMPORTS",
    "DEPENDS_ON",
    "TRIGGERS",
    "DECLARES",
    "EXPOSED_BY",
    "CONFIGURED_BY",
)
CLASSIFICATIONS = ("observed", "derived", "inferred", "conflicting")
UNRESOLVED_REASONS = (
    "target_not_resolved",
    "ambiguous_identity",
    "missing_endpoint",
    "unsupported_relationship",
)


@dataclass
class EngineeringModel:
    """Deterministic engineering knowledge model.

    V1 populates ``engineering`` only. ``architecture`` and ``mappings``
    remain empty placeholders for later milestones.
    """

    schema_version: str = SCHEMA_VERSION
    repository: Dict[str, Any] = field(
        default_factory=lambda: {"name": "", "url": "", "commit_sha": ""}
    )
    analysis: Dict[str, Any] = field(
        default_factory=lambda: {"strategy": "deterministic", "analyzers": []}
    )
    engineering: Dict[str, Any] = field(
        default_factory=lambda: {
            "nodes": [],
            "relationships": [],
            "unresolved": [],
            "metadata": {},
        }
    )
    architecture: Dict[str, Any] = field(
        default_factory=lambda: {
            "nodes": [],
            "relationships": [],
            "unresolved": [],
            "metadata": {},
        }
    )
    mappings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        return self.engineering.get("nodes") or []

    @property
    def relationships(self) -> List[Dict[str, Any]]:
        return self.engineering.get("relationships") or []

    @property
    def unresolved(self) -> List[Dict[str, Any]]:
        return self.engineering.get("unresolved") or []

    def counts(self) -> Dict[str, int]:
        nodes = self.nodes
        rels = self.relationships
        unresolved = self.unresolved
        evidence = sum(len(n.get("evidence") or []) for n in nodes)
        evidence += sum(len(r.get("evidence") or []) for r in rels)
        evidence += sum(len(u.get("evidence") or []) for u in unresolved)
        return {
            "nodes": len(nodes),
            "relationships": len(rels),
            "unresolved": len(unresolved),
            "evidence": evidence,
            "nodes_artifact": sum(
                1 for n in nodes if n.get("kind") == "artifact"
            ),
            "nodes_code_entity": sum(
                1 for n in nodes if n.get("kind") == "code_entity"
            ),
            "nodes_external_package": sum(
                1 for n in nodes if n.get("kind") == "external_package"
            ),
            "nodes_tool": sum(1 for n in nodes if n.get("kind") == "tool"),
            "nodes_mcp_server": sum(
                1 for n in nodes if n.get("kind") == "mcp_server"
            ),
            "rel_contains": sum(1 for r in rels if r.get("kind") == "CONTAINS"),
            "rel_imports": sum(1 for r in rels if r.get("kind") == "IMPORTS"),
            "rel_depends_on": sum(
                1 for r in rels if r.get("kind") == "DEPENDS_ON"
            ),
            "rel_triggers": sum(1 for r in rels if r.get("kind") == "TRIGGERS"),
            "rel_declares": sum(1 for r in rels if r.get("kind") == "DECLARES"),
            "rel_exposed_by": sum(
                1 for r in rels if r.get("kind") == "EXPOSED_BY"
            ),
            "rel_configured_by": sum(
                1 for r in rels if r.get("kind") == "CONFIGURED_BY"
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": dict(self.repository),
            "analysis": {
                "strategy": self.analysis.get("strategy", "deterministic"),
                "analyzers": list(self.analysis.get("analyzers") or []),
            },
            "engineering": {
                "nodes": list(self.engineering.get("nodes") or []),
                "relationships": list(
                    self.engineering.get("relationships") or []
                ),
                "unresolved": list(self.engineering.get("unresolved") or []),
                "metadata": dict(self.engineering.get("metadata") or {}),
            },
            "architecture": {
                "nodes": list(self.architecture.get("nodes") or []),
                "relationships": list(
                    self.architecture.get("relationships") or []
                ),
                "unresolved": list(self.architecture.get("unresolved") or []),
                "metadata": dict(self.architecture.get("metadata") or {}),
            },
            "mappings": list(self.mappings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "EngineeringModel":
        return cls(
            schema_version=document.get("schema_version") or SCHEMA_VERSION,
            repository=dict(document.get("repository") or {}),
            analysis=dict(document.get("analysis") or {}),
            engineering=dict(document.get("engineering") or {}),
            architecture=dict(document.get("architecture") or {}),
            mappings=list(document.get("mappings") or []),
            metadata=dict(document.get("metadata") or {}),
        )


def empty_graph_section() -> Dict[str, Any]:
    return {"nodes": [], "relationships": [], "unresolved": [], "metadata": {}}


__all__ = [
    "CLASSIFICATIONS",
    "RELATIONSHIP_KINDS",
    "SCHEMA_VERSION",
    "UNRESOLVED_REASONS",
    "EngineeringModel",
    "empty_graph_section",
]
