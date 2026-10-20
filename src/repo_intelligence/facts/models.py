"""Domain model for repository_facts.v2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = "repository_facts.v2"

# Deterministic presentation order for known fact types. Unknown types sort last.
FACT_TYPE_ORDER = (
    "file",
    "directory",
    "entity",
    "import",
    "configuration",
    "dependency",
    "artifact",
    "documentation",
    "test",
    "interaction",
    "tool",
    "mcp_server",
)


@dataclass
class SnapshotRef:
    """Provenance link back to the M1 RepositorySnapshot."""

    commit_sha: str
    snapshot_id: str
    repository_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "snapshot_id": self.snapshot_id,
            "repository_url": self.repository_url,
        }


@dataclass
class AnalyzersInfo:
    """Which analyzer suite produced this artifact."""

    version: str
    name: str = "repository-intelligence"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass
class RepositoryFacts:
    """Deterministic inventory of what exists in a repository snapshot.

    Root envelope: schema_version, snapshot, repository, facts, analyzers.

    Single-source rule:
      - ``repository`` = identity/metadata only (name, url)
      - ``facts[]``    = canonical repository observations (files, directories,
        entities, imports, configuration, dependencies, agents, MCP, docs, tests, …)
      - ``counts()``   = derived CLI/reporting summaries computed from facts[]
        (not part of the JSON contract, not an alternate storage)
    """

    schema_version: str = SCHEMA_VERSION
    snapshot: SnapshotRef = field(
        default_factory=lambda: SnapshotRef(commit_sha="", snapshot_id="")
    )
    repository: Dict[str, Any] = field(
        default_factory=lambda: {"name": "", "url": ""}
    )
    facts: List[Dict[str, Any]] = field(default_factory=list)
    analyzers: List[AnalyzersInfo] = field(default_factory=list)

    def facts_of_type(self, fact_type: str) -> List[Dict[str, Any]]:
        return [f for f in self.facts if f.get("fact_type") == fact_type]

    def facts_from_source(self, source: str) -> List[Dict[str, Any]]:
        return [f for f in self.facts if f.get("source") == source]

    def find_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        for fact in self.facts:
            if fact.get("fact_id") == fact_id:
                return fact
        return None

    def counts(self) -> Dict[str, int]:
        """Derived reporting/CLI summaries computed only from ``facts[]``.

        Not part of the RepositoryFacts JSON contract and not canonical
        observation storage. Pipeline::

            facts[]  →  counts()  →  reporting metrics

        Never::

            facts[]  +  separate inventories  →  counts()
        """
        by_type = Counter(f.get("fact_type") or "" for f in self.facts)
        by_source = Counter(f.get("source") or "" for f in self.facts)
        entity_kinds = Counter(
            f.get("kind") or ""
            for f in self.facts
            if f.get("fact_type") == "entity"
        )
        evidence_count = 0
        for fact in self.facts:
            evidence_count += len(fact.get("evidence") or [])
        return {
            "facts": len(self.facts),
            "files": by_type.get("file", 0),
            "directories": by_type.get("directory", 0),
            "entities": by_type.get("entity", 0),
            "python_modules": entity_kinds.get("module", 0),
            "python_symbols": entity_kinds.get("class", 0)
            + entity_kinds.get("function", 0),
            "imports": by_type.get("import", 0),
            "configuration": by_type.get("configuration", 0),
            "dependencies": by_type.get("dependency", 0),
            "agents": by_source.get("agents", 0),
            "mcp": by_source.get("mcp", 0),
            "documentation": by_type.get("documentation", 0),
            "tests": by_type.get("test", 0),
            "interactions": by_type.get("interaction", 0),
            "tools": by_type.get("tool", 0),
            "mcp_servers": by_type.get("mcp_server", 0),
            "evidence": evidence_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.to_dict(),
            "repository": {
                "name": self.repository.get("name", ""),
                "url": self.repository.get("url", ""),
            },
            "facts": [dict(f) for f in self.facts],
            "analyzers": [a.to_dict() for a in self.analyzers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryFacts":
        snapshot = data.get("snapshot", {}) or {}
        analyzers_raw = data.get("analyzers", []) or []
        repository = data.get("repository", {}) or {}
        analyzers: List[AnalyzersInfo] = []
        if isinstance(analyzers_raw, dict):
            # Tolerate a lone object (should not appear in valid v2 docs).
            analyzers_raw = [analyzers_raw]
        for item in analyzers_raw:
            analyzers.append(
                AnalyzersInfo(
                    name=item.get("name", "repository-intelligence"),
                    version=item.get("version", ""),
                )
            )
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            snapshot=SnapshotRef(
                commit_sha=snapshot.get("commit_sha", ""),
                snapshot_id=snapshot.get("snapshot_id", ""),
                repository_url=snapshot.get("repository_url", ""),
            ),
            repository={
                "name": repository.get("name", ""),
                "url": repository.get("url", ""),
            },
            facts=[dict(f) for f in data.get("facts", [])],
            analyzers=analyzers,
        )


def sort_facts(facts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic fact ordering: known types by vocabulary order, then id."""

    def key(fact: Dict[str, Any]):
        fact_type = fact.get("fact_type") or ""
        try:
            type_rank = FACT_TYPE_ORDER.index(fact_type)
        except ValueError:
            type_rank = len(FACT_TYPE_ORDER)
        return (type_rank, fact_type, fact.get("fact_id") or "")

    return sorted(facts, key=key)
