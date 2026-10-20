"""Explicit interaction observations (M2 → candidate TRIGGERS facts).

Detects only explicit trigger statements (e.g. ``Spawned by the Data
Onboarding Agent``, ``Spawn Metadata Agent``, ``agent(metadata)``) and
resolves both endpoints to existing fact_ids. Mere mentions, co-location,
or name similarity never produce an interaction fact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .evidence import make_evidence
from .filesystem import iter_repository_paths

# Explicit-direction statements only (no free-text co-occurrence).
_SPAWNED_BY = re.compile(
    r"[Ss]pawned by ([^\.\n\|\-]+)",
)
# Captures a leading entity even when trailing clause text follows
# (e.g. "the Data Onboarding Agent in Phase 3 (profiling)").
_SPAWN_ENTITY = re.compile(
    r"(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+"
    r"(?:Agent|agent|workflow|Workflow|orchestrator|Orchestrator))\b"
)
_SPAWN_TARGET = re.compile(
    r"\b[Ss]pawn(?:s|ed|ing)?\s+(?!by\b)(?:the\s+)?"
    r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+"
    r"(?:Agent|agent|workflow|Workflow|orchestrator|Orchestrator))\b"
)
_AGENT_CALL = re.compile(
    r"\bagent\(\s*['\"]?([a-z0-9][a-z0-9_\-]*)['\"]?\s*\)"
)
_SELF_ID = re.compile(
    r"\bYou are (?:the\s+)?"
    r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+"
    r"(?:Agent|agent|workflow|Workflow|orchestrator|Orchestrator))\b"
)
_FRONTMATTER_NAME = re.compile(r"(?m)^name:\s*([A-Za-z0-9][A-Za-z0-9_\-]*)\s*$")

# Endpoint facts that can participate in interactions (open kinds under artifact).
_ENDPOINT_KINDS = frozenset(
    {
        "agent_definition",
        "agent_command",
        "agent_config",
        "agent_skill",
    }
)

_TEXT_SUFFIXES = (".md", ".markdown", ".rst", ".adoc", ".txt", ".yml", ".yaml", ".json")
_MAX_READ_BYTES = 1_000_000


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _normalize_alias(value: str) -> str:
    """Casefolded alias key with collapsed whitespace."""
    return " ".join((value or "").strip().split()).casefold()


def _strip_leading_the(value: str) -> str:
    text = (value or "").strip()
    if text.lower().startswith("the "):
        return text[4:].strip()
    return text


def _split_spawned_by_clause(clause: str) -> List[str]:
    """Extract explicit peer entities from a ``Spawned by …`` clause.

    Handles trailing context (``… in Phase 3 (profiling) and Step 4.2``) and
    multi-peer lists (``X or the DevOps workflow``). Only phrases that name
    an Agent/workflow/orchestrator are kept — never bare nouns.
    """
    # Split multi-peer conjunctions first so each segment is scanned alone.
    segments = re.split(r"\s+or\s+|,\s*|\s+and\s+", clause)
    entities: List[str] = []
    seen = set()
    for segment in segments:
        cleaned = _strip_leading_the(segment.strip())
        for match in _SPAWN_ENTITY.finditer(cleaned):
            entity = _strip_leading_the(match.group(1).strip())
            key = _normalize_alias(entity)
            if not entity or key in seen:
                continue
            seen.add(key)
            entities.append(entity)
    return entities


def _candidate_text_paths(root: Path) -> List[str]:
    files, _ = iter_repository_paths(root)
    out: List[str] = []
    for rel in files:
        posix = _posix(rel)
        name = Path(posix).name.lower()
        if name.endswith(_TEXT_SUFFIXES):
            out.append(posix)
    return sorted(out)


def _read_text(path: Path) -> Optional[str]:
    try:
        if not path.is_file():
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > _MAX_READ_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None


def _alias_records(path: str, text: str) -> List[Dict[str, str]]:
    """Explicit self-identity aliases only (frontmatter name + 'You are the X')."""
    aliases: List[Dict[str, str]] = []
    seen = set()

    def add(alias: str) -> None:
        cleaned = " ".join(alias.strip().split())
        if not cleaned:
            return
        key = _normalize_alias(cleaned)
        if not key or key in seen:
            return
        seen.add(key)
        aliases.append({"path": path, "alias": cleaned})

    for match in _FRONTMATTER_NAME.finditer(text):
        add(match.group(1))
        # metadata-agent → also register bare 'metadata' when suffix present.
        raw = match.group(1)
        if raw.endswith("-agent") or raw.endswith("_agent"):
            add(raw[: -len("agent")].rstrip("-_"))
        elif raw.endswith("-workflow") or raw.endswith("_workflow"):
            add(raw[: -len("workflow")].rstrip("-_"))

    for match in _SELF_ID.finditer(text):
        add(match.group(1))

    return aliases


def _statements_for_text(path: str, text: str) -> List[Dict[str, Any]]:
    """Extract explicit trigger statements with 1-based line evidence."""
    statements: List[Dict[str, Any]] = []
    lines = text.splitlines()

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        for match in _SPAWNED_BY.finditer(line):
            for entity in _split_spawned_by_clause(match.group(1)):
                statements.append(
                    {
                        "path": path,
                        "line_start": line_no,
                        "line_end": line_no,
                        "direction": "spawned_by",
                        "peer_label": entity,
                    }
                )

        for match in _SPAWN_TARGET.finditer(line):
            statements.append(
                {
                    "path": path,
                    "line_start": line_no,
                    "line_end": line_no,
                    "direction": "spawns",
                    "peer_label": match.group(1).strip(),
                }
            )

        for match in _AGENT_CALL.finditer(line):
            statements.append(
                {
                    "path": path,
                    "line_start": line_no,
                    "line_end": line_no,
                    "direction": "agent_call",
                    "peer_label": match.group(1).strip(),
                }
            )

    return statements


def analyze_interactions(root: Path) -> Dict[str, Any]:
    """Scan repository text for explicit interaction statements + aliases.

    Returns ``{"statements": [...], "aliases": [...]}``. Pure observation —
    endpoint resolution happens after FactRecords exist.
    """
    root = Path(root)
    statements: List[Dict[str, Any]] = []
    aliases: List[Dict[str, str]] = []

    for rel in _candidate_text_paths(root):
        text = _read_text(root / rel)
        if not text:
            continue
        aliases.extend(_alias_records(rel, text))
        statements.extend(_statements_for_text(rel, text))

    statements.sort(
        key=lambda s: (s.get("path") or "", s.get("line_start") or 0, s.get("direction") or "", s.get("peer_label") or "")
    )
    aliases.sort(key=lambda a: (a.get("path") or "", _normalize_alias(a.get("alias") or "")))
    return {"statements": statements, "aliases": aliases}


def _endpoint_facts(
    facts: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fact in facts:
        if fact.get("fact_type") != "artifact":
            continue
        if fact.get("kind") not in _ENDPOINT_KINDS:
            continue
        attrs = fact.get("attributes") or {}
        path = attrs.get("path")
        if path and fact.get("fact_id"):
            out.append(fact)
    return out


def build_endpoint_registry(
    facts: List[Dict[str, Any]],
    aliases: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Map normalized labels → endpoint fact_ids.

    Priority (first wins; later collisions are ignored rather than made
    ambiguous):

    1. path / stem keys derived from endpoint facts
    2. aliases from ``agent_definition`` paths (self-identity of the agent)
    3. aliases from other endpoint kinds (commands, configs)

    Final map only contains unambiguous single-fact_id keys.
    """
    endpoints = _endpoint_facts(facts)
    by_path: Dict[str, Dict[str, Any]] = {
        str((f.get("attributes") or {}).get("path") or ""): f
        for f in endpoints
    }
    registry: Dict[str, str] = {}

    def register(label: str, fact_id: str) -> None:
        key = _normalize_alias(label)
        if not key or not fact_id:
            return
        # First wins — path/stem and higher-priority aliases lock the key.
        registry.setdefault(key, fact_id)

    for fact in endpoints:
        path = str((fact.get("attributes") or {}).get("path") or "")
        fact_id = str(fact.get("fact_id") or "")
        if not path or not fact_id:
            continue
        register(path, fact_id)
        stem = Path(path).stem
        register(stem, fact_id)
        if stem.endswith("-agent") or stem.endswith("_agent"):
            register(stem[: -len("agent")].rstrip("-_"), fact_id)
        if stem.endswith("-workflow") or stem.endswith("_workflow"):
            register(stem[: -len("workflow")].rstrip("-_"), fact_id)

    def alias_sort_key(item: Dict[str, str]) -> Tuple[int, str, str]:
        path = str(item.get("path") or "")
        fact = by_path.get(path)
        kind = str((fact or {}).get("kind") or "")
        # agent_definition self-ids beat command prompt templates.
        rank = 0 if kind == "agent_definition" else 1
        return (rank, path, _normalize_alias(str(item.get("alias") or "")))

    for item in sorted(aliases, key=alias_sort_key):
        path = str(item.get("path") or "")
        alias = str(item.get("alias") or "")
        fact = by_path.get(path)
        if fact is None or not alias:
            continue
        register(alias, str(fact.get("fact_id") or ""))

    # Rebuild as list values; drop nothing here (first-wins prevents collisions).
    return {key: [fact_id] for key, fact_id in registry.items() if fact_id}


def _resolve_label(
    registry: Dict[str, List[str]], label: str
) -> Optional[str]:
    key = _normalize_alias(label)
    if not key:
        return None
    ids = registry.get(key)
    if ids and len(ids) == 1:
        return ids[0]
    if key.startswith("the "):
        ids = registry.get(key[4:])
        if ids and len(ids) == 1:
            return ids[0]
    return None


def _evidence_artifact_type(path: str, facts: List[Dict[str, Any]]) -> str:
    """Map evidence path to controlled ARTIFACT_TYPES vocabulary."""
    for fact in facts:
        if fact.get("fact_type") != "artifact":
            continue
        attrs = fact.get("attributes") or {}
        if attrs.get("path") != path:
            continue
        kind = fact.get("kind")
        if kind in ("agent_definition", "agent_command", "agent_config", "agent_skill"):
            return "agent_definition"
        if kind in ("mcp_config", "mcp_server_source", "mcp_documentation"):
            return "mcp_definition"
        if kind in ("python",):
            return "python"
        if kind in ("markdown", "shell"):
            return "file"
    for fact in facts:
        if fact.get("fact_type") in ("documentation", "test", "configuration"):
            attrs = fact.get("attributes") or {}
            if attrs.get("path") == path:
                return {
                    "documentation": "documentation",
                    "test": "test",
                    "configuration": "configuration",
                }[fact.get("fact_type")]
    if path.endswith(".py"):
        return "python"
    return "file"


def _fact_ids_by_path(facts: List[Dict[str, Any]]) -> Dict[str, str]:
    """Primary fact_id for a path (endpoint artifacts preferred)."""
    preferred: Dict[str, str] = {}
    fallback: Dict[str, str] = {}
    for fact in facts:
        fact_id = fact.get("fact_id")
        attrs = fact.get("attributes") or {}
        path = attrs.get("path")
        if not path or not fact_id:
            continue
        if fact.get("fact_type") == "artifact" and fact.get("kind") in _ENDPOINT_KINDS:
            preferred[path] = fact_id
        else:
            fallback.setdefault(path, fact_id)
    out = dict(fallback)
    out.update(preferred)
    return out


def build_interaction_facts(
    facts: List[Dict[str, Any]],
    interaction_analysis: Dict[str, Any],
    commit: str,
) -> List[Dict[str, Any]]:
    """Resolve explicit statements into interaction FactRecords.

    Emits ``fact_type=interaction``, ``kind=triggers`` only when both
    endpoints resolve to existing fact_ids. Unresolved / ambiguous /
    non-explicit cases are skipped (no fabricated edges).
    """
    statements = list((interaction_analysis or {}).get("statements") or [])
    aliases = list((interaction_analysis or {}).get("aliases") or [])
    if not statements:
        return []

    registry = build_endpoint_registry(facts, aliases)
    by_path = _fact_ids_by_path(facts)
    endpoint_paths = {
        str((f.get("attributes") or {}).get("path") or "")
        for f in _endpoint_facts(facts)
    }
    fact_id_set = {f.get("fact_id") for f in facts if f.get("fact_id")}

    # Deterministic order regardless of analyzer stability.
    statements = sorted(
        statements,
        key=lambda s: (
            s.get("path") or "",
            int(s.get("line_start") or 0),
            s.get("direction") or "",
            s.get("peer_label") or "",
        ),
    )

    seen_pairs = set()
    out: List[Dict[str, Any]] = []

    for stmt in statements:
        path = str(stmt.get("path") or "")
        direction = str(stmt.get("direction") or "")
        peer_label = str(stmt.get("peer_label") or "")
        line_start = stmt.get("line_start")
        line_end = stmt.get("line_end") or line_start

        if not path or not peer_label:
            continue

        peer_fact_id = _resolve_label(registry, peer_label)
        if peer_fact_id is None:
            continue

        if direction == "spawned_by":
            # This agent/command artifact is spawned by peer → peer triggers it.
            # Require an endpoint target so prose about embedded skills/docs
            # does not become a TRIGGERS edge on a bare file fact.
            if path not in endpoint_paths:
                continue
            source_fact_id = peer_fact_id
            target_fact_id = by_path.get(path)
        elif direction in ("spawns", "agent_call"):
            # This artifact spawns/calls peer → this artifact triggers peer.
            source_fact_id = by_path.get(path)
            target_fact_id = peer_fact_id
        else:
            continue

        if not source_fact_id or not target_fact_id:
            continue
        if source_fact_id not in fact_id_set or target_fact_id not in fact_id_set:
            continue
        if source_fact_id == target_fact_id:
            continue

        pair = (source_fact_id, target_fact_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        source_path = _path_for_fact_id(facts, source_fact_id)
        target_path = _path_for_fact_id(facts, target_fact_id)
        artifact_type = _evidence_artifact_type(path, facts)
        evidence = [
            make_evidence(
                path=path,
                artifact_type=artifact_type,
                commit=commit,
                line_start=int(line_start) if line_start else None,
                line_end=int(line_end) if line_end else None,
                symbol=None,
            )
        ]

        from .records import make_fact  # local import avoids cycle at module load

        out.append(
            make_fact(
                fact_type="interaction",
                kind="triggers",
                source="interactions",
                attributes={
                    "source_fact_id": source_fact_id,
                    "target_fact_id": target_fact_id,
                    "source_path": source_path,
                    "target_path": target_path,
                    "relation": "triggers",
                    "statement": direction,
                },
                evidence=evidence,
            )
        )

    return out


def _path_for_fact_id(facts: List[Dict[str, Any]], fact_id: str) -> str:
    for fact in facts:
        if fact.get("fact_id") == fact_id:
            return str((fact.get("attributes") or {}).get("path") or "")
    return ""


__all__ = [
    "analyze_interactions",
    "build_endpoint_registry",
    "build_interaction_facts",
]
