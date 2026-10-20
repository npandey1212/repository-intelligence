"""M11.2 — Deterministic query API over the engineering graph.

A small, provider-independent, repository-independent interface over
the ``networkx.MultiDiGraph`` produced by ``build_engineering_graph``
— the controlled surface the agent-facing tool (and any future
consumer) uses INSTEAD of raw NetworkX operations.

Two modes of the single entry point:

    query_engineering_graph(graph, start_node_id=...)
        → bounded neighborhood traversal (direction, relationship-kind
          and node-kind filters, depth limit, hard result caps)

    query_engineering_graph(graph)
        → node-kind summary (what kinds/subkinds the model contains,
          counts + bounded id samples) — the discovery path for
          choosing a start_node_id

Named query API (the small learnable surface over the same
implementation; the agent-facing tool builds on it):

    get_node(graph, node_id)                     → node record | None
    neighbors(graph, node_id, direction, ...)    → adjacent node records
    relationships(graph, node_id, direction, ...)→ incident relationships
    find_nodes(graph, kind=...)                  → nodes by kind/kind-subkind
    find_nodes_by_property(graph, name, value)   → nodes by exact property
    traverse(graph, node_id, ...)                → bounded walk (payload of
                                                   query_engineering_graph)

Safety properties (enforced here, not by convention):

* read-only — the graph is never mutated; no repository access; no
  LLM calls; no code evaluation of any kind;
* bounded — ``max_depth`` ≤ MAX_QUERY_DEPTH and hard caps on visited
  nodes / returned relationships (oversized walks stop early and are
  reported via ``truncated``);
* deterministic — same graph + same parameters → byte-identical
  ``to_dict()`` (sorted nodes, sorted relationships, sorted kinds);
* vocabulary-bound — relationship filters must be inside the existing
  EngineeringModel vocabulary (CONTAINS, IMPORTS,_DEPENDS_ON,
  TRIGGERS); node-kind filters use the generic ``kind`` /
  ``kind/subkind`` representation of the model, never specific to any
  names.

Failures raise ``GraphQueryError`` with a stable machine-readable
``code`` (INVALID_ARGUMENT / NODE_NOT_FOUND) — the tool layer
converts them into structured ToolResult failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ..engineering.models import RELATIONSHIP_KINDS

QUERY_DIRECTIONS = ("out", "in", "both")
DEFAULT_DIRECTION = "out"
DEFAULT_QUERY_DEPTH = 1
MAX_QUERY_DEPTH = 6
MAX_QUERY_NODES = 100
MAX_QUERY_RELATIONSHIPS = 200
SUMMARY_SAMPLE_LIMIT = 5

MODE_TRAVERSAL = "traversal"
MODE_SUMMARY = "summary"


class GraphQueryError(Exception):
    """Deterministic query failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class EngineeringGraphQueryResult:
    """Structured, JSON-ready outcome of one graph query."""

    mode: str
    query: Dict[str, Any] = field(default_factory=dict)
    start_node: Optional[Dict[str, Any]] = None
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    kinds: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "query": dict(self.query),
            "start_node": dict(self.start_node) if self.start_node else None,
            "nodes": [dict(n) for n in self.nodes],
            "relationships": [dict(r) for r in self.relationships],
            "evidence": [dict(e) for e in self.evidence],
            "kinds": [dict(k) for k in self.kinds],
            "counts": dict(self.counts),
            "truncated": bool(self.truncated),
        }


# ---------------------------------------------------------------------------
# Input validation (deterministic failures, never silent coercion)
# ---------------------------------------------------------------------------


def _invalid(message: str) -> GraphQueryError:
    return GraphQueryError("INVALID_ARGUMENT", message)


def _validate_graph(graph: Any) -> None:
    if not isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        raise _invalid(
            "graph must be a NetworkX DiGraph/MultiDiGraph built by "
            "build_engineering_graph."
        )


def _validate_direction(direction: Any) -> str:
    if not isinstance(direction, str) or direction not in QUERY_DIRECTIONS:
        raise _invalid(
            "direction must be one of {}; got {!r}.".format(
                ", ".join(QUERY_DIRECTIONS), direction
            )
        )
    return direction


def _validate_max_depth(max_depth: Any) -> int:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise _invalid(
            "max_depth must be an integer between 0 and {}; got {!r}.".format(
                MAX_QUERY_DEPTH, max_depth
            )
        )
    if max_depth < 0 or max_depth > MAX_QUERY_DEPTH:
        raise _invalid(
            "max_depth must be between 0 and {}; got {}.".format(
                MAX_QUERY_DEPTH, max_depth
            )
        )
    return max_depth


def _validate_relationship_kinds(relationship_kinds: Any) -> Optional[List[str]]:
    if relationship_kinds is None:
        return None
    if not isinstance(relationship_kinds, (list, tuple)):
        raise _invalid(
            "relationship_kinds must be a list of relationship kinds or null."
        )
    selected: List[str] = []
    for item in relationship_kinds:
        if not isinstance(item, str) or item not in RELATIONSHIP_KINDS:
            raise _invalid(
                "unknown relationship kind {!r}; allowed: {}.".format(
                    item, ", ".join(RELATIONSHIP_KINDS)
                )
            )
        if item not in selected:
            selected.append(item)
    return selected or None


def _validate_node_kinds(node_kinds: Any) -> Optional[List[Tuple[str, Optional[str]]]]:
    """Node-kind filters are generic ``kind`` or ``kind/subkind``
    strings — matched exactly against the model's own node fields
    (never a hard-coded vocabulary)."""
    if node_kinds is None:
        return None
    if not isinstance(node_kinds, (list, tuple)):
        raise _invalid(
            "node_kinds must be a list of 'kind' or 'kind/subkind' strings "
            "or null."
        )
    filters: List[Tuple[str, Optional[str]]] = []
    for item in node_kinds:
        if not isinstance(item, str) or not item.strip():
            raise _invalid(
                "node_kinds entries must be non-empty strings; got {!r}.".format(
                    item
                )
            )
        parts = item.strip().split("/")
        if len(parts) > 2 or any(not part for part in parts):
            raise _invalid(
                "node_kinds entries must be 'kind' or 'kind/subkind'; "
                "got {!r}.".format(item)
            )
        pair = (parts[0], parts[1] if len(parts) == 2 else None)
        if pair not in filters:
            filters.append(pair)
    return filters or None


def _node_matches(
    kind: Any,
    subkind: Any,
    filters: Optional[List[Tuple[str, Optional[str]]]],
) -> bool:
    if not filters:
        return True
    for want_kind, want_subkind in filters:
        if kind == want_kind and (want_subkind is None or subkind == want_subkind):
            return True
    return False


# ---------------------------------------------------------------------------
# Projections (the same shape for every result entry)
# ---------------------------------------------------------------------------


def _node_entry(graph: nx.Graph, node_id: str, depth: int) -> Dict[str, Any]:
    data = graph.nodes[node_id]
    return {
        "id": node_id,
        "depth": depth,
        "kind": data.get("kind"),
        "subkind": data.get("subkind"),
        "classification": data.get("classification"),
        "attributes": dict(data.get("attributes") or {}),
        "evidence": [dict(e) for e in data.get("evidence") or []],
    }


def _relationship_entry(data: Dict[str, Any], depth: int) -> Dict[str, Any]:
    return {
        "id": data.get("id"),
        "kind": data.get("kind"),
        "source": data.get("source"),
        "target": data.get("target"),
        "classification": data.get("classification"),
        "depth": depth,
        "evidence": [dict(e) for e in data.get("evidence") or []],
    }


def _aggregate_evidence(
    nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Flat, annotated evidence view: every evidence entry of the
    returned nodes (id order) followed by every evidence entry of the
    returned relationships (depth/id order), each carrying its origin
    entity. Evidence is copied from the model — never regenerated,
    never upgraded (classifications stay exactly as recorded)."""
    evidence: List[Dict[str, Any]] = []
    for entry in nodes:
        for index, item in enumerate(entry["evidence"]):
            annotated = dict(item)
            annotated["origin"] = "node:{}".format(entry["id"])
            annotated["index"] = index
            evidence.append(annotated)
    for entry in relationships:
        for index, item in enumerate(entry["evidence"]):
            annotated = dict(item)
            annotated["origin"] = "relationship:{}".format(entry["id"])
            annotated["index"] = index
            evidence.append(annotated)
    return evidence


def _incidents(
    graph: nx.MultiDiGraph, node_id: str, direction: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """Edges visible from ``node_id`` in ``direction`` as
    ``(neighbor_id, edge_data)`` — deterministic (sorted by edge id)."""
    seen: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    if direction in ("out", "both"):
        for _, neighbor, _key, data in graph.out_edges(node_id, keys=True, data=True):
            seen[data.get("id") or ""] = (neighbor, data)
    if direction in ("in", "both"):
        for predecessor, _edge_source, _key, data in graph.in_edges(
            node_id, keys=True, data=True
        ):
            seen.setdefault(
                data.get("id") or "", (predecessor, data)
            )
    return [seen[key] for key in sorted(seen)]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _summary(
    graph: nx.MultiDiGraph, query: Dict[str, Any]
) -> EngineeringGraphQueryResult:
    """Node-kind census: counts + bounded id samples per kind/subkind,
    sorted by (kind, subkind) — how a caller discovers valid
    start_node_ids without guessing."""
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for node_id in sorted(graph.nodes):
        data = graph.nodes[node_id]
        key = (str(data.get("kind")), str(data.get("subkind")))
        bucket = buckets.setdefault(
            key,
            {
                "kind": key[0],
                "subkind": key[1],
                "node_count": 0,
                "sample_node_ids": [],
            },
        )
        bucket["node_count"] += 1
        if len(bucket["sample_node_ids"]) < SUMMARY_SAMPLE_LIMIT:
            bucket["sample_node_ids"].append(node_id)
    kinds = [buckets[key] for key in sorted(buckets)]
    return EngineeringGraphQueryResult(
        mode=MODE_SUMMARY,
        query=query,
        start_node=None,
        nodes=[],
        relationships=[],
        evidence=[],
        kinds=kinds,
        counts={
            "nodes": graph.number_of_nodes(),
            "relationships": graph.number_of_edges(),
            "unresolved": int(graph.graph.get("unresolved_count") or 0),
            "kinds": len(kinds),
        },
        truncated=False,
    )


def _traversal(
    graph: nx.MultiDiGraph,
    start_node_id: str,
    direction: str,
    relationship_kinds: Optional[List[str]],
    node_filters: Optional[List[Tuple[str, Optional[str]]]],
    max_depth: int,
    query: Dict[str, Any],
) -> EngineeringGraphQueryResult:
    """Bounded breadth-first neighborhood walk.

    Semantics (documented, tested):

    * the start node is always returned (anchor), whatever the
      node-kind filter says;
    * node_kinds limits the walk itself — nodes that do not match are
      neither visited, recorded, nor traversed through (so results
      stay a connected subgraph; use a kind-level filter such as
      ``"artifact"`` to cross subkinds);
    * relationship_kinds limits which edges are followed/recorded;
    * relationship ``depth`` = the deeper of its two endpoint depths;
    * caps: ≤ MAX_QUERY_NODES visited, ≤ MAX_QUERY_RELATIONSHIPS
      recorded — hitting either stops the walk and sets ``truncated``.
    """
    if start_node_id not in graph:
        raise GraphQueryError(
            "NODE_NOT_FOUND",
            "no node with id {!r} in the engineering graph; run the "
            "node-kind summary (omit start_node_id) to discover ids.".format(
                start_node_id
            ),
        )

    depths: Dict[str, int] = {start_node_id: 0}
    relationships_out: Dict[str, Dict[str, Any]] = {}
    truncated = False
    frontier = [start_node_id]
    level = 0

    while frontier and level < max_depth:
        next_frontier: List[str] = []
        for current in sorted(frontier):
            for neighbor, data in _incidents(graph, current, direction):
                kind = data.get("kind")
                if relationship_kinds is not None and kind not in relationship_kinds:
                    continue
                relationship_id = data.get("id")
                if relationship_id in relationships_out:
                    continue  # same edge seen from both endpoints
                neighbor_visited = neighbor in depths
                if not neighbor_visited:
                    neighbor_data = graph.nodes[neighbor]
                    if not _node_matches(
                        neighbor_data.get("kind"),
                        neighbor_data.get("subkind"),
                        node_filters,
                    ):
                        continue  # filter: not visited, not recorded
                    if len(depths) >= MAX_QUERY_NODES:
                        truncated = True
                        continue
                if len(relationships_out) >= MAX_QUERY_RELATIONSHIPS:
                    truncated = True
                    continue
                depth = max(depths[current], depths.get(neighbor, level + 1))
                relationships_out[relationship_id] = _relationship_entry(
                    data, depth
                )
                if not neighbor_visited:
                    depths[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
        level += 1

    node_entries = [
        _node_entry(graph, node_id, depths[node_id]) for node_id in sorted(depths)
    ]
    relationship_entries = sorted(
        relationships_out.values(), key=lambda r: (r["depth"], r["id"] or "")
    )
    evidence = _aggregate_evidence(node_entries, relationship_entries)
    return EngineeringGraphQueryResult(
        mode=MODE_TRAVERSAL,
        query=query,
        start_node=_node_entry(graph, start_node_id, 0),
        nodes=node_entries,
        relationships=relationship_entries,
        evidence=evidence,
        kinds=[],
        counts={
            "nodes": len(node_entries),
            "relationships": len(relationship_entries),
            "evidence": len(evidence),
        },
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_engineering_graph(
    graph: nx.MultiDiGraph,
    start_node_id: Optional[str] = None,
    direction: str = DEFAULT_DIRECTION,
    relationship_kinds: Optional[List[str]] = None,
    node_kinds: Optional[List[str]] = None,
    max_depth: int = DEFAULT_QUERY_DEPTH,
) -> EngineeringGraphQueryResult:
    """Run one deterministic, bounded, read-only graph query.

    With ``start_node_id``: traverse the neighborhood (``direction``:
    ``out`` | ``in`` | ``both``) up to ``max_depth`` hops, optionally
    filtered by ``relationship_kinds`` (EngineeringModel vocabulary)
    and ``node_kinds`` (``kind`` or ``kind/subkind``). Without it:
    return the node-kind summary used to discover ids.

    Raises ``GraphQueryError`` (code ``INVALID_ARGUMENT`` or
    ``NODE_NOT_FOUND``) for invalid input or an unknown start node.
    Never mutates ``graph``.
    """
    _validate_graph(graph)
    validated_direction = _validate_direction(direction)
    validated_depth = _validate_max_depth(max_depth)
    validated_relationship_kinds = _validate_relationship_kinds(relationship_kinds)
    validated_node_kinds = _validate_node_kinds(node_kinds)

    if start_node_id is not None:
        if not isinstance(start_node_id, str) or not start_node_id.strip():
            raise _invalid(
                "start_node_id must be a non-empty string; got {!r}.".format(
                    start_node_id
                )
            )
        start_node_id = start_node_id.strip()

    query = {
        "start_node_id": start_node_id,
        "direction": validated_direction,
        "relationship_kinds": (
            list(validated_relationship_kinds)
            if validated_relationship_kinds is not None
            else None
        ),
        "node_kinds": (
            ["/".join(x for x in pair if x) for pair in validated_node_kinds]
            if validated_node_kinds is not None
            else None
        ),
        "max_depth": validated_depth,
    }

    if start_node_id is None:
        # Summary mode takes NO traversal parameters — explicit input
        # is never silently ignored.
        if (
            validated_relationship_kinds is not None
            or validated_node_kinds is not None
            or validated_direction != DEFAULT_DIRECTION
            or validated_depth != DEFAULT_QUERY_DEPTH
        ):
            raise _invalid(
                "the node-kind summary accepts no traversal parameters; "
                "omit them along with start_node_id."
            )
        query = {
            "start_node_id": None,
            "direction": None,
            "relationship_kinds": None,
            "node_kinds": None,
            "max_depth": None,
        }
        return _summary(graph, query)

    if not isinstance(start_node_id, str) or not start_node_id.strip():
        raise _invalid(
            "start_node_id must be a non-empty string; got {!r}.".format(
                start_node_id
            )
        )
    return _traversal(
        graph,
        start_node_id,
        validated_direction,
        validated_relationship_kinds,
        validated_node_kinds,
        validated_depth,
        query,
    )


# ---------------------------------------------------------------------------
# Named query API (per-operation façade over the same implementation)
# ---------------------------------------------------------------------------


def _node_id_argument(node_id: Any) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise _invalid(
            "node_id must be a non-empty string; got {!r}.".format(node_id)
        )
    return node_id.strip()


def _require_node(graph: nx.MultiDiGraph, node_id: Any) -> str:
    node_id = _node_id_argument(node_id)
    if node_id not in graph:
        raise GraphQueryError(
            "NODE_NOT_FOUND",
            "no node with id {!r} in the engineering graph; run "
            "find_nodes() or the node-kind summary to discover ids.".format(
                node_id
            ),
        )
    return node_id


def get_node(graph: nx.MultiDiGraph, node_id: Any) -> Optional[Dict[str, Any]]:
    """Node record for ``node_id`` (entry shape, ``depth`` 0), or
    ``None`` when the id is not in the graph — a plain lookup does not
    raise for an unknown id."""
    _validate_graph(graph)
    resolved = _node_id_argument(node_id)
    if resolved not in graph:
        return None
    return _node_entry(graph, resolved, 0)


def neighbors(
    graph: nx.MultiDiGraph,
    node_id: Any,
    direction: str = DEFAULT_DIRECTION,
    relationship_kinds: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Unique adjacent node records in ``direction`` (``depth`` 1),
    sorted by id. ``relationship_kinds`` restricts which incident
    relationships are followed (default: all). Raises
    ``GraphQueryError`` (NODE_NOT_FOUND / INVALID_ARGUMENT)."""
    _validate_graph(graph)
    resolved = _require_node(graph, node_id)
    validated_direction = _validate_direction(direction)
    validated_kinds = _validate_relationship_kinds(relationship_kinds)
    neighbor_ids = sorted(
        {
            neighbor
            for neighbor, data in _incidents(graph, resolved, validated_direction)
            if validated_kinds is None or data.get("kind") in validated_kinds
        }
    )
    return [_node_entry(graph, neighbor_id, 1) for neighbor_id in neighbor_ids]


def relationships(
    graph: nx.MultiDiGraph,
    node_id: Any,
    direction: str = DEFAULT_DIRECTION,
    relationship_kinds: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Incident relationship records in ``direction`` (``depth`` 1),
    sorted by relationship id. Raises ``GraphQueryError``
    (NODE_NOT_FOUND / INVALID_ARGUMENT)."""
    _validate_graph(graph)
    resolved = _require_node(graph, node_id)
    validated_direction = _validate_direction(direction)
    validated_kinds = _validate_relationship_kinds(relationship_kinds)
    entries: List[Dict[str, Any]] = []
    for _neighbor, data in _incidents(graph, resolved, validated_direction):
        if validated_kinds is not None and data.get("kind") not in validated_kinds:
            continue
        entries.append(_relationship_entry(data, 1))
    return entries


def find_nodes(
    graph: nx.MultiDiGraph, kind: Any = None
) -> List[Dict[str, Any]]:
    """All node records sorted by id — or only those matching
    ``kind``: a ``"kind"``/``"kind/subkind"`` string, or a list of
    them (the model's own generic representation)."""
    _validate_graph(graph)
    if kind is None:
        filters = None
    else:
        filters = _validate_node_kinds([kind] if isinstance(kind, str) else kind)
    entries: List[Dict[str, Any]] = []
    for node_id in sorted(graph.nodes):
        data = graph.nodes[node_id]
        if _node_matches(data.get("kind"), data.get("subkind"), filters):
            entries.append(_node_entry(graph, node_id, 0))
    return entries


def find_nodes_by_property(
    graph: nx.MultiDiGraph, name: Any, value: Any
) -> List[Dict[str, Any]]:
    """Node records whose property ``name`` equals ``value`` exactly,
    sorted by id. ``name`` is a top-level node field (``id``,
    ``kind``, ``subkind``, ``classification``, ...) or a key of the
    node's own ``attributes`` — the property vocabulary is always the
    model's, never hard-coded per repository. A node without that
    property simply does not match (no error)."""
    _validate_graph(graph)
    if not isinstance(name, str) or not name.strip():
        raise _invalid(
            "name must be a non-empty string; got {!r}.".format(name)
        )
    resolved_name = name.strip()
    entries: List[Dict[str, Any]] = []
    for node_id in sorted(graph.nodes):
        data = graph.nodes[node_id]
        if resolved_name in data:
            matched = data.get(resolved_name) == value
        else:
            attributes = data.get("attributes")
            matched = (
                isinstance(attributes, dict)
                and resolved_name in attributes
                and attributes.get(resolved_name) == value
            )
        if matched:
            entries.append(_node_entry(graph, node_id, 0))
    return entries


def traverse(
    graph: nx.MultiDiGraph,
    node_id: Any,
    direction: str = DEFAULT_DIRECTION,
    relationship_kinds: Optional[List[str]] = None,
    node_kinds: Optional[List[str]] = None,
    max_depth: int = DEFAULT_QUERY_DEPTH,
) -> Dict[str, Any]:
    """Bounded neighborhood walk — the named entry point for exactly
    what ``query_engineering_graph(graph, start_node_id=node_id, ...)``
    performs, returning its JSON-ready ``to_dict()`` (start_node,
    nodes, relationships, evidence, counts, truncated)."""
    return query_engineering_graph(
        graph,
        start_node_id=node_id,
        direction=direction,
        relationship_kinds=relationship_kinds,
        node_kinds=node_kinds,
        max_depth=max_depth,
    ).to_dict()


__all__ = [
    "DEFAULT_DIRECTION",
    "DEFAULT_QUERY_DEPTH",
    "EngineeringGraphQueryResult",
    "GraphQueryError",
    "MAX_QUERY_DEPTH",
    "MAX_QUERY_NODES",
    "MAX_QUERY_RELATIONSHIPS",
    "MODE_SUMMARY",
    "MODE_TRAVERSAL",
    "QUERY_DIRECTIONS",
    "SUMMARY_SAMPLE_LIMIT",
    "find_nodes",
    "find_nodes_by_property",
    "get_node",
    "neighbors",
    "query_engineering_graph",
    "relationships",
    "traverse",
]
