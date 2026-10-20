"""Model-facing projection of ``query_engineering_graph`` payloads.

The canonical graph tool returns the full query contract: every node
carries its attributes and inline evidence, every relationship repeats
its evidence, and a third flat evidence view annotates each entry with
its origin. That triple duplication is correct for persistence and the
canonical tool's own tests — but as a MODEL OBSERVATION it inflates the
request context (a depth-2 traversal of a hub artifact serializes to
~139 KB before a single token of reasoning is added).

``project_graph_result_for_model`` re-serializes one canonical payload
into ``model_graph_compact.v1``:

* ``mode``/``query``/``start_node``/``kinds``/``counts``/``truncated``
  keep the query's own contract and vocabulary verbatim (``start_node``
  is compacted like a node but never dropped);
* nodes keep ``id``/``depth``/``kind``/``subkind``/``classification``
  and reference evidence by id — ``attributes`` are dropped;
* relationships keep their identity fields verbatim and reference
  evidence by id;
* one deduplicated top-level ``evidence`` table replaces the three
  duplicated copies; each row carries
  ``evidence_id = "ev:" + sha256(canonical)[:12]`` (canonical =
  the entry minus its ``origin``/``index`` annotations);
* a ``payload`` block records the projection name, the byte budget,
  the serialized body size, and whether truncation happened.

Truncation runs only when the compact body plus its ``payload`` block
still exceeds ``max_bytes``: the DEEPEST nodes are dropped first (never
the start node), then every relationship pointing at a dropped node,
then evidence no remaining entity references — deterministically, one
node at a time. When even the irreducible envelope does not fit, a
``GraphProjectionError("MODEL_PAYLOAD_TOO_LARGE")`` is raised and the
tool wrapper turns it into a structured ``ToolResult`` failure, so the
model receives an actionable error instead of an oversized observation.

Payloads that are NOT canonical (no ``mode`` key — the documented
``tool(arguments)`` seam used by fixture bindings) pass through
unchanged: the projection applies only to the real query contract, and
the canonical tool itself stays byte-identical.

``model_facing_graph_result`` / ``model_facing_graph_tool`` are the two
wiring seams (live bindings + tool registry); failures, provenance
(``source``), and the registered tool NAME all stay exactly as the
canonical contract defines them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .protocol import ToolError, ToolResult
from .query_engineering_graph import query_engineering_graph_tool

MODEL_GRAPH_PROJECTION = "model_graph_compact.v1"
DEFAULT_MODEL_GRAPH_MAX_BYTES = 24_000

REFINEMENT_HINT = (
    "payload truncated to fit the model budget; re-query with a "
    "narrower start_node, direction, node_kinds or "
    "relationship_kinds filters, or max_depth 1."
)

_ANNOTATION_KEYS = frozenset({"origin", "index"})
_NODE_IDENTITY_KEYS = ("kind", "subkind", "classification")


class GraphProjectionError(Exception):
    """Deterministic projection failure carrying a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _serialized_bytes(payload: Any) -> int:
    return len(json.dumps(payload))


def _canonical(entry: Mapping[str, Any]) -> str:
    identity = {
        key: value
        for key, value in entry.items()
        if key not in _ANNOTATION_KEYS
    }
    return json.dumps(
        identity, sort_keys=True, ensure_ascii=False, default=str
    )


def _evidence_id(entry: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()
    return "ev:{}".format(digest[:12])


def _compact_node(
    entry: Mapping[str, Any], evidence_ids: List[str]
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "id": entry.get("id"),
        "depth": entry.get("depth"),
    }
    for key in _NODE_IDENTITY_KEYS:
        value = entry.get(key)
        if value is not None:
            compact[key] = value
    if evidence_ids:
        compact["evidence_ids"] = evidence_ids
    return compact


def _compact_relationship(
    entry: Mapping[str, Any], evidence_ids: List[str]
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "id": entry.get("id"),
        "kind": entry.get("kind"),
        "source": entry.get("source"),
        "target": entry.get("target"),
        "depth": entry.get("depth"),
    }
    classification = entry.get("classification")
    if classification is not None:
        compact["classification"] = classification
    if evidence_ids:
        compact["evidence_ids"] = evidence_ids
    return compact


def _build_body(
    data: Mapping[str, Any],
    survivors: Optional[Set[str]],
    dropped: Set[str],
) -> Dict[str, Any]:
    """Compact body for one survivor set (``None`` keeps everything).

    The evidence table is rebuilt from the entities actually kept, so
    dropped nodes take their evidence with them. Already-compact inputs
    (``evidence_ids`` instead of inline ``evidence`` — the output of a
    previous projection) resolve their carried ids against the input's
    own table, which makes re-projecting a projected payload safe.
    """
    table: Dict[str, Dict[str, Any]] = {}
    seed_index: Dict[str, Dict[str, Any]] = {}
    for row in data.get("evidence") or []:
        if isinstance(row, Mapping):
            evidence_id = row.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id:
                seed_index.setdefault(
                    evidence_id,
                    {
                        key: value
                        for key, value in row.items()
                        if key not in _ANNOTATION_KEYS
                    },
                )

    def register(entry: Mapping[str, Any]) -> str:
        evidence_id = _evidence_id(entry)
        if evidence_id not in table:
            row: Dict[str, Any] = {"evidence_id": evidence_id}
            row.update(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in _ANNOTATION_KEYS
                }
            )
            table[evidence_id] = row
        return evidence_id

    def carry(evidence_id: str) -> str:
        if evidence_id not in table:
            seeded = seed_index.get(evidence_id)
            if seeded is not None:
                table[evidence_id] = dict(seeded)
        return evidence_id

    def ids_for(entry: Mapping[str, Any]) -> List[str]:
        collected = [register(item) for item in entry.get("evidence") or []]
        collected.extend(
            carry(evidence_id)
            for evidence_id in entry.get("evidence_ids") or []
            if isinstance(evidence_id, str)
        )
        seen: Set[str] = set()
        ordered: List[str] = []
        for evidence_id in collected:
            if evidence_id not in seen:
                seen.add(evidence_id)
                ordered.append(evidence_id)
        return ordered

    body: Dict[str, Any] = {
        "mode": data.get("mode"),
        "query": dict(data.get("query") or {}),
    }

    start = data.get("start_node")
    if isinstance(start, Mapping):
        body["start_node"] = _compact_node(start, ids_for(start))
    else:
        body["start_node"] = None

    nodes_out: List[Dict[str, Any]] = []
    for entry in data.get("nodes") or []:
        if survivors is not None and entry.get("id") not in survivors:
            continue
        nodes_out.append(_compact_node(entry, ids_for(entry)))
    body["nodes"] = nodes_out

    rels_out: List[Dict[str, Any]] = []
    for entry in data.get("relationships") or []:
        if survivors is not None and (
            entry.get("source") in dropped
            or entry.get("target") in dropped
        ):
            continue
        rels_out.append(_compact_relationship(entry, ids_for(entry)))
    body["relationships"] = rels_out

    body["evidence"] = list(table.values())
    body["kinds"] = [
        dict(entry) for entry in data.get("kinds") or []
    ]
    body["counts"] = dict(data.get("counts") or {})
    body["truncated"] = bool(data.get("truncated"))
    return body


def _seal(
    body: Dict[str, Any],
    max_bytes: int,
    truncated: bool,
    original_body_bytes: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "projection": MODEL_GRAPH_PROJECTION,
        "max_bytes": max_bytes,
        "body_bytes": _serialized_bytes(body),
        "truncated": bool(truncated),
    }
    if truncated:
        payload["original_body_bytes"] = original_body_bytes
        payload["kept"] = {
            "nodes": len(body.get("nodes") or []),
            "relationships": len(body.get("relationships") or []),
            "evidence": len(body.get("evidence") or []),
        }
    sealed = dict(body)
    sealed["payload"] = payload
    if truncated:
        sealed["refinement_hint"] = REFINEMENT_HINT
    return sealed


def project_graph_result_for_model(
    data: Mapping[str, Any],
    max_bytes: int = DEFAULT_MODEL_GRAPH_MAX_BYTES,
) -> Dict[str, Any]:
    """Compact one canonical graph payload to fit ``max_bytes``.

    Non-canonical payloads (no ``mode`` key) pass through unchanged;
    canonical payloads are compacted, then deterministically truncated
    (deepest nodes first, start node never) until the SEALED result —
    body plus ``payload`` block — fits ``max_bytes``.
    ``GraphProjectionError`` is raised when even the irreducible
    envelope cannot fit.
    """
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise ValueError(
            "max_bytes must be a positive integer; got {!r}.".format(
                max_bytes
            )
        )
    if not isinstance(data, Mapping):
        raise TypeError(
            "data must be a mapping; got {}.".format(
                type(data).__name__
            )
        )
    if "mode" not in data:
        return dict(data)

    source_nodes = list(data.get("nodes") or [])
    original_ids = {
        entry.get("id")
        for entry in source_nodes
        if entry.get("id") is not None
    }
    start = data.get("start_node")
    start_id = start.get("id") if isinstance(start, Mapping) else None
    protected = {start_id} if start_id is not None else set()

    body = _build_body(data, None, set())
    original_body_bytes = _serialized_bytes(body)
    sealed = _seal(body, max_bytes, False, original_body_bytes)
    if _serialized_bytes(sealed) <= max_bytes:
        return sealed

    rank: Dict[Any, Tuple[int, int]] = {}
    for index, entry in enumerate(source_nodes):
        node_id = entry.get("id")
        if node_id is None or node_id in protected:
            continue
        depth = entry.get("depth")
        if isinstance(depth, bool) or not isinstance(depth, int):
            depth = 0
        rank[node_id] = (depth, index)
    candidates = sorted(
        rank,
        key=lambda node_id: (-rank[node_id][0], -rank[node_id][1]),
    )

    survivors: Set[str] = set(original_ids)
    for victim in candidates:
        survivors.discard(victim)
        dropped = original_ids - survivors
        body = _build_body(data, survivors, dropped)
        sealed = _seal(body, max_bytes, True, original_body_bytes)
        if _serialized_bytes(sealed) <= max_bytes:
            return sealed

    minimal = _build_body(data, set(), original_ids)
    final_bytes = _serialized_bytes(
        _seal(minimal, max_bytes, True, original_body_bytes)
    )
    raise GraphProjectionError(
        "MODEL_PAYLOAD_TOO_LARGE",
        "the compact graph projection is still {} bytes with only the "
        "start node left; the budget is {} bytes. Refine the query "
        "(direction, kinds, or max_depth 1).".format(
            final_bytes, max_bytes
        ),
    )


def model_facing_graph_result(
    result: ToolResult,
    max_bytes: Optional[int] = None,
) -> ToolResult:
    """Apply the projection to one canonical tool result.

    Failures and ``data is None`` pass through untouched; a projection
    failure becomes a structured ``ToolResult`` failure with code
    ``MODEL_PAYLOAD_TOO_LARGE`` (``source`` stays ``None`` on failure,
    per the ``ToolResult`` contract). Successes keep their provenance
    and tool name, with only ``data`` replaced.
    """
    if not isinstance(result, ToolResult):
        raise TypeError(
            "result must be a ToolResult; got {}.".format(
                type(result).__name__
            )
        )
    if not result.success or result.data is None:
        return result
    budget = (
        DEFAULT_MODEL_GRAPH_MAX_BYTES
        if max_bytes is None
        else max_bytes
    )
    try:
        projected = project_graph_result_for_model(result.data, budget)
    except GraphProjectionError as exc:
        return ToolResult(
            success=False,
            tool=result.tool,
            data=None,
            source=None,
            error=ToolError(code=exc.code, message=exc.message),
        )
    return replace(result, data=projected)


def model_facing_graph_tool(
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    output_root: Optional[Path] = None,
    engineering_model_path_arg: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> ToolResult:
    """The registry-facing graph tool: canonical query + projection.

    Signature mirrors ``query_engineering_graph_tool`` (the dispatcher
    passes the arguments mapping positionally; application context
    stays keyword-only), so the registered tool name and input contract
    are unchanged.
    """
    result = query_engineering_graph_tool(
        arguments,
        output_root=output_root,
        engineering_model_path_arg=engineering_model_path_arg,
        snapshot_id=snapshot_id,
    )
    return model_facing_graph_result(result, max_bytes=max_bytes)
