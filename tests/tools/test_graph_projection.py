"""Model-facing graph projection — H3 of the context-overflow work.

Covers the ``model_graph_compact.v1`` contract: the canonical query
contract (``mode``/``query``/``counts``/``truncated``/``kinds``)
passes through verbatim while nodes/relationships drop ``attributes``
and reference a single deduplicated evidence table by
``evidence_id``; the ``payload`` block reports budget and truncation;
truncation drops the deepest nodes first and never the start node;
foreign (fixture-seam) payloads pass through untouched; an
irreducibly large payload becomes ``MODEL_PAYLOAD_TOO_LARGE`` instead
of an oversized observation; both wiring seams (tool registry + live
bindings wrapper) are pinned. Integration against the pinned
engineering-graph artifact skips when the artifact is absent.

Offline — no network, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.tool_dispatcher import (
    default_tool_dispatcher,
    tool_registry,
)
from repo_intelligence.tools import ToolError, ToolResult
from repo_intelligence.tools.graph_projection import (
    DEFAULT_MODEL_GRAPH_MAX_BYTES,
    MODEL_GRAPH_PROJECTION,
    GraphProjectionError,
    model_facing_graph_result,
    model_facing_graph_tool,
    project_graph_result_for_model,
)
from repo_intelligence.tools.query_engineering_graph import (
    query_engineering_graph_tool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADOP_GRAPH = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_graph.json"
)

COMMIT = "c" * 40
A = "artifact:directory:pkg"
B = "artifact:file:pkg/mod.py"


def _evidence(path, line=None, symbol=None):
    entry = {
        "path": path,
        "artifact_type": "file",
        "commit": COMMIT,
    }
    if line is not None:
        entry["line_start"] = line
    if symbol is not None:
        entry["symbol"] = symbol
    return entry


def _node(node_id, depth, evidence=(), attrs=None):
    return {
        "id": node_id,
        "depth": depth,
        "kind": "artifact",
        "subkind": "file",
        "classification": "observed",
        "attributes": dict(attrs or {"path": node_id, "name": None}),
        "evidence": [dict(item) for item in evidence],
    }


def _small_traversal():
    """A canonical two-node traversal with shared evidence — the shape
    the real tool returns (attributes + inline evidence + annotated
    aggregate view)."""
    ev_a = _evidence("pkg/mod.py", line=10, symbol="run")
    ev_b = _evidence("pkg/other.py", line=22)
    node_a = _node(A, 0, [ev_a])
    node_b = _node(B, 1, [ev_b])
    relationship = {
        "id": "rel:CONTAINS:{}:{}".format(A, B),
        "kind": "CONTAINS",
        "source": A,
        "target": B,
        "classification": "derived",
        "depth": 1,
        "evidence": [dict(ev_a), dict(ev_b)],
    }
    aggregate = []
    for entry, origin in ((node_a, A), (node_b, B)):
        for index, item in enumerate(entry["evidence"]):
            row = dict(item)
            row["origin"] = "node:{}".format(origin)
            row["index"] = index
            aggregate.append(row)
    return {
        "mode": "traversal",
        "query": {"start_node_id": A, "direction": "out", "max_depth": 1},
        "start_node": node_a,
        "nodes": [node_a, node_b],
        "relationships": [relationship],
        "evidence": aggregate,
        "kinds": [],
        "counts": {
            "nodes": 2,
            "relationships": 1,
            "evidence": 2,
            "unresolved": 0,
        },
        "truncated": False,
    }


def _large_traversal(node_count=60, rel_count=80):
    """A synthetic hub traversal whose COMPACT form still exceeds the
    default budget — forces deterministic truncation."""
    nodes = []
    for index in range(node_count):
        depth = index % 4
        evidence = [
            _evidence(
                "src/module_{0}/file_{1}.py".format(index, item),
                line=item * 7,
                symbol="handler_{0}_{1}".format(index, item),
            )
            for item in range(3)
        ]
        node = _node(
            "artifact:file:module_{0}".format(index), depth, evidence
        )
        node["attributes"] = {
            "path": "src/module_{0}/file_0.py".format(index),
            "qualified_name": "module_{0}.handler".format(index),
            "fact_type": "file",
            "fact_kind": "repository_file",
        }
        nodes.append(node)
    relationships = []
    for index in range(rel_count):
        source = nodes[index % node_count]
        target = nodes[(index * 7 + 3) % node_count]
        relationships.append(
            {
                "id": "rel:IMPORTS:{}:{}".format(
                    source["id"], target["id"]
                ),
                "kind": "IMPORTS",
                "source": source["id"],
                "target": target["id"],
                "classification": "derived",
                "depth": target["depth"],
                "evidence": [dict(e) for e in source["evidence"][:1]],
            }
        )
    start = dict(nodes[0])
    start["depth"] = 0
    return {
        "mode": "traversal",
        "query": {
            "start_node_id": start["id"],
            "direction": "both",
            "max_depth": 3,
        },
        "start_node": start,
        "nodes": nodes,
        "relationships": relationships,
        "evidence": [],
        "kinds": [],
        "counts": {
            "nodes": node_count,
            "relationships": rel_count,
            "evidence": node_count * 3,
            "unresolved": 5,
        },
        "truncated": False,
    }


def _sealed_body_size(projected):
    return len(
        json.dumps(
            {
                key: value
                for key, value in projected.items()
                if key not in ("payload", "refinement_hint")
            }
        )
    )


def _evidence_table(projected):
    return {row["evidence_id"] for row in projected["evidence"]}


def _entity_entries(projected):
    entries = list(projected["nodes"])
    if projected["start_node"] is not None:
        entries.append(projected["start_node"])
    entries.extend(projected["relationships"])
    return entries


# ---------------------------------------------------------------------------
# Pass-through (foreign payloads) and input validation
# ---------------------------------------------------------------------------


def test_foreign_payload_passes_through_unchanged():
    foreign = {"nodes": ["real graph"]}
    assert project_graph_result_for_model(foreign) == foreign
    assert "payload" not in project_graph_result_for_model(foreign)


def test_foreign_payload_ignores_the_byte_budget():
    foreign = {"nodes": ["real graph"]}
    assert project_graph_result_for_model(foreign, max_bytes=8) == foreign


def test_non_mapping_data_is_rejected():
    with pytest.raises(TypeError):
        project_graph_result_for_model(None)
    with pytest.raises(TypeError):
        project_graph_result_for_model(["not", "a", "mapping"])


def test_invalid_byte_budgets_are_rejected():
    for bad in (0, -1, True, "24000", 1.5, None):
        with pytest.raises(ValueError):
            project_graph_result_for_model(_small_traversal(), bad)


def test_wrapper_requires_a_tool_result():
    with pytest.raises(TypeError):
        model_facing_graph_result({"mode": "traversal"})


# ---------------------------------------------------------------------------
# Compact contract (small payload: no truncation)
# ---------------------------------------------------------------------------


def test_small_payload_keeps_the_query_contract_verbatim():
    data = _small_traversal()
    projected = project_graph_result_for_model(data)

    assert projected["mode"] == "traversal"
    assert projected["query"] == data["query"]
    assert projected["counts"] == data["counts"]
    assert projected["truncated"] is False
    assert projected["kinds"] == []
    assert projected["start_node"]["id"] == A
    assert [node["id"] for node in projected["nodes"]] == [A, B]
    rel = projected["relationships"][0]
    assert rel["id"] == data["relationships"][0]["id"]
    assert rel["kind"] == "CONTAINS"
    assert rel["source"] == A
    assert rel["target"] == B


def test_attributes_are_dropped_and_identity_fields_survive():
    projected = project_graph_result_for_model(_small_traversal())

    for entry in _entity_entries(projected):
        assert "attributes" not in entry
        assert "evidence" not in entry
    assert projected["start_node"]["kind"] == "artifact"
    assert projected["start_node"]["subkind"] == "file"
    assert projected["start_node"]["classification"] == "observed"
    assert projected["nodes"][1]["depth"] == 1


def test_evidence_is_deduplicated_into_one_referenced_table():
    data = _small_traversal()
    projected = project_graph_result_for_model(data)

    table = _evidence_table(projected)
    assert len(projected["evidence"]) == 2
    assert len(table) == 2
    for row in projected["evidence"]:
        assert row["evidence_id"].startswith("ev:")
        assert "origin" not in row
        assert "index" not in row
        assert row["path"] in {"pkg/mod.py", "pkg/other.py"}
    for entry in _entity_entries(projected):
        for evidence_id in entry.get("evidence_ids") or []:
            assert evidence_id in table
    node_a = projected["nodes"][0]
    node_b = projected["nodes"][1]
    assert node_a["evidence_ids"] != node_b["evidence_ids"]
    rel = projected["relationships"][0]
    assert set(rel["evidence_ids"]) == set(
        node_a["evidence_ids"] + node_b["evidence_ids"]
    )


def test_payload_block_reports_the_projection_and_its_bytes():
    projected = project_graph_result_for_model(_small_traversal())

    payload = projected["payload"]
    assert payload["projection"] == MODEL_GRAPH_PROJECTION
    assert payload["max_bytes"] == DEFAULT_MODEL_GRAPH_MAX_BYTES
    assert payload["truncated"] is False
    assert payload["body_bytes"] == _sealed_body_size(projected)
    assert "refinement_hint" not in projected
    assert "kept" not in payload
    assert "original_body_bytes" not in payload


def test_projection_is_deterministic():
    data = _large_traversal()
    first = project_graph_result_for_model(data)
    second = project_graph_result_for_model(
        json.loads(json.dumps(data))
    )
    assert first == second


# ---------------------------------------------------------------------------
# Budget truncation (large payload)
# ---------------------------------------------------------------------------


def test_large_payload_fits_the_default_budget():
    data = _large_traversal()
    assert len(json.dumps(data)) > DEFAULT_MODEL_GRAPH_MAX_BYTES

    projected = project_graph_result_for_model(data)
    serialized = json.dumps(projected)

    assert len(serialized) <= DEFAULT_MODEL_GRAPH_MAX_BYTES
    payload = projected["payload"]
    assert payload["truncated"] is True
    untruncated = project_graph_result_for_model(
        data, max_bytes=10_000_000
    )
    assert untruncated["payload"]["truncated"] is False
    assert (
        payload["original_body_bytes"]
        == untruncated["payload"]["body_bytes"]
    )
    assert payload["original_body_bytes"] >= payload["body_bytes"]
    assert payload["kept"]["nodes"] == len(projected["nodes"])
    assert payload["kept"]["relationships"] == len(
        projected["relationships"]
    )
    assert payload["kept"]["evidence"] == len(projected["evidence"])
    assert "refinement_hint" in projected
    assert projected["counts"] == data["counts"]


def test_truncation_drops_deepest_nodes_first_and_keeps_the_start():
    data = _large_traversal()
    projected = project_graph_result_for_model(data)

    kept_ids = {node["id"] for node in projected["nodes"]}
    original_ids = {node["id"] for node in data["nodes"]}
    dropped_ids = original_ids - kept_ids
    assert dropped_ids, "expected the budget to drop nodes"
    assert projected["start_node"]["id"] == data["start_node"]["id"]
    assert data["start_node"]["id"] in kept_ids

    depth_of = {node["id"]: node["depth"] for node in data["nodes"]}
    kept_depths = [depth_of[node_id] for node_id in kept_ids]
    dropped_depths = [depth_of[node_id] for node_id in dropped_ids]
    assert min(dropped_depths) >= max(kept_depths)

    surviving = kept_ids | {data["start_node"]["id"]}
    for rel in projected["relationships"]:
        assert rel["source"] in surviving
        assert rel["target"] in surviving


def test_truncation_respects_a_custom_budget():
    data = _large_traversal()
    budget = 8_000
    projected = project_graph_result_for_model(data, budget)

    assert len(json.dumps(projected)) <= budget
    assert projected["payload"]["max_bytes"] == budget
    assert projected["payload"]["truncated"] is True
    assert len(projected["nodes"]) < len(data["nodes"])


def test_irreducibly_large_payload_raises_payload_too_large():
    data = _small_traversal()
    with pytest.raises(GraphProjectionError) as excinfo:
        project_graph_result_for_model(data, max_bytes=120)
    assert excinfo.value.code == "MODEL_PAYLOAD_TOO_LARGE"
    assert "budget" in excinfo.value.message


def test_reprojecting_a_projected_payload_stays_resolvable():
    first = project_graph_result_for_model(_large_traversal())
    second = project_graph_result_for_model(first)
    third = project_graph_result_for_model(second)

    for projected in (first, second, third):
        table = _evidence_table(projected)
        for entry in _entity_entries(projected):
            for evidence_id in entry.get("evidence_ids") or []:
                assert evidence_id in table
        assert projected["start_node"]["id"] == first["start_node"]["id"]


# ---------------------------------------------------------------------------
# ToolResult wrapper + registry seam
# ---------------------------------------------------------------------------


def test_wrapper_replaces_data_and_keeps_provenance():
    result = ToolResult(
        success=True,
        tool="query_engineering_graph",
        data=_small_traversal(),
        source={"snapshot_id": "pinned_snapshot"},
    )
    wrapped = model_facing_graph_result(result)

    assert wrapped.success is True
    assert wrapped.tool == "query_engineering_graph"
    assert wrapped.source is result.source
    assert wrapped.data is not result.data
    assert wrapped.data["payload"]["projection"] == MODEL_GRAPH_PROJECTION
    assert "payload" not in result.data


def test_wrapper_passes_canonical_failures_through_untouched():
    failure = ToolResult(
        success=False,
        tool="query_engineering_graph",
        error=ToolError(
            code="ENGINEERING_GRAPH_NOT_FOUND",
            message="graph artifact is absent.",
        ),
    )
    assert model_facing_graph_result(failure) is failure


def test_wrapper_passes_empty_success_data_through():
    result = ToolResult(success=True, tool="query_engineering_graph")
    assert model_facing_graph_result(result) is result


def test_wrapper_turns_budget_exhaustion_into_a_structured_failure():
    result = ToolResult(
        success=True,
        tool="query_engineering_graph",
        data=_small_traversal(),
        source={"snapshot_id": "pinned_snapshot"},
    )
    wrapped = model_facing_graph_result(result, max_bytes=120)

    assert wrapped.success is False
    assert wrapped.tool == "query_engineering_graph"
    assert wrapped.data is None
    assert wrapped.source is None
    assert wrapped.error is not None
    assert wrapped.error.code == "MODEL_PAYLOAD_TOO_LARGE"
    assert wrapped.error.message


def test_registry_binds_the_model_facing_graph_tool():
    registry = tool_registry()
    assert registry["query_engineering_graph"] is model_facing_graph_tool
    assert set(registry) == {
        "get_architecture_context",
        "search_repository",
        "read_file",
        "query_engineering_graph",
    }
    assert set(default_tool_dispatcher().tool_names) == set(registry)


def test_registry_tool_passes_canonical_failures_through(tmp_path):
    result = model_facing_graph_tool(
        {},
        engineering_model_path_arg=tmp_path / "absent_engineering.json",
    )
    assert result.success is False
    assert result.data is None
    assert result.error is not None


# ---------------------------------------------------------------------------
# Integration: pinned artifact (production run layout)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adop_output_root():
    if not ADOP_GRAPH.is_file():
        pytest.skip("pinned engineering graph artifact absent")
    return PROJECT_ROOT / "artifacts"


def test_pinned_hub_traversal_fits_the_model_budget(adop_output_root):
    arguments = {
        "start_node_id": "artifact:file:tests/unit/test_agent_tracer.py",
        "direction": "both",
        "max_depth": 2,
    }
    raw = query_engineering_graph_tool(
        arguments, output_root=adop_output_root
    )
    assert raw.success is True
    raw_bytes = len(json.dumps(raw.data))
    assert raw_bytes > DEFAULT_MODEL_GRAPH_MAX_BYTES

    projected = model_facing_graph_result(raw)
    assert projected.success is True
    assert projected.tool == "query_engineering_graph"
    assert projected.source is raw.source
    assert len(json.dumps(projected.data)) <= DEFAULT_MODEL_GRAPH_MAX_BYTES
    assert projected.data["payload"]["truncated"] is True
    assert projected.data["counts"] == raw.data["counts"]
    assert projected.data["start_node"]["id"] == arguments["start_node_id"]

    table = _evidence_table(projected.data)
    for entry in _entity_entries(projected.data):
        for evidence_id in entry.get("evidence_ids") or []:
            assert evidence_id in table

    repeat = model_facing_graph_result(
        query_engineering_graph_tool(arguments, output_root=adop_output_root)
    )
    assert repeat.data == projected.data
