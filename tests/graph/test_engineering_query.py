"""M11.2/M11.3 — deterministic engineering graph query API tests.

Levels:
- unit (synthetic A/B/C model, spec §23): node lookup, out/in/both
  traversal, depth limits, relationship/node-kind filtering,
  nonexistent node, empty result, deterministic ordering, hard caps,
  summary mode, input validation, graph non-mutation, evidence
  preservation, named query API (get_node, neighbors, relationships,
  find_nodes, find_nodes_by_property, traverse)
- integration: pinned ADOP EngineeringModel (skipped when absent)

Offline — no network, no LLM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.graph import (
    MAX_QUERY_NODES,
    MAX_QUERY_RELATIONSHIPS,
    GraphQueryError,
    build_engineering_graph,
    find_nodes,
    find_nodes_by_property,
    get_node,
    neighbors,
    query_engineering_graph,
    relationships,
    traverse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADOP_ENG_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_model.json"
)

COMMIT = "b" * 40

A = "artifact:directory:pkg"
B = "artifact:file:pkg/mod.py"
C = "code_entity:pkg/mod.py"


def _mini_engineering_document():
    """A --CONTAINS--> B --IMPORTS--> C (plus one unresolved claim)."""
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {"name": "synthetic", "url": "", "commit_sha": COMMIT},
        "analysis": {"strategy": "deterministic", "analyzers": []},
        "engineering": {
            "nodes": [
                {
                    "id": A,
                    "kind": "artifact",
                    "subkind": "directory",
                    "classification": "observed",
                    "evidence": [
                        {"path": "pkg", "artifact_type": "directory", "commit": COMMIT}
                    ],
                    "attributes": {"path": "pkg"},
                },
                {
                    "id": B,
                    "kind": "artifact",
                    "subkind": "file",
                    "classification": "observed",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": COMMIT,
                        }
                    ],
                    "attributes": {"path": "pkg/mod.py"},
                },
                {
                    "id": C,
                    "kind": "code_entity",
                    "subkind": "module",
                    "classification": "derived",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": COMMIT,
                            "symbol": "pkg.mod",
                        }
                    ],
                    "attributes": {"qualified_name": "pkg.mod"},
                },
            ],
            "relationships": [
                {
                    "id": "rel:CONTAINS:{}:{}".format(A, B),
                    "kind": "CONTAINS",
                    "source": A,
                    "target": B,
                    "classification": "derived",
                    "evidence": [
                        {"path": "pkg", "artifact_type": "directory", "commit": COMMIT}
                    ],
                },
                {
                    "id": "rel:IMPORTS:{}:{}".format(B, C),
                    "kind": "IMPORTS",
                    "source": B,
                    "target": C,
                    "classification": "observed",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": COMMIT,
                            "line_start": 1,
                            "symbol": "mod",
                        }
                    ],
                },
            ],
            "unresolved": [
                {
                    "source": B,
                    "target": A,
                    "kind": "DEPENDS_ON",
                    "reason": "ambiguous_identity",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": COMMIT,
                            "line_start": 3,
                        }
                    ],
                }
            ],
            "metadata": {},
        },
        "architecture": {
            "nodes": [],
            "relationships": [],
            "unresolved": [],
            "metadata": {},
        },
        "mappings": [],
        "metadata": {},
    }


def _mini_graph():
    return build_engineering_graph(_mini_engineering_document())


def _star_graph(leaf_count, kinds_per_leaf=1):
    """Center node with ``leaf_count`` leaves (optionally multi-edged)."""
    edge_kinds = ("CONTAINS", "IMPORTS", "DEPENDS_ON", "TRIGGERS")[
        :kinds_per_leaf
    ]
    document = _mini_engineering_document()
    document["engineering"] = {
        "nodes": [
            {
                "id": "artifact:directory:center",
                "kind": "artifact",
                "subkind": "directory",
                "classification": "observed",
                "evidence": [
                    {"path": "c", "artifact_type": "directory", "commit": COMMIT}
                ],
                "attributes": {},
            }
        ],
        "relationships": [],
        "unresolved": [],
        "metadata": {},
    }
    for index in range(leaf_count):
        leaf_id = "artifact:file:leaf_{:03d}.txt".format(index)
        document["engineering"]["nodes"].append(
            {
                "id": leaf_id,
                "kind": "artifact",
                "subkind": "file",
                "classification": "observed",
                "evidence": [
                    {
                        "path": leaf_id.split(":", 2)[2],
                        "artifact_type": "file",
                        "commit": COMMIT,
                    }
                ],
                "attributes": {},
            }
        )
        for kind in edge_kinds:
            document["engineering"]["relationships"].append(
                {
                    "id": "rel:{}:artifact:directory:center:{}".format(
                        kind, leaf_id
                    ),
                    "kind": kind,
                    "source": "artifact:directory:center",
                    "target": leaf_id,
                    "classification": "observed",
                    "evidence": [
                        {"path": "c", "artifact_type": "directory", "commit": COMMIT}
                    ],
                }
            )
    document["engineering"]["relationships"].sort(
        key=lambda r: (r["kind"], r["source"], r["target"], r["id"])
    )
    document["engineering"]["nodes"].sort(key=lambda n: n["id"])
    return build_engineering_graph(document)


@pytest.fixture(scope="module")
def adop_graph():
    if not ADOP_ENG_MODEL.is_file():
        pytest.skip("ADOP EngineeringModel absent")
    return build_engineering_graph(
        json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))
    )


# ---------------------------------------------------------------------------
# Node lookup and traversal (synthetic, spec §23)
# ---------------------------------------------------------------------------


def test_node_lookup_at_depth_zero_returns_only_the_start_node():
    result = query_engineering_graph(_mini_graph(), start_node_id=A, max_depth=0)
    assert result.mode == "traversal"
    assert result.start_node["id"] == A
    assert result.start_node["depth"] == 0
    assert [n["id"] for n in result.nodes] == [A]
    assert result.relationships == []
    assert result.counts == {"nodes": 1, "relationships": 0, "evidence": 1}
    assert result.truncated is False


def test_outgoing_traversal_walks_the_abc_chain():
    graph = _mini_graph()
    depth1 = query_engineering_graph(graph, start_node_id=A, direction="out")
    assert [n["id"] for n in depth1.nodes] == [A, B]
    assert [r["kind"] for r in depth1.relationships] == ["CONTAINS"]
    assert depth1.relationships[0]["depth"] == 1

    depth2 = query_engineering_graph(
        graph, start_node_id=A, direction="out", max_depth=2
    )
    assert [n["id"] for n in depth2.nodes] == [A, B, C]
    assert [r["kind"] for r in depth2.relationships] == ["CONTAINS", "IMPORTS"]
    assert [r["depth"] for r in depth2.relationships] == [1, 2]
    assert depth2.nodes[2]["id"] == C and depth2.nodes[2]["depth"] == 2


def test_incoming_traversal_finds_predecessors():
    result = query_engineering_graph(_mini_graph(), start_node_id=C, direction="in")
    assert [n["id"] for n in result.nodes] == [B, C]
    assert result.relationships[0]["source"] == B
    assert result.relationships[0]["target"] == C


def test_bidirectional_traversal_sees_both_sides():
    result = query_engineering_graph(_mini_graph(), start_node_id=B, direction="both")
    assert [n["id"] for n in result.nodes] == [A, B, C]
    kinds = [r["kind"] for r in result.relationships]
    assert kinds == ["CONTAINS", "IMPORTS"]


def test_depth_limit_bounds_the_walk():
    graph = _mini_graph()
    one = query_engineering_graph(graph, start_node_id=A, max_depth=1)
    assert set(n["id"] for n in one.nodes) == {A, B}
    two = query_engineering_graph(graph, start_node_id=A, max_depth=2)
    assert set(n["id"] for n in two.nodes) == {A, B, C}


def test_relationship_kind_filter():
    graph = _mini_graph()
    contains_only = query_engineering_graph(
        graph, start_node_id=B, direction="both", relationship_kinds=["CONTAINS"]
    )
    assert [r["kind"] for r in contains_only.relationships] == ["CONTAINS"]
    assert set(n["id"] for n in contains_only.nodes) == {A, B}

    multiple = query_engineering_graph(
        graph,
        start_node_id=B,
        direction="both",
        relationship_kinds=["CONTAINS", "DEPENDS_ON"],
    )
    # DEPENDS_ON exists only as an unresolved claim → never returned
    assert [r["kind"] for r in multiple.relationships] == ["CONTAINS"]


def test_node_kind_filter_restricts_the_walk():
    graph = _mini_graph()
    files_only = query_engineering_graph(
        graph, start_node_id=A, direction="out", node_kinds=["artifact/file"]
    )
    assert [n["id"] for n in files_only.nodes] == [A, B]  # start anchor kept
    assert [r["kind"] for r in files_only.relationships] == ["CONTAINS"]

    # strict filter blocks traversal through non-matching intermediates:
    # C matches code_entity/module but B (artifact/file) does not
    blocked = query_engineering_graph(
        graph, start_node_id=A, direction="out", node_kinds=["code_entity/module"]
    )
    assert [n["id"] for n in blocked.nodes] == [A]
    assert blocked.relationships == []


def test_kind_level_filter_crosses_subkinds():
    """A kind-level filter matches every subkind of that kind and
    stops at other kinds (results stay a connected subgraph)."""
    graph = _mini_graph()
    result = query_engineering_graph(
        graph, start_node_id=A, direction="out", max_depth=2, node_kinds=["artifact"]
    )
    assert [n["id"] for n in result.nodes] == [A, B]  # C is code_entity
    assert [r["kind"] for r in result.relationships] == ["CONTAINS"]


def test_nonexistent_start_node_raises_node_not_found():
    with pytest.raises(GraphQueryError) as excinfo:
        query_engineering_graph(_mini_graph(), start_node_id="artifact:file:nope")
    assert excinfo.value.code == "NODE_NOT_FOUND"
    assert "nope" in str(excinfo.value)


def test_empty_result_is_a_valid_outcome():
    # C has no outgoing edges
    result = query_engineering_graph(
        _mini_graph(), start_node_id=C, direction="out", max_depth=2
    )
    assert [n["id"] for n in result.nodes] == [C]
    assert result.relationships == []
    assert result.evidence != []  # the start node keeps its own evidence


def test_unresolved_claim_is_never_returned_by_any_query():
    graph = _mini_graph()
    for direction in ("out", "in", "both"):
        result = query_engineering_graph(
            graph, start_node_id=B, direction=direction, max_depth=6
        )
        for relationship in result.relationships:
            assert relationship["kind"] != "DEPENDS_ON"


# ---------------------------------------------------------------------------
# Determinism, caps, safety
# ---------------------------------------------------------------------------


def test_query_is_deterministic_across_runs_and_builds():
    first = query_engineering_graph(
        _mini_graph(), start_node_id=B, direction="both", max_depth=2
    ).to_dict()
    second = query_engineering_graph(
        _mini_graph(), start_node_id=B, direction="both", max_depth=2
    ).to_dict()
    assert first == second  # identical structure AND ordering
    ids = [n["id"] for n in first["nodes"]]
    assert ids == sorted(ids)


def test_results_are_sorted_and_json_serializable():
    result = query_engineering_graph(
        _mini_graph(), start_node_id=B, direction="both", max_depth=2
    )
    payload = json.loads(json.dumps(result.to_dict(), sort_keys=True))
    assert payload["mode"] == "traversal"
    ids = [n["id"] for n in payload["nodes"]]
    assert ids == sorted(ids)
    depths = [r["depth"] for r in payload["relationships"]]
    assert depths == sorted(depths)


def test_query_never_mutates_the_graph():
    graph = _mini_graph()
    before_nodes = copy.deepcopy(dict(graph.nodes(data=True)))
    before_edges = copy.deepcopy(
        {(u, v, k): dict(d) for u, v, k, d in graph.edges(keys=True, data=True)}
    )
    before_graph_attrs = copy.deepcopy(dict(graph.graph))
    query_engineering_graph(graph, start_node_id=B, direction="both", max_depth=2)
    query_engineering_graph(graph)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert {
        (u, v, k): dict(d) for u, v, k, d in graph.edges(keys=True, data=True)
    } == before_edges
    assert dict(graph.graph) == before_graph_attrs


def test_node_cap_stops_the_walk_and_reports_truncation():
    graph = _star_graph(150)
    result = query_engineering_graph(
        graph, start_node_id="artifact:directory:center", max_depth=1
    )
    assert result.truncated is True
    assert len(result.nodes) == MAX_QUERY_NODES
    assert len(result.relationships) == MAX_QUERY_NODES - 1


def test_relationship_cap_stops_the_walk_and_reports_truncation():
    graph = _star_graph(99, kinds_per_leaf=4)
    result = query_engineering_graph(
        graph, start_node_id="artifact:directory:center", max_depth=1
    )
    assert result.truncated is True
    assert len(result.relationships) == MAX_QUERY_RELATIONSHIPS
    assert len(result.nodes) == 100


# ---------------------------------------------------------------------------
# Evidence preservation (spec §14)
# ---------------------------------------------------------------------------


def test_results_preserve_evidence_classification_and_origin():
    result = query_engineering_graph(
        _mini_graph(), start_node_id=B, direction="both", max_depth=2
    )
    # classifications travel with the entities — never upgraded
    by_id = {r["id"]: r for r in result.relationships}
    assert by_id["rel:CONTAINS:{}:{}".format(A, B)]["classification"] == "derived"
    assert by_id["rel:IMPORTS:{}:{}".format(B, C)]["classification"] == "observed"
    # flat evidence view carries origins + model-provided fields
    origins = {e["origin"] for e in result.evidence}
    assert "node:{}".format(C) in origins
    assert "relationship:rel:IMPORTS:{}:{}".format(B, C) in origins
    for entry in result.evidence:
        assert "commit" in entry
        assert entry["commit"] == COMMIT


def test_evidence_is_copied_not_shared_with_the_graph():
    graph = _mini_graph()
    result = query_engineering_graph(graph, start_node_id=A, max_depth=2)
    result.nodes[0]["evidence"][0]["commit"] = "MUTATED"
    result.relationships[0]["evidence"].clear()
    assert graph.nodes[A]["evidence"][0]["commit"] == COMMIT
    assert graph.edges[A, B, "rel:CONTAINS:{}:{}".format(A, B)]["evidence"] != []


# ---------------------------------------------------------------------------
# Summary mode (discovery path)
# ---------------------------------------------------------------------------


def test_summary_without_start_node_lists_kinds_and_samples():
    result = query_engineering_graph(_mini_graph())
    assert result.mode == "summary"
    assert result.start_node is None
    kinds = [(k["kind"], k["subkind"]) for k in result.kinds]
    assert kinds == [
        ("artifact", "directory"),
        ("artifact", "file"),
        ("code_entity", "module"),
    ]
    assert result.counts == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
        "kinds": 3,
    }
    samples = [s for k in result.kinds for s in k["sample_node_ids"]]
    assert set(samples) <= {A, B, C}
    assert len(samples) == 3


def test_summary_is_sorted_and_bounded():
    graph = _star_graph(150)
    result = query_engineering_graph(graph)
    assert result.mode == "summary"
    kinds = [(k["kind"], k["subkind"]) for k in result.kinds]
    assert kinds == sorted(kinds)
    files = [k for k in result.kinds if k["subkind"] == "file"][0]
    assert len(files["sample_node_ids"]) == 5  # bounded sample
    assert files["sample_node_ids"] == sorted(files["sample_node_ids"])
    assert files["node_count"] == 150


def test_summary_rejects_traversal_parameters():
    with pytest.raises(GraphQueryError) as excinfo:
        query_engineering_graph(_mini_graph(), direction="in")
    assert excinfo.value.code == "INVALID_ARGUMENT"
    with pytest.raises(GraphQueryError):
        query_engineering_graph(_mini_graph(), node_kinds=["artifact/file"])
    with pytest.raises(GraphQueryError):
        query_engineering_graph(_mini_graph(), relationship_kinds=["CONTAINS"])
    with pytest.raises(GraphQueryError):
        query_engineering_graph(_mini_graph(), max_depth=2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_direction_depth_and_kinds_raise_invalid_argument():
    graph = _mini_graph()
    for kwargs in (
        {"start_node_id": A, "direction": "sideways"},
        {"start_node_id": A, "max_depth": -1},
        {"start_node_id": A, "max_depth": 99},
        {"start_node_id": A, "max_depth": "2"},
        {"start_node_id": A, "relationship_kinds": ["USES_TOOL"]},
        {"start_node_id": A, "relationship_kinds": "CONTAINS"},
        {"start_node_id": A, "node_kinds": ["artifact/extra/stuff"]},
        {"start_node_id": A, "node_kinds": [""]},
        {"start_node_id": A, "node_kinds": [42]},
        {"start_node_id": ""},
        {"start_node_id": 7},
    ):
        with pytest.raises(GraphQueryError) as excinfo:
            query_engineering_graph(graph, **kwargs)
        assert excinfo.value.code == "INVALID_ARGUMENT", kwargs


def test_invalid_graph_argument_raises_invalid_argument():
    with pytest.raises(GraphQueryError) as excinfo:
        query_engineering_graph(None, start_node_id=A)
    assert excinfo.value.code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# Repository-agnosticism (spec §23)
# ---------------------------------------------------------------------------


def test_synthetic_query_api_needs_no_sample_repository():
    graph = _mini_graph()
    result = query_engineering_graph(graph, start_node_id=A, direction="out", max_depth=2)
    assert [n["id"] for n in result.nodes] == [A, B, C]
    assert [r["kind"] for r in result.relationships] == ["CONTAINS", "IMPORTS"]
    serialized = json.dumps(result.to_dict())
    for sample_specific in ("adop", "agentic", "dag-agent", ".mcp.json", "sample-"):
        assert sample_specific not in serialized.lower()


# ---------------------------------------------------------------------------
# Integration: pinned ADOP EngineeringModel
# ---------------------------------------------------------------------------


def test_adop_directory_contains_walk(adop_graph):
    result = query_engineering_graph(
        adop_graph,
        start_node_id="artifact:directory:.claude/agents",
        direction="out",
        max_depth=1,
    )
    assert result.start_node["id"] == "artifact:directory:.claude/agents"
    assert result.counts["nodes"] >= 2
    for relationship in result.relationships:
        assert relationship["kind"] == "CONTAINS"
        assert relationship["source"] == "artifact:directory:.claude/agents"


def test_adop_incoming_triggers_walk(adop_graph):
    dag = "artifact:agent_definition:.claude/agents/dag-agent.md"
    result = query_engineering_graph(adop_graph, start_node_id=dag, direction="in")
    assert result.start_node["id"] == dag
    assert result.relationships  # commands/docs reference the agent
    for relationship in result.relationships:
        assert relationship["target"] == dag
        assert relationship["kind"] == "TRIGGERS"


def test_adop_multi_edge_pair_is_queryable(adop_graph):
    source = "artifact:file:shared/utils/deterministic_yaml.py"
    result = query_engineering_graph(
        adop_graph, start_node_id=source, direction="out", max_depth=1
    )
    kinds = [r["kind"] for r in result.relationships]
    assert "CONTAINS" in kinds and "IMPORTS" in kinds  # both same-pair edges


def test_adop_summary_counts_match_the_model(adop_graph):
    result = query_engineering_graph(adop_graph)
    assert result.mode == "summary"
    assert result.counts["nodes"] == 1113
    assert result.counts["relationships"] == 1627
    assert result.counts["unresolved"] == 449
    kinds = [(k["kind"], k["subkind"]) for k in result.kinds]
    assert ("artifact", "agent_definition") in kinds
    assert ("artifact", "mcp_config") in kinds


# ---------------------------------------------------------------------------
# Named query API (M11.3 façade, synthetic)
# ---------------------------------------------------------------------------


def test_get_node_returns_an_entry_or_none():
    graph = _mini_graph()
    entry = get_node(graph, A)
    assert entry is not None
    assert entry["id"] == A and entry["depth"] == 0
    assert entry["kind"] == "artifact" and entry["evidence"]
    assert get_node(graph, "artifact:file:ghost") is None


def test_get_node_rejects_invalid_id_and_invalid_graph():
    graph = _mini_graph()
    for bad_id in ("", "   ", 7, None):
        with pytest.raises(GraphQueryError) as excinfo:
            get_node(graph, bad_id)
        assert excinfo.value.code == "INVALID_ARGUMENT"
    with pytest.raises(GraphQueryError) as excinfo:
        get_node(object(), A)
    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_neighbors_follow_direction_and_relationship_filter():
    graph = _mini_graph()
    assert [n["id"] for n in neighbors(graph, A, "out")] == [B]
    assert neighbors(graph, A, "in") == []  # the unresolved claim is not an edge
    assert [n["id"] for n in neighbors(graph, B, "both")] == [A, C]
    assert [n["id"] for n in neighbors(graph, B, "out", ["IMPORTS"])] == [C]
    assert neighbors(graph, B, "out", ["CONTAINS"]) == []
    assert neighbors(graph, A, "out")[0]["depth"] == 1


def test_neighbors_missing_node_and_invalid_arguments():
    graph = _mini_graph()
    with pytest.raises(GraphQueryError) as excinfo:
        neighbors(graph, "artifact:file:ghost")
    assert excinfo.value.code == "NODE_NOT_FOUND"
    with pytest.raises(GraphQueryError) as excinfo:
        neighbors(graph, A, "sideways")
    assert excinfo.value.code == "INVALID_ARGUMENT"
    with pytest.raises(GraphQueryError):
        neighbors(graph, A, relationship_kinds=["USES_TOOL"])


def test_relationships_are_sorted_filtered_and_keep_evidence():
    graph = _mini_graph()
    entries = relationships(graph, B, "both")
    ids = [entry["id"] for entry in entries]
    assert ids == sorted(ids)
    assert [entry["kind"] for entry in entries] == ["CONTAINS", "IMPORTS"]
    for entry in entries:
        assert entry["depth"] == 1
        assert entry["evidence"] and entry["classification"]
    filtered = relationships(graph, B, "out", ["TRIGGERS"])
    assert filtered == []
    with pytest.raises(GraphQueryError) as excinfo:
        relationships(graph, "artifact:file:ghost")
    assert excinfo.value.code == "NODE_NOT_FOUND"


def test_find_nodes_filters_by_kind_and_kind_subkind():
    graph = _mini_graph()
    assert [n["id"] for n in find_nodes(graph)] == [A, B, C]
    assert [n["id"] for n in find_nodes(graph, "artifact")] == [A, B]
    assert [n["id"] for n in find_nodes(graph, "artifact/file")] == [B]
    assert [
        n["id"] for n in find_nodes(graph, ["artifact/file", "code_entity/module"])
    ] == [B, C]
    for bad in ("artifact/extra/stuff", ["artifact/", 42] , 42, ""):
        with pytest.raises(GraphQueryError) as excinfo:
            find_nodes(graph, bad)
        assert excinfo.value.code == "INVALID_ARGUMENT"
    with pytest.raises(GraphQueryError):
        find_nodes(object())


def test_find_nodes_by_property_matches_top_level_and_attributes():
    graph = _mini_graph()
    assert [
        n["id"] for n in find_nodes_by_property(graph, "subkind", "file")
    ] == [B]
    assert [
        n["id"] for n in find_nodes_by_property(graph, "kind", "code_entity")
    ] == [C]
    # attributes key the node tree does not carry top-level
    assert [
        n["id"] for n in find_nodes_by_property(graph, "qualified_name", "pkg.mod")
    ] == [C]
    assert find_nodes_by_property(graph, "subkind", "class") == []
    assert find_nodes_by_property(graph, "not_a_property", "x") == []
    for bad_name in ("", "  ", 7, None):
        with pytest.raises(GraphQueryError) as excinfo:
            find_nodes_by_property(graph, bad_name, "x")
        assert excinfo.value.code == "INVALID_ARGUMENT"
    with pytest.raises(GraphQueryError):
        find_nodes_by_property(object(), "kind", "artifact")


def test_traverse_is_the_named_entry_for_the_single_query_result():
    graph = _mini_graph()
    expected = query_engineering_graph(
        graph, start_node_id=A, direction="both", max_depth=2
    ).to_dict()
    assert traverse(graph, A, "both", max_depth=2) == expected
    assert traverse(graph, A)["start_node"]["id"] == A
    with pytest.raises(GraphQueryError) as excinfo:
        traverse(graph, "artifact:file:ghost")
    assert excinfo.value.code == "NODE_NOT_FOUND"
    with pytest.raises(GraphQueryError) as excinfo:
        traverse(graph, A, "sideways")
    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_named_query_api_is_read_only_and_deterministic():
    graph = _mini_graph()
    nodes_before = {
        node_id: dict(data) for node_id, data in graph.nodes(data=True)
    }
    edges_before = sorted(
        (s, t, k, dict(d)) for s, t, k, d in graph.edges(keys=True, data=True)
    )
    graph_meta_before = dict(graph.graph)

    first = (
        get_node(graph, A),
        [n["id"] for n in neighbors(graph, B, "both")],
        [r["id"] for r in relationships(graph, B, "both")],
        [n["id"] for n in find_nodes(graph, "artifact")],
        [n["id"] for n in find_nodes_by_property(graph, "kind", "artifact")],
        traverse(graph, A, "both", max_depth=2),
    )
    second = (
        get_node(graph, A),
        [n["id"] for n in neighbors(graph, B, "both")],
        [r["id"] for r in relationships(graph, B, "both")],
        [n["id"] for n in find_nodes(graph, "artifact")],
        [n["id"] for n in find_nodes_by_property(graph, "kind", "artifact")],
        traverse(graph, A, "both", max_depth=2),
    )
    assert first == second  # deterministic
    assert {node_id: dict(data) for node_id, data in graph.nodes(data=True)} == (
        nodes_before
    )
    assert sorted(
        (s, t, k, dict(d)) for s, t, k, d in graph.edges(keys=True, data=True)
    ) == edges_before
    assert dict(graph.graph) == graph_meta_before  # never mutated


# ---------------------------------------------------------------------------
# Named query API (M11.3 façade, pinned ADOP EngineeringModel)
# ---------------------------------------------------------------------------


def test_adop_named_api_finds_mcp_and_agent_nodes(adop_graph):
    mcp = find_nodes(adop_graph, "artifact/mcp_config")
    assert [n["id"] for n in mcp] == ["artifact:mcp_config:.mcp.json"]
    assert len(find_nodes(adop_graph, "artifact/agent_definition")) == 11
    assert len(find_nodes_by_property(adop_graph, "kind", "artifact")) == 424
    node = get_node(adop_graph, "artifact:mcp_config:.mcp.json")
    assert node is not None
    assert node["subkind"] == "mcp_config"
    assert node["evidence"]
    assert get_node(adop_graph, "artifact:file:not-in-the-repo.txt") is None


def test_adop_named_api_relationships_and_traverse(adop_graph):
    dag = "artifact:agent_definition:.claude/agents/dag-agent.md"
    triggers = relationships(adop_graph, dag, "both", ["TRIGGERS"])
    assert triggers
    assert all(entry["kind"] == "TRIGGERS" for entry in triggers)
    assert all(entry["target"] == dag for entry in triggers)  # all incoming
    contained = neighbors(
        adop_graph, "artifact:directory:.claude/agents", "out", ["CONTAINS"]
    )
    assert len(contained) >= 5
    result = traverse(adop_graph, dag, direction="both", max_depth=2)
    assert result["start_node"]["id"] == dag
    assert result["counts"]["nodes"] >= 1
    node_ids = [entry["id"] for entry in result["nodes"]]
    assert node_ids == sorted(node_ids)
    assert result["truncated"] in (True, False)
