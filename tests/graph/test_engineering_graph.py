"""M11.1 — EngineeringModel → NetworkX projection tests.

Levels:
- unit: node/relationship projection, attributes, evidence,
  classifications, unresolved exclusion, duplicate ids, invalid
  endpoints, invariants, non-mutation
- integration: pinned ADOP EngineeringModel (skipped when absent)
- repository-agnostic: synthetic A/B/C model (spec §23) — proves the
  implementation is driven by the EngineeringModel contract, not by
  the sample repository

Offline — no network, no LLM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.engineering import EngineeringModel, EngineeringModelError
from repo_intelligence.graph import (
    build_engineering_graph,
    engineering_graph_invariant_problems,
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


def _mini_engineering_document():
    """Synthetic model shaped exactly like spec §23:

        A (directory)
         └── CONTAINS → B (file)
               └── IMPORTS → C (module)

    plus one unresolved claim (B → A, both endpoints exist) that must
    never become an edge.
    """
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {"name": "synthetic", "url": "", "commit_sha": COMMIT},
        "analysis": {"strategy": "deterministic", "analyzers": []},
        "engineering": {
            "nodes": [
                {
                    "id": "artifact:directory:pkg",
                    "kind": "artifact",
                    "subkind": "directory",
                    "classification": "observed",
                    "evidence": [
                        {"path": "pkg", "artifact_type": "directory", "commit": COMMIT}
                    ],
                    "attributes": {"path": "pkg"},
                },
                {
                    "id": "artifact:file:pkg/mod.py",
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
                    "id": "code_entity:pkg/mod.py",
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
                    "attributes": {"qualified_name": "pkg.mod", "name": "mod"},
                },
            ],
            "relationships": [
                {
                    "id": "rel:CONTAINS:artifact:directory:pkg:artifact:file:pkg/mod.py",
                    "kind": "CONTAINS",
                    "source": "artifact:directory:pkg",
                    "target": "artifact:file:pkg/mod.py",
                    "classification": "derived",
                    "evidence": [
                        {"path": "pkg", "artifact_type": "directory", "commit": COMMIT}
                    ],
                },
                {
                    "id": "rel:IMPORTS:artifact:file:pkg/mod.py:code_entity:pkg/mod.py",
                    "kind": "IMPORTS",
                    "source": "artifact:file:pkg/mod.py",
                    "target": "code_entity:pkg/mod.py",
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
                    "source": "artifact:file:pkg/mod.py",
                    "target": "artifact:directory:pkg",
                    "kind": "DEPENDS_ON",
                    "reason": "ambiguous_identity",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": COMMIT,
                            "line_start": 3,
                            "symbol": "pkg",
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


def _multi_edge_document():
    """Same node pair with two canonical relationships (real-model
    pattern: a file both CONTAINS and IMPORTS its code entity)."""
    document = _mini_engineering_document()
    document["engineering"]["relationships"].append(
        {
            "id": "rel:CONTAINS:artifact:file:pkg/mod.py:code_entity:pkg/mod.py",
            "kind": "CONTAINS",
            "source": "artifact:file:pkg/mod.py",
            "target": "code_entity:pkg/mod.py",
            "classification": "derived",
            "evidence": [
                {"path": "pkg/mod.py", "artifact_type": "python", "commit": COMMIT}
            ],
        }
    )
    return document


def _load_adop_engineering():
    return json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def adop_document():
    if not ADOP_ENG_MODEL.is_file():
        pytest.skip("ADOP EngineeringModel absent")
    return _load_adop_engineering()


# ---------------------------------------------------------------------------
# Projection (synthetic, repository-agnostic)
# ---------------------------------------------------------------------------


def test_node_projection_uses_canonical_ids_and_retains_metadata():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    assert set(graph.nodes) == {
        "artifact:directory:pkg",
        "artifact:file:pkg/mod.py",
        "code_entity:pkg/mod.py",
    }
    node = graph.nodes["artifact:file:pkg/mod.py"]
    assert node["kind"] == "artifact"
    assert node["subkind"] == "file"
    assert node["classification"] == "observed"
    assert node["attributes"] == {"path": "pkg/mod.py"}
    assert node["evidence"] == [
        {"path": "pkg/mod.py", "artifact_type": "python", "commit": COMMIT}
    ]
    code = graph.nodes["code_entity:pkg/mod.py"]
    assert code["classification"] == "derived"
    assert code["attributes"]["qualified_name"] == "pkg.mod"


def test_relationship_projection_keys_are_canonical_ids():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    assert graph.number_of_edges() == 2
    keys = {key for _, _, key in graph.edges(keys=True)}
    assert keys == {
        "rel:CONTAINS:artifact:directory:pkg:artifact:file:pkg/mod.py",
        "rel:IMPORTS:artifact:file:pkg/mod.py:code_entity:pkg/mod.py",
    }
    edge = graph.edges[
        "artifact:file:pkg/mod.py", "code_entity:pkg/mod.py",
        "rel:IMPORTS:artifact:file:pkg/mod.py:code_entity:pkg/mod.py",
    ]
    assert edge["kind"] == "IMPORTS"
    assert edge["classification"] == "observed"
    assert edge["source"] == "artifact:file:pkg/mod.py"
    assert edge["target"] == "code_entity:pkg/mod.py"
    assert edge["evidence"][0]["symbol"] == "mod"


def test_duplicate_pair_relationships_are_preserved():
    """A plain DiGraph would drop one of two same-pair edges."""
    document = _multi_edge_document()
    graph = build_engineering_graph(document)
    assert graph.number_of_edges() == 3
    pair_keys = [
        key
        for _, target, key in graph.out_edges("artifact:file:pkg/mod.py", keys=True)
        if target == "code_entity:pkg/mod.py"
    ]
    assert len(pair_keys) == 2
    kinds = {
        graph.edges["artifact:file:pkg/mod.py", "code_entity:pkg/mod.py", key][
            "kind"
        ]
        for key in pair_keys
    }
    assert kinds == {"CONTAINS", "IMPORTS"}


def test_unresolved_claim_never_becomes_an_edge():
    """Both endpoints of the unresolved claim exist as nodes — it must
    still not appear as an edge (the adapter is a projection)."""
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    assert graph.number_of_edges() == 2  # only the two canonical rels
    triples = {
        (data["source"], data["target"], data["kind"])
        for _, _, _, data in graph.edges(keys=True, data=True)
    }
    assert (
        "artifact:file:pkg/mod.py",
        "artifact:directory:pkg",
        "DEPENDS_ON",
    ) not in triples
    # unresolved information retained as graph metadata, not edges
    assert graph.graph["unresolved_count"] == 1


def test_adapter_does_not_mutate_the_model():
    document = _mini_engineering_document()
    before = copy.deepcopy(document["engineering"])
    graph = build_engineering_graph(document)
    for node_id in graph.nodes:
        graph.nodes[node_id]["kind"] = "MUTATED"
    for _, _, key, data in graph.edges(keys=True, data=True):
        data["kind"] = "MUTATED"
        data["evidence"] = []
        data["classification"] = "MUTATED"
    graph.add_node("artifact:file:injected")
    assert document["engineering"] == before


def test_projection_accepts_engineering_model_object():
    model = EngineeringModel.from_dict(_mini_engineering_document())
    graph = build_engineering_graph(model)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_projection_accepts_bare_engineering_section():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document["engineering"])
    assert graph.number_of_nodes() == 3


# ---------------------------------------------------------------------------
# Structural failures (never silent)
# ---------------------------------------------------------------------------


def test_node_without_id_is_rejected():
    document = _mini_engineering_document()
    del document["engineering"]["nodes"][0]["id"]
    with pytest.raises(EngineeringModelError):
        build_engineering_graph(document)


def test_duplicate_node_id_is_rejected():
    document = _mini_engineering_document()
    document["engineering"]["nodes"].append(
        dict(document["engineering"]["nodes"][0])
    )
    with pytest.raises(EngineeringModelError):
        build_engineering_graph(document)


def test_duplicate_relationship_id_is_rejected():
    document = _mini_engineering_document()
    document["engineering"]["relationships"].append(
        dict(document["engineering"]["relationships"][0])
    )
    with pytest.raises(EngineeringModelError):
        build_engineering_graph(document)


def test_relationship_without_endpoint_is_rejected():
    document = _mini_engineering_document()
    del document["engineering"]["relationships"][0]["target"]
    with pytest.raises(EngineeringModelError):
        build_engineering_graph(document)


def test_relationship_with_unknown_endpoint_is_rejected():
    document = _mini_engineering_document()
    document["engineering"]["relationships"][0]["target"] = "artifact:file:ghost"
    with pytest.raises(EngineeringModelError):
        build_engineering_graph(document)


def test_document_without_engineering_section_is_rejected():
    with pytest.raises(EngineeringModelError):
        build_engineering_graph({"schema_version": "repository_knowledge_model.v1"})


# ---------------------------------------------------------------------------
# Invariants (the established graph_invariant_problems pattern)
# ---------------------------------------------------------------------------


def test_invariants_hold_for_valid_projection():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    assert engineering_graph_invariant_problems(graph, document) == []
    # the bare section works too (same convention as the model input)
    assert (
        engineering_graph_invariant_problems(graph, document["engineering"]) == []
    )


def test_invariants_detect_invented_and_missing_nodes():
    document = _mini_engineering_document()
    invented = build_engineering_graph(document)
    invented.add_node("artifact:file:invented")
    problems = engineering_graph_invariant_problems(invented, document)
    assert "invented_graph_node: 'artifact:file:invented'" in problems
    assert any(p.startswith("graph_nodes_mismatch:") for p in problems)

    missing = build_engineering_graph(document)
    missing.remove_node("code_entity:pkg/mod.py")  # also drops its edge
    problems = engineering_graph_invariant_problems(missing, document)
    assert "missing_graph_node: 'code_entity:pkg/mod.py'" in problems
    assert any(p.startswith("graph_nodes_mismatch:") for p in problems)
    assert any(p.startswith("graph_edges_mismatch:") for p in problems)


def test_invariants_detect_invented_relationship():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    graph.add_edge(
        "artifact:directory:pkg",
        "code_entity:pkg/mod.py",
        key="rel:USES_TOOL:a:b",
        id="rel:USES_TOOL:a:b",
        kind="USES_TOOL",
        source="artifact:directory:pkg",
        target="code_entity:pkg/mod.py",
        classification="observed",
        evidence=[],
    )
    problems = engineering_graph_invariant_problems(graph, document)
    assert "invented_graph_relationship: 'rel:USES_TOOL:a:b'" in problems
    assert any(
        p.startswith("relationship_kind_not_in_vocabulary:") for p in problems
    )


def test_invariants_detect_kind_vocabulary_violation():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    key = "rel:CONTAINS:artifact:directory:pkg:artifact:file:pkg/mod.py"
    graph.edges[
        "artifact:directory:pkg", "artifact:file:pkg/mod.py", key
    ]["kind"] = "CALLS"
    problems = engineering_graph_invariant_problems(graph, document)
    assert any(
        "relationship_kind_not_in_vocabulary: 'CALLS'" in p
        for p in problems
    )


def test_invariants_detect_unresolved_claim_leak():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    leaked_id = "rel:DEPENDS_ON:artifact:file:pkg/mod.py:artifact:directory:pkg"
    graph.add_edge(
        "artifact:file:pkg/mod.py",
        "artifact:directory:pkg",
        key=leaked_id,
        id=leaked_id,
        kind="DEPENDS_ON",
        source="artifact:file:pkg/mod.py",
        target="artifact:directory:pkg",
        classification="observed",
        evidence=[],
    )
    problems = engineering_graph_invariant_problems(graph, document)
    assert "unresolved_relationship_edge: {!r}".format(leaked_id) in problems


def test_invariants_detect_edge_key_mismatch():
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    # re-key an edge so key != canonical relationship id
    data = dict(
        graph.edges[
            "artifact:directory:pkg",
            "artifact:file:pkg/mod.py",
            "rel:CONTAINS:artifact:directory:pkg:artifact:file:pkg/mod.py",
        ]
    )
    graph.remove_edge(
        "artifact:directory:pkg",
        "artifact:file:pkg/mod.py",
        "rel:CONTAINS:artifact:directory:pkg:artifact:file:pkg/mod.py",
    )
    graph.add_edge(
        "artifact:directory:pkg", "artifact:file:pkg/mod.py", key="wrong", **data
    )
    problems = engineering_graph_invariant_problems(graph, document)
    assert any(p.startswith("edge_key_mismatch:") for p in problems)


# ---------------------------------------------------------------------------
# Repository-agnosticism (spec §23)
# ---------------------------------------------------------------------------


def test_synthetic_abc_graph_needs_no_sample_repository():
    """A --CONTAINS--> B --IMPORTS--> C built from the generic contract
    only — no sample-repository names anywhere in the inputs or the
    resulting graph."""
    document = _mini_engineering_document()
    graph = build_engineering_graph(document)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert engineering_graph_invariant_problems(graph, document) == []
    # A→B and B→C exist; A→C does not (no invented transitive edge)
    assert graph.has_edge("artifact:directory:pkg", "artifact:file:pkg/mod.py")
    assert graph.has_edge("artifact:file:pkg/mod.py", "code_entity:pkg/mod.py")
    assert not graph.has_edge("artifact:directory:pkg", "code_entity:pkg/mod.py")
    serialized = json.dumps(document)
    for sample_specific in (
        "adop",
        "agentic",
        "dag-agent",
        ".mcp.json",
        "sample-",
    ):
        assert sample_specific not in serialized


def test_projection_sources_are_repository_agnostic():
    root = PROJECT_ROOT / "src" / "repo_intelligence"
    for relative in (
        "graph/engineering_adapter.py",
        "graph/engineering_graph_persist.py",
        "graph/engineering_query.py",
        "tools/query_engineering_graph.py",
    ):
        path = root / relative
        if not path.is_file():  # query layer lands in the next step
            continue
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("adop", "agentic", ".mcp.json", "dag-agent", "sample-"):
            assert banned not in text, "{} contains {!r}".format(relative, banned)


# ---------------------------------------------------------------------------
# Integration: pinned ADOP EngineeringModel
# ---------------------------------------------------------------------------


def test_adop_engineering_graph_projection(adop_document):
    engineering = adop_document["engineering"]
    graph = build_engineering_graph(adop_document)
    assert graph.number_of_nodes() == len(engineering["nodes"]) == 1113
    assert graph.number_of_edges() == len(engineering["relationships"]) == 1627
    assert graph.graph["unresolved_count"] == len(engineering["unresolved"]) == 449
    assert engineering_graph_invariant_problems(graph, adop_document) == []


def test_adop_engineering_graph_keeps_multi_edge(adop_document):
    """The real model contains one same-pair CONTAINS+IMPORTS edge
    pair — a DiGraph projection would have lost one of them."""
    graph = build_engineering_graph(adop_document)
    source = "artifact:file:shared/utils/deterministic_yaml.py"
    target = "code_entity:shared/utils/deterministic_yaml.py"
    pair_keys = [
        key
        for s, t, key in graph.out_edges(source, keys=True)
        if t == target
    ]
    assert len(pair_keys) == 2
    kinds = {
        graph.edges[source, target, key]["kind"] for key in pair_keys
    }
    assert kinds == {"CONTAINS", "IMPORTS"}


def test_adop_unresolved_claims_are_not_edges(adop_document):
    graph = build_engineering_graph(adop_document)
    edge_ids = {
        data["id"] for _, _, _, data in graph.edges(keys=True, data=True)
    }
    for claim in adop_document["engineering"]["unresolved"]:
        derived = "rel:{}:{}:{}".format(
            claim["kind"], claim["source"], claim["target"]
        )
        assert derived not in edge_ids
    assert graph.number_of_edges() == 1627  # 449 unresolved added nothing
