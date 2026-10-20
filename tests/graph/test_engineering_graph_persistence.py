"""M11.1/M11.2 — persisted engineering_graph.json artifact tests.

Levels:
- unit (synthetic A/B/C model, spec §23): document layout, provenance,
  verbatim content preservation, canonical ordering, determinism,
  validation (ids, endpoints, counts), write/read round-trip,
  projection equivalence (model → graph vs artifact → graph),
  generation entry point, CLI, graph non-mutation
- integration: pinned ADOP EngineeringModel (skipped when absent),
  including byte-identical regeneration of the on-disk artifact

Offline — no network, no LLM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID, engineering_graph_path
from repo_intelligence.engineering import EngineeringModelError
from repo_intelligence.graph import (
    ENGINEERING_GRAPH_SCHEMA_VERSION,
    build_engineering_graph,
    build_engineering_graph_document,
    engineering_graph_document_problems,
    engineering_graph_invariant_problems,
    generate_engineering_graph_document,
    load_engineering_graph_document,
    project_engineering_graph,
    validate_engineering_graph_document,
    write_engineering_graph_document,
)
from repo_intelligence.graph import engineering_graph_persist as persist_module

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
    """A --CONTAINS--> B --IMPORTS--> C (plus one unresolved claim),
    with deliberately unsorted node/relationship lists."""
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {"name": "synthetic", "url": "", "commit_sha": COMMIT},
        "analysis": {"strategy": "deterministic", "analyzers": []},
        "engineering": {
            "nodes": [
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
                    "attributes": {"qualified_name": "pkg.mod", "name": "mod"},
                },
                {
                    "id": A,
                    "kind": "artifact",
                    "subkind": "directory",
                    "classification": "observed",
                    "evidence": [
                        {
                            "path": "pkg",
                            "artifact_type": "directory",
                            "commit": COMMIT,
                        }
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
            ],
            "relationships": [
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
                            "line_end": 1,
                        }
                    ],
                },
                {
                    "id": "rel:CONTAINS:{}:{}".format(A, B),
                    "kind": "CONTAINS",
                    "source": A,
                    "target": B,
                    "classification": "derived",
                    "evidence": [
                        {
                            "path": "pkg",
                            "artifact_type": "directory",
                            "commit": COMMIT,
                        }
                    ],
                },
            ],
            "unresolved": [
                {
                    "source": C,
                    "target": A,
                    "kind": "DEPENDS_ON",
                    "reason": "target_not_resolved",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": COMMIT,
                            "line_start": 3,
                            "line_end": 3,
                        }
                    ],
                }
            ],
        },
    }


@pytest.fixture(scope="module")
def adop_document():
    if not ADOP_ENG_MODEL.is_file():
        pytest.skip("ADOP EngineeringModel absent")
    return json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Build (synthetic, repository-agnostic)
# ---------------------------------------------------------------------------


def test_module_exposes_the_m11_1_api():
    assert ENGINEERING_GRAPH_SCHEMA_VERSION == "engineering_graph.v1"
    for name in (
        "build_engineering_graph_document",
        "write_engineering_graph_document",
        "load_engineering_graph_document",
        "generate_engineering_graph_document",
        "engineering_graph_document_problems",
        "validate_engineering_graph_document",
    ):
        assert name in persist_module.__all__
        assert callable(getattr(persist_module, name))


def test_build_document_has_layout_metadata_and_provenance():
    document = build_engineering_graph_document(
        _mini_engineering_document(), snapshot_id="synthetic-run"
    )
    assert list(document.keys()) == [
        "schema_version",
        "provenance",
        "metadata",
        "nodes",
        "relationships",
        "unresolved",
    ]
    assert document["schema_version"] == ENGINEERING_GRAPH_SCHEMA_VERSION
    assert document["metadata"]["counts"] == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
    }
    provenance = document["provenance"]
    assert provenance["source_artifact"] == "engineering_model.json"
    assert provenance["source_schema_version"] == (
        "repository_knowledge_model.v1"
    )
    assert provenance["repository"]["commit_sha"] == COMMIT
    assert provenance["snapshot_id"] == "synthetic-run"
    assert provenance["generator"] == (
        "repo_intelligence.graph.engineering_graph_persist"
    )


def test_build_preserves_content_verbatim_and_does_not_mutate_model():
    model = _mini_engineering_document()
    snapshot = copy.deepcopy(model)
    document = build_engineering_graph_document(model)
    assert model == snapshot  # source of truth untouched
    nodes = {node["id"]: node for node in document["nodes"]}
    for original in snapshot["engineering"]["nodes"]:
        assert nodes[original["id"]] == original
    relationships = {
        relationship["id"]: relationship
        for relationship in document["relationships"]
    }
    for original in snapshot["engineering"]["relationships"]:
        assert relationships[original["id"]] == original
    assert document["unresolved"] == snapshot["engineering"]["unresolved"]
    # deep copies, not shared references
    assert document["nodes"][0] is not snapshot["engineering"]["nodes"][2]


def test_build_orders_nodes_and_relationships_by_id():
    document = build_engineering_graph_document(_mini_engineering_document())
    node_ids = [node["id"] for node in document["nodes"]]
    assert node_ids == sorted(node_ids) == [A, B, C]
    relationship_ids = [
        relationship["id"] for relationship in document["relationships"]
    ]
    assert relationship_ids == sorted(relationship_ids)


def test_build_is_deterministic_regardless_of_input_order():
    first = build_engineering_graph_document(
        _mini_engineering_document(), snapshot_id="run"
    )
    reshuffled = _mini_engineering_document()
    reshuffled["engineering"]["nodes"].reverse()
    reshuffled["engineering"]["relationships"].reverse()
    second = build_engineering_graph_document(reshuffled, snapshot_id="run")
    assert json.dumps(first, indent=2, ensure_ascii=False) == json.dumps(
        second, indent=2, ensure_ascii=False
    )


def test_build_rejects_missing_engineering_section():
    with pytest.raises(EngineeringModelError):
        build_engineering_graph_document({"schema_version": "x"})


def test_build_rejects_non_object_input():
    with pytest.raises(EngineeringModelError):
        build_engineering_graph_document(["not", "a", "document"])


def test_build_rejects_duplicate_node_id():
    model = _mini_engineering_document()
    model["engineering"]["nodes"].append(
        copy.deepcopy(model["engineering"]["nodes"][0])
    )
    with pytest.raises(EngineeringModelError) as excinfo:
        build_engineering_graph_document(model)
    assert "duplicate node id" in str(excinfo.value)


def test_build_rejects_relationship_with_missing_id():
    model = _mini_engineering_document()
    del model["engineering"]["relationships"][0]["id"]
    with pytest.raises(EngineeringModelError):
        build_engineering_graph_document(model)


def test_build_rejects_relationship_with_missing_endpoint():
    model = _mini_engineering_document()
    del model["engineering"]["relationships"][0]["target"]
    with pytest.raises(EngineeringModelError):
        build_engineering_graph_document(model)


def test_build_rejects_relationship_with_unknown_endpoint():
    model = _mini_engineering_document()
    model["engineering"]["relationships"][0]["target"] = "artifact:file:ghost"
    with pytest.raises(EngineeringModelError) as excinfo:
        build_engineering_graph_document(model)
    assert "not among graph nodes" in str(excinfo.value)


def test_build_from_bare_engineering_section():
    document = build_engineering_graph_document(
        _mini_engineering_document()["engineering"]
    )
    assert document["metadata"]["counts"]["nodes"] == 3
    assert document["provenance"]["source_schema_version"] is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_valid_document_has_no_problems():
    document = build_engineering_graph_document(_mini_engineering_document())
    assert engineering_graph_document_problems(document) == []
    validate_engineering_graph_document(document)  # no raise


def test_problems_report_wrong_schema_duplicate_id_and_counts_mismatch():
    document = build_engineering_graph_document(_mini_engineering_document())

    wrong_schema = copy.deepcopy(document)
    wrong_schema["schema_version"] = "repository_knowledge_model.v1"
    assert any(
        "schema_version" in problem
        for problem in engineering_graph_document_problems(wrong_schema)
    )

    duplicate = copy.deepcopy(document)
    duplicate["nodes"][1] = copy.deepcopy(duplicate["nodes"][0])
    assert any(
        "duplicate node id" in problem
        for problem in engineering_graph_document_problems(duplicate)
    )

    bad_counts = copy.deepcopy(document)
    bad_counts["metadata"]["counts"]["relationships"] = 99
    assert any(
        "metadata.counts.relationships" in problem
        for problem in engineering_graph_document_problems(bad_counts)
    )

    non_list = copy.deepcopy(document)
    non_list["relationships"] = {}
    assert any(
        "must be a list" in problem
        for problem in engineering_graph_document_problems(non_list)
    )

    with pytest.raises(EngineeringModelError):
        validate_engineering_graph_document(wrong_schema)
    assert engineering_graph_document_problems("not a document") == [
        "document root is not a JSON object"
    ]


# ---------------------------------------------------------------------------
# Write / load round-trip
# ---------------------------------------------------------------------------


def test_write_uses_project_serialization_convention(tmp_path):
    document = build_engineering_graph_document(_mini_engineering_document())
    destination = tmp_path / "nested" / "engineering_graph.json"
    written = write_engineering_graph_document(document, destination)
    assert written == destination and destination.is_file()
    text = destination.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert text.startswith('{\n  "schema_version": "engineering_graph.v1"')
    reparsed = json.loads(text)
    assert reparsed == document


def test_load_round_trip_returns_an_equal_document(tmp_path):
    document = build_engineering_graph_document(
        _mini_engineering_document(), snapshot_id="run-1"
    )
    written = write_engineering_graph_document(
        document, tmp_path / "engineering_graph.json"
    )
    assert load_engineering_graph_document(written) == document


def test_load_missing_file_is_an_error(tmp_path):
    with pytest.raises(EngineeringModelError) as excinfo:
        load_engineering_graph_document(tmp_path / "absent.json")
    assert "Cannot read engineering graph" in str(excinfo.value)


def test_load_invalid_json_is_an_error(tmp_path):
    broken = tmp_path / "engineering_graph.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(EngineeringModelError) as excinfo:
        load_engineering_graph_document(broken)
    assert "not valid JSON" in str(excinfo.value)


def test_load_structurally_invalid_document_is_an_error(tmp_path):
    document = build_engineering_graph_document(_mini_engineering_document())
    document["metadata"]["counts"]["nodes"] = 42
    destination = tmp_path / "engineering_graph.json"
    destination.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(EngineeringModelError) as excinfo:
        load_engineering_graph_document(destination)
    assert "metadata.counts.nodes" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Projection equivalence (M11.2): model → graph == artifact → graph
# ---------------------------------------------------------------------------


def test_projection_from_graph_document_equals_projection_from_model():
    model = _mini_engineering_document()
    document = build_engineering_graph_document(model)
    from_model = build_engineering_graph(model)
    from_document = project_engineering_graph(document)
    assert set(from_model.nodes) == set(from_document.nodes)
    assert {
        (s, t, k): dict(d)
        for s, t, k, d in from_model.edges(keys=True, data=True)
    } == {
        (s, t, k): dict(d)
        for s, t, k, d in from_document.edges(keys=True, data=True)
    }
    for node_id in from_model.nodes:
        assert from_model.nodes[node_id] == from_document.nodes[node_id]
    assert from_model.graph == from_document.graph
    assert engineering_graph_invariant_problems(from_document, document) == []


def test_projection_from_graph_document_rejects_wrong_schema():
    document = build_engineering_graph_document(_mini_engineering_document())
    document["schema_version"] = "engineering_graph.v2"
    with pytest.raises(EngineeringModelError) as excinfo:
        project_engineering_graph(document)
    assert "engineering_graph.v1" in str(excinfo.value)
    with pytest.raises(EngineeringModelError):
        project_engineering_graph(
            _mini_engineering_document()  # a model document, not a graph doc
        )


def test_unresolved_claim_never_becomes_an_edge_in_the_artifact_projection():
    document = build_engineering_graph_document(_mini_engineering_document())
    assert len(document["unresolved"]) == 1  # carried as data
    graph = project_engineering_graph(document)
    edge_ids = {
        data["id"] for _, _, _, data in graph.edges(keys=True, data=True)
    }
    assert "rel:DEPENDS_ON:{}:{}".format(C, A) not in edge_ids
    assert graph.number_of_edges() == 2
    assert graph.graph["unresolved_count"] == 1


def test_projection_from_graph_document_does_not_mutate_the_document():
    document = build_engineering_graph_document(_mini_engineering_document())
    snapshot = copy.deepcopy(document)
    project_engineering_graph(document)
    assert document == snapshot


# ---------------------------------------------------------------------------
# Generation entry point + CLI
# ---------------------------------------------------------------------------


def test_generate_writes_the_graph_artifact_beside_the_model(tmp_path):
    model_path = tmp_path / "engineering_model.json"
    model_path.write_text(
        json.dumps(_mini_engineering_document(), indent=2) + "\n",
        encoding="utf-8",
    )
    document, written = generate_engineering_graph_document(
        model_path=model_path
    )
    assert written == tmp_path / "engineering_graph.json"
    assert written.is_file()
    assert document["metadata"]["counts"] == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
    }
    assert load_engineering_graph_document(written) == document
    # the model file itself is untouched
    assert json.loads(model_path.read_text(encoding="utf-8")) == (
        _mini_engineering_document()
    )


def test_generate_from_missing_model_is_an_error(tmp_path):
    with pytest.raises(EngineeringModelError):
        generate_engineering_graph_document(
            model_path=tmp_path / "absent.json"
        )


def test_cli_writes_the_artifact_and_prints_a_summary(tmp_path, capsys):
    model_path = tmp_path / "engineering_model.json"
    model_path.write_text(
        json.dumps(_mini_engineering_document()), encoding="utf-8"
    )
    exit_code = persist_module.main(
        [
            "--model",
            str(model_path),
            "--output",
            str(tmp_path / "out" / "engineering_graph.json"),
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == ENGINEERING_GRAPH_SCHEMA_VERSION
    assert summary["counts"] == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
    }
    assert Path(summary["path"]).is_file()


def test_cli_failure_exits_with_six(tmp_path, capsys):
    exit_code = persist_module.main(
        ["--model", str(tmp_path / "absent.json")]
    )
    assert exit_code == 6
    assert "EngineeringModel not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Integration: pinned ADOP EngineeringModel
# ---------------------------------------------------------------------------


def test_adop_graph_document_counts_and_verbatim_content(adop_document):
    document = build_engineering_graph_document(
        adop_document, snapshot_id=PRIMARY_SNAPSHOT_ID
    )
    assert document["metadata"]["counts"] == {
        "nodes": 1113,
        "relationships": 1627,
        "unresolved": 449,
    }
    assert engineering_graph_document_problems(document) == []
    original_nodes = {
        node["id"]: node for node in adop_document["engineering"]["nodes"]
    }
    assert {node["id"]: node for node in document["nodes"]} == original_nodes
    original_relationships = {
        relationship["id"]: relationship
        for relationship in adop_document["engineering"]["relationships"]
    }
    assert {
        relationship["id"]: relationship
        for relationship in document["relationships"]
    } == original_relationships
    assert document["unresolved"] == adop_document["engineering"]["unresolved"]
    assert document["provenance"]["snapshot_id"] == PRIMARY_SNAPSHOT_ID


def test_adop_graph_document_regeneration_is_byte_identical():
    first = build_engineering_graph_document(
        json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8")),
        snapshot_id=PRIMARY_SNAPSHOT_ID,
    )
    second = build_engineering_graph_document(
        json.loads(ADOP_ENG_MODEL.read_text(encoding="utf-8")),
        snapshot_id=PRIMARY_SNAPSHOT_ID,
    )
    dumps = lambda document: json.dumps(  # noqa: E731
        document, indent=2, ensure_ascii=False, sort_keys=False
    )
    assert dumps(first) == dumps(second)


def test_adop_projection_from_artifact_equals_projection_from_model(
    adop_document,
):
    document = build_engineering_graph_document(
        adop_document, snapshot_id=PRIMARY_SNAPSHOT_ID
    )
    from_model = build_engineering_graph(adop_document)
    from_document = project_engineering_graph(document)
    assert from_document.number_of_nodes() == 1113
    assert from_document.number_of_edges() == 1627
    assert from_document.graph["unresolved_count"] == 449
    assert set(from_model.nodes) == set(from_document.nodes)
    assert {
        (s, t, k): dict(d)
        for s, t, k, d in from_model.edges(keys=True, data=True)
    } == {
        (s, t, k): dict(d)
        for s, t, k, d in from_document.edges(keys=True, data=True)
    }
    assert engineering_graph_invariant_problems(from_document, document) == []


def test_adop_artifact_projection_keeps_the_duplicate_pair_and_drops_unresolved(
    adop_document,
):
    document = build_engineering_graph_document(adop_document)
    graph = project_engineering_graph(document)
    source = "artifact:file:shared/utils/deterministic_yaml.py"
    target = "code_entity:shared/utils/deterministic_yaml.py"
    kinds = {
        graph.edges[source, target, key]["kind"]
        for key in graph[source][target]
    }
    assert kinds == {"CONTAINS", "IMPORTS"}
    edge_ids = {
        data["id"] for _, _, _, data in graph.edges(keys=True, data=True)
    }
    for claim in adop_document["engineering"]["unresolved"]:
        derived = "rel:{}:{}:{}".format(
            claim["kind"], claim["source"], claim["target"]
        )
        assert derived not in edge_ids
    assert graph.number_of_edges() == 1627


def test_adop_on_disk_artifact_matches_regeneration(adop_document):
    """The committed artifact is exactly what the generator produces —
    persisted and regeneration are the same bytes (spec §17)."""
    destination = engineering_graph_path(
        PROJECT_ROOT / "artifacts", PRIMARY_SNAPSHOT_ID
    )
    if not destination.is_file():
        pytest.skip("engineering_graph.json not generated yet")
    on_disk = destination.read_text(encoding="utf-8")
    regenerated = json.dumps(
        build_engineering_graph_document(
            adop_document, snapshot_id=PRIMARY_SNAPSHOT_ID
        ),
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
    )
    assert on_disk == regenerated + "\n"
    loaded = load_engineering_graph_document(destination)
    assert loaded["metadata"]["counts"] == {
        "nodes": 1113,
        "relationships": 1627,
        "unresolved": 449,
    }
