"""M5 NetworkX graph runtime and diagram export tests.

Levels:
- unit: adapter invariants, projection, exporters, validation
- integration: pinned ADOP ArchitectureModel (skipped when absent)
- regression: full suite (run separately via tests/)

Tests 1–12 cover definition-of-done items for the graph → diagram boundary.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from repo_intelligence.architecture import validate_knowledge_model_or_raise
from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.graph import (
    DiagramValidationError,
    build_architecture_graph,
    build_diagram_projection,
    create_diagram_artifacts,
    export_graphviz,
    export_mermaid,
    graph_invariant_problems,
    graphviz_available,
    validate_diagram_projection,
    validate_diagram_projection_or_raise,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADOP_ARCH_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "architecture"
    / "architecture_model.json"
)
ADOP_COMMIT = "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
DOA_ID = "concept:agent:data-onboarding-agent"


def _load_adop_architecture():
    return json.loads(ADOP_ARCH_MODEL.read_text(encoding="utf-8"))


def _mini_architecture_document():
    """Small architecture with one relationship + one unresolved claim."""
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {"name": "mini", "commit_sha": "b" * 40},
        "analysis": {},
        "engineering": {"nodes": [], "relationships": [], "unresolved": []},
        "architecture": {
            "nodes": [
                {
                    "id": "concept:agent:alpha",
                    "kind": "concept",
                    "subkind": "agent",
                    "attributes": {"name": "Alpha Agent"},
                    "classification": "derived",
                    "evidence": [{"path": "a.md", "commit": "b" * 40}],
                },
                {
                    "id": "concept:agent:beta",
                    "kind": "concept",
                    "subkind": "agent",
                    "attributes": {"name": "Beta Agent"},
                    "classification": "derived",
                    "evidence": [{"path": "b.md", "commit": "b" * 40}],
                },
            ],
            "relationships": [
                {
                    "id": "rel:TRIGGERS:concept:agent:alpha:concept:agent:beta",
                    "kind": "TRIGGERS",
                    "source": "concept:agent:alpha",
                    "target": "concept:agent:beta",
                    "classification": "derived",
                    "evidence": [{"path": "a.md", "commit": "b" * 40}],
                }
            ],
            "unresolved": [
                {
                    "kind": "TRIGGERS",
                    "source": "artifact:file:ghost.md",
                    "target": "concept:agent:alpha",
                    "reason": "missing_endpoint",
                    "evidence": [{"path": "ghost.md", "commit": "b" * 40}],
                }
            ],
            "metadata": {},
        },
        "mappings": [],
        "metadata": {},
    }


@pytest.fixture(scope="module")
def adop_document():
    if not ADOP_ARCH_MODEL.is_file():
        pytest.skip("ADOP ArchitectureModel absent")
    return _load_adop_architecture()


# ---------------------------------------------------------------------------
# M5.1 — NetworkX adapter
# ---------------------------------------------------------------------------


def test_networkx_node_count(adop_document):
    """Test 1 — ArchitectureModel nodes == NetworkX nodes (12 on current ADOP)."""
    arch_nodes = adop_document["architecture"]["nodes"]
    graph = build_architecture_graph(adop_document)
    assert graph.number_of_nodes() == len(arch_nodes)
    assert graph.number_of_nodes() == 12
    assert graph_invariant_problems(
        graph, adop_document["architecture"]
    ) == []


def test_networkx_edge_count(adop_document):
    """Test 2 — ArchitectureModel relationships == NetworkX edges (7)."""
    arch_rels = adop_document["architecture"]["relationships"]
    graph = build_architecture_graph(adop_document)
    assert graph.number_of_edges() == len(arch_rels)
    assert graph.number_of_edges() == 7
    assert graph_invariant_problems(
        graph, adop_document["architecture"]
    ) == []


def test_networkx_node_identity(adop_document):
    """Test 3 — every NetworkX node ID is an ArchitectureModel node ID."""
    arch_ids = {n["id"] for n in adop_document["architecture"]["nodes"]}
    graph = build_architecture_graph(adop_document)
    for node_id in graph.nodes:
        assert node_id in arch_ids
    assert set(graph.nodes) == arch_ids


def test_networkx_edge_identity(adop_document):
    """Test 4 — every NetworkX edge corresponds to a canonical relationship."""
    arch_rel_ids = {
        r["id"] for r in adop_document["architecture"]["relationships"]
    }
    graph = build_architecture_graph(adop_document)
    for _, _, data in graph.edges(data=True):
        assert data.get("id") in arch_rel_ids
        assert data.get("kind") == "TRIGGERS"
        assert data.get("source") in set(graph.nodes)
        assert data.get("target") in set(graph.nodes)


def test_unresolved_claims_not_in_graph():
    """Test 5 — unresolved claims never become NetworkX edges."""
    document = _mini_architecture_document()
    graph = build_architecture_graph(document)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    edge_ids = {data["id"] for _, _, data in graph.edges(data=True)}
    assert edge_ids == {
        "rel:TRIGGERS:concept:agent:alpha:concept:agent:beta"
    }
    # ghost.md source is unresolved only
    assert not any(
        data.get("source") == "artifact:file:ghost.md"
        for _, _, data in graph.edges(data=True)
    )


def test_adapter_does_not_mutate_architecture(adop_document):
    """Adapter must not modify ArchitectureModel sections."""
    before = copy.deepcopy(
        {
            "architecture": adop_document["architecture"],
            "mappings": adop_document.get("mappings"),
        }
    )
    graph = build_architecture_graph(adop_document)
    # mutate graph attrs
    for node_id in graph.nodes:
        graph.nodes[node_id]["name"] = "MUTATED"
    for _, _, data in graph.edges(data=True):
        data["kind"] = "MUTATED"
        data["evidence"] = []
    graph.add_node("concept:agent:injected")
    assert adop_document["architecture"] == before["architecture"]
    assert adop_document.get("mappings") == before["mappings"]
    assert DOA_ID in {
        n["id"] for n in adop_document["architecture"]["nodes"]
    }


# ---------------------------------------------------------------------------
# Diagram projection
# ---------------------------------------------------------------------------


def test_diagram_projection_completeness(adop_document):
    """Test 6 — projection nodes/relationships mirror ArchitectureModel."""
    graph = build_architecture_graph(adop_document)
    projection = build_diagram_projection(graph)
    arch = adop_document["architecture"]

    assert len(projection.nodes) == len(arch["nodes"])
    assert len(projection.relationships) == len(arch["relationships"])

    proj_arch_ids = [n["architecture_node_id"] for n in projection.nodes]
    arch_ids = [n["id"] for n in arch["nodes"]]
    assert sorted(proj_arch_ids) == sorted(arch_ids)

    proj_rel_ids = [
        r["architecture_relationship_id"] for r in projection.relationships
    ]
    arch_rel_ids = [r["id"] for r in arch["relationships"]]
    assert sorted(proj_rel_ids) == sorted(arch_rel_ids)

    # labels come from ArchitectureModel attributes.name
    name_by_id = {
        n["id"]: n["attributes"]["name"] for n in arch["nodes"]
    }
    for node in projection.nodes:
        assert node["label"] == name_by_id[node["architecture_node_id"]]

    assert validate_diagram_projection(projection, adop_document) == []


def test_diagram_ids_deterministic_from_sorted_canonical_ids():
    document = _mini_architecture_document()
    # reverse node order — projection must still assign by sorted arch id
    document["architecture"]["nodes"] = list(
        reversed(document["architecture"]["nodes"])
    )
    graph = build_architecture_graph(document)
    projection = build_diagram_projection(graph)
    sorted_ids = sorted(
        n["architecture_node_id"] for n in projection.nodes
    )
    assert sorted_ids == [
        "concept:agent:alpha",
        "concept:agent:beta",
    ]
    by_arch = projection.diagram_id_by_architecture_id()
    assert by_arch["concept:agent:alpha"] == "node_001"
    assert by_arch["concept:agent:beta"] == "node_002"


def test_projection_rejects_invented_relationship(adop_document):
    graph = build_architecture_graph(adop_document)
    projection = build_diagram_projection(graph)
    projection.relationships.append(
        {
            "architecture_relationship_id": "rel:TRIGGERS:not:real:not:real",
            "source_architecture_node_id": "concept:agent:fake",
            "target_architecture_node_id": "concept:agent:fake2",
            "source_diagram_id": "node_999",
            "target_diagram_id": "node_998",
            "label": "TRIGGERS",
        }
    )
    problems = validate_diagram_projection(projection, adop_document)
    assert problems
    with pytest.raises(DiagramValidationError):
        validate_diagram_projection_or_raise(projection, adop_document)


# ---------------------------------------------------------------------------
# M5.2 / M5.3 — exporters
# ---------------------------------------------------------------------------


def test_mermaid_export(adop_document):
    """Test 7 — Mermaid includes all nodes, 7 relationships, TRIGGERS labels."""
    graph = build_architecture_graph(adop_document)
    projection = build_diagram_projection(graph)
    mermaid = export_mermaid(projection)

    assert mermaid.startswith("flowchart TD")
    for node in projection.nodes:
        assert node["diagram_id"] in mermaid
        assert node["label"] in mermaid
    triggers_edges = [
        r
        for r in adop_document["architecture"]["relationships"]
        if r["kind"] == "TRIGGERS"
    ]
    assert len(triggers_edges) == 7
    for rel in projection.relationships:
        assert (
            "{} -->|{}| {}".format(
                rel["source_diagram_id"],
                rel["label"],
                rel["target_diagram_id"],
            )
            in mermaid
        )
    assert mermaid.count("-->|TRIGGERS|") == 7
    # no unresolved / no invented labels from unresolved claims
    assert "ghost" not in mermaid
    assert "artifact:file" not in mermaid
    # labels are display names, not raw architecture ids as primary label
    assert "Data Onboarding Agent" in mermaid
    assert DOA_ID not in mermaid


def test_graphviz_export(adop_document):
    """Test 8 — DOT includes all nodes/relationships with correct labels."""
    graph = build_architecture_graph(adop_document)
    projection = build_diagram_projection(graph)
    dot = export_graphviz(projection)

    assert dot.startswith("digraph architecture {")
    assert dot.rstrip().endswith("}")
    for node in projection.nodes:
        assert '{} [label="{}"];'.format(node["diagram_id"], node["label"]) in dot
    for rel in projection.relationships:
        assert (
            '{} -> {} [label="{}"];'.format(
                rel["source_diagram_id"],
                rel["target_diagram_id"],
                rel["label"],
            )
            in dot
        )
    assert dot.count('label="TRIGGERS"') == 7
    assert "ghost" not in dot
    assert "artifact:file" not in dot
    assert "Data Onboarding Agent" in dot
    assert DOA_ID not in dot


def test_deterministic_mermaid(adop_document):
    """Test 9 — repeated Mermaid export is byte-identical."""
    graph = build_architecture_graph(adop_document)
    first = export_mermaid(build_diagram_projection(graph))
    second = export_mermaid(build_diagram_projection(build_architecture_graph(adop_document)))
    assert first == second


def test_deterministic_dot(adop_document):
    """Test 10 — repeated DOT export is byte-identical."""
    first = export_graphviz(
        build_diagram_projection(build_architecture_graph(adop_document))
    )
    second = export_graphviz(
        build_diagram_projection(build_architecture_graph(adop_document))
    )
    assert first == second


def test_mutation_safety(adop_document):
    """Test 11 — mutating NetworkX leaves ArchitectureModel unchanged."""
    before = copy.deepcopy(adop_document["architecture"])
    graph = build_architecture_graph(adop_document)
    projection = build_diagram_projection(graph)
    graph.add_edge(
        DOA_ID,
        DOA_ID,
        id="rel:TRIGGERS:injected:injected",
        kind="INJECTED",
    )
    graph.remove_node(list(graph.nodes)[0])
    projection.nodes[0]["label"] = "HACKED"
    projection.relationships.append({"architecture_relationship_id": "x"})
    assert adop_document["architecture"] == before


# ---------------------------------------------------------------------------
# Artifact / CLI pipeline
# ---------------------------------------------------------------------------


def test_create_diagram_artifacts(tmp_path, adop_document):
    projection, paths = create_diagram_artifacts(
        architecture_model_path=ADOP_ARCH_MODEL,
        output_root=tmp_path,
        snapshot_id=PRIMARY_SNAPSHOT_ID,
        render_svg_output=False,
    )
    assert paths["mermaid"].is_file()
    assert paths["graphviz"].is_file()
    assert paths["validation"].is_file()
    assert paths["mermaid"].name == "architecture.mmd"
    assert paths["graphviz"].name == "architecture.dot"
    assert paths["validation"].name == "validation.json"
    assert paths["mermaid"].parent.name == "diagram"
    assert paths["graphviz"].parent.name == "diagram"

    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["problems"] == []
    assert validation["counts"]["architecture_nodes"] == 12
    assert validation["counts"]["diagram_nodes"] == 12
    assert validation["counts"]["architecture_relationships"] == 7
    assert validation["counts"]["diagram_relationships"] == 7
    assert validation["counts"]["architecture_unresolved"] == 0
    assert validation["commit"] == ADOP_COMMIT

    mermaid = paths["mermaid"].read_text(encoding="utf-8")
    dot = paths["graphviz"].read_text(encoding="utf-8")
    assert mermaid.count("-->|TRIGGERS|") == 7
    assert dot.count('label="TRIGGERS"') == 7
    assert "Data Onboarding Agent" in mermaid
    assert "Data Onboarding Agent" in dot

    # re-export byte-stable
    _, paths2 = create_diagram_artifacts(
        architecture_model_path=ADOP_ARCH_MODEL,
        output_root=tmp_path,
        snapshot_id=PRIMARY_SNAPSHOT_ID,
        render_svg_output=False,
    )
    assert paths["mermaid"].read_bytes() == paths2["mermaid"].read_bytes()
    assert paths["graphviz"].read_bytes() == paths2["graphviz"].read_bytes()
    assert len(projection.nodes) == 12
    assert len(projection.relationships) == 7


def test_adop_integration_doa_graph(adop_document):
    """Integration: real ADOP artifact → DOA + 7 TRIGGERS in graph and exports."""
    validate_knowledge_model_or_raise(
        adop_document, expected_commit_sha=ADOP_COMMIT, require_architecture=True
    )
    graph = build_architecture_graph(adop_document)
    assert DOA_ID in graph
    outbound = list(graph.out_edges(DOA_ID, data=True))
    assert len(outbound) == 7
    assert {data["kind"] for _, _, data in outbound} == {"TRIGGERS"}
    targets = {target for _, target, _ in outbound}
    assert targets == {
        "concept:agent:dag-agent",
        "concept:agent:iac-agent",
        "concept:agent:metadata-agent",
        "concept:agent:ontology-agent",
        "concept:agent:ontology-staging-agent",
        "concept:agent:quality-agent",
        "concept:agent:transformation-agent",
    }

    projection = build_diagram_projection(graph)
    doa_proj = next(
        n
        for n in projection.nodes
        if n["architecture_node_id"] == DOA_ID
    )
    assert doa_proj["label"] == "Data Onboarding Agent"
    mermaid = export_mermaid(projection)
    assert mermaid.count("-->|TRIGGERS|") == 7


@pytest.mark.skipif(
    not shutil.which("dot"), reason="Graphviz dot not installed"
)
def test_optional_svg_rendering(tmp_path, adop_document):
    """Optional: DOT → SVG when Graphviz is available (not a hard dependency)."""
    projection, paths = create_diagram_artifacts(
        architecture_model_path=ADOP_ARCH_MODEL,
        output_root=tmp_path,
        snapshot_id=PRIMARY_SNAPSHOT_ID,
        render_svg_output=True,
    )
    assert graphviz_available()
    assert "svg" in paths
    assert paths["svg"].is_file()
    assert paths["svg"].suffix == ".svg"
    assert paths["svg"].read_text(encoding="utf-8", errors="ignore").lstrip().startswith(
        "<?xml"
    ) or "<svg" in paths["svg"].read_text(encoding="utf-8", errors="ignore")


def test_graphviz_absence_is_soft(tmp_path, adop_document):
    """SVG is optional — core export works without Graphviz."""
    # force no-svg path regardless of environment
    _, paths = create_diagram_artifacts(
        architecture_model_path=ADOP_ARCH_MODEL,
        output_root=tmp_path,
        snapshot_id=PRIMARY_SNAPSHOT_ID,
        render_svg_output=False,
    )
    assert paths["mermaid"].is_file()
    assert paths["graphviz"].is_file()
    assert "svg" not in paths
