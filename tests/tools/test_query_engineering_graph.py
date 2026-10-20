"""M11.3/M11.4 — Tool 4: query_engineering_graph (read-only).

Covers the generic ToolResult protocol, argument contract, both query
modes (summary + traversal), provenance, deterministic results,
structured failures (query codes pass through), artifact immutability,
the standalone CLI, run independence, and repository-agnosticism of the
tool source — plus M11.4 hardening: the agent-facing tool reads the
persisted engineering_graph.json and only that; a missing graph
artifact fails with ENGINEERING_GRAPH_NOT_FOUND and never falls back
to engineering_model.json; a corrupted graph artifact fails loudly
without being repaired. Integration tests against the pinned artifact
skip when it is absent.

Offline — no network, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo_intelligence.artifacts import PRIMARY_SNAPSHOT_ID
from repo_intelligence.graph import (
    build_engineering_graph_document,
    write_engineering_graph_document,
)
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.query_engineering_graph import (
    TOOL_NAME,
    main,
    query_engineering_graph_tool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_SRC_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "tools"
)
TOOL_SOURCE = TOOLS_SRC_DIR / "query_engineering_graph.py"
ADOP_ENG_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_model.json"
)
ADOP_GRAPH = (
    PROJECT_ROOT
    / "artifacts"
    / PRIMARY_SNAPSHOT_ID
    / "engineering"
    / "engineering_graph.json"
)

# Repository/run-specific literals that must never appear in tool source.
FORBIDDEN_TOOL_SOURCE_TERMS = (
    "main_2026-09-22",
    "ef11dac",
    "PRIMARY_SNAPSHOT_ID",
    "ADOP",
    "adop",
    "sample-Agentic-Ai-Data-Operations",
    "data-onboarding",
    "aws-samples",
    "airflow",
    "glue",
    "mcp",
    ".claude",
    "dag-agent",
)

COMMIT = "a" * 40

A = "artifact:directory:pkg"
B = "artifact:file:pkg/mod.py"
C = "code_entity:pkg/mod.py"


def _document(commit_sha=COMMIT):
    """Synthetic A --CONTAINS--> B --IMPORTS--> C model (+1 unresolved)."""
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {
            "name": "synthetic",
            "url": "",
            "commit_sha": commit_sha,
        },
        "engineering": {
            "nodes": [
                {
                    "id": A,
                    "kind": "artifact",
                    "subkind": "directory",
                    "classification": "observed",
                    "evidence": [
                        {"path": "pkg", "artifact_type": "directory", "commit": commit_sha}
                    ],
                    "attributes": {},
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
                            "commit": commit_sha,
                        }
                    ],
                    "attributes": {},
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
                            "commit": commit_sha,
                        }
                    ],
                    "attributes": {},
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
                        {"path": "pkg", "artifact_type": "directory", "commit": commit_sha}
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
                            "commit": commit_sha,
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
                            "commit": commit_sha,
                        }
                    ],
                }
            ],
            "metadata": {},
        },
    }


def _model_path(root: Path, snapshot_id: str) -> Path:
    return root / snapshot_id / "engineering" / "engineering_model.json"


def _graph_path(root: Path, snapshot_id: str) -> Path:
    return root / snapshot_id / "engineering" / "engineering_graph.json"


def _write_graph_artifact(
    root: Path, snapshot_id: str, document=None
) -> Path:
    """Persist engineering_graph.json beside the run's model (M11.1)."""
    graph_document = build_engineering_graph_document(
        document if document is not None else _document(),
        snapshot_id=snapshot_id,
    )
    destination = _graph_path(root, snapshot_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_engineering_graph_document(graph_document, destination)
    return destination


def _write_run(
    root: Path,
    snapshot_id: str,
    document=None,
    write_engineering=True,
    raw_text=None,
    write_graph=True,
):
    """Create the artifact layout ``discover_snapshot_id`` requires
    (facts + architecture) plus the engineering model and — by default
    — its derived graph artifact, which is the only artifact the
    agent-facing run-context path reads."""
    run_dir = root / snapshot_id
    (run_dir / "facts").mkdir(parents=True, exist_ok=True)
    (run_dir / "architecture").mkdir(parents=True, exist_ok=True)
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    if write_engineering:
        (run_dir / "engineering").mkdir(parents=True, exist_ok=True)
        payload = (
            raw_text
            if raw_text is not None
            else json.dumps(_document() if document is None else document)
        )
        _model_path(root, snapshot_id).write_text(payload, encoding="utf-8")
        if write_graph:
            _write_graph_artifact(root, snapshot_id, document=document)
    return run_dir


# ---------------------------------------------------------------------------
# Successful queries (summary + traversal)
# ---------------------------------------------------------------------------


def test_traversal_success_returns_standard_tool_result(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool(
        {"start_node_id": A, "direction": "out", "max_depth": 2},
        output_root=root,
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.tool == TOOL_NAME == "query_engineering_graph"
    assert result.error is None
    assert result.source is not None
    data = result.data or {}
    assert data["mode"] == "traversal"
    assert [n["id"] for n in data["nodes"]] == [A, B, C]
    assert [r["kind"] for r in data["relationships"]] == ["CONTAINS", "IMPORTS"]
    assert data["start_node"]["id"] == A
    assert data["query"]["direction"] == "out"
    assert data["truncated"] is False


def test_empty_arguments_run_the_node_kind_summary(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool({}, output_root=root)

    assert result.success is True
    data = result.data or {}
    assert data["mode"] == "summary"
    assert data["start_node"] is None
    assert data["counts"] == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
        "kinds": 3,
    }
    kinds = [(k["kind"], k["subkind"]) for k in data["kinds"]]
    assert kinds == [
        ("artifact", "directory"),
        ("artifact", "file"),
        ("code_entity", "module"),
    ]


def test_none_arguments_accepted_as_summary(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool(None, output_root=root)

    assert result.success is True
    assert (result.data or {}).get("mode") == "summary"


def test_filters_and_depth_pass_through_to_the_query(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    depth_zero = query_engineering_graph_tool(
        {"start_node_id": A, "max_depth": 0}, output_root=root
    )
    assert depth_zero.success is True
    assert [n["id"] for n in depth_zero.data["nodes"]] == [A]
    assert depth_zero.data["relationships"] == []

    filtered = query_engineering_graph_tool(
        {
            "start_node_id": B,
            "direction": "both",
            "relationship_kinds": ["IMPORTS"],
            "node_kinds": ["code_entity/module"],
            "max_depth": 2,
        },
        output_root=root,
    )
    assert filtered.success is True
    assert [r["kind"] for r in filtered.data["relationships"]] == ["IMPORTS"]
    assert [n["id"] for n in filtered.data["nodes"]] == [B, C]


# ---------------------------------------------------------------------------
# Provenance (source)
# ---------------------------------------------------------------------------


def test_source_identifies_run_specific_artifact(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root
    )

    source = result.source or {}
    assert source["artifact"].endswith(
        "run_alpha/engineering/engineering_graph.json"
    )
    assert Path(source["artifact"]).is_file()
    assert source["artifact"] != str(_model_path(root, "run_alpha"))
    assert source["derived_from"] == "engineering_graph.json"
    assert source["schema_version"] == "engineering_graph.v1"
    assert source["model_schema_version"] == "repository_knowledge_model.v1"
    assert source["commit_sha"] == COMMIT
    assert source["snapshot_id"] == "run_alpha"
    assert source["counts"] == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
    }


# ---------------------------------------------------------------------------
# M11.4 — persisted graph artifact only (no engineering_model fallback)
# ---------------------------------------------------------------------------


def test_explicit_snapshot_id_selects_pinned_graph_when_runs_are_ambiguous(
    tmp_path,
):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")
    _write_run(root, "run_beta")

    # Automatic discovery would reject two usable runs. Application code
    # can bind the investigator to the exact snapshot it already verified.
    result = query_engineering_graph_tool(
        {"start_node_id": A},
        output_root=root,
        snapshot_id="run_alpha",
    )

    assert result.success is True
    assert result.source is not None
    assert result.source["snapshot_id"] == "run_alpha"
    assert result.source["artifact"] == str(_graph_path(root, "run_alpha"))


def test_run_context_reads_the_persisted_graph_artifact(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")
    destination = _graph_path(root, "run_alpha")
    before = destination.read_bytes()

    result = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root
    )

    assert result.success is True
    source = result.source or {}
    assert source["artifact"] == str(destination)
    assert source["derived_from"] == "engineering_graph.json"
    assert source["schema_version"] == "engineering_graph.v1"
    assert source["model_schema_version"] == "repository_knowledge_model.v1"
    assert source["commit_sha"] == COMMIT
    assert source["snapshot_id"] == "run_alpha"
    assert source["counts"] == {
        "nodes": 3,
        "relationships": 2,
        "unresolved": 1,
    }
    assert destination.read_bytes() == before  # read-only


def test_graph_artifact_and_explicit_model_path_return_identical_query_data(
    tmp_path,
):
    """The persisted artifact and the sanctioned explicit (testing
    only) model path are equivalent projections by contract."""
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")
    arguments = {"start_node_id": A, "direction": "both", "max_depth": 2}

    via_graph = query_engineering_graph_tool(arguments, output_root=root)
    via_explicit = query_engineering_graph_tool(
        arguments, engineering_model_path_arg=_model_path(root, "run_alpha")
    )

    assert via_graph.success is True and via_explicit.success is True
    assert via_graph.data == via_explicit.data  # equivalent projections
    assert via_graph.source is not None and via_explicit.source is not None
    assert via_graph.source["derived_from"] == "engineering_graph.json"
    assert via_explicit.source["derived_from"] == "engineering_model.json"
    assert via_graph.source["counts"] == via_explicit.source["counts"]


def test_missing_graph_artifact_fails_even_though_the_model_is_present(
    tmp_path,
):
    """The focused no-fallback contract: the model exists, the graph
    artifact does not → ENGINEERING_GRAPH_NOT_FOUND, no answer, and
    neither artifact is written or repaired."""
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha", write_graph=False)
    model_before = _model_path(root, "run_alpha").read_bytes()

    result = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root
    )

    assert result.success is False
    assert result.data is None and result.source is None
    assert result.error is not None
    assert result.error.code == "ENGINEERING_GRAPH_NOT_FOUND"
    assert "engineering_graph.json" in result.error.message
    # the model was never projected into an answer, and nothing was
    # generated in place
    assert not _graph_path(root, "run_alpha").exists()
    assert _model_path(root, "run_alpha").read_bytes() == model_before


def test_missing_graph_artifact_fails_without_even_reading_the_model(
    tmp_path,
):
    """A syntactically broken model still yields
    ENGINEERING_GRAPH_NOT_FOUND (not MODEL_INVALID): the run-context
    path never parses engineering_model.json at all."""
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha", raw_text="{ not json", write_graph=False)

    result = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "ENGINEERING_GRAPH_NOT_FOUND"
    assert not _graph_path(root, "run_alpha").exists()


def test_corrupted_graph_artifact_is_model_invalid_and_not_repaired(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")  # a valid model is present…
    destination = _graph_path(root, "run_alpha")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("{ not json", encoding="utf-8")

    result = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root
    )

    # …but the corrupted graph artifact fails loudly: no silent
    # fallback, no in-place repair.
    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_INVALID"
    assert destination.read_text(encoding="utf-8") == "{ not json"
    assert _model_path(root, "run_alpha").is_file()


def test_model_document_in_the_graph_artifact_slot_is_model_invalid(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")
    destination = _graph_path(root, "run_alpha")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_document()), encoding="utf-8")

    result = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_INVALID"
    assert "engineering_graph.v1" in result.error.message


def test_snapshot_id_derived_for_explicit_model_path(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_beta")

    result = query_engineering_graph_tool(
        {"start_node_id": A},
        engineering_model_path_arg=_model_path(root, "run_beta"),
    )

    assert result.success is True
    assert (result.source or {}).get("snapshot_id") == "run_beta"


def test_snapshot_id_null_when_path_not_in_run_layout(tmp_path):
    loose = tmp_path / "loose_model.json"
    loose.write_text(json.dumps(_document()), encoding="utf-8")

    result = query_engineering_graph_tool(
        {"start_node_id": A}, engineering_model_path_arg=loose
    )

    assert result.success is True
    assert (result.source or {}).get("snapshot_id") is None


def test_run_independence(tmp_path):
    root_one = tmp_path / "root_one"
    root_two = tmp_path / "root_two"
    _write_run(root_one, "first_run")
    _write_run(root_two, "second_run", document=_document(commit_sha="c" * 40))

    result_one = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root_one
    )
    result_two = query_engineering_graph_tool(
        {"start_node_id": A}, output_root=root_two
    )

    assert result_one.success and result_two.success
    assert result_one.source["snapshot_id"] == "first_run"
    assert result_two.source["snapshot_id"] == "second_run"
    assert result_one.source["commit_sha"] == COMMIT
    assert result_two.source["commit_sha"] == "c" * 40


# ---------------------------------------------------------------------------
# Determinism + immutability
# ---------------------------------------------------------------------------


def test_repeated_queries_return_identical_results(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")
    arguments = {"start_node_id": B, "direction": "both", "max_depth": 2}

    first = query_engineering_graph_tool(arguments, output_root=root)
    second = query_engineering_graph_tool(arguments, output_root=root)

    assert first.success and second.success
    assert first.data == second.data
    assert first.source == second.source


def test_tool_never_mutates_the_engineering_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    run_dir = _write_run(root, "run_alpha")
    model_artifact = _model_path(root, "run_alpha")
    graph_artifact = _graph_path(root, "run_alpha")
    model_before = model_artifact.read_bytes()
    graph_before = graph_artifact.read_bytes()
    before_listing = sorted(
        str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
    )

    traversal = query_engineering_graph_tool(
        {"start_node_id": B, "direction": "both", "max_depth": 2},
        output_root=root,
    )
    summary = query_engineering_graph_tool({}, output_root=root)

    assert traversal.success and summary.success
    assert model_artifact.read_bytes() == model_before
    assert graph_artifact.read_bytes() == graph_before
    assert (
        sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*"))
        == before_listing
    )


# ---------------------------------------------------------------------------
# Structured failures
# ---------------------------------------------------------------------------


def test_non_mapping_arguments_rejected(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool(["not", "a", "mapping"], output_root=root)

    assert result.success is False
    assert result.tool == TOOL_NAME
    assert result.data is None and result.source is None
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENT"


def test_unknown_argument_rejected(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool(
        {"start_node_id": A, "bogus": 1}, output_root=root
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENT"
    assert "bogus" in result.error.message


def test_invalid_query_values_use_query_error_codes(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    bad_direction = query_engineering_graph_tool(
        {"start_node_id": A, "direction": "sideways"}, output_root=root
    )
    assert bad_direction.success is False
    assert bad_direction.error is not None
    assert bad_direction.error.code == "INVALID_ARGUMENT"

    unknown_node = query_engineering_graph_tool(
        {"start_node_id": "artifact:file:does-not-exist"}, output_root=root
    )
    assert unknown_node.success is False
    assert unknown_node.error is not None
    assert unknown_node.error.code == "NODE_NOT_FOUND"

    summary_misuse = query_engineering_graph_tool({"direction": "in"}, output_root=root)
    assert summary_misuse.success is False
    assert summary_misuse.error is not None
    assert summary_misuse.error.code == "INVALID_ARGUMENT"


def test_missing_engineering_artifact(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha", write_engineering=False)

    result = query_engineering_graph_tool({"start_node_id": A}, output_root=root)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "ENGINEERING_GRAPH_NOT_FOUND"
    assert "engineering_graph.json" in result.error.message
    # the tool never creates/regenerates any artifact
    assert not _graph_path(root, "run_alpha").exists()
    assert not _model_path(root, "run_alpha").exists()


def test_missing_explicit_model_path(tmp_path):
    result = query_engineering_graph_tool(
        {"start_node_id": A},
        engineering_model_path_arg=tmp_path / "nope.json",
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_NOT_FOUND"


def test_unresolvable_run_context(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    result = query_engineering_graph_tool({}, output_root=empty_root)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_NOT_FOUND"


def test_multiple_runs_fail_closed(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_one")
    _write_run(root, "run_two")

    result = query_engineering_graph_tool({}, output_root=root)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_NOT_FOUND"
    assert "Multiple" in result.error.message


def test_invalid_json_model_document_is_model_invalid(tmp_path):
    """MODEL_INVALID stays reachable through the sanctioned explicit
    (testing/standalone) model path — the only path that reads a model
    document at all."""
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha", raw_text="{ not json", write_graph=False)

    result = query_engineering_graph_tool(
        {"start_node_id": A},
        engineering_model_path_arg=_model_path(root, "run_alpha"),
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_INVALID"


def test_wrong_schema_version_is_model_invalid(tmp_path):
    root = tmp_path / "artifacts"
    document = _document()
    document["schema_version"] = "something_else.v9"
    _write_run(root, "run_alpha", document=document, write_graph=False)

    result = query_engineering_graph_tool(
        {"start_node_id": A},
        engineering_model_path_arg=_model_path(root, "run_alpha"),
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_INVALID"


def test_broken_model_structure_is_model_invalid(tmp_path):
    root = tmp_path / "artifacts"
    document = _document()
    document["engineering"]["nodes"] = [
        {"kind": "artifact", "subkind": "file"}  # missing id
    ]
    document["engineering"]["relationships"] = []
    _write_run(root, "run_alpha", document=document, write_graph=False)

    result = query_engineering_graph_tool(
        {}, engineering_model_path_arg=_model_path(root, "run_alpha")
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "MODEL_INVALID"


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def test_main_prints_json_and_exits_zero(tmp_path, capsys):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    exit_code = main(
        [
            "--output-root",
            str(root),
            "--start-node-id",
            A,
            "--direction",
            "out",
            "--max-depth",
            "2",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["tool"] == "query_engineering_graph"
    assert [n["id"] for n in payload["data"]["nodes"]] == [A, B, C]


def test_main_summary_mode(tmp_path, capsys):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    exit_code = main(["--output-root", str(root)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["data"]["mode"] == "summary"


def test_main_failure_exits_six(tmp_path, capsys):
    exit_code = main(
        ["--engineering-model-path", str(tmp_path / "missing.json")]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 6
    assert payload["success"] is False
    assert payload["error"]["code"] == "MODEL_NOT_FOUND"


# ---------------------------------------------------------------------------
# Repository-agnosticism (spec §23)
# ---------------------------------------------------------------------------


def test_tool_source_has_no_hardcoded_run_or_domain_terms():
    text = TOOL_SOURCE.read_text(encoding="utf-8")
    for term in FORBIDDEN_TOOL_SOURCE_TERMS:
        assert term not in text, "hard-coded term {!r} in tool source".format(term)


def test_query_results_carry_no_repository_specific_ids(tmp_path):
    root = tmp_path / "artifacts"
    _write_run(root, "run_alpha")

    result = query_engineering_graph_tool(
        {"start_node_id": B, "direction": "both", "max_depth": 2},
        output_root=root,
    )

    serialized = json.dumps(result.to_dict())
    for term in FORBIDDEN_TOOL_SOURCE_TERMS:
        assert term not in serialized


# ---------------------------------------------------------------------------
# Integration: pinned artifact (production run layout)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adop_output_root():
    if not ADOP_GRAPH.is_file():
        pytest.skip("pinned engineering graph artifact absent")
    return PROJECT_ROOT / "artifacts"


def test_pinned_artifact_summary(adop_output_root):
    assert ADOP_GRAPH.is_file() and ADOP_ENG_MODEL.is_file()
    result = query_engineering_graph_tool({}, output_root=adop_output_root)

    assert result.success is True
    counts = (result.data or {}).get("counts", {})
    assert counts["nodes"] == 1113
    assert counts["relationships"] == 1627
    assert counts["unresolved"] == 449
    assert (result.source or {}).get("snapshot_id") == PRIMARY_SNAPSHOT_ID
    # even with the canonical model present on disk, the agent-facing
    # tool reads the persisted graph artifact
    assert (result.source or {}).get("derived_from") == "engineering_graph.json"
    assert (result.source or {}).get("artifact") == str(ADOP_GRAPH)


def test_pinned_artifact_agent_triggers_walk(adop_output_root):
    dag = "artifact:agent_definition:.claude/agents/dag-agent.md"
    result = query_engineering_graph_tool(
        {"start_node_id": dag, "direction": "in"},
        output_root=adop_output_root,
    )

    assert result.success is True
    relationships = (result.data or {}).get("relationships", [])
    assert relationships
    for relationship in relationships:
        assert relationship["target"] == dag
        assert relationship["kind"] == "TRIGGERS"


def test_pinned_artifact_deterministic(adop_output_root):
    arguments = {"start_node_id": "artifact:directory:.claude/agents", "direction": "out"}
    first = query_engineering_graph_tool(arguments, output_root=adop_output_root)
    second = query_engineering_graph_tool(arguments, output_root=adop_output_root)

    assert first.success and second.success
    assert first.data == second.data
