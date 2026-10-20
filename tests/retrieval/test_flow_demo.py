"""Spec §34/§35/§37/§39/§44 — the mandatory flow diagnostic.

Drives ``agentrun_repository_retrieval_flow_demo.py`` end to end on
synthetic fixture repositories (offline, hashing embeddings, no M17,
no OpenAI):

* PASS run: every required §35 section, the exact summary lines, the
  line-numbered read output, and ``FLOW COMPLETE``;
* ``flow_result.json`` carries the §37 contract and every validation
  flag is true;
* exit codes: 0 PASS, 1 FAIL (structured, no traceback), 6 config
  errors (bad top-k, conflicting modes, live without key, missing
  run, ``--no-build`` without index);
* the explicit index-build step (never a query side effect);
* §44 dual fixtures: two repositories, two indexes, isolated results.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from conftest import enable_search, install_pinned_run
from repo_intelligence.graph import (
    build_engineering_graph_document,
    write_engineering_graph_document,
)

DEMO_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_repository_retrieval_flow_demo.py"
)

DEFAULT_QUERY = "How do the agents, tools, and MCP servers interact?"
HASHING = "hashing://v1/128"

SPEC44_FILES = {
    "agents/agent.py": (
        'TOOLS = ("search_notes", "fetch_note")\n'
        "\n"
        "def use_tool(name):\n"
        "    # the agent uses the tool through this entry point\n"
        "    return TOOLS\n"
    ),
    "tools/tool.py": (
        "def search_notes(query):\n"
        "    # tool implementation the agent calls\n"
        "    return query\n"
    ),
    "mcp/server.py": (
        "SERVERS = {}\n"
        "# mcp server registration surface\n"
    ),
    "docs/overview.md": "# Overview\n\nThe agent uses the tool.\n",
    "unrelated/notes.txt": "gardening calendar placeholder\n",
    "pkg/mod.py": "def main():\n    return 42\n",
}


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "agentrun_repository_retrieval_flow_demo", str(DEMO_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


def _engineering_model(name: str, commit: str) -> Dict:
    return {
        "schema_version": "repository_knowledge_model.v1",
        "repository": {
            "name": name,
            "url": "https://example.invalid/fixture.git",
            "commit_sha": commit,
        },
        "engineering": {
            "nodes": [
                {
                    "id": "artifact:directory:pkg",
                    "kind": "artifact",
                    "subkind": "directory",
                    "classification": "observed",
                    "evidence": [],
                    "attributes": {},
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
                            "commit": commit,
                        }
                    ],
                    "attributes": {},
                },
            ],
            "relationships": [
                {
                    "id": (
                        "rel:CONTAINS:artifact:directory:pkg"
                        ":artifact:file:pkg/mod.py"
                    ),
                    "kind": "CONTAINS",
                    "source": "artifact:directory:pkg",
                    "target": "artifact:file:pkg/mod.py",
                    "classification": "derived",
                    "evidence": [
                        {
                            "path": "pkg/mod.py",
                            "artifact_type": "python",
                            "commit": commit,
                        }
                    ],
                }
            ],
            "unresolved": [],
            "metadata": {},
        },
    }


def _flow_run(
    tmp_path: Path,
    monkeypatch,
    files: Optional[Dict[str, str]] = None,
    run_id: str = "run_fixture_alpha",
    name: str = "fixture-repository",
    commit: str = "1" * 40,
    graph: bool = True,
    index: bool = True,
    store: Optional[Path] = None,
    artifacts: str = "artifacts",
) -> Tuple[Path, Path]:
    output_root = tmp_path / artifacts
    checkout = install_pinned_run(
        output_root,
        dict(SPEC44_FILES if files is None else files),
        run_id=run_id,
        repository_name=name,
        commit_sha=commit,
    )
    if graph:
        run_dir = output_root / run_id
        engineering = run_dir / "engineering"
        engineering.mkdir(parents=True, exist_ok=True)
        model = _engineering_model(name, commit)
        (engineering / "engineering_model.json").write_text(
            json.dumps(model), encoding="utf-8"
        )
        write_engineering_graph_document(
            build_engineering_graph_document(
                model, snapshot_id=run_id
            ),
            engineering / "engineering_graph.json",
        )
    if index:
        enable_search(
            checkout,
            monkeypatch,
            store if store is not None else tmp_path,
        )
    elif store is not None:
        monkeypatch.setenv(
            "REPO_INTELLIGENCE_DB_STORE", str(store)
        )
    return checkout, output_root


def _run(args: List[str], capsys) -> Tuple[int, str]:
    code = demo.main(args)
    return code, capsys.readouterr().out


def _base_args(output_root: Path, tmp_path: Path) -> List[str]:
    return [
        "--output-root",
        str(output_root),
        "--output",
        str(tmp_path / "flow_out"),
    ]


def _result_json(tmp_path: Path) -> Dict:
    path = tmp_path / "flow_out" / "flow_result.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# §35 output contract (offline PASS)
# ---------------------------------------------------------------------------


def test_flow_passes_offline_and_prints_every_section(
    tmp_path, monkeypatch, capsys
):
    checkout, output_root = _flow_run(tmp_path, monkeypatch)
    code, out = _run(_base_args(output_root, tmp_path), capsys)
    assert code == 0

    banner = "=" * 60
    # every §35 section, preceded by its banner (numbered prefixes
    # are allowed: "1. REPOSITORY RETRIEVAL")
    for title in (
        "QUERY",
        "REPOSITORY RETRIEVAL",
        "ENGINEERING GRAPH",
        "COMBINED EVIDENCE",
        "AGENT EVIDENCE SELECTION",
        "READ_FILE VERIFICATION",
        "VERIFIED EVIDENCE",
        "SUMMARY",
        "FLOW COMPLETE",
    ):
        pattern = re.compile(
            r"{}\n(?:\d+\.\s)?{}\n{}".format(
                re.escape(banner), re.escape(title), re.escape(banner)
            )
        )
        assert pattern.search(out), title

    # §39: honest offline mode
    assert "Reasoning mode:\nOFFLINE / deterministic" in out
    assert DEFAULT_QUERY in out

    # §35 summary — exact machine-readable lines
    for line in (
        "Repository retrieval: PASS",
        "Engineering graph: PASS",
        "Evidence selection: PASS",
        "Read-file verification: PASS",
        "Provenance: PASS",
        "Overall: PASS",
    ):
        assert line in out, line

    # §7 line-numbered read output
    assert re.search(r"^\d+ \| ", out, re.MULTILINE)
    assert "Verification:\nSUCCESS" in out
    # bounded, honest provenance on the retrieval stage
    assert "fixture-repository" in out
    assert "1" * 40 in out
    # RAG actually surfaced implementation files
    assert "agents/agent.py" in out or "tools/tool.py" in out
    # graph relationship actually printed
    assert "CONTAINS" in out


def test_flow_result_json_carries_the_spec_contract(
    tmp_path, monkeypatch, capsys
):
    checkout, output_root = _flow_run(tmp_path, monkeypatch)
    code, out = _run(_base_args(output_root, tmp_path), capsys)
    assert code == 0
    payload = _result_json(tmp_path)

    # §37 keys
    assert payload["query"] == DEFAULT_QUERY
    assert set(payload["repository"]) == {
        "name",
        "commit_sha",
        "snapshot_id",
    }
    assert payload["repository"]["name"] == "fixture-repository"
    assert payload["repository"]["commit_sha"] == "1" * 40
    assert payload["repository"]["snapshot_id"] == (
        "run_fixture_alpha"
    )
    assert payload["retrieval"]["count"] == len(
        payload["retrieval"]["results"]
    )
    assert payload["retrieval"]["count"] >= 1
    assert isinstance(payload["graph"]["relationships"], list)
    assert payload["graph"]["relationships"]
    assert payload["selected_evidence"]
    assert payload["verified_evidence"]
    # §37 required validation keys (the implementation keeps an
    # extra `selection` check beside them)
    assert {
        "retrieval",
        "graph",
        "verification",
        "provenance",
    } <= set(payload["validation"])
    assert all(payload["validation"].values())
    assert payload["verdict"] == "PASS"
    assert payload["mode"] == "OFFLINE / deterministic"

    # every selected candidate was read back from the pinned checkout
    for entry in payload["verified_evidence"]:
        assert entry["verified"] is True
        assert (checkout / entry["path"]).is_file()
    for entry in payload["selected_evidence"]:
        assert (checkout / entry["path"]).is_file()


def test_flow_writes_json_and_summary_beside_the_output_arg(
    tmp_path, monkeypatch, capsys
):
    _, output_root = _flow_run(tmp_path, monkeypatch)
    code, out = _run(_base_args(output_root, tmp_path), capsys)
    assert code == 0
    json_path = tmp_path / "flow_out" / "flow_result.json"
    md_path = tmp_path / "flow_out" / "flow_summary.md"
    assert json_path.is_file() and md_path.is_file()
    # §35/§9: analysis artifact, never under runtime/
    for saved in (str(json_path), str(md_path)):
        assert "runtime" not in saved
    summary = md_path.read_text(encoding="utf-8")
    assert summary.startswith("# Repository retrieval flow — PASS")
    assert "- query: {}".format(DEFAULT_QUERY) in summary
    assert "- retrieval: PASS" in summary
    assert "- verification: PASS" in summary
    assert str(json_path) in out
    assert str(md_path) in out


def test_flow_honors_query_and_top_k_overrides(
    tmp_path, monkeypatch, capsys
):
    _, output_root = _flow_run(tmp_path, monkeypatch)
    args = _base_args(output_root, tmp_path) + [
        "--query",
        "agent uses the tool",
        "--top-k",
        "3",
    ]
    code, out = _run(args, capsys)
    assert code == 0
    assert "agent uses the tool" in out
    payload = _result_json(tmp_path)
    assert payload["query"] == "agent uses the tool"
    assert len(payload["selected_evidence"]) == 3
    assert payload["retrieval"]["count"] == 3


def test_flow_verbose_dumps_stage_intermediates(
    tmp_path, monkeypatch, capsys
):
    _, output_root = _flow_run(tmp_path, monkeypatch)
    args = _base_args(output_root, tmp_path) + ["--verbose"]
    code, out = _run(args, capsys)
    assert code == 0
    assert "graph kinds census" in out
    assert "retrieval data" in out


# ---------------------------------------------------------------------------
# §10 explicit build (never a query side effect)
# ---------------------------------------------------------------------------


def test_flow_builds_a_missing_index_explicitly(
    tmp_path, monkeypatch, capsys
):
    store = tmp_path / "fresh_store"
    _, output_root = _flow_run(
        tmp_path, monkeypatch, index=False, store=store
    )
    args = _base_args(output_root, tmp_path) + [
        "--embedding-model",
        HASHING,
    ]
    code, out = _run(args, capsys)
    assert code == 0
    assert (
        "index build (explicit, never a query side effect):" in out
    )
    payload = _result_json(tmp_path)
    assert payload["retrieval"]["explicit_build"] is not None
    report = payload["retrieval"]["explicit_build"]
    assert report["embedder"] == HASHING
    assert report["chunks_indexed"] >= 1
    assert all(payload["validation"].values())


def test_flow_no_build_fails_actionably(tmp_path, monkeypatch, capsys):
    store = tmp_path / "empty_store"
    store.mkdir()
    _, output_root = _flow_run(
        tmp_path, monkeypatch, index=False, store=store
    )
    args = _base_args(output_root, tmp_path) + [
        "--embedding-model",
        HASHING,
        "--no-build",
    ]
    code, out = _run(args, capsys)
    assert code == 6
    assert "index missing and --no-build" in out
    assert "Traceback" not in out


# ---------------------------------------------------------------------------
# Exit codes (spec §1/§10/§36)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", (0, 99, -1))
def test_flow_rejects_bad_top_k(tmp_path, monkeypatch, capsys, top_k):
    _, output_root = _flow_run(tmp_path, monkeypatch)
    args = _base_args(output_root, tmp_path) + [
        "--top-k",
        str(top_k),
    ]
    code, out = _run(args, capsys)
    assert code == 6
    assert "--top-k must be between 1 and 50" in out


def test_flow_rejects_conflicting_modes(tmp_path, monkeypatch, capsys):
    _, output_root = _flow_run(tmp_path, monkeypatch)
    args = _base_args(output_root, tmp_path) + [
        "--offline",
        "--live-reasoning",
    ]
    code, out = _run(args, capsys)
    assert code == 6
    assert "--offline / --live-reasoning" in out


def test_flow_live_without_key_fails_cleanly(
    tmp_path, monkeypatch, capsys
):
    # neuter BOTH key sources: delenv alone is defeated by a dev
    # checkout whose .env carries a real working key (load_env would
    # re-inject it mid-run and the live run would succeed, exit 0)
    from repo_intelligence.llm import config as llm_config

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_config, "load_env", lambda *a, **k: {})
    _, output_root = _flow_run(tmp_path, monkeypatch)
    args = _base_args(output_root, tmp_path) + [
        "--live-reasoning",
    ]
    code, out = _run(args, capsys)
    assert code == 6
    assert "live reasoning unavailable" in out
    assert "Traceback" not in out


def test_flow_live_key_check_precedes_snapshot_lookup(
    tmp_path, monkeypatch, capsys
):
    """Fail-fast ordering: with no key the flow must report the
    configuration problem (exit 6) WITHOUT touching snapshots — a
    nonexistent --output-root would otherwise mask the real cause
    with 'no run snapshot under ...'."""
    from repo_intelligence.llm import config as llm_config

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_config, "load_env", lambda *a, **k: {})
    empty = tmp_path / "no_runs"
    empty.mkdir()
    args = [
        "--output-root",
        str(empty),
        "--output",
        str(tmp_path / "flow_out"),
        "--live-reasoning",
    ]
    code, out = _run(args, capsys)
    assert code == 6
    assert "live reasoning unavailable" in out
    assert "OPENAI_API_KEY" in out
    assert "no run snapshot" not in out
    assert "Traceback" not in out


def test_flow_missing_run_is_a_config_failure(
    tmp_path, monkeypatch, capsys
):
    empty = tmp_path / "no_runs"
    empty.mkdir()
    code, out = _run(
        ["--output-root", str(empty), "--output",
         str(tmp_path / "flow_out")],
        capsys,
    )
    assert code == 6
    assert "no run snapshot under" in out
    assert "Traceback" not in out


def test_flow_missing_graph_artifact_fails_the_check(
    tmp_path, monkeypatch, capsys
):
    _, output_root = _flow_run(tmp_path, monkeypatch, graph=False)
    code, out = _run(_base_args(output_root, tmp_path), capsys)
    assert code == 1
    assert "graph summary failed" in out
    assert "ENGINEERING_GRAPH_NOT_FOUND" in out
    assert "Overall: FAIL" in out
    assert "Traceback" not in out
    payload = _result_json(tmp_path)
    assert payload["validation"]["graph"] is False
    assert payload["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# §44 — repository-agnostic dual fixtures
# ---------------------------------------------------------------------------


def test_flow_dual_fixtures_stay_isolated(
    tmp_path, monkeypatch, capsys
):
    files_b = dict(SPEC44_FILES)
    files_b["agents/agent.py"] = (
        "OTHER_WIRING = 'gamma marker wiring'\n"
        "def act():\n"
        "    return OTHER_WIRING\n"
    )
    files_b["docs/notes.md"] = "# Other\n\ngamma marker lives here\n"
    checkout_a, root_a = _flow_run(
        tmp_path,
        monkeypatch,
        run_id="run_fixture_a",
        name="fixture-repository",
        commit="1" * 40,
        artifacts="artifacts_a",
    )
    checkout_b, root_b = _flow_run(
        tmp_path,
        monkeypatch,
        files=files_b,
        run_id="run_fixture_b",
        name="other-repository",
        commit="2" * 40,
        artifacts="artifacts_b",
    )

    code_a, out_a = _run(
        _base_args(root_a, tmp_path), capsys
    )
    assert code_a == 0
    payload_a = _result_json(tmp_path)
    (tmp_path / "flow_out").rename(tmp_path / "flow_out_a")

    code_b, out_b = _run(
        _base_args(root_b, tmp_path), capsys
    )
    assert code_b == 0
    payload_b = _result_json(tmp_path)

    assert payload_a["repository"]["name"] == "fixture-repository"
    assert payload_b["repository"]["name"] == "other-repository"
    assert payload_a["repository"]["commit_sha"] == "1" * 40
    assert payload_b["repository"]["commit_sha"] == "2" * 40

    # every retrieval result carries its OWN repository's identity
    for item in payload_a["retrieval"]["results"]:
        assert item["repository_name"] == "fixture-repository"
        assert item["commit_sha"] == "1" * 40
        assert (checkout_a / item["path"]).is_file()
        assert "gamma marker" not in item["excerpt"]
    for item in payload_b["retrieval"]["results"]:
        assert item["repository_name"] == "other-repository"
        assert item["commit_sha"] == "2" * 40
        assert (checkout_b / item["path"]).is_file()
        assert "agent uses the tool through" not in item["excerpt"]

    # no cross-contamination anywhere in the serialized payloads
    text_a = json.dumps(payload_a)
    text_b = json.dumps(payload_b)
    assert "other-repository" not in text_a
    assert "gamma marker" not in text_a
    assert "2" * 40 not in text_a
    assert "fixture-repository" not in text_b
    assert "1" * 40 not in text_b

    # both indexes coexist in one store and stay separate
    from repo_intelligence.retrieval import load_index
    from repo_intelligence.retrieval.models import IndexIdentity

    for name, commit in (
        ("fixture-repository", "1" * 40),
        ("other-repository", "2" * 40),
    ):
        identity = IndexIdentity(
            repository_name=name,
            commit_sha=commit,
            embedding_model=HASHING,
        )
        loaded = load_index(identity)
        assert loaded.ntotal >= 1
    assert load_index(
        IndexIdentity(
            repository_name="fixture-repository",
            commit_sha="1" * 40,
            embedding_model=HASHING,
        )
    ).index_dir != load_index(
        IndexIdentity(
            repository_name="other-repository",
            commit_sha="2" * 40,
            embedding_model=HASHING,
        )
    ).index_dir


# ---------------------------------------------------------------------------
# §38 — independent of M17
# ---------------------------------------------------------------------------


def test_flow_script_never_imports_the_m17_stack():
    source = DEMO_PATH.read_text(encoding="utf-8")
    for token in (
        "multiagent",
        "orchestrat",
        "agent_loop",
        "run_investigator",
    ):
        assert token not in source, token
