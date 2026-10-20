"""Focused tests for Tool 2: search_repository (M18 RAG retrieval).

The M18 rewrite replaces the former lexical walk with repository- and
commit-aware semantic retrieval over a prebuilt index. These tests
keep the established ToolResult protocol, input-contract, path-safety
and provenance suites, and replace the lexical-specific assertions
with retrieval-layer equivalents (spec §33/§36 A–O, §44 dual
fixtures):

    A–C  chunking determinism lives in tests/retrieval/; here the
         tool surfaces those chunks as evidence
    D–F  metadata/commit/repository isolation (dual fixtures)
    G    search_repository() returns RAG evidence (§26 fields)
    H    missing index → structured actionable error
    I    retrieved range → read_file() → exact source
    L    evidence provenance on every result
    N    representative query leaves runtime/ clean
    O    default root resolves the named snapshot layout

Everything runs offline on deterministic ``hashing://v1/128``
embeddings (no model download, no network, no OpenAI) through the
shared ``tests/conftest.py`` helpers ``install_pinned_run`` /
``enable_search``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from conftest import (
    DEFAULT_COMMIT,
    DEFAULT_NAME,
    DEFAULT_RUN_ID,
    DEFAULT_URL,
    enable_search,
    install_pinned_run,
    symlink_or_skip,
    try_symlink,
)
from repo_intelligence.facts import create_facts
from repo_intelligence.retrieval.models import RetrievalResult
from repo_intelligence.snapshot import create_snapshot
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.protocol import (
    MAX_TOOL_RESPONSE_CHARS,
    response_chars,
)
from repo_intelligence.tools.search_repository import (
    DEFAULT_MAX_RESULTS,
    EVIDENCE_NOTE,
    MAX_FILE_BYTES,
    MAX_RESULTS_LIMIT,
    TOOL_NAME,
    _bound_payload,
    main,
    search_repository,
)

import repo_intelligence.tools.search_repository as search_module

TOOLS_SRC_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "repo_intelligence"
    / "tools"
)

RUN_ID = DEFAULT_RUN_ID
REPO_NAME = DEFAULT_NAME
COMMIT_SHA = DEFAULT_COMMIT
REPO_URL = DEFAULT_URL

# Generic synthetic checkout. Content deliberately uses generic
# concepts only (checkout flow) — never repository-specific terms.
CHECKOUT_FILES = {
    "README.md": (
        "# Sample Project\n"
        "\n"
        "This project manages the checkout flow for demo orders.\n"
        "See the guide for details.\n"
    ),
    "docs/guide.md": (
        "# Guide\n"
        "\n"
        "The checkout flow starts here.\n"
        "It validates the cart.\n"
        "Then it reserves stock.\n"
        "Finally it confirms payment.\n"
        "\n"
        "Unrelated closing notes.\n"
    ),
    "docs/api/reference.md": (
        "# API Reference\n"
        "\n"
        "POST /checkout starts a checkout flow run.\n"
        "GET /checkout returns its status.\n"
    ),
    "docs/faq.md": (
        "# FAQ\n"
        "\n"
        "Q: when does checkout start?\n"
        "A: checkout starts when the cart is ready.\n"
        "Q: can checkout fail?\n"
        "A: yes, checkout can fail while reserving stock.\n"
        "Q: is checkout reusable?\n"
        "A: checkout is reusable after a reset.\n"
    ),
    "src/app.py": (
        "def start_checkout():\n"
        "    # checkout flow entry point\n"
        "    return run_checkout()\n"
        "\n"
        "def run_checkout():\n"
        "    return None\n"
    ),
    "src/client.ts": (
        "export function requestCheckout(): void {\n"
        "  // waits for the checkout flow\n"
        "}\n"
    ),
    "data/sample.csv": "id,checkout\n1,done\n",
    "notes.log": "checkout happened\n",
}

# Spec §44 dual fixture: the repository-agnostic acceptance shape.
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
}

# Second fixture repository for isolation (spec §36 F / §44).
OTHER_FILES = {
    "agents/agent.py": (
        "OTHER_WIRING = 'gamma marker wiring'\n"
        "def act():\n"
        "    return OTHER_WIRING\n"
    ),
    "docs/notes.md": "# Other\n\ngamma marker lives here only\n",
}


def _add_noise(checkout: Path) -> None:
    """Binary / oversized / ignored-dir / extension / symlink cases."""
    (checkout / "binary.txt").write_bytes(
        b"checkout \x00\x01 secret-marker\n"
    )
    (checkout / "oversized.txt").write_text(
        "oversizedmarker checkout\n" + "z" * MAX_FILE_BYTES,
        encoding="utf-8",
    )
    (checkout / "node_modules").mkdir(parents=True, exist_ok=True)
    (checkout / "node_modules" / "index.js").write_text(
        "var nodemodmarker = checkout;\n", encoding="utf-8"
    )
    (checkout / ".venv").mkdir(parents=True, exist_ok=True)
    (checkout / ".venv" / "util.py").write_text(
        "venvmarker = checkout\n", encoding="utf-8"
    )
    outside = checkout.parent.parent / "outside.md"
    outside.write_text(
        "outsidemarker checkout flow outside\n", encoding="utf-8"
    )
    # best-effort: a host without symlink privilege still gets the
    # binary/oversized/ignored-dir/extension noise cases
    try_symlink(outside.resolve(), checkout / "linked.md")


def _fixture(
    tmp_path: Path,
    monkeypatch,
    files: Optional[Dict[str, str]] = None,
    run_id: str = RUN_ID,
    repository_name: str = REPO_NAME,
    commit_sha: str = COMMIT_SHA,
    url: str = REPO_URL,
    noise: bool = False,
) -> Tuple[Path, Path]:
    """Pinned run + named checkout + isolated hashing index."""
    output_root = tmp_path / "artifacts"
    checkout = install_pinned_run(
        output_root,
        dict(CHECKOUT_FILES if files is None else files),
        run_id=run_id,
        repository_name=repository_name,
        commit_sha=commit_sha,
        url=url,
    )
    if noise:
        _add_noise(checkout)
    enable_search(checkout, monkeypatch, tmp_path)
    return checkout, output_root


def _search(
    arguments, checkout: Path, output_root: Path
) -> ToolResult:
    return search_repository(
        arguments, repository_root=checkout, output_root=output_root
    )


def _tree_state(root: Path) -> Dict[str, Tuple[bytes, int]]:
    """bytes + mtime_ns of every entry (read-only proof)."""
    state: Dict[str, Tuple[bytes, int]] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir() and not path.is_symlink():
            state[rel] = (b"", path.lstat().st_mtime_ns)
        else:
            data = path.read_bytes() if path.is_file() else b""
            state[rel] = (data, path.lstat().st_mtime_ns)
    return state


def _all_excerpts(data: Dict) -> List[str]:
    return [str(item.get("excerpt", "")) for item in data["results"]]


# ---------------------------------------------------------------------------
# Generic ToolResult protocol (shared by all tools)
# ---------------------------------------------------------------------------


def test_result_shape_is_generic(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    assert isinstance(result, ToolResult)
    payload = result.to_dict()
    assert list(payload) == ["success", "tool", "data", "source", "error"]
    assert payload["tool"] == TOOL_NAME == "search_repository"
    assert payload["success"] is True
    assert payload["error"] is None


# ---------------------------------------------------------------------------
# RAG results (spec §26 / §36 G)
# ---------------------------------------------------------------------------


def test_results_carry_the_evidence_contract_fields(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    assert result.success is True
    data = result.data
    assert data["query"] == "checkout flow"
    assert data["returned"] == DEFAULT_MAX_RESULTS == 5
    assert data["returned"] <= data["total_matches"]
    assert data["retrieval"]["mode"] == "semantic"
    assert data["retrieval"]["embedding_model"].startswith("hashing://")
    for item in data["results"]:
        # §26 result contract + the established ``path`` alias every
        # evidence consumer reads.
        assert sorted(item) == [
            "chunk_id",
            "commit_sha",
            "excerpt",
            "file_path",
            "language",
            "line_end",
            "line_start",
            "path",
            "repository_name",
            "similarity_score",
            "snapshot_id",
        ]
        assert item["path"] == item["file_path"]
        assert isinstance(item["line_start"], int)
        assert isinstance(item["line_end"], int)
        assert item["line_end"] >= item["line_start"]
        assert item["repository_name"] == REPO_NAME
        assert item["commit_sha"] == COMMIT_SHA
        assert item["snapshot_id"] == RUN_ID
        assert isinstance(item["chunk_id"], str) and item["chunk_id"]
        assert isinstance(item["similarity_score"], float)
        assert isinstance(item["excerpt"], str)


def test_results_rank_by_similarity_with_stable_tie_break(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    keys = [
        (
            -item["similarity_score"],
            item["file_path"],
            item["line_start"],
            item["chunk_id"],
        )
        for item in result.data["results"]
    ]
    assert keys == sorted(keys)


def test_repeated_runs_return_identical_results(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    first = _search({"query": "checkout flow"}, checkout, output_root)
    second = _search({"query": "checkout flow"}, checkout, output_root)
    assert first.data == second.data


def test_query_is_case_insensitive_under_hashing(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    base = _search({"query": "checkout flow"}, checkout, output_root)
    upper = _search({"query": "CHECKOUT FLOW"}, checkout, output_root)
    mixed = _search({"query": "Checkout Flow"}, checkout, output_root)
    assert base.success and upper.success and mixed.success
    assert (
        base.data["results"]
        == upper.data["results"]
        == mixed.data["results"]
    )


def test_no_overlap_query_is_bounded_with_zero_scores(
    tmp_path, monkeypatch
):
    """Semantic retrieval returns bounded candidates, never a dump —
    a query with no overlap only ever surfaces generic chunks, all
    of them below the top-ranked content of a matching query."""
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "zzz-no-such-marker"}, checkout, output_root
    )
    assert result.success is True
    assert result.data["returned"] == DEFAULT_MAX_RESULTS
    assert result.error is None
    assert not any(
        "zzz-no-such-marker" in text
        for text in _all_excerpts(result.data)
    )
    matched = _search({"query": "checkout flow"}, checkout, output_root)
    assert (
        result.data["results"][-1]["similarity_score"]
        < matched.data["results"][0]["similarity_score"]
    )


def test_top_results_mention_the_queried_concept(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    assert result.success is True
    hits = [
        item
        for item in result.data["results"]
        if "checkout" in item["excerpt"].lower()
    ]
    assert hits, "at least one returned chunk mentions checkout"
    top = result.data["results"][0]
    assert top["similarity_score"] > 0.0
    for item in result.data["results"]:
        assert (checkout / item["path"]).is_file()


# ---------------------------------------------------------------------------
# Ordering, limits, truncation
# ---------------------------------------------------------------------------


def test_max_results_caps_and_reports_truncation(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout flow", "max_results": 2},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["returned"] == 2
    assert result.data["total_matches"] > 2
    assert result.data["truncated"] is True


def test_max_results_upper_bound_is_allowed(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "max_results": MAX_RESULTS_LIMIT},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["returned"] <= MAX_RESULTS_LIMIT


# ---------------------------------------------------------------------------
# Path filter + path safety
# ---------------------------------------------------------------------------


def test_path_restriction_directory(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "path": "docs"},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["results"]
    for item in result.data["results"]:
        assert item["path"].startswith("docs/")


def test_path_restriction_file(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "path": "src/app.py"},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["results"]
    for item in result.data["results"]:
        assert item["path"] == "src/app.py"


def test_path_filter_reads_explicitly_named_file(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "flow", "path": "docs/guide.md"},
        checkout,
        output_root,
    )
    assert result.success is True
    guide_lines = CHECKOUT_FILES["docs/guide.md"].splitlines()
    for item in result.data["results"]:
        assert item["path"] == "docs/guide.md"
        assert 1 <= item["line_start"] <= len(guide_lines)
        assert item["line_start"] <= item["line_end"] <= len(
            guide_lines
        )


@pytest.mark.parametrize(
    "path",
    ("../x", "../../etc/passwd", "docs/../../x", "a/../../b"),
)
def test_path_traversal_is_rejected(tmp_path, monkeypatch, path):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "path": path},
        checkout,
        output_root,
    )
    assert result.success is False
    assert result.error.code == "PATH_OUTSIDE_REPOSITORY"


def test_path_symlink_escape_is_rejected(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch, noise=True)
    # the link's subject: create it here, or skip on hosts that
    # cannot create symlinks (never assert PATH_NOT_FOUND vacuously)
    symlink_or_skip(
        (checkout.parent.parent / "outside.md").resolve(),
        checkout / "linked.md",
    )
    result = _search(
        {"query": "checkout", "path": "linked.md"},
        checkout,
        output_root,
    )
    # the symlink resolves outside the checkout → rejected, never read
    assert result.success is False
    assert result.error.code in (
        "PATH_OUTSIDE_REPOSITORY",
        "PATH_NOT_FOUND",
    )


def test_symlinked_file_outside_repo_is_never_indexed(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch, noise=True)
    symlink_or_skip(
        (checkout.parent.parent / "outside.md").resolve(),
        checkout / "linked.md",
    )
    result = _search(
        {"query": "outsidemarker checkout"},
        checkout,
        output_root,
    )
    assert result.success is True
    paths = [item["path"] for item in result.data["results"]]
    assert "linked.md" not in paths
    assert not any(
        "outsidemarker" in text for text in _all_excerpts(result.data)
    )


def _deny_symlink(*args, **kwargs):
    """Simulate WinError 1314 (no SeCreateSymbolicLinkPrivilege)."""
    raise OSError(1314, "A required privilege is not held by the client")


def test_add_noise_works_without_symlink_privilege(tmp_path, monkeypatch):
    """A denied symlink must not cascade: the noise fixture still
    builds, so only the two symlink-subject tests are affected."""
    checkout = (
        tmp_path / "artifacts" / RUN_ID / "repository" / REPO_NAME
    )
    checkout.mkdir(parents=True)
    monkeypatch.setattr(os, "symlink", _deny_symlink)
    _add_noise(checkout)
    assert (checkout / "binary.txt").is_file()
    assert (checkout / "node_modules").is_dir()
    assert not (checkout / "linked.md").is_symlink()


def test_path_not_found(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "path": "nope.md"},
        checkout,
        output_root,
    )
    assert result.success is False
    assert result.error.code == "PATH_NOT_FOUND"


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"query": ""},
        {"query": "   "},
        {"query": None},
        {"query": 7},
        {"query": ["checkout"]},
    ),
)
def test_invalid_query(tmp_path, monkeypatch, arguments):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(arguments, checkout, output_root)
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"


def test_query_must_be_a_string_even_for_non_mapping_arguments(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search("checkout flow", checkout, output_root)
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"


@pytest.mark.parametrize(
    "max_results", (0, -1, 51, 1.5, True, "3", [], {})
)
def test_invalid_max_results(tmp_path, monkeypatch, max_results):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "max_results": max_results},
        checkout,
        output_root,
    )
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"


def test_unknown_argument_is_rejected(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "nonsense": 1},
        checkout,
        output_root,
    )
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"


def test_non_string_path_is_rejected(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search(
        {"query": "checkout", "path": 5},
        checkout,
        output_root,
    )
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"


def test_null_or_empty_path_means_no_filter(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    unfiltered = _search({"query": "checkout"}, checkout, output_root)
    null_path = _search(
        {"query": "checkout", "path": None},
        checkout,
        output_root,
    )
    empty_path = _search(
        {"query": "checkout", "path": ""},
        checkout,
        output_root,
    )
    assert unfiltered.success and null_path.success and empty_path.success
    assert null_path.data["results"] == unfiltered.data["results"]
    assert empty_path.data["results"] == unfiltered.data["results"]


# ---------------------------------------------------------------------------
# File discovery policy (index-time, shared file policy §14)
# ---------------------------------------------------------------------------


def test_binary_files_are_not_indexed(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch, noise=True)
    result = _search({"query": "secret-marker"}, checkout, output_root)
    assert result.success is True
    assert "binary.txt" not in [
        item["path"] for item in result.data["results"]
    ]
    assert not any(
        "secret-marker" in text for text in _all_excerpts(result.data)
    )


def test_oversized_files_are_not_indexed(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch, noise=True)
    result = _search(
        {"query": "oversizedmarker"}, checkout, output_root
    )
    assert result.success is True
    assert "oversized.txt" not in [
        item["path"] for item in result.data["results"]
    ]


def test_ignored_directories_are_not_indexed(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch, noise=True)
    for marker in ("nodemodmarker", "venvmarker"):
        result = _search({"query": marker}, checkout, output_root)
        assert result.success is True
        paths = [item["path"] for item in result.data["results"]]
        assert not any(
            path.startswith(("node_modules/", ".venv/"))
            for path in paths
        )
        assert not any(
            marker in text for text in _all_excerpts(result.data)
        )


def test_non_searchable_extensions_are_not_indexed(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch, noise=True)
    result = _search({"query": "checkout happened"}, checkout, output_root)
    assert result.success is True
    paths = [item["path"] for item in result.data["results"]]
    assert "notes.log" not in paths
    assert "data/sample.csv" not in paths


# ---------------------------------------------------------------------------
# Repository selection, provenance, independence, read-only
# ---------------------------------------------------------------------------


def test_repository_not_found_for_missing_explicit_root(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = search_repository(
        {"query": "checkout"},
        repository_root=tmp_path / "no-such-checkout",
        output_root=output_root,
    )
    assert result.success is False
    assert result.error.code == "REPOSITORY_NOT_FOUND"
    assert result.data is None


def test_repository_not_found_without_run_or_root(tmp_path):
    empty = tmp_path / "empty_out"
    empty.mkdir()
    result = search_repository({"query": "checkout"}, output_root=empty)
    assert result.success is False
    assert result.error.code == "REPOSITORY_NOT_FOUND"
    assert "--repository-root" in result.error.message


def test_default_repository_root_convention(tmp_path, monkeypatch):
    """Spec §36 O: the default root resolves the run's NAMED checkout
    (``<output>/<run>/repository/<name>/``) and searches it."""
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    assert checkout.parent.name == "repository"
    assert checkout.name == REPO_NAME
    result = search_repository(
        {"query": "checkout flow"}, output_root=output_root
    )
    assert result.success is True
    assert result.data["returned"] == DEFAULT_MAX_RESULTS
    for item in result.data["results"]:
        assert (checkout / item["path"]).is_file()


def test_source_reports_run_snapshot_provenance(
    tmp_path, monkeypatch
):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    assert result.source == {
        "repository": REPO_NAME,
        "commit_sha": COMMIT_SHA,
        "snapshot_id": RUN_ID,
        "repository_url": REPO_URL,
    }


def test_source_is_null_when_run_snapshot_is_unavailable(
    tmp_path, monkeypatch
):
    checkout, _ = _fixture(tmp_path, monkeypatch)
    empty = tmp_path / "empty_out"
    empty.mkdir()
    result = search_repository(
        {"query": "checkout"}, repository_root=checkout, output_root=empty
    )
    assert result.success is True
    assert result.source == {
        "repository": None,
        "commit_sha": None,
        "snapshot_id": None,
        "repository_url": None,
    }


def test_commit_isolation_same_repository_two_commits(
    tmp_path, monkeypatch
):
    """Spec §36 F: index for A@commit-1, A@commit-2 — a query never
    returns chunks carrying the other commit's provenance."""
    commit_one = "1" * 40
    commit_two = "2" * 40
    files_one = {
        "agents/agent.py": "ALPHA_ONE = 'alpha marker one'\n",
        "docs/a.md": "# A\n\nalpha marker one doc\n",
    }
    files_two = {
        "agents/agent.py": "ALPHA_TWO = 'alpha marker two'\n",
        "docs/a.md": "# A\n\nalpha marker two doc\n",
    }
    # one shared store — the two commits' indexes coexist and must
    # stay isolated by repository_key
    out = tmp_path / "artifacts"
    checkout_one = install_pinned_run(
        out,
        files_one,
        run_id="run_a1",
        repository_name="fixture-repository",
        commit_sha=commit_one,
    )
    checkout_two = install_pinned_run(
        out,
        files_two,
        run_id="run_a2",
        repository_name="fixture-repository",
        commit_sha=commit_two,
    )
    enable_search(checkout_one, monkeypatch, tmp_path)
    enable_search(checkout_two, monkeypatch, tmp_path)

    one = search_repository(
        {"query": "alpha marker"},
        repository_root=checkout_one,
        output_root=out,
    )
    two = search_repository(
        {"query": "alpha marker"},
        repository_root=checkout_two,
        output_root=out,
    )
    assert one.success and two.success
    assert one.data["retrieval"]["commit_sha"] == commit_one
    assert two.data["retrieval"]["commit_sha"] == commit_two
    for item in one.data["results"]:
        assert item["commit_sha"] == commit_one
        assert "alpha marker two" not in item["excerpt"]
    for item in two.data["results"]:
        assert item["commit_sha"] == commit_two
        assert "alpha marker one" not in item["excerpt"]


def test_repository_isolation_dual_fixtures(
    tmp_path, monkeypatch
):
    """Spec §44: fixture-repository (agents/tools/mcp/docs/unrelated)
    and a second repository with different names — the same code
    works unchanged and indexes stay isolated."""
    query = "How does the agent use the tool?"
    out = tmp_path / "artifacts"
    fixture_a = install_pinned_run(
        out,
        SPEC44_FILES,
        run_id="run_fixture_repository",
        repository_name="fixture-repository",
        commit_sha="1" * 40,
    )
    fixture_b = install_pinned_run(
        out,
        OTHER_FILES,
        run_id="run_other_repository",
        repository_name="other-repository",
        commit_sha="2" * 40,
    )
    # one shared store — repo isolation comes from repository_key
    enable_search(fixture_a, monkeypatch, tmp_path)
    enable_search(fixture_b, monkeypatch, tmp_path)

    in_a = search_repository(
        {"query": query}, repository_root=fixture_a, output_root=out
    )
    in_b = search_repository(
        {"query": query}, repository_root=fixture_b, output_root=out
    )
    assert in_a.success and in_b.success
    for item in in_a.data["results"]:
        assert item["repository_name"] == "fixture-repository"
        assert item["commit_sha"] == "1" * 40
        assert "gamma marker" not in item["excerpt"]
        assert (fixture_a / item["path"]).is_file()
    # the shared query still surfaces A's agent/tool sources
    a_paths = [item["path"] for item in in_a.data["results"]]
    assert any(path.startswith(("agents/", "tools/")) for path in a_paths)
    for item in in_b.data["results"]:
        assert item["repository_name"] == "other-repository"
        assert item["commit_sha"] == "2" * 40
        assert "checkout flow" not in item["excerpt"]
        assert (fixture_b / item["path"]).is_file()
    # B's unique marker only comes back from B's index
    b_hits = search_repository(
        {"query": "gamma marker"},
        repository_root=fixture_b,
        output_root=out,
    )
    a_hits = search_repository(
        {"query": "gamma marker"},
        repository_root=fixture_a,
        output_root=out,
    )
    assert any(
        "gamma marker" in item["excerpt"]
        for item in b_hits.data["results"]
    )
    assert not any(
        "gamma marker" in item["excerpt"]
        for item in a_hits.data["results"]
    )


def test_search_is_read_only(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    before = _tree_state(checkout)
    result = _search({"query": "checkout"}, checkout, output_root)
    assert result.success is True
    assert _tree_state(checkout) == before


def test_queries_never_mutate_the_index(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    store = tmp_path / "db_store"
    before = _tree_state(store)
    for query in ("checkout flow", "guide", "checkout"):
        result = _search({"query": query}, checkout, output_root)
        assert result.success is True
    assert _tree_state(store) == before


def test_representative_query_leaves_runtime_clean(
    tmp_path, monkeypatch
):
    """Spec §36 N: a query must not CREATE agent_memory.db,
    repository_query.db, a FAISS index or a RAG metadata DB under
    runtime/. Asserted as a before/after diff so a pre-existing
    legacy artifact in a dev checkout is not falsely attributed to
    the query — the query still may not add anything."""
    runtime = Path(__file__).resolve().parents[2] / "runtime"
    names = (
        "agent_memory.db",
        "repository_query.db",
        "index.faiss",
        "metadata.db",
    )

    def _present() -> set:
        found = set()
        for name in names:
            if (runtime / name).exists():
                found.add(name)
            for path in runtime.glob("**/{}".format(name)):
                found.add(str(path.relative_to(runtime)))
        return found

    before = _present()
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    assert result.success is True
    created = _present() - before
    assert created == set(), "query created under runtime/: {}".format(
        sorted(created)
    )


# ---------------------------------------------------------------------------
# Failure behavior (structured, no stack traces)
# ---------------------------------------------------------------------------


def test_missing_index_returns_actionable_structured_error(
    tmp_path, monkeypatch
):
    """Spec §36 H: missing index → structured, actionable error."""
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "other_store")
    )
    result = _search({"query": "checkout"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "RAG_INDEX_NOT_FOUND"
    assert "Build it with" in result.error.message
    assert "python -m repo_intelligence.retrieval.index" in (
        result.error.message
    )
    assert result.data is None


@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root bypasses directory permissions",
)
def test_unreadable_subdirectory_does_not_break_query(
    tmp_path, monkeypatch
):
    """Queries read the index, not the checkout — an unreadable
    directory cannot corrupt retrieval (read_file reports its own
    error when verification actually opens the tree)."""
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    locked = checkout / "locked"
    locked.mkdir()
    (locked / "note.md").write_text(
        "checkout inside locked\n", encoding="utf-8"
    )
    os.chmod(str(locked), 0o000)
    try:
        result = _search({"query": "checkout"}, checkout, output_root)
        assert result.success is True
        assert result.data["returned"] == DEFAULT_MAX_RESULTS
    finally:
        os.chmod(str(locked), 0o755)


# ---------------------------------------------------------------------------
# Layer constraints: no network/LLM imports
# ---------------------------------------------------------------------------


def test_tool_sources_have_no_network_or_llm_imports():
    forbidden = (
        "import socket",
        "import urllib",
        "import http",
        "import requests",
        "from requests",
        "openai",
        "langchain",
    )
    for path in sorted(TOOLS_SRC_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, "{} imports {!r}".format(path, token)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def test_standalone_cli_success(tmp_path, capsys):
    from conftest import build_hashing_index

    checkout = install_pinned_run(
        tmp_path / "artifacts", dict(CHECKOUT_FILES)
    )
    store = tmp_path / "rag_store"
    # first run: no index yet → actionable failure (exit 6)
    exit_code = main(
        [
            "--query",
            "checkout",
            "--path",
            "docs",
            "--max-results",
            "4",
            "--repository-root",
            str(checkout),
            "--output-root",
            str(tmp_path / "artifacts"),
            "--store",
            str(store),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 6
    assert payload["error"]["code"] == "RAG_INDEX_NOT_FOUND"

    # explicit build, then the query succeeds
    build_hashing_index(checkout, store_root=store)
    exit_code = main(
        [
            "--query",
            "checkout",
            "--path",
            "docs",
            "--max-results",
            "4",
            "--repository-root",
            str(checkout),
            "--output-root",
            str(tmp_path / "artifacts"),
            "--store",
            str(store),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["tool"] == "search_repository"
    assert payload["data"]["query"] == "checkout"
    assert 1 <= payload["data"]["returned"] <= 4
    for entry in payload["data"]["results"]:
        assert entry["path"].startswith("docs/")
    assert payload["source"]["repository"] == REPO_NAME
    assert payload["source"]["commit_sha"] == COMMIT_SHA
    assert payload["error"] is None


def test_standalone_cli_failure_is_structured(tmp_path, capsys):
    exit_code = main(
        [
            "--query",
            "checkout",
            "--repository-root",
            str(tmp_path / "missing"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 6
    assert payload["success"] is False
    assert payload["tool"] == "search_repository"
    assert payload["data"] is None
    assert payload["source"] is None
    assert payload["error"]["code"] == "REPOSITORY_NOT_FOUND"
    assert payload["error"]["message"]


# ---------------------------------------------------------------------------
# Read-file verification (spec §36 I)
# ---------------------------------------------------------------------------


def test_retrieved_range_reads_back_exactly(tmp_path, monkeypatch):
    """RAG candidate → read_file(path, line_start, line_end) returns
    exactly those lines of the pinned checkout."""
    from repo_intelligence.tools.read_file import read_file

    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)
    assert result.success is True
    for item in result.data["results"][:3]:
        read = read_file(
            {
                "path": item["path"],
                "line_start": item["line_start"],
                "line_end": item["line_end"],
            },
            output_root=output_root,
        )
        assert read.success is True
        assert read.data["path"] == item["path"]
        assert read.data["line_start"] == item["line_start"]
        assert read.data["line_end"] == item["line_end"]
        expected = (checkout / item["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        expected_range = expected[
            item["line_start"] - 1 : item["line_end"]
        ]
        assert read.data["content"] == "\n".join(expected_range)


# ---------------------------------------------------------------------------
# Pinned snapshot integration (M1 persists; tools read the frozen snapshot)
# ---------------------------------------------------------------------------


def _pinned_run(tmp_path: Path, monkeypatch):
    """Full M1 + M2 pipeline against a local origin repo.

    Returns (snapshot, output_root, origin); the checkout also gets
    an isolated hashing index so search_repository answers.
    """
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=str(origin),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"],
        cwd=str(origin),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=str(origin),
        check=True,
        capture_output=True,
        text=True,
    )
    # exact bytes: LF on any host, matching read-back expectations
    (origin / "README.md").write_bytes(
        b"# Sample Project\n\nManages the checkout flow "
        b"for demo orders.\n"
    )
    (origin / "notes.txt").write_bytes(
        b"prepare\nvalidate\ncheckout flow reserve\nconfirm\n"
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=str(origin),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=str(origin),
        check=True,
        capture_output=True,
        text=True,
    )

    out = tmp_path / "artifacts"
    snapshot, snap_path = create_snapshot(
        repository_url=str(origin), ref="main", output_root=out
    )
    create_facts(snapshot_path=snap_path, output_root=out)
    arch = out / snapshot.snapshot_id / "architecture"
    arch.mkdir(parents=True, exist_ok=True)
    (arch / "architecture_model.json").write_text("{}", encoding="utf-8")

    from repo_intelligence.artifacts import repository_dir

    pinned = repository_dir(out, snapshot.snapshot_id)
    enable_search(pinned, monkeypatch, tmp_path)
    return snapshot, out, origin


def test_search_operates_against_persisted_m1_snapshot(
    tmp_path, monkeypatch
):
    snapshot, out, origin = _pinned_run(tmp_path, monkeypatch)
    from repo_intelligence.artifacts import repository_dir

    pinned = repository_dir(out, snapshot.snapshot_id)
    assert pinned.is_dir()

    result = search_repository({"query": "checkout flow"}, output_root=out)
    assert result.success is True
    assert result.data["returned"] >= 1
    for entry in result.data["results"]:
        assert (pinned / entry["path"]).is_file()
        assert entry["commit_sha"] == snapshot.snapshot.commit_sha
    assert result.source == {
        "repository": origin.name,
        "commit_sha": snapshot.snapshot.commit_sha,
        "snapshot_id": snapshot.snapshot_id,
        "repository_url": str(origin),
    }

    # determinism: the same query against the same snapshot is stable
    again = search_repository({"query": "checkout flow"}, output_root=out)
    assert again.data == result.data


def test_search_never_reads_outside_pinned_repository(
    tmp_path, monkeypatch
):
    snapshot, out, _ = _pinned_run(tmp_path, monkeypatch)
    run_dir = out / snapshot.snapshot_id
    # Decoys inside the run directory but OUTSIDE the pinned
    # repository: neither can be returned (the index only ever
    # contains checkout content).
    (run_dir / "decoy.md").write_text(
        "checkout flow decoy line\n", encoding="utf-8"
    )
    (run_dir / "facts" / "repository_facts.json").write_text(
        '{"note": "checkout flow inside facts"}', encoding="utf-8"
    )

    from repo_intelligence.artifacts import repository_dir

    pinned = repository_dir(out, snapshot.snapshot_id)
    result = search_repository({"query": "checkout flow"}, output_root=out)
    assert result.success is True
    assert result.data["returned"] >= 1
    for entry in result.data["results"]:
        assert entry["path"] != "decoy.md"
        assert (pinned / entry["path"]).is_file()


# ---------------------------------------------------------------------------
# Response ceiling (bounded tool output)
# ---------------------------------------------------------------------------


def _fake_results(count: int, excerpt_len: int = 600) -> List[RetrievalResult]:
    """Deterministic ranked results with full-size excerpts."""
    return [
        RetrievalResult(
            file_path="docs/guide.md",
            line_start=index + 1,
            line_end=index + 6,
            chunk_id="chunk-{:03d}".format(index),
            score=round(0.9 - index * 0.01, 6),
            language="markdown",
            repository_name=REPO_NAME,
            commit_sha=COMMIT_SHA,
            snapshot_id="snapshot-1",
            excerpt="evidence {:03d}: ".format(index) + "x" * excerpt_len,
        )
        for index in range(count)
    ]


def _payload(results: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "query": "checkout flow",
        "results": results,
        "total_matches": len(results),
        "returned": len(results),
        "truncated": False,
        "evidence_note": EVIDENCE_NOTE,
        "omitted_results": 0,
        "note": None,
        "retrieval": {
            "mode": "semantic",
            "embedding_model": "hashing://v1/128",
            "repository": REPO_NAME,
            "commit_sha": COMMIT_SHA,
            "snapshot_id": "snapshot-1",
            "index_chunks": len(results),
        },
    }


def test_bound_payload_drops_whole_results_from_the_tail():
    originals = [item.to_dict() for item in _fake_results(60)]
    result = _bound_payload(_payload(originals), {"repository": REPO_NAME})

    assert result.success is True
    data = result.data
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    kept = data["results"]
    assert 0 < len(kept) < 60
    assert data["returned"] == len(kept)
    assert data["omitted_results"] == 60 - len(kept)
    assert data["returned"] + data["omitted_results"] == 60
    assert data["truncated"] is True
    # head of the SAME ranked list, in order, never re-ranked and never
    # cut: every delivered result is byte-identical to its input
    assert kept == originals[: len(kept)]
    assert data["note"] is not None
    assert (
        "{} of 60 results omitted".format(data["omitted_results"])
        in data["note"]
    )
    assert "{}-character response ceiling".format(
        MAX_TOOL_RESPONSE_CHARS
    ) in data["note"]


def test_bound_payload_passes_a_fitting_payload_through():
    originals = [item.to_dict() for item in _fake_results(3, excerpt_len=80)]
    source = {"repository": REPO_NAME, "commit_sha": COMMIT_SHA}
    result = _bound_payload(_payload(originals), source)

    assert result.success is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert result.source == source
    data = result.data
    assert data["results"] == originals
    assert data["returned"] == 3
    assert data["omitted_results"] == 0
    assert data["note"] is None
    assert data["truncated"] is False  # the caller's condition, kept


def test_bound_payload_counts_the_envelope_not_only_the_results():
    """Two results that fit AS CONTENT can still break the ceiling once
    the envelope is added — the bound counts both."""
    originals = [
        item.to_dict() for item in _fake_results(2, excerpt_len=11_900)
    ]
    assert sum(len(item["excerpt"]) for item in originals) < (
        MAX_TOOL_RESPONSE_CHARS
    )

    result = _bound_payload(_payload(originals), {"repository": REPO_NAME})
    data = result.data
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["returned"] == 1  # the highest-ranked result survives
    assert data["omitted_results"] == 1
    assert data["results"] == originals[:1]


def test_query_over_the_ceiling_is_a_structured_failure():
    payload = _payload([])
    payload["query"] = "z" * (MAX_TOOL_RESPONSE_CHARS + 1000)
    result = _bound_payload(payload, {"repository": REPO_NAME})

    assert result.success is False
    assert result.error.code == "QUERY_TOO_LONG"
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert len(result.error.message) < 500
    assert "z" * 50 not in result.error.message  # the query is not echoed


def test_oversized_search_payload_is_capped_end_to_end(
    tmp_path, monkeypatch
):
    """The ceiling holds through the public tool: identity and index
    resolution stay real, only the ranking call is stubbed with more
    results than one response can carry."""
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    fake = _fake_results(40)

    class _Response:
        def __init__(self, results):
            self.results = results
            self.total_matches = len(results)
            self.index_chunks = len(results)
            self.scan_capped = False

    class _Service:
        def __init__(self, store_root=None):
            self.store_root = store_root

        def search(self, query, identity, top_k=5, path_filter=None):
            return _Response(list(fake[:top_k]))

    monkeypatch.setattr(search_module, "RetrievalService", _Service)

    result = _search(
        {"query": "checkout flow", "max_results": 40}, checkout, output_root
    )
    assert result.success is True
    data = result.data
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["omitted_results"] > 0
    assert data["returned"] == len(data["results"])
    assert data["returned"] + data["omitted_results"] == 40
    assert data["truncated"] is True
    assert data["note"] is not None
    assert data["results"] == [
        item.to_dict() for item in fake[: data["returned"]]
    ]
    assert data["retrieval"]["repository"] == REPO_NAME
    assert data["retrieval"]["commit_sha"] == COMMIT_SHA


def test_ordinary_search_stays_within_the_ceiling(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path, monkeypatch)
    result = _search({"query": "checkout flow"}, checkout, output_root)

    assert result.success is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    data = result.data
    assert data["omitted_results"] == 0
    assert data["note"] is None
    assert data["returned"] == len(data["results"])
