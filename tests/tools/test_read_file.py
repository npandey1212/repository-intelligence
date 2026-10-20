"""Focused tests for Tool 3: read_file (read-only).

Tool-layer tests are intentionally separate from the deterministic M1–M6
foundation tests: they exercise the generic ToolResult protocol, the
tool's input/line-range/error contracts, path safety, binary/encoding/
size policies, repository independence, provenance, and read-only
behavior against a small synthetic checkout with generic content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

from conftest import symlink_or_skip, try_symlink
from repo_intelligence.facts import create_facts
from repo_intelligence.snapshot import create_snapshot
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.protocol import (
    MAX_TOOL_RESPONSE_CHARS,
    response_chars,
)
from repo_intelligence.tools.read_file import TOOL_NAME, main, read_file
from repo_intelligence.tools.search_repository import (
    MAX_FILE_BYTES,
    resolve_repository_root,
)

TOOLS_SRC_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "tools"
)

RUN_ID = "run_fixture_alpha"
REPO_NAME = "sample-repo-alpha"
COMMIT_SHA = "b" * 40

EXAMPLE_TEXT = "alpha\nbeta\ngamma\ndelta\nepsilon\n"  # 5 lines

CHECKOUT_FILES = {
    "README.md": (
        "# Sample Project\n"
        "\n"
        "This project manages the checkout flow for demo orders.\n"
    ),
    "example.txt": EXAMPLE_TEXT,
    "notes.txt": "one\ntwo",  # no trailing newline
    "empty.txt": "",
    "docs/workflow.md": (
        "# Workflow\n"
        "\n"
        "1. prepare\n"
        "2. validate\n"
        "3. reserve\n"
        "4. confirm\n"
        "\n"
        "Notes.\n"
    ),
    "docs/api/reference.md": "# API Reference\n\nGET /status returns state.\n",
    "data/sample.csv": "id,name\n1,first\n",
}


def _write_run(root: Path, run_id: str = RUN_ID) -> Path:
    """Minimal existing-run layout for provenance and run resolution."""
    run_dir = root / run_id
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {
                    "name": REPO_NAME,
                    "url": "https://example.invalid/sample.git",
                },
                "snapshot": {"ref": "main", "commit_sha": COMMIT_SHA},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_checkout(
    root: Path, files: Optional[Dict[str, str]] = None
) -> Path:
    """Materialize a synthetic checkout (plus policy/escape cases)."""
    root.mkdir(parents=True, exist_ok=True)
    standard = files is None
    if standard:
        files = dict(CHECKOUT_FILES)
        # symlink escape: outside file + inside symlink (best-effort —
        # hosts that cannot create links still get a usable checkout;
        # the symlink-subject test creates its own link or skips)
        outside = root.parent / "outside.md"
        outside.write_text(
            "outsidemarker outside content\n", encoding="utf-8"
        )
        try_symlink(outside.resolve(), root / "linked.md")
        # binary file (NUL byte in the sniff window)
        files_raw = {
            "binary.txt": b"PK\x03\x04\x00\x01 binary payload\n",
            # invalid UTF-8, but no NUL byte
            "nonutf8.txt": b"caf\xe9 latin-1 text\n",
        }
    else:
        files_raw = {}
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            # exact bytes: write_text would use os.linesep on Windows
            path.write_bytes(content.encode("utf-8"))
    for rel, content in files_raw.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if standard:
        # oversized file (over the shared byte cap)
        (root / "oversized.txt").write_text(
            "start\n" + "z" * MAX_FILE_BYTES, encoding="utf-8"
        )
    return root


def _fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Standard test pair: (checkout, artifacts output root with a run)."""
    output_root = tmp_path / "artifacts"
    _write_run(output_root)
    checkout = _write_checkout(tmp_path / "checkout")
    return checkout, output_root


def _read(arguments, checkout: Path, output_root: Path) -> ToolResult:
    return read_file(
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


# ---------------------------------------------------------------------------
# Generic ToolResult protocol (shared by all tools)
# ---------------------------------------------------------------------------


def test_result_shape_is_generic(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt"}, checkout, output_root)
    assert isinstance(result, ToolResult)
    payload = result.to_dict()
    assert list(payload) == ["success", "tool", "data", "source", "error"]
    assert payload["tool"] == TOOL_NAME == "read_file"
    assert payload["success"] is True
    assert payload["error"] is None
    # evidence only — the exact documented data keys, nothing more
    assert sorted(result.data) == [
        "char_limit",
        "content",
        "line_end",
        "line_start",
        "next_line_start",
        "note",
        "partial_line",
        "path",
        "returned_chars",
        "total_chars",
        "truncated",
    ]


# ---------------------------------------------------------------------------
# Basic reading + full file
# ---------------------------------------------------------------------------


def test_reads_normal_text_file(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "README.md"}, checkout, output_root)
    assert result.success is True
    expected = CHECKOUT_FILES["README.md"]
    assert result.data["path"] == "README.md"
    assert result.data["content"] == expected
    assert result.data["line_start"] == 1
    assert result.data["line_end"] == 3  # final line reported


def test_full_file_reports_final_line(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt"}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == EXAMPLE_TEXT  # byte-faithful
    assert result.data["line_start"] == 1
    assert result.data["line_end"] == 5


def test_full_file_without_trailing_newline(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "notes.txt"}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == "one\ntwo"
    assert result.data["line_end"] == 2


def test_empty_file(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "empty.txt"}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == ""
    assert result.data["line_start"] == 1
    assert result.data["line_end"] == 0  # no lines to report


def test_reads_any_text_extension(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    # read_file has no extension whitelist: an explicitly named file is read
    result = _read({"path": "data/sample.csv"}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == CHECKOUT_FILES["data/sample.csv"]


def test_backslash_path_is_normalized(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "docs\\workflow.md"}, checkout, output_root)
    assert result.success is True
    assert result.data["path"] == "docs/workflow.md"


# ---------------------------------------------------------------------------
# Line ranges (1-based, inclusive)
# ---------------------------------------------------------------------------


def test_range_returns_exact_spec_example(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read(
        {"path": "example.txt", "line_start": 2, "line_end": 4},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["content"] == "beta\ngamma\ndelta"
    assert result.data["line_start"] == 2
    assert result.data["line_end"] == 4


def test_line_start_only_runs_to_end_of_file(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt", "line_start": 4}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == "delta\nepsilon"
    assert result.data["line_start"] == 4
    assert result.data["line_end"] == 5


def test_line_end_only_runs_from_first_line(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt", "line_end": 2}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == "alpha\nbeta"
    assert result.data["line_start"] == 1
    assert result.data["line_end"] == 2


def test_single_line_range(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read(
        {"path": "example.txt", "line_start": 3, "line_end": 3},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["content"] == "gamma"
    assert result.data["line_start"] == 3
    assert result.data["line_end"] == 3


def test_first_line_range(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read(
        {"path": "example.txt", "line_start": 1, "line_end": 1},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["content"] == "alpha"


def test_last_line_range(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read(
        {"path": "example.txt", "line_start": 5, "line_end": 5},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["content"] == "epsilon"
    assert result.data["line_end"] == 5


def test_explicit_full_range_joins_lines_without_trailing_newline(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    full = _read({"path": "example.txt"}, checkout, output_root)
    ranged = _read(
        {"path": "example.txt", "line_start": 1, "line_end": 5},
        checkout,
        output_root,
    )
    assert ranged.success is True
    assert ranged.data["line_start"] == 1
    assert ranged.data["line_end"] == 5
    assert ranged.data["content"] == "alpha\nbeta\ngamma\ndelta\nepsilon"
    # documented difference: full read is byte-faithful (keeps the final
    # newline), a range read returns exactly the selected lines
    assert full.data["content"] == ranged.data["content"] + "\n"


def test_range_past_end_of_file_is_empty(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt", "line_start": 99}, checkout, output_root)
    assert result.success is True
    assert result.data["content"] == ""
    assert result.data["line_start"] == 1
    assert result.data["line_end"] == 0


def test_range_end_beyond_file_is_clamped(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read(
        {"path": "example.txt", "line_start": 4, "line_end": 999},
        checkout,
        output_root,
    )
    assert result.success is True
    assert result.data["content"] == "delta\nepsilon"
    assert result.data["line_end"] == 5  # actual, not requested


# ---------------------------------------------------------------------------
# Line validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "example.txt", "line_start": 0},
        {"path": "example.txt", "line_start": -1},
        {"path": "example.txt", "line_end": 0},
        {"path": "example.txt", "line_end": -3},
        {"path": "example.txt", "line_start": "2"},
        {"path": "example.txt", "line_start": 2.5},
        {"path": "example.txt", "line_start": True},
        {"path": "example.txt", "line_end": [4]},
        {"path": "example.txt", "line_start": 4, "line_end": 2},
    ],
)
def test_invalid_line_bounds(tmp_path, arguments):
    checkout, output_root = _fixture(tmp_path)
    result = _read(arguments, checkout, output_root)
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"
    assert result.data is None


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        {},
        {"path": ""},
        {"path": "   "},
        {"path": 42},
        {"path": None},
    ],
)
def test_invalid_path_argument(tmp_path, arguments):
    checkout, output_root = _fixture(tmp_path)
    result = _read(arguments, checkout, output_root)
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"
    assert result.data is None


def test_unknown_argument_is_rejected(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read(
        {"path": "example.txt", "encoding": "utf-8"},
        checkout,
        output_root,
    )
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"
    assert "encoding" in result.error.message


def test_non_mapping_arguments(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = read_file(
        "example.txt", repository_root=checkout, output_root=output_root
    )
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# Missing files and non-files
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    for path in ("missing.md", "docs/missing.md"):
        result = _read({"path": path}, checkout, output_root)
        assert result.success is False
        assert result.error.code == "FILE_NOT_FOUND"
        assert result.data is None


def test_missing_directory_component(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "docs/nope/example.txt"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "PATH_NOT_FOUND"


def test_intermediate_component_that_is_a_file(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "README.md/example.txt"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "PATH_NOT_FOUND"


def test_directory_is_not_a_file(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    for path in ("docs", "docs/", "."):
        result = _read({"path": path}, checkout, output_root)
        assert result.success is False
        assert result.error.code == "NOT_A_FILE"
        assert result.data is None


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../file.txt",
        "../../etc/passwd",
        "/etc/passwd",
        "docs/../../outside.md",
        "..\\..\\win.txt",
        "C:/secret.txt",
        "C:\\secret.txt",
    ],
)
def test_path_traversal_is_rejected(tmp_path, path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": path}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "PATH_OUTSIDE_REPOSITORY"
    assert result.data is None


def test_symlink_escape_is_rejected(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    # linked.md exists inside the checkout but resolves outside it
    # (create it here so a link-less host skips instead of failing)
    symlink_or_skip(
        (tmp_path / "outside.md").resolve(), checkout / "linked.md"
    )
    assert (checkout / "linked.md").is_symlink()
    result = _read({"path": "linked.md"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "PATH_OUTSIDE_REPOSITORY"
    # the outside content is never returned anywhere
    assert "outsidemarker" not in str(result.to_dict())


def _deny_symlink(*args, **kwargs):
    """Simulate WinError 1314 (no SeCreateSymbolicLinkPrivilege)."""
    raise OSError(1314, "A required privilege is not held by the client")


def test_checkout_fixture_works_without_symlink_privilege(
    tmp_path, monkeypatch
):
    """A denied symlink must not cascade: the shared checkout fixture
    still builds, so only the symlink-subject test is affected."""
    monkeypatch.setattr(os, "symlink", _deny_symlink)
    checkout = _write_checkout(tmp_path / "checkout")
    assert (checkout / "README.md").is_file()
    assert not (checkout / "linked.md").is_symlink()


def test_symlink_subject_skips_when_links_are_denied(tmp_path, monkeypatch):
    """The skip path itself: denial produces a skip with a reason,
    never a fabricated link and never a silent pass."""
    monkeypatch.setattr(os, "symlink", _deny_symlink)
    outcome = None
    try:
        symlink_or_skip(
            str(tmp_path / "outside.md"), str(tmp_path / "linked.md")
        )
    except BaseException as exc:  # noqa: BLE001 — skip is BaseException
        outcome = exc
    assert outcome is not None, "expected a skip when symlinks are denied"
    assert "symlink" in str(outcome)


# ---------------------------------------------------------------------------
# Binary / encoding / size policy
# ---------------------------------------------------------------------------


def test_binary_file_is_rejected(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "binary.txt"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "BINARY_FILE"
    assert result.data is None


def test_unsupported_encoding_is_rejected(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "nonutf8.txt"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "UNSUPPORTED_ENCODING"
    assert result.data is None


def test_oversized_file_is_rejected(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    big = checkout / "oversized.txt"
    assert big.stat().st_size > MAX_FILE_BYTES
    result = _read({"path": "oversized.txt"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "FILE_TOO_LARGE"
    assert str(MAX_FILE_BYTES) in result.error.message


def _emulate_windows_read_denial(target, monkeypatch) -> None:
    """Patch ``Path.read_bytes`` to raise PermissionError for *target*.

    Windows chmod cannot express "unreadable": mode 0o000 only
    toggles the read-only attribute (writes blocked, reads allowed);
    a real Windows denial is an ACL. This emulates that OS-level
    denial for exactly one file so the READ_FAILED contract can be
    asserted on Windows too.
    """
    original_read_bytes = Path.read_bytes
    denied = Path(target).resolve()

    def _denied(path_self, *args, **kwargs):
        if Path(path_self).resolve() == denied:
            raise PermissionError(
                13, "Permission denied", str(path_self)
            )
        return original_read_bytes(path_self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _denied)


@pytest.mark.skipif(
    os.geteuid() == 0 if hasattr(os, "geteuid") else False,
    reason="root bypasses file permissions",
)
def test_unreadable_file_is_read_failed(tmp_path, monkeypatch):
    checkout, output_root = _fixture(tmp_path)
    target = checkout / "example.txt"
    if os.name == "nt":
        # real Windows denial is an ACL (chmod cannot deny reads);
        # emulate it for this file, keep the same assertions
        _emulate_windows_read_denial(target, monkeypatch)
    else:
        os.chmod(str(target), 0o000)
    try:
        result = _read({"path": "example.txt"}, checkout, output_root)
        assert result.success is False
        assert result.error.code == "READ_FAILED"
        assert result.data is None
        assert result.error.message
    finally:
        os.chmod(str(target), 0o644)


def test_windows_read_denial_emulation_is_targeted(
    tmp_path, monkeypatch
):
    """Regression for the nt-branch above: the emulated denial must
    hit exactly the denied file, leave every other file readable,
    and still surface as READ_FAILED through the tool."""
    checkout, output_root = _fixture(tmp_path)
    target = checkout / "example.txt"
    _emulate_windows_read_denial(target, monkeypatch)
    with pytest.raises(PermissionError):
        target.read_bytes()
    other = checkout / "docs" / "workflow.md"
    assert other.read_bytes()  # untouched files stay readable
    result = _read({"path": "example.txt"}, checkout, output_root)
    assert result.success is False
    assert result.error.code == "READ_FAILED"


# ---------------------------------------------------------------------------
# Determinism, independence, provenance, read-only
# ---------------------------------------------------------------------------


def test_repeated_reads_are_identical(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    arguments = {"path": "docs/workflow.md", "line_start": 3, "line_end": 6}
    payloads = [
        _read(arguments, checkout, output_root).to_dict() for _ in range(3)
    ]
    assert payloads[0] == payloads[1] == payloads[2]


def test_repository_independence(tmp_path):
    checkout_a, output_root = _fixture(tmp_path)
    checkout_b = _write_checkout(
        tmp_path / "checkout_beta",
        {"README.md": "# Beta Project\n\ndifferent content line\n"},
    )
    read_b = read_file(
        {"path": "README.md"}, repository_root=checkout_b, output_root=output_root
    )
    read_a = read_file(
        {"path": "README.md"}, repository_root=checkout_a, output_root=output_root
    )
    assert read_b.success and "different content line" in read_b.data["content"]
    assert read_a.success and "checkout flow" in read_a.data["content"]
    assert read_a.data["content"] != read_b.data["content"]


def test_source_reports_run_snapshot_provenance(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt"}, checkout, output_root)
    assert result.source == {
        "repository": REPO_NAME,
        "commit_sha": COMMIT_SHA,
        "snapshot_id": RUN_ID,
        "repository_url": "https://example.invalid/sample.git",
    }


def test_source_is_null_when_run_snapshot_is_unavailable(tmp_path):
    checkout, _ = _fixture(tmp_path)
    empty = tmp_path / "empty_out"
    empty.mkdir()
    result = read_file(
        {"path": "example.txt"}, repository_root=checkout, output_root=empty
    )
    assert result.success is True
    assert result.source == {
        "repository": None,
        "commit_sha": None,
        "snapshot_id": None,
        "repository_url": None,
    }


def test_default_repository_root_convention(tmp_path):
    output_root = tmp_path / "artifacts"
    _write_run(output_root)
    _write_checkout(output_root / RUN_ID / "repository")
    result = read_file({"path": "example.txt"}, output_root=output_root)
    assert result.success is True
    assert result.data["content"] == EXAMPLE_TEXT


def test_repository_not_found(tmp_path):
    output_root = tmp_path / "artifacts"
    _write_run(output_root)
    result = read_file(
        {"path": "example.txt"},
        repository_root=tmp_path / "no-such-checkout",
        output_root=output_root,
    )
    assert result.success is False
    assert result.error.code == "REPOSITORY_NOT_FOUND"


def test_read_is_read_only(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    before = _tree_state(checkout)
    full = _read({"path": "example.txt"}, checkout, output_root)
    ranged = _read(
        {"path": "docs/workflow.md", "line_start": 3, "line_end": 6},
        checkout,
        output_root,
    )
    _read({"path": "missing.md"}, checkout, output_root)
    assert full.success and ranged.success
    assert _tree_state(checkout) == before


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def test_standalone_cli_full_file(tmp_path, capsys):
    checkout, output_root = _fixture(tmp_path)
    exit_code = main(
        [
            "--path",
            "example.txt",
            "--repository-root",
            str(checkout),
            "--output-root",
            str(output_root),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["tool"] == "read_file"
    assert payload["data"] == {
        "path": "example.txt",
        "line_start": 1,
        "line_end": 5,
        "content": EXAMPLE_TEXT,
        "truncated": False,
        "returned_chars": len(EXAMPLE_TEXT),
        "total_chars": len(EXAMPLE_TEXT),
        "char_limit": MAX_TOOL_RESPONSE_CHARS,
        "next_line_start": None,
        "partial_line": False,
        "note": None,
    }
    assert payload["source"]["repository"] == REPO_NAME
    assert payload["source"]["commit_sha"] == COMMIT_SHA
    assert payload["error"] is None


def test_standalone_cli_with_range(tmp_path, capsys):
    checkout, output_root = _fixture(tmp_path)
    exit_code = main(
        [
            "--path",
            "example.txt",
            "--line-start",
            "2",
            "--line-end",
            "4",
            "--repository-root",
            str(checkout),
            "--output-root",
            str(output_root),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["data"]["content"] == "beta\ngamma\ndelta"
    assert payload["data"]["line_start"] == 2
    assert payload["data"]["line_end"] == 4


def test_standalone_cli_failure_is_structured(tmp_path, capsys):
    checkout, output_root = _fixture(tmp_path)
    exit_code = main(
        [
            "--path",
            "missing.md",
            "--repository-root",
            str(checkout),
            "--output-root",
            str(output_root),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 6
    assert payload["success"] is False
    assert payload["tool"] == "read_file"
    assert payload["data"] is None
    assert payload["source"] is None
    assert payload["error"]["code"] == "FILE_NOT_FOUND"
    assert payload["error"]["message"]


# ---------------------------------------------------------------------------
# Pinned snapshot integration (M1 persists; tools read the frozen snapshot)
# ---------------------------------------------------------------------------


def _pinned_run(tmp_path: Path):
    """Full M1 + M2 pipeline against a local origin repo.

    Returns (snapshot, output_root, origin). An architecture placeholder
    is written so the run is discoverable (architecture generation is
    out of scope for the tool-layer tests).
    """
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"], cwd=str(origin),
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=str(origin),
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=str(origin),
        check=True, capture_output=True, text=True,
    )
    # exact bytes: LF on any host, matching read-back expectations
    (origin / "README.md").write_bytes(
        b"# Sample Project\n\ncontent line one\ncontent line two\n"
    )
    subprocess.run(
        ["git", "add", "."], cwd=str(origin),
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=str(origin),
        check=True, capture_output=True, text=True,
    )

    out = tmp_path / "artifacts"
    snapshot, snap_path = create_snapshot(
        repository_url=str(origin), ref="main", output_root=out
    )
    create_facts(snapshot_path=snap_path, output_root=out)
    arch = out / snapshot.snapshot_id / "architecture"
    arch.mkdir(parents=True, exist_ok=True)
    (arch / "architecture_model.json").write_text("{}", encoding="utf-8")
    return snapshot, out, origin


def test_read_file_reads_from_persisted_m1_snapshot(tmp_path):
    snapshot, out, origin = _pinned_run(tmp_path)
    from repo_intelligence.artifacts import repository_dir

    pinned = repository_dir(out, snapshot.snapshot_id)
    assert pinned.is_dir()

    # Default resolution: no explicit repository_root — the pinned
    # snapshot of the current run is used automatically.
    result = read_file({"path": "README.md"}, output_root=out)
    assert result.success is True
    expected = (origin / "README.md").read_text(encoding="utf-8")
    assert result.data["content"] == expected
    assert result.source == {
        "repository": origin.name,
        "commit_sha": snapshot.snapshot.commit_sha,
        "snapshot_id": snapshot.snapshot_id,
        "repository_url": str(origin),
    }

    # Alignment with search_repository: both tools resolve the same
    # pinned repository directory by default.
    resolved, error = resolve_repository_root(None, out)
    assert error is None
    assert resolved == pinned.resolve()


def test_read_file_roundtrips_crlf_bytes(tmp_path):
    """CRLF content is returned verbatim (byte-faithful contract):
    the full read equals the raw bytes, and a range read keeps
    ``\\r`` attached to each selected line — LF-derived expectations
    must never be compared against CRLF-ondisk fixtures."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    data = b"# Title\r\nline two\r\nline three\r\n"
    (checkout / "crlf.md").write_bytes(data)

    full = _read({"path": "crlf.md"}, checkout, tmp_path)
    assert full.success is True
    assert full.data["content"].encode("utf-8") == data

    ranged = _read(
        {"path": "crlf.md", "line_start": 2, "line_end": 2},
        checkout,
        tmp_path,
    )
    assert ranged.success is True
    assert ranged.data["content"] == "line two\r"


# ---------------------------------------------------------------------------
# Response ceiling (bounded tool output)
# ---------------------------------------------------------------------------


def _big_lines(checkout: Path, name: str, count: int, width: int = 79) -> str:
    """A multi-line file of exactly ``count`` lines, ``width`` chars each."""
    text = "\n".join(
        "l{:04d}".format(index) + "x" * (width - 5)
        for index in range(count)
    )
    (checkout / name).write_text(text, encoding="utf-8")
    return text


def test_complete_read_reports_the_bounded_contract_fields(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    result = _read({"path": "example.txt"}, checkout, output_root)
    data = result.data
    assert data["truncated"] is False
    assert data["returned_chars"] == data["total_chars"] == len(EXAMPLE_TEXT)
    assert data["char_limit"] == MAX_TOOL_RESPONSE_CHARS
    assert data["next_line_start"] is None
    assert data["partial_line"] is False
    assert data["note"] is None
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS


def test_oversized_selection_is_bounded_to_the_ceiling(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    text = _big_lines(checkout, "big.txt", 600)
    lines = text.split("\n")

    result = _read({"path": "big.txt"}, checkout, output_root)
    data = result.data
    assert result.success is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is True
    assert data["total_chars"] == len(text)
    assert data["returned_chars"] < data["total_chars"]
    assert data["char_limit"] == MAX_TOOL_RESPONSE_CHARS
    assert data["partial_line"] is False
    # the delivered range covers the COMPLETE lines actually returned —
    # never the whole requested file
    assert data["line_start"] == 1
    assert data["line_end"] < 600
    assert data["content"] == "\n".join(lines[: data["line_end"]])
    assert data["returned_chars"] == len(data["content"])
    assert data["next_line_start"] == data["line_end"] + 1
    assert data["note"] is not None
    assert "resume at line {}".format(data["next_line_start"]) in data["note"]


def test_continuation_windows_reassemble_the_whole_file(tmp_path):
    """Each bounded read names where to resume; walking those windows
    returns every line exactly once and never re-delivers content."""
    checkout, output_root = _fixture(tmp_path)
    text = _big_lines(checkout, "big.txt", 600)

    windows = []
    previous_end: Optional[int] = None
    start: Optional[int] = None
    for _ in range(50):
        arguments: Dict[str, object] = {"path": "big.txt"}
        if start is not None:
            arguments["line_start"] = start
        result = _read(arguments, checkout, output_root)
        assert result.success is True
        assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
        data = result.data
        if previous_end is None:
            assert data["line_start"] == 1
        else:
            assert data["line_start"] == previous_end + 1
        assert data["line_start"] <= data["line_end"]
        previous_end = data["line_end"]
        windows.append(data["content"])
        if data["next_line_start"] is None:
            break
        start = data["next_line_start"]
    else:
        raise AssertionError("continuation never reached the end of the file")

    assert "\n".join(windows) == text


def test_range_read_reports_only_the_delivered_window(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    text = _big_lines(checkout, "big.txt", 600)
    lines = text.split("\n")

    result = _read(
        {"path": "big.txt", "line_start": 100, "line_end": 600},
        checkout,
        output_root,
    )
    data = result.data
    assert result.success is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is True
    assert data["line_start"] == 100
    assert 100 < data["line_end"] < 600  # delivered, never the request
    assert data["next_line_start"] == data["line_end"] + 1
    assert data["content"] == "\n".join(lines[99 : data["line_end"]])
    assert data["total_chars"] == len("\n".join(lines[99:600]))
    assert data["returned_chars"] == len(data["content"])


def test_single_line_over_the_ceiling_yields_empty_range(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    huge = "z" * (MAX_TOOL_RESPONSE_CHARS * 2)
    (checkout / "huge.txt").write_text(huge, encoding="utf-8")

    result = _read({"path": "huge.txt"}, checkout, output_root)
    data = result.data
    assert result.success is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is True
    assert data["partial_line"] is True
    assert data["line_start"] == 1
    assert data["line_end"] == 0  # nothing was delivered COMPLETE
    assert data["next_line_start"] is None  # re-reading hits the ceiling
    assert data["content"].startswith("zzz")
    assert len(data["content"]) < len(huge)
    assert data["returned_chars"] == len(data["content"])
    assert data["total_chars"] == len(huge)
    assert data["note"] is not None
    assert "no complete line" in data["note"]


def test_partial_line_in_a_range_read_keeps_the_region(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    text = "intro\n" + "z" * (MAX_TOOL_RESPONSE_CHARS * 2)
    (checkout / "mixed.txt").write_text(text, encoding="utf-8")

    result = _read({"path": "mixed.txt", "line_start": 2}, checkout, output_root)
    data = result.data
    assert result.success is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is True
    assert data["partial_line"] is True
    assert data["line_start"] == 2
    assert data["line_end"] == 1  # empty delivered range: start - 1
    assert data["next_line_start"] is None
    assert data["content"].startswith("zzz")


def test_the_ceiling_counts_the_envelope_not_only_the_content(tmp_path):
    """A selection that fits the ceiling AS TEXT can still break it once
    the ToolResult envelope is added — the bound has to count both."""
    checkout, output_root = _fixture(tmp_path)
    text = "\n".join("y" * 79 for _ in range(300))  # 23999 chars
    (checkout / "edge.txt").write_text(text, encoding="utf-8")
    assert len(text) < MAX_TOOL_RESPONSE_CHARS

    result = _read({"path": "edge.txt"}, checkout, output_root)
    data = result.data
    assert data["truncated"] is True
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["returned_chars"] < len(text)


def test_bounded_output_is_deterministic(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    _big_lines(checkout, "big.txt", 600)
    first = _read({"path": "big.txt"}, checkout, output_root)
    second = _read({"path": "big.txt"}, checkout, output_root)
    assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())


def test_truncated_read_keeps_provenance(tmp_path):
    checkout, output_root = _fixture(tmp_path)
    _big_lines(checkout, "big.txt", 600)
    result = _read({"path": "big.txt"}, checkout, output_root)
    assert result.success is True
    assert result.source["repository"] == REPO_NAME
    assert result.source["commit_sha"] == COMMIT_SHA
    assert result.data["truncated"] is True
