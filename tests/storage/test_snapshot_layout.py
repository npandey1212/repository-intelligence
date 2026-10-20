"""M18 snapshot layout: ``repository/<name>/`` with legacy fallback.

The pinned checkout lives in a subdirectory named from the snapshot's
own ``repository.name`` (Test O). Pre-M18 flat run directories keep
resolving exactly as before (no fixture or artifact rewrite), and an
unusable name never escapes the per-run container.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from repo_intelligence.artifacts import (
    repository_dir,
    repository_dir_for_name,
    repository_dir_for_snapshot,
)
from repo_intelligence.snapshot import create_snapshot
from repo_intelligence.snapshot.builder import (
    _remove_quietly,
    _rmtree_force,
    repository_name_from_url,
)

_NOT_ROOT = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file permission bits",
)


def _git(args, cwd):
    subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def make_local_repo(path: Path) -> Path:
    """Small local Git repo (name comes from the path basename)."""
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test User"], path)
    (path / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(["add", "."], path)
    _git(["commit", "-q", "-m", "initial commit"], path)
    return path


def _write_snapshot_json(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"repository": {"name": name, "url": "https://x/y.git"}}
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# named layout written by create_snapshot (Test O)
# ---------------------------------------------------------------------------


def test_create_snapshot_persists_named_checkout(tmp_path: Path):
    repo = make_local_repo(tmp_path / "sample-fixture-repo")
    out = tmp_path / "artifacts"
    snapshot, artifact = create_snapshot(
        repository_url=str(repo), ref="main", output_root=out
    )
    assert snapshot.repository.name == "sample-fixture-repo"

    container = out / snapshot.snapshot_id / "repository"
    named = container / "sample-fixture-repo"
    assert named.is_dir()
    assert (named / ".git").exists()
    assert (named / "README.md").is_file()
    # exactly one entry in the container: the named checkout
    assert {p.name for p in container.iterdir()} == {"sample-fixture-repo"}

    # every reader agrees on the named path
    assert repository_dir(out, snapshot.snapshot_id) == named
    assert repository_dir_for_snapshot(artifact) == named
    assert repository_dir_for_name(
        out, snapshot.snapshot_id, snapshot.repository.name
    ) == named


def test_writer_replaces_stale_flat_layout(tmp_path: Path):
    repo = make_local_repo(tmp_path / "sample-fixture-repo")
    out = tmp_path / "artifacts"
    # simulate a pre-M18 flat run directory for the same snapshot id
    # (id is deterministic: created before we know it — so write the
    # flat junk AFTER the first run and re-run)
    snapshot, _ = create_snapshot(
        repository_url=str(repo), ref="main", output_root=out
    )
    container = out / snapshot.snapshot_id / "repository"
    (container / "stale-flat-file.txt").write_text("junk", encoding="utf-8")

    second, _ = create_snapshot(
        repository_url=str(repo), ref="main", output_root=out
    )
    assert second.snapshot_id == snapshot.snapshot_id
    assert not (container / "stale-flat-file.txt").exists()
    assert {p.name for p in container.iterdir()} == {"sample-fixture-repo"}
    assert (container / "sample-fixture-repo" / "README.md").is_file()


# ---------------------------------------------------------------------------
# repository name extraction (Windows paths must yield a bare name)
# ---------------------------------------------------------------------------


def test_repository_name_from_url_handles_windows_paths():
    # backslash paths become the basename, not the whole path —
    # a full-path name overflows MAX_PATH once used as an index key
    assert repository_name_from_url(r"C:\tmp\sample-repo") == (
        "sample-repo"
    )
    assert repository_name_from_url("C:/tmp/sample-repo/") == (
        "sample-repo"
    )
    assert repository_name_from_url(r"C:\tmp\sample-repo.git") == (
        "sample-repo"
    )
    # POSIX paths, URLs and bare names behave exactly as before
    assert repository_name_from_url("/a/b/sample-repo") == (
        "sample-repo"
    )
    assert repository_name_from_url(
        "https://x/y/sample-repo.git"
    ) == "sample-repo"
    assert repository_name_from_url("sample-repo") == "sample-repo"
    assert repository_name_from_url("") == ""


# ---------------------------------------------------------------------------
# force-removal helpers (Windows read-only Git objects, WinError 5)
# ---------------------------------------------------------------------------


def _read_only_git_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    objects = root / ".git" / "objects" / "aa"
    objects.mkdir(parents=True)
    (objects / "blob").write_bytes(b"git object")
    (objects / "blob").chmod(0o444)
    objects.chmod(0o555)
    return root


@_NOT_ROOT
def test_rmtree_force_removes_read_only_git_objects(tmp_path: Path):
    root = _read_only_git_tree(tmp_path)
    # a plain rmtree refuses on this tree (PermissionError, the
    # WinError 5 path); the force variant must still fully remove it
    with pytest.raises(OSError):
        shutil.rmtree(str(root))
    _rmtree_force(root)
    assert not root.exists()


@_NOT_ROOT
def test_remove_quietly_removes_read_only_artifact(tmp_path: Path):
    # verification-rollback cleanup must not silently leave content:
    # ignore_errors would swallow the PermissionError and keep a
    # misleading container behind
    root = _read_only_git_tree(tmp_path)
    _remove_quietly(root)
    assert not root.exists()


@_NOT_ROOT
def test_rmtree_force_reraises_when_removal_is_impossible(
    tmp_path, monkeypatch
):
    from repo_intelligence.snapshot import builder as builder_mod

    root = _read_only_git_tree(tmp_path)
    # simulate a host where unlocking is itself impossible (ACL or
    # open handle): the retry fails again and the original OSError
    # must surface — never a silent "success"
    monkeypatch.setattr(builder_mod, "_unlock", lambda path: None)
    with pytest.raises(OSError):
        builder_mod._rmtree_force(root)


# ---------------------------------------------------------------------------
# byte-faithful clone (machine autocrlf must not rewrite pinned bytes)
# ---------------------------------------------------------------------------


def test_pinned_snapshot_is_byte_faithful_under_autocrlf(
    tmp_path, monkeypatch
):
    """A machine-wide ``core.autocrlf=true`` must not rewrite pinned
    bytes: the clone forces ``core.autocrlf=false``, so the persisted
    checkout matches the origin blob exactly — on CRLF-translation
    hosts (Windows) as well as LF ones."""
    cfg = tmp_path / "user.gitconfig"
    cfg.write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    origin = tmp_path / "origin"
    origin.mkdir()
    _git(["init", "-q", "-b", "main"], origin)
    _git(["config", "user.email", "t@example.com"], origin)
    _git(["config", "user.name", "T"], origin)
    data = b"# Sample\n\ncontent one\ncontent two\n"
    (origin / "README.md").write_bytes(data)
    _git(["add", "."], origin)
    _git(["commit", "-q", "-m", "init"], origin)

    out = tmp_path / "artifacts"
    snapshot, _artifact = create_snapshot(
        repository_url=str(origin), ref="main", output_root=out
    )
    pinned = repository_dir(out, snapshot.snapshot_id)
    assert (pinned / "README.md").read_bytes() == data


# ---------------------------------------------------------------------------
# legacy flat fallback (existing artifacts + fixtures unchanged)
# ---------------------------------------------------------------------------


def test_legacy_flat_layout_without_snapshot_json(tmp_path: Path):
    run = tmp_path / "artifacts" / "run-1"
    container = run / "repository"
    container.mkdir(parents=True)
    (container / "README.md").write_text("# legacy", encoding="utf-8")
    assert repository_dir(tmp_path / "artifacts", "run-1") == container
    assert repository_dir_for_snapshot(
        run / "snapshot" / "repository_snapshot.json"
    ) == container


def test_legacy_flat_layout_wins_over_prospective_named(tmp_path: Path):
    run = tmp_path / "artifacts" / "run-2"
    _write_snapshot_json(
        run / "snapshot" / "repository_snapshot.json", "sample-x"
    )
    container = run / "repository"
    container.mkdir(parents=True)
    (container / "README.md").write_text("# legacy", encoding="utf-8")
    # JSON knows the name, but the checkout on disk is still flat:
    # existing data wins so old artifacts keep resolving.
    assert repository_dir(tmp_path / "artifacts", "run-2") == container


def test_prospective_named_path_when_checkout_absent(tmp_path: Path):
    run = tmp_path / "artifacts" / "run-3"
    _write_snapshot_json(
        run / "snapshot" / "repository_snapshot.json", "sample-x"
    )
    expected = run / "repository" / "sample-x"
    assert repository_dir(tmp_path / "artifacts", "run-3") == expected
    assert repository_dir_for_snapshot(
        run / "snapshot" / "repository_snapshot.json"
    ) == expected
    assert not expected.exists()  # prospective: readers treat as missing


def test_container_without_name_resolves_to_container(tmp_path: Path):
    run = tmp_path / "artifacts" / "run-4"
    run.mkdir(parents=True)
    (run / "repository_snapshot.json").write_text(
        "{}", encoding="utf-8"
    )
    container = run / "repository"
    assert repository_dir_for_snapshot(
        run / "repository_snapshot.json"
    ) == container


def test_unsafe_repository_name_never_escapes_container(tmp_path: Path):
    out = tmp_path / "artifacts"
    assert repository_dir_for_name(out, "run-5", "../evil") == (
        out / "run-5" / "repository"
    )
    assert repository_dir_for_name(out, "run-5", "a/b") == (
        out / "run-5" / "repository"
    )
    assert repository_dir_for_name(out, "run-5", "  ") == (
        out / "run-5" / "repository"
    )
    run = out / "run-6"
    _write_snapshot_json(
        run / "snapshot" / "repository_snapshot.json", "../evil"
    )
    assert repository_dir(out, "run-6") == run / "repository"


def test_unreadable_snapshot_json_falls_back(tmp_path: Path):
    run = tmp_path / "artifacts" / "run-7"
    (run / "snapshot").mkdir(parents=True)
    (run / "snapshot" / "repository_snapshot.json").write_text(
        "{not json", encoding="utf-8"
    )
    assert repository_dir(tmp_path / "artifacts", "run-7") == (
        run / "repository"
    )


# ---------------------------------------------------------------------------
# create_snapshot cleanup path on verification failure
# ---------------------------------------------------------------------------


def test_failed_verification_leaves_no_container(
    tmp_path: Path, monkeypatch
):
    from repo_intelligence.snapshot import builder as builder_mod

    repo = make_local_repo(tmp_path / "sample-fixture-repo")
    out = tmp_path / "artifacts"
    monkeypatch.setattr(
        builder_mod,
        "_verify_pinned_repository",
        lambda pinned, sha: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )
    with pytest.raises(RuntimeError):
        create_snapshot(
            repository_url=str(repo), ref="main", output_root=out
        )
    for run_dir in out.iterdir():
        assert not (run_dir / "repository").exists()
