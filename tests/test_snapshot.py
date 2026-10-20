"""M1 contract tests.

Uses local temporary Git repositories so tests never depend on GitHub.
One integration-style test exercises the real end-to-end Git workflow
(clone → resolve → snapshot → validate → artifact) via create_snapshot.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from repo_intelligence.artifacts import repository_dir
from repo_intelligence.repository.git import (
    CloneError,
    GitMetadata,
    RefResolutionError,
    checkout_ref,
    clone_repository,
    get_head_sha,
    inspect_workspace,
    is_detached_head,
    resolve_ref,
)
from repo_intelligence.snapshot import (
    RepositorySnapshot,
    SchemaValidationError,
    build_snapshot,
    create_snapshot,
    validate_snapshot_document,
    validate_snapshot_invariants,
)
from repo_intelligence.snapshot.models import sanitize_ref_for_id


# ---------------------------------------------------------------------------
# Local Git fixtures
# ---------------------------------------------------------------------------

def _git(args, cwd):
    subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def make_local_repo(path: Path) -> Path:
    """Create a small local Git repo with a commit and a v1.0.0 tag."""
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test User"], path)
    (path / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(["add", "."], path)
    _git(["commit", "-q", "-m", "initial commit"], path)
    _git(["tag", "v1.0.0"], path)
    return path


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    return make_local_repo(tmp_path / "origin")


def _head_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _short_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# 1. Valid repository + valid ref (local)
# ---------------------------------------------------------------------------

def test_valid_repository_and_ref(local_repo: Path, tmp_path: Path):
    snapshot, artifact = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    assert snapshot.repository.url == str(local_repo)
    assert snapshot.repository.name == "origin"
    assert snapshot.snapshot.ref == "v1.0.0"
    assert artifact.is_file()


# ---------------------------------------------------------------------------
# 2. Invalid ref
# ---------------------------------------------------------------------------

def test_invalid_ref(local_repo: Path, tmp_path: Path):
    with pytest.raises(RefResolutionError):
        create_snapshot(
            repository_url=str(local_repo),
            ref="no-such-tag-or-branch",
            output_root=tmp_path / "snapshots",
        )


def test_invalid_ref_does_not_write_artifact(local_repo: Path, tmp_path: Path):
    out = tmp_path / "snapshots"
    with pytest.raises(RefResolutionError):
        create_snapshot(
            repository_url=str(local_repo),
            ref="does-not-exist",
            output_root=out,
        )
    assert not out.exists() or list(out.iterdir()) == []


# ---------------------------------------------------------------------------
# 3. Invalid repository
# ---------------------------------------------------------------------------

def test_invalid_repository(tmp_path: Path):
    with pytest.raises(CloneError):
        create_snapshot(
            repository_url=str(tmp_path / "nope"),
            ref="main",
            output_root=tmp_path / "snapshots",
        )


# ---------------------------------------------------------------------------
# 4. Commit SHA resolution
# ---------------------------------------------------------------------------

def test_commit_sha_resolution(local_repo: Path, tmp_path: Path):
    expected = _head_sha(local_repo)
    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    assert snapshot.snapshot.commit_sha == expected
    assert len(snapshot.snapshot.commit_sha) == 40
    assert snapshot.snapshot.commit_short_sha == _short_sha(local_repo)
    assert snapshot.snapshot.commit_sha.startswith(snapshot.snapshot.commit_short_sha)


def test_resolve_ref_variants(local_repo: Path, tmp_path: Path):
    dest = tmp_path / "clone"
    clone_repository(str(local_repo), dest)
    expected = _head_sha(local_repo)
    assert resolve_ref(dest, "v1.0.0") == expected
    assert resolve_ref(dest, "main") == expected
    assert resolve_ref(dest, expected) == expected
    assert checkout_ref(dest, "v1.0.0") == expected


# ---------------------------------------------------------------------------
# 5. Working-tree state
# ---------------------------------------------------------------------------

def test_working_tree_clean_on_fresh_clone(local_repo: Path, tmp_path: Path):
    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    assert snapshot.workspace.working_tree_clean is True


def test_inspect_workspace_detects_dirty_tree(local_repo: Path, tmp_path: Path):
    dest = tmp_path / "clone"
    clone_repository(str(local_repo), dest)
    (dest / "dirty.txt").write_text("change", encoding="utf-8")
    meta = inspect_workspace(dest)
    assert meta.working_tree_clean is False


def test_on_dirty_fail_allows_clean_tree(local_repo: Path, tmp_path: Path):
    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
        on_dirty="fail",
    )
    assert snapshot.workspace.working_tree_clean is True


def test_on_dirty_fail_raises_when_metadata_dirty(
    local_repo: Path, tmp_path: Path, monkeypatch
):
    from repo_intelligence.snapshot import builder as builder_mod
    from repo_intelligence.snapshot.builder import SnapshotError

    real_inspect = builder_mod.git_adapter.inspect_workspace

    def dirty_inspect(repo_path):
        meta = real_inspect(repo_path)
        return GitMetadata(
            commit_sha=meta.commit_sha,
            commit_short_sha=meta.commit_short_sha,
            working_tree_clean=False,
        )

    monkeypatch.setattr(builder_mod.git_adapter, "inspect_workspace", dirty_inspect)

    with pytest.raises(SnapshotError):
        create_snapshot(
            repository_url=str(local_repo),
            ref="v1.0.0",
            output_root=tmp_path / "snapshots-fail",
            on_dirty="fail",
        )

    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots-warn",
        on_dirty="warn",
    )
    assert snapshot.workspace.working_tree_clean is False


# ---------------------------------------------------------------------------
# 6. Snapshot identifier generation
# ---------------------------------------------------------------------------

def test_snapshot_id_format(local_repo: Path, tmp_path: Path):
    snapshot, artifact = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    snapshot_id = snapshot.snapshot_id
    date = snapshot.snapshot.captured_at.split("T", 1)[0]
    expected = "v1.0.0_{}_{}".format(date, snapshot.snapshot.commit_short_sha)
    assert snapshot_id == expected
    assert artifact.parent.name == "snapshot"
    assert artifact.parent.parent.name == expected
    assert artifact.name == "repository_snapshot.json"


def test_snapshot_id_for_branch_ref_is_not_release_style():
    snap = build_snapshot(
        repository_url="https://example.com/org/repo.git",
        ref="main",
        metadata=GitMetadata(
            commit_sha="a" * 40,
            commit_short_sha="a" * 7,
            working_tree_clean=True,
        ),
        captured_at="2026-09-22T12:00:00+00:00",
    )
    assert snap.snapshot_id == "main_2026-09-22_aaaaaaa"


def test_sanitize_ref_slashes():
    assert sanitize_ref_for_id("feature/foo") == "feature-foo"


# ---------------------------------------------------------------------------
# 7. JSON schema validation
# ---------------------------------------------------------------------------

def test_schema_valid_document(local_repo: Path, tmp_path: Path):
    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    validate_snapshot_document(snapshot.to_dict())  # should not raise


def test_schema_rejects_bad_sha():
    snap = build_snapshot(
        repository_url="https://example.com/x/y.git",
        ref="main",
        metadata=GitMetadata(
            commit_sha="b" * 40,
            commit_short_sha="bbbbbbb",
            working_tree_clean=True,
        ),
        captured_at="2026-09-22T00:00:00+00:00",
    )
    doc = snap.to_dict()
    doc["snapshot"]["commit_sha"] = "not-a-sha"
    with pytest.raises(SchemaValidationError):
        validate_snapshot_document(doc)


def test_schema_rejects_wrong_schema_version():
    snap = build_snapshot(
        repository_url="https://example.com/x/y.git",
        ref="main",
        metadata=GitMetadata(
            commit_sha="c" * 40,
            commit_short_sha="ccccccc",
            working_tree_clean=True,
        ),
        captured_at="2026-09-22T00:00:00+00:00",
    )
    doc = snap.to_dict()
    doc["schema_version"] = "repository_snapshot.v0"
    with pytest.raises(SchemaValidationError):
        validate_snapshot_document(doc)


def test_invariants_require_fields():
    snap = RepositorySnapshot()
    with pytest.raises(SchemaValidationError):
        validate_snapshot_invariants(snap)


def test_schema_file_exists_and_declares_v1():
    from repo_intelligence.snapshot.builder import SCHEMA_PATH

    assert SCHEMA_PATH.is_file()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "repository_snapshot.v1"


# ---------------------------------------------------------------------------
# 8. Artifact location/content
# ---------------------------------------------------------------------------

def test_artifact_location_and_content(local_repo: Path, tmp_path: Path):
    out = tmp_path / "snapshots"
    snapshot, artifact = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=out,
    )
    assert artifact == out / snapshot.snapshot_id / "snapshot" / "repository_snapshot.json"
    assert artifact.is_file()
    # schema copies live beside the snapshot for the pinned snapshot id
    schema_dir = out / snapshot.snapshot_id / "schemas"
    assert (schema_dir / "repository_snapshot.v1.json").is_file()
    assert (schema_dir / "repository_facts.v2.json").is_file()

    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["schema_version"] == "repository_snapshot.v1"
    assert data["repository"]["url"] == str(local_repo)
    assert data["snapshot"]["ref"] == "v1.0.0"
    assert data["snapshot"]["commit_sha"] == _head_sha(local_repo)
    assert data["snapshot"]["commit_short_sha"] == _short_sha(local_repo)
    assert data["workspace"]["working_tree_clean"] is True
    assert data["analyzer"]["name"] == "repository-intelligence"
    assert data["analyzer"]["version"]
    validate_snapshot_document(data)


def test_pinned_checkout_persisted_beside_snapshot_artifact(
    local_repo: Path, tmp_path: Path
):
    out = tmp_path / "snapshots"
    snapshot, artifact = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=out,
    )
    run_dir = out / snapshot.snapshot_id
    # M18: the checkout lives in a subdir named after the snapshot's
    # repository.name, inside the per-run repository/ container.
    pinned = repository_dir(out, snapshot.snapshot_id)
    assert pinned.parent == run_dir / "repository"
    assert pinned.name == snapshot.repository.name

    # The exact checkout (with Git metadata) is persisted under the run dir.
    assert pinned.is_dir()
    assert (pinned / ".git").exists()
    assert (pinned / "README.md").is_file()
    assert (pinned / "README.md").read_text(encoding="utf-8") == (
        "# fixture\n"
    )

    # The snapshot JSON dir itself still holds only the JSON — the clone
    # is never stored inside the snapshot/ artifact folder.
    entries = {p.name for p in artifact.parent.iterdir()}
    assert entries == {"repository_snapshot.json"}
    assert not (artifact.parent / ".git").exists()


def test_persisted_head_equals_recorded_commit_and_is_detached(
    local_repo: Path, tmp_path: Path
):
    out = tmp_path / "snapshots"
    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=out,
    )
    pinned = repository_dir(out, snapshot.snapshot_id)
    assert get_head_sha(pinned) == snapshot.snapshot.commit_sha
    assert snapshot.snapshot.commit_sha == _head_sha(local_repo)
    assert is_detached_head(pinned) is True


def test_persisted_head_mismatch_fails_and_leaves_no_artifact(
    local_repo: Path, tmp_path: Path, monkeypatch
):
    from repo_intelligence.snapshot import builder as builder_mod
    from repo_intelligence.snapshot.builder import SnapshotError

    out = tmp_path / "snapshots"
    real_get_head_sha = builder_mod.git_adapter.get_head_sha

    def corrupt_persisted_head(repo_path):
        # Simulate a corrupted persisted checkout: only the persisted
        # copy reports the wrong HEAD; the temporary clone still works.
        if str(Path(repo_path)).startswith(str(out)):
            return "0" * 40
        return real_get_head_sha(repo_path)

    monkeypatch.setattr(
        builder_mod.git_adapter, "get_head_sha", corrupt_persisted_head
    )

    with pytest.raises(SnapshotError, match="does not match"):
        create_snapshot(
            repository_url=str(local_repo),
            ref="v1.0.0",
            output_root=out,
        )

    # No misleading artifact is left behind.
    run_dirs = [p for p in out.iterdir()] if out.exists() else []
    for run_dir in run_dirs:
        assert not (run_dir / "snapshot" / "repository_snapshot.json").exists()
        assert not (run_dir / "repository").exists()


def test_persisted_repository_overwritten_deterministically_on_rerun(
    local_repo: Path, tmp_path: Path
):
    out = tmp_path / "snapshots"
    first, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=out,
    )
    second, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=out,
    )
    # same repo + ref + same day → same snapshot id; the persisted
    # checkout is replaced in place and stays verified.
    assert first.snapshot_id == second.snapshot_id
    pinned = repository_dir(out, second.snapshot_id)
    assert get_head_sha(pinned) == second.snapshot.commit_sha
    assert is_detached_head(pinned) is True


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip(local_repo: Path, tmp_path: Path):
    snapshot, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    restored = RepositorySnapshot.from_dict(snapshot.to_dict())
    assert restored.to_dict() == snapshot.to_dict()


# ---------------------------------------------------------------------------
# Reproducibility: same immutable commit → same identity
# ---------------------------------------------------------------------------

def test_reproducibility_same_identity(local_repo: Path, tmp_path: Path):
    s1, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "out1",
    )
    s2, _ = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=tmp_path / "out2",
    )
    assert s1.snapshot.commit_sha == s2.snapshot.commit_sha
    assert s1.repository.url == s2.repository.url
    assert s1.snapshot.ref == s2.snapshot.ref
    # captured_at may differ; identity fields above must match
    assert s1.snapshot_id == s2.snapshot_id


# ---------------------------------------------------------------------------
# Integration-style: full Git workflow through create_snapshot
# ---------------------------------------------------------------------------

def test_integration_full_workflow(local_repo: Path, tmp_path: Path):
    """End-to-end: clone local remote, resolve tag, produce validated artifact."""
    out = tmp_path / "snapshots"
    snapshot, artifact = create_snapshot(
        repository_url=str(local_repo),
        ref="v1.0.0",
        output_root=out,
    )
    data = json.loads(artifact.read_text(encoding="utf-8"))
    validate_snapshot_document(data)
    assert data["snapshot"]["commit_sha"] == _head_sha(local_repo)
    assert data["workspace"]["working_tree_clean"] is True
    assert artifact.is_relative_to(out) if hasattr(artifact, "is_relative_to") else str(
        artifact
    ).startswith(str(out))


def test_build_snapshot_from_metadata_only():
    meta = GitMetadata(
        commit_sha="d" * 40,
        commit_short_sha="ddddddd",
        working_tree_clean=False,
    )
    snap = build_snapshot(
        repository_url="https://github.com/aws-samples/"
        "sample-Agentic-Ai-Data-Operations.git",
        ref="v1.0.0",
        metadata=meta,
        captured_at="2026-09-22T08:00:00+00:00",
    )
    assert snap.repository.name == "sample-Agentic-Ai-Data-Operations"
    assert snap.workspace.working_tree_clean is False
    validate_snapshot_document(snap.to_dict())
