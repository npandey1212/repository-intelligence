"""Build, validate, and persist RepositorySnapshot artifacts.

Pipeline:

    Git operations → Git metadata → RepositorySnapshot
        → schema validation → persist exact checkout
          (artifacts/<snapshot_id>/repository/<name>/, verified detached HEAD)
        → JSON under artifacts/<snapshot_id>/snapshot/
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft202012Validator

from .. import __version__
from ..artifacts import (
    ARTIFACTS_ROOT,
    REPOSITORY_DIR_NAME,
    copy_schema_artifacts,
    repository_dir_for_name,
    resolve_output_root,
    snapshot_path as artifact_snapshot_path,
)
from ..repository import git as git_adapter
from ..repository.git import GitError, GitMetadata
from .models import (
    AnalyzerInfo,
    RepositoryInfo,
    RepositorySnapshot,
    SnapshotInfo,
    WorkspaceInfo,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "repository_snapshot.v1.json"
)
# Absolute default: <project-root>/artifacts (not CWD-relative).
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class SnapshotError(Exception):
    """Raised when a snapshot cannot be built or is invalid."""


class SchemaValidationError(SnapshotError):
    """Raised when a snapshot fails JSON Schema or invariant checks."""


def repository_name_from_url(repository_url: str) -> str:
    """Derive a repository name from a URL or local path.

    Windows separators are normalized first so a local Windows path
    (``C:\\tmp\\sample-repo``) yields the bare name — previously the
    whole path became the name, which overflows MAX_PATH once it is
    used as an on-disk key.
    """
    cleaned = repository_url.replace("\\", "/").rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    name = cleaned.rsplit("/", 1)[-1]
    return name or repository_url


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(schema_path) if schema_path else SCHEMA_PATH
    if not path.is_file():
        raise SnapshotError("Schema not found: {}".format(path))
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_snapshot_document(
    document: Dict[str, Any], schema_path: Optional[Path] = None
) -> None:
    """Validate *document* against the repository_snapshot.v1 schema."""
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    if errors:
        messages = [
            "{}: {}".format(
                "/".join(str(p) for p in error.path) or "<root>",
                error.message,
            )
            for error in errors
        ]
        raise SchemaValidationError(
            "Snapshot failed schema validation:\n  - "
            + "\n  - ".join(messages)
        )


def validate_snapshot_invariants(snapshot: RepositorySnapshot) -> None:
    """Check business rules that the schema alone cannot express."""
    problems = []

    if not _SHA_RE.match(snapshot.snapshot.commit_sha or ""):
        problems.append(
            "commit_sha must be a 40-character lowercase hex SHA, got {!r}".format(
                snapshot.snapshot.commit_sha
            )
        )
    if not snapshot.snapshot.commit_short_sha:
        problems.append("commit_short_sha is missing")
    if not snapshot.repository.url:
        problems.append("repository.url is missing")
    if not snapshot.repository.name:
        problems.append("repository.name is missing")
    if not snapshot.snapshot.ref:
        problems.append("snapshot.ref is missing")
    if not snapshot.analyzer.name or not snapshot.analyzer.version:
        problems.append("analyzer name/version is missing")
    if not isinstance(snapshot.workspace.working_tree_clean, bool):
        problems.append("workspace.working_tree_clean must be a boolean")

    if snapshot.snapshot.captured_at:
        try:
            datetime.fromisoformat(
                snapshot.snapshot.captured_at.replace("Z", "+00:00")
            )
        except ValueError:
            problems.append(
                "captured_at is not a valid ISO-8601 timestamp: {!r}".format(
                    snapshot.snapshot.captured_at
                )
            )
    else:
        problems.append("captured_at is missing")

    if snapshot.snapshot.commit_short_sha and snapshot.snapshot.commit_sha:
        if not snapshot.snapshot.commit_sha.startswith(
            snapshot.snapshot.commit_short_sha
        ):
            problems.append(
                "commit_short_sha {!r} is not a prefix of commit_sha {!r}".format(
                    snapshot.snapshot.commit_short_sha,
                    snapshot.snapshot.commit_sha,
                )
            )

    if problems:
        raise SchemaValidationError(
            "Snapshot failed invariant checks:\n  - " + "\n  - ".join(problems)
        )


def build_snapshot(
    repository_url: str,
    ref: str,
    metadata: GitMetadata,
    captured_at: Optional[str] = None,
) -> RepositorySnapshot:
    """Construct an in-memory RepositorySnapshot from Git metadata."""
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).isoformat()

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(
            name=repository_name_from_url(repository_url),
            url=repository_url,
        ),
        snapshot=SnapshotInfo(
            ref=ref,
            commit_sha=metadata.commit_sha,
            commit_short_sha=metadata.commit_short_sha,
            captured_at=captured_at,
        ),
        workspace=WorkspaceInfo(
            working_tree_clean=metadata.working_tree_clean
        ),
        analyzer=AnalyzerInfo(name="repository-intelligence", version=__version__),
    )
    validate_snapshot_invariants(snapshot)
    validate_snapshot_document(snapshot.to_dict())
    return snapshot


def write_snapshot_artifact(
    snapshot: RepositorySnapshot,
    output_root: Optional[Path] = None,
) -> Path:
    """Write the snapshot JSON under ``<output_root>/<snapshot_id>/snapshot/``.

    When *output_root* is None, uses ``<project-root>/artifacts`` (not CWD).
    """
    out_root = resolve_output_root(output_root)
    artifact_path = artifact_snapshot_path(out_root, snapshot.snapshot_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as fh:
        json.dump(snapshot.to_dict(), fh, indent=2)
        fh.write("\n")
    copy_schema_artifacts(out_root, snapshot.snapshot_id)
    return artifact_path


def _unlock(path: Path) -> None:
    """Add owner read/write bits so a refusal to delete can be retried.

    Windows raises PermissionError (WinError 5) when deleting a
    read-only file (Git object files are 0444) or an entry inside a
    read-only directory; chmod with the existing mode plus owner
    read/write is the portable unlock. Failures to unlock are
    ignored — the retry below then surfaces the original OSError.
    """
    try:
        mode = os.stat(str(path)).st_mode
        os.chmod(str(path), mode | stat.S_IWUSR | stat.S_IREAD)
    except OSError:
        pass


def _unlink_force(target: Path) -> None:
    """Unlink once; on refusal, unlock the path and retry once."""
    try:
        target.unlink()
    except OSError:
        _unlock(target)
        target.unlink()


def _rmtree_force(target: Path) -> None:
    """``shutil.rmtree`` that clears read-only trees, not silently.

    Unlike ``ignore_errors=True``, a path that still cannot be
    removed after the unlock-and-retry re-raises, so callers never
    treat a failed rollback as success (a leftover container would be
    a misleading artifact).
    """
    def onerror(func, path, exc_info):
        exc = exc_info[1]
        if not isinstance(exc, OSError):
            raise exc
        failing = Path(path)
        _unlock(failing)
        _unlock(failing.parent)
        func(path)  # retry — a second refusal propagates

    shutil.rmtree(str(target), onerror=onerror)


def _persist_pinned_repository(
    workspace: Path, out_root: Path, snapshot: RepositorySnapshot
) -> Path:
    """Copy the exact checkout to ``<output>/<id>/repository/<name>/``.

    The subdirectory is the snapshot's own ``repository.name`` (M18 —
    one pinned checkout per repository identity; never a hard-coded
    project name). The whole per-snapshot ``repository/`` container is
    replaced so repeated runs also clear any pre-M18 flat layout and
    stay deterministic. The copy preserves Git metadata (including the
    detached HEAD set by ``checkout_ref``) so the persisted checkout
    stays pinned to the recorded commit.
    """
    container = (
        out_root / snapshot.snapshot_id / REPOSITORY_DIR_NAME
    )
    target = repository_dir_for_name(
        out_root, snapshot.snapshot_id, snapshot.repository.name
    )
    if container.exists():
        _rmtree_force(container)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(workspace), str(target), symlinks=True)
    return target


def _verify_pinned_repository(pinned: Path, expected_sha: str) -> None:
    """Verify persisted HEAD equality and detached state of *pinned*.

    Raises SnapshotError when the invariant cannot be established; the
    caller must then fail the operation rather than keep a misleading
    artifact.
    """
    actual = git_adapter.get_head_sha(pinned)
    if actual != expected_sha:
        raise SnapshotError(
            "Persisted repository HEAD {} does not match snapshot commit {} "
            "({}); refusing to keep a misleading artifact.".format(
                actual, expected_sha, pinned
            )
        )
    if not git_adapter.is_detached_head(pinned):
        raise SnapshotError(
            "Persisted repository at {} is not detached at the pinned "
            "commit; refusing to keep a mutable checkout.".format(pinned)
        )


def _remove_quietly(path: Optional[Path]) -> None:
    """Best-effort removal of a partially written artifact path.

    Best-effort, not blind: read-only trees are unlocked and retried
    first (``_rmtree_force``), so a Windows rollback after a failed
    verification actually empties the container instead of silently
    skipping it. Only a genuine second refusal is swallowed.
    """
    if path is None:
        return
    target = Path(path)
    try:
        if target.is_dir():
            _rmtree_force(target)
        elif target.exists():
            _unlink_force(target)
    except OSError:
        pass


def _remove_empty_repository_container(pinned: Path) -> None:
    """Drop the per-run ``repository/`` container left empty by a rollback.

    The named checkout may be gone after ``_remove_quietly``; an empty
    container would linger as a misleading artifact directory, so it is
    removed — only when empty, never with content.
    """
    container = Path(pinned).parent
    if container.name != REPOSITORY_DIR_NAME or not container.is_dir():
        return
    try:
        if not any(container.iterdir()):
            container.rmdir()
    except OSError:
        pass


def create_snapshot(
    repository_url: str,
    ref: str,
    output_root: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
    on_dirty: str = "warn",
) -> Tuple[RepositorySnapshot, Path]:
    """Full M1 pipeline: clone → resolve ref → snapshot → persist → write.

    *output_root* None → ``<project-root>/artifacts``; otherwise the explicit
    override is used exactly as given.

    *on_dirty* controls behavior when the working tree is not clean:

    - ``"warn"``  (default): record ``working_tree_clean: false`` and continue.
    - ``"fail"``: raise SnapshotError.
    - ``"ignore"``: record the state but emit no warning (still recorded).

    The exact checkout (detached at the recorded commit) is persisted to
    ``<output>/<snapshot_id>/repository/<name>/`` and verified against
    ``snapshot.commit_sha`` before the temporary clone is removed. If any
    step fails, the operation raises and no misleading snapshot artifact
    or persisted checkout is left behind.
    """
    out_root = resolve_output_root(output_root)
    if on_dirty not in ("warn", "fail", "ignore"):
        raise ValueError(
            "on_dirty must be one of 'warn', 'fail', 'ignore'; got {!r}".format(
                on_dirty
            )
        )

    git_adapter.ensure_git_available()

    tmp_base = Path(workspace_root) if workspace_root else None
    with tempfile.TemporaryDirectory(
        prefix="repo-intelligence-",
        dir=str(tmp_base) if tmp_base else None,
    ) as tmp:
        workspace = Path(tmp) / "repository"
        try:
            git_adapter.clone_repository(repository_url, workspace)
        except GitError:
            raise

        try:
            git_adapter.checkout_ref(workspace, ref)
        except GitError:
            raise

        metadata = git_adapter.inspect_workspace(workspace)

        if not metadata.working_tree_clean:
            if on_dirty == "fail":
                raise SnapshotError(
                    "Working tree is dirty after checkout of ref {!r}; "
                    "refusing to create snapshot (on_dirty='fail').".format(ref)
                )
            if on_dirty == "warn":
                import sys

                print(
                    "warning: working tree is not clean for ref {!r}; "
                    "snapshot will record working_tree_clean=false".format(ref),
                    file=sys.stderr,
                )

        snapshot = build_snapshot(repository_url, ref, metadata)
        pinned = _persist_pinned_repository(workspace, out_root, snapshot)
        artifact_path: Optional[Path] = None
        try:
            artifact_path = write_snapshot_artifact(snapshot, out_root)
            _verify_pinned_repository(pinned, snapshot.snapshot.commit_sha)
        except Exception:
            _remove_quietly(pinned)
            _remove_quietly(artifact_path)
            _remove_empty_repository_container(pinned)
            raise

    assert artifact_path is not None
    return snapshot, artifact_path
