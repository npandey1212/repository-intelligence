"""Which index belongs to which checkout (M18 §21).

Identity — ``(repository_name, commit_sha)`` — comes from the M1
snapshot record only, never from guessing at directory names or Git
state:

* **Beside the checkout** (structural): the pinned checkout lives at
  ``<run>/repository/<name>`` (or the legacy flat
  ``<run>/repository``), and the record sits at
  ``<run>/snapshot/repository_snapshot.json`` — one hop away for
  either layout.
* **From the run context**: when an explicit checkout override is in
  play, the current run's record is used *only if* that run's
  resolved checkout is literally the same directory — a checkout
  outside the run never inherits another repository's identity.
* **Unpinned explicit subject** (M17 ``--repository <dir>``): a
  directory that is not a run's pinned checkout has no record *by
  design* (``pinned_commit`` stays empty). Its identity is the
  directory name plus an ``unpinned:`` digest of its absolute path —
  scoped so two different unpinned checkouts never share an index.
  Pinned runs are never affected: anything inside a run's
  ``repository/`` container without a readable record is a broken
  pin and fails loudly instead.

Resolution failing means ``RAG_IDENTITY_NOT_RESOLVED`` with an
actionable message; it is never satisfied with a partial guess.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..artifacts import (
    REPOSITORY_DIR_NAME,
    repository_dir,
    resolve_output_root,
    snapshot_path,
)
from ..context.models import ArchitectureContextError, discover_snapshot_id
from .embeddings import DEFAULT_EMBEDDING_IDENTITY
from .errors import RAG_IDENTITY_NOT_RESOLVED, RetrievalError
from .models import IndexIdentity

__all__ = [
    "identity_for_snapshot",
    "resolve_index_identity",
]


def _load_record(path: Path) -> Optional[Dict[str, str]]:
    """``{name, url, commit, snapshot_id}`` from a snapshot JSON."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    repository = data.get("repository")
    snapshot = data.get("snapshot")
    if not isinstance(repository, dict) or not isinstance(snapshot, dict):
        return None
    name = repository.get("name")
    commit = snapshot.get("commit_sha")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(commit, str) or not commit:
        return None
    url = repository.get("url")
    # <output>/<run_id>/snapshot/repository_snapshot.json
    run_id = path.parent.parent.name
    return {
        "name": name,
        "url": url if isinstance(url, str) else "",
        "commit": commit,
        "snapshot_id": run_id,
    }


def _record_beside(root: Path) -> Optional[Dict[str, str]]:
    """Structural lookup: the record that pinned *root*."""
    if root.parent.name == REPOSITORY_DIR_NAME:
        # M18 named layout: <run>/repository/<name>
        run_dir = root.parent.parent
        named = True
    elif root.name == REPOSITORY_DIR_NAME:
        # legacy flat layout: <run>/repository
        run_dir = root.parent
        named = False
    else:
        return None
    path = snapshot_path(run_dir.parent, run_dir.name)
    if not path.is_file():
        return None
    record = _load_record(path)
    if record is None:
        return None
    if named and record["name"] != root.name:
        # the record describes a different checkout — foreign directory
        return None
    return record


def _record_from_run_context(
    root: Path, output_root: Optional[Path]
) -> Optional[Dict[str, str]]:
    """The current run's record, but only when it pinned *root* itself."""
    root_out = resolve_output_root(output_root)
    try:
        run_id = discover_snapshot_id(root_out)
    except ArchitectureContextError:
        return None
    candidate = repository_dir(root_out, run_id)
    try:
        if not candidate.is_dir():
            return None
        if candidate.resolve() != root:
            return None
    except OSError:
        return None
    return _load_record(snapshot_path(root_out, run_id))


def resolve_index_identity(
    repository_root: Path,
    output_root: Optional[Path] = None,
    embedding_model: Optional[str] = None,
) -> IndexIdentity:
    """Identity of the checkout at *repository_root* (or raise)."""
    root = Path(repository_root).resolve()
    model = embedding_model or DEFAULT_EMBEDDING_IDENTITY
    record = _record_beside(root)
    if record is None:
        record = _record_from_run_context(root, output_root)
    if record is None:
        inside_container = (
            root.name == REPOSITORY_DIR_NAME
            or root.parent.name == REPOSITORY_DIR_NAME
        )
        if not inside_container:
            # Unpinned explicit subject (M17 ``--repository <dir>``):
            # no snapshot record exists by design. Scope the index to
            # THIS directory via a path digest so two different
            # unpinned checkouts can never collide.
            digest = hashlib.sha1(
                str(root).encode("utf-8")
            ).hexdigest()[:12]
            return IndexIdentity(
                repository_name=root.name,
                commit_sha="unpinned:{}".format(digest),
                embedding_model=model,
                repository_url="",
                snapshot_id="",
            )
        raise RetrievalError(
            RAG_IDENTITY_NOT_RESOLVED,
            "cannot determine repository identity for checkout {}: no "
            "repository_snapshot.json record was found beside it or in "
            "the current run. Build the index after "
            "'repo-intelligence snapshot' has pinned the checkout "
            "(expected record at <run>/snapshot/"
            "repository_snapshot.json).".format(root),
            details={"repository_root": str(root)},
        )
    return IndexIdentity(
        repository_name=record["name"],
        commit_sha=record["commit"],
        embedding_model=model,
        repository_url=record["url"],
        snapshot_id=record["snapshot_id"],
    )


def identity_for_snapshot(
    output_root: Optional[Path],
    snapshot_id: str,
    embedding_model: Optional[str] = None,
) -> IndexIdentity:
    """Identity of an existing snapshot artifact (writer-side path)."""
    root_out = resolve_output_root(output_root)
    record = _load_record(snapshot_path(root_out, snapshot_id))
    if record is None:
        raise RetrievalError(
            RAG_IDENTITY_NOT_RESOLVED,
            "no readable snapshot record for run {!r} under {}. Run "
            "'repo-intelligence snapshot' first.".format(
                snapshot_id, root_out
            ),
            details={"snapshot_id": snapshot_id},
        )
    return IndexIdentity(
        repository_name=record["name"],
        commit_sha=record["commit"],
        embedding_model=(
            embedding_model
            if embedding_model
            else DEFAULT_EMBEDDING_IDENTITY
        ),
        repository_url=record["url"],
        snapshot_id=record["snapshot_id"],
    )
