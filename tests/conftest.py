"""Shared M18 test helpers (retrieval-aware fixture utilities).

Plain functions, importable as ``from conftest import enable_search``
because pytest prepends this directory to ``sys.path`` (no
``tests/__init__.py`` — same mechanism the per-directory doubles use).

Two building blocks:

``install_pinned_run``   write a full run layout (facts + architecture
                         + snapshot record + NAMED checkout at
                         ``<output>/<run>/repository/<name>/``) with
                         arbitrary fixture files — the dual-fixture
                         shape of spec §44.

``enable_search``        point ``REPO_INTELLIGENCE_DB_STORE`` at a
                         per-test tmp directory and build the
                         deterministic hashing index for that
                         checkout's identity, so ``search_repository``
                         works offline with no model download and no
                         writes to the project's real db_store/.

Determinism: tests use ``hashing://v1/128`` (SHA-1 bucketing, no
network, no model). SentenceTransformers-backed behavior is covered
by the few explicitly model-backed tests that importorskip.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import pytest

from repo_intelligence.retrieval import (
    build_index,
    embedder_for,
    resolve_index_identity,
)
from repo_intelligence.retrieval.models import BuildReport
from repo_intelligence.storage import DB_STORE_ENV_VAR

HASHING_MODEL = "hashing://v1/128"
DEFAULT_RUN_ID = "run_fixture_alpha"
DEFAULT_COMMIT = "1" * 40
DEFAULT_NAME = "fixture-repository"
DEFAULT_URL = "https://example.invalid/fixture.git"

__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_NAME",
    "DEFAULT_RUN_ID",
    "DEFAULT_URL",
    "HASHING_MODEL",
    "build_hashing_index",
    "enable_search",
    "install_pinned_run",
    "symlink_or_skip",
    "try_symlink",
]


def try_symlink(target, link) -> bool:
    """Best-effort symlink: *link* -> *target*; True when it exists.

    Windows hosts without ``SeCreateSymbolicLinkPrivilege`` reject
    ``os.symlink`` (WinError 1314). Shared fixtures call this so a
    missing privilege degrades to "no link" instead of failing every
    test that touches the fixture. *link* is replaced when present.
    """
    link = Path(link)
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        return False
    return True


def symlink_or_skip(target, link) -> None:
    """Create a symlink or ``pytest.skip`` — for symlink-subject tests.

    Never fabricates a substitute: a host that cannot create links
    skips the case (with the reason) rather than passing vacuously,
    so the security contract stays fully asserted wherever links are
    available.
    """
    if not try_symlink(target, link):
        pytest.skip(
            "symlinks are not permitted on this host (Windows needs "
            "SeCreateSymbolicLinkPrivilege or Developer Mode)"
        )


def install_pinned_run(
    output_root: Path,
    files: Dict[str, str],
    run_id: str = DEFAULT_RUN_ID,
    repository_name: str = DEFAULT_NAME,
    commit_sha: str = DEFAULT_COMMIT,
    url: str = DEFAULT_URL,
    ref: str = "main",
) -> Path:
    """Create ``<output_root>/<run_id>/`` with the full M1 layout.

    Writes the facts/architecture markers, the snapshot record, and
    the pinned checkout at ``repository/<repository_name>/`` (M18
    layout); returns the checkout path. Does NOT build an index —
    pair with :func:`enable_search`.
    """
    run_dir = Path(output_root) / run_id
    (run_dir / "facts").mkdir(parents=True, exist_ok=True)
    (run_dir / "architecture").mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshot").mkdir(parents=True, exist_ok=True)
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
                "repository": {"name": repository_name, "url": url},
                "snapshot": {"ref": ref, "commit_sha": commit_sha},
            }
        ),
        encoding="utf-8",
    )
    checkout = run_dir / "repository" / repository_name
    checkout.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # exact bytes: Path.write_text would translate "\n" to
        # os.linesep on Windows and break byte-faithful read-back
        path.write_bytes(content.encode("utf-8"))
    return checkout


def build_hashing_index(
    checkout: Path,
    identity_repository_name: Optional[str] = None,
    commit_sha: Optional[str] = None,
    store_root: Optional[Path] = None,
    run_id: str = "",
    dimension: int = 128,
) -> BuildReport:
    """Low-level: build a hashing index for an explicit identity.

    Used when a test builds an index into an explicit store root
    (no environment mutation); most tests want :func:`enable_search`.
    """
    from repo_intelligence.retrieval.models import IndexIdentity

    name = identity_repository_name or Path(checkout).name
    identity = IndexIdentity(
        repository_name=name,
        commit_sha=commit_sha or DEFAULT_COMMIT,
        embedding_model=HASHING_MODEL,
        snapshot_id=run_id,
    )
    return build_index(
        Path(checkout),
        identity,
        store_root=Path(store_root) if store_root else None,
        embedder=embedder_for(
            "hashing://v1/{}".format(dimension)
        ),
    )


def enable_search(
    checkout: Path,
    monkeypatch,
    tmp_path: Path,
    model: str = HASHING_MODEL,
) -> BuildReport:
    """Make ``search_repository`` work for *checkout* in this test.

    Isolates the whole db_store under *tmp_path* (never the project's
    real ``db_store/``), resolves the checkout's identity (pinned
    record or unpinned subject), and builds the hashing index into
    that isolated store. Queries later resolve the same store through
    the same environment variable and adopt this single index.
    """
    monkeypatch.setenv(DB_STORE_ENV_VAR, str(Path(tmp_path) / "db_store"))
    identity = resolve_index_identity(Path(checkout), None)
    identity = replace(identity, embedding_model=model)
    return build_index(
        Path(checkout),
        identity,
        store_root=None,
        embedder=embedder_for(model),
    )
