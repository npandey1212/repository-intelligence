"""Centralized durable-storage layout under ``db_store/`` (M18).

One home for every path that names persistent local databases and
indexes, so no module constructs ``PROJECT_ROOT / "db_store" / ...``
itself. The three top-level trees have distinct ownership
(authoritative: ``docs/storage-layout.md``):

    artifacts/   durable repository-analysis outputs (M1–M6 facts,
                 engineering model/graph, architecture, diagrams, the
                 pinned repository snapshot)
    db_store/    durable local databases and indexes
                   memory/   agent/application memory (M14) and the
                             repository-query memory database
                   rag/      repository retrieval indexes (M18):
                             metadata.db + vector_store/<key>/...
    runtime/     ephemeral execution state and telemetry (traces,
                 context snapshots, run scratch) — never a durable
                 database, never a retrieval index

Every helper is absolute and project-root-relative (CWD-independent),
matching the ``default_artifacts_root()`` convention in ``artifacts``.
The whole tree can be relocated — for tests and for installations that
keep state outside the source tree — with ONE environment variable:

    REPO_INTELLIGENCE_DB_STORE=<absolute path>

Migration of the legacy ``runtime/agent_memory/`` databases is
explicit, deterministic, and never destructive
(``migrate_legacy_memory_databases`` / ``python -m
repo_intelligence.storage``): a legacy file is moved only when no
database of that name already exists at the new location.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import PROJECT_ROOT

__all__ = [
    "DB_STORE_ENV_VAR",
    "DB_STORE_DIR_NAME",
    "MEMORY_DIR_NAME",
    "RAG_DIR_NAME",
    "VECTOR_STORE_DIR_NAME",
    "MEMORY_DB_FILENAME",
    "QUERY_DB_FILENAME",
    "RAG_METADATA_DB_FILENAME",
    "StorageMigrationError",
    "default_db_store_root",
    "default_memory_store_root",
    "default_rag_store_root",
    "default_memory_db_path",
    "default_query_db_path",
    "default_rag_metadata_db_path",
    "rag_vector_store_path",
    "migrate_legacy_memory_databases",
    "main",
]

# Layout constants (single source of truth).
DB_STORE_ENV_VAR = "REPO_INTELLIGENCE_DB_STORE"
DB_STORE_DIR_NAME = "db_store"
MEMORY_DIR_NAME = "memory"
RAG_DIR_NAME = "rag"
VECTOR_STORE_DIR_NAME = "vector_store"
MEMORY_DB_FILENAME = "agent_memory.db"
QUERY_DB_FILENAME = "repository_query.db"
RAG_METADATA_DB_FILENAME = "metadata.db"

# Legacy location of the persistent memory databases (M14 era).
LEGACY_MEMORY_DIR_NAME = "runtime"
LEGACY_MEMORY_SUBDIR = "agent_memory"
MIGRATABLE_DB_FILENAMES = (MEMORY_DB_FILENAME, QUERY_DB_FILENAME)


class StorageMigrationError(Exception):
    """Deterministic migration refusal — never destroys or duplicates."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


def default_db_store_root() -> Path:
    """Durable-database root: ``<project-root>/db_store``.

    Absolute and CWD-independent. The ``REPO_INTELLIGENCE_DB_STORE``
    environment variable (when set to a non-empty value) replaces the
    default for the whole tree — the one relocation seam used by tests
    and by installations that keep state outside the source tree.
    """
    override = os.environ.get(DB_STORE_ENV_VAR)
    if override:
        return Path(override)
    return PROJECT_ROOT / DB_STORE_DIR_NAME


def default_memory_store_root() -> Path:
    """Persistent memory databases: ``<db_store>/memory``."""
    return default_db_store_root() / MEMORY_DIR_NAME


def default_rag_store_root() -> Path:
    """Persistent retrieval indexes: ``<db_store>/rag``."""
    return default_db_store_root() / RAG_DIR_NAME


def default_memory_db_path() -> Path:
    """Default agent-memory database: ``<db_store>/memory/agent_memory.db``.

    This replaces the legacy ``runtime/agent_memory/agent_memory.db``
    default (storage-location migration only — the M14 schema,
    semantics, and lifecycle are unchanged).
    """
    return default_memory_store_root() / MEMORY_DB_FILENAME


def default_query_db_path() -> Path:
    """Repository-query memory DB: ``<db_store>/memory/`` + query file."""
    return default_memory_store_root() / QUERY_DB_FILENAME


def default_rag_metadata_db_path() -> Path:
    """RAG chunk-metadata database: ``<db_store>/rag/metadata.db``."""
    return default_rag_store_root() / RAG_METADATA_DB_FILENAME


def rag_vector_store_path(
    repository_key: str,
    commit_sha: str,
    embedding_key: Optional[str] = None,
    store_root: Optional[Path] = None,
) -> Path:
    """One index dir: ``<db_store>/rag/vector_store/<repo>/<commit>[/<emb>]``.

    ``repository_key`` and ``embedding_key`` are sanitized, stable
    directory tokens (see ``retrieval.models.store_key``); the
    embedding-model segment keeps two indexes for the same commit built
    with different models from ever sharing files.
    """
    if store_root is not None:
        root = Path(store_root)
    else:
        root = default_rag_store_root()
    path = root / VECTOR_STORE_DIR_NAME / repository_key / commit_sha
    if embedding_key:
        path = path / embedding_key
    return path


def legacy_memory_dir() -> Path:
    """Pre-M18 location: ``<project-root>/runtime/agent_memory``."""
    return PROJECT_ROOT / LEGACY_MEMORY_DIR_NAME / LEGACY_MEMORY_SUBDIR


def migrate_legacy_memory_databases(
    target_root: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Move legacy ``runtime/agent_memory/*.db`` databases to db_store.

    Deterministic and non-destructive:

    * a legacy database with no counterpart at the destination is
      **moved** (parents created);
    * a legacy database whose counterpart already exists is an error
      (``StorageMigrationError``, code ``MIGRATION_TARGET_EXISTS``) —
      the caller sees both locations instead of a silent merge, and
      nothing is deleted;
    * a legacy file that is absent is simply reported as skipped.

    Returns ``{"moved": [...], "skipped": [...]}`` (file names).
    """
    destination = (
        Path(target_root)
        if target_root is not None
        else default_memory_store_root()
    )
    legacy_dir = legacy_memory_dir()
    moved: List[str] = []
    skipped: List[str] = []
    for filename in MIGRATABLE_DB_FILENAMES:
        legacy = legacy_dir / filename
        if not legacy.is_file():
            skipped.append(filename)
            continue
        target = destination / filename
        if target.exists():
            raise StorageMigrationError(
                "MIGRATION_TARGET_EXISTS",
                "refusing to migrate {}: {} already exists. Inspect both "
                "databases and remove or rename the one you do not need; "
                "migration never overwrites or deletes data.".format(
                    legacy, target
                ),
            )
        destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(target))
        moved.append(filename)
    return {"moved": moved, "skipped": skipped}


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m repo_intelligence.storage`` — migrate legacy DBs."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.storage",
        description=(
            "Move legacy runtime/agent_memory/ databases into the "
            "centralized db_store/memory/ location (never overwrites, "
            "never deletes)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which databases WOULD move, without moving anything",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        legacy_dir = legacy_memory_dir()
        destination = default_memory_store_root()
        plan = {
            "legacy_dir": str(legacy_dir),
            "destination": str(destination),
            "would_move": [
                name
                for name in MIGRATABLE_DB_FILENAMES
                if (legacy_dir / name).is_file()
                and not (destination / name).exists()
            ],
            "would_refuse": [
                name
                for name in MIGRATABLE_DB_FILENAMES
                if (legacy_dir / name).is_file()
                and (destination / name).exists()
            ],
            "already_migrated": [
                name
                for name in MIGRATABLE_DB_FILENAMES
                if (destination / name).exists()
            ],
        }
        print(json.dumps(plan, indent=2))
        return 0

    try:
        report = migrate_legacy_memory_databases()
    except StorageMigrationError as exc:
        print(
            "storage migration refused [{}]: {}".format(exc.code, exc.message),
            file=sys.stderr,
        )
        return 1
    print(
        "memory databases migrated into {}: moved={}, skipped={}".format(
            default_memory_store_root(),
            ", ".join(report["moved"]) or "none",
            ", ".join(report["skipped"]) or "none",
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
