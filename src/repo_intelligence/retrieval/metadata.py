"""SQLite chunk metadata for retrieval indexes (M18 §20).

One shared database (``<db_store>/rag/metadata.db``) holds one row per
indexed chunk, keyed by index identity — the FAISS file carries the
vectors, this table carries what they mean: where the chunk lives, its
line range, content hash, language, and the embedded text itself (so
excerpts come from the indexed state, never from a working tree that
may have drifted).

Connections are opened and closed per operation (same discipline as
the memory store): no pooling, no threading assumptions, and two OS
processes observe each other's committed writes through the file
system. Writes for one identity are replace-writes (DELETE then
INSERT inside one transaction) so rebuilding an index never leaves
rows from a previous chunking of the same identity.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..storage import default_rag_metadata_db_path
from .models import Chunk, IndexIdentity

__all__ = [
    "count_chunks",
    "delete_identity_rows",
    "metadata_db_path",
    "read_chunks",
    "write_chunks",
]

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    repository_name TEXT NOT NULL,
    repository_url TEXT NOT NULL DEFAULT '',
    commit_sha TEXT NOT NULL,
    snapshot_id TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL,
    vector_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (
        repository_name, commit_sha, embedding_model, vector_id
    )
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_chunks_identity
ON chunks (repository_name, commit_sha, embedding_model)
"""


def metadata_db_path(store_root: Optional[Path] = None) -> Path:
    """``<db_store>/rag/metadata.db`` (or ``<store_root>/metadata.db``)."""
    if store_root is not None:
        return Path(store_root) / "metadata.db"
    return default_rag_metadata_db_path()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.execute(_CREATE_CHUNKS)
    connection.execute(_CREATE_INDEX)
    return connection


def _identity_clause(identity: IndexIdentity):
    return (
        "repository_name = ? AND commit_sha = ? "
        "AND embedding_model = ?"
    ), (
        identity.repository_name,
        identity.commit_sha,
        identity.embedding_model,
    )


def write_chunks(
    identity: IndexIdentity,
    chunks: Iterable[Chunk],
    db_path: Optional[Path] = None,
) -> int:
    """Replace all metadata rows for *identity*; return rows written.

    ``chunks`` must be in vector order — row ``vector_id`` is the
    position of the chunk's vector inside the FAISS index, so the two
    halves of the index can never drift apart.
    """
    path = Path(db_path) if db_path is not None else metadata_db_path()
    rows = list(chunks)
    clause, params = _identity_clause(identity)
    with _connect(path) as connection:
        connection.execute(
            "DELETE FROM chunks WHERE " + clause, params
        )
        connection.executemany(
            "INSERT INTO chunks ("
            "repository_name, repository_url, commit_sha, snapshot_id, "
            "embedding_model, vector_id, chunk_id, file_path, "
            "line_start, line_end, content_hash, language, content"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    identity.repository_name,
                    identity.repository_url,
                    identity.commit_sha,
                    identity.snapshot_id,
                    identity.embedding_model,
                    index,
                    chunk.chunk_id,
                    chunk.file_path,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.content_hash,
                    chunk.language,
                    chunk.content,
                )
                for index, chunk in enumerate(rows)
            ],
        )
    return len(rows)


def read_chunks(
    identity: IndexIdentity,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """All rows for *identity* in vector order (``[]`` when absent)."""
    path = Path(db_path) if db_path is not None else metadata_db_path()
    if not path.is_file():
        return []
    clause, params = _identity_clause(identity)
    connection = sqlite3.connect(str(path))
    try:
        cursor = connection.execute(
            "SELECT vector_id, chunk_id, file_path, line_start, "
            "line_end, content_hash, language, content, "
            "repository_url, snapshot_id "
            "FROM chunks WHERE "
            + clause
            + " ORDER BY vector_id ASC",
            params,
        )
        return [
            {
                "vector_id": row[0],
                "chunk_id": row[1],
                "file_path": row[2],
                "line_start": row[3],
                "line_end": row[4],
                "content_hash": row[5],
                "language": row[6],
                "content": row[7],
                "repository_url": row[8],
                "snapshot_id": row[9],
            }
            for row in cursor.fetchall()
        ]
    finally:
        connection.close()


def count_chunks(
    identity: IndexIdentity,
    db_path: Optional[Path] = None,
) -> int:
    """Number of metadata rows for *identity* (0 when absent)."""
    path = Path(db_path) if db_path is not None else metadata_db_path()
    if not path.is_file():
        return 0
    clause, params = _identity_clause(identity)
    connection = sqlite3.connect(str(path))
    try:
        cursor = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE " + clause, params
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


def delete_identity_rows(
    identity: IndexIdentity,
    db_path: Optional[Path] = None,
) -> int:
    """Remove all rows for *identity* (used by ``delete_index``)."""
    path = Path(db_path) if db_path is not None else metadata_db_path()
    if not path.is_file():
        return 0
    clause, params = _identity_clause(identity)
    with _connect(path) as connection:
        cursor = connection.execute(
            "DELETE FROM chunks WHERE " + clause, params
        )
        return int(cursor.rowcount or 0)
