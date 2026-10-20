"""SQLite persistence for M14 memory (the lowest layer).

A thin, deterministic store over one local database file — no
connection pooling, no async, no cache (M14 §66): one ``sqlite3``
connection per operation, opened and closed, so two OS processes
observe each other's committed writes through the file system (the
M14 §36 cross-process proof).

Default location (M18 storage centralization — durable local
database, NEVER ``artifacts/`` and NEVER a durable file under
``runtime/``)::

    <project-root>/db_store/memory/agent_memory.db

``db_store/`` is git-ignored; the schema, semantics, and lifecycle
are unchanged from M14 — only the default path moved (one seam in
``default_memory_db_path``). Legacy databases under
``runtime/agent_memory/`` are migrated explicitly via
``repo_intelligence.storage.migrate_legacy_memory_databases`` —
never silently, never destructively.

Schema (smallest inspectable relational shape for the contract —
M14 §29)::

    memories            one row per stored Memory
        memory_id        PK  (opaque mem_<uuid>)
        owner_type       owner.type
        owner_id         owner.id
        memory_type      semantic | episodic
        statement        content.statement
        subject          retrieval key (nullable)
        repository       scope.repository (nullable)
        "commit"         scope.commit     (nullable, quoted: keyword)
        status           verified | superseded | invalid
        created_at       ISO-8601 UTC
        last_verified_at ISO-8601 UTC
        source_run_id    run that formed the memory (nullable)
        supersedes       memory_id this record replaced (nullable)
        superseded_by    memory_id that replaced this record

    memory_evidence     provenance rows (FK → memories, ordered)
        memory_id        FK
        position         evidence order within the memory
        path             repository-relative path
        line_start       nullable (same semantics as M13 EvidenceRef)
        line_end         nullable

Evidence is a SEPARATE TABLE, not a JSON column: provenance is the
part of the contract most likely to be queried or inspected
per-reference ("which memories cite this file?"), rows keep the
M13 ``EvidenceRef`` semantics without parsing, and the relationship
stays visible in a diagram — one extra table is the simplest
inspectable choice (decision recorded in docs/m14).

Lifecycle representation lives IN ``memories`` (status + supersedes /
superseded_by): no event log, no version graph — a superseded memory
is the SAME row flipped with a pointer to its replacement (M14 §28).

Initialization is deterministic and idempotent (``CREATE TABLE IF NOT
EXISTS``): a clean checkout + ``initialize()`` is enough; there is no
manually prepared database (M14 §30). An operation against a database
whose schema was never created raises ``MEMORY_STORE_NOT_INITIALIZED``
— initialization is explicit, never silently assumed.
"""

from __future__ import annotations

import datetime
import sqlite3
import uuid
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from ..evidence import EvidenceRef
from .errors import (
    MEMORY_NOT_FOUND,
    MEMORY_STORE_NOT_INITIALIZED,
    MEMORY_STORE_UNAVAILABLE,
    MemoryOperationError,
)
from .models import Memory, MemoryOwner, MemoryScope

__all__ = [
    "AGENT_MEMORY_DIR_NAME",
    "MEMORY_DB_FILE_NAME",
    "MemoryStore",
    "default_memory_db_path",
    "new_memory_id",
]

# Legacy location constants, kept for existing importers: the default
# path now comes from repo_intelligence.storage (M18); the historical
# runtime/agent_memory/ directory is migrated explicitly, never
# written to anew.
AGENT_MEMORY_DIR_NAME = "agent_memory"
MEMORY_DB_FILE_NAME = "agent_memory.db"

CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    statement TEXT NOT NULL,
    subject TEXT,
    repository TEXT,
    "commit" TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_verified_at TEXT NOT NULL,
    source_run_id TEXT,
    supersedes TEXT,
    superseded_by TEXT
)
"""

CREATE_MEMORY_EVIDENCE = """
CREATE TABLE IF NOT EXISTS memory_evidence (
    memory_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    PRIMARY KEY (memory_id, position),
    FOREIGN KEY (memory_id) REFERENCES memories (memory_id)
)
"""

INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_memories_repository "
    "ON memories (repository)",
    "CREATE INDEX IF NOT EXISTS idx_memories_owner "
    "ON memories (owner_type, owner_id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_status "
    "ON memories (status)",
    "CREATE INDEX IF NOT EXISTS idx_memories_subject "
    "ON memories (subject)",
    "CREATE INDEX IF NOT EXISTS idx_memories_memory_type "
    "ON memories (memory_type)",
)


def default_memory_db_path() -> Path:
    """Default database: ``<project-root>/db_store/memory/agent_memory.db``.

    Delegates to the single centralized layout helper
    (``repo_intelligence.storage``) — absolute, CWD-independent, and
    relocatable via ``REPO_INTELLIGENCE_DB_STORE``. The parent
    directory is created by ``initialize()``.
    """
    from ..storage import default_memory_db_path as _central

    return _central()


def _utcnow() -> str:
    """UTC timestamp, ISO-8601 with millisecond precision (sortable) —
    the same representation the AgentTrace uses."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _unavailable(exc: Exception) -> MemoryOperationError:
    return MemoryOperationError(
        MEMORY_STORE_UNAVAILABLE,
        "memory store unavailable: {}".format(exc),
    )


class MemoryStore:
    """Raw SQLite operations below the Memory Manager.

    This layer knows rows and SQL; it does NOT validate contracts,
    generate ids, or decide lifecycle policy (M14 §19/§20). Every
    public operation is:

        open (sqlite3.Error → MEMORY_STORE_UNAVAILABLE)
          → schema check (missing → MEMORY_STORE_NOT_INITIALIZED)
          → one statement / one transaction
          → close
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self._db_path = (
            Path(db_path) if db_path is not None else default_memory_db_path()
        )

    @property
    def db_path(self) -> Path:
        return self._db_path

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Create the schema if absent (idempotent, deterministic).

        Creates the parent directory when needed (the default runtime
        tree may not exist on a clean checkout). Opening a database at
        an unwritable path raises ``MEMORY_STORE_UNAVAILABLE`` — never
        a fallback location.
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
        except (sqlite3.Error, OSError) as exc:
            raise _unavailable(exc) from exc
        try:
            conn.execute(CREATE_MEMORIES)
            conn.execute(CREATE_MEMORY_EVIDENCE)
            for statement in INDEX_STATEMENTS:
                conn.execute(statement)
            conn.commit()
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(str(self._db_path))
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            conn.close()
            raise _unavailable(exc) from exc
        return conn

    def _require_initialized(self, conn: sqlite3.Connection) -> None:
        """Schema presence IS the initialized state — so a second OS
        process (which never called ``initialize``) can still operate
        on a database this one created (M14 §36)."""
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'memories'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        if row is None:
            raise MemoryOperationError(
                MEMORY_STORE_NOT_INITIALIZED,
                "memory store {} has no schema; call initialize() "
                "first".format(self._db_path),
            )

    def initialized(self) -> bool:
        """True when the schema exists (portable across processes)."""
        if not self._db_path.exists():
            return False
        try:
            conn = self._connect()
        except MemoryOperationError:
            return False
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'memories'"
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            return False
        finally:
            conn.close()

    # -- writes ------------------------------------------------------------

    def insert(self, memory: Memory) -> None:
        """Persist one already-valid Memory (one transaction)."""
        conn = self._connect()
        try:
            self._require_initialized(conn)
            conn.execute(
                "INSERT INTO memories (memory_id, owner_type, owner_id, "
                "memory_type, statement, subject, repository, \"commit\", "
                "status, created_at, last_verified_at, source_run_id, "
                "supersedes, superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?)",
                (
                    memory.memory_id,
                    memory.owner.type,
                    memory.owner.id,
                    memory.memory_type,
                    memory.statement,
                    memory.subject,
                    memory.scope.repository,
                    memory.scope.commit,
                    memory.status,
                    memory.created_at,
                    memory.last_verified_at,
                    memory.source_run_id,
                    memory.supersedes,
                    memory.superseded_by,
                ),
            )
            for position, ref in enumerate(memory.evidence):
                conn.execute(
                    "INSERT INTO memory_evidence (memory_id, position, "
                    "path, line_start, line_end) VALUES (?, ?, ?, ?, ?)",
                    (
                        memory.memory_id,
                        position,
                        ref.path,
                        ref.line_start,
                        ref.line_end,
                    ),
                )
            conn.commit()
        except MemoryOperationError:
            raise
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            conn.close()

    def mark_verified(self, memory_id: str, verified_at: str) -> None:
        """Record a successful re-verification timestamp."""
        self._update(
            "UPDATE memories SET last_verified_at = ? WHERE memory_id = ?",
            (verified_at, memory_id),
        )

    def mark_superseded(self, memory_id: str, superseded_by: str) -> None:
        """Point the old memory at its replacement (status flip)."""
        self._update(
            "UPDATE memories SET status = 'superseded', superseded_by = ? "
            "WHERE memory_id = ?",
            (superseded_by, memory_id),
        )

    def set_status(self, memory_id: str, status: str) -> None:
        """Set an explicit lifecycle status (e.g. 'invalid')."""
        self._update(
            "UPDATE memories SET status = ? WHERE memory_id = ?",
            (status, memory_id),
        )

    def _update(self, sql: str, params: Sequence[Any]) -> None:
        conn = self._connect()
        try:
            self._require_initialized(conn)
            cursor = conn.execute(sql, tuple(params))
            if cursor.rowcount == 0:
                raise MemoryOperationError(
                    MEMORY_NOT_FOUND,
                    "memory {!r} not found".format(params[-1]),
                )
            conn.commit()
        except MemoryOperationError:
            raise
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            conn.close()

    # -- reads -------------------------------------------------------------

    def fetch(self, memory_id: str) -> Optional[Memory]:
        """One memory by id, or ``None`` (caller raises NOT_FOUND)."""
        conn = self._connect()
        try:
            self._require_initialized(conn)
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            return self._decode(conn, row)
        except MemoryOperationError:
            raise
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            conn.close()

    def query(
        self,
        repository: Optional[str] = None,
        commit: Optional[str] = None,
        owner: Optional[MemoryOwner] = None,
        memory_type: Optional[str] = None,
        subject: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Memory]:
        """Deterministic filtered read.

        ``None`` = dimension not filtered. Ordering is
        ``created_at, memory_id`` — stable for a given database, so
        the same query repeated returns the same sequence (M14 §24).
        """
        clauses: List[str] = []
        params: List[Any] = []
        if repository is not None:
            clauses.append("repository = ?")
            params.append(repository)
        if commit is not None:
            clauses.append("\"commit\" = ?")
            params.append(commit)
        if owner is not None:
            clauses.append("owner_type = ? AND owner_id = ?")
            params.extend([owner.type, owner.id])
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, memory_id ASC LIMIT ?"
        params.append(int(limit))

        conn = self._connect()
        try:
            self._require_initialized(conn)
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [self._decode(conn, row) for row in rows]
        except MemoryOperationError:
            raise
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            conn.close()

    # -- decoding ----------------------------------------------------------

    def _decode(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Memory:
        evidence_rows = conn.execute(
            "SELECT path, line_start, line_end FROM memory_evidence "
            "WHERE memory_id = ? ORDER BY position ASC",
            (row["memory_id"],),
        ).fetchall()
        evidence = tuple(
            EvidenceRef(
                path=evidence_row["path"],
                line_start=evidence_row["line_start"],
                line_end=evidence_row["line_end"],
            )
            for evidence_row in evidence_rows
        )
        return Memory(
            memory_id=row["memory_id"],
            owner=MemoryOwner(type=row["owner_type"], id=row["owner_id"]),
            memory_type=row["memory_type"],
            statement=row["statement"],
            scope=MemoryScope(
                repository=row["repository"], commit=row["commit"]
            ),
            evidence=evidence,
            status=row["status"],
            created_at=row["created_at"],
            last_verified_at=row["last_verified_at"],
            subject=row["subject"],
            source_run_id=row["source_run_id"],
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
        )


def new_memory_id() -> str:
    """Opaque application-generated id: ``mem_<32 hex>``.

    UUID-based, never semantically derived from content (M14 §12):
    identity stays independent of what the memory says, so the model
    can neither predict nor influence it.
    """
    return "mem_" + uuid.uuid4().hex
