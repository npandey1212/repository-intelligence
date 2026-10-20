"""M14 — SQLite persistence, initialization, errors (§7/§30/§31/§36).

 * deterministic idempotent schema creation (clean checkout works)
 * operation before initialize → MEMORY_STORE_NOT_INITIALIZED
 * store → fetch round-trip preserves the full contract (provenance,
   scope, evidence order)
 * unknown id → MEMORY_NOT_FOUND (never a default)
 * unwritable location → MEMORY_STORE_UNAVAILABLE (never a fallback)
 * the default database lives under ``runtime/`` (never ``artifacts/``)
 * CROSS-PROCESS PROOF: Process A stores and exits; Process B opens
   the same file and retrieves — real persistence through SQLite,
   not in-process object lifetime (§36)
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from repo_intelligence.evidence import EvidenceRef
from repo_intelligence.memory import (
    MEMORY_NOT_FOUND,
    MEMORY_STORE_NOT_INITIALIZED,
    MEMORY_STORE_UNAVAILABLE,
    MemoryCandidate,
    MemoryManager,
    MemoryOperationError,
    MemoryOwner,
    MemoryScope,
    default_memory_db_path,
)

OWNER = MemoryOwner(type="agent", id="repo_intelligence.agent")
REPO = "example-repository"
COMMIT = "a" * 40


def _candidate(statement="example-repository exposes widget:sync", **overrides):
    fields = dict(
        owner=OWNER,
        memory_type="semantic",
        statement=statement,
        scope=MemoryScope(repository=REPO, commit=COMMIT),
        evidence=(EvidenceRef(path="agents/widget.md", line_start=4, line_end=9),),
        subject="widget",
        source_run_id="run-1",
    )
    fields.update(overrides)
    return MemoryCandidate(**fields)


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------


def test_initialize_creates_schema_deterministically(tmp_path):
    db = tmp_path / "memory.db"
    manager = MemoryManager(db)
    assert manager.is_initialized() is False
    manager.initialize()
    assert manager.is_initialized() is True
    # idempotent: a second initialize is a no-op, never an error
    manager.initialize()
    conn = sqlite3.connect(str(db))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"memories", "memory_evidence"} <= tables
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    assert "idx_memories_repository" in indexes
    conn.close()


def test_initialize_creates_missing_parent_directory(tmp_path):
    db = tmp_path / "agent_memory" / "nested" / "memory.db"
    MemoryManager(db).initialize()
    assert db.exists()


def test_operation_before_initialize_is_structured_not_silent(tmp_path):
    manager = MemoryManager(tmp_path / "fresh.db")
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.store(_candidate())
    assert excinfo.value.code == MEMORY_STORE_NOT_INITIALIZED
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.search()
    assert excinfo.value.code == MEMORY_STORE_NOT_INITIALIZED


def test_store_unavailable_is_structured_and_has_no_fallback(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file in the way")
    manager = MemoryManager(blocker / "memory.db")
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.initialize()
    assert excinfo.value.code == MEMORY_STORE_UNAVAILABLE
    # no silent fallback location was created
    assert list(tmp_path.iterdir()) == [blocker]


def test_default_database_lives_under_db_store_not_artifacts_or_runtime():
    path = default_memory_db_path()
    # M18 storage centralization: durable databases live under
    # db_store/memory/, never under artifacts/ (analysis outputs) and
    # never as a durable file under runtime/ (ephemeral state).
    assert path.parts[-3:] == ("db_store", "memory", "agent_memory.db")
    assert "runtime" not in path.parts
    assert "artifacts" not in path.parts
    assert path.is_absolute()


def test_default_database_honors_db_store_env_seam(monkeypatch, tmp_path):
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "store"))
    path = default_memory_db_path()
    assert path == tmp_path / "store" / "memory" / "agent_memory.db"


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_store_then_get_preserves_the_full_contract(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    stored = manager.store(_candidate())
    fetched = manager.get(stored.memory_id)
    assert fetched == stored
    assert fetched.owner == OWNER
    assert fetched.scope == MemoryScope(repository=REPO, commit=COMMIT)
    assert fetched.evidence == (
        EvidenceRef(path="agents/widget.md", line_start=4, line_end=9),
    )
    assert fetched.subject == "widget"
    assert fetched.source_run_id == "run-1"
    assert fetched.status == "verified"
    assert fetched.supersedes is None
    assert fetched.superseded_by is None


def test_evidence_order_is_preserved(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    stored = manager.store(
        _candidate(
            evidence=(
                EvidenceRef(path="z.py", line_start=1, line_end=2),
                EvidenceRef(path="a.py"),
                EvidenceRef(path="m.py", line_start=9),
            )
        )
    )
    fetched = manager.get(stored.memory_id)
    assert [ref.path for ref in fetched.evidence] == ["z.py", "a.py", "m.py"]


def test_get_unknown_id_raises_not_found(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.get("mem_does_not_exist")
    assert excinfo.value.code == MEMORY_NOT_FOUND
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.get("")
    assert excinfo.value.code == MEMORY_NOT_FOUND


# ---------------------------------------------------------------------------
# cross-process proof (§36)
# ---------------------------------------------------------------------------

_STORE_IN_CHILD = """
import sys
sys.path.insert(0, {src!r})
from repo_intelligence.memory import (
    MemoryCandidate,
    MemoryManager,
    MemoryOwner,
    MemoryScope,
)

manager = MemoryManager(sys.argv[1])
manager.initialize()
stored = manager.store(
    MemoryCandidate(
        owner=MemoryOwner(type="agent", id="repo_intelligence.agent"),
        memory_type="semantic",
        statement="child-process memory",
        scope=MemoryScope(
            repository="example-repository", commit="c" * 40
        ),
    )
)
print(stored.memory_id)
"""


def test_memory_survives_a_process_boundary(tmp_path):
    """Process A stores + exits; Process B (this test process, a
    genuinely different interpreter) opens the same SQLite file and
    retrieves the memory — actual persistence, not shared objects."""
    db = tmp_path / "memory.db"
    src = str(Path(__file__).resolve().parents[2] / "src")
    script = _STORE_IN_CHILD.format(src=src)
    child = subprocess.run(
        [sys.executable, "-c", script, str(db)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    memory_id = child.stdout.strip()
    assert memory_id.startswith("mem_")

    # this process never created the schema and never saw the object
    manager = MemoryManager(db)
    assert manager.is_initialized() is True
    fetched = manager.get(memory_id)
    assert fetched.statement == "child-process memory"
    assert fetched.scope.commit == "c" * 40
    assert manager.search(repository="example-repository") == (fetched,)
