"""M18 storage layout: centralized db_store/ helpers + explicit migration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repo_intelligence import storage
from repo_intelligence.storage import (
    StorageMigrationError,
    default_db_store_root,
    default_memory_db_path,
    default_memory_store_root,
    default_query_db_path,
    default_rag_metadata_db_path,
    default_rag_store_root,
    migrate_legacy_memory_databases,
    rag_vector_store_path,
)


# ---------------------------------------------------------------------------
# centralized paths (Test M)
# ---------------------------------------------------------------------------


def test_default_paths_are_absolute_project_relative():
    for path in (
        default_db_store_root(),
        default_memory_store_root(),
        default_rag_store_root(),
        default_memory_db_path(),
        default_query_db_path(),
        default_rag_metadata_db_path(),
    ):
        assert path.is_absolute()
        assert "db_store" in path.parts


def test_memory_and_query_dbs_share_memory_tree():
    assert default_memory_db_path() == (
        default_memory_store_root() / "agent_memory.db"
    )
    assert default_query_db_path() == (
        default_memory_store_root() / "repository_query.db"
    )


def test_paths_never_live_under_runtime_or_artifacts():
    for path in (
        default_memory_db_path(),
        default_query_db_path(),
        default_rag_metadata_db_path(),
    ):
        assert "runtime" not in path.parts
        assert "artifacts" not in path.parts


def test_env_seam_relocates_whole_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "store"))
    assert default_db_store_root() == tmp_path / "store"
    assert default_memory_db_path() == (
        tmp_path / "store" / "memory" / "agent_memory.db"
    )
    assert default_rag_metadata_db_path() == (
        tmp_path / "store" / "rag" / "metadata.db"
    )
    assert rag_vector_store_path("repo", "abc") == (
        tmp_path / "store" / "rag" / "vector_store" / "repo" / "abc"
    )


def test_env_seam_absent_means_project_root(monkeypatch):
    monkeypatch.delenv("REPO_INTELLIGENCE_DB_STORE", raising=False)
    root = default_db_store_root()
    assert root == storage.PROJECT_ROOT / "db_store"


def test_vector_store_path_embedding_segment(monkeypatch, tmp_path):
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "store"))
    plain = rag_vector_store_path("r", "c1")
    assert plain == (
        tmp_path / "store" / "rag" / "vector_store" / "r" / "c1"
    )
    with_model = rag_vector_store_path("r", "c1", "hashing-v1")
    assert with_model == plain / "hashing-v1"
    explicit = rag_vector_store_path(
        "r", "c1", "m", store_root=tmp_path / "other"
    )
    assert str(explicit).startswith(str(tmp_path / "other"))


# ---------------------------------------------------------------------------
# explicit legacy migration (never silent, never destructive)
# ---------------------------------------------------------------------------


@pytest.fixture
def legacy_env(monkeypatch, tmp_path):
    """Legacy runtime/agent_memory/ with two real (tiny) databases."""
    legacy = tmp_path / "runtime" / "agent_memory"
    legacy.mkdir(parents=True)
    for name in ("agent_memory.db", "repository_query.db"):
        (legacy / name).write_bytes(b"SQLite format 3\x00legacy-bytes")
    monkeypatch.setattr(storage, "legacy_memory_dir", lambda: legacy)
    target = tmp_path / "store" / "memory"
    return {"legacy": legacy, "target": target}


def test_migration_moves_both_databases(legacy_env):
    report = migrate_legacy_memory_databases(legacy_env["target"])
    assert sorted(report["moved"]) == [
        "agent_memory.db",
        "repository_query.db",
    ]
    assert report["skipped"] == []
    for name in ("agent_memory.db", "repository_query.db"):
        assert not (legacy_env["legacy"] / name).exists()
        moved = legacy_env["target"] / name
        assert moved.is_file()
        assert moved.read_bytes().endswith(b"legacy-bytes")


def test_migration_is_idempotent_when_already_moved(legacy_env):
    migrate_legacy_memory_databases(legacy_env["target"])
    report = migrate_legacy_memory_databases(legacy_env["target"])
    assert report["moved"] == []
    assert sorted(report["skipped"]) == [
        "agent_memory.db",
        "repository_query.db",
    ]


def test_migration_refuses_when_target_exists(legacy_env):
    (legacy_env["target"]).mkdir(parents=True)
    (legacy_env["target"] / "agent_memory.db").write_bytes(b"new-data")
    with pytest.raises(StorageMigrationError) as excinfo:
        migrate_legacy_memory_databases(legacy_env["target"])
    assert excinfo.value.code == "MIGRATION_TARGET_EXISTS"
    # nothing destroyed on either side
    assert (legacy_env["legacy"] / "agent_memory.db").is_file()
    assert (
        legacy_env["target"] / "agent_memory.db"
    ).read_bytes() == b"new-data"
    # the second file was not moved either (all-or-nothing refusal)
    assert (legacy_env["legacy"] / "repository_query.db").is_file()


def test_migration_without_legacy_files_reports_skipped(
    monkeypatch, tmp_path
):
    missing = tmp_path / "no-such-dir"
    monkeypatch.setattr(storage, "legacy_memory_dir", lambda: missing)
    report = migrate_legacy_memory_databases(tmp_path / "target")
    assert report == {
        "moved": [],
        "skipped": ["agent_memory.db", "repository_query.db"],
    }
    assert not (tmp_path / "target").exists()


# ---------------------------------------------------------------------------
# CLI: python -m repo_intelligence.storage
# ---------------------------------------------------------------------------


def test_storage_cli_dry_run_prints_plan(
    monkeypatch, tmp_path, capsys
):
    legacy = tmp_path / "runtime" / "agent_memory"
    legacy.mkdir(parents=True)
    (legacy / "agent_memory.db").write_bytes(b"legacy")
    monkeypatch.setattr(storage, "legacy_memory_dir", lambda: legacy)
    monkeypatch.setenv(
        "REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "store")
    )
    rc = storage.main(["--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["would_move"] == ["agent_memory.db"]
    assert out["would_refuse"] == []
    assert (legacy / "agent_memory.db").is_file()  # nothing moved


def test_storage_cli_module_entrypoint_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "repo_intelligence.storage", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert "would_move" in plan
    # platform-neutral comparison: Windows destinations contain "\"
    assert Path(plan["destination"]).as_posix().endswith(
        "db_store/memory"
    )
