"""Spec §33 D/E/F — index persistence, identity resolution, isolation.

Three layers pinned here:

* identity — record beside the checkout (structural), run-context
  match, unpinned fallback for M17-style explicit subjects, and loud
  failure inside a ``repository/`` container without a record;
* index — build is idempotent (``reused``), persistence survives a
  fresh load, ``delete_index`` removes every half (vectors +
  metadata), and commit/repository isolation is structural (separate
  index directories keyed by repository_key + commit);
* service — bounded ``top_k``, path filtering, honest counts, and
  strictly read-only queries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (
    DEFAULT_COMMIT,
    DEFAULT_NAME,
    DEFAULT_RUN_ID,
    install_pinned_run,
)
from repo_intelligence.retrieval import (
    active_index_identity,
    build_index,
    delete_index,
    embedder_for,
    load_index,
    resolve_index_identity,
)
from repo_intelligence.retrieval.errors import RetrievalError
from repo_intelligence.retrieval.identity import identity_for_snapshot
from repo_intelligence.retrieval.metadata import (
    count_chunks,
    delete_identity_rows,
    read_chunks,
    write_chunks,
)
from repo_intelligence.retrieval.models import Chunk, IndexIdentity
from repo_intelligence.retrieval.service import RetrievalService

HASHING = "hashing://v1/128"

FILES = {
    "docs/guide.md": (
        "# Guide\n\nThe checkout flow starts here.\n"
        "It validates the cart.\n"
    ),
    "src/app.py": (
        "def start_checkout():\n"
        "    return None\n"
    ),
}


def _pinned(tmp_path: Path, **kwargs) -> Path:
    return install_pinned_run(tmp_path / "artifacts", dict(FILES), **kwargs)


def _identity_for(checkout: Path) -> IndexIdentity:
    from dataclasses import replace

    base = resolve_index_identity(checkout, None)
    return replace(base, embedding_model=HASHING)


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def test_record_beside_checkout_resolves_identity(tmp_path):
    checkout = _pinned(tmp_path)
    identity = resolve_index_identity(checkout, None)
    assert identity.repository_name == DEFAULT_NAME
    assert identity.commit_sha == DEFAULT_COMMIT
    assert identity.snapshot_id == DEFAULT_RUN_ID
    assert identity.repository_url.startswith("https://")
    # no explicit model → the production default (ST) is recorded
    assert identity.embedding_model.startswith(
        "sentence-transformers://"
    )


def test_resolve_with_explicit_model_wins(tmp_path):
    checkout = _pinned(tmp_path)
    identity = resolve_index_identity(checkout, None, HASHING)
    assert identity.embedding_model == HASHING


def test_legacy_flat_checkout_resolves_identity(tmp_path):
    run_dir = tmp_path / "artifacts" / "run_legacy"
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "repository").mkdir()
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        '{"schema_version": "repository_snapshot.v1",'
        ' "repository": {"name": "legacy-repo", "url": ""},'
        ' "snapshot": {"ref": "main", "commit_sha": "%s"}}'
        % ("c" * 40),
        encoding="utf-8",
    )
    (run_dir / "repository" / "README.md").write_text(
        "x\n", encoding="utf-8"
    )
    identity = resolve_index_identity(run_dir / "repository", None)
    assert identity.repository_name == "legacy-repo"
    assert identity.commit_sha == "c" * 40
    assert identity.snapshot_id == "run_legacy"


def test_unpinned_directory_gets_path_scoped_identity(tmp_path):
    checkout = tmp_path / "standalone_checkout"
    checkout.mkdir()
    (checkout / "README.md").write_text("hello\n", encoding="utf-8")
    identity = resolve_index_identity(checkout, None, HASHING)
    assert identity.repository_name == "standalone_checkout"
    assert identity.commit_sha.startswith("unpinned:")
    assert len(identity.commit_sha) == len("unpinned:") + 12
    assert identity.snapshot_id == ""
    # a second directory with the same name elsewhere never collides
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    other = other_root / "standalone_checkout"
    other.mkdir()
    other_identity = resolve_index_identity(other, None, HASHING)
    assert other_identity.commit_sha != identity.commit_sha


def test_container_without_record_fails_loudly(tmp_path):
    root = tmp_path / "artifacts" / "run_x" / "repository" / "co"
    root.mkdir(parents=True)
    (root / "README.md").write_text("x\n", encoding="utf-8")
    with pytest.raises(RetrievalError) as exc:
        resolve_index_identity(root, None)
    assert exc.value.code == "RAG_IDENTITY_NOT_RESOLVED"
    assert "repository_snapshot.json" in exc.value.message


def test_foreign_named_checkout_fails_loudly(tmp_path):
    """A checkout whose name disagrees with the run's record is not
    silently given the record's identity."""
    checkout = _pinned(tmp_path, repository_name="real-name")
    foreign = checkout.parent / "some-other-dir"
    foreign.mkdir()
    (foreign / "README.md").write_text("x\n", encoding="utf-8")
    with pytest.raises(RetrievalError) as exc:
        resolve_index_identity(foreign, None)
    assert exc.value.code == "RAG_IDENTITY_NOT_RESOLVED"


def test_identity_for_snapshot_reads_the_artifact(tmp_path):
    _pinned(tmp_path)
    identity = identity_for_snapshot(
        tmp_path / "artifacts", DEFAULT_RUN_ID, HASHING
    )
    assert identity.repository_name == DEFAULT_NAME
    assert identity.commit_sha == DEFAULT_COMMIT
    assert identity.embedding_model == HASHING


def test_identity_for_snapshot_missing_run_fails(tmp_path):
    empty = tmp_path / "artifacts"
    empty.mkdir()
    with pytest.raises(RetrievalError) as exc:
        identity_for_snapshot(empty, "run_nope")
    assert exc.value.code == "RAG_IDENTITY_NOT_RESOLVED"


# ---------------------------------------------------------------------------
# Build / persistence / reuse
# ---------------------------------------------------------------------------


def test_build_index_reports_the_contract(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    report = build_index(
        checkout, _identity_for(checkout), embedder=embedder_for(HASHING)
    )
    assert report.reused is False
    assert report.files_indexed == len(FILES)
    assert report.chunks_indexed >= report.files_indexed
    assert report.embedder == HASHING
    assert report.dimension == 128
    assert Path(report.index_dir).is_dir()
    payload = report.to_dict()
    assert payload["identity"]["repository_name"] == DEFAULT_NAME
    assert payload["identity"]["commit_sha"] == DEFAULT_COMMIT
    assert payload["chunks_indexed"] == report.chunks_indexed


def test_build_is_idempotent_and_force_rebuilds(
    tmp_path, monkeypatch
):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    first = build_index(
        checkout, identity, embedder=embedder_for(HASHING)
    )
    second = build_index(
        checkout, identity, embedder=embedder_for(HASHING)
    )
    assert second.reused is True
    assert second.chunks_indexed == first.chunks_indexed
    forced = build_index(
        checkout, identity, embedder=embedder_for(HASHING), force=True
    )
    assert forced.reused is False
    assert forced.chunks_indexed == first.chunks_indexed


def test_index_survives_a_fresh_process_load(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))
    # brand-new service object (no caches) reads the same bytes
    response = RetrievalService().search("checkout flow", identity)
    assert response.results
    assert response.index_chunks > 0
    assert response.results[0].repository_name == DEFAULT_NAME
    assert response.results[0].commit_sha == DEFAULT_COMMIT


def test_delete_index_removes_vectors_and_metadata(
    tmp_path, monkeypatch
):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    report = build_index(
        checkout, identity, embedder=embedder_for(HASHING)
    )
    assert count_chunks(identity) > 0
    delete_index(identity)
    assert not Path(report.index_dir).exists()
    assert count_chunks(identity) == 0
    with pytest.raises(RetrievalError) as exc:
        load_index(identity)
    assert exc.value.code == "RAG_INDEX_NOT_FOUND"


def test_missing_checkout_fails_the_build(tmp_path):
    bogus = IndexIdentity(
        repository_name="ghost",
        commit_sha="f" * 40,
        embedding_model=HASHING,
    )
    with pytest.raises(RetrievalError) as exc:
        build_index(
            tmp_path / "absent", bogus, embedder=embedder_for(HASHING)
        )
    assert exc.value.code == "RAG_REPOSITORY_NOT_FOUND"


# ---------------------------------------------------------------------------
# Commit and repository isolation
# ---------------------------------------------------------------------------


def test_commit_isolation_two_index_directories(
    tmp_path, monkeypatch
):
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    one = install_pinned_run(
        tmp_path / "artifacts",
        FILES,
        run_id="run_one",
        commit_sha="1" * 40,
    )
    two = install_pinned_run(
        tmp_path / "artifacts",
        {"docs/guide.md": "second commit content\n"},
        run_id="run_two",
        commit_sha="2" * 40,
    )
    id_one = _identity_for(one)
    id_two = _identity_for(two)
    assert id_one.commit_sha != id_two.commit_sha
    build_index(one, id_one, embedder=embedder_for(HASHING))
    build_index(two, id_two, embedder=embedder_for(HASHING))
    loaded_one = load_index(id_one)
    loaded_two = load_index(id_two)
    assert loaded_one.index_dir != loaded_two.index_dir
    paths_one = {r["file_path"] for r in loaded_one.records}
    paths_two = {r["file_path"] for r in loaded_two.records}
    assert "src/app.py" in paths_one
    assert "src/app.py" not in paths_two
    # neither index ever carries the other commit's content
    for record in loaded_one.records:
        assert record["file_path"] in FILES
        # chunk content is the joined line range (no trailing \n)
        assert record["content"] == FILES[record["file_path"]].rstrip(
            "\n"
        )
    for record in loaded_two.records:
        assert record["content"] == "second commit content"


def test_repository_isolation_in_one_store(tmp_path, monkeypatch):
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    repo_a = install_pinned_run(
        tmp_path / "artifacts",
        {"docs/a.md": "alpha repository marker\n"},
        run_id="run_a",
        repository_name="repo-alpha",
        commit_sha="a" * 40,
    )
    repo_b = install_pinned_run(
        tmp_path / "artifacts",
        {"docs/b.md": "beta repository marker\n"},
        run_id="run_b",
        repository_name="repo-beta",
        commit_sha="b" * 40,
    )
    id_a = _identity_for(repo_a)
    id_b = _identity_for(repo_b)
    build_index(repo_a, id_a, embedder=embedder_for(HASHING))
    build_index(repo_b, id_b, embedder=embedder_for(HASHING))
    assert load_index(id_a).index_dir != load_index(id_b).index_dir
    response = RetrievalService().search("alpha marker", id_a)
    paths = {item.file_path for item in response.results}
    assert paths == {"docs/a.md"}
    response_b = RetrievalService().search("alpha marker", id_b)
    assert response_b.results
    excerpts = " ".join(
        item.excerpt for item in response_b.results
    )
    assert "alpha repository marker" not in excerpts


# ---------------------------------------------------------------------------
# active_index_identity (model adoption / ambiguity)
# ---------------------------------------------------------------------------


def test_active_identity_adopts_the_single_non_default_model(
    tmp_path, monkeypatch
):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))
    adopted = active_index_identity(identity, None)
    assert adopted.embedding_model == HASHING


def test_active_identity_keeps_a_matching_model_untouched(
    tmp_path, monkeypatch
):
    """When an index exists for the identity's own model, adoption
    is a no-op (the production default-model path)."""
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))
    assert active_index_identity(identity, None) == identity


def test_active_identity_reports_ambiguity(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity_128 = _identity_for(checkout)
    build_index(
        checkout, identity_128, embedder=embedder_for(HASHING)
    )
    from dataclasses import replace

    identity_64 = replace(
        identity_128, embedding_model="hashing://v1/64"
    )
    build_index(
        checkout, identity_64, embedder=embedder_for("hashing://v1/64")
    )
    # an identity on ANOTHER model (the production default) sees two
    # candidate indexes and must refuse to guess
    default_identity = resolve_index_identity(checkout, None)
    with pytest.raises(RetrievalError) as exc:
        active_index_identity(default_identity, None)
    assert exc.value.code == "RAG_INDEX_AMBIGUOUS"
    assert "hashing://v1/128" in exc.value.message
    assert "hashing://v1/64" in exc.value.message
    # the identity that owns one of the indexes never raises
    assert active_index_identity(identity_128, None) == identity_128


def test_active_identity_untouched_when_no_index_exists(
    tmp_path, monkeypatch
):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    unchanged = active_index_identity(identity, None)
    assert unchanged == identity


# ---------------------------------------------------------------------------
# Service contract
# ---------------------------------------------------------------------------


def test_service_search_bounds(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))
    service = RetrievalService()
    for bad in (0, -1, 51, True, 1.5):
        with pytest.raises(RetrievalError) as exc:
            service.search("query", identity, top_k=bad)
        assert exc.value.code == "RAG_INVALID_ARGUMENT"
    for bad_query in ("", "   ", None, 5, ["q"]):
        with pytest.raises(RetrievalError) as exc:
            service.search(bad_query, identity)
        assert exc.value.code == "RAG_INVALID_ARGUMENT"
    with pytest.raises(RetrievalError) as exc:
        service.search("query", identity, path_filter=7)
    assert exc.value.code == "RAG_INVALID_ARGUMENT"


def test_service_top_k_and_honest_counts(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))
    service = RetrievalService()
    small = service.search("checkout", identity, top_k=1)
    assert len(small.results) == 1
    big = service.search("checkout", identity, top_k=50)
    assert len(big.results) <= 50
    assert small.total_matches == big.total_matches
    assert big.index_chunks == big.total_matches


def test_service_path_filter_restricts_results(
    tmp_path, monkeypatch
):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))
    response = RetrievalService().search(
        "checkout", identity, path_filter="docs"
    )
    assert response.results
    for item in response.results:
        assert item.file_path.startswith("docs/")
    assert response.total_matches <= response.index_chunks


def test_service_is_strictly_read_only(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    store = tmp_path / "db_store"
    monkeypatch.setenv("REPO_INTELLIGENCE_DB_STORE", str(store))
    identity = _identity_for(checkout)
    build_index(checkout, identity, embedder=embedder_for(HASHING))

    def snapshot(root: Path):
        state = {}
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if path.is_file():
                state[rel] = path.read_bytes()
            else:
                state[rel] = b"dir"
        return state

    before_store = snapshot(store)
    before_checkout = snapshot(checkout)
    service = RetrievalService()
    for query in ("checkout", "guide", "flow"):
        service.search(query, identity, top_k=50)
    assert snapshot(store) == before_store
    assert snapshot(checkout) == before_checkout


def test_service_missing_index_is_actionable(tmp_path, monkeypatch):
    checkout = _pinned(tmp_path)
    monkeypatch.setenv(
        "REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "empty_store")
    )
    identity = _identity_for(checkout)
    with pytest.raises(RetrievalError) as exc:
        RetrievalService().search("checkout", identity)
    assert exc.value.code == "RAG_INDEX_NOT_FOUND"
    assert "python -m repo_intelligence.retrieval.index" in (
        exc.value.message
    )


# ---------------------------------------------------------------------------
# Metadata store (SQLite)
# ---------------------------------------------------------------------------


def _chunk(path="a.md", start=1, end=4):
    return Chunk(
        file_path=path,
        line_start=start,
        line_end=end,
        content="fixture line one\nfixture line two",
        language="markdown",
    )


def test_metadata_write_read_count_delete(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "db_store")
    )
    identity = IndexIdentity(
        repository_name="meta-repo",
        commit_sha="d" * 40,
        embedding_model=HASHING,
        snapshot_id="run_meta",
    )
    chunks = [_chunk(), _chunk("b.md", 2, 5)]
    written = write_chunks(identity, chunks)
    assert written == 2
    assert count_chunks(identity) == 2
    read = read_chunks(identity)
    assert [r["file_path"] for r in read] == ["a.md", "b.md"]
    assert read[0]["chunk_id"] == _chunk().chunk_id
    assert read[0]["content"].startswith("fixture line one")
    assert read[0]["snapshot_id"] == "run_meta"
    delete_identity_rows(identity)
    assert count_chunks(identity) == 0
    assert read_chunks(identity) == []


def test_metadata_is_scoped_to_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "REPO_INTELLIGENCE_DB_STORE", str(tmp_path / "db_store")
    )
    one = IndexIdentity(
        repository_name="repo-one",
        commit_sha="1" * 40,
        embedding_model=HASHING,
        snapshot_id="",
    )
    two = IndexIdentity(
        repository_name="repo-two",
        commit_sha="2" * 40,
        embedding_model=HASHING,
        snapshot_id="",
    )
    write_chunks(one, [_chunk()])
    write_chunks(two, [_chunk("other.md", 1, 2)])
    assert count_chunks(one) == 1
    assert count_chunks(two) == 1
    assert read_chunks(one)[0]["file_path"] == "a.md"
    assert read_chunks(two)[0]["file_path"] == "other.md"
    # a third identity never sees either row
    stranger = IndexIdentity(
        repository_name="repo-three",
        commit_sha="3" * 40,
        embedding_model=HASHING,
    )
    assert count_chunks(stranger) == 0
