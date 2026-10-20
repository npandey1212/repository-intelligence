"""M14 — lifecycle, staleness, superseding (§26/§27/§28).

 * verify(): TRI-STATE scope comparison against the caller's current
   commit — ``fresh`` (both known+equal) records ``last_verified_at``;
   ``stale`` (both known+differ) changes nothing (never silently
   treated as current, never auto-rewritten); ``unknown`` (a commit is
   missing) is never reported fresh and writes no timestamp
 * stale memory is not deleted or hidden: it stays retrievable until
   the caller explicitly supersedes or invalidates it
 * supersede(): old row kept + flipped + pointer to the replacement,
   new row carries ``supersedes`` — one inspectable pointer each way
 * scope mismatch across repositories → MEMORY_SCOPE_MISMATCH
 * illegal transitions → MEMORY_INVALID; invalidation is explicit,
   idempotent, and never a hard delete
 * is_stale() / verification_status(): pure predicates — unknown
   never counts as stale, and never counts as fresh either

All synthetic (``example-repository``), all LLM-free.
"""

from __future__ import annotations

import pytest

from repo_intelligence.memory import (
    MEMORY_INVALID,
    MEMORY_NOT_FOUND,
    MEMORY_SCOPE_MISMATCH,
    MEMORY_VERIFICATION_FRESH,
    MEMORY_VERIFICATION_STALE,
    MEMORY_VERIFICATION_UNKNOWN,
    MemoryCandidate,
    MemoryManager,
    MemoryOperationError,
    MemoryOwner,
    MemoryScope,
    is_stale,
    verification_status,
)

OWNER = MemoryOwner(type="agent", id="repo_intelligence.agent")
REPO = "example-repository"
OTHER_REPO = "example-other-repository"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def _candidate(
    statement="example-repository exposes widget:sync",
    repository=REPO,
    commit=COMMIT_A,
    owner=OWNER,
):
    return MemoryCandidate(
        owner=owner,
        memory_type="semantic",
        statement=statement,
        scope=MemoryScope(repository=repository, commit=commit),
        subject="widget",
    )


@pytest.fixture()
def manager(tmp_path):
    memory_manager = MemoryManager(tmp_path / "memory.db")
    memory_manager.initialize()
    return memory_manager


# ---------------------------------------------------------------------------
# verify + staleness (§26)
# ---------------------------------------------------------------------------


def test_verify_same_commit_is_fresh_and_records_last_verified(
    manager, monkeypatch
):
    ticks = iter(
        [
            "2026-10-01T10:00:00.000Z",  # store: created_at
            "2026-10-01T10:00:00.000Z",  # store: last_verified_at
            "2026-10-01T12:00:00.000Z",  # verify bump
        ]
    )
    monkeypatch.setattr(
        "repo_intelligence.memory.manager._utcnow",
        lambda: next(ticks),
    )
    stored = manager.store(_candidate())
    outcome = manager.verify(stored.memory_id, COMMIT_A)
    assert outcome.stale is False
    assert outcome.status == "verified"
    assert outcome.scope_commit == COMMIT_A
    assert outcome.current_commit == COMMIT_A
    refreshed = manager.get(stored.memory_id)
    assert refreshed.last_verified_at == "2026-10-01T12:00:00.000Z"
    assert refreshed.created_at == "2026-10-01T10:00:00.000Z"
    assert refreshed.statement == stored.statement


def test_verify_other_commit_is_stale_and_changes_nothing(manager):
    stored = manager.store(_candidate())
    outcome = manager.verify(stored.memory_id, COMMIT_B)
    assert outcome.stale is True
    after = manager.get(stored.memory_id)
    assert after.status == "verified"  # reported, not resolved
    assert after.last_verified_at == stored.last_verified_at
    assert after.statement == stored.statement
    assert after.superseded_by is None  # never auto-superseded


def test_stale_memory_is_not_silently_currented_it_stays_retrievable(
    manager,
):
    stored = manager.store(_candidate())
    manager.verify(stored.memory_id, COMMIT_B)
    # still exactly where it was — the caller decides its fate
    assert [m.memory_id for m in manager.search(commit=COMMIT_A)] == [
        stored.memory_id
    ]


def test_verify_unknown_id_raises_not_found(manager):
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.verify("mem_missing", COMMIT_A)
    assert excinfo.value.code == MEMORY_NOT_FOUND


def test_verify_equal_commits_is_fresh_and_bumps_last_verified(
    manager, monkeypatch
):
    # Deterministic clock (same convention as the fresh/stale tests
    # above): timestamps are millisecond-precision, so two real calls
    # inside one millisecond are legitimately equal. The test controls
    # time instead of sleeping or weakening the assertion.
    ticks = iter(
        [
            "2026-10-01T10:00:00.000Z",  # store: created_at
            "2026-10-01T10:00:00.000Z",  # store: last_verified_at
            "2026-10-01T12:00:00.000Z",  # verify bump
        ]
    )
    monkeypatch.setattr(
        "repo_intelligence.memory.manager._utcnow",
        lambda: next(ticks),
    )
    stored = manager.store(_candidate())
    before = manager.get(stored.memory_id)
    outcome = manager.verify(stored.memory_id, COMMIT_A)
    assert outcome.verification_status == MEMORY_VERIFICATION_FRESH
    assert outcome.stale is False
    after = manager.get(stored.memory_id)
    assert after.last_verified_at == "2026-10-01T12:00:00.000Z"
    assert after.last_verified_at > before.last_verified_at


def test_verify_different_commits_is_stale_and_writes_nothing(manager):
    stored = manager.store(_candidate())
    before = manager.get(stored.memory_id)
    outcome = manager.verify(stored.memory_id, COMMIT_B)
    assert outcome.verification_status == MEMORY_VERIFICATION_STALE
    assert outcome.stale is True
    after = manager.get(stored.memory_id)
    assert after.last_verified_at == before.last_verified_at


def test_verify_missing_current_commit_is_unknown_not_fresh(manager):
    stored = manager.store(_candidate())
    before = manager.get(stored.memory_id)
    for missing in (None, ""):
        outcome = manager.verify(stored.memory_id, missing)
        assert outcome.verification_status == MEMORY_VERIFICATION_UNKNOWN
        assert outcome.stale is False
        # unknown must never be written up as a completed check
        after = manager.get(stored.memory_id)
        assert after.last_verified_at == before.last_verified_at


def test_verify_rejects_a_wrongly_typed_current_commit(manager):
    stored = manager.store(_candidate())
    with pytest.raises(ValueError):
        manager.verify(stored.memory_id, 42)


def test_is_stale_pure_predicate():
    stored_like = type(
        "M",
        (),
        {"scope": MemoryScope(repository=REPO, commit=COMMIT_A)},
    )()
    assert is_stale(stored_like, COMMIT_B) is True
    assert is_stale(stored_like, COMMIT_A) is False
    unscoped = type("M", (), {"scope": MemoryScope()})()
    assert is_stale(unscoped, COMMIT_B) is False
    assert (
        verification_status(stored_like, COMMIT_B)
        == MEMORY_VERIFICATION_STALE
    )
    assert (
        verification_status(stored_like, COMMIT_A)
        == MEMORY_VERIFICATION_FRESH
    )
    # a missing commit is UNKNOWN — not fresh, not stale
    assert (
        verification_status(stored_like, None)
        == MEMORY_VERIFICATION_UNKNOWN
    )
    assert (
        verification_status(unscoped, COMMIT_B)
        == MEMORY_VERIFICATION_UNKNOWN
    )


def test_verify_unscoped_memory_never_claims_freshness_or_staleness(
    manager,
):
    stored = manager.store(_candidate(commit=None, repository=None))
    before = manager.get(stored.memory_id)
    outcome = manager.verify(stored.memory_id, COMMIT_B)
    assert outcome.verification_status == MEMORY_VERIFICATION_UNKNOWN
    assert outcome.stale is False
    after = manager.get(stored.memory_id)
    assert after.last_verified_at == before.last_verified_at


# ---------------------------------------------------------------------------
# supersede (§28)
# ---------------------------------------------------------------------------


def test_supersede_records_an_inspectable_relationship(manager):
    old = manager.store(_candidate("old finding at commit A"))
    new = manager.supersede(
        old.memory_id,
        _candidate(
            "updated finding at commit B",
            commit=COMMIT_B,
        ),
    )
    assert new.supersedes == old.memory_id
    assert new.superseded_by is None
    assert new.status == "verified"
    assert new.statement == "updated finding at commit B"
    assert new.scope.commit == COMMIT_B

    flipped = manager.get(old.memory_id)
    assert flipped.status == "superseded"
    assert flipped.superseded_by == new.memory_id
    assert flipped.statement == "old finding at commit A"  # kept, not lost
    # both directions resolve through get()
    assert manager.get(new.supersedes).superseded_by == new.memory_id
    assert manager.get(flipped.superseded_by).supersedes == flipped.memory_id


def test_superseded_memory_is_searchable_as_superseded_but_not_verified(
    manager,
):
    old = manager.store(_candidate())
    manager.supersede(old.memory_id, _candidate("replacement"))
    assert manager.search(status="verified")[0].supersedes == old.memory_id
    superseded = manager.search(status="superseded")
    assert [m.memory_id for m in superseded] == [old.memory_id]


def test_supersede_across_repositories_is_a_scope_mismatch(manager):
    old = manager.store(_candidate(repository=REPO))
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.supersede(
            old.memory_id, _candidate(repository=OTHER_REPO)
        )
    assert excinfo.value.code == MEMORY_SCOPE_MISMATCH
    assert manager.get(old.memory_id).status == "verified"  # untouched


def test_supersede_an_invalid_memory_is_rejected(manager):
    old = manager.store(_candidate())
    manager.invalidate(old.memory_id)
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.supersede(old.memory_id, _candidate("replacement"))
    assert excinfo.value.code == MEMORY_INVALID


def test_supersede_an_already_superseded_memory_is_rejected(manager):
    old = manager.store(_candidate())
    manager.supersede(old.memory_id, _candidate("replacement one"))
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.supersede(old.memory_id, _candidate("replacement two"))
    assert excinfo.value.code == MEMORY_INVALID


def test_supersede_unknown_id_raises_not_found(manager):
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.supersede("mem_missing", _candidate())
    assert excinfo.value.code == MEMORY_NOT_FOUND


# ---------------------------------------------------------------------------
# invalidate (§27)
# ---------------------------------------------------------------------------


def test_invalidate_is_explicit_idempotent_and_never_deletes(manager):
    stored = manager.store(_candidate())
    invalidated = manager.invalidate(stored.memory_id)
    assert invalidated.status == "invalid"
    assert invalidated.statement == stored.statement
    again = manager.invalidate(stored.memory_id)
    assert again.status == "invalid"
    # still inspectable by id — memory is retained, not erased
    assert manager.get(stored.memory_id).status == "invalid"
    assert manager.search(status="invalid")[0].memory_id == stored.memory_id


def test_invalidate_unknown_id_raises_not_found(manager):
    with pytest.raises(MemoryOperationError) as excinfo:
        manager.invalidate("mem_missing")
    assert excinfo.value.code == MEMORY_NOT_FOUND


# ---------------------------------------------------------------------------
# full lifecycle walk (§27 vocabulary)
# ---------------------------------------------------------------------------


def test_create_store_retrieve_verify_supersede_invalidate(manager):
    created = manager.store(_candidate("first"))          # CREATE + STORE
    fetched = manager.get(created.memory_id)              # RETRIEVE
    fresh = manager.verify(fetched.memory_id, COMMIT_A)   # VERIFY
    assert fresh.stale is False
    replacement = manager.supersede(                      # SUPERSEDE
        fetched.memory_id, _candidate("second", commit=COMMIT_B)
    )
    stale = manager.verify(replacement.memory_id, COMMIT_A)
    assert stale.stale is True                            # staleness visible
    manager.invalidate(replacement.memory_id)             # INVALIDATE
    statuses = {
        manager.get(created.memory_id).status,
        manager.get(replacement.memory_id).status,
    }
    assert statuses == {"superseded", "invalid"}
