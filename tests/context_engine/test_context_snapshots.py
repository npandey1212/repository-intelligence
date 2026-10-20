"""M15 — context snapshots (§11/§42/§45 negative test persistence).

 * ``engine.snapshot`` binds a selection to run/step identity; the
   step MUST match the package's step (a snapshot cannot be relabeled)
 * ``engine.persist`` writes ``<root>/<run_id>/context_<step>.json``
   whose decoded content equals ``snapshot.to_dict()`` — the file IS
   the canonical projection
 * unsafe run/step identifiers (separators, leading dots) → contract
   violation (the path is built from caller-supplied ids)
 * bound-engine scope guard applies to snapshot AND persist
 * I/O failure → CONTEXT_SNAPSHOT_FAILURE (explicit, never a silently
   skipped artifact)
 * default root lives under ``runtime/context`` (regenerable runtime
   state, never ``artifacts/``)

Everything synthetic and LLM-free; files only under ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.context_engine import (
    CONTEXT_CONTRACT_VIOLATION,
    CONTEXT_SCOPE_MISMATCH,
    CONTEXT_SNAPSHOT_FAILURE,
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextOperationError,
    TaskContext,
    default_context_root,
    new_candidate_id,
)

REPO = "example-repository"
COMMIT = "a" * 40


def _task(repository=REPO) -> TaskContext:
    return TaskContext(
        question="How is the widget registry wired?",
        repository=repository,
        commit=COMMIT,
    )


def _candidate(**overrides) -> ContextCandidate:
    fields = dict(
        candidate_id=new_candidate_id(),
        source_type="evidence",
        content="observed wiring",
        repository_scope=REPO,
        commit_scope=COMMIT,
    )
    fields.update(overrides)
    return ContextCandidate(**fields)


def _selection(engine=None, step_id="001", repository=REPO):
    engine = engine or ContextEngine()
    return engine.select(
        _task(repository=repository),
        [_candidate()],
        step_id=step_id,
    )


# ---------------------------------------------------------------------------
# snapshot binding
# ---------------------------------------------------------------------------


def test_snapshot_carries_run_step_and_full_audit(tmp_path):
    engine = ContextEngine(snapshot_root=tmp_path)
    selection = _selection(engine)
    snapshot = engine.snapshot(selection, run_id="run_42", step_id="001")
    assert snapshot.run_id == "run_42"
    assert snapshot.step_id == "001"
    assert snapshot.snapshot_id.startswith("snap_")
    assert snapshot.created_at.endswith("Z")
    assert snapshot.context_id == selection.package.context_id
    assert len(snapshot.selected_items) == 1
    assert len(snapshot.selection_decisions) == 1
    payload = snapshot.to_dict()
    assert payload["run_id"] == "run_42"
    assert payload["step_id"] == "001"
    assert payload["excluded_candidates"] == []


def test_snapshot_step_must_match_package_step():
    engine = ContextEngine()
    selection = _selection(engine, step_id="001")
    with pytest.raises(ContextOperationError) as excinfo:
        engine.snapshot(selection, run_id="run_1", step_id="002")
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_snapshot_rejects_non_selection_and_unsafe_ids():
    engine = ContextEngine()
    selection = _selection(engine)
    with pytest.raises(ContextOperationError) as excinfo:
        engine.snapshot({"package": None}, run_id="run_1",
                        step_id="001")
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION
    for bad in ("../escape", "a/b", "", "  ", ".hidden", "a\\b"):
        with pytest.raises(ContextOperationError) as excinfo:
            engine.snapshot(selection, run_id=bad, step_id="001")
        assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION
        with pytest.raises(ContextOperationError) as excinfo:
            engine.snapshot(selection, run_id="run_1", step_id=bad)
        assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_select_snapshot_convenience_pairs_selection_and_record():
    engine = ContextEngine()
    snapshot = engine.select_snapshot(
        _task(), [_candidate()], run_id="run_9", step_id="001"
    )
    assert snapshot.run_id == "run_9"
    assert snapshot.step_id == "001"
    assert snapshot.selected_items


# ---------------------------------------------------------------------------
# persistence (§42)
# ---------------------------------------------------------------------------


def test_persist_writes_canonical_json_under_run_directory(tmp_path):
    engine = ContextEngine(snapshot_root=tmp_path)
    selection = _selection(engine)
    snapshot = engine.snapshot(selection, run_id="run_42", step_id="001")
    path = engine.persist(snapshot)
    assert path == tmp_path / "run_42" / "context_001.json"
    assert path.exists()
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded == snapshot.to_dict()
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_persist_is_repeatable_for_the_same_step(tmp_path):
    engine = ContextEngine(snapshot_root=tmp_path)
    snapshot = engine.select_snapshot(
        _task(), [_candidate()], run_id="run_1", step_id="001"
    )
    first = engine.persist(snapshot)
    second = engine.persist(snapshot)
    assert first == second
    assert json.loads(first.read_text(encoding="utf-8")) == (
        snapshot.to_dict()
    )


def test_multiple_steps_persist_side_by_side(tmp_path):
    engine = ContextEngine(snapshot_root=tmp_path)
    for step in ("001", "002"):
        snapshot = engine.select_snapshot(
            _task(), [_candidate()], run_id="run_1", step_id=step
        )
        engine.persist(snapshot)
    files = sorted(p.name for p in (tmp_path / "run_1").iterdir())
    assert files == ["context_001.json", "context_002.json"]


def test_persist_io_failure_is_structured(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("occupied", encoding="utf-8")
    engine = ContextEngine(snapshot_root=blocker)
    snapshot = engine.select_snapshot(
        _task(), [_candidate()], run_id="run_1", step_id="001"
    )
    with pytest.raises(ContextOperationError) as excinfo:
        engine.persist(snapshot)
    assert excinfo.value.code == CONTEXT_SNAPSHOT_FAILURE


def test_persist_rejects_non_snapshot(tmp_path):
    engine = ContextEngine(snapshot_root=tmp_path)
    with pytest.raises(ContextOperationError) as excinfo:
        engine.persist({"snapshot_id": "snap_x"})
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_default_root_is_runtime_context():
    root = default_context_root()
    assert root.parts[-2:] == ("runtime", "context")
    assert "artifacts" not in root.parts


def test_engine_defaults_use_default_root():
    engine = ContextEngine()
    assert engine.snapshot_root == default_context_root()


# ---------------------------------------------------------------------------
# scope guard on snapshot/persist
# ---------------------------------------------------------------------------


def test_bound_engine_persist_refuses_foreign_snapshot(tmp_path):
    bound = ContextEngine(repository=REPO, snapshot_root=tmp_path)
    other = ContextEngine(repository="another-repo",
                          snapshot_root=tmp_path)
    foreign_task = TaskContext(
        question="q", repository="another-repo", commit=COMMIT
    )
    selection = other.select(
        foreign_task,
        [_candidate(repository_scope="another-repo", commit_scope=None)],
    )
    with pytest.raises(ContextOperationError) as excinfo:
        bound.snapshot(selection, run_id="run_1", step_id="001")
    assert excinfo.value.code == CONTEXT_SCOPE_MISMATCH
    snapshot = other.snapshot(selection, run_id="run_1", step_id="001")
    with pytest.raises(ContextOperationError) as excinfo:
        bound.persist(snapshot)
    assert excinfo.value.code == CONTEXT_SCOPE_MISMATCH


def test_budget_override_per_select_does_not_mutate_engine_default():
    engine = ContextEngine(budget=ContextBudget(max_characters=1000))
    # an impossible per-call budget fails explicitly...
    with pytest.raises(ContextOperationError) as excinfo:
        engine.select(
            _task(),
            [_candidate(content="x" * 10)],
            budget=ContextBudget(max_characters=5),
        )
    assert excinfo.value.code == "CONTEXT_BUDGET_EXCEEDED"
    # ...and the engine's own default is untouched
    selection = engine.select(_task(), [_candidate(content="x" * 10)])
    assert selection.package.budget.max_characters == 1000
    assert engine.budget.max_characters == 1000
