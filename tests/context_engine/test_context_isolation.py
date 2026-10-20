"""M15 — repository isolation, freshness, authority (§4/§14/§24,
§45 negative tests 2–4).

 * repository A candidate + repository B task → excluded
   OUT_OF_SCOPE — never leaks into B's context (negative test 4)
 * candidate pinned to a repo + UNscoped task → excluded (nothing
   verifies it); scope-neutral candidates stay eligible
 * a repository-BOUND engine refuses a foreign task/snapshot with
   CONTEXT_SCOPE_MISMATCH (operation-level guard)
 * fresh memory eligible; stale memory → STALE_MEMORY, never
   selected as current (negative test 2)
 * unknown-freshness memory (missing commit on either side) →
   UNKNOWN_FRESHNESS — never silently fresh (negative test 3);
   ``candidate_from_memory`` maps the M14 tri-state faithfully and
   ``current_commit=None`` is ALWAYS unknown
 * §14 hierarchy: current evidence outranks historical memory when
   the budget can only keep one

Everything synthetic (``example-repository``) and LLM-free.
"""

from __future__ import annotations

import pytest

from repo_intelligence.context_engine import (
    CONTEXT_BUDGET_EXCEEDED,
    CONTEXT_SCOPE_MISMATCH,
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextOperationError,
    REASON_OUT_OF_SCOPE,
    REASON_STALE_MEMORY,
    REASON_UNKNOWN_FRESHNESS,
    TaskContext,
    candidate_from_memory,
    candidate_from_task,
    new_candidate_id,
    scope_matches,
    select_context,
)
from repo_intelligence.memory import (
    Memory,
    MemoryOwner,
    MemoryScope,
    verification_status,
)

REPO_A = "repo-a"
REPO_B = "repo-b"
COMMIT_1 = "1" * 40
COMMIT_2 = "2" * 40


def _task(repository=REPO_A, commit=COMMIT_1, **overrides) -> TaskContext:
    return TaskContext(
        question="How does repo-a wire its tools?",
        repository=repository,
        commit=commit,
        **overrides,
    )


def _cand(repository=REPO_A, commit=COMMIT_1, **overrides) -> ContextCandidate:
    fields = dict(
        candidate_id=new_candidate_id(),
        source_type="evidence",
        content="observed wiring",
        repository_scope=repository,
        commit_scope=commit,
    )
    fields.update(overrides)
    return ContextCandidate(**fields)


def _memory(repository=REPO_A, commit=COMMIT_1, statement="known fact"):
    return Memory(
        memory_id="mem_fixed_1",
        owner=MemoryOwner(type="agent", id="repo_intelligence.agent"),
        memory_type="episodic",
        statement=statement,
        scope=MemoryScope(repository=repository, commit=commit),
        evidence=(),
        status="verified",
        created_at="2026-10-01T00:00:00.000Z",
        last_verified_at="2026-10-01T00:00:00.000Z",
    )


def _decisions(selection):
    return {d.candidate_id: d for d in selection.decisions}


# ---------------------------------------------------------------------------
# repository isolation (§45 negative test 4)
# ---------------------------------------------------------------------------


def test_repo_a_candidate_never_enters_repo_b_task():
    task = _task(repository=REPO_B, commit=COMMIT_1)
    candidate = _cand(candidate_id="from_a", repository=REPO_A)
    selection = select_context(task, [candidate], ContextBudget())
    decisions = _decisions(selection)
    assert decisions["from_a"].selected is False
    assert decisions["from_a"].reason == REASON_OUT_OF_SCOPE
    assert decisions["from_a"].scope_match is False
    assert selection.package.selected_items == ()


def test_scoped_candidate_is_excluded_for_an_unscoped_task():
    task = _task(repository=None, commit=None)
    candidate = _cand(candidate_id="pinned", repository=REPO_A)
    assert scope_matches(task, candidate) is False
    selection = select_context(task, [candidate], ContextBudget())
    assert _decisions(selection)["pinned"].reason == REASON_OUT_OF_SCOPE


def test_scope_neutral_candidate_is_eligible_for_any_task():
    task = _task(repository=REPO_B, commit=COMMIT_1)
    candidate = _cand(
        candidate_id="neutral", repository=None, commit=None,
        source_type="tool_result", content="generic output",
    )
    selection = select_context(task, [candidate], ContextBudget())
    assert _decisions(selection)["neutral"].selected is True


def test_commit_mismatch_excludes_non_memory_candidate():
    task = _task(commit=COMMIT_1)
    candidate = _cand(candidate_id="old_state", commit=COMMIT_2)
    selection = select_context(task, [candidate], ContextBudget())
    assert _decisions(selection)["old_state"].reason == REASON_OUT_OF_SCOPE


def test_bound_engine_refuses_foreign_task_and_snapshot():
    engine = ContextEngine(repository=REPO_A)
    assert engine.repository == REPO_A
    with pytest.raises(ContextOperationError) as excinfo:
        engine.select(_task(repository=REPO_B), [])
    assert excinfo.value.code == CONTEXT_SCOPE_MISMATCH
    # unscoped task is allowed (the engine cannot contradict a scope
    # the caller did not declare) — the guard is about mismatches
    selection = engine.select(
        TaskContext(question="q", repository=None, commit=None), []
    )
    assert selection.package.repository_scope is None


def test_bound_engine_refuses_foreign_snapshot():
    engine = ContextEngine(repository=REPO_A)
    foreign_engine = ContextEngine(repository=REPO_B)
    selection = foreign_engine.select(_task(repository=REPO_B), [])
    # snapshot() against the WRONG binding fails explicitly
    with pytest.raises(ContextOperationError) as excinfo:
        engine.snapshot(selection, run_id="run_1", step_id="001")
    assert excinfo.value.code == CONTEXT_SCOPE_MISMATCH
    # and a snapshot built by a matching engine persists fine
    good = engine.snapshot(
        engine.select(_task(repository=REPO_A), []),
        run_id="run_1",
        step_id="001",
    )
    assert good.repository_scope == REPO_A


def test_cross_repository_memory_candidate_is_out_of_scope():
    task = _task(repository=REPO_A, commit=COMMIT_1)
    candidate = candidate_from_memory(
        _memory(repository=REPO_B, commit=COMMIT_1),
        current_commit=COMMIT_1,
    )
    selection = select_context(task, [candidate], ContextBudget())
    decisions = _decisions(selection)
    (decision,) = decisions.values()
    # even a FRESH memory from another repository is a scope leak —
    # isolation wins over freshness
    assert decision.reason == REASON_OUT_OF_SCOPE
    assert selection.package.selected_items == ()


# ---------------------------------------------------------------------------
# freshness (§45 negative tests 2 and 3)
# ---------------------------------------------------------------------------


def test_fresh_memory_is_selected_with_verified_reason():
    task = _task(commit=COMMIT_1)
    candidate = candidate_from_memory(
        _memory(commit=COMMIT_1), current_commit=COMMIT_1
    )
    assert candidate.freshness == "fresh"
    selection = select_context(task, [candidate], ContextBudget())
    (decision,) = _decisions(selection).values()
    assert decision.selected is True
    assert decision.reason == "VERIFIED_MEMORY"
    assert decision.freshness == "fresh"


def test_stale_memory_is_never_selected_as_current():
    # §45 negative test 2: a stale memory as current context must fail
    task = _task(commit=COMMIT_2)
    candidate = candidate_from_memory(
        _memory(commit=COMMIT_1), current_commit=COMMIT_2
    )
    assert candidate.freshness == "stale"
    selection = select_context(task, [candidate], ContextBudget())
    (decision,) = _decisions(selection).values()
    assert decision.selected is False
    assert decision.reason == REASON_STALE_MEMORY
    assert selection.package.selected_items == ()


def test_unknown_freshness_is_never_silently_fresh():
    # §45 negative test 3: missing commit on either side → unknown
    task = _task(commit=COMMIT_1)
    unknown = candidate_from_memory(
        _memory(commit=None), current_commit=COMMIT_1
    )
    assert unknown.freshness == "unknown"
    selection = select_context(task, [unknown], ContextBudget())
    (decision,) = _decisions(selection).values()
    assert decision.reason == REASON_UNKNOWN_FRESHNESS
    assert decision.selected is False

    # same when the CURRENT commit is unknown
    also_unknown = candidate_from_memory(
        _memory(commit=COMMIT_1), current_commit=None
    )
    assert also_unknown.freshness == "unknown"

    # and a candidate that simply never declared freshness is unknown
    undeclared = ContextCandidate(
        candidate_id=new_candidate_id(),
        source_type="memory",
        content="no freshness declared",
        repository_scope=REPO_A,
    )
    selection = select_context(task, [undeclared], ContextBudget())
    (decision,) = _decisions(selection).values()
    assert decision.reason == REASON_UNKNOWN_FRESHNESS


def test_verification_status_mapping_matches_m14_tri_state():
    memory = _memory(commit=COMMIT_1)
    assert verification_status(memory, COMMIT_1) == "fresh"
    assert verification_status(memory, COMMIT_2) == "stale"
    assert verification_status(memory, None) == "unknown"
    assert verification_status(
        Memory(
            memory_id="mem_x",
            owner=memory.owner,
            memory_type="episodic",
            statement="s",
            scope=MemoryScope(repository=REPO_A, commit=None),
            evidence=(),
            status="verified",
            created_at="2026-10-01T00:00:00.000Z",
            last_verified_at="2026-10-01T00:00:00.000Z",
        ),
        COMMIT_1,
    ) == "unknown"


def test_adapter_reuses_identity_scope_and_evidence():
    from repo_intelligence.evidence import EvidenceRef

    memory = _memory(commit=COMMIT_1, statement="registry exposes x")
    memory = Memory(
        memory_id="mem_ev",
        owner=memory.owner,
        memory_type=memory.memory_type,
        statement=memory.statement,
        scope=memory.scope,
        evidence=(EvidenceRef(path="agents/widget.md", line_start=4,
                              line_end=9),),
        status=memory.status,
        created_at=memory.created_at,
        last_verified_at=memory.last_verified_at,
    )
    candidate = candidate_from_memory(memory, current_commit=COMMIT_1)
    assert candidate.memory_id == "mem_ev"
    assert candidate.source_id == "mem_ev"
    assert candidate.repository_scope == REPO_A
    assert candidate.commit_scope == COMMIT_1
    assert candidate.evidence[0].path == "agents/widget.md"
    assert candidate.freshness == "fresh"


# ---------------------------------------------------------------------------
# authority hierarchy (§14)
# ---------------------------------------------------------------------------


def test_current_evidence_outranks_historical_memory():
    # budget only fits ONE of the two: evidence must win
    evidence = _cand(candidate_id="ev", content="current observation")
    memory = candidate_from_memory(
        _memory(commit=COMMIT_1, statement="historical recall"),
        current_commit=COMMIT_1,
        candidate_id="mem_cand",
    )
    task = _task()
    selection = select_context(
        task,
        [memory, evidence],
        ContextBudget(max_items=2, max_characters=25),
    )
    selected = [
        item.candidate_id
        for item in selection.package.selected_items
    ]
    # evidence (§14: current observation) is selected; the memory is
    # budget-blocked with a RECORDED reason, never a silent drop
    assert selected == ["ev"]
    decisions = _decisions(selection)
    assert decisions["mem_cand"].selected is False
    assert decisions["mem_cand"].reason in (
        "LOW_PRIORITY", "BUDGET_EXCEEDED"
    )


def test_evidence_authority_outranks_memory_in_sort_order():
    evidence = _cand(candidate_id="ev", content="e")
    memory = candidate_from_memory(
        _memory(commit=COMMIT_1), current_commit=COMMIT_1,
        candidate_id="mem_cand",
    )
    selection = select_context(
        _task(), [memory, evidence], ContextBudget()
    )
    selected = [
        item.candidate_id for item in selection.package.selected_items
    ]
    # presentation: evidence section before memory section
    assert selected.index("ev") < selected.index("mem_cand")


def test_engine_with_impossible_tiny_budget_fails_explicitly():
    engine = ContextEngine(budget=ContextBudget(max_characters=2))
    with pytest.raises(ContextOperationError) as excinfo:
        engine.select(_task(), [_cand(content="too long to fit")])
    assert excinfo.value.code == CONTEXT_BUDGET_EXCEEDED


def test_candidate_from_task_carries_scope_and_question():
    task = _task()
    candidate = candidate_from_task(task)
    assert candidate.source_type == "task"
    assert candidate.content == task.question
    assert candidate.repository_scope == REPO_A
    assert candidate.commit_scope == COMMIT_1
    selection = select_context(task, [candidate], ContextBudget())
    item = selection.package.selected_items[0]
    assert item.selection_reason == "TASK_DIRECT"
