"""M15 — context contract + candidate validation (§8/§9/§10/§12).

Contract enforcement is the application's authority — never the
model's:

 * TaskContext: question required, scope optional, tag sequences
   validated (CONTEXT_CONTRACT_VIOLATION)
 * ContextCandidate: source_type vocabulary, non-empty content,
   valid freshness/authority/priority/evidence — one structured code
   (CONTEXT_INVALID_CANDIDATE)
 * authority resolution: source default from the §14 hierarchy, an
   explicit override wins
 * identity helpers: positional ``ctxitem_<NNN>`` ids, opaque
   ``ctx_``/``cand_``/``snap_`` ids, uniqueness
 * ContextBudget: hard validation of every component + remaining
   properties + ``with_usage``
 * ContextPackage: budget-usage invariant (recorded usage IS the
   items, never over max_characters)
 * ContextSnapshot: excluded ids MUST be exactly the non-selected
   decisions
 * to_dict: the documented JSON projections (canonical shapes)
 * reason/source vocabularies are closed and disjoint

Everything synthetic (``example-repository``) and LLM-free.
"""

from __future__ import annotations

import pytest

from repo_intelligence.context_engine import (
    CONTEXT_CONTRACT_VIOLATION,
    CONTEXT_FRESHNESS_STATUSES,
    CONTEXT_INVALID_CANDIDATE,
    CONTEXT_SOURCE_TYPES,
    DEFAULT_AUTHORITY_BY_SOURCE,
    EXCLUSION_REASONS,
    SELECTION_REASON_BY_SOURCE,
    SELECTION_REASONS,
    AUTHORITY_RANKS,
    ContextBudget,
    ContextCandidate,
    ContextItem,
    ContextOperationError,
    ContextPackage,
    ContextSelection,
    ContextSelectionDecision,
    ContextSnapshot,
    TaskContext,
    new_candidate_id,
    new_context_id,
    new_item_id,
    new_snapshot_id,
    reason_for,
    section_rank,
    validate_candidate,
)
from repo_intelligence.evidence import EvidenceRef

REPO = "example-repository"
COMMIT = "a" * 40


def _candidate(**overrides) -> ContextCandidate:
    fields = dict(
        candidate_id="cand_1",
        source_type="evidence",
        content="widget registry declares widget:sync",
        source_id="agents/widget.md",
        repository_scope=REPO,
        commit_scope=COMMIT,
        evidence=(EvidenceRef(path="agents/widget.md", line_start=4,
                              line_end=9),),
    )
    fields.update(overrides)
    return ContextCandidate(**fields)


# ---------------------------------------------------------------------------
# TaskContext
# ---------------------------------------------------------------------------


def test_task_context_valid_is_accepted():
    task = TaskContext(
        question="How does the widget registry work?",
        repository=REPO,
        commit=COMMIT,
        open_questions=("who owns the registry?",),
        constraints=("do not modify tools/",),
        relevance=("widget",),
    )
    assert task.question.startswith("How does")
    assert task.open_questions == ("who owns the registry?",)
    assert task.relevance == ("widget",)


@pytest.mark.parametrize(
    "overrides",
    [
        {"question": ""},
        {"question": "   "},
        {"question": 42},
        {"repository": ""},
        {"commit": "  "},
        {"open_questions": "not-a-sequence"},
        {"open_questions": ("",)},
        {"constraints": ("",)},
        {"relevance": (42,)},
        {"relevance": "tag-as-string"},
    ],
)
def test_task_context_contract_violations(overrides):
    fields = dict(question="valid")
    fields.update(overrides)
    with pytest.raises(ContextOperationError) as excinfo:
        TaskContext(**fields)
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


# ---------------------------------------------------------------------------
# candidate validation
# ---------------------------------------------------------------------------


def test_valid_candidate_is_accepted_unchanged():
    candidate = _candidate()
    assert validate_candidate(candidate) is candidate


def test_non_candidate_is_rejected_with_structured_code():
    with pytest.raises(ContextOperationError) as excinfo:
        validate_candidate({"content": "a mapping, not a candidate"})
    assert excinfo.value.code == CONTEXT_INVALID_CANDIDATE


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_id": ""},
        {"candidate_id": "   "},
        {"source_type": "dream"},
        {"source_type": ""},
        {"source_type": 7},
        {"content": ""},
        {"content": "   "},
        {"content": None},
        {"source_id": ""},
        {"repository_scope": "  "},
        {"commit_scope": ""},
        {"memory_id": ""},
        {"authority": "telepathy"},
        {"freshness": "warm"},
        {"freshness": "yes"},
        {"priority": "high"},
        {"priority": True},
        {"relevance": "not-a-sequence"},
        {"relevance": ("",)},
        {"evidence": "path-as-string"},
        {"evidence": ("agents/widget.md",)},
        {"evidence": (EvidenceRef(path=""),)},
        {"evidence": (EvidenceRef(path="x", line_start=0),)},
        {"evidence": (EvidenceRef(path="x", line_start=5, line_end=2),)},
        {"created_at": ""},
    ],
)
def test_candidate_contract_violations(overrides):
    with pytest.raises(ContextOperationError) as excinfo:
        _candidate(**overrides)
    assert excinfo.value.code == CONTEXT_INVALID_CANDIDATE


def test_every_source_type_has_a_default_authority_and_reason():
    for source_type in CONTEXT_SOURCE_TYPES:
        assert source_type in DEFAULT_AUTHORITY_BY_SOURCE
        assert source_type in SELECTION_REASON_BY_SOURCE
        authority = DEFAULT_AUTHORITY_BY_SOURCE[source_type]
        assert authority in AUTHORITY_RANKS
        reason = SELECTION_REASON_BY_SOURCE[source_type]
        assert reason in SELECTION_REASONS


def test_reason_and_source_vocabularies_are_disjoint():
    assert set(SELECTION_REASONS) & set(EXCLUSION_REASONS) == set()
    assert set(SELECTION_REASONS) | set(EXCLUSION_REASONS)
    assert set(CONTEXT_FRESHNESS_STATUSES) == {"fresh", "stale", "unknown"}


def test_authority_resolution_defaults_and_override():
    memory = _candidate(source_type="memory")
    assert memory.authority_value == "memory"
    assert memory.authority_rank == AUTHORITY_RANKS["memory"]
    override = _candidate(source_type="memory", authority="evidence")
    assert override.authority_value == "evidence"
    assert override.authority_rank == AUTHORITY_RANKS["evidence"]
    # §14 ordering: evidence > graph/fact > memory > tool_result >
    # inference
    assert (
        AUTHORITY_RANKS["evidence"]
        > AUTHORITY_RANKS["graph_fact"]
        > AUTHORITY_RANKS["memory"]
        > AUTHORITY_RANKS["tool_result"]
        > AUTHORITY_RANKS["inference"]
    )


def test_reason_for_is_source_based():
    assert reason_for(_candidate(source_type="task")) == "TASK_DIRECT"
    assert reason_for(_candidate(source_type="graph")) == "GRAPH_NEIGHBOR"
    assert reason_for(_candidate(source_type="tool_result")) == (
        "RECENT_TOOL_RESULT"
    )


def test_section_rank_covers_sources_and_ranks_unknown_last():
    for source_type in CONTEXT_SOURCE_TYPES:
        assert section_rank(source_type) in range(10, 91)
    assert section_rank("task") < section_rank("memory")
    assert section_rank("evidence") < section_rank("memory")
    assert section_rank("mystery") > section_rank("tool_result")


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------


def test_ids_are_opaque_and_prefixed():
    context_id = new_context_id()
    snapshot_id = new_snapshot_id()
    candidate_id = new_candidate_id()
    assert context_id.startswith("ctx_")
    assert snapshot_id.startswith("snap_")
    assert candidate_id.startswith("cand_")
    assert len({context_id, snapshot_id, candidate_id}) == 3


def test_item_ids_are_positional_and_deterministic():
    assert new_item_id(1) == "ctxitem_001"
    assert new_item_id(12) == "ctxitem_012"
    assert new_item_id(1) == new_item_id(1)


@pytest.mark.parametrize("position", [0, -1, "1", True, None])
def test_item_ids_reject_invalid_positions(position):
    with pytest.raises(ContextOperationError) as excinfo:
        new_item_id(position)
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_budget_defaults_and_remaining_properties():
    budget = ContextBudget()
    assert budget.max_items == 20
    assert budget.max_characters == 12000
    assert budget.remaining_items == 20
    assert budget.remaining_characters == 12000
    filled = budget.with_usage(3, 450)
    assert filled.selected_items == 3
    assert filled.estimated_characters == 450
    assert filled.remaining_items == 17
    assert filled.remaining_characters == 12000 - 450
    assert filled.max_items == 20
    # original unchanged (frozen + pure helper)
    assert budget.selected_items == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_items": 0},
        {"max_items": -1},
        {"max_items": "20"},
        {"max_items": True},
        {"max_characters": 0},
        {"max_characters": -5},
        {"selected_items": -1},
        {"estimated_characters": -1},
    ],
)
def test_budget_contract_violations(overrides):
    with pytest.raises(ContextOperationError) as excinfo:
        ContextBudget(**overrides)
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_budget_to_dict_shape():
    payload = ContextBudget(
        max_items=10, max_characters=500, selected_items=2,
        estimated_characters=120,
    ).to_dict()
    assert payload == {
        "max_items": 10,
        "selected_items": 2,
        "remaining_items": 8,
        "max_characters": 500,
        "estimated_characters": 120,
        "remaining_characters": 380,
    }


# ---------------------------------------------------------------------------
# package / selection / snapshot invariants
# ---------------------------------------------------------------------------


def _item(item_id="ctxitem_001", content="x" * 10) -> ContextItem:
    return ContextItem(
        context_item_id=item_id,
        candidate_id="cand_1",
        source_type="task",
        content=content,
        selection_reason="TASK_DIRECT",
        authority="task",
    )


def _package(items=(), max_chars=1000, max_items=5) -> ContextPackage:
    estimated = sum(len(item.content) for item in items)
    return ContextPackage(
        context_id="ctx_fixed",
        task="question?",
        step_id="001",
        budget=ContextBudget(
            max_items=max_items,
            max_characters=max_chars,
            selected_items=len(items),
            estimated_characters=estimated,
        ),
        selected_items=tuple(items),
    )


def test_package_budget_usage_mismatch_is_rejected():
    with pytest.raises(ContextOperationError) as excinfo:
        ContextPackage(
            context_id="ctx_x",
            task="q",
            step_id="001",
            budget=ContextBudget(selected_items=3,
                                  estimated_characters=0),
        )
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_package_estimated_mismatch_and_over_max_are_rejected():
    with pytest.raises(ContextOperationError):
        ContextPackage(
            context_id="ctx_x",
            task="q",
            step_id="001",
            budget=ContextBudget(
                max_characters=5,
                selected_items=1,
                estimated_characters=4,  # item below is 10 chars
            ),
            selected_items=(_item(),),
        )
    with pytest.raises(ContextOperationError):
        ContextPackage(
            context_id="ctx_x",
            task="q",
            step_id="001",
            budget=ContextBudget(
                max_characters=5,
                selected_items=1,
                estimated_characters=10,
            ),
            selected_items=(_item(),),
        )


def test_package_valid_and_projection_shape():
    package = _package(items=(_item(),))
    assert package.selected_items[0].context_item_id == "ctxitem_001"
    payload = package.to_dict()
    assert payload["task"] == {"question": "question?"}
    assert payload["step_id"] == "001"
    assert payload["budget"]["estimated_characters"] == 10
    assert len(payload["selected_items"]) == 1
    assert payload["selected_items"][0]["selection_reason"] == (
        "TASK_DIRECT"
    )


def _decision(candidate_id="cand_1", selected=True) -> ContextSelectionDecision:
    return ContextSelectionDecision(
        candidate_id=candidate_id,
        selected=selected,
        reason="TASK_DIRECT" if selected else "OUT_OF_SCOPE",
        authority="task",
        scope_match=True,
        priority=0,
        budget_cost=10,
    )


def test_selection_excludes_non_selected_decision_ids():
    selection = ContextSelection(
        package=_package(),
        decisions=(_decision(), _decision("cand_2", selected=False)),
    )
    assert selection.excluded_candidate_ids == ("cand_2",)
    payload = selection.to_dict()
    assert payload["excluded_candidates"] == ["cand_2"]
    assert len(payload["decisions"]) == 2


def test_snapshot_requires_excluded_to_match_decisions():
    package = _package()
    snapshot = ContextSnapshot(
        snapshot_id="snap_1",
        run_id="run_1",
        step_id="001",
        created_at="2026-10-01T00:00:00.000Z",
        package=package,
        selection_decisions=(_decision(), _decision("cand_2",
                                                    selected=False)),
        excluded_candidates=("cand_2",),
    )
    assert snapshot.excluded_candidates == ("cand_2",)
    assert snapshot.context_id == package.context_id
    assert snapshot.task == package.task
    assert snapshot.selected_items == package.selected_items
    with pytest.raises(ContextOperationError) as excinfo:
        ContextSnapshot(
            snapshot_id="snap_2",
            run_id="run_1",
            step_id="001",
            created_at="2026-10-01T00:00:00.000Z",
            package=package,
            selection_decisions=(_decision(), _decision("cand_2",
                                                        selected=False)),
            excluded_candidates=(),  # under-reports the exclusion
        )
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_snapshot_projection_covers_minimum_fields():
    package = _package(items=(_item(),))
    snapshot = ContextSnapshot(
        snapshot_id="snap_1",
        run_id="run_1",
        step_id="001",
        created_at="2026-10-01T00:00:00.000Z",
        package=package,
        selection_decisions=(_decision(),),
        excluded_candidates=(),
    )
    payload = snapshot.to_dict()
    for key in (
        "snapshot_id",
        "context_id",
        "run_id",
        "step_id",
        "created_at",
        "task",
        "repository_scope",
        "commit_scope",
        "selected_items",
        "selection_decisions",
        "excluded_candidates",
        "budget",
    ):
        assert key in payload
    assert payload["task"] == {"question": "question?"}
