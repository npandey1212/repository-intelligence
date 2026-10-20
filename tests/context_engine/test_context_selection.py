"""M15 — deterministic selection policy (§7/§12/§13/§22/§23/§30).

The pipeline, stage by stage, over scripted candidates (no LLM):

 * one SELECT reason per source class, one decision per candidate
 * presentation order: documented section order, authority, priority
 * DETERMINISM: same inputs → identical items, ids, order, decisions,
   summary (only context_id/timestamps vary)
 * exact-identity dedup (first winner keeps its place; later
   duplicate → DUPLICATE with ``duplicate_of``)
 * budget: exact boundary fits, one-char-over excluded
   BUDGET_EXCEEDED (never a silent overage), item-count cut →
   LOW_PRIORITY, eligible-but-nothing-fits → structured
   CONTEXT_BUDGET_EXCEEDED raise (the §45 negative test)
 * priority breaks authority ties; untagged is relevance-neutral
 * evaluation metrics + §47 report are deterministic counts/ratios

Everything synthetic (``example-repository``) and LLM-free.
"""

from __future__ import annotations

import pytest

from repo_intelligence.context_engine import (
    CONTEXT_BUDGET_EXCEEDED,
    CONTEXT_CONTRACT_VIOLATION,
    ContextBudget,
    ContextCandidate,
    ContextOperationError,
    REASON_BUDGET_EXCEEDED,
    REASON_DUPLICATE,
    REASON_IRRELEVANT,
    REASON_LOW_PRIORITY,
    REASON_OUT_OF_SCOPE,
    REASON_STALE_MEMORY,
    REASON_UNKNOWN_FRESHNESS,
    SELECTION_REASON_BY_SOURCE,
    TaskContext,
    evaluate_context,
    new_candidate_id,
    render_evaluation_report,
    select_context,
)

REPO = "example-repository"
COMMIT = "a" * 40


def _task(**overrides) -> TaskContext:
    fields = dict(
        question="How is the widget registry wired?",
        repository=REPO,
        commit=COMMIT,
    )
    fields.update(overrides)
    return TaskContext(**fields)


def _cand(**overrides) -> ContextCandidate:
    fields = dict(
        candidate_id=new_candidate_id(),
        source_type="evidence",
        content="content {}".format(overrides.get("candidate_id", "?")),
        repository_scope=REPO,
        commit_scope=COMMIT,
    )
    fields.update(overrides)
    return ContextCandidate(**fields)


def _fixed(cid, **overrides) -> ContextCandidate:
    return _cand(candidate_id=cid, **overrides)


def _select(candidates, task=None, budget=None, step_id="001"):
    return select_context(
        task if task is not None else _task(),
        candidates,
        budget if budget is not None else ContextBudget(),
        step_id,
    )


def _by_candidate(selection):
    return {
        decision.candidate_id: decision
        for decision in selection.decisions
    }


# ---------------------------------------------------------------------------
# reasons + decisions
# ---------------------------------------------------------------------------


def test_one_select_reason_per_source_class():
    task = _task()
    candidates = [
        _fixed("c_task", source_type="task", content="the question"),
        _fixed("c_repo", source_type="repository", content="readme line"),
        _fixed("c_fact", source_type="fact", content="a fact"),
        _fixed("c_graph", source_type="graph", content="a edge"),
        _fixed("c_evidence", source_type="evidence", content="evidence"),
        _fixed("c_memory", source_type="memory", content="memory",
               freshness="fresh"),
        _fixed("c_tool", source_type="tool_result", content="output"),
        _fixed("c_state", source_type="agent_state", content="state"),
    ]
    selection = _select(candidates, task=task)
    decisions = _by_candidate(selection)
    assert set(decisions) == {
        "c_task", "c_repo", "c_fact", "c_graph", "c_evidence", "c_memory",
        "c_tool", "c_state",
    }
    for candidate in candidates:
        decision = decisions[candidate.candidate_id]
        assert decision.selected is True
        assert decision.reason == SELECTION_REASON_BY_SOURCE[
            candidate.source_type
        ]
    assert len(selection.package.selected_items) == 8


def test_exclusion_reasons_cover_every_stage():
    task = _task(relevance=("widget",))
    candidates = [
        _fixed("c_ok", content="matches", relevance=("widget",)),
        _fixed("c_scope", source_type="memory",
               repository_scope="other-repo"),
        _fixed("c_stale", source_type="memory", freshness="stale"),
        _fixed("c_unknown", source_type="memory"),  # no freshness → unknown
        _fixed("c_irrelevant", source_type="fact",
               relevance=("unrelated",)),
        _fixed("c_dup", source_type="memory", source_id="same-record",
               freshness="fresh"),
        _fixed("c_dup2", source_type="memory", source_id="same-record",
               freshness="fresh"),
    ]
    selection = _select(candidates, task=task)
    decisions = _by_candidate(selection)
    assert decisions["c_ok"].selected is True
    assert decisions["c_scope"].reason == REASON_OUT_OF_SCOPE
    assert decisions["c_stale"].reason == REASON_STALE_MEMORY
    assert decisions["c_unknown"].reason == REASON_UNKNOWN_FRESHNESS
    assert decisions["c_irrelevant"].reason == REASON_IRRELEVANT
    assert decisions["c_dup2"].reason == REASON_DUPLICATE
    assert decisions["c_dup2"].duplicate_of == "c_dup"
    # exactly one decision per candidate, input order preserved
    assert [d.candidate_id for d in selection.decisions] == [
        c.candidate_id for c in candidates
    ]
    # excluded candidates never become items; the dedup WINNER does
    item_ids = {item.candidate_id for item in
                selection.package.selected_items}
    assert item_ids == {"c_ok", "c_dup"}


def test_untagged_candidates_are_relevance_neutral():
    task = _task(relevance=("widget",))
    candidates = [
        _fixed("c_tagged", relevance=("widget",)),
        _fixed("c_untagged"),  # no tags → neutral, not irrelevant
        _fixed("c_task_no_tags"),  # task tags exist, candidate none
    ]
    selection = _select(candidates, task=task)
    decisions = _by_candidate(selection)
    assert all(d.selected for d in decisions.values())


# ---------------------------------------------------------------------------
# presentation order + priority
# ---------------------------------------------------------------------------


def test_presentation_follows_documented_section_order():
    task = _task()
    candidates = [
        _fixed("c_tool", source_type="tool_result", content="z output"),
        _fixed("c_memory", source_type="memory", content="m",
               freshness="fresh"),
        _fixed("c_repo", source_type="repository", content="r"),
        _fixed("c_task", source_type="task", content="q"),
        _fixed("c_evidence", source_type="evidence", content="e"),
    ]
    selection = _select(candidates, task=task)
    order = [item.source_type for item in
             selection.package.selected_items]
    assert order == ["task", "repository", "evidence", "memory",
                     "tool_result"]
    # positional ids follow presentation order
    assert [
        item.context_item_id for item in selection.package.selected_items
    ] == ["ctxitem_001", "ctxitem_002", "ctxitem_003", "ctxitem_004",
          "ctxitem_005"]


def test_priority_breaks_authority_ties_inside_a_section():
    task = _task()
    candidates = [
        _fixed("c_low", source_type="evidence", content="low",
               priority=1),
        _fixed("c_high", source_type="evidence", content="high",
               priority=10),
    ]
    selection = _select(candidates, task=task)
    assert [
        item.candidate_id for item in selection.package.selected_items
    ] == ["c_high", "c_low"]


def test_authority_ranks_sections_within_budget_tightness():
    # budget only fits the two smallest: evidence + memory beat
    # tool_result under §14, and the section order still shows.
    candidates = [
        _fixed("c_tool", source_type="tool_result", content="t" * 8),
        _fixed("c_memory", source_type="memory", content="m" * 8,
               freshness="fresh"),
        _fixed("c_evidence", source_type="evidence", content="e" * 8),
    ]
    budget = ContextBudget(max_items=2, max_characters=100)
    selection = _select(candidates, budget=budget)
    selected = {item.candidate_id for item in
                selection.package.selected_items}
    assert selected == {"c_evidence", "c_memory"}
    decisions = _by_candidate(selection)
    assert decisions["c_tool"].reason == REASON_LOW_PRIORITY


# ---------------------------------------------------------------------------
# determinism (§45 negative test: nondeterministic selection)
# ---------------------------------------------------------------------------


def test_same_inputs_produce_identical_selection():
    def build():
        return [
            _fixed("c_1", source_type="task", content="q"),
            _fixed("c_2", content="evidence", priority=5),
            _fixed("c_3", source_type="memory", content="m",
                   freshness="fresh"),
            _fixed("c_4", source_type="repository", content="r"),
        ]

    first = _select(build(), step_id="001")
    second = _select(build(), step_id="001")
    assert first.package.selected_items == second.package.selected_items
    assert first.decisions == second.decisions
    assert first.package.selection_summary == (
        second.package.selection_summary
    )
    assert first.package.budget == second.package.budget
    assert first.package.step_id == second.package.step_id
    # positional item identity is content-derived, not clock-derived
    assert [i.context_item_id for i in first.package.selected_items] == [
        i.context_item_id for i in second.package.selected_items
    ]


def test_selection_is_independent_of_candidate_input_order():
    candidates = [
        _fixed("c_1", source_type="task", content="q"),
        _fixed("c_2", content="evidence"),
        _fixed("c_3", source_type="memory", content="m",
               freshness="fresh"),
    ]
    forward = _select(list(candidates))
    backward = _select(list(reversed(candidates)))
    assert [i.candidate_id for i in forward.package.selected_items] == [
        i.candidate_id for i in backward.package.selected_items
    ]


# ---------------------------------------------------------------------------
# budget (§12 — never silently exceeded)
# ---------------------------------------------------------------------------


def test_budget_exact_boundary_fits():
    budget = ContextBudget(max_items=10, max_characters=10)
    fits = _select([_fixed("c1", content="a" * 10)], budget=budget)
    assert fits.package.budget.estimated_characters == 10
    assert fits.package.selected_items[0].candidate_id == "c1"


def test_budget_one_char_over_is_excluded_when_something_fits():
    budget = ContextBudget(max_items=10, max_characters=10)
    selection = _select(
        [
            _fixed("small", source_type="task", content="s" * 4),
            _fixed("big", source_type="evidence", content="a" * 11),
        ],
        budget=budget,
    )
    decisions = _by_candidate(selection)
    assert decisions["small"].selected is True
    assert decisions["big"].reason == REASON_BUDGET_EXCEEDED
    assert selection.package.budget.estimated_characters == 4
    assert selection.package.budget.estimated_characters <= budget.max_characters


def test_budget_overflow_is_recorded_never_silent():
    budget = ContextBudget(max_items=10, max_characters=25)
    candidates = [
        _fixed("big", content="b" * 20),
        _fixed("small", content="s" * 5),
    ]
    selection = _select(candidates, budget=budget)
    decisions = _by_candidate(selection)
    # greedy in selection order: big fits first, small must fit too
    assert decisions["big"].selected is True
    assert decisions["small"].selected is True
    assert selection.package.budget.estimated_characters == 25
    # with one char less, the later candidate is EXCLUDED with a
    # recorded reason — the package never exceeds the ceiling
    selection = _select(
        [
            _fixed("small", source_type="task", content="s" * 5),
            _fixed("big", source_type="evidence", content="b" * 20),
        ],
        budget=ContextBudget(max_items=10, max_characters=24),
    )
    decisions = _by_candidate(selection)
    assert decisions["big"].reason == REASON_BUDGET_EXCEEDED
    assert selection.package.budget.estimated_characters == 5


def test_eligible_but_nothing_fits_raises_structured_budget_error():
    # §45 negative test: the budget can never be silently exceeded,
    # and an impossible budget is an EXPLICIT failure, not an empty
    # package.
    budget = ContextBudget(max_items=5, max_characters=3)
    with pytest.raises(ContextOperationError) as excinfo:
        _select([_fixed("c1", content="way too long")], budget=budget)
    assert excinfo.value.code == CONTEXT_BUDGET_EXCEEDED
    assert "refusing" in str(excinfo.value)


def test_empty_candidate_list_yields_empty_package_not_an_error():
    selection = _select([])
    assert selection.package.selected_items == ()
    assert selection.decisions == ()
    assert selection.package.selection_summary["considered"] == 0
    assert selection.package.budget.estimated_characters == 0


def test_all_excluded_yields_empty_package_without_error():
    candidates = [_fixed("c1", repository_scope="elsewhere")]
    selection = _select(candidates)
    assert selection.package.selected_items == ()
    assert selection.package.selection_summary["excluded"] == 1


# ---------------------------------------------------------------------------
# contract guards (duplicate ids, bad inputs)
# ---------------------------------------------------------------------------


def test_duplicate_candidate_ids_are_rejected():
    candidates = [_fixed("same"), _fixed("same")]
    with pytest.raises(ContextOperationError) as excinfo:
        _select(candidates)
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


@pytest.mark.parametrize("bad_input", ["not-a-sequence", {"a": 1}, 7])
def test_non_sequence_candidates_are_rejected(bad_input):
    with pytest.raises(ContextOperationError) as excinfo:
        _select(bad_input)
    assert excinfo.value.code == CONTEXT_CONTRACT_VIOLATION


def test_invalid_candidate_fails_the_whole_selection():
    # a non-candidate inside the sequence — validate_candidate is the
    # application-side authority and rejects it with its own code
    candidates = [_fixed("c1"), "not-a-candidate"]
    with pytest.raises(ContextOperationError) as excinfo:
        _select(candidates)
    assert excinfo.value.code == "CONTEXT_INVALID_CANDIDATE"


# ---------------------------------------------------------------------------
# summary + evaluation metrics (§46/§47)
# ---------------------------------------------------------------------------


def test_selection_summary_counts():
    candidates = [
        _fixed("c_task", source_type="task", content="q"),
        _fixed("c_ev", source_type="evidence", content="e"),
        _fixed("c_scope", source_type="memory",
               repository_scope="other"),
        _fixed("c_dup_a", source_type="memory", source_id="m1",
               freshness="fresh"),
        _fixed("c_dup_b", source_type="memory", source_id="m1",
               freshness="fresh"),
    ]
    selection = _select(candidates)
    summary = selection.package.selection_summary
    assert summary["considered"] == 5
    assert summary["selected"] == 3
    assert summary["excluded"] == 2
    assert summary["duplicates"] == 1
    assert summary["budget_blocked"] == 0
    assert summary["by_source"] == {"task": 1, "evidence": 1, "memory": 1}
    assert summary["by_reason"][REASON_OUT_OF_SCOPE] == 1
    assert summary["by_reason"][REASON_DUPLICATE] == 1


def test_evaluate_context_metrics_are_deterministic():
    candidates = [
        _fixed("c_task", source_type="task", content="q"),
        _fixed("c_mem", source_type="memory", content="m",
               freshness="fresh"),
        _fixed("c_stale", source_type="memory", content="s",
               freshness="stale"),
        _fixed("c_scope", source_type="graph",
               repository_scope="elsewhere"),
    ]
    selection = _select(candidates)
    first = evaluate_context(selection)
    second = evaluate_context(selection)
    assert first == second
    assert first["considered"] == 4
    assert first["selected"] == 2
    assert first["excluded"] == 2
    assert first["memory_items_selected"] == 1
    assert first["stale_items_excluded"] == 1
    assert first["out_of_scope_excluded"] == 1
    assert first["isolation_ok"] is True
    assert 0 <= first["budget_utilization"] <= 100
    report = render_evaluation_report(first)
    for line in (
        "Context evaluation report",
        "* Candidates considered: 4",
        "* Selected: 2",
        "* Excluded: 2",
        "* Budget utilization:",
        "* Duplicates: 0",
        "* Evidence coverage:",
        "* Isolation: PASS",
    ):
        assert line in report


def test_evidence_coverage_counts_selected_items_with_refs():
    from repo_intelligence.evidence import EvidenceRef

    candidates = [
        _fixed("c_ev", source_type="evidence", content="e",
               evidence=(EvidenceRef(path="a.json", line_start=1,
                                     line_end=2),)),
        _fixed("c_task", source_type="task", content="q"),
    ]
    metrics = evaluate_context(_select(candidates))
    assert metrics["selected"] == 2
    assert metrics["evidence_coverage"] == 50.0
