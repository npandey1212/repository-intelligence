"""The M15 deterministic selection policy — explicit rules, no model.

One function (``select_context``) runs the documented pipeline over
the candidates and returns the full audit:

    1. VALIDATE        every candidate against the contract
                       (CONTEXT_INVALID_CANDIDATE — never repaired,
                       never coerced) + unique candidate ids
    2. SCOPE           repository/commit filter — a candidate pinned
                       to another repository (or an unscoped task
                       that cannot host a pinned candidate) is
                       excluded OUT_OF_SCOPE (M15 §4)
    3. FRESHNESS       tri-state, preserved exactly (M15 §24):
                       stale → STALE_MEMORY, unknown →
                       UNKNOWN_FRESHNESS. A memory without a known
                       commit is UNKNOWN — never silently fresh.
    4. RELEVANCE       candidate tags vs task tags: both tagged and
                       disjoint → IRRELEVANT; untagged is neutral
    5. ORDER           deterministic sort key — scope rank, then
                       §14 authority rank, then task-relevance match,
                       then priority, then source type + source
                       identity/content (total order; never opaque
                       id randomness, M15 §30/§24)
    6. DEDUP           exact identity only (source_type + source_id,
                       else source_type + content) — first winner in
                       sort order keeps its place, later duplicates
                       are excluded DUPLICATE (M15 §22)
    7. BUDGET          greedy fill under max_items + max_characters
                       (M15 §12): count-exhausted → LOW_PRIORITY,
                       character overflow → BUDGET_EXCEEDED. If
                       eligible candidates exist and NOT ONE fits the
                       character budget, the policy raises
                       CONTEXT_BUDGET_EXCEEDED — an empty package is
                       only ever the honest result of zero eligible
                       candidates, never of a silent overflow.
    8. PRESENTATION    selected items re-ordered into the documented
                       section order (M15 §23) and given positional
                       ``ctxitem_<NNN>`` ids — the model reads the
                       package in exactly this order.

Every candidate receives exactly one decision (selected OR excluded,
with a stable reason). Same inputs → same items, same order, same
reasons, same decisions: only ids/timestamps vary (M15 §30). This
module imports no LLM, no database, no filesystem — the policy is a
pure function.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import (
    CONTEXT_BUDGET_EXCEEDED,
    CONTEXT_CONTRACT_VIOLATION,
    ContextOperationError,
)
from .models import (
    CONTEXT_FRESHNESS_STALE,
    CONTEXT_FRESHNESS_UNKNOWN,
    ContextBudget,
    ContextCandidate,
    ContextItem,
    ContextPackage,
    ContextSelection,
    ContextSelectionDecision,
    REASON_BUDGET_EXCEEDED,
    REASON_DUPLICATE,
    REASON_IRRELEVANT,
    REASON_LOW_PRIORITY,
    REASON_OUT_OF_SCOPE,
    REASON_STALE_MEMORY,
    REASON_UNKNOWN_FRESHNESS,
    TaskContext,
    new_context_id,
    new_item_id,
    reason_for,
    section_rank,
    validate_candidate,
)

__all__ = [
    "dedup_key",
    "freshness_of",
    "presentation_sort_key",
    "relevance_matches",
    "scope_matches",
    "select_context",
]


def scope_matches(task: TaskContext, candidate: ContextCandidate) -> bool:
    """Stage 2 — repository/commit scope filter (M15 §4).

    * candidate with NO repository scope → scope-neutral: it makes no
      repository-specific claim, so it is eligible for any task;
    * candidate pinned to a repository → the task must declare the
      SAME repository; an unscoped task cannot host a pinned
      candidate (nothing verifies it) and a differently-scoped task
      must never receive it (isolation invariant);
    * non-memory candidates additionally: a commit both known and
      different → out of scope (the content describes another state).

    Memory candidates are NOT commit-checked here: their commit
    difference is exactly the freshness stage's job, so the reason a
    caller sees is STALE_MEMORY (informative) rather than
    OUT_OF_SCOPE — repository mismatch still excludes them first.
    """
    if candidate.repository_scope is None:
        return True
    if task.repository is None:
        return False
    if candidate.repository_scope != task.repository:
        return False
    if candidate.source_type != "memory":
        if candidate.commit_scope is not None and task.commit is not None:
            if candidate.commit_scope != task.commit:
                return False
    return True


def freshness_of(candidate: ContextCandidate) -> Optional[str]:
    """Stage 3 input — the candidate's effective freshness.

    Memory candidates default to UNKNOWN when no freshness was
    positively established (M15 §24 — absence of a comparison is
    never fresh). Non-memory candidates carry freshness only when a
    caller explicitly declared it.
    """
    if candidate.source_type == "memory":
        if candidate.freshness is None:
            return CONTEXT_FRESHNESS_UNKNOWN
        return candidate.freshness
    return candidate.freshness


def relevance_matches(task: TaskContext, candidate: ContextCandidate) -> bool:
    """Stage 4 — task-relevance filter (M15 §13): only a candidate
    whose tags are KNOWN and ALL disjoint with the task's tags is
    irrelevant; untagged on either side is neutral (absence of tags
    is not evidence of irrelevance)."""
    if not candidate.relevance or not task.relevance:
        return True
    return bool(set(candidate.relevance) & set(task.relevance))


def dedup_key(candidate: ContextCandidate) -> Tuple[str, str]:
    """Stage 6 — exact identity (M15 §22): same source record when
    there is one, else identical content within the same source
    type. Never semantic similarity — two different memories saying
    similar things are NOT duplicates."""
    if candidate.source_id is not None:
        return (candidate.source_type, candidate.source_id)
    return (candidate.source_type, candidate.content)


def selection_sort_key(
    task: TaskContext, candidate: ContextCandidate
) -> Tuple[Any, ...]:
    """Stage 5 — the total selection order (M15 §30): higher scope
    rank, higher §14 authority, task-relevance match, higher
    priority first; then source type and the candidate's SOURCE
    identity + content so equal-priority candidates can never swap
    between runs. Opaque candidate ids stay unique but play no part:
    §24 forbids order from depending on id randomness."""
    if task.repository is not None and (
        candidate.repository_scope == task.repository
    ):
        scope_rank = 2
    else:
        scope_rank = 1
    relevance_match = 0
    if relevance_matches(task, candidate) and (
        task.relevance and candidate.relevance
    ):
        relevance_match = 1
    return (
        -scope_rank,
        -candidate.authority_rank,
        -relevance_match,
        -candidate.priority,
        candidate.source_type,
        candidate.source_id or "",
        candidate.content,
    )


def presentation_sort_key(candidate: ContextCandidate) -> Tuple[Any, ...]:
    """Stage 8 — documented presentation order (M15 §23): section
    rank first, then authority/priority and source identity inside
    the section (never the opaque id — §24 determinism)."""
    return (
        section_rank(candidate.source_type),
        -candidate.authority_rank,
        -candidate.priority,
        candidate.source_id or "",
        candidate.content,
    )


def _excluded(
    candidate: ContextCandidate, reason: str, scope_ok: bool,
    duplicate_of: Optional[str] = None,
) -> ContextSelectionDecision:
    return ContextSelectionDecision(
        candidate_id=candidate.candidate_id,
        selected=False,
        reason=reason,
        authority=candidate.authority_value,
        scope_match=scope_ok,
        freshness=freshness_of(candidate),
        priority=candidate.priority,
        budget_cost=len(candidate.content),
        duplicate_of=duplicate_of,
    )


def _selected(
    candidate: ContextCandidate, scope_ok: bool,
) -> ContextSelectionDecision:
    return ContextSelectionDecision(
        candidate_id=candidate.candidate_id,
        selected=True,
        reason=reason_for(candidate),
        authority=candidate.authority_value,
        scope_match=scope_ok,
        freshness=freshness_of(candidate),
        priority=candidate.priority,
        budget_cost=len(candidate.content),
    )


def select_context(
    task: TaskContext,
    candidates: Sequence[ContextCandidate],
    budget: ContextBudget,
    step_id: str = "001",
) -> ContextSelection:
    """Run the full documented pipeline (M15 §7) and return the
    package + every candidate's decision.

    Raises ``CONTEXT_INVALID_CANDIDATE`` (contract violation on one
    candidate), ``CONTEXT_CONTRACT_VIOLATION`` (bad task/budget/
    sequence/duplicate candidate ids), or ``CONTEXT_BUDGET_EXCEEDED``
    (eligible candidates exist but not one fits the character
    budget). Never a silent over-budget or under-reported selection.
    """
    if not isinstance(task, TaskContext):
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "task must be a TaskContext, got {!r}".format(
                type(task).__name__
            ),
        )
    if not isinstance(budget, ContextBudget):
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "budget must be a ContextBudget, got {!r}".format(
                type(budget).__name__
            ),
        )
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, (tuple, list)
    ):
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "candidates must be a sequence of ContextCandidate, got "
            "{!r}".format(type(candidates).__name__),
        )
    # Stage 1 — validate every candidate; ids must be unique within
    # one selection (decisions are keyed by candidate_id).
    seen_ids: set = set()
    for candidate in candidates:
        validate_candidate(candidate)
        if candidate.candidate_id in seen_ids:
            raise ContextOperationError(
                CONTEXT_CONTRACT_VIOLATION,
                "duplicate candidate_id {!r} within one selection "
                "(every candidate must be individually decidable)".format(
                    candidate.candidate_id
                ),
            )
        seen_ids.add(candidate.candidate_id)

    # Stage 2-4 — hard filters, in the documented order.
    decisions: Dict[str, ContextSelectionDecision] = {}
    pending: List[ContextCandidate] = []
    for candidate in candidates:
        scope_ok = scope_matches(task, candidate)
        if not scope_ok:
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_OUT_OF_SCOPE, scope_ok
            )
            continue
        freshness = freshness_of(candidate)
        if freshness == CONTEXT_FRESHNESS_STALE:
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_STALE_MEMORY, scope_ok
            )
            continue
        if freshness == CONTEXT_FRESHNESS_UNKNOWN:
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_UNKNOWN_FRESHNESS, scope_ok
            )
            continue
        if not relevance_matches(task, candidate):
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_IRRELEVANT, scope_ok
            )
            continue
        pending.append(candidate)

    # Stage 5 — total deterministic order.
    ordered = sorted(pending, key=lambda c: selection_sort_key(task, c))

    # Stage 6 — exact-identity dedup (first winner in sort order).
    winners: List[ContextCandidate] = []
    seen_keys: Dict[Tuple[str, str], str] = {}
    for candidate in ordered:
        key = dedup_key(candidate)
        winner_id = seen_keys.get(key)
        if winner_id is not None:
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_DUPLICATE, True, duplicate_of=winner_id
            )
            continue
        seen_keys[key] = candidate.candidate_id
        winners.append(candidate)

    # Stage 7 — greedy budget fill (never over budget, never silent).
    selected: List[ContextCandidate] = []
    characters = 0
    for candidate in winners:
        if len(selected) >= budget.max_items:
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_LOW_PRIORITY, True
            )
            continue
        cost = len(candidate.content)
        if characters + cost > budget.max_characters:
            decisions[candidate.candidate_id] = _excluded(
                candidate, REASON_BUDGET_EXCEEDED, True
            )
            continue
        selected.append(candidate)
        characters += cost

    if pending and not selected:
        # Eligible candidates existed and NOT ONE fits — that is the
        # budget being impossible, not a legitimate empty selection.
        raise ContextOperationError(
            CONTEXT_BUDGET_EXCEEDED,
            "no eligible candidate fits the character budget ({} "
            "characters): smallest eligible candidate is {} "
            "characters — refusing to return an empty or over-budget "
            "package (M15 §12)".format(
                budget.max_characters,
                min(len(c.content) for c in winners),
            ),
        )

    for candidate in selected:
        decisions[candidate.candidate_id] = _selected(candidate, True)

    # Stage 8 — presentation order + positional item ids.
    presentation = sorted(selected, key=presentation_sort_key)
    items: Tuple[ContextItem, ...] = tuple(
        ContextItem(
            context_item_id=new_item_id(position),
            candidate_id=candidate.candidate_id,
            source_type=candidate.source_type,
            content=candidate.content,
            selection_reason=reason_for(candidate),
            authority=candidate.authority_value,
            source_id=candidate.source_id,
            memory_id=candidate.memory_id,
            repository_scope=candidate.repository_scope,
            commit_scope=candidate.commit_scope,
            freshness=freshness_of(candidate),
            priority=candidate.priority,
            relevance=candidate.relevance,
            evidence=candidate.evidence,
        )
        for position, candidate in enumerate(presentation, start=1)
    )
    estimated = sum(len(candidate.content) for candidate in presentation)

    summary = _summary(candidates, decisions, presentation)
    package = ContextPackage(
        context_id=new_context_id(),
        task=task.question,
        step_id=step_id,
        budget=budget.with_usage(len(items), estimated),
        repository_scope=task.repository,
        commit_scope=task.commit,
        selected_items=items,
        open_questions=task.open_questions,
        constraints=task.constraints,
        selection_summary=summary,
    )
    ordered_decisions = tuple(
        decisions[candidate.candidate_id] for candidate in candidates
    )
    return ContextSelection(package=package, decisions=ordered_decisions)


def _summary(
    candidates: Sequence[ContextCandidate],
    decisions: Dict[str, ContextSelectionDecision],
    presentation: Sequence[ContextCandidate],
) -> Dict[str, Any]:
    """Deterministic selection summary for reports/traces: counts by
    source (selected) and by reason (all decisions), built in stable
    iteration order (M15 §46)."""
    by_source: Dict[str, int] = {}
    for candidate in presentation:
        by_source[candidate.source_type] = (
            by_source.get(candidate.source_type, 0) + 1
        )
    by_reason: Dict[str, int] = {}
    for candidate in candidates:
        decision = decisions[candidate.candidate_id]
        by_reason[decision.reason] = by_reason.get(decision.reason, 0) + 1
    excluded = [
        decisions[candidate.candidate_id]
        for candidate in candidates
        if not decisions[candidate.candidate_id].selected
    ]
    budget_blocked = sum(
        1
        for decision in excluded
        if decision.reason
        in (REASON_BUDGET_EXCEEDED, REASON_LOW_PRIORITY)
    )
    return {
        "considered": len(candidates),
        "selected": len(presentation),
        "excluded": len(excluded),
        "by_source": by_source,
        "by_reason": by_reason,
        "duplicates": by_reason.get(REASON_DUPLICATE, 0),
        "budget_blocked": budget_blocked,
    }
