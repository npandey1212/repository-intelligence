"""Deterministic context metrics (M15 §46) — no LLM judge, no
semantic scoring.

Same inputs → same numbers: every metric is a count or a ratio over
the package/decision structures the policy already produced. The
report (``render_evaluation_report``) is the §47 quality output the
demo prints — every number traceable to a contract field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from .errors import CONTEXT_CONTRACT_VIOLATION, ContextOperationError
from .models import (
    REASON_BUDGET_EXCEEDED,
    REASON_DUPLICATE,
    REASON_IRRELEVANT,
    REASON_LOW_PRIORITY,
    REASON_OUT_OF_SCOPE,
    REASON_STALE_MEMORY,
    REASON_UNKNOWN_FRESHNESS,
    ContextSelection,
    ContextSnapshot,
)

__all__ = [
    "evaluate_context",
    "evaluate_context_run",
    "render_evaluation_report",
]

_SelectionLike = Union[ContextSelection, ContextSnapshot]


def evaluate_context(selection: _SelectionLike) -> Dict[str, Any]:
    """Compute the deterministic metric set over one selection or
    snapshot (M15 §46):

    * considered / selected / excluded — decision counts
    * context_size / budget_utilization — what entered the context
      against the ceiling (percent, 2 decimals)
    * duplicate_count / budget_blocked_count — how much the dedup and
      budget stages actually did
    * evidence_coverage — share of SELECTED items carrying at least
      one ``EvidenceRef`` (percent, 2 decimals; 0.0 for an empty
      package)
    * memory/stale/unknown/out-of-scope counts — the freshness and
      isolation behaviour made numeric
    * isolation_ok — True iff every selected decision was scope-
      matched (a selected out-of-scope item would be a bug, so this
      metric is a boolean assertion, not a score)
    * by_source / by_reason — full count maps from the summary
    """
    package, decisions = _unpack(selection)
    items = package.selected_items
    budget = package.budget
    excluded = [decision for decision in decisions if not decision.selected]
    by_reason: Dict[str, int] = {}
    for decision in decisions:
        by_reason[decision.reason] = by_reason.get(decision.reason, 0) + 1
    with_evidence = sum(1 for item in items if item.evidence)
    memory_selected = sum(
        1 for item in items if item.source_type == "memory"
    )
    utilization = round(
        100.0 * budget.estimated_characters / budget.max_characters, 2
    )
    coverage = round(100.0 * with_evidence / len(items), 2) if items else 0.0
    return {
        "considered": len(decisions),
        "selected": len(items),
        "excluded": len(excluded),
        "context_size": budget.estimated_characters,
        "budget_max": budget.max_characters,
        "budget_utilization": utilization,
        "duplicate_count": by_reason.get(REASON_DUPLICATE, 0),
        "budget_blocked_count": by_reason.get(REASON_BUDGET_EXCEEDED, 0)
        + by_reason.get(REASON_LOW_PRIORITY, 0),
        "evidence_coverage": coverage,
        "memory_items_selected": memory_selected,
        "stale_items_excluded": by_reason.get(REASON_STALE_MEMORY, 0),
        "unknown_items_excluded": by_reason.get(
            REASON_UNKNOWN_FRESHNESS, 0
        ),
        "out_of_scope_excluded": by_reason.get(REASON_OUT_OF_SCOPE, 0),
        "irrelevant_excluded": by_reason.get(REASON_IRRELEVANT, 0),
        "isolation_ok": all(
            decision.scope_match
            for decision in decisions
            if decision.selected
        ),
        "by_source": dict(package.selection_summary.get("by_source") or {}),
        "by_reason": by_reason,
    }


def evaluate_context_run(
    snapshots: Sequence[ContextSnapshot],
) -> Dict[str, Any]:
    """Deterministic run-level metrics over EVERY selection one agent
    run performed (M15 §29) — the numbers that make context EVOLUTION
    observable, still counts/ratios over recorded structures (no LLM
    judge):

    * ``context_selection_count`` / ``context_snapshot_count`` — how
      many selections the run made and persisted (equal: one snapshot
      per selection by construction)
    * ``context_evolution_count`` — reasoning boundaries where a NEW
      context was created after the first (``len - 1``, never negative)
    * ``selected_count`` / ``excluded_count`` / ``duplicate_count`` /
      ``memory_items_selected`` / ``stale_items_excluded`` /
      ``unknown_items_excluded`` — summed across all steps (total
      decisions the run produced)
    * ``context_size`` / ``budget_utilization`` / ``evidence_coverage``
      — the CURRENT (last step) context, since that is what the final
      reasoning step read
    * ``isolation_ok`` — True iff EVERY step's selection was
      scope-clean
    * ``steps`` — the per-step ``evaluate_context`` dicts, in step
      order, for step-by-step inspection

    An empty sequence (a run that never selected) returns zero counts
    and ``isolation_ok=True`` (no violations can exist in no
    selections) — never an error, never a fabricated step.
    """
    if isinstance(snapshots, (str, bytes)) or not isinstance(
        snapshots, (tuple, list)
    ):
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "snapshots must be a sequence of ContextSnapshot, got "
            "{!r}".format(type(snapshots).__name__),
        )
    steps: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        steps.append(evaluate_context(snapshot))

    def _total(key: str) -> int:
        return sum(int(step[key]) for step in steps)

    last = steps[-1] if steps else None
    return {
        "context_selection_count": len(steps),
        "context_snapshot_count": len(steps),
        "context_evolution_count": max(0, len(steps) - 1),
        "selected_count": _total("selected"),
        "excluded_count": _total("excluded"),
        "context_size": last["context_size"] if last else 0,
        "budget_utilization": last["budget_utilization"] if last else 0.0,
        "duplicate_count": _total("duplicate_count"),
        "evidence_coverage": last["evidence_coverage"] if last else 0.0,
        "memory_items_selected": _total("memory_items_selected"),
        "stale_items_excluded": _total("stale_items_excluded"),
        "unknown_items_excluded": _total("unknown_items_excluded"),
        "isolation_ok": all(step["isolation_ok"] for step in steps),
        "steps": steps,
    }


def render_evaluation_report(metrics: Dict[str, Any]) -> str:
    """The §47 quality report — deterministic lines over the metric
    dict (stable field presentation, no prose beyond labels)."""
    lines: List[str] = [
        "Context evaluation report",
        "* Candidates considered: {}".format(metrics["considered"]),
        "* Selected: {}".format(metrics["selected"]),
        "* Excluded: {}".format(metrics["excluded"]),
        "* Context size: {} characters (max {})".format(
            metrics["context_size"], metrics["budget_max"]
        ),
        "* Budget utilization: {}%".format(metrics["budget_utilization"]),
        "* Duplicates: {}".format(metrics["duplicate_count"]),
        "* Budget blocked: {}".format(metrics["budget_blocked_count"]),
        "* Evidence coverage: {}%".format(metrics["evidence_coverage"]),
        "* Memory items selected: {}".format(
            metrics["memory_items_selected"]
        ),
        "* Stale excluded: {}".format(metrics["stale_items_excluded"]),
        "* Unknown-freshness excluded: {}".format(
            metrics["unknown_items_excluded"]
        ),
        "* Out-of-scope excluded: {}".format(
            metrics["out_of_scope_excluded"]
        ),
        "* Isolation: {}".format(
            "PASS" if metrics["isolation_ok"] else "FAIL"
        ),
    ]
    by_reason = metrics.get("by_reason") or {}
    if by_reason:
        parts = [
            "{}={}".format(reason, by_reason[reason])
            for reason in sorted(by_reason)
        ]
        lines.append("* By reason: {}".format(", ".join(parts)))
    return "\n".join(lines)


def _unpack(selection: _SelectionLike):
    """Accept a ContextSelection or a ContextSnapshot — both are the
    package + decision audit; anything else is a contract error."""
    if isinstance(selection, ContextSelection):
        return selection.package, selection.decisions
    if isinstance(selection, ContextSnapshot):
        return selection.package, selection.selection_decisions
    raise ContextOperationError(
        CONTEXT_CONTRACT_VIOLATION,
        "expected a ContextSelection or ContextSnapshot, got {!r}".format(
            type(selection).__name__
        ),
    )
