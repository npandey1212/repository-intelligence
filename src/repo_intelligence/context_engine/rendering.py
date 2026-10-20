"""Deterministic projection of a ``ContextPackage`` into model
context (M15 §10/§28).

One job: turn the selected package into the text that enters the
execution's existing ``messages`` — header, task, every item with
its provenance and selection reason, budget line, plus the
open questions/constraints that flow through untouched. The renderer
is PURE (same package → same text), provider-neutral (no LLM client,
SDK, or provider term — callers wrap the text in their own message
type), and it renders ONLY what the policy selected: excluded
candidates, decision audits, and summaries live in snapshots and
traces, never in the model's context (M15 §55 — no hidden context,
and no leaking rejected material either).

Each item line shows WHY it is here (source type, selection reason,
authority, freshness for remembered material, scope, evidence) —
the execution sees the reasoning behind its own context instead of
receiving an unexplained blob (M15 §41).
"""

from __future__ import annotations

from typing import List

from .errors import CONTEXT_RENDER_FAILURE, ContextOperationError
from .models import ContextItem, ContextPackage

__all__ = [
    "CONTEXT_HEADER",
    "render_context_text",
]

CONTEXT_HEADER = (
    "Context selected for this reasoning step (deterministic policy — "
    "each item shows why it is included; remembered material is prior "
    "information, current evidence wins on conflict):"
)


def render_context_text(package: ContextPackage) -> str:
    """Header + budget line + one block per selected item, in the
    package's presentation order (M15 §23 — the model reads exactly
    this order).

    Raises ``CONTEXT_RENDER_FAILURE`` for a non-package argument: a
    render failure is explicit, never an empty string that would
    silently become an empty context.
    """
    if not isinstance(package, ContextPackage):
        raise ContextOperationError(
            CONTEXT_RENDER_FAILURE,
            "package must be a ContextPackage, got {!r}".format(
                type(package).__name__
            ),
        )
    budget = package.budget
    lines: List[str] = []
    scope = "unscoped"
    if package.repository_scope is not None:
        scope = package.repository_scope
        if package.commit_scope is not None:
            scope = "{}@{}".format(scope, _short_commit(package.commit_scope))
    lines.append(CONTEXT_HEADER)
    lines.append(
        "{} | step {} | scope: {} | budget: {}/{} chars, "
        "{}/{} items".format(
            package.context_id,
            package.step_id,
            scope,
            budget.estimated_characters,
            budget.max_characters,
            budget.selected_items,
            budget.max_items,
        )
    )
    lines.append("Task: {}".format(package.task))
    lines.append("")
    if not package.selected_items:
        lines.append("(no context items selected for this step)")
    for item in package.selected_items:
        lines.extend(_render_item(item))
    if package.open_questions:
        lines.append("")
        lines.append("Open questions:")
        for question in package.open_questions:
            lines.append("* {}".format(question))
    if package.constraints:
        lines.append("")
        lines.append("Constraints:")
        for constraint in package.constraints:
            lines.append("* {}".format(constraint))
    return "\n".join(lines)


def _render_item(item: ContextItem) -> List[str]:
    """One item block: identity/provenance line + indented content
    + evidence references. Multi-line content keeps its lines."""
    header_parts = [
        item.context_item_id,
        item.source_type,
        item.selection_reason,
        "authority {}".format(item.authority),
    ]
    if item.freshness is not None:
        header_parts.append("freshness {}".format(item.freshness))
    if item.priority:
        header_parts.append("priority {}".format(item.priority))
    if item.source_id is not None:
        header_parts.append("source {}".format(item.source_id))
    if item.repository_scope is not None:
        scope = item.repository_scope
        if item.commit_scope is not None:
            scope = "{}@{}".format(scope, _short_commit(item.commit_scope))
        header_parts.append("scope {}".format(scope))
    lines = ["[{}] {}".format(
        item.context_item_id, " | ".join(header_parts[1:])
    )]
    for content_line in item.content.splitlines() or [item.content]:
        lines.append("  {}".format(content_line))
    for ref in item.evidence:
        if ref.line_start is not None and ref.line_end is not None:
            lines.append(
                "  evidence: {}:{}-{}".format(
                    ref.path, ref.line_start, ref.line_end
                )
            )
        else:
            lines.append("  evidence: {}".format(ref.path))
    return lines


def _short_commit(commit: str) -> str:
    """Full commits stay intact when short (synthetic values in
    tests); real 40-hex shas are shown at 12 chars — the snapshot
    keeps the full value."""
    return commit if len(commit) <= 12 else commit[:12]
