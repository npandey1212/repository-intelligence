"""Adapters: existing M13/M14 records → context candidates (M15 §15).

The engine consumes memory as RECORDS — the adapter turns an
already-retrieved ``Memory`` into a ``ContextCandidate`` and carries
the M14 tri-state freshness across with it. This module never opens
the database (it does not import ``sqlite3``): callers pass whatever
``MemoryManager.search`` returned. Repository scope travels as
candidate scope, so the policy's isolation filter can act on it —
and a memory whose commit is unknown is ADAPTED AS UNKNOWN, never
as fresh (M15 §24).

Tool observations follow the same record-in/candidate-out rule
(``candidates_from_tool_result``): a ``ToolResult`` becomes candidate
source material — never a memory write (storage stays M14's
evidence-backed formation policy) — and when the observation carries
an inspectable path + line range, its ``EvidenceRef`` provenance
travels on to ``ContextItem.evidence_refs``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..evidence import EvidenceRef
from ..memory import Memory, verification_status
from .models import ContextCandidate, TaskContext, new_candidate_id

__all__ = [
    "candidate_from_memory",
    "candidate_from_task",
    "candidates_from_tool_result",
]


def candidate_from_task(
    task: TaskContext,
    priority: int = 0,
    candidate_id: Optional[str] = None,
) -> ContextCandidate:
    """The current question as a candidate — source type ``task``,
    authority ``task`` (it outranks everything: it is what the step
    is being asked to do). Its relevance tags travel with it, so the
    relevance stage can match other candidates against the same
    tags. *candidate_id* is optional: composition roots that need a
    STABLE id (deterministic tests, demo fixtures) pass one; the
    default generates a fresh opaque id."""
    return ContextCandidate(
        candidate_id=candidate_id or new_candidate_id(),
        source_type="task",
        content=task.question,
        source_id="task",
        repository_scope=task.repository,
        commit_scope=task.commit,
        priority=priority,
        relevance=task.relevance,
    )


def candidate_from_memory(
    memory: Memory,
    current_commit: Optional[str] = None,
    relevance: Sequence[str] = (),
    candidate_id: Optional[str] = None,
) -> ContextCandidate:
    """One stored memory as a candidate — source type ``memory``,
    carrying its identity (``memory_id`` feeds dedup), its pinned
    scope, its evidence references, and its FRESHNESS computed by the
    M14 tri-state rule against *current_commit*:

    * both commits known and equal  → ``fresh``
    * both known and different       → ``stale``
    * either missing                 → ``unknown`` (never silently
      fresh — M15 §24).

    ``current_commit=None`` therefore always adapts as UNKNOWN: only
    a positively established comparison can make a memory eligible as
    current context. *candidate_id* behaves as in
    ``candidate_from_task`` (stable id when the caller needs one).
    """
    return ContextCandidate(
        candidate_id=candidate_id or new_candidate_id(),
        source_type="memory",
        content=memory.statement,
        source_id=memory.memory_id,
        memory_id=memory.memory_id,
        repository_scope=memory.scope.repository,
        commit_scope=memory.scope.commit,
        evidence=tuple(memory.evidence),
        freshness=verification_status(memory, current_commit),
        relevance=tuple(relevance),
    )


def _clip(text: str, limit: int = 140) -> str:
    """Deterministic display clip for candidate summaries — content
    itself is never rewritten beyond this presentation bound."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def candidates_from_tool_result(
    tool_result: Any,
    task: TaskContext,
    candidate_id: Optional[str] = None,
    evidence_candidate_id: Optional[str] = None,
) -> List[ContextCandidate]:
    """One executed tool's observation → candidate source material
    (M15 §7/§8).

    Returns exactly the offers an observation makes to the NEXT
    reasoning step:

    * always one ``tool_result`` candidate (authority 200) — a brief,
      bounded summary of what the tool returned, pinned to the task's
      repository/commit so the isolation filter can act on it and
      deduplicating on the observation's identity (tool + target +
      inspected range), so re-offering the same observation across
      steps stays a ``DUPLICATE`` instead of growing the context;
    * plus one ``evidence`` candidate (authority 500) when the payload
      carries an inspectable ``path`` + integer ``line_start`` /
      ``line_end`` — carrying the M13 ``EvidenceRef`` provenance, so
      the selected item and its context entry stay traceable to
      source → candidate → evidence → repository/commit.

    This is an ADAPTATION, never a write: a tool result does not
    become memory here (memory formation stays M14's evidence-backed
    policy), and nothing is pre-selected — the deterministic policy
    still decides (M15 §7). *tool_result* is duck-typed (``tool``,
    ``success``, ``data`` — the ``ToolResult`` protocol), so this
    module never imports the tool runtime.

    *candidate_id* / *evidence_candidate_id* are optional stable ids
    for composition roots that need them (demo fixtures, tests); the
    default generates opaque ``cand_<uuid>`` ids.
    """
    tool_name = str(getattr(tool_result, "tool", None) or "tool")
    success = bool(getattr(tool_result, "success", False))
    data = getattr(tool_result, "data", None) or {}
    if not isinstance(data, dict):
        data = {}
    path = str(data.get("path") or "")
    line_start = data.get("line_start")
    line_end = data.get("line_end")
    content = str(data.get("content") or "")
    bounds = ""
    if isinstance(line_start, int) and isinstance(line_end, int):
        bounds = " lines {}-{}".format(line_start, line_end)
    target = "{}{}".format(path or "-", bounds)
    if success:
        summary = "{} {} ({} chars): {}".format(
            tool_name, target, len(content), _clip(content)
        )
        source_id = "tool:{}:{}".format(tool_name, path or "-")
        if bounds:
            source_id = "{}:{}-{}".format(source_id, line_start, line_end)
    else:
        error = getattr(tool_result, "error", None)
        detail = ""
        if error is not None:
            detail = str(
                getattr(error, "message", None) or error
            )
        summary = "{} {} failed{}".format(
            tool_name,
            target,
            ": {}".format(_clip(detail)) if detail else "",
        )
        source_id = "tool:{}:{}:failed".format(tool_name, path or "-")
    candidates: List[ContextCandidate] = [
        ContextCandidate(
            candidate_id=candidate_id or new_candidate_id(),
            source_type="tool_result",
            content=summary or "{} observation".format(tool_name),
            source_id=source_id,
            repository_scope=task.repository,
            commit_scope=task.commit,
            relevance=task.relevance,
        )
    ]
    if success and path and isinstance(line_start, int) and isinstance(
        line_end, int
    ):
        candidates.append(
            ContextCandidate(
                candidate_id=evidence_candidate_id or new_candidate_id(),
                source_type="evidence",
                content=(
                    "Inspected {}:{}-{} — current-run evidence for the "
                    "investigation.".format(path, line_start, line_end)
                ),
                source_id="evidence:{}:{}-{}".format(
                    path, line_start, line_end
                ),
                repository_scope=task.repository,
                commit_scope=task.commit,
                evidence=(
                    EvidenceRef(
                        path=path,
                        line_start=line_start,
                        line_end=line_end,
                    ),
                ),
                priority=10,
                relevance=task.relevance,
            )
        )
    return candidates
