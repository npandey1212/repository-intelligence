"""Deterministic projection of retrieved memory into agent context.

One job: turn a bounded tuple of stored ``Memory`` records into the
text that enters an execution's existing ``messages`` (M14 §21) — no
prompt redesign, no context ranking (M15 is explicitly deferred).

The header states the two rules the model must not violate (M14
§43/§44), in the same declarative style as ``STRUCTURED_ANSWER_INSTRUCTION``
(output format, never a reasoning style):

* retrieved memory is PRIOR information, not current evidence;
* current observations outrank it when they conflict.

Rendering is pure and deterministic: same memories → same text; the
metadata (id, type, status, scope) is shown so the execution can see
*why* a memory is in context — never hidden reasoning, never a
generated summary (the statement is the model's own prior output,
reused verbatim).
"""

from __future__ import annotations

from typing import Optional, Sequence

from .models import Memory
from .manager import is_stale

__all__ = [
    "MEMORY_CONTEXT_HEADER",
    "render_memory_context",
]

MEMORY_CONTEXT_HEADER = (
    "Persistent memory retrieved before this execution (prior "
    "conclusions from earlier runs — NOT current evidence; verify "
    "against observations from this run, and current evidence wins "
    "on conflict):"
)


def render_memory_context(
    memories: Sequence[Memory], current_commit: Optional[str] = None
) -> str:
    """Header + one line per memory (insertion order preserved).

    With *current_commit* given, memories whose scope commit differs
    are visibly flagged stale — the execution sees the scope
    difference before it reasons (M14 §26), it is never laundered
    into freshness.
    """
    if not memories:
        return ""
    lines = [MEMORY_CONTEXT_HEADER]
    for memory in memories:
        scope = "unscoped"
        if memory.scope.repository is not None:
            scope = memory.scope.repository
            if memory.scope.commit is not None:
                scope = "{}@{}".format(
                    scope, _short_commit(memory.scope.commit)
                )
        line = "* {} | {} | {} | scope: {} | {}".format(
            memory.memory_id,
            memory.memory_type,
            memory.status,
            scope,
            memory.statement,
        )
        if current_commit is not None and is_stale(memory, current_commit):
            line += " [stale: created at another commit]"
        lines.append(line)
    return "\n".join(lines)


def _short_commit(commit: str) -> str:
    """Full commits stay intact when short (synthetic values in tests);
    real 40-hex shas are shown at 12 chars — the trace/record keeps
    the full value."""
    return commit if len(commit) <= 12 else commit[:12]
