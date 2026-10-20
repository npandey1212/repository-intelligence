"""Structured context-engineering failures (M15).

One exception type carrying a stable machine-readable ``code`` — the
same convention as ``MemoryOperationError`` (M14 §31) and
``GraphQueryError``: the message is for humans, the ``code`` is what
tests, the trace, and composition roots branch on.

Codes:

    CONTEXT_CONTRACT_VIOLATION   a TaskContext / ContextPackage /
                                 ContextBudget / snapshot argument
                                 violates the context contract (empty
                                 question, unsafe run/step id, budget
                                 below 1, non-package passed to the
                                 renderer…)
    CONTEXT_INVALID_CANDIDATE    one ContextCandidate violates the
                                 candidate contract (unknown source
                                 type, empty content, bad freshness/
                                 authority value, malformed evidence…)
    CONTEXT_BUDGET_EXCEEDED      the budget invariant would be broken:
                                 eligible candidates exist but not one
                                 of them fits the character budget (the
                                 engine NEVER silently returns an empty
                                 or over-budget package)
    CONTEXT_SCOPE_MISMATCH       an operation crossed repository
                                 scopes: an engine bound to one
                                 repository was asked to select for
                                 another repository's task, or a
                                 snapshot pinned to another repository
                                 was handed to ``persist``
    CONTEXT_RENDER_FAILURE       the renderer could not project a
                                 package into context text (a non-
                                 package argument or a corrupted item)
    CONTEXT_SNAPSHOT_FAILURE     a snapshot could not be written to
                                 the runtime tree (I/O failure,
                                 unwritable path)

Never a silent fallback: a context failure never degrades into
handing the model a *different* context than the one the engine
selected (M15 §55 — no hidden context). Callers decide whether a
context failure is fatal for their run; the failure itself is always
explicit and traceable.
"""

from __future__ import annotations

__all__ = [
    "CONTEXT_BUDGET_EXCEEDED",
    "CONTEXT_CONTRACT_VIOLATION",
    "CONTEXT_INVALID_CANDIDATE",
    "CONTEXT_RENDER_FAILURE",
    "CONTEXT_SCOPE_MISMATCH",
    "CONTEXT_SNAPSHOT_FAILURE",
    "ContextOperationError",
]

CONTEXT_CONTRACT_VIOLATION = "CONTEXT_CONTRACT_VIOLATION"
CONTEXT_INVALID_CANDIDATE = "CONTEXT_INVALID_CANDIDATE"
CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
CONTEXT_SCOPE_MISMATCH = "CONTEXT_SCOPE_MISMATCH"
CONTEXT_RENDER_FAILURE = "CONTEXT_RENDER_FAILURE"
CONTEXT_SNAPSHOT_FAILURE = "CONTEXT_SNAPSHOT_FAILURE"


class ContextOperationError(Exception):
    """Deterministic context failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
