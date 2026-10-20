"""Structured memory failures (M14 §31).

One exception type carrying a stable machine-readable ``code`` — the
same convention as ``GraphQueryError`` (INVALID_ARGUMENT /
NODE_NOT_FOUND): the message is for humans, the ``code`` is what
tests, the trace, and composition roots branch on.

Codes:

    MEMORY_NOT_FOUND             get()/lifecycle call for an id that
                                 is not in the store
    MEMORY_CONTRACT_VIOLATION    a candidate or stored record violates
                                 the memory contract (bad owner, bad
                                 type, empty statement, bad evidence…)
    MEMORY_SCOPE_MISMATCH        an operation crossed repository scopes
                                 or lacked the repository scope it
                                 requires (e.g. superseding an ADOP
                                 memory from another repository, or
                                 repository-derived retrieval with no
                                 repository scope)
    MEMORY_INVALID               the operation is not legal for this
                                 memory's status (e.g. superseding an
                                 already superseded/invalid memory)
    MEMORY_STORE_UNAVAILABLE     the database could not be opened or
                                 written (I/O failure, unwritable path)
    MEMORY_STORE_NOT_INITIALIZED the database exists but the schema was
                                 never created (an operation ran before
                                 ``initialize()``)

Never a silent fallback: a store failure never degrades into
pretending the memory existed (no "search the repository instead"
short-circuit). Callers decide whether a memory failure is fatal for
their run; the failure itself is always explicit.
"""

from __future__ import annotations

__all__ = [
    "MEMORY_CONTRACT_VIOLATION",
    "MEMORY_INVALID",
    "MEMORY_NOT_FOUND",
    "MEMORY_SCOPE_MISMATCH",
    "MEMORY_STORE_NOT_INITIALIZED",
    "MEMORY_STORE_UNAVAILABLE",
    "MemoryOperationError",
]

MEMORY_NOT_FOUND = "MEMORY_NOT_FOUND"
MEMORY_CONTRACT_VIOLATION = "MEMORY_CONTRACT_VIOLATION"
MEMORY_SCOPE_MISMATCH = "MEMORY_SCOPE_MISMATCH"
MEMORY_INVALID = "MEMORY_INVALID"
MEMORY_STORE_UNAVAILABLE = "MEMORY_STORE_UNAVAILABLE"
MEMORY_STORE_NOT_INITIALIZED = "MEMORY_STORE_NOT_INITIALIZED"


class MemoryOperationError(Exception):
    """Deterministic memory failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
