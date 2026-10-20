"""Persistent Agent Memory (M14) — repository-agnostic.

One capability M13 did not have: **selected validated information
survives the end of one agent execution and can be retrieved by a
later, separate execution** — across Python processes, through one
local SQLite file.

    Agent ── memory candidate / retrieval request ──► MemoryManager
                                                       │ validate
                                                       │ generate mem_ id
                                                       │ deterministic query
                                                       │ lifecycle
                                                       ▼
                                                    SQLite
                                          db_store/memory/agent_memory.db

Boundaries (M14 §19/§20):

    Agent            reasoning and action (never imports sqlite3)
    MemoryManager    memory lifecycle (never calls an LLM)
    SQLite           persistence (this package's store layer)

This package owns the memory CONTRACT (``models``), structured errors
(``errors``), the SQLite store (``store``), the manager API
(``manager`` — store/get/search/verify/supersede/invalidate), the
deterministic memory formation policy after M13 validation
(``formation``), and deterministic context rendering (``rendering``).

Deliberately NOT in M14 (deferred — docs/m14): embeddings, vector
search, semantic/RAG retrieval, ranking, consolidation, autonomous
extraction, multi-agent shared memory, decay/forgetting, distributed
or managed memory services. Retrieval is exact/controlled matching
over repository, commit, owner, memory_type, subject, status —
deterministic and inspectable.

Provider-neutral by construction: nothing here imports an LLM client,
SDK, or model class; memory operates on application-level structures
only (M14 §64). No secrets, keys, raw tool dumps, or hidden reasoning
are ever stored — a memory holds deliberately selected information
(M14 §67).
"""

from __future__ import annotations

from .errors import (
    MEMORY_CONTRACT_VIOLATION,
    MEMORY_INVALID,
    MEMORY_NOT_FOUND,
    MEMORY_SCOPE_MISMATCH,
    MEMORY_STORE_NOT_INITIALIZED,
    MEMORY_STORE_UNAVAILABLE,
    MemoryOperationError,
)
from .formation import MAX_MEMORY_CANDIDATES, select_memory_candidates
from .manager import MemoryManager, is_stale, verification_status
from .models import (
    DEFAULT_SEARCH_LIMIT,
    MEMORY_STATUSES,
    MEMORY_STATUS_INVALID,
    MEMORY_STATUS_SUPERSEDED,
    MEMORY_STATUS_VERIFIED,
    MEMORY_TYPES,
    MEMORY_TYPE_EPISODIC,
    MEMORY_TYPE_SEMANTIC,
    MEMORY_VERIFICATION_FRESH,
    MEMORY_VERIFICATION_STALE,
    MEMORY_VERIFICATION_STATUSES,
    MEMORY_VERIFICATION_UNKNOWN,
    Memory,
    MemoryCandidate,
    MemoryContext,
    MemoryOwner,
    MemoryScope,
    MemoryVerification,
    candidate_from,
    validate_candidate,
)
from .rendering import MEMORY_CONTEXT_HEADER, render_memory_context
from .store import MemoryStore, default_memory_db_path, new_memory_id

__all__ = [
    "DEFAULT_SEARCH_LIMIT",
    "MAX_MEMORY_CANDIDATES",
    "MEMORY_CONTRACT_VIOLATION",
    "MEMORY_CONTEXT_HEADER",
    "MEMORY_INVALID",
    "MEMORY_NOT_FOUND",
    "MEMORY_SCOPE_MISMATCH",
    "MEMORY_STATUSES",
    "MEMORY_STATUS_INVALID",
    "MEMORY_STATUS_SUPERSEDED",
    "MEMORY_STATUS_VERIFIED",
    "MEMORY_STORE_NOT_INITIALIZED",
    "MEMORY_STORE_UNAVAILABLE",
    "MEMORY_TYPES",
    "MEMORY_TYPE_EPISODIC",
    "MEMORY_TYPE_SEMANTIC",
    "MEMORY_VERIFICATION_FRESH",
    "MEMORY_VERIFICATION_STALE",
    "MEMORY_VERIFICATION_STATUSES",
    "MEMORY_VERIFICATION_UNKNOWN",
    "Memory",
    "MemoryCandidate",
    "MemoryContext",
    "MemoryManager",
    "MemoryOperationError",
    "MemoryOwner",
    "MemoryScope",
    "MemoryStore",
    "MemoryVerification",
    "candidate_from",
    "default_memory_db_path",
    "is_stale",
    "new_memory_id",
    "render_memory_context",
    "select_memory_candidates",
    "validate_candidate",
    "verification_status",
]
