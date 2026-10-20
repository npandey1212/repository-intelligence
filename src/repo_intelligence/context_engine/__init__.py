"""Context Engineering (M15) — deterministic, inspectable selection
of what information enters the model context for a reasoning step.

    task + memory + repository/graph evidence + tool results + state
        │
        ▼  ContextEngine.select  (explicit rules — §7 pipeline)
    ContextSelection
    ├── ContextPackage   what the model sees (items, open questions,
    │                    constraints, budget WITH usage — §29)
    └── decisions        one per candidate: selected or excluded,
                         each with a stable reason (§13)
        │
        ▼  render_context_text  (provider-neutral text — §28)
    existing messages (M7 Message)  →  LLMRequest  →  model
        │
        ▼  ContextEngine.snapshot + persist  (inspectable record — §11)
    runtime/context/<run_id>/context_<step>.json

Boundaries (M15 §15–§17):

    Agent            reasoning and action (never runs the policy)
    ContextEngine    selection authority (never calls an LLM, never
                     opens a database — memory arrives as RECORDS
                     through ``adapters``)
    Policy           pure function (validate → scope → freshness →
                     relevance → order → dedup → budget →
                     presentation)
    Snapshots        JSON only (the memory DB stays the ONLY SQLite
                     surface — §43)

This package owns the context CONTRACT (``models``), structured
errors (``errors``), the deterministic policy (``policy``), the
engine API (``engine``), the renderer (``rendering``), deterministic
metrics (``evaluation``), and the M13/M14 record adapters
(``adapters``).

Deliberately NOT in M15 (deferred — docs/m15): vector DB,
embeddings, semantic retrieval, RAG/GraphRAG, an LLM-based context
selector, learned ranking, summarization agents, multi-agent
anything, orchestration frameworks (§63), and a second trace/evidence
system (``AgentTrace.context_events`` extends the one trace; evidence
REUSES M13 ``EvidenceRef``).

Provider-neutral by construction: nothing here imports an LLM client,
SDK, or model class — the renderer produces text; callers wrap it in
their own message type (M15 §28). No secrets, raw tool dumps, or
hidden reasoning enter a package: only deliberately selected,
provenanced information (M15 §55/§67).
"""

from __future__ import annotations

from .adapters import (
    candidate_from_memory,
    candidate_from_task,
    candidates_from_tool_result,
)
from .engine import ContextEngine, default_context_root
from .errors import (
    CONTEXT_BUDGET_EXCEEDED,
    CONTEXT_CONTRACT_VIOLATION,
    CONTEXT_INVALID_CANDIDATE,
    CONTEXT_RENDER_FAILURE,
    CONTEXT_SCOPE_MISMATCH,
    CONTEXT_SNAPSHOT_FAILURE,
    ContextOperationError,
)
from .evaluation import (
    evaluate_context,
    evaluate_context_run,
    render_evaluation_report,
)
from .models import (
    AUTHORITY_EVIDENCE,
    AUTHORITY_GRAPH_FACT,
    AUTHORITY_INFERENCE,
    AUTHORITY_MEMORY,
    AUTHORITY_RANKS,
    AUTHORITY_REPOSITORY,
    AUTHORITY_STATE,
    AUTHORITY_TASK,
    AUTHORITY_TOOL_RESULT,
    CONTEXT_FRESHNESS_FRESH,
    CONTEXT_FRESHNESS_STATUSES,
    CONTEXT_FRESHNESS_STALE,
    CONTEXT_FRESHNESS_UNKNOWN,
    CONTEXT_SOURCE_TYPES,
    DEFAULT_AUTHORITY_BY_SOURCE,
    DEFAULT_CONTEXT_MAX_CHARACTERS,
    DEFAULT_CONTEXT_MAX_ITEMS,
    EXCLUSION_REASONS,
    SELECTION_REASON_BY_SOURCE,
    SELECTION_REASONS,
    SOURCE_SECTION_ORDER,
    REASON_AGENT_STATE,
    REASON_BUDGET_EXCEEDED,
    REASON_CURRENT_EVIDENCE,
    REASON_DUPLICATE,
    REASON_GRAPH_NEIGHBOR,
    REASON_IRRELEVANT,
    REASON_LOW_PRIORITY,
    REASON_OUT_OF_SCOPE,
    REASON_RECENT_TOOL_RESULT,
    REASON_REPOSITORY_FACT,
    REASON_REPOSITORY_SCOPE,
    REASON_STALE_MEMORY,
    REASON_TASK_DIRECT,
    REASON_UNKNOWN_FRESHNESS,
    REASON_VERIFIED_MEMORY,
    ContextBudget,
    ContextCandidate,
    ContextItem,
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
from .policy import (
    dedup_key,
    freshness_of,
    presentation_sort_key,
    relevance_matches,
    scope_matches,
    select_context,
)
from .rendering import CONTEXT_HEADER, render_context_text

__all__ = [
    "AUTHORITY_EVIDENCE",
    "AUTHORITY_GRAPH_FACT",
    "AUTHORITY_INFERENCE",
    "AUTHORITY_MEMORY",
    "AUTHORITY_RANKS",
    "AUTHORITY_REPOSITORY",
    "AUTHORITY_STATE",
    "AUTHORITY_TASK",
    "AUTHORITY_TOOL_RESULT",
    "CONTEXT_BUDGET_EXCEEDED",
    "CONTEXT_CONTRACT_VIOLATION",
    "CONTEXT_FRESHNESS_FRESH",
    "CONTEXT_FRESHNESS_STATUSES",
    "CONTEXT_FRESHNESS_STALE",
    "CONTEXT_FRESHNESS_UNKNOWN",
    "CONTEXT_HEADER",
    "CONTEXT_INVALID_CANDIDATE",
    "CONTEXT_RENDER_FAILURE",
    "CONTEXT_SCOPE_MISMATCH",
    "CONTEXT_SNAPSHOT_FAILURE",
    "CONTEXT_SOURCE_TYPES",
    "DEFAULT_AUTHORITY_BY_SOURCE",
    "DEFAULT_CONTEXT_MAX_CHARACTERS",
    "DEFAULT_CONTEXT_MAX_ITEMS",
    "EXCLUSION_REASONS",
    "REASON_AGENT_STATE",
    "REASON_BUDGET_EXCEEDED",
    "REASON_CURRENT_EVIDENCE",
    "REASON_DUPLICATE",
    "REASON_GRAPH_NEIGHBOR",
    "REASON_IRRELEVANT",
    "REASON_LOW_PRIORITY",
    "REASON_OUT_OF_SCOPE",
    "REASON_RECENT_TOOL_RESULT",
    "REASON_REPOSITORY_FACT",
    "REASON_REPOSITORY_SCOPE",
    "REASON_STALE_MEMORY",
    "REASON_TASK_DIRECT",
    "REASON_UNKNOWN_FRESHNESS",
    "REASON_VERIFIED_MEMORY",
    "SELECTION_REASON_BY_SOURCE",
    "SELECTION_REASONS",
    "SOURCE_SECTION_ORDER",
    "ContextBudget",
    "ContextCandidate",
    "ContextEngine",
    "ContextItem",
    "ContextOperationError",
    "ContextPackage",
    "ContextSelection",
    "ContextSelectionDecision",
    "ContextSnapshot",
    "TaskContext",
    "candidate_from_memory",
    "candidate_from_task",
    "candidates_from_tool_result",
    "dedup_key",
    "default_context_root",
    "evaluate_context",
    "evaluate_context_run",
    "freshness_of",
    "new_candidate_id",
    "new_context_id",
    "new_item_id",
    "new_snapshot_id",
    "presentation_sort_key",
    "reason_for",
    "relevance_matches",
    "render_context_text",
    "render_evaluation_report",
    "scope_matches",
    "section_rank",
    "select_context",
    "validate_candidate",
]
