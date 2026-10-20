"""The M15 context-engineering contracts — small, explicit,
repository-agnostic.

Context engineering is the deterministic decision about **what
information enters the model context for a reasoning step** (M15 §6):

    task + retrieved memory + repository/graph evidence + tool
    results + agent state
        │
        ▼  ContextEngine.select  (explicit rules, no model)
    ContextSelection
    ├── ContextPackage          what the model will see
    │   ├── context_id          ctx_<opaque uuid>
    │   ├── task                the current question
    │   ├── step_id             which reasoning step this serves
    │   ├── selected_items[]    ContextItem (presentation-ordered)
    │   ├── open_questions[]    never silently dropped
    │   ├── constraints[]       never silently dropped
    │   └── budget              ContextBudget (max + ACTUAL usage)
    └── decisions[]             one per candidate — selected AND
                                excluded, each with a stable reason

    ContextSnapshot              persisted record of one selection
    (package + all decisions + excluded ids — M15 §11/§29)

Contract rules (validated application-side, never by the LLM):

* a candidate NEVER invents selection — it is only *considered*;
  the engine alone assigns reasons, order, and budget outcomes;
* every candidate receives exactly ONE decision (selected or
  excluded, with a stable machine-readable reason) — nothing is
  dropped without a traceable reason;
* freshness is tri-state and preserved exactly (fresh / stale /
  unknown — a memory without a known commit is UNKNOWN, never
  silently fresh: M15 §24);
* budget is a hard invariant: ``estimated_characters`` equals the
  sum of selected item contents and can never exceed
  ``max_characters`` (M15 §12);
* evidence REUSES ``EvidenceRef`` (M13) — no second evidence model
  (M15 §16);
* ids and timestamps are generated here (``new_context_id``,
  ``_utcnow``) so tests can pin them — the engine itself never
  writes a database or a file (M15 §15).

Source types (M15 §8 vocabulary): every candidate declares exactly
one — task, repository, fact, graph, evidence, memory, tool_result,
agent_state. Authority ranks the SOURCE CLASSES of the §14 hierarchy
(direct repository evidence > deterministic graph/fact > validated
memory > tool result > inference); the current TASK outranks all of
it because it is what the step is being asked to do.

No LLM client, SDK, or model class is imported anywhere in this
package (provider-neutral by construction, M15 §28): the contracts
are application-level structures only. No secrets, no raw tool
dumps, no hidden-reasoning fields — a package holds deliberately
selected information only (M15 §55).
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..evidence import EvidenceRef
from .errors import (
    CONTEXT_CONTRACT_VIOLATION,
    CONTEXT_INVALID_CANDIDATE,
    ContextOperationError,
)

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
    "CONTEXT_FRESHNESS_FRESH",
    "CONTEXT_FRESHNESS_STATUSES",
    "CONTEXT_FRESHNESS_STALE",
    "CONTEXT_FRESHNESS_UNKNOWN",
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
    "SELECTION_REASONS",
    "SELECTION_REASON_BY_SOURCE",
    "SOURCE_SECTION_ORDER",
    "CONTEXT_SOURCE_TYPES",
    "ContextBudget",
    "ContextCandidate",
    "ContextItem",
    "ContextPackage",
    "ContextSelection",
    "ContextSelectionDecision",
    "ContextSnapshot",
    "TaskContext",
    "new_candidate_id",
    "new_context_id",
    "new_item_id",
    "new_snapshot_id",
    "reason_for",
    "section_rank",
    "validate_candidate",
]

# -- source types (M15 §8) ---------------------------------------------------

CONTEXT_SOURCE_TYPES = (
    "task",
    "repository",
    "fact",
    "graph",
    "evidence",
    "memory",
    "tool_result",
    "agent_state",
)

# -- authority hierarchy (M15 §14) -------------------------------------------
# direct repository evidence > deterministic graph/fact > validated
# memory > tool result > inference. The numeric ranks make the
# hierarchy a deterministic sort key; the strings are what renders
# and records show. ``agent_state`` (execution bookkeeping) sits
# between tool results and inference; ``task`` is the question the
# step serves and therefore outranks everything else.

AUTHORITY_TASK = "task"
AUTHORITY_REPOSITORY = "repository"
AUTHORITY_EVIDENCE = "evidence"
AUTHORITY_GRAPH_FACT = "graph_fact"
AUTHORITY_MEMORY = "memory"
AUTHORITY_TOOL_RESULT = "tool_result"
AUTHORITY_STATE = "agent_state"
AUTHORITY_INFERENCE = "inference"

AUTHORITY_RANKS: Dict[str, int] = {
    AUTHORITY_TASK: 600,
    AUTHORITY_EVIDENCE: 500,
    AUTHORITY_REPOSITORY: 450,
    AUTHORITY_GRAPH_FACT: 400,
    AUTHORITY_MEMORY: 300,
    AUTHORITY_TOOL_RESULT: 200,
    AUTHORITY_STATE: 150,
    AUTHORITY_INFERENCE: 100,
}

DEFAULT_AUTHORITY_BY_SOURCE: Dict[str, str] = {
    "task": AUTHORITY_TASK,
    "repository": AUTHORITY_REPOSITORY,
    "fact": AUTHORITY_GRAPH_FACT,
    "graph": AUTHORITY_GRAPH_FACT,
    "evidence": AUTHORITY_EVIDENCE,
    "memory": AUTHORITY_MEMORY,
    "tool_result": AUTHORITY_TOOL_RESULT,
    "agent_state": AUTHORITY_STATE,
}

# -- freshness (M15 §24 — same tri-state semantics as M14 §26) ----------------

CONTEXT_FRESHNESS_FRESH = "fresh"
CONTEXT_FRESHNESS_STALE = "stale"
CONTEXT_FRESHNESS_UNKNOWN = "unknown"
CONTEXT_FRESHNESS_STATUSES = (
    CONTEXT_FRESHNESS_FRESH,
    CONTEXT_FRESHNESS_STALE,
    CONTEXT_FRESHNESS_UNKNOWN,
)

# -- selection / exclusion reasons (M15 §13) ---------------------------------
# Reasons are STABLE machine-readable strings: tests, the trace, and
# the demo branch on them; prose never replaces them.

REASON_TASK_DIRECT = "TASK_DIRECT"
REASON_REPOSITORY_SCOPE = "REPOSITORY_SCOPE"
REASON_REPOSITORY_FACT = "REPOSITORY_FACT"
REASON_CURRENT_EVIDENCE = "CURRENT_EVIDENCE"
REASON_GRAPH_NEIGHBOR = "GRAPH_NEIGHBOR"
REASON_VERIFIED_MEMORY = "VERIFIED_MEMORY"
REASON_RECENT_TOOL_RESULT = "RECENT_TOOL_RESULT"
REASON_AGENT_STATE = "AGENT_STATE"

SELECTION_REASONS = (
    REASON_TASK_DIRECT,
    REASON_REPOSITORY_SCOPE,
    REASON_REPOSITORY_FACT,
    REASON_CURRENT_EVIDENCE,
    REASON_GRAPH_NEIGHBOR,
    REASON_VERIFIED_MEMORY,
    REASON_RECENT_TOOL_RESULT,
    REASON_AGENT_STATE,
)

REASON_OUT_OF_SCOPE = "OUT_OF_SCOPE"
REASON_STALE_MEMORY = "STALE_MEMORY"
REASON_UNKNOWN_FRESHNESS = "UNKNOWN_FRESHNESS"
REASON_DUPLICATE = "DUPLICATE"
REASON_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
REASON_LOW_PRIORITY = "LOW_PRIORITY"
REASON_IRRELEVANT = "IRRELEVANT"

EXCLUSION_REASONS = (
    REASON_OUT_OF_SCOPE,
    REASON_STALE_MEMORY,
    REASON_UNKNOWN_FRESHNESS,
    REASON_DUPLICATE,
    REASON_BUDGET_EXCEEDED,
    REASON_LOW_PRIORITY,
    REASON_IRRELEVANT,
)

# Why a source CLASS enters context (deterministic — a candidate's
# reason depends only on its declared source, never on content).
SELECTION_REASON_BY_SOURCE: Dict[str, str] = {
    "task": REASON_TASK_DIRECT,
    "repository": REASON_REPOSITORY_SCOPE,
    "fact": REASON_REPOSITORY_FACT,
    "evidence": REASON_CURRENT_EVIDENCE,
    "graph": REASON_GRAPH_NEIGHBOR,
    "memory": REASON_VERIFIED_MEMORY,
    "tool_result": REASON_RECENT_TOOL_RESULT,
    "agent_state": REASON_AGENT_STATE,
}

# -- presentation order (M15 §23 — documented, never incidental) --------------
# The model reads context in this section order regardless of which
# candidates won the budget: the current task first, then the
# repository it runs against, then live state, current evidence,
# deterministic facts/graph, remembered material, and finally raw
# tool output.

SOURCE_SECTION_ORDER: Dict[str, int] = {
    "task": 10,
    "repository": 20,
    "agent_state": 30,
    "evidence": 40,
    "fact": 50,
    "graph": 60,
    "memory": 70,
    "tool_result": 80,
}

# -- budget defaults (M15 §12) ------------------------------------------------

DEFAULT_CONTEXT_MAX_ITEMS = 20
DEFAULT_CONTEXT_MAX_CHARACTERS = 12000


# -- generation helpers -------------------------------------------------------

def _utcnow() -> str:
    """UTC timestamp, ISO-8601 with millisecond precision (sortable) —
    the same representation the AgentTrace and the memory store use.
    Module-level so tests can pin time deterministically."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def new_context_id() -> str:
    """Opaque ``ctx_<uuid>`` — content-independent identity (same rule
    as memory ids: identity is never derived from content)."""
    return "ctx_" + uuid.uuid4().hex


def new_candidate_id() -> str:
    """Opaque ``cand_<uuid>`` for caller-built candidates."""
    return "cand_" + uuid.uuid4().hex


def new_snapshot_id() -> str:
    """Opaque ``snap_<uuid>`` for persisted selections."""
    return "snap_" + uuid.uuid4().hex


def new_item_id(position: int) -> str:
    """Deterministic positional ``ctxitem_<NNN>`` — item identity IS
    its presentation position (M15 §23/§30: same inputs → same ids)."""
    if isinstance(position, bool) or not isinstance(position, int):
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "item position must be an integer, got {!r}".format(position),
        )
    if position < 1:
        raise ContextOperationError(
            CONTEXT_CONTRACT_VIOLATION,
            "item position must be >= 1, got {!r}".format(position),
        )
    return "ctxitem_{:03d}".format(position)


# -- validation helpers -------------------------------------------------------

def _contract_violation(message: str) -> ContextOperationError:
    return ContextOperationError(CONTEXT_CONTRACT_VIOLATION, message)


def _invalid_candidate(message: str) -> ContextOperationError:
    return ContextOperationError(CONTEXT_INVALID_CANDIDATE, message)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _contract_violation(
            "{} must be a non-empty string, got {!r}".format(label, value)
        )
    return value


def _optional_text(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _contract_violation(
            "{} must be None or a non-empty string, got {!r}".format(
                label, value
            )
        )
    return value


def _text_tuple(value: Any, label: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise _contract_violation(
            "{} must be a sequence of non-empty strings, got {!r}".format(
                label, type(value).__name__
            )
        )
    result = []
    for index, element in enumerate(value):
        if not isinstance(element, str) or not element.strip():
            raise _contract_violation(
                "{}[{}] must be a non-empty string, got {!r}".format(
                    label, index, element
                )
            )
        result.append(element)
    return tuple(result)


def _validate_evidence(evidence: Any) -> Tuple[EvidenceRef, ...]:
    if isinstance(evidence, (str, bytes)) or not isinstance(
        evidence, (tuple, list)
    ):
        raise _invalid_candidate(
            "evidence must be a sequence of EvidenceRef, got {!r}".format(
                type(evidence).__name__
            )
        )
    refs: Tuple[EvidenceRef, ...] = ()
    for index, ref in enumerate(evidence):
        if not isinstance(ref, EvidenceRef):
            raise _invalid_candidate(
                "evidence[{}] must be an EvidenceRef, got {!r}".format(
                    index, type(ref).__name__
                )
            )
        if not isinstance(ref.path, str) or not ref.path.strip():
            raise _invalid_candidate(
                "evidence[{}].path must be a non-empty string".format(index)
            )
        for bound in ("line_start", "line_end"):
            value = getattr(ref, bound)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise _invalid_candidate(
                    "evidence[{}].{} must be None or a positive integer, "
                    "got {!r}".format(index, bound, value)
                )
        if (
            ref.line_start is not None
            and ref.line_end is not None
            and ref.line_start > ref.line_end
        ):
            raise _invalid_candidate(
                "evidence[{}] line_start {} > line_end {}".format(
                    index, ref.line_start, ref.line_end
                )
            )
        refs = refs + (ref,)
    return refs


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _contract_violation(
            "{} must be an integer >= 1, got {!r}".format(label, value)
        )
    return value


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _contract_violation(
            "{} must be an integer >= 0, got {!r}".format(label, value)
        )
    return value


# -- contracts ----------------------------------------------------------------

@dataclass(frozen=True)
class TaskContext:
    """The current reasoning step's question + scope (M15 §9).

    The task is the ANCHOR of every selection: its ``repository``/
    ``commit`` define the scope filter, its ``relevance`` tags feed
    the relevance stage, its ``open_questions``/``constraints`` flow
    into the package untouched (never silently dropped — M15 §18).
    """

    question: str
    repository: Optional[str] = None
    commit: Optional[str] = None
    open_questions: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    relevance: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _optional_text(self.repository, "repository")
        _optional_text(self.commit, "commit")
        object.__setattr__(
            self, "open_questions", _text_tuple(self.open_questions,
                                                "open_questions")
        )
        object.__setattr__(
            self, "constraints", _text_tuple(self.constraints, "constraints")
        )
        object.__setattr__(
            self, "relevance", _text_tuple(self.relevance, "relevance")
        )


@dataclass(frozen=True)
class ContextCandidate:
    """What one source offers to a reasoning step (M15 §8) —
    considered, never self-selected.

    ``candidate_id`` is caller-assigned (``new_candidate_id()``):
    unlike memory ids (manager-generated), candidates are input
    objects assembled by the composition root BEFORE any engine call,
    so their identity must be stable across the select/snapshot pair.
    """

    candidate_id: str
    source_type: str
    content: str
    source_id: Optional[str] = None
    repository_scope: Optional[str] = None
    commit_scope: Optional[str] = None
    authority: Optional[str] = None
    evidence: Tuple[EvidenceRef, ...] = ()
    memory_id: Optional[str] = None
    freshness: Optional[str] = None
    priority: int = 0
    relevance: Tuple[str, ...] = ()
    created_at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        validate_candidate(self)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "relevance", tuple(self.relevance))

    @property
    def authority_value(self) -> str:
        """Resolved authority string: an explicit override wins,
        otherwise the source type's default in the §14 hierarchy."""
        if self.authority is not None:
            return self.authority
        return DEFAULT_AUTHORITY_BY_SOURCE.get(self.source_type,
                                               AUTHORITY_INFERENCE)

    @property
    def authority_rank(self) -> int:
        return AUTHORITY_RANKS[self.authority_value]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready projection (candidate inventory — §33 — and
        test/diagnostic output; decisions and items carry their own
        projections)."""
        return {
            "candidate_id": self.candidate_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "content": self.content,
            "repository_scope": self.repository_scope,
            "commit_scope": self.commit_scope,
            "authority": self.authority_value,
            "evidence": [ref.to_dict() for ref in self.evidence],
            "memory_id": self.memory_id,
            "freshness": self.freshness,
            "priority": self.priority,
            "relevance": list(self.relevance),
            "created_at": self.created_at,
        }


def validate_candidate(candidate: Any) -> ContextCandidate:
    """Enforce the candidate contract on one object (M15 §8).

    Raises ``ContextOperationError(CONTEXT_INVALID_CANDIDATE)`` for
    any violation — validation is application-side; the model never
    validates its own context. Returns the candidate unchanged for
    fluent use (the same shape as M14's ``validate_candidate``).
    """
    if not isinstance(candidate, ContextCandidate):
        raise _invalid_candidate(
            "candidate must be a ContextCandidate, got {!r}".format(
                type(candidate).__name__
            )
        )
    if not isinstance(candidate.candidate_id, str) or not (
        candidate.candidate_id.strip()
    ):
        raise _invalid_candidate(
            "candidate_id must be a non-empty string, got {!r}".format(
                candidate.candidate_id
            )
        )
    if candidate.source_type not in CONTEXT_SOURCE_TYPES:
        raise _invalid_candidate(
            "source_type must be one of {}, got {!r}".format(
                ", ".join(CONTEXT_SOURCE_TYPES), candidate.source_type
            )
        )
    if not isinstance(candidate.content, str) or not candidate.content.strip():
        raise _invalid_candidate(
            "content must be a non-empty string, got {!r}".format(
                candidate.content
            )
        )
    for label, value in (
        ("source_id", candidate.source_id),
        ("repository_scope", candidate.repository_scope),
        ("commit_scope", candidate.commit_scope),
        ("memory_id", candidate.memory_id),
    ):
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise _invalid_candidate(
                "{} must be None or a non-empty string, got {!r}".format(
                    label, value
                )
            )
    if candidate.authority is not None:
        if candidate.authority not in AUTHORITY_RANKS:
            raise _invalid_candidate(
                "authority must be one of {}, got {!r}".format(
                    ", ".join(sorted(AUTHORITY_RANKS)),
                    candidate.authority,
                )
            )
    if candidate.freshness is not None:
        if candidate.freshness not in CONTEXT_FRESHNESS_STATUSES:
            raise _invalid_candidate(
                "freshness must be one of {}, got {!r}".format(
                    ", ".join(CONTEXT_FRESHNESS_STATUSES),
                    candidate.freshness,
                )
            )
    if (
        isinstance(candidate.priority, bool)
        or not isinstance(candidate.priority, int)
    ):
        raise _invalid_candidate(
            "priority must be an integer, got {!r}".format(candidate.priority)
        )
    if isinstance(candidate.relevance, (str, bytes)) or not isinstance(
        candidate.relevance, (tuple, list)
    ):
        raise _invalid_candidate(
            "relevance must be a sequence of non-empty strings, got "
            "{!r}".format(type(candidate.relevance).__name__)
        )
    for index, tag in enumerate(candidate.relevance):
        if not isinstance(tag, str) or not tag.strip():
            raise _invalid_candidate(
                "relevance[{}] must be a non-empty string, got {!r}".format(
                    index, tag
                )
            )
    if not isinstance(candidate.created_at, str) or not candidate.created_at:
        raise _invalid_candidate(
            "created_at must be a non-empty string, got {!r}".format(
                candidate.created_at
            )
        )
    _validate_evidence(candidate.evidence)
    return candidate


def reason_for(candidate: ContextCandidate) -> str:
    """The stable SELECT reason for a candidate's source class
    (deterministic — content plays no part)."""
    return SELECTION_REASON_BY_SOURCE.get(candidate.source_type,
                                          AUTHORITY_INFERENCE)


def section_rank(source_type: str) -> int:
    """Presentation section rank (M15 §23). Unknown sources rank last
    (after every documented section)."""
    return SOURCE_SECTION_ORDER.get(source_type, 90)


@dataclass(frozen=True)
class ContextItem:
    """One SELECTED candidate in presentation order (M15 §10) — the
    unit the renderer projects into model context.

    ``context_item_id`` is the positional ``ctxitem_<NNN>`` assigned
    after the presentation sort, so identity encodes the exact order
    the model will read (M15 §23/§30).
    """

    context_item_id: str
    candidate_id: str
    source_type: str
    content: str
    selection_reason: str
    authority: str
    source_id: Optional[str] = None
    memory_id: Optional[str] = None
    repository_scope: Optional[str] = None
    commit_scope: Optional[str] = None
    freshness: Optional[str] = None
    priority: int = 0
    relevance: Tuple[str, ...] = ()
    evidence: Tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record (canonical contract projection)."""
        return {
            "context_item_id": self.context_item_id,
            "candidate_id": self.candidate_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "memory_id": self.memory_id,
            "content": self.content,
            "repository_scope": self.repository_scope,
            "commit_scope": self.commit_scope,
            "authority": self.authority,
            "freshness": self.freshness,
            "priority": self.priority,
            "relevance": list(self.relevance),
            "selection_reason": self.selection_reason,
            "evidence": [ref.to_dict() for ref in self.evidence],
        }


@dataclass(frozen=True)
class ContextSelectionDecision:
    """Exactly one decision per candidate (M15 §13) — selected or
    excluded, each with a stable reason. Nothing is dropped
    silently: the decision list is the full audit of the selection.

    ``budget_cost`` is the candidate's character cost (``len`` of its
    content) as considered by the budget stage — recorded even for
    candidates excluded earlier, so reports can show what each
    candidate would have consumed.
    """

    candidate_id: str
    selected: bool
    reason: str
    authority: str
    scope_match: bool
    priority: int
    budget_cost: int
    freshness: Optional[str] = None
    duplicate_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record (canonical contract projection)."""
        return {
            "candidate_id": self.candidate_id,
            "selected": self.selected,
            "reason": self.reason,
            "authority": self.authority,
            "scope_match": self.scope_match,
            "freshness": self.freshness,
            "priority": self.priority,
            "budget_cost": self.budget_cost,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(frozen=True)
class ContextBudget:
    """Hard selection budget (M15 §12): item count + characters,
    with the ACTUAL usage recorded — a budget is only trustworthy
    when the consumption travels with it (never a bare maximum).

    ``selected_items``/``estimated_characters`` are the usage
    (defaults 0); ``with_usage`` derives the filled copy the policy
    stores on the package. Every component is validated: a budget
    can never be silently invalid (a 0/negative budget would make
    "no candidate fits" indistinguishable from a real overflow).
    """

    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS
    max_characters: int = DEFAULT_CONTEXT_MAX_CHARACTERS
    selected_items: int = 0
    estimated_characters: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.max_items, "max_items")
        _require_positive_int(self.max_characters, "max_characters")
        _require_non_negative_int(self.selected_items, "selected_items")
        _require_non_negative_int(
            self.estimated_characters, "estimated_characters"
        )

    @property
    def remaining_items(self) -> int:
        return max(0, self.max_items - self.selected_items)

    @property
    def remaining_characters(self) -> int:
        return max(0, self.max_characters - self.estimated_characters)

    def with_usage(self, selected_items: int,
                   estimated_characters: int) -> "ContextBudget":
        """The same budget with observed usage filled in (used by the
        policy — inputs are validated by the constructor)."""
        return ContextBudget(
            max_items=self.max_items,
            max_characters=self.max_characters,
            selected_items=selected_items,
            estimated_characters=estimated_characters,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record — maximum + usage + remaining for both
        dimensions (the §29/§47 shape)."""
        return {
            "max_items": self.max_items,
            "selected_items": self.selected_items,
            "remaining_items": self.remaining_items,
            "max_characters": self.max_characters,
            "estimated_characters": self.estimated_characters,
            "remaining_characters": self.remaining_characters,
        }


@dataclass(frozen=True)
class ContextPackage:
    """WHAT the model sees for one reasoning step (M15 §10/§29).

    Construction invariant (enforced): the budget's recorded usage
    matches the selected items exactly — a package can never claim a
    usage its items do not have (M15 §12).
    """

    context_id: str
    task: str
    step_id: str
    budget: ContextBudget
    repository_scope: Optional[str] = None
    commit_scope: Optional[str] = None
    selected_items: Tuple[ContextItem, ...] = ()
    open_questions: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    selection_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        _require_text(self.context_id, "context_id")
        _require_text(self.task, "task")
        _require_text(self.step_id, "step_id")
        _optional_text(self.repository_scope, "repository_scope")
        _optional_text(self.commit_scope, "commit_scope")
        if not isinstance(self.budget, ContextBudget):
            raise _contract_violation(
                "budget must be a ContextBudget, got {!r}".format(
                    type(self.budget).__name__
                )
            )
        if isinstance(self.selected_items, (str, bytes)) or not isinstance(
            self.selected_items, (tuple, list)
        ):
            raise _contract_violation(
                "selected_items must be a sequence of ContextItem, got "
                "{!r}".format(type(self.selected_items).__name__)
            )
        for index, item in enumerate(self.selected_items):
            if not isinstance(item, ContextItem):
                raise _contract_violation(
                    "selected_items[{}] must be a ContextItem, got "
                    "{!r}".format(index, type(item).__name__)
                )
        object.__setattr__(self, "selected_items",
                           tuple(self.selected_items))
        object.__setattr__(
            self, "open_questions", _text_tuple(self.open_questions,
                                                "open_questions")
        )
        object.__setattr__(
            self, "constraints", _text_tuple(self.constraints, "constraints")
        )
        if not isinstance(self.selection_summary, dict):
            raise _contract_violation(
                "selection_summary must be a dict, got {!r}".format(
                    type(self.selection_summary).__name__
                )
            )
        if not isinstance(self.created_at, str) or not self.created_at:
            raise _contract_violation(
                "created_at must be a non-empty string, got {!r}".format(
                    self.created_at
                )
            )
        # Budget-usage invariant (M15 §12): recorded usage IS the
        # items — enforced here, at construction, not trusted later.
        if self.budget.selected_items != len(self.selected_items):
            raise _contract_violation(
                "budget.selected_items ({}) does not match {} selected "
                "items".format(
                    self.budget.selected_items, len(self.selected_items)
                )
            )
        estimated = sum(len(item.content) for item in self.selected_items)
        if self.budget.estimated_characters != estimated:
            raise _contract_violation(
                "budget.estimated_characters ({}) does not match the sum "
                "of selected item contents ({})".format(
                    self.budget.estimated_characters, estimated
                )
            )
        if self.budget.estimated_characters > self.budget.max_characters:
            raise _contract_violation(
                "budget.estimated_characters ({}) exceeds max_characters "
                "({})".format(
                    self.budget.estimated_characters,
                    self.budget.max_characters,
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record (canonical §29 projection)."""
        return {
            "context_id": self.context_id,
            "task": {"question": self.task},
            "step_id": self.step_id,
            "repository_scope": self.repository_scope,
            "commit_scope": self.commit_scope,
            "selected_items": [item.to_dict() for item in self.selected_items],
            "open_questions": list(self.open_questions),
            "constraints": list(self.constraints),
            "budget": self.budget.to_dict(),
            "selection_summary": dict(self.selection_summary),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ContextSelection:
    """The engine's verdict on ONE selection request (M15 §10): the
    package that would enter the model + every candidate's decision.

    Returned by ``ContextEngine.select``; ``ContextEngine.snapshot``
    turns it into a persisted record once the run/step identity is
    known.
    """

    package: ContextPackage
    decisions: Tuple[ContextSelectionDecision, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.package, ContextPackage):
            raise _contract_violation(
                "package must be a ContextPackage, got {!r}".format(
                    type(self.package).__name__
                )
            )
        if isinstance(self.decisions, (str, bytes)) or not isinstance(
            self.decisions, (tuple, list)
        ):
            raise _contract_violation(
                "decisions must be a sequence of "
                "ContextSelectionDecision, got {!r}".format(
                    type(self.decisions).__name__
                )
            )
        for index, decision in enumerate(self.decisions):
            if not isinstance(decision, ContextSelectionDecision):
                raise _contract_violation(
                    "decisions[{}] must be a ContextSelectionDecision, "
                    "got {!r}".format(index, type(decision).__name__)
                )
        object.__setattr__(self, "decisions", tuple(self.decisions))

    @property
    def excluded_candidate_ids(self) -> Tuple[str, ...]:
        return tuple(
            decision.candidate_id
            for decision in self.decisions
            if not decision.selected
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record: package + full decision audit."""
        return {
            "package": self.package.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "excluded_candidates": list(self.excluded_candidate_ids),
        }


@dataclass(frozen=True)
class ContextSnapshot:
    """The persisted record of ONE selection (M15 §11): package +
    every decision + the excluded ids, bound to a run and step.

    The snapshot is the inspection artifact — after the fact anyone
    can answer "what entered the model at step N of run R, and what
    was deliberately left out, and why" without re-running anything.
    """

    snapshot_id: str
    run_id: str
    step_id: str
    created_at: str
    package: ContextPackage
    selection_decisions: Tuple[ContextSelectionDecision, ...] = ()
    excluded_candidates: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.snapshot_id, "snapshot_id")
        _require_text(self.run_id, "run_id")
        _require_text(self.step_id, "step_id")
        _require_text(self.created_at, "created_at")
        if not isinstance(self.package, ContextPackage):
            raise _contract_violation(
                "package must be a ContextPackage, got {!r}".format(
                    type(self.package).__name__
                )
            )
        if isinstance(self.selection_decisions, (str, bytes)) or not isinstance(
            self.selection_decisions, (tuple, list)
        ):
            raise _contract_violation(
                "selection_decisions must be a sequence of "
                "ContextSelectionDecision, got {!r}".format(
                    type(self.selection_decisions).__name__
                )
            )
        for index, decision in enumerate(self.selection_decisions):
            if not isinstance(decision, ContextSelectionDecision):
                raise _contract_violation(
                    "selection_decisions[{}] must be a "
                    "ContextSelectionDecision, got {!r}".format(
                        index, type(decision).__name__
                    )
                )
        object.__setattr__(self, "selection_decisions",
                           tuple(self.selection_decisions))
        object.__setattr__(
            self, "excluded_candidates", _text_tuple(
                self.excluded_candidates, "excluded_candidates"
            )
        )
        # Internal consistency: the excluded ids ARE the non-selected
        # decisions — a snapshot can never under-report exclusions.
        expected = tuple(
            decision.candidate_id
            for decision in self.selection_decisions
            if not decision.selected
        )
        if self.excluded_candidates != expected:
            raise _contract_violation(
                "excluded_candidates must list exactly the non-selected "
                "decision ids, got {} vs {}".format(
                    self.excluded_candidates, expected
                )
            )

    # -- delegating views (the snapshot IS the package + audit) ---------

    @property
    def context_id(self) -> str:
        return self.package.context_id

    @property
    def task(self) -> str:
        return self.package.task

    @property
    def repository_scope(self) -> Optional[str]:
        return self.package.repository_scope

    @property
    def commit_scope(self) -> Optional[str]:
        return self.package.commit_scope

    @property
    def selected_items(self) -> Tuple[ContextItem, ...]:
        return self.package.selected_items

    @property
    def budget(self) -> ContextBudget:
        return self.package.budget

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready record (flattened §11/§29 projection — every
        minimum field present at the top level)."""
        return {
            "snapshot_id": self.snapshot_id,
            "context_id": self.context_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "created_at": self.created_at,
            "task": {"question": self.task},
            "repository_scope": self.repository_scope,
            "commit_scope": self.commit_scope,
            "selected_items": [item.to_dict() for item in
                               self.selected_items],
            "open_questions": list(self.package.open_questions),
            "constraints": list(self.package.constraints),
            "selection_decisions": [
                decision.to_dict() for decision in self.selection_decisions
            ],
            "excluded_candidates": list(self.excluded_candidates),
            "budget": self.budget.to_dict(),
            "selection_summary": dict(self.package.selection_summary),
        }
