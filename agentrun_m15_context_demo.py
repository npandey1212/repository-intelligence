"""M15 — context engineering demonstration (one story, nine phases).

    python agentrun_m15_context_demo.py            # offline (default)
    python agentrun_m15_context_demo.py --live     # real model

The M15 demonstration (spec §31–§41/§57): deterministic, inspectable
selection of what information enters the model's context for one
reasoning step — candidates → selection policy → ContextPackage →
render → model → tool → new candidates → new snapshot.

Story (§57):
    1  candidate inventory      (§33)  every source offers a candidate
    2  selection snapshot #1    (§34)  step 001: the policy decides
    3  model step               (§35)  render → model → response
    4  tool step                (§36)  dispatcher reads a real file
    5  snapshot #2 (evolution)  (§37)  step 002: new candidates enter
    6  memory interaction       (§38)  fresh selected; stale and
                                       unknown-freshness excluded
    7  budget demonstration     (§39)  hard ceilings; an impossible
                                       budget caught explicitly
    8  full agent run           (§40)  execution boundary: retrieve
                                       → select → render → investigate
                                       → M13 validation → memory
                                       formation → context events
    9  evaluation report        (§41)  deterministic metrics (§47)

What this script does NOT do: no second agent, no selection loop of
its own, no policy the package does not expose, no LLM judge — a
composition root over ``repo_intelligence`` (M10/M13/M14 demo
convention). Memory SQLite stays the only database; context snapshots
are JSON files (§43).

Modes:
    offline (default)  fully deterministic: scripted model responses,
                       the REAL tool dispatcher reading a real file,
                       the REAL engineering graph artifact when
                       present (a small synthetic document otherwise),
                       the REAL ContextEngine. No API key, no network.
    --live             the same story with a real model: requires
                       OPENAI_API_KEY and the existing M6
                       Architecture Context (never inspect or commit
                       .env).

Exit codes:
    0  every demonstration condition held
    1  a demonstration condition failed (isolation violated, snapshot
       missing, claims unsupported, context events absent, ...)
    6  configuration/execution failure (missing OPENAI_API_KEY in
       --live mode, missing Architecture Context, LLMError,
       AgentError, StructuredOutputValidationError,
       MemoryOperationError, GraphQueryError, ...)

The offline test suite (tests/context_engine/test_m15_context_demo.py)
drives the phase functions and main() with scripted doubles only —
never the network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydantic import BaseModel, Field, constr  # noqa: E402

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    AgentExecutionConfig,
    JsonFileTraceSink,
)
from repo_intelligence.context import (  # noqa: E402
    ArchitectureContextError,
)
from repo_intelligence.context_engine import (  # noqa: E402
    CONTEXT_BUDGET_EXCEEDED,
    REASON_STALE_MEMORY,
    REASON_UNKNOWN_FRESHNESS,
    REASON_VERIFIED_MEMORY,
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextOperationError,
    TaskContext,
    candidate_from_memory,
    candidate_from_task,
    candidates_from_tool_result,
    evaluate_context,
    evaluate_context_run,
    render_context_text,
    render_evaluation_report,
)
from repo_intelligence.engineering.validator import (  # noqa: E402
    EngineeringModelError,
)
from repo_intelligence.evidence import (  # noqa: E402
    STATUS_SUPPORTED,
)
from repo_intelligence.graph import (  # noqa: E402
    GraphQueryError,
    build_engineering_graph,
    load_engineering_graph_document,
    project_engineering_graph,
    query_engineering_graph,
)
from repo_intelligence.llm import (  # noqa: E402
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputValidationError,
    ToolCall,
)
from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.llm.openai_client import (  # noqa: E402
    DEFAULT_MODEL,
    OpenAILLMClient,
)
from repo_intelligence.memory import (  # noqa: E402
    MemoryCandidate,
    MemoryContext,
    MemoryManager,
    MemoryOperationError,
    MemoryOwner,
    MemoryScope,
)
from repo_intelligence.planning import (  # noqa: E402
    PlanValidationError,
    load_architecture_context,
)
from repo_intelligence.tool_dispatcher import (  # noqa: E402
    default_tool_dispatcher,
)

RULE = "=" * 72
SECTION = "-" * 52

MAX_TOOL_CALLS = 100

EXIT_SUCCESS = 0
EXIT_DEMO_FAILED = 1
EXIT_EXECUTION_FAILED = 6

# Repository NAME as DATA (§10): the context implementation is
# repository-agnostic; this is the label the demonstration uses.
REPOSITORY = "ADOP"
SUBJECT = "agents-tools-mcp"

# Offline scope: a visibly synthetic commit (the offline story never
# reads the Architecture Context; --live uses the real one).
OFFLINE_COMMIT = "c" * 40
OFFLINE_LEGACY_COMMIT = "c" * 32 + "0" * 8

# Fixed demonstration identity: repeated offline runs overwrite the
# same snapshot files (deterministic artifact paths — §42).
DEMO_RUN_ID = "m15_demo"
STEP_INITIAL = "001"
STEP_EVOLVED = "002"

# Demonstration budgets (§39): the default engine budget, a tight
# "what if" budget, and one that can never fit anything.
DEFAULT_BUDGET = ContextBudget(max_items=64, max_characters=6000)
TINY_BUDGET = ContextBudget(max_items=64, max_characters=400)
IMPOSSIBLE_BUDGET = ContextBudget(max_characters=5)

DEFAULT_QUESTION = (
    "How are the ADOP agents, tools, and MCP servers connected? "
    "Use the repository files you inspect as evidence for every "
    "claim."
)

# Where the demonstration looks for a real engineering graph (M17's
# canonical artifact); none found → a small synthetic document keeps
# the offline story hermetic.
GRAPH_GLOB = "artifacts/*/engineering/engineering_graph.json"

SYNTHETIC_GRAPH_DOCUMENT: Dict[str, Any] = {
    "schema_version": "engineering_graph.v1",
    "nodes": [
        {
            "id": "agent:context_demo",
            "kind": "agent",
            "label": "context demo agent",
        },
        {
            "id": "tool:read_file",
            "kind": "tool",
            "label": "read_file",
        },
        {
            "id": "mcp_server:demo",
            "kind": "mcp_server",
            "label": "demo MCP server",
        },
    ],
    "relationships": [
        {
            "id": "rel:USES:agent:context_demo:tool:read_file",
            "kind": "USES",
            "source": "agent:context_demo",
            "target": "tool:read_file",
        },
        {
            "id": "rel:EXPOSES:mcp_server:demo:tool:read_file",
            "kind": "EXPOSES",
            "source": "mcp_server:demo",
            "target": "tool:read_file",
        },
    ],
    "unresolved": [],
}

# The demonstration's three seeded memories (§38): one fresh, one
# scoped to an earlier commit (stale), one whose commit is unknown —
# the tri-state freshness story in one store.
FRESH_MEMORY_STATEMENT = (
    "Investigation outcome: ADOP agents, tools, and MCP servers are "
    "connected through declarative agent definitions that reference "
    "tools by name."
)
STALE_MEMORY_STATEMENT = (
    "Earlier investigation outcome: ADOP agents and MCP servers are "
    "wired together through a central registry file."
)
UNKNOWN_MEMORY_STATEMENT = (
    "Unscoped observation: widget-style tools are declared next to "
    "their agent definitions."
)

# Files the tool step (and the agent's own investigation) may read —
# first existing candidate keeps the story runnable anywhere.
READ_TARGET_CANDIDATES = (
    "README.md",
    "pyproject.toml",
    "src/repo_intelligence/agent/agent.py",
)

LEGACY_MEMORY_TYPE = "episodic"
DEMO_DB_PATH = PROJECT_ROOT / "runtime" / "agent_memory" / "m15_demo.db"


def legacy_commit_for(current_commit: str) -> str:
    """Deterministic 'earlier commit' derived from the current sha."""
    return current_commit[:32] + "0" * 8


# -- claim-bearing contract (M13 shape — demo-local copy) -------------------


class EvidenceReference(BaseModel):
    """One inspected-file pointer inside a claim (§3 contract)."""

    path: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Repository-relative path of a file the agent ACTUALLY "
            "INSPECTED during this run (read via read_file). Files "
            "that only appeared in search results, graph output, the "
            "architecture context, or the plan are NOT evidence."
        ),
    )
    line_start: Optional[int] = Field(
        None,
        description=(
            "First line read (only when both bounds are known; "
            "omit both otherwise)."
        ),
    )
    line_end: Optional[int] = Field(
        None,
        description="Last line read (same rule as line_start).",
    )


class InvestigationClaim(BaseModel):
    """One important factual claim, grounded in inspected files (§3)."""

    statement: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One specific factual conclusion of this investigation — "
            "no more than the inspected evidence supports; represent "
            "uncertainty as uncertainty rather than inventing it away."
        ),
    )
    evidence_refs: List[EvidenceReference] = Field(
        ...,
        min_items=1,
        description=(
            "The repository files actually inspected during this run "
            "that support this claim (at least one). The application "
            "validates every reference against this run's trace; a "
            "reference to a never-inspected file is recorded as "
            "CLAIM_EVIDENCE_INVALID."
        ),
    )


class InvestigationSummary(BaseModel):
    """The final structured response for this investigation.

    Output EXACTLY one JSON object matching this schema and nothing
    else. Every field is required: state the finding, list the
    files inspected as evidence_paths, and give at least one claim.
    A response missing any field — in particular "claims" — fails
    validation and the run fails. Conclusions and evidence
    references only; never hidden reasoning.
    """

    finding: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "A self-contained explanation of the investigation "
            "result — precise where the inspected evidence is "
            "precise, explicitly uncertain where it is not."
        ),
    )
    evidence_paths: List[constr(strip_whitespace=True, min_length=1)] = (
        Field(
            ...,
            min_items=1,
            description=(
                "Repository-relative paths of every file inspected as "
                "supporting evidence during this run."
            ),
        )
    )
    claims: List[InvestigationClaim] = Field(
        ...,
        min_items=1,
        description=(
            "REQUIRED, at least one: the important factual claims "
            "backing the finding; each cites the files this run "
            "inspected. The response fails validation without it — "
            "do not omit this field."
        ),
    )


# -- offline model (scripted, deterministic — never the network) ------------


def _text(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _structured(payload: Dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


class OfflineModel:
    """Scripted LLM double: answers strictly from its queue, never
    touches the network, fails loudly if asked more times than
    scripted (a demonstration condition, not a silent default)."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._queue: List[LLMResponse] = list(responses)
        self.requests: List[Any] = []
        self.model = "offline-scripted"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("model invoked more times than scripted")
        return self._queue.pop(0)


def build_offline_model(read_target: Path) -> OfflineModel:
    """The five deterministic responses of the offline story, in
    consumption order: the §35 model step, then the agent's PLANNED
    run (§40) — planner payload, the P1 tool request, P1's step
    result, and the final claim payload. Claim line bounds are
    computed from the SAME file the REAL dispatcher will read, so
    the M13 validator agrees."""
    total_lines = _count_lines(read_target)
    claim_end = min(9, total_lines)
    payload = {
        "finding": (
            "README.md is this repository's entry-point documentation "
            "(offline scripted conclusion)."
        ),
        "evidence_paths": ["README.md"],
        "claims": [
            {
                "statement": (
                    "The repository root contains README.md, read "
                    "lines 1-{} as evidence.".format(claim_end)
                ),
                "evidence_refs": [
                    {
                        "path": "README.md",
                        "line_start": 1,
                        "line_end": claim_end,
                    }
                ],
            }
        ],
    }
    plan = {
        "objective": DEFAULT_QUESTION,
        "steps": [
            {
                "id": "P1",
                "goal": (
                    "Inspect the repository entry-point documentation "
                    "and report what was inspected."
                ),
            },
            {
                "id": "P2",
                "goal": (
                    "Summarize the inspected documentation as "
                    "evidence-backed claims."
                ),
            },
        ],
    }
    return OfflineModel(
        # 1 — the §35 model step (consumed before the agent runs)
        _text(
            "Offline scripted model response: the selected context "
            "is deterministic — this reply was prepared without any "
            "network call."
        ),
        # 2 — the planner (M10: validated BEFORE any tool executes)
        _structured(plan),
        # 3 — step P1 asks for the documented inspection
        _tool_calls(
            ToolCall(
                id="call_read_001",
                name="read_file",
                arguments={"path": "README.md"},
            )
        ),
        # 4 — P1's result (the executed tool's observation)
        _text(
            "README.md inspected via read_file: lines 1-{} read; "
            "ready to summarize the evidence.".format(claim_end)
        ),
        # 5 — step P2's final structured answer
        _structured(payload),
    )


def _count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 1


def demo_read_target() -> Optional[Path]:
    """The first existing candidate file the tool step can read."""
    for name in READ_TARGET_CANDIDATES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            return candidate
    return None


# -- composition ------------------------------------------------------------


def build_task(question: str, current_commit: str) -> TaskContext:
    """The one task every phase selects for: this repository, this
    commit, the investigation's relevance tags."""
    return TaskContext(
        question=question,
        repository=REPOSITORY,
        commit=current_commit,
        relevance=("agents", "tools", "mcp"),
    )


def build_engine(snapshot_root=None) -> ContextEngine:
    """The demonstration's selection authority: bound to this
    repository (isolation guard) with the demonstration budget."""
    return ContextEngine(
        budget=DEFAULT_BUDGET, repository=REPOSITORY,
        snapshot_root=snapshot_root,
    )


def build_memory_manager(db_path=None) -> MemoryManager:
    """The demonstration's memory store: the demo database unless a
    path is given (tests inject temporary databases). Schema
    creation is deterministic and idempotent (M14 §30)."""
    manager = MemoryManager(db_path if db_path is not None else DEMO_DB_PATH)
    manager.initialize()
    return manager


def build_memory_context(current_commit: str) -> MemoryContext:
    """The agent's memory configuration — repository scope at the
    current commit, the demonstration's subject, this agent's owner."""
    return MemoryContext(
        scope=MemoryScope(repository=REPOSITORY, commit=current_commit),
        subject=SUBJECT,
        owner=MemoryOwner(type="agent", id="repo_intelligence.agent"),
    )


def build_agent(
    llm_client,
    tool_dispatcher,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
) -> Agent:
    """Compose the EXISTING Agent with the injected MemoryManager and
    ContextEngine — one budget, one run, one claim-bearing contract,
    one memory boundary, one context boundary (M14 §22 / M15 §5)."""
    return Agent(
        llm_client=llm_client,
        tool_dispatcher=tool_dispatcher,
        trace_sink=trace_sink,
        execution_config=AgentExecutionConfig(max_tool_calls=MAX_TOOL_CALLS),
        structured_output_model=InvestigationSummary,
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )


def seed_demo_memories(
    manager: MemoryManager, current_commit: str
) -> Dict[str, str]:
    """Seed (once) the three demonstration memories and return their
    ids keyed fresh/stale/unknown. Idempotent: an already-seeded
    database keeps its ids — the story reuses the same three rows."""
    existing = manager.search(
        repository=REPOSITORY, subject=SUBJECT, status="verified"
    )
    by_label: Dict[str, str] = {}
    for memory in existing:
        if memory.scope.commit == current_commit:
            by_label.setdefault("fresh", memory.memory_id)
        elif memory.scope.commit == legacy_commit_for(current_commit):
            by_label.setdefault("stale", memory.memory_id)
        elif memory.scope.commit is None:
            by_label.setdefault("unknown", memory.memory_id)
    if len(by_label) == 3:
        return by_label

    owner = MemoryOwner(type="agent", id="repo_intelligence.agent")
    seeds = (
        (
            "fresh",
            FRESH_MEMORY_STATEMENT,
            MemoryScope(repository=REPOSITORY, commit=current_commit),
        ),
        (
            "stale",
            STALE_MEMORY_STATEMENT,
            MemoryScope(
                repository=REPOSITORY,
                commit=legacy_commit_for(current_commit),
            ),
        ),
        (
            "unknown",
            UNKNOWN_MEMORY_STATEMENT,
            MemoryScope(repository=REPOSITORY, commit=None),
        ),
    )
    for label, statement, scope in seeds:
        if label in by_label:
            continue
        stored = manager.store(
            MemoryCandidate(
                owner=owner,
                memory_type=LEGACY_MEMORY_TYPE,
                statement=statement,
                scope=scope,
                subject=SUBJECT,
            )
        )
        by_label[label] = stored.memory_id
    return by_label


def build_graph(graph_path=None):
    """(graph, source label) — an explicit path, else the newest
    real artifact under ``artifacts/``, else the small synthetic
    document. Deterministic either way (M15 §17: the graph enters
    only through its own query abstraction)."""
    if graph_path is not None:
        document = load_engineering_graph_document(Path(graph_path))
        return project_engineering_graph(document), _relative(graph_path)
    found = sorted(PROJECT_ROOT.glob(GRAPH_GLOB))
    if found:
        chosen = found[-1]
        document = load_engineering_graph_document(chosen)
        return project_engineering_graph(document), _relative(chosen)
    return (
        build_engineering_graph(dict(SYNTHETIC_GRAPH_DOCUMENT)),
        "(synthetic fallback — no engineering graph artifact found)",
    )


def pick_query_start(graph) -> Optional[str]:
    """Deterministic neighborhood query start: prefer mcp_server,
    then tool, then any node (sorted) — never 'whatever the dict
    gave first'."""
    for kind in ("mcp_server", "tool", "code_entity", "artifact"):
        ids = sorted(
            node_id
            for node_id, data in graph.nodes(data=True)
            if data.get("kind") == kind
        )
        if ids:
            return ids[0]
    ids = sorted(graph.nodes)
    return ids[0] if ids else None


def graph_candidates(task: TaskContext, graph) -> List[ContextCandidate]:
    """Two graph-derived candidates (§17): a kind discovery summary
    and one bounded neighborhood — both through
    ``query_engineering_graph``, never raw graph access."""
    candidates: List[ContextCandidate] = []
    summary = query_engineering_graph(graph)
    kind_parts = [
        "{}:{}={}".format(
            entry.get("kind"), entry.get("subkind") or "-",
            entry.get("node_count"),
        )
        for entry in (summary.kinds or [])[:4]
    ]
    if not kind_parts:
        kind_parts = [
            "{}={}".format(key, summary.counts.get(key))
            for key in sorted(summary.counts)
        ]
    candidates.append(
        ContextCandidate(
            candidate_id="cand_graph_summary",
            source_type="graph",
            content=(
                "Graph discovery: {nodes} nodes, {relationships} "
                "relationships, {unresolved} unresolved claims; kinds "
                "(first groups): {kinds}.".format(
                    nodes=summary.counts.get("nodes", 0),
                    relationships=summary.counts.get("relationships", 0),
                    unresolved=summary.counts.get("unresolved", 0),
                    kinds=", ".join(kind_parts),
                )
            ),
            source_id="graph:summary",
            repository_scope=task.repository,
            priority=0,
            relevance=task.relevance,
        )
    )
    start = pick_query_start(graph)
    if start is not None:
        neighborhood = query_engineering_graph(
            graph, start_node_id=start, direction="both", max_depth=2
        )
        sample = sorted(
            str(node.get("id")) for node in neighborhood.nodes
        )[:3]
        candidates.append(
            ContextCandidate(
                candidate_id="cand_graph_neighborhood",
                source_type="graph",
                content=(
                    "Graph neighborhood of {} (both directions, depth "
                    "2): {} nodes, {} relationships; e.g. {}.".format(
                        start,
                        len(neighborhood.nodes),
                        len(neighborhood.relationships),
                        ", ".join(sample) or "(none)",
                    )
                ),
                source_id="graph:{}".format(start),
                repository_scope=task.repository,
                priority=0,
                relevance=task.relevance,
            )
        )
    return candidates


# -- printing (recorded data only) ------------------------------------------


def _clip(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "\u2026"


def _relative(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _scope_label(scope) -> str:
    if scope.repository_scope is None:
        return "(unscoped)"
    if scope.commit_scope is None:
        return scope.repository_scope
    return "{}@{}".format(
        scope.repository_scope, str(scope.commit_scope)[:12]
    )


def print_candidates(candidates, title, stream=None) -> None:
    stream = stream or sys.stdout
    print(title, file=stream)
    print(SECTION, file=stream)
    print(
        "{:<24} {:<11} {:<12} {:<16} {:<9} {}".format(
            "candidate", "source", "authority", "scope", "freshness",
            "content",
        ),
        file=stream,
    )
    for candidate in candidates:
        print(
            "{:<24} {:<11} {:<12} {:<16} {:<9} {}".format(
                candidate.candidate_id,
                candidate.source_type,
                candidate.authority_value,
                _clip(_scope_label(candidate), 16),
                candidate.freshness or "-",
                _clip(candidate.content, 72),
            ),
            file=stream,
        )
    print(file=stream)


def print_package(snapshot, path, stream=None) -> None:
    stream = stream or sys.stdout
    package = snapshot.package
    print("Context package (persisted snapshot)", file=stream)
    print(SECTION, file=stream)
    print(
        "context: {}   run: {}   step: {}".format(
            package.context_id, snapshot.run_id, snapshot.step_id
        ),
        file=stream,
    )
    print("artifact: {}".format(_relative(path) if path else "(not "
                                "persisted)"), file=stream)
    print("scope: {}".format(_scope_label(package)), file=stream)
    budget = package.budget
    print(
        "budget: {}/{} items, {}/{} characters".format(
            budget.selected_items,
            budget.max_items,
            budget.estimated_characters,
            budget.max_characters,
        ),
        file=stream,
    )
    print(
        "selection summary: {}".format(
            json.dumps(package.selection_summary, sort_keys=True)
        ),
        file=stream,
    )
    print("selected items (presentation order)", file=stream)
    if not package.selected_items:
        print("  (none)", file=stream)
    for item in package.selected_items:
        print(
            "  {} {} [{}] {}".format(
                item.context_item_id,
                item.source_type,
                item.selection_reason,
                _clip(item.content, 96),
            ),
            file=stream,
        )
    print(file=stream)


def print_decisions(snapshot, stream=None) -> None:
    stream = stream or sys.stdout
    print("Every candidate's decision (nothing dropped silently)",
          file=stream)
    print(SECTION, file=stream)
    for decision in snapshot.selection_decisions:
        line = "* {} -> {}".format(
            decision.candidate_id,
            "SELECTED" if decision.selected else "excluded",
        )
        line += " | {}".format(decision.reason)
        line += " | authority {}".format(decision.authority)
        if decision.freshness is not None:
            line += " | freshness {}".format(decision.freshness)
        if decision.duplicate_of is not None:
            line += " | duplicate of {}".format(decision.duplicate_of)
        print(line, file=stream)
    print(file=stream)


def print_diff(added, removed, stream=None) -> None:
    stream = stream or sys.stdout
    print("Evolution diff (step 001 -> step 002)", file=stream)
    print(SECTION, file=stream)
    print(
        "added: {}".format(", ".join(added) if added else "(none)"),
        file=stream,
    )
    print(
        "removed: {}".format(", ".join(removed) if removed else "(none)"),
        file=stream,
    )
    print(file=stream)


def print_memory_outcomes(outcomes, stream=None) -> None:
    stream = stream or sys.stdout
    print("Memory interaction (freshness tri-state — §38)", file=stream)
    print(SECTION, file=stream)
    for label in ("fresh", "stale", "unknown"):
        entry = outcomes.get(label)
        if entry is None:
            print("* {}: (missing)".format(label), file=stream)
            continue
        print(
            "* {} memory {} -> {} | reason {}".format(
                label,
                entry["candidate_id"],
                "SELECTED" if entry["selected"] else "excluded",
                entry["reason"],
            ),
            file=stream,
        )
    print(file=stream)


def print_context_events(trace, stream=None) -> None:
    stream = stream or sys.stdout
    print("Context events (this run's trace — §40)", file=stream)
    print(SECTION, file=stream)
    events = getattr(trace, "context_events", []) or []
    if not events:
        print("(none)", file=stream)
    for event in events:
        line = "* {}".format(event.get("operation"))
        if event.get("step_id"):
            line += " — step {}".format(event["step_id"])
        if event.get("candidate_count") is not None:
            line += " — candidates: {}".format(event["candidate_count"])
        if event.get("selected_count") is not None:
            line += " — selected: {}".format(event["selected_count"])
        if event.get("excluded_count") is not None:
            line += " — excluded: {}".format(event["excluded_count"])
        if event.get("characters") is not None:
            line += " — characters: {}".format(event["characters"])
        if event.get("reasons"):
            parts = [
                "{}={}".format(reason, event["reasons"][reason])
                for reason in sorted(event["reasons"])
            ]
            line += " — reasons: {}".format(", ".join(parts))
        error = event.get("error")
        if isinstance(error, dict):
            line += " — ERROR {}: {}".format(
                error.get("code"), error.get("message")
            )
        print(line, file=stream)
    print(file=stream)


def print_claims(result, trace, stream=None) -> None:
    stream = stream or sys.stdout
    print("Claims and evidence (current run)", file=stream)
    print(SECTION, file=stream)
    validation = trace.evidence_validation or {}
    results = validation.get("results") or []
    if not results:
        print("(no structured claims)", file=stream)
    for index, entry in enumerate(results, start=1):
        claim = entry.get("claim") or {}
        print("{}. {}".format(index, claim.get("statement")), file=stream)
        for ref in claim.get("evidence_refs") or []:
            path = ref.get("path")
            if ref.get("line_start") is not None:
                path = "{}:{}-{}".format(
                    path, ref.get("line_start"), ref.get("line_end")
                )
            print("   evidence: {}".format(path), file=stream)
        print("   validation: {}".format(entry.get("status")), file=stream)
    print(file=stream)


def print_execution(trace, stream=None) -> None:
    stream = stream or sys.stdout
    print("Execution", file=stream)
    print(SECTION, file=stream)
    for iteration in trace.iterations:
        for result in iteration.tool_results:
            print(
                "tool: {} ({})".format(
                    result.get("tool"),
                    "ok" if result.get("success") else "failed",
                ),
                file=stream,
            )
    print(
        "iterations: {}   termination: {}".format(
            len(trace.iterations), trace.termination_reason
        ),
        file=stream,
    )
    print(file=stream)


def print_plan(trace, stream=None) -> None:
    """Print the run's plan from the trace — planner execution,
    plan generation, and deterministic validation (§40), all read
    from recorded state, never re-derived."""
    stream = stream or sys.stdout
    print("=== PLANNER ===", file=stream)
    print("Plan (planner iteration ran before any tool)", file=stream)
    print(SECTION, file=stream)
    plan = getattr(trace, "plan", None) or {}
    if not plan:
        print("(no plan recorded)", file=stream)
        print(file=stream)
        return
    print("objective: {}".format(plan.get("objective")), file=stream)
    steps = plan.get("steps") or []
    print("plan generated: {} step(s)".format(len(steps)), file=stream)
    for step in steps:
        print(
            "* {} [{}] {}".format(
                step.get("id"), step.get("status"), step.get("goal")
            ),
            file=stream,
        )
        if step.get("result"):
            print(
                "  result: {}".format(_clip(step.get("result"), 240)),
                file=stream,
            )
    print(
        "plan validated: status {}".format(plan.get("status")),
        file=stream,
    )
    print(file=stream)


def print_memory_events(trace, stream=None) -> None:
    stream = stream or sys.stdout
    print("Memory operations (this run's trace)", file=stream)
    print(SECTION, file=stream)
    events = getattr(trace, "memory_events", []) or []
    if not events:
        print("(none)", file=stream)
    for event in events:
        line = "* {}".format(event.get("operation"))
        if event.get("count") is not None:
            line += " — {} retrieved".format(event["count"])
        if event.get("memory_id"):
            line += " — {}".format(event["memory_id"])
        if event.get("verification_status"):
            line += " — verification: {}".format(
                event["verification_status"]
            )
        elif event.get("stale") is not None:
            line += " — stale: {}".format(
                "yes" if event["stale"] else "no"
            )
        if event.get("status"):
            line += " — status {}".format(event["status"])
        error = event.get("error")
        if isinstance(error, dict):
            line += " — ERROR {}: {}".format(
                error.get("code"), error.get("message")
            )
        print(line, file=stream)
    print(file=stream)


def _has_error(events) -> bool:
    return any(
        isinstance(event.get("error"), dict) for event in (events or [])
    )


# -- phase 1: candidate inventory (§33) -------------------------------------


def run_inventory_phase(
    task,
    manager,
    memory_ids,
    graph,
    graph_label,
    stream=None,
) -> Dict[str, Any]:
    """Every source offers what it has — BEFORE any policy runs."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 1: CANDIDATE INVENTORY (§33)", file=stream)
    print(RULE, file=stream)
    print(
        "task: {}".format(_clip(task.question, 96)), file=stream
    )
    print(
        "scope: {}@{}   graph: {}".format(
            REPOSITORY, task.commit[:12], graph_label
        ),
        file=stream,
    )
    print("memory database: {}".format(_relative(manager.db_path)),
          file=stream)
    print(file=stream)

    candidates: List[ContextCandidate] = [
        candidate_from_task(task, priority=0, candidate_id="cand_task"),
        ContextCandidate(
            candidate_id="cand_repo",
            source_type="repository",
            content=(
                "Repository {} at commit {}: agents, tools, and MCP "
                "servers are the documented subject of this "
                "investigation.".format(REPOSITORY, task.commit[:12])
            ),
            source_id="repository:{}".format(REPOSITORY),
            repository_scope=task.repository,
            commit_scope=task.commit,
            priority=5,
            relevance=task.relevance,
        ),
    ]
    for label in ("fresh", "stale", "unknown"):
        memory_id = memory_ids.get(label)
        if not memory_id:
            continue
        memory = manager.get(memory_id)
        candidates.append(
            candidate_from_memory(
                memory,
                current_commit=task.commit,
                relevance=task.relevance,
                candidate_id="cand_mem_{}".format(label),
            )
        )
    candidates.extend(graph_candidates(task, graph))

    print_candidates(
        candidates,
        "Candidate inventory — what each source OFFERS (§33)",
        stream=stream,
    )
    by_source: Dict[str, int] = {}
    for candidate in candidates:
        by_source[candidate.source_type] = (
            by_source.get(candidate.source_type, 0) + 1
        )
    ok = (
        by_source.get("task") == 1
        and by_source.get("repository") == 1
        and by_source.get("memory") == 3
        and by_source.get("graph", 0) >= 1
    )
    return {
        "task": task,
        "candidates": candidates,
        "by_source": by_source,
        "ok": ok,
    }


# -- phase 2: selection snapshot #1 (§34) -----------------------------------


def run_selection_phase(
    engine, task, candidates, run_id, step_id, stream=None
) -> Dict[str, Any]:
    """The policy decides; the snapshot persists the full audit."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "M15 — Phase 2: SELECTION SNAPSHOT #1 (step {}) (§34)".format(
            step_id
        ),
        file=stream,
    )
    print(RULE, file=stream)
    print("=== STEP 1 ===", file=stream)

    snapshot = engine.select_snapshot(
        task, candidates, run_id, step_id
    )
    path = engine.persist(snapshot)
    print_package(snapshot, path, stream=stream)
    print_decisions(snapshot, stream=stream)
    metrics = evaluate_context(snapshot)
    ok = metrics["selected"] > 0 and metrics["isolation_ok"]
    return {
        "snapshot": snapshot,
        "path": path,
        "metrics": metrics,
        "ok": ok,
    }


# -- phase 3: model step (§35) ----------------------------------------------


def run_model_step_phase(llm, package, question, stream=None) -> Dict[str, Any]:
    """Render → model → response: the package is exactly what the
    model receives."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 3: MODEL STEP (§35)", file=stream)
    print(RULE, file=stream)

    rendered = render_context_text(package)
    print("Rendered context (the model input)", file=stream)
    print(SECTION, file=stream)
    print(_clip(rendered, 900), file=stream)
    print(
        "… [{} characters total — clipped for display only]".format(
            len(rendered)
        ),
        file=stream,
    )
    print(file=stream)

    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="Question: {}\n\n{}".format(question, rendered),
            )
        ]
    )
    response = llm.complete(request)
    print("Model response", file=stream)
    print(SECTION, file=stream)
    print(_clip(response.message.content, 400) or "(empty)",
          file=stream)
    print(file=stream)
    ok = bool(response.message.content and response.message.content.strip())
    return {
        "rendered": rendered,
        "request": request,
        "response": response,
        "ok": ok,
    }


# -- phase 4: tool step (§36) -----------------------------------------------


def run_tool_step_phase(dispatcher, read_target, stream=None) -> Dict[str, Any]:
    """The dispatcher executes against the real repository (offline
    too — no network involved)."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 4: TOOL STEP (§36)", file=stream)
    print(RULE, file=stream)

    if read_target is None:
        print("No readable target file found — tool step skipped.",
              file=stream)
        print(file=stream)
        return {"result": None, "call": None, "ok": False}

    relative = _relative(read_target)
    call = ToolCall(
        id="call_read_tool_step",
        name="read_file",
        arguments={"path": relative},
    )
    result = dispatcher.dispatch(call)
    data = result.data or {}
    print(
        "dispatched: {}({})".format(call.name, relative), file=stream
    )
    line_start = data.get("line_start")
    line_end = data.get("line_end")
    if line_start is None or line_end is None:
        print(
            "success: {}".format(result.success), file=stream
        )
    else:
        print(
            "success: {}   lines {}-{} ({} characters)".format(
                result.success,
                line_start,
                line_end,
                len(str(data.get("content") or "")),
            ),
            file=stream,
        )
    print("content preview: {}".format(_clip(data.get("content"), 200)),
          file=stream)
    print(file=stream)
    return {"result": result, "call": call, "ok": bool(result.success)}


# -- phase 5: snapshot #2, context evolution (§37) --------------------------


def tool_candidates(task, tool_result) -> List[ContextCandidate]:
    """What the tool step now offers — the SAME adaptation the real
    Agent applies inside its loop (``candidates_from_tool_result``,
    M15 §7/§8): the raw result (authority 200) plus one evidence
    pointer (authority 500) into the inspected range, pinned to this
    task's repository/commit. The demonstration composes the shared
    adapter with stable ids instead of re-implementing it."""
    return candidates_from_tool_result(
        tool_result,
        task,
        candidate_id="cand_tool_read",
        evidence_candidate_id="cand_evidence_read",
    )


def run_evolution_phase(
    engine, task, prior_candidates, tool_result, run_id, stream=None
) -> Dict[str, Any]:
    """New candidates arrive after the tool step: select again for
    step 002 and diff the two packages."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 5: SNAPSHOT #2 — CONTEXT EVOLUTION (§37)",
          file=stream)
    print(RULE, file=stream)
    print("=== STEP 2 ===", file=stream)

    new_candidates = tool_candidates(task, tool_result)
    candidates = list(prior_candidates) + new_candidates
    snapshot = engine.select_snapshot(
        task, candidates, run_id, STEP_EVOLVED
    )
    path = engine.persist(snapshot)
    print_package(snapshot, path, stream=stream)
    print_decisions(snapshot, stream=stream)
    return {
        "snapshot": snapshot,
        "path": path,
        "candidates": candidates,
        "new_candidates": new_candidates,
        "metrics": evaluate_context(snapshot),
        "ok": True,
    }


def diff_snapshots(snapshot1, snapshot2) -> Dict[str, List[str]]:
    """Candidate-level evolution: selected ids added/removed between
    the two persisted steps."""
    selected1 = {
        decision.candidate_id
        for decision in snapshot1.selection_decisions
        if decision.selected
    }
    selected2 = {
        decision.candidate_id
        for decision in snapshot2.selection_decisions
        if decision.selected
    }
    return {
        "added": sorted(selected2 - selected1),
        "removed": sorted(selected1 - selected2),
    }


# -- phase 6: memory interaction (§38) --------------------------------------


def run_memory_phase(snapshot, memory_ids, stream=None) -> Dict[str, Any]:
    """Freshness made visible: the same three memories' decisions in
    the persisted audit."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 6: MEMORY INTERACTION (§38)", file=stream)
    print(RULE, file=stream)

    decisions = {
        decision.candidate_id: decision
        for decision in snapshot.selection_decisions
    }
    outcomes: Dict[str, Dict[str, Any]] = {}
    for label in ("fresh", "stale", "unknown"):
        candidate_id = "cand_mem_{}".format(label)
        decision = decisions.get(candidate_id)
        if decision is None:
            outcomes[label] = {
                "candidate_id": candidate_id,
                "selected": False,
                "reason": "(absent from this selection)",
            }
            continue
        outcomes[label] = {
            "candidate_id": candidate_id,
            "selected": decision.selected,
            "reason": decision.reason,
            "freshness": decision.freshness,
        }
    print_memory_outcomes(outcomes, stream=stream)
    ok = (
        outcomes.get("fresh", {}).get("selected") is True
        and outcomes.get("fresh", {}).get("reason") == REASON_VERIFIED_MEMORY
        and outcomes.get("stale", {}).get("selected") is False
        and outcomes.get("stale", {}).get("reason") == REASON_STALE_MEMORY
        and outcomes.get("unknown", {}).get("selected") is False
        and outcomes.get("unknown", {}).get("reason")
        == REASON_UNKNOWN_FRESHNESS
    )
    if not ok:
        print(
            "MEMORY INTERACTION CONDITION FAILED (see reasons above).",
            file=stream,
        )
    print(file=stream)
    return {"outcomes": outcomes, "ok": ok}


# -- phase 7: budget demonstration (§39) ------------------------------------


def run_budget_phase(engine, task, candidates, stream=None) -> Dict[str, Any]:
    """Hard ceilings in action: a tight budget that still selects
    within its limit, and an impossible budget that fails loudly."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 7: BUDGET DEMONSTRATION (§39)", file=stream)
    print(RULE, file=stream)

    tiny = engine.select(
        task, candidates, budget=TINY_BUDGET, step_id=STEP_INITIAL
    )
    budget = tiny.package.budget
    print(
        "Tight budget (max {} items, {} chars): selected {} items, "
        "{} chars".format(
            budget.max_items,
            budget.max_characters,
            budget.selected_items,
            budget.estimated_characters,
        ),
        file=stream,
    )
    blocked = [
        decision
        for decision in tiny.decisions
        if decision.reason == "BUDGET_EXCEEDED"
    ]
    print(
        "budget-blocked candidates: {}".format(
            ", ".join(decision.candidate_id for decision in blocked)
            or "(none)"
        ),
        file=stream,
    )
    print(
        "package within budget: {}".format(
            budget.estimated_characters <= budget.max_characters
            and budget.selected_items <= budget.max_items
        ),
        file=stream,
    )
    print(file=stream)

    impossible_error: Optional[str] = None
    try:
        engine.select(
            task, candidates, budget=IMPOSSIBLE_BUDGET,
            step_id=STEP_INITIAL,
        )
    except ContextOperationError as exc:
        impossible_error = exc.code
    print("Impossible budget (max {} items, {} chars)".format(
        IMPOSSIBLE_BUDGET.max_items,
        IMPOSSIBLE_BUDGET.max_characters,
    ), file=stream)
    print(SECTION, file=stream)
    if impossible_error is not None:
        print(
            "caught ContextOperationError: {} (never a silent empty "
            "or over-budget package)".format(impossible_error),
            file=stream,
        )
    else:
        print(
            "UNEXPECTED: no error was raised for an impossible "
            "budget.",
            file=stream,
        )
    print(file=stream)

    ok = (
        budget.estimated_characters <= budget.max_characters
        and budget.selected_items <= budget.max_items
        and budget.selected_items > 0
        and bool(blocked)
        and impossible_error == CONTEXT_BUDGET_EXCEEDED
    )
    return {
        "tiny": tiny,
        "blocked": blocked,
        "impossible_error": impossible_error,
        "ok": ok,
    }


# -- phase 8: full agent run (§40) ------------------------------------------


def run_agent_phase(
    llm,
    dispatcher,
    manager,
    memory_context,
    engine,
    question,
    trace_sink=None,
    stream=None,
    architecture_context=None,
    architecture_context_record=None,
) -> Dict[str, Any]:
    """One complete investigation with BOTH boundaries live: the
    planner runs first (validated plan, recorded on the trace),
    then M14 memory retrieval + context selection at the execution
    edge, M13 validation after the loop, deterministic memory
    formation — all inside ONE agent run id."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 8: PLANNER + FULL AGENT RUN (§40)", file=stream)
    print(RULE, file=stream)
    print(
        "Objective:", file=stream,
    )
    print(question, file=stream)
    print(file=stream)

    agent = build_agent(
        llm, dispatcher, trace_sink, manager, memory_context, engine
    )
    result = agent.run_planned(
        question, architecture_context, architecture_context_record
    )
    trace = agent.last_trace

    print_plan(trace, stream=stream)
    print_execution(trace, stream=stream)
    print_claims(result, trace, stream=stream)
    print_memory_events(trace, stream=stream)
    print_context_events(trace, stream=stream)

    supported = [
        entry
        for entry in result.evidence_validation
        if entry.status == STATUS_SUPPORTED
    ]
    context_events = getattr(trace, "context_events", []) or []
    operations = set(event.get("operation") for event in context_events)
    stored = any(
        event.get("operation") == "memory.store"
        for event in (getattr(trace, "memory_events", []) or [])
    )

    # §17/§18 — the REAL run's per-step lifecycle, printed from
    # recorded data: every reasoning boundary this agent executed,
    # its persisted snapshot, and the context the model actually
    # read at that step.
    snapshots = list(getattr(agent, "last_context_snapshots", []) or [])
    step_selects = [
        event
        for event in context_events
        if event.get("operation") == "context.select"
        and event.get("step_id")
    ]
    step_ids = [event["step_id"] for event in step_selects]
    triggers = {
        event["step_id"]: event.get("trigger") for event in step_selects
    }
    print("Agent context steps (real execution — §17)", file=stream)
    print(SECTION, file=stream)
    if not snapshots:
        print("(none)", file=stream)
    for index, snapshot in enumerate(snapshots, start=1):
        metrics = evaluate_context(snapshot)
        print(
            "=== STEP {} ===  (step {}, trigger {})".format(
                index,
                snapshot.step_id,
                triggers.get(snapshot.step_id) or "-",
            ),
            file=stream,
        )
        print(
            "candidates: {}   selected: {}   excluded: {}".format(
                metrics["considered"],
                metrics["selected"],
                metrics["excluded"],
            ),
            file=stream,
        )
        print(
            "context: {}".format(snapshot.package.context_id), file=stream
        )
        print(
            "rendered: {}".format(
                _clip(render_context_text(snapshot.package), 400)
            ),
            file=stream,
        )
        snapshot_path = None
        if engine is not None and hasattr(engine, "snapshot_root"):
            snapshot_path = (
                Path(engine.snapshot_root)
                / trace.run_id
                / "context_{}.json".format(snapshot.step_id)
            )
        print(
            "snapshot: {}".format(
                _relative(snapshot_path) if snapshot_path else "(none)"
            ),
            file=stream,
        )
        print(file=stream)
    run_metrics = evaluate_context_run(snapshots)
    if len(snapshots) >= 2:
        first_selected = {
            decision.candidate_id
            for decision in snapshots[0].selection_decisions
            if decision.selected
        }
        last_selected = {
            decision.candidate_id
            for decision in snapshots[-1].selection_decisions
            if decision.selected
        }
        print(
            "agent context evolution: +{} added, -{} removed "
            "(evolution steps: {})".format(
                len(last_selected - first_selected),
                len(first_selected - last_selected),
                run_metrics["context_evolution_count"],
            ),
            file=stream,
        )
    else:
        print(
            "agent context evolution: none "
            "(evolution steps: {})".format(
                run_metrics["context_evolution_count"]
            ),
            file=stream,
        )
    print(
        "snapshot ownership: {} snapshot(s) belong to run {}".format(
            len(snapshots), trace.run_id
        ),
        file=stream,
    )
    print(file=stream)

    # The demonstration condition now REQUIRES the per-step
    # lifecycle (§17): selection at the boundary (001), a
    # re-selection after the executed tool (002, …), one persisted
    # snapshot per boundary, and no recorded context error.
    evolution_ok = (
        len(step_ids) >= 2
        and step_ids[0] == "001"
        and step_ids == sorted(step_ids)
        and len(step_ids) == len(set(step_ids))
    )
    snapshots_ok = len(snapshots) == len(step_ids)
    if snapshots_ok and engine is not None and hasattr(
        engine, "snapshot_root"
    ):
        snapshots_ok = all(
            (
                Path(engine.snapshot_root)
                / trace.run_id
                / "context_{}.json".format(snapshot.step_id)
            ).is_file()
            for snapshot in snapshots
        )
    # §40/§7 — the run must be a PLAN-driven run whose validated
    # plan executed, and every persisted snapshot must belong to
    # this exact trace run (same run_id, §14 ownership).
    plan = getattr(trace, "plan", None) or {}
    plan_steps = plan.get("steps") or []
    plan_ok = (
        plan.get("status") == "completed"
        and len(plan_steps) >= 2
        and all(step.get("id") for step in plan_steps)
    )
    linkage_ok = bool(snapshots) and all(
        snapshot.run_id == trace.run_id for snapshot in snapshots
    )
    ok = (
        bool(result.evidence_validation)
        and len(supported) == len(result.evidence_validation)
        and "context.select" in operations
        and "context.render" in operations
        and evolution_ok
        and snapshots_ok
        and plan_ok
        and linkage_ok
        and agent.last_context_error is None
        and not _has_error(getattr(trace, "memory_events", []) or [])
        and stored
    )
    if not ok:
        print("AGENT RUN CONDITION FAILED (see output above).",
              file=stream)
    print(file=stream)
    return {
        "agent": agent,
        "trace": trace,
        "result": result,
        "plan": plan,
        "context_events": context_events,
        "snapshots": snapshots,
        "step_ids": step_ids,
        "metrics": run_metrics,
        "ok": ok,
    }


# -- phase 9: evaluation report (§41/§47) -----------------------------------


def run_evaluation_phase(snapshot1, snapshot2, stream=None) -> Dict[str, Any]:
    """The deterministic quality report — no LLM judge, no prose
    scoring: counts, coverage, isolation, reasons."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M15 — Phase 9: EVALUATION REPORT (§41/§47)", file=stream)
    print(RULE, file=stream)

    metrics1 = evaluate_context(snapshot1)
    metrics2 = evaluate_context(snapshot2)
    report1 = render_evaluation_report(metrics1)
    report2 = render_evaluation_report(metrics2)
    print("Snapshot #1 (step {})".format(snapshot1.step_id), file=stream)
    print(SECTION, file=stream)
    print(report1, file=stream)
    print(file=stream)
    print("Snapshot #2 (step {})".format(snapshot2.step_id), file=stream)
    print(SECTION, file=stream)
    print(report2, file=stream)
    print(file=stream)
    ok = metrics1["isolation_ok"] and metrics2["isolation_ok"]
    return {
        "metrics1": metrics1,
        "metrics2": metrics2,
        "report1": report1,
        "report2": report2,
        "ok": ok,
    }


# -- orchestration ----------------------------------------------------------


def run_demo(
    llm,
    dispatcher,
    manager,
    engine,
    memory_ids,
    current_commit,
    question,
    graph,
    graph_label,
    read_target,
    run_id=DEMO_RUN_ID,
    trace_sink=None,
    stream=None,
    architecture_context=None,
    architecture_context_record=None,
) -> Dict[str, Any]:
    """All nine phases over one shared story (§57). Returns every
    phase outcome plus the aggregate ``ok``."""
    stream = stream or sys.stdout
    task = build_task(question, current_commit)

    inventory = run_inventory_phase(
        task, manager, memory_ids, graph, graph_label, stream=stream
    )
    selection = run_selection_phase(
        engine, task, inventory["candidates"], run_id, STEP_INITIAL,
        stream=stream,
    )
    model_step = run_model_step_phase(
        llm, selection["snapshot"].package, question, stream=stream
    )
    tool_step = run_tool_step_phase(dispatcher, read_target, stream=stream)
    if tool_step["ok"]:
        evolution = run_evolution_phase(
            engine,
            task,
            inventory["candidates"],
            tool_step["result"],
            run_id,
            stream=stream,
        )
    else:
        evolution = {
            "snapshot": selection["snapshot"],
            "path": selection["path"],
            "candidates": inventory["candidates"],
            "new_candidates": [],
            "metrics": selection["metrics"],
            "ok": False,
        }
    diff = diff_snapshots(
        selection["snapshot"], evolution["snapshot"]
    )
    print_diff(diff["added"], diff["removed"], stream=stream)

    memory_phase = run_memory_phase(
        evolution["snapshot"], memory_ids, stream=stream
    )
    budget_phase = run_budget_phase(
        engine, task, evolution["candidates"], stream=stream
    )
    agent_phase = run_agent_phase(
        llm,
        dispatcher,
        manager,
        build_memory_context(current_commit),
        engine,
        question,
        trace_sink=trace_sink,
        stream=stream,
        architecture_context=architecture_context,
        architecture_context_record=architecture_context_record,
    )
    evaluation = run_evaluation_phase(
        selection["snapshot"], evolution["snapshot"], stream=stream
    )

    # §41 — what the demonstration proved, from recorded data only.
    print(RULE, file=stream)
    print("M15 demonstration summary (§41)", file=stream)
    print(RULE, file=stream)
    print(
        "candidates considered: {} -> {}".format(
            evaluation["metrics1"]["considered"],
            evaluation["metrics2"]["considered"],
        ),
        file=stream,
    )
    print(
        "selected: {} -> {}".format(
            evaluation["metrics1"]["selected"],
            evaluation["metrics2"]["selected"],
        ),
        file=stream,
    )
    print(
        "context evolution: +{} added, -{} removed".format(
            len(diff["added"]), len(diff["removed"])
        ),
        file=stream,
    )
    print(
        "memory: fresh SELECTED, stale excluded, unknown excluded — "
        "{}".format("yes" if memory_phase["ok"] else "NO"),
        file=stream,
    )
    print(
        "budget respected: {}".format("yes" if budget_phase["ok"]
                                      else "NO"),
        file=stream,
    )
    print(
        "agent run (claims validated, context events recorded): "
        "{}".format("yes" if agent_phase["ok"] else "NO"),
        file=stream,
    )
    plan_steps = (agent_phase.get("plan") or {}).get("steps") or []
    print(
        "plan executed: {}".format(
            " -> ".join(
                step.get("id") or "?" for step in plan_steps
            )
            or "(none)"
        ),
        file=stream,
    )
    print(
        "agent context steps: {} — evolution steps: {} "
        "(snapshots persisted per reasoning boundary)".format(
            " -> ".join(agent_phase.get("step_ids") or []) or "(none)",
            (agent_phase.get("metrics") or {}).get(
                "context_evolution_count", 0
            ),
        ),
        file=stream,
    )
    print(
        "isolation: {}".format(
            "PASS" if evaluation["ok"] else "FAIL"
        ),
        file=stream,
    )
    print("artifacts: {}".format(_relative(selection["path"])),
          file=stream)
    if evolution["path"] != selection["path"]:
        print("           {}".format(_relative(evolution["path"])),
              file=stream)
    for snapshot in agent_phase.get("snapshots") or []:
        if engine is None or not hasattr(engine, "snapshot_root"):
            continue
        agent_path = (
            Path(engine.snapshot_root)
            / agent_phase["trace"].run_id
            / "context_{}.json".format(snapshot.step_id)
        )
        print("           {}".format(_relative(agent_path)), file=stream)
    print(file=stream)

    ok = all(
        phase["ok"]
        for phase in (
            inventory,
            selection,
            model_step,
            tool_step,
            evolution,
            memory_phase,
            budget_phase,
            agent_phase,
            evaluation,
        )
    )
    return {
        "task": task,
        "inventory": inventory,
        "selection": selection,
        "model_step": model_step,
        "tool_step": tool_step,
        "evolution": evolution,
        "diff": diff,
        "memory_phase": memory_phase,
        "budget_phase": budget_phase,
        "agent_phase": agent_phase,
        "evaluation": evaluation,
        "ok": ok,
    }


# -- main -------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m15_context_demo.py",
        description=(
            "M15 end-to-end demonstration: deterministic context "
            "selection over real candidates — inventory, snapshots, "
            "model step, tool step, memory interaction, budget, and "
            "a full agent run with trace-recorded context events."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use a real model + the M6 Architecture Context "
        "(requires OPENAI_API_KEY); default is fully offline",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation question (default: the ADOP "
        "agents/tools/MCP connection question)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id for --live (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--db",
        default=None,
        help="memory database path (default: runtime/agent_memory/"
        "m15_demo.db — the demonstration's OWN database)",
    )
    parser.add_argument(
        "--snapshots",
        default=None,
        help="context snapshot root (default: runtime/context)",
    )
    parser.add_argument(
        "--graph",
        default=None,
        help="engineering graph artifact path (default: newest "
        "artifacts/*/engineering/engineering_graph.json, else a "
        "small synthetic document)",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="agent trace root (default: runtime/agent_runs)",
    )
    args = parser.parse_args(argv)

    question = args.question or DEFAULT_QUESTION

    # configuration (explicit .env load at the composition root —
    # same convention as the M9/M10/M13/M14 demos)
    if args.live:
        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "M15 demonstration (--live) requires OPENAI_API_KEY.\n"
                "Copy .env.example to .env, set OPENAI_API_KEY (never "
                "commit .env), and re-run — or run the default offline "
                "demonstration:\n"
                "    python agentrun_m15_context_demo.py",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        try:
            context_text, context_record = load_architecture_context()
        except ArchitectureContextError as exc:
            print("Demonstration failed.", file=sys.stderr)
            print(
                "ArchitectureContextError: {}".format(exc), file=sys.stderr
            )
            print(
                "Re-run M6 to regenerate the Architecture Context.",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        current_commit = (context_record or {}).get("commit_sha") or ""
        if not current_commit:
            print(
                "Demonstration failed: Architecture Context record has "
                "no commit_sha (cannot scope context).",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        try:
            llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
        except LLMError as exc:
            print("Demonstration failed.", file=sys.stderr)
            print("LLMError: {}".format(exc), file=sys.stderr)
            return EXIT_EXECUTION_FAILED
        mode_label = "live (real model)"
    else:
        current_commit = OFFLINE_COMMIT
        # Offline runs have no M6 artifact loaded — the planner
        # still receives the step-001 context (memory + task), the
        # same input the live run's Architecture Context augments.
        context_text = None
        context_record = None
        llm_client = None  # built below, after the read target exists
        mode_label = "offline (deterministic, scripted model)"

    try:
        manager = build_memory_manager(args.db)
    except MemoryOperationError as exc:
        print("Memory store failed: {}".format(exc.code), file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    memory_ids = seed_demo_memories(manager, current_commit)

    try:
        graph, graph_label = build_graph(args.graph)
    except (EngineeringModelError, GraphQueryError) as exc:
        print("Graph load failed: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    read_target = demo_read_target()
    if llm_client is None:
        if read_target is None:
            print(
                "Demonstration failed: no readable target file "
                "(looked for: {}).".format(
                    ", ".join(READ_TARGET_CANDIDATES)
                ),
                file=sys.stderr,
            )
            return EXIT_DEMO_FAILED
        llm_client = build_offline_model(read_target)

    engine = build_engine(args.snapshots)
    trace_sink = JsonFileTraceSink(args.traces) if args.traces else (
        JsonFileTraceSink()
    )

    print("M15 context engineering demonstration — {}".format(mode_label))
    print(
        "scope: {}@{}   run: {}".format(
            REPOSITORY, current_commit[:12], DEMO_RUN_ID
        )
    )
    print(file=sys.stdout)

    try:
        outcome = run_demo(
            llm_client,
            default_tool_dispatcher(),
            manager,
            engine,
            memory_ids,
            current_commit,
            question,
            graph,
            graph_label,
            read_target,
            run_id=DEMO_RUN_ID,
            trace_sink=trace_sink,
            architecture_context=context_text,
            architecture_context_record=context_record,
        )
    except (
        AgentError,
        LLMError,
        StructuredOutputValidationError,
        PlanValidationError,
        MemoryOperationError,
        ContextOperationError,
        GraphQueryError,
        EngineeringModelError,
    ) as exc:
        print("Demonstration execution failed.", file=sys.stderr)
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    if not outcome["ok"]:
        failed = [
            name
            for name in (
                "inventory",
                "selection",
                "model_step",
                "tool_step",
                "evolution",
                "memory_phase",
                "budget_phase",
                "agent_phase",
                "evaluation",
            )
            if not outcome[name]["ok"]
        ]
        print(
            "Demonstration condition failed in: {}.".format(
                ", ".join(failed)
            ),
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED
    print(
        "exit code: {} ({} — snapshots under {})".format(
            EXIT_SUCCESS, mode_label, _relative(engine.snapshot_root)
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
