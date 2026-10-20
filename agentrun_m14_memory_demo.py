"""M14 — persistent agent memory (two executions, two processes).

    python agentrun_m14_memory_demo.py
    python agentrun_m14_memory_demo.py --phase store
    python agentrun_m14_memory_demo.py --phase retrieve

The M14 demonstration (spec §35/§36/§37): selected validated
information from ONE planned investigation survives the end of that
execution in SQLite and is retrieved by a GENUINELY NEW execution —
a separate OS process that never saw Task 1's conversation.

    Execution 1 (this process, phase ``store``)
        Task 1 planned investigation → tools → evidence → M13 claim
        validation → deterministic memory formation policy
        → MemoryManager.store()
        → SQLite  (+ one clearly-labelled legacy memory scoped to an
        earlier commit, for the stale-memory demonstration)
        → process work ends

    Execution 2 (a NEW OS process, phase ``retrieve`` — spawned by
    this script, not a function call)
        Task 2 planned investigation → MemoryManager.search() →
        retrieved memory enters the planner request and messages →
        current investigation still runs → compare with current
        evidence → verify → stale legacy memory superseded (or
        retained), explicitly

What this script does NOT do: no second agent, no planner, no loop,
no validator of its own — a composition root over ``repo_intelligence``
(M10/M13 demo convention). Memory lifecycle calls here are the
APPLICATION-driven decisions the Agent deliberately does not make
itself (supersede/retain).

Exit codes:
    0  both phases completed (memory stored + retrieved + lifecycle OK)
    1  demonstration condition failed (e.g. nothing retrievable —
       Task 1 stored no memory, or a memory-store error was recorded)
    6  configuration/execution failure (missing OPENAI_API_KEY,
       missing Architecture Context, LLMError, AgentError,
       PlanValidationError, StructuredOutputValidationError,
       MemoryOperationError)

Credentials/configuration: requires OPENAI_API_KEY and the existing
M6 Architecture Context (same as the M13 demo — never inspect or
commit .env). The offline test suite
(tests/memory/test_m14_memory_demo.py) drives the phase functions and
main() with scripted doubles only — never the network.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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
from repo_intelligence.evidence import (  # noqa: E402
    STATUS_SUPPORTED,
)
from repo_intelligence.llm import (  # noqa: E402
    LLMError,
    StructuredOutputValidationError,
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

PHASE_STORE = "store"
PHASE_RETRIEVE = "retrieve"
PHASE_BOTH = "both"

# Repository NAME as DATA (§10): the memory implementation is
# repository-agnostic; this is the label the demonstration uses.
REPOSITORY = "ADOP"
SUBJECT = "agents-tools-mcp"

# Task 1: the M13 controlled investigation objective (claims required).
DEFAULT_OBJECTIVE_STORE = (
    "Explain how the ADOP agents, tools, and MCP servers are connected. "
    "Use the repository tools to gather evidence from files you "
    "actually inspect — do not rely on the initial context alone. "
    "Record your conclusion as structured output using explicit "
    "claims, where every claim contains the repository-relative paths "
    "of the files that support it."
)

# Task 2: memory is a starting point, never a substitute (§44).
DEFAULT_OBJECTIVE_RECHECK = (
    "Re-examine how the ADOP agents, tools, and MCP servers are "
    "connected, and state whether those connections still hold in the "
    "CURRENT repository state. Prior memory, when present, is only a "
    "starting point: confirm it against files you actually inspect "
    "with the repository tools, and let current evidence win on any "
    "conflict. Record your conclusion as structured output using "
    "explicit claims, where every claim contains the "
    "repository-relative paths of the files that support it."
)

# The clearly-labelled legacy memory for the stale demonstration: an
# "earlier investigation" outcome recorded at an OLDER commit (the
# current sha with its tail zeroed — visibly synthetic, printed as
# such). Repository evolution is simulated deterministically; the
# mechanism (scope mismatch → stale → supersede) is real.
LEGACY_STATEMENT = (
    "Earlier investigation: the ADOP agents, tools, and MCP servers "
    "are connected through agent definitions and MCP tool declarations."
)
LEGACY_MEMORY_TYPE = "episodic"


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


# -- composition -----------------------------------------------------------


def build_memory_manager(db_path=None) -> MemoryManager:
    """The demonstration's memory store: default runtime path unless
    given one (tests inject temporary databases). Schema creation is
    deterministic and idempotent (M14 §30) — a second OS process then
    finds the schema through ``sqlite_master`` without initializing."""
    manager = MemoryManager(db_path)
    manager.initialize()
    return manager


def build_memory_context(current_commit: str) -> MemoryContext:
    """One execution's memory configuration — injected with the
    manager (M14 §22): this repository scope at the current commit,
    the demonstration's subject, and this agent's owner (the same
    identity ``run_retrieve_phase`` searches with)."""
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
) -> Agent:
    """Compose the EXISTING Agent with the injected MemoryManager —
    one budget, one planned run, one claim-bearing contract, one
    memory boundary (M14 §22: no sqlite3 inside the Agent)."""
    return Agent(
        llm_client=llm_client,
        tool_dispatcher=tool_dispatcher,
        trace_sink=trace_sink,
        execution_config=AgentExecutionConfig(max_tool_calls=MAX_TOOL_CALLS),
        structured_output_model=InvestigationSummary,
        memory_manager=memory_manager,
        memory_context=memory_context,
    )


# -- printing (recorded data only) -----------------------------------------


def print_memories(memories, title, stream=None) -> None:
    stream = stream or sys.stdout
    print(title, file=stream)
    print(SECTION, file=stream)
    if not memories:
        print("(none)", file=stream)
    for memory in memories:
        print("* {} | {} | {}".format(
            memory.memory_id, memory.memory_type, memory.status
        ), file=stream)
        print("    scope: {}".format(_scope_label(memory)), file=stream)
        print("    {}".format(memory.statement), file=stream)
    print(file=stream)


def _scope_label(memory) -> str:
    if memory.scope.repository is None:
        return "(unscoped)"
    if memory.scope.commit is None:
        return memory.scope.repository
    return "{}@{}".format(memory.scope.repository, memory.scope.commit)


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


def print_plan(trace, stream=None) -> None:
    stream = stream or sys.stdout
    plan = trace.plan or {}
    print("Plan", file=stream)
    print(SECTION, file=stream)
    print("status: {}".format(plan.get("status")), file=stream)
    for step in plan.get("steps") or []:
        print(
            "{} [{}] {}".format(
                step.get("id"),
                step.get("status"),
                step.get("goal"),
            ),
            file=stream,
        )
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


def _has_memory_error(trace) -> bool:
    return any(
        isinstance(event.get("error"), dict)
        for event in (getattr(trace, "memory_events", []) or [])
    )


def _relative(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# -- phase 1: store --------------------------------------------------------


def run_store_phase(
    agent,
    manager,
    objective,
    context_text,
    context_record,
    current_commit,
    stream=None,
) -> dict:
    """Execution 1: planned investigation → M13 validation →
    deterministic memory formation policy (the Agent stores) → one
    legacy memory for the stale demonstration. Returns ``{"trace", "result",
    "memory_ids", "legacy_memory", "ok"}``."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M14 — Execution 1: TASK 1 → STORE", file=stream)
    print(RULE, file=stream)
    print("Objective:", file=stream)
    print(objective, file=stream)
    print(file=stream)
    print(
        "repository scope: {}@{}   max_tool_calls: {}".format(
            REPOSITORY, current_commit[:12], MAX_TOOL_CALLS
        ),
        file=stream,
    )
    print("memory database: {}".format(_relative(manager.db_path)), file=stream)
    print(file=stream)

    result = agent.run_planned(objective, context_text, context_record)
    trace = agent.last_trace

    print_plan(trace, stream=stream)
    print_execution(trace, stream=stream)
    print_claims(result, trace, stream=stream)
    print_memory_events(trace, stream=stream)

    stored = [
        event["memory_id"]
        for event in trace.memory_events
        if event.get("operation") == "memory.store"
        and event.get("memory_id")
    ]
    print("Stored by this execution", file=stream)
    print(SECTION, file=stream)
    print_memories(
        [manager.get(memory_id) for memory_id in stored],
        "Persisted memories",
        stream=stream,
    )

    # The stale-memory fixture: an "earlier investigation" outcome at
    # an older commit (clearly labelled — repository evolution is
    # simulated for the demonstration, the mechanism is real).
    legacy = manager.store(
        MemoryCandidate(
            owner=MemoryOwner(type="agent", id="repo_intelligence.agent"),
            memory_type=LEGACY_MEMORY_TYPE,
            statement=LEGACY_STATEMENT,
            scope=MemoryScope(
                repository=REPOSITORY,
                commit=legacy_commit_for(current_commit),
            ),
            subject=SUBJECT,
        )
    )
    print("Legacy memory (stale-memory fixture)", file=stream)
    print(SECTION, file=stream)
    print(
        "stored at an EARLIER commit (simulated repository history): "
        "{}@{}".format(REPOSITORY, legacy.scope.commit),
        file=stream,
    )
    print("* {} | {}".format(legacy.memory_id, legacy.statement), file=stream)
    print(file=stream)
    print(
        "Execution 1 complete — its conversation state ends here; only "
        "the SQLite file survives.",
        file=stream,
    )

    ok = stored and not _has_memory_error(trace)
    return {
        "trace": trace,
        "result": result,
        "memory_ids": stored,
        "legacy_memory": legacy,
        "ok": bool(ok),
    }


# -- phase 2: retrieve (runs in a NEW OS process) --------------------------


def run_retrieve_phase(
    agent,
    manager,
    objective,
    context_text,
    context_record,
    current_commit,
    stream=None,
) -> dict:
    """Execution 2: explicit search → the Agent's own retrieval at its
    execution boundary → current investigation → verify → stale
    decision (supersede when current evidence supports a replacement,
    otherwise retain). This process never held Task 1's messages."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("M14 — Execution 2: NEW PROCESS → RETRIEVE → INVESTIGATE", file=stream)
    print(RULE, file=stream)
    print(
        "OS pid {}: Task 1's messages, plan, and tool results never "
        "existed in this process — only the SQLite file was shared.".format(
            os.getpid()
        ),
        file=stream,
    )
    print(file=stream)

    # §13/§54: discovery through search(), not through ids.
    owner = MemoryOwner(type="agent", id="repo_intelligence.agent")
    prior = manager.search(
        repository=REPOSITORY,
        owner=owner,
        subject=SUBJECT,
        status="verified",
        limit=10,
    )
    print_memories(
        prior,
        "Retrieved persistent memory (MemoryManager.search)",
        stream=stream,
    )

    print("Objective:", file=stream)
    print(objective, file=stream)
    print(file=stream)

    result = agent.run_planned(objective, context_text, context_record)
    trace = agent.last_trace

    print_plan(trace, stream=stream)
    print_execution(trace, stream=stream)
    print_claims(result, trace, stream=stream)
    print_memory_events(trace, stream=stream)

    # scope freshness of what was retrieved (§26) — TRI-STATE
    # (fresh / stale / unknown), explicit and printed. Unknown is
    # never treated as fresh: it cannot be decided either way.
    print("Memory verification (scope check: fresh/stale/unknown)",
          file=stream)
    print(SECTION, file=stream)
    stale: List = []
    fresh: List = []
    unknown: List = []
    for memory in prior:
        outcome = manager.verify(memory.memory_id, current_commit)
        label = {
            "fresh": "fresh",
            "stale": "STALE",
            "unknown": "UNKNOWN",
        }[outcome.verification_status]
        print(
            "* {} {} — scope {} vs current {}".format(
                outcome.memory_id,
                label,
                outcome.scope_commit,
                outcome.current_commit,
            ),
            file=stream,
        )
        if outcome.verification_status == "stale":
            stale.append(memory)
        elif outcome.verification_status == "fresh":
            fresh.append(memory)
        else:
            unknown.append(memory)
    if not prior:
        print("(no prior memory was retrieved)", file=stream)
    print(file=stream)

    # explicit application decision (§26/§27): current evidence beats
    # old memory — supersede when this run validated supporting
    # claims; otherwise retain the old memory for its own scope.
    decisions: List[str] = []
    supported = [
        entry
        for entry in result.evidence_validation
        if entry.status == STATUS_SUPPORTED
    ]
    print("Stale-memory decision", file=stream)
    print(SECTION, file=stream)
    if not stale:
        print(
            "No stale memory — retrieved scope matches the current "
            "commit.",
            file=stream,
        )
    elif supported:
        current_claim = supported[0]
        for old in stale:
            replacement = manager.supersede(
                old.memory_id,
                MemoryCandidate(
                    owner=owner,
                    memory_type=LEGACY_MEMORY_TYPE,
                    statement=(
                        "Re-established at current commit {}: {}".format(
                            current_commit[:12],
                            current_claim.claim.statement,
                        )
                    ),
                    scope=MemoryScope(
                        repository=REPOSITORY, commit=current_commit
                    ),
                    evidence=tuple(current_claim.claim.evidence_refs),
                    subject=SUBJECT,
                    source_run_id=trace.run_id,
                ),
            )
            decisions.append("superseded")
            print(
                "* SUPERSEDED: {}".format(old.memory_id),
                file=stream,
            )
            print(
                "    old ({}): {}".format(
                    manager.get(old.memory_id).status, old.statement
                ),
                file=stream,
            )
            print(
                "    new: {} @ {}".format(
                    replacement.memory_id, current_commit[:12]
                ),
                file=stream,
            )
            print(
                "    reason: current evidence supports the finding "
                "({}, {})".format(
                    current_claim.claim.statement,
                    current_claim.status,
                ),
                file=stream,
            )
    else:
        for old in stale:
            decisions.append("retained")
            print(
                "* RETAINED: {} stays valid for its own scope — this "
                "run produced no validated replacement.".format(
                    old.memory_id
                ),
                file=stream,
            )
    print(file=stream)

    print_memories(
        manager.search(repository=REPOSITORY, subject=SUBJECT),
        "Memory store after Execution 2",
        stream=stream,
    )

    ok = bool(prior) and not _has_memory_error(trace)
    return {
        "trace": trace,
        "result": result,
        "prior": prior,
        "stale": stale,
        "fresh": fresh,
        "unknown": unknown,
        "decisions": decisions,
        "ok": ok,
    }


# -- process boundary (§36) -------------------------------------------------


def _spawn_retrieve_phase(model=None, db_path=None, question=None) -> int:
    """Launch phase 2 as a GENUINELY NEW OS process (the §36 proof):
    same SQLite file, no shared memory, no shared conversation."""
    argv = [sys.executable, str(PROJECT_ROOT / Path(__file__).name)]
    argv += ["--phase", PHASE_RETRIEVE]
    if model:
        argv += ["--model", model]
    if db_path:
        argv += ["--db", str(db_path)]
    if question:
        argv += ["--question", question]
    completed = subprocess.run(argv)
    return completed.returncode


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m14_memory_demo.py",
        description=(
            "M14 end-to-end demonstration: a planned investigation "
            "stores selected validated memory in SQLite; a SECOND "
            "OS PROCESS retrieves it, uses it as starting context, "
            "and validates current evidence against it (including a "
            "stale-memory supersede)."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=(PHASE_BOTH, PHASE_STORE, PHASE_RETRIEVE),
        default=PHASE_BOTH,
        help="both (default) runs store here, then spawns a new "
        "process for retrieve",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override Task 1's objective (default: the ADOP "
        "agents/tools/MCP investigation)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--db",
        default=None,
        help="memory database path (default: runtime/agent_memory/"
        "agent_memory.db)",
    )
    args = parser.parse_args(argv)

    # configuration (explicit .env load at the composition root —
    # same convention as the M9/M10/M13 demos)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M14 demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_m14_memory_demo.py",
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED

    if args.phase == PHASE_BOTH:
        # Execution 1 (store) runs in THIS process; execution 2 runs in
        # a NEW OS process (the §36 proof) — spawned with the same
        # configuration; it loads its own context and opens the SAME
        # SQLite file independently.
        print(
            "Phase store runs in THIS process; phase retrieve runs in "
            "a NEW OS process."
        )
        store_argv = ["--phase", PHASE_STORE]
        if args.question:
            store_argv += ["--question", args.question]
        if args.model:
            store_argv += ["--model", args.model]
        if args.db:
            store_argv += ["--db", args.db]
        store_exit = main(store_argv)
        if store_exit != EXIT_SUCCESS:
            return store_exit
        return _spawn_retrieve_phase(
            model=args.model, db_path=args.db, question=args.question
        )

    try:
        context_text, context_record = load_architecture_context()
    except ArchitectureContextError as exc:
        print("Agent execution failed.", file=sys.stderr)
        print("ArchitectureContextError: {}".format(exc), file=sys.stderr)
        print("Re-run M6 to regenerate the Architecture Context.", file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    current_commit = (context_record or {}).get("commit_sha") or ""
    if not current_commit:
        print(
            "Agent execution failed: Architecture Context record has no "
            "commit_sha (cannot scope memory).",
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED

    store_objective = args.question or DEFAULT_OBJECTIVE_STORE
    if args.phase == PHASE_RETRIEVE and args.question:
        recheck_objective = args.question
    else:
        recheck_objective = DEFAULT_OBJECTIVE_RECHECK

    try:
        llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        print("Agent execution failed.", file=sys.stderr)
        print("LLMError: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    try:
        manager = build_memory_manager(args.db)
    except MemoryOperationError as exc:
        print("Memory store failed: {}".format(exc.code), file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    memory_context = build_memory_context(current_commit)
    trace_sink = JsonFileTraceSink()
    agent = build_agent(
        llm_client,
        default_tool_dispatcher(),
        trace_sink,
        manager,
        memory_context,
    )

    if args.phase == PHASE_STORE:
        try:
            outcome = run_store_phase(
                agent,
                manager,
                store_objective,
                context_text,
                context_record,
                current_commit,
            )
        except (
            AgentError,
            LLMError,
            StructuredOutputValidationError,
            PlanValidationError,
            MemoryOperationError,
        ) as exc:
            print("Agent execution failed: {}".format(exc), file=sys.stderr)
            return EXIT_EXECUTION_FAILED
        if not outcome["ok"]:
            print(
                "Execution 1 did not persist memory (see errors above).",
                file=sys.stderr,
            )
            return EXIT_DEMO_FAILED
        print("exit code: {} (phase store)".format(EXIT_SUCCESS))
        return EXIT_SUCCESS

    try:
        outcome = run_retrieve_phase(
            agent,
            manager,
            recheck_objective,
            context_text,
            context_record,
            current_commit,
        )
    except (
        AgentError,
        LLMError,
        StructuredOutputValidationError,
        PlanValidationError,
        MemoryOperationError,
    ) as exc:
        print("Agent execution failed: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED
    if not outcome["ok"]:
        print(
            "Execution 2 retrieved NO prior memory — the two executions "
            "did not communicate through SQLite.",
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED
    print(
        "exit code: {} (phase retrieve: {} retrieved, {} stale, "
        "decision: {})".format(
            EXIT_SUCCESS,
            len(outcome["prior"]),
            len(outcome["stale"]),
            ", ".join(outcome["decisions"]) or "none-needed",
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
