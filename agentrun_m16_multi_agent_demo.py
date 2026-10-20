"""M16 — multi-agent foundations demonstration (one story, six phases).

    python agentrun_m16_multi_agent_demo.py                         # offline
    python agentrun_m16_multi_agent_demo.py --repository \
        tests/fixtures/construction                                 # fixture mode
    python agentrun_m16_multi_agent_demo.py --live                  # real model

The M16 demonstration (spec §15–§35): TWO specialized agents
collaborating through typed contracts — Investigator (tools, typed
``InvestigationReport``) → Analyst (NO tools, typed
``ArchitectureFinding``) → deterministic Validator (eight rules).

Story:
    1  permissions        both layers per role (definitions + registry)
    2  workflow run       one sequential collaboration, no bus/broker
    3  investigation      typed report, evidence refs, run id
    4  typed handoff      six messages, scope on every hop, payload types
    5  analysis           conclusions constrained to the report's
                          evidence; investigation id echoed
    6  validation/status  eight checks, task status, collaboration
                          events on state + agent traces (+ artifacts)

What this script does NOT do: no second orchestration framework, no
parallelism, no message broker, no LLM judge — a composition root over
``repo_intelligence`` (demo convention). The validator is pure code
over recorded state; the model never touches it.

Modes:
    offline (default)  fully deterministic: scripted model responses,
                       the REAL tool dispatcher reading a real file
                       (the pinned M1 snapshot by default, or
                       ``--repository <root>`` bound through the
                       dispatcher seam), the REAL MemoryManager /
                       ContextEngine / MultiAgentRunner. No API key.
    --live             the same workflow with a real model for BOTH
                       agents: requires OPENAI_API_KEY (never inspect
                       or commit .env). Deterministic validation may
                       reject a live answer — that is the point (exit 1,
                       never a fabricated success).

Scope labels (repository/commit) are demonstration DATA (§10 pattern);
the offline story never reads the Architecture Context.

Exit codes:
    0  every demonstration condition held
    1  a demonstration condition failed (tool isolation not visible,
        scope not preserved, validation failed, events missing, ...)
    6  configuration/execution failure (missing OPENAI_API_KEY in
        --live, missing readable target file, ContractError, ...)

The offline test suite (tests/multiagent/test_m16_multi_agent_demo.py)
drives the phase functions and main() with scripted doubles only —
never the network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    JsonFileTraceSink,
)
from repo_intelligence.context_engine import (  # noqa: E402
    ContextBudget,
    ContextEngine,
)
from repo_intelligence.llm import (  # noqa: E402
    LLMError,
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
    MemoryContext,
    MemoryManager,
    MemoryOwner,
    MemoryScope,
)
from repo_intelligence.multiagent import (  # noqa: E402
    ArchitectureFinding,
    COMPLETED,
    ContractError,
    InvestigationReport,
    MultiAgentRunner,
    RULE_ORDER,
    RunnerConfig,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (  # noqa: E402
    ANALYST_TOOLS,
    INVESTIGATOR_TOOLS,
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.tools.read_file import read_file  # noqa: E402
from repo_intelligence.tools.search_repository import (  # noqa: E402
    search_repository,
)

RULE = "=" * 72
SECTION = "-" * 52

EXIT_SUCCESS = 0
EXIT_DEMO_FAILED = 1
EXIT_EXECUTION_FAILED = 6

# Repository NAME as DATA (§10): the implementation is
# repository-agnostic; this is the label the demonstration uses.
REPOSITORY = "ADOP"
SUBJECT = "agents-tools-mcp"

# Offline scope: a visibly synthetic commit (demo labels only).
OFFLINE_COMMIT = "c" * 40

# Fixed demonstration identity: repeated offline runs overwrite the
# same trace artifacts (deterministic paths).
DEMO_RUN_ID = "m16_demo"

# The shared typed id of one offline collaboration (the live run's id
# comes from the Investigator's report; the Analyst must echo it).
OFFLINE_INVESTIGATION_ID = "inv-m16-demo-001"

DEFAULT_QUESTION = (
    "Explain how the agents, tools, and MCP servers in this "
    "repository interact."
)

# Candidate evidence files — default mode looks inside the project
# (names mirror the pinned snapshot), fixture mode inside --repository.
DEFAULT_READ_CANDIDATES = (
    "README.md",
    "pyproject.toml",
    "src/repo_intelligence/agent/agent.py",
)
FIXTURE_READ_CANDIDATES = (
    "src/service.py",
    "README.md",
    "main.py",
    "src/main.py",
)

DEMO_DB_PATH = PROJECT_ROOT / "runtime" / "agent_memory" / "m16_demo.db"
DEFAULT_BUDGET = ContextBudget(max_items=64, max_characters=6000)


# -- scripted model --------------------------------------------------------


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
    touches the network, fails loudly if over-invoked."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._queue: List[LLMResponse] = list(responses)
        self.requests: List[Any] = []
        self.model = "offline-scripted"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("model invoked more times than scripted")
        return self._queue.pop(0)


def offline_report_payload(
    question: str, repository: str, commit: str, evidence_path: str
) -> Dict[str, Any]:
    return {
        "investigation_id": OFFLINE_INVESTIGATION_ID,
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "findings": [
            {
                "statement": (
                    "{} was inspected and records the investigated "
                    "behavior (offline scripted finding).".format(
                        evidence_path
                    )
                ),
                "evidence_refs": [{"path": evidence_path}],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [{"path": evidence_path}],
        "investigated_sources": [evidence_path],
        "status": "complete",
    }


def offline_finding_payload(
    question: str, repository: str, commit: str, evidence_path: str
) -> Dict[str, Any]:
    return {
        "investigation_id": OFFLINE_INVESTIGATION_ID,
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "conclusions": [
            {
                "statement": (
                    "The inspected evidence shows the interaction the "
                    "question asks about (offline scripted "
                    "conclusion)."
                ),
                "classification": "observed",
                "evidence_refs": [{"path": evidence_path}],
            }
        ],
        "evidence_refs": [{"path": evidence_path}],
        "summary": "Evidence-backed, report-constrained conclusion.",
        "status": "complete",
    }


def build_offline_investigator(
    question: str, repository: str, commit: str, evidence_path: str
) -> OfflineModel:
    """Two deterministic responses: one real tool request (the REAL
    dispatcher reads the target), then the typed InvestigationReport
    whose refs match what that read will gather."""
    return OfflineModel(
        _tool_calls(
            ToolCall(
                id="call_m16_read",
                name="read_file",
                arguments={"path": evidence_path},
            )
        ),
        _structured(
            offline_report_payload(
                question, repository, commit, evidence_path
            )
        ),
    )


def build_offline_analyst(
    question: str, repository: str, commit: str, evidence_path: str
) -> OfflineModel:
    """One deterministic response: the typed ArchitectureFinding that
    echoes the investigation id and cites only the report's evidence."""
    return OfflineModel(
        _structured(
            offline_finding_payload(
                question, repository, commit, evidence_path
            )
        )
    )


# -- composition ------------------------------------------------------------


def demo_read_target(
    repository: Optional[str],
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """Find the evidence file for the offline story.

    Returns ``(absolute_path, repository_relative_path, root_label)``.
    ``root_label`` is ``None`` for the default (pinned snapshot)
    mode — bindings stay empty; a non-``None`` label binds the
    read-only text tools to that root through the dispatcher seam.
    """
    if repository is not None:
        root = Path(repository).expanduser()
        if not root.is_dir():
            return None, None, None
        for name in FIXTURE_READ_CANDIDATES:
            candidate = root / name
            if candidate.is_file():
                return candidate, name, str(root)
        return None, None, str(root)
    for name in DEFAULT_READ_CANDIDATES:
        candidate = PROJECT_ROOT / name
        if candidate.is_file():
            return candidate, name, None
    return None, None, None


def fixture_bindings(root: Path) -> Dict[str, Any]:
    """Bind the two read-only text tools to a foreign repository root
    via the documented ``tool(arguments)`` closure seam; the other two
    remain the pinned defaults (never called by the offline story)."""

    def read(arguments):
        return read_file(arguments, repository_root=root)

    def search(arguments):
        return search_repository(arguments, repository_root=root)

    bindings: Dict[str, Any] = {
        "read_file": read,
        "search_repository": search,
    }
    return bindings


def build_memory_manager(db_path: Optional[str] = None) -> MemoryManager:
    manager = MemoryManager(
        db_path if db_path is not None else str(DEMO_DB_PATH)
    )
    manager.initialize()
    return manager


def build_memory_context(
    repository: str, current_commit: str
) -> MemoryContext:
    return MemoryContext(
        scope=MemoryScope(repository=repository, commit=current_commit),
        subject=SUBJECT,
        owner=MemoryOwner(type="agent", id="repo_intelligence.agent"),
    )


def build_engine(
    repository: str, snapshot_root: Optional[str] = None
) -> ContextEngine:
    return ContextEngine(
        budget=DEFAULT_BUDGET,
        repository=repository,
        snapshot_root=snapshot_root,
    )


def build_workflow(
    question: str,
    repository: str,
    commit: str,
    investigator_llm,
    analyst_llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
    config: Optional[RunnerConfig] = None,
):
    """Compose the two EXISTING Agent instances with their role
    permissions and shared memory/context/trace dependencies, then the
    deterministic runner over them."""
    investigator = Agent(
        investigator_llm,
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=bindings
        ),
        trace_sink=trace_sink,
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )
    analyst = Agent(
        analyst_llm,
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )
    runner = MultiAgentRunner(
        investigator,
        analyst,
        trace_sink=trace_sink,
        config=config,
        clock=time.monotonic,
    )
    return runner, investigator, analyst


# -- printing helpers -------------------------------------------------------


def _clip(text: Optional[str], limit: int = 120) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


# -- phases -----------------------------------------------------------------


def run_permissions_phase(stream=None) -> Dict[str, Any]:
    """Phase 1 — both permission layers per role, printed before the
    workflow runs (what each model will be offered, what may execute)."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 1 — role permissions (definitions + registry)", file=stream)
    print(RULE, file=stream)
    inv_definitions = tool_definitions_for("investigator")
    ana_definitions = tool_definitions_for("analyst")
    inv_registry = restricted_dispatcher("investigator").tool_names
    ana_registry = restricted_dispatcher("analyst").tool_names
    print(
        "investigator: {} definitions [{}]; registry [{}]".format(
            len(inv_definitions),
            ", ".join(tool.name for tool in inv_definitions),
            ", ".join(inv_registry),
        ),
        file=stream,
    )
    print(
        "analyst:      {} definitions [{}]; registry [{}]".format(
            len(ana_definitions),
            ", ".join(tool.name for tool in ana_definitions) or "none",
            ", ".join(ana_registry) or "none",
        ),
        file=stream,
    )
    probe = restricted_dispatcher("analyst").dispatch(
        ToolCall(id="probe", name="read_file", arguments={"path": "README.md"})
    )
    print(
        "analyst probe read_file -> {}".format(
            "TOOL_NOT_FOUND (refused)"
            if probe.error is not None and probe.error.code == "TOOL_NOT_FOUND"
            else "EXECUTED (isolation violated)"
        ),
        file=stream,
    )
    ok = (
        len(INVESTIGATOR_TOOLS) == 4
        and ANALYST_TOOLS == ()
        and len(inv_definitions) == 4
        and ana_definitions == []
        and probe.error is not None
        and probe.error.code == "TOOL_NOT_FOUND"
    )
    print("permissions condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_investigation_phase(state, stream=None) -> Dict[str, Any]:
    """Phase 3 — the Investigator's typed evidence package."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 3 — investigation (typed InvestigationReport)", file=stream)
    print(RULE, file=stream)
    report = state.report
    if report is None:
        print("no report (status {})".format(state.status), file=stream)
        print("investigation condition: NO", file=stream)
        return {"ok": False}
    print("investigation_id: {}".format(report.investigation_id), file=stream)
    print("run id:           {}".format(state.investigator_run_id), file=stream)
    print("findings:         {}".format(len(report.findings)), file=stream)
    print(
        "evidence refs:    {}".format(
            ", ".join(
                ref.path
                + (
                    ""
                    if ref.line_start is None
                    else ":{}-{}".format(ref.line_start, ref.line_end)
                )
                for ref in report.evidence_refs
            )
        ),
        file=stream,
    )
    print(
        "open questions:   {}".format(len(report.open_questions)),
        file=stream,
    )
    for finding in report.findings:
        print(
            "  * [{}] {}".format(
                finding.support, _clip(finding.statement)
            ),
            file=stream,
        )
    ok = (
        isinstance(report, InvestigationReport)
        and bool(report.findings)
        and bool(report.evidence_refs)
        and bool(state.investigator_run_id)
    )
    print("investigation condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_handoff_phase(state, stream=None) -> Dict[str, Any]:
    """Phase 4 — the typed handoff: message order, payload types, and
    scope on every hop."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 4 — typed handoff (messages, scopes, payload types)", file=stream)
    print(RULE, file=stream)
    expected = [
        "INVESTIGATION_REQUEST",
        "INVESTIGATION_RESULT",
        "ANALYSIS_REQUEST",
        "ANALYSIS_RESULT",
        "VALIDATION_REQUEST",
        "VALIDATION_RESULT",
    ]
    observed = [message.message_type for message in state.messages]
    type_ok = observed == expected
    scope_ok = all(
        message.repository_scope == state.task.repository_scope
        and message.commit_scope == state.task.commit_scope
        for message in state.messages
    )
    payload_types = [type(message.payload).__name__ for message in state.messages]
    for message in state.messages:
        print(
            "  {} -> {}: {} (payload {})".format(
                message.sender_agent,
                message.recipient_agent,
                message.message_type,
                type(message.payload).__name__,
            ),
            file=stream,
        )
    print(
        "sequence: {}".format("six in order" if type_ok else "MISMATCH: {}".format(observed)),
        file=stream,
    )
    print(
        "scope on every hop: {}".format("preserved" if scope_ok else "BROKEN"),
        file=stream,
    )
    print(
        "payload types: {}".format(", ".join(payload_types)),
        file=stream,
    )
    ok = type_ok and scope_ok
    print("handoff condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_analysis_phase(state, stream=None) -> Dict[str, Any]:
    """Phase 5 — the Analyst's typed output, constrained to the
    report's evidence and id."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 5 — analysis (typed ArchitectureFinding)", file=stream)
    print(RULE, file=stream)
    finding = state.finding
    if finding is None or state.report is None:
        print("no finding (status {})".format(state.status), file=stream)
        print("analysis condition: NO", file=stream)
        return {"ok": False}
    print("investigation_id: {}".format(finding.investigation_id), file=stream)
    print(
        "id echoed:        {}".format(
            "yes"
            if finding.investigation_id == state.report.investigation_id
            else "NO"
        ),
        file=stream,
    )
    print("conclusions:      {}".format(len(finding.conclusions)), file=stream)
    for conclusion in finding.conclusions:
        print(
            "  * [{}] {} ({})".format(
                conclusion.classification,
                _clip(conclusion.statement),
                ", ".join(ref.path for ref in conclusion.evidence_refs)
                or "no refs",
            ),
            file=stream,
        )
    print("summary: {}".format(_clip(finding.summary)), file=stream)
    ok = (
        isinstance(finding, ArchitectureFinding)
        and finding.investigation_id == state.report.investigation_id
        and bool(finding.conclusions)
    )
    print("analysis condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_validation_phase(state, stream=None) -> Dict[str, Any]:
    """Phase 6 — the deterministic verdict: eight named checks, no
    model in the loop."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 6 — deterministic validation (no LLM)", file=stream)
    print(RULE, file=stream)
    verdict = state.validation
    if verdict is None:
        print("no validation (status {})".format(state.status), file=stream)
        print("validation condition: NO", file=stream)
        return {"ok": False}
    for check in verdict.checks:
        print(
            "  [{}] {:<22} {}".format(
                "ok" if check.ok else "FAIL",
                check.rule,
                "" if check.ok else _clip(check.detail, 200),
            ),
            file=stream,
        )
    rules = [check.rule for check in verdict.checks]
    order_ok = rules == list(RULE_ORDER)
    all_ok = all(check.ok for check in verdict.checks)
    print(
        "tool failures: {}".format(
            len(verdict.tool_failures)
            if verdict.tool_failures
            else "none"
        ),
        file=stream,
    )
    ok = order_ok and all_ok and verdict.status == "passed"
    print("validation condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_status_phase(state, trace_root, stream=None) -> Dict[str, Any]:
    """Phase 7 — task status, the collaboration timeline, and the
    trace artifacts that carry the same events."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 7 — status, events, trace artifacts", file=stream)
    print(RULE, file=stream)
    print("task status: {}".format(state.status), file=stream)
    print(
        "task id:     {}   messages: {}".format(
            state.task_id, len(state.messages)
        ),
        file=stream,
    )
    for event in state.collaboration_events:
        print(
            "  {:>3} {:<20} actor={:<12} {}".format(
                event["sequence"],
                event["operation"],
                event["actor"],
                event.get("message_type") or event.get("status") or "",
            ),
            file=stream,
        )
    sequences = [
        event["sequence"] for event in state.collaboration_events
    ]
    ordered = sequences == sorted(sequences) and len(sequences) == len(
        set(sequences)
    )
    trace_ok = True
    for role, run_id in (
        ("investigator", state.investigator_run_id),
        ("analyst", state.analyst_run_id),
    ):
        if run_id is None:
            trace_ok = False
            continue
        trace_path = Path(trace_root) / run_id / "trace.json"
        markdown_path = Path(trace_root) / run_id / "trace.md"
        if not trace_path.is_file():
            print("trace ({}): MISSING {}".format(role, trace_path), file=stream)
            trace_ok = False
            continue
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        events = payload.get("collaboration_events") or []
        markdown = (
            markdown_path.read_text(encoding="utf-8")
            if markdown_path.is_file()
            else ""
        )
        print(
            "trace ({}): {} collaboration events{}".format(
                role,
                len(events),
                ", markdown section present"
                if "## Collaboration" in markdown
                else ", markdown section MISSING",
            ),
            file=stream,
        )
        if not events or "## Collaboration" not in markdown:
            trace_ok = False
    ok = (
        state.status == COMPLETED
        and bool(state.collaboration_events)
        and ordered
        and trace_ok
    )
    print("status condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


# -- demo -------------------------------------------------------------------


def run_demo(
    question: str,
    repository: str,
    commit: str,
    investigator_llm,
    analyst_llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
    config: Optional[RunnerConfig] = None,
    trace_root=None,
    stream=None,
) -> Dict[str, Any]:
    """All six phases over one shared story. Returns every phase
    outcome plus the aggregate ``ok``."""
    stream = stream or sys.stdout
    permissions = run_permissions_phase(stream=stream)

    print(RULE, file=stream)
    print("Phase 2 — workflow run (Investigator → Analyst → Validator)", file=stream)
    print(RULE, file=stream)
    task = TaskPayload(
        question=question,
        repository_scope=repository,
        commit_scope=commit,
    )
    runner, investigator, analyst = build_workflow(
        question,
        repository,
        commit,
        investigator_llm,
        analyst_llm,
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
        bindings=bindings,
        config=config,
    )
    print(
        "scope: repository={} commit={}".format(
            repository, commit[:12]
        ),
        file=stream,
    )
    try:
        state = runner.run(task)
    except ContractError as exc:
        print("ContractError: {}".format(exc), file=sys.stderr)
        return {
            "permissions": permissions,
            "execution_failed": True,
            "ok": False,
        }
    print(
        "workflow finished: status={} messages={} events={}".format(
            state.status,
            len(state.messages),
            len(state.collaboration_events),
        ),
        file=stream,
    )

    investigation = run_investigation_phase(state, stream=stream)
    handoff = run_handoff_phase(state, stream=stream)
    analysis = run_analysis_phase(state, stream=stream)
    validation = run_validation_phase(state, stream=stream)
    status = run_status_phase(
        state,
        trace_root if trace_root is not None else "runtime/agent_runs",
        stream=stream,
    )

    ok = all(
        phase["ok"]
        for phase in (
            permissions,
            investigation,
            handoff,
            analysis,
            validation,
            status,
        )
    ) and not state.error

    print(RULE, file=stream)
    print("M16 demonstration summary", file=stream)
    print(RULE, file=stream)
    print(
        "permissions:    {}".format("yes" if permissions["ok"] else "NO"),
        file=stream,
    )
    print(
        "investigation:  {}".format("yes" if investigation["ok"] else "NO"),
        file=stream,
    )
    print(
        "handoff:        {}".format("yes" if handoff["ok"] else "NO"),
        file=stream,
    )
    print(
        "analysis:       {}".format("yes" if analysis["ok"] else "NO"),
        file=stream,
    )
    print(
        "validation:     {}".format("yes" if validation["ok"] else "NO"),
        file=stream,
    )
    print(
        "status/events:  {}".format("yes" if status["ok"] else "NO"),
        file=stream,
    )
    if state.error:
        print("workflow error: {}".format(state.error), file=stream)
    print(
        "final status:   {} (validated collaboration = "
        "COMPLETED only)".format(state.status),
        file=stream,
    )
    return {
        "state": state,
        "investigator": investigator,
        "analyst": analyst,
        "permissions": permissions,
        "investigation": investigation,
        "handoff": handoff,
        "analysis": analysis,
        "validation": validation,
        "status": status,
        "ok": ok,
    }


# -- main -------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m16_multi_agent_demo.py",
        description=(
            "M16 end-to-end demonstration: two specialized agents "
            "collaborating through typed contracts with deterministic "
            "validation — offline and deterministic by default."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use a real model for both agents "
        "(requires OPENAI_API_KEY); default is fully offline",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation question (default: the "
        "agents/tools/MCP interaction question)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id for --live (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--repository",
        default=None,
        help="foreign repository root: binds read_file/search_repository "
        "to it (e.g. tests/fixtures/construction); default reads the "
        "pinned M1 snapshot",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="memory database path (default: runtime/agent_memory/"
        "m16_demo.db — the demonstration's OWN database)",
    )
    parser.add_argument(
        "--snapshots",
        default=None,
        help="context snapshot root (default: runtime/context)",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="agent trace root (default: runtime/agent_runs)",
    )
    args = parser.parse_args(argv)

    question = args.question or DEFAULT_QUESTION

    # configuration (explicit .env load at the composition root —
    # same convention as the M10/M13/M14/M15 demos)
    if args.live:
        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "M16 demonstration (--live) requires OPENAI_API_KEY.\n"
                "Copy .env.example to .env, set OPENAI_API_KEY (never "
                "commit .env), and re-run — or run the default offline "
                "demonstration:\n"
                "    python agentrun_m16_multi_agent_demo.py",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        try:
            investigator_llm = OpenAILLMClient(
                model=args.model or DEFAULT_MODEL
            )
            analyst_llm = OpenAILLMClient(
                model=args.model or DEFAULT_MODEL
            )
        except LLMError as exc:
            print("Demonstration failed.", file=sys.stderr)
            print("LLMError: {}".format(exc), file=sys.stderr)
            return EXIT_EXECUTION_FAILED
        mode_label = "live (real model for both agents)"
    else:
        investigator_llm = None
        analyst_llm = None
        mode_label = "offline (deterministic, scripted model)"

    read_target, evidence_path, bound_root = demo_read_target(
        args.repository
    )
    if args.repository is not None and bound_root is None:
        print(
            "Demonstration failed: --repository {} is not a "
            "directory.".format(args.repository),
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED
    if evidence_path is None:
        print(
            "Demonstration failed: no readable target file under {} "
            "(looked for: {}).".format(
                args.repository or PROJECT_ROOT,
                ", ".join(
                    FIXTURE_READ_CANDIDATES
                    if args.repository
                    else DEFAULT_READ_CANDIDATES
                ),
            ),
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED

    repository_label = (
        Path(args.repository).name if args.repository else REPOSITORY
    )

    if investigator_llm is None:
        investigator_llm = build_offline_investigator(
            question, repository_label, OFFLINE_COMMIT, evidence_path
        )
        analyst_llm = build_offline_analyst(
            question, repository_label, OFFLINE_COMMIT, evidence_path
        )

    try:
        manager = build_memory_manager(args.db)
    except Exception as exc:
        print("Memory store failed: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    memory_context = build_memory_context(
        repository_label, OFFLINE_COMMIT
    )
    engine = build_engine(repository_label, args.snapshots)
    trace_sink = JsonFileTraceSink(args.traces) if args.traces else (
        JsonFileTraceSink()
    )
    trace_root = (
        Path(args.traces)
        if args.traces
        else Path("runtime") / "agent_runs"
    )
    bindings = (
        fixture_bindings(Path(bound_root)) if bound_root else None
    )

    print("M16 multi-agent demonstration — {}".format(mode_label))
    print(
        "scope: {}@{}   investigation: {}   target: {}".format(
            repository_label,
            OFFLINE_COMMIT[:12],
            OFFLINE_INVESTIGATION_ID
            if investigator_llm.model == "offline-scripted"
            else "(assigned by the model's report)",
            evidence_path,
        ),
    )
    print(file=sys.stdout)

    try:
        outcome = run_demo(
            question,
            repository_label,
            OFFLINE_COMMIT,
            investigator_llm,
            analyst_llm,
            trace_sink,
            manager,
            memory_context,
            engine,
            bindings=bindings,
            trace_root=trace_root,
        )
    except (
        AgentError,
        LLMError,
        StructuredOutputValidationError,
        ContractError,
    ) as exc:
        print("Demonstration execution failed.", file=sys.stderr)
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    if outcome.get("execution_failed"):
        return EXIT_EXECUTION_FAILED

    if not outcome["ok"]:
        failed = [
            name
            for name in (
                "permissions",
                "investigation",
                "handoff",
                "analysis",
                "validation",
                "status",
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
        "exit code: {} ({}; traces under {})".format(
            EXIT_SUCCESS, mode_label, trace_root
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
