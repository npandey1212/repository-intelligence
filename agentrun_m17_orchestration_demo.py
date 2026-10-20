"""M17 — multi-agent orchestration demonstration (one story, seven phases).

    python agentrun_m17_orchestration_demo.py                       # offline
    python agentrun_m17_orchestration_demo.py --repository \
        tests/fixtures/construction                                 # fixture mode
    python agentrun_m17_orchestration_demo.py --live                # real model

The M17 demonstration (spec §4–§56): the application takes ONE plan
(M10 planner + deterministic plan gate), assigns each step a role with
a reason, executes the steps sequentially over the two M16 agents under
explicit bounds, and finishes in ONE sealed terminal state — with the
M16 eight-rule validator as the only verdict.

Story:
    1  plan gate       M10 planner → ``validate_plan`` → plan P1–P4
    2  roles           deterministic step → role, with a reason
    3  execution       sequential steps, typed outputs, bounded runs
    4  handoffs        six messages, scope on every hop, budget counters
    5  validation      the M16 verdict over the MERGED evidence package
    6  failure         two explicit terminal failures — an invalid plan
                       runs NO agent; a small message bound stops the
                       run with the analysis step BLOCKED
    7  status/traces   terminal status, event timeline, orchestration
                       events mirrored onto each agent's own trace

What this script does NOT do: no second planning framework, no second
agent framework, no parallelism, no broker, no LLM judge — a
composition root over ``repo_intelligence`` (demo convention).

Modes:
    offline (default)  fully deterministic: scripted planner, per-step
                       scripted agent responses, the REAL tool
                       dispatcher reading a real file (the pinned
                       snapshot by default, or ``--repository <root>``
                       bound through the dispatcher seam), the REAL
                       MemoryManager / ContextEngine / Orchestrator.
                       No API key.
    --live             the same story with a real model for the planner
                       and both agents: requires OPENAI_API_KEY (never
                       inspect or commit .env). Deterministic validation
                       may reject a live answer — that is the point
                       (exit 1, never a fabricated success).

Exit codes:
    0  every demonstration condition held
    1  a demonstration condition failed (plan gate, role assignment,
        terminal status, blocked-step semantics, validation, ...)
    6  configuration/execution failure (missing OPENAI_API_KEY in
        --live, missing readable target file, planner/agent raised, ...)

The offline test suite (tests/orchestration/test_m17_demo.py) drives
the phase functions and main() with scripted doubles only — never the
network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    ContractError,
    InvestigationReport,
    RULE_ORDER,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (  # noqa: E402
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import (  # noqa: E402
    BUDGET_EXCEEDED,
    COMPLETED,
    PLAN_FAILED,
    OrchestrationConfig,
    Orchestrator,
    assign_roles,
    is_terminal,
)
from repo_intelligence.planning.models import (  # noqa: E402
    InvestigationPlan,
    PlanStep,
)
from repo_intelligence.planning.planner import (  # noqa: E402
    build_planning_request,
    plan_from_response,
)
from repo_intelligence.planning.validation import (  # noqa: E402
    PlanValidationError,
    validate_plan,
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

OFFLINE_COMMIT = "c" * 40
OFFLINE_INVESTIGATION_ID = "inv-m17-demo-001"
DEFAULT_QUESTION = (
    "Explain how the application boundary is enforced in this "
    "repository."
)

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

DEMO_DB_PATH = PROJECT_ROOT / "runtime" / "agent_memory" / "m17_demo.db"
DEFAULT_BUDGET = ContextBudget(max_items=64, max_characters=6000)

# The plan the offline planner produces (spec §46: one plan, four
# steps, evidence before interpretation).
OFFLINE_PLAN_STEPS = (
    ("P1", "Identify the application entry points"),
    ("P2", "Investigate the tool dispatcher and role permissions"),
    ("P3", "Collect evidence about the typed message handoffs"),
    ("P4", "Explain how the boundary is enforced across the workflow"),
)

EXPECTED_HANDOFFS = [
    "INVESTIGATION_REQUEST",
    "INVESTIGATION_RESULT",
    "ANALYSIS_REQUEST",
    "ANALYSIS_RESULT",
    "VALIDATION_REQUEST",
    "VALIDATION_RESULT",
]


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


def offline_plan_payload(
    objective: str, repository: str, commit: str
) -> Dict[str, Any]:
    return {
        "objective": objective,
        "steps": [
            {"id": step_id, "goal": goal}
            for step_id, goal in OFFLINE_PLAN_STEPS
        ],
    }


def offline_report_payload(
    question: str,
    repository: str,
    commit: str,
    evidence_path: str,
    index: int,
) -> Dict[str, Any]:
    """One Investigator step's evidence package — every step inspects
    the same demonstration file from its own angle, so the merge shows
    findings concatenated and refs de-duplicated."""
    return {
        "investigation_id": "inv-m17-demo-{:03d}".format(index),
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "findings": [
            {
                "statement": (
                    "step {} inspected {} and recorded its own angle "
                    "(offline scripted finding).".format(
                        OFFLINE_PLAN_STEPS[index - 1][0], evidence_path
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
    question: str,
    repository: str,
    commit: str,
    evidence_path: str,
) -> Dict[str, Any]:
    return {
        "investigation_id": OFFLINE_INVESTIGATION_ID,
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "conclusions": [
            {
                "statement": (
                    "the boundary is enforced by the role-restricted "
                    "dispatcher (offline scripted conclusion)."
                ),
                "classification": "observed",
                "evidence_refs": [{"path": evidence_path}],
            }
        ],
        "evidence_refs": [{"path": evidence_path}],
        "summary": "Evidence-first, interpretation last.",
        "status": "complete",
    }


def build_offline_planner(
    objective: str, repository: str, commit: str
) -> OfflineModel:
    return OfflineModel(
        _structured(offline_plan_payload(objective, repository, commit))
    )


def build_offline_investigator(
    question: str, repository: str, commit: str, evidence_path: str
) -> OfflineModel:
    """Three Investigator steps: read the evidence file, then report."""
    responses: List[LLMResponse] = []
    for index in range(1, len(OFFLINE_PLAN_STEPS)):
        responses.append(
            _tool_calls(
                ToolCall(
                    id="call_1",
                    name="read_file",
                    arguments={"path": evidence_path},
                )
            )
        )
        responses.append(
            _structured(
                offline_report_payload(
                    question, repository, commit, evidence_path, index
                )
            )
        )
    return OfflineModel(*responses)


def build_offline_analyst(
    question: str, repository: str, commit: str, evidence_path: str
) -> OfflineModel:
    return OfflineModel(
        _structured(
            offline_finding_payload(question, repository, commit, evidence_path)
        )
    )


# -- composition -----------------------------------------------------------


def demo_read_target(
    repository: Optional[str],
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """Find the evidence file for the offline story.

    Returns ``(absolute_path, relative_path, bound_root)``; a
    non-``None`` root means the read-only text tools must be bound to
    it through the dispatcher seam.
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
    via the documented ``tool(arguments)`` closure seam."""

    def read(arguments):
        return read_file(arguments, repository_root=root)

    def search(arguments):
        return search_repository(arguments, repository_root=root)

    return {"read_file": read, "search_repository": search}


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


def build_agents(
    investigator_llm,
    analyst_llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
) -> Tuple[Agent, Agent]:
    """Compose the two EXISTING Agent instances with role permissions
    and the shared memory/context/trace dependencies."""
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
    return investigator, analyst


def build_orchestrator(
    investigator: Agent,
    analyst: Agent,
    trace_sink=None,
    config: Optional[OrchestrationConfig] = None,
) -> Orchestrator:
    kwargs: Dict[str, Any] = {}
    if trace_sink is not None:
        kwargs["trace_sink"] = trace_sink
    if config is not None:
        kwargs["config"] = config
    return Orchestrator(investigator, analyst, **kwargs)


# -- printing helpers ------------------------------------------------------


def _clip(text: Optional[str], limit: int = 100) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def _header(title: str, stream) -> None:
    print(RULE, file=stream)
    print(title, file=stream)
    print(RULE, file=stream)


# -- phases ----------------------------------------------------------------


def run_plan_phase(
    planner, objective: str, stream=None
) -> Dict[str, Any]:
    """Phase 1 — the M10 planner produces ONE plan and the
    deterministic plan gate decides whether execution may start."""
    stream = stream or sys.stdout
    _header("Phase 1 — plan gate (M10 planner → validate_plan)", stream)
    request = build_planning_request(objective)
    response = planner.complete(request)
    plan = plan_from_response(response)
    error = None
    try:
        validate_plan(plan)
    except PlanValidationError as exc:
        error = str(exc)
    print("objective: {}".format(_clip(plan.objective, 80)), file=stream)
    print("steps:     {}".format(len(plan.steps)), file=stream)
    for step in plan.steps:
        print(
            "  {}  {} [{}]".format(step.id, _clip(step.goal), step.status),
            file=stream,
        )
    print(
        "plan gate: {}".format(error if error else "SUCCESS"),
        file=stream,
    )
    ok = (
        error is None
        and len(plan.steps) == len(OFFLINE_PLAN_STEPS)
        and [step.id for step in plan.steps] == [
            item[0] for item in OFFLINE_PLAN_STEPS
        ]
    )
    print("plan condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok, "plan": plan, "error": error}


def run_roles_phase(plan: InvestigationPlan, stream=None) -> Dict[str, Any]:
    """Phase 2 — deterministic role assignment, every step with a
    human-readable reason (spec §15)."""
    stream = stream or sys.stdout
    _header("Phase 2 — role assignment (deterministic, observable)", stream)
    assignments = assign_roles(plan.steps)
    for item in assignments:
        print(
            "  {}  {:<12} {}".format(
                item.step_id, item.role, _clip(item.reason)
            ),
            file=stream,
        )
    roles = [item.role for item in assignments]
    ok = (
        len(assignments) == len(plan.steps)
        and roles[0] == "INVESTIGATOR"
        and roles[-1] == "ANALYST"
        and all(item.reason.strip() for item in assignments)
    )
    print(
        "roles: {} investigator / {} analyst".format(
            roles.count("INVESTIGATOR"), roles.count("ANALYST")
        ),
        file=stream,
    )
    print("roles condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok, "assignments": assignments}


def run_execution_phase(
    task: TaskPayload,
    plan: InvestigationPlan,
    orchestrator: Orchestrator,
    stream=None,
) -> Dict[str, Any]:
    """Phase 3 — the sequential step loop: assignment, agent run,
    typed output, per-step result (bounded, no parallelism)."""
    stream = stream or sys.stdout
    _header("Phase 3 — execution (sequential steps over two agents)", stream)
    run = orchestrator.execute(task, plan)
    for result in run.step_results:
        print(
            "  {}  {:<12} {:<8} attempts={} run={}".format(
                result.step_id,
                result.agent_role,
                result.status,
                result.attempts,
                (result.agent_run_id or "-")[:12],
            ),
            file=stream,
        )
    if run.blocked_steps:
        print(
            "  blocked: {}".format(", ".join(run.blocked_steps)),
            file=stream,
        )
    if run.failures:
        print("  failures: {}".format("; ".join(run.failures)), file=stream)
    print(
        "status: {} (termination {})".format(
            run.status, run.termination_reason
        ),
        file=stream,
    )
    ok = (
        run.status == COMPLETED
        and run.termination_reason == "SUCCESS"
        and [result.status for result in run.step_results]
        == ["completed"] * len(plan.steps)
        and run.failed_steps == []
        and run.blocked_steps == []
    )
    print("execution condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok, "run": run}


def run_handoff_phase(run, stream=None) -> Dict[str, Any]:
    """Phase 4 — the six typed handoffs, scope on every hop, and the
    budget counters the orchestrator recorded."""
    stream = stream or sys.stdout
    _header("Phase 4 — typed handoffs and explicit bounds", stream)
    observed = [message.message_type for message in run.messages]
    for index, message in enumerate(run.messages, start=1):
        print(
            "  {}. {:<22} {} -> {}  scope={}@{}".format(
                index,
                message.message_type,
                message.sender_agent,
                message.recipient_agent,
                message.repository_scope,
                message.commit_scope[:12],
            ),
            file=stream,
        )
    print(
        "agent runs:   {}".format(len(run.agent_runs)),
        file=stream,
    )
    print(
        "tool calls:   {}".format(run.tool_calls_total),
        file=stream,
    )
    print(
        "messages:     {} / {}".format(
            len(run.messages), OrchestrationConfig().max_messages
        ),
        file=stream,
    )
    print("retries:      {}".format(run.retries), file=stream)
    scopes_ok = all(
        message.repository_scope == run.task.repository_scope
        and message.commit_scope == run.task.commit_scope
        for message in run.messages
    )
    ok = observed == EXPECTED_HANDOFFS and scopes_ok
    print("handoff condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_validation_phase(run, stream=None) -> Dict[str, Any]:
    """Phase 5 — the M16 validator's eight deterministic rules over the
    MERGED report, the finding, and the merged evidence authority."""
    stream = stream or sys.stdout
    _header("Phase 5 — deterministic validation (the M16 verdict)", stream)
    result = run.validation
    if result is None:
        print("no validation result", file=stream)
        print("validation condition: NO", file=stream)
        return {"ok": False}
    print(
        "merged report: {} findings, {} evidence ref(s)".format(
            len(run.report.findings) if run.report else 0,
            len(run.report.evidence_refs) if run.report else 0,
        ),
        file=stream,
    )
    print(
        "report id:     {}".format(
            run.report.investigation_id if run.report else "-"
        ),
        file=stream,
    )
    print(
        "finding id:    {}".format(
            run.finding.investigation_id if run.finding else "-"
        ),
        file=stream,
    )
    for check in result.checks:
        print(
            "  [{}] {:<22} {}".format(
                "PASS" if check.ok else "FAIL",
                check.rule,
                _clip(check.detail) if not check.ok else "",
            ),
            file=stream,
        )
    print("verdict: {}".format(result.status), file=stream)
    ok = (
        result.status == "passed"
        and [check.rule for check in result.checks] == list(RULE_ORDER)
        and all(check.ok for check in result.checks)
        and run.finding is not None
        and run.report is not None
        and run.finding.investigation_id == run.report.investigation_id
    )
    print("validation condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_failure_phase(
    task: TaskPayload,
    plan_payload: Dict[str, Any],
    agent_factory: Callable[[], Tuple[Agent, Agent]],
    trace_sink=None,
    stream=None,
) -> Dict[str, Any]:
    """Phase 6 — two explicit terminal failures (spec §42/§45).

    (a) an invalid plan (a step that IS a tool call) fails the plan
        gate: NO agent runs at all;
    (b) a deliberately small message bound stops the run before the
        analysis step: the evidence is preserved, the last step is
        BLOCKED, and the terminal status is BUDGET_EXCEEDED — never a
        fabricated success.
    """
    stream = stream or sys.stdout
    _header("Phase 6 — failure semantics (explicit terminal states)", stream)

    # (a) plan gate: no agent may execute after a rejected plan
    invalid = InvestigationPlan(
        objective=task.question,
        steps=[
            PlanStep(
                id="P1", goal="read_file(path='README.md')"
            )
        ],
    )
    investigator, analyst = agent_factory()
    orchestrator = build_orchestrator(
        investigator, analyst, trace_sink=trace_sink
    )
    first = orchestrator.execute(task, invalid)
    invocations = len(investigator._llm_client.requests) + len(
        analyst._llm_client.requests
    )
    print(
        "  (a) invalid plan    -> {} / {} (agent runs {})".format(
            first.status, first.termination_reason, invocations
        ),
        file=stream,
    )
    story_a = (
        first.status == PLAN_FAILED
        and first.termination_reason == "PLAN_FAILED"
        and invocations == 0
        and first.step_results == []
    )

    # (b) message bound: analysis blocked, evidence preserved
    investigator, analyst = agent_factory()
    orchestrator = build_orchestrator(
        investigator,
        analyst,
        trace_sink=trace_sink,
        config=OrchestrationConfig(max_messages=2),
    )
    bounded_plan = InvestigationPlan.parse_obj(dict(plan_payload))
    second = orchestrator.execute(task, bounded_plan)
    print(
        "  (b) messages <= 2   -> {} / {} (blocked {})".format(
            second.status,
            second.termination_reason,
            ", ".join(second.blocked_steps) or "none",
        ),
        file=stream,
    )
    story_b = (
        second.status == BUDGET_EXCEEDED
        and second.termination_reason == "BUDGET_EXCEEDED"
        and second.blocked_steps == ["P4"]
        and second.report is not None
        and second.finding is None
        and second.validation is None
    )
    ok = story_a and story_b
    print("failure condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok, "invalid_plan_run": first, "bounded_run": second}


def run_status_phase(
    run, investigator: Agent, analyst: Agent, stream=None
) -> Dict[str, Any]:
    """Phase 7 — the sealed terminal status, the event timeline, and
    the orchestration events mirrored onto each agent's own trace."""
    stream = stream or sys.stdout
    _header("Phase 7 — status, events, and trace mirroring", stream)
    operations = run.event_operations()
    sequences = [event["sequence"] for event in run.events]
    print(
        "terminal:   {} ({})".format(run.status, run.termination_reason),
        file=stream,
    )
    print(
        "plan:       {}".format(
            ", ".join(
                "{}={}".format(step.id, step.status) for step in run.plan.steps
            )
        ),
        file=stream,
    )
    print(
        "events:     {} (first {}, last {})".format(
            len(operations), operations[0], operations[-1]
        ),
        file=stream,
    )
    for agent, label in ((investigator, "investigator"), (analyst, "analyst")):
        trace = agent.last_trace
        mirrored = (
            [event["operation"] for event in trace.orchestration_events]
            if trace is not None
            else []
        )
        print(
            "  {} trace: {} mirrored event(s){}".format(
                label,
                len(mirrored),
                "" if not mirrored else " [{}]".format(", ".join(mirrored)),
            ),
            file=stream,
        )
    from repo_intelligence.agent.trace import render_markdown

    markdown = (
        render_markdown(investigator.last_trace)
        if investigator.last_trace is not None
        else ""
    )
    section_present = "## Orchestration" in markdown
    print(
        "orchestration markdown section present: {}".format(
            "yes" if section_present else "NO"
        ),
        file=stream,
    )
    ok = (
        is_terminal(run.status)
        and sequences == sorted(sequences)
        and len(sequences) == len(set(sequences))
        and operations[0] == "orchestration.start"
        and section_present
        and bool(
            investigator.last_trace is not None
            and investigator.last_trace.orchestration_events
        )
        and bool(
            analyst.last_trace is not None
            and analyst.last_trace.orchestration_events
        )
    )
    print("status condition: {}".format("yes" if ok else "NO"), file=stream)
    return {"ok": ok}


def run_demo(
    objective: str,
    repository: str,
    commit: str,
    planner,
    agent_factory: Callable[[], Tuple[Agent, Agent]],
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
    trace_root=None,
    stream=None,
) -> Dict[str, Any]:
    """All seven phases over one shared story (spec §46/§56)."""
    stream = stream or sys.stdout
    task = TaskPayload(
        question=objective,
        repository_scope=repository,
        commit_scope=commit,
    )

    plan_phase = run_plan_phase(planner, objective, stream=stream)
    plan: InvestigationPlan = plan_phase["plan"]
    roles_phase = run_roles_phase(plan, stream=stream)

    investigator, analyst = agent_factory()
    orchestrator = build_orchestrator(
        investigator, analyst, trace_sink=trace_sink
    )
    execution = run_execution_phase(
        task, plan, orchestrator, stream=stream
    )
    run = execution["run"]

    handoff = run_handoff_phase(run, stream=stream)
    validation = run_validation_phase(run, stream=stream)
    failure = run_failure_phase(
        task, plan_phase["plan"].dict(), agent_factory, trace_sink, stream=stream
    )
    status = run_status_phase(
        run, investigator, analyst, stream=stream
    )

    phases = {
        "plan": plan_phase,
        "roles": roles_phase,
        "execution": execution,
        "handoff": handoff,
        "validation": validation,
        "failure": failure,
        "status": status,
    }
    ok = all(phase["ok"] for phase in phases.values())

    print(RULE, file=stream)
    print("M17 demonstration summary", file=stream)
    print(RULE, file=stream)
    for name in (
        "plan",
        "roles",
        "execution",
        "handoff",
        "validation",
        "failure",
        "status",
    ):
        print(
            "{:<15} {}".format(
                "{}:".format(name), "yes" if phases[name]["ok"] else "NO"
            ),
            file=stream,
        )
    print(
        "final status:   {} (terminal = {} only)".format(
            run.status, run.termination_reason
        ),
        file=stream,
    )
    print(
        "agent runs:     {} (bounded: {})".format(
            len(run.agent_runs),
            OrchestrationConfig().max_agent_runs,
        ),
        file=stream,
    )
    return {
        "run": run,
        "plan": plan,
        "investigator": investigator,
        "analyst": analyst,
        "phases": phases,
        "ok": ok,
    }


# -- main -------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m17_orchestration_demo.py",
        description=(
            "M17 end-to-end demonstration: one plan executed by "
            "deterministically assigned agents under explicit bounds, "
            "one sealed terminal state — offline and deterministic by "
            "default."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use a real model for the planner and both agents "
        "(requires OPENAI_API_KEY); default is fully offline",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the objective/question (default: {})".format(
            _clip(DEFAULT_QUESTION, 60)
        ),
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
        "pinned snapshot",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="memory database path (default: runtime/agent_memory/"
        "m17_demo.db — the demonstration's OWN database)",
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

    objective = args.question or DEFAULT_QUESTION

    if args.live:
        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "M17 demonstration (--live) requires OPENAI_API_KEY.\n"
                "Copy .env.example to .env, set OPENAI_API_KEY (never "
                "commit .env), and re-run — or run the default offline "
                "demonstration:\n"
                "    python agentrun_m17_orchestration_demo.py",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        mode_label = "live (real model for planner and both agents)"
    else:
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

    # -- composition (live clients or scripted doubles) --------------------
    live_clients: Optional[Tuple[Any, Any, Any]] = None
    if args.live:
        try:
            planner = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
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
        live_clients = (planner, investigator_llm, analyst_llm)
    else:
        planner = build_offline_planner(
            objective, repository_label, OFFLINE_COMMIT
        )
        investigator_llm = build_offline_investigator(
            objective, repository_label, OFFLINE_COMMIT, evidence_path
        )
        analyst_llm = build_offline_analyst(
            objective, repository_label, OFFLINE_COMMIT, evidence_path
        )

    try:
        manager = build_memory_manager(args.db)
    except Exception as exc:
        print("Memory store failed: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED
    memory_context = build_memory_context(repository_label, OFFLINE_COMMIT)
    engine = build_engine(repository_label, args.snapshots)
    trace_sink = (
        JsonFileTraceSink(args.traces) if args.traces else JsonFileTraceSink()
    )
    trace_root = (
        Path(args.traces)
        if args.traces
        else Path("runtime") / "agent_runs"
    )
    bindings = fixture_bindings(Path(bound_root)) if bound_root else None

    def agent_factory() -> Tuple[Agent, Agent]:
        if live_clients is not None:
            _, investigator_llm, analyst_llm = live_clients
        else:
            investigator_llm = build_offline_investigator(
                objective, repository_label, OFFLINE_COMMIT, evidence_path
            )
            analyst_llm = build_offline_analyst(
                objective, repository_label, OFFLINE_COMMIT, evidence_path
            )
        return build_agents(
            investigator_llm,
            analyst_llm,
            trace_sink,
            manager,
            memory_context,
            engine,
            bindings=bindings,
        )

    print("M17 multi-agent orchestration demonstration — {}".format(mode_label))
    print(
        "scope: {}@{}   target: {}".format(
            repository_label, OFFLINE_COMMIT[:12], evidence_path
        ),
    )
    print(file=sys.stdout)

    try:
        outcome = run_demo(
            objective,
            repository_label,
            OFFLINE_COMMIT,
            planner,
            agent_factory,
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
        PlanValidationError,
    ) as exc:
        print("Demonstration execution failed.", file=sys.stderr)
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    if not outcome["ok"]:
        failed = [
            name
            for name, phase in outcome["phases"].items()
            if not phase["ok"]
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
