"""M19 — contract-driven agent coordinator demonstration.

    python agentrun_m19_orchestrator_agents_demo.py
    python agentrun_m19_orchestrator_agents_demo.py --repository \
        tests/fixtures/construction

The M19 demonstration: the orchestrator is a COORDINATOR over an
explicit task boundary — every step becomes an immutable
``AgentTaskRequest`` dispatched through the ``AgentExecutor`` port,
every execution comes back as a typed ``AgentTaskResult`` classified
by ONE failure taxonomy, and only the coordinator commits
``TaskRecord`` state (readiness from the pure scheduler, assembly
through ``RunAssembler``, the M16 verdict as the only validator).

Story:
    1  contract      one AgentTaskRequest — summary only, the prompt
                     text is never printed; the chain dependencies
                     from ``sequential_dependencies``
    2  dispatch      the executor boundary runs the REAL scripted
                     investigator once → typed output, trace id,
                     measured usage (never fabricated)
    3  coordinated   one full Orchestrator.execute: one TaskRecord
                     per step, pending → ready → running →
                     completed, results committed from results
    4  readiness     the pure scheduler: the ready set after each
                     completion, plan order preserved
    5  failure       a broken investigator → TaskFailure category
                     ``structured_output`` (retryable, retried,
                     then committed FAILED) and successors BLOCKED
                     with their blocking reason
    6  recovery      the SAME transient failure retried under the
                     same bound → the second attempt commits
                     (attempts 2, retries 1, cumulative usage)
    7  assembly      merged report + finding + the eight-rule M16
                     verdict over the merged evidence package

Modes: fully offline and deterministic by default (scripted models,
the REAL tool dispatcher reading a real file from the pinned
snapshot — or ``--repository <root>`` through the dispatcher seam).
There is no ``--live`` flag here: live runs go through
``agentrun_repository_query.py --live``.

Exit codes:
    0  every demonstration condition held
    1  a demonstration condition failed (contract, dispatch,
        records, readiness, failure, recovery, assembly, verdict)
    6  configuration/execution failure (unreadable evidence file,
        scripted model over-invoked, participant raised, ...)

The offline test suite (tests/orchestration/test_m19_demo.py) drives
the phase functions and main() with scripted doubles only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent import Agent  # noqa: E402
from repo_intelligence.llm import (  # noqa: E402
    LLMResponse,
    Message,
    ToolCall,
)
from repo_intelligence.multiagent import (  # noqa: E402
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (  # noqa: E402
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import (  # noqa: E402
    ANALYST,
    COMPLETED,
    INCOMPLETE,
    INVESTIGATOR,
    TASK_COMPLETED,
    TASK_BLOCKED,
    TASK_FAILED,
    AgentTaskExecutor,
    AgentTaskRequest,
    Orchestrator,
    TaskRecord,
    ready_tasks,
    sequential_dependencies,
)
from repo_intelligence.planning.models import (  # noqa: E402
    InvestigationPlan,
    PlanStep,
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

EXIT_SUCCESS = 0
EXIT_DEMO_FAILED = 1
EXIT_EXECUTION_FAILED = 6

REPOSITORY = "ADOP"
OFFLINE_COMMIT = "c" * 40
OFFLINE_INVESTIGATION_ID = "inv-m19-demo-001"
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

# The offline plan: evidence before interpretation (three steps).
OFFLINE_PLAN_STEPS = (
    ("P1", "Identify the application entry points"),
    ("P2", "Investigate the tool dispatcher and role permissions"),
    ("P3", "Explain how the boundary is enforced across the workflow"),
)


# -- scripted model --------------------------------------------------------


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
    question: str,
    evidence_path: str,
    index: int,
) -> Dict[str, Any]:
    return {
        "investigation_id": "inv-m19-demo-{:03d}".format(index),
        "question": question,
        "repository_scope": REPOSITORY,
        "commit_scope": OFFLINE_COMMIT,
        "findings": [
            {
                "statement": (
                    "step {} inspected {} (offline scripted "
                    "finding).".format(
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
    question: str, evidence_path: str
) -> Dict[str, Any]:
    return {
        "investigation_id": OFFLINE_INVESTIGATION_ID,
        "question": question,
        "repository_scope": REPOSITORY,
        "commit_scope": OFFLINE_COMMIT,
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


def read_then_report(
    question: str, evidence_path: str, index: int
) -> List[LLMResponse]:
    return [
        _tool_calls(
            ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": evidence_path},
            )
        ),
        _structured(offline_report_payload(question, evidence_path, index)),
    ]


def build_offline_investigator(
    question: str, evidence_path: str, rounds: int = 2
) -> OfflineModel:
    responses: List[LLMResponse] = []
    for index in range(1, rounds + 1):
        responses.extend(read_then_report(question, evidence_path, index))
    return OfflineModel(*responses)


def build_offline_analyst(
    question: str, evidence_path: str
) -> OfflineModel:
    return OfflineModel(
        _structured(offline_finding_payload(question, evidence_path))
    )


# -- composition -----------------------------------------------------------


def demo_read_target(
    repository: Optional[str],
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """Find the evidence file: ``(absolute_path, relative, bound_root)``;
    a non-``None`` root means the read-only tools are bound to it."""
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


def build_agents(
    investigator_llm,
    analyst_llm,
    bindings: Optional[Dict[str, Any]] = None,
) -> Tuple[Agent, Agent]:
    """The two EXISTING agents with role permissions — no memory or
    context wiring here: this demonstration isolates the M19 task
    boundary (M14/M15 are exercised by the M17 demonstration)."""
    investigator = Agent(
        investigator_llm,
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=bindings
        ),
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    analyst = Agent(
        analyst_llm,
        tool_dispatcher=restricted_dispatcher("analyst"),
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
    )
    return investigator, analyst


def build_plan(objective: str) -> InvestigationPlan:
    return InvestigationPlan(
        objective=objective,
        steps=[
            PlanStep(id=step_id, goal=goal)
            for step_id, goal in OFFLINE_PLAN_STEPS
        ],
    )


def build_task(objective: str) -> TaskPayload:
    return TaskPayload(
        question=objective,
        repository_scope=REPOSITORY,
        commit_scope=OFFLINE_COMMIT,
    )


def _broken_investigator_factory(
    objective: str, evidence_path: str, bindings: Optional[Dict[str, Any]]
) -> Tuple[Agent, Agent]:
    """A payload the structured-output contract REJECTS (the failure
    story: classified, retried, committed FAILED)."""
    broken = offline_report_payload(objective, evidence_path, 1)
    del broken["findings"]
    return build_agents(
        OfflineModel(_structured(broken), _structured(broken)),
        build_offline_analyst(objective, evidence_path),
        bindings=bindings,
    )


def _recovering_investigator_factory(
    objective: str, evidence_path: str, bindings: Optional[Dict[str, Any]]
) -> Tuple[Agent, Agent]:
    """The SAME rejected payload first, then healthy rounds: the
    bounded retry RECOVERS (attempt 1 fails, attempt 2 commits, P2
    runs normally afterwards)."""
    broken = offline_report_payload(objective, evidence_path, 1)
    del broken["findings"]
    return build_agents(
        OfflineModel(
            _structured(broken),
            *read_then_report(objective, evidence_path, 1),
            *read_then_report(objective, evidence_path, 2),
        ),
        build_offline_analyst(objective, evidence_path),
        bindings=bindings,
    )


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


def _condition(ok: bool, stream) -> None:
    print("condition: {}".format("yes" if ok else "NO"), file=stream)


# -- phases ----------------------------------------------------------------


def run_contract_phase(task: TaskPayload, stream=None) -> Dict[str, Any]:
    """Phase 1 — the immutable task envelope: summary-only output,
    chain dependencies from the ONE place order becomes dependency."""
    stream = stream or sys.stdout
    _header("Phase 1 — task contract (request summary, dependencies)", stream)
    request = AgentTaskRequest(
        run_id="demo-run-contract",
        task_id="P1",
        attempt=1,
        role=INVESTIGATOR,
        assignment_reason="first evidence step, chain head",
        question=task.question,
        goal=OFFLINE_PLAN_STEPS[0][1],
        prompt=(
            "TASK-SCOPED PROMPT (never printed) answer: {} | "
            "repository: {} | commit: {}".format(
                task.question, task.repository_scope, task.commit_scope
            )
        ),
        repository_scope=task.repository_scope,
        commit_scope=task.commit_scope,
        depends_on=(),
    )
    summary = request.summary()
    for key in sorted(summary):
        print("  {:<14} {}".format(key, summary[key]), file=stream)
    blob = repr(summary)
    print(
        "prompt withheld: {}".format(
            "yes" if "TASK-SCOPED PROMPT" not in blob else "NO"
        ),
        file=stream,
    )
    chain = sequential_dependencies(
        [step_id for step_id, _ in OFFLINE_PLAN_STEPS]
    )
    print("chain: {}".format(chain), file=stream)
    ok = (
        "prompt" not in summary
        and "question" not in summary
        and "TASK-SCOPED PROMPT" not in blob
        and chain == {"P1": [], "P2": ["P1"], "P3": ["P2"]}
        and summary["role"] == INVESTIGATOR
    )
    _condition(ok, stream)
    return {"ok": ok}


def run_dispatch_phase(
    task: TaskPayload,
    evidence_path: str,
    bindings: Optional[Dict[str, Any]],
    stream=None,
) -> Dict[str, Any]:
    """Phase 2 — ONE dispatch through the executor boundary: typed
    output, this-call trace, measured usage."""
    stream = stream or sys.stdout
    evidence_path = str(evidence_path)
    _header("Phase 2 — executor dispatch (typed result, usage)", stream)
    investigator, analyst = build_agents(
        build_offline_investigator(task.question, evidence_path, rounds=1),
        build_offline_analyst(task.question, evidence_path),
        bindings=bindings,
    )
    executor = AgentTaskExecutor.for_agents(investigator, analyst)
    request = AgentTaskRequest(
        run_id="demo-run-dispatch",
        task_id="P1",
        attempt=1,
        role=INVESTIGATOR,
        assignment_reason="dispatch demonstration",
        question=task.question,
        goal=OFFLINE_PLAN_STEPS[0][1],
        prompt="Dispatch once for P1: {}".format(task.question),
        repository_scope=task.repository_scope,
        commit_scope=task.commit_scope,
    )
    result = executor.execute(request)
    print("  status        {}".format(result.status), file=stream)
    print(
        "  output        {}".format(type(result.output).__name__),
        file=stream,
    )
    print(
        "  typed         {}".format(
            "yes" if isinstance(result.output, InvestigationReport)
            else "NO"
        ),
        file=stream,
    )
    print(
        "  trace id      {}".format(
            "present" if result.agent_run_id else "missing"
        ),
        file=stream,
    )
    print(
        "  usage         {} tool call(s), {} model iteration(s)".format(
            result.usage.tool_calls, result.usage.model_iterations
        ),
        file=stream,
    )
    print(
        "  tokens        {}".format(
            "not reported (stays None)"
            if result.usage.total_tokens is None
            else str(result.usage.total_tokens)
        ),
        file=stream,
    )
    ok = (
        result.status == TASK_COMPLETED
        and isinstance(result.output, InvestigationReport)
        and result.agent_run_id is not None
        and result.trace is not None
        and result.usage.tool_calls == 1
        and result.usage.model_iterations == 2
        and result.usage.total_tokens is None
    )
    _condition(ok, stream)
    return {"ok": ok, "result": result}


def _print_records(run, stream) -> None:
    for record in run.tasks:
        print(
            "  {:<3} {:<12} {:<10} deps={:<8} attempts={} "
            "tools={} category={}".format(
                record.task_id,
                record.role,
                record.status,
                str(record.depends_on),
                record.attempts,
                record.usage.tool_calls if record.usage else "-",
                record.failure_category or "-",
            ),
            file=stream,
        )


def run_coordinated_phase(
    task: TaskPayload,
    objective: str,
    evidence_path: str,
    bindings: Optional[Dict[str, Any]],
    stream=None,
) -> Dict[str, Any]:
    """Phase 3 — one full orchestrated run: a TaskRecord per step,
    committed ONLY by the coordinator from the results."""
    stream = stream or sys.stdout
    evidence_path = str(evidence_path)
    _header("Phase 3 — coordinated run (task records committed)", stream)
    plan = build_plan(objective)
    error = None
    try:
        validate_plan(plan)
    except PlanValidationError as exc:
        error = str(exc)
    print("plan gate: {}".format(error if error else "SUCCESS"), file=stream)
    investigator, analyst = build_agents(
        build_offline_investigator(objective, evidence_path, rounds=2),
        build_offline_analyst(objective, evidence_path),
        bindings=bindings,
    )
    orchestrator = Orchestrator(investigator, analyst)
    run = orchestrator.execute(task, plan)
    _print_records(run, stream)
    print(
        "records:       {} (one per plan step)".format(len(run.tasks)),
        file=stream,
    )
    print(
        "terminal:      {} / {}".format(
            run.status, run.termination_reason
        ),
        file=stream,
    )
    checks = run.validation.checks if run.validation is not None else []
    print(
        "verdict:       {} ({}/8 rules)".format(
            run.validation_status, sum(1 for c in checks if c.ok)
        ),
        file=stream,
    )
    outputs_typed = all(
        isinstance(record.output, (InvestigationReport, ArchitectureFinding))
        for record in run.tasks
    )
    ok = (
        error is None
        and run.status == COMPLETED
        and run.termination_reason == "SUCCESS"
        and len(run.tasks) == len(OFFLINE_PLAN_STEPS)
        and [record.status for record in run.tasks] == [TASK_COMPLETED] * 3
        and [record.depends_on for record in run.tasks]
        == [[], ["P1"], ["P2"]]
        and outputs_typed
        and all(record.usage is not None for record in run.tasks)
        and run.validation_status == "passed"
        and len(checks) == 8
        and all(check.ok for check in checks)
    )
    _condition(ok, stream)
    return {
        "ok": ok,
        "run": run,
        "plan": plan,
        "investigator": investigator,
        "analyst": analyst,
    }


def run_readiness_phase(stream=None) -> Dict[str, Any]:
    """Phase 4 — the pure scheduler: ready sets as completions land,
    plan order preserved, terminal tasks never re-offered."""
    stream = stream or sys.stdout
    _header("Phase 4 — readiness (pure scheduler over task records)", stream)
    step_ids = [step_id for step_id, _ in OFFLINE_PLAN_STEPS]
    dependencies = sequential_dependencies(step_ids)
    records = [
        TaskRecord(
            task_id=step_id,
            role=INVESTIGATOR if index < 2 else ANALYST,
            depends_on=list(dependencies[step_id]),
        )
        for index, step_id in enumerate(step_ids)
    ]
    stages = []
    print(
        "  start         {}".format(
            [record.task_id for record in ready_tasks(records)]
        ),
        file=stream,
    )
    stages.append([record.task_id for record in ready_tasks(records)])
    records[0].status = TASK_COMPLETED
    print(
        "  P1 completed  {}".format(
            [record.task_id for record in ready_tasks(records)]
        ),
        file=stream,
    )
    stages.append([record.task_id for record in ready_tasks(records)])
    records[1].status = TASK_COMPLETED
    print(
        "  P2 completed  {}".format(
            [record.task_id for record in ready_tasks(records)]
        ),
        file=stream,
    )
    stages.append([record.task_id for record in ready_tasks(records)])
    records[2].status = TASK_COMPLETED
    print(
        "  P3 completed  {}".format(
            [record.task_id for record in ready_tasks(records)]
        ),
        file=stream,
    )
    stages.append([record.task_id for record in ready_tasks(records)])
    ok = stages == [["P1"], ["P2"], ["P3"], []]
    _condition(ok, stream)
    return {"ok": ok}


def run_failure_phase(
    task: TaskPayload,
    objective: str,
    evidence_path: str,
    bindings: Optional[Dict[str, Any]],
    stream=None,
) -> Dict[str, Any]:
    """Phase 5 — a rejected structured payload: classified ONCE,
    retried to the bound, committed FAILED, successors BLOCKED."""
    stream = stream or sys.stdout
    evidence_path = str(evidence_path)
    _header("Phase 5 — failure (one taxonomy, blocked successors)", stream)
    plan = build_plan(objective)
    investigator, analyst = _broken_investigator_factory(
        objective, evidence_path, bindings
    )
    orchestrator = Orchestrator(investigator, analyst)
    run = orchestrator.execute(task, plan)
    _print_records(run, stream)
    print(
        "terminal:      {} / {}".format(
            run.status, run.termination_reason
        ),
        file=stream,
    )
    print(
        "retries:       {}".format(run.retries), file=stream
    )
    print(
        "analyst runs:  {}".format(len(analyst._llm_client.requests)),
        file=stream,
    )
    p1, p2, p3 = run.tasks
    ok = (
        run.status == INCOMPLETE
        and run.termination_reason == "AGENT_FAILED"
        and p1.status == TASK_FAILED
        and p1.failure_category == "structured_output"
        and p1.error.startswith("StructuredOutputValidationError")
        and p2.status == TASK_BLOCKED
        and p2.error == "P1_FAILED"
        and p3.status == TASK_BLOCKED
        and p3.error == "P1_FAILED"
        and run.retries == 1
        and analyst._llm_client.requests == []
    )
    _condition(ok, stream)
    return {"ok": ok, "run": run}


def run_recovery_phase(
    task: TaskPayload,
    objective: str,
    evidence_path: str,
    bindings: Optional[Dict[str, Any]],
    stream=None,
) -> Dict[str, Any]:
    """Phase 6 — the SAME transient failure, retried under the SAME
    bound: attempt 2 commits (recovery, not just retry exhaustion)."""
    stream = stream or sys.stdout
    evidence_path = str(evidence_path)
    _header("Phase 6 — recovery (bounded retry then committed)", stream)
    plan = build_plan(objective)
    investigator, analyst = _recovering_investigator_factory(
        objective, evidence_path, bindings
    )
    orchestrator = Orchestrator(investigator, analyst)
    run = orchestrator.execute(task, plan)
    _print_records(run, stream)
    print(
        "terminal:      {} / {}".format(
            run.status, run.termination_reason
        ),
        file=stream,
    )
    print("retries:       {}".format(run.retries), file=stream)
    p1, p2, p3 = run.tasks
    usage = p1.usage
    usage_total = p1.usage_total
    print(
        "p1 attempts:   {} (recent {} iteration(s), "
        "cumulative {} iteration(s))".format(
            p1.attempts,
            usage.model_iterations if usage else None,
            usage_total.model_iterations if usage_total else None,
        ),
        file=stream,
    )
    ok = (
        run.status == COMPLETED
        and run.termination_reason == "SUCCESS"
        and run.retries == 1
        and p1.attempts == 2
        and p1.status == TASK_COMPLETED
        and p1.failure_category is None
        and p1.output is not None
        and usage is not None
        and usage.model_iterations == 2
        and usage_total is not None
        and usage_total.model_iterations == 3
        and p2.status == TASK_COMPLETED
        and p3.status == TASK_COMPLETED
    )
    _condition(ok, stream)
    return {"ok": ok, "run": run}


def run_assembly_phase(run, stream=None) -> Dict[str, Any]:
    """Phase 7 — explicit assembly: merged evidence package, the
    finding paired with it, and the M16 verdict as the only judge."""
    stream = stream or sys.stdout
    _header("Phase 7 — assembly (merged package + M16 verdict)", stream)
    report = run.report
    finding = run.finding
    print(
        "  report        {}".format(
            report.investigation_id if report else "missing"
        ),
        file=stream,
    )
    print(
        "  findings      {}".format(
            len(report.findings) if report else 0
        ),
        file=stream,
    )
    print(
        "  evidence refs {} (de-duplicated across steps)".format(
            len(report.evidence_refs) if report else 0
        ),
        file=stream,
    )
    print(
        "  finding       {}".format(
            _clip(finding.summary, 60) if finding else "missing"
        ),
        file=stream,
    )
    print(
        "  ids match     {}".format(
            "yes"
            if report and finding
            and report.investigation_id == finding.investigation_id
            else "NO"
        ),
        file=stream,
    )
    verdict = run.validation
    print(
        "  verdict       {}".format(
            run.validation_status if verdict else "missing"
        ),
        file=stream,
    )
    ok = (
        report is not None
        and finding is not None
        and report.investigation_id == OFFLINE_INVESTIGATION_ID
        and report.status == "complete"
        and len(report.findings) == 2
        and len(report.evidence_refs) == 1
        and report.investigation_id == finding.investigation_id
        and verdict is not None
        and verdict.status == "passed"
        and len(verdict.checks) == 8
        and all(check.ok for check in verdict.checks)
    )
    _condition(ok, stream)
    return {"ok": ok}


# -- demonstration ---------------------------------------------------------


def run_demo(
    task: TaskPayload,
    objective: str,
    evidence_path: str,
    bindings: Optional[Dict[str, Any]],
    stream=None,
) -> Dict[str, Any]:
    """All seven phases over one shared story."""
    stream = stream or sys.stdout
    evidence_path = str(evidence_path)
    contract = run_contract_phase(task, stream=stream)
    dispatch = run_dispatch_phase(
        task, evidence_path, bindings, stream=stream
    )
    coordinated = run_coordinated_phase(
        task, objective, evidence_path, bindings, stream=stream
    )
    readiness = run_readiness_phase(stream=stream)
    failure = run_failure_phase(
        task, objective, evidence_path, bindings, stream=stream
    )
    recovery = run_recovery_phase(
        task, objective, evidence_path, bindings, stream=stream
    )
    assembly = run_assembly_phase(
        coordinated["run"], stream=stream
    )
    phases = {
        "contract": contract,
        "dispatch": dispatch,
        "coordinated": coordinated,
        "readiness": readiness,
        "failure": failure,
        "recovery": recovery,
        "assembly": assembly,
    }
    ok = all(phase["ok"] for phase in phases.values())

    print(RULE, file=stream)
    print("M19 demonstration summary", file=stream)
    print(RULE, file=stream)
    for name in phases:
        print(
            "{:<16} {}".format(
                "{}:".format(name),
                "yes" if phases[name]["ok"] else "NO",
            ),
            file=stream,
        )
    run = coordinated["run"]
    print(
        "final status:  {} (terminal = {} only)".format(
            run.status, run.termination_reason
        ),
        file=stream,
    )
    print(
        "task records:  {} committed by the coordinator".format(
            len(run.tasks)
        ),
        file=stream,
    )
    return {
        "ok": ok,
        "phases": phases,
        "run": run,
        "plan": coordinated["plan"],
    }


# -- main -------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m19_orchestrator_agents_demo.py",
        description=(
            "M19 demonstration: the coordinator dispatches typed task "
            "contracts through the executor boundary — offline and "
            "deterministic by default."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the objective/question (default: {})".format(
            _clip(DEFAULT_QUESTION, 60)
        ),
    )
    parser.add_argument(
        "--repository",
        default=None,
        help="foreign repository root: binds read_file/search_repository "
        "to it (e.g. tests/fixtures/construction); default reads the "
        "pinned snapshot",
    )
    args = parser.parse_args(argv)
    objective = args.question or DEFAULT_QUESTION

    read_target, evidence_path, bound_root = demo_read_target(
        args.repository
    )
    if read_target is None:
        print(
            "M19 demonstration: no readable evidence file (looked for "
            "{}{}).".format(
                ", ".join(DEFAULT_READ_CANDIDATES)
                if args.repository is None
                else ", ".join(FIXTURE_READ_CANDIDATES),
                " in {!r}".format(args.repository)
                if args.repository is not None
                else "",
            ),
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED
    bindings = (
        fixture_bindings(Path(bound_root)) if bound_root else None
    )
    print("evidence file: {}".format(evidence_path), file=sys.stdout)

    task = build_task(objective)
    try:
        outcome = run_demo(
            task, objective, evidence_path, bindings, stream=sys.stdout
        )
    except AssertionError as exc:
        print(
            "M19 demonstration execution failed: {}".format(exc),
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED
    except Exception as exc:  # participant/planner raised — config 6
        print(
            "M19 demonstration execution failed: {}: {}".format(
                type(exc).__name__, exc
            ),
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED
    return EXIT_SUCCESS if outcome["ok"] else EXIT_DEMO_FAILED


if __name__ == "__main__":
    sys.exit(main())
