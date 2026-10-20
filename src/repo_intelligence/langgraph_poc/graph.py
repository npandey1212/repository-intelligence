"""The middleware-protected LangGraph ``create_agent`` POC.

Outer workflow (a real ``StateGraph`` â€” explicit nodes, explicit
edges, shared typed state):

```text
START -> planner -> query_router -> orchestrator_select
      -> investigator -> orchestrator_receive -> final_response -> END
```

with conditional failure routing after planner / query_router /
orchestrator_select / investigator / orchestrator_receive: any failure
jumps to ``final_response``, which reports an EXPLICIT ``FAILED``
outcome â€” a safety failure can never be mistaken for a successful
investigation.

Node responsibilities:

* ``planner`` â€” the EXISTING planning implementation
  (``planning.planner.build_planning_request`` +
  ``plan_from_response`` + ``planning.validation.validate_plan``)
  through an injected provider-neutral ``LLMClient``. This POC then
  requires the four-step shape the POC promises: ids P1-P4, each a
  meaningful, question-derived goal (the planner model authors the
  goals; the application only validates).
* ``query_router`` â€” the EXISTING ``orchestration.roles.assign_roles``
  contract with explicit overrides: P1-P3 -> INVESTIGATOR, P4 -> ANALYST.
* ``analyst`` (when injected) â€” a tool-free ``create_agent`` consumes all
  three validated reports and returns a typed ``ArchitectureFinding``.
* ``deterministic_validation`` â€” validates scope, source-report IDs,
  report completeness, and evidence provenance without an LLM.
* ``orchestrator_select`` â€” selects the next unfinished Investigator step.
  In multi-step mode it dispatches P1-P3 in plan order. If an Analyst is
  injected, P4 runs only after all three reports complete successfully.
* ``investigator`` â€” runs the ``create_agent``-constructed agent
  (see :func:`build_create_agent_investigator`) on the SELECTED TASK
  only, with the three ``react_safety``-backed middleware. The
  structured ``InvestigationReport`` is read from the agent state's
  ``structured_response`` (the create_agent structured-output path)
  and type-checked; anything else is an explicit failure.
* ``orchestrator_receive`` â€” re-validates each report against the
  EXISTING ``InvestigationReport`` contract and records that step completed
  only on valid success.
* ``final_response`` â€” the simplest result the graph state supports
  (no new response model): ``{"status", "failure", "error",
  "report", "completed_step_ids"}``.

Two independent limits, deliberately separate:

* the GRAPH recursion limit (``config={"recursion_limit": ...}`` at
  ``graph.invoke``) bounds how many OUTER graph steps may run;
* the Investigator's middleware (``ReActLoopGuard``) bounds the
  internal model/tool loop, and the request budget bounds each model
  request. Neither replaces the other: a generous graph recursion
  limit cannot bypass the agent's own ceilings, and the agent's
  ceilings cannot keep a malformed graph looping forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
)

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - optional environment
    raise ImportError(
        "The LangGraph create_agent POC requires the 'langgraph' and "
        "'langchain' packages. Install the project's optional graph "
        "dependencies before importing this module."
    ) from exc

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ..agent.request_budget import (
    RequestBudgetExceededError,
    RequestBudgetPolicy,
)
from ..agents.architecture_investigator import ARCHITECTURE_TOOLS
from ..llm.client import LLMClient
from ..multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
    ValidationCheck,
    ValidationResult,
)
from ..orchestration.roles import (
    AGENT_ROLES,
    ANALYST,
    INVESTIGATOR,
    RoleAssignment,
    assign_roles,
)
from ..planning.models import InvestigationPlan
from ..planning.planner import build_planning_request, plan_from_response
from ..planning.validation import validate_plan
from ..react_safety import ReActSafetyConfig
from .middleware import (
    ExecutionLimitMiddleware,
    ObservationBoundingMiddleware,
    ReActExecutionLimitExceeded,
    RequestBudgetMiddleware,
)
from .tools import investigator_tool_definitions, to_langchain_tool

__all__ = [
    "AGENT_RECURSION_LIMIT",
    "CREATE_AGENT_INVESTIGATION_PROMPT_V1",
    "CREATE_AGENT_ANALYST_PROMPT_V1",
    "GRAPH_RECURSION_LIMIT",
    "LangGraphPocState",
    "POC_PLAN_STEP_IDS",
    "POC_ROLE_OVERRIDES",
    "PocInvestigator",
    "PocAnalyst",
    "build_create_agent_analyst",
    "build_create_agent_investigator",
    "build_poc_graph",
    "run_poc",
    "validate_poc_synthesis",
]

# This create_agent POC returns the existing multi-agent handoff contract,
# not the separate ArchitectureReport used by the architecture investigator.
# Keeping the prompt aligned with response_format avoids asking the model to
# produce a schema that the receiving boundary will reject.
CREATE_AGENT_INVESTIGATION_PROMPT_V1 = """\
ROLE
You are a read-only repository investigator. Collect evidence for the assigned
task and return exactly one structured InvestigationReport matching the
provided schema. Do not return an ArchitectureReport.

SCOPE AND EVIDENCE
- Investigate only the repository and commit named in the TASK block.
- Use the provided tools; never modify files or infer undocumented behavior.
- Search is for discovery only. Before citing a path or line range, read that
  source range with read_file.
- Prefer the smallest set of tool calls that answers the assigned step.
- Do not inspect architecture context, graph data, or unrelated files unless
  the assigned task specifically requires them.
- Once evidence is sufficient, stop calling tools and produce the report.
- If a detail cannot be established from the evidence, state that as an
  uncertainty instead of continuing open-ended exploration.

INVESTIGATIONREPORT CONTRACT
- Echo question, repository_scope, and commit_scope exactly from the TASK.
- Provide a non-empty investigation_id.
- findings is a list of evidence-backed statements. Each finding contains
  statement, evidence_refs, support, and optional uncertainty. support must be
  one of SUPPORTED, UNSUPPORTED, or UNKNOWN.
- Every evidence reference must point to a source actually read during this run.
- evidence_refs must be the exact union of every finding's evidence_refs.
  Match each reference by the full tuple (path, line_start, line_end); do not
  omit a finding-level reference or change its line bounds in the aggregate.
- investigated_sources lists the source paths actually read.
- Use status "complete" only when the assigned task is answered from evidence;
  otherwise use "incomplete".
- Include open_questions and uncertainties when appropriate.
"""

# The POC's plan shape: exactly these four step ids. The planner model
# authors the GOALS (meaningful, question-derived); the application
# validates the shape â€” never fabricates a plan.
POC_PLAN_STEP_IDS = ("P1", "P2", "P3", "P4")

# Router overrides for this POC: evidence steps to the Investigator,
# the final step to the Analyst (metadata only â€” no Analyst runs).
POC_ROLE_OVERRIDES: Dict[str, str] = {
    "P1": INVESTIGATOR,
    "P2": INVESTIGATOR,
    "P3": INVESTIGATOR,
    "P4": ANALYST,
}

# Outer-graph step budget (planner + router + two orchestrator phases
# + investigator + receive + final + conditional bookkeeping). This
# bounds GRAPH execution only â€” it is not a token control and not a
# substitute for the Investigator's middleware limits.
GRAPH_RECURSION_LIMIT = 25

# The Investigator agent's OWN LangGraph step budget. LangGraph counts
# graph-node transitions (model and tool nodes), not just model iterations;
# a value of 12 can therefore stop a normal ReAct loop before the project's
# middleware guard reaches its configured model/tool ceiling and raises the
# intended typed ReActExecutionLimitExceeded. Keep this framework-level
# failsafe above the middleware iteration budget so the middleware remains
# the primary, inspectable stop condition.
# LangGraph counts model/tool node transitions, not just ReAct iterations.
# With a 12-tool-call middleware budget, 20 graph steps can terminate the
# agent after roughly 10 model/tool cyclesâ€”before middleware can trigger its
# bounded finalization turn. Keep this failsafe above that path; the middleware
# limits remain the primary, inspectable control.
AGENT_RECURSION_LIMIT = 32

ROUTING_CONTINUE = "continue"
ROUTING_SUCCESS = "success"
ROUTING_FAIL = "fail"


class LangGraphPocState(TypedDict, total=False):
    """The POC graph state â€” existing models only, no duplicates.

    Focused: the question, the validated plan, the assignments, the
    selected step/task, the validated report, the completed step ids,
    the routing/failure bookkeeping, and the final response. No
    repository contents, no unbounded tool output, no conversation
    history.
    """

    question: str
    repository_scope: str
    commit_scope: str
    plan: InvestigationPlan
    role_assignments: List[RoleAssignment]
    selected_step_id: str
    selected_step_goal: str
    selected_task: TaskPayload
    investigator_report: InvestigationReport
    investigator_reports: List[InvestigationReport]
    completed_step_ids: List[str]
    multi_step: bool
    routing: str
    failure: str
    error: str
    execution_snapshot: Dict[str, Any]
    repair_attempts_by_step: Dict[str, int]
    architecture_finding: ArchitectureFinding
    validation_result: ValidationResult
    final: Dict[str, Any]


@dataclass
class PocInvestigator:
    """The create_agent-constructed Investigator plus its middleware
    handles (inspection seam for tests and callers â€” records,
    guard snapshots, and the compiled agent itself)."""

    agent: Any
    budget: RequestBudgetMiddleware
    execution: ExecutionLimitMiddleware
    observation: ObservationBoundingMiddleware
    tool_names: List[str] = field(default_factory=list)
    # Optional because callers/tests may construct the wrapper without repair.
    # build_create_agent_investigator supplies the bounded, tool-free repair agent.
    repair_agent: Any = None


@dataclass
class PocAnalyst:
    """Tool-free create_agent wrapper for P4 synthesis."""

    agent: Any


def _is_architecture_finding_validation_error(exc: Exception) -> bool:
    """Return true only for the bounded P4 structured-output repair case."""
    message = "{}: {}".format(type(exc).__name__, exc).lower()
    return (
        "structuredoutputvalidationerror" in message
        or (
            "failed to parse structured output" in message
            and "architecturefinding" in message
        )
    )


CREATE_AGENT_ANALYST_PROMPT_V1 = """\\
ROLE
You are the architecture synthesis Analyst. You do not investigate the
repository and you have no tools. Use only the supplied P1-P3
InvestigationReport objects. Do not invent source paths, line ranges, tools,
protocols, or behavior.

TASK
Synthesize the overall interaction model across agent roles, tool usage,
and MCP communication. Separate direct findings from inferences, preserve
open questions, and avoid claims not supported by the supplied reports.

OUTPUT CONTRACT — REQUIRED FIELDS
Return exactly one ArchitectureFinding, not an InvestigationReport. The
ArchitectureFinding schema requires ALL of these fields:
- investigation_id: the first report's investigation_id (anchor ID)
- question: copy the original question from the supplied task exactly
- repository_scope: copy repository_scope exactly
- commit_scope: copy commit_scope exactly
- conclusions: a non-empty list of conclusions
- evidence_refs: the exact union of all conclusion evidence_refs
- open_questions: a list (empty only when none remain)
- summary: a concise synthesis of P1-P3
- source_investigation_ids: exact ordered list of all supplied report IDs
- status: "complete" only when the synthesis is complete; otherwise "incomplete"

Every conclusion must cite one or more evidence_refs copied exactly from the
supplied reports. Do not omit question or substitute InvestigationReport
fields such as findings/investigated_sources. Do not invent schema fields.
Do not claim that evidence proves more than the Investigator findings establish.
"""


def build_create_agent_analyst(model: Any) -> PocAnalyst:
    """Build a P4 Analyst with no repository tools and typed output."""
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=CREATE_AGENT_ANALYST_PROMPT_V1,
        response_format=ArchitectureFinding,
    )
    return PocAnalyst(agent=agent)


def build_create_agent_investigator(
    model: Any,
    bindings: Mapping[str, Callable[..., Any]],
    *,
    prompt: str = CREATE_AGENT_INVESTIGATION_PROMPT_V1,
    safety_config: Optional[ReActSafetyConfig] = None,
    request_budget_policy: Optional[RequestBudgetPolicy] = None,
    observation_max_chars: Optional[int] = None,
    tool_names: Optional[Sequence[str]] = None,
) -> PocInvestigator:
    """Construct the Investigator with the REAL ``create_agent`` API.

    *model* is any ``BaseChatModel`` (the caller supplies the
    project's provider configuration â€” e.g. ``ChatOpenAI`` built from
    the same ``OPENAI_API_KEY`` the ``llm`` package loads â€” so no
    provider SDK is imported here).
    *bindings* maps tool name -> the EXISTING already-bound project
    tool callable (``repository_query.live_bindings`` output or an
    equivalent fixture binding); the tool functions themselves are
    never re-implemented.

    Middleware order (documented, test-pinned): budget FIRST
    (outermost) so a budget rejection raises before the execution
    guard records a model call â€” a rejection never consumes the
    model-call allowance; then the execution guard; then observation
    bounding.
    """
    selected_names = (
        list(tool_names)
        if tool_names is not None
        else [name for name in ARCHITECTURE_TOOLS if name in bindings]
    )
    # A requested name WITHOUT an existing binding is never silently
    # wrapped â€” the Investigator runs only with real, already-bound
    # project tools.
    selected_names = [name for name in selected_names if name in bindings]
    definitions = investigator_tool_definitions(selected_names)
    lc_tools = []
    for name in selected_names:
        definition = definitions.get(name)
        lc_tools.append(
            to_langchain_tool(
                name,
                bindings[name],
                description=(
                    definition.description if definition is not None
                    else ""
                ),
                input_schema=(
                    dict(definition.input_schema)
                    if definition is not None
                    else None
                ),
            )
        )

    # An explicit RequestBudgetPolicy wins. Otherwise honor the existing
    # ReActSafetyConfig's per-request token ceiling, falling back to the
    # project's standard request policy when no stricter ceiling is set.
    effective_budget_policy = request_budget_policy
    if effective_budget_policy is None and safety_config is not None:
        effective_budget_policy = safety_config.request_budget_policy()
    budget = RequestBudgetMiddleware(effective_budget_policy)
    execution = ExecutionLimitMiddleware(
        safety_config,
        request_budget=budget,
        investigation_tool_names=selected_names,
    )
    effective_observation_max_chars = observation_max_chars
    if effective_observation_max_chars is None and safety_config is not None:
        effective_observation_max_chars = safety_config.max_observation_chars
    observation = ObservationBoundingMiddleware(
        effective_observation_max_chars
    )

    middleware = [budget, execution, observation]
    agent = create_agent(
        model=model,
        tools=lc_tools,
        system_prompt=prompt,
        middleware=middleware,
        response_format=InvestigationReport,
    )
    # A single bounded repair path for schema-validation failures. The repair
    # agent has no repository tools, so it can correct the structured contract
    # without restarting repository exploration. It shares the same budget and
    # execution middleware so the retry does not silently reset safety limits.
    repair_agent = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            prompt
            + "\n\nSTRUCTURED OUTPUT REPAIR MODE\n"
            + "You receive the prior rejected report and its validation error. "
            + "Return only a corrected InvestigationReport. Do not call tools, "
            + "invent evidence, or change source references. Preserve the "
            + "original findings and make evidence_refs the exact union of all "
            + "finding evidence_refs using (path, line_start, line_end)."
        ),
        middleware=middleware,
        response_format=InvestigationReport,
    )
    return PocInvestigator(
        agent=agent,
        budget=budget,
        execution=execution,
        observation=observation,
        tool_names=selected_names,
        repair_agent=repair_agent,
    )


def _investigator_task_message(
    question: str,
    repository_scope: str,
    commit_scope: str,
    step_goal: str,
) -> str:
    """The human-task message in the existing direct-mode TASK-block
    convention (system prompt carries the versioned instructions)."""
    return (
        "TASK\n"
        "question: {question}\n"
        "repository_scope: {repository}\n"
        "commit_scope: {commit}\n"
        "step_goal: {goal}".format(
            question=question,
            repository=repository_scope,
            commit=commit_scope,
            goal=step_goal,
        )
    )


def _validate_poc_plan(plan: InvestigationPlan) -> None:
    """POC plan shape: exactly the four ids P1-P4, non-empty goals
    (``validate_plan`` already checked goal-level-ness and
    uniqueness)."""
    ids = tuple(step.id for step in plan.steps)
    if ids != POC_PLAN_STEP_IDS:
        raise ValueError(
            "this POC requires exactly the four plan steps {}, got "
            "{}".format(", ".join(POC_PLAN_STEP_IDS), list(ids))
        )
    for step in plan.steps:
        if not step.goal.strip():
            raise ValueError(
                "plan step {!r} has an empty goal".format(step.id)
            )


def _validate_assignments(
    plan: InvestigationPlan,
    assignments: List[RoleAssignment],
) -> None:
    """Every plan step has EXACTLY one valid assignment."""
    step_ids = [step.id for step in plan.steps]
    assigned = [item.step_id for item in assignments]
    if len(assignments) != len(step_ids) or sorted(assigned) != sorted(
        step_ids
    ):
        raise ValueError(
            "assignments must cover every step exactly once: plan {} "
            "assignments {}".format(step_ids, assigned)
        )
    for item in assignments:
        if item.role not in AGENT_ROLES:
            raise ValueError(
                "unsupported role {!r} for step {!r}".format(
                    item.role, item.step_id
                )
            )


def _find_structured_output_validation_error(exc: BaseException) -> Optional[BaseException]:
    """Find LangChain's structured-output validation exception in a cause chain."""
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "StructuredOutputValidationError":
            return current
        current = current.__cause__ or current.__context__
    return None


def _repair_message(task_message: str, error: BaseException) -> str:
    """Build a bounded repair-only input from the rejected structured payload."""
    rejected = getattr(error, "tool_input", None)
    if rejected is None:
        rejected = getattr(error, "raw_output", None)
    try:
        rejected_text = json.dumps(rejected, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rejected_text = str(rejected)
    # Keep the repair prompt bounded even if a provider returns an oversized
    # structured payload. The retry is not another repository investigation.
    rejected_text = rejected_text[:12000]
    return (
        task_message
        + "\n\nSTRUCTURED OUTPUT REPAIR REQUEST\n"
        + "The prior InvestigationReport was rejected by schema validation. "
        + "Correct the report using the original payload below. Do not use "
        + "repository tools or conduct new exploration. Preserve the findings "
        + "and include every finding evidence reference in the aggregate list, "
        + "matching (path, line_start, line_end) exactly.\n"
        + "Validation error:\n"
        + str(error)[:4000]
        + "\nRejected payload (may be truncated):\n"
        + rejected_text
    )




def _report_ref_keys(report: InvestigationReport) -> set:
    return {
        (ref.path, ref.line_start, ref.line_end)
        for ref in report.evidence_refs
    }


def validate_poc_synthesis(
    reports: Sequence[InvestigationReport],
    finding: ArchitectureFinding,
    *,
    repository_scope: str,
    commit_scope: str,
) -> ValidationResult:
    """Validate P4 provenance and structure without an LLM.

    This checks evidence linkage, not semantic entailment: it verifies that
    cited references came from P1-P3 and are attached to Investigator
    findings, but cannot prove that natural-language claims are logically
    entailed by the underlying source text.
    """
    reports = list(reports)
    anchor_id = reports[0].investigation_id if reports else "missing-report"
    checks: List[ValidationCheck] = []

    def add(rule: str, ok: bool, detail: str = "") -> None:
        checks.append(ValidationCheck(
            rule=rule, ok=ok, detail="" if ok else detail[:400]
        ))

    expected_ids = [report.investigation_id for report in reports]
    add(
        "three_investigation_reports",
        len(reports) == 3,
        "expected P1-P3 reports, received {}".format(len(reports)),
    )
    add(
        "investigation_reports_complete",
        len(reports) == 3 and all(
            report.status == "complete" and bool(report.findings)
            for report in reports
        ),
        "every P1-P3 report must be complete and contain findings",
    )
    scopes_match = (
        all(report.repository_scope == repository_scope
            and report.commit_scope == commit_scope for report in reports)
        and finding.repository_scope == repository_scope
        and finding.commit_scope == commit_scope
    )
    add(
        "scope_match",
        scopes_match,
        "P4 output and all reports must match the requested repository/commit scope",
    )
    add(
        "source_report_ids",
        finding.source_investigation_ids == expected_ids,
        "P4 source_investigation_ids must exactly match P1-P3 report IDs in order",
    )
    add(
        "anchor_report_id",
        bool(reports) and finding.investigation_id == anchor_id,
        "P4 investigation_id must equal the first report ID as the anchor",
    )
    add(
        "nonempty_synthesis",
        bool(finding.summary.strip()) and bool(finding.conclusions),
        "P4 must provide a summary and at least one conclusion",
    )

    report_ref_keys = (
        set().union(*(_report_ref_keys(report) for report in reports))
        if reports else set()
    )
    conclusion_ref_keys = {
        (ref.path, ref.line_start, ref.line_end)
        for conclusion in finding.conclusions
        for ref in conclusion.evidence_refs
    }
    aggregate_ref_keys = {
        (ref.path, ref.line_start, ref.line_end)
        for ref in finding.evidence_refs
    }
    add(
        "evidence_refs_from_reports",
        conclusion_ref_keys.issubset(report_ref_keys),
        "one or more P4 conclusion evidence references are absent from P1-P3 report aggregates",
    )
    add(
        "aggregate_matches_conclusions",
        aggregate_ref_keys == conclusion_ref_keys,
        "P4 evidence_refs must be the exact union of conclusion evidence_refs",
    )

    investigator_finding_refs = set()
    for report in reports:
        for report_finding in report.findings:
            investigator_finding_refs.update(
                (ref.path, ref.line_start, ref.line_end)
                for ref in report_finding.evidence_refs
            )
    conclusion_links_ok = bool(finding.conclusions) and all(
        bool(conclusion.evidence_refs)
        and any(
            (ref.path, ref.line_start, ref.line_end)
            in investigator_finding_refs
            for ref in conclusion.evidence_refs
        )
        for conclusion in finding.conclusions
    )
    add(
        "conclusion_finding_links",
        conclusion_links_ok,
        "each P4 conclusion must cite evidence attached to at least one P1-P3 finding",
    )

    return ValidationResult(
        investigation_id=anchor_id,
        status="passed" if all(check.ok for check in checks) else "failed",
        checks=checks,
    )


def build_poc_graph(
    *,
    planner_client: LLMClient,
    investigator: PocInvestigator,
    analyst: Optional[PocAnalyst] = None,
    multi_step: bool = False,
) -> Any:
    """Compile the POC ``StateGraph`` over the injected planner
    client (provider-neutral) and the ``create_agent`` Investigator.
    Dependencies are injected â€” nothing here reaches for a provider.
    If ``analyst`` is supplied with ``multi_step=True``, P4 runs after
    P1-P3 and its result must pass deterministic provenance validation.
    """

    def planner(state: LangGraphPocState) -> Dict[str, Any]:
        question = state.get("question") or ""
        try:
            request = build_planning_request(question)
            response = planner_client.complete(request)
            plan = plan_from_response(response)
            validate_plan(plan)
            _validate_poc_plan(plan)
        except Exception as exc:  # explicit, never a fabricated plan
            return {
                "failure": "PLANNER_FAILED",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "routing": ROUTING_FAIL,
            }
        return {"plan": plan, "routing": ROUTING_CONTINUE}

    def query_router(state: LangGraphPocState) -> Dict[str, Any]:
        plan = state.get("plan")
        try:
            assignments = assign_roles(
                plan.steps, role_overrides=dict(POC_ROLE_OVERRIDES)
            )
            _validate_assignments(plan, assignments)
        except Exception as exc:
            return {
                "failure": "ROUTER_FAILED",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "routing": ROUTING_FAIL,
            }
        return {
            "role_assignments": assignments,
            "routing": ROUTING_CONTINUE,
        }

    def orchestrator_select(state: LangGraphPocState) -> Dict[str, Any]:
        plan = state.get("plan")
        roles = {
            item.step_id: item
            for item in (state.get("role_assignments") or [])
        }
        try:
            completed = set(state.get("completed_step_ids") or [])
            investigator_steps = [
                step for step in plan.steps
                if step.id in POC_PLAN_STEP_IDS[:3]
                and roles.get(step.id) is not None
                and roles[step.id].role == INVESTIGATOR
            ]
            eligible_steps = investigator_steps if multi_step else investigator_steps[:1]
            step = next(
                (candidate for candidate in eligible_steps
                 if candidate.id not in completed),
                None,
            )
            if step is None:
                raise ValueError("no unfinished Investigator step remains")
            if step.id not in POC_PLAN_STEP_IDS[:3]:
                raise ValueError(
                    "unexpected Investigator step id {!r}".format(step.id)
                )
            assignment = roles.get(step.id)
            if assignment is None or assignment.role != INVESTIGATOR:
                raise ValueError(
                    "step {!r} is not assigned to the INVESTIGATOR".format(
                        step.id
                    )
                )
            task = TaskPayload(
                question=state.get("question") or "",
                repository_scope=state.get("repository_scope") or "",
                commit_scope=state.get("commit_scope") or "",
            )
        except Exception as exc:
            return {
                "failure": "SELECTION_FAILED",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "routing": ROUTING_FAIL,
            }
        return {
            "selected_step_id": step.id,
            "selected_step_goal": step.goal,
            "selected_task": task,
            "routing": ROUTING_CONTINUE,
        }

    def investigator_node(state: LangGraphPocState) -> Dict[str, Any]:
        task: TaskPayload = state["selected_task"]
        step_id: str = state["selected_step_id"]
        plan: InvestigationPlan = state["plan"]
        goal = state.get("selected_step_goal") or next(
            step.goal for step in plan.steps if step.id == step_id
        )
        message = HumanMessage(
            _investigator_task_message(
                task.question,
                task.repository_scope,
                task.commit_scope,
                goal,
            )
        )
        repair_attempts = 0
        # Each plan step gets a fresh budget. A validation repair, if needed,
        # shares that same invocation budget and is allowed exactly once.
        investigator.execution.reset_for_invocation()
        try:
            result = investigator.agent.invoke(
                {"messages": [message]},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except RequestBudgetExceededError as exc:
            investigator.execution.note_budget_rejection(exc.record)
            return {
                "failure": "REQUEST_BUDGET_EXCEEDED",
                "error": str(exc),
                "routing": ROUTING_FAIL,
            }
        except ReActExecutionLimitExceeded as exc:
            return {
                "failure": exc.reason,
                "error": str(exc),
                "execution_snapshot": exc.snapshot,
                "routing": ROUTING_FAIL,
            }
        except Exception as exc:
            validation_error = _find_structured_output_validation_error(exc)
            if validation_error is None:
                return {
                    "failure": "INVESTIGATOR_FAILED",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "execution_snapshot": investigator.execution.guard.snapshot(),
                    "routing": ROUTING_FAIL,
                }
            repair_attempts = 1
            if investigator.repair_agent is None:
                return {
                    "failure": "STRUCTURED_OUTPUT_REPAIR_UNAVAILABLE",
                    "error": str(validation_error),
                    "repair_attempts": 0,
                    "repair_attempts_by_step": {
                        **(state.get("repair_attempts_by_step") or {}),
                        step_id: 0,
                    },
                    "execution_snapshot": investigator.execution.guard.snapshot(),
                    "routing": ROUTING_FAIL,
                }
            try:
                repair_input = HumanMessage(
                    _repair_message(message.content, validation_error)
                )
                result = investigator.repair_agent.invoke(
                    {"messages": [repair_input]},
                    config={"recursion_limit": min(AGENT_RECURSION_LIMIT, 8)},
                )
            except Exception as repair_exc:
                return {
                    "failure": "STRUCTURED_OUTPUT_REPAIR_FAILED",
                    "error": (
                        "initial validation error: {}\nrepair attempt failed: {}: {}"
                    ).format(
                        validation_error,
                        type(repair_exc).__name__,
                        repair_exc,
                    ),
                    "repair_attempts": repair_attempts,
                    "repair_attempts_by_step": {
                        **(state.get("repair_attempts_by_step") or {}),
                        step_id: repair_attempts,
                    },
                    "execution_snapshot": investigator.execution.guard.snapshot(),
                    "routing": ROUTING_FAIL,
                }
        report = result.get("structured_response")
        if not isinstance(report, InvestigationReport):
            return {
                "failure": "MISSING_STRUCTURED_OUTPUT",
                "error": (
                    "structured_response was {!r}, not a validated "
                    "InvestigationReport".format(type(report).__name__)
                ),
                "repair_attempts": repair_attempts,
                "repair_attempts_by_step": {
                    **(state.get("repair_attempts_by_step") or {}),
                    step_id: repair_attempts,
                },
                "routing": ROUTING_FAIL,
            }
        reports = list(state.get("investigator_reports") or [])
        reports.append(report)
        return {
            "investigator_report": report,
            "investigator_reports": reports,
            "repair_attempts": repair_attempts,
            "repair_attempts_by_step": {
                **(state.get("repair_attempts_by_step") or {}),
                step_id: repair_attempts,
            },
            "routing": ROUTING_SUCCESS,
        }

    def orchestrator_receive(state: LangGraphPocState) -> Dict[str, Any]:
        report = state.get("investigator_report")
        step_id = state.get("selected_step_id") or ""
        try:
            if not isinstance(report, InvestigationReport):
                raise ValueError("no validated report to receive")
            if not step_id:
                raise ValueError("no selected step to complete")
            # Re-validate against the existing contract (defense in
            # depth: the receiving boundary validates again before
            # recording success â€” never on failure).
            InvestigationReport.model_validate(report.model_dump())
        except Exception as exc:
            return {
                "failure": "RECEIVE_FAILED",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "routing": ROUTING_FAIL,
            }
        completed = list(state.get("completed_step_ids") or [])
        if step_id not in completed:
            completed.append(step_id)
        return {
            "completed_step_ids": completed,
            "routing": ROUTING_CONTINUE,
        }

    def analyst_node(state: LangGraphPocState) -> Dict[str, Any]:
        reports = list(state.get("investigator_reports") or [])
        plan = state.get("plan")
        step = next((item for item in plan.steps if item.id == "P4"), None)
        if analyst is None:
            return {
                "failure": "ANALYST_UNAVAILABLE",
                "error": "P4 was selected but no Analyst was injected",
                "routing": ROUTING_FAIL,
            }
        if len(reports) != 3 or any(
            report.status != "complete" for report in reports
        ):
            return {
                "failure": "ANALYST_INPUT_INCOMPLETE",
                "error": "P4 requires three complete P1-P3 InvestigationReports",
                "routing": ROUTING_FAIL,
            }
        payload = {
            "question": state.get("question") or "",
            "repository_scope": state.get("repository_scope") or "",
            "commit_scope": state.get("commit_scope") or "",
            "step_goal": step.goal if step is not None else (
                "Synthesize P1-P3 reports into the interaction model"
            ),
            "reports": [report.model_dump(mode="json") for report in reports],
        }
        message = HumanMessage(
            "SYNTHESIS TASK\\nUse only the supplied reports. Return an "
            "ArchitectureFinding, not an InvestigationReport.\\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        invoke_config = {
            "recursion_limit": min(AGENT_RECURSION_LIMIT, 8)
        }
        try:
            result = analyst.agent.invoke(
                {"messages": [message]},
                config=invoke_config,
            )
        except Exception as first_exc:
            # One bounded schema-repair retry for P4 only. The Analyst has
            # no repository tools; this retry cannot restart exploration.
            error_text = "{}: {}".format(
                type(first_exc).__name__, first_exc
            )
            if not _is_architecture_finding_validation_error(first_exc):
                return {
                    "failure": "ANALYST_FAILED",
                    "error": error_text,
                    "routing": ROUTING_FAIL,
                }
            repair_message = HumanMessage(
                "Your previous response failed ArchitectureFinding "
                "structured-output validation. Correct the schema error "
                "and return a complete ArchitectureFinding. The required "
                "field 'question' was missing. Include every required field: "
                "investigation_id, question, repository_scope, commit_scope, "
                "conclusions, evidence_refs, open_questions, summary, "
                "source_investigation_ids, and status. Do not return an "
                "InvestigationReport. Preserve all scopes and report IDs "
                "from the original task; cite only evidence references "
                "copied from P1-P3 reports. Validation error: "
                + error_text[:1600]
            )
            try:
                result = analyst.agent.invoke(
                    {"messages": [message, repair_message]},
                    config=invoke_config,
                )
            except Exception as repair_exc:
                return {
                    "failure": "STRUCTURED_OUTPUT_REPAIR_FAILED",
                    "error": (
                        "Initial P4 schema validation failed; one repair "
                        "attempt also failed. {}: {}".format(
                            type(repair_exc).__name__, repair_exc
                        )
                    ),
                    "routing": ROUTING_FAIL,
                }
        finding = result.get("structured_response")
        if not isinstance(finding, ArchitectureFinding):
            return {
                "failure": "ANALYST_MISSING_STRUCTURED_OUTPUT",
                "error": (
                    "structured_response was {!r}, not ArchitectureFinding"
                    .format(type(finding).__name__)
                ),
                "routing": ROUTING_FAIL,
            }
        return {
            "architecture_finding": finding,
            "routing": ROUTING_SUCCESS,
        }

    def deterministic_validation_node(
        state: LangGraphPocState,
    ) -> Dict[str, Any]:
        reports = list(state.get("investigator_reports") or [])
        finding = state.get("architecture_finding")
        if not isinstance(finding, ArchitectureFinding):
            return {
                "failure": "SYNTHESIS_VALIDATION_FAILED",
                "error": "P4 did not produce an ArchitectureFinding",
                "routing": ROUTING_FAIL,
            }
        validation = validate_poc_synthesis(
            reports,
            finding,
            repository_scope=state.get("repository_scope") or "",
            commit_scope=state.get("commit_scope") or "",
        )
        if validation.status != "passed":
            failed = [
                "{}: {}".format(check.rule, check.detail)
                for check in validation.checks if not check.ok
            ]
            return {
                "validation_result": validation,
                "failure": "SYNTHESIS_VALIDATION_FAILED",
                "error": "; ".join(failed) or "P4 validation failed",
                "routing": ROUTING_FAIL,
            }
        completed = list(state.get("completed_step_ids") or [])
        if "P4" not in completed:
            completed.append("P4")
        return {
            "validation_result": validation,
            "completed_step_ids": completed,
            "routing": ROUTING_SUCCESS,
        }

    def final_response(state: LangGraphPocState) -> Dict[str, Any]:
        failure = state.get("failure")
        completed = list(state.get("completed_step_ids") or [])
        reports = list(state.get("investigator_reports") or [])
        selected_step_id = state.get("selected_step_id")
        step_statuses = {step_id: "completed" for step_id in completed}
        if failure and selected_step_id and selected_step_id not in completed:
            step_statuses[selected_step_id] = "failed"
        if analyst is not None and multi_step:
            if "P4" in completed:
                step_statuses["P4"] = "completed"
            elif all(step_id in completed for step_id in POC_PLAN_STEP_IDS[:3]):
                step_statuses["P4"] = "failed" if failure else "pending"
            else:
                step_statuses["P4"] = "pending"
        if failure:
            status = "PARTIAL" if completed else "FAILED"
        else:
            status = "SUCCESS"
        final = {
            "status": status,
            "failure": failure,
            "error": state.get("error") or "",
            # Keep the original single-report field for compatibility;
            # in multi-step mode it is the most recently completed report.
            "report": state.get("investigator_report") if reports else None,
            "reports": reports,
            "step_statuses": step_statuses,
            "execution_snapshot": state.get("execution_snapshot"),
            "repair_attempts": sum(
                (state.get("repair_attempts_by_step") or {}).values()
            ),
            "repair_attempts_by_step": state.get("repair_attempts_by_step") or {},
            "architecture_finding": state.get("architecture_finding"),
            "validation_result": state.get("validation_result"),
            "completed_step_ids": completed,
        }
        return {"final": final}

    def _route_continue_or_fail(state: LangGraphPocState) -> str:
        return (
            "continue"
            if state.get("routing") == ROUTING_CONTINUE
            else "final_response"
        )

    def _route_after_receive(state: LangGraphPocState) -> str:
        if state.get("routing") != ROUTING_CONTINUE:
            return "final_response"
        if not multi_step:
            return "final_response"
        completed = set(state.get("completed_step_ids") or [])
        pending = [step_id for step_id in POC_PLAN_STEP_IDS[:3]
                   if step_id not in completed]
        if pending:
            return "continue"
        return "analyst" if analyst is not None else "final_response"

    def _route_analyst_or_fail(state: LangGraphPocState) -> str:
        return (
            "validate"
            if state.get("routing") == ROUTING_SUCCESS
            else "final_response"
        )

    def _route_validation_or_final(state: LangGraphPocState) -> str:
        return "final_response"

    def _route_success_or_fail(state: LangGraphPocState) -> str:
        return (
            ROUTING_SUCCESS
            if state.get("routing") == ROUTING_SUCCESS
            else "final_response"
        )

    graph: StateGraph = StateGraph(LangGraphPocState)
    graph.add_node("planner", planner)
    graph.add_node("query_router", query_router)
    graph.add_node("orchestrator_select", orchestrator_select)
    graph.add_node("investigator", investigator_node)
    graph.add_node("orchestrator_receive", orchestrator_receive)
    if analyst is not None:
        graph.add_node("analyst", analyst_node)
        graph.add_node("deterministic_validation", deterministic_validation_node)
    graph.add_node("final_response", final_response)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_continue_or_fail,
        {"continue": "query_router", "final_response": "final_response"},
    )
    graph.add_conditional_edges(
        "query_router",
        _route_continue_or_fail,
        {
            "continue": "orchestrator_select",
            "final_response": "final_response",
        },
    )
    graph.add_conditional_edges(
        "orchestrator_select",
        _route_continue_or_fail,
        {"continue": "investigator", "final_response": "final_response"},
    )
    graph.add_conditional_edges(
        "investigator",
        _route_success_or_fail,
        {
            "success": "orchestrator_receive",
            "final_response": "final_response",
        },
    )
    graph.add_conditional_edges(
        "orchestrator_receive",
        _route_after_receive,
        {
            "continue": "orchestrator_select",
            "analyst": "analyst",
            "final_response": "final_response",
        },
    )
    if analyst is not None:
        graph.add_conditional_edges(
            "analyst",
            _route_analyst_or_fail,
            {"validate": "deterministic_validation",
             "final_response": "final_response"},
        )
        graph.add_conditional_edges(
            "deterministic_validation",
            _route_validation_or_final,
            {"final_response": "final_response"},
        )
    graph.add_edge("final_response", END)
    return graph.compile()


def run_poc(
    question: str,
    *,
    planner_client: LLMClient,
    investigator: PocInvestigator,
    analyst: Optional[PocAnalyst] = None,
    repository_scope: str = "",
    commit_scope: str = "",
    recursion_limit: int = GRAPH_RECURSION_LIMIT,
    multi_step: bool = False,
) -> Dict[str, Any]:
    """Execute the POC graph on one question.

    When ``multi_step=True``, run P1-P3 sequentially. If an Analyst is
    supplied, execute P4 synthesis and deterministic validation afterwards.
    ``recursion_limit`` bounds OUTER graph steps; middleware bounds each
    Investigator invocation independently.
    """
    graph = build_poc_graph(
        planner_client=planner_client,
        investigator=investigator,
        analyst=analyst,
        multi_step=multi_step,
    )
    initial: Dict[str, Any] = {
        "question": question,
        "repository_scope": repository_scope,
        "commit_scope": commit_scope,
        "multi_step": multi_step,
        "investigator_reports": [],
        "completed_step_ids": [],
    }
    return graph.invoke(
        initial, config={"recursion_limit": recursion_limit}
    )
