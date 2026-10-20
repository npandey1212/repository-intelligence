"""Middleware-protected LangGraph ``create_agent`` POC tests.

Behavioral proof for the POC brief:

* graph tests — planner -> router -> select -> REAL ``create_agent``
  investigator -> receive -> final response, explicit failures, and
  the reused existing contracts;
* middleware tests — the three ``react_safety``-backed middleware
  (budget / observation bounding / execution limits) behaving exactly
  as the project's generic controls promise, inside the real agent
  loop.

No live provider is ever reached: every model call is a scripted
in-process ``BaseChatModel``; the planner is a scripted ``LLMClient``.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional

import pytest

langchain = pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from repo_intelligence.agent.request_budget import RequestBudgetPolicy
from repo_intelligence.langgraph_poc import (
    CREATE_AGENT_INVESTIGATION_PROMPT_V1,
    AGENT_RECURSION_LIMIT,
    GRAPH_RECURSION_LIMIT,
    POC_PLAN_STEP_IDS,
    PocInvestigator,
    build_create_agent_analyst,
    build_create_agent_investigator,
    run_poc,
)
from repo_intelligence.langgraph_poc.middleware import (
    ReActExecutionLimitExceeded,
    llm_request_from_model_request,
)
from repo_intelligence.multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
)
from repo_intelligence.orchestration.roles import ANALYST, INVESTIGATOR
from repo_intelligence.planning.models import InvestigationPlan
from repo_intelligence.react_safety import (
    MAX_TOOL_RESPONSE_CHARS,
    ReActLoopGuard,
    ReActSafetyConfig,
    ReActTermination,
)
from repo_intelligence.tools.protocol import ToolResult

QUESTION = "How does the ledger subsystem persist entries?"
REPO_SCOPE = "poc-fixture"
COMMIT_SCOPE = "0000000"

PLAN_PAYLOAD = {
    "objective": QUESTION,
    "steps": [
        {
            "id": "P1",
            "goal": "Identify the ledger components and responsibilities",
        },
        {
            "id": "P2",
            "goal": "Trace the request flow through the ledger",
        },
        {
            "id": "P3",
            "goal": "Map the ledger's external tool integrations",
        },
        {
            "id": "P4",
            "goal": "Synthesize the architecture diagram specification",
        },
    ],
}

REPORT_ARGS = {
    "investigation_id": "inv-poc-1",
    "question": QUESTION,
    "repository_scope": REPO_SCOPE,
    "commit_scope": COMMIT_SCOPE,
    "findings": [
        {
            "statement": "The LedgerClient wraps persistence",
            "evidence_refs": [
                {
                    "path": "src/payments/ledger.py",
                    "line_start": 1,
                    "line_end": 5,
                }
            ],
            "support": "SUPPORTED",
        }
    ],
    "evidence_refs": [
        {"path": "src/payments/ledger.py", "line_start": 1, "line_end": 5}
    ],
    "open_questions": [],
    "uncertainties": [],
    "investigated_sources": ["src/payments/ledger.py"],
    "status": "complete",
}


class SpyChatModel(BaseChatModel):
    """Scripted model: pops a queue of ``AIMessage`` replies and
    records every request it receives (messages snapshot)."""

    queue: List[BaseMessage] = Field(default_factory=list)
    calls: List[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "spy"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append({"messages": list(messages)})
        if not self.queue:
            raise AssertionError("model queue exhausted")
        message = self.queue.pop(0)
        return ChatResult(
            generations=[ChatGeneration(message=message)]
        )


class FakePlannerClient:
    """Provider-neutral scripted planner ``LLMClient``."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.requests: List[Any] = []

    def complete(self, request: Any) -> Any:
        from repo_intelligence.llm.models import LLMResponse, Message

        self.requests.append(request)
        return LLMResponse(
            message=Message(role="assistant", content=""),
            structured_output=(
                dict(self.payload)
                if isinstance(self.payload, dict) else self.payload
            ),
        )


def _tool_turn(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": dict(args),
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _structured_turn(args: dict) -> AIMessage:
    return _tool_turn("InvestigationReport", args, "call-structured")


def _read_file_binding(arguments: Any) -> ToolResult:
    return ToolResult(
        tool="read_file",
        success=True,
        data={
            "path": str((arguments or {}).get("path", "")),
            "content": "line1\nline2\nline3",
        },
        source=None,
    )


def _huge_string_binding(text: str) -> Any:
    def binding(arguments: Any) -> str:
        return text

    return binding


def _happy_spy() -> SpyChatModel:
    return SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py",
                 "line_start": 1, "line_end": 3},
                "call-1",
            ),
            _structured_turn(REPORT_ARGS),
        ]
    )


def _investigator(
    spy: SpyChatModel,
    bindings: Any = None,
    **kwargs: Any,
) -> PocInvestigator:
    if bindings is None:
        bindings = {"read_file": _read_file_binding}
    return build_create_agent_investigator(spy, bindings, **kwargs)


def _run(
    spy: SpyChatModel,
    planner_payload: Any = PLAN_PAYLOAD,
    investigator: Optional[PocInvestigator] = None,
    **kwargs: Any,
) -> Any:
    inv = investigator or _investigator(spy)
    planner = FakePlannerClient(planner_payload)
    return run_poc(
        QUESTION,
        planner_client=planner,
        investigator=inv,
        repository_scope=REPO_SCOPE,
        commit_scope=COMMIT_SCOPE,
        **kwargs,
    )


def test_create_agent_default_prompt_matches_structured_report_contract() -> None:
    assert "InvestigationReport" in CREATE_AGENT_INVESTIGATION_PROMPT_V1
    assert "Do not return an ArchitectureReport" in CREATE_AGENT_INVESTIGATION_PROMPT_V1
    assert "evidence_refs must aggregate" in CREATE_AGENT_INVESTIGATION_PROMPT_V1
    assert "stop calling tools" in CREATE_AGENT_INVESTIGATION_PROMPT_V1


# -- graph tests ---------------------------------------------------------------


def test_happy_path_completes_with_validated_report() -> None:
    spy = _happy_spy()
    state = _run(spy)
    final = state["final"]
    assert final["status"] == "SUCCESS"
    assert final["failure"] is None
    assert final["completed_step_ids"] == ["P1"]
    report = final["report"]
    assert isinstance(report, InvestigationReport)
    assert report.investigation_id == "inv-poc-1"
    assert report.findings[0].support == "SUPPORTED"


def test_planner_uses_existing_contract_and_four_step_shape() -> None:
    spy = _happy_spy()
    planner = FakePlannerClient(PLAN_PAYLOAD)
    state = run_poc(
        QUESTION,
        planner_client=planner,
        investigator=_investigator(spy),
        repository_scope=REPO_SCOPE,
        commit_scope=COMMIT_SCOPE,
    )
    plan = state["plan"]
    assert isinstance(plan, InvestigationPlan)
    assert tuple(step.id for step in plan.steps) == POC_PLAN_STEP_IDS
    # The existing provider-neutral planning request was sent verbatim.
    request = planner.requests[0]
    assert request.tools == []
    assert request.structured_output is not None
    assert request.structured_output.name == "InvestigationPlan"
    assert plan.objective == QUESTION


def test_router_assigns_expected_roles_and_covers_every_step() -> None:
    spy = _happy_spy()
    state = _run(spy)
    assignments = {
        item.step_id: item.role for item in state["role_assignments"]
    }
    assert assignments == {
        "P1": INVESTIGATOR,
        "P2": INVESTIGATOR,
        "P3": INVESTIGATOR,
        "P4": ANALYST,
    }


def test_only_first_step_p1_executes() -> None:
    spy = _happy_spy()
    state = _run(spy)
    assert state["selected_step_id"] == "P1"
    task = state["selected_task"]
    assert isinstance(task, TaskPayload)
    assert task.question == QUESTION
    assert state["completed_step_ids"] == ["P1"]
    # Exactly one TASK human message was ever shown to the model —
    # P2-P4 never reached the investigator.
    human_contents = {
        message.content
        for call in spy.calls
        for message in call["messages"]
        if isinstance(message, HumanMessage)
    }
    assert len(human_contents) == 1
    only = next(iter(human_contents)) or ""
    assert "step_goal: Identify the ledger components" in only
    assert "P2" not in only


def test_multi_step_runs_p1_p2_p3_with_distinct_goals_and_reports() -> None:
    queue = []
    for index, step_id in enumerate(("P1", "P2", "P3"), start=1):
        queue.append(_tool_turn(
            "read_file",
            {"path": "src/payments/{}.py".format(step_id.lower())},
            "{}-tool".format(step_id),
        ))
        report_args = dict(REPORT_ARGS)
        report_args["investigation_id"] = "inv-{}".format(step_id.lower())
        queue.append(_structured_turn(report_args))

    spy = SpyChatModel(queue=queue)
    state = _run(spy, multi_step=True)
    final = state["final"]

    assert final["status"] == "SUCCESS"
    assert final["failure"] is None
    assert final["completed_step_ids"] == ["P1", "P2", "P3"]
    assert [report.investigation_id for report in final["reports"]] == [
        "inv-p1", "inv-p2", "inv-p3"
    ]
    assert final["step_statuses"] == {
        "P1": "completed", "P2": "completed", "P3": "completed"
    }

    human_messages = [
        message.content
        for call in spy.calls
        for message in call["messages"]
        if isinstance(message, HumanMessage)
    ]
    combined = "\n".join(str(message) for message in human_messages)
    assert "step_goal: Identify the ledger components and responsibilities" in combined
    assert "step_goal: Trace the request flow through the ledger" in combined
    assert "step_goal: Map the ledger's external tool integrations" in combined


def test_structured_report_validation_gets_one_tool_free_repair() -> None:
    invalid_report = copy.deepcopy(REPORT_ARGS)
    invalid_report["investigation_id"] = "inv-p2-invalid"
    # Finding cites a source, but the report aggregate intentionally omits it.
    invalid_report["evidence_refs"] = []

    valid_p2 = copy.deepcopy(REPORT_ARGS)
    valid_p2["investigation_id"] = "inv-p2-repaired"
    valid_p3 = copy.deepcopy(REPORT_ARGS)
    valid_p3["investigation_id"] = "inv-p3"

    spy = SpyChatModel(queue=[
        # P1 succeeds normally.
        _tool_turn("read_file", {"path": "src/payments/p1.py"}, "p1-tool"),
        _structured_turn(REPORT_ARGS),
        # P2's first report violates the aggregate evidence invariant.
        _tool_turn("read_file", {"path": "src/payments/p2.py"}, "p2-tool"),
        _structured_turn(invalid_report),
        # The no-tools repair agent receives this corrected structured result.
        _structured_turn(valid_p2),
        # P3 proceeds normally after repaired P2.
        _tool_turn("read_file", {"path": "src/payments/p3.py"}, "p3-tool"),
        _structured_turn(valid_p3),
    ])

    state = _run(spy, multi_step=True)
    final = state["final"]

    assert final["status"] == "SUCCESS"
    assert final["completed_step_ids"] == ["P1", "P2", "P3"]
    assert [r.investigation_id for r in final["reports"]] == [
        "inv-poc-1", "inv-p2-repaired", "inv-p3"
    ]
    assert final["repair_attempts"] == 1
    assert final["repair_attempts_by_step"]["P2"] == 1


def test_multi_step_failure_preserves_completed_reports() -> None:
    queue = [
        _tool_turn("read_file", {"path": "src/payments/p1.py"}, "p1-tool"),
        _structured_turn(REPORT_ARGS),
        # The P2 turn is malformed as a report tool call; the create_agent
        # path cannot produce a valid structured InvestigationReport.
        _tool_turn("read_file", {"path": "src/payments/p2.py"}, "p2-tool"),
    ]
    spy = SpyChatModel(queue=queue)
    state = _run(spy, multi_step=True)
    final = state["final"]

    assert final["status"] == "PARTIAL"
    assert final["completed_step_ids"] == ["P1"]
    assert len(final["reports"]) == 1
    assert final["step_statuses"]["P1"] == "completed"
    assert final["step_statuses"]["P2"] == "failed"


def test_investigator_runs_via_real_create_agent() -> None:
    spy = _happy_spy()
    inv = _investigator(spy)
    assert inv.tool_names == ["read_file"]
    assert inv.agent is not None
    # The three middleware are live objects attached to this
    # investigator (order pinned: budget, execution, observation).
    from langchain.agents.middleware import AgentMiddleware

    assert isinstance(inv.budget, AgentMiddleware)
    assert isinstance(inv.execution, AgentMiddleware)
    assert isinstance(inv.observation, AgentMiddleware)
    state = _run(spy, investigator=inv)
    assert state["final"]["status"] == "SUCCESS"
    # The REAL create_agent loop executed a real tool call: the spy
    # saw a ToolMessage for read_file before the structured turn.
    second_call_messages = spy.calls[1]["messages"]
    tool_contents = [
        message.content
        for message in second_call_messages
        if type(message).__name__ == "ToolMessage"
    ]
    assert tool_contents
    assert "line1" in tool_contents[0]


def _architecture_finding_args(
    source_ids: List[str],
    *,
    evidence_path: str = "src/payments/ledger.py",
) -> dict:
    ref = {"path": evidence_path, "line_start": 1, "line_end": 5}
    return {
        "investigation_id": source_ids[0],
        "source_investigation_ids": list(source_ids),
        "question": QUESTION,
        "repository_scope": REPO_SCOPE,
        "commit_scope": COMMIT_SCOPE,
        "conclusions": [
            {
                "statement": "The ledger component is invoked through a defined request flow.",
                "classification": "derived",
                "evidence_refs": [ref],
            }
        ],
        "evidence_refs": [ref],
        "open_questions": [],
        "summary": "P1-P3 evidence describes the ledger components, request flow, and integrations.",
        "status": "complete",
    }


def test_m52_analyst_synthesizes_three_reports_and_validation_passes() -> None:
    queue = []
    for index, step_id in enumerate(("P1", "P2", "P3"), start=1):
        queue.append(_tool_turn(
            "read_file",
            {"path": "src/payments/{}.py".format(step_id.lower())},
            "{}-tool".format(step_id),
        ))
        report_args = dict(REPORT_ARGS)
        report_args["investigation_id"] = "inv-{}".format(step_id.lower())
        queue.append(_structured_turn(report_args))
    investigator_spy = SpyChatModel(queue=queue)
    analyst_spy = SpyChatModel(queue=[
        _tool_turn(
            "ArchitectureFinding",
            _architecture_finding_args(["inv-p1", "inv-p2", "inv-p3"]),
            "analyst-structured",
        )
    ])
    analyst = build_create_agent_analyst(analyst_spy)
    state = _run(investigator_spy, multi_step=True, analyst=analyst)
    final = state["final"]

    assert final["status"] == "SUCCESS"
    assert final["failure"] is None
    assert final["completed_step_ids"] == ["P1", "P2", "P3", "P4"]
    assert final["step_statuses"] == {
        "P1": "completed", "P2": "completed",
        "P3": "completed", "P4": "completed",
    }
    assert isinstance(final["architecture_finding"], ArchitectureFinding)
    assert final["architecture_finding"].source_investigation_ids == [
        "inv-p1", "inv-p2", "inv-p3"
    ]
    assert final["validation_result"].status == "passed"
    assert len(final["validation_result"].checks) >= 8
    # Analyst receives all three reports and no repository tools.
    analyst_human = [
        message.content for call in analyst_spy.calls
        for message in call["messages"] if isinstance(message, HumanMessage)
    ]
    assert len(analyst_human) == 1
    assert all(report_id in analyst_human[0]
               for report_id in ("inv-p1", "inv-p2", "inv-p3"))


def test_m52_analyst_repairs_missing_question_once() -> None:
    queue = []
    for index, step_id in enumerate(("P1", "P2", "P3"), start=1):
        queue.append(_tool_turn(
            "read_file",
            {"path": "src/payments/{}.py".format(step_id.lower())},
            "{}-tool".format(step_id),
        ))
        report_args = dict(REPORT_ARGS)
        report_args["investigation_id"] = "inv-{}".format(step_id.lower())
        queue.append(_structured_turn(report_args))
    investigator_spy = SpyChatModel(queue=queue)

    invalid_args = _architecture_finding_args(
        ["inv-p1", "inv-p2", "inv-p3"]
    )
    invalid_args.pop("question")
    valid_args = _architecture_finding_args(
        ["inv-p1", "inv-p2", "inv-p3"]
    )
    analyst_spy = SpyChatModel(queue=[
        _tool_turn("ArchitectureFinding", invalid_args, "analyst-invalid"),
        _tool_turn("ArchitectureFinding", valid_args, "analyst-repaired"),
    ])
    analyst = build_create_agent_analyst(analyst_spy)
    state = _run(investigator_spy, multi_step=True, analyst=analyst)
    final = state["final"]

    assert final["status"] == "SUCCESS"
    assert final["completed_step_ids"] == ["P1", "P2", "P3", "P4"]
    assert final["step_statuses"]["P4"] == "completed"
    assert isinstance(final["architecture_finding"], ArchitectureFinding)
    assert final["architecture_finding"].question == QUESTION
    # The retry is bounded to one additional model call and explicitly
    # identifies the missing required field.
    analyst_messages = [
        message.content for call in analyst_spy.calls
        for message in call["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert any("field 'question' was missing" in content
               for content in analyst_messages)


def test_m52_deterministic_validation_rejects_foreign_evidence() -> None:
    reports = [
        InvestigationReport.model_validate({
            **REPORT_ARGS, "investigation_id": "inv-p1"
        }),
        InvestigationReport.model_validate({
            **REPORT_ARGS, "investigation_id": "inv-p2"
        }),
        InvestigationReport.model_validate({
            **REPORT_ARGS, "investigation_id": "inv-p3"
        }),
    ]
    finding = ArchitectureFinding.model_validate(
        _architecture_finding_args(
            ["inv-p1", "inv-p2", "inv-p3"],
            evidence_path="docs/not-in-reports.md",
        )
    )
    from repo_intelligence.langgraph_poc.graph import validate_poc_synthesis
    result = validate_poc_synthesis(
        reports, finding,
        repository_scope=REPO_SCOPE,
        commit_scope=COMMIT_SCOPE,
    )
    assert result.status == "failed"
    assert any(
        check.rule == "evidence_refs_from_reports" and not check.ok
        for check in result.checks
    )


def test_analyst_step_is_metadata_only() -> None:
    spy = _happy_spy()
    state = _run(spy)
    # The P4 ANALYST assignment exists but nothing analyst-like ran:
    # the final response carries only the investigator's report and
    # only P1 as completed.
    assert any(
        item.step_id == "P4" and item.role == ANALYST
        for item in state["role_assignments"]
    )
    assert state["final"]["completed_step_ids"] == ["P1"]
    assert state["final"]["report"] is state["investigator_report"]


def test_invalid_planner_payload_fails_explicitly() -> None:
    spy = SpyChatModel(queue=[])
    state = _run(spy, planner_payload={"objective": QUESTION})
    final = state["final"]
    assert final["status"] == "FAILED"
    assert final["failure"] == "PLANNER_FAILED"
    assert final["report"] is None
    assert spy.calls == []


def test_three_step_plan_fails_poc_shape_check() -> None:
    payload = {
        "objective": QUESTION,
        "steps": [
            {"id": "P1", "goal": "Identify the ledger components"},
            {"id": "P2", "goal": "Trace the request flow"},
            {"id": "P3", "goal": "Synthesize the diagram"},
        ],
    }
    spy = SpyChatModel(queue=[])
    state = _run(spy, planner_payload=payload)
    assert state["final"]["failure"] == "PLANNER_FAILED"
    assert "P4" in state["final"]["error"] or "P1" in state["final"]["error"]
    assert spy.calls == []


def test_tool_level_goal_fails_existing_plan_validation() -> None:
    payload = {
        "objective": QUESTION,
        "steps": [
            {"id": "P1", "goal": "call read_file(src/ledger.py)"},
            {"id": "P2", "goal": "Trace the request flow"},
            {"id": "P3", "goal": "Map external integrations"},
            {"id": "P4", "goal": "Synthesize the diagram"},
        ],
    }
    spy = SpyChatModel(queue=[])
    state = _run(spy, planner_payload=payload)
    assert state["final"]["failure"] == "PLANNER_FAILED"
    assert "tool" in state["final"]["error"].lower()
    assert spy.calls == []


def test_empty_scopes_fail_selection_explicitly() -> None:
    spy = SpyChatModel(queue=[])
    inv = _investigator(spy)
    state = run_poc(
        QUESTION,
        planner_client=FakePlannerClient(PLAN_PAYLOAD),
        investigator=inv,
    )
    assert state["final"]["failure"] == "SELECTION_FAILED"
    assert spy.calls == []


def test_graph_and_agent_recursion_limits_are_distinct() -> None:
    # The create_agent graph counts model/tool nodes separately. Its
    # failsafe must allow the middleware's default model-call ceiling to
    # raise the typed, inspectable ReAct limit error first.
    assert AGENT_RECURSION_LIMIT >= ReActSafetyConfig().max_model_calls + 2
    assert GRAPH_RECURSION_LIMIT > AGENT_RECURSION_LIMIT
    assert GRAPH_RECURSION_LIMIT >= 20


# -- middleware tests ----------------------------------------------------------


def test_budget_rejects_initial_request_before_model() -> None:
    spy = SpyChatModel(queue=[_structured_turn(REPORT_ARGS)])
    policy = RequestBudgetPolicy(
        max_input_tokens=10, context_capacity_tokens=None
    )
    inv = _investigator(spy, request_budget_policy=policy)
    state = _run(spy, investigator=inv)
    final = state["final"]
    assert final["failure"] == "REQUEST_BUDGET_EXCEEDED"
    # The provider (spy) was never reached — nothing sent, nothing
    # retried.
    assert spy.calls == []
    # A rejection does NOT consume the model-call allowance (the
    # guard records zero model calls), but the loop is marked
    # terminated so nothing further can run.
    snapshot = inv.execution.guard.snapshot()
    assert snapshot["model_calls"] == 0
    assert snapshot["termination_reason"] == (
        ReActTermination.REQUEST_BUDGET_EXCEEDED
    )
    assert inv.budget.rejected
    assert inv.budget.rejected[0]["disposition"] == "rejected"
    assert inv.budget.rejected[0]["totals"]["total_tokens_est"] > 10


def _first_call_estimate() -> int:
    """Probe the first investigator request size with a generous
    policy, then report its token estimate."""
    probe_spy = SpyChatModel(
        queue=[_structured_turn(copy.deepcopy(REPORT_ARGS))]
    )
    probe_inv = build_create_agent_investigator(
        probe_spy,
        {"read_file": _read_file_binding},
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    probe_inv.agent.invoke(
        {"messages": [HumanMessage("probe")]}
    )
    return probe_inv.budget.records[0]["totals"]["total_tokens_est"]


def test_safety_config_request_budget_is_used_without_separate_policy() -> None:
    """The existing ReAct config's per-request ceiling must not be
    silently ignored when callers omit the duplicate policy argument."""
    spy = SpyChatModel(queue=[_structured_turn(REPORT_ARGS)])
    inv = _investigator(
        spy,
        safety_config=ReActSafetyConfig(max_request_input_tokens=10),
    )
    state = _run(spy, investigator=inv)
    assert state["final"]["status"] == "FAILED"
    assert state["final"]["failure"] == "REQUEST_BUDGET_EXCEEDED"
    assert spy.calls == []
    assert inv.budget.rejected[0]["budget"]["effective_input_budget_tokens"] == 10


def test_cumulative_budget_uses_estimates_and_blocks_next_provider_call() -> None:
    """Without provider usage metadata, the middleware must feed the
    request-budget estimate into ReActLoopGuard; the second request is
    rejected before the model sees it when the cumulative ceiling would
    be exceeded."""
    probe_spy = _happy_spy()
    probe_inv = _investigator(
        probe_spy,
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    probe_state = _run(probe_spy, investigator=probe_inv)
    assert probe_state["final"]["status"] == "SUCCESS"
    first_request_estimate = probe_inv.budget.records[0]["totals"][
        "total_tokens_est"
    ]
    assert first_request_estimate > 0

    spy = _happy_spy()
    inv = _investigator(
        spy,
        safety_config=ReActSafetyConfig(
            max_cumulative_input_tokens=first_request_estimate + 10
        ),
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    state = _run(spy, investigator=inv)
    assert state["final"]["status"] == "FAILED"
    assert state["final"]["failure"] == ReActTermination.CUMULATIVE_BUDGET_EXCEEDED
    assert len(spy.calls) == 1  # second provider call was blocked locally
    snapshot = inv.execution.guard.snapshot()
    assert snapshot["model_calls"] == 1
    assert snapshot["cumulative_estimated_input_tokens"] == first_request_estimate
    assert snapshot["termination_reason"] == ReActTermination.CUMULATIVE_BUDGET_EXCEEDED


def test_budget_rejects_second_request_after_tool_observation() -> None:
    spy = SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py",
                 "line_start": 1, "line_end": 3},
                "call-1",
            ),
            _structured_turn(REPORT_ARGS),
        ]
    )
    first_est = _first_call_estimate()
    assert first_est > 0
    big = "x" * 20_000
    policy = RequestBudgetPolicy(
        max_input_tokens=first_est + 200,
        context_capacity_tokens=None,
    )
    inv = _investigator(
        spy,
        bindings={"read_file": _huge_string_binding(big)},
        request_budget_policy=policy,
    )
    state = _run(spy, investigator=inv)
    final = state["final"]
    assert final["failure"] == "REQUEST_BUDGET_EXCEEDED"
    # First call was accepted and the tool ran; the SECOND call —
    # after the oversized observation joined history — was rejected
    # before any further model call.
    assert len(spy.calls) == 1
    dispositions = [r["disposition"] for r in inv.budget.records]
    assert dispositions == ["accepted", "rejected"]
    snapshot = inv.execution.guard.snapshot()
    assert snapshot["model_calls"] == 1
    assert inv.execution.guard.terminal_status() == (
        ReActTermination.REQUEST_BUDGET_EXCEEDED
    )


def test_budget_is_outermost_of_the_execution_guard() -> None:
    """With ``max_model_calls=1`` AND an oversized second request,
    the BUDGET rejection wins (it runs first) — proving the
    documented middleware order, not just the sum of the parts."""
    spy = SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py",
                 "line_start": 1, "line_end": 3},
                "call-1",
            ),
            _structured_turn(REPORT_ARGS),
        ]
    )
    first_est = _first_call_estimate()
    policy = RequestBudgetPolicy(
        max_input_tokens=first_est + 200,
        context_capacity_tokens=None,
    )
    inv = _investigator(
        spy,
        bindings={"read_file": _huge_string_binding("y" * 20_000)},
        safety_config=ReActSafetyConfig(max_model_calls=1),
        request_budget_policy=policy,
    )
    state = _run(spy, investigator=inv)
    assert state["final"]["failure"] == "REQUEST_BUDGET_EXCEEDED"
    assert state["final"]["failure"] != (
        ReActTermination.MODEL_CALL_LIMIT_REACHED
    )


def test_oversized_observation_bounded_before_next_model_call() -> None:
    big = "Z" * 50_000
    spy = SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py",
                 "line_start": 1, "line_end": 3},
                "call-1",
            ),
            _structured_turn(REPORT_ARGS),
        ]
    )
    inv = _investigator(
        spy,
        bindings={"read_file": _huge_string_binding(big)},
        observation_max_chars=4_000,
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    state = _run(spy, investigator=inv)
    assert state["final"]["status"] == "SUCCESS"
    # The middleware recorded the bounding decision.
    assert inv.observation.records
    record = inv.observation.records[0]
    assert record["truncated"] is True
    assert record["original_chars"] == 50_000
    assert record["returned_chars"] <= 4_000
    assert record["note"]
    # The SECOND model call saw the BOUNDED tool message, never the
    # 50k original.
    tool_contents = [
        message.content
        for message in spy.calls[1]["messages"]
        if type(message).__name__ == "ToolMessage"
    ]
    assert tool_contents
    bounded = tool_contents[0]
    assert isinstance(bounded, str)
    assert len(bounded) < 6_000
    assert "Z" * 4_100 not in bounded
    assert "observation bounded" in bounded or "truncated" in bounded


def test_safety_config_observation_limit_is_used_by_default() -> None:
    spy = SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py", "line_start": 1, "line_end": 3},
                "call-1",
            ),
            _structured_turn(REPORT_ARGS),
        ]
    )
    inv = _investigator(
        spy,
        bindings={"read_file": _huge_string_binding("Q" * 5_000)},
        safety_config=ReActSafetyConfig(max_observation_chars=1_000),
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    state = _run(spy, investigator=inv)
    assert state["final"]["status"] == "SUCCESS"
    assert inv.observation.max_chars == 1_000
    assert inv.observation.records[0]["returned_chars"] <= 1_000


def test_iteration_limit_is_enforced_inside_create_agent() -> None:
    spy = SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py", "line_start": 1, "line_end": 3},
                "call-1",
            ),
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py", "line_start": 4, "line_end": 6},
                "call-2",
            ),
        ]
    )
    inv = _investigator(
        spy,
        safety_config=ReActSafetyConfig(max_iterations=1, max_model_calls=10),
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    state = _run(spy, investigator=inv)
    assert state["final"]["status"] == "FAILED"
    assert state["final"]["failure"] == ReActTermination.ITERATION_LIMIT_REACHED
    assert len(spy.calls) == 1
    assert inv.execution.guard.snapshot()["iterations"] == 1


def test_execution_limit_stops_after_two_model_calls() -> None:
    turns = [
        _tool_turn(
            "read_file",
            {"path": "src/payments/ledger.py",
             "line_start": 1, "line_end": 3},
            "call-{}".format(index),
        )
        for index in range(4)
    ]
    spy = SpyChatModel(queue=turns)
    inv = _investigator(
        spy,
        safety_config=ReActSafetyConfig(max_model_calls=2),
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    state = _run(spy, investigator=inv)
    final = state["final"]
    assert final["failure"] == ReActTermination.MODEL_CALL_LIMIT_REACHED
    assert final["status"] == "FAILED"
    assert len(spy.calls) == 2  # exactly the allowance, never more
    snapshot = inv.execution.guard.snapshot()
    assert snapshot["model_calls"] == 2
    assert snapshot["termination_reason"] == (
        ReActTermination.MODEL_CALL_LIMIT_REACHED
    )


def test_budget_failure_is_a_single_attempt_no_retry() -> None:
    spy = SpyChatModel(queue=[_structured_turn(REPORT_ARGS)])
    policy = RequestBudgetPolicy(
        max_input_tokens=10, context_capacity_tokens=None
    )
    inv = _investigator(spy, request_budget_policy=policy)
    planner = FakePlannerClient(PLAN_PAYLOAD)
    state = run_poc(
        QUESTION,
        planner_client=planner,
        investigator=inv,
        repository_scope=REPO_SCOPE,
        commit_scope=COMMIT_SCOPE,
    )
    # One planner request, zero model calls, one failed attempt —
    # the graph does not retry a budget rejection.
    assert len(planner.requests) == 1
    assert spy.calls == []
    assert state["final"]["failure"] == "REQUEST_BUDGET_EXCEEDED"
    assert inv.execution.guard.snapshot()["model_calls"] == 0


def test_safety_failure_flows_to_failed_final_response() -> None:
    spy = SpyChatModel(
        queue=[
            _tool_turn(
                "read_file",
                {"path": "src/payments/ledger.py",
                 "line_start": 1, "line_end": 3},
                "call-1",
            )
        ]
    )
    inv = _investigator(
        spy,
        safety_config=ReActSafetyConfig(max_model_calls=1),
        request_budget_policy=RequestBudgetPolicy(
            max_input_tokens=10**6, context_capacity_tokens=None
        ),
    )
    state = _run(spy, investigator=inv)
    final = state["final"]
    # A safety stop can NEVER flow onward as a valid result.
    assert final["status"] == "FAILED"
    assert final["report"] is None
    assert final["completed_step_ids"] == []
    assert final["error"]


def test_missing_structured_output_is_explicit_failure() -> None:
    # The model replies with plain content and no structured tool
    # call; create_agent eventually returns without a validated
    # report (or raises) — either way the graph fails explicitly.
    spy = SpyChatModel(
        queue=[AIMessage(content="I decline to structure that.")]
    )
    inv = _investigator(spy)
    state = _run(spy, investigator=inv)
    final = state["final"]
    assert final["status"] == "FAILED"
    assert final["failure"] in {
        "MISSING_STRUCTURED_OUTPUT",
        "INVESTIGATOR_FAILED",
    }


def test_llm_request_projection_is_provider_neutral() -> None:
    from langchain.agents.middleware import ModelRequest
    from langchain_core.tools import StructuredTool

    spy = SpyChatModel(queue=[])

    tool = StructuredTool(
        name="read_file",
        description="read a file",
        func=lambda **kwargs: "{}",
        args_schema={"type": "object", "properties": {}},
    )
    model_request = ModelRequest(
        model=spy,
        messages=[HumanMessage("find the ledger")],
        system_message=None,
        tools=[tool],
    )
    projected = llm_request_from_model_request(model_request)
    assert [message.role for message in projected.messages] == ["user"]
    assert projected.messages[0].content == "find the ledger"
    assert [definition.name for definition in projected.tools] == [
        "read_file"
    ]


def test_middleware_reuse_no_duplicate_safety_implementation() -> None:
    import inspect

    import repo_intelligence.langgraph_poc.middleware as module

    source = inspect.getsource(module)
    # One budget implementation: the middleware delegates to the
    # existing machinery, and nothing provider-specific appears.
    assert "enforce_request_budget_or_raise" in source
    assert "bound_observation" in source
    assert "ReActLoopGuard" in source
    assert "CHARS_PER_TOKEN" not in source
    # No provider SDK import anywhere in the module (the only
    # permitted mention of a provider path is the docstring pointer).
    assert "import openai" not in source
    assert "from openai" not in source
    assert "ChatOpenAI" not in source
    assert "gpt-" not in source.lower()
    # And the guard it configures is the project's own class.
    inv = _investigator(SpyChatModel(queue=[]))
    assert isinstance(inv.execution.guard, ReActLoopGuard)
    # Default ceiling untouched.
    assert inv.observation.max_chars == MAX_TOOL_RESPONSE_CHARS
    assert inv.execution.guard.snapshot()["config"]["max_model_calls"] == 16


def test_default_budget_policy_is_the_project_standard() -> None:
    from repo_intelligence.agent.request_budget import (
        enforce_request_budget_or_raise,
    )
    from repo_intelligence.llm.models import LLMRequest, Message

    inv = _investigator(SpyChatModel(queue=[]))
    assert inv.budget.policy is None  # None = global standard, never weakened
    assert RequestBudgetPolicy().max_input_tokens is None
    # The global default effective budget stays the pinned value.
    _, record = enforce_request_budget_or_raise(
        LLMRequest(messages=[Message(role="user", content="hi")]),
        RequestBudgetPolicy(),
    )
    assert record["budget"]["effective_input_budget_tokens"] == 95_232


def test_build_uses_only_existing_bound_tool_definitions() -> None:
    spy = SpyChatModel(queue=[])
    inv = _investigator(spy)  # binding only for read_file
    assert inv.tool_names == ["read_file"]
    # A requested name WITHOUT a binding is never silently wrapped.
    inv2 = build_create_agent_investigator(
        spy,
        {"read_file": _read_file_binding},
        tool_names=["read_file", "query_engineering_graph"],
    )
    assert inv2.tool_names == ["read_file"]
