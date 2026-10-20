"""M10 — sequential plan-and-execute through the EXISTING Agent (spec
§10–§20, §30, §38).

``Agent.run_planned`` is plan MODE of the same Agent — same
ToolDispatcher, same tool definitions, same conversation state, same
one M8.5 budget, same trace, same error semantics. What is new:

    planner request (one LLM call, no tools)      → InvestigationPlan
    deterministic validate_plan                   → or explicit failure
    plan recorded in the existing AgentTrace      → BEFORE any tool
    each step executed sequentially               → existing ReAct
                                                   loop, iterations
                                                   tagged plan_step_id
    one AgentRunResult                            → final step's result

Spec §38 coverage:
    planning contract tests 1–6   (missing/empty/duplicate/malformed →
                                   explicit failure, NO tool execution)
    ordering                      (plan creation before first tool)
    sequential execution          (P1 → P2 → P3; P2 cannot start
                                   before P1 is terminal)
    ReAct inside a step           (tool → observation → tool → result)
    tool-call bound               (one global AgentExecutionConfig)
    failure                       (planner failure, validation
                                   failure, step failure, plan
                                   invalidation, limit reached —
                                   none silently becomes success)

Non-plan runs are untouched (existing M8.x suite, run through the
same loop path).
"""

from __future__ import annotations

import copy
import json
from typing import List

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
)
from repo_intelligence.llm import (
    LLMError,
    LLMResponse,
    Message,
    StructuredOutputValidationError,
    ToolCall,
)
from repo_intelligence.planning import PlanValidationError
from repo_intelligence.tools import ToolResult

QUESTION = "Explain how the agents, tools, and MCP servers are connected."
CONTEXT = (
    "# Architecture Context\n\n- Contract: `architecture_context.v1`\n"
)
CONTEXT_RECORD = {
    "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/"
    "architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
    "snapshot_id": "main_2026-09-22_ef11dac",
}


# -- per-file test doubles (no conftest in this repo) ----------------------


class FakeLLMClient:
    """Queued responses (or Exceptions to raise); records requests.

    ``on_request`` optionally observes every request at call time —
    used to assert chronological trace state (spec §38 ordering).
    """

    def __init__(self, *responses, on_request=None) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"
        self._on_request = on_request

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if self._on_request is not None:
            self._on_request(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeDispatcher:
    """Records dispatches; canned results first, else generic success.

    ``on_dispatch`` optionally observes every dispatch at call time —
    used to prove the plan was already recorded (spec §38 ordering).
    """

    def __init__(self, *results, on_dispatch=None) -> None:
        self._queue = list(results)
        self.calls = []
        self._on_dispatch = on_dispatch

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._on_dispatch is not None:
            self._on_dispatch(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


# -- scripted responses ----------------------------------------------------


def _call(call_id, name="search_repository", arguments=None):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _plan_response(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _text(content):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _plan_payload(steps, objective=QUESTION):
    return {"objective": objective, "steps": steps}


class Finding(BaseModel):
    """M8.9 run contract used by the structured-final-step test."""

    finding: str
    evidence_paths: List[str]


TWO_STEP_PLAN = [
    {"id": "P1", "goal": "Identify relevant agent definitions"},
    {"id": "P2", "goal": "Produce an evidence-backed explanation"},
]


def two_step_script():
    return (
        _plan_response(_plan_payload(TWO_STEP_PLAN)),
        _tool_calls(_call("c1", "search_repository", {"query": "agent"})),
        _text("identified 3 agent definition files"),
        _tool_calls(_call("c2", "read_file", {"path": "agents/dag.md"})),
        _text("final evidence-backed answer"),
    )


def build_agent(responses, results=(), on_request=None, on_dispatch=None, **kwargs):
    client = FakeLLMClient(*responses, on_request=on_request)
    dispatcher = FakeDispatcher(*results, on_dispatch=on_dispatch)
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_tool_calls=kwargs.pop("max_tool_calls", 8)
        ),
        **kwargs,
    )
    return agent, client, dispatcher


def _plan_statuses(trace):
    return [
        (step["id"], step["status"])
        for step in (trace.plan or {}).get("steps", [])
    ]


# -- planning contract (spec §38 items 1–6) --------------------------------


def test_valid_planner_response_executes_the_whole_plan():
    agent, client, dispatcher = build_agent(two_step_script())
    result = agent.run_planned(QUESTION, CONTEXT, CONTEXT_RECORD)

    assert result.answer == "final evidence-backed answer"
    trace = agent.last_trace
    assert trace.plan["objective"] == QUESTION
    assert trace.plan["validation"] == {"status": "SUCCESS"}
    assert trace.plan["status"] == "completed"
    assert _plan_statuses(trace) == [("P1", "completed"), ("P2", "completed")]
    assert trace.plan["steps"][0]["result"] == "identified 3 agent definition files"
    assert trace.termination_reason == "FINAL_ANSWER"
    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
    ]
    # planning used the same client, no tools, before execution
    assert client.requests[0].tools == []
    assert client.requests[0].structured_output.name == "InvestigationPlan"


def test_planner_payload_missing_objective_fails_schema_validation():
    agent, client, dispatcher = build_agent(
        (_plan_response({"steps": [{"id": "P1", "goal": "g"}]}),)
    )
    with pytest.raises(StructuredOutputValidationError):
        agent.run_planned(QUESTION, CONTEXT, CONTEXT_RECORD)

    assert dispatcher.calls == []
    trace = agent.last_trace
    assert trace.termination_reason == "PLAN_VALIDATION_FAILED"
    assert trace.plan["validation"]["status"] == "FAILED"
    assert trace.plan["status"] == "invalidated"
    assert trace.iterations[0].structured_validation["status"] == "FAILED"


def test_empty_steps_fail_application_validation():
    agent, client, dispatcher = build_agent(
        (_plan_response(_plan_payload([])),)
    )
    with pytest.raises(PlanValidationError):
        agent.run_planned(QUESTION)

    assert dispatcher.calls == []
    trace = agent.last_trace
    assert trace.termination_reason == "PLAN_VALIDATION_FAILED"
    assert trace.plan["validation"]["status"] == "FAILED"
    assert trace.plan["validation"]["stage"] == "application"
    assert trace.plan["steps"] == []
    assert trace.plan["status"] == "invalidated"


def test_duplicate_step_ids_fail_application_validation():
    agent, client, dispatcher = build_agent(
        (
            _plan_response(
                _plan_payload(
                    [
                        {"id": "P1", "goal": "identify the agents"},
                        {"id": "P1", "goal": "map the tools"},
                    ]
                )
            ),
        )
    )
    with pytest.raises(PlanValidationError):
        agent.run_planned(QUESTION)
    assert dispatcher.calls == []
    assert agent.last_trace.termination_reason == "PLAN_VALIDATION_FAILED"
    assert "duplicate" in agent.last_trace.plan["validation"]["error"]


def test_malformed_structured_plan_fails_explicitly():
    agent, client, dispatcher = build_agent((_text("no plan for you"),))
    with pytest.raises(StructuredOutputValidationError):
        agent.run_planned(QUESTION)
    assert dispatcher.calls == []
    trace = agent.last_trace
    assert trace.termination_reason == "PLAN_VALIDATION_FAILED"
    assert trace.plan["validation"]["status"] == "FAILED"
    assert trace.plan["steps"] == []


def test_invalid_tool_level_plan_never_executes_repository_tools():
    agent, client, dispatcher = build_agent(
        (
            _plan_response(
                _plan_payload(
                    [
                        {
                            "id": "P1",
                            "goal": 'call search_repository(query="agent")',
                        }
                    ]
                )
            ),
        )
    )
    with pytest.raises(PlanValidationError):
        agent.run_planned(QUESTION)
    # the ONLY LLM call was the planning request; zero dispatches
    assert len(client.requests) == 1
    assert dispatcher.calls == []


# -- ordering (spec §38): plan creation before first tool execution -------


def test_plan_is_recorded_before_the_first_tool_execution():
    holder = {}
    first_dispatch_state = {}

    def on_request(request):
        # while planning: nothing has executed, nothing is recorded yet
        if len(holder["client"].requests) == 1:
            assert holder["dispatcher"].calls == []
            assert holder["agent"].last_trace.plan is None

    def on_dispatch(tool_call):
        if "plan" not in first_dispatch_state:
            first_dispatch_state["plan"] = copy.deepcopy(
                holder["agent"].last_trace.plan
            )

    agent, client, dispatcher = build_agent(
        two_step_script(), on_request=on_request, on_dispatch=on_dispatch
    )
    holder.update(agent=agent, client=client, dispatcher=dispatcher)
    agent.run_planned(QUESTION, CONTEXT, CONTEXT_RECORD)

    plan_at_first_dispatch = first_dispatch_state["plan"]
    assert plan_at_first_dispatch is not None
    assert plan_at_first_dispatch["validation"] == {"status": "SUCCESS"}
    # P1 already in progress; P2 still pending — chronological state
    # inside the ONE trace: plan recorded → tool execution
    assert [
        (s["id"], s["status"]) for s in plan_at_first_dispatch["steps"]
    ] == [("P1", "in_progress"), ("P2", "pending")]
    # architecture context recorded before execution too
    assert agent.last_trace.architecture_context == CONTEXT_RECORD


# -- sequential execution (spec §16/§38) -----------------------------------


def test_steps_execute_in_order_and_p2_waits_for_p1():
    observed = []

    def on_request(request):
        user_text = " ".join(
            message.content or ""
            for message in request.messages
            if message.role == "user"
        )
        observed.append(
            {
                "mentions_p2": "Plan step P2" in user_text,
                "plan": copy.deepcopy(holder["agent"].last_trace.plan),
            }
        )

    holder = {}
    agent, client, dispatcher = build_agent(
        two_step_script(), on_request=on_request
    )
    holder["agent"] = agent
    agent.run_planned(QUESTION)

    p2_requests = [index for index, item in enumerate(observed) if item["mentions_p2"]]
    assert p2_requests, "no request ever introduced plan step P2"
    first_p2 = p2_requests[0]
    # P1 reached its terminal status BEFORE any request mentioned P2
    plan_before_p2 = observed[first_p2]["plan"]
    assert [
        (s["id"], s["status"]) for s in plan_before_p2["steps"]
    ] == [("P1", "completed"), ("P2", "in_progress")]
    # and no earlier request mentioned P2 at all
    assert not any(item["mentions_p2"] for item in observed[:first_p2])

    trace = agent.last_trace
    # iterations are tagged and contiguous per step (P1 block, P2 block)
    assert [
        iteration.plan_step_id for iteration in trace.iterations
    ] == [None, "P1", "P1", "P2", "P2"]


# -- ReAct inside a step (spec §15/§38) ------------------------------------


def test_one_step_runs_the_existing_react_behavior():
    script = (
        _plan_response(
            _plan_payload([{"id": "P1", "goal": "Identify agent definitions"}])
        ),
        _tool_calls(_call("c1", "search_repository", {"query": "agent"})),
        _tool_calls(_call("c2", "read_file", {"path": "agents/dag.md"})),
        _text("agents: dag-agent, report-agent"),
    )
    agent, client, dispatcher = build_agent(script)
    result = agent.run_planned(QUESTION)

    assert result.answer == "agents: dag-agent, report-agent"
    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
    ]
    trace = agent.last_trace
    assert [iteration.plan_step_id for iteration in trace.iterations] == [
        None,
        "P1",
        "P1",
        "P1",
    ]
    # iteration 2: tool call executed and observed
    assert len(trace.iterations[1].tool_results) == 1
    assert trace.iterations[1].observations[0]["tool"] == "search_repository"
    # iteration 3's request carries the first observation (conversation
    # state preserved across the step's ReAct cycle)
    third_request_messages = trace.iterations[2].request["messages"]
    tool_messages = [
        m for m in third_request_messages if m.get("role") == "tool"
    ]
    assert tool_messages and tool_messages[0]["tool_call_id"] == "c1"
    assert len(trace.iterations[2].tool_results) == 1
    # iteration 4: the step's textual result closes the step
    assert trace.iterations[3].response_type == "FINAL_ANSWER"
    assert trace.iterations[3].final_answer == "agents: dag-agent, report-agent"
    assert trace.plan["status"] == "completed"
    assert trace.plan["steps"][0]["status"] == "completed"


# -- tool-call bound (spec §19/§38): ONE global budget ---------------------


def test_single_max_tool_calls_bound_controls_the_whole_plan():
    script = (
        _plan_response(_plan_payload(TWO_STEP_PLAN)),
        _tool_calls(_call("c1")),
        _tool_calls(_call("c2")),
        _text("never reached"),
    )
    agent, client, dispatcher = build_agent(script, max_tool_calls=1)

    with pytest.raises(AgentError) as excinfo:
        agent.run_planned(QUESTION)
    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"

    # exactly one execution — the second call was blocked, not run
    assert len(dispatcher.calls) == 1
    trace = agent.last_trace
    assert trace.termination_reason == "EXECUTION_LIMIT_REACHED"
    summary = trace.summary
    assert summary.max_tool_calls == 1
    assert summary.tool_calls_executed == 1
    assert summary.tool_calls_blocked == 1
    # the interrupted step/plan are honestly blocked — never success
    assert trace.plan["status"] == "blocked"
    assert _plan_statuses(trace) == [("P1", "blocked"), ("P2", "pending")]


# -- failure paths (spec §29/§38): none silently becomes success -----------


def test_planner_llm_failure_records_failure_and_runs_no_tool():
    agent, client, dispatcher = build_agent(
        (LLMError("provider unavailable"),)
    )
    with pytest.raises(LLMError):
        agent.run_planned(QUESTION, CONTEXT, CONTEXT_RECORD)
    assert dispatcher.calls == []
    trace = agent.last_trace
    assert trace.termination_reason == "LLM_ERROR"
    assert trace.plan is None
    assert trace.architecture_context == CONTEXT_RECORD
    assert trace.iterations[0].response_type == "ERROR"


def test_step_llm_failure_marks_step_and_plan_failed():
    script = (
        _plan_response(_plan_payload(TWO_STEP_PLAN)),
        _tool_calls(_call("c1")),
        LLMError("provider unavailable mid-step"),
    )
    agent, client, dispatcher = build_agent(script)
    with pytest.raises(LLMError):
        agent.run_planned(QUESTION)
    assert dispatcher.calls and len(dispatcher.calls) == 1
    trace = agent.last_trace
    assert trace.termination_reason == "LLM_ERROR"
    assert trace.plan["status"] == "failed"
    assert _plan_statuses(trace) == [("P1", "failed"), ("P2", "pending")]


def test_empty_step_result_invalidates_the_plan_without_replanning():
    script = (
        _plan_response(_plan_payload(TWO_STEP_PLAN)),
        _tool_calls(_call("c1")),
        _text("   "),
    )
    agent, client, dispatcher = build_agent(script)
    result = agent.run_planned(QUESTION)

    # a result is returned, but it records the invalidation — it is
    # NOT a successful investigation
    assert result.answer == ""
    trace = agent.last_trace
    assert trace.termination_reason == "PLAN_INVALIDATED"
    assert trace.plan["status"] == "invalidated"
    assert _plan_statuses(trace) == [("P1", "invalidated"), ("P2", "pending")]
    # exactly one planning request + two execution requests — the
    # planner was never called again (no replanning, spec §17)
    assert len(client.requests) == 3


def test_structured_final_step_uses_the_m89_contract_only_at_the_end():
    payload = {"finding": "agents use tools via mcp servers", "evidence_paths": ["a"]}
    script = (
        _plan_response(_plan_payload(TWO_STEP_PLAN)),
        _text("step one complete"),
        LLMResponse(
            message=Message(role="assistant", content=json.dumps(payload)),
            finish_reason="stop",
            structured_output=payload,
        ),
    )
    agent, client, dispatcher = build_agent(
        script, structured_output_model=Finding
    )
    result = agent.run_planned(QUESTION)

    assert result.structured_output.finding == payload["finding"]
    assert client.requests[0].structured_output.name == "InvestigationPlan"
    # P1 (intermediate): free-form step result — no run contract
    assert client.requests[1].structured_output is None
    # P2 (final step): the M8.9 run contract travels again
    assert client.requests[2].structured_output.name == "Finding"
    trace = agent.last_trace
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    assert trace.iterations[-1].structured_validation["status"] == "SUCCESS"
    assert trace.plan["status"] == "completed"


def test_non_plan_runs_never_gain_plan_fields():
    script = (_tool_calls(_call("c1")), _text("plain answer"))
    agent, client, dispatcher = build_agent(script)
    result = agent.run_result("plain question")
    assert result.answer == "plain answer"
    trace = agent.last_trace
    assert trace.plan is None
    assert trace.architecture_context is None
    assert all(
        iteration.plan_step_id is None for iteration in trace.iterations
    )


def test_run_planned_validates_input_before_any_llm_or_tool_call():
    agent, client, dispatcher = build_agent(())
    with pytest.raises(AgentError):
        agent.run_planned("   ")
    assert client.requests == []
    assert dispatcher.calls == []
