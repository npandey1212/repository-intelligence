"""M10 — end-to-end planned investigation with a scripted LLM (spec §39).

One deterministic test of the complete learning path, doubles only:

    OBJECTIVE
      + M6 ARCHITECTURE CONTEXT (passed directly to the planner)
      ↓
    LLM PLANNER (one request, no tools, M8.9 structured output)
      ↓
    STRUCTURED PLAN → deterministic validation
      ↓
    PLAN RECORDED in the existing AgentTrace (before any tool)
      ↓
    P1 → existing ReAct → ToolDispatcher → repository evidence
      ↓
    P2 → existing ReAct → ToolDispatcher → repository evidence
      ↓
    P3 → existing ReAct → final answer
      ↓
    ONE AgentRunResult + ONE AgentTrace (JSON canonical + Markdown
    projection both exposing the plan and plan-step correlation)

The real-model version of this path is the human-runnable
``agentrun_m10_plan_demo.py``; this test never touches the network.
"""

from __future__ import annotations

import copy
import json

from repo_intelligence.agent import (
    Agent,
    AgentExecutionConfig,
    render_markdown,
)
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

OBJECTIVE = "Explain how the agents, tools, and MCP servers are connected."
CONTEXT = (
    "# Architecture Context\n\n"
    "- Contract: `architecture_context.v1`\n\n"
    "## Architecture\n\n"
    "- agents: dag-agent, report-agent\n"
    "- tools: search_repository, read_file\n"
    "- mcp servers: configured in .mcp.json\n"
)
CONTEXT_RECORD = {
    "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/"
    "architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
    "snapshot_id": "main_2026-09-22_ef11dac",
}

PLAN_PAYLOAD = {
    "objective": OBJECTIVE,
    "steps": [
        {"id": "P1", "goal": "Identify the relevant agent definitions"},
        {"id": "P2", "goal": "Map agents to the tools or capabilities they use"},
        {"id": "P3", "goal": "Produce an evidence-backed explanation"},
    ],
}


# -- per-file test doubles (no conftest in this repo) ----------------------


class FakeLLMClient:
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
        return self._queue.pop(0)


class FakeDispatcher:
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


def _call(call_id, name, arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _plan(payload):
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


def _tool(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _result(tool_name, data, path):
    return ToolResult(
        success=True,
        tool=tool_name,
        data=data,
        source={
            "path": path,
            "commit_sha": CONTEXT_RECORD["commit_sha"],
            "snapshot_id": CONTEXT_RECORD["snapshot_id"],
        },
    )


SCRIPT = (
    _plan(PLAN_PAYLOAD),
    # P1: ReAct — search, then step result
    _tool(_call("c1", "search_repository", {"query": "agent"})),
    _text("agent definitions live in .claude/agents/"),
    # P2: ReAct — read, then step result
    _tool(_call("c2", "read_file", {"path": ".claude/agents/dag-agent.md"})),
    _text("dag-agent uses search_repository and read_file"),
    # P3: ReAct — read MCP config, then the final answer
    _tool(_call("c3", "read_file", {"path": ".mcp.json"})),
    _text(
        "The dag-agent invokes search_repository/read_file, which are "
        "exposed to MCP servers configured in .mcp.json."
    ),
)

RESULTS = (
    _result("search_repository", {"matches": ["dag-agent"]}, "agents/dag.md"),
    _result("read_file", {"content": "frontmatter"}, ".claude/agents/dag-agent.md"),
    _result("read_file", {"content": "{}"}, ".mcp.json"),
)


def run_scenario(max_tool_calls=6):
    snapshots = []
    holder = {}

    def on_request(request):
        # chronological proof: no dispatch before the plan exists, and
        # plan-step ordering holds at every request boundary
        snapshots.append(
            {
                "requests_seen": len(holder["client"].requests),
                "dispatches_so_far": len(holder["dispatcher"].calls),
                "plan": copy.deepcopy(holder["agent"].last_trace.plan),
            }
        )

    def on_dispatch(tool_call):
        snapshots.append(
            {
                "dispatch": tool_call.id,
                "plan": copy.deepcopy(holder["agent"].last_trace.plan),
            }
        )

    client = FakeLLMClient(*SCRIPT, on_request=on_request)
    dispatcher = FakeDispatcher(*RESULTS, on_dispatch=on_dispatch)
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=max_tool_calls),
    )
    holder.update(agent=agent, client=client, dispatcher=dispatcher)
    result = agent.run_planned(OBJECTIVE, CONTEXT, CONTEXT_RECORD)
    return agent, client, dispatcher, result, snapshots


# -- the complete path (spec §39) ------------------------------------------


def test_complete_planned_investigation_end_to_end():
    agent, client, dispatcher, result, snapshots = run_scenario()
    trace = agent.last_trace

    # -- ONE result, ONE trace, honest answer ------------------------
    assert result.answer.startswith("The dag-agent invokes")
    assert result.claims == ()
    assert result.structured_output is None
    assert trace is agent.last_trace
    assert len(trace.iterations) == 7  # 1 planning + 3 steps x 2

    # -- planner input: architecture context passed directly (§8) ----
    planning_request = client.requests[0]
    assert planning_request.tools == []
    assert planning_request.structured_output.name == "InvestigationPlan"
    assert CONTEXT in planning_request.messages[1].content
    assert trace.architecture_context == CONTEXT_RECORD

    # -- plan recorded before the first tool execution ---------------
    first_dispatch = next(s for s in snapshots if "dispatch" in s)
    assert first_dispatch["plan"] is not None
    assert first_dispatch["plan"]["validation"] == {"status": "SUCCESS"}
    assert [
        (s["id"], s["status"]) for s in first_dispatch["plan"]["steps"]
    ] == [("P1", "in_progress"), ("P2", "pending"), ("P3", "pending")]
    # and no dispatch happened while planning
    planning_snap = snapshots[0]
    assert planning_snap["dispatches_so_far"] == 0

    # -- sequential lifecycle, all completed --------------------------
    assert trace.plan["status"] == "completed"
    assert [(s["id"], s["status"]) for s in trace.plan["steps"]] == [
        ("P1", "completed"),
        ("P2", "completed"),
        ("P3", "completed"),
    ]
    assert trace.plan["steps"][0]["result"] == (
        "agent definitions live in .claude/agents/"
    )

    # -- existing ReAct executes the actual work, step by step -------
    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
        "read_file",
    ]
    assert [iteration.plan_step_id for iteration in trace.iterations] == [
        None,
        "P1",
        "P1",
        "P2",
        "P2",
        "P3",
        "P3",
    ]
    # every step's iteration pair: tool executed, then step result
    for index in (1, 3, 5):
        assert len(trace.iterations[index].tool_results) == 1
        assert trace.iterations[index].observations
        following = trace.iterations[index + 1]
        assert following.response_type == "FINAL_ANSWER"
    # repository provenance observed from actual tool results
    assert trace.repository is not None
    assert trace.repository["snapshot_id"] == CONTEXT_RECORD["snapshot_id"]

    # -- one authoritative budget -------------------------------------
    summary = trace.summary
    assert summary.max_tool_calls == 6
    assert summary.tool_calls_executed == 3
    assert summary.tool_calls_blocked == 0
    assert summary.termination_reason == "FINAL_ANSWER"

    # chronological snapshots: once a step starts, every earlier step
    # is already completed (strict sequential execution, no jumping)
    for snapshot in snapshots:
        plan = snapshot["plan"]
        if plan is None:
            continue
        statuses = [step["status"] for step in plan["steps"]]
        for position, status in enumerate(statuses):
            if status != "pending":
                assert all(
                    earlier == "completed" for earlier in statuses[:position]
                ), statuses


# -- projections (spec §23/§40/AC 5, 12) -----------------------------------


def test_final_trace_json_and_markdown_both_expose_the_plan():
    agent, client, dispatcher, result, _ = run_scenario()
    trace = agent.last_trace

    payload = json.loads(trace.to_json())
    assert payload["plan"]["objective"] == OBJECTIVE
    assert [step["id"] for step in payload["plan"]["steps"]] == [
        "P1",
        "P2",
        "P3",
    ]
    assert payload["architecture_context"] == CONTEXT_RECORD
    assert [
        iteration.get("plan_step_id") for iteration in payload["iterations"]
    ] == [None, "P1", "P1", "P2", "P2", "P3", "P3"]
    # existing ReAct stages remain on every execution iteration
    for iteration in payload["iterations"][1:]:
        for stage in ("request", "llm_response", "tool_results", "observations"):
            assert stage in iteration

    markdown = render_markdown(trace)
    assert markdown.index("## Plan") < markdown.index("## Iteration 1")
    assert "### Plan Steps" in markdown
    assert "**completed**" in markdown
    assert CONTEXT_RECORD["artifact"] in markdown
    assert "Plan step: `P2`" in markdown
    assert "### Tool Executions" in markdown
    assert "## Run Summary" in markdown

    # no hidden reasoning in either projection (spec §24)
    for projection in (trace.to_json(), markdown):
        for banned in BANNED_REASONING_FIELDS:
            assert '"{}"'.format(banned) not in projection
