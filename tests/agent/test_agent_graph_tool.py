"""M11 — Agent integration for query_engineering_graph (spec §20).

Offline (fake LLM + controlled artifact fixtures, per-file doubles — no
conftest in this repo):

* the 4th tool is described to the LLM in every Agent request;
* agent sources keep the import boundary (no tools/graph/engineering
  imports — descriptions carry their contracts as literals, pinned to
  the real contracts by tests);
* a returned ToolCall executes through the real application dispatcher
  (the real graph tool over a controlled run) and its ToolResult comes
  back as an observation;
* failures are observed, not fatal;
* ``run_planned`` keeps the M10 semantics: exactly one planner call,
  the plan exists before the first tool, and the graph tool executes
  with a ``plan_step_id`` correlation in the trace.

No network, no API key.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import repo_intelligence.artifacts as artifacts
from repo_intelligence.agent import (
    AGENT_TOOL_DEFINITIONS,
    Agent,
    AgentExecutionConfig,
)
from repo_intelligence.graph import (
    build_engineering_graph_document,
    write_engineering_graph_document,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tool_dispatcher import default_tool_dispatcher

QUESTION = "How are the agents connected in this repository?"
ANSWER = "The agents are wired through the engineering graph."
CONTEXT = (
    "# Architecture Context\n\n"
    "- Contract: `architecture_context.v1`\n\n"
    "## Architecture\n\n"
    "- agents: alpha-agent, beta-agent\n"
)
CONTEXT_RECORD = {
    "artifact": "artifacts/run_fixture_graph/architecture_context/"
    "architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "b" * 40,
    "snapshot_id": "run_fixture_graph",
}
OBJECTIVE = "Explain how the agents are connected."

AGENT_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "agent"
)
# The agent package may describe tools but never import the tool /
# graph / engineering layers (enforced here for all three layers).
LAYER_IMPORT_RE = re.compile(
    r"(?:from|import)\s+"
    r"(?:repo_intelligence\.(?:tools|graph|engineering)|"
    r"\.{1,2}(?:tools|graph|engineering))\b"
)

EXPECTED_TOOL_NAMES = [
    "get_architecture_context",
    "search_repository",
    "read_file",
    "query_engineering_graph",
]

RUN_ID = "run_fixture_graph"
A = "artifact:directory:pkg"
B = "artifact:file:pkg/mod.py"


# -- per-file doubles ------------------------------------------------------


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


def _text(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name, arguments=None, call_id="call_1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _plan(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _observation(request, index=-1):
    tool_messages = [
        message for message in request.messages if message.role == "tool"
    ]
    return json.loads(tool_messages[index].content)


def _graph_arguments(**overrides):
    arguments = {"start_node_id": A, "direction": "out", "max_depth": 1}
    arguments.update(overrides)
    return arguments


# -- controlled run fixture (facts + architecture + engineering) -----------


def _write_graph_run(tmp_path: Path, monkeypatch):
    """Minimal pinned-run layout under tmp_path; only the artifacts root
    is redirected — resolution (discover run → persisted graph
    artifact) is the production path."""
    root = tmp_path / "artifacts"
    run_dir = root / RUN_ID
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir()
    (run_dir / "engineering").mkdir()
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "engineering" / "engineering_model.json").write_text(
        json.dumps(_ENGINEERING_MODEL), encoding="utf-8"
    )
    write_engineering_graph_document(
        build_engineering_graph_document(
            _ENGINEERING_MODEL, snapshot_id=RUN_ID
        ),
        run_dir / "engineering" / "engineering_graph.json",
    )
    monkeypatch.setattr(artifacts, "default_artifacts_root", lambda: root)
    return run_dir


_ENGINEERING_MODEL = {
    "schema_version": "repository_knowledge_model.v1",
    "repository": {
        "name": "sample-repo-alpha",
        "url": "https://example.invalid/sample.git",
        "commit_sha": "b" * 40,
    },
    "engineering": {
        "nodes": [
            {
                "id": A,
                "kind": "artifact",
                "subkind": "directory",
                "classification": "observed",
                "evidence": [
                    {"path": "pkg", "artifact_type": "directory", "commit": "b" * 40}
                ],
                "attributes": {},
            },
            {
                "id": B,
                "kind": "artifact",
                "subkind": "file",
                "classification": "observed",
                "evidence": [
                    {
                        "path": "pkg/mod.py",
                        "artifact_type": "python",
                        "commit": "b" * 40,
                    }
                ],
                "attributes": {},
            },
        ],
        "relationships": [
            {
                "id": "rel:CONTAINS:{}:{}".format(A, B),
                "kind": "CONTAINS",
                "source": A,
                "target": B,
                "classification": "derived",
                "evidence": [
                    {"path": "pkg", "artifact_type": "directory", "commit": "b" * 40}
                ],
            }
        ],
        "unresolved": [],
        "metadata": {},
    },
}


# ---------------------------------------------------------------------------
# Definitions reach the LLM
# ---------------------------------------------------------------------------


def test_agent_request_describes_the_graph_tool():
    fake = FakeLLMClient(_text())
    Agent(llm_client=fake).run(QUESTION)

    names = [tool.name for tool in fake.requests[0].tools]
    assert names == EXPECTED_TOOL_NAMES
    assert names[-1] == "query_engineering_graph"
    assert names == [tool.name for tool in AGENT_TOOL_DEFINITIONS]


def test_graph_tool_definition_advertises_both_modes():
    definition = AGENT_TOOL_DEFINITIONS[-1]
    assert definition.name == "query_engineering_graph"
    schema = definition.input_schema
    assert "required" not in schema  # {} → node-kind summary
    assert set(schema["properties"]) == {
        "start_node_id",
        "direction",
        "relationship_kinds",
        "node_kinds",
        "max_depth",
    }


# ---------------------------------------------------------------------------
# Import boundary (agent describes, never imports)
# ---------------------------------------------------------------------------


def test_agent_sources_import_no_tools_graph_or_engineering_layers():
    for path in sorted(AGENT_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert LAYER_IMPORT_RE.search(text) is None, path


# ---------------------------------------------------------------------------
# Real dispatcher executes the graph tool (ReAct loop, M8.4)
# ---------------------------------------------------------------------------


def test_graph_tool_call_executes_and_is_observed(tmp_path, monkeypatch):
    _write_graph_run(tmp_path, monkeypatch)
    fake = FakeLLMClient(
        _tool(_call("query_engineering_graph", _graph_arguments())),
        _text(),
    )
    dispatcher = default_tool_dispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert len(fake.requests) == 2
    observation = _observation(fake.requests[1])
    assert observation["success"] is True
    assert observation["tool"] == "query_engineering_graph"
    assert observation["data"]["mode"] == "traversal"
    assert [node["id"] for node in observation["data"]["nodes"]] == [A, B]
    assert [rel["kind"] for rel in observation["data"]["relationships"]] == [
        "CONTAINS"
    ]
    assert observation["source"]["snapshot_id"] == RUN_ID
    assert observation["error"] is None


def test_graph_tool_summary_call_executes(tmp_path, monkeypatch):
    _write_graph_run(tmp_path, monkeypatch)
    fake = FakeLLMClient(
        _tool(_call("query_engineering_graph", {})),
        _text(),
    )

    Agent(
        llm_client=fake, tool_dispatcher=default_tool_dispatcher()
    ).run(QUESTION)

    observation = _observation(fake.requests[1])
    assert observation["success"] is True
    assert observation["data"]["mode"] == "summary"
    assert observation["data"]["counts"]["nodes"] == 2


def test_graph_tool_failure_is_observed_not_fatal(tmp_path, monkeypatch):
    _write_graph_run(tmp_path, monkeypatch)
    fake = FakeLLMClient(
        _tool(_call("query_engineering_graph", {"start_node_id": "nope"})),
        _text(),
    )

    outcome = Agent(
        llm_client=fake, tool_dispatcher=default_tool_dispatcher()
    ).run(QUESTION)

    assert outcome == ANSWER
    observation = _observation(fake.requests[1])
    assert observation["success"] is False
    assert observation["tool"] == "query_engineering_graph"
    assert observation["error"]["code"] == "NODE_NOT_FOUND"


def test_registry_and_definitions_stay_in_sync(tmp_path, monkeypatch):
    _write_graph_run(tmp_path, monkeypatch)
    dispatcher = default_tool_dispatcher()

    assert tuple(dispatcher.tool_names) == tuple(sorted(EXPECTED_TOOL_NAMES))
    assert set(dispatcher.tool_names) == {
        tool.name for tool in AGENT_TOOL_DEFINITIONS
    }


# ---------------------------------------------------------------------------
# M10 semantics unchanged with the 4th tool (run_planned)
# ---------------------------------------------------------------------------


def test_run_planned_executes_graph_tool_with_plan_correlation(
    tmp_path, monkeypatch
):
    _write_graph_run(tmp_path, monkeypatch)
    plan_payload = {
        "objective": OBJECTIVE,
        "steps": [{"id": "P1", "goal": "Inspect the engineering graph"}],
    }
    fake = FakeLLMClient(
        _plan(plan_payload),
        _tool(_call("query_engineering_graph", _graph_arguments(), call_id="c1")),
        _text("pkg/mod.py lives under pkg/."),
    )
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=default_tool_dispatcher(),
        execution_config=AgentExecutionConfig(max_tool_calls=6),
    )

    result = agent.run_planned(OBJECTIVE, CONTEXT, CONTEXT_RECORD)
    trace = agent.last_trace

    # exactly one planner request — no tools, plan before any tool
    planning_requests = [r for r in fake.requests if not r.tools]
    assert len(planning_requests) == 1
    assert fake.requests[0].tools == []
    assert all(request.tools for request in fake.requests[1:])

    # the graph tool executed once, inside the plan step
    assert result.answer == "pkg/mod.py lives under pkg/."
    assert trace.plan["status"] == "completed"
    assert [(s["id"], s["status"]) for s in trace.plan["steps"]] == [
        ("P1", "completed")
    ]
    assert [iteration.plan_step_id for iteration in trace.iterations] == [
        None,
        "P1",
        "P1",
    ]
    planning_iteration = trace.iterations[0]
    assert planning_iteration.tool_results == []  # plan precedes any tool
    tool_iteration = trace.iterations[1]
    assert len(tool_iteration.tool_results) == 1
    assert tool_iteration.tool_results[0]["tool"] == "query_engineering_graph"
    assert tool_iteration.tool_results[0]["success"] is True

    # the canonical trace projection exposes the executed tool
    payload = json.loads(trace.to_json())
    executed = [
        result_entry.get("tool")
        for iteration in payload["iterations"]
        for result_entry in iteration.get("tool_results", [])
    ]
    assert executed == ["query_engineering_graph"]
    assert payload["plan"]["objective"] == OBJECTIVE

    # one authoritative budget: one tool call executed, none blocked
    summary = trace.summary
    assert summary.max_tool_calls == 6
    assert summary.tool_calls_executed == 1
    assert summary.tool_calls_blocked == 0
    assert summary.termination_reason == "FINAL_ANSWER"
