"""Context-budget preflight at every planning entry point.

The real overflow (a ~128,882-token request rejected by the provider
with ``context_length_exceeded``) proved that a request could leave
this process without any local preflight.  This file pins the boundary
at the CALL sites rather than only at the helper:

* the production planning path ``_build_plan`` / ``run_repository_query``
* the executable LangGraph planning module ``langgraph.planner``
* the tool-result lifecycle that feeds the NEXT model request
* the diagnostics that must tell a local rejection from a provider
  failure, and a rejection from a silently truncated request

Offline — no network, no live provider.  Every client here is scripted
and every "the provider was not called" claim is made on a request
list the fake client records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest

import repo_intelligence.langgraph.planner as lg_planner
import repo_intelligence.llm.openai_client as openai_client_module
from repo_intelligence import repository_query as rq
from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
    trace_sink as trace_sink_module,
)
from repo_intelligence.agent.request_budget import (
    DISPOSITION_ACCEPTED,
    DISPOSITION_REJECTED,
    DISPOSITION_REDUCED,
    RequestBudgetExceededError,
    RequestBudgetPolicy,
    build_request_budget_record,
    enforce_request_budget,
    enforce_request_budget_or_raise,
)
from repo_intelligence.agents import visualization_artifacts as viz_module
from repo_intelligence.context_engine import engine as context_engine_module
from repo_intelligence.llm import (
    LLMError,
    LLMRequest,
    Message,
    spec_for_model,
)
from repo_intelligence.planning.models import InvestigationPlan
from repo_intelligence.planning.planner import PLANNING_SYSTEM_INSTRUCTION
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.graph_projection import (
    DEFAULT_MODEL_GRAPH_MAX_BYTES,
    model_facing_graph_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARCH_MCP_ROOT = FIXTURES / "arch_mcp"
GRAPH_RUNNER = (
    PROJECT_ROOT
    / "src"
    / "repo_intelligence"
    / "orchestration"
    / "langgraph_runner.py"
)

QUESTION = "How do the agents, tools, and MCP servers interact?"

# Larger than the default input budget (95,232 estimated tokens =
# 380,928 characters).  It is expressed as the only content a planner
# request can grow in — the objective — because the planner request
# carries no tool definitions and no chat history.
OVERSIZED_QUESTION = "x" * 400_851

# The graph observation below is far larger than this request budget;
# the compact projection is not.
TIGHT_POLICY_TOKENS = 12_000


# -- fixtures and doubles --------------------------------------------------


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every root a run can write to, pointed at ``tmp_path``."""
    monkeypatch.setattr(
        viz_module,
        "default_artifacts_root",
        lambda: tmp_path / "artifacts",
    )
    monkeypatch.setattr(
        trace_sink_module,
        "default_trace_root",
        lambda: tmp_path / "runtime" / "agent_runs",
    )
    monkeypatch.setattr(
        context_engine_module,
        "default_context_root",
        lambda: tmp_path / "runtime" / "context",
    )
    monkeypatch.setattr(
        rq,
        "default_query_db_path",
        lambda: tmp_path / "runtime" / "agent_memory" / "query.db",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


class _RecordingPlanner:
    """Scripted planner double.

    Records every request it is asked to send.  With no scripted
    response it fails loudly, so a test that expects a locally
    REJECTED request reports ``provider was called`` instead of
    passing on an empty queue.
    """

    model = "scripted-planner"

    def __init__(self, response: Any = None) -> None:
        self.requests: List[LLMRequest] = []
        self._response = response

    def complete(self, request: LLMRequest) -> Any:
        self.requests.append(request)
        if self._response is None:
            raise AssertionError("provider was called for a rejected request")
        return self._response


class _FailingPlanner:
    """Planner double that fails the way a PROVIDER call fails."""

    model = "failing-planner"

    def __init__(self, error: BaseException) -> None:
        self.requests: List[LLMRequest] = []
        self._error = error

    def complete(self, request: LLMRequest) -> Any:
        self.requests.append(request)
        raise self._error


def _spy_planner(monkeypatch) -> List[LLMRequest]:
    """Keep the REAL offline model set but expose what its planner sent.

    ``run_repository_query`` builds the model set internally; the spy
    only records the requests its planner submits, so a test can prove
    none was submitted or that the one that was is unchanged.
    """
    captured: List[LLMRequest] = []
    real_build = rq.build_offline_models

    def build(question, repository, commit, entry_target):
        models = real_build(question, repository, commit, entry_target)
        planner = models["planner"]
        original = planner.complete

        def complete(request):
            captured.append(request)
            return original(request)

        planner.complete = complete
        return models

    monkeypatch.setattr(rq, "build_offline_models", build)
    return captured


def _request_for_observation(body: str) -> LLMRequest:
    """One loop turn's request: instruction + objective + an
    observation, with the real ``InvestigationPlan`` response contract
    so the schema overhead is measured too."""
    return LLMRequest(
        messages=[
            Message(role="system", content=PLANNING_SYSTEM_INSTRUCTION),
            Message(role="user", content=QUESTION),
            Message(role="tool", content=body),
        ],
        tools=[],
        structured_output=spec_for_model(InvestigationPlan),
    )


def _evidence(path: str, line: int) -> dict:
    return {
        "path": path,
        "artifact_type": "file",
        "commit": "c" * 40,
        "line_start": line,
        "line_end": line + 40,
    }


def _large_graph_result() -> ToolResult:
    """A canonical traversal payload well past the request budget."""
    nodes = []
    evidence_rows = []
    for index in range(70):
        entries = [
            _evidence(
                "src/module_{0}/file_{1}.py".format(index, item),
                item * 7,
            )
            for item in range(3)
        ]
        node_id = "artifact:file:module_{0}".format(index)
        nodes.append(
            {
                "id": node_id,
                "depth": index % 4,
                "kind": "artifact",
                "subkind": "file",
                "classification": "observed",
                "attributes": {
                    "path": "src/module_{0}/file_0.py".format(index),
                    "qualified_name": "module_{0}.handler".format(index),
                    "fact_type": "file",
                    "detail": "repository file detail " * 8,
                },
                "evidence": entries,
            }
        )
        for position, row in enumerate(entries):
            annotated = dict(row)
            annotated["origin"] = "node:{0}".format(node_id)
            annotated["index"] = position
            evidence_rows.append(annotated)
    start = dict(nodes[0])
    start["depth"] = 0
    relationships = [
        {
            "id": "rel:IMPORTS:{0}:{1}".format(
                nodes[index % len(nodes)]["id"],
                nodes[(index * 7 + 3) % len(nodes)]["id"],
            ),
            "kind": "IMPORTS",
            "source": nodes[index % len(nodes)]["id"],
            "target": nodes[(index * 7 + 3) % len(nodes)]["id"],
            "classification": "derived",
            "depth": 1,
            "evidence": [dict(nodes[index % len(nodes)]["evidence"][0])],
        }
        for index in range(90)
    ]
    return ToolResult(
        success=True,
        tool="query_engineering_graph",
        data={
            "mode": "traversal",
            "query": {
                "start_node_id": start["id"],
                "direction": "both",
                "max_depth": 3,
            },
            "start_node": start,
            "nodes": nodes,
            "relationships": relationships,
            "evidence": evidence_rows,
            "kinds": [],
            "counts": {
                "nodes": len(nodes),
                "relationships": len(relationships),
                "evidence": len(evidence_rows),
                "unresolved": 3,
            },
            "truncated": False,
        },
    )


# -- 1 / 3 / 4: the production planner is behind the preflight -------------


def test_oversized_planner_request_is_rejected_before_the_provider():
    planner = _RecordingPlanner()

    with pytest.raises(RequestBudgetExceededError) as excinfo:
        rq._build_plan(planner, OVERSIZED_QUESTION)

    assert planner.requests == []  # zero provider calls
    record = excinfo.value.record
    assert record["disposition"] == DISPOSITION_REJECTED
    assert record["within_budget"] is False
    assert (
        record["totals"]["total_tokens_est"]
        > record["budget"]["effective_input_budget_tokens"]
    )
    assert record["rejection_reason"] == "protected_request_exceeds_budget"
    assert record["estimation"]["is_estimate"] is True


def test_planner_within_budget_reaches_the_provider_and_returns_a_plan():
    expected = rq.build_planning_request(QUESTION)
    planner = _RecordingPlanner(
        rq.structured_response(rq.offline_plan_payload(QUESTION))
    )

    plan = rq._build_plan(planner, QUESTION)

    assert len(planner.requests) == 1
    sent = planner.requests[0]
    # The preflight changed nothing a within-budget planner may see.
    assert sent == expected
    assert sent.structured_output.schema == expected.structured_output.schema
    assert [message.content for message in sent.messages] == [
        message.content for message in expected.messages
    ]
    assert plan.objective == QUESTION
    assert [step.id for step in plan.steps] == ["S1", "S2", "S3"]


def test_production_planning_never_reaches_the_provider_when_over_budget(
    sandbox, monkeypatch
):
    sent = _spy_planner(monkeypatch)

    result = rq.run_repository_query(
        OVERSIZED_QUESTION,
        str(ARCH_MCP_ROOT),
        diagram=False,
        artifacts=False,
    )

    assert sent == []  # zero provider calls end to end
    assert result.status == rq.RUN_PLAN_FAILED
    assert result.termination_reason == "PLAN_FAILED"
    assert rq.exit_code(result) == rq.EXIT_INVESTIGATION_FAILED
    assert result.failures
    assert result.failures[0].startswith("RequestBudgetExceededError:")
    assert "LOCAL context-budget preflight" in result.failures[0]


def test_production_planning_request_is_unchanged_by_the_preflight(
    sandbox, monkeypatch
):
    sent = _spy_planner(monkeypatch)

    result = rq.run_repository_query(QUESTION, str(ARCH_MCP_ROOT))

    assert result.status == rq.COMPLETED
    assert result.termination_reason == "SUCCESS"
    assert result.failures == []
    assert rq.exit_code(result) == rq.EXIT_SUCCESS
    assert len(sent) == 1
    # The ordinary planning request is byte-identical to the request
    # the application built before this gate existed.
    assert sent[0] == rq.build_planning_request(QUESTION)


# -- 8 / 10: diagnostics and no silent data loss ---------------------------


def test_local_rejection_is_distinguishable_from_a_provider_failure(
    sandbox, monkeypatch
):
    rejected = rq.run_repository_query(
        OVERSIZED_QUESTION,
        str(ARCH_MCP_ROOT),
        diagram=False,
        artifacts=False,
    )
    assert rejected.termination_reason == "PLAN_FAILED"
    assert rejected.failures[0].startswith("RequestBudgetExceededError:")
    assert "LOCAL context-budget preflight" in rejected.failures[0]

    # The same entry point, this time failing INSIDE the provider call.
    provider_error = LLMError(
        "400: This model's maximum context length is 128000 tokens. "
        "However, your messages resulted in 128882 tokens."
    )

    def failing_models(question, repository, commit, entry_target):
        return {
            key: _FailingPlanner(provider_error)
            for key in ("arch", "flow", "viz", "planner")
        }

    monkeypatch.setattr(rq, "build_offline_models", failing_models)
    failed = rq.run_repository_query(
        QUESTION, str(ARCH_MCP_ROOT), diagram=False, artifacts=False
    )
    assert failed.termination_reason == "PLAN_FAILED"
    assert failed.failures[0].startswith("LLMError:")
    assert "LOCAL context-budget preflight" not in failed.failures[0]

    # Same status, two failures a reader can tell apart by origin.
    assert rejected.failures[0] != failed.failures[0]


def test_rejection_preserves_the_protected_planner_content_unclipped():
    request = rq.build_planning_request(OVERSIZED_QUESTION)

    with pytest.raises(RequestBudgetExceededError) as excinfo:
        enforce_request_budget_or_raise(request, subject="planner request")

    record = excinfo.value.record
    assert record["rejection_reason"] == "protected_request_exceeds_budget"
    # Nothing was clipped: there is no reducible observation, and the
    # instruction / objective were never rewritten to fit.
    assert record["reduction"]["tool_messages_clipped"] == 0
    assert record["messages"][1]["content_characters"] == len(
        OVERSIZED_QUESTION
    )
    assert request.messages[0].content == PLANNING_SYSTEM_INSTRUCTION
    assert request.messages[1].content == OVERSIZED_QUESTION
    assert "truncated" not in request.messages[1].content


# -- 6: the tool-result lifecycle bounds the next model request ------------


def test_large_tool_result_is_bounded_before_the_next_model_request():
    raw = _large_graph_result()
    raw_body = json.dumps(raw.data)
    policy = RequestBudgetPolicy(max_input_tokens=TIGHT_POLICY_TOKENS)
    assert len(raw_body) > policy.effective_input_budget_characters

    projected = model_facing_graph_result(raw)
    projected_body = json.dumps(projected.data)
    assert len(projected_body) <= DEFAULT_MODEL_GRAPH_MAX_BYTES
    assert len(projected_body) < len(raw_body)

    # The projected observation arrives at the preflight already small
    # enough: accepted, byte-identical, nothing dropped.
    final, record = enforce_request_budget_or_raise(
        _request_for_observation(projected_body),
        policy,
        subject="graph observation",
    )
    assert record["disposition"] == DISPOSITION_ACCEPTED
    assert final.messages[-1].content == projected_body

    # The raw observation cannot ship as-is: H2 clips it to a valid
    # JSON envelope instead of sending it or failing the turn.
    clipped, raw_record = enforce_request_budget(
        _request_for_observation(raw_body), policy
    )
    assert raw_record["disposition"] == DISPOSITION_REDUCED
    assert raw_record["reduction"]["tool_messages_clipped"] == 1
    envelope = json.loads(clipped.messages[-1].content)
    assert envelope["truncated"] is True
    assert envelope["original_characters"] == len(raw_body)

    # Whatever leaves the process fits the budget, by estimate.
    for request in (final, clipped):
        probe = build_request_budget_record(request, policy)
        assert probe["within_budget"] is True


# -- 7: the executable LangGraph planning path -----------------------------


def _prepare_graph_planner(
    monkeypatch,
    architecture_context: str,
    response: Any = None,
) -> List[LLMRequest]:
    """Drive ``langgraph.planner.get_plan`` with a scripted context and
    a counting client.  Returns the client's request list."""
    calls: List[LLMRequest] = []
    monkeypatch.setattr(
        lg_planner,
        "read_optional_file",
        lambda *args, **kwargs: architecture_context,
    )
    monkeypatch.setattr(lg_planner, "print_request", lambda request: None)
    monkeypatch.setattr(lg_planner, "load_env", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-tests-only")

    class _CountingClient:
        def __init__(self, model: Any = None) -> None:
            self.model = model

        def complete(self, request: LLMRequest) -> Any:
            calls.append(request)
            if response is None:
                raise AssertionError(
                    "provider called for an over-budget request"
                )
            return response

    monkeypatch.setattr(
        openai_client_module, "OpenAILLMClient", _CountingClient
    )
    return calls


def test_langgraph_planner_rejects_an_oversized_request_before_the_provider(
    monkeypatch, capsys
):
    calls = _prepare_graph_planner(monkeypatch, "c" * 400_851)

    code = lg_planner.get_plan("How do agents and MCP servers interact?")

    captured = capsys.readouterr()
    assert calls == []
    assert code == 4
    assert "RequestBudgetExceededError" in captured.out
    assert "LOCAL context-budget preflight" in captured.out


def test_langgraph_planner_sends_a_within_budget_request(
    monkeypatch, capsys
):
    objective = "How do agents and MCP servers interact?"
    calls = _prepare_graph_planner(
        monkeypatch,
        "short architecture context",
        rq.structured_response(rq.offline_plan_payload(objective)),
    )

    plan = lg_planner.get_plan(objective)

    capsys.readouterr()
    assert len(calls) == 1
    assert calls[0] == rq.build_planning_request(
        objective, "short architecture context", None
    )
    assert [step.id for step in plan.steps] == ["S1", "S2", "S3"]


def test_the_orchestration_graph_makes_no_direct_provider_call():
    """The task-level graph driver submits NO model request itself.

    Every model call on that path goes through ``AgentTaskExecutor`` →
    ``Agent.run_result`` → the agent's own gate, so the graph file
    must never grow a direct provider call.
    """
    source = GRAPH_RUNNER.read_text(encoding="utf-8")
    for needle in (".complete(", "chat.completions", "OpenAILLMClient"):
        assert needle not in source


# -- 5 / 9: the specialist boundary is untouched ---------------------------


def test_specialist_agent_oversized_request_never_reaches_the_provider():
    """The specialist path keeps its own gate (H2): a too-large agent
    request fails locally with a typed termination instead of at the
    SDK, and no request is built for the provider.
    """
    client = _RecordingPlanner()
    agent = Agent(
        llm_client=client,
        execution_config=AgentExecutionConfig(max_request_input_tokens=40),
    )

    with pytest.raises(AgentError) as excinfo:
        agent.run_result(OVERSIZED_QUESTION)

    assert client.requests == []
    assert excinfo.value.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
    record = agent.last_trace.iterations[0].request_budget
    assert record["disposition"] == DISPOSITION_REJECTED
    assert record["within_budget"] is False
    # The specialist failure is typed by termination reason, not by a
    # free-text message, so it stays distinguishable from an LLMError.
    assert agent.last_trace.termination_reason == (
        "CONTEXT_BUDGET_EXCEEDED"
    )
