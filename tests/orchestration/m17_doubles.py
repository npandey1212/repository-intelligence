"""Shared scripted doubles for the M17 orchestration tests.

Same discipline as ``tests/multiagent/test_runner.py``: queued
``LLMResponse``s, a canned read-only dispatcher, one ``TaskPayload`` —
no network, no API key, no timing dependence. Everything the
orchestrator decides (roles, budgets, retries, terminal status) is
asserted against this scripted ground truth.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from repo_intelligence.agent import Agent, AgentExecutionConfig
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.multiagent import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import Orchestrator, OrchestrationConfig
from repo_intelligence.planning.models import InvestigationPlan, PlanStep
from repo_intelligence.tools import ToolResult

REPOSITORY = "example-repository"
COMMIT = "c" * 40
QUESTION = "Where is the application boundary enforced?"
README_REF = {"path": "README.md"}
SERVICE_REF = {"path": "src/service.py"}


class FakeLLMClient:
    """Queued responses (or queued exceptions); refuses to be
    over-invoked."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def structured(payload: Dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def call(name: str, arguments: Dict[str, Any]) -> ToolCall:
    return ToolCall(id="call_1", name=name, arguments=dict(arguments))


def tool_calls(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def report_payload(**overrides: Any) -> Dict[str, Any]:
    data = {
        "investigation_id": "inv-1",
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "findings": [
            {
                "statement": "README.md documents the dispatcher boundary",
                "evidence_refs": [dict(README_REF)],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [dict(README_REF)],
        "investigated_sources": ["README.md"],
        "status": "complete",
    }
    data.update(overrides)
    return data


def finding_payload(**overrides: Any) -> Dict[str, Any]:
    data = {
        "investigation_id": "inv-1",
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "conclusions": [
            {
                "statement": "the boundary is application-owned",
                "classification": "observed",
                "evidence_refs": [dict(README_REF)],
            }
        ],
        "evidence_refs": [dict(README_REF)],
        "summary": "One dispatcher, one boundary.",
        "status": "complete",
    }
    data.update(overrides)
    return data


def investigator_bindings():
    """Hermetic read-only tools (canned successes for any path)."""

    def read_file(arguments):
        path = arguments.get("path") or "README.md"
        return ToolResult(
            success=True,
            tool="read_file",
            data={"path": path, "content": "the dispatcher owns the boundary"},
        )

    def generic(name):
        def tool(arguments):
            return ToolResult(
                success=True,
                tool=name,
                data={
                    "path": arguments.get("path") or "README.md",
                    "content": "x",
                },
            )

        return tool

    return {
        "read_file": read_file,
        "search_repository": generic("search_repository"),
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def investigate_read_then_report(
    payload: Optional[Dict[str, Any]] = None,
) -> list:
    """The default Investigator script: read README.md, then report."""
    return [
        tool_calls(call("read_file", dict(README_REF))),
        structured(payload if payload is not None else report_payload()),
    ]


def build(
    inv_responses=None,
    ana_responses=None,
    config: Optional[OrchestrationConfig] = None,
    clock=None,
    trace_sink=None,
    inv_kwargs=None,
    role_assigner=None,
):
    """Fresh Orchestrator over scripted agents (never a shared client)."""
    if inv_responses is None:
        inv_responses = investigate_read_then_report()
    if ana_responses is None:
        ana_responses = [structured(finding_payload())]
    kwargs: Dict[str, Any] = dict(inv_kwargs or {})
    investigator = Agent(
        FakeLLMClient(*inv_responses),
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=investigator_bindings()
        ),
        trace_sink=trace_sink,
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
        **kwargs,
    )
    analyst = Agent(
        FakeLLMClient(*ana_responses),
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
    )
    orch_kwargs: Dict[str, Any] = {}
    if config is not None:
        orch_kwargs["config"] = config
    if clock is not None:
        orch_kwargs["clock"] = clock
    if trace_sink is not None:
        orch_kwargs["trace_sink"] = trace_sink
    if role_assigner is not None:
        orch_kwargs["role_assigner"] = role_assigner
    return (
        Orchestrator(investigator, analyst, **orch_kwargs),
        investigator,
        analyst,
    )


def two_step_plan(**kwargs) -> InvestigationPlan:
    """Evidence step then interpretation step (the happy-path shape)."""
    steps = kwargs.pop(
        "steps",
        [
            PlanStep(id="P1", goal="Identify the application boundary"),
            PlanStep(id="P2", goal="Explain how the boundary is enforced"),
        ],
    )
    return InvestigationPlan(
        objective=kwargs.pop("objective", QUESTION), steps=steps, **kwargs
    )


def task(**overrides) -> TaskPayload:
    data = {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }
    data.update(overrides)
    return TaskPayload(**data)


def rule(result, name: str):
    matches = [check for check in result.checks if check.rule == name]
    assert len(matches) == 1, name
    return matches[0]


def agent_budget_config(**overrides) -> AgentExecutionConfig:
    return AgentExecutionConfig(**overrides)
