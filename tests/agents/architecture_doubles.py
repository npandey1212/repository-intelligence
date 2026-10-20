"""Shared scripted doubles for the Architecture Investigator tests.

Same discipline as ``tests/orchestration/m17_doubles.py``: queued
``LLMResponse``s, fixture-bound read-only tools, one plan/task — no
network, no API key, no timing dependence. Everything the specialist
decides (prompt wrapping, handoff projection, evidence checks) is
asserted against this scripted ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from repo_intelligence.agent import Agent, AgentExecutionConfig
from repo_intelligence.agents import (
    ARCHITECTURE_TOOLS,
    ArchitectureInvestigator,
    ArchitectureReport,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.multiagent import (
    ArchitectureFinding,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.planning.models import InvestigationPlan, PlanStep
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.read_file import read_file as real_read_file
from repo_intelligence.tools.search_repository import (
    search_repository as real_search_repository,
)

REPOSITORY = "example-repository"
COMMIT = "c" * 40
QUESTION = "Explain the architecture of this repository."

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "arch_simple"
MCP_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "arch_mcp"
CONFLICT_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "arch_conflict"
)

RESEARCHER_PATH = "agents/researcher.py"
MCP_COLLECTOR_PATH = "agents/collector.py"
MCP_CONFIG_PATH = ".mcp.json"


class FakeLLMClient:
    """Queued responses; refuses to be over-invoked."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests: List[Any] = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


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
    """A minimal valid ArchitectureReport payload (arch_simple evidence)."""
    data = {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "summary": "One agent reads notes through two registered tools.",
        "components": [
            {
                "id": "researcher",
                "name": "Researcher",
                "type": "agent",
                "responsibility": "answers questions from notes",
                "evidence_refs": [{"path": RESEARCHER_PATH}],
                "classification": "observed",
            },
            {
                "id": "search_notes",
                "name": "search_notes",
                "type": "tool",
                "responsibility": "lexical note search",
                "evidence_refs": [
                    {"path": RESEARCHER_PATH},
                    {"path": "tools/search_notes.py"},
                ],
                "classification": "observed",
            },
        ],
        "relationships": [
            {
                "source": "researcher",
                "target": "search_notes",
                "relationship_type": "DECLARES",
                "evidence_refs": [{"path": RESEARCHER_PATH}],
                "classification": "observed",
                "confidence": "high",
            }
        ],
        "findings": [
            {
                "statement": "the researcher imports its tools directly",
                "evidence_refs": [{"path": RESEARCHER_PATH}],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [
            {"path": RESEARCHER_PATH},
            {"path": "tools/search_notes.py"},
        ],
        "uncertainties": [
            "no runtime invocation evidence exists in the repository"
        ],
        "status": "complete",
    }
    data.update(overrides)
    return data


def mcp_report_payload(**overrides: Any) -> Dict[str, Any]:
    """An ArchitectureReport over the arch_mcp fixture."""
    data = {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "summary": "One agent exposes one tool through one MCP server.",
        "components": [
            {
                "id": "collector",
                "name": "Collector",
                "type": "agent",
                "responsibility": "collects records",
                "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                "classification": "observed",
            },
            {
                "id": "fetch_record",
                "name": "fetch_record",
                "type": "tool",
                "responsibility": "reads one record",
                "evidence_refs": [{"path": "tools/fetch_record.py"}],
                "classification": "observed",
            },
            {
                "id": "records",
                "name": "records",
                "type": "mcp_server",
                "responsibility": "serves fetch_record over MCP",
                "evidence_refs": [{"path": MCP_CONFIG_PATH}],
                "classification": "observed",
            },
        ],
        "relationships": [
            {
                "source": "collector",
                "target": "fetch_record",
                "relationship_type": "DECLARES",
                "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                "classification": "observed",
                "confidence": "high",
            },
            {
                "source": "fetch_record",
                "target": "records",
                "relationship_type": "EXPOSED_BY",
                "evidence_refs": [{"path": MCP_CONFIG_PATH}],
                "classification": "derived",
                "confidence": "medium",
            },
        ],
        "findings": [
            {
                "statement": "the collector imports the record tool",
                "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [
            {"path": MCP_COLLECTOR_PATH},
            {"path": "tools/fetch_record.py"},
            {"path": MCP_CONFIG_PATH},
        ],
        "uncertainties": [],
        "status": "complete",
    }
    data.update(overrides)
    return data


def finding_payload(investigation_id: str, **overrides: Any) -> Dict[str, Any]:
    """The Analyst's structured payload (M16 ArchitectureFinding)."""
    data = {
        "investigation_id": investigation_id,
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "conclusions": [
            {
                "statement": "the repository's agents reach tools directly",
                "classification": "observed",
                "evidence_refs": [{"path": RESEARCHER_PATH}],
            }
        ],
        "evidence_refs": [{"path": RESEARCHER_PATH}],
        "summary": "Agents declare tools; no MCP wiring is present.",
        "status": "complete",
    }
    data.update(overrides)
    return data


def derived_id(payload: Dict[str, Any]) -> str:
    """The handoff id the projection will derive from this payload."""
    return ArchitectureReport(
        question=payload["question"],
        repository_scope=payload["repository_scope"],
        commit_scope=payload["commit_scope"],
    ).derived_investigation_id()


def fixture_bindings(root: Path) -> Dict[str, Any]:
    """Real read-only tools over a foreign fixture root (the M16/M17
    binding seam); graph/context tools answer with canned successes."""

    def read(arguments):
        return real_read_file(arguments, repository_root=root)

    def search(arguments):
        return real_search_repository(arguments, repository_root=root)

    def generic(name):
        def tool(arguments):
            return ToolResult(
                success=True, tool=name, data={"path": "", "content": ""}
            )

        return tool

    return {
        "read_file": read,
        "search_repository": search,
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def canned_bindings() -> Dict[str, Any]:
    """Hermetic read-only tools (canned successes for any path)."""

    def read_file(arguments):
        path = arguments.get("path") or "README.md"
        return ToolResult(
            success=True,
            tool="read_file",
            data={"path": path, "content": "agents declare their tools"},
        )

    def generic(name):
        def tool(arguments):
            return ToolResult(
                success=True,
                tool=name,
                data={"path": arguments.get("path") or "", "content": ""},
            )

        return tool

    return {
        "read_file": read_file,
        "search_repository": generic("search_repository"),
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def build_investigator(
    *responses,
    bindings: Optional[Dict[str, Any]] = None,
    trace_sink=None,
    execution_config: Optional[AgentExecutionConfig] = None,
) -> ArchitectureInvestigator:
    """The specialist over a scripted Agent (the composition the demo
    and the orchestrator both use)."""
    agent = Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=restricted_dispatcher(
            "investigator",
            bindings=bindings if bindings is not None else canned_bindings(),
        ),
        trace_sink=trace_sink,
        execution_config=execution_config,
        structured_output_model=ArchitectureReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    return ArchitectureInvestigator(agent=agent, tools=ARCHITECTURE_TOOLS)


def build_analyst(*responses, trace_sink=None) -> Agent:
    return Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
    )


def read_then_report(
    payload: Optional[Dict[str, Any]] = None,
    path: str = RESEARCHER_PATH,
) -> list:
    """The direct-mode script: one read, then the typed report."""
    return [
        tool_calls(call("read_file", {"path": path})),
        structured(payload if payload is not None else report_payload()),
    ]


def plan(**overrides: Any) -> InvestigationPlan:
    """Evidence before interpretation: three INVESTIGATOR steps then
    one ANALYST step (the architecture question's natural shape)."""
    steps = overrides.pop(
        "steps",
        [
            PlanStep(id="P1", goal="Identify the repository's agents and tools"),
            PlanStep(
                id="P2",
                goal="Investigate the MCP configuration and its wiring",
            ),
            PlanStep(
                id="P3",
                goal="Collect evidence for component relationships",
            ),
            PlanStep(
                id="P4",
                goal="Explain how agents, tools, and MCP servers interact",
            ),
        ],
    )
    return InvestigationPlan(
        objective=overrides.pop("objective", QUESTION), steps=steps, **overrides
    )


def plan_payload(**overrides: Any) -> Dict[str, Any]:
    steps = overrides.pop(
        "steps",
        [
            {"id": entry.id, "goal": entry.goal} for entry in plan().steps
        ],
    )
    data = {"objective": QUESTION, "steps": steps}
    data.update(overrides)
    return data


def task(**overrides: Any) -> TaskPayload:
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
