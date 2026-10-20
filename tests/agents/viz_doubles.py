"""Shared scripted doubles for the Architecture Visualization Agent tests.

Same discipline as ``tests/agents/architecture_doubles.py`` — whose
generic pieces (queued ``LLMResponse``s, fixture bindings, the task /
plan / rule helpers) are IMPORTED here, not copied.  Everything the
specialist decides (envelope parsing, handoff projection, diagram
validation, artifact persistence) is asserted against this scripted
ground truth: one arch_mcp-based report pair, one DiagramSpec payload
that matches it, and payloads that deliberately do not.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from architecture_doubles import (  # noqa: F401  (shared test utils)
    COMMIT,
    MCP_COLLECTOR_PATH,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    REPOSITORY,
    FakeLLMClient,
    build_investigator,
    fixture_bindings,
    mcp_report_payload,
    rule,
    structured,
    tool_calls,
)
from flow_doubles import (  # noqa: F401  (shared test utils)
    build_flow_investigator,
    mcp_payload,
    read_then_report,
)

from repo_intelligence.agent import Agent
from repo_intelligence.agents import (
    ArchitectureVisualizationAgent,
    DiagramSpec,
    VisualizationRequest,
)
from repo_intelligence.agents.models.architecture_report import ArchitectureReport
from repo_intelligence.agents.models.flow_dependency_report import FlowDependencyReport
from repo_intelligence.llm import ToolCall
from repo_intelligence.multiagent import TaskPayload
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)

VIZ_QUESTION = "How does the collector agent reach the records MCP server?"
FETCH_RECORD_PATH = "tools/fetch_record.py"

# The runner's investigator question labels a step as
# "step: <id> — <goal>" (em dash U+2014) — the route key.  The label
# sits on its own line inside the question, hence MULTILINE.
_STEP_LABEL = re.compile("^step: (\\S+) \u2014 ", re.MULTILINE)

__all__ = [
    "FETCH_RECORD_PATH",
    "MCP_COLLECTOR_PATH",
    "MCP_CONFIG_PATH",
    "MCP_FIXTURE_ROOT",
    "VIZ_QUESTION",
    "ArchitectureVisualizationAgent",
    "RoutedInvestigator",
    "RoutedInvestigatorError",
    "build_flow",
    "build_visualization_agent",
    "build_investigator",
    "fixture_bindings",
    "invalid_edge_spec_payload",
    "out_of_scope_spec_payload",
    "read_paths",
    "read_then_report",
    "rule",
    "spec_payload",
    "structured",
    "viz_arch_payload",
    "viz_flow_payload",
    "viz_plan_payload",
    "viz_request",
    "viz_task",
]


def viz_arch_payload(**overrides: Any) -> Dict[str, Any]:
    """The Architecture Investigator's report over the arch_mcp
    fixture, pinned to the shared visualization scope."""
    data = mcp_report_payload()
    data.update(
        {"repository_scope": REPOSITORY, "commit_scope": COMMIT}
    )
    data.update(overrides)
    return data


def viz_flow_payload(**overrides: Any) -> Dict[str, Any]:
    """The Flow / Dependency Investigator's report over the SAME
    fixture and the SAME scope (one visualization, one pinned state)."""
    data = mcp_payload()
    data.update(
        {"repository_scope": REPOSITORY, "commit_scope": COMMIT}
    )
    data.update(overrides)
    return data


def viz_request(**overrides: Any) -> VisualizationRequest:
    """A direct-mode request carrying both typed reports."""
    data: Dict[str, Any] = {
        "question": VIZ_QUESTION,
        "diagram_type": "AGENT_TOOL_MCP",
        "architecture_report": ArchitectureReport(**viz_arch_payload()),
        "flow_report": FlowDependencyReport(**viz_flow_payload()),
    }
    data.update(overrides)
    return VisualizationRequest(**data)


def spec_payload(**overrides: Any) -> Dict[str, Any]:
    """The DiagramSpec payload the scripted model returns — every
    node, edge and reference exists in the supplied reports."""
    data: Dict[str, Any] = {
        "diagram_type": "AGENT_TOOL_MCP",
        "title": "Collector agent through fetch_record to the records "
        "MCP server",
        "question": VIZ_QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "nodes": [
            {
                "id": "collector",
                "label": "Collector",
                "node_type": "agent",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
            },
            {
                "id": "fetch_record",
                "label": "fetch_record",
                "node_type": "tool",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": FETCH_RECORD_PATH}],
            },
            {
                "id": "records",
                "label": "records",
                "node_type": "mcp_server",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": MCP_CONFIG_PATH}],
            },
        ],
        "edges": [
            {
                "id": "collector-declares-fetch_record",
                "source": "collector",
                "target": "fetch_record",
                "relationship_type": "DECLARES",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
            },
            {
                "id": "fetch_record-exposed_by-records",
                "source": "fetch_record",
                "target": "records",
                "relationship_type": "EXPOSED_BY",
                "classification": "derived",
                "confidence": "medium",
                "evidence_refs": [{"path": MCP_CONFIG_PATH}],
            },
        ],
        "groups": [
            {
                "id": "agent-side",
                "label": "Agent side",
                "node_ids": ["collector", "fetch_record"],
            }
        ],
        "legend": [
            {
                "kind": "node_type",
                "term": "agent",
                "meaning": "autonomous actor",
            },
            {
                "kind": "relationship_type",
                "term": "EXPOSED_BY",
                "meaning": "exposed through a server",
            },
        ],
        "evidence_refs": [
            {"path": MCP_COLLECTOR_PATH},
            {"path": FETCH_RECORD_PATH},
            {"path": MCP_CONFIG_PATH},
        ],
        "uncertainties": [
            "the wire format between the tool and the MCP server is "
            "not established"
        ],
        "status": "complete",
    }
    data.update(overrides)
    return data


def out_of_scope_spec_payload(**overrides: Any) -> Dict[str, Any]:
    """A DiagramSpec that CONSTRUCTS but carries a component the
    investigation never established — the validator's job to reject."""
    data = spec_payload()
    data["nodes"] = list(data["nodes"]) + [
        {
            "id": "notes_store",
            "label": "notes store",
            "node_type": "unknown",
            "classification": "unknown",
            "confidence": "unknown",
        }
    ]
    data.update(overrides)
    return data


def invalid_edge_spec_payload(**overrides: Any) -> Dict[str, Any]:
    """A DiagramSpec that invents an edge the reports never
    established (collapsing the tool out of the chain)."""
    data = spec_payload()
    data["edges"] = list(data["edges"]) + [
        {
            "id": "collector-imports-records",
            "source": "collector",
            "target": "records",
            "relationship_type": "IMPORTS",
            "classification": "observed",
            "confidence": "high",
            "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
        }
    ]
    data.update(overrides)
    return data


def read_paths(*paths: str) -> Any:
    """One scripted tool step: real read requests with unique ids
    (the files the later evidence index must see as inspected)."""
    return tool_calls(
        *[
            ToolCall(
                id="call_viz_read_{}".format(index),
                name="read_file",
                arguments={"path": path},
            )
            for index, path in enumerate(paths)
        ]
    )


def build_visualization_agent(
    *responses, trace_sink=None
) -> ArchitectureVisualizationAgent:
    """The specialist over a scripted Agent with NO tools — the
    analyst-slot composition the direct path and the orchestrator
    both use."""
    agent = Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=DiagramSpec,
        tool_definitions=tool_definitions_for("analyst"),
    )
    return ArchitectureVisualizationAgent(agent=agent, tools=())


def build_flow(
    *responses, bindings: Optional[Dict[str, Any]] = None, trace_sink=None
):
    """The Flow / Dependency Investigator over the arch_mcp fixture."""
    return build_flow_investigator(
        *responses,
        bindings=bindings
        if bindings is not None
        else fixture_bindings(MCP_FIXTURE_ROOT),
        trace_sink=trace_sink,
    )


def viz_task(**overrides: Any) -> TaskPayload:
    data: Dict[str, Any] = {
        "question": VIZ_QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }
    data.update(overrides)
    return TaskPayload(**data)


def viz_plan_payload(**overrides: Any) -> Dict[str, Any]:
    """Three steps: two INVESTIGATOR goals (architecture, flow) then
    one ANALYST goal (the visualization reads the goal pattern the
    repository's role assigner uses)."""
    steps = overrides.pop(
        "steps",
        [
            {
                "id": "S1",
                "goal": "Document the architecture components and their "
                "relationships in this repository.",
            },
            {
                "id": "S2",
                "goal": "Trace the path from the collector agent to the "
                "records MCP server.",
            },
            {
                "id": "S3",
                "goal": "Interpret the investigated findings as one "
                "diagram for the user's question.",
            },
        ],
    )
    data: Dict[str, Any] = {"objective": VIZ_QUESTION, "steps": steps}
    data.update(overrides)
    return data


class RoutedInvestigatorError(Exception):
    """A question with no routed step — explicit, never a guess."""


class RoutedInvestigator:
    """Test-only adapter: one investigator slot, two specialists.

    The generic orchestrator holds exactly ONE investigator, so this
    router reads the plan's step label the runner embedded
    (``step: <id> — <goal>``) and delegates to that step's
    specialist.  Retry-safe (the same step routes to the same
    specialist) and trace-safe (``last_trace`` is the specialist that
    actually ran).  Neither the orchestrator nor the runner knows a
    router exists — the orchestration infrastructure stays
    unmodified.
    """

    def __init__(self, route: Dict[str, Any]) -> None:
        self.route = dict(route)
        self._active: Optional[Any] = None
        self.calls: List[str] = []

    @property
    def last_trace(self) -> Any:
        return self._active.last_trace if self._active is not None else None

    def run_result(self, question: str) -> Any:
        match = _STEP_LABEL.search(question)
        step_id = match.group(1) if match is not None else None
        specialist = self.route.get(step_id or "")
        if specialist is None:
            raise RoutedInvestigatorError(
                "no specialist routed for step {!r}".format(step_id)
            )
        self._active = specialist
        self.calls.append(step_id or "")
        return specialist.run_result(question)
