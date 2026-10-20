"""Shared scripted doubles for the Flow / Dependency Investigator tests.

Same discipline as ``tests/agents/architecture_doubles.py`` — whose
generic pieces (queued ``LLMResponse``s, fixture bindings, the
scripted analyst, the plan/task/rule helpers) are IMPORTED here, not
copied.  Everything the specialist decides (path rules, projection,
evidence checks) is asserted against this scripted ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from architecture_doubles import (  # noqa: F401  (shared test utils)
    FakeLLMClient,
    fixture_bindings,
    structured,
    tool_calls,
)

from repo_intelligence.agent import Agent
from repo_intelligence.agents import (
    FLOW_DEPENDENCY_TOOLS,
    FlowDependencyInvestigator,
    FlowDependencyReport,
)
from repo_intelligence.llm import ToolCall
from repo_intelligence.multiagent import TaskPayload
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.planning.models import InvestigationPlan, PlanStep

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FLOW_CHAIN_ROOT = FIXTURES / "flow_chain"
FLOW_DISCONNECTED_ROOT = FIXTURES / "flow_disconnected"
FLOW_CONFLICT_ROOT = FIXTURES / "flow_conflict"
MCP_FIXTURE_ROOT = FIXTURES / "arch_mcp"

FLOW_REPOSITORY = "example-flow-repository"
FLOW_COMMIT = "c" * 40
FLOW_QUESTION = "How does the producer reach the publisher?"
MCP_QUESTION = "How does the collector agent reach the records MCP server?"

# flow_chain files (multi-hop: producer -> transform -> publisher)
PRODUCER_PATH = "src/producer.py"
TRANSFORM_PATH = "src/transform.py"
PUBLISHER_PATH = "src/publisher.py"

# flow_disconnected files (no edge between them)
ALPHA_PATH = "src/alpha.py"
BETA_PATH = "src/beta.py"

# flow_conflict files (docs claim a direct edge; code routes around)
API_PATH = "src/api.py"
BRIDGE_PATH = "src/bridge.py"
RENDERER_PATH = "src/renderer.py"
WIRING_DOC_PATH = "docs/wiring.md"

# arch_mcp files (agent -> tool -> MCP server)
COLLECTOR_PATH = "agents/collector.py"
FETCH_RECORD_PATH = "tools/fetch_record.py"
MCP_CONFIG_PATH = ".mcp.json"


def ref(path: str) -> Dict[str, Any]:
    return {"path": path}


def step(
    source: str,
    target: str,
    relationship_type: str,
    paths: List[str],
    classification: str = "observed",
    confidence: str = "high",
) -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relationship_type": relationship_type,
        "evidence_refs": [ref(path) for path in paths],
        "classification": classification,
        "confidence": confidence,
    }


def finding(
    statement: str, paths: List[str], support: str = "SUPPORTED"
) -> Dict[str, Any]:
    return {
        "statement": statement,
        "evidence_refs": [ref(path) for path in paths],
        "support": support,
    }


def flow_payload(**overrides: Any) -> Dict[str, Any]:
    """A complete multi-hop FOUND payload over the flow_chain fixture
    (producer -> transform -> publisher, evidence in hand)."""
    data: Dict[str, Any] = {
        "question": FLOW_QUESTION,
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "producer",
        "target": "publisher",
        "path_status": "FOUND",
        "summary": "producer reaches publisher through transform.",
        "path": [
            step(
                "producer",
                "transform",
                "IMPORTS",
                [PRODUCER_PATH],
            ),
            step(
                "transform",
                "publisher",
                "IMPORTS",
                [TRANSFORM_PATH],
            ),
        ],
        "alternative_paths": [],
        "relationships": [
            step("producer", "transform", "IMPORTS", [PRODUCER_PATH]),
            step(
                "transform", "publisher", "IMPORTS", [TRANSFORM_PATH]
            ),
        ],
        "findings": [
            finding(
                "producer reaches publisher through transform",
                [PRODUCER_PATH],
            )
        ],
        "evidence_refs": [ref(PRODUCER_PATH), ref(TRANSFORM_PATH)],
        "uncertainties": [],
        "status": "complete",
    }
    data.update(overrides)
    return data


def direct_payload(**overrides: Any) -> Dict[str, Any]:
    """A single-step direct flow (Test 1): producer -> transform."""
    data = flow_payload(
        source="producer",
        target="transform",
        summary="producer reaches transform directly.",
        path=[step("producer", "transform", "IMPORTS", [PRODUCER_PATH])],
        relationships=[
            step("producer", "transform", "IMPORTS", [PRODUCER_PATH])
        ],
        findings=[
            finding(
                "producer reaches transform directly", [PRODUCER_PATH]
            )
        ],
        evidence_refs=[ref(PRODUCER_PATH)],
    )
    data.update(overrides)
    return data


def mcp_payload(**overrides: Any) -> Dict[str, Any]:
    """Agent -> Tool -> MCP Server with intermediates preserved
    (Test 3) over the arch_mcp fixture."""
    data: Dict[str, Any] = {
        "question": MCP_QUESTION,
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "collector",
        "target": "records",
        "path_status": "FOUND",
        "summary": (
            "collector reaches the records MCP server through the "
            "fetch_record tool."
        ),
        "path": [
            step("collector", "fetch_record", "IMPORTS",
                 [COLLECTOR_PATH]),
            step("fetch_record", "records", "EXPOSED_BY",
                 [MCP_CONFIG_PATH]),
        ],
        "alternative_paths": [],
        "relationships": [
            step("collector", "fetch_record", "IMPORTS",
                 [COLLECTOR_PATH]),
            step("fetch_record", "records", "EXPOSED_BY",
                 [MCP_CONFIG_PATH]),
        ],
        "findings": [
            finding(
                "the collector's record fetch is exposed over MCP",
                [MCP_CONFIG_PATH],
            )
        ],
        "evidence_refs": [ref(COLLECTOR_PATH), ref(MCP_CONFIG_PATH)],
        "uncertainties": [],
        "status": "complete",
    }
    data.update(overrides)
    return data


def no_path_payload(
    path_status: str = "NOT_FOUND", **overrides: Any
) -> Dict[str, Any]:
    """alpha and beta exist but share no supported edge (Test 4)."""
    data: Dict[str, Any] = {
        "question": "How does alpha reach beta?",
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "alpha",
        "target": "beta",
        "path_status": path_status,
        "summary": (
            "no path between alpha and beta is established."
            if path_status == "NOT_FOUND"
            else "insufficient evidence either way."
        ),
        "path": [],
        "alternative_paths": [],
        "relationships": [],
        "findings": [
            finding(
                "alpha and beta are isolated modules",
                [ALPHA_PATH],
                support=(
                    "SUPPORTED"
                    if path_status == "NOT_FOUND"
                    else "UNKNOWN"
                ),
            )
        ],
        "evidence_refs": [ref(ALPHA_PATH)],
        "uncertainties": [
            (
                "no repository artifact links alpha to beta"
                if path_status == "NOT_FOUND"
                else "neither module references the other; absence of "
                "imports alone cannot settle the question"
            )
        ],
        "status": "complete",
    }
    data.update(overrides)
    return data


def conflicting_payload(**overrides: Any) -> Dict[str, Any]:
    """Docs claim api -> renderer directly; the code routes through
    the bridge (Test 6): both candidates preserved."""
    data: Dict[str, Any] = {
        "question": "How does the api reach the renderer?",
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "api",
        "target": "renderer",
        "path_status": "CONFLICTING",
        "summary": (
            "wiring docs claim a direct edge; the code routes api "
            "through bridge."
        ),
        "path": [
            step("api", "renderer", "DEPENDS_ON", [WIRING_DOC_PATH],
                 classification="conflicting", confidence="low"),
        ],
        "alternative_paths": [
            [
                step("api", "bridge", "IMPORTS", [API_PATH]),
                step("bridge", "renderer", "IMPORTS", [BRIDGE_PATH]),
            ]
        ],
        "relationships": [
            step("api", "renderer", "DEPENDS_ON", [WIRING_DOC_PATH],
                 classification="conflicting", confidence="low"),
            step("api", "bridge", "IMPORTS", [API_PATH]),
            step("bridge", "renderer", "IMPORTS", [BRIDGE_PATH]),
        ],
        "findings": [
            finding(
                "docs and code disagree about the api->renderer edge",
                [WIRING_DOC_PATH, API_PATH],
                support="SUPPORTED",
            )
        ],
        "evidence_refs": [
            ref(WIRING_DOC_PATH),
            ref(API_PATH),
            ref(BRIDGE_PATH),
        ],
        "uncertainties": [
            "documentation asserts a direct edge the code does not "
            "implement"
        ],
        "status": "complete",
    }
    data.update(overrides)
    return data


def contains_payload(**overrides: Any) -> Dict[str, Any]:
    """A structural CONTAINS edge stays CONTAINS (Test 5) — never
    relabelled as a call."""
    data = direct_payload(
        source="producer",
        target="transform",
        path=[
            step("producer", "transform", "CONTAINS", [PRODUCER_PATH],
                 classification="derived", confidence="medium")
        ],
        relationships=[
            step("producer", "transform", "CONTAINS", [PRODUCER_PATH],
                 classification="derived", confidence="medium")
        ],
        path_status="FOUND",
        summary="structurally contained, not a demonstrated call.",
        findings=[],
    )
    data.update(overrides)
    return data


# -- orchestrated story (arch_mcp: sub-path reports, one per step) --------


def mcp_endpoint_payload() -> Dict[str, Any]:
    """Step P1: endpoints identified, path not yet established."""
    return {
        "question": MCP_QUESTION,
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "collector",
        "target": "records",
        "path_status": "UNKNOWN",
        "summary": "endpoints identified; path pending further steps.",
        "path": [],
        "alternative_paths": [],
        "relationships": [],
        "findings": [
            finding(
                "collector is the flow's agent endpoint",
                [COLLECTOR_PATH],
            ),
            finding(
                "records is the flow's MCP server endpoint",
                [MCP_CONFIG_PATH],
            ),
        ],
        "evidence_refs": [ref(COLLECTOR_PATH), ref(MCP_CONFIG_PATH)],
        "uncertainties": [],
        "status": "complete",
    }


def mcp_leg_one_payload() -> Dict[str, Any]:
    """Step P2: the agent -> tool leg (FOUND sub-path)."""
    return {
        "question": MCP_QUESTION,
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "collector",
        "target": "fetch_record",
        "path_status": "FOUND",
        "summary": "collector reaches fetch_record.",
        "path": [
            step("collector", "fetch_record", "IMPORTS",
                 [COLLECTOR_PATH])
        ],
        "alternative_paths": [],
        "relationships": [
            step("collector", "fetch_record", "IMPORTS",
                 [COLLECTOR_PATH])
        ],
        "findings": [],
        "evidence_refs": [ref(COLLECTOR_PATH)],
        "uncertainties": [],
        "status": "complete",
    }


def mcp_leg_two_payload() -> Dict[str, Any]:
    """Step P3: the tool -> MCP server leg (FOUND sub-path)."""
    return {
        "question": MCP_QUESTION,
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "source": "fetch_record",
        "target": "records",
        "path_status": "FOUND",
        "summary": "fetch_record is exposed by the records server.",
        "path": [
            step("fetch_record", "records", "EXPOSED_BY",
                 [MCP_CONFIG_PATH])
        ],
        "alternative_paths": [],
        "relationships": [
            step("fetch_record", "records", "EXPOSED_BY",
                 [MCP_CONFIG_PATH])
        ],
        "findings": [],
        "evidence_refs": [ref(MCP_CONFIG_PATH)],
        "uncertainties": [],
        "status": "complete",
    }


def mcp_finding_payload() -> Dict[str, Any]:
    """Step P4: the Analyst's conclusion echoing the derived id."""
    derived = FlowDependencyReport(
        **mcp_endpoint_payload()
    ).derived_investigation_id()
    return {
        "investigation_id": derived,
        "question": MCP_QUESTION,
        "repository_scope": FLOW_REPOSITORY,
        "commit_scope": FLOW_COMMIT,
        "conclusions": [
            {
                "statement": (
                    "the agent reaches its MCP capability through the "
                    "tool leg the evidence shows (scripted "
                    "conclusion)."
                ),
                "classification": "observed",
                "evidence_refs": [ref(COLLECTOR_PATH)],
            }
        ],
        "evidence_refs": [ref(COLLECTOR_PATH)],
        "summary": "Evidence-backed flow conclusion.",
        "status": "complete",
    }


def flow_plan(**overrides: Any) -> InvestigationPlan:
    """Evidence before interpretation: three INVESTIGATOR steps then
    one ANALYST step (the flow question's natural shape)."""
    steps = overrides.pop(
        "steps",
        [
            PlanStep(
                id="P1",
                goal="Identify the source and target components",
            ),
            PlanStep(
                id="P2",
                goal="Investigate how the agent reaches its tool",
            ),
            PlanStep(
                id="P3",
                goal="Investigate how the tool is exposed to the "
                "MCP server",
            ),
            PlanStep(
                id="P4",
                goal="Explain how the agent reaches the MCP server",
            ),
        ],
    )
    return InvestigationPlan(
        objective=overrides.pop("objective", MCP_QUESTION),
        steps=steps,
        **overrides,
    )


def flow_plan_payload() -> Dict[str, Any]:
    return {
        "objective": MCP_QUESTION,
        "steps": [
            {"id": entry.id, "goal": entry.goal}
            for entry in flow_plan().steps
        ],
    }


def flow_task() -> TaskPayload:
    return TaskPayload(
        question=MCP_QUESTION,
        repository_scope=FLOW_REPOSITORY,
        commit_scope=FLOW_COMMIT,
    )


def read_then_report(paths: List[str], payload: Dict[str, Any]) -> list:
    """One script: real tool request(s) with unique ids, then the
    typed report whose refs match what those reads gather."""
    return [
        tool_calls(
            *[
                ToolCall(
                    id="call_flow_read_{}".format(index),
                    name="read_file",
                    arguments={"path": path},
                )
                for index, path in enumerate(paths)
            ]
        ),
        structured(payload),
    ]


def build_flow_investigator(
    *responses,
    bindings: Optional[Dict[str, Any]] = None,
    trace_sink=None,
) -> FlowDependencyInvestigator:
    """The specialist over a scripted Agent (the composition the demo
    and the orchestrator both use)."""
    agent = Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=restricted_dispatcher(
            "investigator",
            bindings=bindings if bindings is not None else
            fixture_bindings(FLOW_CHAIN_ROOT),
        ),
        trace_sink=trace_sink,
        structured_output_model=FlowDependencyReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    return FlowDependencyInvestigator(
        agent=agent, tools=FLOW_DEPENDENCY_TOOLS
    )
