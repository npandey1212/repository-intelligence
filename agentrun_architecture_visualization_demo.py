#!/usr/bin/env python
"""Architecture Visualization Agent demonstration — direct +
orchestrated (§24).

What this script demonstrates (spec §24/§29):

1. the DIRECT path — the versioned specialist prompt over the existing
   Agent runtime producing a typed ``DiagramSpec`` from the two typed
   investigation reports, then the deterministic six-rule diagram
   validation and the existing renderer's Mermaid/SVG output;
2. the ORCHESTRATED path — the SAME specialists injected into the
   M10 → M17 planner/orchestrator (Architecture Investigator → Flow /
   Dependency Investigator → Architecture Visualization Agent) with the
   typed handoff to the M16 validator and the deterministic verdict;
3. the persistent artifact set — diagram_spec.json, architecture.mmd,
   architecture.svg (where Graphviz is installed) and run_summary.md —
   verified file by file after the run (spec §25).

Nine phases (spec §24): scope, architecture investigation, flow
investigation, visualization, validation, rendering, artifact
persistence, M17, trace.  One generic orchestrator, unmodified: the
investigator slot routes the two specialists by the plan's own step
labels, the analyst slot holds the visualizer, and the renderer stays a
deterministic stage.

Offline and deterministic by default: scripted models, the tiny
``tests/fixtures/arch_mcp`` fixture (or any root via ``--repository``),
real read-only tools, real validator.  ``--live`` uses a real model
(requires ``OPENAI_API_KEY``).

Exit codes: 0 = every condition holds; 1 = a demonstration condition
failed; 6 = execution failed (including --live without an API key).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    JsonFileTraceSink,
)
from repo_intelligence.agent.trace import render_markdown  # noqa: E402
from repo_intelligence.agents import (  # noqa: E402
    ARCHITECTURE_TOOLS,
    ARCHITECTURE_VISUALIZATION_PROMPT_V1,
    FLOW_DEPENDENCY_TOOLS,
    PATH_STATUSES,
    SVG_UNAVAILABLE_NOTICE,
    VISUALIZATION_PROMPT_VERSION,
    VISUALIZATION_TOOLS,
    ArchitectureInvestigator,
    ArchitectureInvestigatorError,
    ArchitectureReport,
    ArchitectureVisualizationAgent,
    ArchitectureVisualizationAgentError,
    DiagramScope,
    DiagramSpec,
    FlowDependencyInvestigator,
    FlowDependencyInvestigatorError,
    FlowDependencyReport,
    RunSummaryContext,
    VisualizationArtifactError,
    VisualizationRequest,
    verify_artifacts,
    write_visualization_artifacts,
)
from repo_intelligence.context_engine import (  # noqa: E402
    ContextBudget,
    ContextEngine,
)
from repo_intelligence.evidence.models import (  # noqa: E402
    STATUS_SUPPORTED,
)
from repo_intelligence.evidence.validator import (  # noqa: E402
    gather_evidence,
)
from repo_intelligence.llm import (  # noqa: E402
    LLMError,
    LLMResponse,
    Message,
    StructuredOutputValidationError,
    ToolCall,
)
from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.llm.openai_client import (  # noqa: E402
    DEFAULT_MODEL,
    OpenAILLMClient,
)
from repo_intelligence.memory import (  # noqa: E402
    MemoryContext,
    MemoryManager,
    MemoryOwner,
    MemoryScope,
)
from repo_intelligence.multiagent import (  # noqa: E402
    ArchitectureFinding,
    ContractError,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (  # noqa: E402
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import (  # noqa: E402
    COMPLETED,
    Orchestrator,
)
from repo_intelligence.planning.planner import (  # noqa: E402
    build_planning_request,
    plan_from_response,
)
from repo_intelligence.planning.validation import (  # noqa: E402
    PlanValidationError,
    validate_plan,
)
from repo_intelligence.tools import ToolResult  # noqa: E402
from repo_intelligence.tools.read_file import read_file  # noqa: E402
from repo_intelligence.tools.search_repository import (  # noqa: E402
    search_repository,
)

RULE = "=" * 72
SECTION = "-" * 52

EXIT_SUCCESS = 0
EXIT_DEMO_FAILED = 1
EXIT_EXECUTION_FAILED = 6

SUBJECT = "architecture-visualization-agent"

# Offline scope: a visibly synthetic commit (demo labels only).
OFFLINE_COMMIT = "c" * 40

DEFAULT_QUESTION = (
    "How does the collector agent reach the records MCP server?"
)

# The default investigation target is the capability's OWN tiny
# evaluation fixture (§30): deterministic, honest, self-contained.
# Pass --repository to investigate any other checkout.
DEFAULT_REPOSITORY = PROJECT_ROOT / "tests" / "fixtures" / "arch_mcp"
READ_CANDIDATES = (
    "agents/collector.py",
    "tools/fetch_record.py",
    "agents/researcher.py",
    "src/producer.py",
    "src/api.py",
    "src/service.py",
    "README.md",
    "main.py",
    "src/main.py",
)
TOOL_CANDIDATE = "tools/fetch_record.py"
MCP_CONFIG_NAME = ".mcp.json"

DEMO_DB_PATH = (
    PROJECT_ROOT / "runtime" / "agent_memory" /
    "architecture_visualization_demo.db"
)
DEFAULT_BUDGET = ContextBudget(max_items=64, max_characters=6000)

# The runner's investigator question labels a step as
# "step: <id> — <goal>" (em dash U+2014).  The label sits on its own
# line inside the question, hence MULTILINE; the goal rides along so
# the demo router can fall back to it for a live planner's ids.
_STEP_LABEL = re.compile("^step: (\\S+) \u2014 (.*)$", re.MULTILINE)


# -- scripted model --------------------------------------------------------


def _structured(payload: Dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


class OfflineModel:
    """Scripted LLM double: answers strictly from its queue, never
    touches the network, fails loudly if over-invoked."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._queue: List[LLMResponse] = list(responses)
        self.requests: List[Any] = []
        self.model = "offline-scripted"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("model invoked more times than scripted")
        return self._queue.pop(0)


def _clip(text: Any, limit: int = 88) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


# -- offline story ---------------------------------------------------------


def _ref(path: str) -> Dict[str, str]:
    return {"path": path}


def _unique(paths: List[str]) -> List[str]:
    seen: List[str] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _step(
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
        "evidence_refs": [_ref(path) for path in paths],
        "classification": classification,
        "confidence": confidence,
    }


def demo_read_target(repository: Optional[str]) -> Any:
    """Resolve (root, target_relative_path, bound_root).

    Default mode reads the capability's own arch_mcp fixture; a
    ``--repository`` root is searched for the first readable candidate.
    """
    root = (
        Path(repository).expanduser()
        if repository is not None
        else DEFAULT_REPOSITORY
    )
    if not root.is_dir():
        return None, None, str(root)
    for name in READ_CANDIDATES:
        candidate = root / name
        if candidate.is_file():
            return root, name, str(root)
    return root, None, str(root)


def demo_story(root: Path, target: str) -> Dict[str, Any]:
    """Resolve the offline story for this root + target.

    When the root carries an MCP configuration, the story is the
    agent → tool → MCP server chain the question asks about (the
    intermediate hops are kept — never collapsed).  Without MCP the
    story stays honest: the source endpoint is identified, no path is
    invented, and the diagram carries exactly one node.
    """
    source = Path(target).stem
    tool: Optional[str] = None
    server: Optional[str] = None
    config = root / MCP_CONFIG_NAME
    if config.is_file():
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}
        servers = data.get("mcpServers") or {}
        if servers:
            server = sorted(servers)[0]
            names = servers[server].get("tools") or []
            tool = names[0] if names else "{}_tool".format(server)
    tool_path = (
        TOOL_CANDIDATE if (root / TOOL_CANDIDATE).is_file() else None
    )
    mcp = tool is not None and server is not None
    arch_reads = [target] + (
        [tool_path] if tool_path else []
    ) + ([MCP_CONFIG_NAME] if mcp else [])
    flow_reads = [target] + ([MCP_CONFIG_NAME] if mcp else [])
    if mcp:
        legs = [
            (source, tool, "IMPORTS", [target]),
            (tool, server, "EXPOSED_BY", [MCP_CONFIG_NAME]),
        ]
        reads = _unique(arch_reads + flow_reads)
        return {
            "target": target,
            "tool_path": tool_path,
            "mcp": True,
            "source": source,
            "destination": server,
            "tool": tool,
            "server": server,
            "legs": legs,
            "arch_reads": _unique(arch_reads),
            "flow_reads": _unique(flow_reads),
            "reads": reads,
        }
    return {
        "target": target,
        "tool_path": tool_path,
        "mcp": False,
        "source": source,
        "destination": None,
        "tool": None,
        "server": None,
        "legs": [],
        "arch_reads": _unique(arch_reads),
        "flow_reads": _unique(flow_reads),
        "reads": _unique(arch_reads),
    }


# -- offline payloads ------------------------------------------------------


def offline_arch_payload(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> Dict[str, Any]:
    """The DIRECT-mode ArchitectureReport over this story — every
    reference points at a file this run actually reads."""
    source = story["source"]
    components: List[Dict[str, Any]] = [
        {
            "id": source,
            "name": source.capitalize(),
            "type": "agent",
            "responsibility": "runs the recorded workflow",
            "evidence_refs": [_ref(story["target"])],
            "classification": "observed",
        }
    ]
    relationships: List[Dict[str, Any]] = []
    if story["mcp"]:
        tool = story["tool"]
        server = story["server"]
        tool_ref = story["tool_path"] or MCP_CONFIG_NAME
        components.extend(
            [
                {
                    "id": tool,
                    "name": tool,
                    "type": "tool",
                    "responsibility": "performs one recorded operation",
                    "evidence_refs": [_ref(tool_ref)],
                    "classification": "observed",
                },
                {
                    "id": server,
                    "name": server,
                    "type": "mcp_server",
                    "responsibility": "serves {} over MCP".format(tool),
                    "evidence_refs": [_ref(MCP_CONFIG_NAME)],
                    "classification": "observed",
                },
            ]
        )
        relationships.extend(
            [
                {
                    "source": source,
                    "target": tool,
                    "relationship_type": "DECLARES",
                    "evidence_refs": [_ref(story["target"])],
                    "classification": "observed",
                    "confidence": "high",
                },
                {
                    "source": tool,
                    "target": server,
                    "relationship_type": "EXPOSED_BY",
                    "evidence_refs": [_ref(MCP_CONFIG_NAME)],
                    "classification": "derived",
                    "confidence": "medium",
                },
            ]
        )
    findings = [
        {
            "statement": (
                "{} declares its tooling (offline scripted finding)"
                .format(source)
                if story["mcp"]
                else "{} is the entry component "
                "(offline scripted finding)".format(source)
            ),
            "evidence_refs": [_ref(story["target"])],
            "support": STATUS_SUPPORTED,
        }
    ]
    return {
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "summary": (
            "Read-only inspection established the chain "
            "{source} -> {tool} -> {server} "
            "(offline scripted report).".format(
                source=source, tool=story["tool"],
                server=story["server"],
            )
            if story["mcp"]
            else "Read-only inspection of {target} identified the "
            "entry component; no MCP configuration exists "
            "(offline scripted report).".format(target=story["target"])
        ),
        "components": components,
        "relationships": relationships,
        "findings": findings,
        "evidence_refs": [_ref(path) for path in story["arch_reads"]],
        "uncertainties": (
            []
            if story["mcp"]
            else [
                "no MCP configuration exists in this repository, so "
                "no tool or server hop is established"
            ]
        ),
        "status": "complete",
    }


def offline_flow_payload(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> Dict[str, Any]:
    """The DIRECT-mode FlowDependencyReport over this story — the
    ordered path with every intermediate hop preserved."""
    path = [
        _step(source, target, kind, list(paths))
        for source, target, kind, paths in story["legs"]
    ]
    evidence_paths = story["flow_reads"]
    return {
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "source": story["source"],
        "target": story["destination"],
        "path_status": "FOUND" if path else "UNKNOWN",
        "summary": (
            "Read-only inspection of {target} +{mcp} established the "
            "path {source} -> {tool} -> {server} "
            "(offline scripted report).".format(
                target=story["target"],
                mcp=MCP_CONFIG_NAME,
                source=story["source"],
                tool=story["tool"],
                server=story["server"],
            )
            if story["mcp"]
            else "Read-only inspection of {target} identified the "
            "source endpoint; no destination hop is established "
            "(offline scripted report).".format(target=story["target"])
        ),
        "path": path,
        "alternative_paths": [],
        "relationships": list(path),
        "findings": [
            {
                "statement": (
                    "{} is the flow's source endpoint "
                    "(offline scripted finding).".format(story["source"])
                ),
                "evidence_refs": [_ref(story["target"])],
                "support": STATUS_SUPPORTED,
            }
        ],
        "evidence_refs": [_ref(item) for item in evidence_paths],
        "uncertainties": (
            []
            if story["mcp"]
            else [
                "no MCP configuration exists in this repository, so "
                "the flow's destination could not be established "
                "from static evidence"
            ]
        ),
        "status": "complete",
    }


def offline_spec_payload(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> Dict[str, Any]:
    """The DiagramSpec payload the scripted model returns — every
    node, edge and reference exists in the supplied reports."""
    source = story["source"]
    nodes: List[Dict[str, Any]] = [
        {
            "id": source,
            "label": source,
            "node_type": "agent",
            "classification": "observed",
            "confidence": "high",
            "evidence_refs": [_ref(story["target"])],
        }
    ]
    edges: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    legend: List[Dict[str, Any]] = [
        {
            "kind": "node_type",
            "term": "agent",
            "meaning": "autonomous actor",
        }
    ]
    uncertainties: List[str] = []
    if story["mcp"]:
        tool = story["tool"]
        server = story["server"]
        tool_ref = story["tool_path"] or MCP_CONFIG_NAME
        nodes.extend(
            [
                {
                    "id": tool,
                    "label": tool,
                    "node_type": "tool",
                    "classification": "observed",
                    "confidence": "high",
                    "evidence_refs": [_ref(tool_ref)],
                },
                {
                    "id": server,
                    "label": server,
                    "node_type": "mcp_server",
                    "classification": "observed",
                    "confidence": "high",
                    "evidence_refs": [_ref(MCP_CONFIG_NAME)],
                },
            ]
        )
        edges.extend(
            [
                {
                    "id": "{}-declares-{}".format(source, tool),
                    "source": source,
                    "target": tool,
                    "relationship_type": "DECLARES",
                    "classification": "observed",
                    "confidence": "high",
                    "evidence_refs": [_ref(story["target"])],
                },
                {
                    "id": "{}-exposed_by-{}".format(tool, server),
                    "source": tool,
                    "target": server,
                    "relationship_type": "EXPOSED_BY",
                    "classification": "derived",
                    "confidence": "medium",
                    "evidence_refs": [_ref(MCP_CONFIG_NAME)],
                },
            ]
        )
        groups = [
            {
                "id": "agent-side",
                "label": "Agent side",
                "node_ids": [source, tool],
            }
        ]
        legend.append(
            {
                "kind": "relationship_type",
                "term": "EXPOSED_BY",
                "meaning": "exposed through a server",
            }
        )
        uncertainties = [
            "the wire format between {} and the {} MCP server is not "
            "established".format(tool, server)
        ]
        title = "{} through {} to the {} MCP server".format(
            source, tool, server
        )
        diagram_type = "AGENT_TOOL_MCP"
    else:
        title = "{} architecture".format(source)
        diagram_type = "COMPONENT"
        uncertainties = [
            "no MCP configuration exists in this repository, so the "
            "destination of {} could not be established".format(source)
        ]
    return {
        "diagram_type": diagram_type,
        "title": title,
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "legend": legend,
        "evidence_refs": [
            _ref(path) for path in story["reads"]
        ],
        "uncertainties": uncertainties,
        "status": "complete",
    }


def offline_plan_payload(question: str) -> Dict[str, Any]:
    """Three steps: two INVESTIGATOR goals (architecture, flow) then
    one ANALYST goal (the role assigner reads the goal pattern)."""
    return {
        "objective": question,
        "steps": [
            {
                "id": "S1",
                "goal": "Document the architecture components and "
                "their relationships in this repository.",
            },
            {
                "id": "S2",
                "goal": "Trace the path from the collector agent to "
                "the records MCP server.",
            },
            {
                "id": "S3",
                "goal": "Interpret the investigated findings as one "
                "diagram for the user's question.",
            },
        ],
    }


# -- offline models --------------------------------------------------------


def _read_calls(paths: List[str], suffix: str) -> LLMResponse:
    return _tool_calls(
        *[
            ToolCall(
                id="call_viz_read_{}_{}".format(suffix, index),
                name="read_file",
                arguments={"path": path},
            )
            for index, path in enumerate(paths)
        ]
    )


def build_offline_models(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> Dict[str, Any]:
    """Every scripted participant: the three DIRECT runs, the
    planner, and the three ORCHESTRATED runs (separate queues —
    each phase consumes exactly its own responses)."""
    arch_payload = offline_arch_payload(
        question, repository, commit, story
    )
    flow_payload = offline_flow_payload(
        question, repository, commit, story
    )
    spec_payload = offline_spec_payload(
        question, repository, commit, story
    )
    return {
        "arch": OfflineModel(
            _read_calls(story["arch_reads"], "arch"),
            _structured(arch_payload),
        ),
        "flow": OfflineModel(
            _read_calls(story["flow_reads"], "flow"),
            _structured(flow_payload),
        ),
        "viz": OfflineModel(_structured(spec_payload)),
        "planner": OfflineModel(_structured(
            offline_plan_payload(question)
        )),
        "m17_arch": OfflineModel(
            _read_calls(story["arch_reads"], "m17_arch"),
            _structured(arch_payload),
        ),
        "m17_flow": OfflineModel(
            _read_calls(story["flow_reads"], "m17_flow"),
            _structured(flow_payload),
        ),
        "m17_viz": OfflineModel(_structured(spec_payload)),
    }


def build_live_models(model_id: str) -> Dict[str, Any]:
    """--live: a real model for every participant (the story scripts
    do not apply; the specialists read the repository for real)."""
    clients = {
        key: OpenAILLMClient(model=model_id)
        for key in (
            "arch",
            "flow",
            "viz",
            "planner",
            "m17_arch",
            "m17_flow",
            "m17_viz",
        )
    }
    return clients


# -- composition -----------------------------------------------------------


def fixture_bindings(root: Path) -> Dict[str, Any]:
    """Bind the read-only text tools to the investigation root via the
    documented ``tool(arguments)`` closure seam; graph/context tools
    answer with canned successes."""

    def read(arguments):
        return read_file(arguments, repository_root=root)

    def search(arguments):
        return search_repository(arguments, repository_root=root)

    def generic(name):
        def tool(arguments):
            return ToolResult(
                success=True,
                tool=name,
                data={"path": "", "content": ""},
            )

        return tool

    return {
        "read_file": read,
        "search_repository": search,
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def build_memory_manager(db_path: Optional[str] = None) -> MemoryManager:
    manager = MemoryManager(
        db_path if db_path is not None else str(DEMO_DB_PATH)
    )
    manager.initialize()
    return manager


def build_memory_context(
    repository: str, current_commit: str
) -> MemoryContext:
    return MemoryContext(
        scope=MemoryScope(repository=repository, commit=current_commit),
        subject=SUBJECT,
        owner=MemoryOwner(
            type="agent",
            id="repo_intelligence.agents.architecture_visualization",
        ),
    )


def build_engine(
    repository: str, snapshot_root: Optional[str] = None
) -> ContextEngine:
    return ContextEngine(
        budget=DEFAULT_BUDGET,
        repository=repository,
        snapshot_root=snapshot_root,
    )


def build_architect(
    llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
) -> ArchitectureInvestigator:
    """The Architecture Investigator over the existing Agent runtime —
    the ONE composition the direct phase and the orchestrator use."""
    agent = Agent(
        llm,
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=bindings
        ),
        trace_sink=trace_sink,
        structured_output_model=ArchitectureReport,
        tool_definitions=tool_definitions_for("investigator"),
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )
    return ArchitectureInvestigator(
        agent=agent, tools=ARCHITECTURE_TOOLS
    )


def build_flow_investigator(
    llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
) -> FlowDependencyInvestigator:
    agent = Agent(
        llm,
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=bindings
        ),
        trace_sink=trace_sink,
        structured_output_model=FlowDependencyReport,
        tool_definitions=tool_definitions_for("investigator"),
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )
    return FlowDependencyInvestigator(
        agent=agent, tools=FLOW_DEPENDENCY_TOOLS
    )


def build_visualizer(
    llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
) -> ArchitectureVisualizationAgent:
    """The visualizer over the existing Agent runtime with NO tools —
    the analyst-slot composition both paths use."""
    agent = Agent(
        llm,
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=DiagramSpec,
        tool_definitions=tool_definitions_for("analyst"),
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )
    return ArchitectureVisualizationAgent(agent=agent, tools=())


class RoutedInvestigatorError(Exception):
    """A question with no routed step — explicit, never a guess."""


class RoutedInvestigator:
    """Demo adapter: one investigator slot, two specialists.

    The generic orchestrator holds exactly ONE investigator, so this
    router reads the plan's step label the runner embedded
    (``step: <id> — <goal>``) and delegates to that step's
    specialist: by the plan's own step id first (the offline plan's
    S1/S2), then by the goal's content — the fallback a live planner's
    step ids need.  Retry-safe and trace-safe; neither the
    orchestrator nor the runner knows a router exists, so the
    orchestration infrastructure stays unmodified.
    """

    def __init__(self, route: Dict[str, Any]) -> None:
        self.route = dict(route)
        self._active: Optional[Any] = None
        self.calls: List[str] = []

    @property
    def last_trace(self) -> Any:
        return self._active.last_trace if self._active is not None else None

    def _by_goal(self, goal: str) -> Any:
        lower = goal.lower()
        if any(
            word in lower
            for word in ("architecture", "component", "relationship")
        ):
            return self.route.get("arch")
        if any(
            word in lower
            for word in ("path", "trace", "reach", "dependency")
        ):
            return self.route.get("flow")
        return None

    def run_result(self, question: str) -> Any:
        match = _STEP_LABEL.search(question)
        step_id = match.group(1) if match is not None else None
        goal = match.group(2) if match is not None else ""
        specialist = self.route.get(step_id or "") or self._by_goal(goal)
        if specialist is None:
            raise RoutedInvestigatorError(
                "no specialist routed for step {!r}".format(step_id)
            )
        self._active = specialist
        self.calls.append(step_id or "")
        return specialist.run_result(question)


# -- phases ----------------------------------------------------------------


def run_scope_phase(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    visualizer: ArchitectureVisualizationAgent,
    stream=None,
) -> Dict[str, Any]:
    """Phase 1 — the question, the repository-agnostic scope, the
    versioned prompt, and the one visible tool list."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print("Phase 1 — question & scope", file=stream)
    print(RULE, file=stream)
    print("question:   {}".format(_clip(question, 120)), file=stream)
    print(
        "scope:      repository={} commit={}".format(
            repository, commit[:12]
        ),
        file=stream,
    )
    print("target:     {}".format(story["target"]), file=stream)
    print(
        "story:      {}".format(
            "agent -> tool -> MCP server"
            if story["mcp"]
            else "source endpoint only (destination UNKNOWN)"
        ),
        file=stream,
    )
    print(
        "prompt:     ARCHITECTURE_VISUALIZATION_PROMPT_V{} ({})".format(
            VISUALIZATION_PROMPT_VERSION[1:],
            len(ARCHITECTURE_VISUALIZATION_PROMPT_V1),
        ),
        file=stream,
    )
    print(
        "tools:      architecture={} flow={} visualizer={}".format(
            len(ARCHITECTURE_TOOLS),
            len(FLOW_DEPENDENCY_TOOLS),
            "(none)" if not visualizer.tools
            else ", ".join(visualizer.tools),
        ),
        file=stream,
    )
    ok = (
        VISUALIZATION_PROMPT_VERSION == "V1"
        and VISUALIZATION_TOOLS == ()
        and visualizer.tools == []
        and len(ARCHITECTURE_TOOLS) > 0
        and len(FLOW_DEPENDENCY_TOOLS) > 0
        and "DiagramSpec" in ARCHITECTURE_VISUALIZATION_PROMPT_V1
        and "orchestrat"
        not in ARCHITECTURE_VISUALIZATION_PROMPT_V1.lower()
        and "planner"
        not in ARCHITECTURE_VISUALIZATION_PROMPT_V1.lower()
    )
    print(
        "scope condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok}


def run_architecture_phase(
    architect: ArchitectureInvestigator,
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    stream=None,
) -> Dict[str, Any]:
    """Phase 2 — ArchitectureInvestigator → ArchitectureReport."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 2 — architecture investigation "
        "(investigate -> ArchitectureReport)",
        file=stream,
    )
    print(RULE, file=stream)
    report = architect.investigate(question, repository, commit)
    print(
        "investigation id: {}".format(
            report.derived_investigation_id()
        ),
        file=stream,
    )
    print(
        "scope:            repository={} commit={}".format(
            report.repository_scope, report.commit_scope[:12]
        ),
        file=stream,
    )
    print(
        "components:       {} ({})".format(
            len(report.components),
            ", ".join(component.id for component in report.components),
        ),
        file=stream,
    )
    print(
        "relationships:    {} ({})".format(
            len(report.relationships),
            "; ".join(
                "{} {} {}".format(
                    relationship.source,
                    relationship.relationship_type,
                    relationship.target,
                )
                for relationship in report.relationships
            )
            or "none",
        ),
        file=stream,
    )
    print(
        "findings:         {} statement(s)".format(len(report.findings)),
        file=stream,
    )
    ids = {component.id for component in report.components}
    expected = {story["source"]}
    if story["mcp"]:
        expected |= {story["tool"], story["server"]}
    ok = (
        isinstance(report, ArchitectureReport)
        and report.status == "complete"
        and report.repository_scope == repository
        and report.commit_scope == commit
        and expected <= ids
    )
    print(
        "architecture condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {"ok": ok, "report": report}


def run_flow_phase(
    investigator: FlowDependencyInvestigator,
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    stream=None,
) -> Dict[str, Any]:
    """Phase 3 — FlowDependencyInvestigator → FlowDependencyReport."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 3 — flow investigation "
        "(investigate -> FlowDependencyReport)",
        file=stream,
    )
    print(RULE, file=stream)
    report = investigator.investigate(question, repository, commit)
    print(
        "investigation id: {}".format(
            report.derived_investigation_id()
        ),
        file=stream,
    )
    print(
        "path_status:      {}".format(report.path_status), file=stream
    )
    print(
        "endpoints:        {} -> {}".format(
            report.source,
            report.target if report.target is not None else "UNKNOWN",
        ),
        file=stream,
    )
    print(
        "steps:            {}".format(
            "; ".join(
                "{} {} {}".format(
                    step.source, step.relationship_type, step.target
                )
                for step in report.path
            )
            or "none",
        ),
        file=stream,
    )
    hops_ok = (
        [
            "{} {} {}".format(
                step.source, step.relationship_type, step.target
            )
            for step in report.path
        ]
        == [
            "{} {} {}".format(
                source, kind, target
            )
            for source, target, kind, _ in story["legs"]
        ]
    )
    if story["mcp"]:
        status_ok = report.path_status == "FOUND"
        endpoints_ok = (
            report.source == story["source"]
            and report.target == story["server"]
        )
    else:
        status_ok = report.path_status in PATH_STATUSES
        endpoints_ok = report.target is None
    ok = (
        isinstance(report, FlowDependencyReport)
        and report.status == "complete"
        and report.repository_scope == repository
        and report.commit_scope == commit
        and status_ok
        and endpoints_ok
        and hops_ok
    )
    print(
        "flow condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok, "report": report}


def run_visualization_phase(
    visualizer: ArchitectureVisualizationAgent,
    architecture_report: ArchitectureReport,
    flow_report: FlowDependencyReport,
    question: str,
    repository: str,
    commit: str,
    diagram_type: str,
    stream=None,
) -> Dict[str, Any]:
    """Phase 4 — ArchitectureVisualizationAgent → DiagramSpec over the
    typed request (structured outputs, never raw repository content)."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 4 — visualization (request -> DiagramSpec)",
        file=stream,
    )
    print(RULE, file=stream)
    request = VisualizationRequest(
        question=question,
        diagram_type=diagram_type,
        architecture_report=architecture_report,
        flow_report=flow_report,
    )
    spec = visualizer.visualize(request)
    print("diagram_type: {}".format(spec.diagram_type), file=stream)
    print("title:        {}".format(_clip(spec.title)), file=stream)
    print(
        "nodes:        {} ({})".format(
            len(spec.nodes),
            ", ".join(
                "{} [{}]".format(node.id, node.node_type)
                for node in spec.nodes
            ),
        ),
        file=stream,
    )
    print(
        "edges:        {} ({})".format(
            len(spec.edges),
            "; ".join(
                "{} {} {}".format(
                    edge.source, edge.relationship_type, edge.target
                )
                for edge in spec.edges
            )
            or "none",
        ),
        file=stream,
    )
    print(
        "groups: {}  legend: {}  evidence refs: {}  uncertainties: "
        "{}".format(
            len(spec.groups),
            len(spec.legend),
            len(spec.evidence_refs),
            len(spec.uncertainties),
        ),
        file=stream,
    )
    scope = DiagramScope.from_request(request)
    envelope_ok = (
        spec.question == request.question
        and spec.repository_scope == repository
        and spec.commit_scope == commit
    )
    ids = {node.id for node in spec.nodes}
    ok = (
        isinstance(spec, DiagramSpec)
        and spec.diagram_type == request.diagram_type
        and spec.status == "complete"
        and envelope_ok
        and ids <= set(scope.node_ids)
        and visualizer.tools == []
    )
    print(
        "envelope from request: {}".format(
            "yes" if envelope_ok else "NO"
        ),
        file=stream,
    )
    print(
        "visualization condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {"ok": ok, "spec": spec, "request": request}


def run_validation_phase(
    visualizer: ArchitectureVisualizationAgent,
    spec: DiagramSpec,
    request: VisualizationRequest,
    architect: ArchitectureInvestigator,
    investigator: FlowDependencyInvestigator,
    stream=None,
) -> Dict[str, Any]:
    """Phase 5 — the deterministic six-rule verdict (no model) over
    the investigators' own evidence."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 5 — validation (six deterministic rules, no model)",
        file=stream,
    )
    print(RULE, file=stream)
    scope = DiagramScope.from_request(request)
    merged_trace = {
        "iterations": [
            iteration.to_dict()
            for trace in (architect.last_trace, investigator.last_trace)
            if trace is not None
            for iteration in trace.iterations
        ]
    }
    evidence = gather_evidence(merged_trace)
    print(
        "scope: {} node(s), {} edge(s), {} evidence key(s), "
        "diagram_type {}".format(
            len(scope.node_ids),
            len(scope.edges),
            len(scope.evidence_keys),
            scope.requested_diagram_type,
        ),
        file=stream,
    )
    print(
        "evidence: {} inspected path(s) from the investigators' "
        "traces (the visualizer reads nothing)".format(
            len(evidence.inspected)
        ),
        file=stream,
    )
    result = visualizer.validate(spec, request=request, evidence=evidence)
    for check in result.checks:
        print(
            "  [{}] {:<24} {}".format(
                "ok" if check.ok else "FAIL",
                check.rule,
                check.detail,
            ),
            file=stream,
        )
    print("validation: {}".format(result.status), file=stream)
    ok = result.status == "passed" and all(
        check.ok for check in result.checks
    )
    print(
        "validation condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {"ok": ok, "result": result, "scope": scope}


def run_rendering_phase(spec: DiagramSpec, stream=None) -> Dict[str, Any]:
    """Phase 6 — the EXISTING M5 renderer: deterministic Mermaid from
    the spec projection; SVG only where Graphviz is installed."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 6 — rendering (existing M5 renderer, deterministic)",
        file=stream,
    )
    print(RULE, file=stream)
    projection = spec.to_diagram_projection()
    mermaid = render_mermaid_text(projection)
    rerendered = render_mermaid_text(spec.to_diagram_projection())
    print(
        "mermaid ({} lines):".format(len(mermaid.splitlines())),
        file=stream,
    )
    for line in mermaid.splitlines():
        print("  {}".format(line), file=stream)
    if graphviz_available():
        print(
            "svg: available (Graphviz dot found) — written in Phase 7",
            file=stream,
        )
    else:
        print("svg: {}".format(SVG_UNAVAILABLE_NOTICE), file=stream)
    ids = {node.id for node in spec.nodes}
    ok = (
        "flowchart TD" in mermaid
        and rerendered == mermaid
        and len(ids) == len(spec.nodes)
        and all(node.label in mermaid for node in spec.nodes)
        and all(
            edge.relationship_type in mermaid for edge in spec.edges
        )
    )
    print(
        "deterministic re-render: {}".format(
            "identical" if rerendered == mermaid else "DIFFERENT"
        ),
        file=stream,
    )
    print(
        "rendering condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {"ok": ok, "mermaid": mermaid}


def run_artifacts_phase(
    spec: DiagramSpec,
    validation,
    architecture_report: ArchitectureReport,
    flow_report: FlowDependencyReport,
    trace_root: Path,
    directory: Optional[str],
    stream=None,
) -> Dict[str, Any]:
    """Phase 7 — the four persistent artifacts of ONE validated spec,
    verified file by file (spec §25)."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 7 — artifact persistence (validated spec only)",
        file=stream,
    )
    print(RULE, file=stream)
    artifacts = write_visualization_artifacts(
        spec,
        validation,
        context=RunSummaryContext(
            architecture_report=architecture_report,
            flow_report=flow_report,
            evidence_notes=[],
            m17_agents=[],
            m17_status="pending — run in Phase 8",
            trace_location=str(trace_root),
        ),
        directory=directory,
    )
    print("directory: {}".format(artifacts.directory), file=stream)
    statuses = verify_artifacts(artifacts)
    ok = True
    for status in statuses:
        if status.name == "architecture.svg" and not status.exists:
            print(
                "  {:<20} exists: NO   {}".format(
                    status.name, status.note or "not produced"
                ),
                file=stream,
            )
            ok = ok and not artifacts.svg_available
            continue
        print(
            "  {:<20} exists: {} ({} bytes)".format(
                status.name,
                "yes" if status.exists else "NO",
                status.bytes,
            ),
            file=stream,
        )
        ok = ok and status.exists and status.bytes > 0
    print(
        "artifacts condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {"ok": ok, "artifacts": artifacts, "statuses": statuses}


def run_m17_phase(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    models: Dict[str, Any],
    composition: Dict[str, Any],
    direct: Dict[str, Any],
    trace_root: Path,
    artifact_directory: Optional[str],
    stream=None,
) -> Dict[str, Any]:
    """Phase 8 — the SAME capability through M10 → M17 → M16: plan,
    role assignment, routed specialists, typed handoff, deterministic
    validation — then the run_summary is refreshed with these facts."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 8 — orchestrated execution "
        "(plan -> routed investigators -> visualizer -> validator)",
        file=stream,
    )
    print(RULE, file=stream)

    planner_response = models["planner"].complete(
        build_planning_request(question)
    )
    investigation_plan = plan_from_response(planner_response)
    validate_plan(investigation_plan)
    print(
        "plan: {} steps — {}".format(
            len(investigation_plan.steps),
            ", ".join(step.id for step in investigation_plan.steps),
        ),
        file=stream,
    )

    architect = build_architect(
        models["m17_arch"],
        composition["sink"],
        composition["manager"],
        composition["memory_context"],
        composition["engine"],
        bindings=composition["bindings"],
    )
    flow_investigator = build_flow_investigator(
        models["m17_flow"],
        composition["sink"],
        composition["manager"],
        composition["memory_context"],
        composition["engine"],
        bindings=composition["bindings"],
    )
    visualizer = build_visualizer(
        models["m17_viz"],
        composition["sink"],
        composition["manager"],
        composition["memory_context"],
        composition["engine"],
    )
    router = RoutedInvestigator(
        {
            "S1": architect,
            "S2": flow_investigator,
            "arch": architect,
            "flow": flow_investigator,
        }
    )
    orchestrator = Orchestrator(
        router, visualizer, trace_sink=composition["sink"]
    )
    task = TaskPayload(
        question=question,
        repository_scope=repository,
        commit_scope=commit,
    )
    run = orchestrator.execute(task, investigation_plan)

    assignments = [
        event
        for event in run.events
        if event["operation"] == "orchestration.step.assign"
    ]
    roles = [event["agent_role"] for event in assignments]
    print("roles: {}".format(" \u2192 ".join(roles) or "none"),
          file=stream)
    for step, assign in zip(run.plan.steps, assignments):
        print(
            "  {} {:<12} {:<14} {}".format(
                step.id,
                assign["agent_role"],
                step.status,
                _clip(step.goal),
            ),
            file=stream,
        )
    for result in run.step_results:
        print(
            "  {} {} -> {}".format(
                result.step_id,
                result.agent_role,
                type(result.output).__name__,
            ),
            file=stream,
        )

    if run.report is not None:
        print(
            "merged report: {} statement(s), {} evidence ref(s)".format(
                len(run.report.findings),
                len(run.report.evidence_refs),
            ),
            file=stream,
        )
    spec = visualizer.last_spec
    if spec is not None:
        print(
            "diagram: {} node(s), {} edge(s)".format(
                len(spec.nodes), len(spec.edges)
            ),
            file=stream,
        )
    if run.validation is not None:
        print("validation: {}".format(run.validation.status),
              file=stream)
        for check in run.validation.checks:
            print(
                "  [{}] {:<18} {}".format(
                    "ok" if check.ok else "FAIL",
                    check.rule,
                    "" if check.ok else _clip(check.detail, 160),
                ),
                file=stream,
            )

    refreshed = False
    if (
        direct.get("spec") is not None
        and direct.get("validation") is not None
        and direct["validation"].status == "passed"
    ):
        try:
            write_visualization_artifacts(
                direct["spec"],
                direct["validation"],
                context=RunSummaryContext(
                    architecture_report=direct.get(
                        "architecture_report"
                    ),
                    flow_report=direct.get("flow_report"),
                    evidence_notes=[],
                    m17_agents=[
                        "INVESTIGATOR (architecture investigation)",
                        "INVESTIGATOR (flow / dependency "
                        "investigation)",
                        "ANALYST (visualization handoff)",
                        "validator (deterministic M16 verdict)",
                        "renderer (deterministic Mermaid/DOT/SVG "
                        "stage)",
                    ],
                    m17_status=run.status,
                    trace_location=str(trace_root),
                ),
                directory=artifact_directory,
            )
            refreshed = True
            print(
                "run_summary.md: refreshed with the M17 facts "
                "(status: {})".format(run.status),
                file=stream,
            )
        except VisualizationArtifactError as exc:
            print(
                "run_summary.md refresh: NO ({})".format(exc),
                file=stream,
            )
    else:
        print(
            "run_summary.md refresh: skipped (the direct validation "
            "did not pass)",
            file=stream,
        )

    expected_roles = ["INVESTIGATOR", "INVESTIGATOR", "ANALYST"]
    steps_ok = all(
        step.status == "completed" for step in run.plan.steps
    )
    outputs_ok = [
        type(result.output).__name__
        for result in run.step_results
    ] == [
        "InvestigationReport",
        "InvestigationReport",
        "ArchitectureFinding",
    ]
    checks_ok = (
        run.validation is not None
        and run.validation.status == "passed"
        and all(check.ok for check in run.validation.checks)
    )
    ok = (
        run.status == COMPLETED
        and run.termination_reason == "SUCCESS"
        and roles == expected_roles
        and steps_ok
        and outputs_ok
        and checks_ok
        and isinstance(run.finding, ArchitectureFinding)
        and spec is not None
        and router.calls == ["S1", "S2"]
        and refreshed
    )
    print(
        "status: {}   termination: {}   tool calls: {}".format(
            run.status, run.termination_reason, run.tool_calls_total
        ),
        file=stream,
    )
    print(
        "orchestrated condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {
        "ok": ok,
        "run": run,
        "visualizer": visualizer,
        "roles": roles,
        "refreshed": refreshed,
    }


def run_trace_phase(
    direct_traces: Dict[str, Any],
    m17_visualizer: ArchitectureVisualizationAgent,
    run,
    trace_root: Path,
    summary_path: Optional[str],
    stream=None,
) -> Dict[str, Any]:
    """Phase 9 — the ONE trace system, the repository's ACTUAL role
    names, and the persisted summary checked against the run."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 9 — trace (ONE trace system; actual role names)",
        file=stream,
    )
    print(RULE, file=stream)

    print("role chain:", file=stream)
    print("  INVESTIGATOR  (architecture investigation)", file=stream)
    print(
        "  INVESTIGATOR  (flow / dependency investigation)", file=stream
    )
    print("  ANALYST       (visualization handoff)", file=stream)
    print(
        "  validator     (deterministic M16 verdict)", file=stream
    )
    print(
        "  renderer      (deterministic Mermaid / DOT / SVG stage)",
        file=stream,
    )

    disk_ok = True
    for label, trace in direct_traces.items():
        if trace is None:
            print("trace ({}): MISSING".format(label), file=stream)
            disk_ok = False
            continue
        trace_path = Path(trace_root) / trace.run_id / "trace.json"
        markdown_path = Path(trace_root) / trace.run_id / "trace.md"
        has_json = trace_path.is_file()
        markdown = (
            markdown_path.read_text(encoding="utf-8")
            if markdown_path.is_file()
            else ""
        )
        print(
            "trace ({}): {} json={} md={} run-summary={}".format(
                label,
                trace.run_id,
                has_json,
                bool(markdown),
                "## Run Summary" in markdown,
            ),
            file=stream,
        )
        disk_ok = (
            disk_ok
            and has_json
            and bool(markdown)
            and "## Run Summary" in markdown
        )

    orchestrated = m17_visualizer.last_trace
    markdown_ok = True
    if orchestrated is None:
        print("trace (orchestrated visualizer): MISSING", file=stream)
        disk_ok = False
        markdown_ok = False
    else:
        trace_path = (
            Path(trace_root) / orchestrated.run_id / "trace.json"
        )
        markdown_path = (
            Path(trace_root) / orchestrated.run_id / "trace.md"
        )
        markdown = (
            markdown_path.read_text(encoding="utf-8")
            if markdown_path.is_file()
            else ""
        )
        print(
            "trace (orchestrated visualizer): {} json={} md={} "
            "orchestration={}".format(
                orchestrated.run_id,
                trace_path.is_file(),
                bool(markdown),
                "## Orchestration" in markdown,
            ),
            file=stream,
        )
        disk_ok = (
            disk_ok
            and trace_path.is_file()
            and bool(markdown)
        )
        markdown_ok = "## Orchestration" in markdown

    mirrored_count = (
        len(
            getattr(
                orchestrated, "orchestration_events", None
            )
            or []
        )
        if orchestrated is not None
        else 0
    )
    sequences = [event["sequence"] for event in run.events]
    ordered = (
        sequences == sorted(sequences)
        and len(sequences) == len(set(sequences))
    )
    in_memory = render_markdown(m17_visualizer.last_trace)
    print(
        "mirrored orchestration events: {} (markdown: {} chars)".format(
            mirrored_count, len(in_memory)
        ),
        file=stream,
    )
    print(
        "event sequences monotonic + unique: {}".format(
            "yes" if ordered else "NO"
        ),
        file=stream,
    )
    print(
        "role chain in events: {}".format(
            " \u2192 ".join(
                event["agent_role"]
                for event in run.events
                if event["operation"] == "orchestration.step.assign"
            )
            or "none"
        ),
        file=stream,
    )

    summary_ok = False
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary_text = (
            summary_file.read_text(encoding="utf-8")
            if summary_file.is_file()
            else ""
        )
        expected_line = "Status: {}".format(run.status)
        summary_ok = (
            "## M17 Execution" in summary_text
            and expected_line in summary_text
        )
        print(
            "run_summary.md M17 status line: {}".format(
                "present ({})".format(run.status)
                if summary_ok
                else "MISSING"
            ),
            file=stream,
        )

    ok = (
        disk_ok
        and markdown_ok
        and mirrored_count > 0
        and ordered
        and "## Orchestration" in in_memory
        and "orchestration.step.assign" in in_memory
        and summary_ok
    )
    print(
        "trace condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok}


# -- rendering helpers -----------------------------------------------------


def render_mermaid_text(projection) -> str:
    from repo_intelligence.graph.exporters import render_mermaid

    return render_mermaid(projection)


def graphviz_available() -> bool:
    from repo_intelligence.graph.exporters import (
        graphviz_available as available,
    )

    return available()


# -- demo ------------------------------------------------------------------


def run_demo(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    models: Dict[str, Any],
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]],
    trace_root: Path,
    artifact_directory: Optional[str],
    stream=None,
) -> Dict[str, Any]:
    """Every phase over one shared story; returns each outcome plus
    the aggregate ``ok``."""
    stream = stream or sys.stdout

    composition = {
        "sink": trace_sink,
        "manager": memory_manager,
        "memory_context": memory_context,
        "engine": context_engine,
        "bindings": bindings,
    }

    architect = build_architect(
        models["arch"],
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
        bindings=bindings,
    )
    flow_investigator = build_flow_investigator(
        models["flow"],
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
        bindings=bindings,
    )
    visualizer = build_visualizer(
        models["viz"],
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
    )

    scope = run_scope_phase(
        question, repository, commit, story, visualizer, stream=stream
    )
    architecture = run_architecture_phase(
        architect, question, repository, commit, story, stream=stream
    )
    flow = run_flow_phase(
        flow_investigator,
        question,
        repository,
        commit,
        story,
        stream=stream,
    )
    if (
        architecture.get("report") is not None
        and flow.get("report") is not None
    ):
        visualization = run_visualization_phase(
            visualizer,
            architecture["report"],
            flow["report"],
            question,
            repository,
            commit,
            "AGENT_TOOL_MCP" if story["mcp"] else "COMPONENT",
            stream=stream,
        )
    else:
        visualization = {"ok": False, "spec": None, "request": None}
    if visualization.get("request") is not None:
        validation = run_validation_phase(
            visualizer,
            visualization["spec"],
            visualization["request"],
            architect,
            flow_investigator,
            stream=stream,
        )
    else:
        validation = {"ok": False, "result": None, "scope": None}
    if visualization.get("spec") is not None:
        rendering = run_rendering_phase(
            visualization["spec"], stream=stream
        )
    else:
        rendering = {"ok": False, "mermaid": ""}
    if (
        validation.get("result") is not None
        and validation["result"].status == "passed"
        and visualization.get("spec") is not None
    ):
        artifacts = run_artifacts_phase(
            visualization["spec"],
            validation["result"],
            architecture.get("report"),
            flow.get("report"),
            trace_root,
            artifact_directory,
            stream=stream,
        )
    else:
        verdict = (
            validation["result"].status
            if validation.get("result") is not None
            else "missing"
        )
        print(RULE, file=stream)
        print(
            "Phase 7 — artifact persistence (validated spec only)",
            file=stream,
        )
        print(RULE, file=stream)
        print(
            "validation: {} — the artifacts of a rejected diagram "
            "are never written".format(verdict),
            file=stream,
        )
        print("artifacts condition: NO", file=stream)
        artifacts = {"ok": False, "artifacts": None, "statuses": []}
    orchestrated = run_m17_phase(
        question,
        repository,
        commit,
        story,
        models,
        composition,
        {
            "spec": visualization.get("spec"),
            "validation": validation.get("result"),
            "architecture_report": architecture.get("report"),
            "flow_report": flow.get("report"),
        },
        trace_root,
        artifact_directory,
        stream=stream,
    )
    if orchestrated.get("run") is not None:
        refreshed_summary: Optional[str] = None
        if artifacts.get("artifacts") is not None:
            refreshed_summary = artifacts["artifacts"].run_summary
        trace = run_trace_phase(
            {
                "direct architecture": architect.last_trace,
                "direct flow": flow_investigator.last_trace,
                "direct visualizer": visualizer.last_trace,
            },
            orchestrated["visualizer"],
            orchestrated["run"],
            trace_root,
            refreshed_summary,
            stream=stream,
        )
    else:
        trace = {"ok": False}

    phases = {
        "scope": scope,
        "architecture": architecture,
        "flow": flow,
        "visualization": visualization,
        "validation": validation,
        "rendering": rendering,
        "artifacts": artifacts,
        "m17": orchestrated,
        "trace": trace,
    }
    ok = all(entry["ok"] for entry in phases.values())

    print(RULE, file=stream)
    print(
        "Architecture Visualization Agent demonstration summary",
        file=stream,
    )
    print(RULE, file=stream)
    for name, entry in phases.items():
        print(
            "{:<15} {}".format(
                name + ":", "yes" if entry["ok"] else "NO"
            ),
            file=stream,
        )
    print(
        "all conditions: {}".format("yes" if ok else "NO"), file=stream
    )

    print(RULE, file=stream)
    print("ARTIFACTS (spec §25)", file=stream)
    print(RULE, file=stream)
    if artifacts.get("artifacts") is not None:
        for status in artifacts["statuses"]:
            if status.exists:
                print(
                    "  {:<20} exists: yes  ({} bytes)  {}".format(
                        status.name, status.bytes, status.path
                    ),
                    file=stream,
                )
            else:
                print(
                    "  {:<20} exists: NO   {}".format(
                        status.name, status.note or status.path
                    ),
                    file=stream,
                )
    else:
        print("  (no artifacts written)", file=stream)

    return dict(
        phases,
        ok=ok,
        artifact_set=artifacts.get("artifacts"),
    )


# -- main ------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_architecture_visualization_demo.py",
        description=(
            "Architecture Visualization Agent demonstration: the "
            "direct specialist path and full planner/orchestrator/"
            "validator orchestration — offline and deterministic by "
            "default."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the visualization question (default: the "
        "agent/tool/MCP reachability question)",
    )
    parser.add_argument(
        "--repository",
        default=None,
        help="repository root to investigate (default: "
        "tests/fixtures/arch_mcp — the capability's own tiny "
        "evaluation fixture)",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="commit label for the scope (default: offline synthetic "
        "commit)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="artifact directory (default: <project-root>/"
        "artifacts/architecture_visualization)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use a real model for every participant (requires "
        "OPENAI_API_KEY); default is fully offline",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id for --live (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--db",
        default=None,
        help="memory database path (default: runtime/agent_memory/"
        "architecture_visualization_demo.db — the demonstration's "
        "OWN database)",
    )
    parser.add_argument(
        "--snapshots",
        default=None,
        help="context snapshot root (default: runtime/context)",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="agent trace root (default: runtime/agent_runs)",
    )
    args = parser.parse_args(argv)

    question = args.question or DEFAULT_QUESTION

    if args.live:
        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Architecture Visualization Agent demonstration "
                "(--live) requires OPENAI_API_KEY.\n"
                "Copy .env.example to .env, set OPENAI_API_KEY (never "
                "commit .env), and re-run — or run the default offline "
                "demonstration:\n"
                "    python agentrun_architecture_visualization_demo.py",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        mode_label = "live (real model for every participant)"
    else:
        mode_label = "offline (deterministic, scripted models)"

    root, target, bound_root = demo_read_target(args.repository)
    if args.repository is not None and root is None:
        print(
            "Demonstration failed: --repository {} is not a "
            "directory.".format(args.repository),
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED
    if target is None:
        print(
            "Demonstration failed: no readable target file under {} "
            "(looked for: {}).".format(
                root, ", ".join(READ_CANDIDATES)
            ),
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED

    repository_label = root.name
    commit = args.commit or OFFLINE_COMMIT
    story = demo_story(root, target)

    if args.live:
        models = build_live_models(args.model or DEFAULT_MODEL)
    else:
        models = build_offline_models(
            question, repository_label, commit, story
        )

    try:
        manager = build_memory_manager(args.db)
    except Exception as exc:
        print("Memory store failed: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    memory_context = build_memory_context(repository_label, commit)
    engine = build_engine(repository_label, args.snapshots)
    trace_sink = (
        JsonFileTraceSink(args.traces)
        if args.traces
        else JsonFileTraceSink()
    )
    trace_root = (
        Path(args.traces)
        if args.traces
        else Path("runtime") / "agent_runs"
    )
    bindings = fixture_bindings(root)

    print(
        "Architecture Visualization Agent demonstration — {}".format(
            mode_label
        )
    )
    print(
        "scope: {}@{}   target: {}   mcp: {}".format(
            repository_label,
            commit[:12],
            target,
            "yes" if story["mcp"] else "no",
        ),
    )
    print(file=sys.stdout)

    try:
        outcome = run_demo(
            question,
            repository_label,
            commit,
            story,
            models,
            trace_sink,
            manager,
            memory_context,
            engine,
            bindings,
            trace_root,
            args.output,
        )
    except (
        AgentError,
        LLMError,
        StructuredOutputValidationError,
        ContractError,
        ArchitectureInvestigatorError,
        FlowDependencyInvestigatorError,
        ArchitectureVisualizationAgentError,
        VisualizationArtifactError,
        PlanValidationError,
        RoutedInvestigatorError,
    ) as exc:
        print("Demonstration execution failed.", file=sys.stderr)
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    if not outcome["ok"]:
        failed = [
            name
            for name in (
                "scope",
                "architecture",
                "flow",
                "visualization",
                "validation",
                "rendering",
                "artifacts",
                "m17",
                "trace",
            )
            if not outcome[name]["ok"]
        ]
        print(
            "Demonstration condition failed in: {}.".format(
                ", ".join(failed)
            ),
            file=sys.stderr,
        )
        return EXIT_DEMO_FAILED
    print(
        "exit code: {} ({}; traces under {}; artifacts under "
        "{})".format(
            EXIT_SUCCESS,
            mode_label,
            trace_root,
            outcome["artifact_set"].directory
            if outcome.get("artifact_set") is not None
            else "artifacts/architecture_visualization",
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
