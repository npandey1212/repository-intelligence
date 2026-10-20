#!/usr/bin/env python
"""Flow / Dependency Investigator demonstration — direct + orchestrated (§28).

What this script demonstrates (spec §29):

1. the DIRECT path — the versioned specialist prompt over the existing
   Agent runtime producing a typed ``FlowDependencyReport`` (source,
   target, path_status, ordered path), then the deterministic M13
   evidence check and the human-readable FLOW INVESTIGATION view;
2. the ORCHESTRATED path — the SAME specialist injected as the
   Investigator of the M10 → M17 planner/orchestrator, with the typed
   projection handing every path step to the M16 Analyst and the
   deterministic validator judging the collaboration.

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
    FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1,
    FLOW_DEPENDENCY_TOOLS,
    FLOW_PROMPT_VERSION,
    PATH_STATUSES,
    FlowDependencyInvestigator,
    FlowDependencyInvestigatorError,
    FlowDependencyReport,
    build_flow_dependency_investigator_prompt,
)
from repo_intelligence.context_engine import (  # noqa: E402
    ContextBudget,
    ContextEngine,
)
from repo_intelligence.evidence.models import (  # noqa: E402
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    ClaimValidation,
)
from repo_intelligence.evidence.validator import (  # noqa: E402
    gather_evidence,
)
from repo_intelligence.engineering import RELATIONSHIP_KINDS  # noqa: E402
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
    COMPLETED,
    ArchitectureFinding,
    ContractError,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (  # noqa: E402
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import Orchestrator  # noqa: E402
from repo_intelligence.planning.planner import (  # noqa: E402
    build_planning_request,
    plan_from_response,
)
from repo_intelligence.planning.validation import (  # noqa: E402
    PlanValidationError,
    validate_plan,
)
from repo_intelligence.tools.read_file import read_file  # noqa: E402
from repo_intelligence.tools.search_repository import (  # noqa: E402
    search_repository,
)

RULE = "=" * 72
SECTION = "-" * 52

EXIT_SUCCESS = 0
EXIT_DEMO_FAILED = 1
EXIT_EXECUTION_FAILED = 6

SUBJECT = "flow-dependency-investigator"

# Offline scope: a visibly synthetic commit (demo labels only).
OFFLINE_COMMIT = "c" * 40

DEFAULT_QUESTION = (
    "How does an agent interact with a tool and an MCP server in "
    "this repository?"
)

# The default investigation target is the capability's OWN tiny
# evaluation fixture (§30): deterministic, honest, self-contained.
# Pass --repository to investigate any other checkout.
DEFAULT_REPOSITORY = PROJECT_ROOT / "tests" / "fixtures" / "arch_mcp"
READ_CANDIDATES = (
    "agents/collector.py",
    "agents/researcher.py",
    "src/producer.py",
    "src/api.py",
    "src/service.py",
    "README.md",
    "main.py",
    "src/main.py",
)
MCP_CONFIG_NAME = ".mcp.json"

DEMO_DB_PATH = (
    PROJECT_ROOT / "runtime" / "agent_memory" /
    "flow_investigator_demo.db"
)
DEFAULT_BUDGET = ContextBudget(max_items=64, max_characters=6000)


# -- scripted model --------------------------------------------------------


def _text(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


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


def _ref(path: str) -> Dict[str, Any]:
    return {"path": path}


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


def demo_read_target(
    repository: Optional[str],
) -> Any:
    """Resolve (root, target_relative_path) for the investigation.

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
    """Resolve the offline flow story for this root + target.

    When the root carries an MCP configuration, the story is the
    agent → tool → MCP server chain the question asks about (the
    intermediate hops are kept — never collapsed).  Without MCP the
    story stays honest: the source endpoint is identified, and the
    destination is recorded as UNKNOWN rather than invented.
    """
    source = Path(target).stem
    server: Optional[str] = None
    tool: Optional[str] = None
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
    if server is not None and tool is not None:
        legs = [
            (source, tool, "IMPORTS", [target]),
            (tool, server, "EXPOSED_BY", [MCP_CONFIG_NAME]),
        ]
        reads = [target, MCP_CONFIG_NAME]
        return {
            "target": target,
            "mcp": True,
            "source": source,
            "destination": server,
            "tool": tool,
            "server": server,
            "legs": legs,
            "reads": reads,
        }
    return {
        "target": target,
        "mcp": False,
        "source": source,
        "destination": None,
        "tool": None,
        "server": None,
        "legs": [],
        "reads": [target],
    }


def _materialize(legs: List[tuple]) -> List[Dict[str, Any]]:
    return [
        _step(source, target, kind, list(paths))
        for source, target, kind, paths in legs
    ]


def offline_flow_payload(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    summary: str = "",
    findings: Optional[List[Dict[str, Any]]] = None,
    uncertainties: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The DIRECT-mode FlowDependencyReport: endpoints, ordered path,
    findings — every reference points at a file this run reads."""
    legs = story["legs"]
    path = _materialize(legs)
    evidence_paths: List[str] = []
    for path_item in story["reads"]:
        if path_item not in evidence_paths:
            evidence_paths.append(path_item)
    if findings is None:
        findings = [
            {
                "statement": (
                    "{} is the flow's source endpoint "
                    "(offline scripted finding).".format(
                        story["source"]
                    )
                ),
                "evidence_refs": [_ref(story["reads"][0])],
                "support": STATUS_SUPPORTED,
            }
        ]
    if uncertainties is None:
        uncertainties = (
            []
            if story["mcp"]
            else [
                "no MCP configuration exists in this repository, so "
                "the flow's destination could not be established "
                "from static evidence"
            ]
        )
    if not summary:
        if story["mcp"]:
            summary = (
                "Read-only inspection of {target} +{mcp} established "
                "the path {source} -> {tool} -> {server} "
                "(offline scripted report).".format(
                    target=story["target"],
                    mcp=MCP_CONFIG_NAME,
                    source=story["source"],
                    tool=story["tool"],
                    server=story["server"],
                )
            )
        else:
            summary = (
                "Read-only inspection of {target} identified the "
                "source endpoint; no path to a destination is "
                "established (offline scripted report).".format(
                    target=story["target"]
                )
            )
    return {
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "source": story["source"],
        "target": story["destination"],
        "path_status": "FOUND" if legs else "UNKNOWN",
        "summary": summary,
        "path": path,
        "alternative_paths": [],
        "relationships": list(path),
        "findings": findings,
        "evidence_refs": [_ref(item) for item in evidence_paths],
        "uncertainties": uncertainties,
        "status": "complete",
    }


def offline_step_endpoints_payload(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> Dict[str, Any]:
    """Orchestrated step P1: endpoints identified, path pending."""
    findings = [
        {
            "statement": "{} is the flow's source endpoint".format(
                story["source"]
            ),
            "evidence_refs": [_ref(story["reads"][0])],
            "support": STATUS_SUPPORTED,
        }
    ]
    if story["destination"] is not None:
        findings.append(
            {
                "statement": "{} is the flow's destination endpoint"
                .format(story["destination"]),
                "evidence_refs": [_ref(MCP_CONFIG_NAME)],
                "support": STATUS_SUPPORTED,
            }
        )
    payload = offline_flow_payload(
        question,
        repository,
        commit,
        story,
        summary="Step P1: endpoints identified from direct reads.",
        findings=findings,
        uncertainties=[],
    )
    payload["path"] = []
    payload["relationships"] = []
    payload["path_status"] = "UNKNOWN"
    return payload


def offline_step_leg_payload(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    leg_index: int,
) -> Dict[str, Any]:
    """Orchestrated step P2/P3: ONE leg of the path (or the honest
    UNKNOWN answer when no leg exists)."""
    legs = story["legs"]
    if leg_index < len(legs):
        source, target, _, paths = legs[leg_index]
        payload = offline_flow_payload(
            question,
            repository,
            commit,
            story,
            summary="Step P{}: leg {} -> {} established.".format(
                leg_index + 2, source, target
            ),
            findings=[],
            uncertainties=[],
        )
        payload["source"] = source
        payload["target"] = target
        payload["path"] = [_step(source, target, _, list(paths))]
        payload["relationships"] = list(payload["path"])
        payload["path_status"] = "FOUND"
        payload["evidence_refs"] = [_ref(path) for path in paths]
        return payload
    payload = offline_flow_payload(
        question,
        repository,
        commit,
        story,
        summary=(
            "Step P{}: no further leg is established from the "
            "evidence.".format(leg_index + 2)
        ),
        findings=[
            {
                "statement": (
                    "the inspected file shows the source endpoint "
                    "but no destination hop"
                ),
                "evidence_refs": [_ref(story["reads"][0])],
                "support": STATUS_SUPPORTED,
            }
        ],
        uncertainties=[
            "no destination hop is established from static evidence"
        ],
    )
    payload["path"] = []
    payload["relationships"] = []
    payload["path_status"] = "UNKNOWN"
    return payload


def offline_plan_payload(question: str) -> Dict[str, Any]:
    """Four steps: three INVESTIGATOR, one ANALYST (the flow
    question's natural evidence-then-interpretation shape)."""
    return {
        "objective": question,
        "steps": [
            {
                "id": "P1",
                "goal": (
                    "Identify the source and target components of the "
                    "flow"
                ),
            },
            {
                "id": "P2",
                "goal": (
                    "Investigate the first leg of the flow with "
                    "evidence"
                ),
            },
            {
                "id": "P3",
                "goal": (
                    "Investigate the remaining leg of the flow with "
                    "evidence"
                ),
            },
            {
                "id": "P4",
                "goal": (
                    "Explain how the flow reaches its destination"
                ),
            },
        ],
    }


def offline_finding_payload(
    question: str,
    repository: str,
    commit: str,
    investigation_id: str,
    story: Dict[str, Any],
) -> Dict[str, Any]:
    if story["mcp"]:
        statement = (
            "The inspected files show the chain {source} -> {tool} -> "
            "{server} (offline scripted conclusion).".format(
                source=story["source"],
                tool=story["tool"],
                server=story["server"],
            )
        )
    else:
        statement = (
            "The inspected file shows the source endpoint; no "
            "destination hop is demonstrated (offline scripted "
            "conclusion)."
        )
    return {
        "investigation_id": investigation_id,
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "conclusions": [
            {
                "statement": statement,
                "classification": "observed",
                "evidence_refs": [_ref(story["reads"][0])],
            }
        ],
        "evidence_refs": [_ref(story["reads"][0])],
        "summary": "Evidence-backed flow conclusion.",
        "status": "complete",
    }


# -- offline models --------------------------------------------------------


def _read_calls(story: Dict[str, Any], suffix: str) -> List[ToolCall]:
    return [
        ToolCall(
            id="call_flow_read_{}_{}".format(suffix, index),
            name="read_file",
            arguments={"path": path},
        )
        for index, path in enumerate(story["reads"])
    ]


def build_offline_direct_model(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> OfflineModel:
    """One real tool request (the REAL dispatcher reads the fixture),
    then the typed report whose refs match that read."""
    return OfflineModel(
        _tool_calls(*_read_calls(story, "direct")),
        _structured(
            offline_flow_payload(question, repository, commit, story)
        ),
    )


def build_offline_investigator_model(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> OfflineModel:
    """Three orchestrated steps: endpoints, leg one, leg two (or the
    honest UNKNOWN answers when the repository has no second hop)."""
    return OfflineModel(
        _tool_calls(*_read_calls(story, "s1")),
        _structured(
            offline_step_endpoints_payload(
                question, repository, commit, story
            )
        ),
        _tool_calls(*_read_calls(story, "s2")),
        _structured(
            offline_step_leg_payload(
                question, repository, commit, story, 0
            )
        ),
        _tool_calls(*_read_calls(story, "s3")),
        _structured(
            offline_step_leg_payload(
                question, repository, commit, story, 1
            )
        ),
    )


def build_offline_analyst_model(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
) -> OfflineModel:
    """One typed finding echoing the derived correlation id."""
    derived = FlowDependencyReport(
        **offline_step_endpoints_payload(
            question, repository, commit, story
        )
    ).derived_investigation_id()
    return OfflineModel(
        _structured(
            offline_finding_payload(
                question, repository, commit, derived, story
            )
        )
    )


def build_offline_planner_model(question: str) -> OfflineModel:
    return OfflineModel(_structured(offline_plan_payload(question)))


# -- composition -----------------------------------------------------------


def fixture_bindings(root: Path) -> Dict[str, Any]:
    """Bind the read-only text tools to the investigation root via the
    documented ``tool(arguments)`` closure seam."""

    def read(arguments):
        return read_file(arguments, repository_root=root)

    def search(arguments):
        return search_repository(arguments, repository_root=root)

    return {"read_file": read, "search_repository": search}


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
            id="repo_intelligence.agents.flow_dependency_investigator",
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


def build_specialist(
    llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]] = None,
) -> FlowDependencyInvestigator:
    """The specialist over the existing Agent runtime — the ONE
    composition both the direct phase and the orchestrator use."""
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


def build_analyst(
    llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
) -> Agent:
    return Agent(
        llm,
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )


# -- phases ----------------------------------------------------------------


def run_scope_phase(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    specialist: FlowDependencyInvestigator,
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
        file=stream)
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
        "prompt:     FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V{} ({})".format(
            FLOW_PROMPT_VERSION[1:],
            len(FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1),
        ),
        file=stream,
    )
    print(
        "tools:      {}".format(", ".join(specialist.tools)),
        file=stream,
    )
    prompt_preview = build_flow_dependency_investigator_prompt(
        question=question,
        repository_scope=repository,
        commit_scope=commit,
    )
    print(
        "task block: {}".format(
            _clip(prompt_preview.splitlines()[-1])
        ),
        file=stream,
    )
    ok = (
        FLOW_PROMPT_VERSION == "V2"
        and specialist.tools == list(FLOW_DEPENDENCY_TOOLS)
        and len(specialist.tools) == 4
        and question in prompt_preview
        and "orchestrat"
        not in FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1.lower()
        and "planner"
        not in FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1.lower()
    )
    print(
        "scope condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok}


def run_direct_phase(
    specialist: FlowDependencyInvestigator,
    question: str,
    repository: str,
    commit: str,
    stream=None,
) -> Dict[str, Any]:
    """Phase 2 — direct execution: typed ``FlowDependencyReport``."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 2 — direct execution "
        "(investigate -> FlowDependencyReport)",
        file=stream,
    )
    print(RULE, file=stream)
    report = specialist.investigate(question, repository, commit)
    trace = specialist.last_trace
    print("investigation id: {}".format(
        report.derived_investigation_id()
    ), file=stream)
    print("run id:           {}".format(
        trace.run_id if trace is not None else "none"
    ), file=stream)
    print("status:           {}".format(report.status), file=stream)
    print(
        "path_status:      {}".format(report.path_status), file=stream
    )
    print(
        "source -> target: {} -> {}".format(
            report.source, report.target if report.target else "?"
        ),
        file=stream,
    )
    print(
        "path steps:       {}   relationships: {}   findings: "
        "{}".format(
            len(report.path),
            len(report.relationships),
            len(report.findings),
        ),
        file=stream,
    )
    print(
        "evidence refs:    {}".format(
            ", ".join(ref.path for ref in report.evidence_refs)
            or "none"
        ),
        file=stream,
    )
    for step in report.path:
        print(
            "  * {} {} {} [{}] ({})".format(
                step.source,
                step.relationship_type,
                step.target,
                step.classification,
                step.confidence,
            ),
            file=stream,
        )
    for finding in report.findings:
        print(
            "  * [{}] {}".format(finding.support, _clip(finding.statement)),
            file=stream,
        )
    for uncertainty in report.uncertainties:
        print("  ? {}".format(_clip(uncertainty)), file=stream)
    consistent = bool(report.path) == (
        report.path_status == "FOUND"
    )
    ok = (
        isinstance(report, FlowDependencyReport)
        and bool(report.evidence_refs)
        and report.status == "complete"
        and report.path_status in PATH_STATUSES
        and consistent
    )
    print(
        "direct condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok, "report": report}


def _status_counts(validations: List[ClaimValidation]) -> Dict[str, int]:
    counts = {STATUS_SUPPORTED: 0, STATUS_UNSUPPORTED: 0, STATUS_UNKNOWN: 0}
    for validation in validations:
        counts[validation.status] = counts.get(validation.status, 0) + 1
    return counts


def run_evidence_phase(
    specialist: FlowDependencyInvestigator,
    report: FlowDependencyReport,
    stream=None,
) -> Dict[str, Any]:
    """Phase 3 — the deterministic evidence check (M13 claim
    validation, no model in the loop)."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 3 — evidence check (M13 validate_claims, no LLM)",
        file=stream,
    )
    print(RULE, file=stream)
    validations = specialist.validate(report)
    counts = _status_counts(validations)
    for validation in validations:
        print(
            "  [{:<11}] {}".format(
                validation.status,
                _clip(validation.claim.statement),
            ),
            file=stream,
        )
        if validation.status != STATUS_SUPPORTED:
            print(
                "               {}{}".format(
                    validation.reason or "no eligible evidence",
                    " ({})".format(validation.code)
                    if validation.code
                    else "",
                ),
                file=stream,
            )
    print(
        "claims: {} supported, {} unsupported, {} unknown".format(
            counts[STATUS_SUPPORTED],
            counts[STATUS_UNSUPPORTED],
            counts[STATUS_UNKNOWN],
        ),
        file=stream,
    )
    ok = (
        bool(validations)
        and counts[STATUS_UNSUPPORTED] == 0
        and counts[STATUS_UNKNOWN] == 0
    )
    print(
        "evidence condition: {}".format("yes" if ok else "NO"),
        file=stream,
    )
    return {"ok": ok, "validations": validations}


def run_view_phase(
    question: str,
    repository: str,
    commit: str,
    report: FlowDependencyReport,
    validations: List[ClaimValidation],
    specialist: FlowDependencyInvestigator,
    stream=None,
) -> Dict[str, Any]:
    """Phase 4 — the human-readable FLOW INVESTIGATION view (§24):
    the ordered path drawn as an ASCII chain, evidence, validator."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 4 — human-readable flow view (FLOW INVESTIGATION)",
        file=stream,
    )
    print(RULE, file=stream)
    print("FLOW INVESTIGATION", file=stream)
    print(SECTION, file=stream)
    print("Question:", file=stream)
    print("  {}".format(_clip(question, 120)), file=stream)
    print(
        "Repository: {}@{}".format(repository, commit[:12]), file=stream
    )
    print(file=stream)

    inspected: List[str] = []
    if specialist.last_trace is not None:
        index = gather_evidence(specialist.last_trace)
        seen = set()
        for item in index.inspected:
            if item.path not in seen:
                seen.add(item.path)
                inspected.append(item.path)
    print("1. Investigator", file=stream)
    if inspected:
        for path in inspected:
            print("   \u2713 read {}".format(path), file=stream)
    else:
        print("   \u2717 no inspected reads", file=stream)
    print(
        "   \u2713 path_status {}".format(report.path_status),
        file=stream,
    )

    print("2. Path", file=stream)
    if report.path:
        width = max(
            len(step.source) for step in report.path
        ) + 2
        for index, step in enumerate(report.path):
            print(
                "   {}".format(step.source.rjust(width)),
                file=stream,
            )
            print(
                "   {} \u2502 {}  [{}]  ({})".format(
                    " " * (width - 1),
                    step.relationship_type,
                    step.classification,
                    step.evidence_refs[0].path,
                )
                if step.evidence_refs
                else "   {} \u2502 {}  ({})".format(
                    " " * (width - 1),
                    step.relationship_type,
                    "no evidence",
                ),
                file=stream,
            )
            print(
                "   {} \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "> {}".format(" " * (width - 1), step.target),
                file=stream,
            )
    else:
        print(
            "   (no path established — {})".format(
                report.path_status
            ),
            file=stream,
        )

    print("3. Evidence", file=stream)
    print(
        "   \u2713 {} evidence reference(s)".format(
            len(report.evidence_refs)
        ),
        file=stream,
    )
    for step in report.path:
        print(
            "   * {} {} {} [{}]".format(
                step.source,
                step.relationship_type,
                step.target,
                step.classification,
            ),
            file=stream,
        )
    for finding in report.findings:
        print(
            "   * [{}] {}".format(
                finding.support, _clip(finding.statement)
            ),
            file=stream,
        )
    for uncertainty in report.uncertainties:
        print("   ? {}".format(_clip(uncertainty)), file=stream)

    print("4. Analyst", file=stream)
    print(
        "   \u2014 not applicable in direct mode "
        "(the orchestrated run uses the Analyst)",
        file=stream,
    )

    print("5. Validator", file=stream)
    counts = _status_counts(validations)
    print(
        "   \u2713 {} statement(s) supported".format(
            counts[STATUS_SUPPORTED]
        ),
        file=stream,
    )
    if counts[STATUS_UNSUPPORTED]:
        print(
            "   \u2717 {} statement(s) unsupported".format(
                counts[STATUS_UNSUPPORTED]
            ),
            file=stream,
        )
    print(file=stream)
    print("Result: {}".format(report.path_status), file=stream)

    ok = (
        bool(inspected)
        and counts[STATUS_UNSUPPORTED] == 0
        and report.status == "complete"
        and (
            bool(report.path)
            == (report.path_status == "FOUND")
        )
    )
    print(
        "view condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok}


def run_orchestrated_phase(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    planner_llm,
    investigator_llm,
    analyst_llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]],
    stream=None,
) -> Dict[str, Any]:
    """Phase 5 — the SAME specialist through M10 → M17 → M16: plan,
    role assignment, typed handoff, deterministic validation."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 5 — orchestrated execution "
        "(plan -> specialist steps -> Analyst -> validator)",
        file=stream,
    )
    print(RULE, file=stream)

    planner_response = planner_llm.complete(
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

    specialist = build_specialist(
        investigator_llm,
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
        bindings=bindings,
    )
    analyst = build_analyst(
        analyst_llm,
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
    )
    orchestrator = Orchestrator(specialist, analyst, trace_sink=trace_sink)
    task = TaskPayload(
        question=question,
        repository_scope=repository,
        commit_scope=commit,
    )
    run = orchestrator.execute(task, investigation_plan)

    assigns = [
        event
        for event in run.events
        if event["operation"] == "orchestration.step.assign"
    ]
    roles = [event["agent_role"] for event in assigns]
    print("roles: {}".format(" \u2192 ".join(roles) or "none"), file=stream)
    for step, assign in zip(run.plan.steps, assigns):
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

    statements: List[str] = []
    if run.report is not None:
        statements = [
            finding.statement for finding in run.report.findings
        ]
        leg_statements = [
            statement
            for statement in statements
            if any(
                " {} ".format(kind) in statement
                for kind in RELATIONSHIP_KINDS
            )
        ]
        print(
            "merged report: {} statement(s), {} path leg(s), "
            "{} evidence ref(s)".format(
                len(statements),
                len(leg_statements),
                len(run.report.evidence_refs),
            ),
            file=stream,
        )
    if run.validation is not None:
        print(
            "validation: {}".format(run.validation.status), file=stream
        )
        for check in run.validation.checks:
            print(
                "  [{}] {:<18} {}".format(
                    "ok" if check.ok else "FAIL",
                    check.rule,
                    "" if check.ok else _clip(check.detail, 160),
                ),
                file=stream,
            )
    if run.finding is not None:
        derived = FlowDependencyReport(
            **offline_step_endpoints_payload(
                question, repository, commit, story
            )
        ).derived_investigation_id()
        echo_ok = run.finding.investigation_id == derived
        print(
            "finding: id echo {} — {}".format(
                "yes" if echo_ok else "NO",
                _clip(run.finding.summary),
            ),
            file=stream,
        )
    else:
        echo_ok = False

    expected_roles = [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    checks_ok = (
        run.validation is not None
        and run.validation.status == "passed"
        and all(check.ok for check in run.validation.checks)
    )
    steps_ok = all(
        step.status == "completed" for step in run.plan.steps
    )
    # The intermediate hops must survive the typed handoff: every
    # scripted leg appears as its own statement, and no collapsed
    # source -> destination edge may be invented.
    legs_ok = True
    if story["mcp"]:
        expected_legs = [
            "{} {} {}".format(
                source, kind, target
            )
            for source, target, kind, _ in story["legs"]
        ]
        legs_ok = all(
            any(expected in statement for statement in statements)
            for expected in expected_legs
        )
        collapsed = "{} EXPOSED_BY {}".format(
            story["source"], story["server"]
        )
        legs_ok = legs_ok and not any(
            collapsed in statement for statement in statements
        )
    ok = (
        run.status == COMPLETED
        and run.termination_reason == "SUCCESS"
        and roles == expected_roles
        and steps_ok
        and checks_ok
        and echo_ok
        and legs_ok
        and run.finding is not None
        and run.report is not None
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
        "specialist": specialist,
        "analyst": analyst,
    }


def run_trace_phase(
    direct_specialist: FlowDependencyInvestigator,
    orchestrated_specialist: FlowDependencyInvestigator,
    orchestrated_analyst: Agent,
    run,
    trace_root: Path,
    stream=None,
) -> Dict[str, Any]:
    """Phase 6 — the ONE trace system: on disk for every participant,
    with the orchestration events mirrored into the agent traces."""
    stream = stream or sys.stdout
    print(RULE, file=stream)
    print(
        "Phase 6 — trace summary (disk artifacts + mirrored events)",
        file=stream,
    )
    print(RULE, file=stream)
    traces = [
        ("direct investigator", direct_specialist.last_trace),
        ("orchestrated investigator", orchestrated_specialist.last_trace),
        ("orchestrated analyst", orchestrated_analyst.last_trace),
    ]
    disk_ok = True
    markdown_ok = True
    for label, trace in traces:
        if trace is None:
            print("trace ({}): MISSING".format(label), file=stream)
            disk_ok = False
            markdown_ok = False
            continue
        trace_path = Path(trace_root) / trace.run_id / "trace.json"
        markdown_path = Path(trace_root) / trace.run_id / "trace.md"
        has_json = trace_path.is_file()
        markdown = (
            markdown_path.read_text(encoding="utf-8")
            if markdown_path.is_file()
            else ""
        )
        has_md = bool(markdown)
        print(
            "trace ({}): {} json={} md={} run-summary={}".format(
                label,
                trace.run_id,
                has_json,
                has_md,
                "## Run Summary" in markdown,
            ),
            file=stream,
        )
        disk_ok = disk_ok and has_json and has_md
        markdown_ok = markdown_ok and "## Run Summary" in markdown
        if label != "direct investigator":
            mirrored = "## Orchestration" in markdown
            print(
                "  orchestration section: {}".format(
                    "present" if mirrored else "MISSING"
                ),
                file=stream,
            )
            markdown_ok = markdown_ok and mirrored

    mirrored_events = orchestrated_specialist.last_trace
    mirrored_count = (
        len(getattr(mirrored_events, "orchestration_events", []) or [])
        if mirrored_events is not None
        else 0
    )
    sequences = [event["sequence"] for event in run.events]
    ordered = (
        sequences == sorted(sequences)
        and len(sequences) == len(set(sequences))
    )
    in_memory = render_markdown(orchestrated_specialist.last_trace)
    print(
        "mirrored orchestration events: {} (markdown: {})".format(
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
    ok = (
        disk_ok
        and markdown_ok
        and mirrored_count > 0
        and ordered
        and "## Orchestration" in in_memory
    )
    print(
        "trace condition: {}".format("yes" if ok else "NO"), file=stream
    )
    return {"ok": ok}


# -- demo ------------------------------------------------------------------


def run_demo(
    question: str,
    repository: str,
    commit: str,
    story: Dict[str, Any],
    direct_llm,
    planner_llm,
    investigator_llm,
    analyst_llm,
    trace_sink,
    memory_manager,
    memory_context,
    context_engine,
    bindings: Optional[Dict[str, Any]],
    trace_root: Path,
    stream=None,
) -> Dict[str, Any]:
    """Every phase over one shared story; returns each outcome plus
    the aggregate ``ok``."""
    stream = stream or sys.stdout

    direct_specialist = build_specialist(
        direct_llm,
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
        bindings=bindings,
    )
    scope = run_scope_phase(
        question,
        repository,
        commit,
        story,
        direct_specialist,
        stream=stream,
    )
    direct = run_direct_phase(
        direct_specialist, question, repository, commit, stream=stream
    )
    evidence = (
        run_evidence_phase(
            direct_specialist, direct["report"], stream=stream
        )
        if direct.get("report") is not None
        else {"ok": False, "validations": []}
    )
    view = (
        run_view_phase(
            question,
            repository,
            commit,
            direct["report"],
            evidence["validations"],
            direct_specialist,
            stream=stream,
        )
        if direct.get("report") is not None
        else {"ok": False}
    )
    orchestrated = run_orchestrated_phase(
        question,
        repository,
        commit,
        story,
        planner_llm,
        investigator_llm,
        analyst_llm,
        trace_sink,
        memory_manager,
        memory_context,
        context_engine,
        bindings,
        stream=stream,
    )
    if orchestrated.get("run") is not None and orchestrated.get(
        "specialist"
    ) is not None:
        trace = run_trace_phase(
            direct_specialist,
            orchestrated["specialist"],
            orchestrated["analyst"],
            orchestrated["run"],
            trace_root,
            stream=stream,
        )
    else:
        trace = {"ok": False}

    phases = {
        "scope": scope,
        "direct": direct,
        "evidence": evidence,
        "view": view,
        "orchestrated": orchestrated,
        "trace": trace,
    }
    ok = all(entry["ok"] for entry in phases.values())

    print(RULE, file=stream)
    print("Flow / Dependency Investigator demonstration summary", file=stream)
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
    return dict(phases, ok=ok)


# -- main ------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_flow_dependency_investigator_demo.py",
        description=(
            "Flow / Dependency Investigator demonstration: direct "
            "specialist execution and full planner/orchestrator/"
            "analyst orchestration — offline and deterministic by "
            "default."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the flow question (default: the agent/tool/"
        "MCP interaction question)",
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
        "flow_investigator_demo.db — the demonstration's OWN database)",
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
                "Flow / Dependency Investigator demonstration (--live) "
                "requires OPENAI_API_KEY.\n"
                "Copy .env.example to .env, set OPENAI_API_KEY (never "
                "commit .env), and re-run — or run the default offline "
                "demonstration:\n"
                "    python agentrun_flow_dependency_investigator_demo.py",
                file=sys.stderr,
            )
            return EXIT_EXECUTION_FAILED
        try:
            planner_llm = OpenAILLMClient(
                model=args.model or DEFAULT_MODEL
            )
            specialist_llm = OpenAILLMClient(
                model=args.model or DEFAULT_MODEL
            )
            analyst_llm = OpenAILLMClient(
                model=args.model or DEFAULT_MODEL
            )
        except LLMError as exc:
            print("Demonstration failed.", file=sys.stderr)
            print("LLMError: {}".format(exc), file=sys.stderr)
            return EXIT_EXECUTION_FAILED
        direct_llm = specialist_llm
        investigator_llm = specialist_llm
        mode_label = "live (real model for every participant)"
    else:
        planner_llm = None
        direct_llm = None
        investigator_llm = None
        analyst_llm = None
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

    if direct_llm is None:
        direct_llm = build_offline_direct_model(
            question, repository_label, commit, story
        )
        investigator_llm = build_offline_investigator_model(
            question, repository_label, commit, story
        )
        analyst_llm = build_offline_analyst_model(
            question, repository_label, commit, story
        )
        planner_llm = build_offline_planner_model(question)

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
        "Flow / Dependency Investigator demonstration — {}".format(
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
            direct_llm,
            planner_llm,
            investigator_llm,
            analyst_llm,
            trace_sink,
            manager,
            memory_context,
            engine,
            bindings,
            trace_root,
        )
    except (
        AgentError,
        LLMError,
        StructuredOutputValidationError,
        ContractError,
        FlowDependencyInvestigatorError,
        PlanValidationError,
    ) as exc:
        print("Demonstration execution failed.", file=sys.stderr)
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    if not outcome["ok"]:
        failed = [
            name
            for name in (
                "scope",
                "direct",
                "evidence",
                "view",
                "orchestrated",
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
        "exit code: {} ({}; traces under {})".format(
            EXIT_SUCCESS, mode_label, trace_root
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
