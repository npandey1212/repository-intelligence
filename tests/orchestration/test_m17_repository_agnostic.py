"""M17 repository-agnosticism (spec §34/§35 — no repository branch).

1. ``src/repo_intelligence/orchestration/`` contains NO
   repository-specific identifiers: the orchestrator decides roles,
   budgets, and terminal states from contracts and application config
   only. Demo constants live in the demo script alone.
2. A full orchestrated run works against a NON-ADOP fixture root bound
   through the dispatcher layer — foreign repositories enter only via
   the documented ``tool(arguments)`` binding seam.
"""

from __future__ import annotations

import re
from pathlib import Path

from repo_intelligence.agent import Agent
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
from repo_intelligence.orchestration import COMPLETED, Orchestrator
from repo_intelligence.planning.models import InvestigationPlan, PlanStep
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.read_file import read_file
from repo_intelligence.tools.search_repository import search_repository

from m17_doubles import FakeLLMClient, finding_payload, report_payload, structured

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "construction"
PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "repo_intelligence"
    / "orchestration"
)

REPOSITORY = "construction-fixture"
COMMIT = "d" * 40
QUESTION = "What does Service provide?"
SERVICE_PATH = "src/service.py"


def _report_payload():
    return report_payload(
        investigation_id="inv-fixture",
        question=QUESTION,
        repository_scope=REPOSITORY,
        commit_scope=COMMIT,
        findings=[
            {
                "statement": "service.py defines the Service class",
                "evidence_refs": [{"path": SERVICE_PATH}],
                "support": "SUPPORTED",
            }
        ],
        evidence_refs=[{"path": SERVICE_PATH}],
        investigated_sources=[SERVICE_PATH],
    )


def _finding_payload():
    return finding_payload(
        investigation_id="inv-fixture",
        question=QUESTION,
        repository_scope=REPOSITORY,
        commit_scope=COMMIT,
        conclusions=[
            {
                "statement": "Service.run is the fixture entry point",
                "classification": "observed",
                "evidence_refs": [{"path": SERVICE_PATH}],
            }
        ],
        evidence_refs=[{"path": SERVICE_PATH}],
        summary="A small fixture service.",
    )


def _tool_call(name, arguments) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name=name, arguments=arguments)],
        ),
        finish_reason="tool_calls",
    )


def _fixture_bindings():
    """Bind the read-only text tools to the fixture root through the
    documented ``tool(arguments)`` closure seam."""

    def read(arguments):
        return read_file(arguments, repository_root=FIXTURE_ROOT)

    def search(arguments):
        return search_repository(arguments, repository_root=FIXTURE_ROOT)

    def generic(name):
        def tool(arguments):
            return ToolResult(success=True, tool=name, data={})

        return tool

    return {
        "read_file": read,
        "search_repository": search,
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def test_no_repository_specific_identifiers_in_package():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"adop", text, flags=re.IGNORECASE):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append("{}:{}".format(path.name, line))
    assert offenders == [], "repository identifiers found: {}".format(
        offenders
    )


def test_orchestration_has_no_framework_or_broker_dependencies():
    """No second framework enters by import — the orchestrator is
    application code over the existing Agent + planner + validator.

    The LangGraph orchestration milestone designates
    ``langgraph_runner.py`` as the ONE sanctioned framework adapter
    (the graph owns task-level control flow over the same contracts).
    That file may import ``langgraph`` and nothing else from the
    banned list; every other module in the package — including the
    adapter — stays free of crewai/autogen/sqlalchemy/brokers, and
    every other module stays free of ``langgraph`` entirely.
    """
    banned_always = (
        r"crewai|autogen|sqlalchemy|redis|pika|celery|requests|httpx"
    )
    full_pattern = re.compile(
        r"^\s*(?:from|import)\s+(langgraph|" + banned_always + r")\b",
        re.IGNORECASE,
    )
    adapter_pattern = re.compile(
        r"^\s*(?:from|import)\s+(?:" + banned_always + r")\b",
        re.IGNORECASE,
    )
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        pattern = (
            adapter_pattern
            if path.name == "langgraph_runner.py"
            else full_pattern
        )
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                offenders.append(
                    "{}:{}:{}".format(path.name, number, line.strip())
                )
    assert offenders == []


def test_plan_executes_against_non_repository_fixture():
    investigator = Agent(
        FakeLLMClient(
            _tool_call("read_file", {"path": SERVICE_PATH}),
            structured(_report_payload()),
        ),
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=_fixture_bindings()
        ),
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    analyst = Agent(
        FakeLLMClient(structured(_finding_payload())),
        tool_dispatcher=restricted_dispatcher("analyst"),
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
    )
    orchestrator = Orchestrator(investigator, analyst)
    plan = InvestigationPlan(
        objective=QUESTION,
        steps=[
            PlanStep(id="P1", goal="Identify the Service entry points"),
            PlanStep(id="P2", goal="Explain what Service provides"),
        ],
    )
    run = orchestrator.execute(
        TaskPayload(
            question=QUESTION,
            repository_scope=REPOSITORY,
            commit_scope=COMMIT,
        ),
        plan,
    )
    assert run.status == COMPLETED
    assert run.validation.status == "passed"
    assert run.termination_reason == "SUCCESS"
    # the tool read the FIXTURE file, not a pinned default snapshot:
    # its content appears in the recorded tool result.
    contents = [
        result.get("data", {}).get("content", "")
        for iteration in investigator.last_trace.iterations
        for result in iteration.tool_results
        if result.get("success")
    ]
    assert any("Service helpers used by construction" in text for text in contents)
    # the orchestrator's decisions reference only generic ids/roles
    for event in run.events:
        assert event["operation"].startswith("orchestration.")
        if "agent_role" in event:
            assert event["agent_role"] in ("INVESTIGATOR", "ANALYST")
