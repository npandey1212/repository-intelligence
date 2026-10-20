"""M16 spec §23 test 13 — repository-agnosticism.

1. The multi-agent package must contain NO repository-specific
   identifiers (no "ADOP" anywhere in ``src/repo_intelligence/
   multiagent/``): roles, contracts, and validation are generic; demo
   constants live only in the demo script.
2. A full workflow must run against a NON-ADOP fixture root bound
   through the dispatcher layer — proving foreign repositories enter
   only via the documented ``tool(arguments)`` binding seam, never by
   forking the code.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from repo_intelligence.agent import Agent
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.multiagent import (
    COMPLETED,
    ArchitectureFinding,
    InvestigationReport,
    MultiAgentRunner,
    TaskPayload,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.tools import ToolResult
from repo_intelligence.tools.read_file import read_file
from repo_intelligence.tools.search_repository import search_repository

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "construction"
PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "multiagent"
)

REPOSITORY = "construction-fixture"
COMMIT = "d" * 40
QUESTION = "What does Service provide?"
SERVICE_PATH = "src/service.py"


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


def _structured(payload) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(name, arguments) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name=name, arguments=arguments)],
        ),
        finish_reason="tool_calls",
    )


def _report_payload():
    return {
        "investigation_id": "inv-fixture",
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "findings": [
            {
                "statement": "service.py defines the Service class",
                "evidence_refs": [{"path": SERVICE_PATH}],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [{"path": SERVICE_PATH}],
        "investigated_sources": [SERVICE_PATH],
        "status": "complete",
    }


def _finding_payload():
    return {
        "investigation_id": "inv-fixture",
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "conclusions": [
            {
                "statement": "Service.run is the fixture entry point",
                "classification": "observed",
                "evidence_refs": [{"path": SERVICE_PATH}],
            }
        ],
        "evidence_refs": [{"path": SERVICE_PATH}],
        "summary": "A small fixture service.",
        "status": "complete",
    }


def _fixture_bindings():
    """Bind the two read-only text tools to the fixture root via the
    documented ``tool(arguments)`` closure seam; the other two stay
    generic canned successes (never called by the scripted run)."""

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


def test_workflow_runs_against_non_repository_fixture():
    investigator = Agent(
        FakeLLMClient(
            _tool_calls("read_file", {"path": SERVICE_PATH}),
            _structured(_report_payload()),
        ),
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=_fixture_bindings()
        ),
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    analyst = Agent(
        FakeLLMClient(_structured(_finding_payload())),
        tool_dispatcher=restricted_dispatcher("analyst"),
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
    )
    runner = MultiAgentRunner(investigator, analyst, clock=time.monotonic)
    state = runner.run(
        TaskPayload(
            question=QUESTION,
            repository_scope=REPOSITORY,
            commit_scope=COMMIT,
        )
    )
    assert state.status == COMPLETED
    assert state.validation.status == "passed"
    # The tool read the FIXTURE file, not the pinned default snapshot:
    # its content appears in the investigator's recorded tool result.
    contents = [
        result.get("data", {}).get("content", "")
        for iteration in investigator.last_trace.iterations
        for result in iteration.tool_results
        if result.get("success")
    ]
    assert any("Service helpers used by construction" in text for text in contents)
