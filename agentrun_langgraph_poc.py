"""agentrun_langgraph_poc.py — middleware-protected LangGraph
``create_agent`` POC demonstration (offline by default).

Runs the POC workflow end to end — planner -> query router ->
orchestrator select -> REAL ``create_agent`` investigator (with the
three ``react_safety``-backed middleware) -> orchestrator receive ->
tool-free P4 Analyst -> deterministic synthesis validation -> final response — on scripted in-process models. No network, no
provider SDK, deterministic output.

``--live`` switches the planner to the project's ``OpenAILLMClient``
and the investigator to a caller-supplied ``ChatOpenAI`` model (both
built from the SAME ``OPENAI_API_KEY`` configuration the rest of the
project uses), and binds the REAL project tools against the pinned
repository via ``repository_query.live_bindings``. Live runs need a
valid key; this environment has none — offline is the verification
path.

Exit codes: 0 = SUCCESS, 1 = FAILED (explicit failure category
printed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from repo_intelligence.langgraph_poc import (
    build_create_agent_analyst,
    build_create_agent_investigator,
    run_poc,
)

QUESTION = (
    "How do the agents, tools, and MCP servers interact?"
)
REPO_SCOPE = "this repository"
COMMIT_SCOPE = "pinned snapshot"

PLAN_PAYLOAD = {
    "objective": QUESTION,
    "steps": [
        {
            "id": "P1",
            "goal": "Identify the agent components and responsibilities",
        },
        {
            "id": "P2",
            "goal": "Trace how tools are registered and dispatched",
        },
        {
            "id": "P3",
            "goal": "Map how MCP servers integrate with the agents",
        },
        {
            "id": "P4",
            "goal": "Synthesize the architecture diagram specification",
        },
    ],
}

REPORT_ARGS = {
    "investigation_id": "inv-poc-demo",
    "question": QUESTION,
    "repository_scope": REPO_SCOPE,
    "commit_scope": COMMIT_SCOPE,
    "findings": [
        {
            "statement": (
                "Agents dispatch read-only tools through a registry"
            ),
            "evidence_refs": [
                {
                    "path": "src/repo_intelligence/tool_dispatcher.py",
                    "line_start": 1,
                    "line_end": 5,
                }
            ],
            "support": "SUPPORTED",
        }
    ],
    "evidence_refs": [
        {
            "path": "src/repo_intelligence/tool_dispatcher.py",
            "line_start": 1,
            "line_end": 5,
        }
    ],
    "open_questions": [],
    "uncertainties": [],
    "investigated_sources": [
        "src/repo_intelligence/tool_dispatcher.py"
    ],
    "status": "complete",
}


# -- offline scripted doubles -------------------------------------------------


def _build_offline_investigator() -> Any:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    class SpyChatModel(BaseChatModel):
        queue: List[BaseMessage] = Field(default_factory=list)
        calls: List[dict] = Field(default_factory=list)

        @property
        def _llm_type(self) -> str:
            return "spy"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            self.calls.append({"messages": list(messages)})
            if not self.queue:
                raise AssertionError("model queue exhausted")
            return ChatResult(
                generations=[
                    ChatGeneration(message=self.queue.pop(0))
                ]
            )

    from repo_intelligence.tools.protocol import ToolResult

    def fake_binding(arguments: Any) -> ToolResult:
        return ToolResult(
            tool="read_file",
            success=True,
            data={
                "path": str((arguments or {}).get("path", "")),
                "content": "dispatch: registry[name](arguments)",
            },
            source=None,
        )

    spy = SpyChatModel(
        queue=[
            message
            for index in range(1, 4)
            for message in (
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "read_file",
                        "args": {"path": "src/tool_dispatcher.py"},
                        "id": "call-{}-tool".format(index),
                        "type": "tool_call",
                    }],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "InvestigationReport",
                        "args": dict(REPORT_ARGS, investigation_id="inv-poc-demo-{}".format(index)),
                        "id": "call-{}-report".format(index),
                        "type": "tool_call",
                    }],
                ),
            )
        ]
    )
    inv = build_create_agent_investigator(
        spy, {"read_file": fake_binding}
    )
    analyst_ref = {
        "path": "src/repo_intelligence/tool_dispatcher.py",
        "line_start": 1,
        "line_end": 5,
    }
    analyst_spy = SpyChatModel(
        queue=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "ArchitectureFinding",
                    "args": {
                        "investigation_id": "inv-poc-demo-1",
                        "source_investigation_ids": [
                            "inv-poc-demo-1",
                            "inv-poc-demo-2",
                            "inv-poc-demo-3",
                        ],
                        "question": QUESTION,
                        "repository_scope": REPO_SCOPE,
                        "commit_scope": COMMIT_SCOPE,
                        "conclusions": [{
                            "statement": (
                                "The POC investigation used a repository tool "
                                "through the configured tool binding."
                            ),
                            "classification": "derived",
                            "evidence_refs": [analyst_ref],
                        }],
                        "evidence_refs": [analyst_ref],
                        "open_questions": [],
                        "summary": (
                            "The three investigation reports describe tool "
                            "dispatch and the repository evidence it reads."
                        ),
                        "status": "complete",
                    },
                    "id": "call-analyst-report",
                    "type": "tool_call",
                }],
            )
        ]
    )
    analyst = build_create_agent_analyst(analyst_spy)
    return inv, analyst, spy


class _ScriptedPlanner:
    def __init__(self) -> None:
        self.requests: List[Any] = []

    def complete(self, request: Any) -> Any:
        from repo_intelligence.llm.models import LLMResponse, Message

        self.requests.append(request)
        return LLMResponse(
            message=Message(role="assistant", content=""),
            structured_output=dict(PLAN_PAYLOAD),
        )


# -- live wiring (needs OPENAI_API_KEY) ---------------------------------------


def _build_live(
    repository: Path,
) -> tuple:
    from langchain_openai import ChatOpenAI

    from repo_intelligence.llm.config import get_openai_api_key
    from repo_intelligence.llm.openai_client import (
        DEFAULT_MODEL,
        OpenAILLMClient,
    )
    from repo_intelligence.repository_query import live_bindings

    api_key = get_openai_api_key()
    if not api_key:
        raise SystemExit(
            "--live requires OPENAI_API_KEY in the environment "
            "(none found)"
        )
    model = ChatOpenAI(model=DEFAULT_MODEL, api_key=api_key)
    planner = OpenAILLMClient(model=DEFAULT_MODEL, max_retries=0)
    bindings = live_bindings(repository)
    inv = build_create_agent_investigator(model, bindings)
    analyst_model = ChatOpenAI(model=DEFAULT_MODEL, api_key=api_key)
    analyst = build_create_agent_analyst(analyst_model)
    return inv, analyst, planner


def _report(state: Dict[str, Any], live: bool) -> int:
    final = state.get("final") or {}
    status = final.get("status")
    print("")
    print("=== LangGraph create_agent POC result ===")
    print("mode: {}".format("live" if live else "offline"))
    print("status: {}".format(status))
    print("failure: {}".format(final.get("failure")))
    if final.get("error"):
        print("error: {}".format(final["error"]))
    plan = state.get("plan")
    if plan is not None:
        print(
            "plan steps: {}".format(
                ", ".join(step.id for step in plan.steps)
            )
        )
    assignments = state.get("role_assignments") or []
    print(
        "assignments: {}".format(
            ", ".join(
                "{}={}".format(item.step_id, item.role)
                for item in assignments
            )
        )
    )
    print("completed: {}".format(final.get("completed_step_ids")))
    report = final.get("report")
    if report is not None:
        print("report id: {}".format(report.investigation_id))
        print("report status: {}".format(report.status))
        print("findings: {}".format(len(report.findings)))
        for finding in report.findings:
            print("  - [{}] {}".format(finding.support, finding.statement))
    synthesis = final.get("architecture_finding")
    if synthesis is not None:
        print("P4 synthesis: {}".format(synthesis.summary))
        for conclusion in synthesis.conclusions:
            print("  - [{}] {}".format(
                conclusion.classification, conclusion.statement
            ))
    validation = final.get("validation_result")
    if validation is not None:
        print("P4 validation: {}".format(validation.status))
        for check in validation.checks:
            print("  - {} {}: {}".format(
                "PASS" if check.ok else "FAIL", check.rule, check.detail
            ))
    return 0 if status == "SUCCESS" else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Middleware-protected LangGraph create_agent POC "
            "(offline by default)"
        )
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="repository root for --live tool bindings",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="run only P1 (default: run P1-P3 then P4 synthesis/validation)",
    )
    args = parser.parse_args(argv)

    if args.live:
        inv, analyst, planner = _build_live(args.repository)
    else:
        try:
            inv, analyst, _spy = _build_offline_investigator()
        except ImportError as exc:
            print(
                "offline demo requires the optional poc extra "
                "(langchain/langgraph): {}".format(exc),
                file=sys.stderr,
            )
            return 2
        planner = _ScriptedPlanner()

    state = run_poc(
        QUESTION,
        planner_client=planner,
        investigator=inv,
        analyst=analyst,
        repository_scope=REPO_SCOPE,
        commit_scope=COMMIT_SCOPE,
        multi_step=not args.single_step,
    )

    if args.verbose:
        print("=== safety instrumentation ===")
        print(
            "budget records: {}".format(
                [
                    "{}:{}".format(
                        record.get("disposition"),
                        record.get("totals", {}).get(
                            "total_tokens_est"
                        ),
                    )
                    for record in inv.budget.records
                ]
            )
        )
        print(
            "observation records: {}".format(
                [
                    "truncated={} chars {}/{}".format(
                        record.get("truncated"),
                        record.get("returned_chars"),
                        record.get("original_chars"),
                    )
                    for record in inv.observation.records
                ]
            )
        )
        print(
            "guard snapshot: {}".format(inv.execution.guard.snapshot())
        )

    return _report(state, live=args.live)


if __name__ == "__main__":
    sys.exit(main())
