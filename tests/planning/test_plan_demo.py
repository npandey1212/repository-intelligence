"""M10 demonstration entry-point tests (wiring only — fully offline).

Verify ``agentrun_m10_plan_demo.py`` is wired to the EXISTING
implementation — it must not become a second planner, validator,
loop, or evaluator:

 1  the module imports and exposes the demo API (no side effects)
 2  the existing Agent is constructed through the demo's build path
    (one budget, M8.9 structured-output contract, real Agent class)
 3  the EXISTING Architecture Context loader supplies the planner's
    orientation input — passed directly, never as a tool call
 4  one offline planned run through the demo composition: planner
    request (no tools) → validated plan in the trace → sequential
    step execution under the one budget → completed AgentRunResult;
    then the EXISTING Evaluator is invoked exactly once with the
    completed run + trace + criteria (spied), the artifact is written
    by the existing sink, and evaluation performs no LLM call of its
    own — with the honest run-mode ceiling visible
    (claim_traceability UNKNOWN, objective coverage evaluated)
 5  exit codes reflect the evaluation status (PASS/PARTIAL → 0,
    FAIL → 1) and the planning/execution failures (→ 6)
 6  missing API key → clear execution failure (exit 6), before any
    client construction or network use; missing Architecture Context
    artifact → clear failure (exit 6), still before client
    construction
 7  the full offline demo path (main) runs through the EXISTING
    planner loader + Agent + evaluator and prints the recorded plan

No API key, no network, no real model: scripted doubles only (the
live demonstration is the script itself, never the test suite).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from repo_intelligence.agent import Agent, AgentRunResult
from repo_intelligence.context import ArchitectureContextError
from repo_intelligence.evaluation import (
    EVALUATION_STATUSES,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationResult,
    JsonFileEvaluationSink,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "agentrun_m10_plan_demo.py"

CONTEXT_TEXT = (
    "# Architecture Context\n\n- Contract: `architecture_context.v1`\n\n"
    "## Architecture\n\n- agents: dag-agent\n- mcp: .mcp.json\n"
)
CONTEXT_RECORD = {
    "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/"
    "architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
    "snapshot_id": "main_2026-09-22_ef11dac",
}

PLAN_PAYLOAD = {
    "objective": "demo objective",
    "steps": [
        {"id": "P1", "goal": "Identify the relevant definitions"},
        {"id": "P2", "goal": "Produce an evidence-backed explanation"},
    ],
}


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "agentrun_m10_plan_demo", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # register before exec: Pydantic resolves the model's annotations
    # through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


# -- per-file test doubles (no conftest in this repo) ----------------------


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        return ToolResult(
            success=True,
            tool=tool_call.name,
            data={"path": "module.py", "content": "# evidence"},
        )


def _plan_response():
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(PLAN_PAYLOAD)),
        finish_reason="stop",
        structured_output=PLAN_PAYLOAD,
    )


def _tool_response():
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="read_file",
                    arguments={"path": "module.py"},
                )
            ],
        ),
        finish_reason="tool_calls",
    )


def _step_result_response():
    return LLMResponse(
        message=Message(
            role="assistant",
            content="module.py implements the dispatcher.",
        ),
        finish_reason="stop",
    )


def _structured_response():
    payload = {
        "finding": "Requests pass through a single dispatcher.",
        "evidence_paths": ["module.py"],
    }
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _offline_planned_run(objective="demo objective"):
    """One completed planned run through the demo's Agent composition:
    plan → P1 tool + step result → P2 structured final answer."""
    fake = FakeLLMClient(
        _plan_response(),
        _tool_response(),
        _step_result_response(),
        _structured_response(),
    )
    dispatcher = FakeDispatcher()
    agent = demo.build_agent(fake, dispatcher, None)
    result = agent.run_planned(objective, CONTEXT_TEXT, CONTEXT_RECORD)
    return result, agent, fake, dispatcher


# ---------------------------------------------------------------------------
# 1 — module imports and exposes the demo API
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "main",
        "build_agent",
        "build_criteria",
        "evaluate_run",
        "exit_code_for",
        "print_evaluation",
        "print_plan",
        "DEFAULT_OBJECTIVE",
        "MAX_TOOL_CALLS",
        "InvestigationSummary",
    ):
        assert hasattr(demo, name), name
    # the script is an entry point: planning/evaluation types come
    # from the existing packages
    assert demo.Evaluator.__name__ == "Evaluator"
    assert demo.load_architecture_context.__name__ == (
        "load_architecture_context"
    )
    assert issubclass(demo.InvestigationSummary, object)


def test_demo_has_no_planner_validator_or_loop_of_its_own():
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    # composition-root only: no tool-level get_architecture_context
    # call before planning (the existing loader is passed directly)
    assert "get_architecture_context" not in source
    # no second implementation surfaces in the script
    for banned in (
        "def validate_plan",
        "def plan_from_response",
        "class investigationplan",
        "while ",
        "def _run_loop",
        "def complete(",
    ):
        assert banned not in source, banned


# ---------------------------------------------------------------------------
# 2 — the existing Agent, constructed through the demo path
# ---------------------------------------------------------------------------


def test_build_agent_constructs_the_existing_agent():
    fake = FakeLLMClient(_plan_response())
    agent = demo.build_agent(fake, FakeDispatcher(), None)
    assert isinstance(agent, Agent)
    config = agent._execution_config  # the Agent's own config object
    assert config.max_tool_calls == demo.MAX_TOOL_CALLS
    assert config.structured_claims is False  # M8.9 mode, not M8.8
    assert agent._structured_output_model is demo.InvestigationSummary


# ---------------------------------------------------------------------------
# 3 — offline planned run: direct context, no tools during planning,
#     plan in the trace, sequential execution, one budget
# ---------------------------------------------------------------------------


def test_offline_planned_run_through_the_demo_composition():
    result, agent, fake, dispatcher = _offline_planned_run()
    assert isinstance(result, AgentRunResult)
    trace = agent.last_trace

    # the planner's orientation input was passed directly, tools=[],
    # as the FIRST request — before any dispatch
    planning_request = fake.requests[0]
    assert planning_request.tools == []
    assert planning_request.structured_output.name == "InvestigationPlan"
    assert CONTEXT_TEXT in planning_request.messages[1].content
    assert len(dispatcher.calls) == 1
    assert trace.architecture_context == CONTEXT_RECORD

    # validated plan recorded, then executed sequentially
    assert trace.plan is not None
    assert trace.plan["validation"] == {"status": "SUCCESS"}
    assert [(s["id"], s["status"]) for s in trace.plan["steps"]] == [
        ("P1", "completed"),
        ("P2", "completed"),
    ]
    assert [i.plan_step_id for i in trace.iterations] == [
        None,
        "P1",
        "P1",
        "P2",
    ]
    # one budget shared across plan steps
    assert trace.summary.max_tool_calls == demo.MAX_TOOL_CALLS
    assert trace.summary.tool_calls_executed == 1
    assert result.structured_output is not None
    assert result.structured_output.evidence_paths == ["module.py"]


# ---------------------------------------------------------------------------
# 4 — existing Evaluator invoked once, artifact written, no LLM during
#     evaluation, honest run-mode ceiling
# ---------------------------------------------------------------------------


def test_evaluate_run_invokes_existing_evaluator_and_writes_artifact(
    tmp_path, monkeypatch
):
    result, agent, fake, dispatcher = _offline_planned_run()
    criteria = demo.build_criteria(demo.DEFAULT_OBJECTIVE)
    assert isinstance(criteria, EvaluationCriteria)

    calls = []
    real_evaluator = demo.Evaluator

    class SpyEvaluator:
        @staticmethod
        def evaluate(*args, **kwargs):
            calls.append((args, kwargs))
            return real_evaluator.evaluate(*args, **kwargs)

    monkeypatch.setattr(demo, "Evaluator", SpyEvaluator)

    requests_before = len(fake.requests)
    dispatches_before = len(dispatcher.calls)
    sink = JsonFileEvaluationSink(root=tmp_path)
    evaluation = demo.evaluate_run(
        result, agent.last_trace, criteria, sink
    )

    # exactly one invocation of the existing evaluator, with the
    # completed run + its trace + the explicit criteria
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs == {}
    assert len(args) == 3
    assert args[0] is result
    assert args[1] is agent.last_trace
    assert args[2] is criteria

    # no LLM call / tool dispatch during evaluation (consumer only)
    assert len(fake.requests) == requests_before
    assert len(dispatcher.calls) == dispatches_before

    # a real EvaluationResult in the existing vocabulary
    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.status in EVALUATION_STATUSES
    assert len(evaluation.criteria) == 5

    # honest run-mode ceiling (M8.9 mode: no M8.8 claims)
    assert (
        evaluation.criterion("objective_coverage").status == STATUS_PASS
    )
    assert (
        evaluation.criterion("claim_traceability").status
        == STATUS_UNKNOWN
    )
    assert evaluation.criterion("final_result").status == STATUS_PASS
    assert (
        evaluation.criterion("evidence_gathered").status == STATUS_PASS
    )
    assert evaluation.criterion("execution_bound").status == STATUS_PASS
    assert evaluation.status == STATUS_PARTIAL

    # artifact written by the existing sink, round-trips through the
    # existing model
    json_path = sink.json_path(evaluation.run_id)
    md_path = sink.markdown_path(evaluation.run_id)
    assert json_path.is_file() and md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert EvaluationResult.from_dict(payload) == evaluation
    assert payload["run_id"] == agent.last_trace.run_id


# ---------------------------------------------------------------------------
# 5 — exit codes reflect the evaluation status / planning failures
# ---------------------------------------------------------------------------


def _evaluation_with_status(status: str) -> EvaluationResult:
    return EvaluationResult(
        status=status,
        criteria=(),
        summary="",
        run_id="run-x",
    )


def test_exit_codes_reflect_evaluation_status():
    assert (
        demo.exit_code_for(_evaluation_with_status(STATUS_PASS))
        == demo.EXIT_SUCCESS
    )
    assert (
        demo.exit_code_for(_evaluation_with_status(STATUS_PARTIAL))
        == demo.EXIT_SUCCESS
    )
    assert (
        demo.exit_code_for(_evaluation_with_status(STATUS_FAIL))
        == demo.EXIT_EVALUATION_FAILED
    )
    assert demo.EXIT_EXECUTION_FAILED == 6  # existing runner convention


def test_print_plan_projects_the_recorded_plan(capsys):
    class Trace:
        architecture_context = dict(CONTEXT_RECORD)
        plan = {
            "objective": "demo objective",
            "status": "completed",
            "validation": {"status": "SUCCESS"},
            "steps": [
                {
                    "id": "P1",
                    "goal": "Identify the relevant definitions",
                    "status": "completed",
                    "result": "found them",
                },
                {
                    "id": "P2",
                    "goal": "Produce an explanation",
                    "status": "completed",
                },
            ],
        }

    demo.print_plan(Trace())
    out = capsys.readouterr().out
    assert "validation: SUCCESS" in out
    assert "P1 [completed]" in out
    assert "found them" in out
    assert CONTEXT_RECORD["snapshot_id"] in out


# ---------------------------------------------------------------------------
# 6 — missing API key / missing Architecture Context → exit 6, no
#     client construction, no network
# ---------------------------------------------------------------------------


def test_main_without_api_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = demo.main([])

    assert code == demo.EXIT_EXECUTION_FAILED
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_main_without_architecture_context_fails_before_client(
    monkeypatch, capsys
):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-this-test")

    def _boom(*args, **kwargs):
        raise ArchitectureContextError("artifact missing")

    monkeypatch.setattr(demo, "load_architecture_context", _boom)

    def _no_client(*args, **kwargs):
        raise AssertionError("LLM client must not be constructed")

    monkeypatch.setattr(demo, "OpenAILLMClient", _no_client)

    code = demo.main([])

    assert code == demo.EXIT_EXECUTION_FAILED
    err = capsys.readouterr().err
    assert "ArchitectureContextError" in err
    assert "Re-run M6" in err


# ---------------------------------------------------------------------------
# 7 — full offline demo path: existing loader + Agent + evaluator,
#     plan printed from the recorded trace
# ---------------------------------------------------------------------------


def test_main_runs_the_complete_offline_planned_path(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-this-test")
    monkeypatch.setattr(
        demo,
        "load_architecture_context",
        lambda *a, **k: (CONTEXT_TEXT, dict(CONTEXT_RECORD)),
    )

    fake = FakeLLMClient(
        _plan_response(),
        _tool_response(),
        _step_result_response(),
        _structured_response(),
    )
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(
        demo, "OpenAILLMClient", lambda model=None: fake
    )
    monkeypatch.setattr(
        demo, "default_tool_dispatcher", lambda: dispatcher
    )
    # no trace files in the repository during tests; evaluation still
    # written through the existing sink in a temp directory
    monkeypatch.setattr(demo, "JsonFileTraceSink", lambda: None)
    monkeypatch.setattr(
        demo,
        "JsonFileEvaluationSink",
        lambda: JsonFileEvaluationSink(root=tmp_path),
    )

    code = demo.main([])

    assert code == demo.EXIT_SUCCESS
    out = capsys.readouterr().out
    # one planner request, then the existing loop's requests
    assert fake.requests[0].tools == []
    assert fake.requests[0].structured_output.name == "InvestigationPlan"
    assert len(fake.requests) == 4
    assert len(dispatcher.calls) == 1
    # the recorded plan is printed from the trace (no re-derivation)
    assert "Plan" in out
    assert "validation: SUCCESS" in out
    assert "P1 [completed]" in out
    assert "P2 [completed]" in out
    assert "plan-step timeline" in out
    assert "Overall:" in out
    assert "PARTIAL" in out  # honest M8.9 ceiling, exit still 0
    assert "exit code: 0" in out
