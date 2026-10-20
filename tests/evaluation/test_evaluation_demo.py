"""M9 demonstration entry-point tests (wiring only — fully offline).

Verify ``agentrun_m9_evaluation_demo.py`` is wired to the EXISTING
implementation — it must not become a second evaluator or a second
loop:

 1  the module imports and exposes the demo API (no side effects)
 2  the existing Agent is constructed through the demo's build path
    (one budget, M8.9 structured-output contract, real Agent class)
 3  the EXISTING Evaluator is invoked exactly once with the completed
    run + trace + criteria (spied), the artifact is written by the
    existing sink, and evaluation performs no LLM call of its own —
    with the honest run-mode ceiling visible (claim_traceability
    UNKNOWN, objective coverage evaluated)
 4  exit codes reflect the evaluation status (PASS/PARTIAL → 0,
    FAIL → 1)
 5  missing API key → clear execution failure (exit 6), before any
    client construction or network use

No API key, no network, no real model: scripted doubles only (the
live demonstration is the script itself, never the test suite).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from repo_intelligence.agent import Agent, AgentRunResult
from repo_intelligence.evaluation import (
    EVALUATION_STATUSES,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationCriterionResult,
    EvaluationResult,
    JsonFileEvaluationSink,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "agentrun_m9_evaluation_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "agentrun_m9_evaluation_demo", SCRIPT_PATH
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


def _offline_run():
    """One completed run through the demo's Agent composition."""
    fake = FakeLLMClient(_tool_response(), _structured_response())
    dispatcher = FakeDispatcher()
    agent = demo.build_agent(fake, dispatcher, None)
    result = agent.run_result(demo.DEFAULT_OBJECTIVE)
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
        "DEFAULT_OBJECTIVE",
        "MAX_TOOL_CALLS",
        "InvestigationSummary",
    ):
        assert hasattr(demo, name), name
    # the script is an entry point: evaluation types come from M9
    assert demo.Evaluator.__name__ == "Evaluator"
    assert issubclass(demo.InvestigationSummary, object)


def test_demo_objective_is_repository_agnostic():
    lowered = (
        SCRIPT_PATH.read_text(encoding="utf-8").lower()
        + demo.DEFAULT_OBJECTIVE.lower()
    )
    for term in (
        "adop",
        "agentic-ai",
        "dag-agent",
        "servers.yaml",
        "sample-agentic",
        "get_architecture_context",
        "aws",
        "tool-registry",
    ):
        assert term not in lowered, term


# ---------------------------------------------------------------------------
# 2 — the existing Agent, constructed through the demo path
# ---------------------------------------------------------------------------


def test_build_agent_constructs_the_existing_agent():
    fake = FakeLLMClient(_structured_response())
    agent = demo.build_agent(fake, FakeDispatcher(), None)
    assert isinstance(agent, Agent)
    config = agent._execution_config  # the Agent's own config object
    assert config.max_tool_calls == demo.MAX_TOOL_CALLS
    assert config.structured_claims is False  # M8.9 mode, not M8.8
    assert agent._structured_output_model is demo.InvestigationSummary


# ---------------------------------------------------------------------------
# 3 — existing Evaluator invoked once, artifact written, no LLM during
#     evaluation, honest run-mode ceiling
# ---------------------------------------------------------------------------


def test_evaluate_run_invokes_existing_evaluator_and_writes_artifact(
    tmp_path, monkeypatch
):
    result, agent, fake, dispatcher = _offline_run()
    assert isinstance(result, AgentRunResult)
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

    # this run mode: coverage evaluated from the typed result,
    # traceability honestly UNKNOWN (no M8.8 claims in M8.9 mode)
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
    # overall cannot be full PASS while a criterion is UNKNOWN
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
# 4 — exit codes reflect the evaluation status
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


def test_console_projection_uses_existing_labels_and_vocabulary(
    capsys,
):
    evaluation = EvaluationResult(
        status=STATUS_PARTIAL,
        criteria=(
            EvaluationCriterionResult(
                "final_result", STATUS_PASS, "final answer present"
            ),
            EvaluationCriterionResult(
                "evidence_gathered", STATUS_PASS, "1 successful"
            ),
            EvaluationCriterionResult(
                "claim_traceability",
                STATUS_UNKNOWN,
                "run recorded no claims",
            ),
            EvaluationCriterionResult(
                "execution_bound", STATUS_PASS, "executed 1 of 10"
            ),
            EvaluationCriterionResult(
                "objective_coverage", STATUS_PASS, "all 2 present"
            ),
        ),
        summary="4 PASS, 0 PARTIAL, 1 UNKNOWN, 0 FAIL",
        run_id="run-x",
    )
    demo.print_evaluation(evaluation)
    out = capsys.readouterr().out
    assert "Final result:" in out
    assert "Claim traceability:" in out
    assert "Objective coverage:" in out
    assert "Overall:" in out
    assert "PARTIAL" in out
    assert "run recorded no claims" in out  # explanation for UNKNOWN


# ---------------------------------------------------------------------------
# 5 — missing API key → clear execution failure, no network
# ---------------------------------------------------------------------------


def test_main_without_api_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = demo.main([])

    assert code == demo.EXIT_EXECUTION_FAILED
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
