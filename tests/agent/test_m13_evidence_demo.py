"""M13 demonstration entry-point tests (wiring only — fully offline).

Verify ``agentrun_m13_evidence_demo.py`` is wired to the EXISTING
implementation — it must not become a second planner, validator,
loop, claim extractor, or evaluator:

 1  the module imports and exposes the demo API (no side effects)
 2  §16 controlled experiment: the SAME objective and budget as the
    M13 baseline (the M10 plan-and-execute demo)
 3  the existing Agent is constructed through the demo's build path
    (one budget, claim-bearing M8.9 contract, real Agent class)
 4  the structured schema (§18): valid + multiple claims accepted;
    empty claims and missing evidence paths rejected — deterministic,
    application-side
 5  one offline planned run through the demo composition → claims in
    the final result, inspected-only validation SUPPORTED, then the
    EXISTING Evaluator invoked exactly once (spied) — with
    claim_traceability PASS and overall PASS (the baseline ceiling)
 6  the claims chain is printed from recorded data only (§19)
 7  exit codes reflect the evaluation status (PASS/PARTIAL → 0,
    FAIL → 1); an invalid claim fails the evaluation (exit 1) with
    CLAIM_EVIDENCE_INVALID visible
 8  missing API key / missing Architecture Context → clear exit 6,
    before any client construction
 9  no second implementation in the script source

No API key, no network, no real model: scripted doubles only (the
live demonstration is the script itself, never the test suite).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_intelligence.agent import Agent, AgentRunResult
from repo_intelligence.context import ArchitectureContextError
from repo_intelligence.evaluation import (
    EVALUATION_STATUSES,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    EvaluationCriteria,
    EvaluationCriterionResult,
    EvaluationResult,
    JsonFileEvaluationSink,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "agentrun_m13_evidence_demo.py"
BASELINE_SCRIPT_PATH = PROJECT_ROOT / "agentrun_m10_plan_demo.py"

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

VALID_CLAIMS_PAYLOAD = {
    "finding": "Requests pass through a single dispatcher.",
    "evidence_paths": ["module.py"],
    "claims": [
        {
            "statement": "module.py implements the dispatcher.",
            "evidence_refs": [
                {"path": "module.py", "line_start": 1, "line_end": 20}
            ],
        }
    ],
}

INVALID_CLAIM_PAYLOAD = {
    "finding": "A file the run never opened explains everything.",
    "evidence_paths": ["other.py"],
    "claims": [
        {
            "statement": "other.py explains everything.",
            "evidence_refs": [{"path": "other.py"}],
        }
    ],
}


def _load_demo(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # register before exec: Pydantic resolves the model's annotations
    # through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo(SCRIPT_PATH, "agentrun_m13_evidence_demo")


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
            data={
                "path": "module.py",
                "line_start": 1,
                "line_end": 20,
                "content": "# evidence",
                "total_lines": 100,
            },
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


def _structured_response(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _offline_planned_run(payload=VALID_CLAIMS_PAYLOAD):
    """One completed planned run through the demo's Agent composition:
    plan → P1 tool + step result → P2 claim-bearing structured
    final answer."""
    fake = FakeLLMClient(
        _plan_response(),
        _tool_response(),
        _step_result_response(),
        _structured_response(payload),
    )
    dispatcher = FakeDispatcher()
    agent = demo.build_agent(fake, dispatcher, None)
    result = agent.run_planned(
        "demo objective", CONTEXT_TEXT, CONTEXT_RECORD
    )
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
        "print_claims",
        "DEFAULT_OBJECTIVE",
        "MAX_TOOL_CALLS",
        "InvestigationSummary",
        "InvestigationClaim",
        "EvidenceReference",
    ):
        assert hasattr(demo, name), name
    assert demo.Evaluator.__name__ == "Evaluator"
    assert demo.load_architecture_context.__name__ == (
        "load_architecture_context"
    )
    assert issubclass(demo.InvestigationSummary, object)


def test_controlled_experiment_matches_the_baseline(  # §16 / §11
):
    # the exact controlled-investigation objective the M13 experiment
    # specifies — same question as the baseline, with the required
    # final result made explicit (claims + supporting paths)
    assert demo.DEFAULT_OBJECTIVE == (
        "Explain how the ADOP agents, tools, and MCP servers are "
        "connected. Use the repository tools to gather evidence from "
        "files you actually inspect — do not rely on the initial "
        "context alone. Record your conclusion as structured output "
        "using explicit claims, where every claim contains the "
        "repository-relative paths of the files that support it."
    )
    shared_question = (
        "Explain how the ADOP agents, tools, and MCP servers are "
        "connected. Use the repository tools to gather evidence from "
        "files you actually inspect — do not rely on the initial "
        "context alone. Record your conclusion as structured output"
    )
    baseline = _load_demo(
        BASELINE_SCRIPT_PATH, "agentrun_m10_plan_demo_for_m13"
    )
    # same investigation question as the M10 baseline; the M13 run
    # makes the final-result contract explicit instead of M10's
    # field coaching, same tool budget
    assert demo.DEFAULT_OBJECTIVE.startswith(shared_question)
    assert baseline.DEFAULT_OBJECTIVE.startswith(shared_question)
    assert demo.MAX_TOOL_CALLS == baseline.MAX_TOOL_CALLS


def test_demo_has_no_second_implementation():
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    for banned in (
        "def validate_plan",
        "def gather_evidence",
        "def validate_claims",
        "claims_from_payload",
        "def _run_loop",
        "class investigationplan",
        "while ",
        "def complete(",
        "llm-as-judge",
    ):
        assert banned not in source, banned


# ---------------------------------------------------------------------------
# 2 — the existing Agent, constructed through the demo path
# ---------------------------------------------------------------------------


def test_build_agent_constructs_the_existing_agent():
    fake = FakeLLMClient(_plan_response())
    agent = demo.build_agent(fake, FakeDispatcher(), None)
    assert isinstance(agent, Agent)
    config = agent._execution_config
    assert config.max_tool_calls == demo.MAX_TOOL_CALLS == 60
    assert config.structured_claims is False  # one typed contract (M13)
    assert agent._structured_output_model is demo.InvestigationSummary


# ---------------------------------------------------------------------------
# 3 — structured schema (§18): valid, multiple, empty, missing evidence
# ---------------------------------------------------------------------------


def test_schema_accepts_valid_and_multiple_claims():
    model = demo.InvestigationSummary(
        finding="Agents connect through declared tool references.",
        evidence_paths=["a.md", "b.md"],
        claims=[
            {
                "statement": "first claim",
                "evidence_refs": [{"path": "a.md"}],
            },
            {
                "statement": "second claim",
                "evidence_refs": [
                    {"path": "b.md", "line_start": 1, "line_end": 2}
                ],
            },
        ],
    )
    assert model.finding.startswith("Agents")
    assert len(model.claims) == 2
    assert model.claims[1].evidence_refs[0].line_end == 2


def test_schema_rejects_empty_claim():
    base = {
        "finding": "f",
        "evidence_paths": ["a.md"],
        "claims": [
            {"statement": "s", "evidence_refs": [{"path": "a.md"}]}
        ],
    }

    for empty in ("", "   "):
        payload = dict(base)
        payload["claims"] = [
            {"statement": empty, "evidence_refs": [{"path": "a.md"}]}
        ]
        with pytest.raises(ValidationError):
            demo.InvestigationSummary(**payload)


def test_schema_rejects_missing_evidence_paths():
    # the claims array itself is required (§14: success means claims)
    with pytest.raises(ValidationError):
        demo.InvestigationSummary(
            finding="f", evidence_paths=["a.md"]
        )
    # every claim must cite at least one inspected file
    for claim in (
        {"statement": "no refs key"},
        {"statement": "empty refs", "evidence_refs": []},
    ):
        with pytest.raises(ValidationError):
            demo.InvestigationSummary(
                finding="f", evidence_paths=["a.md"], claims=[claim]
            )


def test_schema_requires_finding_and_evidence_paths():
    with pytest.raises(ValidationError):
        demo.InvestigationSummary(
            evidence_paths=["a.md"],
            claims=[
                {"statement": "s", "evidence_refs": [{"path": "a.md"}]}
            ],
        )
    with pytest.raises(ValidationError):
        demo.InvestigationSummary(
            finding="f",
            evidence_paths=[],
            claims=[
                {"statement": "s", "evidence_refs": [{"path": "a.md"}]}
            ],
        )


def test_model_descriptions_carry_the_structured_instruction():
    schema = demo.InvestigationSummary.schema()
    definitions = schema.get("$defs", schema.get("definitions", {}))
    claim_schema = definitions["InvestigationClaim"]
    assert "inspected" in schema["properties"]["claims"][
        "description"
    ].lower()
    assert "uncertainty" in claim_schema["properties"]["statement"][
        "description"
    ]
    assert "CLAIM_EVIDENCE_INVALID" in claim_schema["properties"][
        "evidence_refs"
    ]["description"]


def test_build_criteria_require_claims_and_evidence():
    criteria = demo.build_criteria(demo.DEFAULT_OBJECTIVE)
    assert isinstance(criteria, EvaluationCriteria)
    assert [item.key for item in criteria.objective_criteria] == [
        "finding",
        "evidence_paths",
        "claims",
    ]


# ---------------------------------------------------------------------------
# 4 — offline planned run: claims in the result, validation, evaluation
# ---------------------------------------------------------------------------


def test_offline_planned_run_produces_evidence_backed_claims():
    result, agent, fake, dispatcher = _offline_planned_run()
    assert isinstance(result, AgentRunResult)
    trace = agent.last_trace

    # plan recorded before any dispatch (existing M10 semantics)
    assert trace.plan is not None
    assert trace.plan["validation"] == {"status": "SUCCESS"}
    assert [i.plan_step_id for i in trace.iterations] == [
        None,
        "P1",
        "P1",
        "P2",
    ]
    assert len(dispatcher.calls) == 1

    # the M13 contract produced claims + inspected-only validation
    assert isinstance(
        result.structured_output, demo.InvestigationSummary
    )
    assert len(result.claims) == 1
    assert result.claims[0].statement == (
        "module.py implements the dispatcher."
    )
    assert len(result.evidence_validation) == 1
    entry = result.evidence_validation[0]
    assert entry.status == "SUPPORTED"
    assert entry.code is None
    assert entry.matched_evidence[0].tool == "read_file"
    # the whole chain is on the one trace, ready to inspect (§19)
    block = trace.to_dict()["evidence_validation"]
    assert block["results"][0]["status"] == "SUPPORTED"


def test_evaluate_run_yields_traceability_and_overall_pass(
    tmp_path, monkeypatch
):
    result, agent, fake, dispatcher = _offline_planned_run()
    criteria = demo.build_criteria(demo.DEFAULT_OBJECTIVE)

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

    # exactly one invocation of the existing evaluator; evaluation is
    # a pure consumer (no LLM call, no tool dispatch)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs == {}
    assert args[0] is result
    assert args[1] is agent.last_trace
    assert args[2] is criteria
    assert len(fake.requests) == requests_before
    assert len(dispatcher.calls) == dispatches_before

    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.status in EVALUATION_STATUSES
    assert len(evaluation.criteria) == 5
    # M13: traceability is MEASURED (the baseline reported UNKNOWN)
    assert (
        evaluation.criterion("claim_traceability").status == STATUS_PASS
    )
    assert (
        evaluation.criterion("objective_coverage").status == STATUS_PASS
    )
    assert evaluation.status == STATUS_PASS

    # artifact written by the existing sink, round-trips
    json_path = sink.json_path(evaluation.run_id)
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert EvaluationResult.from_dict(payload) == evaluation
    assert payload["run_id"] == agent.last_trace.run_id


# ---------------------------------------------------------------------------
# 5 — the claims chain printed from recorded data only (§19)
# ---------------------------------------------------------------------------


def test_print_claims_projects_the_recorded_chain(capsys):
    result, agent, fake, dispatcher = _offline_planned_run()
    demo.print_claims(result, agent.last_trace)
    out = capsys.readouterr().out

    assert "Claims and evidence" in out
    assert "module.py implements the dispatcher." in out
    assert "evidence: module.py:1-20" in out
    assert "validation: SUPPORTED" in out
    assert "inspected: module.py:1-20" in out
    assert "iteration" in out and "read_file" in out
    assert "1 supported, 0 unsupported, 0 unevaluable" in out


def test_print_claims_reports_an_invalid_claim(capsys):
    result, agent, fake, dispatcher = _offline_planned_run(
        INVALID_CLAIM_PAYLOAD
    )
    demo.print_claims(result, agent.last_trace)
    out = capsys.readouterr().out

    assert "validation: UNSUPPORTED [CLAIM_EVIDENCE_INVALID]" in out
    assert "never gathered" in out
    assert "0 supported, 1 unsupported, 0 unevaluable" in out


def test_print_evaluation_projects_existing_vocabulary(capsys):
    evaluation = EvaluationResult(
        status=STATUS_PASS,
        criteria=(
            EvaluationCriterionResult(
                "claim_traceability",
                STATUS_PASS,
                "1 of 1 claim(s) supported",
            ),
        ),
        summary="5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL",
        run_id="run-x",
    )
    demo.print_evaluation(evaluation)
    flat = " ".join(capsys.readouterr().out.split())
    assert "Claim traceability: PASS" in flat
    assert "Overall: PASS" in flat
    assert "5 PASS" in flat


# ---------------------------------------------------------------------------
# 6 — exit codes / invalid claim fails the evaluation (exit 1)
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
# 7 — missing API key / missing Architecture Context → exit 6
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
# 8 — full offline demo paths: success (exit 0) and invalid claim
#     (exit 1), both printed from the recorded run
# ---------------------------------------------------------------------------


def _wire_main(monkeypatch, tmp_path, payload):
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
        _structured_response(payload),
    )
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(demo, "OpenAILLMClient", lambda model=None: fake)
    monkeypatch.setattr(
        demo, "default_tool_dispatcher", lambda: dispatcher
    )
    monkeypatch.setattr(demo, "JsonFileTraceSink", lambda: None)
    monkeypatch.setattr(
        demo,
        "JsonFileEvaluationSink",
        lambda: JsonFileEvaluationSink(root=tmp_path),
    )
    return fake, dispatcher


def test_main_runs_the_complete_offline_planned_path(
    monkeypatch, capsys, tmp_path
):
    fake, dispatcher = _wire_main(
        monkeypatch, tmp_path, VALID_CLAIMS_PAYLOAD
    )

    code = demo.main([])

    assert code == demo.EXIT_SUCCESS
    out = capsys.readouterr().out
    # one planner request, then the existing loop's requests
    assert fake.requests[0].tools == []
    assert fake.requests[0].structured_output.name == "InvestigationPlan"
    assert len(fake.requests) == 4
    assert len(dispatcher.calls) == 1
    # the recorded chain (§19): plan → execution → claims → validation
    assert "validation: SUCCESS" in out
    assert "P1 [completed]" in out
    assert "Claims and evidence" in out
    assert "validation: SUPPORTED" in out
    # M13: every criterion established — no honest-but-weak ceiling
    flat = " ".join(out.split())
    assert "Claim traceability: PASS" in flat
    assert "Overall: PASS" in flat
    assert "Evaluation completed: PASS" in flat
    assert "exit code: 0" in flat


def test_main_fails_the_evaluation_when_a_claim_is_invalid(
    monkeypatch, capsys, tmp_path
):
    _wire_main(monkeypatch, tmp_path, INVALID_CLAIM_PAYLOAD)

    code = demo.main([])

    assert code == demo.EXIT_EVALUATION_FAILED
    out = capsys.readouterr().out
    assert "CLAIM_EVIDENCE_INVALID" in out
    flat = " ".join(out.split())
    assert "Claim traceability: FAIL" in flat
    assert "Overall: FAIL" in flat
    assert "Investigation completed, but evaluation failed." in flat
    assert "exit code: 1 (evaluation: FAIL)" in flat
