"""M9 integration tests — a completed real Agent run, evaluated.

End-to-end through the REAL Agent (scripted offline model + canned
dispatcher — no key, no network), then ``Evaluator.evaluate`` over the
run's own artifacts. Evaluation happens strictly AFTER the run and is
pure: same trace object, same verdict, and zero interaction with the
model or dispatcher while evaluating.

    1 claims run        M8.8 validation result CONSUMED (traceability
                        PASS "1 of 1", coverage honestly UNKNOWN — a
                        claims run has no structured result)
    2 structured run    M8.9 typed result CONSUMED (objective coverage
                        evaluated field-by-field, never reparsed)
    3 budget exhausted  EXECUTION_LIMIT_REACHED run: bound PASS while
                        the run as a whole FAILs (spec §16/§17: compliance
                        ≠ success)
    4 no side effects   evaluating changes neither the LLM request
                        count nor the dispatcher call count (AC-010,
                        AC-011)
    5 determinism       evaluating the same completed run twice →
                        byte-identical payloads
    6 repo-agnostic     generic config objective, no repository terms

Regression: the full existing suite (tests/agent, tests/evidence,
tests/llm) runs with this file — see the documented command.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
)
from repo_intelligence.evaluation import (
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_EVIDENCE_GATHERED,
    CRITERION_EXECUTION_BOUND,
    CRITERION_FINAL_RESULT,
    CRITERION_OBJECTIVE_COVERAGE,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    Evaluator,
    ObjectiveCriterion,
)
from repo_intelligence.evidence import STATUS_SUPPORTED
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

QUESTION = "Explain how configuration reaches the application."
ANSWER = "config.yaml is read by loader.py and passed to application.py."
PATH_CONFIG = "config.yaml"

STATEMENT = "Configuration is read from config.yaml."


class ConfigSummary(BaseModel):
    """Generic structured contract (repo-agnostic — spec §20)."""

    entry_point: str
    configuration_source: str
    loader: str


FULL_STRUCTURED = ConfigSummary(
    entry_point="application.py:main()",
    configuration_source="config.yaml",
    loader="loader.py",
)
INCOMPLETE_STRUCTURED = ConfigSummary(
    entry_point="application.py:main()",
    configuration_source="config.yaml",
    loader="",  # expected but empty → objective coverage FAIL
)


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
    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _read(path=PATH_CONFIG, start=1, end=20):
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": path,
            "line_start": start,
            "line_end": end,
            "content": "# config excerpt\n",
        },
        source={"repository": "generic-repo", "commit_sha": "a" * 40},
    )


def _tool_response(call_id="c1", path=PATH_CONFIG):
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="read_file",
                    arguments={"path": path, "line_start": 1, "line_end": 20},
                )
            ],
        ),
        finish_reason="tool_calls",
    )


def _text_response(content):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _structured_response(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _claims_final(answer=ANSWER, statement=STATEMENT, path=PATH_CONFIG):
    payload = {
        "answer": answer,
        "claims": [
            {
                "statement": statement,
                "evidence_refs": [{"path": path, "line_start": 1, "line_end": 20}],
            }
        ],
    }
    return _text_response(json.dumps(payload))


def _claims_run():
    """Real Agent run producing M8.8-validated structured claims."""
    fake = FakeLLMClient(_tool_response(), _claims_final())
    dispatcher = FakeDispatcher(_read())
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_tool_calls=8, structured_claims=True
        ),
    )
    result = agent.run_result(QUESTION)
    return result, agent, fake, dispatcher


def _structured_run(structured=FULL_STRUCTURED):
    """Real Agent run producing an M8.9 typed structured result."""
    payload = structured.dict()
    fake = FakeLLMClient(_tool_response(), _structured_response(payload))
    dispatcher = FakeDispatcher(_read())
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        structured_output_model=ConfigSummary,
    )
    result = agent.run_result(QUESTION)
    return result, agent, fake, dispatcher


def _full_criteria():
    return EvaluationCriteria(
        objective=QUESTION,
        objective_criteria=(
            ObjectiveCriterion("entry_point"),
            ObjectiveCriterion("configuration_source"),
            ObjectiveCriterion("loader"),
        ),
    )


# ---------------------------------------------------------------------------
# 1 — claims run: M8.8 consumed, never recomputed
# ---------------------------------------------------------------------------


def test_claims_run_evaluation_consumes_evidence_validation():
    result, agent, fake, dispatcher = _claims_run()
    assert len(result.evidence_validation) == 1
    assert result.evidence_validation[0].status == STATUS_SUPPORTED

    evaluation = Evaluator.evaluate(result, agent.last_trace)

    traceability = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert traceability.status == STATUS_PASS
    assert "1 of 1" in traceability.explanation
    assert evaluation.criterion(CRITERION_FINAL_RESULT).status == STATUS_PASS
    assert (
        evaluation.criterion(CRITERION_EVIDENCE_GATHERED).status == STATUS_PASS
    )
    # honest ceiling: a claims run has no structured result → UNKNOWN,
    # never a silent PASS (a claims run cannot define field coverage)
    coverage = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert coverage.status == STATUS_UNKNOWN
    # overall is not PASS: an unevaluable criterion is not a success
    assert evaluation.status != STATUS_PASS
    assert evaluation.run_id == agent.last_trace.to_dict()["run_id"]


def test_claims_evaluation_does_not_revalidate():
    """The recorded statuses are consumed as-is: evaluation performs
    zero additional validation work (nothing to observe but a stable
    verdict over unchanged inputs)."""
    result, agent, fake, dispatcher = _claims_run()
    before = result.evidence_validation
    evaluation = Evaluator.evaluate(result, agent.last_trace)
    assert result.evidence_validation == before
    assert len(evaluation.criterion(CRITERION_CLAIM_TRACEABILITY).explanation) > 0


# ---------------------------------------------------------------------------
# 2 — structured run: M8.9 consumed, never reparsed
# ---------------------------------------------------------------------------


def test_structured_run_objective_coverage_all_present():
    result, agent, fake, dispatcher = _structured_run()
    assert isinstance(result.structured_output, ConfigSummary)

    evaluation = Evaluator.evaluate(
        result, agent.last_trace, _full_criteria()
    )
    coverage = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert coverage.status == STATUS_PASS
    assert "all 3" in coverage.explanation
    # honest ceiling (M8.9 spec): a structured-output run has no
    # claims → traceability is UNKNOWN, so overall is PARTIAL — the
    # evaluator never pretends an unevaluable criterion passed
    assert evaluation.criterion(CRITERION_CLAIM_TRACEABILITY).status == (
        STATUS_UNKNOWN
    )
    assert evaluation.status == STATUS_PARTIAL


def test_structured_run_objective_coverage_missing_field_fails():
    result, agent, fake, dispatcher = _structured_run(INCOMPLETE_STRUCTURED)
    evaluation = Evaluator.evaluate(
        result, agent.last_trace, _full_criteria()
    )
    coverage = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert coverage.status == STATUS_FAIL
    assert "loader" in coverage.explanation
    assert evaluation.status == STATUS_FAIL


def test_structured_run_without_expected_criteria_is_unknown():
    result, agent, fake, dispatcher = _structured_run()
    evaluation = Evaluator.evaluate(result, agent.last_trace)
    coverage = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert coverage.status == STATUS_UNKNOWN
    assert evaluation.status == STATUS_PARTIAL


# ---------------------------------------------------------------------------
# 3 — budget exhausted: compliance PASS, run FAIL (spec §16/§17)
# ---------------------------------------------------------------------------


def test_limit_reached_run_respects_bound_but_fails_objective():
    fake = FakeLLMClient(
        _tool_response(call_id="c1"),
        _tool_response(call_id="c2"),
    )
    dispatcher = FakeDispatcher(_read())
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=1),
    )
    with pytest.raises(AgentError) as excinfo:
        agent.run_result(QUESTION)
    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"

    # no AgentRunResult exists — evaluate the completed trace directly
    evaluation = Evaluator.evaluate(None, agent.last_trace, _full_criteria())

    assert (
        evaluation.criterion(CRITERION_EXECUTION_BOUND).status == STATUS_PASS
    )
    assert evaluation.criterion(CRITERION_FINAL_RESULT).status == STATUS_FAIL
    assert evaluation.status == STATUS_FAIL
    bound = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert "1 of 1" in bound.explanation
    assert "blocked" in bound.explanation


# ---------------------------------------------------------------------------
# 4 — evaluation touches neither model nor tools (AC-010, AC-011)
# ---------------------------------------------------------------------------


def test_evaluation_dispatches_no_tool_and_no_llm_call():
    result, agent, fake, dispatcher = _claims_run()
    requests_before = len(fake.requests)
    calls_before = len(dispatcher.calls)

    for _ in range(3):  # repeat: still zero side effects
        Evaluator.evaluate(result, agent.last_trace, _full_criteria())

    assert len(fake.requests) == requests_before
    assert len(dispatcher.calls) == calls_before


# ---------------------------------------------------------------------------
# 5 — determinism over a real run's artifacts
# ---------------------------------------------------------------------------


def test_real_run_evaluation_is_deterministic():
    result, agent, fake, dispatcher = _structured_run()
    first = Evaluator.evaluate(
        result, agent.last_trace, _full_criteria()
    )
    second = Evaluator.evaluate(
        result, agent.last_trace, _full_criteria()
    )
    assert first.to_json() == second.to_json()
    # and the trace mapping form gives the same verdict as the object
    via_mapping = Evaluator.evaluate(
        result, agent.last_trace.to_dict(), _full_criteria()
    )
    assert via_mapping.to_json() == first.to_json()


# ---------------------------------------------------------------------------
# 6 — repository-agnostic: nothing about this fixture's repository
# ---------------------------------------------------------------------------


def test_evaluation_payload_free_of_repository_terminology():
    result, agent, fake, dispatcher = _structured_run()
    evaluation = Evaluator.evaluate(
        result, agent.last_trace, _full_criteria()
    )
    blob = json.dumps(evaluation.to_dict()).lower()
    for term in ("adop", "mcp", "claude", "dag-agent", "openai"):
        assert term not in blob, term
