"""M9 — deterministic evaluator tests (spec §8, §16, §17, §20).

The five criteria, their failure semantics, and the multi-dimensional
overall result — no LLM, no tools, no repository, no network:

 1  final result: exists / missing / empty answer
 2  evidence gathered: successes count, failures never, zero ≠ pass
 3  claim traceability: all supported / some unknown / unsupported /
    none produced — the M8.8 result is CONSUMED, never recomputed
 4  execution bound: zero / below / at limit; blocked ≠ executed;
    missing summary → UNKNOWN; over-limit → FAIL
 5  objective coverage: all present / missing field / empty expected
    criteria → UNKNOWN / typed (M8.9) result consumed as-is
 6  overall aggregation: all pass → PASS; one fail → FAIL; multiple
    fails → FAIL; any non-clean state → PARTIAL
 7  Run A / Run B / Run C (spec §17): execution compliance ≠ success
 8  determinism: same inputs → identical payload
 9  repo-agnostic fixture (spec §20): generic config-loading objective
10  source guarantee: evaluation performs no LLM/tool/agent-loop work
    and does not reimplement M8.8
11  input errors: malformed execution data raises; evaluation
    outcomes never raise

Loop mechanics: tests/agent/. Evidence algorithm: tests/evidence/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import AgentRunResult
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.evaluation import (
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_EVIDENCE_GATHERED,
    CRITERION_EXECUTION_BOUND,
    CRITERION_FINAL_RESULT,
    CRITERION_OBJECTIVE_COVERAGE,
    CRITERIONS,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationResult,
    Evaluator,
    ObjectiveCriterion,
)
from repo_intelligence.evidence import (
    STATUS_SUPPORTED,
    STATUS_UNKNOWN as VALIDATION_UNKNOWN,
    STATUS_UNSUPPORTED,
    Claim,
    ClaimValidation,
    EvidenceRef,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_PACKAGE = PROJECT_ROOT / "src" / "repo_intelligence" / "evaluation"

SOURCE = {
    "repository": "sample-repo-alpha",
    "commit_sha": "b" * 40,
}


# ---------------------------------------------------------------------------
# Deterministic execution fixtures (same shapes AgentTrace.to_dict emits)
# ---------------------------------------------------------------------------


def _tool_result(name, success=True, call_id="c1", data=None):
    return {
        "tool": name,
        "tool_call_id": call_id,
        "success": success,
        "data": data if data is not None else {"path": "file.txt"},
        "source": dict(SOURCE) if success else None,
        "error": None if success else {"code": "E"},
    }


def _iteration(n, tool_results=()):
    return {
        "iteration": n,
        "response_type": "TOOL_CALLS",
        "tool_results": list(tool_results),
    }


def _trace(
    run_id="run-eval",
    iterations=None,
    max_tool_calls=8,
    tool_calls_executed=2,
    tool_calls_blocked=0,
    with_summary=True,
    termination_reason="FINAL_ANSWER",
):
    """A trace mapping: what happened, recorded once, read-only."""
    if iterations is None:
        iterations = [
            _iteration(1, [_tool_result("read_file", call_id="c1")]),
            _iteration(2, [_tool_result("read_file", success=False, call_id="c2")]),
            _iteration(3),
        ]
    payload = {
        "run_id": run_id,
        "iterations": iterations,
    }
    if with_summary:
        payload["summary"] = {
            "total_iterations": len(iterations),
            "termination_reason": termination_reason,
            "max_tool_calls": max_tool_calls,
            "tool_calls_executed": tool_calls_executed,
            "tool_calls_blocked": tool_calls_blocked,
        }
    return payload


class ConfigSummary(BaseModel):
    """Generic repo-agnostic structured result (spec §20 example)."""

    entry_point: str
    configuration_source: str
    loader: str


def _result(
    answer="Configuration flows from config.yaml through loader.py.",
    claims=(),
    validation=(),
    structured=None,
):
    return AgentRunResult(
        answer=answer,
        claims=tuple(claims),
        evidence_validation=tuple(validation),
        structured_output=structured,
    )


def _supported(statement="Config is loaded from config.yaml."):
    claim = Claim(
        statement=statement,
        evidence_refs=(EvidenceRef(path="config.yaml"),),
    )
    return ClaimValidation(
        claim=claim,
        status=STATUS_SUPPORTED,
        matched_evidence=(),
        reason="matched gathered evidence: read_file config.yaml",
    )


def _validation(status):
    return ClaimValidation(
        claim=Claim(statement="s"),
        status=status,
        matched_evidence=(),
        reason="deterministic reason",
    )


def _criteria(*keys, objective="Explain how configuration is loaded."):
    return EvaluationCriteria(
        objective=objective,
        objective_criteria=tuple(ObjectiveCriterion(key) for key in keys),
    )


GOOD_STRUCTURED = ConfigSummary(
    entry_point="main.py:load_config()",
    configuration_source="config.yaml",
    loader="loader.py",
)


# ---------------------------------------------------------------------------
# 1 — final result
# ---------------------------------------------------------------------------


def test_final_result_present_passes():
    evaluation = Evaluator.evaluate(_result(), _trace())
    item = evaluation.criterion(CRITERION_FINAL_RESULT)
    assert item.status == STATUS_PASS
    assert "characters" in item.explanation


def test_final_result_missing_fails():
    evaluation = Evaluator.evaluate(None, _trace())
    item = evaluation.criterion(CRITERION_FINAL_RESULT)
    assert item.status == STATUS_FAIL
    assert "no final result" in item.explanation


def test_final_result_blank_answer_fails():
    evaluation = Evaluator.evaluate(_result(answer="   "), _trace())
    assert evaluation.criterion(CRITERION_FINAL_RESULT).status == STATUS_FAIL


# ---------------------------------------------------------------------------
# 2 — evidence gathered
# ---------------------------------------------------------------------------


def test_successful_tool_execution_counts_as_evidence():
    evaluation = Evaluator.evaluate(_result(), _trace())
    item = evaluation.criterion(CRITERION_EVIDENCE_GATHERED)
    assert item.status == STATUS_PASS
    assert "1 successful" in item.explanation  # the failed c2 doesn't count


def test_final_answer_alone_is_not_evidence():
    trace = _trace(
        iterations=[_iteration(1, [_tool_result("read_file", success=False)])],
        tool_calls_executed=1,
    )
    evaluation = Evaluator.evaluate(_result(), trace)
    item = evaluation.criterion(CRITERION_EVIDENCE_GATHERED)
    assert item.status == STATUS_FAIL
    assert "none succeeded" in item.explanation


def test_zero_tool_calls_fails_evidence_criterion():
    trace = _trace(iterations=[_iteration(1)], tool_calls_executed=0)
    evaluation = Evaluator.evaluate(_result(), trace)
    item = evaluation.criterion(CRITERION_EVIDENCE_GATHERED)
    assert item.status == STATUS_FAIL
    assert "no evidence" in item.explanation


# ---------------------------------------------------------------------------
# 3 — claim traceability (M8.8 consumed, never recomputed)
# ---------------------------------------------------------------------------


def test_all_claims_supported_passes():
    result = _result(validation=(_supported(), _supported("Two.")))
    evaluation = Evaluator.evaluate(result, _trace())
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_PASS
    assert "2 of 2" in item.explanation


def test_some_unknown_claims_is_partial():
    result = _result(
        validation=(
            _supported(),
            _validation(VALIDATION_UNKNOWN),
        )
    )
    evaluation = Evaluator.evaluate(result, _trace())
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_PARTIAL
    assert "1 of 2" in item.explanation


def test_unsupported_claims_fail_when_none_supported():
    result = _result(
        validation=(_validation(STATUS_UNSUPPORTED), _validation(STATUS_UNSUPPORTED))
    )
    evaluation = Evaluator.evaluate(result, _trace())
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_FAIL
    assert "0 of 2" in item.explanation


def test_mixture_with_supported_and_unsupported_is_partial():
    result = _result(
        validation=(_supported(), _validation(STATUS_UNSUPPORTED))
    )
    evaluation = Evaluator.evaluate(result, _trace())
    assert (
        evaluation.criterion(CRITERION_CLAIM_TRACEABILITY).status
        == STATUS_PARTIAL
    )


def test_no_claims_recorded_is_unknown_not_pass():
    evaluation = Evaluator.evaluate(_result(), _trace())
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_UNKNOWN
    assert "no claims" in item.explanation


def test_validation_statuses_are_consumed_verbatim():
    """The evaluator must not recompute M8.8 — a recorded status is
    taken as-is (proof: statuses here would only survive if consumed)."""
    statuses = (STATUS_SUPPORTED, STATUS_UNSUPPORTED, VALIDATION_UNKNOWN)
    result = _result(validation=tuple(_validation(s) for s in statuses))
    evaluation = Evaluator.evaluate(result, _trace())
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_PARTIAL
    assert "1 of 3" in item.explanation


# ---------------------------------------------------------------------------
# 4 — execution bound (M8.5 semantics: blocked ≠ executed)
# ---------------------------------------------------------------------------


def test_zero_executions_within_bound_passes():
    evaluation = Evaluator.evaluate(
        _result(), _trace(tool_calls_executed=0, tool_calls_blocked=0)
    )
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert item.status == STATUS_PASS
    assert "0 of 8" in item.explanation


def test_below_limit_passes():
    evaluation = Evaluator.evaluate(
        _result(), _trace(max_tool_calls=5, tool_calls_executed=3)
    )
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert item.status == STATUS_PASS
    assert "3 of 5" in item.explanation


def test_exactly_at_limit_passes():
    evaluation = Evaluator.evaluate(
        _result(), _trace(max_tool_calls=5, tool_calls_executed=5)
    )
    assert (
        evaluation.criterion(CRITERION_EXECUTION_BOUND).status == STATUS_PASS
    )


def test_blocked_calls_never_count_as_executions():
    """At the limit with blocked follow-ups: executed stays == max,
    blocked is reported separately (M8.5 semantics preserved)."""
    evaluation = Evaluator.evaluate(
        _result(),
        _trace(max_tool_calls=5, tool_calls_executed=5, tool_calls_blocked=3),
    )
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert item.status == STATUS_PASS
    assert "3 blocked" in item.explanation
    payload = evaluation.to_dict()
    by_id = {c["criterion"]: c for c in payload["criteria"]}
    assert "blocked call(s) not counted" in by_id[CRITERION_EXECUTION_BOUND][
        "explanation"
    ]


def test_executed_over_limit_fails_defensively():
    evaluation = Evaluator.evaluate(
        _result(), _trace(max_tool_calls=2, tool_calls_executed=5)
    )
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert item.status == STATUS_FAIL
    assert "exceeds" in item.explanation


def test_missing_summary_is_unknown():
    evaluation = Evaluator.evaluate(_result(), _trace(with_summary=False))
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert item.status == STATUS_UNKNOWN
    assert "summary not recorded" in item.explanation


def test_summary_without_budget_is_unknown():
    trace = _trace()
    del trace["summary"]["max_tool_calls"]
    evaluation = Evaluator.evaluate(_result(), trace)
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    assert item.status == STATUS_UNKNOWN
    assert "budget" in item.explanation


# ---------------------------------------------------------------------------
# 5 — objective coverage (explicit criteria; M8.9 result consumed)
# ---------------------------------------------------------------------------


def test_all_expected_fields_present_passes():
    result = _result(structured=GOOD_STRUCTURED)
    evaluation = Evaluator.evaluate(
        result,
        _trace(),
        _criteria(
            "entry_point", "configuration_source", "loader"
        ),
    )
    item = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert item.status == STATUS_PASS
    assert "all 3" in item.explanation


def test_missing_expected_field_fails_and_names_it():
    result = _result(structured=GOOD_STRUCTURED)
    evaluation = Evaluator.evaluate(
        result,
        _trace(),
        _criteria("entry_point", "missing_field"),
    )
    item = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert item.status == STATUS_FAIL
    assert "missing_field" in item.explanation


def test_empty_text_field_counts_as_missing():
    result = _result(
        structured=ConfigSummary(
            entry_point=" ",
            configuration_source="config.yaml",
            loader="loader.py",
        )
    )
    evaluation = Evaluator.evaluate(
        result, _trace(), _criteria("entry_point")
    )
    item = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert item.status == STATUS_FAIL
    assert "entry_point" in item.explanation


def test_zero_is_present_not_missing():
    class CountResult(BaseModel):
        findings: int

    result = _result(structured=CountResult(findings=0))
    evaluation = Evaluator.evaluate(
        result, _trace(), _criteria("findings")
    )
    assert (
        evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE).status
        == STATUS_PASS
    )


def test_empty_expected_criteria_is_unknown_not_pass():
    evaluation = Evaluator.evaluate(_result(), _trace())
    item = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert item.status == STATUS_UNKNOWN
    assert "cannot be evaluated" in item.explanation


def test_no_structured_result_is_unknown():
    evaluation = Evaluator.evaluate(
        _result(), _trace(), _criteria("entry_point")
    )
    item = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert item.status == STATUS_UNKNOWN
    assert "no structured result" in item.explanation


def test_structured_result_consumed_as_mapping_without_reparsing():
    """A dict-shaped validated result is used directly — the evaluator
    never treats the answer text as JSON to decode."""
    result = _result(
        answer='{"entry_point": "raw json text, ignored"}',
        structured={"entry_point": "main.py"},
    )
    evaluation = Evaluator.evaluate(
        result, _trace(), _criteria("entry_point")
    )
    item = evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE)
    assert item.status == STATUS_PASS
    assert "1 expected" in item.explanation


# ---------------------------------------------------------------------------
# 6 — overall aggregation
# ---------------------------------------------------------------------------


def test_all_pass_overall_is_pass():
    evaluation = Evaluator.evaluate(
        _result(validation=(_supported(),), structured=GOOD_STRUCTURED),
        _trace(),
        _criteria("entry_point", "configuration_source", "loader"),
    )
    assert evaluation.status == STATUS_PASS
    assert evaluation.summary == "5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL"


def test_single_failure_fails_whole_evaluation():
    evaluation = Evaluator.evaluate(None, _trace())
    assert evaluation.status == STATUS_FAIL
    assert "1 FAIL" in evaluation.summary
    assert "5 PASS" not in evaluation.summary


def test_multiple_failures_still_one_fail_status():
    evaluation = Evaluator.evaluate(
        None,
        _trace(iterations=[_iteration(1)], tool_calls_executed=0),
        _criteria("entry_point"),
    )
    failed = [
        item.criterion
        for item in evaluation.criteria
        if item.status == STATUS_FAIL
    ]
    assert CRITERION_FINAL_RESULT in failed
    assert CRITERION_EVIDENCE_GATHERED in failed
    assert evaluation.status == STATUS_FAIL
    assert evaluation.summary == (
        "1 PASS, 0 PARTIAL, 2 UNKNOWN, 2 FAIL"
    )


def test_partial_and_unknown_states_are_not_overall_pass():
    evaluation = Evaluator.evaluate(
        _result(validation=(_supported(), _validation(VALIDATION_UNKNOWN))),
        _trace(),
    )
    assert evaluation.criterion(CRITERION_CLAIM_TRACEABILITY).status == (
        STATUS_PARTIAL
    )
    assert evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE).status == (
        STATUS_UNKNOWN
    )
    assert evaluation.status == STATUS_PARTIAL
    assert evaluation.summary == "3 PASS, 1 PARTIAL, 1 UNKNOWN, 0 FAIL"


# ---------------------------------------------------------------------------
# 7 — Run A / Run B / Run C (spec §17)
# ---------------------------------------------------------------------------


def run_a_success():
    """Valid answer + evidence + supported claims + objective + bound."""
    return Evaluator.evaluate(
        _result(validation=(_supported(),), structured=GOOD_STRUCTURED),
        _trace(max_tool_calls=8, tool_calls_executed=4),
        _criteria("entry_point", "configuration_source", "loader"),
    )


def run_b_no_evidence():
    """Valid answer, no repository evidence."""
    return Evaluator.evaluate(
        _result(),
        _trace(iterations=[_iteration(1)], tool_calls_executed=0),
        _criteria("entry_point"),
    )


def run_c_limit_reached_objective_incomplete():
    """Evidence + supported claims, tool-call limit reached,
    objective incomplete (structured result present but the loader
    field is empty) → bound PASS, coverage FAIL, overall FAIL."""
    return Evaluator.evaluate(
        _result(
            validation=(_supported(),),
            structured=ConfigSummary(
                entry_point="main.py:load_config()",
                configuration_source="config.yaml",
                loader="",
            ),
        ),
        _trace(max_tool_calls=5, tool_calls_executed=5),
        _criteria("entry_point", "configuration_source", "loader"),
    )


def test_run_a_passes_every_criterion():
    evaluation = run_a_success()
    assert evaluation.status == STATUS_PASS
    for item in evaluation.criteria:
        assert item.status == STATUS_PASS, item.criterion


def test_run_b_fails_on_missing_evidence():
    evaluation = run_b_no_evidence()
    assert evaluation.criterion(CRITERION_EVIDENCE_GATHERED).status == (
        STATUS_FAIL
    )
    assert evaluation.status == STATUS_FAIL


def test_run_c_execution_compliance_is_not_success():
    evaluation = run_c_limit_reached_objective_incomplete()
    assert (
        evaluation.criterion(CRITERION_EXECUTION_BOUND).status == STATUS_PASS
    )
    assert (
        evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE).status
        == STATUS_FAIL
    )
    assert evaluation.status == STATUS_FAIL


# ---------------------------------------------------------------------------
# 8 — determinism
# ---------------------------------------------------------------------------


def test_evaluation_is_deterministic():
    first = run_a_success()
    second = run_a_success()
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_evaluation_does_not_mutate_inputs():
    trace = _trace()
    before = json.dumps(trace, sort_keys=True)
    result = _result()
    Evaluator.evaluate(result, trace, _criteria("entry_point"))
    assert json.dumps(trace, sort_keys=True) == before
    assert result == _result()


# ---------------------------------------------------------------------------
# 9 — repo-agnostic acceptance fixture (spec §20)
# ---------------------------------------------------------------------------


def test_repo_agnostic_config_objective():
    """Generic configuration objective with zero ADOP/MCP/tool
    vocabulary anywhere in the fixture or the verdict."""
    result = _result(
        validation=(_supported("Config is loaded from config.yaml."),),
        structured=ConfigSummary(
            entry_point="app.main:main()",
            configuration_source="config.yaml",
            loader="loader.py",
        ),
    )
    trace = _trace(
        run_id="config-run",
        iterations=[
            _iteration(1, [_tool_result("read_config", call_id="c1")]),
            _iteration(2, [_tool_result("load", call_id="c2")]),
            _iteration(3, [_tool_result("start", call_id="c3")]),
        ],
        tool_calls_executed=3,
    )
    evaluation = Evaluator.evaluate(
        result,
        trace,
        EvaluationCriteria(
            objective="Explain how configuration reaches the application.",
            objective_criteria=(
                ObjectiveCriterion("entry_point"),
                ObjectiveCriterion("configuration_source"),
                ObjectiveCriterion("loader"),
            ),
        ),
    )
    assert evaluation.status == STATUS_PASS
    blob = json.dumps(evaluation.to_dict()).lower()
    for repository_term in (
        "adop",
        "mcp",
        "dag-agent",
        "servers.yaml",
        "sample-agentic",
        "get_architecture_context",
    ):
        assert repository_term not in blob


# ---------------------------------------------------------------------------
# 10 — source guarantee: consumer, not producer
# ---------------------------------------------------------------------------


def test_evaluation_source_performs_no_llm_tool_or_loop_work():
    forbidden_snippets = (
        "import openai",
        "from ..llm",
        "from repo_intelligence.llm",
        ".complete(",
        ".dispatch(",
        "StructuredOutputSpec(",
        "validate_claims(",
        "gather_evidence(",
        "def _run_loop",
        "class Agent",
    )
    for path in sorted(EVAL_PACKAGE.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in source, (
                "{} must not contain {!r}".format(path.name, snippet)
            )


def test_evaluation_source_declares_evaluation_only_imports():
    """Cross-package imports: evidence vocabulary (consumed statuses)
    and the trace-sink default root — nothing else beyond stdlib and
    typing_extensions."""
    import re

    allowed = {
        "__future__",
        "json",
        "dataclasses",
        "typing",
        "typing_extensions",
        "pathlib",
    }
    pattern = re.compile(r"(?:from|import)\s+([.\w]+)")
    for path in sorted(EVAL_PACKAGE.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            match = pattern.match(stripped)
            if not match:
                continue
            module = match.group(1)
            if module.startswith("."):
                continue  # intra-package / in-project relative import
            root = module.split(".")[0]
            assert root in allowed, (
                "{}: unexpected import root {!r}".format(path.name, root)
            )


def test_no_banned_reasoning_vocabulary_in_evaluation_payload():
    evaluation = run_a_success()
    for key in json.loads(evaluation.to_json()):
        assert key not in BANNED_REASONING_FIELDS


# ---------------------------------------------------------------------------
# 11 — input errors vs evaluation outcomes
# ---------------------------------------------------------------------------


def test_malformed_execution_data_raises():
    with pytest.raises(TypeError):
        Evaluator.evaluate(_result(), 42)
    with pytest.raises(ValueError) as excinfo:
        Evaluator.evaluate(_result(), {"summary": {}})
    assert "iterations" in str(excinfo.value)


def test_evaluation_outcomes_are_returned_not_raised():
    """A run with nothing at all still yields a full evaluation."""
    evaluation = Evaluator.evaluate(None, {"run_id": "x", "iterations": []})
    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.status == STATUS_FAIL
    assert len(evaluation.criteria) == len(CRITERIONS)


def test_to_dict_object_accepts_trace_like_objects():
    class TraceLike:
        def to_dict(self):
            return _trace(run_id="obj-run")

    evaluation = Evaluator.evaluate(_result(), TraceLike())
    assert evaluation.run_id == "obj-run"
