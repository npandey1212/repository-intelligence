"""M9 — evaluation model tests (deterministic data contract).

Proves the smallest evaluation model is exactly what the spec asks
for: explicit status vocabulary, fixed five-criterion layout,
round-trippable serialization, no numeric score, no hidden-reasoning
vocabulary:

 1  valid evaluation result (full fixed shape)
 2  invalid overall/criterion state → ValueError
 3  serialization / deserialization round-trip
 4  criterion lookup + fixed order
 5  counts() in fixed vocabulary order
 6  EvaluationCriteria / ObjectiveCriterion input validation
 7  no score anywhere (no numeric field in to_dict())
 8  no hidden-reasoning fields (banned vocabulary)
 9  determinism: same inputs → byte-identical to_json()

Evaluator behavior lives in ``test_evaluator.py``; the sink in
``test_evaluation_sink.py``.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.evaluation import (
    CRITERIONS,
    EVALUATION_STATUSES,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationCriterionResult,
    EvaluationResult,
    ObjectiveCriterion,
)

EXTRA_BANNED_FIELDS = ("reasoning", "reasoning_trace")

ALL_PASS = (
    EvaluationCriterionResult("final_result", STATUS_PASS, "answer present"),
    EvaluationCriterionResult(
        "evidence_gathered", STATUS_PASS, "2 successful tool execution(s)"
    ),
    EvaluationCriterionResult(
        "claim_traceability", STATUS_PASS, "1 of 1 claim(s) supported"
    ),
    EvaluationCriterionResult(
        "execution_bound", STATUS_PASS, "executed 2 of 8 permitted"
    ),
    EvaluationCriterionResult(
        "objective_coverage", STATUS_PASS, "all 3 expected field(s) present"
    ),
)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


# ---------------------------------------------------------------------------
# 1 — valid evaluation result
# ---------------------------------------------------------------------------


def test_valid_evaluation_result_full_shape():
    result = EvaluationResult(
        status=STATUS_PASS,
        criteria=ALL_PASS,
        summary="5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL",
        run_id="run-1",
    )
    payload = result.to_dict()
    assert payload["status"] == STATUS_PASS
    assert [item["criterion"] for item in payload["criteria"]] == list(CRITERIONS)
    assert payload["summary"] == "5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL"
    assert payload["run_id"] == "run-1"
    # criteria carry their own explanation — never a bare status
    for item in payload["criteria"]:
        assert item["explanation"]


# ---------------------------------------------------------------------------
# 2 — invalid states → ValueError
# ---------------------------------------------------------------------------


def test_invalid_overall_status_raises():
    with pytest.raises(ValueError) as excinfo:
        EvaluationResult(status="OK")
    assert "overall status" in str(excinfo.value)


def test_invalid_criterion_status_raises():
    with pytest.raises(ValueError) as excinfo:
        EvaluationCriterionResult(
            criterion="final_result", status="MAYBE"
        )
    assert "status must be one of" in str(excinfo.value)
    # the vocabulary is echoed, so failures teach the contract
    for status in EVALUATION_STATUSES:
        assert status in str(excinfo.value)


def test_empty_criterion_id_raises():
    with pytest.raises(ValueError):
        EvaluationCriterionResult(criterion="  ", status=STATUS_PASS)


def test_non_criterion_item_in_result_raises():
    with pytest.raises(ValueError) as excinfo:
        EvaluationResult(status=STATUS_FAIL, criteria=("final_result",))
    assert "EvaluationCriterionResult" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3 — serialization / deserialization round-trip
# ---------------------------------------------------------------------------


def test_round_trip_to_dict_and_from_dict():
    result = EvaluationResult(
        status=STATUS_PARTIAL,
        criteria=(
            EvaluationCriterionResult("final_result", STATUS_PASS, "ok"),
            EvaluationCriterionResult(
                "objective_coverage", STATUS_UNKNOWN, "no criteria"
            ),
        ),
        summary="1 PASS, 0 PARTIAL, 1 UNKNOWN, 0 FAIL",
        run_id="run-2",
    )
    restored = EvaluationResult.from_dict(result.to_dict())
    assert restored == result
    assert restored.to_json() == result.to_json()


def test_from_dict_defaults_for_missing_fields():
    restored = EvaluationResult.from_dict({"status": STATUS_FAIL})
    assert restored.criteria == ()
    assert restored.summary == ""
    assert restored.run_id == ""


# ---------------------------------------------------------------------------
# 4 — criterion lookup + fixed order
# ---------------------------------------------------------------------------


def test_criterion_lookup_and_missing_raises():
    result = EvaluationResult(status=STATUS_PASS, criteria=ALL_PASS)
    assert result.criterion("execution_bound").status == STATUS_PASS
    with pytest.raises(KeyError):
        result.criterion("quality_score")


def test_result_criteria_follow_fixed_order():
    result = EvaluationResult(status=STATUS_PASS, criteria=ALL_PASS)
    assert [item.criterion for item in result.criteria] == list(CRITERIONS)
    assert list(CRITERIONS) == [
        "final_result",
        "evidence_gathered",
        "claim_traceability",
        "execution_bound",
        "objective_coverage",
    ]


# ---------------------------------------------------------------------------
# 5 — counts in fixed vocabulary order
# ---------------------------------------------------------------------------


def test_counts_by_status():
    criteria = (
        EvaluationCriterionResult("a", STATUS_PASS),
        EvaluationCriterionResult("b", STATUS_PASS),
        EvaluationCriterionResult("c", STATUS_FAIL),
        EvaluationCriterionResult("d", STATUS_PARTIAL),
        EvaluationCriterionResult("e", STATUS_UNKNOWN),
    )
    counts = EvaluationResult.counts(criteria)
    assert counts[STATUS_PASS] == 2
    assert counts[STATUS_FAIL] == 1
    assert counts[STATUS_PARTIAL] == 1
    assert counts[STATUS_UNKNOWN] == 1


def test_counts_empty_is_all_zero():
    counts = EvaluationResult.counts(())
    assert counts == {
        STATUS_PASS: 0,
        STATUS_PARTIAL: 0,
        STATUS_UNKNOWN: 0,
        STATUS_FAIL: 0,
    }


# ---------------------------------------------------------------------------
# 6 — criteria input validation (what success means)
# ---------------------------------------------------------------------------


def test_objective_criterion_requires_non_empty_key():
    with pytest.raises(ValueError):
        ObjectiveCriterion(key="")
    with pytest.raises(ValueError):
        ObjectiveCriterion(key="   ")


def test_evaluation_criteria_round_trip():
    criteria = EvaluationCriteria(
        objective="Explain how configuration is loaded.",
        objective_criteria=(
            ObjectiveCriterion("entry_point", "where loading starts"),
            ObjectiveCriterion("configuration_source"),
        ),
    )
    restored = EvaluationCriteria.from_dict(criteria.to_dict())
    assert restored == criteria
    assert restored.objective_criteria[0].description == (
        "where loading starts"
    )


def test_empty_criteria_defaults():
    criteria = EvaluationCriteria()
    assert criteria.objective == ""
    assert criteria.objective_criteria == ()
    assert EvaluationCriteria.from_dict({}) == criteria


# ---------------------------------------------------------------------------
# 7 — no score anywhere
# ---------------------------------------------------------------------------


def test_no_numeric_score_field_in_payload():
    result = EvaluationResult(
        status=STATUS_PASS,
        criteria=ALL_PASS,
        summary="5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL",
        run_id="run-3",
    )
    payload = result.to_dict()
    assert set(payload) == {"status", "criteria", "summary", "run_id"}
    for item in payload["criteria"]:
        assert set(item) == {"criterion", "status", "explanation"}
    forbidden = {"score", "quality", "confidence", "rating", "points"}
    assert not (set(payload) & forbidden)
    # and no number is ever produced as the evaluation verdict
    assert isinstance(payload["status"], str)


# ---------------------------------------------------------------------------
# 8 — no hidden-reasoning vocabulary anywhere
# ---------------------------------------------------------------------------


def test_no_hidden_reasoning_fields_anywhere():
    result = EvaluationResult(
        status=STATUS_PASS,
        criteria=ALL_PASS,
        summary="5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL",
        run_id="run-4",
    )
    for key in _walk_keys(json.loads(result.to_json())):
        assert key not in BANNED_REASONING_FIELDS, key
        assert key not in EXTRA_BANNED_FIELDS, key
    # dataclass fields, not just serialized keys
    for field_name in EvaluationResult.__dataclass_fields__:
        assert field_name not in BANNED_REASONING_FIELDS
    for field_name in EvaluationCriterionResult.__dataclass_fields__:
        assert field_name not in BANNED_REASONING_FIELDS


# ---------------------------------------------------------------------------
# 9 — determinism
# ---------------------------------------------------------------------------


def test_same_inputs_serialize_identically():
    def build():
        return EvaluationResult(
            status=STATUS_PARTIAL,
            criteria=(
                EvaluationCriterionResult("final_result", STATUS_PASS, "ok"),
                EvaluationCriterionResult(
                    "evidence_gathered", STATUS_FAIL, "no evidence"
                ),
            ),
            summary="1 PASS, 0 PARTIAL, 0 UNKNOWN, 1 FAIL",
            run_id="run-5",
        )

    first, second = build(), build()
    assert first.to_json() == second.to_json()
    assert first.to_dict() == second.to_dict()
