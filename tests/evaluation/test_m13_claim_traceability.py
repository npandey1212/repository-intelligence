"""M13 — claim traceability is a measurable M9 criterion (spec §15/§18).

The criterion itself is the established ``CRITERION_CLAIM_TRACEABILITY``
(consumed from ``AgentRunResult.evidence_validation``, never
recomputed — no LLM, no judge). M13 tightens what the numbers mean:

 * PASS when every claim's evidence was valid (inspected) —
 * FAIL when evidence is **invalid** (UNSUPPORTED, recorded with
   ``CLAIM_EVIDENCE_INVALID``) or **missing** (claims recorded but
   none reference evaluable evidence — previously UNKNOWN, now an
   answerable "no"),
 * UNKNOWN only when the run recorded no claims at all (nothing to
   measure), and
 * a claim-bearing typed run can now reach overall PASS — the M13
   run mode removes the baseline's honest-but-weak UNKNOWN ceiling.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from repo_intelligence.agent import AgentRunResult
from repo_intelligence.evaluation import (
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_OBJECTIVE_COVERAGE,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    Evaluator,
    ObjectiveCriterion,
)
from repo_intelligence.evidence import (
    CLAIM_EVIDENCE_INVALID,
    STATUS_SUPPORTED,
    STATUS_UNKNOWN as VALIDATION_UNKNOWN,
    STATUS_UNSUPPORTED,
    Claim,
    ClaimValidation,
    EvidenceRef,
)

SOURCE = {
    "repository": "sample-repo-alpha",
    "commit_sha": "b" * 40,
}


# ---------------------------------------------------------------------------
# Fixtures (same shapes the frozen evaluator tests use)
# ---------------------------------------------------------------------------


def _trace():
    return {
        "run_id": "run-m13",
        "iterations": [
            {
                "iteration": 1,
                "response_type": "TOOL_CALLS",
                "tool_results": [
                    {
                        "tool": "read_file",
                        "tool_call_id": "c1",
                        "success": True,
                        "data": {"path": "file.txt"},
                        "source": dict(SOURCE),
                        "error": None,
                    }
                ],
            }
        ],
        "summary": {
            "total_iterations": 1,
            "termination_reason": "FINAL_ANSWER",
            "max_tool_calls": 8,
            "tool_calls_executed": 1,
            "tool_calls_blocked": 0,
        },
    }


def _result(validation=(), structured=None):
    return AgentRunResult(
        answer="The repository connects agents through declared tools.",
        claims=tuple(entry.claim for entry in validation),
        evidence_validation=tuple(validation),
        structured_output=structured,
    )


def _supported(statement="ontology-agent declares an MCP tool."):
    return ClaimValidation(
        claim=Claim(
            statement=statement,
            evidence_refs=(EvidenceRef(path=".claude/agents/x.md"),),
        ),
        status=STATUS_SUPPORTED,
        matched_evidence=(),
        reason="matched gathered evidence: read_file x.md",
    )


def _validation(status, code=None, statement="s"):
    return ClaimValidation(
        claim=Claim(statement=statement),
        status=status,
        matched_evidence=(),
        reason="deterministic reason",
        code=code,
    )


class EvidenceBackedFinding(BaseModel):
    """The M13 contract shape the objective-coverage criterion reads."""

    finding: str
    evidence_paths: List[str]
    claims: List[dict]


def _criteria():
    return EvaluationCriteria(
        objective="Explain how agents, tools, and MCP servers connect.",
        objective_criteria=(
            ObjectiveCriterion("finding", "the connection explained"),
            ObjectiveCriterion("evidence_paths", "inspected files"),
            ObjectiveCriterion("claims", "grounded claims"),
        ),
    )


# ---------------------------------------------------------------------------
# 1 — evidence valid / invalid / missing (§18 evaluator)
# ---------------------------------------------------------------------------


def test_claim_traceability_passes_when_evidence_is_valid():
    evaluation = Evaluator.evaluate(
        _result(validation=(_supported(), _supported("Two."))), _trace()
    )
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_PASS
    assert "2 of 2" in item.explanation


def test_claim_traceability_fails_when_evidence_is_invalid():
    evaluation = Evaluator.evaluate(
        _result(
            validation=(
                _validation(
                    STATUS_UNSUPPORTED, code=CLAIM_EVIDENCE_INVALID
                ),
            )
        ),
        _trace(),
    )
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_FAIL
    assert "0 of 1" in item.explanation
    assert evaluation.status == STATUS_FAIL


def test_claim_traceability_fails_when_evidence_is_missing():
    """Claims were recorded, but not one references evaluable
    evidence — an answerable question with a negative answer (M13:
    this used to hide behind UNKNOWN)."""
    evaluation = Evaluator.evaluate(
        _result(
            validation=(
                _validation(VALIDATION_UNKNOWN),
                _validation(VALIDATION_UNKNOWN),
            )
        ),
        _trace(),
    )
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_FAIL
    assert "none reference evaluable evidence" in item.explanation
    assert evaluation.status == STATUS_FAIL


def test_claim_traceability_partial_when_only_some_claims_trace():
    evaluation = Evaluator.evaluate(
        _result(
            validation=(
                _supported(),
                _validation(
                    STATUS_UNSUPPORTED, code=CLAIM_EVIDENCE_INVALID
                ),
            )
        ),
        _trace(),
    )
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_PARTIAL
    assert "1 of 2" in item.explanation
    # one FAILed criterion would sink the run; PARTIAL keeps it visible
    assert evaluation.status == STATUS_PARTIAL


def test_no_claims_still_reports_unknown_never_pass():
    evaluation = Evaluator.evaluate(_result(), _trace())
    item = evaluation.criterion(CRITERION_CLAIM_TRACEABILITY)
    assert item.status == STATUS_UNKNOWN
    assert "no claims" in item.explanation


# ---------------------------------------------------------------------------
# 2 — the M13 run mode can reach overall PASS (baseline ceiling gone)
# ---------------------------------------------------------------------------


def test_claim_bearing_typed_run_reaches_overall_pass():
    structured = EvidenceBackedFinding(
        finding="Agents connect to MCP servers only through declared "
        "tool references; access is per-agent, not universal.",
        evidence_paths=[
            ".claude/agents/ontology-agent.md",
            "tool-registry/servers.yaml",
        ],
        claims=[{"statement": "ontology-agent declares an MCP tool."}],
    )
    evaluation = Evaluator.evaluate(
        _result(validation=(_supported(),), structured=structured),
        _trace(),
        _criteria(),
    )

    assert (
        evaluation.criterion(CRITERION_CLAIM_TRACEABILITY).status
        == STATUS_PASS
    )
    assert (
        evaluation.criterion(CRITERION_OBJECTIVE_COVERAGE).status
        == STATUS_PASS
    )
    # every criterion established — the baseline could only reach
    # PARTIAL (claim_traceability UNKNOWN in M8.9 mode)
    assert evaluation.status == STATUS_PASS


def test_evaluation_consumes_recorded_statuses_without_revalidating():
    result = _result(validation=(_supported(),))
    before = result.evidence_validation
    evaluation = Evaluator.evaluate(result, _trace())

    assert result.evidence_validation == before
    assert (
        len(evaluation.criterion(CRITERION_CLAIM_TRACEABILITY).explanation)
        > 0
    )
