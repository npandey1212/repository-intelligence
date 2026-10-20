"""M9 investigation-evaluation data model (deterministic, offline).

The smallest useful representation for evaluating a *completed*
Agent run against explicit success criteria:

    EvaluationResult
    ├── status            overall PASS / FAIL / PARTIAL / UNKNOWN
    ├── criteria[]        EvaluationCriterionResult, fixed order
    │   ├── criterion     final_result · evidence_gathered ·
    │   │                 claim_traceability · execution_bound ·
    │   │                 objective_coverage
    │   ├── status        PASS / FAIL / PARTIAL / UNKNOWN
    │   └── explanation   observable fact, never a hidden rationale
    ├── summary           deterministic counts of the criterion states
    └── run_id            which run this evaluation belongs to

Evaluation criteria — what success means for ONE run:

    EvaluationCriteria
    ├── objective            the question the run set out to answer
    └── objective_criteria[] expected result fields (ObjectiveCriterion
                             { key, description }) — the explicit
                             definition of objective coverage

Design boundaries (M9):

* **No score.** There is no number anywhere in this model: a single
  score hides *why* a run succeeded or failed. Each criterion states
  its own status plus a short, observable explanation.
* **Status vocabulary** mirrors the M8.8 evidence model (a small
  Literal + frozenset, not free-form strings): PASS/FAIL are decided,
  PARTIAL is evaluated-but-incomplete, UNKNOWN means the criterion
  *cannot* be evaluated from the given inputs (e.g. no claims were
  produced, no expected objective criteria were defined) — UNKNOWN is
  never silently upgraded to PASS.
* **Deterministic and reproducible**: no timestamps, no randomness,
  no model output — the same inputs serialize to the same payload.
* **Consumption, not reimplementation**: claim traceability reads the
  M8.8 ``ClaimValidation`` statuses it is given; structured coverage
  reads the M8.9 validated result; execution bounds read the recorded
  summary. This model stores evaluation *results*, never evidence
  matching or answer text parsing.

No hidden reasoning fields exist anywhere in this model (the same
banned vocabulary the execution trace rejects is asserted by tests).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Tuple

from typing_extensions import Literal

__all__ = [
    "CRITERION_CLAIM_TRACEABILITY",
    "CRITERION_EVIDENCE_GATHERED",
    "CRITERION_EXECUTION_BOUND",
    "CRITERION_FINAL_RESULT",
    "CRITERION_LABELS",
    "CRITERION_OBJECTIVE_COVERAGE",
    "CRITERIONS",
    "EVALUATION_STATUSES",
    "STATUS_FAIL",
    "STATUS_PARTIAL",
    "STATUS_PASS",
    "STATUS_UNKNOWN",
    "EvaluationCriteria",
    "EvaluationCriterionResult",
    "EvaluationResult",
    "EvaluationStatus",
    "ObjectiveCriterion",
]

EvaluationStatus = Literal["PASS", "FAIL", "PARTIAL", "UNKNOWN"]

# Controlled vocabulary — small and stable (the smallest set that
# keeps "failed" distinct from "evaluated but incomplete" distinct
# from "cannot be evaluated from these inputs").
STATUS_PASS: EvaluationStatus = "PASS"
STATUS_FAIL: EvaluationStatus = "FAIL"
STATUS_PARTIAL: EvaluationStatus = "PARTIAL"
STATUS_UNKNOWN: EvaluationStatus = "UNKNOWN"
EVALUATION_STATUSES = frozenset(
    {STATUS_PASS, STATUS_FAIL, STATUS_PARTIAL, STATUS_UNKNOWN}
)

# The five criteria, in fixed order (deterministic result layout).
CRITERION_FINAL_RESULT = "final_result"
CRITERION_EVIDENCE_GATHERED = "evidence_gathered"
CRITERION_CLAIM_TRACEABILITY = "claim_traceability"
CRITERION_EXECUTION_BOUND = "execution_bound"
CRITERION_OBJECTIVE_COVERAGE = "objective_coverage"
CRITERIONS = (
    CRITERION_FINAL_RESULT,
    CRITERION_EVIDENCE_GATHERED,
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_EXECUTION_BOUND,
    CRITERION_OBJECTIVE_COVERAGE,
)

# Display labels for the Markdown projection (ids stay machine-stable).
CRITERION_LABELS: Dict[str, str] = {
    CRITERION_FINAL_RESULT: "Final result",
    CRITERION_EVIDENCE_GATHERED: "Evidence gathered",
    CRITERION_CLAIM_TRACEABILITY: "Claim traceability",
    CRITERION_EXECUTION_BOUND: "Execution bound",
    CRITERION_OBJECTIVE_COVERAGE: "Objective coverage",
}


def _validate_status(status: str, what: str) -> None:
    if status not in EVALUATION_STATUSES:
        raise ValueError(
            "{} status must be one of {}, got {!r}".format(
                what, ", ".join(sorted(EVALUATION_STATUSES)), status
            )
        )


@dataclass(frozen=True)
class ObjectiveCriterion:
    """One explicit success condition for the run's objective.

    ``key`` is a required field of the run's structured result
    (M8.9): objective coverage checks that every expected key is
    present and substantive. Evaluation without an explicit definition
    of success is not attempted — an empty list means "not defined",
    never "everything passed".
    """

    key: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError(
                "objective criterion requires a non-empty key, "
                "got {!r}".format(self.key)
            )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"key": self.key}
        if self.description:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectiveCriterion":
        return cls(
            key=data.get("key", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class EvaluationCriteria:
    """What success means for one evaluation — supplied, never guessed.

    ``objective`` records the question the run set out to answer
    (context for the report). ``objective_criteria`` defines objective
    coverage: the fields the final structured result must contain.
    Criteria are pure input: identical criteria + identical run →
    identical evaluation.
    """

    objective: str = ""
    objective_criteria: Tuple[ObjectiveCriterion, ...] = field(
        default_factory=tuple
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "objective_criteria": [
                item.to_dict() for item in self.objective_criteria
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationCriteria":
        return cls(
            objective=data.get("objective", ""),
            objective_criteria=tuple(
                ObjectiveCriterion.from_dict(item)
                for item in (data.get("objective_criteria") or [])
            ),
        )


@dataclass(frozen=True)
class EvaluationCriterionResult:
    """The evaluation outcome for ONE criterion."""

    criterion: str
    status: EvaluationStatus
    explanation: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, str) or not self.criterion.strip():
            raise ValueError(
                "criterion result requires a non-empty criterion id, "
                "got {!r}".format(self.criterion)
            )
        _validate_status(self.status, "criterion {!r}".format(self.criterion))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "status": self.status,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "EvaluationCriterionResult":
        return cls(
            criterion=data.get("criterion", ""),
            status=data.get("status", ""),
            explanation=data.get("explanation", ""),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """The deterministic evaluation of one completed Agent run.

    ``criteria`` always holds exactly one result per known criterion
    (``CRITERIONS`` order), so two evaluations of equivalent inputs
    serialize to identical payloads — the reproducibility property of
    M9 (same run + same criteria → same evaluation).
    """

    status: EvaluationStatus
    criteria: Tuple[EvaluationCriterionResult, ...] = ()
    summary: str = ""
    run_id: str = ""

    def __post_init__(self) -> None:
        _validate_status(self.status, "overall")
        for item in self.criteria:
            if not isinstance(item, EvaluationCriterionResult):
                raise ValueError(
                    "criteria must be EvaluationCriterionResult items, "
                    "got {!r}".format(type(item).__name__)
                )
            _validate_status(item.status, "criterion {!r}".format(item.criterion))

    def criterion(self, criterion_id: str) -> EvaluationCriterionResult:
        """The result for *criterion_id* (raises KeyError if absent)."""
        for item in self.criteria:
            if item.criterion == criterion_id:
                return item
        raise KeyError(criterion_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "criteria": [item.to_dict() for item in self.criteria],
            "summary": self.summary,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        return cls(
            status=data.get("status", ""),
            criteria=tuple(
                EvaluationCriterionResult.from_dict(item)
                for item in (data.get("criteria") or [])
            ),
            summary=data.get("summary", ""),
            run_id=data.get("run_id", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        """Canonical JSON form (mirrors ``AgentTrace.to_json``)."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def counts(
        criteria: Iterable[EvaluationCriterionResult],
    ) -> Dict[str, int]:
        """State counts in fixed vocabulary order (summary input)."""
        counts = {
            STATUS_PASS: 0,
            STATUS_PARTIAL: 0,
            STATUS_UNKNOWN: 0,
            STATUS_FAIL: 0,
        }
        for item in criteria:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts
