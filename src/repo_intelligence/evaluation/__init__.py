"""Deterministic investigation evaluation (M9).

Evaluate a *completed* Agent run against explicit success criteria —
offline, after the run, once:

    AgentRunResult ─┐
                    ├──► Evaluator.evaluate ──► EvaluationResult ──► Sink
    AgentTrace      │         (no LLM, no tools, no second agent)
    or its mapping  ─┘

Public surface:

    ObjectiveCriterion            explicit success condition (a field
                                  the structured result must contain)
    EvaluationCriteria            what success means for this run
                                  (objective + expected fields)
    EvaluationCriterionResult     one criterion's status + explanation
    EvaluationResult              the full evaluation (fixed vocabulary
                                  PASS / FAIL / PARTIAL / UNKNOWN —
                                  never a numeric score)
    CRITERIONS, CRITERION_*       the five fixed criteria, stable ids
    Evaluator.evaluate            pure, deterministic post-run entry
                                  point (consumes, never recomputes,
                                  the M8.8 evidence result and the
                                  M8.9 structured result)
    render_markdown               projection of to_dict() — not a 2nd
                                  source of truth
    EvaluationSink,               persistence (evaluation.{json,md}
    JsonFileEvaluationSink        beside the run's trace)

Boundaries (this package):

* no LLM / provider calls, no tool dispatch, no repository reads,
  no retry, no repair, no second agent loop, no planner, no memory;
* never re-enters or mutates the Agent run it evaluates;
* no generic numeric ``score`` — every criterion states its own
  status plus a short, observable explanation;
* no hidden-reasoning vocabulary anywhere (banned-word assertions
  are enforced by tests);
* repo-agnostic: no ADOP / MCP / tool / path / domain hard-coding.
"""

from .evaluator import Evaluator, render_markdown
from .models import (
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_EVIDENCE_GATHERED,
    CRITERION_EXECUTION_BOUND,
    CRITERION_FINAL_RESULT,
    CRITERION_LABELS,
    CRITERION_OBJECTIVE_COVERAGE,
    CRITERIONS,
    EVALUATION_STATUSES,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationCriterionResult,
    EvaluationResult,
    EvaluationStatus,
    ObjectiveCriterion,
)
from .sink import EvaluationSink, JsonFileEvaluationSink

__all__ = [
    "CRITERION_CLAIM_TRACEABILITY",
    "CRITERION_EVIDENCE_GATHERED",
    "CRITERION_EXECUTION_BOUND",
    "CRITERION_FINAL_RESULT",
    "CRITERION_LABELS",
    "CRITERION_OBJECTIVE_COVERAGE",
    "CRITERIONS",
    "EVALUATION_STATUSES",
    "EvaluationCriteria",
    "EvaluationCriterionResult",
    "EvaluationResult",
    "EvaluationSink",
    "EvaluationStatus",
    "Evaluator",
    "JsonFileEvaluationSink",
    "ObjectiveCriterion",
    "STATUS_FAIL",
    "STATUS_PARTIAL",
    "STATUS_PASS",
    "STATUS_UNKNOWN",
    "render_markdown",
]
