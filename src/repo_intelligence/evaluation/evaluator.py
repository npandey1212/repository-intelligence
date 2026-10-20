"""Deterministic investigation evaluation (M9).

Evaluate a *completed* Agent run against explicit success criteria —
offline, no LLM, no tools, no second agent, no score:

    AgentRunResult ─┐
                    ├──► Evaluator.evaluate(...) ──► EvaluationResult
    execution       │        (post-run, deterministic)
    (AgentTrace or  │
     its mapping)   ─┘

The evaluator is a **consumer**, never a producer:

* it does not investigate (no repository reads, no tool dispatch),
* it does not call a model (no client, no provider imports),
* it does not continue or repair the Agent's run — it evaluates what
  already happened, once, after the run finished.

What each criterion reads (existing artifacts, never recomputed):

    final result        AgentRunResult.answer
    evidence gathered   recorded successful ToolResults in the trace
    claim traceability  the M8.8 ClaimValidation statuses already on
                        the result (consumed, NOT re-run)
    execution bound     the recorded run summary (max / executed /
                        blocked — M8.5 semantics: blocked calls were
                        never executed, so they are not executions)
    objective coverage  the M8.9 validated structured result (consumed
                        as a mapping or model — never re-parsed JSON)

Reproducibility: ``evaluate`` is a pure function of its inputs — same
completed run + same criteria → byte-identical ``to_dict()`` output
(no timestamps, no randomness, no model in the loop).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ..evidence import STATUS_SUPPORTED, STATUS_UNSUPPORTED
from .models import (
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_EVIDENCE_GATHERED,
    CRITERION_EXECUTION_BOUND,
    CRITERION_FINAL_RESULT,
    CRITERION_LABELS,
    CRITERION_OBJECTIVE_COVERAGE,
    STATUS_FAIL,
    STATUS_PARTIAL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationCriterionResult,
    EvaluationResult,
    EvaluationStatus,
)

__all__ = ["Evaluator", "render_markdown"]


# ---------------------------------------------------------------------------
# Execution-data access (same duck-typing rule as the M8.8 validator)
# ---------------------------------------------------------------------------


def _as_mapping(item: Any) -> Mapping[str, Any]:
    """Accept a mapping or an object exposing ``to_dict()``."""
    if isinstance(item, Mapping):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError(
        "execution data must be a mapping or to_dict()-able; got "
        "{!r}".format(type(item).__name__)
    )


def _iterations_of(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    iterations = payload.get("iterations")
    if iterations is None:
        raise ValueError(
            "execution data has no 'iterations' — not an AgentTrace or "
            "trace mapping"
        )
    return [item for item in iterations if isinstance(item, Mapping)]


# ---------------------------------------------------------------------------
# Criterion 1 — a final result exists
# ---------------------------------------------------------------------------


def _final_result(result: Any) -> EvaluationCriterionResult:
    if result is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_FINAL_RESULT,
            status=STATUS_FAIL,
            explanation="run produced no final result",
        )
    answer = getattr(result, "answer", None)
    if not isinstance(answer, str) or not answer.strip():
        return EvaluationCriterionResult(
            criterion=CRITERION_FINAL_RESULT,
            status=STATUS_FAIL,
            explanation="final answer is missing or empty",
        )
    return EvaluationCriterionResult(
        criterion=CRITERION_FINAL_RESULT,
        status=STATUS_PASS,
        explanation="final answer present ({} characters)".format(len(answer)),
    )


# ---------------------------------------------------------------------------
# Criterion 2 — evidence was gathered (successful tool executions)
# ---------------------------------------------------------------------------


def _evidence_gathered(payload: Mapping[str, Any]) -> EvaluationCriterionResult:
    total = 0
    succeeded = 0
    for iteration in _iterations_of(payload):
        for tool_result in iteration.get("tool_results") or []:
            if not isinstance(tool_result, Mapping):
                continue
            total += 1
            if tool_result.get("success"):
                succeeded += 1
    if succeeded:
        return EvaluationCriterionResult(
            criterion=CRITERION_EVIDENCE_GATHERED,
            status=STATUS_PASS,
            explanation="{} successful tool execution(s) recorded".format(
                succeeded
            ),
        )
    if total:
        return EvaluationCriterionResult(
            criterion=CRITERION_EVIDENCE_GATHERED,
            status=STATUS_FAIL,
            explanation=(
                "{} tool execution(s) recorded but none succeeded — a "
                "final answer alone is not evidence".format(total)
            ),
        )
    return EvaluationCriterionResult(
        criterion=CRITERION_EVIDENCE_GATHERED,
        status=STATUS_FAIL,
        explanation="no successful tool execution recorded — no evidence "
        "gathered",
    )


# ---------------------------------------------------------------------------
# Criterion 3 — claim traceability (the M8.8 result, consumed as-is)
# ---------------------------------------------------------------------------


def _claim_traceability(result: Any) -> EvaluationCriterionResult:
    if result is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_CLAIM_TRACEABILITY,
            status=STATUS_UNKNOWN,
            explanation="no run result available to evaluate",
        )
    validations = getattr(result, "evidence_validation", None) or ()
    if not validations:
        return EvaluationCriterionResult(
            criterion=CRITERION_CLAIM_TRACEABILITY,
            status=STATUS_UNKNOWN,
            explanation=(
                "run recorded no claims — traceability not applicable "
                "(structured claims not requested or none produced)"
            ),
        )
    total = len(validations)
    supported = sum(
        1 for item in validations if getattr(item, "status", None) == STATUS_SUPPORTED
    )
    unsupported = sum(
        1
        for item in validations
        if getattr(item, "status", None) == STATUS_UNSUPPORTED
    )
    if supported == total:
        return EvaluationCriterionResult(
            criterion=CRITERION_CLAIM_TRACEABILITY,
            status=STATUS_PASS,
            explanation="{} of {} claim(s) supported by gathered evidence".format(
                supported, total
            ),
        )
    if supported == 0 and unsupported == 0:
        # Claims WERE recorded but not one references evaluable
        # evidence (no refs / insufficient metadata) — the question
        # "can this conclusion be traced back to the run's evidence?"
        # was answerable and the answer is no. M13 §15/§18: that is a
        # FAIL, never an honest-looking UNKNOWN (UNKNOWN stays for
        # runs that recorded no claims at all — nothing to measure).
        return EvaluationCriterionResult(
            criterion=CRITERION_CLAIM_TRACEABILITY,
            status=STATUS_FAIL,
            explanation=(
                "{} claim(s) recorded, but none reference evaluable "
                "evidence — nothing traces back to the run's "
                "evidence".format(total)
            ),
        )
    if supported == 0:
        return EvaluationCriterionResult(
            criterion=CRITERION_CLAIM_TRACEABILITY,
            status=STATUS_FAIL,
            explanation=(
                "0 of {} claim(s) traceable to gathered evidence "
                "({} unsupported)".format(total, unsupported)
            ),
        )
    return EvaluationCriterionResult(
        criterion=CRITERION_CLAIM_TRACEABILITY,
        status=STATUS_PARTIAL,
        explanation=(
            "{} of {} claim(s) supported; the remainder are not fully "
            "traceable".format(supported, total)
        ),
    )


# ---------------------------------------------------------------------------
# Criterion 4 — execution respected the configured bound (M8.5 facts)
# ---------------------------------------------------------------------------


def _execution_bound(payload: Mapping[str, Any]) -> EvaluationCriterionResult:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return EvaluationCriterionResult(
            criterion=CRITERION_EXECUTION_BOUND,
            status=STATUS_UNKNOWN,
            explanation="run summary not recorded — configured bound unknown",
        )
    max_tool_calls = summary.get("max_tool_calls")
    executed = summary.get("tool_calls_executed")
    blocked = summary.get("tool_calls_blocked") or 0
    if max_tool_calls is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_EXECUTION_BOUND,
            status=STATUS_UNKNOWN,
            explanation="no configured tool-call budget recorded",
        )
    if executed is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_EXECUTION_BOUND,
            status=STATUS_UNKNOWN,
            explanation="execution counts not recorded in the run summary",
        )
    if executed > max_tool_calls:
        return EvaluationCriterionResult(
            criterion=CRITERION_EXECUTION_BOUND,
            status=STATUS_FAIL,
            explanation=(
                "executed {} tool call(s) exceeds the configured maximum "
                "{}".format(executed, max_tool_calls)
            ),
        )
    return EvaluationCriterionResult(
        criterion=CRITERION_EXECUTION_BOUND,
        status=STATUS_PASS,
        explanation=(
            "executed {} of {} permitted tool call(s); {} blocked call(s) "
            "not counted as executions".format(executed, max_tool_calls, blocked)
        ),
    )


# ---------------------------------------------------------------------------
# Criterion 5 — objective coverage (explicit expected criteria)
# ---------------------------------------------------------------------------


def _substantive(value: Any) -> bool:
    """A present field: not None, not blank text, not an empty container.

    Numbers count even when zero — only "absent / empty" means the
    expected criterion was not addressed.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _structured_as_mapping(structured: Any) -> Optional[Mapping[str, Any]]:
    """The validated M8.9 result as a mapping — never re-parsed."""
    if isinstance(structured, Mapping):
        return structured
    as_dict = getattr(structured, "dict", None)  # Pydantic v1 model
    if callable(as_dict):
        data = as_dict()
        if isinstance(data, Mapping):
            return data
    return None


def _objective_coverage(
    result: Any, criteria: EvaluationCriteria
) -> EvaluationCriterionResult:
    expected = criteria.objective_criteria
    if not expected:
        return EvaluationCriterionResult(
            criterion=CRITERION_OBJECTIVE_COVERAGE,
            status=STATUS_UNKNOWN,
            explanation=(
                "no expected objective criteria defined — objective "
                "coverage cannot be evaluated"
            ),
        )
    if result is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_OBJECTIVE_COVERAGE,
            status=STATUS_UNKNOWN,
            explanation="no run result available to evaluate",
        )
    structured = getattr(result, "structured_output", None)
    if structured is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_OBJECTIVE_COVERAGE,
            status=STATUS_UNKNOWN,
            explanation=(
                "run produced no structured result — expected fields "
                "cannot be checked"
            ),
        )
    data = _structured_as_mapping(structured)
    if data is None:
        return EvaluationCriterionResult(
            criterion=CRITERION_OBJECTIVE_COVERAGE,
            status=STATUS_UNKNOWN,
            explanation=(
                "structured result is not a mapping or model — expected "
                "fields cannot be checked"
            ),
        )
    missing = [
        item.key for item in expected if not _substantive(data.get(item.key))
    ]
    if missing:
        return EvaluationCriterionResult(
            criterion=CRITERION_OBJECTIVE_COVERAGE,
            status=STATUS_FAIL,
            explanation="missing or empty expected field(s): {}".format(
                ", ".join(missing)
            ),
        )
    return EvaluationCriterionResult(
        criterion=CRITERION_OBJECTIVE_COVERAGE,
        status=STATUS_PASS,
        explanation="all {} expected field(s) present in the structured "
        "result".format(len(expected)),
    )


# ---------------------------------------------------------------------------
# Overall status + summary (deterministic aggregation)
# ---------------------------------------------------------------------------


def _overall(
    results: Sequence[EvaluationCriterionResult],
) -> EvaluationStatus:
    """Any FAIL fails the run; otherwise a non-clean state is PARTIAL;
    PASS requires every criterion to PASS (never inherited from an
    UNKNOWN — success must be established, not assumed)."""
    counts = EvaluationResult.counts(results)
    if counts[STATUS_FAIL]:
        return STATUS_FAIL
    if counts[STATUS_PARTIAL] or counts[STATUS_UNKNOWN]:
        return STATUS_PARTIAL
    return STATUS_PASS


def _summary(results: Sequence[EvaluationCriterionResult]) -> str:
    counts = EvaluationResult.counts(results)
    return "{} PASS, {} PARTIAL, {} UNKNOWN, {} FAIL".format(
        counts[STATUS_PASS],
        counts[STATUS_PARTIAL],
        counts[STATUS_UNKNOWN],
        counts[STATUS_FAIL],
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Deterministic post-run investigation evaluator.

    ``Evaluator.evaluate(result, execution, criteria)`` is a static,
    stateless entry point: give it the completed run's result (or
    ``None`` when the run produced none) plus its execution record
    (an ``AgentTrace`` or its JSON mapping) and the explicit success
    criteria. It never mutates its inputs and never performs I/O
    beyond returning the result — persisting an evaluation is a
    separable concern (``evaluation.sink``).
    """

    @staticmethod
    def evaluate(
        result: Any,
        execution: Any,
        criteria: Optional[EvaluationCriteria] = None,
    ) -> EvaluationResult:
        """Evaluate one completed run → ``EvaluationResult``.

        *result* is an ``AgentRunResult`` (or ``None`` for a run that
        produced no final result). *execution* is the run's
        ``AgentTrace`` or its mapping. *criteria* supplies the
        explicit success definition (default: none defined →
        objective coverage is UNKNOWN, never a silent PASS).

        Raises ``TypeError``/``ValueError`` only for malformed
        execution data (not an AgentTrace or trace mapping); every
        *evaluation* outcome is returned as data, never raised.
        """
        if criteria is None:
            criteria = EvaluationCriteria()
        payload = _as_mapping(execution)
        results = [
            _final_result(result),
            _evidence_gathered(payload),
            _claim_traceability(result),
            _execution_bound(payload),
            _objective_coverage(result, criteria),
        ]
        run_id = payload.get("run_id") or ""
        return EvaluationResult(
            status=_overall(results),
            criteria=tuple(results),
            summary=_summary(results),
            run_id=str(run_id),
        )


# ---------------------------------------------------------------------------
# Markdown projection (rendered from the result — never a 2nd source)
# ---------------------------------------------------------------------------


def render_markdown(evaluation: EvaluationResult) -> str:
    """Human-readable projection of ``EvaluationResult.to_dict()``."""
    lines = [
        "# Investigation Evaluation",
        "",
        "* Run: `{}`".format(evaluation.run_id or "-"),
        "",
    ]
    for item in evaluation.criteria:
        label = CRITERION_LABELS.get(item.criterion, item.criterion)
        lines.append("**{}:** `{}`".format(label, item.status))
        if item.status != STATUS_PASS and item.explanation:
            lines.append("  * {}".format(item.explanation))
    lines.extend(
        [
            "",
            "**Overall:** `{}`".format(evaluation.status),
            "",
            evaluation.summary,
            "",
        ]
    )
    return "\n".join(lines)
