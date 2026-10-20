"""M9 — investigation evaluation demonstration (deterministic, offline).

Shows the M9 lesson end to end — no LLM, no network, no tools, no
second agent loop. An Agent's completed run is evaluated SEPARATELY
from execution:

    Objective
      ↓
    Agent (investigates)                ← M8.x, frozen
      ↓
    Trace + claims/structured result
      ↓
    M8.8 evidence validation            ← claim-to-evidence question
      ↓
    M9 Evaluator.evaluate               ← "was the investigation
      ↓                                     successful?"
    EvaluationResult (PASS/FAIL/PARTIAL/UNKNOWN per criterion)

Demonstrations (spec §17, §28):

    Run A   complete run  → every criterion PASS, overall PASS
    Run C   at the limit with an incomplete objective → execution
            bound PASS while objective coverage FAIL → overall FAIL
            ("an Agent can execute correctly and still fail the
            investigation objective")
    Run B   final answer, zero repository evidence → evidence FAIL
    Fixture the captured M8.7 trace evaluated with honest UNKNOWNs
            (criteria that cannot be evaluated are never PASS)

Checks (all required):

    1  Run A: all five criteria PASS, overall PASS
    2  Run A: deterministic counts summary
    3  Run C: execution bound PASS at the configured limit
    4  Run C: objective coverage FAIL names the missing field
    5  Run C: overall FAIL (compliance is not success)
    6  Run B: evidence gathered FAIL with a final answer present
    7  determinism: same inputs → byte-identical evaluation JSON
    8  real M8.7 trace fixture: evidence + bound PASS, unevaluable
       criteria UNKNOWN (never silently PASS), overall PARTIAL
    9  blocked tool calls are never counted as executions
   10  no numeric score anywhere in any payload
   11  no hidden reasoning vocabulary in payloads or markdown
   12  repository-agnostic verdict (no ADOP/MCP/tool terminology)
   13  artifact sink: evaluation.{json,md} written beside the trace,
       JSON round-trips, markdown is a projection of that JSON

Usage (from the project root; fully offline):

    python agentrun_evaluation_test.py

Exit codes: 0 = all checks passed, 6 = failure.

See also: agentrun_evidence_test.py (M8.8 claim traceability),
agentrun_react_test.py (M8.6 mechanics). This script evaluates a run
AFTER it finished — it never calls a model, never dispatches a tool,
and never re-enters the Agent loop.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydantic import BaseModel  # noqa: E402

from repo_intelligence.agent import AgentRunResult  # noqa: E402
from repo_intelligence.agent.trace import (  # noqa: E402
    BANNED_REASONING_FIELDS,
)
from repo_intelligence.evaluation import (  # noqa: E402
    CRITERION_CLAIM_TRACEABILITY,
    CRITERION_EVIDENCE_GATHERED,
    CRITERION_EXECUTION_BOUND,
    CRITERION_FINAL_RESULT,
    CRITERION_LABELS,
    CRITERION_OBJECTIVE_COVERAGE,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_UNKNOWN,
    EvaluationCriteria,
    EvaluationResult,
    Evaluator,
    JsonFileEvaluationSink,
    ObjectiveCriterion,
    render_markdown,
)
from repo_intelligence.evidence import (  # noqa: E402
    STATUS_SUPPORTED,
    Claim,
    ClaimValidation,
    EvidenceRef,
)

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "m87_react_trace.json"
RULE = "-" * 72
SCORE_KEYS = ("score", "quality", "confidence", "rating", "points")


# ---------------------------------------------------------------------------
# Generic repo-agnostic structured contract (spec §20 vocabulary)
# ---------------------------------------------------------------------------


class ConfigSummary(BaseModel):
    entry_point: str
    configuration_source: str
    loader: str


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


# ---------------------------------------------------------------------------
# Deterministic run artifacts (the shapes a completed run produces)
# ---------------------------------------------------------------------------


def _supported(statement: str, path: str) -> ClaimValidation:
    return ClaimValidation(
        claim=Claim(
            statement=statement,
            evidence_refs=(EvidenceRef(path=path, line_start=1, line_end=20),),
        ),
        status=STATUS_SUPPORTED,
        matched_evidence=(),
        reason="matched gathered evidence: read_file {}".format(path),
    )


def _trace(
    run_id: str,
    successes: int,
    max_tool_calls: int = 8,
    executed: Optional[int] = None,
    blocked: int = 0,
) -> Dict[str, Any]:
    """A completed run's execution record (trace-shaped mapping)."""
    if executed is None:
        executed = successes
    results = [
        {
            "tool": "read_file",
            "tool_call_id": "c{}".format(number),
            "success": True,
            "data": {"path": "config.yaml"},
        }
        for number in range(1, successes + 1)
    ]
    return {
        "run_id": run_id,
        "iterations": [
            {"iteration": 1, "tool_results": results},
            {"iteration": 2, "tool_results": []},
        ],
        "summary": {
            "termination_reason": "FINAL_ANSWER",
            "max_tool_calls": max_tool_calls,
            "tool_calls_executed": executed,
            "tool_calls_blocked": blocked,
        },
    }


def _criteria() -> EvaluationCriteria:
    return EvaluationCriteria(
        objective="Explain how configuration reaches the application.",
        objective_criteria=(
            ObjectiveCriterion(
                "entry_point", "where configuration loading begins"
            ),
            ObjectiveCriterion(
                "configuration_source", "the configuration file"
            ),
            ObjectiveCriterion("loader", "the component that loads it"),
        ),
    )


def run_a() -> EvaluationResult:
    """Complete run: answer + evidence + supported claims + full
    structured result + within the bound + objective satisfied."""
    result = AgentRunResult(
        answer="config.yaml is read by loader.py into application.py.",
        evidence_validation=(
            _supported(
                "Configuration is read from config.yaml.", "config.yaml"
            ),
        ),
        structured_output=ConfigSummary(
            entry_point="application.py:main()",
            configuration_source="config.yaml",
            loader="loader.py",
        ),
    )
    return Evaluator.evaluate(
        result, _trace("run-a", successes=3, executed=3), _criteria()
    )


def run_c() -> EvaluationResult:
    """At the configured limit, evidence-backed claims, but the
    objective is incomplete (loader field empty)."""
    result = AgentRunResult(
        answer="config.yaml is read by loader.py into application.py.",
        evidence_validation=(
            _supported(
                "Configuration is read from config.yaml.", "config.yaml"
            ),
        ),
        structured_output=ConfigSummary(
            entry_point="application.py:main()",
            configuration_source="config.yaml",
            loader="",
        ),
    )
    return Evaluator.evaluate(
        result,
        _trace("run-c", successes=3, max_tool_calls=5, executed=5),
        _criteria(),
    )


def run_b() -> EvaluationResult:
    """A final answer with zero successful repository evidence."""
    result = AgentRunResult(
        answer="Configuration is loaded somehow (no tools consulted)."
    )
    return Evaluator.evaluate(
        result, _trace("run-b", successes=0, executed=0), _criteria()
    )


# ---------------------------------------------------------------------------
# Spec §28 presentation block
# ---------------------------------------------------------------------------


def print_block(evaluation: EvaluationResult) -> None:
    print("Investigation Evaluation")
    print()
    for item in evaluation.criteria:
        label = CRITERION_LABELS.get(item.criterion, item.criterion) + ":"
        print("{:<24}{}".format(label, item.status))
        if item.status != STATUS_PASS:
            print("    {}".format(item.explanation))
    print()
    print("{:<24}{}".format("Overall:", evaluation.status))
    print("    {}".format(evaluation.summary))


def main() -> int:
    started = time.time()
    print("=" * 72)
    print("M9 — Investigation Evaluation (deterministic, offline)")
    print("=" * 72)
    print(
        "objective: {}".format(_criteria().objective)
    )
    print(
        "expected: {}".format(
            ", ".join(
                item.key for item in _criteria().objective_criteria
            )
        )
    )
    print()

    evaluations = {
        "A": run_a(),
        "B": run_b(),
        "C": run_c(),
    }

    print(RULE)
    print("Run A — complete investigation (spec §17 run A)")
    print(RULE)
    print_block(evaluations["A"])
    print()
    print(
        "    note: claims + structured output are separate run modes in "
        "one real Agent"
    )
    print(
        "    (documented M8.9 ceiling) — this fixture assembles both "
        "artifact kinds"
    )
    print()

    print(RULE)
    print(
        "Run C — at the limit, objective incomplete (spec §17 run C)"
    )
    print(RULE)
    print_block(evaluations["C"])
    print()
    print(
        "    lesson: the Agent executed correctly AND still failed the "
        "investigation"
    )
    print()

    print(RULE)
    print("Run B — final answer, no repository evidence")
    print(RULE)
    print_block(evaluations["B"])
    print()

    # Captured M8.7 trace: real tool results over the pinned checkout.
    fixture_payload: Dict[str, Any] = {}
    fixture_note = "fixture not found: {}".format(FIXTURE_PATH)
    if FIXTURE_PATH.is_file():
        fixture_payload = json.loads(
            FIXTURE_PATH.read_text(encoding="utf-8")
        )
        final_answer = (fixture_payload.get("summary") or {}).get(
            "final_answer"
        )
        fixture_result = AgentRunResult(
            answer=final_answer or ""
            if isinstance(final_answer, str)
            else ""
        )
        fixture_eval = Evaluator.evaluate(
            fixture_result,
            fixture_payload,
            EvaluationCriteria(
                objective="Investigate the captured repository run."
            ),
        )
        evaluations["fixture"] = fixture_eval
        print(RULE)
        print("Captured M8.7 trace fixture — honest UNKNOWNs")
        print(RULE)
        print_block(fixture_eval)
        print(
            "    no claims recorded -> traceability UNKNOWN; no expected "
            "criteria -> coverage"
        )
        print(
            "    UNKNOWN; neither is ever upgraded to PASS just because "
            "the run looks fine"
        )
        print()

    # Artifact sink: evaluation.{json, md} beside the trace tree.
    sink_ok = False
    sink_detail = ""
    with tempfile.TemporaryDirectory() as tmp:
        sink = JsonFileEvaluationSink(root=tmp)
        sink.write(evaluations["C"])
        json_path = Path(tmp) / "run-c" / "evaluation.json"
        md_path = Path(tmp) / "run-c" / "evaluation.md"
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            restored = EvaluationResult.from_dict(payload)
            sink_ok = (
                restored == evaluations["C"]
                and md_path.is_file()
                and md_path.read_text(encoding="utf-8")
                == render_markdown(evaluations["C"])
            )
            sink_detail = (
                "evaluation.json round-trips; evaluation.md projects it"
                if sink_ok
                else "artifact mismatch"
            )
        except Exception as exc:  # pragma: no cover - report, not crash
            sink_detail = "artifact error: {}".format(exc)

    # Determinism over the fixture payload when available.
    deterministic = True
    if fixture_payload:
        first = Evaluator.evaluate(
            AgentRunResult(answer=""),
            fixture_payload,
            EvaluationCriteria(),
        ).to_json()
        second = Evaluator.evaluate(
            AgentRunResult(answer=""),
            fixture_payload,
            EvaluationCriteria(),
        ).to_json()
        deterministic = first == second
    else:
        deterministic = (
            run_a().to_json() == run_a().to_json()
        )

    payloads = [
        evaluation.to_dict() for evaluation in evaluations.values()
    ]
    if "fixture" in evaluations:
        payloads.append(evaluations["fixture"].to_dict())
    markdowns = [
        render_markdown(evaluation)
        for evaluation in evaluations.values()
    ]

    no_score = True
    for payload in payloads:
        blob = json.dumps(payload).lower()
        for key in SCORE_KEYS:
            if '"{}"'.format(key) in blob:
                no_score = False
        if not isinstance(payload["status"], str):
            no_score = False

    no_banned = True
    for payload in payloads:
        for key in _walk_keys(payload):
            if key in BANNED_REASONING_FIELDS or key in (
                "reasoning",
                "reasoning_trace",
            ):
                no_banned = False
    for text in markdowns:
        lowered = text.lower()
        for banned in BANNED_REASONING_FIELDS + (
            "reasoning",
            "reasoning_trace",
        ):
            if banned in lowered:
                no_banned = False

    repo_free = True
    for payload in payloads:
        blob = json.dumps(payload).lower()
        for term in ("adop", "mcp", "dag-agent", "servers.yaml", "openai"):
            if term in blob:
                repo_free = False

    a = evaluations["A"]
    b = evaluations["B"]
    c = evaluations["C"]

    checks = [
        (
            "Run A: all five criteria PASS, overall PASS",
            a.status == STATUS_PASS
            and all(item.status == STATUS_PASS for item in a.criteria),
            a.summary,
        ),
        (
            "Run A: deterministic counts summary",
            a.summary == "5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL",
            a.summary,
        ),
        (
            "Run C: execution bound PASS at the configured limit",
            c.criterion(CRITERION_EXECUTION_BOUND).status == STATUS_PASS,
            c.criterion(CRITERION_EXECUTION_BOUND).explanation,
        ),
        (
            "Run C: objective coverage FAIL names the missing field",
            c.criterion(CRITERION_OBJECTIVE_COVERAGE).status == STATUS_FAIL
            and "loader"
            in c.criterion(CRITERION_OBJECTIVE_COVERAGE).explanation,
            c.criterion(CRITERION_OBJECTIVE_COVERAGE).explanation,
        ),
        (
            "Run C: overall FAIL (compliance is not success)",
            c.status == STATUS_FAIL
            and c.criterion(CRITERION_EXECUTION_BOUND).status == STATUS_PASS,
            c.summary,
        ),
        (
            "Run B: evidence gathered FAIL with a final answer present",
            b.criterion(CRITERION_FINAL_RESULT).status == STATUS_PASS
            and b.criterion(CRITERION_EVIDENCE_GATHERED).status == STATUS_FAIL,
            b.criterion(CRITERION_EVIDENCE_GATHERED).explanation,
        ),
        (
            "determinism: same inputs -> byte-identical evaluation JSON",
            deterministic,
            "evaluated twice, identical to_json()",
        ),
        (
            "M8.7 trace fixture: PASS where evaluable, UNKNOWN where "
            "not (never silently PASS)",
            "fixture" in evaluations
            and evaluations["fixture"].criterion(
                CRITERION_EVIDENCE_GATHERED
            ).status
            == STATUS_PASS
            and evaluations["fixture"].criterion(
                CRITERION_CLAIM_TRACEABILITY
            ).status
            == STATUS_UNKNOWN
            and evaluations["fixture"].criterion(
                CRITERION_OBJECTIVE_COVERAGE
            ).status
            == STATUS_UNKNOWN
            and evaluations["fixture"].status == "PARTIAL",
            fixture_note,
        ),
        (
            "blocked tool calls are never counted as executions",
            _check_blocked_not_counted(),
            "at-limit fixture with blocked calls stays PASS "
            "(executed <= max, blocked reported separately)",
        ),
        (
            "no numeric score anywhere in any payload",
            no_score,
            "score keys absent; status vocabulary is PASS/FAIL/"
            "PARTIAL/UNKNOWN",
        ),
        (
            "no hidden reasoning vocabulary in payloads or markdown",
            no_banned,
            "banned fields absent from JSON keys and rendered text",
        ),
        (
            "repository-agnostic verdict (no ADOP/MCP/tool terminology)",
            repo_free,
            "payloads free of repository-specific terms",
        ),
        (
            "artifact sink: evaluation.{json,md} beside the trace, "
            "JSON round-trips, markdown projects it",
            sink_ok,
            sink_detail,
        ),
    ]

    print("Checks")
    failures = 0
    for number, (label, ok, detail) in enumerate(checks, start=1):
        verdict = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        dots = "." * max(3, 62 - len(label))
        print(
            "[{}/{}] {} {}  {}".format(
                number, len(checks), label, dots, verdict
            )
        )
        if not ok:
            print("         {}".format(detail))
    print(RULE)
    elapsed = time.time() - started
    if failures:
        print(
            "Result: FAIL - {}/{} checks failed ({:.2f}s)".format(
                failures, len(checks), elapsed
            )
        )
        return 6
    print(
        "Result: PASS - {}/{} checks ({:.2f}s)".format(
            len(checks), len(checks), elapsed
        )
    )
    return 0


def _check_blocked_not_counted() -> bool:
    """M8.5 semantics: 5 of 5 executed, 3 blocked follow-ups → PASS."""
    evaluation = Evaluator.evaluate(
        AgentRunResult(answer="done."),
        _trace("run-bound", successes=2, max_tool_calls=5, executed=5, blocked=3),
        EvaluationCriteria(),
    )
    item = evaluation.criterion(CRITERION_EXECUTION_BOUND)
    return (
        item.status == STATUS_PASS
        and "3 blocked" in item.explanation
        and "not counted" in item.explanation
    )


if __name__ == "__main__":
    sys.exit(main())
