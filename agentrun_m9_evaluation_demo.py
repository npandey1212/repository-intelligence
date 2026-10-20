"""M9 — end-to-end investigation evaluation demonstration (real model).

One human-runnable script that shows the M9 learning goal happening
for a REAL investigation: execution and evaluation as two separate,
observable phases — with no second evaluator, no second loop, and no
LLM call outside the Agent:

    python agentrun_m9_evaluation_demo.py
    python agentrun_m9_evaluation_demo.py --model gpt-4o-mini
    python agentrun_m9_evaluation_demo.py --question "Investigate …"

What it does (each step reuses the existing implementation):

    1  compose the existing Agent          (OpenAILLMClient +
                                             default_tool_dispatcher +
                                             JsonFileTraceSink,
                                             one budget, M8.9
                                             structured-output contract)
    2  run ONE investigation               agent.run_result(objective)
                                             — every Decision → Action →
                                             Observation cycle belongs to
                                             the existing M8.4 loop
    3  obtain the completed AgentRunResult (+ agent.last_trace)
    4  invoke the EXISTING M9 evaluator    Evaluator.evaluate(result,
                                             trace, criteria) — once,
                                             strictly after the run
    5  print the evaluation                per-criterion statuses from
                                             the existing vocabulary
    6  persist the evaluation              JsonFileEvaluationSink →
                                             runtime/agent_runs/<run-id>/
                                             evaluation.{json,md}
    7  exit code reflects the outcome      (mapping below)

This script does NOT implement evaluation. It is an entry point into
the existing M9 package (`repo_intelligence.evaluation`): it supplies
the completed run, the trace, and the explicit success criteria, and
reads back the EvaluationResult. It contains no criteria logic, no
status vocabulary, and no serialization of its own beyond console
formatting.

Run mode (faithful to the architecture — spec §7 of the M9 prompt):
this run uses the M8.9 structured-output mode, which produces a typed
result but NO M8.8 structured claims. The evaluator therefore reports

    claim_traceability = UNKNOWN

— exactly what the run actually supports. Nothing is manufactured to
make all five criteria PASS: UNKNOWN is reported, never upgraded.
Objective coverage IS evaluated against the expected fields of the
structured result (the M8.9 contract sent with this run).

Exit codes:
    0  run completed; evaluation PASS or PARTIAL (no criterion FAILED)
       — PARTIAL means some criterion could not be evaluated (e.g.
         claim_traceability UNKNOWN in this mode); it is printed and
         recorded, never silently converted to PASS
    1  run completed, but evaluation FAILED — criterion-level detail
       is printed before exiting ("Investigation completed, but
       evaluation failed.")
    6  agent execution/configuration failure (missing OPENAI_API_KEY,
       LLMError, AgentError, structured-output validation failure) —
       printed as "Agent execution failed." (same convention as
       agentrun_react_live_test.py)

Unexpected programming errors are NOT caught: they propagate with a
traceback rather than masquerading as an evaluation failure.

Credentials/configuration: requires OPENAI_API_KEY (copy .env.example
to .env; never commit .env) — same convention as the M8.7 live
runner; `--question`/`--model` mirror its flags. Without a key the
script fails CLEARLY with exit 6 and never falls back to a fake
model. The offline test suite never runs this script against the
network (tests/evaluation/test_evaluation_demo.py wires the demo path
with scripted doubles only).

Repository-agnostic: the objective, the structured contract
(``finding`` / ``evidence_paths``) and the evaluation contain no
repository names, tool names, MCP/server terminology or file-path
rules — the existing tools and the evaluator's existing contracts
determine the behavior.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydantic import BaseModel  # noqa: E402

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    AgentExecutionConfig,
    JsonFileTraceSink,
)
from repo_intelligence.evaluation import (  # noqa: E402
    CRITERION_LABELS,
    STATUS_FAIL,
    STATUS_PASS,
    EvaluationCriteria,
    EvaluationResult,
    Evaluator,
    JsonFileEvaluationSink,
    ObjectiveCriterion,
)
from repo_intelligence.llm import (  # noqa: E402
    LLMError,
    StructuredOutputValidationError,
)
from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.llm.openai_client import (  # noqa: E402
    DEFAULT_MODEL,
    OpenAILLMClient,
)
from repo_intelligence.tool_dispatcher import (  # noqa: E402
    default_tool_dispatcher,
)

RULE = "=" * 72
SECTION = "-" * 52

# The single M8.5 tool-call limit for this run (no second budget).
MAX_TOOL_CALLS = 10

EXIT_SUCCESS = 0
EXIT_EVALUATION_FAILED = 1
EXIT_EXECUTION_FAILED = 6

# Generic, repository-agnostic investigation objective (spec: the demo
# must not hard-code repository names, tools, or file paths). It names
# the structured fields the M8.9 contract below requires, so the model
# knows what to record.
DEFAULT_OBJECTIVE = (
    "Investigate this repository and explain how one clearly "
    "identifiable implementation concern is implemented. Use the "
    "repository tools to gather evidence from files you actually "
    "inspect — do not rely on the initial context alone. Record your "
    "conclusion as structured output: 'finding' = a self-contained "
    "explanation of the concern and how it works; 'evidence_paths' = "
    "the repository-relative paths of the files that support it."
)


class InvestigationSummary(BaseModel):
    """Generic M8.9 output contract for this run (repo-agnostic).

    Also the explicit definition of objective coverage: M9 checks that
    every expected field exists and is substantive — success must be
    defined, not assumed.
    """

    finding: str
    evidence_paths: List[str]


def build_criteria(objective: str) -> EvaluationCriteria:
    """The existing M9 input type — what success means for this run.

    The five evaluation criteria themselves are fixed by the existing
    evaluator; this only supplies the run's objective and the expected
    structured-result fields.
    """
    return EvaluationCriteria(
        objective=objective,
        objective_criteria=(
            ObjectiveCriterion(
                "finding",
                "how the identified implementation concern works",
            ),
            ObjectiveCriterion(
                "evidence_paths",
                "files inspected as supporting evidence",
            ),
        ),
    )


def build_agent(llm_client, tool_dispatcher, trace_sink) -> Agent:
    """Compose the EXISTING Agent — no loop, dispatcher, or retry of
    our own: one budget, one run, M8.9 structured output."""
    return Agent(
        llm_client=llm_client,
        tool_dispatcher=tool_dispatcher,
        trace_sink=trace_sink,
        execution_config=AgentExecutionConfig(
            max_tool_calls=MAX_TOOL_CALLS,
        ),
        structured_output_model=InvestigationSummary,
    )


def evaluate_run(result, trace, criteria, evaluation_sink) -> EvaluationResult:
    """Invoke the existing M9 evaluator ONCE, after the run — then
    persist through the existing sink (no serialization of our own)."""
    evaluation = Evaluator.evaluate(result, trace, criteria)
    evaluation_sink.write(evaluation)
    return evaluation


def exit_code_for(evaluation: EvaluationResult) -> int:
    """0 = no criterion FAILED (PASS or PARTIAL); 1 = FAIL.

    PARTIAL is an honest incomplete verdict (some criterion UNKNOWN —
    e.g. claim_traceability in this run mode), not a failed
    investigation; the printed/serialized status distinguishes it.
    """
    if evaluation.status == STATUS_FAIL:
        return EXIT_EVALUATION_FAILED
    return EXIT_SUCCESS


def print_evaluation(evaluation: EvaluationResult, stream=None) -> None:
    """Console projection using the existing labels + vocabulary."""
    stream = stream or sys.stdout
    print("Evaluation", file=stream)
    print(SECTION, file=stream)
    for item in evaluation.criteria:
        label = CRITERION_LABELS.get(item.criterion, item.criterion) + ":"
        print("{:<21}{}".format(label, item.status), file=stream)
        if item.status != STATUS_PASS and item.explanation:
            print("    {}".format(item.explanation), file=stream)
    print(file=stream)
    print("{:<21}{}".format("Overall:", evaluation.status), file=stream)
    print("    {}".format(evaluation.summary), file=stream)


def _relative(path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _print_trace_paths(trace_sink, trace, stream) -> None:
    """Where this run's trace was written (file sinks only)."""
    json_path = getattr(trace_sink, "json_path", None)
    if trace is None or not callable(json_path):
        return
    path = json_path(trace.run_id)
    if path.exists():
        print("trace: {}".format(_relative(path)), file=stream)


def _warn_if_checkout_missing() -> None:
    """Advisory only: search/read need the M1 pinned checkout.

    A missing checkout is not a script failure — the tools return a
    structured failure observation and the existing Agent/evaluator
    semantics report it honestly (evidence_gathered may FAIL).
    """
    from repo_intelligence.tools.search_repository import (
        resolve_repository_root,
    )

    root, failure = resolve_repository_root(None, None)
    if failure is None and root is not None:
        return
    print(
        "warning: pinned repository checkout not found — "
        "search_repository/read_file will return REPOSITORY_NOT_FOUND "
        "observations (re-run M1 to restore it).",
        file=sys.stderr,
    )


def _print_execution(trace) -> None:
    print("Agent execution", file=sys.stdout)
    print(SECTION, file=sys.stdout)
    summary = trace.summary if trace is not None else None
    if summary is None:
        print("run summary not recorded", file=sys.stdout)
        return
    print("iterations: {}".format(summary.total_iterations))
    print(
        "tool calls: {} (executed {}, blocked {})".format(
            summary.total_tool_calls,
            summary.tool_calls_executed,
            summary.tool_calls_blocked,
        )
    )
    print("max tool calls: {}".format(summary.max_tool_calls))
    print("termination: {}".format(summary.termination_reason))
    print("run id: {}".format(trace.run_id))


def _execution_failed(exc: BaseException, agent, trace_sink) -> int:
    """Expected failures exit clearly (existing error information,
    trace path when available) — never as an evaluation failure."""
    print("Agent execution failed.", file=sys.stderr)
    print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
    _print_trace_paths(trace_sink, agent.last_trace, sys.stderr)
    return EXIT_EXECUTION_FAILED


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m9_evaluation_demo.py",
        description=(
            "M9 end-to-end demonstration: one real-model investigation "
            "through the existing Agent, then the existing M9 "
            "evaluator over the completed run (offline evaluation, "
            "evaluation.{json,md} beside the trace)."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation objective (default: generic "
        "implementation-concern investigation)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    args = parser.parse_args(argv)
    objective = args.question or DEFAULT_OBJECTIVE

    # configuration (explicit .env load at the composition root —
    # same convention as agentrun_react_live_test.py)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M9 demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_m9_evaluation_demo.py",
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED

    try:
        llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        print("Agent execution failed.", file=sys.stderr)
        print("LLMError: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    model_id = args.model or DEFAULT_MODEL
    trace_sink = JsonFileTraceSink()
    agent = build_agent(
        llm_client, default_tool_dispatcher(), trace_sink
    )
    criteria = build_criteria(objective)

    print(RULE)
    print("M9 — Investigation Evaluation (end-to-end demonstration)")
    print(RULE)
    print("Objective:")
    print(objective)
    print()
    print(
        "model: {}   max_tool_calls: {}   run mode: M8.9 structured "
        "output".format(model_id, MAX_TOOL_CALLS)
    )
    _warn_if_checkout_missing()

    # ONE run — every Decision → Action → Observation cycle belongs to
    # the existing Agent loop; evaluation happens strictly after this.
    try:
        result = agent.run_result(objective)
    except AgentError as exc:
        return _execution_failed(exc, agent, trace_sink)
    except LLMError as exc:
        return _execution_failed(exc, agent, trace_sink)
    except StructuredOutputValidationError as exc:
        return _execution_failed(exc, agent, trace_sink)

    trace = agent.last_trace
    print()
    _print_execution(trace)
    print()
    print(
        "run mode note: this mode produces no M8.8 structured claims — "
        "claim_traceability is evaluated as UNKNOWN (reported, never "
        "manufactured)."
    )
    print()

    # EXECUTION → EVALUATION: completed run + explicit criteria in,
    # one EvaluationResult out, persisted by the existing sink.
    evaluation_sink = JsonFileEvaluationSink()
    evaluation = evaluate_run(result, trace, criteria, evaluation_sink)
    print_evaluation(evaluation)

    print()
    print("Artifacts")
    print(SECTION)
    if trace is not None:
        for name in ("trace.json", "trace.md"):
            path = trace_sink.trace_dir(trace.run_id) / name
            if path.exists():
                print("trace:      {}".format(_relative(path)))
    for name in ("evaluation.json", "evaluation.md"):
        path = evaluation_sink.evaluation_dir(evaluation.run_id) / name
        if path.exists():
            print("evaluation: {}".format(_relative(path)))

    print()
    if evaluation.status == STATUS_FAIL:
        print("Investigation completed, but evaluation failed.")
        print("Failed criteria:")
        for item in evaluation.criteria:
            if item.status == STATUS_FAIL:
                print(
                    "  {}: {}".format(
                        CRITERION_LABELS.get(
                            item.criterion, item.criterion
                        ),
                        item.explanation,
                    )
                )
        code = exit_code_for(evaluation)
        print("exit code: {} (evaluation: FAIL)".format(code))
        return code
    if evaluation.status == "PARTIAL":
        print(
            "Evaluation completed: PARTIAL — no criterion FAILED; "
            "unevaluable criteria are reported as UNKNOWN (never "
            "converted to PASS)."
        )
    else:
        print(
            "Evaluation completed: PASS — every criterion established."
        )
    print("exit code: {} (evaluation: {})".format(EXIT_SUCCESS, evaluation.status))
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
