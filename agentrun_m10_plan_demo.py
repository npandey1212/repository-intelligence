"""M10 — explicit plan-and-execute demonstration (real model).

One human-runnable script that shows the M10 learning goal happening
for a REAL investigation — planning, validation, and execution as one
chronological run with one trace (spec §39):

    python agentrun_m10_plan_demo.py
    python agentrun_m10_plan_demo.py --model gpt-4o-mini
    python agentrun_m10_plan_demo.py --question "Investigate …"

What it does (each step reuses the existing implementation):

    1  load the existing M6 Architecture Context
       planning.load_architecture_context() → markdown + record
       (artifact, schema version, pinned commit, snapshot id)
    2  compose the existing Agent              (OpenAILLMClient +
                                                 default_tool_dispatcher +
                                                 JsonFileTraceSink,
                                                 one budget, M8.9
                                                 structured-output contract)
    3  run ONE planned investigation           agent.run_planned(
                                                 objective, context, record)
       — ONE planner LLM call (no tools) → deterministic validation →
         plan recorded on the ONE AgentTrace BEFORE the first tool
         execution → each step executed sequentially by the existing
         ReAct loop under the existing max_tool_calls bound
    4  obtain the completed AgentRunResult     (+ agent.last_trace:
                                                 plan, plan_step_id
                                                 iteration correlation)
    5  print the plan and the execution        plan validation status,
                                                 per-step outcomes,
                                                 step-by-step timeline
    6  invoke the EXISTING M9 evaluator        (once, strictly after the
                                                 run — optional stage of
                                                 this same run, no second
                                                 loop, no LLM call)
    7  persist through the existing sinks      trace.{json,md} +
                                                 evaluation.{json,md}
    8  exit code reflects the outcome          (mapping below)

This script does NOT implement planning, validation, execution, or
evaluation. It is an entry point into the existing implementation
(`repo_intelligence.planning` for the plan contract/loader, the
existing `Agent.run_planned` for the run): it supplies the objective,
the existing Architecture Context artifact, and the success criteria,
and reads back the results. It contains no planner prompt, no
validation rules, no loop, and no serialization of its own beyond
console formatting.

Exit codes:
    0  run completed; evaluation PASS or PARTIAL (no criterion FAILED)
       — PARTIAL means some criterion could not be evaluated (e.g.
         claim_traceability UNKNOWN in M8.9 mode); it is printed and
         recorded, never silently converted into PASS
    1  run completed, but evaluation FAILED — criterion-level detail
       is printed before exiting ("Investigation completed, but
       evaluation failed.")
    6  planning/execution/configuration failure (missing
       OPENAI_API_KEY, missing Architecture Context artifact,
       LLMError, AgentError, PlanValidationError,
       StructuredOutputValidationError) — printed as "Agent
       execution failed." (same convention as the M9 demo)

Unexpected programming errors are NOT caught: they propagate with a
traceback rather than masquerading as an evaluation failure.

Failure semantics are visible, not hidden: a planner failure exits 6
with the recorded PLAN_VALIDATION_FAILED reason on the written trace;
no repository tool ever runs without a validated plan in the trace.

Credentials/configuration: requires OPENAI_API_KEY and the existing
M6 Architecture Context artifact (copy .env.example to .env; never
commit .env; re-run M6 to regenerate). Without a key or context the
script fails CLEARLY with exit 6 and never falls back to a fake
model. The offline test suite never runs this script against the
network (tests/planning/test_plan_demo.py wires the demo path with
scripted doubles only).
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
from repo_intelligence.context import (  # noqa: E402
    ArchitectureContextError,
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
from repo_intelligence.planning import (  # noqa: E402
    PlanValidationError,
    load_architecture_context,
)
from repo_intelligence.tool_dispatcher import (  # noqa: E402
    default_tool_dispatcher,
)

RULE = "=" * 72
SECTION = "-" * 52

# The single M8.5 tool-call limit for this run (no second budget; the
# plan spans several steps but shares this ONE bound — sized so a
# multi-step plan has room to finish within it).
MAX_TOOL_CALLS = 60

EXIT_SUCCESS = 0
EXIT_EVALUATION_FAILED = 1
EXIT_EXECUTION_FAILED = 6

# M10 explicitly allows a repository-aware objective for the demo
# (spec §7): the plan is produced by the LLM over the existing
# Architecture Context, not hard-coded to any repository structure.
DEFAULT_OBJECTIVE = (
    "Explain how the ADOP agents, tools, and MCP servers are connected. "
    "Use the repository tools to gather evidence from files you "
    "actually inspect — do not rely on the initial context alone. "
    "Record your conclusion as structured output: 'finding' = a "
    "self-contained explanation of the connection; 'evidence_paths' = "
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


def build_agent(llm_client, tool_dispatcher, trace_sink) -> Agent:
    """Compose the EXISTING Agent — no loop, dispatcher, or planner of
    our own: one budget, one run, M8.9 structured output. The plan is
    produced inside ``run_planned`` by the one planner LLM call."""
    return Agent(
        llm_client=llm_client,
        tool_dispatcher=tool_dispatcher,
        trace_sink=trace_sink,
        execution_config=AgentExecutionConfig(
            max_tool_calls=MAX_TOOL_CALLS,
        ),
        structured_output_model=InvestigationSummary,
    )


def build_criteria(objective: str) -> EvaluationCriteria:
    """The existing M9 input type — what success means for this run."""
    return EvaluationCriteria(
        objective=objective,
        objective_criteria=(
            ObjectiveCriterion(
                "finding",
                "how the agents, tools, and MCP servers are connected",
            ),
            ObjectiveCriterion(
                "evidence_paths",
                "files inspected as supporting evidence",
            ),
        ),
    )


def evaluate_run(result, trace, criteria, evaluation_sink) -> EvaluationResult:
    """Invoke the existing M9 evaluator ONCE, after the run — then
    persist through the existing sink (no serialization of our own)."""
    evaluation = Evaluator.evaluate(result, trace, criteria)
    evaluation_sink.write(evaluation)
    return evaluation


def exit_code_for(evaluation: EvaluationResult) -> int:
    """0 = no criterion FAILED (PASS or PARTIAL); 1 = FAIL."""
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


def print_plan(trace, stream=None) -> None:
    """Console projection of the plan recorded on the trace — what was
    planned and validated, in order, with each step's outcome.

    Reads ONLY ``trace.plan`` / ``trace.architecture_context`` (the
    authoritative recorded payloads); it does not re-derive or
    re-validate anything.
    """
    stream = stream or sys.stdout
    print("Plan", file=stream)
    print(SECTION, file=stream)
    context = getattr(trace, "architecture_context", None)
    if context:
        print("context:   {}".format(context.get("artifact")), file=stream)
        print(
            "commit:    {}".format(context.get("commit_sha")), file=stream
        )
        print(
            "snapshot:  {}".format(context.get("snapshot_id")), file=stream
        )
    plan = getattr(trace, "plan", None)
    if plan is None:
        print("no plan recorded on this trace", file=stream)
        return
    validation = plan.get("validation")
    if validation:
        print(
            "validation: {}".format(validation.get("status")), file=stream
        )
    print("status:    {}".format(plan.get("status")), file=stream)
    print("steps:", file=stream)
    for step in plan.get("steps", []):
        print(
            "  {} [{}] {}".format(
                step.get("id"), step.get("status"), step.get("goal")
            ),
            file=stream,
        )
        result = step.get("result")
        if result:
            compact = " ".join(str(result).split())
            if len(compact) > 100:
                compact = compact[:97] + "..."
            print("      {}".format(compact), file=stream)


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
    """Advisory only: search/read need the M1 pinned checkout."""
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
    # plan-step correlation: which iteration belonged to which step
    step_ids = [
        getattr(iteration, "plan_step_id", None)
        for iteration in trace.iterations
    ]
    if any(step_id is not None for step_id in step_ids):
        timeline = ", ".join(
            "iter {}={}".format(
                index + 1, step_id if step_id is not None else "-"
            )
            for index, step_id in enumerate(step_ids)
        )
        print("plan-step timeline: {}".format(timeline))


def _execution_failed(exc: BaseException, agent, trace_sink) -> int:
    """Expected failures exit clearly (existing error information,
    trace path when available) — never as an evaluation failure."""
    print("Agent execution failed.", file=sys.stderr)
    print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
    _print_trace_paths(trace_sink, agent.last_trace, sys.stderr)
    return EXIT_EXECUTION_FAILED


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m10_plan_demo.py",
        description=(
            "M10 end-to-end demonstration: one real-model planned "
            "investigation — planner LLM call, deterministic "
            "validation, plan recorded in the existing AgentTrace, "
            "sequential step execution through the existing ReAct "
            "loop, then the existing M9 evaluator over the completed "
            "run (offline evaluation, artifacts beside the trace)."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation objective (default: the "
        "ADOP agents/tools/MCP connection investigation)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    args = parser.parse_args(argv)
    objective = args.question or DEFAULT_OBJECTIVE

    # configuration (explicit .env load at the composition root —
    # same convention as the M9 demo)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M10 demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_m10_plan_demo.py",
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED

    # the planner's orientation input: the EXISTING M6 artifact,
    # passed directly (never a tool call before the plan exists)
    try:
        context_text, context_record = load_architecture_context()
    except ArchitectureContextError as exc:
        print("Agent execution failed.", file=sys.stderr)
        print("ArchitectureContextError: {}".format(exc), file=sys.stderr)
        print("Re-run M6 to regenerate the Architecture Context.", file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    try:
        llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        print("Agent execution failed.", file=sys.stderr)
        print("LLMError: {}".format(exc), file=sys.stderr)
        return EXIT_EXECUTION_FAILED

    model_id = args.model or DEFAULT_MODEL
    trace_sink = JsonFileTraceSink()
    agent = build_agent(llm_client, default_tool_dispatcher(), trace_sink)
    criteria = build_criteria(objective)

    print(RULE)
    print("M10 — Explicit Plan-and-Execute (end-to-end demonstration)")
    print(RULE)
    print("Objective:")
    print(objective)
    print()
    print(
        "model: {}   max_tool_calls: {}   run mode: M10 plan-and-"
        "execute over M8.9 structured output".format(
            model_id, MAX_TOOL_CALLS
        )
    )
    print(
        "architecture context: {} (snapshot {})".format(
            context_record.get("artifact"),
            context_record.get("snapshot_id"),
        )
    )
    _warn_if_checkout_missing()

    # ONE planned run — planner call, validation, plan recorded, then
    # the existing loop executes each step; evaluation happens strictly
    # after this.
    try:
        result = agent.run_planned(
            objective, context_text, context_record
        )
    except AgentError as exc:
        return _execution_failed(exc, agent, trace_sink)
    except LLMError as exc:
        return _execution_failed(exc, agent, trace_sink)
    except StructuredOutputValidationError as exc:
        return _execution_failed(exc, agent, trace_sink)
    except PlanValidationError as exc:
        return _execution_failed(exc, agent, trace_sink)

    trace = agent.last_trace
    print()
    print_plan(trace)
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
    trace_dir = getattr(trace_sink, "trace_dir", None)
    if trace is not None and callable(trace_dir):
        for name in ("trace.json", "trace.md"):
            path = trace_dir(trace.run_id) / name
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
            "converted into PASS)."
        )
    else:
        print(
            "Evaluation completed: PASS — every criterion established."
        )
    print(
        "exit code: {} (evaluation: {})".format(
            EXIT_SUCCESS, evaluation.status
        )
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
