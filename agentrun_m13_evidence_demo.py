"""M13 — evidence-backed agent conclusions (real model, controlled run).

One human-runnable script for the M13 experiment (spec §16/§19):

    python agentrun_m13_evidence_demo.py
    python agentrun_m13_evidence_demo.py --model gpt-4o-mini
    python agentrun_m13_evidence_demo.py --question "Investigate …"

It runs the SAME controlled investigation objective as the M13
baseline (the M10 plan-and-execute demo) against the same pinned
checkout, through the SAME existing implementation — the only change
is the structured-output contract, which now carries evidence-backed
claims:

    1  one existing M6 Architecture Context (loaded directly)
    2  one existing Agent (OpenAILLMClient + default dispatcher +
       JsonFileTraceSink) with a CLAIM-BEARING M8.9 contract —
       InvestigationSummary {finding, evidence_paths, claims[]},
       each claim requiring inspected-file evidence_refs (§12)
    3  ONE planned run: agent.run_planned → the existing planner
       call, deterministic plan validation, plan on the one trace,
       sequential ReAct steps under the one max_tool_calls bound
    4  AFTER the loop the existing deterministic step runs:
       gather_evidence(trace) → validate_claims(...,
       require_inspected=True) → recorded on the trace — the
       application, never the LLM, decides whether each claim's
       evidence was actually inspected (CLAIM_EVIDENCE_INVALID when
       not)
    5  the existing M9 evaluator interprets the recorded result —
       claim_traceability is now measurable (PASS / FAIL / PARTIAL)
       instead of the baseline's honest UNKNOWN
    6  the full chain is printed: objective → plan → tool calls →
       observed evidence → claims → evidence paths → deterministic
       validation → evaluation (§19), and persisted beside the trace

What this script does NOT do: no planner, validator, loop, second
agent, evaluator agent, reflection, or judging step of its own —
it is a composition root over ``repo_intelligence`` (same convention
as the M10/M11/M12 demos) and contains no serialization beyond
console formatting.

Exit codes (same convention as the M10 demo):
    0  run completed; evaluation PASS or PARTIAL (no criterion FAILED)
    1  run completed, but evaluation FAILED — criterion detail printed
       (e.g. claim_traceability FAIL: a claim cited a file that was
       never inspected)
    6  planning/execution/configuration failure (missing
       OPENAI_API_KEY, missing Architecture Context, LLMError,
       AgentError, PlanValidationError,
       StructuredOutputValidationError)

Unexpected programming errors are NOT caught: they propagate with a
traceback rather than masquerading as an evaluation failure.

Credentials/configuration: requires OPENAI_API_KEY and the existing
M6 Architecture Context artifact (copy .env.example to .env; never
commit .env; re-run M6 to regenerate). Without a key or context the
script fails CLEARLY with exit 6 and never falls back to a fake
model. The offline test suite never runs this script against the
network (tests/agent/test_m13_evidence_demo.py wires the demo path
with scripted doubles only).
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

from pydantic import BaseModel, Field, constr  # noqa: E402

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

# The single M8.5 tool-call limit for this run — identical to the
# M13 baseline run so the controlled comparison holds the budget
# constant (spec §16).
MAX_TOOL_CALLS = 60

EXIT_SUCCESS = 0
EXIT_EVALUATION_FAILED = 1
EXIT_EXECUTION_FAILED = 6

# §16: the SAME controlled investigation objective as the baseline.
DEFAULT_OBJECTIVE = (
    "Explain how the ADOP agents, tools, and MCP servers are connected. "
    "Use the repository tools to gather evidence from files you "
    "actually inspect — do not rely on the initial context alone. "
    "Record your conclusion as structured output using explicit "
    "claims, where every claim contains the repository-relative paths "
    "of the files that support it."
)


class EvidenceReference(BaseModel):
    """One inspected-file pointer inside a claim (§3 contract)."""

    path: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Repository-relative path of a file the agent ACTUALLY "
            "INSPECTED during this run (read via read_file). Files "
            "that only appeared in search results, graph output, the "
            "architecture context, or the plan are NOT evidence."
        ),
    )
    line_start: Optional[int] = Field(
        None,
        description=(
            "First line read (only when both bounds are known; "
            "omit both otherwise)."
        ),
    )
    line_end: Optional[int] = Field(
        None,
        description="Last line read (same rule as line_start).",
    )


class InvestigationClaim(BaseModel):
    """One important factual claim, grounded in inspected files (§3)."""

    statement: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One specific factual conclusion of this investigation — "
            "no more than the inspected evidence supports; represent "
            "uncertainty as uncertainty rather than inventing it away."
        ),
    )
    evidence_refs: List[EvidenceReference] = Field(
        ...,
        min_items=1,
        description=(
            "The repository files actually inspected during this run "
            "that support this claim (at least one). The application "
            "validates every reference against this run's trace; a "
            "reference to a never-inspected file is recorded as "
            "CLAIM_EVIDENCE_INVALID."
        ),
    )


class InvestigationSummary(BaseModel):
    """The final structured response for this investigation.

    Output EXACTLY one JSON object matching this schema and nothing
    else. Every field is required: state the finding, list the
    files inspected as evidence_paths, and give at least one claim.
    A response missing any field — in particular "claims" — fails
    validation and the run fails. Conclusions and evidence
    references only; never hidden reasoning.
    """

    finding: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "A self-contained explanation of the investigation "
            "result — precise where the inspected evidence is "
            "precise, explicitly uncertain where it is not."
        ),
    )
    evidence_paths: List[constr(strip_whitespace=True, min_length=1)] = (
        Field(
            ...,
            min_items=1,
            description=(
                "Repository-relative paths of every file inspected as "
                "supporting evidence during this run."
            ),
        )
    )
    claims: List[InvestigationClaim] = Field(
        ...,
        min_items=1,
        description=(
            "REQUIRED, at least one: the important factual claims "
            "backing the finding; each cites the files this run "
            "inspected. The response fails validation without it — "
            "do not omit this field."
        ),
    )


def build_agent(llm_client, tool_dispatcher, trace_sink) -> Agent:
    """Compose the EXISTING Agent — no loop, dispatcher, or planner of
    our own: one budget, one planned run, one claim-bearing M8.9
    contract (the claims are extracted and validated application-side
    after the loop — M13)."""
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
    """The existing M9 input type — what success means for this run:
    the finding, the inspected evidence paths, and the claims
    themselves (§14: a successful result must contain claims)."""
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
            ObjectiveCriterion(
                "claims",
                "evidence-backed claims grounding the finding",
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
                step.get("id"),
                step.get("status"),
                step.get("goal"),
            ),
            file=stream,
        )
        result = step.get("result")
        if result:
            compact = " ".join(str(result).split())
            if len(compact) > 100:
                compact = compact[:97] + "..."
            print("      {}".format(compact), file=stream)


def _call_sites(trace) -> dict:
    """tool_call_id → (iteration number, tool name) from the ONE trace
    — the §5 provenance chain: claim → evidence path → which tool
    call/iteration produced it."""
    sites = {}
    if trace is None:
        return sites
    for iteration in trace.iterations:
        for result in iteration.tool_results:
            call_id = result.get("tool_call_id")
            if call_id is not None:
                sites[call_id] = (
                    iteration.iteration,
                    result.get("tool") or "tool",
                )
    return sites


def print_claims(result, trace, stream=None) -> None:
    """The M13 chain, printed from recorded data only (§19):

    claim → evidence refs → deterministic validation → the tool call
    that actually inspected the file. Never model-authored reasoning.
    """
    stream = stream or sys.stdout
    print("Claims and evidence", file=stream)
    print(SECTION, file=stream)
    claims = result.claims
    if not claims:
        print(
            "the structured payload carried no claims — claim "
            "traceability cannot be established (reported, never "
            "manufactured)",
            file=stream,
        )
        return
    sites = _call_sites(trace)
    validations = list(result.evidence_validation)
    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0}
    for index, claim in enumerate(claims, start=1):
        print("{}. {}".format(index, claim.statement), file=stream)
        for ref in claim.evidence_refs:
            print("   evidence: {}".format(ref.render()), file=stream)
        entry = validations[index - 1] if index - 1 < len(validations) else None
        if entry is None:
            print("   validation: (not evaluated)", file=stream)
            continue
        counts[entry.status] = counts.get(entry.status, 0) + 1
        label = entry.status
        if entry.code:
            label = "{} [{}]".format(label, entry.code)
        print("   validation: {}".format(label), file=stream)
        if entry.reason:
            print("     reason: {}".format(entry.reason), file=stream)
        for item in entry.matched_evidence:
            site = sites.get(item.tool_call_id)
            where = (
                "iteration {} via {}".format(site[0], site[1])
                if site is not None
                else item.tool
            )
            print(
                "     inspected: {} ({})".format(
                    item.ref.render(), where
                ),
                file=stream,
            )
    print(
        "claims: {} — {} supported, {} unsupported, {} unevaluable".format(
            len(claims),
            counts.get("SUPPORTED", 0),
            counts.get("UNSUPPORTED", 0),
            counts.get("UNKNOWN", 0),
        ),
        file=stream,
    )


def _print_execution(trace) -> None:
    print("Agent execution", file=sys.stdout)
    print(SECTION, file=sys.stdout)
    summary = trace.summary if trace is not None else None
    if summary is None:
        print("run summary not recorded", file=sys.stdout)
        return
    print("iterations: {}".format(summary.total_iterations), file=sys.stdout)
    print(
        "tool calls: {} (executed {}, blocked {})".format(
            summary.total_tool_calls,
            summary.tool_calls_executed,
            summary.tool_calls_blocked,
        ),
        file=sys.stdout,
    )
    print("max tool calls: {}".format(summary.max_tool_calls), file=sys.stdout)
    print("termination: {}".format(summary.termination_reason), file=sys.stdout)
    print("run id: {}".format(trace.run_id), file=sys.stdout)
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
        print("plan-step timeline: {}".format(timeline), file=sys.stdout)


def _relative(path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _print_trace_paths(trace_sink, trace, stream=None) -> None:
    """Where this run's trace was written (file sinks only)."""
    stream = stream or sys.stderr
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


def _execution_failed(exc: BaseException, agent, trace_sink) -> int:
    """Expected failures exit clearly (existing error information,
    trace path when available) — never as an evaluation failure."""
    print("Agent execution failed.", file=sys.stderr)
    print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
    _print_trace_paths(trace_sink, agent.last_trace, sys.stderr)
    return EXIT_EXECUTION_FAILED


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m13_evidence_demo.py",
        description=(
            "M13 end-to-end demonstration: the controlled "
            "investigation objective from the baseline, now with an "
            "evidence-backed claim contract — one planned run "
            "through the existing planner and ReAct loop, "
            "application-side inspected-evidence validation after "
            "the loop, then the existing M9 evaluator (offline "
            "evaluation, artifacts beside the trace)."
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
    # same convention as the M9/M10 demos)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M13 demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_m13_evidence_demo.py",
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
    print("M13 — Evidence-Backed Agent Conclusions (end-to-end)")
    print(RULE)
    print("Objective:")
    print(objective)
    print()
    print(
        "model: {}   max_tool_calls: {}   run mode: M10 plan-and-"
        "execute over claim-bearing M8.9 structured output".format(
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
    # the existing loop executes each step; evidence validation runs
    # strictly AFTER the loop (existing deterministic step), and
    # evaluation strictly after that.
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
    print_claims(result, trace)
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
