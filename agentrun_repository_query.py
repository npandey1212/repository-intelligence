#!/usr/bin/env python
"""Repository Query — ask the pinned subject repository a question,
get one evidence-backed answer.

    python agentrun_repository_query.py \\
        "How do the agents, tools, and MCP servers interact?"

What is analyzed
----------------

The SUBJECT is the ADOP repository already captured locally by M1 —
never this analyzer's own source tree, and never a fresh clone:

    repository-intelligence
            │ analyzes
            ▼
    artifacts/main_2026-09-22_ef11dac/repository
            │
            ▼
    ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

The default resolves from THIS FILE's location
(``Path(__file__).resolve().parent``), so it is machine- and
platform-independent (no hard-coded absolute path).  Before anything
runs, the checkout is VERIFIED against the expected commit — from Git
when the working tree has metadata, otherwise from M1's own snapshot
record beside it.  A mismatch (or a missing snapshot) stops the run
with the standing ERROR block and a non-zero exit: the M1
reproducibility guarantee is never silently traded for an analysis.

A thin user-facing entry point over machinery that already exists —
it composes nothing new, it only runs it and renders what happened:

    resolve + verify the pinned subject → plan (M10) →
    M17 Orchestrator: Architecture Investigator →
    Flow / Dependency Investigator → Architecture Visualization Agent
    → M16 deterministic verdict → six-rule diagram validation →
    the four artifacts (validated spec only) → one answer.

Two modes — the caller's decision, never the environment's:

offline (the default)
    fully deterministic: the evidence-driven investigation script
    reads the subject through the REAL read-only tools
    (``read_file``, ``search_repository``), bounds itself to a fixed
    number of calls, and derives every component, relationship, path,
    uncertainty and answer line from what came back.  No network, no
    key, no answer authored in advance.

--live
    the SAME composition talks to the configured provider (the
    provider-neutral ``OpenAILLMClient`` layer).  When the provider
    cannot be constructed the run stops with its own ERROR block and
    a non-zero exit: live mode never falls back to the offline
    script, silently or otherwise.  The banner always says which
    mode ran.

Flags (the whole surface):

    QUERY                  the question (positional, or --query)
    --repository PATH      subject repository to investigate
                           (default: the pinned ADOP snapshot
                           artifacts/main_2026-09-22_ef11dac/repository)
    --live                 use the configured live LLM provider
                           instead of the deterministic offline model
    --verbose              print the per-step progress lines
    --no-artifacts         write no files; the diagram is still
                           validated
    --no-diagram           the diagram is out of scope: no diagram
                           validation, no artifacts, and the answer
                           comes from the merged investigation
                           reports instead of the (unvalidated)
                           diagram finding

Interactive: run with no query and the prompt loops until ``exit``,
``quit`` or end-of-input.

Exit codes (the documented contract):

    0    the query was answered and everything in scope validated
    1    the investigation failed, a requested artifact could not be
         written, or the subject snapshot could not be verified
    2    invalid usage — an empty query, a missing/foreign flag value
    3    validation failed (the M16 collaboration verdict, or the
         six-rule diagram verdict while the diagram is in scope)
    130  interrupted (Ctrl-C)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agents import SVG_UNAVAILABLE_NOTICE  # noqa: E402
from repo_intelligence.repository_query import (  # noqa: E402
    EXIT_INVESTIGATION_FAILED,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    EXIT_USAGE,
    LIVE_CONFIGURATION_UNAVAILABLE,
    ORCHESTRATION_LANGGRAPH,
    ORCHESTRATION_MODES,
    SNAPSHOT_VERIFICATION_FAILED,
    SUBJECT_COMMIT,
    RepositoryQueryResult,
    default_subject_repository,
    exit_code,
    run_repository_query,
)

# The standing banner (the documented §6/§7 shape): the analyzer's own
# name, then a rule — never a machine-specific absolute path.
RULE = "\u2500" * 40

MARK_DONE = "\u2713"   # ✓
MARK_FAILED = "\u2717"  # ✗
MARK_SKIPPED = "\u25cb"  # ○

EXIT_FALLBACK_FAILED = EXIT_INVESTIGATION_FAILED

# step label → the role name the run report uses
RUN_LABELS = (
    ("architecture", "Architecture Investigator"),
    ("flow", "Flow / Dependency Investigator"),
    ("visualization", "Architecture Visualization"),
)


def _clip(text: object, limit: int = 72) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def _bullet(text: str) -> str:
    return "  {}".format(text)


# -- rendering -------------------------------------------------------------


def _print_header(
    question: str, result: RepositoryQueryResult, stream
) -> None:
    print("Repository Intelligence", file=stream)
    print(RULE, file=stream)
    print(file=stream)
    print("Subject repository:", file=stream)
    print(result.subject_repository or "-", file=stream)
    print(file=stream)
    print("Pinned commit:", file=stream)
    print(result.pinned_commit or "-", file=stream)
    print(file=stream)
    print("Question:", file=stream)
    print(_clip(question, 160), file=stream)
    print(file=stream)
    print(
        "Repository: {}   target: {}".format(
            result.repository or "-",
            result.target or "-",
        ),
        file=stream,
    )
    if result.mode == "live":
        print(
            "Mode:       live (real model, OPENAI_API_KEY present)",
            file=stream,
        )
    else:
        print(
            "Mode:       offline (deterministic, scripted models — "
            "no network)",
            file=stream,
        )
    print(file=stream)


def _print_live_failure(
    result: RepositoryQueryResult, stream
) -> None:
    """``--live`` without a usable provider: the standing ERROR block.
    The two lines the contract promises come first and verbatim, the
    reason follows, and no investigation is executed — live mode never
    falls back to the offline script."""
    print("Repository Intelligence", file=stream)
    print(RULE, file=stream)
    print(file=stream)
    print("ERROR: Live mode unavailable.", file=stream)
    print(file=stream)
    if result.failures:
        for line in result.failures[0].splitlines():
            print(line, file=stream)
        print(file=stream)
        if len(result.failures) > 1:
            print("Reason:", file=stream)
            print(
                "  {}".format(_clip(result.failures[1], 140)),
                file=stream,
            )
            print(file=stream)
    print("No investigation was executed.", file=stream)


def _print_verification_failure(
    result: RepositoryQueryResult, stream
) -> None:
    """The standing ERROR block (§6): a subject snapshot that cannot
    be verified is never analyzed — the block says so and stops."""
    resolved = result.subject_repository
    if not resolved:
        try:
            resolved = default_subject_repository()
        except Exception:  # pragma: no cover - defensive fallback
            resolved = "-"
    print("Repository Intelligence", file=stream)
    print(RULE, file=stream)
    print(file=stream)
    print(
        "ERROR: Subject repository snapshot verification failed.",
        file=stream,
    )
    print(file=stream)
    print("Expected commit:", file=stream)
    print("  {}".format(SUBJECT_COMMIT), file=stream)
    print(file=stream)
    print("Resolved repository:", file=stream)
    print("  {}".format(resolved), file=stream)
    print(file=stream)
    if result.failures:
        print("Reason:", file=stream)
        for failure in result.failures:
            print("  {}".format(_clip(failure, 140)), file=stream)
        print(file=stream)
    print("No investigation was executed.", file=stream)


def _print_progress(
    result: RepositoryQueryResult, stream, no_artifacts: bool = False
) -> None:
    """Post-hoc progress: every line is read back from the run's own
    records — there is no live event stream to stream from, and none
    is invented."""
    total = len(result.steps)
    for index, step in enumerate(result.steps, start=1):
        print(
            "[{}/{}] {} {} {} \u2192 {} (run {})".format(
                index,
                total,
                step.step_id,
                step.role or "-",
                step.label or "-",
                step.output_type or "no structured output",
                step.run_id or "none",
            ),
            file=stream,
        )
    collaboration = result.collaboration_status or "not reached"
    diagram = _diagram_label(result)
    print(
        "[validation] collaboration {}; diagram {}".format(
            collaboration, diagram
        ),
        file=stream,
    )
    if result.artifacts:
        print(
            "[artifacts] {} ({} file(s))".format(
                result.artifact_directory,
                len([row for row in result.artifacts if row.exists]),
            ),
            file=stream,
        )
    elif result.artifacts_error:
        print(
            "[artifacts] FAILED — {}".format(result.artifacts_error),
            file=stream,
        )
    elif no_artifacts:
        print("[artifacts] skipped (--no-artifacts)", file=stream)
    else:
        print("[artifacts] not written", file=stream)
    print(file=stream)


def _diagram_label(result: RepositoryQueryResult) -> str:
    if not result.diagram_in_scope:
        return "skipped (--no-diagram)"
    if result.diagram_status is None:
        return "not reached"
    return result.diagram_status


def _print_result(
    result: RepositoryQueryResult, stream, no_artifacts: bool = False
) -> None:
    print(
        "Status:     {}   termination: {}".format(
            result.status, result.termination_reason or "-"
        ),
        file=stream,
    )
    print(file=stream)

    print("Final answer", file=stream)
    print(RULE, file=stream)
    if result.answer:
        print(result.answer, file=stream)
        for statement in result.conclusions:
            print(_bullet("- {}".format(statement)), file=stream)
    else:
        print(
            "(no answer — the investigation did not complete)",
            file=stream,
        )
    print(file=stream)

    print("Evidence", file=stream)
    print(RULE, file=stream)
    if result.evidence:
        for line in result.evidence:
            print(_bullet("- {}".format(line)), file=stream)
    else:
        print("  (none recorded)", file=stream)
    print(file=stream)

    print("Uncertainties", file=stream)
    print(RULE, file=stream)
    if result.uncertainties:
        for item in result.uncertainties:
            print(_bullet("- {}".format(_clip(item, 100))), file=stream)
    else:
        print("  (none recorded)", file=stream)
    print(file=stream)

    _print_execution_summary(result, stream)
    _print_validation(result, stream)
    _print_artifacts(result, stream, no_artifacts)
    _print_trace(result, stream)

    if result.failures:
        print("Failures", file=stream)
        print(RULE, file=stream)
        for failure in result.failures:
            print(_bullet("- {}".format(_clip(failure, 140))), file=stream)
        print(file=stream)


def _run_name(label: str) -> str:
    for key, name in RUN_LABELS:
        if label == key:
            return name
    return label or "-"


def _print_execution_summary(
    result: RepositoryQueryResult, stream
) -> None:
    """The agent execution summary: what ran, which investigation it
    produced, and which run id belongs to which specialist."""
    heading = "Agent execution summary"
    print(heading, file=stream)
    print(RULE, file=stream)
    if result.steps:
        for step in result.steps:
            if step.status == "completed":
                mark = MARK_DONE
            elif step.status == "failed":
                mark = MARK_FAILED
            else:
                mark = MARK_SKIPPED
            print(
                "  {} {:<4} {:<13} {:<14} {}".format(
                    mark,
                    step.step_id,
                    step.role or "-",
                    step.label or "-",
                    _clip(step.goal, 54),
                ),
                file=stream,
            )
    else:
        print("  (no plan steps executed)", file=stream)
    print(file=stream)
    print(
        "Investigation ID: {}".format(result.investigation_id or "-"),
        file=stream,
    )
    print(file=stream)
    print("Runs:", file=stream)
    rows = [step for step in result.steps if step.run_id]
    if not rows:
        print("  (no agent run recorded)", file=stream)
    else:
        for step in rows:
            print(
                "  {}: {}".format(_run_name(step.label), step.run_id),
                file=stream,
            )
    print(file=stream)


def _print_validation(result: RepositoryQueryResult, stream) -> None:
    print("Validation", file=stream)
    print(RULE, file=stream)
    collaboration = result.collaboration_status
    if collaboration is None:
        print("  collaboration: not reached", file=stream)
    else:
        print(
            "  collaboration: {} (M16 deterministic verdict)".format(
                collaboration
            ),
            file=stream,
        )
    if not result.diagram_in_scope:
        print(
            "  diagram:       skipped (--no-diagram)", file=stream
        )
    elif result.diagram_status is None:
        print("  diagram:       not reached", file=stream)
    else:
        print(
            "  diagram:       {} (six deterministic rules)".format(
                result.diagram_status
            ),
            file=stream)
    print(file=stream)


def _print_artifacts(
    result: RepositoryQueryResult, stream, no_artifacts: bool = False
) -> None:
    print("Artifacts", file=stream)
    print(RULE, file=stream)
    if not result.diagram_in_scope:
        print(
            "  (skipped — --no-diagram puts the diagram out of scope)",
            file=stream,
        )
        print(file=stream)
        return
    if no_artifacts:
        print(
            "  (not written — --no-artifacts; validation still ran)",
            file=stream,
        )
        print(file=stream)
        return
    if not result.artifacts:
        if result.artifacts_error:
            print(
                "  {} write failed — {}".format(
                    MARK_FAILED, result.artifacts_error
                ),
                file=stream,
            )
        else:
            print(
                "  (none written — validation did not pass)",
                file=stream,
            )
        print(file=stream)
        return
    for row in result.artifacts:
        if row.exists:
            print(
                "  {} {}  {}".format(MARK_DONE, row.name, row.path),
                file=stream,
            )
        elif row.name == "architecture.svg":
            # Graphviz is optional: an SVG that does not exist is
            # reported as unavailable — never claimed, never required.
            notice = row.note or SVG_UNAVAILABLE_NOTICE
            if notice.startswith("NOT AVAILABLE"):
                notice = notice[len("NOT AVAILABLE") :].lstrip(" \u2014")
            print(
                "  {} {} \u2014 unavailable \u2014 {}".format(
                    MARK_SKIPPED, row.name, notice
                ),
                file=stream,
            )
        else:
            print(
                "  {} {} \u2014 not written".format(
                    MARK_FAILED, row.name
                ),
                file=stream,
            )
    print(file=stream)


def _print_trace(result: RepositoryQueryResult, stream) -> None:
    print("Trace", file=stream)
    print(RULE, file=stream)
    if not result.trace_paths:
        print("  (no trace recorded)", file=stream)
        print(file=stream)
        return
    for path in result.trace_paths:
        print("  {}".format(path), file=stream)
    print(file=stream)


# -- execution -------------------------------------------------------------


def _run_once(
    question: str,
    args: argparse.Namespace,
    progress: bool,
    stream,
) -> int:
    try:
        result = run_repository_query(
            question,
            repository=args.repository,
            mode="live" if args.live else "offline",
            diagram=not args.no_diagram,
            artifacts=not args.no_artifacts,
            orchestration=args.orchestration,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # last resort: no traceback for users
        print("error: {}: {}".format(type(exc).__name__, exc), file=stream)
        if args.verbose:
            traceback.print_exc(file=stream)
        return EXIT_FALLBACK_FAILED

    if result.termination_reason == LIVE_CONFIGURATION_UNAVAILABLE:
        # --live without a usable provider: the standing ERROR block,
        # no fallback, no investigation, non-zero exit.
        _print_live_failure(result, stream)
    elif result.termination_reason == SNAPSHOT_VERIFICATION_FAILED:
        # never analyze an unverified subject — the standing ERROR
        # block, no investigation, non-zero exit.
        _print_verification_failure(result, stream)
    else:
        _print_header(question, result, stream)
        if args.orchestration == ORCHESTRATION_LANGGRAPH:
            # non-default backend: say so (the result contract itself
            # is backend-independent, so the banner carries it)
            print(
                "Orchestration: langgraph (M21 opt-in task loop)",
                file=stream,
            )
            print(file=stream)
        if progress:
            _print_progress(result, stream, args.no_artifacts)
        _print_result(result, stream, args.no_artifacts)
    code = exit_code(result)
    print("exit code: {}".format(code), file=stream)
    return code


def _interactive(
    args: argparse.Namespace, stream
) -> int:
    print(
        "Repository Intelligence — interactive (type exit or quit "
        "to leave, end-of-input to finish)",
        file=stream,
    )
    print(file=stream)
    while True:
        try:
            line = input("query> ")
        except EOFError:
            print(file=stream)
            return EXIT_SUCCESS
        text = line.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            return EXIT_SUCCESS
        # one failed query never ends the session — the loop decides,
        # not the query
        _run_once(text, args, progress=True, stream=stream)
        print(file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python agentrun_repository_query.py",
        description=(
            "Ask the pinned subject repository one natural-language "
            "question and get one evidence-backed answer with its "
            "validation status, run ids, trace location and artifacts "
            "— offline and deterministic by default."
        ),
        epilog=(
            "Default subject: artifacts/main_2026-09-22_ef11dac/"
            "repository — M1's local ADOP snapshot, verified against "
            "commit ef11dac8618b6344d6a5e4add6ef51cd2811d7e0 before "
            "anything runs.  It is never cloned, and this project's "
            "own source tree is never the subject."
        ),
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="the question to answer",
    )
    parser.add_argument(
        "--query",
        dest="query_option",
        default=None,
        help="the question to answer (same as the positional form)",
    )
    parser.add_argument(
        "--repository",
        default=None,
        help=(
            "subject repository to investigate (default: the pinned "
            "ADOP snapshot artifacts/main_2026-09-22_ef11dac/"
            "repository)"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Use the configured live LLM provider instead of the "
            "deterministic offline model."
        ),
    )
    parser.add_argument(
        "--orchestration",
        choices=list(ORCHESTRATION_MODES),
        default=ORCHESTRATION_MODES[0],
        help=(
            "task-loop backend: 'legacy' (default, M17 Orchestrator "
            "loop) or 'langgraph' (opt-in M21 LangGraph task loop — "
            "same planner, specialists, budgets, and result "
            "contract); requires the optional 'langgraph' dependency"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the per-step progress lines of the run",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="write no files (the diagram is still validated)",
    )
    parser.add_argument(
        "--no-diagram",
        action="store_true",
        help="put the diagram out of scope: no diagram validation, "
        "no artifacts (the answer always comes from the merged "
        "investigation reports)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.query is not None and args.query_option is not None:
        print(
            "error: pass the query either as a positional argument "
            "or with --query, not both.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    question = (
        args.query if args.query is not None else args.query_option
    )
    if question is not None and not question.strip():
        print(
            "Error: repository query cannot be empty.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if args.repository is not None and not (
        Path(args.repository).expanduser().is_dir()
    ):
        print(
            "error: --repository {} is not a directory.".format(
                args.repository
            ),
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        if question is None:
            return _interactive(args, sys.stdout)
        return _run_once(
            question.strip(),
            args,
            progress=args.verbose,
            stream=sys.stdout,
        )
    except KeyboardInterrupt:
        print(file=sys.stderr)
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
