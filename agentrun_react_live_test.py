"""M8.7 — live ReAct investigation (real LLM, existing Agent chain)
+ M8.8 integration — structured claims validated after the run.

M8.6 (`agentrun_react_test.py`) proved adaptive, observation-driven
tool selection with a deterministic scripted model — fully offline,
with assertions. This script runs the SAME behavior against a real
model, demonstrated end-to-end:

    python agentrun_react_live_test.py

What it is — one composition root, one run:

    OpenAILLMClient  (real model; `Agent.run_result` performs every LLM call)
    default_tool_dispatcher()  (the real application dispatcher)
    JsonFileTraceSink()        (writes runtime/agent_runs/<run_id>/trace.json)
    AgentExecutionConfig(max_tool_calls=10, structured_claims=True)
    agent.run_result(objective)   (called exactly once)

There is NO second loop, NO orchestration layer, NO React engine, NO
second budget, and NO new evidence framework: every
Decision → Action → Observation cycle inside the run belongs to the
Agent's existing M8.4 loop. Tool actions are expected to be derived
from prior observations (the objective explicitly forbids relying
solely on the initial architecture context).

M8.7 + M8.8 end-to-end (what this runner demonstrates):

    Objective
       ↓
    LLM  →  get_architecture_context / read_file(...)   (bounded investigation)
       ↓
    LLM  →  final response = ONE explicit JSON object
            {"answer": "...", "claims": [{"statement": "...",
             "evidence_refs": [{path, line_start, line_end}]}]}
       ↓
    the loop finishes; the trace already holds every ToolResult
       ↓
    M8.8 post-run step (inside run_result, deterministic, offline):
       gather_evidence(trace) → validate_claims(claims, evidence)
       → recorded on the trace as `evidence_validation`
       ↓
    human-readable final answer + per-claim validation report

The model is NOT the authority for "was this evidence gathered?" —
the existing deterministic validator is. A failed tool result
(e.g. `read_file("tool-registry/README.md")` → `FILE_NOT_FOUND`) can
never satisfy an evidence reference; a claim citing it comes out
`UNSUPPORTED`. Validation never dispatches a tool and cannot change
the answer. The exact tool calls and claims remain model-dependent
and bounded by the single `max_tool_calls`.

The objective asks the model to investigate the agents, tools, and
MCP (Model Context Protocol) connections defined in this repository
(the ADOP / Agentic-AI Data-Operations project).

Output (concise — never raw LLM request/response payloads):

    objective
    numbered tool calls (name, exact arguments, success/failed/blocked)
    counts: iterations, tool calls (executed/blocked), termination
    trace path
    evidence validation (one block per claim: status + deterministic reason)
    final answer

Credentials/configuration:

    Requires OPENAI_API_KEY (copy .env.example to .env and set it).
    Without a key this script fails CLEARLY with exit code 6 — it never
    falls back to a fake model, and the offline test suite never runs
    this script. Failures during the run follow the existing Agent
    semantics (LLMError / AgentError → exit 6, trace still written);
    there is no M8.7 retry logic. A model that ignores the structured
    contract does not fail the run either: the raw answer stands and
    the validation block is simply empty (reported below).

M8.6 vs M8.7 (only the model source differs — identical loop,
dispatcher, tools, config, and trace schema):

    agentrun_react_test.py      M8.6 scripted model, offline, asserts
                                the adaptation (8 checks)
    agentrun_react_live_test.py M8.7 real model, live, demonstrates
                                the adaptation + M8.8 validation
                                (prints the run)
    agentrun_test.py            M8.4/M8.5 harness, retained
    agentrun_evidence_test.py   M8.8 offline validator demo (fixture)

Usage (from the project root, after copying .env.example to .env):

    python agentrun_react_live_test.py
    python agentrun_react_live_test.py --model gpt-4o-mini

Exit codes: 0 = final answer reached, 6 = failure (no key, LLMError,
AgentError).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    AgentExecutionConfig,
    AgentRunResult,
    JsonFileTraceSink,
)
from repo_intelligence.llm import LLMError  # noqa: E402
from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.llm.openai_client import (  # noqa: E402
    DEFAULT_MODEL,
    OpenAILLMClient,
)
from repo_intelligence.tool_dispatcher import (  # noqa: E402
    default_tool_dispatcher,
)

RULE = "=" * 72

# The single M8.5 tool-call limit for this run (no second budget).
MAX_TOOL_CALLS = 12

DEFAULT_OBJECTIVE = (
    "Investigate which agents, tools, and MCP (Model Context Protocol) "
    "connections are defined in this repository (the ADOP / "
    "Agentic-AI Data-Operations project). Start from the architecture "
    "context, but do not rely solely on it: use the repository tools to "
    "gather supporting evidence from files in the repository itself, and "
    "write your final answer with specific references to the files you "
    "actually inspected (citing paths from the evidence), naming the "
    "agent definitions, tool registry entries, and MCP server "
    "configuration you found."
)

EXIT_FAILURE = 6


def _print_trace_path(agent: Agent, sink: object, stream) -> None:
    """Print where this run's trace was written (file sinks only)."""
    trace = agent.last_trace
    json_path = getattr(sink, "json_path", None)
    if trace is None or not callable(json_path):
        return
    path: Path = json_path(trace.run_id)
    if path.exists():
        print("trace: {}".format(path), file=stream)


def _tool_call_lines(trace) -> List[str]:
    """Numbered tool calls from the execution trace (observable only).

    Executed calls are paired with their ToolResult by tool_call_id;
    budget-blocked calls (M8.5) are marked and never listed as run.
    """
    lines: List[str] = []
    for iteration in trace.iterations:
        blocked_ids = {
            entry.get("tool_call_id")
            for entry in (iteration.blocked_tool_calls or [])
        }
        results = {
            entry.get("tool_call_id"): entry
            for entry in (iteration.tool_results or [])
        }
        for call in (iteration.tool_calls or []):
            name = call.get("name")
            arguments = call.get("arguments") or {}
            rendered = json.dumps(arguments, sort_keys=True)
            if call.get("id") in blocked_ids:
                status = "blocked (tool-call limit, not executed)"
            else:
                result = results.get(call.get("id")) or {}
                status = "success" if result.get("success") else "failed"
            lines.append("{}. {}({}) -> {}".format(
                len(lines) + 1, name, rendered, status
            ))
    return lines


def _print_validation(result: AgentRunResult) -> None:
    """M8.8 outcome: one block per claim — status + the validator's
    deterministic reason (evidence matching, never model reasoning)."""
    print("Evidence validation (M8.8):")
    if not result.evidence_validation:
        print(
            "  (no structured claims produced — the final response did "
            "not follow the JSON contract; nothing to validate)"
        )
        return
    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNKNOWN": 0}
    for entry in result.evidence_validation:
        counts[entry.status] = counts.get(entry.status, 0) + 1
        print("  [{status}] {statement}".format(
            status=entry.status, statement=entry.claim.statement
        ))
        print("      {reason}".format(reason=entry.reason))
    print("  totals: {claims} claim(s) — {supported} supported, "
          "{unsupported} unsupported, {unknown} unknown".format(
              claims=len(result.evidence_validation),
              supported=counts["SUPPORTED"],
              unsupported=counts["UNSUPPORTED"],
              unknown=counts["UNKNOWN"],
          ))


def _warn_if_checkout_missing() -> None:
    """Advisory only: search/read need the M1 pinned checkout.

    A missing checkout is not a script failure — the tools return a
    structured failure observation and the existing Agent semantics
    decide what happens next (same as any other tool failure).
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_react_live_test.py",
        description=(
            "M8.7 live demonstration: one real-LLM ReAct investigation "
            "through the existing Agent (single run_result call, real "
            "dispatcher, read-only tools, max_tool_calls=10) with the "
            "M8.8 post-run validation of structured claims."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation objective (default: ADOP "
        "agents/tools/MCP investigation)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    args = parser.parse_args(argv)
    objective = args.question or DEFAULT_OBJECTIVE

    # configuration (explicit .env load at the composition root)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M8.7 live demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_react_live_test.py",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    # composition: concrete implementations chosen outside the Agent
    try:
        llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        print("LLMError: {}".format(exc), file=sys.stderr)
        return EXIT_FAILURE

    model_id = args.model or DEFAULT_MODEL
    sink = JsonFileTraceSink()
    agent = Agent(
        llm_client=llm_client,
        tool_dispatcher=default_tool_dispatcher(),
        trace_sink=sink,
        execution_config=AgentExecutionConfig(
            max_tool_calls=MAX_TOOL_CALLS,
            # M8.8: the final response is the explicit
            # {"answer", "claims"} contract; claims are validated
            # deterministically after the loop, against this run's
            # trace only.
            structured_claims=True,
        ),
    )

    print(RULE)
    print("M8.7 + M8.8 — live investigation with structured claims "
          "(real LLM, existing Agent)")
    print(RULE)
    print("objective: {}".format(objective))
    print("model: {}   max_tool_calls: {}".format(model_id, MAX_TOOL_CALLS))
    print()
    _warn_if_checkout_missing()

    # ONE run — every Decision → Action → Observation cycle belongs to
    # the Agent's existing M8.4 loop (no orchestration of our own);
    # the M8.8 validation runs inside run_result after that loop.
    try:
        result = agent.run_result(objective)
    except AgentError as exc:
        print("AgentError: {}".format(exc), file=sys.stderr)
        _print_trace_path(agent, sink, sys.stderr)
        return EXIT_FAILURE
    except LLMError as exc:
        print("LLMError: {}".format(exc), file=sys.stderr)
        _print_trace_path(agent, sink, sys.stderr)
        return EXIT_FAILURE
    outcome = result.answer

    trace = agent.last_trace
    summary = trace.summary
    print()
    print("Execution summary")
    print("-" * len("Execution summary"))
    if summary is not None:
        print("iterations: {}".format(summary.total_iterations))
        print(
            "tool calls: {} (executed {}, blocked {})".format(
                summary.total_tool_calls,
                summary.tool_calls_executed,
                summary.tool_calls_blocked,
            )
        )
        print("termination: {}".format(summary.termination_reason))
    if trace is not None:
        _print_trace_path(agent, sink, sys.stdout)
    print()
    tool_lines = _tool_call_lines(trace) if trace is not None else []
    if tool_lines:
        print("Tool calls:")
        for line in tool_lines:
            print("  {}".format(line))
        print()
    _print_validation(result)
    print()
    print("Final answer:")
    print(outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
