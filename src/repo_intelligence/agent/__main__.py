"""Standalone Agent command (real LLM smoke test).

Demonstrates the intended dependency chain — and nothing else:

    Agent → LLMClient → OpenAILLMClient → OpenAI
    Agent → ToolDispatcher → application tool   (M8.4 multi-step loop)
    Agent → AgentExecutionConfig                (M8.5 tool-call budget)
    Agent → AgentTrace → JsonFileTraceSink      (execution trace)

The concrete client, dispatcher, budget, and trace sink are chosen
here, at the composition root, never inside the Agent. One question =
one Agent invocation: whenever the model requests a tool, the dispatcher
executes that one registered tool and the ToolResult observation is fed
back to the model — the model decides the next step. The loop stops when
the model returns a final answer (or a structured failure occurs; the
tool-call budget is ``--max-tool-calls``, default 8 — a tool requested
after exhaustion is NOT executed and the run terminates with
``EXECUTION_LIMIT_REACHED``, exit code 6).

Every run also writes its execution trace (observable events only) to
``runtime/agent_runs/<run_id>/trace.json`` (canonical) plus a
``trace.md`` projection under the project root — execution telemetry
lives under ``runtime/``, deliberately separate from the durable
``artifacts/`` outputs; the path is printed after the answer and on
the error stream.

Usage (from the project root, after copying .env.example to .env):

    python -m repo_intelligence.agent
    python -m repo_intelligence.agent --question "What does M2 produce?"
    python -m repo_intelligence.agent --model gpt-4o-mini
    python -m repo_intelligence.agent --max-tool-calls 3

Exit codes: 0 success, 6 structured failure (LLMError / AgentError).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from ..llm import LLMError
from ..llm.config import load_env
from ..llm.openai_client import DEFAULT_MODEL, OpenAILLMClient
from ..tool_dispatcher import default_tool_dispatcher
from .agent import Agent, AgentError, AgentExecutionConfig
from .trace_sink import JsonFileTraceSink

DEFAULT_QUESTION = "What is the purpose of this repository?"

__all__ = ["DEFAULT_QUESTION", "main"]


def _non_negative_int(value: str) -> int:
    """argparse type: the tool-call budget must be an integer >= 0."""
    try:
        budget = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "must be an integer, got {!r}".format(value)
        )
    if budget < 0:
        raise argparse.ArgumentTypeError(
            "must be >= 0, got {}".format(budget)
        )
    return budget


class _AnnouncingDispatcher:
    """CLI-only wrapper around the real dispatcher.

    Records each executed loop step (tool name + success) so the CLI
    can show what the loop did between the question and the final
    answer. Execution itself is fully delegated — the real registry
    and the real tools are unchanged.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.steps: List[tuple] = []

    def dispatch(self, tool_call):
        result = self._inner.dispatch(tool_call)
        self.steps.append((result.tool or tool_call.name, result.success))
        return result


def _build_trace_sink() -> JsonFileTraceSink:
    """Composition root: persist every run's trace as JSON (canonical)
    plus a Markdown projection under
    ``runtime/agent_runs/<run_id>/trace.{json,md}`` (the sink's
    default runtime root — execution telemetry, not artifacts)."""
    return JsonFileTraceSink()


def _print_trace_path(agent: Agent, sink: object, stream) -> None:
    """Print where this run's trace was written, when a file sink has
    produced one (test/in-memory sinks print nothing)."""
    trace = agent.last_trace
    json_path = getattr(sink, "json_path", None)
    if trace is None or not callable(json_path):
        return
    path: Path = json_path(trace.run_id)
    if path.exists():
        print("Trace: {}".format(path), file=stream)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.agent",
        description=(
            "Run the Agent against the real LLM: load .env, construct "
            "OpenAILLMClient and the default tool dispatcher, inject "
            "them into the Agent, ask one question, and print the final "
            "answer reached by the multi-step tool loop (each executed "
            "tool step is listed)."
        ),
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="question to ask (default: {!r})".format(DEFAULT_QUESTION),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--max-tool-calls",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=(
            "execution budget: maximum tool executions for this run "
            "(default: {}; 0 disables tool execution)".format(
                AgentExecutionConfig().max_tool_calls
            )
        ),
    )
    args = parser.parse_args(argv)

    # configuration (explicit .env load at the composition root)
    load_env()

    # composition: concrete implementations are chosen outside the Agent
    try:
        llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        print("LLMError: {}".format(exc), file=sys.stderr)
        return 6

    execution_config = (
        AgentExecutionConfig(max_tool_calls=args.max_tool_calls)
        if args.max_tool_calls is not None
        else AgentExecutionConfig()
    )
    dispatcher = _AnnouncingDispatcher(default_tool_dispatcher())
    trace_sink = _build_trace_sink()
    agent = Agent(
        llm_client=llm_client,
        tool_dispatcher=dispatcher,
        trace_sink=trace_sink,
        execution_config=execution_config,
    )

    try:
        outcome = agent.run(args.question)
    except AgentError as exc:
        print("AgentError: {}".format(exc), file=sys.stderr)
        _print_trace_path(agent, trace_sink, sys.stderr)
        return 6
    except LLMError as exc:
        print("LLMError: {}".format(exc), file=sys.stderr)
        _print_trace_path(agent, trace_sink, sys.stderr)
        return 6

    print("Question:")
    print(args.question)
    print()

    if dispatcher.steps:
        print("Tool steps:")
        for index, (name, success) in enumerate(dispatcher.steps, start=1):
            print("  {}. {} -> success={}".format(index, name, success))
        print()

    print("Answer:")
    print(outcome)
    print()
    _print_trace_path(agent, trace_sink, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
