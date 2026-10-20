"""Standalone Agent tool-capability test (M8.4 multi-step loop).

Checks, without pytest and without any network, that the Agent can use
its four read-only tools through the application-owned tool dispatcher
and feed the ToolResult observations back to the LLM (the M8.4 loop):

    1. Dispatcher construction   default registry = exactly the
                                 4 read-only tools, structural contract
    2. Observation reaches LLM   ToolCall → dispatch → ToolResult →
                                 conversation → 2nd LLM call → answer
    3. Exactly-once dispatch     one dispatch per tool step, tool
                                 definitions on every request
    4. Unknown tool fails safely LLM-requested unknown name →
                                 TOOL_NOT_FOUND observed, registry intact
    5. Text answer path          normal answers still return str,
                                 no dispatch happens
    6. ReAct loop (offline)      full cycle trace printed by the real
                                 Agent loop: every request → response →
                                 observation until a final answer

Steps 1-6 are required and run offline: a fake LLMClient supplies
canned responses while the Agent loop, the dispatcher, and the tools
are the real ones. Step 6 prints every request/response cycle of the
production M8.4 loop (tracing wrappers observe the real Agent — the
script runs no loop of its own).

Optional:

    --live    run the real M8.4 Agent loop against the real model
              (needs OPENAI_API_KEY): every request-response cycle and
              every tool observation is printed as it happens, until
              the model gives a final answer or the M8.5 execution
              budget (AgentExecutionConfig.max_tool_calls) is reached.
              Informational only — never changes the exit code.

Every cycle is printed in full:

    --- request        (full message history sent to the LLM)
    --- response       (finish_reason, latency, content, tool calls)
    --- observation    (the ToolResult the dispatcher executed)

Telemetry (automatic): every Agent run this script performs is
recorded as a structured execution trace and persisted under
runtime/agent_runs/<run_id>/ (trace.json = canonical,
trace.md = projection) through the observability MVP's
JsonFileTraceSink — runtime execution telemetry, deliberately kept
outside artifacts/. After the steps, a telemetry summary lists each
recorded run — iterations, tool calls, termination reason, and the
trace path. Tracing is best-effort: a trace-write failure never
changes the script's checks or exit code.

Usage (from the project root):

    python agentrun_test.py
    python agentrun_test.py --live --model gpt-4o-mini \
        --question "tell me about onboarding"
    python agentrun_test.py --live --full

Options:
    --full     print cycles untruncated (default clips very long
               blocks at 1200 characters)

Exit codes: 0 = all required checks passed, 6 = failure (structured).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    AgentExecutionConfig,
    AgentTrace,
    JsonFileTraceSink,
    ToolDispatcher,
)
from repo_intelligence.llm import (  # noqa: E402
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
)
from repo_intelligence.llm.config import get_openai_api_key, load_env  # noqa: E402
from repo_intelligence.llm.openai_client import DEFAULT_MODEL, OpenAILLMClient  # noqa: E402
from repo_intelligence.tool_dispatcher import default_tool_dispatcher  # noqa: E402
from repo_intelligence.tools import ToolResult  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 6
TOTAL_STEPS = 6
REQUIRED_STEPS = 6
NAME_WIDTH = 42
RULE = "-" * 70

REGISTERED_TOOLS = (
    "get_architecture_context",
    "query_engineering_graph",
    "read_file",
    "search_repository",
)

DEFAULT_QUESTION = "What is the purpose of this repository?"
DEFAULT_CLIP = 1200  # characters per printed block; --full disables


class _FakeLLMClient:
    """Canned ``LLMClient`` for offline steps (no network).

    Returns the queued responses in order and records every request,
    so a step can prove how often the LLM was actually invoked and
    what conversation it received.
    """

    def __init__(self, responses: List[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: List[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(
                "_FakeLLMClient: complete() called with no queued response "
                "(more LLM cycles than this step allows)"
            )
        return self._responses.pop(0)


class _CountingDispatcher:
    """Wraps a real dispatcher and counts dispatch() calls (exactly-once)."""

    def __init__(self, inner: ToolDispatcher) -> None:
        self._inner = inner
        self.calls = 0

    def dispatch(self, tool_call: ToolCall) -> ToolResult:
        self.calls += 1
        return self._inner.dispatch(tool_call)


class _TracingLLMClient:
    """Wraps an LLMClient and prints every request/response cycle the
    real Agent produces (observations appear via the tracing
    dispatcher)."""

    def __init__(self, inner, limit: Optional[int]) -> None:
        self._inner = inner
        self._limit = limit
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        tag = "LLM call {}".format(self.calls)
        _print_request(tag, request.messages, request, self._limit)
        started = time.monotonic()
        response = self._inner.complete(request)
        latency = time.monotonic() - started
        _print_response(tag, response, latency, self._limit)
        return response


class _TracingDispatcher:
    """Wraps the real dispatcher and prints each ToolResult observation
    as the Agent loop produces it."""

    def __init__(self, inner: ToolDispatcher, limit: Optional[int]) -> None:
        self._inner = inner
        self._limit = limit
        self.dispatches = 0

    def dispatch(self, tool_call: ToolCall) -> ToolResult:
        self.dispatches += 1
        result = self._inner.dispatch(tool_call)
        _print_observation(self.dispatches, tool_call, result, self._limit)
        return result


class _Telemetry:
    """Script-wide execution-trace telemetry (observability MVP).

    Behaves as the Agent's ``TraceSink``: every ``Agent.run``
    performed with this object is persisted to
    ``<root>/<run_id>/trace.json`` (+ ``trace.md``) by the wrapped
    ``JsonFileTraceSink`` (default root: project
    ``runtime/agent_runs/``), and the same traces are kept
    in memory so the script can print a summary. One telemetry object
    is created per invocation — tracking is automatic, never
    hand-started per step.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self._sink = JsonFileTraceSink(root)
        self.records: List[Tuple[str, AgentTrace]] = []

    # -- TraceSink protocol (Agent.run calls this on every run) ----------

    def write(self, trace: AgentTrace) -> None:
        self._sink.write(trace)

    # -- bookkeeping -------------------------------------------------------

    def capture(self, agent: Agent, label: str) -> None:
        """Remember the trace the agent just produced, under *label*."""
        if agent.last_trace is not None:
            self.records.append((label, agent.last_trace))

    def json_path(self, run_id: str) -> Path:
        return self._sink.json_path(run_id)

    @property
    def root(self) -> Path:
        return self._sink.root

    # -- reporting ---------------------------------------------------------

    def print_summary(self) -> None:
        print(RULE)
        print("Telemetry — Agent execution traces (automatic):")
        if not self.records:
            print("  (no Agent runs recorded)")
            return
        print("  {} run(s) written: {}/<run_id>/trace.json".format(
            len(self.records), _display_path(self.root)
        ))
        for index, (label, trace) in enumerate(self.records, start=1):
            summary = trace.summary
            if summary is None:
                print("  [{}] {}  run={}  (unfinalized)".format(
                    index, label, trace.run_id
                ))
                continue
            print(
                "  [{}] {}  run={}  iterations={}  tool_calls={}  "
                "termination={}".format(
                    index,
                    label,
                    trace.run_id,
                    summary.total_iterations,
                    summary.total_tool_calls,
                    summary.termination_reason,
                )
            )
            if trace.sink_error is not None:
                print("      sink_error: {}".format(trace.sink_error))
            else:
                print("      {}".format(
                    _display_path(self.json_path(trace.run_id))
                ))


def _display_path(path: Path) -> str:
    """Project-relative path when possible (shorter, portable output)."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _run_with_telemetry(
    telemetry: _Telemetry, agent: Agent, question: str, label: str
) -> str:
    """Run one step's question and capture its execution trace.

    Capture happens in ``finally`` so even a failing/raising run is
    recorded (the Agent finalizes its trace before propagating) — and
    the trace write itself can never change the step's outcome.
    """
    try:
        return agent.run(question)
    finally:
        telemetry.capture(agent, label)


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _row(number: str, name: str, status: str, detail: str = "") -> None:
    dots = "." * max(3, NAME_WIDTH - len(name))
    line = "[{}/{}] {} {}  {}".format(number, TOTAL_STEPS, name, dots, status)
    if detail:
        line += "  " + detail
    print(line)


def _finish(passed: int, started: float) -> int:
    elapsed = time.monotonic() - started
    print(RULE)
    if passed >= REQUIRED_STEPS:
        print("Result: PASS - {}/{} required checks ({:.2f}s)".format(
            passed, REQUIRED_STEPS, elapsed
        ))
        return EXIT_OK
    print("Result: FAIL - {}/{} required checks ({:.2f}s)".format(
        passed, REQUIRED_STEPS, elapsed
    ))
    return EXIT_FAIL


# ---------------------------------------------------------------------------
# Cycle printing (every request/response/observation of the real loop)
# ---------------------------------------------------------------------------


def _clip(text: Optional[str], limit: Optional[int]) -> str:
    if text is None:
        return ""
    if limit is None or len(text) <= limit:
        return text
    return "{}\n... [{} chars truncated — rerun with --full]".format(
        text[:limit], len(text) - limit
    )


def _print_block(indent: str, text: Optional[str], limit: Optional[int]) -> None:
    clipped = _clip(text, limit)
    if clipped == "":
        print(indent + "(none)")
        return
    for line in clipped.splitlines():
        print(indent + line)


def _print_request(tag: str, messages: List[Message], request: LLMRequest,
                   limit: Optional[int]) -> None:
    print("--- request ({})".format(tag))
    for message in messages:
        head = "[{}]".format(message.role)
        if message.tool_call_id is not None:
            head += " (tool_call_id={})".format(message.tool_call_id)
        text = _clip(message.content, limit)
        if message.tool_calls:
            if text:
                print("  {} content:".format(head))
                for line in text.splitlines():
                    print("    " + line)
            for call in message.tool_calls:
                print("  {} tool_call: {}({}) id={}".format(
                    head,
                    call.name,
                    json.dumps(call.arguments),
                    call.id,
                ))
            continue
        if not text:
            print("  {} (none)".format(head))
        elif "\n" in text:
            print("  {} content:".format(head))
            for line in text.splitlines():
                print("    " + line)
        else:
            print("  {} {}".format(head, text))
    tool_names = ", ".join(tool.name for tool in request.tools)
    print("  tools offered: {}".format(tool_names))


def _print_response(tag: str, response: LLMResponse, latency: float,
                    limit: Optional[int]) -> None:
    print("--- response ({}, finish_reason={}, {:.2f}s)".format(
        tag, response.finish_reason, latency
    ))
    if response.message.content:
        print("  [assistant] content:")
        _print_block("    ", response.message.content, limit)
    for index, call in enumerate(response.tool_calls, start=1):
        print("  [assistant] tool_call #{}: {}({}) id={}".format(
            index,
            call.name,
            json.dumps(call.arguments),
            call.id,
        ))
    if not response.message.content and not response.tool_calls:
        print("  [assistant] (empty: no content, no tool calls)")


def _print_observation(index: int, call: ToolCall, result: ToolResult,
                       limit: Optional[int]) -> None:
    print("--- observation (dispatch #{} → {})".format(index, call.name))
    _print_block("  ", json.dumps(result.to_dict(), indent=2), limit)
    print()


# ---------------------------------------------------------------------------
# Required offline steps
# ---------------------------------------------------------------------------


def _step1(_telemetry: _Telemetry) -> bool:
    dispatcher = default_tool_dispatcher()
    names = tuple(dispatcher.tool_names)
    contract = isinstance(dispatcher, ToolDispatcher)
    registry_ok = names == REGISTERED_TOOLS
    if contract and registry_ok:
        _row(
            "1",
            "Dispatcher construction",
            "PASS",
            "registry: {}".format(", ".join(names)),
        )
        return True
    _row(
        "1",
        "Dispatcher construction",
        "FAIL",
        "contract={} registry={}".format(contract, names),
    )
    return False


def _step2(telemetry: _Telemetry) -> bool:
    # M8.4: the ToolResult must become an observation on the NEXT
    # request — so a tool step costs exactly one extra LLM round trip.
    fake = _FakeLLMClient(
        [
            _tool_response(
                ToolCall(
                    id="call_offline_1",
                    name="get_architecture_context",
                    arguments={},
                )
            ),
            _text_response("Architecture context was observed."),
        ]
    )
    agent = Agent(
        fake, default_tool_dispatcher(), trace_sink=telemetry
    )
    outcome = _run_with_telemetry(
        telemetry, agent, "Describe the architecture.",
        "step 2 — observation reaches the LLM",
    )
    observation = ""
    if len(fake.requests) == 2:
        tool_messages = [m for m in fake.requests[1].messages if m.role == "tool"]
        if tool_messages:
            observation = tool_messages[0].content
    ok = (
        isinstance(outcome, str)
        and len(fake.requests) == 2
        and '"tool": "get_architecture_context"' in observation
        and '"success": true' in observation
    )
    if ok:
        _row(
            "2",
            "Observation reaches the LLM",
            "PASS",
            "ToolResult observed by the LLM (2 LLM calls)",
        )
        return True
    _row(
        "2",
        "Observation reaches the LLM",
        "FAIL",
        "outcome={} llm_calls={} observed={}".format(
            type(outcome).__name__, len(fake.requests), bool(observation)
        ),
    )
    return False


def _step3(telemetry: _Telemetry) -> bool:
    fake = _FakeLLMClient(
        [
            _tool_response(
                ToolCall(
                    id="call_offline_2",
                    name="get_architecture_context",
                    arguments={},
                )
            ),
            _text_response("done"),
        ]
    )
    counting = _CountingDispatcher(default_tool_dispatcher())
    agent = Agent(fake, counting, trace_sink=telemetry)
    outcome = _run_with_telemetry(
        telemetry, agent, "Describe the architecture.",
        "step 3 — exactly-once dispatch per step",
    )
    ok = (
        isinstance(outcome, str)
        and len(fake.requests) == 2
        and len(fake.requests[0].tools) == len(REGISTERED_TOOLS)
        and counting.calls == 1
    )
    if ok:
        _row(
            "3",
            "Exactly-once dispatch per step",
            "PASS",
            "2 LLM calls, 1 dispatch, {} tool definitions".format(
                len(fake.requests[0].tools)
            ),
        )
        return True
    _row(
        "3",
        "Exactly-once dispatch per step",
        "FAIL",
        "llm_calls={} dispatches={} tools={}".format(
            len(fake.requests), counting.calls,
            len(fake.requests[0].tools) if fake.requests else 0,
        ),
    )
    return False


def _step4(telemetry: _Telemetry) -> bool:
    dispatcher = default_tool_dispatcher()
    registry_before = tuple(dispatcher.tool_names)
    fake = _FakeLLMClient(
        [
            _tool_response(
                ToolCall(
                    id="call_offline_3",
                    name="eval",
                    arguments={"code": "__import__('os')"},
                )
            ),
            _text_response("That tool does not exist here."),
        ]
    )
    agent = Agent(fake, dispatcher, trace_sink=telemetry)
    outcome = _run_with_telemetry(
        telemetry, agent, "Run this code.",
        "step 4 — unknown tool fails safely",
    )
    observation = ""
    if len(fake.requests) == 2:
        tool_messages = [m for m in fake.requests[1].messages if m.role == "tool"]
        if tool_messages:
            observation = tool_messages[0].content
    ok = (
        isinstance(outcome, str)
        and '"code": "TOOL_NOT_FOUND"' in observation
        and tuple(dispatcher.tool_names) == registry_before
    )
    if ok:
        _row(
            "4",
            "Unknown tool fails safely",
            "PASS",
            "code=TOOL_NOT_FOUND observed, registry unchanged",
        )
        return True
    _row(
        "4",
        "Unknown tool fails safely",
        "FAIL",
        "outcome={} observed={!r}".format(
            type(outcome).__name__, observation[:60]
        ),
    )
    return False


def _step5(telemetry: _Telemetry) -> bool:
    fake = _FakeLLMClient([_text_response("This repository is intelligence tooling.")])
    counting = _CountingDispatcher(default_tool_dispatcher())
    agent = Agent(fake, counting, trace_sink=telemetry)
    outcome = _run_with_telemetry(
        telemetry, agent, "Answer without tools.",
        "step 5 — text answer path unchanged",
    )
    ok = (
        isinstance(outcome, str)
        and bool(outcome)
        and counting.calls == 0
        and len(fake.requests) == 1
    )
    if ok:
        _row(
            "5",
            "Text answer path unchanged",
            "PASS",
            "returns str, 0 dispatches",
        )
        return True
    _row(
        "5",
        "Text answer path unchanged",
        "FAIL",
        "type={} dispatches={}".format(type(outcome).__name__, counting.calls),
    )
    return False


def _step6(
    question: str, limit: Optional[int], telemetry: _Telemetry
) -> bool:
    """Full offline ReAct loop trace driven by the REAL M8.4 Agent:
    canned tool request → real dispatch → observation fed back →
    canned final answer. All cycles are printed by the tracing
    wrappers (the script runs no loop of its own); the run itself is
    also recorded as a persisted execution trace (telemetry)."""
    print(RULE)
    print("ReAct loop trace (offline: fake LLM, REAL Agent M8.4 loop)")
    print("question: {}".format(question))
    print(
        "execution budget (max_tool_calls)={}".format(
            AgentExecutionConfig().max_tool_calls
        )
    )
    fake = _FakeLLMClient(
        [
            _tool_response(
                ToolCall(
                    id="call_react_1",
                    name="get_architecture_context",
                    arguments={},
                )
            ),
            _text_response(
                "The repository intelligence platform persists pinned "
                "evidence artifacts; architecture context is available "
                "via the get_architecture_context tool (offline canned "
                "final answer after one observation)."
            ),
        ]
    )
    tracer = _TracingLLMClient(fake, limit)
    dispatcher = _TracingDispatcher(default_tool_dispatcher(), limit)
    agent = Agent(tracer, dispatcher, trace_sink=telemetry)
    try:
        outcome = _run_with_telemetry(
            telemetry, agent, question, "step 6 — ReAct loop (offline)"
        )
    except AssertionError as exc:
        _row("6", "ReAct loop (trace above)", "FAIL", str(exc)[:80])
        return False
    ok = (
        isinstance(outcome, str)
        and tracer.calls == 2
        and dispatcher.dispatches == 1
    )
    if ok:
        _row(
            "6",
            "ReAct loop (trace above)",
            "PASS",
            "{} LLM calls, {} dispatch → final answer".format(
                tracer.calls, dispatcher.dispatches
            ),
        )
        return True
    _row(
        "6",
        "ReAct loop (trace above)",
        "FAIL",
        "outcome={} llm_calls={} dispatches={}".format(
            type(outcome).__name__, tracer.calls, dispatcher.dispatches
        ),
    )
    return False


# ---------------------------------------------------------------------------
# Optional live loop (informational — never changes the exit code)
# ---------------------------------------------------------------------------


def _live(
    args: argparse.Namespace, limit: Optional[int], telemetry: _Telemetry
) -> None:
    load_env()
    if not get_openai_api_key():
        print("[live] ReAct loop (trace above)  SKIP  "
              "OPENAI_API_KEY not set")
        return

    print(RULE)
    print("ReAct loop trace (live: {}, REAL Agent M8.4 loop)".format(args.model))
    print("question: {}".format(args.question))
    print(
        "execution budget (max_tool_calls)={}".format(
            AgentExecutionConfig().max_tool_calls
        )
    )
    tracer = None
    dispatcher = None
    try:
        tracer = _TracingLLMClient(OpenAILLMClient(model=args.model), limit)
        dispatcher = _TracingDispatcher(default_tool_dispatcher(), limit)
        agent = Agent(tracer, dispatcher, trace_sink=telemetry)
        _run_with_telemetry(
            telemetry, agent, args.question,
            "live — ReAct loop ({})".format(args.model),
        )
    except (LLMError, AgentError) as exc:
        print("[live] ReAct loop (trace above)  WARN  {}".format(
            str(exc)[:80]
        ))
        return
    except KeyboardInterrupt:
        print("[live] ReAct loop (trace above)  WARN  interrupted")
        return
    print("[live] ReAct loop (trace above)  PASS  "
          "{} LLM calls, {} dispatch(es), final answer returned".format(
              tracer.calls, dispatcher.dispatches
          ))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    mode = "offline (fake LLM, REAL Agent loop + dispatcher/tools)"
    if args.live:
        mode += " + live loop"
    # Automatic telemetry: one sink per invocation; every Agent run in
    # this script is traced and persisted without further wiring.
    telemetry = _Telemetry()
    print("Agent tool-capability test (M8.4)")
    print("=================================")
    print("mode: {}".format(mode))
    print("model: {}".format(args.model))
    print("telemetry: execution traces → {}/<run_id>/trace.json".format(
        _display_path(telemetry.root)
    ))
    print()

    limit = None if args.full else DEFAULT_CLIP
    steps = (
        lambda: _step1(telemetry),
        lambda: _step2(telemetry),
        lambda: _step3(telemetry),
        lambda: _step4(telemetry),
        lambda: _step5(telemetry),
        lambda: _step6(args.question, limit, telemetry),
    )
    names = (
        "Dispatcher construction",
        "Observation reaches the LLM",
        "Exactly-once dispatch per step",
        "Unknown tool fails safely",
        "Text answer path unchanged",
        "ReAct loop (trace above)",
    )
    passed = 0
    skipped = False
    for index, step in enumerate(steps, start=1):
        if skipped:
            _row(str(index), names[index - 1], "SKIP")
            continue
        if step():
            passed += 1
        else:
            skipped = True

    if args.live:
        print()
        _live(args, limit, telemetry)

    telemetry.print_summary()
    return _finish(passed, started)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_test.py",
        description=(
            "Standalone Agent tool-capability test (M8.4): dispatcher "
            "registry, ToolResult observations reaching the LLM, "
            "exactly-once dispatch, unknown-tool safety, the text "
            "answer path, and a fully printed ReAct loop driven by the "
            "real Agent — offline by default, with an optional live "
            "loop via --live."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="run the real M8.4 Agent loop against the real model, "
        "printing every cycle (requires OPENAI_API_KEY; never changes "
        "the exit code)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model id for --live (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="question for the loop (default: {!r})".format(DEFAULT_QUESTION),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="print cycles untruncated (default clips blocks at {} "
        "characters)".format(DEFAULT_CLIP),
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
