"""Standalone LLM connectivity test (M7 abstraction layer).

Checks, in order, that the LLM layer can actually reach a model:

    1. Configuration      OPENAI_API_KEY present (.env or environment)
    2. Client construction  OpenAILLMClient builds without LLMError
    3. Chat completion    a real round trip through LLMClient.complete()
    4. Tool-call round trip  the model can request a tool (informational)

Steps 1-3 are required; step 4 reports PASS/WARN/SKIP and never changes
the exit code. No Agent, no tool execution, no test framework needed.

Usage (from the project root):

    python llm_test.py                      # real API connectivity
    python llm_test.py --model gpt-4o-mini
    python llm_test.py --offline            # no network: fake provider,
                                            # validates script + layer
    python llm_test.py --prompt "Say ok"

Exit codes: 0 = all required checks passed, 6 = failure (structured).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.llm import (  # noqa: E402
    LLMError,
    LLMRequest,
    Message,
    ToolDefinition,
)
from repo_intelligence.llm.config import get_openai_api_key, load_env  # noqa: E402
from repo_intelligence.llm.openai_client import DEFAULT_MODEL, OpenAILLMClient  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 6
TOTAL_STEPS = 4
REQUIRED_STEPS = 3
NAME_WIDTH = 34
RULE = "-" * 70

DEFAULT_PROMPT = "Reply with exactly: connectivity ok"

SAMPLE_TOOL = ToolDefinition(
    name="search_repository",
    description=(
        "Search repository text for relevant evidence. Read-only; "
        "returns matching file locations."
    ),
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


class _OfflineProvider:
    """Fake provider client for --offline mode (no network).

    Mirrors the SDK shape (chat.completions.create) and returns a text
    response normally, or a tool-call response when tools were offered —
    so both round trips exercise the real translation code.
    """

    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        if kwargs.get("tools"):
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_offline_1",
                        function=SimpleNamespace(
                            name="search_repository",
                            arguments=json.dumps({"query": "deployment docs"}),
                        ),
                    )
                ],
            )
            choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        else:
            message = SimpleNamespace(content="connectivity ok", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=12, completion_tokens=6, total_tokens=18
        )
        return SimpleNamespace(choices=[choice], usage=usage)


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return "{}...{}".format(value[:3], value[-4:])


def _row(number: int, name: str, status: str, detail: str = "") -> None:
    dots = "." * max(3, NAME_WIDTH - len(name))
    line = "[{}/{}] {} {}  {}".format(number, TOTAL_STEPS, name, dots, status)
    if detail:
        line += "  " + detail
    print(line)


def _usage_detail(response) -> str:
    usage = response.usage
    if usage is None:
        return ""
    parts = []
    if usage.input_tokens is not None:
        parts.append("in={}".format(usage.input_tokens))
    if usage.output_tokens is not None:
        parts.append("out={}".format(usage.output_tokens))
    return " | " + " ".join(parts) if parts else ""


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


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    mode = "offline (fake provider, no network)" if args.offline else "online (real API)"
    print("LLM connectivity test")
    print("=====================")
    print("mode: {}".format(mode))
    print("model: {}".format(args.model))
    print()

    passed = 0

    # -- 1. configuration ---------------------------------------------------
    load_env()
    if args.offline:
        _row(1, "Configuration", "PASS", "offline mode - key not required")
        passed += 1
    else:
        key = get_openai_api_key()
        if key:
            _row(1, "Configuration", "PASS", "OPENAI_API_KEY={}".format(_mask(key)))
            passed += 1
        else:
            _row(
                1,
                "Configuration",
                "FAIL",
                "OPENAI_API_KEY not set - copy .env.example to .env and add your key",
            )
            _row(2, "Client construction", "SKIP")
            _row(3, "Chat completion round trip", "SKIP")
            _row(4, "Tool-call round trip", "SKIP")
            return _finish(passed, started)

    # -- 2. client construction --------------------------------------------
    try:
        if args.offline:
            client = OpenAILLMClient(
                model=args.model, provider_client=_OfflineProvider()
            )
        else:
            client = OpenAILLMClient(model=args.model)
        _row(2, "Client construction", "PASS", "model={}".format(args.model))
        passed += 1
    except LLMError as exc:
        _row(2, "Client construction", "FAIL", str(exc))
        _row(3, "Chat completion round trip", "SKIP")
        _row(4, "Tool-call round trip", "SKIP")
        return _finish(passed, started)

    # -- 3. chat completion round trip ---------------------------------------
    try:
        before = time.monotonic()
        response = client.complete(
            LLMRequest(messages=[Message(role="user", content=args.prompt)])
        )
        latency = time.monotonic() - before
        content = response.message.content
        if not content:
            _row(3, "Chat completion round trip", "FAIL", "empty response content")
            _row(4, "Tool-call round trip", "SKIP")
            return _finish(passed, started)
        _row(
            3,
            "Chat completion round trip",
            "PASS",
            "{:.2f}s | finish={}{} | {!r}".format(
                latency,
                response.finish_reason,
                _usage_detail(response),
                content if len(content) <= 60 else content[:57] + "...",
            ),
        )
        passed += 1
    except LLMError as exc:
        _row(3, "Chat completion round trip", "FAIL", str(exc))
        _row(4, "Tool-call round trip", "SKIP")
        return _finish(passed, started)

    # -- 4. tool-call round trip (informational) ------------------------------
    try:
        tool_response = client.complete(
            LLMRequest(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "You must call the provided search tool before "
                            "answering. Never answer without calling it."
                        ),
                    ),
                    Message(
                        role="user",
                        content="Locate documentation about deployment.",
                    ),
                ],
                tools=[SAMPLE_TOOL],
            )
        )
        if tool_response.tool_calls:
            call = tool_response.tool_calls[0]
            _row(
                4,
                "Tool-call round trip",
                "PASS",
                "model requested '{}' with {}".format(
                    call.name, json.dumps(call.arguments)
                ),
            )
        else:
            _row(
                4,
                "Tool-call round trip",
                "SKIP",
                "model answered without a tool call (connectivity already proven)",
            )
    except LLMError as exc:
        _row(4, "Tool-call round trip", "WARN", str(exc))

    return _finish(passed, started)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python llm_test.py",
        description=(
            "Standalone LLM connectivity test: configuration, client "
            "construction, one chat round trip, and an optional "
            "tool-call round trip. No Agent, no tool execution."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="prompt for the chat round trip (default: {!r})".format(DEFAULT_PROMPT),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no network: run the same checks against an in-process fake "
        "provider (validates the script and abstraction layer)",
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
