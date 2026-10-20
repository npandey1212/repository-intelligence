"""Standalone demonstration of the LLM abstraction (no Agent).

Proves the boundary works independently of any agent/tool loop:

    1. Load configuration            (load_env → .env)
    2. Construct OpenAILLMClient     (composition root chooses provider)
    3. Build an LLMRequest           (provider-neutral models)
    4. Preflight the request         (local context-budget gate)
    5. Call complete()               (model inference only)
    6. Print the normalized LLMResponse

Run after setting OPENAI_API_KEY in .env:

    python -m repo_intelligence.llm
    python -m repo_intelligence.llm --tools   # let the model request a
                                              # sample tool (displayed,
                                              # never executed)

Exit codes: 0 success, 6 structured failure (LLMError, or an
over-budget request rejected by the local context-budget preflight).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ..agent import (
    RequestBudgetExceededError,
    enforce_request_budget_or_raise,
)
from .client import LLMError
from .config import load_env
from .models import LLMRequest, LLMResponse, Message, ToolDefinition
from .openai_client import DEFAULT_MODEL, OpenAILLMClient

# Sample tool shown to the model when --tools is given. The purpose is
# purely to demonstrate that a model-requested tool call is *returned*
# in normalized form — nothing executes it here.
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

DEFAULT_PROMPT = "Say hello in one short sentence."


def _build_request(args: argparse.Namespace) -> LLMRequest:
    messages: List[Message] = [
        Message(
            role="system",
            content="You are a concise assistant. Answer plainly.",
        ),
        Message(role="user", content=args.prompt),
    ]
    tools = [SAMPLE_TOOL] if args.tools else []
    return LLMRequest(
        messages=messages,
        tools=tools,
        model=args.model,
        temperature=args.temperature,
    )


def _print_response(response: LLMResponse) -> None:
    print("LLM Response")
    print("------------")
    print("Finish reason: {}".format(response.finish_reason))
    print()
    print("Message:")
    if response.message.content is not None:
        print(response.message.content)
    else:
        print("(no text content)")
    print()
    if response.tool_calls:
        print("Tool calls (requested, not executed):")
        for call in response.tool_calls:
            print(
                "  - {} {} [id={}]".format(
                    call.name, json.dumps(call.arguments), call.id
                )
            )
        print()
    print("Usage:")
    if response.usage is not None:
        print("Input tokens: {}".format(response.usage.input_tokens))
        print("Output tokens: {}".format(response.usage.output_tokens))
        print("Total tokens: {}".format(response.usage.total_tokens))
    else:
        print("(not provided)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.llm",
        description=(
            "Standalone LLM abstraction test: send one request through "
            "the provider-neutral LLMClient and print the normalized "
            "response. No Agent, no tool execution."
        ),
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="user prompt (default: hello)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="sampling temperature"
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="include a sample tool definition so the model may request "
        "it (displayed only — never executed)",
    )
    args = parser.parse_args(argv)

    # 1. configuration (explicit .env load at the composition root)
    load_env()

    # 2. composition: choose the concrete implementation here
    try:
        client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        print("LLMError: {}".format(exc), file=sys.stderr)
        return 6

    # 3. provider-neutral request
    request = _build_request(args)

    # 4. preflight: never send an over-budget request to the provider
    try:
        request, _budget_record = enforce_request_budget_or_raise(
            request, subject="llm module request"
        )
    except RequestBudgetExceededError as exc:
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 6

    # 5. model inference only
    try:
        response = client.complete(request)
    except LLMError as exc:
        print("LLMError: {}".format(exc), file=sys.stderr)
        return 6

    # 6. normalized response
    _print_response(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
