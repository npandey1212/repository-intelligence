#!/usr/bin/env python
"""
Small learning demo for the existing Repository Intelligence planner.

Run from the repository root:
    python planner_request_demo.py --objective "Explain how agents and MCP servers interact"

Inspect the exact request without spending an LLM call:
    python planner_request_demo.py --objective "..." 

Call the model and print the typed plan:
    python planner_request_demo.py --objective "..." --execute

Optionally provide context files:
    python planner_request_demo.py --objective "..." \
        --architecture-context-file path/to/architecture_context.md \
        --memory-context-file path/to/memory_context.md \
        --execute

This script deliberately reuses the project's planner request builder,
response decoder, and deterministic plan validator. It does not execute
the plan and does not invoke any repository tools.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allows running this script from the repository root without installing
# the package in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.planning import (  # noqa: E402
    InvestigationPlan,
    PlanValidationError,
    build_planning_request,
    plan_from_response,
    plan_to_dict,
    validate_plan,
)
from repo_intelligence.planning.planner import PLANNING_SYSTEM_INSTRUCTION  # noqa: E402


DEFAULT_OBJECTIVE = (
    "Explain how the agents, tools, and MCP servers interact in the repository."
)


DEFAULT_ARCHITECTURE_CONTEXT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "main_2026-09-22_ef11dac"
    / "architecture_context"
    / "architecture_context.md"
)


def read_optional_file(path: Optional[str], label: str, default_path: Optional[Path] = None) -> Optional[str]:
    """Read an explicitly supplied file, or a default artifact when configured."""
    file_path = Path(path) if path is not None else default_path
    if file_path is None:
        return None
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.is_file():
        if path is None:
            raise FileNotFoundError(
                f"Default {label} artifact not found: {file_path}\\n"
                "Check that you are running this script from the repository root "
                "and that the generated artifact exists, or pass an explicit file path."
            )
        raise FileNotFoundError(f"{label} file not found: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"{label} file is empty: {file_path}")
    print(f"Loaded {label}: {file_path} ({len(content):,} characters)")
    return content


def print_request(request) -> None:
    """Print the provider-neutral LLMRequest returned by the builder."""
    print("\n" + "=" * 78)
    print("1. WHAT build_planning_request() RETURNS")
    print("=" * 78)
    print(f"Python type: {type(request).__name__}")
    print(f"Number of messages: {len(request.messages)}")
    print(f"Tools available to planner: {len(request.tools)}")
    print(f"Requested model override: {request.model!r}")
    print(f"Temperature: {request.temperature!r}")

    for index, message in enumerate(request.messages, start=1):
        print(f"\n--- Message {index}: role={message.role!r} ---")
        print(message.content or "")

    print("\n--- Structured output contract ---")
    if request.structured_output is None:
        print("No structured-output schema requested.")
    else:
        print(f"Schema name: {request.structured_output.name}")
        print(json.dumps(request.structured_output.schema, indent=2, ensure_ascii=False))


def print_plan(plan: InvestigationPlan) -> None:
    """Print the typed plan that plan_from_response() returns."""
    print("\n" + "=" * 78)
    print("3. TYPED PLAN RETURNED BY plan_from_response()")
    print("=" * 78)
    print(f"Python type: {type(plan).__name__}")
    print("\nPlan JSON:")
    print(json.dumps(plan_to_dict(plan), indent=2, ensure_ascii=False))

    print("\nExecution order (the list order is sequential):")
    for index, step in enumerate(plan.steps, start=1):
        print(f"  {index}. {step.id}: {step.goal} [status={step.status}]")

    print("\nRole assignment:")
    print("  This plan schema has no agent/role field.")
    print("  In the M17 workflow, role assignment happens later in orchestration.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the existing planner request and optionally call the LLM."
    )
    parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help="Goal to give the planner.",
    )
    parser.add_argument(
        "--architecture-context-file",
        help=(
            "UTF-8 architecture-context file. Defaults to "
            "artifacts/main_2026-09-22_ef11dac/architecture_context/architecture_context.md."
        ),
    )
    parser.add_argument(
        "--memory-context-file",
        help="Optional UTF-8 text file whose contents become memory_context.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the configured LLM, decode the plan, and validate it.",
    )
    parser.add_argument(
        "--model",
        help="Optional OpenAI model override (otherwise the project's default is used).",
    )
    args = parser.parse_args()

    architecture_context = read_optional_file(
        args.architecture_context_file,
        "architecture context",
        default_path=DEFAULT_ARCHITECTURE_CONTEXT_FILE,
    )
    memory_context = read_optional_file(args.memory_context_file, "Memory context")

    # This is the function the user wants to understand.
    request = build_planning_request(
        objective=args.objective,
        architecture_context=architecture_context,
        memory_context=memory_context,
    )
    print_request(request)

    if not args.execute:
        print("\nNo LLM call was made. Re-run with --execute to generate a plan.")
        return 0

    print("\n" + "=" * 78)
    print("2. CALL THE LLM AND DECODE ITS RESPONSE")
    print("=" * 78)

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Configure .env or the environment.")
        return 2

    try:
        from repo_intelligence.llm.openai_client import (  # imported only for live mode
            DEFAULT_MODEL,
            OpenAILLMClient,
        )

        client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
        response = client.complete(request)
        print(f"LLM finish reason: {response.finish_reason!r}")
        print(f"LLM usage: {response.usage!r}")

        plan = plan_from_response(response)
        print_plan(plan)

        print("\n" + "=" * 78)
        print("4. DETERMINISTIC APPLICATION-LEVEL PLAN VALIDATION")
        print("=" * 78)
        validate_plan(plan)
        print("VALIDATION PASSED: the plan is eligible for execution.")
        print("No agent was invoked and no plan step was executed by this demo.")
        return 0

    except PlanValidationError as exc:
        print(f"PLAN VALIDATION FAILED: {exc}")
        print("The plan was not executed.")
        return 3
    except Exception as exc:
        print(f"Planner demo failed: {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
