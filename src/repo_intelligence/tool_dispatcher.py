"""Application-owned tool dispatcher — the M8.3 execution boundary.

One LLM-generated ``ToolCall`` in, one generic ``ToolResult`` out, then
STOP:

    ToolCall → ToolDispatcher → registered application tool → ToolResult

Ownership / security boundary:

* The registry is explicit and application-owned. A tool name coming
  from the LLM resolves only against this registry — never via dynamic
  import, ``eval``, reflection, or any other arbitrary function
  execution. Unknown names become a structured failure.
* ``default_tool_dispatcher()`` registers exactly the four strictly
  read-only tools (``get_architecture_context``, ``search_repository``,
  ``read_file``, ``query_engineering_graph``); nothing registered here
  can execute shell commands, mutate the repository, or write files.
* Arguments are handed to the registered tool, which applies its
  existing validation contract — the dispatcher does not duplicate
  tool-specific validation. Dispatcher-level guards cover malformed
  tool calls (missing name, non-mapping arguments) and turn them into
  structured ``ToolResult`` failures.
* Unexpected exceptions from a tool are converted into a structured
  failure — no stack trace crosses the Agent/tool boundary.

The dispatcher never calls the LLM, never plans, never retries, never
loops, and never summarizes: executing one tool and returning its
result is the entire job (the M8.4 loop lives above it, in the Agent).

Composition: the Agent depends on the structural contract in
``repo_intelligence.agent.dispatch`` and receives an instance by
dependency injection; composition roots (``agent/__main__.py``, tests)
build the concrete dispatcher here.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Tuple

from .llm import ToolCall
from .tools import ToolError, ToolResult
from .tools.get_architecture_context import get_architecture_context
from .tools.graph_projection import model_facing_graph_tool
from .tools.read_file import read_file
from .tools.search_repository import search_repository

# A registered tool: takes its arguments mapping, returns the generic
# ToolResult (tools validate their own input contracts themselves).
ToolFunction = Callable[..., ToolResult]

__all__ = ["ToolDispatcher", "default_tool_dispatcher", "tool_registry"]


class ToolDispatcher:
    """Resolves a ToolCall name against the explicit registry and
    executes exactly one registered tool per ``dispatch``."""

    def __init__(self, registry: Optional[Mapping[str, ToolFunction]] = None):
        """``registry`` maps tool name → callable; ``None`` starts empty
        (register explicitly — nothing is discovered dynamically)."""
        self._registry: Dict[str, ToolFunction] = dict(registry or {})

    @property
    def tool_names(self) -> Tuple[str, ...]:
        """Currently registered tool names (sorted) — the only names
        that can ever execute."""
        return tuple(sorted(self._registry))

    def register(self, name: str, tool: ToolFunction) -> None:
        """Explicitly register one application tool under ``name``."""
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        self._registry[name] = tool

    def dispatch(self, tool_call: ToolCall) -> ToolResult:
        """Execute exactly one ``ToolCall``.

        Expected failures (unknown tool, malformed arguments, tool
        errors, unexpected exceptions) are returned as structured
        ``ToolResult`` failures — this method does not raise for them
        and never executes anything outside the registry.
        """
        name = getattr(tool_call, "name", None)
        if not isinstance(name, str) or not name:
            return ToolResult(
                success=False,
                tool="",
                error=ToolError(
                    code="INVALID_ARGUMENT",
                    message="tool call must carry a non-empty tool name.",
                ),
            )

        tool = self._registry.get(name)
        if tool is None:
            registered = ", ".join(self.tool_names) or "(none)"
            return ToolResult(
                success=False,
                tool=name,
                error=ToolError(
                    code="TOOL_NOT_FOUND",
                    message=(
                        "tool {!r} is not registered; registered tools: {}."
                    ).format(name, registered),
                ),
            )

        arguments = getattr(tool_call, "arguments", None)
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return ToolResult(
                success=False,
                tool=name,
                error=ToolError(
                    code="INVALID_ARGUMENT",
                    message="arguments must be a mapping.",
                ),
            )

        try:
            result = tool(arguments)
        except Exception as exc:  # unexpected — no stack trace crosses the boundary
            return ToolResult(
                success=False,
                tool=name,
                error=ToolError(
                    code="TOOL_FAILED",
                    message="tool {!r} failed during execution: {}: {}.".format(
                        name, type(exc).__name__, exc
                    ),
                ),
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                success=False,
                tool=name,
                error=ToolError(
                    code="TOOL_FAILED",
                    message="tool {!r} did not return a ToolResult.".format(name),
                ),
            )
        return result


def tool_registry() -> Dict[str, ToolFunction]:
    """The M8.3 default registry as a fresh mapping (name → callable).

    Exposed so a composition root can build a BOUNDED subset — e.g. one
    registry per agent role (M16 permissions) — from the same single
    source of truth ``default_tool_dispatcher`` uses. No tool is ever
    discovered dynamically; this returns exactly the four read-only
    tools, with ``query_engineering_graph`` bound to its model-facing
    projection wrapper (``model_graph_compact.v1``) so graph
    observations stay inside the request budget.
    """
    return {
        "get_architecture_context": get_architecture_context,
        "search_repository": search_repository,
        "read_file": read_file,
        "query_engineering_graph": model_facing_graph_tool,
    }


def default_tool_dispatcher() -> ToolDispatcher:
    """The M8.3 registry: exactly the four read-only tools.

    The tools receive only their argument mapping — application context
    (the pinned repository snapshot, the artifact output root) is
    resolved by the tools themselves, never supplied by the LLM.
    """
    return ToolDispatcher(tool_registry())
