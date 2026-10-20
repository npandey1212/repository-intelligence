"""M8.3 tool execution boundary — the contract the Agent depends on.

The Agent receives a dispatcher by dependency injection and knows
nothing about the executable tools themselves:

    Agent ── LLMClient ────────► LLM provider     (decides WHAT to call)
    Agent ── ToolDispatcher ───► application      (executes exactly one
                                                   ToolCall → ToolResult)

This module is only the structural contract: a runtime-checkable
``Protocol`` mirroring the style of ``repo_intelligence.llm.LLMClient``.
The concrete application-owned implementation lives in
``repo_intelligence.tool_dispatcher`` and is wired by composition roots
(``agent/__main__.py``, tests) — the Agent package never imports the
executable tools (enforced by tests).

M8.3 boundary: one ``dispatch`` call executes exactly one tool and
returns its generic ``ToolResult``. The dispatcher itself never sends
the result back to the LLM — the M8.4 loop in ``Agent.run`` appends
it to the conversation as a tool observation.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import Protocol, runtime_checkable

from ..llm import ToolCall

__all__ = ["ToolDispatcher"]


@runtime_checkable
class ToolDispatcher(Protocol):
    """One-shot application tool execution contract.

    Implementations execute exactly one ``ToolCall`` and return the
    generic ``ToolResult``. They never call the LLM, never plan, never
    retry, never loop, and never modify the repository.
    """

    def dispatch(self, tool_call: ToolCall) -> Any:
        ...
