"""Adapter: the project's read-only Investigator tools as real
``langchain`` tools for ``create_agent``.

The project tools keep their EXISTING call shape and safety policy —
one mapping of arguments in, one ``ToolResult`` out (the same
protocol the production Agent's dispatcher uses, including every
per-tool response ceiling and error code). This module only adapts
the CALLING convention: a langchain tool call's parsed arguments are
delivered as the single ``arguments`` mapping, and the ``ToolResult``
comes back as its deterministic JSON string (the exact serialization
the production conversation seam embeds and the request budget
measures). Nothing is re-implemented: the bound functions are the
real tool functions (or the already-bound callables from
``repository_query.live_bindings``).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping, Optional

from langchain_core.tools import StructuredTool

from ..agent.tool_definitions import AGENT_TOOL_DEFINITIONS

__all__ = [
    "investigator_tool_definitions",
    "to_langchain_tool",
]


def investigator_tool_definitions(
    names: Optional[list] = None,
) -> Dict[str, Any]:
    """The existing ``ToolDefinition`` records (name, description,
    JSON schema) for the Investigator's tools — the same definitions
    the production Agent sends with each request."""
    selected = set(names) if names is not None else None
    return {
        definition.name: definition
        for definition in AGENT_TOOL_DEFINITIONS
        if selected is None or definition.name in selected
    }


def to_langchain_tool(
    name: str,
    func: Callable[[Mapping[str, Any]], Any],
    *,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
) -> StructuredTool:
    """Wrap ONE already-bound project tool as a ``StructuredTool``.

    *func* keeps the project signature — a single ``arguments``
    mapping (the bound live/fixture callable, never a re-implementation).
    The tool's result is the ``ToolResult``'s own JSON serialization;
    :class:`langgraph_poc.middleware.ObservationBoundingMiddleware`
    bounds that text before it joins the agent's history.
    """

    def run(**kwargs: Any) -> str:
        result = func(dict(kwargs))
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            return json.dumps(to_dict(), sort_keys=True, default=str)
        if isinstance(result, str):
            return result
        return json.dumps(result, sort_keys=True, default=str)

    schema = dict(input_schema or {"type": "object", "properties": {}})
    return StructuredTool(
        name=name,
        description=description or "project tool {}".format(name),
        func=run,
        args_schema=schema,
    )
