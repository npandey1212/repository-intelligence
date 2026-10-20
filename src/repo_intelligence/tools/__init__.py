"""Read-only agent tools — the layer above the deterministic M1–M6 pipeline.

Convention for every tool in this project:

    src/repo_intelligence/tools/<tool_name>.py
    tests/tools/test_<tool_name>.py
    docs/tools/<tool-name>.md

Every tool takes a ToolCall-style arguments mapping and returns the same
generic ``ToolResult`` (protocol in ``protocol.py``). Tools only read
existing artifacts; the future agent/LLM loop decides how to use them.

Successful results of Tools 1–3 are bounded to one serialized
``MAX_TOOL_RESPONSE_CHARS`` (24,000 characters, envelope included) with
explicit truncation metadata — ``truncated`` plus per-tool resume
fields — so no tool observation can overflow the model context by
itself (the H1/H2 request budget still owns the WHOLE request).

Import style (keeps ``python -m`` entry points warning-free — the
package does not eagerly import tool modules):

    from repo_intelligence.tools import ToolResult
    from repo_intelligence.tools.get_architecture_context import (
        get_architecture_context,
    )

Implemented:
    Tool 1 — get_architecture_context
    Tool 2 — search_repository
    Tool 3 — read_file
    Tool 4 — query_engineering_graph   (M11: EngineeringModel graph
                                        queries — node-kind summary or
                                        bounded neighborhood traversal)

Not implemented (later milestones):
    the LLM tool-selection loop (shipped in M8/M10 — the Agent plans and
    dispatches through repo_intelligence.tool_dispatcher)
"""

from __future__ import annotations

from .protocol import ToolCall, ToolError, ToolResult

__all__ = [
    "ToolCall",
    "ToolError",
    "ToolResult",
]
