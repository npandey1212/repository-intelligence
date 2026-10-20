"""Generic read-only tool protocol: ToolCall / ToolResult.

One shared shape for every agent-accessible tool in this project:

    ToolCall   = { tool_name, arguments }
    ToolResult = { success, tool, data, source, error }

A tool fills ``data``/``source`` (or ``error``) and returns; a future
agent/LLM runtime decides what to do with the result. No tool returns a
tool-specific result type — ``ToolResult`` stays tool-agnostic so later
tools (search_repository, read_file, ...) reuse it unchanged.

No reasoning lives here: this is only the transport contract between the
application/tool runtime and a future agent loop.

Bounded tool output (this module's second responsibility)
---------------------------------------------------------

An agent observation is ``json.dumps(result.to_dict())`` — that exact
string is what the model pays for — so the transport contract also owns
the response ceiling every text-bearing tool enforces:

    MAX_TOOL_RESPONSE_CHARS   ceiling on ONE serialized ToolResult
    response_chars(result)    exact serialized size of one result
    bound_result_text(...)    largest line-aligned prefix that fits

Tools bound their output AT THE SOURCE; the request-budget preflight in
``agent/request_budget`` stays the final safety boundary in front of
every provider call (H1/H2). The ceiling deliberately matches the order
of magnitude of the model-facing graph projection limit
(``tools/graph_projection.DEFAULT_MODEL_GRAPH_MAX_BYTES``), but the two
are separate constants with separate failure modes and are never
coupled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Ceiling on one serialized ToolResult: no text-bearing tool may return
# a success payload bigger than this. Measured on json.dumps of the
# whole result, because that is the observation string the agent sends.
MAX_TOOL_RESPONSE_CHARS = 24_000

# Characters reserved for the truncation note while a tool computes its
# bound: the note is part of the payload, so its worst-case length is
# reserved up front and the ceiling stays a guarantee rather than an
# estimate. Real notes are plain ASCII without quotes or backslashes
# (pinned by tests) and never use more than this.
TRUNCATION_NOTE_RESERVE = 400


@dataclass(frozen=True)
class ToolCall:
    """A request to execute one tool (name + arguments)."""

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolError:
    """Structured tool failure — never a stack trace."""

    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ToolResult:
    """Standard result every tool returns.

    data:    tool-specific payload (dict), None on failure
    source:  provenance describing where the payload came from, None on
             failure
    error:   structured error, None on success
    """

    success: bool
    tool: str
    data: Optional[Dict[str, Any]] = None
    source: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready representation used by CLIs and future runtimes."""
        return {
            "success": self.success,
            "tool": self.tool,
            "data": self.data,
            "source": self.source,
            "error": self.error.to_dict() if self.error is not None else None,
        }


# ---------------------------------------------------------------------------
# Bounded output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundedText:
    """What survived a response ceiling.

    ``text``         the returned text (whole lines whenever any line
                     fits, a prefix of the first line otherwise);
    ``whole_lines``  how many leading whole lines ``text`` contains;
    ``partial_line`` ``text`` ends inside the first line — only ever
                     true when a single line cannot fit at all.
    """

    text: str
    whole_lines: int
    partial_line: bool


def response_chars(result: ToolResult) -> int:
    """Serialized size of a ToolResult, in characters.

    This is exactly the observation string an agent run embeds
    (``json.dumps(result.to_dict())``), so it is the honest measure of
    what one tool call costs the request budget.
    """
    return len(json.dumps(result.to_dict()))


def bound_result_text(
    result: ToolResult, text_key: str, limit: int
) -> BoundedText:
    """Largest line-aligned prefix of ``result.data[text_key]`` whose
    SERIALIZED result still fits in ``limit`` characters.

    The envelope (``success``/``tool``/``source``/``error`` and the
    other ``data`` keys) is measured first; only what remains is offered
    to the text, so a bound returned here guarantees
    ``response_chars(...) <= limit`` for the rebound result.

    Whole lines are preferred — a caller can report the exact line range
    it delivered and continue from ``line_end + 1``. A character prefix
    of the first line is returned only when not even one line fits, and
    is flagged with ``partial_line`` so the caller can say so instead of
    implying the line was read in full.

    Deterministic: same result, same limit → same text.
    """
    if not result.data:
        return BoundedText(text="", whole_lines=0, partial_line=False)

    # Measure against the SAME serialization the caller pays for
    # (``response_chars`` → ``result.to_dict()``) with the text emptied,
    # so a bound returned here guarantees ``response_chars(...) <=
    # limit`` for the rebound result — the envelope (``success``/
    # ``tool``/``source``/``error`` and the other ``data`` keys) counts
    # against the ceiling too. ``to_dict()`` shares ``data``, so the
    # copy has to happen before anything is emptied.
    emptied = result.to_dict()
    emptied["data"] = dict(result.data)
    emptied["data"][text_key] = ""
    # json.dumps of a payload whose text field is "" counts two quote
    # characters the text itself will re-supply.
    overhead = len(json.dumps(emptied)) - len(json.dumps(""))
    room = limit - overhead

    lines: Sequence[str] = str(result.data.get(text_key) or "").split("\n")
    kept: List[str] = []
    used = 0
    for line in lines:
        # Escaping is per character, so the serialized length of a
        # "\n"-joined text is the SUM of the serialized lengths of its
        # lines (each line contributes its own quotes, the separator
        # contributes its own two escaped characters — they cancel).
        cost = len(json.dumps(line))
        if used + cost > room:
            break
        kept.append(line)
        used += cost
    if kept:
        return BoundedText(text="\n".join(kept), whole_lines=len(kept),
                           partial_line=False)

    # No line fits: prefix the first line to the exact room.
    low, high, best = 0, len(lines[0]), ""
    while low <= high:
        mid = (low + high) // 2
        candidate = lines[0][:mid]
        if len(json.dumps(candidate)) <= room:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return BoundedText(text=best, whole_lines=0, partial_line=True)


__all__ = [
    "BoundedText",
    "MAX_TOOL_RESPONSE_CHARS",
    "TRUNCATION_NOTE_RESERVE",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "bound_result_text",
    "response_chars",
]
