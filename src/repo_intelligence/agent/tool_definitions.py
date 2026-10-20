"""Agent-visible tool definitions (M8.2) — descriptions for the LLM.

What this module is: provider-neutral ``ToolDefinition`` values
describing the four existing read-only application tools, sent with
every Agent request so the model knows which capabilities exist.

What it is NOT: an implementation, registry, or dispatcher. The
executable tools live in the separate application tools package; the
Agent package never imports them (enforced by tests). The
schemas mirror the documented tool contracts (docs/tools/*.md), and
tests/agent/test_tool_definitions.py pins them to the tools' actual
validation contracts so descriptions cannot silently drift from
behavior.

M8.2 boundary: definitions only. A returned ToolCall is recognized and
reported — never executed; execution arrives in the next milestone.
"""

from __future__ import annotations

from typing import List

from ..llm import ToolDefinition

GET_ARCHITECTURE_CONTEXT_TOOL = ToolDefinition(
    name="get_architecture_context",
    description=(
        "Retrieve the already-generated high-level architecture context "
        "for the repository. The default response is a compact heading "
        "index (the document's real section headings plus a retrieval "
        "hint) — pass section to receive one heading block's body. "
        "Responses are bounded by a response ceiling (truncated "
        "responses say so and list the available sections)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Optional Markdown heading title to return instead "
                    "of the default heading index (e.g. 'Architecture', "
                    "'Technology'); matched case-insensitively against "
                    "the document's heading titles, exact match only."
                ),
            },
            "max_chars": {
                "type": "integer",
                "minimum": 1000,
                "maximum": 24000,
                "default": 24000,
                "description": (
                    "Lower the response ceiling for this call "
                    "(1000-24000); it can never be raised."
                ),
            },
        },
        "additionalProperties": False,
    },
)

SEARCH_REPOSITORY_TOOL = ToolDefinition(
    name="search_repository",
    description=(
        "Search the pinned repository snapshot for relevant evidence "
        "(semantic retrieval over the repository's prebuilt index; "
        "read-only, never builds an index)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Free-text question or keyword query; ranked by "
                    "semantic similarity against the indexed "
                    "repository chunks."
                ),
            },
            "path": {
                "type": ["string", "null"],
                "description": (
                    "Optional repository-relative file or directory "
                    "filter; null means no filter."
                ),
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
                "description": "Maximum matches to return (1-50).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description=(
        "Retrieve exact text evidence from a file of the pinned "
        "repository snapshot."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Non-empty repository-relative file path.",
            },
            "line_start": {
                "type": "integer",
                "minimum": 1,
                "description": "First line (1-based, inclusive).",
            },
            "line_end": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Last line (1-based, inclusive); must be >= "
                    "line_start."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)

# M11: the engineering graph query. All arguments are optional — an
# empty call runs the node-kind summary (discovery mode); the enum
# values are pinned to the tool/query contracts by tests (this module
# stays free of engineering/tools imports).
QUERY_ENGINEERING_GRAPH_TOOL = ToolDefinition(
    name="query_engineering_graph",
    description=(
        "Query the repository's engineering graph (structured "
        "relationships: CONTAINS, IMPORTS, DEPENDS_ON, TRIGGERS, "
        "DECLARES, EXPOSED_BY, CONFIGURED_BY) with evidence. Call "
        "with {} for the node-kind summary (discover node kinds and "
        "ids), or provide a start_node_id to traverse its bounded "
        "neighborhood — e.g. DECLARES → EXPOSED_BY → CONFIGURED_BY "
        "walks an agent's declared tool/server/configuration path."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "start_node_id": {
                "type": ["string", "null"],
                "minLength": 1,
                "description": (
                    "Node id to traverse from; omit or null for the "
                    "node-kind summary."
                ),
            },
            "direction": {
                "type": "string",
                "enum": ["out", "in", "both"],
                "default": "out",
                "description": "Edge direction for the traversal.",
            },
            "relationship_kinds": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": [
                        "CONTAINS",
                        "IMPORTS",
                        "DEPENDS_ON",
                        "TRIGGERS",
                        "DECLARES",
                        "EXPOSED_BY",
                        "CONFIGURED_BY",
                    ],
                },
                "description": (
                    "Only follow/return these relationship kinds; null "
                    "means all."
                ),
            },
            "node_kinds": {
                "type": ["array", "null"],
                "items": {"type": "string", "minLength": 1},
                "description": (
                    "Only visit nodes matching 'kind' or 'kind/subkind' "
                    "(the start node is always included); null means all."
                ),
            },
            "max_depth": {
                "type": "integer",
                "minimum": 0,
                "maximum": 6,
                "default": 1,
                "description": (
                    "Traversal depth in hops (0 = start node only)."
                ),
            },
        },
        "additionalProperties": False,
    },
)

# Exactly the tools the Agent offers to the LLM — no others.
AGENT_TOOL_DEFINITIONS: List[ToolDefinition] = [
    GET_ARCHITECTURE_CONTEXT_TOOL,
    SEARCH_REPOSITORY_TOOL,
    READ_FILE_TOOL,
    QUERY_ENGINEERING_GRAPH_TOOL,
]

__all__ = [
    "AGENT_TOOL_DEFINITIONS",
    "GET_ARCHITECTURE_CONTEXT_TOOL",
    "QUERY_ENGINEERING_GRAPH_TOOL",
    "READ_FILE_TOOL",
    "SEARCH_REPOSITORY_TOOL",
]
