"""Three independent context-overflow boundaries stay independent.

Step 1D of the bounded-tool-output work: the new per-tool response
ceiling must not be confused with — or quietly replace — the two
pre-existing boundaries, and the preflight must still own everything
the tool ceiling does not cover:

* tool ceiling — ``protocol.MAX_TOOL_RESPONSE_CHARS`` caps ONE
  serialized ``ToolResult`` in CHARACTERS (read_file,
  get_architecture_context, search_repository);
* H3 graph projection — ``DEFAULT_MODEL_GRAPH_MAX_BYTES`` caps the
  MODEL-FACING graph projection in BYTES with its own structured
  failure ``MODEL_PAYLOAD_TOO_LARGE``;
* H1/H2 request budget — ``RequestBudgetPolicy`` measures the WHOLE
  provider request in TOKENS and clips or rejects oversized
  observations before any provider call.

Pinned here: the constants and their namespaces (neither module
imports the other's knob), both existing failure modes unchanged, and
the composition — a tool observation produced under the new ceiling is
ACCEPTED by the preflight while an unbounded observation is still
reduced (protected content is still rejected outright).

Offline — no network, no LLM.
"""

from __future__ import annotations

import json

from repo_intelligence.agent.request_budget import (
    DISPOSITION_ACCEPTED,
    DISPOSITION_REJECTED,
    DISPOSITION_REDUCED,
    RequestBudgetExceededError,
    RequestBudgetPolicy,
    enforce_request_budget,
    enforce_request_budget_or_raise,
)
from repo_intelligence.agent.tool_definitions import (
    AGENT_TOOL_DEFINITIONS,
)
from repo_intelligence.llm import LLMRequest, Message, ToolCall
from repo_intelligence.tools import ToolResult
from repo_intelligence.agent import request_budget as budget_module
from repo_intelligence.tools import graph_projection as projection_module
from repo_intelligence.tools import protocol as protocol_module
from repo_intelligence.tools.graph_projection import (
    DEFAULT_MODEL_GRAPH_MAX_BYTES,
    GraphProjectionError,
    model_facing_graph_result,
    project_graph_result_for_model,
)
from repo_intelligence.tools.protocol import MAX_TOOL_RESPONSE_CHARS

COMMIT = "c" * 40
QUESTION = "How do the agents, tools, and MCP servers interact?"
START_NODE = "artifact:directory:pkg"


def _canonical_traversal(node_count: int = 140) -> dict:
    """A canonical hub traversal whose COMPACT form still exceeds the
    24,000-byte projection budget (forces deterministic truncation)."""
    nodes = []
    relationships = []
    aggregate = []
    previous_id = None
    for index in range(node_count):
        node_id = "artifact:file:pkg/mod{:03d}.py".format(index)
        entry = {
            "path": "pkg/mod{:03d}.py".format(index),
            "artifact_type": "file",
            "commit": COMMIT,
            "line_start": index + 1,
            "symbol": "run_{:03d}".format(index),
        }
        nodes.append(
            {
                "id": node_id,
                "depth": index % 6,
                "kind": "artifact",
                "subkind": "file",
                "classification": "observed",
                "attributes": {
                    "path": entry["path"],
                    "language": "python",
                },
                "evidence": [dict(entry)],
            }
        )
        row = dict(entry)
        row["origin"] = "node:{}".format(node_id)
        row["index"] = 0
        aggregate.append(row)
        if previous_id is not None:
            relationships.append(
                {
                    "id": "rel:CONTAINS:{}:{}".format(previous_id, node_id),
                    "kind": "CONTAINS",
                    "source": previous_id,
                    "target": node_id,
                    "classification": "derived",
                    "depth": index % 6,
                    "evidence": [dict(entry)],
                }
            )
        previous_id = node_id
    return {
        "mode": "traversal",
        "query": {
            "start_node_id": START_NODE,
            "direction": "out",
            "max_depth": 2,
        },
        "start_node": nodes[0],
        "nodes": nodes,
        "relationships": relationships,
        "evidence": aggregate,
        "kinds": [],
        "counts": {
            "nodes": len(nodes),
            "relationships": len(relationships),
            "evidence": len(aggregate),
            "unresolved": 0,
        },
        "truncated": False,
    }


def _request(observation: str, system: str = "system contract") -> LLMRequest:
    """One request shaped like the agent's real seam: the tool
    observation embedded exactly as the loop serializes it."""
    return LLMRequest(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=QUESTION),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="read_file",
                        arguments={"path": "a.py"},
                    )
                ],
            ),
            Message(role="tool", content=observation, tool_call_id="c1"),
        ],
        tools=list(AGENT_TOOL_DEFINITIONS),
        structured_output=None,
    )


def _observation(content: str) -> str:
    """The agent's serialization of one successful read_file result."""
    return json.dumps(
        {
            "success": True,
            "tool": "read_file",
            "data": {
                "path": "a.py",
                "line_start": 1,
                "line_end": 1,
                "content": content,
                "truncated": True,
            },
            "source": {"repository": "sample", "commit_sha": COMMIT},
            "error": None,
        }
    )


# ---------------------------------------------------------------------------
# The constants are distinct knobs with distinct namespaces
# ---------------------------------------------------------------------------


def test_the_three_ceiling_constants_stay_distinct():
    policy = RequestBudgetPolicy()

    # one serialized ToolResult — CHARACTERS
    assert MAX_TOOL_RESPONSE_CHARS == 24_000
    # the model-facing graph projection — BYTES
    assert DEFAULT_MODEL_GRAPH_MAX_BYTES == 24_000
    # the whole provider request — TOKENS (128k window minus the
    # output reservation and the safety margin)
    assert policy.context_capacity_tokens == 128_000
    assert policy.reserved_output_tokens == 16_384
    assert policy.safety_margin_tokens == 16_384
    assert policy.effective_input_budget_tokens == 95_232

    # neither boundary module imports the other's knob
    assert "MAX_TOOL_RESPONSE_CHARS" not in vars(projection_module)
    assert "DEFAULT_MODEL_GRAPH_MAX_BYTES" not in vars(protocol_module)
    assert "MAX_TOOL_RESPONSE_CHARS" not in vars(budget_module)


# ---------------------------------------------------------------------------
# H3 still owns the graph projection (byte limit + typed failure)
# ---------------------------------------------------------------------------


def test_graph_projection_keeps_its_own_byte_limit():
    payload = _canonical_traversal()
    assert len(json.dumps(payload, sort_keys=True)) > (
        DEFAULT_MODEL_GRAPH_MAX_BYTES
    )

    projected = project_graph_result_for_model(payload)
    assert len(json.dumps(projected, sort_keys=True)) <= (
        DEFAULT_MODEL_GRAPH_MAX_BYTES
    )
    assert projected["payload"]["truncated"] is True
    assert projected["payload"]["max_bytes"] == DEFAULT_MODEL_GRAPH_MAX_BYTES
    # the canonical contract passes through untouched
    assert projected["mode"] == "traversal"
    assert projected["query"] == payload["query"]


def test_graph_projection_failure_mode_is_unchanged():
    payload = _canonical_traversal()

    try:
        project_graph_result_for_model(payload, max_bytes=64)
    except GraphProjectionError as exc:
        assert exc.code == "MODEL_PAYLOAD_TOO_LARGE"
    else:
        raise AssertionError("expected GraphProjectionError")

    result = ToolResult(
        success=True,
        tool="query_engineering_graph",
        data=payload,
        source={"artifact": "engineering_graph.json"},
        error=None,
    )
    failed = model_facing_graph_result(result, max_bytes=64)
    assert failed.success is False
    assert failed.data is None
    assert failed.source is None
    assert failed.error.code == "MODEL_PAYLOAD_TOO_LARGE"


# ---------------------------------------------------------------------------
# H1/H2 still own everything the tool ceiling does not cover
# ---------------------------------------------------------------------------


def test_observation_under_the_tool_ceiling_is_accepted():
    """The largest observation the NEW tool ceiling can produce is far
    inside the request budget: the two boundaries compose without the
    preflight ever having to touch a tool observation."""
    observation = _observation("z" * (MAX_TOOL_RESPONSE_CHARS - 500))
    assert len(observation) <= MAX_TOOL_RESPONSE_CHARS

    request = _request(observation)
    final, record = enforce_request_budget(request, RequestBudgetPolicy())

    assert record["disposition"] == DISPOSITION_ACCEPTED
    assert final is request


def test_unbounded_observation_is_still_reduced_before_provider():
    observation = _observation("z" * 500_000)
    request = _request(observation)

    final, record = enforce_request_budget(request, RequestBudgetPolicy())

    assert record["disposition"] == DISPOSITION_REDUCED
    clipped = json.loads(final.messages[-1].content)
    assert clipped["truncated"] is True
    assert clipped["tool"] == "read_file"
    assert clipped["original_characters"] == len(observation)
    assert len(clipped["excerpt"]) < len(observation)
    assert len(final.messages[-1].content) < len(observation)
    # the REDUCED request passes the same preflight it just failed
    settled, _ = enforce_request_budget(final, RequestBudgetPolicy())
    assert settled is final


def test_protected_content_over_the_budget_is_still_rejected():
    observation = _observation("z" * 1000)
    request = _request(observation, system="s" * 500_000)

    final, record = enforce_request_budget(request, RequestBudgetPolicy())
    assert record["disposition"] == DISPOSITION_REJECTED
    # rejected requests are returned UNCHANGED — never a silent trim
    assert final is request

    try:
        enforce_request_budget_or_raise(request, subject="test request")
    except RequestBudgetExceededError as exc:
        assert "test request" in str(exc)
        assert exc.record["disposition"] == DISPOSITION_REJECTED
    else:
        raise AssertionError("expected RequestBudgetExceededError")


def test_the_tool_ceiling_introduced_no_new_tool_arguments():
    """The ceiling is a RESPONSE concern: read_file keeps its
    three-argument input contract (no max_chars-style argument)."""
    from repo_intelligence.tools.read_file import _ALLOWED_ARGUMENTS

    assert _ALLOWED_ARGUMENTS == ("path", "line_start", "line_end")
