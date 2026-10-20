"""M13 — evidence-backed conclusions in the Agent (spec §18 agent layer).

The structured-output contract may carry a ``claims`` array; the
Agent extracts it verbatim after schema validation and runs the
existing deterministic validator — **inspected-only** (§4/§6):

 * final structured result includes claims (typed M8.9 payload)
 * valid claims survive finalization (SUPPORTED, recorded on trace)
 * invalid claims produce a deterministic structured failure
   (UNSUPPORTED + ``CLAIM_EVIDENCE_INVALID``, never an invented pass)
 * search-only discovery does not become inspected evidence at the
   Agent level either
 * a typed contract WITHOUT claims is byte-for-byte pre-M13 behavior
 * no hidden-reasoning fields anywhere in the recorded payloads

Frozen M8–M12 tests (``tests/agent/test_agent_structured_output.py``,
``tests/evidence/*``) stay green: those paths are untouched here.
"""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel

from repo_intelligence.agent import (
    Agent,
    AgentExecutionConfig,
    render_markdown,
)
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.evidence import (
    CLAIM_EVIDENCE_INVALID,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
)
from repo_intelligence.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    ToolCall,
)
from repo_intelligence.tools import ToolResult

QUESTION = "Which files define the dispatcher?"

BANNED_EXTRA = ("reasoning", "reasoning_trace")


# -- claim-bearing structured contract (M13 §3 shape) ----------------------


class ClaimRef(BaseModel):
    path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class ClaimItem(BaseModel):
    statement: str
    evidence_refs: List[ClaimRef]


class ClaimedSummary(BaseModel):
    finding: str
    claims: List[ClaimItem]


class PlainSummary(BaseModel):
    """Pre-M13 contract: no claims field at all."""

    finding: str


READ_PAYLOAD = {
    "finding": "module.py defines the dispatcher.",
    "claims": [
        {
            "statement": "module.py defines the dispatcher.",
            "evidence_refs": [
                {"path": "module.py", "line_start": 1, "line_end": 20}
            ],
        }
    ],
}


# -- per-file test doubles (no conftest in this repo) ----------------------


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


class FakeDispatcher:
    """Canned results per call; default = successful read of module.py."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(
            success=True,
            tool=tool_call.name,
            data={
                "path": "module.py",
                "line_start": 1,
                "line_end": 20,
                "content": "# dispatcher",
                "total_lines": 100,
            },
        )


def _structured_response(payload, content=None):
    if content is None:
        content = json.dumps(payload)
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls_response(*calls):
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def _call(name="read_file", arguments=None, call_id="c1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _read_result():
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": "module.py",
            "line_start": 1,
            "line_end": 20,
            "content": "# dispatcher",
            "total_lines": 100,
        },
    )


def _search_result(path):
    return ToolResult(
        success=True,
        tool="search_repository",
        data={
            "query": "dispatcher",
            "results": [
                {
                    "path": path,
                    "line_start": 1,
                    "line_end": 5,
                    "excerpt": "dispatcher",
                }
            ],
        },
    )


def _run(payload, model=ClaimedSummary, results=None):
    fake = FakeLLMClient(
        _tool_calls_response(_call()),
        _structured_response(payload),
    )
    dispatcher = FakeDispatcher(
        *(results or [_read_result()])
    )
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        structured_output_model=model,
    )
    result = agent.run_result(QUESTION)
    return result, agent, fake, dispatcher


# ---------------------------------------------------------------------------
# 1 — final structured result includes claims; valid claims survive
# ---------------------------------------------------------------------------


def test_typed_payload_with_claims_yields_claims_and_validation():
    result, agent, fake, dispatcher = _run(READ_PAYLOAD)

    typed = result.structured_output
    assert isinstance(typed, ClaimedSummary)
    assert typed.finding == READ_PAYLOAD["finding"]
    assert len(result.claims) == 1
    assert result.claims[0].statement == (
        "module.py defines the dispatcher."
    )

    assert len(result.evidence_validation) == 1
    entry = result.evidence_validation[0]
    assert entry.status == STATUS_SUPPORTED
    assert entry.code is None
    assert entry.matched_evidence[0].tool == "read_file"

    # recorded on the ONE trace before persistence (§19 inspectable)
    block = agent.last_trace.to_dict()["evidence_validation"]
    assert len(block["claims"]) == 1
    assert block["results"][0]["status"] == STATUS_SUPPORTED
    # validation dispatched nothing — still exactly one tool call
    assert [call.id for call in dispatcher.calls] == ["c1"]
    # the request carried the claim-bearing spec on every iteration
    spec = fake.requests[0].structured_output
    assert isinstance(spec, StructuredOutputSpec)
    assert spec.name == "ClaimedSummary"


def test_typed_payload_without_claims_is_unchanged():
    payload = {"finding": "plain pre-M13 contract"}
    result, agent, fake, dispatcher = _run(payload, model=PlainSummary)

    assert result.claims == ()
    assert result.evidence_validation == ()
    assert "evidence_validation" not in agent.last_trace.to_dict()
    assert isinstance(result.structured_output, PlainSummary)


# ---------------------------------------------------------------------------
# 2 — invalid claims produce a deterministic structured failure (§6/§18)
# ---------------------------------------------------------------------------


def test_uninspected_claim_records_claim_evidence_invalid():
    payload = {
        "finding": "A file the run never opened says X.",
        "claims": [
            {
                "statement": "other.py says X.",
                "evidence_refs": [{"path": "other.py"}],
            }
        ],
    }
    result, agent, fake, dispatcher = _run(payload)

    entry = result.evidence_validation[0]
    assert entry.status == STATUS_UNSUPPORTED
    assert entry.code == CLAIM_EVIDENCE_INVALID
    assert "never gathered" in entry.reason

    # the structured failure is recorded on the trace and rendered
    block = agent.last_trace.to_dict()["evidence_validation"]
    assert block["results"][0]["code"] == CLAIM_EVIDENCE_INVALID
    markdown = render_markdown(agent.last_trace)
    assert "code: CLAIM_EVIDENCE_INVALID" in markdown


def test_search_discovery_alone_does_not_support_a_claim():
    payload = {
        "finding": "search found docs/x.md.",
        "claims": [
            {
                "statement": "docs/x.md documents dispatch.",
                "evidence_refs": [{"path": "docs/x.md"}],
            }
        ],
    }
    result, agent, fake, dispatcher = _run(
        payload, results=[_search_result("docs/x.md")]
    )

    entry = result.evidence_validation[0]
    assert entry.status == STATUS_UNSUPPORTED
    assert entry.code == CLAIM_EVIDENCE_INVALID
    assert "never inspected" in entry.reason
    # the file WAS discovered — gathered, just not eligible (§4)
    assert entry.matched_evidence == ()


def test_mixed_claims_validate_independently_and_in_order():
    payload = {
        "finding": "One inspected, one invented.",
        "claims": [
            {
                "statement": "module.py defines the dispatcher.",
                "evidence_refs": [{"path": "module.py"}],
            },
            {
                "statement": "invented.py proves it.",
                "evidence_refs": [{"path": "invented.py"}],
            },
        ],
    }
    result, agent, fake, dispatcher = _run(payload)

    assert [entry.status for entry in result.evidence_validation] == [
        STATUS_SUPPORTED,
        STATUS_UNSUPPORTED,
    ]
    assert [
        entry.code for entry in result.evidence_validation
    ] == [None, CLAIM_EVIDENCE_INVALID]


# ---------------------------------------------------------------------------
# 3 — no hidden reasoning anywhere (§12), deterministic across runs
# ---------------------------------------------------------------------------


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_claims_and_validation_contain_no_hidden_reasoning():
    result, agent, fake, dispatcher = _run(READ_PAYLOAD)

    payload = {
        "structured": result.structured_output.dict(),
        "claims": [claim.to_dict() for claim in result.claims],
        "validation": [
            entry.to_dict() for entry in result.evidence_validation
        ],
        "trace": agent.last_trace.to_dict(),
    }
    keys = list(_walk_keys(payload))
    assert keys
    for banned in tuple(BANNED_REASONING_FIELDS) + BANNED_EXTRA:
        assert banned not in keys, banned


def test_claims_and_validation_are_deterministic():
    first, _, _, _ = _run(READ_PAYLOAD)
    second, _, _, _ = _run(READ_PAYLOAD)

    assert [c.to_dict() for c in first.claims] == [
        c.to_dict() for c in second.claims
    ]
    assert [e.to_dict() for e in first.evidence_validation] == [
        e.to_dict() for e in second.evidence_validation
    ]


def test_execution_config_default_stays_off():
    agent = Agent(FakeLLMClient(), structured_output_model=ClaimedSummary)
    assert agent._execution_config.structured_claims is False
    assert (
        AgentExecutionConfig().structured_claims is False
    )  # frozen default
