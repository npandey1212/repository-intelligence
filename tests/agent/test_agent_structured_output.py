"""M8.9 — provider-neutral structured output (agent layer, spec Tests 1–8).

An Agent constructed with ``structured_output_model`` sends that
model's schema with every request, classifies a terminal structured
response as ``STRUCTURED_OUTPUT`` (never dispatched), validates the
payload application-side into a typed result, and fails EXPLICITLY on
anything that does not match — recorded on the same trace as every
other run. No provider SDK is involved: the doubles below are plain
``LLMClient`` implementations, which is the provider-neutrality claim
itself (Test 8).

Test 9 (spec reaches the adapter) lives in
``tests/llm/test_structured_output.py``; Test 10 is the full-suite
regression run.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import (
    Agent,
    AgentExecutionConfig,
    render_markdown,
)
from repo_intelligence.agent.trace import (
    BANNED_REASONING_FIELDS,
    llm_response_dict,
)
from repo_intelligence.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    StructuredOutputValidationError,
    ToolCall,
)
from repo_intelligence.tools import ToolResult

QUESTION = "Summarize the login failure reports."


# Generic, repo-agnostic output contract (M8.9 docs use the same one).
class TicketSummary(BaseModel):
    title: str
    severity: int


VALID_PAYLOAD = {"title": "login fails on SSO", "severity": 2}
VALID_JSON = json.dumps(VALID_PAYLOAD)


# -- per-file test doubles (no conftest in this repo) ----------------------


class FakeLLMClient:
    """Queued responses; records the provider-neutral requests."""

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
    """Records dispatches; canned results first, else generic success."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _text_response(content="free-form answer"):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _structured_response(payload=None, content=None):
    """A fake structured response: raw JSON text + decoded payload."""
    if payload is None:
        payload = VALID_PAYLOAD
    if content is None:
        content = json.dumps(payload)
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls_response(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name="search_repository", arguments=None):
    return ToolCall(
        id="call_1",
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _structured_agent(*responses, dispatcher=None) -> Agent:
    return Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=dispatcher,
        structured_output_model=TicketSummary,
    )


# ---------------------------------------------------------------------------
# Test 1 — valid structured output → typed result (+ full trace records)
# ---------------------------------------------------------------------------


def test_valid_structured_output_returns_the_typed_result():
    fake = FakeLLMClient(_structured_response())
    agent = Agent(fake, structured_output_model=TicketSummary)

    result = agent.run_result(QUESTION)

    typed = result.structured_output
    assert isinstance(typed, TicketSummary)
    assert typed.title == "login fails on SSO"
    assert typed.severity == 2
    # the raw JSON text stays visible as the human-readable answer
    assert result.answer == VALID_JSON
    assert result.claims == ()  # structured claims stay a separate feature

    # the request carried the provider-neutral spec on every iteration
    spec = fake.requests[0].structured_output
    assert isinstance(spec, StructuredOutputSpec)
    assert spec.name == "TicketSummary"
    assert spec.schema == TicketSummary.schema()

    # trace: STRUCTURED_OUTPUT everywhere it belongs
    trace = agent.last_trace
    iteration = trace.iterations[0]
    assert iteration.response_type == "STRUCTURED_OUTPUT"
    assert iteration.decision == {"type": "STRUCTURED_OUTPUT"}
    assert iteration.structured_validation == {
        "status": "SUCCESS",
        "model": "TicketSummary",
    }
    assert iteration.llm_response["type"] == "STRUCTURED_OUTPUT"
    assert iteration.llm_response["structured_output"] == VALID_PAYLOAD
    assert iteration.request["structured_output"]["name"] == "TicketSummary"
    assert iteration.request["structured_output"]["schema"] == (
        TicketSummary.schema()
    )
    assert iteration.continuation == {
        "decision": "FINAL_ANSWER",
        "reason": "LLM returned structured output",
        "termination_reason": "STRUCTURED_OUTPUT",
    }
    assert trace.summary.termination_reason == "STRUCTURED_OUTPUT"
    assert trace.summary.final_answer == VALID_JSON

    # markdown: response type, payload, validation status, termination
    markdown = render_markdown(trace)
    assert "Type: `STRUCTURED_OUTPUT`" in markdown
    assert "Structured output:" in markdown
    assert '"login fails on SSO"' in markdown
    assert "Validation status: SUCCESS" in markdown
    assert "Termination: `STRUCTURED_OUTPUT`" in markdown


def test_run_returns_the_raw_json_text_for_a_structured_run():
    agent = _structured_agent(_structured_response())
    answer = agent.run(QUESTION)
    assert answer == VALID_JSON  # plain str contract kept; typed via run_result


# ---------------------------------------------------------------------------
# Test 2 — missing field → explicit failure, no default injected
# ---------------------------------------------------------------------------


def test_missing_field_fails_explicitly_without_a_default():
    payload = {"severity": 2}  # 'title' required but absent
    agent = _structured_agent(
        _structured_response(payload=payload, content=json.dumps(payload))
    )

    with pytest.raises(StructuredOutputValidationError) as excinfo:
        agent.run_result(QUESTION)

    assert "TicketSummary" in str(excinfo.value)
    assert excinfo.value.errors  # the field-level detail travels
    trace = agent.last_trace
    iteration = trace.iterations[0]
    assert iteration.response_type == "STRUCTURED_OUTPUT"
    assert iteration.structured_validation["status"] == "FAILED"
    assert trace.summary.termination_reason == (
        "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    )
    assert iteration.continuation["decision"] == "TERMINATE"
    markdown = render_markdown(trace)
    assert "Validation status: FAILED" in markdown


# ---------------------------------------------------------------------------
# Test 3 — wrong field type → explicit failure
# ---------------------------------------------------------------------------


def test_wrong_field_type_fails_explicitly():
    payload = {"title": "login fails on SSO", "severity": "high"}
    agent = _structured_agent(
        _structured_response(payload=payload, content=json.dumps(payload))
    )

    with pytest.raises(StructuredOutputValidationError) as excinfo:
        agent.run_result(QUESTION)

    assert excinfo.value.errors
    assert agent.last_trace.termination_reason == (
        "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    )


def test_structured_output_requested_but_no_json_payload_fails_explicitly():
    # The adapter decoded nothing (prose / broken JSON) — content text
    # exists, structured_output is None: still an explicit failure.
    agent = _structured_agent(_text_response("I will not produce JSON."))

    with pytest.raises(StructuredOutputValidationError) as excinfo:
        agent.run_result(QUESTION)

    assert "no JSON object" in str(excinfo.value)
    iteration = agent.last_trace.iterations[0]
    assert iteration.response_type == "STRUCTURED_OUTPUT"
    assert iteration.llm_response["type"] == "STRUCTURED_OUTPUT"
    assert iteration.structured_validation["status"] == "FAILED"
    assert "structured_output" not in iteration.llm_response  # none arrived
    markdown = render_markdown(agent.last_trace)
    assert "Text: <present \u2014 not a structured payload>" in markdown


# ---------------------------------------------------------------------------
# Test 5 — free-form text behavior is untouched (no spec → nothing changes)
# ---------------------------------------------------------------------------


def test_plain_agent_still_returns_free_form_text():
    fake = FakeLLMClient(_text_response("the plain answer"))
    agent = Agent(fake)

    result = agent.run_result(QUESTION)

    assert result.answer == "the plain answer"
    assert result.structured_output is None
    assert fake.requests[0].structured_output is None  # free-form default
    assert agent.last_trace.summary.termination_reason == "FINAL_ANSWER"
    assert agent.last_trace.iterations[0].response_type == "FINAL_ANSWER"
    assert agent.last_trace.iterations[0].structured_validation is None


# ---------------------------------------------------------------------------
# Test 6 — tool calls still work under a structured-output contract
# ---------------------------------------------------------------------------


def test_tool_calls_are_dispatched_and_the_loop_continues():
    dispatcher = FakeDispatcher()
    agent = _structured_agent(
        _tool_calls_response(_call("search_repository", {"query": "x"})),
        _structured_response(),
        dispatcher=dispatcher,
    )

    result = agent.run_result(QUESTION)

    # tool executed exactly once, then the structured terminal response
    assert [call.name for call in dispatcher.calls] == ["search_repository"]
    assert result.structured_output.title == "login fails on SSO"
    trace = agent.last_trace
    assert len(trace.iterations) == 2
    assert trace.iterations[0].response_type == "TOOL_CALL"
    assert trace.iterations[1].response_type == "STRUCTURED_OUTPUT"
    # the schema was on BOTH requests — it travels the whole conversation
    for iteration in trace.iterations:
        assert iteration.request["structured_output"]["name"] == (
            "TicketSummary"
        )
    assert trace.summary.tool_calls_executed == 1
    assert trace.summary.termination_reason == "STRUCTURED_OUTPUT"


# ---------------------------------------------------------------------------
# Test 7 — STRUCTURED_OUTPUT is not a tool call (classification + dispatch)
# ---------------------------------------------------------------------------


def test_structured_payload_response_is_never_dispatched():
    dispatcher = FakeDispatcher()
    agent = _structured_agent(_structured_response(), dispatcher=dispatcher)

    agent.run_result(QUESTION)

    assert dispatcher.calls == []  # payload ≠ tool request
    assert agent.last_trace.iterations[0].response_type == (
        "STRUCTURED_OUTPUT"
    )


def test_tool_calls_win_the_classification_even_with_a_payload():
    # Defensive ordering: a response carrying both is a TOOL_CALL —
    # tool requests are dispatched, never mistaken for a payload.
    response = _tool_calls_response(_call())
    hybrid = LLMResponse(
        message=response.message,
        finish_reason="tool_calls",
        structured_output=VALID_PAYLOAD,
    )
    record = llm_response_dict(hybrid, structured_requested=True)
    assert record["type"] == "TOOL_CALL"
    # and a structured run's plain payload is STRUCTURED_OUTPUT …
    assert llm_response_dict(
        _structured_response(), structured_requested=True
    )["type"] == "STRUCTURED_OUTPUT"
    # … while a non-structured run's classification is untouched.
    assert llm_response_dict(_text_response())["type"] == "FINAL_ANSWER"
    assert llm_response_dict(_tool_calls_response(_call()))["type"] == (
        "TOOL_CALL"
    )


# ---------------------------------------------------------------------------
# Test 8 — provider neutrality: the Agent never sees a provider
# ---------------------------------------------------------------------------


def test_agent_and_trace_are_provider_neutral():
    fake = FakeLLMClient(_structured_response())
    agent = Agent(fake, structured_output_model=TicketSummary)

    result = agent.run_result(QUESTION)

    # the contract crossed only the provider-neutral seam:
    # LLMRequest.structured_output on the way out, a typed model back
    request = fake.requests[0]
    assert isinstance(request, LLMRequest)
    assert isinstance(request.structured_output, StructuredOutputSpec)
    assert request.structured_output.schema == TicketSummary.schema()
    assert isinstance(result.structured_output, TicketSummary)
    # nothing provider-specific in the trace payload or markdown
    payload = json.dumps(agent.last_trace.to_dict()).lower()
    markdown = render_markdown(agent.last_trace).lower()
    for needle in ("response_format", "json_schema", "openai", "gpt-"):
        assert needle not in payload
        assert needle not in markdown


def test_structured_trace_contains_no_hidden_reasoning_fields():
    agent = _structured_agent(_structured_response())
    agent.run_result(QUESTION)

    payload = agent.last_trace.to_dict()
    serialized = json.dumps(payload).lower()
    for banned in BANNED_REASONING_FIELDS:
        assert banned not in serialized
    assert "thought" not in render_markdown(agent.last_trace).lower()


# ---------------------------------------------------------------------------
# Configuration errors — caught at construction, never mid-run
# ---------------------------------------------------------------------------


def test_non_model_structured_output_type_fails_at_construction():
    with pytest.raises(TypeError):
        Agent(FakeLLMClient(), structured_output_model=dict)


def test_structured_output_model_and_structured_claims_are_exclusive():
    with pytest.raises(ValueError):
        Agent(
            FakeLLMClient(),
            execution_config=AgentExecutionConfig(structured_claims=True),
            structured_output_model=TicketSummary,
        )
