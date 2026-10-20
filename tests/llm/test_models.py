"""Provider-neutral model tests (no provider, no API)."""

from __future__ import annotations

import dataclasses

import pytest

from repo_intelligence.llm.models import (
    ROLES,
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    ToolCall,
    ToolDefinition,
    Usage,
)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


def test_message_construction():
    message = Message(role="user", content="hello")
    assert message.role == "user"
    assert message.content == "hello"
    assert message.tool_calls == []
    assert message.tool_call_id is None


def test_message_fields_are_the_documented_contract():
    assert [field.name for field in dataclasses.fields(Message)] == [
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
    ]


def test_all_documented_roles_are_supported():
    for role in ("system", "user", "assistant", "tool"):
        assert Message(role=role).role == role
    assert set(ROLES) == {"system", "user", "assistant", "tool"}


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        Message(role="function", content="x")


def test_assistant_message_can_carry_tool_calls():
    call = ToolCall(id="call_1", name="search_repository", arguments={"query": "x"})
    message = Message(role="assistant", content=None, tool_calls=[call])
    assert message.tool_calls == [call]


def test_tool_message_can_carry_tool_call_id():
    message = Message(role="tool", content="result", tool_call_id="call_1")
    assert message.tool_call_id == "call_1"


# ---------------------------------------------------------------------------
# ToolDefinition / ToolCall / Usage
# ---------------------------------------------------------------------------


def test_tool_definition_construction():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    tool = ToolDefinition(
        name="search_repository",
        description="Search repository text for relevant evidence.",
        input_schema=schema,
    )
    assert tool.name == "search_repository"
    assert tool.description.startswith("Search repository")
    assert tool.input_schema == schema  # generic JSON-schema dict
    assert [field.name for field in dataclasses.fields(ToolDefinition)] == [
        "name",
        "description",
        "input_schema",
    ]


def test_tool_call_arguments_are_structured_data():
    call = ToolCall(
        id="call_42", name="read_file", arguments={"path": "docs/a.md"}
    )
    assert call.arguments == {"path": "docs/a.md"}
    assert isinstance(call.arguments, dict)
    assert ToolCall(id="c", name="t").arguments == {}


def test_usage_fields_are_optional():
    assert Usage().input_tokens is None
    usage = Usage(input_tokens=11, output_tokens=7, total_tokens=18)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
        11,
        7,
        18,
    )


# ---------------------------------------------------------------------------
# LLMRequest / LLMResponse
# ---------------------------------------------------------------------------


def test_request_construction():
    request = LLMRequest(
        messages=[Message(role="user", content="hi")],
        tools=[ToolDefinition(name="t")],
        model="some-model",
        temperature=0.0,
    )
    assert len(request.messages) == 1
    assert request.tools[0].name == "t"
    assert request.model == "some-model"
    assert request.temperature == 0.0


def test_request_defaults_are_minimal():
    request = LLMRequest(messages=[Message(role="user", content="hi")])
    assert request.tools == []
    assert request.model is None  # falls back to the client's default
    assert request.temperature is None
    assert request.structured_output is None  # free-form text (default)
    assert request.max_tokens is None  # provider default (no output cap)
    assert [field.name for field in dataclasses.fields(LLMRequest)] == [
        "messages",
        "tools",
        "model",
        "temperature",
        "structured_output",
        "max_tokens",
    ]


def test_plain_response_has_no_tool_calls():
    response = LLMResponse(
        message=Message(role="assistant", content="Hello."),
        finish_reason="stop",
        usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
    )
    assert response.tool_calls == []
    assert not response.tool_calls  # agent can branch on truthiness
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 5


def test_response_distinguishes_tool_request():
    call = ToolCall(id="call_1", name="search_repository", arguments={"query": "q"})
    response = LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=[call]),
        finish_reason="tool_calls",
    )
    assert response.tool_calls == [call]
    assert response.tool_calls[0].name == "search_repository"
    assert response.tool_calls[0].arguments == {"query": "q"}
    assert response.message.content is None


def test_response_fields_are_the_documented_contract():
    assert [field.name for field in dataclasses.fields(LLMResponse)] == [
        "message",
        "finish_reason",
        "usage",
        "structured_output",
    ]


def test_models_are_read_only():
    message = Message(role="user", content="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "changed"


# ---------------------------------------------------------------------------
# M8.9 — StructuredOutputSpec / structured output fields
# ---------------------------------------------------------------------------


def test_structured_output_spec_construction():
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    spec = StructuredOutputSpec(name="TicketSummary", schema=schema)
    assert spec.name == "TicketSummary"
    assert spec.schema == schema


def test_structured_output_spec_requires_a_non_empty_name():
    with pytest.raises(ValueError):
        StructuredOutputSpec(name="", schema={})
    with pytest.raises(ValueError):
        StructuredOutputSpec(name="   ", schema={})


def test_structured_output_spec_schema_must_be_a_json_object():
    with pytest.raises(ValueError):
        StructuredOutputSpec(name="TicketSummary", schema=["not", "an", "object"])


def test_structured_output_spec_is_read_only():
    spec = StructuredOutputSpec(name="TicketSummary")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "changed"


def test_request_carries_an_optional_structured_output_spec():
    spec = StructuredOutputSpec(name="TicketSummary", schema={"type": "object"})
    request = LLMRequest(
        messages=[Message(role="user", content="hi")],
        structured_output=spec,
    )
    assert request.structured_output is spec


def test_response_carries_an_optional_structured_payload():
    payload = {"title": "login fails", "severity": 2}
    response = LLMResponse(
        message=Message(role="assistant", content='{"title": "login fails"}'),
        finish_reason="stop",
        structured_output=payload,
    )
    assert response.structured_output == payload


def test_structured_fields_default_to_absent():
    request = LLMRequest(messages=[Message(role="user", content="hi")])
    response = LLMResponse(message=Message(role="assistant", content="hi"))
    assert request.structured_output is None  # free-form requests unchanged
    assert response.structured_output is None  # no payload unless one arrived
