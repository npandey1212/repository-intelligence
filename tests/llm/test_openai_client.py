"""OpenAI adapter tests — provider translation exercised with fakes.

No real API calls: a fake provider client (``provider_client=``) stands
in for the SDK, so the exact translation code paths run offline. An
opt-in smoke test (``RUN_OPENAI_SMOKE=1`` + API key) exercises the real
client; it is skipped in the default suite.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from repo_intelligence.llm import (
    LLMError,
    LLMRequest,
    Message,
    ToolCall,
    ToolDefinition,
    Usage,
)
from repo_intelligence.llm.openai_client import DEFAULT_MODEL, OpenAILLMClient


# ---------------------------------------------------------------------------
# Fake provider client (mirrors the SDK's chat.completions.create shape)
# ---------------------------------------------------------------------------


class _FakeProvider:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _raw_response(
    content=None,
    tool_calls=None,
    finish_reason="stop",
    usage=True,
    choices=True,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    if not choices:
        return SimpleNamespace(choices=[], usage=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    raw_usage = (
        SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
        if usage
        else None
    )
    return SimpleNamespace(choices=[choice], usage=raw_usage)


def _raw_tool_call(
    call_id="call_1",
    name="search_repository",
    arguments='{"query": "onboarding workflow"}',
):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _client(response=None, error=None, **kwargs):
    provider = _FakeProvider(response=response, error=error)
    return OpenAILLMClient(provider_client=provider, **kwargs), provider


# ---------------------------------------------------------------------------
# Request → provider payload
# ---------------------------------------------------------------------------


def test_basic_request_translated_to_chat_completions_payload():
    client, provider = _client(response=_raw_response(content="hi"))
    response = client.complete(
        LLMRequest(
            messages=[
                Message(role="system", content="be brief"),
                Message(role="user", content="hello"),
            ],
            temperature=0.2,
        )
    )
    payload = provider.calls[0]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["temperature"] == 0.2
    assert payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert "tools" not in payload
    assert response.message.role == "assistant"
    assert response.message.content == "hi"
    assert response.finish_reason == "stop"


def test_request_model_overrides_client_default():
    client, provider = _client(response=_raw_response(content="x"))
    client.complete(
        LLMRequest(messages=[Message(role="user", content="hi")], model="req-model")
    )
    assert provider.calls[0]["model"] == "req-model"


def test_temperature_omitted_when_unset():
    client, provider = _client(response=_raw_response(content="x"))
    client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert "temperature" not in provider.calls[0]


def test_max_tokens_translated_to_provider_parameter():
    client, provider = _client(response=_raw_response(content="x"))
    client.complete(
        LLMRequest(
            messages=[Message(role="user", content="hi")],
            max_tokens=512,
        )
    )
    assert provider.calls[0]["max_tokens"] == 512


def test_max_tokens_omitted_when_unset():
    client, provider = _client(response=_raw_response(content="x"))
    client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert "max_tokens" not in provider.calls[0]


def test_max_retries_forwarded_to_sdk_constructor(monkeypatch):
    recorded = {}

    class _RecordingSDK:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _RecordingSDK, raising=True)
    OpenAILLMClient(api_key="sk-test-not-a-real-call", max_retries=0)
    assert recorded.get("max_retries") == 0

    recorded.clear()
    OpenAILLMClient(api_key="sk-test-not-a-real-call")
    assert "max_retries" not in recorded


def test_tool_definitions_translated_to_provider_functions():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    client, provider = _client(response=_raw_response(content="x"))
    client.complete(
        LLMRequest(
            messages=[Message(role="user", content="hi")],
            tools=[
                ToolDefinition(
                    name="search_repository",
                    description="Search repository text.",
                    input_schema=schema,
                )
            ],
        )
    )
    assert provider.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_repository",
                "description": "Search repository text.",
                "parameters": schema,
            },
        }
    ]


def test_assistant_and_tool_history_messages_translated():
    client, provider = _client(response=_raw_response(content="x"))
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.md"})
    client.complete(
        LLMRequest(
            messages=[
                Message(role="assistant", content=None, tool_calls=[call]),
                Message(role="tool", content="file body", tool_call_id="call_1"),
                Message(role="user", content="thanks"),
            ]
        )
    )
    messages = provider.calls[0]["messages"]
    assert messages[0]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "a.md"}),
            },
        }
    ]
    assert messages[1]["tool_call_id"] == "call_1"
    assert messages[2] == {"role": "user", "content": "thanks"}


# ---------------------------------------------------------------------------
# Provider response → LLMResponse
# ---------------------------------------------------------------------------


def test_usage_translated_to_neutral_usage():
    client, _ = _client(response=_raw_response(content="hi"))
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert response.usage == Usage(input_tokens=11, output_tokens=7, total_tokens=18)


def test_missing_usage_becomes_none():
    client, _ = _client(response=_raw_response(content="hi", usage=False))
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert response.usage is None


def test_tool_call_response_translated_to_structured_call():
    client, _ = _client(
        response=_raw_response(
            content=None,
            tool_calls=[_raw_tool_call()],
            finish_reason="tool_calls",
        )
    )
    response = client.complete(
        LLMRequest(
            messages=[Message(role="user", content="find the onboarding docs")],
            tools=[ToolDefinition(name="search_repository")],
        )
    )
    assert response.message.content is None
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        ToolCall(
            id="call_1",
            name="search_repository",
            arguments={"query": "onboarding workflow"},
        )
    ]
    assert response.tool_calls[0].arguments["query"] == "onboarding workflow"


def test_finish_reason_lowercased_and_absent_is_none():
    client, _ = _client(response=_raw_response(content="x", finish_reason="STOP"))
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert response.finish_reason == "stop"

    client, _ = _client(response=_raw_response(content="x", finish_reason=None))
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert response.finish_reason is None


def test_empty_tool_arguments_become_empty_dict():
    client, _ = _client(
        response=_raw_response(
            tool_calls=[_raw_tool_call(arguments="")], finish_reason="tool_calls"
        )
    )
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert response.tool_calls[0].arguments == {}


def test_dict_tool_arguments_accepted():
    client, _ = _client(
        response=_raw_response(
            tool_calls=[_raw_tool_call(arguments={"query": "already-parsed"})],
            finish_reason="tool_calls",
        )
    )
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert response.tool_calls[0].arguments == {"query": "already-parsed"}


# ---------------------------------------------------------------------------
# Failure translation (everything → LLMError)
# ---------------------------------------------------------------------------


def test_provider_exception_translated_to_llm_error():
    cause = RuntimeError("provider is down")
    client, _ = _client(error=cause)
    with pytest.raises(LLMError) as excinfo:
        client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert "LLM request failed" in str(excinfo.value)
    assert "provider is down" in str(excinfo.value)
    assert excinfo.value.__cause__ is cause


def test_response_without_choices_is_llm_error():
    client, _ = _client(response=_raw_response(choices=False))
    with pytest.raises(LLMError, match="no choices"):
        client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))


def test_invalid_tool_argument_json_is_llm_error():
    client, _ = _client(
        response=_raw_response(
            tool_calls=[_raw_tool_call(arguments='{"unclosed": ')],
            finish_reason="tool_calls",
        )
    )
    with pytest.raises(LLMError, match="invalid JSON arguments"):
        client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))


def test_non_object_tool_arguments_are_llm_error():
    client, _ = _client(
        response=_raw_response(
            tool_calls=[_raw_tool_call(arguments="[1, 2]")],
            finish_reason="tool_calls",
        )
    )
    with pytest.raises(LLMError, match="did not decode to a JSON object"):
        client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))


def test_tool_call_without_name_is_llm_error():
    client, _ = _client(
        response=_raw_response(
            tool_calls=[_raw_tool_call(name="")], finish_reason="tool_calls"
        )
    )
    with pytest.raises(LLMError, match="without a name"):
        client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))


def test_missing_api_key_is_llm_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAILLMClient()


def test_missing_provider_sdk_is_translated_to_llm_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # simulate absent SDK
    with pytest.raises(LLMError, match="openai.*package.*required"):
        OpenAILLMClient(api_key="sk-test-not-a-real-call")


# ---------------------------------------------------------------------------
# Opt-in real-API smoke test (skipped in the default suite)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_OPENAI_SMOKE") != "1",
    reason="opt-in real API smoke test (RUN_OPENAI_SMOKE=1 + API key)",
)
def test_real_openai_smoke():
    from repo_intelligence.llm.config import load_env

    load_env()
    client = OpenAILLMClient()
    response = client.complete(
        LLMRequest(
            messages=[Message(role="user", content="Reply with the single word: ok.")]
        )
    )
    assert response.message.content
    assert response.finish_reason == "stop"
    assert response.usage is not None and response.usage.total_tokens
