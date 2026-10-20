"""Standalone CLI (``python -m repo_intelligence.llm``) tests.

The concrete client is monkeypatched so the CLI's composition-root and
output contract are proven without any provider or network.
"""

from __future__ import annotations

import pytest

import repo_intelligence.llm.__main__ as llm_main
from repo_intelligence.llm import (
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)


class _FakeLLMClient:
    """Stands in for OpenAILLMClient at the composition root."""

    response = None
    error = None
    last_request = None

    def __init__(self, model=None):
        self.model = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        type(self).last_request = request
        if type(self).error is not None:
            raise type(self).error
        return type(self).response


@pytest.fixture(autouse=True)
def _fake_client(monkeypatch):
    _FakeLLMClient.response = None
    _FakeLLMClient.error = None
    _FakeLLMClient.last_request = None
    monkeypatch.setattr(llm_main, "OpenAILLMClient", _FakeLLMClient)
    yield _FakeLLMClient


def test_cli_success_exits_zero_with_documented_output(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="Hello!"),
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    exit_code = llm_main.main(["--prompt", "say hello"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LLM Response" in captured.out
    assert "Finish reason: stop" in captured.out
    assert "Hello!" in captured.out
    assert "Input tokens: 10" in captured.out
    assert "Output tokens: 5" in captured.out
    assert "Total tokens: 15" in captured.out
    assert captured.err == ""


def test_cli_prints_requested_tool_calls_without_executing(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_repository",
                    arguments={"query": "onboarding"},
                )
            ],
        ),
        finish_reason="tool_calls",
    )
    exit_code = llm_main.main(["--prompt", "find it", "--tools"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Tool calls (requested, not executed):" in captured.out
    assert "search_repository" in captured.out
    assert '"query": "onboarding"' in captured.out


def test_cli_builds_provider_neutral_request(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="ok")
    )
    llm_main.main(["--prompt", "hello", "--model", "test-model", "--temperature", "0.1"])
    request = _FakeLLMClient.last_request
    assert isinstance(request, LLMRequest)
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[1].content == "hello"
    assert request.temperature == 0.1
    assert request.tools == []  # --tools not given


def test_cli_tools_flag_only_defines_the_sample_tool(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="ok")
    )
    llm_main.main(["--tools"])
    request = _FakeLLMClient.last_request
    assert [tool.name for tool in request.tools] == ["search_repository"]
    assert request.tools[0].input_schema["type"] == "object"


def test_cli_llm_error_exits_six_on_stderr(capsys):
    _FakeLLMClient.error = LLMError("OPENAI_API_KEY is not configured.")
    exit_code = llm_main.main([])
    captured = capsys.readouterr()
    assert exit_code == 6
    assert "LLMError: OPENAI_API_KEY is not configured." in captured.err
    assert captured.out == ""


def test_cli_rejects_an_oversized_prompt_before_the_provider(capsys):
    """The module CLI is a composition root like any other: an
    over-budget request is refused locally, with zero provider calls
    and a failure that reads as a LOCAL preflight, not an LLM error."""
    exit_code = llm_main.main(["--prompt", "x" * 400_851])
    captured = capsys.readouterr()
    assert exit_code == 6
    assert _FakeLLMClient.last_request is None  # zero provider calls
    assert "RequestBudgetExceededError" in captured.err
    assert "LOCAL context-budget preflight" in captured.err
    assert "LLMError" not in captured.err
    assert captured.out == ""


def test_cli_missing_usage_prints_placeholder(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="ok"), finish_reason="stop"
    )
    exit_code = llm_main.main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "(not provided)" in captured.out
