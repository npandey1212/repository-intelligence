"""Minimal Agent tests (M8.1) — fake LLM client only.

No OpenAI API, no API key, no network: a ``FakeLLMClient`` satisfies
the ``LLMClient`` protocol and records every request the Agent sends.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from repo_intelligence.agent import SYSTEM_PROMPT, Agent, AgentError
from repo_intelligence.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
)

QUESTION = "What is the onboarding workflow?"
ANSWER = "The onboarding workflow is a four-step process."

AGENT_SRC = Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "agent"
PROVIDER_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(openai|anthropic|bedrock|litellm|langchain|langgraph)\b",
    re.MULTILINE,
)
TOOLS_IMPORT_RE = re.compile(
    r"(?:from|import)\s+(?:repo_intelligence\.tools|\.{1,2}tools)\b"
)


class FakeLLMClient:
    """Stands in for any provider: records requests, returns one canned
    ``LLMResponse``."""

    def __init__(self, response):
        self.response = response
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.response


def _text_response(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_call_response(name="search_repository", arguments=None):
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name=name,
                    arguments=arguments or {"query": "onboarding workflow"},
                )
            ],
        ),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Test 1 — Agent returns the LLM answer
# ---------------------------------------------------------------------------


def test_agent_returns_llm_answer():
    fake = FakeLLMClient(response=_text_response())
    agent = Agent(llm_client=fake)
    assert agent.run(QUESTION) == "The onboarding workflow is a four-step process."


def test_agent_returns_empty_string_when_model_returns_one():
    fake = FakeLLMClient(response=_text_response(content=""))
    assert Agent(llm_client=fake).run(QUESTION) == ""


# ---------------------------------------------------------------------------
# Test 2 — Agent constructs the request correctly
# ---------------------------------------------------------------------------


def test_agent_constructs_expected_request():
    fake = FakeLLMClient(response=_text_response())
    Agent(llm_client=fake).run(QUESTION)

    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert isinstance(request, LLMRequest)
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.messages[1].content == QUESTION
    assert [tool.name for tool in request.tools] == [  # M8.2: definitions sent
        "get_architecture_context",
        "search_repository",
        "read_file",
        "query_engineering_graph",
    ]
    assert request.model is None  # client default
    assert request.temperature is None


def test_system_prompt_is_minimal_and_tool_free():
    assert SYSTEM_PROMPT.strip()
    lowered = SYSTEM_PROMPT.lower()
    assert "tool" not in lowered
    for tool_name in (
        "search_repository",
        "read_file",
        "get_architecture_context",
    ):
        assert tool_name not in lowered


# ---------------------------------------------------------------------------
# Test 3 — Agent uses the injected LLMClient
# ---------------------------------------------------------------------------


def test_agent_uses_injected_llm_client():
    fake = FakeLLMClient(response=_text_response())
    assert isinstance(fake, LLMClient)  # fake satisfies the neutral contract

    agent = Agent(llm_client=fake)
    assert agent.run(QUESTION) == ANSWER
    assert len(fake.requests) == 1  # the injected instance handled the call


def test_agents_with_different_clients_return_their_own_answers():
    fake_a = FakeLLMClient(response=_text_response("answer A"))
    fake_b = FakeLLMClient(response=_text_response("answer B"))
    assert Agent(llm_client=fake_a).run(QUESTION) == "answer A"
    assert Agent(llm_client=fake_b).run(QUESTION) == "answer B"


def test_agent_sources_have_no_provider_imports():
    for path in sorted(AGENT_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert PROVIDER_IMPORT_RE.findall(text) == [], path


def test_agent_sources_do_not_import_repository_tools():
    for path in sorted(AGENT_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert TOOLS_IMPORT_RE.search(text) is None, path


# ---------------------------------------------------------------------------
# Test 4 — Tool calls are not executed
# ---------------------------------------------------------------------------


def test_tool_calls_are_reported_as_unsupported():
    fake = FakeLLMClient(response=_tool_call_response())
    agent = Agent(llm_client=fake)

    with pytest.raises(AgentError) as excinfo:
        agent.run(QUESTION)

    message = str(excinfo.value)
    assert "not supported" in message
    assert "search_repository" in message
    assert len(fake.requests) == 1  # no execute-and-retry loop


def test_agent_error_is_distinct_from_llm_error():
    assert issubclass(AgentError, Exception)
    assert not issubclass(AgentError, LLMError)


# ---------------------------------------------------------------------------
# Test 5 — Empty / invalid question behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_question", ["", "   ", "\n\t"])
def test_empty_question_rejected_before_reaching_llm(bad_question):
    fake = FakeLLMClient(response=_text_response())
    with pytest.raises(AgentError, match="must not be empty"):
        Agent(llm_client=fake).run(bad_question)
    assert fake.requests == []  # the LLM is never called


@pytest.mark.parametrize("bad_question", [None, 42, ["a question"]])
def test_non_string_question_rejected(bad_question):
    fake = FakeLLMClient(response=_text_response())
    with pytest.raises(AgentError, match="must be a string"):
        Agent(llm_client=fake).run(bad_question)
    assert fake.requests == []


def test_response_without_text_is_agent_error():
    fake = FakeLLMClient(
        response=LLMResponse(message=Message(role="assistant", content=None))
    )
    with pytest.raises(AgentError, match="no textual answer"):
        Agent(llm_client=fake).run(QUESTION)
