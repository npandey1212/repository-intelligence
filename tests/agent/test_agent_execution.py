"""M8.4 Agent → dispatcher boundary tests.

Each requested ToolCall is dispatched through the M8.3 boundary
(exactly once, real registry) and its ToolResult becomes a conversation
observation for the next LLM turn — the run ends with a final answer.
Deep conversation-state/state-machine tests live in
``test_agent_loop.py``; tool-internal behavior in ``tests/tools/``;
dispatcher mechanics in ``tests/test_tool_dispatcher.py``.
"""

from __future__ import annotations

import json

from repo_intelligence.agent import Agent, ToolDispatcher
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult
from repo_intelligence.tool_dispatcher import default_tool_dispatcher

QUESTION = "What components participate in the onboarding workflow?"
ANSWER = "The onboarding workflow uses four components."


class FakeLLMClient:
    """Records every request; returns queued canned responses."""

    def __init__(self, *responses):
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than queued responses")
        return self._queue.pop(0)


class FakeDispatcher:
    """Stands in for the application-owned dispatcher boundary."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._result is None:
            return ToolResult(success=True, tool=tool_call.name, data={})
        return self._result


def _text_response(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_call_response(*tool_calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(tool_calls)),
        finish_reason="tool_calls",
    )


def _search_call(arguments=None):
    return ToolCall(
        id="call_1",
        name="search_repository",
        arguments={"query": "onboarding workflow"}
        if arguments is None
        else arguments,
    )


def _observation_json(request):
    tool_messages = [m for m in request.messages if m.role == "tool"]
    return json.loads(tool_messages[-1].content)


# ---------------------------------------------------------------------------
# ToolResult becomes an observation; the final answer is the outcome
# ---------------------------------------------------------------------------


def test_agent_feeds_tool_result_to_the_llm_and_returns_answer():
    sentinel = ToolResult(
        success=True,
        tool="search_repository",
        data={"total_matches": 0},
        source={"repository": "sample"},
    )
    fake = FakeLLMClient(
        _tool_call_response(_search_call()),
        _text_response(),
    )
    dispatcher = FakeDispatcher(result=sentinel)

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert len(dispatcher.calls) == 1
    observation = _observation_json(fake.requests[1])
    assert observation["tool"] == "search_repository"
    assert observation["data"] == {"total_matches": 0}
    assert observation["source"] == {"repository": "sample"}


def test_agent_executes_registered_tool_end_to_end():
    fake = FakeLLMClient(
        _tool_call_response(
            ToolCall(id="call_1", name="get_architecture_context", arguments={})
        ),
        _text_response(),
    )
    dispatcher = default_tool_dispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    observation = _observation_json(fake.requests[1])
    assert observation["success"] is True
    assert observation["tool"] == "get_architecture_context"
    assert "context" in observation["data"]


def test_search_tool_call_reaches_the_dispatcher_with_expected_arguments():
    arguments = {"query": "onboarding workflow", "max_results": 5}
    fake = FakeLLMClient(
        _tool_call_response(_search_call(arguments)),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert len(dispatcher.calls) == 1
    call = dispatcher.calls[0]
    assert call.name == "search_repository"
    assert call.arguments == arguments


def test_read_file_tool_result_reaches_the_llm_as_observation():
    sentinel = ToolResult(
        success=True,
        tool="read_file",
        data={"path": "docs/workflow.md", "content": "line 1\n"},
        source={"commit_sha": "ef11dac"},
    )
    fake = FakeLLMClient(
        _tool_call_response(
            ToolCall(
                id="call_2",
                name="read_file",
                arguments={"path": "docs/workflow.md"},
            )
        ),
        _text_response(),
    )
    dispatcher = FakeDispatcher(result=sentinel)

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    observation = _observation_json(fake.requests[1])
    assert observation["tool"] == "read_file"
    assert observation["data"]["path"] == "docs/workflow.md"
    assert observation["source"] == {"commit_sha": "ef11dac"}


# ---------------------------------------------------------------------------
# Exactly-once dispatch; one LLM invocation per iteration
# ---------------------------------------------------------------------------


def test_agent_dispatches_exactly_once_per_tool_call_response():
    fake = FakeLLMClient(
        _tool_call_response(_search_call()),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert len(dispatcher.calls) == 1


def test_llm_is_invoked_once_per_iteration():
    fake = FakeLLMClient(
        _tool_call_response(_search_call()),
        _text_response(),
    )

    Agent(llm_client=fake, tool_dispatcher=FakeDispatcher()).run(QUESTION)

    # one call for the tool request + one call after the observation
    assert len(fake.requests) == 2


def test_all_tool_calls_in_a_response_are_executed_in_order():
    first = ToolCall(id="call_1", name="search_repository", arguments={"query": "a"})
    second = ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"})
    fake = FakeLLMClient(
        _tool_call_response(first, second),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
    ]
    assert len(fake.requests) == 2


# ---------------------------------------------------------------------------
# Normal textual responses stay unchanged
# ---------------------------------------------------------------------------


def test_text_answer_unchanged_when_a_dispatcher_is_injected():
    fake = FakeLLMClient(_text_response())
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert isinstance(outcome, str)
    assert dispatcher.calls == []  # nothing was dispatched
    assert len(fake.requests) == 1


# ---------------------------------------------------------------------------
# Dependency injection / structural contract
# ---------------------------------------------------------------------------


def test_agent_accepts_a_fake_dispatcher():
    fake = FakeLLMClient(
        _tool_call_response(_search_call()),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    agent = Agent(llm_client=fake, tool_dispatcher=dispatcher)
    agent.run(QUESTION)

    assert len(dispatcher.calls) == 1


def test_dispatcher_contract_is_structural_and_runtime_checkable():
    assert isinstance(FakeDispatcher(), ToolDispatcher)
    assert not isinstance(object(), ToolDispatcher)
    assert not isinstance("not a dispatcher", ToolDispatcher)
