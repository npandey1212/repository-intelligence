"""M8.4 multi-step tool-calling loop tests — the state machine itself.

A ToolResult becomes a provider-neutral conversation observation and
the LLM decides the next step. These tests cover the loop, the explicit
conversation state, tool-call identity, failure-as-observation, LLM
failure, the M8.5 execution budget, and the canonical pinned-snapshot
integration. Tool-internal behavior lives in ``tests/tools/``;
dispatcher mechanics in ``tests/test_tool_dispatcher.py``; the
M8.1 request-shape baseline in ``tests/agent/test_agent.py``;
the bounded-execution contract in ``tests/agent/test_agent_bounded.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import repo_intelligence.artifacts as artifacts
from conftest import enable_search
from repo_intelligence.agent import (
    SYSTEM_PROMPT,
    Agent,
    AgentError,
    AgentExecutionConfig,
)
from repo_intelligence.llm import LLMError, LLMResponse, Message, ToolCall
from repo_intelligence.tool_dispatcher import default_tool_dispatcher
from repo_intelligence.tools import ToolError, ToolResult

QUESTION = "How does the data onboarding workflow work?"
ANSWER = "The workflow has four deterministic stages."

RUN_ID = "run_fixture_alpha"
COMMIT_SHA = "b" * 40

CONTEXT_TEXT = """# Architecture Context

- Contract: `architecture_context.v1`

## Repository

- Name: `sample-repo-alpha`
- Pinned commit: {}
- Snapshot: {}
""".format(
    COMMIT_SHA, RUN_ID
)

EXPECTED_TOOLS = [
    "get_architecture_context",
    "search_repository",
    "read_file",
    "query_engineering_graph",
]


class FakeLLMClient:
    """Queued responses (or queued exceptions); records every request."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than queued responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    """Records dispatches; returns queued results in order (else a
    generic success for the called tool)."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _text_response(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_calls_response(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name, arguments=None, call_id="call_1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _roles(request):
    return [message.role for message in request.messages]


def _tool_messages(request):
    return [message for message in request.messages if message.role == "tool"]


def _observation(request, index=-1):
    return json.loads(_tool_messages(request)[index].content)


FAILED_READ = ToolResult(
    success=False,
    tool="read_file",
    error=ToolError(code="PATH_NOT_FOUND", message="no such path"),
)


def _controlled_pinned_context(tmp_path: Path, monkeypatch):
    """Deterministic pinned-run context for the real repository tools.

    Builds the full run layout (facts/architecture/context markers,
    snapshot artifact, and the pinned ``repository/`` checkout) under
    *tmp_path* and points the default artifacts root at it. Only the
    root location is redirected — production resolution (discover run →
    repository dir → search/read) and validation are unchanged.
    """
    root = tmp_path / "artifacts"
    run_dir = root / RUN_ID
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "architecture_context").mkdir()
    (run_dir / "repository").mkdir()
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture_context" / "architecture_context.md").write_text(
        CONTEXT_TEXT, encoding="utf-8"
    )
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {
                    "name": "sample-repo-alpha",
                    "url": "https://example.invalid/sample.git",
                },
                "snapshot": {"ref": "main", "commit_sha": COMMIT_SHA},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "repository" / "README.md").write_text(
        "# Sample Project\n\nThe onboarding workflow lives here.\n",
        encoding="utf-8",
    )
    # M18: isolated store + deterministic index for search_repository.
    enable_search(run_dir / "repository", monkeypatch, tmp_path)
    monkeypatch.setattr(artifacts, "default_artifacts_root", lambda: root)
    return run_dir


# ---------------------------------------------------------------------------
# §23 Test 1 — normal final answer (no tool)
# ---------------------------------------------------------------------------


def test_final_answer_requires_a_single_llm_call():
    fake = FakeLLMClient(_text_response())
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert isinstance(outcome, str)
    assert len(fake.requests) == 1
    assert dispatcher.calls == []


# ---------------------------------------------------------------------------
# §23 Test 2 — one tool call then final answer (minimum M8.4 proof)
# ---------------------------------------------------------------------------


def test_one_tool_call_then_final_answer():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "onboarding"})),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert len(fake.requests) == 2  # one extra LLM round trip = M8.4
    assert len(dispatcher.calls) == 1


# ---------------------------------------------------------------------------
# §23 Test 3 — two-step tool sequence, executed in returned order
# ---------------------------------------------------------------------------


def test_two_step_tool_sequence_executed_in_order():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "onboarding"})),
        _tool_calls_response(_call("read_file", {"path": "README.md"})),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert len(fake.requests) == 3
    assert [call.name for call in dispatcher.calls] == [
        "search_repository",
        "read_file",
    ]


# ---------------------------------------------------------------------------
# §23 Test 4 + §35 — canonical sequence against the real dispatcher and
# the pinned repository snapshot (integration)
# ---------------------------------------------------------------------------


def test_canonical_architecture_search_read_sequence(
    tmp_path, monkeypatch
):
    _controlled_pinned_context(tmp_path, monkeypatch)
    fake = FakeLLMClient(
        _tool_calls_response(_call("get_architecture_context", {}, "c1")),
        _tool_calls_response(
            _call("search_repository", {"query": "onboarding workflow"}, "c2")
        ),
        _tool_calls_response(_call("read_file", {"path": "README.md"}, "c3")),
        _text_response(),
    )
    dispatcher = default_tool_dispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert len(fake.requests) == 4

    # Every ToolResult was appended to the conversation as an
    # observation, in step order.
    first, second, third = (fake.requests[i] for i in (1, 2, 3))
    obs1 = _observation(first)
    assert obs1["success"] is True
    assert obs1["tool"] == "get_architecture_context"
    assert "context" in obs1["data"]

    obs2 = _observation(second)
    assert obs2["success"] is True
    assert obs2["tool"] == "search_repository"
    assert obs2["data"]["query"] == "onboarding workflow"
    assert obs2["data"]["total_matches"] >= 1
    # pinned snapshot provenance reached the LLM
    assert obs2["source"]["snapshot_id"] == RUN_ID
    assert obs2["source"]["commit_sha"] == COMMIT_SHA

    obs3 = _observation(third)
    assert obs3["success"] is True
    assert obs3["tool"] == "read_file"
    assert "onboarding workflow" in json.dumps(obs3["data"])


# ---------------------------------------------------------------------------
# §24 — conversation state: exact histories across iterations
# ---------------------------------------------------------------------------


def test_conversation_history_accumulates_in_order():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        _tool_calls_response(_call("read_file", {"path": "README.md"})),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert len(fake.requests) == 3
    # first request: only the initial turns
    assert _roles(fake.requests[0]) == ["system", "user"]
    # second request: assistant tool-call turn + tool observation
    assert _roles(fake.requests[1]) == ["system", "user", "assistant", "tool"]
    # third request: two full action/observation pairs, chronological
    assert _roles(fake.requests[2]) == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    # tool definitions are offered on every iteration
    for request in fake.requests:
        assert [tool.name for tool in request.tools] == EXPECTED_TOOLS


def test_system_prompt_present_in_every_request():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        _text_response(),
    )
    Agent(llm_client=fake, tool_dispatcher=FakeDispatcher()).run(QUESTION)

    for request in fake.requests:
        assert request.messages[0].role == "system"
        assert request.messages[0].content == SYSTEM_PROMPT


def test_user_question_present_in_every_request():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        _tool_calls_response(_call("read_file", {"path": "README.md"})),
        _text_response(),
    )
    Agent(llm_client=fake, tool_dispatcher=FakeDispatcher()).run(QUESTION)

    for request in fake.requests:
        assert request.messages[1].role == "user"
        assert request.messages[1].content == QUESTION


def test_tool_definitions_sent_on_every_request():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        _tool_calls_response(_call("read_file", {"path": "README.md"})),
        _text_response(),
    )
    Agent(llm_client=fake, tool_dispatcher=FakeDispatcher()).run(QUESTION)

    assert len(fake.requests) == 3
    for request in fake.requests:
        assert [tool.name for tool in request.tools] == EXPECTED_TOOLS


# ---------------------------------------------------------------------------
# §25 — tool-call identity across the request boundary
# ---------------------------------------------------------------------------


def test_tool_call_identity_is_preserved():
    fake = FakeLLMClient(
        _tool_calls_response(
            _call("read_file", {"path": "README.md"}, "call_abc123")
        ),
        _text_response(),
    )
    Agent(llm_client=fake, tool_dispatcher=FakeDispatcher()).run(QUESTION)

    follow_up = fake.requests[1]
    assistant_turns = [
        m for m in follow_up.messages if m.role == "assistant" and m.tool_calls
    ]
    assert len(assistant_turns) == 1
    assert assistant_turns[0].tool_calls[0].id == "call_abc123"

    tool_messages = _tool_messages(follow_up)
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_abc123"


# ---------------------------------------------------------------------------
# §26 / §28 / §30 — tool failure is an observation, not a termination;
# no application retry
# ---------------------------------------------------------------------------


def test_failed_tool_result_is_observed_and_loop_continues():
    fake = FakeLLMClient(
        _tool_calls_response(_call("read_file", {"path": "missing.md"})),
        _text_response("No such file was found in the pinned snapshot."),
    )
    dispatcher = FakeDispatcher(FAILED_READ)

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome.startswith("No such file")
    assert len(fake.requests) == 2  # the model got another turn
    observation = _observation(fake.requests[1])
    assert observation["success"] is False
    assert observation["error"]["code"] == "PATH_NOT_FOUND"


def test_no_application_retry_same_tool_call_is_not_reexecuted():
    fake = FakeLLMClient(
        _tool_calls_response(_call("read_file", {"path": "missing.md"})),
        _text_response(),
    )
    dispatcher = FakeDispatcher(FAILED_READ)

    Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    # The Agent did not repeat the failing call itself: exactly one
    # dispatch happened, while the LLM received a second turn. Only a
    # new LLM-generated ToolCall could cause another execution.
    assert len(dispatcher.calls) == 1
    assert len(fake.requests) == 2


def test_llm_recovers_from_failure_by_choosing_another_tool():
    fake = FakeLLMClient(
        _tool_calls_response(_call("read_file", {"path": "missing.md"}, "c1")),
        _tool_calls_response(
            _call("search_repository", {"query": "onboarding"}, "c2")
        ),
        _text_response(),
    )
    dispatcher = FakeDispatcher(
        FAILED_READ,
        ToolResult(success=True, tool="search_repository", data={"total_matches": 1}),
    )

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert [call.name for call in dispatcher.calls] == [
        "read_file",
        "search_repository",
    ]
    assert _observation(fake.requests[1])["success"] is False
    assert _observation(fake.requests[2])["success"] is True


# ---------------------------------------------------------------------------
# §27 — LLM failure during iteration 2 propagates unchanged
# ---------------------------------------------------------------------------


def test_llm_failure_after_a_tool_step_propagates_llm_error():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        LLMError("provider exploded"),
    )
    dispatcher = FakeDispatcher()

    with pytest.raises(LLMError, match="provider exploded"):
        Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert len(fake.requests) == 2
    assert len(dispatcher.calls) == 1


# ---------------------------------------------------------------------------
# §29 — M8.5 execution budget (no hang, no fabricated answer)
# ---------------------------------------------------------------------------


def test_execution_budget_blocks_the_next_tool_call_with_structured_error():
    budget = AgentExecutionConfig().max_tool_calls
    runaway = _tool_calls_response(_call("search_repository", {"query": "a"}))
    fake = FakeLLMClient(*([runaway] * (budget + 1)))
    dispatcher = FakeDispatcher()

    with pytest.raises(AgentError) as excinfo:
        Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    message = str(excinfo.value)
    assert "execution budget exhausted" in message
    assert "NOT executed" in message
    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    # exactly `budget` executions, then the check blocked the next
    # request BEFORE dispatch (never executed, never fabricated)
    assert len(dispatcher.calls) == budget
    assert len(fake.requests) == budget + 1


# ---------------------------------------------------------------------------
# §33 — multiple ToolCalls in one response: deterministic sequential
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_execute_sequentially_in_returned_order():
    first = _call("search_repository", {"query": "a"}, "c1")
    second = _call("read_file", {"path": "README.md"}, "c2")
    fake = FakeLLMClient(
        _tool_calls_response(first, second),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    assert [call.id for call in dispatcher.calls] == ["c1", "c2"]
    follow_up = fake.requests[1]
    assert _roles(follow_up) == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    tool_messages = _tool_messages(follow_up)
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# §31 — regression: the M8.3 one-shot stop no longer happens
# ---------------------------------------------------------------------------


def test_loop_continues_after_a_tool_result_m83_regression():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        _text_response(),
    )
    dispatcher = FakeDispatcher()

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    # Under M8.3 this returned a ToolResult after ONE request. M8.4:
    # the observation went back to the LLM and a final answer is the
    # result of the run.
    assert isinstance(outcome, str)
    assert not isinstance(outcome, ToolResult)
    assert len(fake.requests) == 2


# ---------------------------------------------------------------------------
# §32 / §45 — unknown tool fails safely and is merely observed;
# registry stays the four read-only tools
# ---------------------------------------------------------------------------


def test_unknown_tool_failure_is_observed_and_agent_continues():
    dispatcher = default_tool_dispatcher()
    registry_before = tuple(dispatcher.tool_names)
    fake = FakeLLMClient(
        _tool_calls_response(_call("eval", {"code": "__import__('os')"})),
        _text_response(),
    )

    outcome = Agent(llm_client=fake, tool_dispatcher=dispatcher).run(QUESTION)

    assert outcome == ANSWER
    observation = _observation(fake.requests[1])
    assert observation["success"] is False
    assert observation["error"]["code"] == "TOOL_NOT_FOUND"
    assert tuple(dispatcher.tool_names) == registry_before
    assert registry_before == (
        "get_architecture_context",
        "query_engineering_graph",
        "read_file",
        "search_repository",
    )
