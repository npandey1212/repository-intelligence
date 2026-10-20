"""M14 — Agent ↔ Memory Manager integration (§21/§22/§43/§44).

The execution-boundary capability, end to end with scripted doubles
(no LLM):

 * run WITHOUT a manager → trace shape and messages byte-for-byte
   pre-M14 (no memory_events, no memory message)
 * run 1: validated (SUPPORTED) claims → explicit decision → store;
   UNSUPPORTED claims never persist
 * run 2 (a NEW Agent, fresh conversation, same database): retrieve
   → memory enters the existing ``messages`` (and the planner
   request) → the run STILL gathers current evidence (memory does
   not replace investigation)
 * retrieval is bounded (search_limit), recorded on the trace, and
   verified against the run's commit (staleness visible, never
   silently currented)
 * structured failures: memory_context without a manager fails at
   construction; a failing store is recorded as an error event and
   never crashes the completed investigation
 * no hidden reasoning anywhere in memory events

Everything synthetic (``example-repository``) except the mechanism —
the mechanism is the real implementation.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import Agent, render_markdown
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.evidence import STATUS_SUPPORTED
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.memory import (
    MEMORY_CONTEXT_HEADER,
    MEMORY_SCOPE_MISMATCH,
    MEMORY_STORE_NOT_INITIALIZED,
    MemoryCandidate,
    MemoryContext,
    MemoryManager,
    MemoryOperationError,
    MemoryOwner,
    MemoryScope,
)
from repo_intelligence.tools import ToolResult

QUESTION = "What does example-repository's widget registry expose?"
REPO = "example-repository"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


# -- claim-bearing contract (M13 shape, reused here) ------------------------


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


READ_PAYLOAD = {
    "finding": "example-repository's widget registry exposes widget:sync.",
    "claims": [
        {
            "statement": "widget registry declares widget:sync",
            "evidence_refs": [
                {"path": "agents/widget.md", "line_start": 4, "line_end": 9}
            ],
        }
    ],
}

UNSUPPORTED_PAYLOAD = {
    "finding": "unsupported guess",
    "claims": [
        {
            "statement": "a claim citing only a search hit",
            "evidence_refs": [
                {"path": "agents/widget.md", "line_start": 4, "line_end": 9}
            ],
        }
    ],
}


# -- doubles ---------------------------------------------------------------


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


class FakeDispatcher:
    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return _read_result()


def _structured(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _text(content):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _call(call_id="c1", name="read_file", arguments=None):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments or {"path": "agents/widget.md"},
    )


def _read_result():
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": "agents/widget.md",
            "line_start": 4,
            "line_end": 9,
            "content": "# widget registry\ntools:\n  widget:sync",
            "total_lines": 20,
        },
    )


def _plan(steps, objective=QUESTION):
    return LLMResponse(
        message=Message(
            role="assistant",
            content=json.dumps({"objective": objective, "steps": steps}),
        ),
        finish_reason="stop",
        structured_output={"objective": objective, "steps": steps},
    )


def _manager(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    return manager


def _context(commit=COMMIT_A):
    return MemoryContext(
        scope=MemoryScope(repository=REPO, commit=commit),
        subject="widget",
    )


def _claim_script(payload=READ_PAYLOAD):
    return (
        _tool_calls(_call()),
        _structured(payload),
    )


def _build(manager, context, responses, results=None):
    client = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(*(results or [_read_result()]))
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        structured_output_model=ClaimedSummary,
        memory_manager=manager,
        memory_context=context,
    )
    return agent, client, dispatcher


def _operations(trace):
    return [event["operation"] for event in trace.memory_events]


# ---------------------------------------------------------------------------
# no manager → pre-M14 behavior
# ---------------------------------------------------------------------------


def test_run_without_memory_manager_keeps_trace_and_messages_untouched():
    client = FakeLLMClient(*_claim_script())
    dispatcher = FakeDispatcher(_read_result())
    agent = Agent(
        llm_client=client,
        structured_output_model=ClaimedSummary,
        tool_dispatcher=dispatcher,
    )
    agent.run_result(QUESTION)
    payload = agent.last_trace.to_dict()
    assert "memory_events" not in payload
    assert "## Memory" not in render_markdown(agent.last_trace)
    first_request = client.requests[0]
    assert [m.role for m in first_request.messages] == ["system", "user"]


def test_memory_context_without_manager_fails_at_construction():
    with pytest.raises(ValueError):
        Agent(
            llm_client=FakeLLMClient(),
            memory_context=MemoryContext(),
        )


# ---------------------------------------------------------------------------
# execution 1 — formation (store after M13 validation)
# ---------------------------------------------------------------------------


def test_first_task_stores_a_memory_from_the_validated_claim(tmp_path):
    manager = _manager(tmp_path)
    agent, client, _ = _build(manager, _context(), _claim_script())
    result = agent.run_result(QUESTION)

    assert result.evidence_validation[0].status == STATUS_SUPPORTED
    trace = agent.last_trace
    assert _operations(trace) == ["memory.retrieve", "memory.store"]
    retrieve, store = trace.memory_events
    assert retrieve["count"] == 0
    assert retrieve["agent_id"] == "repo_intelligence.agent"
    assert retrieve["run_id"] == trace.run_id
    assert retrieve["query"]["repository"] == REPO
    assert retrieve["query"]["subject"] == "widget"
    assert store["memory_id"].startswith("mem_")
    assert store["status"] == "verified"
    assert store["repository"] == REPO
    assert store["commit"] == COMMIT_A
    assert store["run_id"] == trace.run_id

    (memory,) = manager.search(repository=REPO, subject="widget")
    assert memory.memory_id == store["memory_id"]
    assert memory.statement == "widget registry declares widget:sync"
    assert memory.source_run_id == trace.run_id
    assert memory.evidence[0].path == "agents/widget.md"


def test_unsupported_claim_is_never_persisted(tmp_path):
    manager = _manager(tmp_path)
    # search-only discovery → validator records UNSUPPORTED (M13)
    search_hit = ToolResult(
        success=True,
        tool="search_repository",
        data={
            "query": "widget",
            "results": [
                {
                    "path": "agents/widget.md",
                    "line_start": 4,
                    "line_end": 9,
                    "excerpt": "widget",
                }
            ],
        },
    )
    agent, _, _ = _build(
        manager,
        _context(),
        _claim_script(UNSUPPORTED_PAYLOAD),
        results=[search_hit],
    )
    result = agent.run_result(QUESTION)
    assert result.evidence_validation[0].status != STATUS_SUPPORTED
    assert _operations(agent.last_trace) == ["memory.retrieve"]
    assert manager.search() == ()


def test_most_observations_leave_no_memory_at_all(tmp_path):
    """A run whose final answer carries NO structured claims forms no
    memory — transient information disappears with the execution."""
    manager = _manager(tmp_path)
    client = FakeLLMClient(
        _tool_calls(_call()),
        _text("a plain textual answer, no claims"),
    )
    dispatcher = FakeDispatcher(_read_result())
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        memory_manager=manager,
        memory_context=_context(),
    )
    agent.run_result(QUESTION)
    assert _operations(agent.last_trace) == ["memory.retrieve"]
    assert manager.search() == ()


# ---------------------------------------------------------------------------
# execution 2 — retrieval into a NEW execution's context
# ---------------------------------------------------------------------------


def _store_first_memory(manager):
    agent, _, _ = _build(manager, _context(), _claim_script())
    agent.run_result(QUESTION)
    return agent


def test_second_task_retrieves_memory_into_its_messages(tmp_path):
    manager = _manager(tmp_path)
    _store_first_memory(manager)

    # a genuinely new execution: new Agent, new client, empty messages
    second, client, dispatcher = _build(
        manager, _context(), _claim_script()
    )
    second.run_result(QUESTION)

    first_request = client.requests[0]
    roles = [m.role for m in first_request.messages]
    assert roles == ["system", "user", "user"]
    memory_message = first_request.messages[2].content
    assert memory_message.startswith(MEMORY_CONTEXT_HEADER)
    assert "widget registry declares widget:sync" in memory_message
    assert "mem_" in memory_message

    retrieve, _, store = second.last_trace.memory_events
    assert retrieve["operation"] == "memory.retrieve"
    assert retrieve["count"] == 1
    memory_id = retrieve["memory_ids"][0]
    assert retrieve["query"]["status"] == "verified"
    # the run still investigated: tools ran, evidence validation ran
    assert dispatcher.calls, "memory must not replace investigation"
    assert second.last_trace.evidence_validation is not None
    # and it formed its own memory from ITS OWN current claim
    assert store["operation"] == "memory.store"
    assert store["memory_id"] != memory_id
    assert len(manager.search()) == 2


def test_second_task_verifies_retrieved_memory_against_current_commit(
    tmp_path,
):
    manager = _manager(tmp_path)
    _store_first_memory(manager)

    # same commit → fresh
    fresh, _, _ = _build(manager, _context(COMMIT_A), _claim_script())
    fresh.run_result(QUESTION)
    ops = _operations(fresh.last_trace)
    assert ops == ["memory.retrieve", "memory.verify", "memory.store"]
    verify = fresh.last_trace.memory_events[1]
    assert verify["stale"] is False
    assert verify["memory_id"].startswith("mem_")

    # different commit → stale is reported, memory untouched
    stale, _, _ = _build(manager, _context(COMMIT_B), _claim_script())
    stale.run_result(QUESTION)
    stale_verify = stale.last_trace.memory_events[1]
    assert stale_verify["stale"] is True
    assert stale_verify["commit"] == COMMIT_B


def test_retrieval_is_bounded_by_search_limit(tmp_path):
    manager = _manager(tmp_path)
    for index in range(6):
        manager.store(
            MemoryCandidate(
                owner=MemoryOwner(type="agent", id="repo_intelligence.agent"),
                memory_type="semantic",
                statement="memory number {}".format(index),
                scope=MemoryScope(repository=REPO, commit=COMMIT_A),
                subject="widget",
            )
        )
    context = MemoryContext(
        scope=MemoryScope(repository=REPO, commit=COMMIT_A),
        subject="widget",
        search_limit=3,
    )
    agent, client, _ = _build(manager, context, _claim_script())
    agent.run_result(QUESTION)
    retrieve = agent.last_trace.memory_events[0]
    assert retrieve["count"] == 3
    assert retrieve["query"]["limit"] == 3
    memory_message = client.requests[0].messages[2].content
    assert memory_message.count("* mem_") == 3


def test_default_context_never_searches_without_a_repository_scope(
    tmp_path,
):
    """Repository-derived execution without a repository scope (no
    memory_context → default owner, empty scope) performs NO memory
    search: an unrestricted cross-repository search must never be a
    silent fallback (M14 isolation invariant). The refusal is
    recorded as a structured MEMORY_SCOPE_MISMATCH failure event,
    and formation still stores with NO invented scope."""
    manager = _manager(tmp_path)
    # a memory exists somewhere — it must NOT be reachable from an
    # execution that has no repository scope to anchor against
    _store_first_memory(manager)
    agent = Agent(
        llm_client=FakeLLMClient(*_claim_script()),
        tool_dispatcher=FakeDispatcher(_read_result()),
        structured_output_model=ClaimedSummary,
        memory_manager=manager,  # no memory_context → default owner/scope
    )
    agent.run_result(QUESTION)
    retrieve, store = agent.last_trace.memory_events
    # structured refusal, explicitly recorded — not a silent empty result
    assert retrieve["operation"] == "memory.retrieve"
    assert retrieve["error"]["code"] == MEMORY_SCOPE_MISMATCH
    assert "query" not in retrieve
    assert agent.last_memory_error.startswith(MEMORY_SCOPE_MISMATCH)
    # store carries NO invented scope
    assert "repository" not in store
    assert "commit" not in store
    # the pre-existing memory was never returned to this execution
    assert retrieve.get("memory_ids", []) == []


# ---------------------------------------------------------------------------
# planned runs (§23 — retrieval before planning, M10 invariant intact)
# ---------------------------------------------------------------------------


def _planned_script():
    steps = [
        {"id": "P1", "goal": "Inspect the widget registry"},
        {"id": "P2", "goal": "Summarize the findings"},
    ]
    return (
        _plan(steps),
        _tool_calls(_call("c1")),
        _text("registry inspected"),
        _structured(READ_PAYLOAD),
    )


def test_planned_run_retrieves_before_planning_and_memory_enters_planner(
    tmp_path,
):
    manager = _manager(tmp_path)
    _store_first_memory(manager)  # a memory exists for the NEW run

    client = FakeLLMClient(*_planned_script())
    dispatcher = FakeDispatcher(_read_result())
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        structured_output_model=ClaimedSummary,
        memory_manager=manager,
        memory_context=_context(),
    )
    result = agent.run_planned(QUESTION)
    assert result.claims

    trace = agent.last_trace
    # chronological boundary: retrieve → verify → store (planning and
    # tool execution happen between retrieve and the post-run steps)
    assert _operations(trace) == [
        "memory.retrieve",
        "memory.verify",
        "memory.store",
    ]
    # retrieval is NOT a tool call: the planner request (iteration 1)
    # already carries the retrieved memory, and the plan exists before
    # any dispatch (M10 invariant preserved)
    planner_request = trace.iterations[0].request
    planner_messages = planner_request["messages"]
    user_content = planner_messages[1]["content"]
    assert MEMORY_CONTEXT_HEADER in user_content
    assert "Architecture Context" not in user_content  # none passed here
    assert "widget registry declares widget:sync" in user_content
    assert dispatcher.calls, "current investigation still runs"
    # the execution messages carry it too (system, user, memory, step)
    live_messages = client.requests[1].messages
    roles = [m.role for m in live_messages]
    assert roles[:3] == ["system", "user", "user"]
    assert MEMORY_CONTEXT_HEADER in live_messages[2].content


def test_planned_run_without_memory_keeps_the_planner_request_identical(
    tmp_path,
):
    client = FakeLLMClient(*_planned_script())
    dispatcher = FakeDispatcher(_read_result())
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        structured_output_model=ClaimedSummary,
    )
    agent.run_planned(QUESTION)
    planner_messages = agent.last_trace.iterations[0].request["messages"]
    user_content = planner_messages[1]["content"]
    assert MEMORY_CONTEXT_HEADER not in user_content
    assert user_content == QUESTION


# ---------------------------------------------------------------------------
# structured failure handling (§31 — observable, never silent)
# ---------------------------------------------------------------------------


def test_uninitialized_store_is_recorded_as_error_events_not_a_crash(
    tmp_path,
):
    broken = MemoryManager(tmp_path / "never_initialized.db")
    client = FakeLLMClient(*_claim_script())
    dispatcher = FakeDispatcher(_read_result())
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        structured_output_model=ClaimedSummary,
        memory_manager=broken,
        memory_context=_context(),
    )
    result = agent.run_result(QUESTION)  # the investigation still stands
    assert result.answer
    operations = _operations(agent.last_trace)
    assert operations == ["memory.retrieve", "memory.store"]
    for event in agent.last_trace.memory_events:
        assert event["error"]["code"] == MEMORY_STORE_NOT_INITIALIZED
    assert MEMORY_STORE_NOT_INITIALIZED in agent.last_memory_error
    # no fallback pretending: no memory message entered the conversation
    assert [m.role for m in client.requests[0].messages] == ["system", "user"]
    # and the store never pretends to hold data: it still refuses
    with pytest.raises(MemoryOperationError) as excinfo:
        broken.search()
    assert excinfo.value.code == MEMORY_STORE_NOT_INITIALIZED


def test_error_events_carry_no_hidden_reasoning(tmp_path):
    manager = _manager(tmp_path)
    agent, _, _ = _build(manager, _context(), _claim_script())
    agent.run_result(QUESTION)
    for event in agent.last_trace.memory_events:
        assert set(event) <= {
            "operation",
            "agent_id",
            "run_id",
            "repository",
            "commit",
            "count",
            "memory_ids",
            "query",
            "memory_id",
            "status",
            "verification_status",
            "stale",
            "error",
        }
        for banned in BANNED_REASONING_FIELDS:
            assert banned not in event


def test_memory_section_renders_on_the_trace(tmp_path):
    manager = _manager(tmp_path)
    agent, _, _ = _build(manager, _context(), _claim_script())
    agent.run_result(QUESTION)
    markdown = render_markdown(agent.last_trace)
    assert "## Memory" in markdown
    assert "`memory.retrieve` \u2014 0 retrieved" in markdown
    assert "`memory.store`" in markdown
    assert "status `verified`" in markdown
