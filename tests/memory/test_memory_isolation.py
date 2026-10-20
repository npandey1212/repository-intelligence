"""M14 — repository-scoped retrieval isolation (§23 isolation invariant).

The repository scope is the ISOLATION ANCHOR of repository-derived
execution (M14 §23):

 * same repository → its memory is retrievable;
 * different repository → NOT retrievable, even when owner and
   subject match exactly (cross-repository isolation);
 * same owner + same subject but a different repository → still only
   the current repository's memory;
 * missing repository scope → NO search at all: an unrestricted
   cross-repository search is never a silent fallback; the refusal is
   recorded as a structured ``MEMORY_SCOPE_MISMATCH`` failure event.

All synthetic (``example-repository``), all LLM-free.
"""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel

from repo_intelligence.agent import Agent
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.memory import (
    MEMORY_SCOPE_MISMATCH,
    MemoryCandidate,
    MemoryContext,
    MemoryManager,
    MemoryOwner,
    MemoryScope,
)
from repo_intelligence.tools import ToolResult

QUESTION = "What does example-repository's widget registry expose?"
REPO = "example-repository"
OTHER_REPO = "example-other-repository"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
AGENT_NAME = "repo_intelligence.agent"
AGENT_OWNER = MemoryOwner(type="agent", id=AGENT_NAME)


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


def _claim_script():
    return (
        _tool_calls(_call()),
        _structured(READ_PAYLOAD),
    )


def _manager(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    return manager


def _context(repository=REPO, commit=COMMIT_A, subject="widget"):
    return MemoryContext(
        scope=MemoryScope(repository=repository, commit=commit),
        subject=subject,
    )


def _candidate(
    statement,
    repository,
    commit=COMMIT_A,
    owner=AGENT_OWNER,
    subject="widget",
):
    return MemoryCandidate(
        owner=owner,
        memory_type="semantic",
        statement=statement,
        scope=MemoryScope(repository=repository, commit=commit),
        subject=subject,
    )


def _run(manager, context=None):
    agent = Agent(
        llm_client=FakeLLMClient(*_claim_script()),
        tool_dispatcher=FakeDispatcher(_read_result()),
        structured_output_model=ClaimedSummary,
        memory_manager=manager,
        memory_context=context,
    )
    agent.run_result(QUESTION)
    return agent


def _retrieve_event(agent):
    return agent.last_trace.memory_events[0]


# ---------------------------------------------------------------------------
# Case A — same repository: the memory IS retrievable
# ---------------------------------------------------------------------------


def test_same_repository_memory_is_retrievable(tmp_path):
    manager = _manager(tmp_path)
    stored = manager.store(
        _candidate("widget registry declares widget:sync", REPO)
    )

    agent = _run(manager, _context())

    event = _retrieve_event(agent)
    assert event["operation"] == "memory.retrieve"
    assert "error" not in event
    assert event["query"]["repository"] == REPO
    assert event["count"] == 1
    assert stored.memory_id in event["memory_ids"]


# ---------------------------------------------------------------------------
# Case B — different repository: NOT retrievable (owner+subject match)
# ---------------------------------------------------------------------------


def test_other_repository_memory_is_not_retrieved(tmp_path):
    manager = _manager(tmp_path)
    stored = manager.store(
        _candidate("other repository's secret widget", OTHER_REPO)
    )

    agent = _run(manager, _context())

    event = _retrieve_event(agent)
    assert event["operation"] == "memory.retrieve"
    assert "error" not in event
    assert event["count"] == 0
    assert stored.memory_id not in event.get("memory_ids", [])
    # the memory itself still exists for ITS repository's executions
    assert manager.get(stored.memory_id).memory_id == stored.memory_id
    assert (
        manager.search(repository=OTHER_REPO)[0].memory_id
        == stored.memory_id
    )


# ---------------------------------------------------------------------------
# Case C — same owner + same subject, different repository: still isolated
# ---------------------------------------------------------------------------


def test_same_owner_and_subject_still_crosses_no_repository_boundary(
    tmp_path,
):
    manager = _manager(tmp_path)
    mine = manager.store(
        _candidate("current repository's widget registry", REPO)
    )
    theirs = manager.store(
        _candidate(
            "other repository's widget registry",
            OTHER_REPO,
            owner=AGENT_OWNER,  # identical owner...
        ),
    )

    agent = _run(manager, _context(subject="widget"))  # ...+ subject

    event = _retrieve_event(agent)
    assert "error" not in event
    assert event["count"] == 1
    assert event["memory_ids"] == [mine.memory_id]
    assert theirs.memory_id not in event["memory_ids"]


# ---------------------------------------------------------------------------
# Case D — missing repository scope: NO search is ever performed
# ---------------------------------------------------------------------------


def test_missing_repository_scope_performs_no_search(tmp_path):
    manager = _manager(tmp_path)
    manager.store(_candidate("a memory that must stay out", REPO))

    calls = []
    real_search = manager.search

    def spy_search(*args, **kwargs):
        calls.append((args, kwargs))
        return real_search(*args, **kwargs)

    manager.search = spy_search

    # no memory_context → default scope: repository is absent
    agent = _run(manager, context=None)

    assert calls == [], "retrieval searched without a repository scope"
    event = _retrieve_event(agent)
    assert event["operation"] == "memory.retrieve"
    assert event["error"]["code"] == MEMORY_SCOPE_MISMATCH
    assert "query" not in event
    assert agent.last_memory_error.startswith(MEMORY_SCOPE_MISMATCH)
