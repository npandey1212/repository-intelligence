"""M15 — Agent ↔ ContextEngine integration (§5/§27/§55/§56).

The PER-REASONING-STEP context capability, end to end with scripted
doubles (no LLM):

 * run WITHOUT an engine → trace shape and messages byte-for-byte
   pre-M15 (no context_events, legacy rendered-memory message)
 * run WITH an engine → ``context.select`` + ``context.render`` on
   the trace; the context message IS the rendered package (visible
   in the recorded request)
 * context is RE-SELECTED after every executed tool batch: the
   ``ToolResult`` becomes candidates, selection runs again for the
   next step, a second snapshot is persisted for the agent's REAL
   reasoning step — context #1 ≠ context #2 because the agent
   observed something new (§14/§15 critical tests)
 * memory retrieval stays separately traced: memory available ≠
   memory selected (§56) — a fresh retrieved memory is both
   ``memory.retrieve``-ed AND selected; the selection decision is a
   separate event
 * a selection failure is recorded as a structured error event, sets
   ``last_context_error``, and produces NO context message — never
   a silent fallback to a different context (§55); the
   investigation still completes
 * tool results never become memory automatically (M14 owns
   formation), and the engine never becomes a second working-memory
   store (§25)
 * invalid engine objects fail at construction
 * no hidden reasoning anywhere in context events

Everything synthetic (``example-repository``) and LLM-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest
from pydantic import BaseModel

from repo_intelligence.agent import Agent, render_markdown
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.context_engine import (
    CONTEXT_BUDGET_EXCEEDED,
    REASON_BUDGET_EXCEEDED,
    REASON_LOW_PRIORITY,
    REASON_RECENT_TOOL_RESULT,
    REASON_CURRENT_EVIDENCE,
    ContextBudget,
    ContextEngine,
    ContextOperationError,
    evaluate_context_run,
)
from repo_intelligence.evidence import STATUS_SUPPORTED
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.memory import (
    MEMORY_CONTEXT_HEADER,
    MemoryContext,
    MemoryManager,
    MemoryScope,
)
from repo_intelligence.tools import ToolResult

QUESTION = "What does example-repository's widget registry expose?"
REPO = "example-repository"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


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
        message=Message(role="assistant", content=None,
                        tool_calls=list(calls)),
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
    return (_tool_calls(_call()), _structured(READ_PAYLOAD))


def _manager(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    return manager


def _context(commit=COMMIT_A):
    return MemoryContext(
        scope=MemoryScope(repository=REPO, commit=commit),
        subject="widget",
    )


def _store_first_memory(manager):
    agent, _, _ = _build(manager, _context(), _claim_script())
    agent.run_result(QUESTION)
    return agent


def _build(manager, context, responses, engine=None):
    client = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(_read_result())
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        structured_output_model=ClaimedSummary,
        memory_manager=manager,
        memory_context=context,
        context_engine=engine,
    )
    return agent, client, dispatcher


def _select_event(trace, step="001"):
    events = [
        e
        for e in trace.context_events
        if e["operation"] == "context.select" and e.get("step_id") == step
    ]
    assert len(events) == 1
    return events[0]


def _render_event(trace, step="001"):
    events = [
        e
        for e in trace.context_events
        if e["operation"] == "context.render" and e.get("step_id") == step
    ]
    assert len(events) == 1
    return events[0]


def _snapshot_path(trace, step, root):
    return Path(root) / trace.run_id / "context_{}.json".format(step)


def _snapshot_json(trace, step, root):
    return json.loads(_snapshot_path(trace, step, root).read_text())


# ---------------------------------------------------------------------------
# without an engine → pre-M15 behavior untouched
# ---------------------------------------------------------------------------


def test_run_without_engine_has_no_context_events(tmp_path):
    manager = _manager(tmp_path)
    _store_first_memory(manager)
    agent, client, _ = _build(manager, _context(), _claim_script())
    agent.run_result(QUESTION)
    payload = agent.last_trace.to_dict()
    assert "context_events" not in payload
    assert "## Context" not in render_markdown(agent.last_trace)
    # legacy memory message byte-for-byte (M14 contract)
    memory_message = client.requests[0].messages[2].content
    assert memory_message.startswith(MEMORY_CONTEXT_HEADER)
    assert "Context selected for this reasoning step" not in (
        memory_message
    )


def test_context_engine_without_select_fails_at_construction():
    with pytest.raises(ValueError):
        Agent(llm_client=FakeLLMClient(), context_engine=object())
    # None stays the documented default (legacy run)
    agent = Agent(llm_client=FakeLLMClient(), context_engine=None)
    assert agent.last_context_error is None


# ---------------------------------------------------------------------------
# with an engine → selection boundary, traced
# ---------------------------------------------------------------------------


def test_engine_run_selects_and_renders_into_the_messages(tmp_path):
    manager = _manager(tmp_path)
    _store_first_memory(manager)  # a fresh memory exists for run 2
    engine = ContextEngine(
        budget=ContextBudget(max_items=10, max_characters=8000),
        repository=REPO,
        snapshot_root=tmp_path / "snapshots",
    )
    agent, client, _ = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    result = agent.run_result(QUESTION)
    assert result.evidence_validation[0].status == STATUS_SUPPORTED
    assert agent.last_context_error is None

    trace = agent.last_trace
    operations = [e["operation"] for e in trace.context_events]
    # one selection per reasoning step: 001 at the execution
    # boundary, 002 after the executed tool batch (M15 §5)
    assert operations == [
        "context.select",
        "context.render",
        "context.select",
        "context.render",
    ]
    assert [e["step_id"] for e in trace.context_events] == [
        "001",
        "001",
        "002",
        "002",
    ]
    select = _select_event(trace)
    assert select["context_id"].startswith("ctx_")
    assert select["step_id"] == "001"
    assert select["run_id"] == trace.run_id
    assert select["candidate_count"] == 2  # task + retrieved memory
    assert select["selected_count"] == 2
    assert select["excluded_count"] == 0
    assert select["trigger"] == "initial"
    render = _render_event(trace)
    assert render["context_id"] == select["context_id"]

    # the context message the model received IS the rendered package
    context_message = client.requests[0].messages[2].content
    assert context_message.startswith(
        "Context selected for this reasoning step"
    )
    assert "widget registry declares widget:sync" in context_message
    assert "TASK_DIRECT" in context_message
    assert "VERIFIED_MEMORY" in context_message
    assert select["context_id"] in context_message
    assert "## Context" in render_markdown(trace)

    # both REAL reasoning steps are inspectable snapshots (§12)
    first = _snapshot_path(trace, "001", engine.snapshot_root)
    second = _snapshot_path(trace, "002", engine.snapshot_root)
    assert first.is_file() and second.is_file()
    assert json.loads(first.read_text())["step_id"] == "001"
    assert json.loads(second.read_text())["step_id"] == "002"
    assert len(agent.last_context_snapshots) == 2


def test_memory_available_is_traced_separately_from_selected(tmp_path):
    # §56: memory retrieval and context selection are two events —
    # what was AVAILABLE (memory.retrieve) and what was SELECTED
    # (context.select with per-candidate decisions).
    manager = _manager(tmp_path)
    _store_first_memory(manager)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, _, _ = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    agent.run_result(QUESTION)
    trace = agent.last_trace
    retrieve = trace.memory_events[0]
    assert retrieve["operation"] == "memory.retrieve"
    assert retrieve["count"] == 1
    select = _select_event(trace)
    # both candidates considered; the memory candidate selected
    reasons = {
        d["candidate_id"]: (d["selected"], d["reason"])
        for d in select["decisions"]
    }
    assert len(reasons) == 2
    selected_reasons = sorted(
        reason for selected, reason in reasons.values() if selected
    )
    assert selected_reasons == ["TASK_DIRECT", "VERIFIED_MEMORY"]
    # ordering on the trace: retrieval BEFORE selection
    operations = (
        [e["operation"] for e in trace.memory_events]
        + [e["operation"] for e in trace.context_events]
    )
    assert operations.index("memory.retrieve") < operations.index(
        "context.select"
    )


def test_stale_memory_is_traced_as_available_but_not_selected(tmp_path):
    # run 1 stores at COMMIT_A; run 2 works from COMMIT_B → the
    # memory is retrieved (available), verified stale, and excluded
    # from the context selection with a recorded reason.
    manager = _manager(tmp_path)
    _store_first_memory(manager)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, client, _ = _build(
        manager, _context(commit=COMMIT_B), _claim_script(),
        engine=engine,
    )
    agent.run_result(QUESTION)
    trace = agent.last_trace
    retrieve = trace.memory_events[0]
    assert retrieve["count"] == 1  # available...
    select = _select_event(trace)
    memory_decisions = [
        d for d in select["decisions"] if d["reason"] in (
            "STALE_MEMORY", "UNKNOWN_FRESHNESS", "VERIFIED_MEMORY",
        )
    ]
    assert len(memory_decisions) == 1
    assert memory_decisions[0]["reason"] == "STALE_MEMORY"
    assert memory_decisions[0]["selected"] is False
    assert select["selected_count"] == 1  # only the task candidate
    context_message = client.requests[0].messages[2].content
    assert "widget registry declares widget:sync" not in (
        context_message
    )
    # the stale memory was still verified on the trace (M14 §26)
    verify_ops = [
        e for e in trace.memory_events
        if e["operation"] == "memory.verify"
    ]
    assert verify_ops and verify_ops[0]["verification_status"] == "stale"


def test_context_failure_is_recorded_never_a_silent_fallback(tmp_path):
    class ExplodingEngine:
        def select(self, *args, **kwargs):
            raise ContextOperationError(
                CONTEXT_BUDGET_EXCEEDED,
                "budget impossible by configuration",
            )

    manager = _manager(tmp_path)
    _store_first_memory(manager)
    agent, client, _ = _build(
        manager, _context(), _claim_script(), engine=ExplodingEngine()
    )
    result = agent.run_result(QUESTION)  # investigation still stands
    assert result.answer
    trace = agent.last_trace
    # §55: ONE recorded failure per reasoning step — the boundary
    # (001) and the post-tool re-selection (002). Never a silent
    # fallback, never a crash.
    assert len(trace.context_events) == 2
    for event, step in zip(trace.context_events, ("001", "002")):
        assert event["operation"] == "context.select"
        assert event["step_id"] == step
        assert event["error"]["code"] == CONTEXT_BUDGET_EXCEEDED
    assert agent.last_context_error.startswith(CONTEXT_BUDGET_EXCEEDED)
    # NO context message entered either request — and no legacy
    # fallback either: the absence is explicit
    for request in client.requests:
        roles = [m.role for m in request.messages]
        assert roles[0] == "system"
        for message in request.messages:
            assert "Context selected for this reasoning step" not in (
                message.content or ""
            )
        assert MEMORY_CONTEXT_HEADER not in request.messages[1].content


def test_context_events_carry_no_hidden_reasoning(tmp_path):
    manager = _manager(tmp_path)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, _, _ = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    agent.run_result(QUESTION)
    for event in agent.last_trace.context_events:
        for field in BANNED_REASONING_FIELDS:
            assert field not in event


# ---------------------------------------------------------------------------
# planned runs (planner input goes through the same engine)
# ---------------------------------------------------------------------------


def _plan(steps, objective=QUESTION):
    return LLMResponse(
        message=Message(
            role="assistant",
            content=json.dumps({"objective": objective, "steps": steps}),
        ),
        finish_reason="stop",
        structured_output={"objective": objective, "steps": steps},
    )


def _text(content):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def test_planned_run_selects_context_before_planning(tmp_path):
    manager = _manager(tmp_path)
    _store_first_memory(manager)
    steps = [
        {"id": "P1", "goal": "Inspect the widget registry"},
        {"id": "P2", "goal": "Summarize the findings"},
    ]
    script = (
        _plan(steps),
        _tool_calls(_call("c1")),
        _text("registry inspected"),
        _structured(READ_PAYLOAD),
    )
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    client = FakeLLMClient(*script)
    agent = Agent(
        llm_client=client,
        tool_dispatcher=FakeDispatcher(_read_result()),
        structured_output_model=ClaimedSummary,
        memory_manager=manager,
        memory_context=_context(),
        context_engine=engine,
    )
    agent.run_planned(QUESTION)
    trace = agent.last_trace
    operations = [e["operation"] for e in trace.context_events]
    # same per-step lifecycle as a non-planned run: 001 for the
    # planner input, 002 after the executed tool batch
    assert operations == [
        "context.select",
        "context.render",
        "context.select",
        "context.render",
    ]
    assert [e["step_id"] for e in trace.context_events] == [
        "001",
        "001",
        "002",
        "002",
    ]
    assert trace.context_events[0]["trigger"] == "initial"
    assert trace.context_events[2]["trigger"] == "tool_result"
    # the PLANNER request already carries the selected context
    planner_messages = trace.iterations[0].request["messages"]
    user_content = planner_messages[1]["content"]
    assert "Context selected for this reasoning step" in user_content
    assert "widget registry declares widget:sync" in user_content
    assert "TASK_DIRECT" in user_content
    # both reasoning steps were persisted as inspectable snapshots
    assert _snapshot_path(trace, "001", engine.snapshot_root).is_file()
    assert _snapshot_path(trace, "002", engine.snapshot_root).is_file()


# ---------------------------------------------------------------------------
# PER-STEP EVOLUTION — mandatory critical tests (M15 gap-closure §14/§15)
# ---------------------------------------------------------------------------


def _build_with_results(manager, context, responses, engine, results):
    client = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(*results)
    agent = Agent(
        llm_client=client,
        tool_dispatcher=dispatcher,
        structured_output_model=ClaimedSummary,
        memory_manager=manager,
        memory_context=context,
        context_engine=engine,
    )
    return agent, client, dispatcher


def _read_result_at(path, line_start, line_end, content):
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
            "content": content,
            "total_lines": 40,
        },
    )


def test_context_is_reselected_after_tool_result(tmp_path):
    """§14/§8 CRITICAL — the capability is per reasoning step, executed
    inside the REAL loop: one selection at the boundary, one after the
    executed tool batch, both traced, both rendered into the
    conversation, both persisted. The six §8 proofs below all come
    from this single execution."""
    manager = _manager(tmp_path)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, client, dispatcher = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    agent.run_result(QUESTION)
    trace = agent.last_trace

    # (1) a repository tool actually executed
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0].name == "read_file"
    # (2) it produced a result — in the trace AND in the working
    # history the model reads next (§9: the history is preserved)
    tool_iterations = [
        iteration
        for iteration in trace.iterations
        if iteration.tool_results
    ]
    assert tool_iterations
    assert all(
        result.get("success")
        for iteration in tool_iterations
        for result in iteration.tool_results
    )
    assert len(client.requests) == 2
    tool_messages = [
        message
        for message in client.requests[1].messages
        if message.role == "tool"
    ]
    assert tool_messages
    assert tool_messages[0].content
    # (3) the result became available to the ContextEngine — the
    # post-tool selection recorded the tool candidate as selected
    snapshot2 = _snapshot_json(trace, "002", engine.snapshot_root)
    assert any(
        decision["reason"] == "RECENT_TOOL_RESULT"
        and decision["selected"]
        for decision in snapshot2["selection_decisions"]
    )
    # (4) a NEW deterministic selection ran because of it — four
    # events, two steps, two causes
    assert [
        (e["operation"], e["step_id"], e["trigger"])
        for e in trace.context_events
    ] == [
        ("context.select", "001", "initial"),
        ("context.render", "001", "initial"),
        ("context.select", "002", "tool_result"),
        ("context.render", "002", "tool_result"),
    ]
    # (5) a new package was rendered (context #2 ≠ context #1) and
    # the observation entered the CONTEXT, not just the transcript
    first_context = client.requests[0].messages[2].content
    second_context = client.requests[1].messages[-1].content
    assert first_context != second_context
    assert "RECENT_TOOL_RESULT" not in first_context
    assert "RECENT_TOOL_RESULT" in second_context
    assert "CURRENT_EVIDENCE" in second_context
    # (6) the NEXT model invocation received the evolved context —
    # the last message of the request that follows the tool batch
    assert client.requests[1].messages[-1].role == "user"
    assert second_context in client.requests[1].messages[-1].content

    # both steps are inspectable snapshots on disk (§12)
    first = _snapshot_path(trace, "001", engine.snapshot_root)
    second = _snapshot_path(trace, "002", engine.snapshot_root)
    assert first.is_file() and second.is_file()
    assert snapshot2["step_id"] == "002"
    # §7 ownership: every persisted snapshot belongs to THIS run
    assert all(
        snapshot.run_id == trace.run_id
        for snapshot in agent.last_context_snapshots
    )
    assert json.loads(first.read_text())["run_id"] == trace.run_id
    assert snapshot2["run_id"] == trace.run_id

    # run-level metrics make the evolution observable (§29)
    metrics = evaluate_context_run(agent.last_context_snapshots)
    assert metrics["context_selection_count"] == 2
    assert metrics["context_evolution_count"] == 1
    assert metrics["isolation_ok"] is True


def test_context_changes_after_tool_result_for_a_meaningful_reason(tmp_path):
    """§15 CRITICAL — context #1 ≠ context #2 because the agent
    observed something: the new candidates are IN the second
    selection, with their recorded reasons."""
    manager = _manager(tmp_path)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, _, _ = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    agent.run_result(QUESTION)

    snapshot1, snapshot2 = agent.last_context_snapshots
    ids1 = [item.source_id for item in snapshot1.package.selected_items]
    ids2 = [item.source_id for item in snapshot2.package.selected_items]
    assert ids1 != ids2
    added = set(ids2) - set(ids1)
    assert any(source_id and source_id.startswith("tool:") for source_id in added)
    assert any(
        source_id and source_id.startswith("evidence:") for source_id in added
    )

    # why they entered: the recorded decision reasons
    by_candidate = {
        decision.candidate_id: decision
        for decision in snapshot2.selection_decisions
    }
    added_reasons = set()
    for item in snapshot2.package.selected_items:
        if item.source_id in added:
            added_reasons.add(by_candidate[item.candidate_id].reason)
    assert added_reasons == {
        REASON_RECENT_TOOL_RESULT,
        REASON_CURRENT_EVIDENCE,
    }
    # nothing silently vanished from step 1 — every earlier candidate
    # is still decided in step 2
    for decision in snapshot1.selection_decisions:
        assert decision.candidate_id in by_candidate


def test_new_candidates_are_excluded_not_forced_when_budget_is_tight(tmp_path):
    """§15 opposite — a tight budget: the new low-authority offers
    are decided and EXCLUDED with a recorded reason. The context
    does NOT change; nothing is forced in, nothing is dropped
    silently."""
    manager = _manager(tmp_path)
    engine = ContextEngine(
        budget=ContextBudget(max_items=1, max_characters=4000),
        repository=REPO,
        snapshot_root=tmp_path / "snapshots",
    )
    agent, _, _ = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    agent.run_result(QUESTION)

    snapshot1, snapshot2 = agent.last_context_snapshots
    ids1 = [item.source_id for item in snapshot1.package.selected_items]
    ids2 = [item.source_id for item in snapshot2.package.selected_items]
    assert ids1 == ids2  # the higher-authority context stands

    offered2 = {
        decision.candidate_id for decision in snapshot2.selection_decisions
    }
    offered1 = {
        decision.candidate_id for decision in snapshot1.selection_decisions
    }
    new_offered = offered2 - offered1
    assert new_offered  # the tool observation WAS offered...
    for candidate_id in new_offered:
        decision = next(
            d
            for d in snapshot2.selection_decisions
            if d.candidate_id == candidate_id
        )
        assert decision.selected is False  # ...and excluded, recorded
        assert decision.reason in (
            REASON_LOW_PRIORITY,
            REASON_BUDGET_EXCEEDED,
        )
    assert evaluate_context_run(
        agent.last_context_snapshots
    )["excluded_count"] >= len(new_offered)


def test_boundary_without_new_information_selects_context_once(tmp_path):
    """The capability is per MEANINGFUL boundary, not per request:
    a run with no executed tool selects exactly once (step 001)."""
    manager = _manager(tmp_path)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, client, _ = _build(
        manager, _context(), (_structured(READ_PAYLOAD),), engine=engine
    )
    agent.run_result(QUESTION)
    assert len(client.requests) == 1
    operations = [
        e["operation"] for e in agent.last_trace.context_events
    ]
    assert operations == ["context.select", "context.render"]
    assert [e["step_id"] for e in agent.last_trace.context_events] == [
        "001",
        "001",
    ]
    assert evaluate_context_run(
        agent.last_context_snapshots
    )["context_evolution_count"] == 0


# ---------------------------------------------------------------------------
# §25 negatives — tool result ≠ memory, engine ≠ working memory
# ---------------------------------------------------------------------------


def test_tool_result_does_not_become_memory(tmp_path):
    """§25 — observations enter CONTEXT (messages + snapshots) but
    never the memory store: formation count follows the VALIDATED
    CLAIMS (one), not the executed tools (two), and only on M14's
    controlled path."""
    manager = _manager(tmp_path)
    _store_first_memory(manager)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, _, dispatcher = _build_with_results(
        manager,
        _context(),
        (
            _tool_calls(_call("c1"), _call("c2")),
            _structured(READ_PAYLOAD),
        ),
        engine,
        (_read_result(), _read_result()),
    )
    agent.run_result(QUESTION)
    trace = agent.last_trace

    assert len(dispatcher.calls) == 2  # two observations entered context
    assert len(agent.last_context_snapshots) == 2  # 001 + 002
    store_events = [
        e for e in trace.memory_events if e["operation"] == "memory.store"
    ]
    assert len(store_events) == 1  # one SUPPORTED claim — not two tools
    # retrieval stays ONE boundary event (§56): no re-retrieval per
    # context step
    retrieve_events = [
        e
        for e in trace.memory_events
        if e["operation"] == "memory.retrieve"
    ]
    assert len(retrieve_events) == 1
    assert retrieve_events[0]["count"] == 1
    # the observation lives in the context message and snapshot only
    snapshot2 = agent.last_context_snapshots[-1]
    assert any(
        item.source_id and item.source_id.startswith("tool:")
        for item in snapshot2.package.selected_items
    )


def test_context_engine_is_not_a_second_working_memory_store(tmp_path):
    """§25 — working memory stays the message list: the engine has
    no message/store API, context arrives as ordinary user messages,
    and the engine records decisions (snapshots), never dialogue."""
    manager = _manager(tmp_path)
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, client, _ = _build(
        manager, _context(), _claim_script(), engine=engine
    )
    agent.run_result(QUESTION)

    # (a) selection/inspection API only — no message channel
    for forbidden in (
        "append_message",
        "add_message",
        "store_message",
        "messages",
        "store",
    ):
        assert not hasattr(engine, forbidden)

    # (b) the conversation stayed one chronological message list;
    #     context messages are ordinary user messages (no new role),
    #     and the tool observation stays in the transcript too
    roles = [m.role for m in client.requests[1].messages]
    assert roles == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert client.requests[1].messages[-1].content.startswith(
        "Context selected for this reasoning step"
    )

    # (c) the engine recorded SELECTIONS, not conversation: one
    #     snapshot per boundary, carrying decisions only
    assert len(agent.last_context_snapshots) == 2
    for snapshot in agent.last_context_snapshots:
        assert not hasattr(snapshot, "messages")
        assert snapshot.selection_decisions


# ---------------------------------------------------------------------------
# §16 full investigation — plan → tool → context → claims → memory
# ---------------------------------------------------------------------------


def test_full_investigation_from_plan_to_memory_formation(tmp_path):
    """§16 — one offline planned investigation, end to end: the plan
    runs a tool, the tool changes the context (001 → 002), the final
    step's claims validate SUPPORTED, M14's controlled path forms
    memory, and a follow-up run retrieves exactly that memory."""
    manager = _manager(tmp_path)
    steps = [
        {"id": "P1", "goal": "Inspect the widget registry"},
        {"id": "P2", "goal": "Summarize the findings"},
    ]
    script = (
        _plan(steps),
        _tool_calls(_call("c1")),
        _text("registry inspected"),
        _structured(READ_PAYLOAD),  # validated → memory forms
        _tool_calls(_call("c1")),  # follow-up run consumes it
        _structured(READ_PAYLOAD),
    )
    engine = ContextEngine(
        repository=REPO, snapshot_root=tmp_path / "snapshots"
    )
    agent, _, _ = _build_with_results(
        manager, _context(), script, engine, (_read_result(), _read_result())
    )

    run1 = agent.run_planned(QUESTION)
    trace1 = agent.last_trace
    assert run1.evidence_validation[0].status == STATUS_SUPPORTED
    # context evolved across the plan's executed tool
    assert [e["step_id"] for e in trace1.context_events] == [
        "001",
        "001",
        "002",
        "002",
    ]
    assert trace1.context_events[2]["trigger"] == "tool_result"
    assert _snapshot_path(trace1, "001", engine.snapshot_root).is_file()
    assert _snapshot_path(trace1, "002", engine.snapshot_root).is_file()
    metrics = evaluate_context_run(agent.last_context_snapshots)
    assert metrics["context_selection_count"] == 2
    assert metrics["context_evolution_count"] == 1
    assert metrics["isolation_ok"] is True
    # M14's controlled path formed memory from the VALIDATED claims
    stores = [
        e for e in trace1.memory_events if e["operation"] == "memory.store"
    ]
    assert len(stores) == 1

    # follow-up run retrieves exactly that memory (fresh, COMMIT_A)
    run2 = agent.run_result(QUESTION)
    trace2 = agent.last_trace
    assert run2.evidence_validation[0].status == STATUS_SUPPORTED
    retrieves = [
        e for e in trace2.memory_events if e["operation"] == "memory.retrieve"
    ]
    assert len(retrieves) == 1
    assert retrieves[0]["count"] == 1
    # the retrieved memory was SELECTED into step 001 of run 2 (§56)
    select1 = _select_event(trace2)
    memory_decisions = [
        d
        for d in select1["decisions"]
        if d["reason"] == "VERIFIED_MEMORY"
    ]
    assert memory_decisions and memory_decisions[0]["selected"] is True


# ---------------------------------------------------------------------------
# §24 determinism — identical selection across runs, id-independent
# ---------------------------------------------------------------------------


def _normalized_selection(snapshots):
    return [
        {
            "selected": [
                (
                    item.source_id,
                    item.authority,
                    item.priority,
                    item.selection_reason,
                    item.content,
                )
                for item in snapshot.package.selected_items
            ],
            "decisions": [
                (
                    decision.reason,
                    decision.selected,
                    decision.authority,
                    decision.priority,
                    decision.budget_cost,
                )
                for decision in snapshot.selection_decisions
            ],
        }
        for snapshot in snapshots
    ]


def test_agent_context_selection_is_deterministic_across_runs(tmp_path):
    """§24 — same repository, same task, same candidates → identical
    selection ORDER and identical decisions across two runs, even
    though every candidate_id is a fresh uuid (the tie-break must
    never depend on it). Two tool observations create genuine ties
    (same source type, same authority, same priority)."""
    responses = (
        _tool_calls(_call("c1"), _call("c2")),
        _structured(READ_PAYLOAD),
    )
    results = (
        _read_result_at("agents/widget.md", 4, 9, "widget:sync"),
        _read_result_at("docs/flow.md", 1, 3, "step one\nstep two"),
    )
    normalized = []
    decision_id_sets = []
    for run in ("one", "two"):
        manager = _manager(tmp_path / run)
        engine = ContextEngine(
            repository=REPO, snapshot_root=tmp_path / run / "snaps"
        )
        agent, _, _ = _build_with_results(
            manager, _context(), responses, engine, results
        )
        agent.run_result(QUESTION)
        normalized.append(
            _normalized_selection(agent.last_context_snapshots)
        )
        decision_id_sets.append(
            set(
                decision.candidate_id
                for snapshot in agent.last_context_snapshots
                for decision in snapshot.selection_decisions
            )
        )

    assert normalized[0] == normalized[1]
    # ...and the ids themselves are run-local uuids — the equality
    # above did NOT come from stable ids (the §24 defect)
    assert decision_id_sets[0].isdisjoint(decision_id_sets[1])
