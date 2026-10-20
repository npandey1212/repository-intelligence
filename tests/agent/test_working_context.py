"""Application working-context ceiling — bounded normal execution.

The reliability gap: a run whose accumulated conversation outgrows the
provider window dies mid-workflow (no stop reason, no compose). The
ceiling closes it in two deterministic, model-free layers — both
driven by the SAME ``max_context_characters`` number:

1. ceiling as configuration   ``DEFAULT_MAX_CONTEXT_CHARACTERS`` and
                              ``AgentExecutionConfig.max_context_characters``
                              (with validation), estimate in characters
2. persistent compaction      above the ceiling the working list is
                              REPLACED (``messages = compacted``): see
                              ``tests/agent/test_context_compaction.py``
3. per-request bound          when compaction cannot shrink the
                              conversation (one oversized unit, or a
                              preamble that carries it),
                              ``restrict_working_context``
                              still shapes THAT request: oldest raw
                              units dropped, the NEWEST interaction
                              always kept, the preamble never dropped,
                              tool turns never split, byte-identical
                              pass-through at/below the ceiling
4. observable                 one structured event per layer, counts
                              only, never message contents:
                              ``context.compaction.*`` (replacement) or
                              ``context.restricted`` (this request)
5. evidence survives          both layers rewrite the REQUEST, never
                              the trace: ``gather_evidence(trace)``
                              still returns this run's refs
6. recent context survives    the last observation stays in the
                              request, the oldest one does not
7. no oversized retry loop    a provider context-length failure is
                              recorded once and never replayed
8. stop reasons               derived from existing terminal state —
                              EVIDENCE_SUFFICIENT / BUDGET_EXCEEDED /
                              CONTEXT_LENGTH_EXCEEDED / ERROR. Neither
                              layer is a stop: a restriction or a
                              compaction is a MARKER (``Stop reason`` /
                              ``Context compactions`` /
                              ``Context restrictions`` in the summary)

Everything is scripted and LLM-free: queued responses, a recording
dispatcher, one question — no network, no API key.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import (
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    STOP_REASONS,
    SYSTEM_PROMPT,
    Agent,
    AgentExecutionConfig,
    AgentTrace,
    estimate_context_characters,
    render_markdown,
    restrict_working_context,
)
from repo_intelligence.evidence import gather_evidence
from repo_intelligence.llm import LLMError, LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

QUESTION = "How do the agents, tools, and MCP servers interact?"
HEAD = (
    Message(role="system", content=SYSTEM_PROMPT),
    Message(role="user", content=QUESTION),
)


class FakeLLMClient:
    """Queued responses (or queued exceptions); records every request."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError(
                "LLM invoked more times than queued responses"
            )
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    """Records every dispatch; returns queued results in order."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _call(name, arguments, call_id):
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def _text(text):
    return LLMResponse(
        message=Message(role="assistant", content=text),
        finish_reason="stop",
    )


def _operations(trace):
    return [event["operation"] for event in trace.lifecycle_events]


def _events(trace, operation):
    return [
        event
        for event in trace.lifecycle_events
        if event["operation"] == operation
    ]


def _event(trace, operation):
    matches = _events(trace, operation)
    assert matches, operation
    return matches[0]


def _observation(artifact, marker):
    """A large ``read_file``-style result: big enough that ONE fits a
    hand-picked ceiling and TWO do not."""
    return ToolResult(
        success=True,
        tool="read_file",
        data={"path": artifact, "content": marker * 5000},
        source={"artifact": artifact},
    )


def _ceiling_one_observation():
    """Preamble + exactly ONE observation fits; a second does not."""
    return estimate_context_characters(HEAD) + 6000


def _two_observations_then_answer(ceiling):
    """Scripted run: read, read, answer — with the ceiling armed."""
    fake = FakeLLMClient(
        _tool_calls(_call("read_file", {"path": "src/a.py"}, "c1")),
        _tool_calls(_call("read_file", {"path": "src/b.py"}, "c2")),
        _text("the run answered from the evidence it had"),
    )
    dispatcher = FakeDispatcher(
        _observation("src/a.py", "A"),
        _observation("src/b.py", "B"),
    )
    agent = Agent(
        fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_context_characters=ceiling
        ),
    )
    return agent, fake, dispatcher


def _fallback_run():
    """Two tiny observations, then one that alone crosses a very small
    ceiling.

    Compaction declines this shape — its state message would cost more
    than the tiny history it would drop — so the per-request bound is
    what keeps the request bounded. That fallback is the point of this
    script.
    """

    def tiny(path):
        return ToolResult(
            success=True,
            tool="read_file",
            data={"path": path, "content": "tiny " + path},
            source={"artifact": path},
        )

    fake = FakeLLMClient(
        _tool_calls(_call("read_file", {"path": "src/a.py"}, "c1")),
        _tool_calls(_call("read_file", {"path": "src/b.py"}, "c2")),
        _tool_calls(_call("read_file", {"path": "src/c.py"}, "c3")),
        _text("the run answered from the evidence it had"),
    )
    dispatcher = FakeDispatcher(
        tiny("src/a.py"),
        tiny("src/b.py"),
        _observation("src/c.py", "C"),
    )
    agent = Agent(
        fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_context_characters=1000),
    )
    return agent, fake, dispatcher


# ---------------------------------------------------------------------------
# 1 — ceiling as configuration
# ---------------------------------------------------------------------------


def test_default_ceiling_is_300k_characters():
    # ~75k tokens at ~4 characters/token: well inside a 128k-token
    # window, with output room and margin — and ONE number, set once.
    assert DEFAULT_MAX_CONTEXT_CHARACTERS == 300_000
    assert (
        AgentExecutionConfig().max_context_characters
        == DEFAULT_MAX_CONTEXT_CHARACTERS
    )


def test_ceiling_must_be_positive():
    with pytest.raises(ValueError, match="max_context_characters"):
        AgentExecutionConfig(max_context_characters=0)


def test_estimate_counts_message_content_and_tool_call_json():
    messages = [
        Message(role="system", content="abc"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="read_file",
                                 arguments={"path": "a"})],
        ),
    ]
    expected = 3 + len("read_file") + len(json.dumps({"path": "a"}))
    assert estimate_context_characters(messages) == expected


# ---------------------------------------------------------------------------
# 3 — deterministic reduction (pure function: the bound)
# ---------------------------------------------------------------------------


def test_below_the_ceiling_is_an_independent_copy():
    messages = list(HEAD)
    restricted = restrict_working_context(
        messages, estimate_context_characters(messages)
    )
    assert restricted == messages
    assert restricted is not messages


def test_restriction_drops_oldest_units_and_keeps_the_newest():
    messages = list(HEAD)
    for index in range(4):
        messages.append(
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c{}".format(index),
                        name="read_file",
                        arguments={"path": "f{}".format(index)},
                    )
                ],
            )
        )
        messages.append(
            Message(
                role="tool",
                tool_call_id="c{}".format(index),
                content="O" * 4000,
            )
        )
    ceiling = estimate_context_characters(HEAD) + 4500

    restricted = restrict_working_context(messages, ceiling)

    assert len(restricted) < len(messages)
    # Preamble untouched, newest observation kept.
    assert restricted[:2] == list(HEAD)
    assert restricted[-1].role == "tool"
    # Only the NEWEST observation survived: c3 in, c0/c1/c2 out.
    kept_ids = [
        message.tool_call_id
        for message in restricted
        if message.role == "tool"
    ]
    assert kept_ids == ["c3"]
    assert estimate_context_characters(restricted) <= ceiling


def test_restriction_never_splits_a_tool_call_from_its_results():
    messages = list(HEAD)
    for index in range(3):
        messages.append(
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c{}".format(index),
                        name="read_file",
                        arguments={"path": "f{}".format(index)},
                    )
                ],
            )
        )
        messages.append(
            Message(
                role="tool",
                tool_call_id="c{}".format(index),
                content="Z" * 4000,
            )
        )
    restricted = restrict_working_context(
        messages, estimate_context_characters(HEAD) + 4500
    )
    tool_ids = [
        message.tool_call_id
        for message in restricted
        if message.role == "tool"
    ]
    requested = [
        call.id
        for message in restricted
        if message.role == "assistant"
        for call in (message.tool_calls or [])
    ]
    assert tool_ids == requested


def test_a_preamble_only_conversation_is_left_alone():
    messages = list(HEAD)
    restricted = restrict_working_context(messages, 10)
    assert restricted == messages


# ---------------------------------------------------------------------------
# 4 — observable, in the shape compaction declines
# ---------------------------------------------------------------------------


def test_restriction_is_the_fallback_when_compaction_cannot_shrink():
    agent, fake, _ = _fallback_run()

    result = agent.run(QUESTION)

    assert result, "restriction must not cost the run its answer"
    trace = agent.last_trace
    # Compaction declined (its state message would not shrink this
    # shape), so the SAME ceiling was applied to the request instead —
    # and a declined compaction never leaves a dangling start event.
    assert trace.context_compactions == 0
    assert _events(trace, "context.compaction.started") == []
    assert trace.context_restrictions == 1

    event = _event(trace, "context.restricted")
    assert event["reason"] == "working_context_ceiling"
    assert event["max_context_characters"] == 1000
    assert event["before_characters"] > event["after_characters"]
    assert event["messages_before"] == 8
    assert event["messages_after"] == 4
    assert event["messages_dropped"] == 4
    assert event["observations_dropped"] == 2
    # Counts of what the trace still holds — restriction never claims
    # evidence was touched.
    assert event["evidence_recorded"] == 3

    # The event is recorded at assembly: the first three requests went
    # out as scripted, the fourth is the bounded one.
    assert [len(request.messages) for request in fake.requests] == [
        2,
        4,
        6,
        4,
    ]

    # Restriction is a request event, not a stop.
    assert trace.termination_reason == "FINAL_ANSWER"
    assert trace.stop_reason == "EVIDENCE_SUFFICIENT"


# ---------------------------------------------------------------------------
# 5 — evidence survives the bound
# ---------------------------------------------------------------------------


def test_restriction_leaves_the_evidence_record_intact():
    agent, _, dispatcher = _fallback_run()

    agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.context_restrictions == 1
    # All three observations are still ON the trace, even though the
    # last request carried only the newest one.
    recorded = sum(
        len(iteration.tool_results) for iteration in trace.iterations
    )
    assert recorded == len(dispatcher.calls) == 3

    evidence = gather_evidence(trace)
    assert evidence.gathered, "restriction must not thin the evidence record"
    assert [ref.path for ref in evidence.gathered] == [
        "src/a.py",
        "src/b.py",
        "src/c.py",
    ]
    # All three were CONTENT reads, so all three are inspections
    # (M13 §4) — including the older ones whose observations the last
    # request no longer carries.
    assert all(ref.inspected for ref in evidence.gathered)



# ---------------------------------------------------------------------------
# 6 — recent context survives
# ---------------------------------------------------------------------------

def test_the_ceiling_keeps_the_newest_observation_only():
    # Mechanism-agnostic: whether the last request was REPLACED
    # (compaction) or merely bounded, the newest observation is the one
    # that survives.
    ceiling = _ceiling_one_observation()
    agent, fake, _ = _two_observations_then_answer(ceiling)

    agent.run(QUESTION)

    bounded = fake.requests[-1]
    contents = [message.content or "" for message in bounded.messages]
    assert any("B" * 100 in text for text in contents)
    assert not any("A" * 100 in text for text in contents)
    # The question itself is durable state and is never the thing
    # that shrinks.
    assert QUESTION in contents

# ---------------------------------------------------------------------------
# 7 — no oversized retry loop
# ---------------------------------------------------------------------------


def test_context_length_failure_is_recorded_once_and_never_replayed():
    fake = FakeLLMClient(
        LLMError("context_length_exceeded: maximum context length exceeded"),
        _text("closed from the state already collected"),
    )
    dispatcher = FakeDispatcher()
    agent = Agent(
        fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(finalize_on_boundary=True),
    )

    result = agent.run(QUESTION)

    assert result
    # The oversized request was NOT retried: one failure, one
    # tool-free finalization pass.
    assert len(fake.requests) == 2
    assert fake.requests[1].tools == []
    assert _operations(agent.last_trace) == [
        "context.length_exceeded",
        "finalization.started",
        "tools.disabled",
        "finalization.completed",
    ]
    assert agent.last_trace.stop_reason == "CONTEXT_LENGTH_EXCEEDED"
    assert len(dispatcher.calls) == 0


# ---------------------------------------------------------------------------
# 8 — stop reasons: derived views over existing terminal state
# ---------------------------------------------------------------------------


def test_stop_reason_vocabulary_is_the_documented_one():
    assert STOP_REASONS == (
        "EVIDENCE_SUFFICIENT",
        "BUDGET_EXCEEDED",
        "CONTEXT_LENGTH_EXCEEDED",
        "ERROR",
    )


def test_stop_reason_is_none_while_the_run_is_open():
    trace = AgentTrace.start(user_question=QUESTION, max_tool_calls=8)
    assert trace.termination_reason is None
    assert trace.stop_reason is None


def test_stop_reason_distinguishes_the_four_terminal_classes():
    answered = AgentTrace.start(user_question=QUESTION)
    answered.terminate("FINAL_ANSWER")
    assert answered.stop_reason == "EVIDENCE_SUFFICIENT"

    structured = AgentTrace.start(user_question=QUESTION)
    structured.terminate("STRUCTURED_OUTPUT")
    assert structured.stop_reason == "EVIDENCE_SUFFICIENT"

    budget = AgentTrace.start(user_question=QUESTION, max_tool_calls=64)
    budget.terminate("EXECUTION_LIMIT_REACHED")
    assert budget.stop_reason == "BUDGET_EXCEEDED"

    failed = AgentTrace.start(user_question=QUESTION)
    failed.terminate("ERROR")
    assert failed.stop_reason == "ERROR"

    # A boundary that closed through finalization is still a BUDGET
    # stop — the finalization pass does not relabel it.
    finalized = AgentTrace.start(user_question=QUESTION, max_tool_calls=64)
    finalized.record_lifecycle_event(
        {
            "operation": "execution.boundary_reached",
            "reason": "max_tool_calls",
            "run_id": finalized.run_id,
            "tool_calls_executed": 64,
            "max_tool_calls": 64,
        }
    )
    finalized.terminate("STRUCTURED_OUTPUT")
    assert finalized.stop_reason == "BUDGET_EXCEEDED"

    # Context exhaustion outranks everything else on the trace.
    length = AgentTrace.start(user_question=QUESTION)
    length.record_lifecycle_event(
        {
            "operation": "context.length_exceeded",
            "reason": "context_length_exceeded",
            "run_id": length.run_id,
            "tool_calls_executed": 3,
            "max_tool_calls": 8,
            "error": "context_length_exceeded",
        }
    )
    length.terminate("FINAL_ANSWER")
    assert length.stop_reason == "CONTEXT_LENGTH_EXCEEDED"


def test_run_summary_reports_the_stop_reason_and_the_context_markers():
    # The run that was REPLACED reports compaction, not restriction...
    compacted, _, _ = _two_observations_then_answer(
        _ceiling_one_observation()
    )
    compacted.run(QUESTION)

    markdown = render_markdown(compacted.last_trace)
    assert "* Stop reason: `EVIDENCE_SUFFICIENT`" in markdown
    assert "* Context compactions: 1" in markdown
    assert "* Context restrictions:" not in markdown

    # ...and the run that fell back to the per-request bound reports
    # restriction. Neither marker is ever the stop reason: the run
    # answered, and the summary says so once.
    bounded, _, _ = _fallback_run()
    bounded.run(QUESTION)

    markdown = render_markdown(bounded.last_trace)
    assert "* Stop reason: `EVIDENCE_SUFFICIENT`" in markdown
    assert "* Context restrictions: 1" in markdown
    assert "* Context compactions:" not in markdown
