"""Persistent working-context compaction (M17 reliability, final change).

The gap this closes: the per-request bound only ever shapes ONE
request. The run's working history still grew forever, so a long run
either outgrows the provider window or pays to re-send its whole
conversation on every turn. The fix stays deliberately small and lives
inside the existing loop:

1. one threshold        the existing ``max_context_characters`` ceiling,
                        checked at request ASSEMBLY — no second limit
2. replacement          above it ``compact_working_context`` REPLACES
                        the working list (``messages = compacted``):
                        grow → threshold → compact → bounded again
3. deterministic        no LLM call, no tokenizer, no summarizer: the
                        state message is built from this run's own
                        trace
4. what survives        system contract + objective, the EvidenceRefs
                        (claim → ref → artifact → pinned commit), failed
                        attempts kept apart, clipped snippets of
                        non-file observations, and the recent tool
                        window (newest unit kept, units never split)
5. observable           a count-only ``context.compaction.started`` →
                        ``context.compaction.completed`` pair carrying
                        before/after estimates — never contents
6. never grows          returns ``None`` when the result would not be
                        smaller; the per-request bound stays the
                        fallback for that one request
7. not a stop           compaction is a marker, never a stop reason

The handoff half of the same contract (a compacted Investigator still
yields the typed M16/M17 ``InvestigationReport``) lives with the
specialist tests in ``tests/agents/test_architecture_investigator_m17.py``.

Everything is scripted and LLM-free: queued responses, a recording
dispatcher, one question — no network, no API key.
"""

from __future__ import annotations

import pytest

from repo_intelligence.agent import (
    Agent,
    AgentExecutionConfig,
    COMPACT_STATE_HEADER,
    DEFAULT_COMPACTION_RECENT_MESSAGES,
    SYSTEM_PROMPT,
    compact_working_context,
    estimate_context_characters,
)
from repo_intelligence.agent.agent import build_compact_state_message
from repo_intelligence.evidence import gather_evidence
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolError, ToolResult

QUESTION = "How do the agents, tools, and MCP servers interact?"
HEAD = (
    Message(role="system", content=SYSTEM_PROMPT),
    Message(role="user", content=QUESTION),
)


class FakeLLMClient:
    """Queued responses; records every request (no network, no key)."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError(
                "LLM invoked more times than queued responses"
            )
        return self._queue.pop(0)


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


def _read_call(path, call_id):
    return _tool_calls(_call("read_file", {"path": path}, call_id))


def _observation(artifact, marker):
    """A big ``read_file``-style result: ONE fits a hand-picked
    ceiling, TWO do not."""
    return ToolResult(
        success=True,
        tool="read_file",
        data={"path": artifact, "content": marker * 5000},
        source={"artifact": artifact},
    )


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


def _states(messages):
    return [
        message.content
        for message in messages
        if (message.content or "").startswith(COMPACT_STATE_HEADER)
    ]


def _run(script, results, ceiling):
    """Script one run: queued responses, queued tool results, one
    ceiling. Returns the agent and its recording client."""
    fake = FakeLLMClient(*script)
    dispatcher = FakeDispatcher(*results)
    agent = Agent(
        fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_context_characters=ceiling),
    )
    return agent, fake, dispatcher


def _ceiling_for():
    """Preamble + room for ONE observation; two of them cross it."""
    return estimate_context_characters(HEAD) + 6000


def _two_reads_then_answer():
    """Read, read, answer — the classic grow-then-cross script."""
    return _run(
        [
            _read_call("src/a.py", "c1"),
            _read_call("src/b.py", "c2"),
            _text("the run answered from the evidence it had"),
        ],
        [_observation("src/a.py", "A"), _observation("src/b.py", "B")],
        _ceiling_for(),
    )


def _three_reads_then_answer():
    """Read, read, read, answer — enough growth to compact TWICE."""
    return _run(
        [
            _read_call("src/a.py", "c1"),
            _read_call("src/b.py", "c2"),
            _read_call("src/c.py", "c3"),
            _text("the run answered from the evidence it had"),
        ],
        [
            _observation("src/a.py", "A"),
            _observation("src/b.py", "B"),
            _observation("src/c.py", "C"),
        ],
        _ceiling_for(),
    )


def _fact_and_failure_then_answer():
    """Read a file, fail a search, query the graph — then answer: one
    reference, one failed attempt, one non-file observation."""
    graph = ToolResult(
        success=True,
        tool="query_engineering_graph",
        data={"edges": [{"source": "agent", "type": "DECLARES",
                         "target": "tool"}] * 40},
    )
    failed = ToolResult(
        success=False,
        tool="search_repository",
        error=ToolError(code="PATH_NOT_FOUND", message="no such path"),
    )
    return _run(
        [
            _read_call("src/a.py", "c1"),
            _tool_calls(_call("search_repository", {"pattern": "zzz"}, "c2")),
            _tool_calls(
                _call("query_engineering_graph", {"q": "declares"}, "c3")
            ),
            _text("the run answered from the evidence it had"),
        ],
        [_observation("src/a.py", "A"), failed, graph],
        _ceiling_for(),
    )


# ---------------------------------------------------------------------------
# 1 — one threshold, no second context limit
# ---------------------------------------------------------------------------


def test_the_ceiling_is_still_the_one_configured_number():
    # Compaction reuses the per-request ceiling: no new limit, no
    # token budget, no provider window guess.
    assert (
        AgentExecutionConfig().max_context_characters == 300_000
    )
    assert DEFAULT_COMPACTION_RECENT_MESSAGES == 8
    assert (
        AgentExecutionConfig().compaction_recent_messages
        == DEFAULT_COMPACTION_RECENT_MESSAGES
    )
    with pytest.raises(ValueError, match="max_context_characters"):
        AgentExecutionConfig(max_context_characters=0)
    with pytest.raises(ValueError, match="compaction_recent_messages"):
        AgentExecutionConfig(compaction_recent_messages=0)


# ---------------------------------------------------------------------------
# 2 — below the threshold the run is untouched
# ---------------------------------------------------------------------------


def test_below_the_threshold_the_run_is_byte_for_byte_untouched():
    # Default ceiling: the same script that would have grown forever
    # (2 → 4 → 6 → 8 messages) is left exactly as scripted.
    fake = FakeLLMClient(
        _read_call("src/a.py", "c1"),
        _read_call("src/b.py", "c2"),
        _read_call("src/c.py", "c3"),
        _text("the run answered from the evidence it had"),
    )
    dispatcher = FakeDispatcher(
        _observation("src/a.py", "A"),
        _observation("src/b.py", "B"),
        _observation("src/c.py", "C"),
    )
    agent = Agent(fake, tool_dispatcher=dispatcher)

    agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.context_compactions == 0
    assert trace.context_restrictions == 0
    assert _events(trace, "context.compaction.started") == []
    assert [len(request.messages) for request in fake.requests] == [
        2,
        4,
        6,
        8,
    ]
    assert not [
        content
        for request in fake.requests
        for content in _states(request.messages)
    ]


# ---------------------------------------------------------------------------
# 3 — replacement (the central property)
# ---------------------------------------------------------------------------


def test_compaction_replaces_the_working_history_and_keeps_using_it():
    agent, fake, _ = _three_reads_then_answer()

    agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.context_compactions == 2
    assert trace.context_restrictions == 0

    # The first crossing: a STARTED → COMPLETED pair with counts only.
    first = _event(trace, "context.compaction.completed")
    assert first["reason"] == "working_context_threshold"
    assert first["messages_after"] < first["messages_before"]
    assert (
        first["after_context_estimate"] < first["before_context_estimate"]
    )
    assert (
        first["after_context_estimate"]
        <= first["max_context_characters"]
    )
    assert first["evidence_count"] == 2
    assert not _events(trace, "context.restricted")

    # Requests before the crossing are the scripted history — the
    # bound and the compaction never rewrite what already went out.
    assert len(fake.requests[0].messages) == 2
    assert len(fake.requests[1].messages) == 4

    # The request after the crossing IS the replacement: fewer
    # messages, a state message, and the counts the event reported.
    replaced = fake.requests[2]
    assert len(replaced.messages) == first["messages_after"]
    assert len(_states(replaced.messages)) == 1
    assert (
        estimate_context_characters(replaced.messages)
        == first["after_context_estimate"]
    )

    # ...and it is the list the run CONTINUES from: the next turn
    # grows out of the compacted history (head + state + newest unit),
    # not out of the original six messages.
    continued = fake.requests[3]
    assert [message.role for message in continued.messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
    ]
    assert len(_states(continued.messages)) == 1


# ---------------------------------------------------------------------------
# 4 — repeated compaction stays bounded (multi-cycle)
# ---------------------------------------------------------------------------


def test_repeated_compaction_keeps_every_request_bounded():
    agent, fake, _ = _three_reads_then_answer()
    ceiling = _ceiling_for()

    agent.run(QUESTION)

    sizes = [
        estimate_context_characters(request.messages)
        for request in fake.requests
    ]
    completed = _events(agent.last_trace, "context.compaction.completed")
    # grow → cross → compact → grow → cross → compact: two cycles,
    # each crossing ABOVE the ceiling...
    assert agent.last_trace.context_compactions == 2
    assert all(
        event["before_context_estimate"] > ceiling for event in completed
    )
    # ...and every request that actually went out under it.
    assert all(size <= ceiling for size in sizes)
    # State messages are SUPERSEDED, never stacked: one per request.
    assert [len(_states(request.messages)) for request in fake.requests] == [
        0,
        0,
        1,
        1,
    ]


# ---------------------------------------------------------------------------
# 5 — evidence survives compaction
# ---------------------------------------------------------------------------


def test_compaction_leaves_the_evidence_record_intact():
    agent, fake, dispatcher = _two_reads_then_answer()

    agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.context_compactions == 1
    # Compaction rewrote the conversation, never the trace: both
    # observations are still recorded, and both are inspections.
    assert sum(
        len(iteration.tool_results) for iteration in trace.iterations
    ) == len(dispatcher.calls) == 2

    evidence = gather_evidence(trace)
    assert [ref.path for ref in evidence.gathered] == [
        "src/a.py",
        "src/b.py",
    ]
    assert all(ref.inspected for ref in evidence.gathered)

    # The durable substitute names every reference: the state message
    # is what the run continues to work from.
    state = _states(fake.requests[-1].messages)[0]
    assert "src/a.py" in state
    assert "src/b.py" in state
    assert "Evidence references (2)" in state


# ---------------------------------------------------------------------------
# 6 — state carries the facts, the failures and the observations
# ---------------------------------------------------------------------------


def test_compaction_state_carries_the_run_s_established_facts():
    agent, fake, _ = _fact_and_failure_then_answer()

    agent.run(QUESTION)

    assert agent.last_trace.context_compactions == 1
    state = _states(fake.requests[-1].messages)[0]

    # The objective, restated so it survives the history it spawned.
    assert "question: {}".format(QUESTION) in state
    # File-addressable facts stay as EvidenceRefs — path + provenance,
    # never the file's content.
    assert "src/a.py (read_file, inspected)" in state
    assert "A" * 100 not in state
    # Failed attempts are recorded apart, so they can never read as
    # positive evidence.
    assert "Failed attempts (1)" in state
    assert "search_repository" in state
    assert "PATH_NOT_FOUND" in state
    # Non-file observations (graph edges, engineering queries) keep a
    # clipped snippet: the only raw output compaction preserves.
    assert "Other observations (1 total" in state
    assert "DECLARES" in state
    # ...and the message closes by pointing back at the references.
    assert "references above" in state


# ---------------------------------------------------------------------------
# 7 — instruction side and recent tool window
# ---------------------------------------------------------------------------


def test_compaction_never_touches_the_instruction_side():
    agent, fake, _ = _two_reads_then_answer()

    agent.run(QUESTION)

    compacted = fake.requests[-1].messages
    # System contract and the user's question are the protected
    # preamble: the very objects the run started with, in order —
    # never the thing that shrinks.
    original = fake.requests[0].messages
    assert compacted[0] is original[0]
    assert compacted[1] is original[1]
    assert compacted[1].content == QUESTION
    # The question is restated inside the state message too.
    assert QUESTION in _states(compacted)[0]


def test_compaction_keeps_the_newest_tool_window():
    agent, fake, _ = _two_reads_then_answer()

    agent.run(QUESTION)

    contents = [
        message.content or "" for message in fake.requests[-1].messages
    ]
    # Newest observation in, oldest out — units are kept whole, so the
    # surviving window is a complete assistant→tool exchange.
    assert any("B" * 100 in text for text in contents)
    assert not any("A" * 100 in text for text in contents)


def test_compaction_drops_the_old_verbose_history_from_the_request():
    agent, fake, _ = _three_reads_then_answer()

    agent.run(QUESTION)

    # After two cycles neither of the older payloads is still being
    # paid for on every turn; only the newest one is.
    last = "\n".join(
        message.content or "" for message in fake.requests[-1].messages
    )
    assert "A" * 100 not in last
    assert "B" * 100 not in last
    assert "C" * 100 in last
    # ...while the trace still holds all three observations.
    trace = agent.last_trace
    assert sum(
        len(iteration.tool_results) for iteration in trace.iterations
    ) == 3
    assert len(gather_evidence(trace).gathered) == 3


def test_the_state_message_is_deterministic_and_trace_derived():
    agent, _, _ = _two_reads_then_answer()

    agent.run(QUESTION)

    trace = agent.last_trace
    first = build_compact_state_message(trace)
    second = build_compact_state_message(trace)
    # Built from the trace, twice → the same bytes: no model call, no
    # summarizer, no chance of an invented finding.
    assert first.content == second.content
    assert first.role == "user"
    assert "question: {}".format(QUESTION) in first.content
    for ref in gather_evidence(trace).gathered:
        assert ref.path in first.content


# ---------------------------------------------------------------------------
# 8 — compaction costs no model call, and says only counts
# ---------------------------------------------------------------------------


def test_compaction_adds_no_llm_requests_and_reveals_no_contents():
    agent, fake, dispatcher = _three_reads_then_answer()

    agent.run(QUESTION)

    # Three tool calls + one final answer = four requests, exactly —
    # two compactions happened without a single extra model call.
    assert agent.last_trace.context_compactions == 2
    assert len(fake.requests) == len(dispatcher.calls) + 1 == 4
    assert len(agent.last_trace.iterations) == 4

    operations = [
        event["operation"]
        for event in agent.last_trace.lifecycle_events
    ]
    assert operations.count("context.compaction.started") == 2
    assert operations.count("context.compaction.completed") == 2
    # A completed pair: the start carries what it crossed, the
    # completion adds what the replacement costs. Counts only —
    # never a message, never an artifact.
    started_keys = {
        "operation",
        "reason",
        "before_context_estimate",
        "messages_before",
        "evidence_count",
        "max_context_characters",
    }
    completed_keys = started_keys | {
        "after_context_estimate",
        "messages_after",
        "failed_attempts_count",
    }
    for event in _events(agent.last_trace, "context.compaction.started"):
        assert set(event) >= started_keys
    for event in _events(agent.last_trace, "context.compaction.completed"):
        assert set(event) >= completed_keys
    for event in _events(agent.last_trace, "context.compaction.started") + (
        _events(agent.last_trace, "context.compaction.completed")
    ):
        for value in event.values():
            assert len(str(value)) < 200


# ---------------------------------------------------------------------------
# 9 — compaction declines rather than grows (pure function)
# ---------------------------------------------------------------------------


def _state():
    return Message(role="user", content=COMPACT_STATE_HEADER + "\nstate")


def _unit(index, payload):
    return [
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
        ),
        Message(role="tool", tool_call_id="c{}".format(index),
                content=payload),
    ]


def test_within_the_threshold_compaction_is_not_even_attempted():
    messages = list(HEAD) + _unit(0, "X" * 4000)
    assert (
        estimate_context_characters(messages) <= 10_000
    )
    assert (
        compact_working_context(messages, 10_000, _state()) is None
    )


def test_one_oversized_unit_is_declined_not_grown():
    # One unit that cannot fit without the state message adding more
    # than it removes: compaction must say no, and the caller keeps
    # the original list (the per-request bound stays the fallback).
    messages = list(HEAD) + _unit(0, "X" * 8000)
    assert (
        compact_working_context(messages, 6_000, _state()) is None
    )


def test_a_preamble_only_conversation_is_left_alone():
    # Nothing to compact: the instruction side is never the target.
    assert compact_working_context(list(HEAD), 10, _state()) is None


def test_a_shrinking_result_keeps_units_whole_and_the_head_intact():
    messages = list(HEAD) + _unit(0, "A" * 4000) + _unit(1, "B" * 4000)

    compacted = compact_working_context(messages, 6_000, _state())

    assert compacted is not None
    assert estimate_context_characters(compacted) < (
        estimate_context_characters(messages)
    )
    assert estimate_context_characters(compacted) <= 6_000
    # Head untouched, newest unit kept whole, oldest unit gone.
    assert compacted[:2] == list(HEAD)
    assert [message.role for message in compacted[2:]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert compacted[-1].tool_call_id == "c1"
    # Repeated state messages can never accumulate: the OLD state
    # message is not carried into the replacement.
    assert _states(compacted) == [_state().content]
