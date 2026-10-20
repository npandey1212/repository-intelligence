"""The hard investigation lifecycle — 16 calls as a real boundary.

The soft target used to be telemetry: one notice, one event, and the
run kept exploring. ``enforce_soft_target`` (opt-in; the M17
composition root sets it) turns that crossing into the investigation
lifecycle the prompt already describes:

    cross the target → REASSESS → FINALIZE
                                       └ ONE_TARGETED_CALL (exactly
                                         one dispatch) → REASSESS
                                         → FINALIZE

1. crossing        the target opens the decision phase —
                   ``investigation_reassessed`` follows
                   ``effort_threshold_reached``, and the response
                   that follows decides from evidence
2. sufficient      decision ``FINALIZE`` with NO extra call — the
                   run stops at the target, never past it
3. insufficient    decision ``ONE_TARGETED_CALL``: exactly ONE of the
                   requested calls executes (the rest of the batch is
                   recorded as blocked, never dispatched, and never
                   presented as part of the conversation), then the
                   run reassesses and finalizes
4. satisfied gap   a gap the evidence already answers finalizes — no
                   ``targeted_call_started``, no redundant search
5. hard ceiling    ``max_tool_calls`` stays 64, the budget is checked
                   FIRST, and every finalization pass is tool-free
                   (``tools_disabled: true``); enforcement never
                   records ``execution.boundary_reached``, so the
                   derived stop reason stays EVIDENCE_SUFFICIENT
6. flag off       without the flag the run is byte-for-byte the
                   pre-lifecycle behavior (notice + event only)

Everything is scripted and LLM-free: queued responses, fixture-bound
read-only tools, one question — no network, no API key.
"""

from __future__ import annotations

import pytest

from architecture_doubles import (
    COMMIT,
    QUESTION,
    REPOSITORY,
    build_investigator,
    call,
    report_payload,
    structured,
    tool_calls,
)

from repo_intelligence.agent import (
    AgentExecutionConfig,
)
from repo_intelligence.llm import ToolCall
from repo_intelligence import repository_query as rq


def _enforced(soft_target=3, max_tool_calls=64):
    """The M17-shaped configuration under test: the hard ceiling is
    unchanged, the soft target opens the decision phase."""
    return AgentExecutionConfig(
        max_tool_calls=max_tool_calls,
        soft_tool_call_target=soft_target,
        finalize_on_boundary=True,
        enforce_soft_target=True,
    )


def _read(path):
    return tool_calls(call("read_file", {"path": path}))


def _events(trace, operation):
    return [
        event
        for event in trace.lifecycle_events
        if event["operation"] == operation
    ]


def _operations(trace):
    return [
        event["operation"] for event in trace.lifecycle_events
    ]


def _decisions(trace):
    return [
        event["decision"]
        for event in _events(trace, "execution.investigation_decision")
    ]


def _assert_no_dangling_tool_calls(specialist):
    """Every tool call shown in an assistant turn has its result in
    the SAME conversation — a provider would reject the alternative,
    and a blocked call must never appear as if it had run."""
    for request in specialist.agent._llm_client.requests:
        requested = set()
        answered = set()
        for message in request.messages:
            if message.role == "assistant":
                requested.update(
                    entry.id for entry in message.tool_calls
                )
            elif message.role == "tool":
                answered.add(message.tool_call_id)
        assert requested <= answered, (
            "tool call(s) {} were presented without a result".format(
                sorted(requested - answered)
            )
        )


# ---------------------------------------------------------------------------
# 1 — the crossing opens the decision phase
# ---------------------------------------------------------------------------


def test_crossing_the_target_opens_the_decision_phase():
    responses = [_read("file{}.py".format(i)) for i in range(3)]
    responses.append(structured(report_payload()))
    specialist = build_investigator(
        *responses, execution_config=_enforced(soft_target=3)
    )

    specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    trace = specialist.agent.last_trace
    operations = _operations(trace)
    assert "execution.effort_threshold_reached" in operations
    assert (
        operations.index("execution.effort_threshold_reached")
        < operations.index("execution.investigation_reassessed")
    )
    reassessed = _events(trace, "execution.investigation_reassessed")
    assert [event["decision_point"] for event in reassessed] == [1]
    # reassessment rather than unrestricted continuation: the
    # response that followed the crossing decided, and nothing else
    # was dispatched
    assert _decisions(trace) == ["FINALIZE"]
    assert trace.summary.tool_calls_executed == 3
    assert _events(trace, "execution.targeted_call_started") == []
    assert trace.termination_reason == "STRUCTURED_OUTPUT"


# ---------------------------------------------------------------------------
# 2 — sufficient evidence: FINALIZE, no additional call
# ---------------------------------------------------------------------------


def test_sufficient_evidence_finalizes_with_no_extra_call():
    responses = [_read("a.py"), _read("b.py"), structured(report_payload())]
    specialist = build_investigator(
        *responses, execution_config=_enforced(soft_target=2)
    )

    specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    trace = specialist.agent.last_trace
    assert _decisions(trace) == ["FINALIZE"]
    decision = _events(trace, "execution.investigation_decision")[0]
    assert decision["reason"] == "evidence_sufficient"
    # no fourth model turn asked for anything: the run stopped at the
    # target, and the queue's last response was the report itself
    assert trace.summary.tool_calls_executed == 2
    assert _events(trace, "execution.targeted_call_started") == []
    # the honest derived view: tools were still available and were
    # not consumed by a budget
    assert trace.stop_reason == "EVIDENCE_SUFFICIENT"


# ---------------------------------------------------------------------------
# 3 — insufficient: exactly ONE targeted call, then FINALIZE
# ---------------------------------------------------------------------------


def test_one_targeted_call_then_reassess_then_finalize():
    responses = [
        _read("a.py"),
        _read("b.py"),  # the target is crossed here (soft target 2)
        # three calls requested — exactly ONE may execute
        tool_calls(
            call("read_file", {"path": "c.py"}),
            ToolCall(
                id="call_2",
                name="search_repository",
                arguments={"query": "collaboration"},
            ),
            ToolCall(
                id="call_3",
                name="search_repository",
                arguments={"query": "memory"},
            ),
        ),
        structured(report_payload()),
    ]
    specialist = build_investigator(
        *responses, execution_config=_enforced(soft_target=2)
    )

    specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    trace = specialist.agent.last_trace
    # two before the boundary + exactly one targeted call
    assert trace.summary.tool_calls_executed == 3
    targeted = _events(trace, "execution.targeted_call_started")
    assert len(targeted) == 1
    assert targeted[0]["tool"] == "read_file"
    # the rest of the batch was recorded, never dispatched
    blocked = [
        entry
        for iteration in trace.iterations
        for entry in iteration.blocked_tool_calls
    ]
    assert len(blocked) == 2
    assert [entry["tool"] for entry in blocked] == [
        "search_repository",
        "search_repository",
    ]
    assert {entry["reason"] for entry in blocked} == {
        "SOFT_TOOL_CALL_TARGET_REACHED"
    }
    # REASSESS → ONE_TARGETED_CALL → REASSESS → FINALIZE
    assert _decisions(trace) == ["ONE_TARGETED_CALL", "FINALIZE"]
    reassessed = _events(trace, "execution.investigation_reassessed")
    assert [event["decision_point"] for event in reassessed] == [1, 2]
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    _assert_no_dangling_tool_calls(specialist)


# ---------------------------------------------------------------------------
# 4 — an already-satisfied gap finalizes without the redundant call
# ---------------------------------------------------------------------------


def test_an_already_satisfied_gap_never_triggers_a_search():
    search = tool_calls(
        call("search_repository", {"query": "agents"})
    )
    responses = [
        search,
        search,  # crossing the target (soft target 2) with a search
        # the evidence already answers the open gap: no third search
        structured(report_payload()),
    ]
    specialist = build_investigator(
        *responses, execution_config=_enforced(soft_target=2)
    )

    specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    trace = specialist.agent.last_trace
    assert _decisions(trace) == ["FINALIZE"]
    assert _events(trace, "execution.targeted_call_started") == []
    assert trace.summary.tool_calls_executed == 2
    assert len(trace.iterations) == 3  # search, search, report
    for iteration in trace.iterations[:-1]:
        assert iteration.blocked_tool_calls == []


# ---------------------------------------------------------------------------
# 5 — the hard ceiling stays 64, and finalization is always tool-free
# ---------------------------------------------------------------------------


def test_the_m17_config_keeps_the_64_call_ceiling_and_enforces():
    assert rq.M17_MAX_TOOL_CALLS == 64
    assert rq.M17_EXECUTION_CONFIG.max_tool_calls == 64
    assert rq.M17_EXECUTION_CONFIG.finalize_on_boundary is True
    assert rq.M17_EXECUTION_CONFIG.soft_tool_call_target == 8
    assert rq.M17_EXECUTION_CONFIG.enforce_soft_target is True


def test_a_forced_finalization_disables_tools_and_keeps_the_budget():
    responses = [
        _read("a.py"),  # soft target 1 → crossing
        _read("b.py"),  # ONE targeted call → reassessed again
        _read("c.py"),  # post-targeted: blocked, tool-free close
        structured(report_payload()),
    ]
    specialist = build_investigator(
        *responses, execution_config=_enforced(soft_target=1)
    )

    report = specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    assert report is not None
    trace = specialist.agent.last_trace
    operations = _operations(trace)
    assert "finalization.started" in operations
    assert "tools.disabled" in operations
    assert operations[-1] == "finalization.completed"
    started = _events(trace, "finalization.started")[0]
    assert started["tools_disabled"] is True
    assert started["reason"] == "execution.investigation_decision"
    disabled = _events(trace, "tools.disabled")[0]
    assert disabled["tools_disabled"] is True
    assert disabled["tool_definitions_offered"] == 0
    # the pass itself carried no tool definitions at all
    assert specialist.agent._llm_client.requests[-1].tools == []
    # enforcement does NOT claim the budget was exceeded
    assert "execution.boundary_reached" not in operations
    assert trace.stop_reason == "EVIDENCE_SUFFICIENT"
    # two dispatched (the crossing call + the targeted one); the
    # post-targeted request never executed
    assert trace.summary.tool_calls_executed == 2
    _assert_no_dangling_tool_calls(specialist)


def test_without_the_flag_the_run_keeps_the_old_behavior():
    """Flag off = byte-for-byte the pre-lifecycle run: the notice and
    the threshold event exist, tools keep executing, and none of the
    lifecycle events appear."""
    responses = [
        _read("a.py"),
        _read("b.py"),
        _read("c.py"),
        structured(report_payload()),
    ]
    specialist = build_investigator(
        *responses,
        execution_config=AgentExecutionConfig(
            max_tool_calls=64,
            soft_tool_call_target=2,
            finalize_on_boundary=True,
            # enforce_soft_target defaults to False
        ),
    )

    specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    trace = specialist.agent.last_trace
    assert trace.summary.tool_calls_executed == 3  # kept going
    operations = _operations(trace)
    assert "execution.effort_threshold_reached" in operations
    for operation in (
        "execution.investigation_reassessed",
        "execution.investigation_decision",
        "execution.targeted_call_started",
    ):
        assert operation not in operations


def test_the_hard_boundary_still_finalizes_with_tools_off():
    """The 64-call ceiling path is untouched: budget blocks the call
    (its own reason), and the pass is tool-free."""
    responses = [
        _read("a.py"),
        _read("b.py"),
        structured(report_payload()),
    ]
    specialist = build_investigator(
        *responses,
        execution_config=AgentExecutionConfig(
            max_tool_calls=1,
            finalize_on_boundary=True,
        ),
    )

    report = specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    assert report is not None
    trace = specialist.agent.last_trace
    operations = _operations(trace)
    assert "execution.boundary_reached" in operations
    assert "tools.disabled" in operations
    blocked = [
        entry
        for iteration in trace.iterations
        for entry in iteration.blocked_tool_calls
    ]
    assert [entry["reason"] for entry in blocked] == [
        "EXECUTION_LIMIT_REACHED"
    ]
    # enforcement stayed silent: no lifecycle events without the flag
    assert _events(trace, "execution.investigation_decision") == []
    assert specialist.agent._llm_client.requests[-1].tools == []


# ---------------------------------------------------------------------------
# 6 — the flag is a boolean, validated like every other field
# ---------------------------------------------------------------------------


def test_enforce_soft_target_must_be_a_boolean():
    with pytest.raises(ValueError) as excinfo:
        AgentExecutionConfig(enforce_soft_target="yes")

    assert "enforce_soft_target" in str(excinfo.value)
