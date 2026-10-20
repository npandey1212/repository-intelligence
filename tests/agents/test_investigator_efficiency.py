"""Soft investigation target + evidence discipline (M17 final).

Two halves of one reliability gap: runs that burned the whole budget
exploring, and runs that only noticed at the hard ceiling that the
evidence had long been sufficient.

1. sufficiency first     a short, sufficient run finalizes before any
                         efficiency signal exists — no notice, no
                         event, nothing to render
2. the soft target       crossing it records ONE count-only
                         ``execution.effort_threshold_reached`` event
                         and appends ONE execution-state notice; the
                         run keeps going, keeps executing tools, and
                         decides from evidence when to stop
3. the notice is once    exactly one notice, appearing in every
                         request from the crossing onward — never
                         repeated, never bloating the conversation
4. the hard ceiling      64 (M17) is unchanged and still the only
                         thing that blocks a tool: crossing the soft
                         target neither replaces nor weakens it
5. duplicate searches    a sufficient broad search finalizes
                         immediately — no scripted second search is
                         ever needed

Everything is scripted and LLM-free: queued responses, fixture-bound
read-only tools, one question — no network, no API key.
"""

from __future__ import annotations

import pytest

from architecture_doubles import (
    QUESTION,
    build_investigator,
    call,
    read_then_report,
    report_payload,
    structured,
    tool_calls,
)

from repo_intelligence.agent import (
    AgentExecutionConfig,
    SOFT_TOOL_CALL_TARGET,
)
from repo_intelligence.agents import (
    ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    ArchitectureReport,
    PROMPT_VERSION,
)
from repo_intelligence import repository_query as rq

NOTICE_MARKER = "INVESTIGATION_EFFORT_THRESHOLD_REACHED"
# The prompt also carries the marker (it anticipates the notice), and
# the specialist's ROLE arrives as a user message too — so the notice
# is identified by the sentence only the appended message contains.
NOTICE_ANCHOR = "reassessment point, not a cutoff"


def _strategy_section() -> str:
    prompt = ARCHITECTURE_INVESTIGATOR_PROMPT_V1
    start = prompt.index("INVESTIGATION STRATEGY")
    return prompt[start:prompt.index("OUTPUT")]


def _long_run(soft_target, max_tool_calls=64, tool_calls_count=5):
    """Five inspections, then the typed report — over the soft target."""
    responses = [
        tool_calls(call("read_file", {"path": "file{}.py".format(index)}))
        for index in range(tool_calls_count)
    ]
    responses.append(structured(report_payload()))
    specialist = build_investigator(
        *responses,
        execution_config=AgentExecutionConfig(
            max_tool_calls=max_tool_calls,
            soft_tool_call_target=soft_target,
            finalize_on_boundary=True,
        ),
    )
    result = specialist.investigate(
        QUESTION, "example-repository", "c" * 40
    )
    return specialist, result


def _notice_occurrences(request) -> int:
    return sum(
        1
        for message in request.messages
        if message.role == "user"
        and NOTICE_ANCHOR in (message.content or "")
    )


def _events(trace, operation):
    return [
        event
        for event in trace.lifecycle_events
        if event["operation"] == operation
    ]


def _markdown(trace):
    from repo_intelligence.agent import render_markdown

    return render_markdown(trace)


# ---------------------------------------------------------------------------
# 1 — evidence sufficiency: finalized long before any signal exists
# ---------------------------------------------------------------------------


def test_sufficient_evidence_finalizes_without_any_efficiency_signal():
    specialist = build_investigator(*read_then_report())

    result = specialist.investigate(
        QUESTION, "example-repository", "c" * 40
    )

    assert result is not None
    assert isinstance(result, ArchitectureReport)
    trace = specialist.agent.last_trace
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    assert trace.stop_reason == "EVIDENCE_SUFFICIENT"
    assert trace.summary.tool_calls_executed == 1
    # nothing crossed, nothing recorded, nothing to render
    assert trace.effort_threshold_reached is False
    assert trace.lifecycle_events == []
    for request in specialist.agent._llm_client.requests:
        assert _notice_occurrences(request) == 0
    assert "* Soft effort target: reached" not in _markdown(trace)



# ---------------------------------------------------------------------------
# 2 — crossing the soft target: recorded, not enforced
# ---------------------------------------------------------------------------


def test_crossing_the_soft_target_records_the_signal_and_keeps_going():
    specialist, result = _long_run(soft_target=3)

    assert result is not None
    trace = specialist.agent.last_trace
    summary = trace.summary

    # the run did NOT stop at the soft target: it kept investigating
    # and finished voluntarily from evidence
    assert summary.tool_calls_executed == 5
    assert summary.tool_calls_blocked == 0
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    assert trace.stop_reason == "EVIDENCE_SUFFICIENT"

    assert trace.effort_threshold_reached is True
    events = _events(trace, "execution.effort_threshold_reached")
    assert len(events) == 1  # once per run, never repeated
    event = events[0]
    assert event["reason"] == "soft_tool_call_target"
    assert event["tool_calls_executed"] == 3
    assert event["soft_tool_call_target"] == 3
    assert event["max_tool_calls"] == 64
    assert event["signal"] == NOTICE_MARKER
    assert event["run_id"] == trace.run_id
    # nothing blocked and nothing terminated by this event
    assert "finalization" not in str(event)
    assert "* Soft effort target: reached" in _markdown(trace)


def test_tools_still_execute_after_the_soft_target_is_crossed():
    specialist, _ = _long_run(soft_target=3, tool_calls_count=5)
    requests = specialist.agent._llm_client.requests

    # before the crossing: no notice yet (three requests, three
    # executions — the target is reached only after the third)
    assert _notice_occurrences(requests[0]) == 0
    assert _notice_occurrences(requests[1]) == 0
    assert _notice_occurrences(requests[2]) == 0
    # from the crossing on: the notice is present, exactly once per
    # request — appended once, carried forward, never repeated
    assert _notice_occurrences(requests[3]) == 1
    assert _notice_occurrences(requests[-1]) == 1
    # and execution continued past it: more tool calls followed
    assert specialist.agent.last_trace.summary.tool_calls_executed == 5


def test_the_notice_is_an_execution_state_message_not_a_new_contract():
    specialist, _ = _long_run(soft_target=3)
    notice = None
    for request in specialist.agent._llm_client.requests:
        for message in request.messages:
            if message.role != "user":
                continue
            if NOTICE_ANCHOR in (message.content or ""):
                notice = message
    assert notice is not None
    assert notice.role == "user"
    assert NOTICE_MARKER in notice.content
    assert "3 tool call(s) have executed" in notice.content
    assert "soft investigation target is 3 of 64" in notice.content
    assert "reassessment point, not a cutoff" in notice.content
    # counters only — never message content, never an answer
    assert "final answer" not in notice.content.lower()


# ---------------------------------------------------------------------------
# 3 — the hard ceiling is unchanged, and still the only ceiling
# ---------------------------------------------------------------------------


def test_the_m17_hard_ceiling_stays_64_and_the_soft_target_is_8():
    assert SOFT_TOOL_CALL_TARGET == 16
    assert rq.M17_MAX_TOOL_CALLS == 64
    assert rq.M17_EXECUTION_CONFIG.max_tool_calls == 64
    assert rq.M17_ORCHESTRATION_CONFIG.max_total_tool_calls == 64
    assert rq.M17_EXECUTION_CONFIG.soft_tool_call_target == 8
    assert rq.M17_EXECUTION_CONFIG.finalize_on_boundary is True
    # the default configuration keeps its own budget and the shared
    # soft target — a soft signal never silently raises the ceiling
    default = AgentExecutionConfig()
    assert default.max_tool_calls == 8
    assert default.soft_tool_call_target == SOFT_TOOL_CALL_TARGET


def test_crossing_the_soft_target_does_not_replace_the_hard_ceiling():
    specialist, _ = _long_run(
        soft_target=1, max_tool_calls=2, tool_calls_count=3
    )
    trace = specialist.agent.last_trace
    summary = trace.summary

    # the soft target was crossed (signal exists) AND the hard
    # ceiling still blocked the third execution
    assert trace.effort_threshold_reached is True
    assert summary.tool_calls_executed == 2
    assert summary.tool_calls_blocked == 1
    operations = [event["operation"] for event in trace.lifecycle_events]
    assert "execution.boundary_reached" in operations
    assert "execution.effort_threshold_reached" in operations
    # the run still closed through the ordinary finalization path
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    assert operations[-1] == "finalization.completed"


def test_a_soft_target_above_the_budget_never_fires():
    specialist = build_investigator(
        *read_then_report(),
        execution_config=AgentExecutionConfig(
            max_tool_calls=1, soft_tool_call_target=64
        ),
    )
    result = specialist.investigate(
        QUESTION, "example-repository", "c" * 40
    )
    assert result is not None
    trace = specialist.agent.last_trace
    assert trace.effort_threshold_reached is False
    assert trace.lifecycle_events == []


def test_the_soft_target_configuration_is_validated():
    with pytest.raises(ValueError, match="soft_tool_call_target"):
        AgentExecutionConfig(soft_tool_call_target=0)
    with pytest.raises(ValueError, match="soft_tool_call_target"):
        AgentExecutionConfig(soft_tool_call_target="16")
    with pytest.raises(ValueError, match="soft_tool_call_target"):
        AgentExecutionConfig(soft_tool_call_target=True)


# ---------------------------------------------------------------------------
# 4 — duplicate searches: one sufficient search, then finalize
# ---------------------------------------------------------------------------


def test_a_sufficient_broad_search_finalizes_without_a_second_search():
    specialist = build_investigator(
        tool_calls(call("search_repository", {"query": "agent"})),
        structured(report_payload()),
    )

    result = specialist.investigate(
        QUESTION, "example-repository", "c" * 40
    )

    assert result is not None
    trace = specialist.agent.last_trace
    assert trace.summary.tool_calls_executed == 1
    assert len(trace.iterations) == 2  # search, then the typed report
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    # the scripted queue would have raised on any extra model call,
    # and only one tool result was ever produced
    assert len(trace.iterations[0].tool_results) == 1
    assert trace.effort_threshold_reached is False


def test_the_prompt_carries_the_objective_and_efficiency_discipline():
    section = _strategy_section()
    # objectives derived from the assignment, from the same reasoning
    # turn — no separate step, no extra model call
    assert "start with the objectives, not the tools" in section
    assert "derived from the assignment, never from a fixed list" in section
    assert "would need its own model call" in section
    assert "mark an objective satisfied, then leave it" in section
    # search only when a concrete unanswered question exists
    assert "search only for missing evidence" in section
    assert "if no concrete" in section
    assert "uncertainty by itself is not sufficient justification" in section
    # the one-time execution-state notice is anticipated in the prompt
    assert NOTICE_MARKER in section
    assert "It is not a cutoff" in section
    assert "never makes insufficient evidence" in section
    # the vocabulary contract still holds
    assert PROMPT_VERSION == "V1"
    lowered = ARCHITECTURE_INVESTIGATOR_PROMPT_V1.lower()
    for banned in ("orchestrat", "planner", "adop"):
        assert banned not in lowered
