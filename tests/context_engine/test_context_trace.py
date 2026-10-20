"""M15 — trace integration (§27/§56): ONE trace, context observable.

 * ``AgentTrace.context_events`` is additive — a trace without
   context events serializes and renders EXACTLY as before (no new
   key, no new section): frozen M13/M14 trace behavior stays intact
 * select + render events are representable on the same trace with
   their observable metadata (context_id, step, counts, budget,
   reasons, decisions)
 * payloads are clipped like every other recorded block and contain
   no hidden-reasoning fields
 * markdown projection: one deterministic ``## Context`` section
   AFTER the Memory section (selection happens after retrieval)
"""

from __future__ import annotations

from repo_intelligence.agent.trace import (
    BANNED_REASONING_FIELDS,
    AgentTrace,
    render_markdown,
)


def _trace():
    return AgentTrace.start(user_question="example question")


def _select_event(operation="context.select", **overrides):
    event = {
        "operation": operation,
        "run_id": "run_1",
        "context_id": "ctx_ab12",
        "step_id": "001",
        "candidate_count": 9,
        "selected_count": 6,
        "excluded_count": 3,
        "budget": {
            "max_items": 20,
            "selected_items": 6,
            "remaining_items": 14,
            "max_characters": 12000,
            "estimated_characters": 4830,
            "remaining_characters": 7170,
        },
        "reasons": {"TASK_DIRECT": 1, "OUT_OF_SCOPE": 2},
    }
    event.update(overrides)
    return event


def test_trace_without_context_events_keeps_exact_legacy_shape():
    trace = _trace()
    trace.terminate("FINAL_ANSWER", final_answer="done")
    trace.finalize()
    payload = trace.to_dict()
    assert "context_events" not in payload
    markdown = render_markdown(trace)
    assert "## Context" not in markdown


def test_events_are_appended_in_order_and_serialized_only_when_present():
    trace = _trace()
    trace.record_context_event(_select_event())
    trace.record_context_event(
        {"operation": "context.render", "context_id": "ctx_ab12",
         "step_id": "001", "characters": 5120}
    )
    payload = trace.to_dict()
    assert [e["operation"] for e in payload["context_events"]] == [
        "context.select",
        "context.render",
    ]
    # deep-copied: mutating the caller's dict never mutates the trace
    event = {"operation": "context.select"}
    trace.record_context_event(event)
    event["operation"] = "tampered"
    assert trace.context_events[-1]["operation"] == "context.select"


def test_event_payloads_are_clipped_like_every_other_block():
    trace = _trace()
    trace.record_context_event(
        {"operation": "context.select", "context_id": "ctx_" + "z" * 5000}
    )
    recorded = trace.context_events[0]
    assert len(recorded["context_id"]) < 5000
    assert "[trace clipped" in recorded["context_id"]


def test_select_and_render_events_render_in_one_context_section():
    trace = _trace()
    trace.record_context_event(_select_event())
    trace.record_context_event(
        {"operation": "context.render", "context_id": "ctx_ab12",
         "step_id": "001", "characters": 5120}
    )
    markdown = render_markdown(trace)
    assert markdown.count("## Context") == 1
    section = markdown[markdown.index("## Context"):]
    assert "`context.select`" in section
    assert "`ctx_ab12`" in section
    assert "step `001`" in section
    assert "9 candidates \u2192 6 selected / 3 excluded" in section
    assert "budget 4830/12000 chars, 6/20 items" in section
    assert "reasons: OUT_OF_SCOPE=2, TASK_DIRECT=1" in section
    assert "`context.render`" in section
    assert "5120 characters" in section
    # the Context section precedes Iteration sections (boundary
    # selection happens before the loop)
    if "## Iteration" in markdown:
        assert markdown.index("## Context") < markdown.index(
            "## Iteration"
        )


def test_context_section_follows_memory_section():
    trace = _trace()
    trace.record_memory_event({"operation": "memory.retrieve", "count": 2})
    trace.record_context_event(_select_event())
    markdown = render_markdown(trace)
    assert markdown.index("## Memory") < markdown.index("## Context")


def test_error_events_render_with_their_structured_code():
    trace = _trace()
    trace.record_context_event(
        {
            "operation": "context.select",
            "step_id": "001",
            "error": {
                "code": "CONTEXT_BUDGET_EXCEEDED",
                "message": "no eligible candidate fits",
            },
        }
    )
    markdown = render_markdown(trace)
    assert (
        "ERROR CONTEXT_BUDGET_EXCEEDED: no eligible candidate fits"
        in markdown
    )


def test_context_events_contain_no_hidden_reasoning():
    trace = _trace()
    trace.record_context_event(_select_event())
    trace.record_context_event(
        {"operation": "context.render", "characters": 100}
    )
    for event in trace.context_events:
        for field in BANNED_REASONING_FIELDS:
            assert field not in event
    markdown = render_markdown(trace)
    for field in BANNED_REASONING_FIELDS:
        assert field not in markdown


def test_empty_event_still_renders_its_operation():
    trace = _trace()
    trace.record_context_event({"operation": "context.snapshot"})
    markdown = render_markdown(trace)
    assert "`context.snapshot`" in markdown
