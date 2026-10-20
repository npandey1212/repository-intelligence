"""M14 — trace integration (§32/§33): ONE trace, memory observable.

 * ``AgentTrace.memory_events`` is additive — a trace without memory
   events serializes and renders EXACTLY as before (no new key, no
   new section): frozen M8–M13 trace behavior stays intact
 * every lifecycle operation is representable on the same trace —
   retrieve / store / verify / supersede / invalidate — with the §32
   metadata (memory_id, agent_id, run_id, repository, commit,
   operation, status)
 * payloads are clipped like every other recorded block and contain
   no hidden-reasoning fields
 * markdown projection: one deterministic ``## Memory`` section
"""

from __future__ import annotations

from repo_intelligence.agent.trace import (
    BANNED_REASONING_FIELDS,
    AgentTrace,
    render_markdown,
)


def _trace():
    return AgentTrace.start(user_question="example question")


def test_trace_without_memory_events_keeps_exact_legacy_shape():
    trace = _trace()
    trace.terminate("FINAL_ANSWER", final_answer="done")
    trace.finalize()
    payload = trace.to_dict()
    assert "memory_events" not in payload
    markdown = render_markdown(trace)
    assert "## Memory" not in markdown


def test_events_are_appended_in_order_and_serialized_only_when_present():
    trace = _trace()
    trace.record_memory_event({"operation": "memory.retrieve", "count": 0})
    trace.record_memory_event(
        {"operation": "memory.store", "memory_id": "mem_x"}
    )
    payload = trace.to_dict()
    assert [e["operation"] for e in payload["memory_events"]] == [
        "memory.retrieve",
        "memory.store",
    ]
    # deep-copied: mutating the caller's dict never mutates the trace
    event = {"operation": "memory.verify"}
    trace.record_memory_event(event)
    event["operation"] = "tampered"
    assert trace.memory_events[-1]["operation"] == "memory.verify"


def test_event_payloads_are_clipped_like_every_other_block():
    trace = _trace()
    trace.record_memory_event(
        {"operation": "memory.store", "memory_id": "mem_" + "z" * 5000}
    )
    recorded = trace.memory_events[0]
    assert len(recorded["memory_id"]) < 5000
    assert "[trace clipped" in recorded["memory_id"]


def test_every_lifecycle_operation_is_representable():
    """§32: retrieval, storage, verification, and superseding are all
    visible on the ONE execution trace — including the
    application-driven supersede/invalidate decisions."""
    trace = _trace()
    events = [
        {
            "operation": "memory.retrieve",
            "memory_ids": ["mem_old"],
            "count": 1,
            "repository": "example-repository",
            "commit": "a" * 40,
        },
        {
            "operation": "memory.verify",
            "memory_id": "mem_old",
            "stale": True,
            "status": "verified",
            "agent_id": "repo_intelligence.agent",
            "run_id": trace.run_id,
        },
        {
            "operation": "memory.supersede",
            "memory_id": "mem_old",
            "status": "superseded",
        },
        {
            "operation": "memory.store",
            "memory_id": "mem_new",
            "status": "verified",
        },
        {
            "operation": "memory.invalidate",
            "memory_id": "mem_new",
            "status": "invalid",
        },
    ]
    for event in events:
        trace.record_memory_event(event)
    markdown = render_markdown(trace)
    assert "## Memory" in markdown
    assert "`memory.retrieve` \u2014 1 retrieved" in markdown
    assert "`memory.verify`" in markdown and "stale: yes" in markdown
    assert "`memory.supersede`" in markdown
    assert "`memory.store`" in markdown
    assert "`memory.invalidate`" in markdown and "status `invalid`" in markdown


def test_verification_renders_as_a_tri_state_not_a_boolean():
    """M14 §26: the scope check is fresh / stale / unknown — the
    trace renders the status itself, and unknown is never flattened
    to "stale: no" (which would read as fresh)."""
    trace = _trace()
    for status in ("fresh", "stale", "unknown"):
        trace.record_memory_event(
            {
                "operation": "memory.verify",
                "memory_id": "mem_{}".format(status),
                "verification_status": status,
                "stale": status == "stale",
            }
        )
    markdown = render_markdown(trace)
    assert "verification `fresh`" in markdown
    assert "verification `stale`" in markdown
    assert "verification `unknown`" in markdown
    assert "stale: no" not in markdown


def test_error_events_render_with_their_structured_code():
    trace = _trace()
    trace.record_memory_event(
        {
            "operation": "memory.store",
            "error": {
                "code": "MEMORY_STORE_UNAVAILABLE",
                "message": "memory store unavailable: disk full",
            },
        }
    )
    markdown = render_markdown(trace)
    assert "ERROR MEMORY_STORE_UNAVAILABLE" in markdown


def test_memory_events_contain_no_hidden_reasoning():
    trace = _trace()
    trace.record_memory_event(
        {
            "operation": "memory.store",
            "memory_id": "mem_x",
            "statement": "observable content only",
        }
    )
    recorded = trace.memory_events[0]
    for banned in BANNED_REASONING_FIELDS:
        assert banned not in recorded
    payload = trace.to_dict()
    assert "reasoning" not in str(payload)
