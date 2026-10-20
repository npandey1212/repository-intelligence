"""M10 — additive plan fields on the EXISTING AgentTrace (spec §21–§26).

Plan information is run-level state on the one trace — never a second
trace object:

    AgentTrace.plan                     structured plan payload
                                        (objective · status ·
                                        validation · steps lifecycle)
    AgentTrace.architecture_context     compact M6 record: artifact,
                                        schema version, pinned commit,
                                        snapshot id (spec §26)
    AgentIteration.plan_step_id         which plan step this iteration
                                        executed (spec §22) — set only
                                        by the Agent's plan mode

Additive guarantees asserted here: a trace that never planned has the
exact payload it had before M10 (no ``plan``/``architecture_context``
keys, no ``plan_step_id`` on iterations), the Markdown projection
renders the plan between the question and the iterations, the JSON
projection carries the same plan, and no banned hidden-reasoning
field can appear anywhere in either projection (spec §24).
"""

from __future__ import annotations

import json

from repo_intelligence.agent import AgentTrace, render_markdown
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.llm import LLMRequest, Message

PLAN_PAYLOAD = {
    "objective": "Explain how the agents, tools, and MCP servers connect.",
    "status": "completed",
    "validation": {"status": "SUCCESS"},
    "steps": [
        {
            "id": "P1",
            "goal": "Identify relevant agent definitions",
            "status": "completed",
            "result": "found agent definition files",
        },
        {
            "id": "P2",
            "goal": "Map agents to the tools they use",
            "status": "completed",
        },
    ],
}

CONTEXT_RECORD = {
    "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/"
    "architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
    "snapshot_id": "main_2026-09-22_ef11dac",
}


def _request():
    return LLMRequest(
        messages=[Message(role="user", content="objective")],
        tools=[],
    )


def _open_trace(**kwargs):
    trace = AgentTrace.start(user_question="objective", **kwargs)
    return trace


# -- plan_step_id on iterations (spec §22) ---------------------------------


def test_iteration_plan_step_id_is_optional_and_additive():
    trace = _open_trace()
    plain = trace.begin_iteration(_request())
    tagged = trace.begin_iteration(_request(), plan_step_id="P1")
    assert plain.plan_step_id is None
    assert "plan_step_id" not in plain.to_dict()
    assert tagged.plan_step_id == "P1"
    assert tagged.to_dict()["plan_step_id"] == "P1"


def test_iteration_numbering_is_unchanged_when_plan_step_id_is_used():
    trace = _open_trace()
    trace.begin_iteration(_request(), plan_step_id="P1")
    trace.begin_iteration(_request(), plan_step_id="P2")
    assert [iteration.iteration for iteration in trace.iterations] == [1, 2]


# -- run-level plan record (spec §21) --------------------------------------


def test_trace_without_plan_has_no_plan_keys():
    trace = _open_trace()
    trace.finalize()
    payload = trace.to_dict()
    assert "plan" not in payload
    assert "architecture_context" not in payload


def test_record_plan_exposes_plan_in_json_projection():
    trace = _open_trace()
    trace.record_plan(PLAN_PAYLOAD)
    trace.finalize()
    payload = trace.to_dict()
    assert payload["plan"] == PLAN_PAYLOAD
    assert json.loads(trace.to_json())["plan"] == PLAN_PAYLOAD
    # plan sits before the iterations (chronological reconstruction)
    keys = list(payload.keys())
    assert keys.index("plan") < keys.index("iterations")


def test_record_architecture_context_is_a_compact_run_level_record():
    trace = _open_trace()
    trace.record_architecture_context(CONTEXT_RECORD)
    trace.record_plan(PLAN_PAYLOAD)
    trace.finalize()
    payload = trace.to_dict()
    assert payload["architecture_context"] == CONTEXT_RECORD
    assert payload["plan"] == PLAN_PAYLOAD


def test_recorded_plan_is_copied_not_aliased():
    trace = _open_trace()
    source = dict(PLAN_PAYLOAD)
    trace.record_plan(source)
    source["objective"] = "mutated afterwards"
    assert trace.plan["objective"] == PLAN_PAYLOAD["objective"]


def test_long_plan_result_is_clipped_by_trace_policy():
    trace = _open_trace()
    payload = dict(PLAN_PAYLOAD)
    payload["steps"] = [
        {
            "id": "P1",
            "goal": "g",
            "status": "completed",
            "result": "x" * 5000,
        }
    ]
    trace.record_plan(payload)
    recorded = trace.plan["steps"][0]["result"]
    assert recorded.startswith("x" * 2000)
    assert "[trace clipped:" in recorded


# -- markdown projection (spec §23/§43 Step 6) -----------------------------


def test_markdown_renders_plan_section_before_iterations():
    trace = _open_trace()
    trace.record_architecture_context(CONTEXT_RECORD)
    trace.record_plan(PLAN_PAYLOAD)
    iteration = trace.begin_iteration(_request(), plan_step_id="P1")
    iteration.close(
        response_type="FINAL_ANSWER",
        decision={"type": "FINAL_ANSWER"},
        reason_summary="Model answered.",
    )
    trace.terminate("FINAL_ANSWER", final_answer="done")
    trace.finalize()
    markdown = render_markdown(trace)
    assert "## Plan" in markdown
    assert markdown.index("## Plan") < markdown.index("## Iteration 1")
    assert PLAN_PAYLOAD["objective"] in markdown
    assert "Identify relevant agent definitions" in markdown
    assert "found agent definition files" in markdown
    assert "`P1`" in markdown and "**completed**" in markdown
    assert "Plan validation: SUCCESS" in markdown
    assert CONTEXT_RECORD["artifact"] in markdown
    assert CONTEXT_RECORD["commit_sha"] in markdown


def test_markdown_renders_plan_step_correlation_on_the_iteration():
    trace = _open_trace()
    trace.record_plan(PLAN_PAYLOAD)
    iteration = trace.begin_iteration(_request(), plan_step_id="P2")
    iteration.close(
        response_type="FINAL_ANSWER",
        decision={"type": "FINAL_ANSWER"},
        reason_summary="Model answered.",
    )
    trace.terminate("FINAL_ANSWER", final_answer="done")
    trace.finalize()
    markdown = render_markdown(trace)
    assert "Plan step: `P2`" in markdown


def test_markdown_without_plan_is_unchanged_in_shape():
    trace = _open_trace()
    iteration = trace.begin_iteration(_request())
    iteration.close(
        response_type="FINAL_ANSWER",
        decision={"type": "FINAL_ANSWER"},
        reason_summary="Model answered.",
        final_answer="done",
        continuation={
            "decision": "FINAL_ANSWER",
            "reason": "LLM returned final answer",
            "termination_reason": "FINAL_ANSWER",
        },
    )
    trace.terminate("FINAL_ANSWER", final_answer="done")
    trace.finalize()
    markdown = render_markdown(trace)
    assert "## Plan" not in markdown
    assert "Plan step:" not in markdown
    assert "Final answer received from LLM." in markdown


def test_markdown_renders_plan_step_result_for_a_step_boundary():
    trace = _open_trace()
    trace.record_plan(PLAN_PAYLOAD)
    iteration = trace.begin_iteration(_request(), plan_step_id="P1")
    iteration.close(
        response_type="FINAL_ANSWER",
        decision={"type": "FINAL_ANSWER"},
        reason_summary="Plan step P1 result.",
        final_answer="step one result",
        continuation={
            "decision": "CONTINUE",
            "reason": "plan step P1 completed; continue plan execution",
            "next_iteration": 2,
        },
    )
    trace.terminate("FINAL_ANSWER", final_answer="final")
    trace.finalize()
    markdown = render_markdown(trace)
    assert "Plan step `P1` result recorded." in markdown
    assert "Continue plan execution." in markdown
    assert "Tool result will be returned to the LLM." not in markdown


def test_failed_plan_validation_renders_in_the_plan_section():
    trace = _open_trace()
    trace.record_plan(
        {
            "objective": "o",
            "status": "invalidated",
            "validation": {
                "status": "FAILED",
                "stage": "application",
                "error": "duplicate step id 'P1'",
            },
            "steps": [],
        }
    )
    trace.terminate("PLAN_VALIDATION_FAILED")
    trace.finalize()
    markdown = render_markdown(trace)
    assert "Plan validation: FAILED" in markdown
    assert "duplicate step id 'P1'" in markdown


# -- no hidden reasoning (spec §24) ----------------------------------------


def test_serialized_trace_with_plan_carries_no_banned_reasoning_fields():
    trace = _open_trace()
    trace.record_architecture_context(CONTEXT_RECORD)
    trace.record_plan(
        {
            "objective": "o",
            "status": "completed",
            "validation": {"status": "SUCCESS"},
            "steps": [
                {
                    "id": "P1",
                    "goal": "g",
                    "status": "completed",
                    "result": "r",
                }
            ],
        }
    )
    iteration = trace.begin_iteration(_request(), plan_step_id="P1")
    iteration.close(
        response_type="FINAL_ANSWER",
        decision={"type": "FINAL_ANSWER"},
        reason_summary="Model answered.",
    )
    trace.terminate("FINAL_ANSWER", final_answer="done")
    trace.finalize()
    for projection in (trace.to_json(), render_markdown(trace)):
        for banned in BANNED_REASONING_FIELDS:
            assert '"{}"'.format(banned) not in projection
