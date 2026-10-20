"""M10 — planner LLM contract (spec §11/§27/§28).

The planner is a provider-neutral structured-output request built from
the existing M8.9 machinery — never an OpenAI-specific object, never a
tool call:

    build_planning_request(objective, architecture_context)
        messages   system planning instruction + user objective +
                   the ALREADY-GENERATED M6 Architecture Context
                   passed directly (spec §8 — the planner never calls
                   get_architecture_context as a tool)
        tools      [] — the planner creates a plan; it executes
                   nothing (spec §12)
        structured one StructuredOutputSpec derived from the
                   InvestigationPlan Pydantic model (spec §28)

    plan_from_response(response)
        decoded payload → typed InvestigationPlan, or an EXPLICIT
        StructuredOutputValidationError (no defaults, no repair,
        spec §29)

    load_architecture_context(...)
        existing M6 artifact resolution (context.models) → the planner
        input text plus the compact trace record (artifact path,
        schema version, pinned commit, snapshot id — spec §26)
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.context.models import ArchitectureContextError
from repo_intelligence.llm import (
    LLMResponse,
    Message,
    StructuredOutputValidationError,
)
from repo_intelligence.planning import (
    PLANNING_SYSTEM_INSTRUCTION,
    InvestigationPlan,
    build_planning_request,
    load_architecture_context,
    plan_from_response,
)

OBJECTIVE = "Explain how the agents, tools, and MCP servers are connected."
CONTEXT_TEXT = "# Architecture Context\n\n- Contract: `architecture_context.v1`\n"


def _structured_response(payload):
    if isinstance(payload, str):
        content = payload
        decoded = json.loads(payload) if payload.strip() else None
    else:
        content = json.dumps(payload)
        decoded = payload
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
        structured_output=decoded,
    )


# -- build_planning_request ------------------------------------------------


def test_request_is_provider_neutral_messages_only():
    request = build_planning_request(OBJECTIVE, CONTEXT_TEXT)
    assert [message.role for message in request.messages] == [
        "system",
        "user",
    ]
    assert request.messages[0].content == PLANNING_SYSTEM_INSTRUCTION
    user_content = request.messages[1].content
    assert OBJECTIVE in user_content
    assert CONTEXT_TEXT in user_content


def test_architecture_context_is_direct_input_not_a_tool():
    request = build_planning_request(OBJECTIVE, CONTEXT_TEXT)
    # no tools: the planner cannot call get_architecture_context (§8)
    assert request.tools == []
    assert CONTEXT_TEXT in request.messages[1].content


def test_request_without_context_still_carries_the_objective():
    request = build_planning_request(OBJECTIVE)
    assert len(request.messages) == 2
    assert request.messages[1].content == OBJECTIVE


def test_request_derives_the_structured_spec_from_the_plan_model():
    request = build_planning_request(OBJECTIVE, CONTEXT_TEXT)
    spec = request.structured_output
    assert spec is not None
    assert spec.name == "InvestigationPlan"
    assert set(spec.schema.get("required") or []) >= {"objective", "steps"}
    # Pydantic v1 uses ``definitions`` while Pydantic v2 uses ``$defs``.
    definitions = spec.schema.get("definitions") or spec.schema.get("$defs") or {}
    step_schema = definitions.get("PlanStep", {}).get("properties", {})
    assert "id" in step_schema and "goal" in step_schema


def test_planning_instruction_forbids_executing_and_wants_goal_level():
    assert "goal-level" in PLANNING_SYSTEM_INSTRUCTION
    assert "typically 3-4 steps" in PLANNING_SYSTEM_INSTRUCTION
    assert "overlapping searches" in PLANNING_SYSTEM_INSTRUCTION
    assert "do not execute" in PLANNING_SYSTEM_INSTRUCTION.lower()


# -- plan_from_response (planning contract tests 1, 2, 5 of spec §38) -----


def test_valid_structured_response_produces_an_investigation_plan():
    payload = {
        "objective": OBJECTIVE,
        "steps": [
            {"id": "P1", "goal": "Identify relevant agent definitions"},
            {"id": "P2", "goal": "Map agents to the tools they use"},
        ],
    }
    plan = plan_from_response(_structured_response(payload))
    assert isinstance(plan, InvestigationPlan)
    assert plan.objective == payload["objective"]
    assert [step.id for step in plan.steps] == ["P1", "P2"]


def test_missing_structured_payload_fails_explicitly():
    response = LLMResponse(
        message=Message(role="assistant", content="I cannot plan."),
        finish_reason="stop",
    )
    with pytest.raises(StructuredOutputValidationError):
        plan_from_response(response)


def test_non_object_payload_fails_explicitly():
    response = LLMResponse(
        message=Message(role="assistant", content="[]"),
        finish_reason="stop",
        structured_output=["not", "an", "object"],
    )
    with pytest.raises(StructuredOutputValidationError):
        plan_from_response(response)


def test_payload_missing_objective_fails_schema_validation():
    with pytest.raises(StructuredOutputValidationError):
        plan_from_response(_structured_response({"steps": [{"id": "P1"}]}))


def test_payload_with_malformed_steps_fails_schema_validation():
    with pytest.raises(StructuredOutputValidationError):
        plan_from_response(
            _structured_response({"objective": OBJECTIVE, "steps": "P1"})
        )
    with pytest.raises(StructuredOutputValidationError):
        plan_from_response(
            _structured_response(
                {"objective": OBJECTIVE, "steps": [{"id": "P1"}]}
            )
        )


def test_objective_empty_string_parses_schema_but_is_rejected_by_validation():
    # schema-level: objective present but blank — deterministic app
    # validation (validate_plan) owns this check, not the parser
    plan = plan_from_response(
        _structured_response({"objective": "", "steps": [{"id": "P1", "goal": "g"}]})
    )
    assert plan.objective == ""


# -- load_architecture_context (spec §7/§26/§36) ---------------------------


def _write_context_artifact(root):
    snapshot = "main_2026-09-22_ef11dac"
    context_dir = root / snapshot / "architecture_context"
    context_dir.mkdir(parents=True)
    text = (
        "# Architecture Context\n\n"
        "- Contract: `architecture_context.v1`\n\n"
        "## Repository\n\n"
        "- Name: `sample`\n"
        "- Pinned commit: `ef11dac8618b6344d6a5e4add6ef51cd2811d7e0`\n"
        "- Snapshot: `{}`\n".format(snapshot)
    )
    (context_dir / "architecture_context.md").write_text(
        text, encoding="utf-8"
    )
    # discovery requires both M2 facts and M4 architecture artifacts
    (root / snapshot / "facts").mkdir(parents=True)
    (root / snapshot / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / snapshot / "architecture").mkdir(parents=True)
    (root / snapshot / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    return snapshot, text


def test_load_architecture_context_returns_text_and_compact_record(tmp_path):
    snapshot, text = _write_context_artifact(tmp_path)
    context_text, record = load_architecture_context(output_root=tmp_path)
    assert context_text == text
    assert record["schema_version"] == "architecture_context.v1"
    assert record["commit_sha"] == (
        "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
    )
    assert record["snapshot_id"] == snapshot
    assert record["artifact"].endswith("architecture_context.md")
    assert "architecture_context" in record["artifact"]


def test_load_architecture_context_missing_artifact_fails(tmp_path):
    with pytest.raises(ArchitectureContextError):
        load_architecture_context(output_root=tmp_path)


def test_load_architecture_context_empty_artifact_fails(tmp_path):
    snapshot, _ = _write_context_artifact(tmp_path)
    (tmp_path / snapshot / "architecture_context" / "architecture_context.md").write_text(
        "   ", encoding="utf-8"
    )
    with pytest.raises(ArchitectureContextError):
        load_architecture_context(output_root=tmp_path)
