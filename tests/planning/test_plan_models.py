"""M10 — plan data model (InvestigationPlan / PlanStep).

The smallest useful structured plan (spec §13): objective + ordered
goal-level steps with an application-managed lifecycle status. The
model is the structured-output contract the planner returns (M8.9
machinery, spec §28), so:

 1  ``objective`` and ``steps`` are REQUIRED (a planner payload that
    omits them fails schema validation explicitly — never defaulted)
 2  ``status``/``result`` are executor-managed fields with deterministic
    defaults (``pending`` / ``None``) — the lifecycle vocabulary is the
    same six-value set spec §13 suggests
 3  planner-supplied extra keys (notes, "reasoning", tool hints …) are
    ignored by parsing and never reach ``plan_to_dict`` — the trace
    payload carries no hidden-reasoning fields (spec §24, asserted
    against the trace's own banned vocabulary)
 4  ``plan_to_dict`` is the trace projection: JSON-ready, validation
    outcome nestable, lifecycle mutations visible (spec §21/§40)

No scheduling, dependency, concurrency, or optimization fields exist
(spec §16/§42).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.planning import (
    PLAN_STATUSES,
    PLAN_STATUS_PENDING,
    InvestigationPlan,
    plan_to_dict,
)

VALID_PLAN = {
    "objective": "Explain how the agents, tools, and MCP servers connect.",
    "steps": [
        {"id": "P1", "goal": "Identify relevant agent definitions"},
        {"id": "P2", "goal": "Map agents to the tools they use"},
    ],
}


# -- construction / schema contract ---------------------------------------


def test_plan_requires_objective_and_steps():
    with pytest.raises(ValidationError):
        InvestigationPlan.parse_obj({"steps": []})
    with pytest.raises(ValidationError):
        InvestigationPlan.parse_obj({"objective": "x"})


def test_valid_plan_parses_from_planner_payload():
    plan = InvestigationPlan.parse_obj(VALID_PLAN)
    assert plan.objective == VALID_PLAN["objective"]
    assert [step.id for step in plan.steps] == ["P1", "P2"]
    assert [step.goal for step in plan.steps] == [
        "Identify relevant agent definitions",
        "Map agents to the tools they use",
    ]


def test_step_lifecycle_defaults_are_pending():
    plan = InvestigationPlan.parse_obj(VALID_PLAN)
    assert plan.status == PLAN_STATUS_PENDING
    for step in plan.steps:
        assert step.status == PLAN_STATUS_PENDING
        assert step.result is None


def test_planner_supplied_lifecycle_values_parse_but_default_payload_is_pending():
    plan = InvestigationPlan.parse_obj(
        {
            "objective": "o",
            "steps": [
                {"id": "P1", "goal": "g1", "status": "completed", "result": "r"}
            ],
        }
    )
    assert plan.steps[0].status == "completed"
    assert plan.steps[0].result == "r"


def test_unknown_planner_keys_are_ignored():
    plan = InvestigationPlan.parse_obj(
        {
            "objective": "o",
            "reasoning": "hidden planner thoughts",
            "steps": [
                {
                    "id": "P1",
                    "goal": "g1",
                    "notes": "extra",
                    "tool": "search_repository",
                }
            ],
        }
    )
    payload = plan_to_dict(plan)
    assert "reasoning" not in payload
    assert "notes" not in (payload["steps"][0] or {})
    assert "tool" not in (payload["steps"][0] or {})


def test_status_vocabulary_is_the_six_suggested_values():
    assert PLAN_STATUSES == frozenset(
        {
            "pending",
            "in_progress",
            "completed",
            "failed",
            "invalidated",
            "blocked",
        }
    )


# -- trace projection (plan_to_dict) --------------------------------------


def test_plan_to_dict_shape_is_json_ready():
    plan = InvestigationPlan.parse_obj(VALID_PLAN)
    payload = plan_to_dict(plan)
    assert payload == {
        "objective": VALID_PLAN["objective"],
        "status": "pending",
        "steps": [
            {
                "id": "P1",
                "goal": "Identify relevant agent definitions",
                "status": "pending",
            },
            {
                "id": "P2",
                "goal": "Map agents to the tools they use",
                "status": "pending",
            },
        ],
    }
    assert json.loads(json.dumps(payload)) == payload


def test_plan_to_dict_nests_validation_outcome():
    plan = InvestigationPlan.parse_obj(VALID_PLAN)
    payload = plan_to_dict(plan, validation={"status": "SUCCESS"})
    assert payload["validation"] == {"status": "SUCCESS"}


def test_plan_to_dict_reflects_lifecycle_mutation():
    plan = InvestigationPlan.parse_obj(VALID_PLAN)
    plan.status = "in_progress"
    plan.steps[0].status = "completed"
    plan.steps[0].result = "found 3 agent definition files"
    payload = plan_to_dict(plan)
    assert payload["status"] == "in_progress"
    assert payload["steps"][0]["status"] == "completed"
    assert payload["steps"][0]["result"] == "found 3 agent definition files"
    # untouched step stays pending
    assert payload["steps"][1]["status"] == "pending"
    assert "result" not in payload["steps"][1]


def test_plan_payload_carries_no_banned_reasoning_fields():
    plan = InvestigationPlan.parse_obj(VALID_PLAN)
    plan.steps[0].status = "completed"
    blob = json.dumps(plan_to_dict(plan, validation={"status": "SUCCESS"}))
    for banned in BANNED_REASONING_FIELDS:
        assert '"{}"'.format(banned) not in blob
