"""M17 role assignment (spec §14/§15/§36/§37).

Deterministic, application-owned, observable: every assignment carries
a human-readable reason; overrides are explicit and rejected when they
name an unsupported role; two post-conditions keep the workflow shape
(evidence first, interpretation last); a plan with no analysis step can
never fabricate an architectural answer.
"""

from __future__ import annotations

import pytest

from repo_intelligence.orchestration.roles import (
    ANALYST,
    INVESTIGATOR,
    RoleAssignment,
    assign_roles,
)
from repo_intelligence.planning.models import PlanStep


def steps(*goals) -> list:
    return [
        PlanStep(id="S{}".format(index + 1), goal=goal)
        for index, goal in enumerate(goals)
    ]


# -- default classifier ----------------------------------------------------


def test_evidence_goals_default_to_investigator():
    for goal in (
        "Identify the application boundary",
        "Investigate how requests enter the system",
        "Understand relationships between modules",
    ):
        assignment = assign_roles(steps(goal))[0]
        assert assignment.role == INVESTIGATOR, goal
        assert assignment.reason == (
            "step requires repository evidence collection"
        )
        assert assignment.reason.strip()


def test_synthesis_goals_default_to_analyst():
    assignments = assign_roles(
        steps("Read the entry points", "Explain how the boundary is enforced")
    )
    assert assignments[0].role == INVESTIGATOR
    assert assignments[1].role == ANALYST
    assert assignments[1].reason == (
        "architecture interpretation requires Analyst role"
    )


def test_interpretation_vocabulary_is_deliberately_narrow():
    for goal in (
        "Summarize the module layout",
        "Synthesize the architectural story",
        "What does load_config() mean for startup",
        "Produce the final conclusions",
    ):
        assignments = assign_roles(steps("Identify the boundary", goal))
        assert assignments[1].role == ANALYST, goal
    for goal in (
        "Identify the entry point",
        "Investigate the call graph",
        "Understand relationships between services",
    ):
        # third step carries the interpretation so post-condition B
        # cannot promote the step under test.
        assignments = assign_roles(
            steps("Identify the boundary", goal, "Explain the outcome")
        )
        assert assignments[1].role == INVESTIGATOR, goal


def test_every_step_is_assigned_in_plan_order_with_a_reason():
    assignments = assign_roles(
        steps("Identify modules", "Summarize the architecture")
    )
    assert [item.step_id for item in assignments] == ["S1", "S2"]
    assert all(isinstance(item, RoleAssignment) for item in assignments)
    assert all(item.reason for item in assignments)


# -- post-conditions -------------------------------------------------------


def test_first_step_becomes_investigator_when_none_classified():
    assignments = assign_roles(
        steps("Explain the architecture", "Summarize the findings")
    )
    assert assignments[0].role == INVESTIGATOR
    assert assignments[0].reason == (
        "evidence collection required before interpretation"
    )
    assert assignments[1].role == ANALYST


def test_last_step_becomes_analyst_when_plan_of_two_has_none():
    assignments = assign_roles(
        steps("Identify the boundary", "Collect the evidence")
    )
    assert assignments[0].role == INVESTIGATOR
    assert assignments[1].role == ANALYST
    assert assignments[1].reason == (
        "final step synthesizes the architecture explanation"
    )


def test_single_step_plan_stays_investigator():
    assignments = assign_roles(steps("Identify the boundary"))
    assert len(assignments) == 1
    assert assignments[0].role == INVESTIGATOR


def test_empty_plan_assigns_nothing():
    assert assign_roles([]) == []


# -- overrides (explicit, validated) ---------------------------------------


def test_explicit_override_wins_and_is_marked_application_assigned():
    assignments = assign_roles(
        steps(
            "Identify the boundary",
            "Collect the evidence",
            "Explain the boundary",
        ),
        role_overrides={"S2": ANALYST},
    )
    assert [item.role for item in assignments] == [
        INVESTIGATOR,
        ANALYST,
        ANALYST,
    ]
    assert assignments[1].reason == (
        "application-assigned role for this step"
    )
    assert assignments[0].reason != assignments[1].reason


def test_unsupported_role_is_rejected_not_coerced():
    with pytest.raises(ValueError) as excinfo:
        assign_roles(
            steps("Identify the boundary", "Explain the boundary"),
            role_overrides={"S1": "PLANNER"},
        )
    message = str(excinfo.value)
    assert "PLANNER" in message
    assert "S1" in message
    assert "INVESTIGATOR" in message and "ANALYST" in message


def test_override_for_an_unknown_step_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        assign_roles(steps("Identify the boundary"), role_overrides={"S9": ANALYST})
    assert "S9" in str(excinfo.value)


def test_override_that_orders_analysis_before_evidence_is_rejected():
    """An ANALYST step with no preceding INVESTIGATOR step could never
    receive a report — the ordering constraint is validated here, at
    assignment time, rather than surfacing as a runtime crash."""
    with pytest.raises(ValueError) as excinfo:
        assign_roles(
            steps("Explain the boundary", "Identify the boundary"),
            role_overrides={"S1": ANALYST},
        )
    assert "S1" in str(excinfo.value)
