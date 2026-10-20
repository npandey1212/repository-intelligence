"""M10 — deterministic plan validation (spec §18).

``validate_plan`` is the application gate between the LLM-produced
plan and any repository tool execution. It is pure, offline, and
deterministic — no model call, no repair, no planner retry:

    objective exists
    at least one step
    step IDs non-empty and unique
    every step has a meaningful (non-empty) goal
    steps are goal-level, not direct tool-call instructions
    no unsupported execution directives (replanning, parallel
        execution, planner re-invocation, spawning)

A failure raises ``PlanValidationError``; the run must terminate with
NO repository tool execution (asserted end-to-end in
``tests/agent/test_agent_planned.py``). Step ordering is the list
order itself — the sequence the executor walks sequentially — so there
is nothing beyond "steps exist and are identified" to check here
(spec §16: no scheduler, no dependency graph).
"""

from __future__ import annotations

import pytest

from repo_intelligence.planning import (
    InvestigationPlan,
    PlanStep,
    PlanValidationError,
    validate_plan,
)


def _plan(*steps, objective="Explain how the components connect."):
    return InvestigationPlan(
        objective=objective,
        steps=[PlanStep(id=step_id, goal=goal) for step_id, goal in steps],
    )


VALID = _plan(
    ("P1", "Identify relevant agent definitions"),
    ("P2", "Map agents to the tools they use"),
    ("P3", "Produce an evidence-backed explanation"),
)


# -- accepting checks ------------------------------------------------------


def test_goal_level_plan_validates():
    assert validate_plan(VALID) is None


def test_step_order_is_the_list_order():
    plan = _plan(("P1", "first sub-goal"), ("P2", "second sub-goal"))
    assert validate_plan(plan) is None
    assert [step.id for step in plan.steps] == ["P1", "P2"]


def test_function_mention_inside_a_goal_is_not_a_tool_instruction():
    plan = _plan(("P1", "Explain how load_config() works"))
    assert validate_plan(plan) is None


# -- objective / steps -----------------------------------------------------


def test_empty_objective_fails():
    plan = _plan(("P1", "identify the agents"), objective="   ")
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan)
    assert "objective" in str(excinfo.value)


def test_empty_steps_fail():
    plan = InvestigationPlan(objective="o", steps=[])
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan)
    assert "step" in str(excinfo.value)


# -- step identity ---------------------------------------------------------


def test_duplicate_step_ids_fail():
    plan = _plan(("P1", "identify the agents"), ("P1", "map the tools"))
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan)
    assert "P1" in str(excinfo.value)


def test_blank_step_id_fails():
    plan = _plan(("  ", "identify the agents"))
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_empty_step_goal_fails():
    plan = _plan(("P1", "   "))
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan)
    assert "P1" in str(excinfo.value)


# -- goal-level, not tool-level (spec §14) ---------------------------------


@pytest.mark.parametrize(
    "goal",
    [
        'call search_repository(query="agent")',
        'search_repository(query="agent")',
        'read_file(path=".mcp.json")',
        'call get_architecture_context',
        'invoke read_file(path="agents/dag-agent.md")',
        '{"name": "search_repository", "arguments": {"query": "agent"}}',
    ],
)
def test_tool_level_goal_is_rejected(goal):
    plan = _plan(("P1", goal))
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan(plan)
    assert "P1" in str(excinfo.value)


def test_good_example_goals_from_the_spec_validate():
    plan = _plan(
        ("P1", "identify relevant agent definitions"),
        ("P2", "map agents to tools"),
        ("P3", "map tools to MCP servers"),
        ("P4", "validate important relationships against implementation"),
        ("P5", "produce an evidence-backed explanation"),
    )
    assert validate_plan(plan) is None


# -- unsupported execution directives (spec §18/§42) -----------------------


@pytest.mark.parametrize(
    "goal",
    [
        "Identify the agents, then replan the investigation",
        "Re-plan the remaining steps after P1",
        "Identify the agents, then call the planner again",
        "Execute the identified steps by parallel execution",
        "Spawn a helper to gather evidence",
    ],
)
def test_unsupported_execution_directive_is_rejected(goal):
    plan = _plan(("P1", goal))
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_domain_words_that_merely_look_suspicious_still_validate():
    plan = _plan(
        ("P1", "identify how parallel data flows are orchestrated"),
        ("P2", "explain the plan document structure"),
    )
    assert validate_plan(plan) is None
