"""M17 orchestration status machine (spec §25/§39).

Explicit, minimal, terminal states sealed: the run moves PENDING →
PLANNING → PLAN_VALIDATED → EXECUTING → VALIDATING, or exits through
an explicit failure state. Illegal transitions raise instead of
corrupting the run record, and no terminal state ever re-opens (once
terminal, no further agent execution is permitted).
"""

from __future__ import annotations

import pytest

from repo_intelligence.orchestration.status import (
    AGENT_FAILED,
    ALLOWED_TRANSITIONS,
    BUDGET_EXCEEDED,
    COMPLETED,
    EXECUTING,
    INCOMPLETE,
    PENDING,
    PLAN_FAILED,
    PLAN_VALIDATED,
    PLANNING,
    RUN_STATUSES,
    TERMINATION_REASONS,
    TIMEOUT,
    VALIDATING,
    VALIDATION_FAILED,
    OrchestrationTransitionError,
    advance,
    can_advance,
    is_terminal,
)

TERMINALS = (
    COMPLETED,
    VALIDATION_FAILED,
    PLAN_FAILED,
    AGENT_FAILED,
    TIMEOUT,
    BUDGET_EXCEEDED,
    INCOMPLETE,
)


def test_happy_path_transitions():
    assert advance(PENDING, PLANNING) == PLANNING
    assert advance(PLANNING, PLAN_VALIDATED) == PLAN_VALIDATED
    assert advance(PLAN_VALIDATED, EXECUTING) == EXECUTING
    assert advance(EXECUTING, VALIDATING) == VALIDATING
    assert advance(VALIDATING, COMPLETED) == COMPLETED


def test_plan_gate_exits_without_executing():
    assert can_advance(PLANNING, PLAN_FAILED)
    assert not can_advance(PLANNING, EXECUTING)


def test_execution_exits_reach_every_failure_state():
    for terminal in (
        VALIDATING,
        AGENT_FAILED,
        INCOMPLETE,
        TIMEOUT,
        BUDGET_EXCEEDED,
    ):
        assert can_advance(EXECUTING, terminal), terminal


def test_validation_exits():
    for terminal in (COMPLETED, VALIDATION_FAILED, INCOMPLETE):
        assert can_advance(VALIDATING, terminal), terminal


def test_every_terminal_status_is_sealed():
    for terminal in TERMINALS:
        assert is_terminal(terminal)
        assert ALLOWED_TRANSITIONS[terminal] == frozenset()


def test_no_terminal_status_is_reachable_from_pending():
    for terminal in TERMINALS:
        assert not can_advance(PENDING, terminal)


def test_illegal_transition_raises_with_the_observed_pair():
    with pytest.raises(OrchestrationTransitionError) as excinfo:
        advance(PENDING, COMPLETED)
    message = str(excinfo.value)
    assert "'PENDING'" in message
    assert "'COMPLETED'" in message


def test_restart_after_terminal_raises():
    with pytest.raises(OrchestrationTransitionError):
        advance(COMPLETED, EXECUTING)
    with pytest.raises(OrchestrationTransitionError):
        advance(VALIDATION_FAILED, VALIDATING)


def test_unknown_status_is_rejected_rather_than_treated_as_terminal():
    with pytest.raises(OrchestrationTransitionError):
        advance("NOT_A_STATUS", COMPLETED)
    assert not is_terminal("NOT_A_STATUS")


def test_status_vocabulary_is_closed():
    assert RUN_STATUSES == frozenset(
        {
            "PENDING",
            "PLANNING",
            "PLAN_VALIDATED",
            "EXECUTING",
            "VALIDATING",
            "COMPLETED",
            "VALIDATION_FAILED",
            "PLAN_FAILED",
            "AGENT_FAILED",
            "TIMEOUT",
            "BUDGET_EXCEEDED",
            "INCOMPLETE",
        }
    )
    assert set(ALLOWED_TRANSITIONS) == set(RUN_STATUSES)


def test_termination_reason_vocabulary_is_explicit():
    assert TERMINATION_REASONS == (
        "SUCCESS",
        "PLAN_FAILED",
        "AGENT_FAILED",
        "VALIDATION_FAILED",
        "TIMEOUT",
        "BUDGET_EXCEEDED",
        "DECLARED_INCOMPLETE",
        "NO_ANALYSIS",
    )
