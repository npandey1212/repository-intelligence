"""M17 orchestrator — plan execution, bounds, and failure semantics.

Every scenario is scripted and LLM-free: queued ``LLMResponse``s, a
canned read-only dispatcher, one ``TaskPayload``. What is proven here:

* the happy path: plan gate → role assignment → sequential steps →
  six typed handoffs → the M16 validator → one sealed terminal state
* every bound is explicit and checked BEFORE the next unit of work
  (plan steps, agent runs, messages, total tool calls, wall clock)
* bounded retry exists ONLY here and never on validation failures,
  never past ``max_retries``, never on an agent's own exhausted budget
* failure semantics: invalid plan runs no agent; a failed
  Investigator step blocks what follows and declares the work
  incomplete; a failed Analyst step never fabricates an answer
* observability: sequence-numbered run events, mirrored onto each
  agent's OWN trace through the one trace system.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import AgentExecutionConfig
from repo_intelligence.agent.trace import (
    ORCHESTRATION_OPERATIONS,
    AgentTrace,
    render_markdown,
)
from repo_intelligence.agent.trace_sink import JsonFileTraceSink
from repo_intelligence.multiagent import (
    InvestigationReport,
    ContractError,
)
from repo_intelligence.orchestration import (
    AGENT_FAILED,
    ANALYST,
    BUDGET_EXCEEDED,
    COMPLETED,
    INCOMPLETE,
    INVESTIGATOR,
    PLAN_FAILED,
    TIMEOUT,
    VALIDATION_FAILED,
    OrchestrationConfig,
    RoleAssignment,
)
from repo_intelligence.planning.models import InvestigationPlan, PlanStep

from m17_doubles import (
    COMMIT,
    README_REF,
    REPOSITORY,
    build,
    call,
    finding_payload,
    investigate_read_then_report,
    report_payload,
    rule,
    structured,
    task,
    tool_calls,
    two_step_plan,
)


def _ops(run):
    return run.event_operations()


def _events(run, operation):
    return [event for event in run.events if event["operation"] == operation]


def assert_no_agent_ran(investigator, analyst):
    assert investigator._llm_client.requests == []
    assert analyst._llm_client.requests == []


# -- happy path ------------------------------------------------------------


def test_happy_path_completes_with_the_six_typed_handoffs():
    orch, investigator, analyst = build()
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert run.final is True
    assert run.completed_at
    assert [message.message_type for message in run.messages] == [
        "INVESTIGATION_REQUEST",
        "INVESTIGATION_RESULT",
        "ANALYSIS_REQUEST",
        "ANALYSIS_RESULT",
        "VALIDATION_REQUEST",
        "VALIDATION_RESULT",
    ]
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert len(run.validation.checks) == 8
    assert all(check.ok for check in run.validation.checks)
    assert run.validation_status == "passed"


def test_plan_lifecycle_is_driven_by_the_executor_not_the_model():
    plan = two_step_plan()
    orch, _, _ = build()
    run = orch.execute(task(), plan)
    assert run.status == COMPLETED
    assert plan.status == "completed"
    assert [step.status for step in plan.steps] == ["completed", "completed"]
    assert all(step.result for step in plan.steps)
    # the executor — never the model — moved every lifecycle field
    assert plan.steps[0].status not in ("pending", "in_progress")


def test_step_results_carry_roles_outputs_and_agent_run_ids():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    assert [result.step_id for result in run.step_results] == ["P1", "P2"]
    assert [result.agent_role for result in run.step_results] == [
        "INVESTIGATOR",
        "ANALYST",
    ]
    assert [result.status for result in run.step_results] == [
        "completed",
        "completed",
    ]
    first, second = run.step_results
    assert isinstance(first.output, InvestigationReport)
    assert first.output.investigation_id == "inv-1"
    assert [ref.path for ref in first.evidence_refs] == ["README.md"]
    assert first.agent_run_id
    assert first.attempts == 1
    assert second.output is not None
    assert second.output.investigation_id == "inv-1"
    assert run.completed_steps == ["P1", "P2"]
    assert run.failed_steps == [] and run.blocked_steps == []
    assert run.current_step is None
    assert run.finding is not None
    assert run.report is not None


def test_role_assignment_is_observable_with_a_reason():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    assignments = _events(run, "orchestration.step.assign")
    assert [event["step_id"] for event in assignments] == ["P1", "P2"]
    assert [event["agent_role"] for event in assignments] == [
        "INVESTIGATOR",
        "ANALYST",
    ]
    for event in assignments:
        assert event["assignment_reason"].strip()
        assert event["step_id"]


def test_events_are_sequence_numbered_and_ordered():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    sequences = [event["sequence"] for event in run.events]
    assert sequences == list(range(1, len(sequences) + 1))
    assert len(sequences) == len(set(sequences))
    operations = _ops(run)
    assert operations[0] == "orchestration.start"
    assert operations[1] == "orchestration.plan.created"
    assert operations[2] == "orchestration.plan.validated"
    assert operations[-1] == "orchestration.complete"
    assert "orchestration.step.start" in operations
    assert "orchestration.step.complete" in operations
    assert "orchestration.handoff.send" in operations
    assert "orchestration.validation.start" in operations
    assert "orchestration.validation.complete" in operations
    for event in run.events:
        assert event["operation"] in ORCHESTRATION_OPERATIONS
        assert event["timestamp"]


def test_agent_runs_are_correlated_to_steps_and_attempts():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    assert [entry["step_id"] for entry in run.agent_runs] == ["P1", "P2"]
    assert [entry["agent_role"] for entry in run.agent_runs] == [
        "INVESTIGATOR",
        "ANALYST",
    ]
    assert all(entry["attempt"] == 1 for entry in run.agent_runs)
    assert all(entry["run_id"] for entry in run.agent_runs)
    assert run.agent_runs[0]["run_id"] == run.step_results[0].agent_run_id


def test_every_message_carries_the_task_scope():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    for message in run.messages:
        assert message.repository_scope == REPOSITORY
        assert message.commit_scope == COMMIT


def test_role_overrides_are_validated_before_any_agent_runs():
    orch, investigator, _ = build()
    with pytest.raises(ValueError):
        orch.execute(
            task(), two_step_plan(), role_overrides={"P1": "PLANNER"}
        )
    assert investigator._llm_client.requests == []


def test_role_assigner_is_injectable():
    from repo_intelligence.orchestration.roles import (
        ANALYST,
        INVESTIGATOR,
        RoleAssignment,
    )

    def fixed(steps, overrides=None):
        return [
            RoleAssignment(
                step_id=step.id,
                role=INVESTIGATOR,
                reason="application-supplied assignment",
            )
            for step in steps
        ]

    orch, _, _ = build(
        role_assigner=fixed,
        inv_responses=investigate_read_then_report()
        + investigate_read_then_report(),
    )
    run = orch.execute(task(), two_step_plan())
    # both steps ran as Investigators: no analysis, never a fabricated
    # architectural answer — the injected decision is what executed.
    assert [result.agent_role for result in run.step_results] == [
        INVESTIGATOR,
        INVESTIGATOR,
    ]
    assert run.status == INCOMPLETE
    assert run.termination_reason == "NO_ANALYSIS"
    assert run.finding is None
    assert run.report is not None
    assert ANALYST not in [
        event.get("agent_role") for event in run.events
    ]


# -- preassigned step roles (the additive execute input) -------------------


def recording_assigner(calls):
    """Spy for the default role assigner: records, then behaves
    exactly like ``assign_roles``."""
    from repo_intelligence.orchestration import assign_roles

    def spy(steps, overrides=None):
        calls.append((list(steps), overrides))
        return assign_roles(steps, overrides)

    return spy


def standard_preassigned():
    return [
        RoleAssignment(
            step_id="P1", role=INVESTIGATOR, reason="caller: evidence"
        ),
        RoleAssignment(
            step_id="P2", role=ANALYST, reason="caller: synthesis"
        ),
    ]


# -- backward compatibility: no new input, no behavior change -------------


def test_execute_without_role_assignments_invokes_the_default_assigner():
    calls = []
    orch, _, _ = build(role_assigner=recording_assigner(calls))
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    assert len(calls) == 1
    assert [step.id for step in calls[0][0]] == ["P1", "P2"]
    assert calls[0][1] is None


def test_role_assignments_none_is_identical_to_omitting_it():
    calls = []
    orch, _, _ = build(role_assigner=recording_assigner(calls))
    run = orch.execute(
        task(), two_step_plan(), role_assignments=None
    )
    assert run.status == COMPLETED
    assert len(calls) == 1
    assert calls[0][1] is None


def test_role_overrides_still_flow_to_the_assigner_when_not_preassigned():
    calls = []
    orch, _, _ = build(role_assigner=recording_assigner(calls))
    run = orch.execute(
        task(), two_step_plan(), role_overrides={"P2": INVESTIGATOR}
    )
    assert run.status == COMPLETED
    assert len(calls) == 1
    assert calls[0][1] == {"P2": INVESTIGATOR}
    # the override reached the assigner; its post-conditions still
    # apply on top (a 2+ step plan with no Analyst gains one)
    assert [result.agent_role for result in run.step_results] == [
        INVESTIGATOR,
        ANALYST,
    ]


def test_default_assignment_reasons_are_unchanged_without_preassignment():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    reasons = [
        event["assignment_reason"]
        for event in _events(run, "orchestration.step.assign")
    ]
    assert reasons == [
        "step requires repository evidence collection",
        "architecture interpretation requires Analyst role",
    ]


def test_execute_signature_keeps_role_overrides_as_the_third_parameter():
    import inspect

    orch, _, _ = build()
    parameters = list(inspect.signature(orch.execute).parameters)
    assert parameters == [
        "task",
        "plan",
        "role_overrides",
        "role_assignments",
    ]


# -- preassigned behavior --------------------------------------------------


def test_preassigned_assignments_execute_exactly_as_supplied():
    orch, _, _ = build()
    run = orch.execute(
        task(), two_step_plan(), role_assignments=standard_preassigned()
    )
    assert run.status == COMPLETED
    assert [result.step_id for result in run.step_results] == ["P1", "P2"]
    assert [result.agent_role for result in run.step_results] == [
        INVESTIGATOR,
        ANALYST,
    ]
    assert run.completed_steps == ["P1", "P2"]
    assert run.termination_reason == "SUCCESS"


def test_preassigned_reasons_are_preserved_verbatim_in_assign_events():
    orch, _, _ = build()
    run = orch.execute(
        task(), two_step_plan(), role_assignments=standard_preassigned()
    )
    assert run.status == COMPLETED
    events = _events(run, "orchestration.step.assign")
    assert [event["step_id"] for event in events] == ["P1", "P2"]
    assert [event["agent_role"] for event in events] == [
        INVESTIGATOR,
        ANALYST,
    ]
    assert [event["assignment_reason"] for event in events] == [
        "caller: evidence",
        "caller: synthesis",
    ]


def test_preassigned_assignments_bypass_the_role_assigner():
    def exploding_assigner(steps, overrides=None):
        raise AssertionError(
            "role_assigner must not run for preassigned input"
        )

    orch, _, _ = build(role_assigner=exploding_assigner)
    run = orch.execute(
        task(), two_step_plan(), role_assignments=standard_preassigned()
    )
    assert run.status == COMPLETED


def test_preassigned_assignments_are_normalized_to_plan_order():
    orch, _, _ = build()
    run = orch.execute(
        task(),
        two_step_plan(),
        role_assignments=[
            RoleAssignment(
                step_id="P2", role=ANALYST, reason="synthesis"
            ),
            RoleAssignment(
                step_id="P1", role=INVESTIGATOR, reason="evidence"
            ),
        ],
    )
    assert run.status == COMPLETED
    assert [
        (event["step_id"], event["agent_role"])
        for event in _events(run, "orchestration.step.assign")
    ] == [("P1", INVESTIGATOR), ("P2", ANALYST)]
    assert [result.agent_role for result in run.step_results] == [
        INVESTIGATOR,
        ANALYST,
    ]


def test_preassigned_all_investigator_is_not_silently_rewritten():
    orch, _, analyst = build(
        inv_responses=investigate_read_then_report()
        + investigate_read_then_report()
    )
    run = orch.execute(
        task(),
        two_step_plan(),
        role_assignments=[
            RoleAssignment(
                step_id="P1", role=INVESTIGATOR, reason="caller: one"
            ),
            RoleAssignment(
                step_id="P2", role=INVESTIGATOR, reason="caller: two"
            ),
        ],
    )
    # no Analyst step was supplied, so none is invented: the run ends
    # with a preserved evidence package, never a fabricated answer.
    assert run.status == INCOMPLETE
    assert run.termination_reason == "NO_ANALYSIS"
    assert run.finding is None
    assert run.report is not None
    assert analyst._llm_client.requests == []
    assert [result.agent_role for result in run.step_results] == [
        INVESTIGATOR,
        INVESTIGATOR,
    ]


# -- preassigned input: deterministic rejection, no agent runs -------------


def test_role_assignments_and_role_overrides_together_are_rejected():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_overrides={"P1": INVESTIGATOR},
            role_assignments=standard_preassigned(),
        )
    assert "not both" in str(excinfo.value)
    assert_no_agent_ran(investigator, analyst)


def test_preassigned_item_must_be_a_role_assignment():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_assignments=[
                {"step_id": "P1", "role": INVESTIGATOR, "reason": "x"}
            ],
        )
    assert "role_assignments[0]" in str(excinfo.value)
    assert "RoleAssignment" in str(excinfo.value)
    assert_no_agent_ran(investigator, analyst)


def test_duplicate_preassigned_step_id_is_rejected():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_assignments=[
                RoleAssignment(
                    step_id="P1", role=INVESTIGATOR, reason="a"
                ),
                RoleAssignment(step_id="P1", role=ANALYST, reason="b"),
                RoleAssignment(step_id="P2", role=ANALYST, reason="c"),
            ],
        )
    assert "duplicate" in str(excinfo.value)
    assert "P1" in str(excinfo.value)
    assert_no_agent_ran(investigator, analyst)


def test_preassigned_coverage_must_be_exact_missing_step_named():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_assignments=[
                RoleAssignment(
                    step_id="P1", role=INVESTIGATOR, reason="only"
                )
            ],
        )
    assert "missing" in str(excinfo.value)
    assert "P2" in str(excinfo.value)
    assert_no_agent_ran(investigator, analyst)


def test_preassigned_unknown_step_id_is_rejected():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_assignments=standard_preassigned()
            + [
                RoleAssignment(
                    step_id="P9", role=INVESTIGATOR, reason="stray"
                )
            ],
        )
    assert "absent from the plan" in str(excinfo.value)
    assert "P9" in str(excinfo.value)
    assert_no_agent_ran(investigator, analyst)


def test_preassigned_unsupported_role_is_rejected_not_coerced():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_assignments=[
                RoleAssignment(
                    step_id="P1", role="PLANNER", reason="typo"
                ),
                RoleAssignment(
                    step_id="P2", role=ANALYST, reason="caller"
                ),
            ],
        )
    message = str(excinfo.value)
    assert "PLANNER" in message
    assert "P1" in message
    assert "INVESTIGATOR" in message and "ANALYST" in message
    assert_no_agent_ran(investigator, analyst)


def test_preassigned_analysis_before_evidence_is_rejected():
    orch, investigator, analyst = build()
    with pytest.raises(ValueError) as excinfo:
        orch.execute(
            task(),
            two_step_plan(),
            role_assignments=[
                RoleAssignment(
                    step_id="P1", role=ANALYST, reason="too early"
                ),
                RoleAssignment(
                    step_id="P2", role=INVESTIGATOR, reason="late"
                ),
            ],
        )
    message = str(excinfo.value)
    assert "ANALYST before any INVESTIGATOR" in message
    assert "P1" in message
    assert_no_agent_ran(investigator, analyst)


# -- the plan gate ---------------------------------------------------------


def test_invalid_plan_fails_before_any_agent_runs():
    plan = InvestigationPlan(objective="objective", steps=[])
    orch, investigator, analyst = build()
    run = orch.execute(task(), plan)
    assert run.status == PLAN_FAILED
    assert run.termination_reason == "PLAN_FAILED"
    assert run.final is True
    assert investigator._llm_client.requests == []
    assert analyst._llm_client.requests == []
    assert plan.status == "failed"
    validated = _events(run, "orchestration.plan.validated")
    assert validated and validated[0]["status"] == "FAILED"
    assert _ops(run)[-1] == "orchestration.failed"
    assert run.step_results == []


def test_over_long_plan_is_rejected_by_the_step_budget():
    steps = [
        PlanStep(id="S{}".format(index), goal="Identify module {}".format(index))
        for index in range(9)
    ]
    plan = InvestigationPlan(objective="objective", steps=steps)
    orch, investigator, _ = build(config=OrchestrationConfig(max_plan_steps=8))
    run = orch.execute(task(), plan)
    assert run.status == PLAN_FAILED
    assert "max_plan_steps" in str(
        _events(run, "orchestration.plan.validated")[0]["errors"]
    )
    assert investigator._llm_client.requests == []


def test_tool_level_goal_is_a_plan_failure_not_a_longer_run():
    plan = InvestigationPlan(
        objective="objective",
        steps=[PlanStep(id="S1", goal="read_file(path='README.md')")],
    )
    orch, investigator, _ = build()
    run = orch.execute(task(), plan)
    assert run.status == PLAN_FAILED
    assert investigator._llm_client.requests == []


def test_config_bounds_are_validated_up_front():
    with pytest.raises(ValueError):
        OrchestrationConfig(max_plan_steps=0)
    with pytest.raises(ValueError):
        OrchestrationConfig(max_agent_runs=0)
    with pytest.raises(ValueError):
        OrchestrationConfig(max_messages=0)
    with pytest.raises(ValueError):
        OrchestrationConfig(max_retries=-1)
    with pytest.raises(ValueError):
        OrchestrationConfig(max_runtime_seconds=0)


# -- failure semantics -----------------------------------------------------


def test_investigator_failure_blocks_downstream_and_declares_incomplete():
    broken = report_payload()
    del broken["findings"]
    orch, investigator, analyst = build(
        inv_responses=[structured(broken), structured(broken)]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    assert run.final is True
    assert run.failed_steps == ["P1"]
    assert run.blocked_steps == ["P2"]
    assert run.finding is None
    assert run.validation is None
    assert run.report is None
    assert analyst._llm_client.requests == []
    assert run.step_result("P1").status == "failed"
    assert run.step_result("P1").attempts == 2
    assert run.step_result("P2").status == "blocked"
    assert plan_blocked(run)
    blocked = _events(run, "orchestration.step.blocked")
    assert [event["step_id"] for event in blocked] == ["P2"]
    assert blocked[0]["reason"] == "P1_FAILED"
    assert _ops(run)[-1] == "orchestration.terminated"


def plan_blocked(run) -> bool:
    return [step.status for step in run.plan.steps] == ["failed", "blocked"]


def test_analyst_failure_yields_agent_failed_with_no_fabricated_finding():
    broken = finding_payload()
    del broken["summary"]
    orch, _, analyst = build(
        ana_responses=[structured(broken), structured(broken)]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == AGENT_FAILED
    assert run.termination_reason == "AGENT_FAILED"
    assert run.report is not None
    assert run.finding is None
    assert run.validation is None
    assert run.failed_steps == ["P2"]
    assert run.step_result("P2").status == "failed"
    assert _ops(run)[-1] == "orchestration.terminated"
    assert run.plan.status == "failed"


def test_agent_tool_budget_maps_to_budget_exceeded_without_retry():
    orch, investigator, _ = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ],
        inv_kwargs={
            "execution_config": AgentExecutionConfig(max_tool_calls=0)
        },
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    assert run.retries == 0
    assert run.step_result("P1").attempts == 1
    assert run.blocked_steps == ["P2"]
    budget = _events(run, "orchestration.budget.exceeded")
    assert budget and budget[0]["budget"] == "agent_max_tool_calls"
    assert _ops(run)[-1] == "orchestration.terminated"


def test_single_step_plan_preserves_evidence_without_fabricating_analysis():
    orch, _, analyst = build()
    plan = InvestigationPlan(
        objective="objective",
        steps=[PlanStep(id="P1", goal="Identify the application boundary")],
    )
    run = orch.execute(task(), plan)
    assert run.status == INCOMPLETE
    assert run.termination_reason == "NO_ANALYSIS"
    assert run.report is not None
    assert run.finding is None
    assert run.validation is None
    assert analyst._llm_client.requests == []
    assert [message.message_type for message in run.messages] == [
        "INVESTIGATION_REQUEST"
    ]


# -- bounded retry (M17 only, never indefinite) ----------------------------


def test_retry_recovers_from_a_broken_structured_response():
    broken = report_payload()
    del broken["findings"]
    orch, investigator, _ = build(
        inv_responses=[
            structured(broken),
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    assert run.retries == 1
    assert run.step_result("P1").attempts == 2
    retries = _events(run, "orchestration.retry")
    assert len(retries) == 1
    assert retries[0]["step_id"] == "P1"
    assert retries[0]["attempt"] == 2
    assert retries[0]["reason"]
    starts = [
        event
        for event in _events(run, "orchestration.agent.start")
        if event["step_id"] == "P1"
    ]
    assert [event["attempt"] for event in starts] == [1, 2]
    assert investigator._llm_client.requests


def test_retry_is_bounded_by_max_retries():
    broken = report_payload()
    del broken["findings"]
    orch, _, _ = build(
        inv_responses=[structured(broken), structured(broken)],
        config=OrchestrationConfig(max_retries=1),
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == INCOMPLETE
    assert run.retries == 1
    assert run.step_result("P1").attempts == 2


def test_retry_can_be_disabled():
    broken = report_payload()
    del broken["findings"]
    orch, _, _ = build(
        inv_responses=[structured(broken), structured(broken)],
        config=OrchestrationConfig(max_retries=0),
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == INCOMPLETE
    assert run.retries == 0
    assert run.step_result("P1").attempts == 1


def test_scope_mismatch_is_retried_then_succeeds():
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload(commit_scope="f" * 40)),
            structured(report_payload()),
        ]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    assert run.retries == 1
    assert run.step_result("P1").attempts == 2
    assert run.report.commit_scope == COMMIT


def test_validation_failures_are_never_retried():
    """The verdict is terminal: outputs exist, validation rejected
    them, and no agent ran again."""
    orch, _, analyst = build(
        ana_responses=[
            structured(
                finding_payload(
                    conclusions=[
                        {
                            "statement": "one path cannot be conflicting",
                            "classification": "conflicting",
                            "evidence_refs": [dict(README_REF)],
                        }
                    ]
                )
            )
        ]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == VALIDATION_FAILED
    assert run.termination_reason == "VALIDATION_FAILED"
    assert run.retries == 0
    assert _events(run, "orchestration.retry") == []
    calls_after_verdict = len(analyst._llm_client.requests)
    assert calls_after_verdict == 1
    assert _ops(run)[-1] == "orchestration.failed"


# -- budgets ---------------------------------------------------------------


def test_message_budget_stops_before_analysis():
    orch, _, analyst = build(config=OrchestrationConfig(max_messages=2))
    run = orch.execute(task(), two_step_plan())
    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    assert run.report is not None
    assert run.finding is None
    assert run.validation is None
    assert analyst._llm_client.requests == []
    assert run.blocked_steps == ["P2"]
    budget = _events(run, "orchestration.budget.exceeded")
    assert budget[0]["budget"] == "max_messages"
    assert budget[0]["limit"] == 2


def test_total_tool_call_budget_stops_before_the_next_step():
    orch, _, analyst = build(
        config=OrchestrationConfig(max_total_tool_calls=1)
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == BUDGET_EXCEEDED
    assert run.tool_calls_total == 1
    assert run.report is not None
    assert run.finding is None
    assert analyst._llm_client.requests == []
    budget = _events(run, "orchestration.budget.exceeded")
    assert budget[0]["budget"] == "max_total_tool_calls"
    assert budget[0]["limit"] == 1


def test_agent_run_budget_counts_every_attempt():
    orch, _, analyst = build(config=OrchestrationConfig(max_agent_runs=1))
    run = orch.execute(task(), two_step_plan())
    assert run.status == BUDGET_EXCEEDED
    assert len(run.agent_runs) == 1
    assert run.report is not None
    assert analyst._llm_client.requests == []
    budget = _events(run, "orchestration.budget.exceeded")
    assert budget[0]["budget"] == "max_agent_runs"
    assert budget[0]["observed"] == 1


def test_wall_clock_budget_maps_to_timeout_between_steps():
    class SteppingClock:
        def __init__(self, step):
            self.value = 0.0
            self.step = step

        def __call__(self):
            current = self.value
            self.value += self.step
            return current

    orch, investigator, analyst = build(
        clock=SteppingClock(10.0),
        config=OrchestrationConfig(max_runtime_seconds=5.0),
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == TIMEOUT
    assert run.termination_reason == "TIMEOUT"
    assert run.agent_runs == []
    assert investigator._llm_client.requests == []
    assert analyst._llm_client.requests == []
    assert run.blocked_steps == ["P1", "P2"]
    assert _events(run, "orchestration.timeout")
    assert _ops(run)[-1] == "orchestration.terminated"


# -- validation ------------------------------------------------------------


def test_validation_failure_is_never_converted_to_success():
    orch, _, _ = build(
        ana_responses=[
            structured(
                finding_payload(
                    conclusions=[
                        {
                            "statement": "one path cannot be conflicting",
                            "classification": "conflicting",
                            "evidence_refs": [dict(README_REF)],
                        }
                    ]
                )
            )
        ]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == VALIDATION_FAILED
    assert run.report is not None and run.finding is not None
    assert rule(run.validation, "classification").ok is False
    assert run.validation.status == "failed"
    assert _events(run, "orchestration.validation.complete")[0][
        "status"
    ] == "failed"
    assert run.plan.status == "failed"


def test_finding_from_another_investigation_fails_the_collaboration_rule():
    orch, _, _ = build(
        ana_responses=[structured(finding_payload(investigation_id="inv-X"))]
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == VALIDATION_FAILED
    assert rule(run.validation, "collaboration").ok is False


def test_declared_incomplete_is_not_a_success():
    orch, _, _ = build(
        inv_responses=investigate_read_then_report(
            report_payload(status="incomplete")
        )
    )
    run = orch.execute(task(), two_step_plan())
    assert run.status == INCOMPLETE
    assert run.termination_reason == "DECLARED_INCOMPLETE"
    assert run.validation.status == "passed"
    assert run.report.status == "incomplete"
    assert run.final is True


def test_validation_runs_after_every_step_over_the_merged_evidence():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    operations = _ops(run)
    assert operations.index("orchestration.step.complete") < (
        operations.index("orchestration.validation.start")
    )
    assert operations.index("orchestration.validation.complete") < (
        operations.index("orchestration.complete")
    )


# -- several Investigator steps, one evidence package ----------------------


def test_multiple_investigator_steps_merge_into_one_evidence_package():
    plan = two_step_plan(
        steps=[
            PlanStep(id="P1", goal="Identify the application boundary"),
            PlanStep(id="P2", goal="Investigate the service entry points"),
            PlanStep(id="P3", goal="Explain how the boundary is enforced"),
        ]
    )
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
            tool_calls(call("read_file", {"path": "src/service.py"})),
            structured(
                report_payload(
                    investigation_id="inv-2",
                    findings=[
                        {
                            "statement": "service.py defines the entry point",
                            "evidence_refs": [{"path": "src/service.py"}],
                            "support": "SUPPORTED",
                        }
                    ],
                    evidence_refs=[{"path": "src/service.py"}],
                    investigated_sources=["src/service.py"],
                )
            ),
        ],
        ana_responses=[
            structured(
                finding_payload(
                    investigation_id="inv-1",
                    conclusions=[
                        {
                            "statement": "the boundary is application-owned",
                            "classification": "observed",
                            "evidence_refs": [dict(README_REF)],
                        },
                        {
                            "statement": "service.py is the entry point",
                            "classification": "observed",
                            "evidence_refs": [{"path": "src/service.py"}],
                        },
                    ],
                    evidence_refs=[
                        dict(README_REF),
                        {"path": "src/service.py"},
                    ],
                )
            )
        ],
    )
    run = orch.execute(task(), plan)
    assert run.status == COMPLETED
    assert [result.agent_role for result in run.step_results] == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    # deterministic merge: findings concatenated in step order, refs
    # de-duplicated, the FIRST investigation id kept.
    assert run.report.investigation_id == "inv-1"
    assert [item.statement for item in run.report.findings] == [
        "README.md documents the dispatcher boundary",
        "service.py defines the entry point",
    ]
    assert [ref.path for ref in run.report.evidence_refs] == [
        "README.md",
        "src/service.py",
    ]
    assert [ref.path for ref in run.step_results[0].evidence_refs] == [
        "README.md"
    ]
    assert [ref.path for ref in run.step_results[1].evidence_refs] == [
        "src/service.py"
    ]
    # evidence_valid proves BOTH traces fed the merged authority: the
    # second conclusion resolves only against the second step's read.
    assert run.validation.status == "passed"
    assert rule(run.validation, "evidence_valid").ok is True
    assert rule(run.validation, "collaboration").ok is True
    assert [message.message_type for message in run.messages] == [
        "INVESTIGATION_REQUEST",
        "INVESTIGATION_RESULT",
        "ANALYSIS_REQUEST",
        "ANALYSIS_RESULT",
        "VALIDATION_REQUEST",
        "VALIDATION_RESULT",
    ]
    assert run.plan.status == "completed"
    assert [step.status for step in run.plan.steps] == [
        "completed",
        "completed",
        "completed",
    ]


# -- observability: the ONE trace system -----------------------------------


def test_orchestration_events_are_mirrored_onto_each_agent_trace(tmp_path):
    sink = JsonFileTraceSink(root=str(tmp_path / "traces"), markdown=True)
    orch, investigator, analyst = build(trace_sink=sink)
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    inv_ops = [
        event["operation"]
        for event in investigator.last_trace.orchestration_events
    ]
    assert inv_ops == [
        "orchestration.step.assign",
        "orchestration.handoff.send",
        "orchestration.agent.start",
        "orchestration.agent.complete",
        "orchestration.step.complete",
    ]
    ana_ops = [
        event["operation"]
        for event in analyst.last_trace.orchestration_events
    ]
    assert "orchestration.step.assign" in ana_ops
    assert "orchestration.handoff.send" in ana_ops
    assert "orchestration.step.complete" in ana_ops
    # mirrored events carry the application's decision metadata
    assignment = investigator.last_trace.orchestration_events[0]
    assert assignment["step_id"] == "P1"
    assert assignment["agent_role"] == "INVESTIGATOR"
    assert assignment["assignment_reason"].strip()
    assert assignment["sequence"]
    # the sink artifact was re-written with the mirrored events
    trace_dir = tmp_path / "traces" / run.step_results[0].agent_run_id
    payload = json.loads(
        (trace_dir / "trace.json").read_text(encoding="utf-8")
    )
    assert payload.get("orchestration_events")
    markdown = (trace_dir / "trace.md").read_text(encoding="utf-8")
    assert "## Orchestration" in markdown
    assert "orchestration.step.assign" in markdown
    # run-level events (start, plan gate, validation, termination) are
    # NOT mirrored: they belong to no single agent's execution.
    mirrored = [
        event["operation"]
        for event in investigator.last_trace.orchestration_events
    ]
    assert "orchestration.complete" not in mirrored
    assert "orchestration.validation.start" not in mirrored


def test_default_trace_serialization_is_backward_compatible():
    trace = AgentTrace.start("question")
    assert trace.orchestration_events == []
    payload = trace.to_dict()
    assert "orchestration_events" not in payload
    assert "## Orchestration" not in render_markdown(trace)


def test_trace_recorder_rejects_unknown_orchestration_operations():
    trace = AgentTrace.start("question")
    with pytest.raises(ValueError):
        trace.record_orchestration_event({"operation": "task.dispatch"})
    trace.record_orchestration_event(
        {"operation": "orchestration.step.assign", "sequence": 1}
    )
    assert trace.orchestration_events[0]["operation"] == (
        "orchestration.step.assign"
    )
    payload = trace.to_dict()
    assert payload["orchestration_events"][0]["sequence"] == 1


def test_run_events_and_trace_events_share_one_vocabulary():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    for operation in _ops(run):
        assert operation in ORCHESTRATION_OPERATIONS


# -- malformed handoffs stay a bug, not an outcome -------------------------


def test_malformed_payload_never_reaches_the_message_boundary():
    from repo_intelligence.multiagent import make_message

    with pytest.raises(ContractError):
        make_message(
            "INVESTIGATION_RESULT",
            "investigator",
            "analyst",
            {"not": "a report"},
            REPOSITORY,
            COMMIT,
        )
