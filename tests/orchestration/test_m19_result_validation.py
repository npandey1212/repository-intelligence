"""M19 boundary validation — every ``AgentTaskResult`` is checked
against its request BEFORE the coordinator commits any state.

``validate_task_result`` (contracts.py, coordinator side of the
contract) pins correlation, the declared types, the two committable
result statuses, and the status/failure/output combinations. A
violation is an implementation bug at the boundary: the runner fails
the task with ``FAILURE_CONTRACT_VIOLATION`` (non-retryable) and
commits NOTHING from the invalid result — no output, usage, trace, or
correlation id.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from repo_intelligence.multiagent import (
    ArchitectureFinding,
    InvestigationReport,
)
from repo_intelligence.orchestration import (
    AGENT_FAILED,
    FAILURE_CONTRACT_VIOLATION,
    INCOMPLETE,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    VALIDATION_FAILED,
    AgentTaskRequest,
    AgentTaskResult,
    Orchestrator,
    TaskContractError,
    TaskFailure,
    TaskUsage,
    validate_task_result,
)

from m17_doubles import (
    build,
    finding_payload,
    report_payload,
    task,
    two_step_plan,
)


def _request(**overrides) -> AgentTaskRequest:
    data = {
        "run_id": "r1",
        "task_id": "P1",
        "attempt": 1,
        "role": "INVESTIGATOR",
        "assignment_reason": "first investigator step",
        "question": "how does it work?",
        "goal": "Identify the boundary",
        "prompt": "build the report",
        "repository_scope": "example-repository",
        "commit_scope": "c" * 40,
    }
    data.update(overrides)
    return AgentTaskRequest(**data)


def _result(**overrides) -> AgentTaskResult:
    data = {
        "run_id": "r1",
        "task_id": "P1",
        "attempt": 1,
        "role": "INVESTIGATOR",
        "status": "completed",
        "output": object(),
    }
    data.update(overrides)
    return AgentTaskResult(**data)


class _Scripted:
    """Fake boundary echoing the request's per-run ``run_id`` (the one
    correlation field unknowable when the script was built)."""

    def __init__(self, *results):
        self.requests = []
        self.results = list(results)

    def execute(self, request):
        self.requests.append(request)
        if not self.results:
            raise AssertionError("executor invoked more times than scripted")
        return replace(self.results.pop(0), run_id=request.run_id)


# -- unit: what passes ------------------------------------------------------


def test_valid_completed_and_failed_results_pass():
    validate_task_result(_request(), _result())
    failure = TaskFailure(
        category="provider",
        error_type="LLMError",
        error_message="LLMError: rate limited",
        retryable=True,
    )
    validate_task_result(
        _request(),
        _result(status="failed", output=None, failure=failure),
    )


def test_completed_without_output_is_allowed():
    # The executor genuinely produced no payload; whether that is a
    # failure is the runner's MISSING_STRUCTURED_OUTPUT policy.
    validate_task_result(_request(), _result(output=None))


# -- unit: what is rejected -------------------------------------------------


def test_non_result_is_rejected():
    with pytest.raises(TaskContractError, match="AgentTaskResult"):
        validate_task_result(_request(), {"status": "completed"})
    with pytest.raises(TaskContractError, match="AgentTaskResult"):
        validate_task_result(_request(), SimpleNamespace(status="completed"))


@pytest.mark.parametrize(
    "field,observed",
    [
        ("run_id", "r-other"),
        ("task_id", "P9"),
        ("attempt", 2),
        ("role", "ANALYST"),
    ],
)
def test_correlation_fields_must_echo_the_request(field, observed):
    with pytest.raises(TaskContractError) as excinfo:
        validate_task_result(_request(), _result(**{field: observed}))
    message = str(excinfo.value)
    assert field in message
    assert "does not echo" in message


def test_usage_must_be_a_task_usage():
    with pytest.raises(TaskContractError, match="TaskUsage"):
        validate_task_result(_request(), _result(usage=None))


def test_evidence_refs_must_be_a_tuple():
    with pytest.raises(TaskContractError, match="tuple"):
        validate_task_result(_request(), _result(evidence_refs=[]))


@pytest.mark.parametrize("status", ["incomplete", "cancelled", "ready"])
def test_reserved_record_statuses_are_never_result_statuses(status):
    with pytest.raises(TaskContractError) as excinfo:
        validate_task_result(_request(), _result(status=status))
    message = str(excinfo.value)
    assert "coordinator-side" in message
    assert status in message


def test_completed_result_may_not_carry_a_failure():
    failure = TaskFailure(
        category="unexpected",
        error_type="RuntimeError",
        error_message="RuntimeError: boom",
        retryable=True,
    )
    with pytest.raises(TaskContractError, match="carries a failure"):
        validate_task_result(_request(), _result(failure=failure))


def test_completed_result_may_not_carry_a_free_form_dict():
    with pytest.raises(TaskContractError, match="free-form dict"):
        validate_task_result(_request(), _result(output={"findings": []}))


def test_failed_result_must_carry_a_task_failure():
    with pytest.raises(TaskContractError, match="TaskFailure"):
        validate_task_result(_request(), _result(status="failed"))


def test_failed_result_may_not_carry_an_output():
    failure = TaskFailure(
        category="unexpected",
        error_type="RuntimeError",
        error_message="RuntimeError: boom",
        retryable=True,
    )
    with pytest.raises(TaskContractError, match="must not carry"):
        validate_task_result(
            _request(),
            _result(status="failed", failure=failure, output=object()),
        )


# -- runner integration: nothing from an invalid result is committed -------


def _report() -> InvestigationReport:
    return InvestigationReport.model_validate(report_payload())


def test_runner_fails_a_correlated_mismatch_without_committing_state():
    mismatched = _result(task_id="WRONG")
    executor = _Scripted(mismatched)
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    assert run.retries == 0
    assert len(executor.requests) == 1  # contract violations never retry
    assert run.agent_runs == []  # no correlation id from the bad result
    assert run.tool_calls_total == 0
    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert p1.attempts == 1
    assert "task_id" in p1.error
    assert p1.output is None
    assert p1.usage is None
    assert p1.usage_total is None
    assert p1.agent_run_id is None
    assert p2.status == TASK_BLOCKED
    assert p2.error == "P1_FAILED"
    assert run.report is None  # the invalid report never merged
    assert run.usage_total is None  # no usage commits from a bad result
    assert "orchestration.retry" not in run.event_operations()
    assert investigator._llm_client.requests == []


def test_runner_rejects_completed_free_form_dict_before_retry_policy():
    # Without contract validation this dict would have become a
    # retryable MISSING_STRUCTURED_OUTPUT; the contract rejects it
    # first, so the task fails exactly once.
    executor = _Scripted(_result(output={"findings": []}))
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == INCOMPLETE
    p1, _ = run.tasks
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert p1.attempts == 1
    assert p1.output is None
    assert run.retries == 0


def test_runner_rejects_a_reserved_result_status():
    executor = _Scripted(_result(status="incomplete"))
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    p1, _ = run.tasks
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert "coordinator-side" in p1.error


def test_analyst_step_violation_fails_the_run_with_agent_failed():
    report = _report()
    finding = ArchitectureFinding.model_validate(finding_payload())
    executor = _Scripted(
        _result(task_id="P1", output=report),
        _result(task_id="P2", role="INVESTIGATOR", output=finding),
    )
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == AGENT_FAILED
    assert run.termination_reason == "AGENT_FAILED"
    p1, p2 = run.tasks
    assert p1.status == TASK_COMPLETED
    assert isinstance(p1.output, InvestigationReport)
    assert p2.status == TASK_FAILED
    assert p2.failure_category == FAILURE_CONTRACT_VIOLATION
    assert "role" in p2.error
    assert p2.output is None
    assert p2.agent_run_id is None
    assert run.finding is None
    assert run.report is not None  # the valid P1 report still merged


def test_valid_results_keep_the_pre_contract_commit_behavior():
    report = _report()
    finding = ArchitectureFinding.model_validate(finding_payload())
    executor = _Scripted(
        _result(task_id="P1", output=report),
        _result(task_id="P2", role="ANALYST", output=finding),
    )
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    p1, p2 = run.tasks
    assert p1.status == TASK_COMPLETED
    assert p2.status == TASK_COMPLETED
    assert p1.usage == TaskUsage()  # scripted result default, committed
    assert run.finding is finding
    # scripted traces carry no inspected evidence — the deterministic
    # validator still decides the run's fate (unchanged M17 behavior).
    assert run.status == VALIDATION_FAILED
    assert len(executor.requests) == 2
