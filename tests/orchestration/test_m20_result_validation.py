"""M20 result validation — runtime value checks before any commit.

``dataclass(frozen=True)`` and type annotations do NOT guarantee
valid runtime field values. ``validate_task_result`` now rejects
malformed ``TaskUsage`` / ``TaskFailure`` VALUES (negative counters,
bool-as-int, non-finite elapsed time, a string or unrelated object as
``failure``) with ``TaskContractError`` — the boundary's own
vocabulary — before the coordinator reads a single field, so a
malformed result can never cause an incidental ``AttributeError`` /
``TypeError`` deep inside validation or arithmetic. Every violation
takes the established non-retryable ``contract_violation`` path and
commits NOTHING; valid measurements from a well-formed FAILED attempt
still commit exactly once.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.orchestration import (
    FAILURE_CONTRACT_VIOLATION,
    INCOMPLETE,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    AgentTaskRequest,
    AgentTaskResult,
    Orchestrator,
    TaskContractError,
    TaskFailure,
    TaskUsage,
    validate_task_result,
)

from m17_doubles import (
    COMMIT,
    REPOSITORY,
    build,
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
        "assignment_reason": "M20 validation check",
        "question": "how does it work?",
        "goal": "Identify the boundary",
        "prompt": "build the report",
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
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


def _failure(**overrides) -> TaskFailure:
    data = {
        "category": "provider",
        "error_type": "LLMError",
        "error_message": "LLMError: rate limited",
        "retryable": True,
    }
    data.update(overrides)
    return TaskFailure(**data)


class _Scripted:
    """Fake boundary echoing the request's per-run ``run_id``."""

    def __init__(self, *results):
        self.requests = []
        self.results = list(results)

    def execute(self, request):
        self.requests.append(request)
        if not self.results:
            raise AssertionError("executor invoked more times than scripted")
        return replace(self.results.pop(0), run_id=request.run_id)


# -- usage field values -----------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_calls", -1),
        ("tool_calls", True),
        ("tool_calls", "2"),
        ("model_iterations", -1),
        ("model_iterations", False),
        ("elapsed_seconds", -0.5),
        ("elapsed_seconds", True),
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
        ("elapsed_seconds", "1.0"),
        ("input_tokens", -1),
        ("input_tokens", True),
        ("output_tokens", -5),
        ("total_tokens", 1.5),
    ],
)
def test_malformed_usage_values_are_rejected(field, value):
    usage = TaskUsage()
    object.__setattr__(usage, field, value)
    with pytest.raises(TaskContractError, match="usage"):
        validate_task_result(_request(), _result(usage=usage))


def test_valid_honest_usage_passes():
    validate_task_result(
        _request(),
        _result(
            usage=TaskUsage(
                tool_calls=2,
                model_iterations=3,
                elapsed_seconds=0.25,
                input_tokens=None,  # unreported stays None
            )
        ),
    )


# -- failure runtime types --------------------------------------------------


@pytest.mark.parametrize("bad", ["boom", 42, object(), {"category": "x"}])
def test_failure_must_be_a_task_failure_before_any_field_read(bad):
    # Completed + garbage failure: must be a TaskContractError, never
    # an AttributeError from reading .category.
    with pytest.raises(TaskContractError, match="TaskFailure"):
        validate_task_result(_request(), _result(failure=bad))
    # Failed + garbage failure: same vocabulary, same safety.
    with pytest.raises(TaskContractError, match="TaskFailure"):
        validate_task_result(
            _request(), _result(status="failed", output=None, failure=bad)
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": 5},
        {"error_type": None},
        {"error_message": 7},
        {"retryable": 1},
        {"termination_reason": 5},
    ],
)
def test_malformed_failure_field_values_are_rejected(overrides):
    with pytest.raises(TaskContractError, match="failure"):
        validate_task_result(
            _request(),
            _result(status="failed", output=None, failure=_failure(**overrides)),
        )


def test_failed_result_with_valid_failure_and_measured_usage_passes():
    validate_task_result(
        _request(),
        _result(
            status="failed",
            output=None,
            failure=_failure(),
            usage=TaskUsage(tool_calls=1, elapsed_seconds=0.1),
        ),
    )


@pytest.mark.parametrize("bad", [42, ("run-x",)])
def test_agent_run_id_must_be_str_or_none(bad):
    with pytest.raises(TaskContractError, match="agent_run_id"):
        validate_task_result(_request(), _result(agent_run_id=bad))


# -- runner integration: contract violation, nothing committed --------------


def test_runner_rejects_a_malformed_usage_without_committing():
    bad_usage = TaskUsage()
    object.__setattr__(bad_usage, "tool_calls", -3)
    executor = _Scripted(_result(usage=bad_usage))
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == INCOMPLETE
    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert "usage" in p1.error
    assert p1.usage is None
    assert p1.usage_total is None
    assert run.tool_calls_total == 0
    assert run.usage_total is None
    assert run.retries == 0  # never retried
    assert p2.status == TASK_BLOCKED
    assert run.report is None


def test_runner_treats_an_unexpected_validator_error_as_a_violation():
    # A TaskUsage whose attribute access explodes models a validator
    # bug (or a hostile result): execute() must still return a
    # terminal run through the contract_violation path — never leak
    # the incidental exception.
    class _BombUsage(TaskUsage):
        @property
        def elapsed_seconds(self):  # type: ignore[override]
            raise RuntimeError("usage exploded")

    # Bypass __init__ (frozen dataclass field assignment vs the
    # property): the instance IS a TaskUsage, but one of its fields
    # explodes on access — exactly the hostile/buggy shape the
    # defensive runner path must survive.
    bomb = object.__new__(_BombUsage)
    executor = _Scripted(_result(usage=bomb))
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == INCOMPLETE
    p1, _ = run.tasks
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert "usage exploded" in p1.error
    assert p1.usage is None
    assert run.tool_calls_total == 0
    assert run.retries == 0


def test_a_valid_failed_attempt_still_commits_its_measured_usage():
    # Only MALFORMED data is excluded: a well-formed failed attempt
    # contributes its honest usage exactly once (then the task fails).
    failure = _failure(category="structured_output", retryable=False)
    executor = _Scripted(
        _result(
            status="failed",
            output=None,
            failure=failure,
            usage=TaskUsage(tool_calls=2, elapsed_seconds=0.5),
        )
    )
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())

    p1, _ = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.usage is not None
    assert p1.usage.tool_calls == 2
    assert p1.usage_total is not None
    assert p1.usage_total.tool_calls == 2
    assert run.tool_calls_total == 2
    assert run.usage_total is not None
    assert run.usage_total.tool_calls == 2


def test_completed_dict_output_still_rejected_before_policy():
    # Free-form dict remains a violation (never a retryable
    # MISSING_STRUCTURED_OUTPUT).
    executor = _Scripted(_result(output={"findings": []}))
    _, investigator, analyst = build()
    orch = Orchestrator(investigator, analyst, executor=executor)
    run = orch.execute(task(), two_step_plan())
    p1, _ = run.tasks
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert p1.output is None


def test_report_payload_still_validates_as_a_real_output():
    # Sanity: a genuine typed report with honest usage passes the
    # hardened validator end to end.
    report = InvestigationReport.model_validate(report_payload())
    validate_task_result(
        _request(),
        _result(output=report, usage=TaskUsage(tool_calls=1)),
    )


def test_nan_and_inf_elapsed_rejected_even_as_float_subclass():
    class _Weird(float):
        pass

    usage = TaskUsage()
    object.__setattr__(usage, "elapsed_seconds", _Weird(math.inf))
    with pytest.raises(TaskContractError, match="elapsed_seconds"):
        validate_task_result(_request(), _result(usage=usage))
