"""M19 task contracts — immutability, failure taxonomy, usage accounting.

Scripted and LLM-free: every classification predicate is the exact one
the M17 retry loop used (``AgentError.termination_reason``,
``is_context_length_error``), so these tests pin the ONE taxonomy that
now decides retryability for every dispatch.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from repo_intelligence.agent import AgentError
from repo_intelligence.llm import LLMError
from repo_intelligence.orchestration import (
    FAILURE_AGENT_BUDGET,
    FAILURE_CONTEXT_EXHAUSTED,
    FAILURE_FINALIZATION,
    FAILURE_PROVIDER,
    FAILURE_UNEXPECTED,
    TASK_STATUSES,
    AgentTaskRequest,
    AgentTaskResult,
    TaskFailure,
    TaskUsage,
    classify_exception,
)


def _request(**overrides) -> AgentTaskRequest:
    data = {
        "run_id": "r1",
        "task_id": "P1",
        "attempt": 1,
        "role": "INVESTIGATOR",
        "assignment_reason": "first investigator step",
        "question": "SECRET_QUESTION how does it work?",
        "goal": "Identify the boundary",
        "prompt": "SECRET_PROMPT build the report",
        "repository_scope": "example-repository",
        "commit_scope": "c" * 40,
        "depends_on": ("P0",),
        "max_tool_calls": 4,
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
    }
    data.update(overrides)
    return AgentTaskResult(**data)


# -- immutability ----------------------------------------------------------


def test_request_is_frozen_so_nothing_can_rewrite_a_dispatch():
    request = _request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.prompt = "rewritten"


def test_result_is_frozen_so_a_committed_outcome_cannot_drift():
    result = _result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "failed"


def test_task_failure_is_frozen():
    failure = classify_exception(RuntimeError("boom"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.retryable = True


def test_task_usage_is_frozen():
    usage = TaskUsage(tool_calls=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        usage.tool_calls = 2


# -- request envelope safety ------------------------------------------------


def test_request_summary_is_metadata_only_never_prompt_text():
    summary = _request().summary()
    blob = repr(summary)
    assert "SECRET_PROMPT" not in blob
    assert "SECRET_QUESTION" not in blob
    assert summary["task_id"] == "P1"
    assert summary["attempt"] == 1
    assert summary["role"] == "INVESTIGATOR"
    assert summary["depends_on"] == ["P0"]
    assert summary["max_tool_calls"] == 4
    assert summary["scope"] == "example-repository@cccccccccccc"


def test_request_carries_task_scoped_dependency_results():
    report = object()
    request = _request(dependency_results=(report,))
    assert request.dependency_results == (report,)
    assert request.depends_on == ("P0",)


# -- the one failure taxonomy ----------------------------------------------


def test_task_status_vocabulary_is_exactly_the_documented_set():
    assert TASK_STATUSES == (
        "pending",
        "ready",
        "running",
        "completed",
        "incomplete",
        "failed",
        "blocked",
        "cancelled",
    )


def test_agent_budget_failure_is_never_retryable():
    failure = classify_exception(
        AgentError("budget gone", termination_reason="EXECUTION_LIMIT_REACHED")
    )
    assert failure.category == FAILURE_AGENT_BUDGET
    assert failure.retryable is False
    assert failure.error_type == "AgentError"
    assert failure.error_message.startswith("AgentError: ")
    assert failure.termination_reason == "EXECUTION_LIMIT_REACHED"


def test_finalization_failure_is_never_retryable():
    failure = classify_exception(
        AgentError("closed early", termination_reason="FINALIZATION_FAILED")
    )
    assert failure.category == FAILURE_FINALIZATION
    assert failure.retryable is False
    assert failure.termination_reason == "FINALIZATION_FAILED"


def test_context_exhaustion_is_never_retryable():
    failure = classify_exception(
        LLMError("model raised context_length_exceeded")
    )
    assert failure.category == FAILURE_CONTEXT_EXHAUSTED
    assert failure.retryable is False


def test_ordinary_provider_failures_stay_retryable():
    failure = classify_exception(LLMError("rate limit exceeded"))
    assert failure.category == FAILURE_PROVIDER
    assert failure.retryable is True


def test_unexpected_exception_defaults_to_retryable_unexpected():
    failure = classify_exception(RuntimeError("boom"))
    assert failure.category == FAILURE_UNEXPECTED
    assert failure.retryable is True
    assert failure.error_message == "RuntimeError: boom"


def test_result_exposes_retryable_and_error_message_from_its_failure():
    result = _result(
        status="failed",
        failure=classify_exception(
            AgentError("budget", termination_reason="EXECUTION_LIMIT_REACHED")
        ),
    )
    assert result.retryable is False
    assert result.error_message.startswith("AgentError: ")
    ok = _result()
    assert ok.retryable is False
    assert ok.error_message is None


# -- usage accounting (honest fields only) ---------------------------------


def test_task_usage_defaults_are_zero_and_tokens_stay_none():
    usage = TaskUsage()
    assert usage.tool_calls == 0
    assert usage.model_iterations == 0
    assert usage.elapsed_seconds == 0.0
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_usage_from_trace_sums_only_reported_provider_tokens():
    from repo_intelligence.orchestration.contracts import usage_from_trace

    def iteration(tool_calls, response):
        return SimpleNamespace(tool_calls=tool_calls, response=response)

    trace = SimpleNamespace(
        iterations=[
            iteration(
                [{"name": "read_file"}],
                {"usage": {"input_tokens": 10, "output_tokens": 2}},
            ),
            iteration([], {}),
            iteration(
                [],
                {"usage": {"input_tokens": 5, "total_tokens": 20}},
            ),
        ]
    )
    usage = usage_from_trace(trace, tool_calls=1, elapsed_seconds=0.5)
    assert usage.tool_calls == 1
    assert usage.model_iterations == 3
    assert usage.elapsed_seconds == 0.5
    assert usage.input_tokens == 15
    assert usage.output_tokens == 2
    assert usage.total_tokens == 20


def test_usage_from_trace_without_reported_usage_keeps_tokens_none():
    from repo_intelligence.orchestration.contracts import usage_from_trace

    trace = SimpleNamespace(
        iterations=[SimpleNamespace(tool_calls=[], response={})]
    )
    usage = usage_from_trace(trace, tool_calls=0, elapsed_seconds=0.1)
    assert usage.model_iterations == 1
    assert usage.input_tokens is None
    assert usage.total_tokens is None
