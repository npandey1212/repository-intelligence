"""Test-only execution policy for the one-investigator smoke notebook.

Sanctioned by the smoke-test brief as a small execution policy/wrapper
over the EXISTING task-execution contract (``AgentExecutor.execute``) —
NOT a second orchestration engine, planner, router, or investigator
framework.  The M21 graph has no built-in "stop after N tasks" switch;
without a broad refactor, this wrapper is the smallest executable
enforcement of the restricted experiment's limits:

* at most ONE executed task, and only the allowed task id (Plan 1);
* the Analyst participant raises if it is ever dispatched;
* the provider adapter is guarded so a "dry run" that tries to make a
  real model call FAILS LOUDLY instead of silently spending money;
* the optional live planner call is hard-capped at a configured number
  of calls, stamps an explicit output-token cap onto every request,
  and records each request/response for honest usage reporting.

Loaded from the notebook (importlib, path-based) and from the focused
tests.  Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

__all__ = [
    "OneTaskPolicyViolation",
    "OneTaskExecutionPolicy",
    "AnalystForbidden",
    "FailingInvestigator",
    "ProviderCallAttempted",
    "ProviderState",
    "PROVIDER_STATE",
    "install_provider_guard",
    "reset_provider_guard",
    "LiveCallLimitExceeded",
    "CallLimitedClient",
    "one_task_plan_payload",
]


# -- one-task execution policy (the enforced max-1 rule) ---------------------


class OneTaskPolicyViolation(RuntimeError):
    """A second (or foreign) task execution attempt was blocked."""


class OneTaskExecutionPolicy:
    """Allow exactly ONE ``AgentTaskRequest`` execution.

    Satisfies the existing ``AgentExecutor`` protocol structurally
    (``execute(request) -> AgentTaskResult``) and delegates everything
    it permits to the wrapped executor.  Any attempt beyond
    *max_executed*, or for a task other than *allowed_task_id*, raises
    :class:`OneTaskPolicyViolation` BEFORE the inner executor is
    touched — the limit is executable code, never a prompt hint.
    """

    def __init__(
        self,
        inner: Any,
        allowed_task_id: str,
        max_executed: int = 1,
    ) -> None:
        if max_executed < 1:
            raise ValueError("max_executed must be >= 1")
        self._inner = inner
        self._allowed_task_id = allowed_task_id
        self._max_executed = max_executed
        self.executed: List[str] = []
        self.blocked: List[str] = []

    @property
    def execute_attempts(self) -> int:
        return len(self.executed) + len(self.blocked)

    def execute(self, request: Any) -> Any:
        task_id = str(getattr(request, "task_id", "None"))
        violation: Optional[str] = None
        if len(self.executed) >= self._max_executed:
            violation = "execution limit of {} task(s) reached".format(
                self._max_executed
            )
        elif task_id != self._allowed_task_id:
            violation = (
                "only task {!r} may execute in this experiment"
                .format(self._allowed_task_id)
            )
        if violation is not None:
            self.blocked.append(task_id)
            raise OneTaskPolicyViolation(
                "one-task smoke policy blocked task {!r}: {}".format(
                    task_id, violation
                )
            )
        self.executed.append(task_id)
        return self._inner.execute(request)


# -- participants that must never run ---------------------------------------


class AnalystForbidden:
    """Analyst double: records any invocation, then raises.

    The record survives the raise, so a test can prove the Analyst
    was never dispatched — while the raise itself keeps a stray
    dispatch loud at the boundary.
    """

    def __init__(self) -> None:
        self.calls: List[Any] = []

    @property
    def last_trace(self) -> Any:
        return None

    def run_result(self, question: str, *, max_tool_calls: Any = None) -> Any:
        self.calls.append(question)
        raise AssertionError(
            "the Analyst must not run in the one-investigator smoke test"
        )


class FailingInvestigator:
    """Investigator double that always fails — the no-retry scenario."""

    def __init__(self, message: str = "scripted smoke failure") -> None:
        self.message = message
        self.calls: List[Any] = []

    @property
    def last_trace(self) -> Any:
        return None

    def run_result(self, question: str, *, max_tool_calls: Any = None) -> Any:
        self.calls.append(question)
        raise RuntimeError(self.message)


# -- provider call guard (zero-provider-call enforcement) -------------------


class ProviderCallAttempted(RuntimeError):
    """A real provider call was attempted while the guard blocked it."""


class ProviderState:
    """Mutable state of the class-level provider guard."""

    def __init__(self) -> None:
        self.limit: int = 0
        self.calls: int = 0
        self.passed: List[Any] = []
        self.installed: bool = False

    @property
    def blocked_calls(self) -> int:
        return max(0, self.calls - self.limit)


PROVIDER_STATE = ProviderState()
_ORIGINAL_COMPLETE: Optional[Any] = None


def _guarded_complete(self: Any, request: Any) -> Any:
    state = PROVIDER_STATE
    attempt = state.calls + 1
    state.calls = attempt
    if attempt > state.limit:
        raise ProviderCallAttempted(
            "provider call #{} blocked by the smoke-test guard "
            "(limit {}): {}".format(
                attempt, state.limit, type(self).__name__
            )
        )
    response = _ORIGINAL_COMPLETE(self, request)
    state.passed.append(request)
    return response


def install_provider_guard(limit: int = 0) -> ProviderState:
    """Patch the ONE provider adapter (``OpenAILLMClient.complete``).

    ``limit=0`` (the dry-run default) makes ANY provider call raise
    :class:`ProviderCallAttempted` — a notebook that merely labels
    itself "dry run" while calling the provider is not acceptable, so
    the guard makes it impossible.  ``limit=1`` lets exactly one
    call pass through (the opt-in live experiment); call #2 raises.
    """
    global _ORIGINAL_COMPLETE
    from repo_intelligence.llm.openai_client import OpenAILLMClient

    reset_provider_guard()
    if _ORIGINAL_COMPLETE is None:
        _ORIGINAL_COMPLETE = OpenAILLMClient.complete
    PROVIDER_STATE.limit = limit
    PROVIDER_STATE.calls = 0
    PROVIDER_STATE.passed = []
    OpenAILLMClient.complete = _guarded_complete
    PROVIDER_STATE.installed = True
    return PROVIDER_STATE


def reset_provider_guard() -> None:
    """Restore the real adapter (test teardown / notebook cleanup)."""
    global _ORIGINAL_COMPLETE
    if not PROVIDER_STATE.installed:
        return
    from repo_intelligence.llm.openai_client import OpenAILLMClient

    if _ORIGINAL_COMPLETE is not None:
        OpenAILLMClient.complete = _ORIGINAL_COMPLETE
    PROVIDER_STATE.installed = False


# -- live-call guard (one-call cap + output cap + usage capture) -------------


class LiveCallLimitExceeded(RuntimeError):
    """The configured live-call limit was exhausted."""


class CallLimitedClient:
    """``LLMClient`` wrapper for the ONE-call live experiment.

    * hard-calls cap: call *max_calls + 1* raises
      :class:`LiveCallLimitExceeded` BEFORE the inner client is
      touched — no silent extra calls;
    * output cap: every request is rewritten with ``max_tokens``
      (the provider's own parameter, via the existing ``LLMRequest``
      contract) when *max_tokens* is set;
    * experiment input ceiling: when *input_budget* (a
      ``RequestBudgetPolicy``) is set, the request is measured with the
      SAME H1 estimator the application uses and an over-ceiling
      request is REJECTED via ``enforce_request_budget_or_raise`` —
      the provider is never invoked, nothing is silently truncated, and
      the rejected record is kept for honest reporting. Only requests
      that actually reach the inner client consume the call allowance;
    * honesty: every (request, response) pair that reached the inner
      client is recorded so the notebook can report provider-reported
      usage without guessing.
    """

    def __init__(
        self,
        inner: Any,
        max_calls: int = 1,
        max_tokens: Optional[int] = None,
        input_budget: Optional[Any] = None,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 or None")
        self._inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.input_budget = input_budget
        self.requests: List[Any] = []
        self.responses: List[Any] = []
        self.input_records: List[Dict[str, Any]] = []
        self.rejected_inputs: List[Dict[str, Any]] = []

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def calls_made(self) -> int:
        return len(self.responses)

    def complete(self, request: Any) -> Any:
        if len(self.requests) >= self.max_calls:
            raise LiveCallLimitExceeded(
                "live smoke limit of {} provider call(s) exhausted; "
                "a second call requires a new, deliberate experiment"
                .format(self.max_calls)
            )
        if self.input_budget is not None:
            from repo_intelligence.agent.request_budget import (
                RequestBudgetExceededError,
                enforce_request_budget_or_raise,
            )

            try:
                request, record = enforce_request_budget_or_raise(
                    request,
                    self.input_budget,
                    subject="live planner request",
                )
            except RequestBudgetExceededError as exc:
                self.rejected_inputs.append(exc.record)
                raise
            self.input_records.append(record)
        if self.max_tokens is not None:
            request = dataclasses.replace(
                request, max_tokens=self.max_tokens
            )
        self.requests.append(request)
        response = self._inner.complete(request)
        self.responses.append(response)
        return response


# -- the fixture plan (exactly one Investigator step) -----------------------


def one_task_plan_payload(question: str) -> Dict[str, Any]:
    """The smoke test's fixture PLAN: exactly one Investigator step.

    Defense-in-depth alongside :class:`OneTaskExecutionPolicy` — the
    POLICY is the enforced limit; this fixture keeps the canonical dry
    run minimal (Plan 1 only, no Analyst step).  The goal text
    classifies as "architecture" under the existing QueryRouter
    keyword families.
    """
    return {
        "objective": question,
        "steps": [
            {
                "id": "P1",
                "goal": "Document the architecture components and "
                "their relationships in this repository.",
            },
        ],
    }
