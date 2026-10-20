"""M19 Orchestrator↔Agent task-execution contracts (framework-neutral).

The narrow, typed boundary between the coordinator (``Orchestrator``)
and ONE agent task execution: one immutable ``AgentTaskRequest`` in,
one immutable ``AgentTaskResult`` out. No shared mutable run state
crosses this boundary (the agent never sees the ``OrchestrationRun``),
and no framework class appears here — LangGraph (or anything else) may
later CALL this boundary; it never defines it.

Lifecycle of one task execution through this contract:

    created/selected → request validated → started → running →
        result returned → result validated (coordinator side) →
        completed | failed

Failure taxonomy (``TaskFailure`` / ``classify_exception``) preserves
the distinctions the M17 retry policy depends on, decided ONCE here so
every executor maps exceptions identically:

    provider                 transient provider/LLM failure — retryable
    context_exhausted        provider context-length error — NEVER
                             retried as-is (same request reproduces it)
    agent_execution_budget   the agent's own tool budget
                             (EXECUTION_LIMIT_REACHED) — never retried
    finalization_failed      a failed tool-free finalization pass —
                             never retried (the run already closed from
                             the state it had)
    structured_output        structured-output validation failure —
                             retried under the same bounded policy as
                             before (a different response may validate)
    invalid_response         a malformed LLM response — retryable
    dispatch_error           tool call with no dispatcher — retryable
                             (configuration, but the bounded retry keeps
                             today's semantics)
    contract_violation       the executor returned a result that breaks
                             this boundary's contract (correlation
                             fields that do not echo the request, an
                             unsupported status, or a status/failure/
                             output combination the coordinator cannot
                             commit) — NEVER retried: it is an
                             implementation bug at the boundary, and a
                             replay would violate it identically
    unexpected               anything else — retryable under the same
                             bounded policy as before

Honesty rules: usage values come from actual measurements only —
tool calls and model iterations are counted from the execution trace;
tokens are summed from provider-reported ``usage`` metadata when
present and stay ``None`` when the provider did not report them (never
fabricated). Elapsed time is the executor's own wall clock around the
participant call. Application-generated reasons and statuses are never
model-authored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Tuple

from ..agent import AgentError
from ..llm import (
    LLMError,
    StructuredOutputValidationError,
    is_context_length_error,
)

__all__ = [
    "FAILURE_AGENT_BUDGET",
    "FAILURE_CONTEXT_EXHAUSTED",
    "FAILURE_CONTRACT_VIOLATION",
    "FAILURE_DISPATCH_ERROR",
    "FAILURE_FINALIZATION",
    "FAILURE_INVALID_RESPONSE",
    "FAILURE_SCOPE_MISMATCH",
    "FAILURE_STRUCTURED_OUTPUT",
    "FAILURE_UNEXPECTED",
    "FAILURE_PROVIDER",
    "TASK_BLOCKED",
    "TASK_CANCELLED",
    "TASK_COMPLETED",
    "TASK_FAILED",
    "TASK_INCOMPLETE",
    "TASK_PENDING",
    "TASK_READY",
    "TASK_RUNNING",
    "TASK_STATUSES",
    "AgentTaskRequest",
    "AgentTaskResult",
    "TaskContractError",
    "TaskFailure",
    "TaskUsage",
    "classify_exception",
    "combine_usage",
    "validate_task_result",
]


def _utcnow() -> str:
    """UTC timestamp in the orchestration event format.

    Deliberately self-contained (same format as
    ``orchestration.models.utcnow``): this module is the BOTTOM of the
    orchestration import graph — models may import it, so it must
    never import models.
    """
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# -- task status vocabulary (one canonical set, coordinator-owned) ---------

TASK_PENDING = "pending"
TASK_READY = "ready"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_INCOMPLETE = "incomplete"
TASK_FAILED = "failed"
TASK_BLOCKED = "blocked"
TASK_CANCELLED = "cancelled"

TASK_STATUSES = (
    TASK_PENDING,
    TASK_READY,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_INCOMPLETE,
    TASK_FAILED,
    TASK_BLOCKED,
    TASK_CANCELLED,
)

# -- failure categories -----------------------------------------------------

FAILURE_PROVIDER = "provider"
FAILURE_CONTEXT_EXHAUSTED = "context_exhausted"
FAILURE_AGENT_BUDGET = "agent_execution_budget"
FAILURE_FINALIZATION = "finalization_failed"
FAILURE_STRUCTURED_OUTPUT = "structured_output"
FAILURE_SCOPE_MISMATCH = "scope_mismatch"
FAILURE_INVALID_RESPONSE = "invalid_response"
FAILURE_DISPATCH_ERROR = "dispatch_error"
FAILURE_UNEXPECTED = "unexpected"
FAILURE_CONTRACT_VIOLATION = "contract_violation"


class TaskContractError(RuntimeError):
    """An executor returned a result that violates the task contract.

    Raised by ``validate_task_result`` on the COORDINATOR side of the
    boundary (a returned result is never trusted before commit):
    correlation fields that do not echo the request, an unsupported
    result status, a ``usage``/``evidence_refs`` value outside the
    declared types, or a status/failure/output combination the runner
    cannot commit. It is an implementation bug at the boundary, not a
    transient failure — the orchestrator fails the task with
    ``FAILURE_CONTRACT_VIOLATION`` (non-retryable) and commits NO
    result-derived state (no output, usage, trace, or correlation id).
    """


@dataclass(frozen=True)
class TaskFailure:
    """Structured classification of one failed task execution.

    ``error_type`` is the exception class name and ``error_message`` is
    the human-readable "Type: message" form — both preserved so logs
    and events stay actionable. ``retryable`` is decided HERE (the one
    taxonomy), never by the convenience of the retry loop.
    """

    category: str
    error_type: str
    error_message: str
    retryable: bool
    termination_reason: Optional[str] = None


def classify_exception(exc: BaseException) -> TaskFailure:
    """Map one exception from a participant call to a ``TaskFailure``.

    Reuses the exact runtime predicates the M17 loop used
    (``AgentError.termination_reason``,
    ``is_context_length_error``) so classification is behavior-preserving,
    not a re-interpretation.
    """
    error_type = type(exc).__name__
    error_message = "{}: {}".format(error_type, exc)
    termination = getattr(exc, "termination_reason", None)
    if isinstance(exc, AgentError) and termination == (
        "EXECUTION_LIMIT_REACHED"
    ):
        return TaskFailure(
            category=FAILURE_AGENT_BUDGET,
            error_type=error_type,
            error_message=error_message,
            retryable=False,
            termination_reason=termination,
        )
    if isinstance(exc, AgentError) and termination == "FINALIZATION_FAILED":
        return TaskFailure(
            category=FAILURE_FINALIZATION,
            error_type=error_type,
            error_message=error_message,
            retryable=False,
            termination_reason=termination,
        )
    if is_context_length_error(exc):
        return TaskFailure(
            category=FAILURE_CONTEXT_EXHAUSTED,
            error_type=error_type,
            error_message=error_message,
            retryable=False,
            termination_reason=termination,
        )
    if isinstance(exc, LLMError):
        return TaskFailure(
            category=FAILURE_PROVIDER,
            error_type=error_type,
            error_message=error_message,
            retryable=True,
            termination_reason=termination,
        )
    if isinstance(exc, StructuredOutputValidationError):
        return TaskFailure(
            category=FAILURE_STRUCTURED_OUTPUT,
            error_type=error_type,
            error_message=error_message,
            retryable=True,
            termination_reason=termination,
        )
    if isinstance(exc, AgentError) and termination == "TOOL_DISPATCH_ERROR":
        return TaskFailure(
            category=FAILURE_DISPATCH_ERROR,
            error_type=error_type,
            error_message=error_message,
            retryable=True,
            termination_reason=termination,
        )
    if isinstance(exc, AgentError) and termination == "INVALID_LLM_RESPONSE":
        return TaskFailure(
            category=FAILURE_INVALID_RESPONSE,
            error_type=error_type,
            error_message=error_message,
            retryable=True,
            termination_reason=termination,
        )
    return TaskFailure(
        category=FAILURE_UNEXPECTED,
        error_type=error_type,
        error_message=error_message,
        retryable=True,
        termination_reason=termination,
    )


@dataclass(frozen=True)
class TaskUsage:
    """Measured usage of ONE task execution (never fabricated).

    ``tool_calls`` / ``model_iterations`` come from the execution
    trace; ``elapsed_seconds`` is the executor's wall clock around the
    participant call. Token fields are summed from provider-reported
    per-iteration usage metadata and stay ``None`` when no provider
    reported any (the trace layer records usage only when the provider
    supplies it).
    """

    tool_calls: int = 0
    model_iterations: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


def combine_usage(*usages: Optional[TaskUsage]) -> TaskUsage:
    """Sum measured usage across attempts of one task (honest fields).

    Counters and elapsed time always sum. Token fields are summed
    only over attempts whose provider REPORTED them and stay ``None``
    while none did — a missing measurement never becomes a fabricated
    ``0``. ``None`` arguments (no attempt committed yet) are allowed.
    """
    tool_calls = 0
    model_iterations = 0
    elapsed_seconds = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    for usage in usages:
        if usage is None:
            continue
        tool_calls += usage.tool_calls
        model_iterations += usage.model_iterations
        elapsed_seconds += usage.elapsed_seconds
        if usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + usage.output_tokens
        if usage.total_tokens is not None:
            total_tokens = (total_tokens or 0) + usage.total_tokens
    return TaskUsage(
        tool_calls=tool_calls,
        model_iterations=model_iterations,
        elapsed_seconds=elapsed_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


@dataclass(frozen=True)
class AgentTaskRequest:
    """One immutable task execution request (the orchestrator's only
    input to an executor).

    Carries the task-scoped envelope — identity/correlation, role and
    assignment reason, repository/commit scope, validated dependency
    ids, the dependency results explicitly allowed as input, and the
    task-scoped prompt. It never carries the mutable
    ``OrchestrationRun``, unrelated task outputs, secrets, or hidden
    reasoning.
    """

    run_id: str
    task_id: str
    attempt: int
    role: str
    assignment_reason: str
    question: str
    goal: str
    prompt: str
    repository_scope: str
    commit_scope: str
    depends_on: Tuple[str, ...] = ()
    dependency_results: Tuple[Any, ...] = ()
    # M20: the PER-ATTEMPT tool-call limit carried to the execution
    # boundary. ``None`` = no task-level override (the participant's
    # own ``AgentExecutionConfig.max_tool_calls`` applies); an int
    # (>= 0, bool rejected) is enforced by the default adapter at the
    # agent's pre-dispatch guard BEFORE any tool executes — one
    # attempt gets one allowance (a retry is a fresh attempt with its
    # own limit). ``0`` = no tool call may execute this attempt.
    # Enforced only when the participant supports the per-call
    # override (Agent.run_result and the specialists do); the
    # run-level bound remains OrchestrationConfig.max_total_tool_calls.
    max_tool_calls: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_tool_calls is None:
            return
        if not isinstance(self.max_tool_calls, int) or isinstance(
            self.max_tool_calls, bool
        ):
            raise ValueError(
                "max_tool_calls must be an integer or None, got {!r}"
                .format(self.max_tool_calls)
            )
        if self.max_tool_calls < 0:
            raise ValueError(
                "max_tool_calls must be >= 0, got {}".format(
                    self.max_tool_calls
                )
            )

    def summary(self) -> dict:
        """A safe, bounded summary for timelines/logs — metadata only,
        never prompt text or dependency payloads."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "role": self.role,
            "scope": "{}@{}".format(
                self.repository_scope, (self.commit_scope or "")[:12]
            ),
            "depends_on": list(self.depends_on),
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass(frozen=True)
class AgentTaskResult:
    """One immutable result of ONE task execution.

    ``output`` is the participant's typed payload (``InvestigationReport``,
    ``ArchitectureFinding``, or a specialist contract such as
    ``ArchitectureReport`` / ``FlowDependencyReport`` / ``DiagramSpec``)
    — never an unvalidated free-form dictionary. ``status`` uses the
    ``TASK_*`` vocabulary; a failure carries a structured
    ``TaskFailure`` (category + retryability decided once).
    ``agent_run_id`` is the execution's trace correlation id, and
    ``trace`` (when the participant exposed one) is that execution's
    own trace object — never a previous run's ``last_trace`` (the
    adapter only returns a trace that appeared/was replaced during
    THIS call). ``trace`` is an in-process reference, not part of any
    serialized boundary.
    """

    run_id: str
    task_id: str
    attempt: int
    role: str
    status: str
    output: Optional[Any] = None
    failure: Optional[TaskFailure] = None
    usage: TaskUsage = field(default_factory=TaskUsage)
    evidence_refs: Tuple[Any, ...] = ()
    agent_run_id: Optional[str] = None
    trace: Optional[Any] = None
    started_at: str = ""
    completed_at: str = ""

    @property
    def retryable(self) -> bool:
        return self.failure is not None and self.failure.retryable

    @property
    def error_message(self) -> Optional[str]:
        return self.failure.error_message if self.failure else None




def _validate_task_usage(usage: Any) -> None:
    """Runtime validation of ONE ``TaskUsage``'s field values.

    A ``frozen=True`` dataclass and type annotations do NOT guarantee
    valid runtime values — every field is checked explicitly so a
    malformed usage can never leak into arithmetic
    (``combine_usage``), budget counters, or committed task state, and
    so the caller sees a ``TaskContractError`` (the boundary's own
    vocabulary) instead of an incidental ``TypeError`` /
    ``ValueError`` from a later ``+`` or comparison.

    Rules: ``tool_calls`` / ``model_iterations`` are non-negative
    ints (``bool`` rejected); ``elapsed_seconds`` is a finite
    non-negative number (``bool`` rejected — a bool elapsed time is
    nonsense, not ``0.0``/``1.0``); token fields are ``None``
    (unreported — never fabricated) or non-negative ints.
    """
    for name in ("tool_calls", "model_iterations"):
        value = getattr(usage, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TaskContractError(
                "result.usage.{} is {!r}; expected int".format(
                    name, value
                )
            )
        if value < 0:
            raise TaskContractError(
                "result.usage.{} is {}; expected >= 0".format(name, value)
            )
    elapsed = usage.elapsed_seconds
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise TaskContractError(
            "result.usage.elapsed_seconds is {!r}; expected number".format(
                elapsed
            )
        )
    if not math.isfinite(elapsed) or elapsed < 0:
        raise TaskContractError(
            "result.usage.elapsed_seconds is {}; expected a finite "
            "value >= 0".format(elapsed)
        )
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, name)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise TaskContractError(
                "result.usage.{} is {!r}; expected int or None".format(
                    name, value
                )
            )
        if value < 0:
            raise TaskContractError(
                "result.usage.{} is {}; expected >= 0".format(name, value)
            )


def _validate_task_failure(failure: Any) -> None:
    """Runtime validation of ONE ``TaskFailure``'s field values.

    Checked BEFORE any attribute is read by the coordinator (a
    malformed failure becomes a ``TaskContractError``, never an
    ``AttributeError`` deep inside validation): ``category`` /
    ``error_type`` / ``error_message`` are strings, ``retryable`` is
    a bool (``int`` rejected — an executor may not smuggle ``1``),
    and ``termination_reason`` is ``None`` or a string.
    """
    if not isinstance(failure, TaskFailure):
        raise TaskContractError(
            "failure is {}; expected TaskFailure".format(
                type(failure).__name__
            )
        )
    for name in ("category", "error_type", "error_message"):
        value = getattr(failure, name)
        if not isinstance(value, str):
            raise TaskContractError(
                "failure.{} is {!r}; expected str".format(name, value)
            )
    if not isinstance(failure.retryable, bool):
        raise TaskContractError(
            "failure.retryable is {!r}; expected bool".format(
                failure.retryable
            )
        )
    if failure.termination_reason is not None and not isinstance(
        failure.termination_reason, str
    ):
        raise TaskContractError(
            "failure.termination_reason is {!r}; expected str or "
            "None".format(failure.termination_reason)
        )


def validate_task_result(
    request: AgentTaskRequest, result: Any
) -> None:
    """Validate ONE returned result against its request (coordinator
    side of the boundary — the runner calls this BEFORE committing any
    result-derived state).

    Raises ``TaskContractError`` on the first violation:

    * ``result`` is not an ``AgentTaskResult``;
    * correlation fields do not echo the request (``run_id``,
      ``task_id``, ``attempt``, ``role``);
    * ``usage`` is not a ``TaskUsage`` or ``evidence_refs`` is not a
      tuple (the declared annotation types);
    * ``usage`` field VALUES are malformed (negative counters, a
      bool/negative/non-finite elapsed time, invalid token counts) —
      rejected here so arithmetic later can never fail incidentally;
    * ``status`` is not one of the two result statuses the coordinator
      commits — ``completed`` / ``failed``. ``incomplete`` /
      ``cancelled`` are coordinator-side TASK_RECORD states, never
      result statuses here (an executor may not invent outcomes);
    * ``completed`` carries a non-``TaskFailure``/``TaskFailure``
      payload, or ``failed`` carries no valid ``TaskFailure`` (a
      string or unrelated object is rejected by runtime type check,
      before any field is read), or ``failed`` carries an ``output``;
    * ``completed`` output is a free-form ``dict`` (the contract
      requires a typed payload — whether it is the EXPECTED type
      stays the runner's orchestration decision);
    * ``agent_run_id`` is neither ``None`` nor a string.

    Deliberately ALLOWED: ``completed`` + ``output=None`` — an
    executor that genuinely produced no payload; the runner classifies
    that as its retryable ``MISSING_STRUCTURED_OUTPUT`` outcome.
    Validation here is about contract shape, not orchestration policy.
    """
    if not isinstance(result, AgentTaskResult):
        raise TaskContractError(
            "executor returned {}; expected AgentTaskResult".format(
                type(result).__name__
            )
        )
    for name in ("run_id", "task_id", "attempt", "role"):
        expected = getattr(request, name)
        observed = getattr(result, name)
        if observed != expected:
            raise TaskContractError(
                "result.{} {!r} does not echo request.{} {!r}".format(
                    name, observed, name, expected
                )
            )
    if not isinstance(result.usage, TaskUsage):
        raise TaskContractError(
            "result.usage is {}; expected TaskUsage".format(
                type(result.usage).__name__
            )
        )
    _validate_task_usage(result.usage)
    if not isinstance(result.evidence_refs, tuple):
        raise TaskContractError(
            "result.evidence_refs is {}; expected tuple".format(
                type(result.evidence_refs).__name__
            )
        )
    if result.status not in (TASK_COMPLETED, TASK_FAILED):
        raise TaskContractError(
            "result.status {!r} is not a committable result status "
            "(expected {!r} or {!r}; {!r}/{!r} are coordinator-side "
            "record states an executor may never return)".format(
                result.status,
                TASK_COMPLETED,
                TASK_FAILED,
                TASK_INCOMPLETE,
                TASK_CANCELLED,
            )
        )
    if result.status == TASK_COMPLETED:
        if result.failure is not None:
            _validate_task_failure(result.failure)
            raise TaskContractError(
                "completed result carries a failure ({}: {})".format(
                    result.failure.category, result.failure.error_message
                )
            )
        if isinstance(result.output, dict):
            raise TaskContractError(
                "completed result output is a free-form dict; the "
                "contract requires a typed payload"
            )
    else:
        # Runtime type check FIRST — a string/unrelated object as
        # ``failure`` is a contract violation, never an incidental
        # AttributeError from reading ``.category`` below.
        _validate_task_failure(result.failure)
        if result.output is not None:
            raise TaskContractError(
                "failed result must not carry an output"
            )
    if result.agent_run_id is not None and not isinstance(
        result.agent_run_id, str
    ):
        raise TaskContractError(
            "result.agent_run_id is {!r}; expected str or None".format(
                result.agent_run_id
            )
        )


def usage_from_trace(
    trace: Any, tool_calls: int, elapsed_seconds: float
) -> TaskUsage:
    """Derive ``TaskUsage`` from one execution trace (honest fields only).

    Iterations and tool calls are counted from the trace; tokens are
    summed from provider-reported per-iteration usage metadata and are
    ``None`` when none was reported.
    """
    iterations = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    if trace is not None:
        iterations = len(getattr(trace, "iterations", ()) or ())
        for iteration in getattr(trace, "iterations", ()) or ():
            meta = getattr(iteration, "response", None) or {}
            usage = meta.get("usage") if hasattr(meta, "get") else None
            if not isinstance(usage, dict):
                continue
            if usage.get("input_tokens") is not None:
                input_tokens = (input_tokens or 0) + int(
                    usage["input_tokens"]
                )
            if usage.get("output_tokens") is not None:
                output_tokens = (output_tokens or 0) + int(
                    usage["output_tokens"]
                )
            if usage.get("total_tokens") is not None:
                total_tokens = (total_tokens or 0) + int(
                    usage["total_tokens"]
                )
    return TaskUsage(
        tool_calls=tool_calls,
        model_iterations=iterations,
        elapsed_seconds=elapsed_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def now_timestamp() -> str:
    """Current UTC timestamp in the orchestration event format."""
    return _utcnow()
