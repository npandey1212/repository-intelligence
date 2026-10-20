"""M19 agent task executor — the Orchestrator↔participant adapter.

One narrow boundary: the coordinator dispatches an immutable
``AgentTaskRequest`` and receives an immutable ``AgentTaskResult``.
Model clients, tool dispatchers, and prompt construction stay OUT of
the Orchestrator — the executor owns "run this one task on this one
participant and report what happened".

Participant selection is deterministic (``ParticipantRegistry``): a
role maps to exactly one registered participant, and an unknown role
is rejected explicitly with the known roles named — never a silent
fallback, never a marketplace.

``AgentTaskExecutor`` (the default adapter) works with BOTH the
generic ``Agent`` and the specialist agents
(``ArchitectureInvestigator``, ``FlowDependencyInvestigator``,
``ArchitectureVisualizationAgent``): each of them exposes
``run_result(question)`` returning an object with
``structured_output``, plus a ``last_trace`` attribute/property — the
adapter needs nothing more, and typed specialist outputs
(``ArchitectureReport`` / ``FlowDependencyReport`` / ``DiagramSpec``)
travel as ``AgentTaskResult.output`` uncoerced. M20: when the request
carries a per-attempt ``max_tool_calls`` limit, the adapter calls
``run_result(question, max_tool_calls=...)`` so the limit reaches the
participant's own pre-tool guard (the None path is the exact M19
single-argument call).

Trace attribution: the adapter snapshots the participant's
``last_trace`` BEFORE the call and only reports a trace that differs
from that snapshot — a result can never inherit a PREVIOUS
execution's trace id. Isolation reality (documented, not assumed
away): participant conversation state is local to each
``run_result`` call, so repeated SEQUENTIAL executions on one
participant are safe; CONCURRENT executions on one participant are
not (the shared ``last_trace`` / error fields would clobber each
other) — until parallel dispatch lands, concurrency isolation belongs
to a factory/worker policy, not to this adapter.

No LangGraph (or other framework) class appears anywhere in this
module; the boundary is plain Python and unit-testable with fake
participants and no network.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from .contracts import (
    TASK_COMPLETED,
    TASK_FAILED,
    AgentTaskRequest,
    AgentTaskResult,
    classify_exception,
    usage_from_trace,
)
from .models import utcnow
from .roles import ANALYST, INVESTIGATOR

__all__ = [
    "AgentExecutor",
    "AgentTaskExecutor",
    "ParticipantRegistry",
]


class AgentExecutor(Protocol):
    """The Orchestrator's dispatch boundary for ONE task execution.

    Implementations MUST return a result for participant failures
    (mapped, never raised) so the coordinator's retry/termination
    policy stays in one place; raising is reserved for boundary
    misuse (wrong request type, unknown role — a bug, not an
    outcome).
    """

    def execute(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Execute one task and return its typed result."""
        ...


class ParticipantRegistry:
    """Deterministic role → participant mapping (explicit, rejectable).

    Selection never falls back: an unknown role raises ``ValueError``
    naming the known roles, so a typo can never silently dispatch to
    the wrong participant.
    """

    def __init__(
        self, participants: Optional[Mapping[str, Any]] = None
    ) -> None:
        self._participants: Dict[str, Any] = {}
        for role, participant in dict(participants or {}).items():
            self.register(role, participant)

    def register(self, role: str, participant: Any) -> None:
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                "role must be a non-empty string, got {!r}".format(role)
            )
        if participant is None:
            raise ValueError(
                "participant for role {!r} must not be None".format(role)
            )
        if role in self._participants:
            raise ValueError(
                "role {!r} is already registered".format(role)
            )
        self._participants[role] = participant

    def select(self, role: str) -> Any:
        if role not in self._participants:
            known = ", ".join(sorted(self._participants)) or "(none)"
            raise ValueError(
                "no participant registered for role {!r} (known "
                "roles: {})".format(role, known)
            )
        return self._participants[role]

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(sorted(self._participants))


def _count_tool_calls(trace: Any, trace_before: Any) -> int:
    """EXECUTED tool calls recorded on THIS execution's trace only.

    An executed dispatch appends exactly one ``tool_results`` entry
    (a failed tool still executed and still counts); ``tool_calls``
    on an iteration are the *requested* calls — a blocked call never
    executes and must never be counted (M20: honest per-attempt
    usage — requested/blocked/planned calls are not executions).
    """
    if trace is None or trace is trace_before:
        return 0
    return sum(
        len(getattr(iteration, "tool_results", ()) or ())
        for iteration in trace.iterations
    )


def _fresh_trace(participant: Any, trace_before: Any) -> Optional[Any]:
    """The participant's trace when it appeared/was replaced during
    THIS call; ``None`` when nothing new ran (never a previous run's
    trace)."""
    trace = getattr(participant, "last_trace", None)
    if trace is None or trace is trace_before:
        return None
    return trace


class AgentTaskExecutor:
    """Default adapter: a run_result-style participant → task result.

    On success the participant's ``structured_output`` becomes
    ``AgentTaskResult.output`` (typed, or ``None`` when the run
    produced no structured payload — the coordinator decides whether
    that is a failure against its expected type). On an exception the
    failure is classified ONCE via ``classify_exception`` and returned
    as a failed result; only boundary misuse raises.
    """

    def __init__(
        self,
        registry: ParticipantRegistry,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._clock = clock

    @classmethod
    def for_agents(
        cls,
        investigator: Any,
        analyst: Any,
        clock: Callable[[], float] = time.monotonic,
    ) -> "AgentTaskExecutor":
        """The M17-compatible executor over the two injected agents."""
        return cls(
            ParticipantRegistry(
                {INVESTIGATOR: investigator, ANALYST: analyst}
            ),
            clock=clock,
        )

    @property
    def registry(self) -> ParticipantRegistry:
        return self._registry

    def execute(self, request: AgentTaskRequest) -> AgentTaskResult:
        if not isinstance(request, AgentTaskRequest):
            raise ValueError(
                "request must be an AgentTaskRequest, got {}".format(
                    type(request).__name__
                )
            )
        participant = self._registry.select(request.role)
        trace_before = getattr(participant, "last_trace", None)
        started_at = utcnow()
        started = self._clock()
        try:
            # M20: a request-level per-attempt tool limit is passed
            # through ONLY when set — the None path is the exact M19
            # call (``run_result(question)``), so every participant
            # and test double that predates the override keeps
            # working unchanged. When set, the participant enforces
            # it at its own pre-tool guard (Agent/specialists do); a
            # participant that does not accept the override raises at
            # the call (boundary misuse, surfaced loudly — never a
            # silently ignored budget).
            if request.max_tool_calls is None:
                result = participant.run_result(request.prompt)
            else:
                result = participant.run_result(
                    request.prompt,
                    max_tool_calls=request.max_tool_calls,
                )
        except Exception as exc:  # participant failure → typed result
            return self._failed_result(
                request, exc, trace_before, started, started_at
            )
        elapsed = self._clock() - started
        trace = _fresh_trace(participant, trace_before)
        output = getattr(result, "structured_output", None)
        evidence_refs: Tuple[Any, ...] = tuple(
            getattr(output, "evidence_refs", ()) or ()
        )
        return AgentTaskResult(
            run_id=request.run_id,
            task_id=request.task_id,
            attempt=request.attempt,
            role=request.role,
            status=TASK_COMPLETED,
            output=output,
            usage=usage_from_trace(
                trace, _count_tool_calls(trace, trace_before), elapsed
            ),
            evidence_refs=evidence_refs,
            agent_run_id=getattr(trace, "run_id", None),
            trace=trace,
            started_at=started_at,
            completed_at=utcnow(),
        )

    def _failed_result(
        self,
        request: AgentTaskRequest,
        exc: BaseException,
        trace_before: Any,
        started: float,
        started_at: str,
    ) -> AgentTaskResult:
        elapsed = self._clock() - started
        # The exception path may still have produced a trace (the
        # Agent finalizes its trace in all outcomes).
        participant = self._registry.select(request.role)
        trace = _fresh_trace(participant, trace_before)
        failure = classify_exception(exc)
        return AgentTaskResult(
            run_id=request.run_id,
            task_id=request.task_id,
            attempt=request.attempt,
            role=request.role,
            status=TASK_FAILED,
            failure=failure,
            usage=usage_from_trace(
                trace, _count_tool_calls(trace, trace_before), elapsed
            ),
            agent_run_id=getattr(trace, "run_id", None),
            trace=trace,
            started_at=started_at,
            completed_at=utcnow(),
        )
