"""Generic, graph-agnostic ReAct context-safety controls.

The original failure mode this closes: a ReAct-style loop kept making
model calls, each call appending tool observations to the history,
until ONE outgoing request exceeded the model's context limit. The
production ``Agent`` loop already defends itself (``_gate_model_request``
on every call, working-context compaction, the tool-protocol response
ceiling) — but those defenses live inside that one heavyweight class.
A hand-rolled ReAct loop, a different graph node, or a future agent
runtime that calls ``LLMClient.complete`` directly inherits NONE of
them.

This module is the reusable boundary those callers opt into. It is
deliberately small and graph-agnostic: no agent names, no task names,
no repository identifiers, no second budget implementation. Three
composable pieces:

* :class:`BudgetedLLMClient` — wrap ANY ``LLMClient``; the COMPLETE
  outgoing request (messages + tool definitions + response-format
  schema) is measured and enforced immediately before the provider
  boundary with the SAME ``agent.request_budget`` machinery the rest
  of the project uses. An oversized request is REJECTED locally —
  never sent, never silently truncated, never retried unchanged. A
  rejection does not touch the provider (and therefore does not
  consume any provider-call allowance); a request that actually
  reaches the inner client does.
* :func:`bound_observation` — deterministically bound ONE tool
  observation before it joins agent history, preserving truncation
  metadata and a continuation hint. Per-observation bounds do NOT
  guarantee the total request fits — the client-boundary budget
  remains the authority on that.
* :class:`ReActLoopGuard` — configurable limits for iterations, tool
  calls, model calls, per-request input tokens, and a cumulative
  input-token allowance, with a clear terminal status on exhaustion.
  No unbounded retry or compaction loop: once a limit is hit the run
  stops safely and inspectably.

* :func:`restrict_history` — a per-request safety net: drop the
  OLDEST non-protected interactions from THAT request only (the
  caller's own transcript is never mutated), keeping tool-call and
  tool-result pairs whole and preserving provenance by keeping the
  newest interactions and protected messages.

Approximation note: the token figures here come from the project's
established ``chars_per_token`` RATIO estimator
(``agent.request_budget``) unless a provider-reported ``Usage`` is
available — they are ESTIMATES, not exact counts, and nothing in this
module claims a monetary cap. Global budget defaults are never
changed by constructing a config; a stricter ceiling is an explicit,
caller-supplied policy.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm.models import LLMRequest, LLMResponse, Message
from .tools.protocol import MAX_TOOL_RESPONSE_CHARS
from .agent.request_budget import (
    RequestBudgetExceededError,
    RequestBudgetPolicy,
    enforce_request_budget_or_raise,
)

__all__ = [
    "BudgetedLLMClient",
    "MAX_TOOL_RESPONSE_CHARS",
    "ReActLoopGuard",
    "ReActSafetyConfig",
    "ReActTermination",
    "RequestBudgetExceededError",
    "bound_observation",
    "restrict_history",
]


class ReActTermination:
    """Terminal statuses for a bounded ReAct run (string constants,
    matching the project's string-style termination vocabularies)."""

    COMPLETED = "COMPLETED"
    ITERATION_LIMIT_REACHED = "ITERATION_LIMIT_REACHED"
    TOOL_CALL_LIMIT_REACHED = "TOOL_CALL_LIMIT_REACHED"
    MODEL_CALL_LIMIT_REACHED = "MODEL_CALL_LIMIT_REACHED"
    REQUEST_BUDGET_EXCEEDED = "REQUEST_BUDGET_EXCEEDED"
    CUMULATIVE_BUDGET_EXCEEDED = "CUMULATIVE_BUDGET_EXCEEDED"


@dataclasses.dataclass(frozen=True)
class ReActSafetyConfig:
    """One configuration for ANY ReAct-style loop.

    Every field is an independent control — a per-request input
    ceiling does NOT by itself bound total run cost, so iterations,
    tool calls, model calls, observation size, and a cumulative
    input-token allowance are separate knobs. ``None`` for a token
    field means "use the application default / unbounded", never
    "unlimited by accident": the request-input field falls back to
    the project's global default policy, and the cumulative field
    must be set explicitly to activate.
    """

    max_iterations: int = 12
    max_tool_calls: int = 12
    max_model_calls: int = 16
    max_observation_chars: int = MAX_TOOL_RESPONSE_CHARS
    max_request_input_tokens: Optional[int] = None
    max_cumulative_input_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_tool_calls",
                     "max_model_calls", "max_observation_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    "{} must be an integer, got {!r}".format(name, value)
                )
            if value < 1:
                raise ValueError(
                    "{} must be >= 1, got {}".format(name, value)
                )
        for name in ("max_request_input_tokens",
                     "max_cumulative_input_tokens"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    "{} must be None or an integer >= 1, got {!r}".format(
                        name, value
                    )
                )

    def request_budget_policy(self) -> RequestBudgetPolicy:
        """The policy this config enforces at the client boundary.

        ``None`` (the default) returns the project's STANDARD default
        policy — the global H1 default is never weakened by using this
        module. A set value returns a capacity-free policy whose
        explicit ``max_input_tokens`` is the caller's stricter local
        ceiling (same estimator, same enforcement machinery).
        """
        if self.max_request_input_tokens is None:
            return RequestBudgetPolicy()
        return RequestBudgetPolicy(
            max_input_tokens=self.max_request_input_tokens,
            context_capacity_tokens=None,
        )


# -- bounded tool observations ------------------------------------------------


def bound_observation(
    text: str,
    max_chars: int,
    *,
    label: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Bound ONE tool observation before it joins agent history.

    Whole lines are kept whenever any line fits (so a caller can
    continue from a reported line boundary); a character prefix of the
    first line is returned only when not even one line fits. The
    returned record carries the original size, the delivered size, the
    limit, a truncation flag, and a continuation hint — the metadata a
    loop needs to re-request targeted content later instead of losing
    the reference. Deterministic: same input, same limit, same output.

    This bounds ONE observation only. It does NOT guarantee the
    accumulated request fits the input budget — that remains the job
    of :class:`BudgetedLLMClient` at the provider boundary.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1, got {}".format(max_chars))
    original = text or ""
    lines = original.split("\n")
    # Reserve a bounded continuation note so the delivered text plus
    # its own metadata always fit the stated ceiling.
    note = "\n[truncated: showing first {} chars of {}; " \
        "re-read with a targeted query to continue]"
    # Conservative reserve: enough for the note with large numbers.
    reserve = len(note.format(max_chars, len(original))) + 16
    room = max(1, max_chars - reserve)
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)  # + newline separator
        if used + cost > room:
            break
        kept.append(line)
        used += cost
    truncated = len(kept) < len(lines)
    if kept:
        body = "\n".join(kept)
    else:
        # Not even one line fits: prefix the first line to the room.
        body = lines[0][:room] if lines else ""
        truncated = True
    bounded = body + note.format(len(body), len(original)) \
        if truncated else body
    record: Dict[str, Any] = {
        "label": label,
        "original_chars": len(original),
        "returned_chars": len(body),
        "char_limit": max_chars,
        "truncated": truncated,
        "note": (
            "observation bounded before joining history; re-read "
            "with a targeted query to continue"
            if truncated else None
        ),
    }
    return bounded, record


# -- per-request history restriction (safety net) -----------------------------


def restrict_history(
    messages: Sequence[Message],
    *,
    max_chars: int,
    protected_roles: Sequence[str] = ("system", "developer"),
    keep_recent: int = 2,
) -> Tuple[List[Message], Dict[str, Any]]:
    """Per-request safety net: drop the OLDEST non-protected
    interactions from THIS request until it fits ``max_chars``.

    The caller's own transcript is NEVER mutated — a new list is
    returned, so evidence already gathered stays in the loop's
    history and can be re-requested later. Tool-call and tool-result
    pairs are kept WHOLE (a result never survives without its call,
    and a call never survives without its result), and at least the
    ``keep_recent`` NEWEST non-protected messages are always kept, so
    the newest interaction — and therefore the loop's immediate
    reasoning state — is never dropped. Protected roles (system /
    developer instructions) are never dropped.

    This is a REQUEST-only bound, not a transcript rewrite: it does
    not summarize, does not call a model, and does not claim the
    dropped content is gone — only that this particular request does
    not carry it. The record reports what was dropped so the decision
    is inspectable.
    """

    def _units(msgs: Sequence[Message]) -> List[List[Message]]:
        """Group assistant tool-call turns with their tool results."""
        grouped: List[List[Message]] = []
        index = 0
        while index < len(msgs):
            message = msgs[index]
            if (
                message.role == "assistant"
                and message.tool_calls
                and index + 1 < len(msgs)
                and msgs[index + 1].role == "tool"
            ):
                pair = [message]
                index += 1
                while (
                    index < len(msgs)
                    and msgs[index].role == "tool"
                ):
                    pair.append(msgs[index])
                    index += 1
                grouped.append(pair)
            else:
                grouped.append([message])
                index += 1
        return grouped

    def _chars(msgs: Sequence[Message]) -> int:
        return sum(
            len(message.content or "")
            + sum(
                len(call.name)
                + len(str(call.arguments or {}))
                for call in message.tool_calls
            )
            for message in msgs
        )

    protected = set(protected_roles)
    grouped = _units(messages)
    total = _chars(messages)
    if total <= max_chars:
        return list(messages), {
            "dropped_units": 0,
            "dropped_chars": 0,
            "original_chars": total,
            "returned_chars": total,
            "char_limit": max_chars,
            "within_limit": True,
        }

    # Indexes of droppable units: non-protected, and NOT among the
    # keep_recent newest non-protected units.
    non_protected = [
        position
        for position, unit in enumerate(grouped)
        if unit[0].role not in protected
    ]
    keep_newest = (
        set(non_protected[-keep_recent:])
        if keep_recent > 0
        else set()
    )
    droppable = [
        position
        for position in non_protected
        if position not in keep_newest
    ]
    dropped_chars = 0
    dropped_units = 0
    drop_set = set()
    # Drop oldest-first until the request fits.
    for position in droppable:
        if total - dropped_chars <= max_chars:
            break
        unit_chars = _chars(grouped[position])
        dropped_chars += unit_chars
        dropped_units += 1
        drop_set.add(position)

    remaining: List[Message] = []
    for position, unit in enumerate(grouped):
        if position in drop_set:
            continue
        remaining.extend(unit)
    returned = _chars(remaining)
    return remaining, {
        "dropped_units": dropped_units,
        "dropped_chars": dropped_chars,
        "original_chars": total,
        "returned_chars": returned,
        "char_limit": max_chars,
        "within_limit": returned <= max_chars,
    }


# -- the client-boundary budget enforcement -----------------------------------


class BudgetedLLMClient:
    """Wrap ANY ``LLMClient``: enforce the COMPLETE request budget
    immediately before the provider boundary.

    Every ``complete()`` call — the first one AND every later ReAct
    iteration — is measured with the project's existing
    ``agent.request_budget`` machinery against the configured policy.
    An oversized request raises :class:`RequestBudgetExceededError`
    before the inner client is touched: the provider is never invoked,
    nothing is silently truncated, and the request is never retried
    unchanged. A request that PASSES the gate is sent byte-identical
    (accepted) or with tool observations clipped to bounded excerpts
    (reduced) — the same accepted / reduced / rejected semantics the
    rest of the project uses, so there is no second budget
    implementation to drift.

    Call accounting is honest: ``calls_reached_provider`` counts only
    requests that actually reached the inner client (so a budget
    rejection does NOT consume a provider-call allowance, while a
    provider failure — which did reach the provider — does).
    """

    def __init__(
        self,
        inner: Any,
        *,
        policy: Optional[RequestBudgetPolicy] = None,
        subject: str = "react request",
    ) -> None:
        self._inner = inner
        self._policy = policy
        self._subject = subject
        self.records: List[Dict[str, Any]] = []
        self.sent: List[LLMRequest] = []
        self.rejected: List[Dict[str, Any]] = []
        self.calls_reached_provider: int = 0

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    @property
    def rejections(self) -> List[Dict[str, Any]]:
        return list(self.rejected)

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            final_request, record = enforce_request_budget_or_raise(
                request, self._policy, subject=self._subject
            )
        except RequestBudgetExceededError as exc:
            # Rejected BEFORE the provider: recorded for inspection,
            # the provider is never touched, and the call is never
            # retried unchanged. The exception is surfaced unchanged.
            self.records.append(exc.record)
            self.rejected.append(exc.record)
            raise
        self.records.append(record)
        self.sent.append(final_request)
        self.calls_reached_provider += 1
        try:
            return self._inner.complete(final_request)
        except Exception:
            # The provider was reached — the call consumes whatever
            # provider-call allowance the caller maintains. The
            # failure itself is surfaced unchanged (never swallowed).
            raise


# -- the bounded-execution guard ----------------------------------------------


class ReActLoopGuard:
    """Configurable execution limits for ANY ReAct-style loop.

    Counts iterations, tool calls, model calls, and cumulative input
    tokens against a :class:`ReActSafetyConfig`. The loop consults
    ``allow_*`` BEFORE each action and calls the matching ``record_*``
    AFTER; on exhaustion the guard sets a clear terminal status and
    every later ``allow_*`` returns ``False`` — there is no unbounded
    retry or compaction loop, only a safe, inspectable stop.

    Cumulative accounting is honest about estimation: when a response
    carries a provider-reported input-token count it is used for that
    call; otherwise the preflight record's ESTIMATED tokens are used,
    and the basis is tracked per call so a report can distinguish
    reported usage from estimates. Neither figure is claimed to be an
    exact provider-side count, and neither is a monetary cap.
    """

    def __init__(self, config: Optional[ReActSafetyConfig] = None) -> None:
        self.config = config if config is not None else ReActSafetyConfig()
        self.iterations: int = 0
        self.tool_calls: int = 0
        self.model_calls: int = 0
        self.cumulative_input_tokens: int = 0
        self.cumulative_reported_input_tokens: int = 0
        self.cumulative_estimated_input_tokens: int = 0
        self.termination_reason: Optional[str] = None
        self._budget_exceeded_record: Optional[Dict[str, Any]] = None

    # -- inspection ---------------------------------------------------------

    @property
    def budget_exceeded_record(self) -> Optional[Dict[str, Any]]:
        return self._budget_exceeded_record

    def terminal_status(self) -> Optional[str]:
        return self.termination_reason

    def stop(self, reason: str) -> None:
        """Mark the run terminated for an explicit reason."""
        if self.termination_reason is None:
            self.termination_reason = reason

    # -- iteration control --------------------------------------------------

    def allow_iteration(self) -> bool:
        if self.termination_reason is not None:
            return False
        if self.iterations >= self.config.max_iterations:
            self.stop(ReActTermination.ITERATION_LIMIT_REACHED)
            return False
        return True

    def begin_iteration(self) -> None:
        self.iterations += 1

    # -- tool-call control --------------------------------------------------

    def allow_tool_call(self) -> bool:
        if self.termination_reason is not None:
            return False
        if self.tool_calls >= self.config.max_tool_calls:
            self.stop(ReActTermination.TOOL_CALL_LIMIT_REACHED)
            return False
        return True

    def record_tool_call(self) -> None:
        """Record a tool attempt without consuming the finalization turn.

        Reaching ``max_tool_calls`` prevents another investigation tool call,
        but does not mark the whole run terminal before the agent has one
        bounded opportunity to synthesize collected evidence. The LangChain
        execution middleware enforces that finalization-only turn by removing
        investigation tools from the model request.
        """
        self.tool_calls += 1

    # -- model-call + cumulative control ------------------------------------

    def allow_model_call(self) -> bool:
        if self.termination_reason is not None:
            return False
        if self.model_calls >= self.config.max_model_calls:
            self.stop(ReActTermination.MODEL_CALL_LIMIT_REACHED)
            return False
        ceiling = self.config.max_cumulative_input_tokens
        if (
            ceiling is not None
            and self.cumulative_input_tokens >= ceiling
        ):
            self.stop(ReActTermination.CUMULATIVE_BUDGET_EXCEEDED)
            return False
        return True

    def record_model_response(
        self,
        response: LLMResponse,
        *,
        budget_record: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Count one model call and accumulate its input cost."""
        self.model_calls += 1
        reported: Optional[int] = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            reported = getattr(usage, "input_tokens", None)
        estimated: Optional[int] = None
        if budget_record is not None:
            totals = budget_record.get("totals") or {}
            estimated = totals.get("total_tokens_est")
        if reported is not None:
            self.cumulative_reported_input_tokens += reported
            self.cumulative_input_tokens += reported
        elif estimated is not None:
            self.cumulative_estimated_input_tokens += estimated
            self.cumulative_input_tokens += estimated

    def record_budget_rejection(
        self, record: Dict[str, Any]
    ) -> None:
        """A request was rejected before the provider: terminate
        safely. The rejection is NOT retried and does not consume a
        provider call."""
        self._budget_exceeded_record = record
        self.stop(ReActTermination.REQUEST_BUDGET_EXCEEDED)

    def snapshot(self) -> Dict[str, Any]:
        """Privacy-safe counters for reporting (no message content)."""
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_reported_input_tokens": (
                self.cumulative_reported_input_tokens
            ),
            "cumulative_estimated_input_tokens": (
                self.cumulative_estimated_input_tokens
            ),
            "termination_reason": self.termination_reason,
            "config": dataclasses.asdict(self.config),
        }
