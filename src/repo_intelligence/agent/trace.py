"""Agent execution trace — observable lifecycle of one Agent run (MVP).

Every ``Agent.run`` automatically builds an ``AgentTrace``: the
question, one numbered entry per LLM decision cycle (the observable
conversation messages sent, request/response metadata, an explicit
``llm_response`` record, the exact ToolCalls requested, the
ToolResults executed with their duration, the observations fed back,
and the explicit continue/terminate decision), and a run summary with
totals and a termination reason. The trace is a plain in-memory data
structure; the Agent keeps the latest one on ``Agent.last_trace``.

M8.7.1 stage model — one iteration reads top-to-bottom as the ordered
interaction stages (observability only; the loop itself is unchanged):

    LLM REQUEST     → ``request`` (messages actually sent)
    LLM RESPONSE    → ``llm_response`` + ``response`` + decision
    ACTION          → ``tool_calls`` (verbatim requested calls)
    TOOL EXECUTION  → dispatch outcome incl. ``duration_ms``
    TOOL RESULT     → ``tool_results`` (success/failure, provenance)
    STATE UPDATE    → ``observations`` (what went back into the
                      conversation) and what the next request carries
    CONTINUATION    → ``continuation`` (CONTINUE / FINAL_ANSWER /
                      TERMINATE — the observable loop decision)
    FINAL ANSWER    → ``final_answer`` (final iteration only)

The M8.8 evidence validation is a run-level section recorded after
the investigation (``evidence_validation``), never an iteration stage.

The trace exposes observable model inputs/outputs and tool activity
for debugging and learning. It does not store hidden chain-of-thought
or private reasoning.

What is captured — observable events only:

    question · iteration numbers · the messages actually sent (roles,
    order, clipped content) · request/response metadata · the explicit
    ``llm_response`` record (type, tool-call count, exact requested
    tool calls) · decision type · tool name + exact arguments ·
    tool-call id · ToolResult (success/failure, error, provenance,
    clipped data) · per-execution ``duration_ms`` · observation
    identity · per-iteration ``continuation`` (CONTINUE /
    FINAL_ANSWER / TERMINATE + reason) · final answer · termination
    reason · totals · M8.5 execution budget (max_tool_calls), tool
    calls blocked because the budget was exhausted (explicitly NOT
    executed — never a fake ToolResult), and executed/blocked counts ·
    M8.8 integration: for structured-claims runs, the model's explicit
    structured claims and the post-run deterministic validation
    results (recorded after the investigation, before persistence) ·
    M8.9 integration: for structured-output runs, the requested schema
    on the request record, the decoded JSON payload on the
    ``llm_response`` record, and the iteration's application-side
    ``structured_validation`` status (SUCCESS/FAILED)

What is deliberately NOT captured: the model's private chain of
thought. There is no ``thought``/``chain_of_thought``/hidden-reasoning
field (see ``BANNED_REASONING_FIELDS``) and none is ever fabricated —
``reason_summary`` holds a short developer-facing description of the
observable action ("Model requested search_repository(...)"), never a
claim about what the model was thinking.

Persistence is a separable concern (``trace_sink.py``): this module
only defines the model, its JSON serialization, and the human-readable
Markdown projection of that same JSON (never a second source of
truth). JSON is the canonical machine-readable format.
"""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .. import __version__ as AGENT_PACKAGE_VERSION
from ..llm import LLMRequest, LLMResponse, ToolCall

__all__ = [
    "AGENT_NAME",
    "BANNED_REASONING_FIELDS",
    "COLLABORATION_OPERATIONS",
    "CONTINUATION_DECISIONS",
    "DECISION_TYPES",
    "LIFECYCLE_OPERATIONS",
    "ORCHESTRATION_OPERATIONS",
    "RESPONSE_TYPES",
    "STOP_REASONS",
    "TERMINATION_REASONS",
    "TRACE_DATA_CHAR_LIMIT",
    "AgentIteration",
    "AgentTrace",
    "TraceSummary",
    "blocked_tool_call_dict",
    "build_decision",
    "continuation_dict",
    "llm_response_dict",
    "message_to_dict",
    "observation_dict",
    "render_markdown",
    "response_meta",
    "tool_call_to_dict",
    "tool_result_to_dict",
]

# Trace identity: "which agent produced this". The version is the
# package version — never invented repository metadata.
AGENT_NAME = "repo_intelligence.agent"

# Observable response/decision vocabulary (small, explicit).
# M8.9 adds STRUCTURED_OUTPUT — the response type of a run that asked
# for one JSON object (distinct from tool requests and free-form text).
RESPONSE_TYPES = (
    "TOOL_CALL",
    "FINAL_ANSWER",
    "ERROR",
    "INVALID_RESPONSE",
    "STRUCTURED_OUTPUT",
)
DECISION_TYPES = (
    "TOOL_CALL",
    "FINAL_ANSWER",
    "ERROR",
    "INVALID_RESPONSE",
    "LIMIT_REACHED",
    "STRUCTURED_OUTPUT",
)
TERMINATION_REASONS = (
    "FINAL_ANSWER",
    "LLM_ERROR",
    "INVALID_LLM_RESPONSE",
    "TOOL_DISPATCH_ERROR",
    "EXECUTION_LIMIT_REACHED",
    "ERROR",
    "STRUCTURED_OUTPUT",
    "STRUCTURED_OUTPUT_VALIDATION_FAILED",
    # M10 plan mode (additive; non-plan runs never terminate here):
    # the produced plan failed deterministic validation (schema or
    # application gate), and a plan step produced no usable result so
    # the deterministic policy invalidated the plan (no replanning).
    "PLAN_VALIDATION_FAILED",
    "PLAN_INVALIDATED",
    # Execution-lifecycle boundary (additive): the run reached an
    # execution boundary and the tool-free finalization pass could not
    # produce a usable structured result — an explicit, non-retryable
    # failure, never a silent success and never an ordinary LLM retry.
    "FINALIZATION_FAILED",
    # Request-budget preflight (additive): the application request
    # budget rejected the request BEFORE any provider call — protected
    # content alone exceeded the budget, so the request was never sent
    # and is never retried unchanged (request_budget.py). The derived
    # stop_reason maps this to CONTEXT_LENGTH_EXCEEDED: same failure
    # class (context exhaustion), predicted locally instead of
    # reported by the provider.
    "CONTEXT_BUDGET_EXCEEDED",
)

# M8.7.1: the observable continue/terminate decision recorded at the
# end of every iteration — the loop's Stage G, never a rationale.
CONTINUATION_DECISIONS = ("CONTINUE", "FINAL_ANSWER", "TERMINATE")

# M16: the closed vocabulary of multi-agent collaboration operations
# recorded on an agent's execution trace (one section of the SAME
# trace — no second trace system, M16 §21). Only the runner ever
# appends these; the model never authors trace events. Sequence
# numbers and timestamps are informational metadata (the
# collaboration record in the task state is authoritative).
COLLABORATION_OPERATIONS = (
    "agent.start",
    "agent.end",
    "message.send",
    "message.receive",
    "analysis.start",
    "analysis.complete",
    "validator.start",
    "validator.complete",
)

# M17: the closed vocabulary of orchestration operations recorded on
# an agent's execution trace (one section of the SAME trace — no
# second trace system, M17 §16). Only the orchestrator ever appends
# these (mirrored from the authoritative OrchestrationRun event log
# for the step windows this agent executed); the model never authors
# trace events. Sequence numbers and timestamps are informational
# metadata (the run's event list is authoritative). Planner/validation/
# termination events that belong to no agent stay on the run state.
ORCHESTRATION_OPERATIONS = (
    "orchestration.start",
    "orchestration.plan.created",
    "orchestration.plan.validated",
    "orchestration.step.start",
    "orchestration.step.assign",
    "orchestration.agent.start",
    "orchestration.agent.complete",
    "orchestration.step.complete",
    "orchestration.step.failed",
    "orchestration.step.blocked",
    "orchestration.handoff.send",
    "orchestration.retry",
    "orchestration.validation.start",
    "orchestration.validation.complete",
    "orchestration.budget.exceeded",
    "orchestration.timeout",
    "orchestration.complete",
    "orchestration.failed",
    "orchestration.terminated",
)

# Execution lifecycle (additive): the closed vocabulary of boundary /
# finalization events recorded on the SAME trace — the smallest
# observable representation of "the run stopped taking tools and
# closed from the state it already had". Only the Agent's loop records
# these; the model never authors trace events. Empty for every run
# that never reached an execution boundary, so all existing traces
# keep their exact shape.
LIFECYCLE_OPERATIONS = (
    "execution.boundary_reached",
    "context.length_exceeded",
    "finalization.started",
    "tools.disabled",
    "finalization.completed",
    "finalization.failed",
    # Bounded normal execution (M17 cleanup): the application
    # working-context ceiling reduced THIS request before it was sent
    # — counts only, never message contents, and never a claim that
    # evidence was touched (the trace record is authoritative).
    "context.restricted",
    # Persistent working-context compaction (M17 final): the working
    # conversation itself was REPLACED above the ceiling (protected
    # preamble + one deterministic state message + a small recent
    # window). Count-only pair, never message contents; distinct from
    # FINALIZATION (a terminal tool-free pass) and from
    # CONTEXT_LENGTH_EXCEEDED (a provider rejection) — a compacted run
    # continues investigating and may finish voluntarily.
    "context.compaction.started",
    "context.compaction.completed",
    # M17 efficiency policy: the run crossed the SOFT investigation
    # target (counters + the notice's marker only — never message
    # content). A reassessment signal, not a boundary: nothing was
    # blocked, nothing terminated, and the hard budget above is the
    # only thing that ever does either.
    "execution.effort_threshold_reached",
    # M17 hard investigation lifecycle (opt-in, enforce_soft_target):
    # the crossing OPENS a bounded decision phase instead of
    # open-ended investigation. reassessed (the run is deciding over
    # the evidence it has), decision (FINALIZE or ONE_TARGETED_CALL),
    # targeted_call_started (the one allowed post-threshold dispatch).
    # Counters + names only — never reasoning, never message content.
    # Enforcement never records execution.boundary_reached: the hard
    # budget was not exceeded, and the derived stop reason stays
    # EVIDENCE_SUFFICIENT for a run that finalizes this way.
    "execution.investigation_reassessed",
    "execution.investigation_decision",
    "execution.targeted_call_started",
    # Request-budget preflight at the model-call boundary (H1/H2):
    # count-only pair recorded when the application request budget
    # changed or stopped the OUTGOING request — ``request_reduced``
    # (tool observations clipped to bounded excerpts, before/after
    # counts, request still sent) and ``budget_exceeded`` (protected
    # content alone over budget — never sent, never retried
    # unchanged). Sizes and dispositions only; never message content,
    # never a provider verdict (the provider was not consulted for a
    # rejected request).
    "context.request_reduced",
    "context.budget_exceeded",
)

# Observable execution distinctions (M17 cleanup) — derived READ-ONLY
# views over terminal state that already exists, never a second status
# system:
#   stop_reason             EVIDENCE_SUFFICIENT | BUDGET_EXCEEDED |
#                           CONTEXT_LENGTH_EXCEEDED | ERROR
#   context_restrictions    the CONTEXT_RESTRICTED marker (the request
#                           was bounded on the way, then execution
#                           continued)
#   context_compactions     the COMPACTION marker (the working
#                           conversation was replaced, then execution
#                           continued) — never a stop reason: a normal
#                           success path may compact several times and
#                           still finish with evidence sufficient
#   effort_threshold_reached the SOFT investigation target was
#                           crossed (one execution-state notice was
#                           sent) — an efficiency marker, never a
#                           stop reason, never a block
STOP_REASONS = (
    "EVIDENCE_SUFFICIENT",
    "BUDGET_EXCEEDED",
    "CONTEXT_LENGTH_EXCEEDED",
    "ERROR",
)

# Field names that must NEVER appear anywhere in a serialized trace —
# the trace records actions and observations, not hidden reasoning.
BANNED_REASONING_FIELDS = (
    "thought",
    "chain_of_thought",
    "private_reasoning",
    "hidden_reasoning",
    "internal_monologue",
    "reasoning",
    "scratchpad",
)

# Tool results may embed source code (read_file can return up to 1 MB).
# Traces are bounded per recorded string so a run cannot grow the
# artifact without limit; clipped values carry an explicit marker.
TRACE_DATA_CHAR_LIMIT = 2000

_CLIP_MARKER = "... [trace clipped: {} chars total]"


def _utcnow() -> str:
    """UTC timestamp, ISO-8601 with millisecond precision (sortable)."""
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _clip_text(value: str, limit: int = TRACE_DATA_CHAR_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + _CLIP_MARKER.format(len(value))


def _clip_deep(value: Any, limit: int = TRACE_DATA_CHAR_LIMIT) -> Any:
    """Recursively clip long strings inside a JSON-ready structure."""
    if isinstance(value, str):
        return _clip_text(value, limit)
    if isinstance(value, dict):
        return {key: _clip_deep(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip_deep(item, limit) for item in value]
    return value


def response_meta(response: LLMResponse) -> Dict[str, Any]:
    """Observable metadata of one LLM response (no conversation dump)."""
    usage = response.usage
    meta: Dict[str, Any] = {
        "finish_reason": response.finish_reason,
        "tool_call_count": len(response.tool_calls),
        "has_text": response.message.content is not None,
        "usage": None,
    }
    if usage is not None:
        meta["usage"] = {
            key: value
            for key, value in (
                ("input_tokens", usage.input_tokens),
                ("output_tokens", usage.output_tokens),
                ("total_tokens", usage.total_tokens),
            )
            if value is not None
        }
    return meta


def llm_response_dict(
    response: LLMResponse, structured_requested: bool = False
) -> Dict[str, Any]:
    """The model's response as received — one explicit record per LLM
    turn: what kind of response it was (``TOOL_CALL``,
    ``STRUCTURED_OUTPUT``, ``FINAL_ANSWER`` or ``INVALID_RESPONSE`` —
    the same classification the loop applies), whether it carried text,
    the exact tool calls it requested (when any), and the decoded JSON
    payload of a structured response (M8.9, when any).

    *structured_requested* is set by the loop for a run whose request
    carried a ``StructuredOutputSpec``: without a payload such a
    response is still classified ``STRUCTURED_OUTPUT`` (the contract
    was structured — validation then reports explicitly whether the
    payload arrived and matched). Tool calls always win the
    classification, structured run or not: they are dispatched, never
    mistaken for a payload.

    Observable output only: the response type, counts and the verbatim
    tool calls/payload. Hidden chain of thought is not part of a
    response and is never recorded (the prose the model DID return
    lives in ``final_answer``/message content, never as "thoughts").
    """
    text = response.message.content
    kind: str
    if response.tool_calls:
        kind = "TOOL_CALL"
    elif response.structured_output is not None or structured_requested:
        kind = "STRUCTURED_OUTPUT"
    elif text is not None:
        kind = "FINAL_ANSWER"
    else:
        kind = "INVALID_RESPONSE"
    payload: Dict[str, Any] = {
        "type": kind,
        "tool_call_count": len(response.tool_calls),
        "has_text": text is not None,
    }
    if response.tool_calls:
        payload["tool_calls"] = [
            tool_call_to_dict(call) for call in response.tool_calls
        ]
    if response.structured_output is not None:
        payload["structured_output"] = _clip_deep(
            dict(response.structured_output)
        )
    return payload


def message_to_dict(message: Any) -> Dict[str, Any]:
    """One conversation message as the provider actually received it.

    Records the observable input state only — role, (clipped)
    content, any requested tool calls and the tool-call id — in
    chronological order. Nothing model-internal is added; long
    contents are clipped by the caller's trace policy.
    """
    payload: Dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [
            tool_call_to_dict(call) for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def tool_call_to_dict(tool_call: ToolCall) -> Dict[str, Any]:
    """Exactly what the application received — never rewritten."""
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": dict(tool_call.arguments),
    }


def tool_result_to_dict(
    tool_call_id: Optional[str],
    result: Dict[str, Any],
    duration_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Record one executed ToolResult (``ToolResult.to_dict()`` form).

    ``data``/``source`` are clipped to the trace size limit; the
    success flag, structured error, and provenance are kept intact.
    ``duration_ms`` (M8.7.1) is the observed dispatch duration of this
    execution in whole milliseconds — recorded only when the Agent
    loop measured it (None for a hand-built or legacy result, so the
    canonical payload of existing traces is unchanged).
    """
    payload: Dict[str, Any] = {
        "tool": result.get("tool"),
        "tool_call_id": tool_call_id,
        "success": bool(result.get("success")),
        "data": _clip_deep(result.get("data")),
        "source": _clip_deep(result.get("source")),
        "error": result.get("error"),
    }
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    return payload


def observation_dict(
    tool: str, success: bool, tool_call_id: Optional[str]
) -> Dict[str, Any]:
    """Identity of the observation appended to the conversation.

    The observation payload itself is the ToolResult recorded above —
    this entry preserves the link (type + tool + call id) without
    duplicating the conversation message.
    """
    return {
        "type": "TOOL_RESULT",
        "tool": tool,
        "success": success,
        "tool_call_id": tool_call_id,
    }


def blocked_tool_call_dict(
    tool_call: ToolCall, executed: int, max_tool_calls: int,
    reason: str = "EXECUTION_LIMIT_REACHED",
) -> Dict[str, Any]:
    """A tool call the model requested but execution blocked (M8.5).

    Explicitly distinct from a ToolResult: the tool was NOT executed,
    so there is no success flag, no data, and no fabricated result —
    only why it was skipped and the budget state. ``reason`` names the
    authority that blocked it: ``EXECUTION_LIMIT_REACHED`` (the hard
    budget) or ``SOFT_TOOL_CALL_TARGET_REACHED`` (the post-threshold
    lifecycle blocked it while one targeted call ran).
    """
    return {
        "tool": tool_call.name,
        "tool_call_id": tool_call.id,
        "reason": reason,
        "tool_calls_executed": executed,
        "max_tool_calls": max_tool_calls,
    }


def build_decision(decision_type: str, tool_names: Sequence[str] = ()) -> Dict[str, Any]:
    """Observable decision record for one iteration.

    ``tool`` (singular) when the model requested exactly one tool,
    ``tools`` (plural) for multi-tool responses — never a hidden
    rationale behind the decision.
    """
    decision: Dict[str, Any] = {"type": decision_type}
    names = [name for name in tool_names if name]
    if len(names) == 1:
        decision["tool"] = names[0]
    elif names:
        decision["tools"] = names
    return decision


def continuation_dict(
    decision: str,
    reason: str,
    next_iteration: Optional[int] = None,
    termination_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """What happens after this iteration — Stage G, observable only.

    ``CONTINUE`` carries ``next_iteration`` (the number of the next
    LLM request); ``FINAL_ANSWER``/``TERMINATE`` may carry the
    existing run-level ``termination_reason`` vocabulary. ``reason``
    is a short developer-facing description of the observable action
    ("tool result added to conversation state"), never a rationale
    behind the model's decision. Recorded only by the Agent loop for
    a real run — a hand-built or legacy trace simply has no
    ``continuation`` block, so its canonical payload is unchanged.
    """
    if decision not in CONTINUATION_DECISIONS:
        raise ValueError(
            "continuation decision must be one of {}".format(
                ", ".join(CONTINUATION_DECISIONS)
            )
        )
    payload: Dict[str, Any] = {"decision": decision, "reason": reason}
    if next_iteration is not None:
        payload["next_iteration"] = next_iteration
    if termination_reason is not None:
        payload["termination_reason"] = termination_reason
    return payload


# ---------------------------------------------------------------------------
# Developer-facing action summaries (observable facts only, §14)
# ---------------------------------------------------------------------------


def summarize_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    if len(tool_calls) == 1:
        call = tool_calls[0]
        return "Model requested {}({}).".format(
            call.name, json.dumps(call.arguments, sort_keys=True)
        )
    names = ", ".join(call.name for call in tool_calls)
    return "Model requested {} tool calls: {}.".format(len(tool_calls), names)


def summarize_final_answer(answer: str) -> str:
    return "Model returned a final textual answer ({} characters).".format(
        len(answer)
    )


def summarize_structured_output(payload: Dict[str, Any]) -> str:
    """Observable description of a validated structured response."""
    return "Model returned structured output ({} top-level key(s)).".format(
        len(payload)
    )


def summarize_structured_output_error(message: str) -> str:
    """Single-line, observable description of a validation failure."""
    return "Structured output validation failed: {}".format(
        " ".join(str(message).split())
    )


def summarize_no_dispatcher(tool_names: Sequence[str]) -> str:
    return (
        "Model requested tool(s) {}, but no ToolDispatcher is "
        "configured — execution is unavailable.".format(", ".join(tool_names))
    )


def summarize_budget_exhausted(
    blocked_names: Sequence[str], executed: int, max_tool_calls: int
) -> str:
    return (
        "Execution budget exhausted ({executed} of {max_calls} tool "
        "calls executed) — blocked {names}: not executed.".format(
            executed=executed,
            max_calls=max_tool_calls,
            names=", ".join(blocked_names),
        )
    )


def summarize_dispatch_error(tool_name: str, exc: Exception) -> str:
    return "Dispatching {} raised {}: {}.".format(
        tool_name, type(exc).__name__, exc
    )


def summarize_llm_error(message: str) -> str:
    return "LLM call failed: {}.".format(message)


def summarize_invalid_response() -> str:
    return "LLM response contained neither text nor tool calls."


def summarize_unexpected_error(message: str) -> str:
    return "Unexpected error during the run: {}.".format(message)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class AgentIteration:
    """One numbered LLM decision cycle (``iteration`` starts at 1).

    An iteration is *open* until ``close`` records the observable
    decision; an LLM failure closes it implicitly via
    ``AgentTrace.record_llm_error``.
    """

    iteration: int
    request: Dict[str, Any]
    response_type: Optional[str] = None
    decision: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    # Explicit record of the LLM response as received (M8.8 follow-up
    # readability): type, tool-call count, has_text and — for a tool
    # call — the exact requested tool calls. Set only by the Agent
    # loop when a response actually arrived (None for a hand-built
    # trace or an LLM failure), so the canonical payload of existing
    # traces is unchanged.
    llm_response: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    # M8.5: calls the model requested here but the execution budget
    # blocked — never executed, therefore never a ToolResult.
    blocked_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None
    reason_summary: Optional[str] = None
    # M8.7.1: the observable continue/terminate decision recorded when
    # the iteration closed (CONTINUE + next_iteration, FINAL_ANSWER or
    # TERMINATE + termination_reason). Set only by the Agent loop for
    # a real run (None for a hand-built or legacy trace, so the
    # canonical payload of existing traces is unchanged).
    continuation: Optional[Dict[str, Any]] = None
    # M8.9: application-side structured-output validation outcome for a
    # STRUCTURED_OUTPUT iteration — {"status": "SUCCESS", "model": ...}
    # or {"status": "FAILED", "error": ...} (recorded by the Agent
    # loop; None everywhere else, so existing traces are unchanged).
    structured_validation: Optional[Dict[str, Any]] = None
    # M10: which plan step this iteration was executing — set only by
    # the Agent's plan mode for iterations inside a step (None for
    # every non-plan run and for the plan-generation iteration, so the
    # canonical payload of existing traces is unchanged). Tool calls
    # and results stay nested in the iteration, so plan-step
    # correlation needs no other id (spec §22).
    plan_step_id: Optional[str] = None
    # H1: the request-budget preflight record for the request this
    # iteration SENT — sizes/roles/budget/disposition only (never
    # message content; request_budget.py). Set only by the Agent's
    # model-call gate for runs through it (None for a hand-built or
    # legacy trace, so the canonical payload of existing traces is
    # unchanged).
    request_budget: Optional[Dict[str, Any]] = None

    @property
    def is_open(self) -> bool:
        return self.response_type is None

    def close(
        self,
        response_type: str,
        decision: Dict[str, Any],
        reason_summary: str,
        final_answer: Optional[str] = None,
        error: Optional[str] = None,
        continuation: Optional[Dict[str, Any]] = None,
        structured_validation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close this iteration with its observable decision."""
        self.response_type = response_type
        self.decision = decision
        self.reason_summary = reason_summary
        if final_answer is not None:
            self.final_answer = final_answer
        if error is not None:
            self.error = error
        if continuation is not None:
            self.continuation = continuation
        if structured_validation is not None:
            self.structured_validation = _clip_deep(
                dict(structured_validation)
            )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "iteration": self.iteration,
            "response_type": self.response_type,
            "decision": dict(self.decision),
            "request": dict(self.request),
            "response": dict(self.response),
        }
        if self.plan_step_id is not None:
            payload["plan_step_id"] = self.plan_step_id
        if self.llm_response is not None:
            payload["llm_response"] = dict(self.llm_response)
        payload["tool_calls"] = [dict(call) for call in self.tool_calls]
        payload["tool_results"] = [dict(result) for result in self.tool_results]
        payload["observations"] = [
            dict(observation) for observation in self.observations
        ]
        if self.blocked_tool_calls:
            payload["blocked_tool_calls"] = [
                dict(entry) for entry in self.blocked_tool_calls
            ]
        if self.final_answer is not None:
            payload["final_answer"] = self.final_answer
        if self.error is not None:
            payload["error"] = self.error
        if self.reason_summary is not None:
            payload["reason_summary"] = self.reason_summary
        if self.continuation is not None:
            payload["continuation"] = dict(self.continuation)
        if self.structured_validation is not None:
            payload["structured_validation"] = dict(self.structured_validation)
        if self.request_budget is not None:
            payload["request_budget"] = _clip_deep(dict(self.request_budget))
        return payload


@dataclass
class TraceSummary:
    """Run totals — ``total_iterations`` counts EVERY LLM decision
    cycle (including the final-answer one); ``total_tool_calls``
    counts the ToolCalls the model requested (= executed + blocked on
    a budget termination). M8.5 bounded-execution fields:
    ``max_tool_calls`` (configured budget; ``None`` for a hand-built
    trace without one), ``tool_calls_executed`` (budget consumed —
    ToolResults recorded), ``tool_calls_blocked`` (requested but NOT
    executed because the budget was exhausted)."""

    total_iterations: int
    total_tool_calls: int
    tools_used: List[str]
    termination_reason: str
    final_answer: Optional[str] = None
    max_tool_calls: Optional[int] = None
    tool_calls_executed: int = 0
    tool_calls_blocked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "total_tool_calls": self.total_tool_calls,
            "tools_used": list(self.tools_used),
            "termination_reason": self.termination_reason,
            "final_answer": self.final_answer,
            "max_tool_calls": self.max_tool_calls,
            "tool_calls_executed": self.tool_calls_executed,
            "tool_calls_blocked": self.tool_calls_blocked,
        }


@dataclass
class AgentTrace:
    """Structured execution trace of one Agent run (provider-neutral)."""

    run_id: str
    started_at: str
    user_question: str
    agent: Dict[str, str] = field(
        default_factory=lambda: {
            "name": AGENT_NAME,
            "version": AGENT_PACKAGE_VERSION,
        }
    )
    completed_at: Optional[str] = None
    repository: Optional[Dict[str, Any]] = None
    # M8.5: configured execution budget (serialized into the summary
    # at finalize; None when a trace is built without one).
    max_tool_calls: Optional[int] = None
    termination_reason: Optional[str] = None
    final_answer: Optional[str] = None
    iterations: List[AgentIteration] = field(default_factory=list)
    summary: Optional[TraceSummary] = None
    sink_error: Optional[str] = None
    # M8.8 integration: structured claims + the post-run deterministic
    # validation outcome — recorded AFTER the investigation completed,
    # only for runs that requested structured claims (None otherwise,
    # so every existing trace keeps its exact shape).
    evidence_validation: Optional[Dict[str, Any]] = None
    # M10 (additive, run level): the explicit structured plan recorded
    # BEFORE the first repository tool execution — objective, status,
    # deterministic validation outcome, and every step's lifecycle
    # ({id, goal, status, result}). None for every non-plan run (spec
    # §21/§40).
    plan: Optional[Dict[str, Any]] = None
    # M10: compact record of the M6 Architecture Context given to the
    # planner (artifact path, schema version, pinned commit, snapshot
    # id — spec §26). None unless a plan run was given the context.
    architecture_context: Optional[Dict[str, Any]] = None
    # M14 (additive): the run's persistent-memory operations, in the
    # order the execution performed them (memory.retrieve → … →
    # memory.store). Each event is observable metadata only —
    # operation, memory_id(s), owner/run/repo/commit, status, stale
    # flag, or a structured error code (§32) — never hidden reasoning.
    # Empty (field omitted from serialization) for every non-memory
    # run, so all existing traces keep their exact shape.
    memory_events: List[Dict[str, Any]] = field(default_factory=list)
    # M15 (additive): the run's context-engineering operations, in
    # the order the execution performed them (context.select →
    # context.render per selection). Each event carries observable
    # metadata only — context_id, step, counts, budget, per-candidate
    # reasons, or a structured error code — never hidden reasoning
    # and never the model's internal state. Empty (field omitted from
    # serialization) for every run that did not inject a
    # ContextEngine, so all existing traces keep their exact shape.
    context_events: List[Dict[str, Any]] = field(default_factory=list)
    # M16 (additive): the run's multi-agent collaboration operations,
    # in the order the runner performed them (agent.start →
    # message.send → … → validator.complete). Each event carries
    # observable metadata only — sequence, actor, message type, status,
    # or a structured reason — never model-internal reasoning and
    # never message prose. Empty (field omitted from serialization)
    # for every single-agent run, so all existing traces keep their
    # exact shape.
    collaboration_events: List[Dict[str, Any]] = field(default_factory=list)
    # M17 (additive): the run's orchestration operations, in the order
    # the orchestrator performed them (step.start → step.assign →
    # agent.start → … → step.complete for the step windows this agent
    # executed). Each event carries observable metadata only — sequence,
    # step id, role, assignment reason, message type, status, budget —
    # never hidden reasoning. Empty (field omitted from serialization)
    # for every run outside an M17 orchestration, so all existing
    # traces keep their exact shape.
    orchestration_events: List[Dict[str, Any]] = field(default_factory=list)
    # Execution lifecycle (additive): the run's boundary/finalization
    # transitions, in the order the loop performed them
    # (execution.boundary_reached → finalization.started →
    # tools.disabled → finalization.completed, or
    # context.length_exceeded → finalization.*). Each event carries
    # observable metadata only — operation, reason, budget counters,
    # evidence count, tools-disabled flag, or a structured error —
    # never model reasoning. Empty (field omitted from serialization)
    # for every run that never reached an execution boundary, so all
    # existing traces keep their exact shape.
    lifecycle_events: List[Dict[str, Any]] = field(default_factory=list)

    # -- construction ------------------------------------------------------

    @classmethod
    def start(
        cls, user_question: str, max_tool_calls: Optional[int] = None
    ) -> "AgentTrace":
        """Start a trace for one run (fresh run id + timestamp).

        ``max_tool_calls`` carries the Agent's configured execution
        budget into the trace (M8.5) — it is never invented when the
        caller does not supply one.
        """
        return cls(
            run_id=uuid.uuid4().hex,
            started_at=_utcnow(),
            user_question=user_question,
            max_tool_calls=max_tool_calls,
        )

    # -- recording ---------------------------------------------------------

    def begin_iteration(
        self,
        request: LLMRequest,
        model: Optional[str] = None,
        plan_step_id: Optional[str] = None,
    ) -> AgentIteration:
        """Append the next open iteration (numbering starts at 1).

        The recorded request carries both its metadata counts and the
        observable conversation messages actually sent (roles,
        chronological order, clipped contents) — the loop's input
        state, never anything model-internal.

        ``plan_step_id`` (M10) tags the iteration with the plan step
        it is executing; ``None`` (every non-plan run, and the
        plan-generation request) leaves the iteration exactly as
        before.
        """
        resolved_model = request.model if request.model is not None else model
        request_record: Dict[str, Any] = {
            "message_count": len(request.messages),
            "tool_definition_count": len(request.tools),
            "model": resolved_model,
            "messages": _clip_deep(
                [message_to_dict(message) for message in request.messages]
            ),
        }
        if request.structured_output is not None:
            # M8.9: which JSON contract this request asked for (the
            # schema as given — recorded, never rewritten).
            request_record["structured_output"] = _clip_deep(
                {
                    "name": request.structured_output.name,
                    "schema": dict(request.structured_output.schema),
                }
            )
        iteration = AgentIteration(
            iteration=len(self.iterations) + 1,
            request=request_record,
            plan_step_id=plan_step_id,
        )
        self.iterations.append(iteration)
        return iteration

    def record_plan(self, plan: Dict[str, Any]) -> None:
        """Record the run-level M10 plan (spec §21).

        Called with the JSON-ready payload from ``plan_to_dict`` —
        BEFORE the first repository tool execution, and again whenever
        a step's lifecycle changes, so chronological trace state reads
        ``plan recorded → tool execution``. Deep-copied and clipped by
        the trace's recording policy; never called for non-plan runs,
        so their payload is unchanged.
        """
        self.plan = _clip_deep(dict(plan))

    def record_architecture_context(self, record: Dict[str, Any]) -> None:
        """Record the compact M6 Architecture Context given to the
        planner (artifact path, schema version, pinned commit,
        snapshot id — spec §26): enough to reconstruct what the
        planner received, without duplicating the context payload."""
        self.architecture_context = _clip_deep(dict(record))

    def note_repository(self, source: Optional[Dict[str, Any]]) -> None:
        """Record repository provenance observed on a ToolResult.

        Only an actual tool result's ``source`` is used — the trace
        never invents a snapshot/commit the run did not touch. The
        first provenance-bearing result wins.
        """
        if self.repository is not None or not source:
            return
        clipped = _clip_deep(dict(source))
        if clipped:
            self.repository = clipped

    def terminate(self, reason: str, final_answer: Optional[str] = None) -> None:
        """Record why the run is ending (called at each terminal path)."""
        self.termination_reason = reason
        if final_answer is not None:
            self.final_answer = final_answer

    def record_llm_error(self, message: str) -> None:
        """Close an open iteration after an ``LLMError`` and terminate."""
        self.terminate("LLM_ERROR")
        if self.iterations and self.iterations[-1].is_open:
            iteration = self.iterations[-1]
            iteration.close(
                response_type="ERROR",
                decision=build_decision("ERROR"),
                reason_summary=summarize_llm_error(message),
                error=message,
                continuation=continuation_dict(
                    "TERMINATE",
                    "LLM call failed",
                    termination_reason="LLM_ERROR",
                ),
            )

    def record_unexpected_error(self, message: str) -> None:
        """Safety net for exceptions without an explicit terminal path."""
        if self.termination_reason is None:
            self.terminate("ERROR")
        if self.iterations and self.iterations[-1].is_open:
            iteration = self.iterations[-1]
            iteration.close(
                response_type="ERROR",
                decision=build_decision("ERROR"),
                reason_summary=summarize_unexpected_error(message),
                error=message,
                continuation=continuation_dict(
                    "TERMINATE",
                    "run terminated with an unexpected error",
                    termination_reason=self.termination_reason or "ERROR",
                ),
            )

    def record_evidence_validation(
        self, claims: Sequence[Any], results: Sequence[Any]
    ) -> None:
        """Attach the M8.8 post-run validation outcome (integration).

        Called only AFTER the investigation loop finished and the
        trace holds every ToolResult: *claims* is the model's explicit
        structured final response (``Claim``-likes exposing
        ``to_dict``), *results* the deterministic validator's output
        (``ClaimValidation``-likes). Each result's ``reason``
        describes evidence matching only ("matched gathered evidence:
        ..."), never model-internal reasoning; the payload is clipped
        like every other recorded string. Validation itself never
        dispatched a tool — this block is its recorded outcome.
        """
        self.evidence_validation = _clip_deep(
            {
                "claims": [claim.to_dict() for claim in claims],
                "results": [result.to_dict() for result in results],
            }
        )

    def record_memory_event(self, event: Dict[str, Any]) -> None:
        """Append one M14 persistent-memory operation (§32).

        The ONE extension for M14 — no second trace system (§33):
        retrieval, verification, storage (and any application-driven
        supersede/invalidate) are recorded on this same execution
        trace as clipped, JSON-ready metadata dicts. The payload
        carries no model-internal reasoning — same observability
        policy as every other recorded block.
        """
        self.memory_events.append(_clip_deep(dict(event)))

    def record_context_event(self, event: Dict[str, Any]) -> None:
        """Append one M15 context-engineering operation.

        The ONE extension for M15 — no second trace system (§27):
        selection and rendering are recorded on this same execution
        trace as clipped, JSON-ready metadata dicts. The payload
        carries counts, budget, per-candidate reasons, and ids — what
        entered the model context and why — never model-internal
        reasoning (same observability policy as every other recorded
        block).
        """
        self.context_events.append(_clip_deep(dict(event)))

    def record_collaboration_event(self, event: Dict[str, Any]) -> None:
        """Append one M16 multi-agent collaboration operation.

        The ONE extension for M16 — no second trace system: stage
        starts/ends and typed handoffs are recorded on this same
        execution trace as clipped, JSON-ready metadata dicts. The
        payload carries sequence, actor, message type, and status —
        what left or arrived and when — never the message prose and
        never model-internal reasoning (same observability policy as
        every other recorded block). Unknown operations raise: only
        the runner records these, so an unknown one is a bug here,
        not model input.
        """
        operation = event.get("operation")
        if operation not in COLLABORATION_OPERATIONS:
            raise ValueError(
                "unknown collaboration operation {!r} (expected one of "
                "{})".format(operation, ", ".join(COLLABORATION_OPERATIONS))
            )
        self.collaboration_events.append(_clip_deep(dict(event)))

    def record_orchestration_event(self, event: Dict[str, Any]) -> None:
        """Append one M17 orchestration operation.

        The ONE extension for M17 — no second trace system: step
        lifecycle, assignment, handoffs, retry, and termination for
        this agent's step windows are recorded on this same execution
        trace as clipped, JSON-ready metadata dicts. The payload
        carries sequence, step id, role, assignment reason, message
        type, and status — what the application decided and why in
        application terms — never model-internal reasoning (same
        observability policy as every other recorded block). Unknown
        operations raise: only the orchestrator records these, so an
        unknown one is a bug here, not model input.
        """
        operation = event.get("operation")
        if operation not in ORCHESTRATION_OPERATIONS:
            raise ValueError(
                "unknown orchestration operation {!r} (expected one of "
                "{})".format(operation, ", ".join(ORCHESTRATION_OPERATIONS))
            )
        self.orchestration_events.append(_clip_deep(dict(event)))

    def record_lifecycle_event(self, event: Dict[str, Any]) -> None:
        """Append one execution-lifecycle transition.

        The smallest possible observability for the boundary/finalization
        phase (same one-trace discipline as M14–M17): the run reached an
        execution boundary, tools were disabled, and a tool-free
        finalization pass closed the run from the state already
        collected. The payload carries operation, reason, budget
        counters and counts — never model reasoning. Unknown operations
        raise: only the Agent's loop records these, so an unknown one is
        a bug here, not model input.
        """
        operation = event.get("operation")
        if operation not in LIFECYCLE_OPERATIONS:
            raise ValueError(
                "unknown lifecycle operation {!r} (expected one of "
                "{})".format(operation, ", ".join(LIFECYCLE_OPERATIONS))
            )
        self.lifecycle_events.append(_clip_deep(dict(event)))

    # -- derived stop/context views (never a second status system) -------

    @property
    def stop_reason(self) -> Optional[str]:
        """Why execution stopped — a READ-ONLY view over terminal state
        this trace already carries (``termination_reason`` +
        ``lifecycle_events``), not a parallel status hierarchy:

            CONTEXT_LENGTH_EXCEEDED  context exhaustion — reported by
                                     the provider or predicted by the
                                     request-budget gate (never
                                     retried as-is)
            BUDGET_EXCEEDED          the tool-call budget/boundary
                                     ended tool use
            EVIDENCE_SUFFICIENT      the run answered voluntarily —
                                     tools were still available and
                                     were not consumed
            ERROR                    any other explicit failure

        ``None`` while the run has no terminal state. Neither context
        COMPACTION nor context RESTRICTION is a stop reason: both are
        request-assembly events (``context.compaction.*``,
        ``context.restricted``) after which the run continues — see
        ``context_compactions`` / ``context_restrictions``. A run may
        compact, keep investigating, and still finish with evidence
        sufficient.

        Context exhaustion outranks everything else on the trace —
        whether the PROVIDER reported it (``context.length_exceeded``)
        or the application's request-budget gate predicted it before
        any call was made (``CONTEXT_BUDGET_EXCEEDED`` termination /
        ``context.budget_exceeded``): same failure class, same derived
        reason, different observer.
        """
        if self.termination_reason is None:
            return None
        operations = {
            event.get("operation") for event in self.lifecycle_events
        }
        if "context.length_exceeded" in operations:
            return "CONTEXT_LENGTH_EXCEEDED"
        if (
            self.termination_reason == "CONTEXT_BUDGET_EXCEEDED"
            or "context.budget_exceeded" in operations
        ):
            return "CONTEXT_LENGTH_EXCEEDED"
        if (
            "execution.boundary_reached" in operations
            or self.termination_reason == "EXECUTION_LIMIT_REACHED"
        ):
            return "BUDGET_EXCEEDED"
        if self.termination_reason in ("FINAL_ANSWER", "STRUCTURED_OUTPUT"):
            return "EVIDENCE_SUFFICIENT"
        return "ERROR"

    @property
    def context_restrictions(self) -> int:
        """How many times this run's request was bounded by the
        application working-context ceiling before it was sent."""
        return sum(
            1
            for event in self.lifecycle_events
            if event.get("operation") == "context.restricted"
        )

    @property
    def context_compactions(self) -> int:
        """How many times this run's WORKING conversation was replaced
        by a compacted one (persistent, above the same ceiling).

        Each compaction is a completed ``started`` → ``completed``
        pair; counting completions gives "did compaction actually
        happen", and the pair's before/after estimates say by how
        much.
        """
        return sum(
            1
            for event in self.lifecycle_events
            if event.get("operation") == "context.compaction.completed"
        )

    @property
    def effort_threshold_reached(self) -> bool:
        """Whether this run crossed the SOFT investigation target.

        One ``execution.effort_threshold_reached`` event is recorded
        at most once per run, when the execution-state notice was
        appended. It is an EFFICIENCY marker (how far the run got
        before deciding on its own, from evidence, to finalize),
        never a stop reason and never a blocked tool: the hard
        ``max_tool_calls`` boundary is the only thing that stops
        tool use, and it is recorded separately as
        ``execution.boundary_reached``.
        """
        return any(
            event.get("operation") == "execution.effort_threshold_reached"
            for event in self.lifecycle_events
        )

    def finalize(self, completed_at: Optional[str] = None) -> None:
        """Seal the trace: completed timestamp + run summary (idempotent)."""
        if self.summary is not None:
            return
        self.completed_at = completed_at or _utcnow()
        if self.termination_reason is None:
            self.terminate("ERROR")
        total_tool_calls = sum(
            len(iteration.tool_calls) for iteration in self.iterations
        )
        # M8.5 budget accounting: executed = ToolResults recorded
        # (budget consumed); blocked = requested but never executed.
        tool_calls_executed = sum(
            len(iteration.tool_results) for iteration in self.iterations
        )
        tool_calls_blocked = sum(
            len(iteration.blocked_tool_calls) for iteration in self.iterations
        )
        tools_used: List[str] = []
        for iteration in self.iterations:
            for result in iteration.tool_results:
                name = result.get("tool")
                if name and name not in tools_used:
                    tools_used.append(name)
        self.summary = TraceSummary(
            total_iterations=len(self.iterations),
            total_tool_calls=total_tool_calls,
            tools_used=tools_used,
            termination_reason=self.termination_reason,
            final_answer=self.final_answer,
            max_tool_calls=self.max_tool_calls,
            tool_calls_executed=tool_calls_executed,
            tool_calls_blocked=tool_calls_blocked,
        )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready representation (canonical machine format input)."""
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "agent": dict(self.agent),
            "user_question": self.user_question,
        }
        if self.repository is not None:
            payload["repository"] = dict(self.repository)
        if self.architecture_context is not None:
            payload["architecture_context"] = dict(self.architecture_context)
        if self.memory_events:
            payload["memory_events"] = [
                dict(event) for event in self.memory_events
            ]
        if self.context_events:
            payload["context_events"] = [
                dict(event) for event in self.context_events
            ]
        if self.collaboration_events:
            payload["collaboration_events"] = [
                dict(event) for event in self.collaboration_events
            ]
        if self.orchestration_events:
            payload["orchestration_events"] = [
                dict(event) for event in self.orchestration_events
            ]
        if self.lifecycle_events:
            payload["lifecycle_events"] = [
                dict(event) for event in self.lifecycle_events
            ]
        if self.plan is not None:
            payload["plan"] = dict(self.plan)
        payload["iterations"] = [
            iteration.to_dict() for iteration in self.iterations
        ]
        if self.summary is not None:
            payload["summary"] = self.summary.to_dict()
        if self.evidence_validation is not None:
            payload["evidence_validation"] = dict(self.evidence_validation)
        if self.sink_error is not None:
            payload["sink_error"] = self.sink_error
        return payload

    def to_json(self, indent: int = 2) -> str:
        """Canonical machine-readable trace format."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown projection (derived from the trace — never a second truth)
# ---------------------------------------------------------------------------


def _render_arguments(arguments: Dict[str, Any]) -> str:
    return json.dumps(arguments, indent=2, ensure_ascii=False, sort_keys=True)


def _message_excerpt(content: str, limit: int = 160) -> str:
    """One-line readable excerpt of a recorded message content."""
    flat = " ".join(str(content).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _status_label(
    result: Optional[Dict[str, Any]], success: Any = None
) -> str:
    """``SUCCESS`` / ``FAILURE (code)`` / ``UNKNOWN`` for one result or
    observation — deterministic labels, never model prose."""
    if success is None:
        success = (result or {}).get("success")
    if success is True:
        return "SUCCESS"
    if success is False:
        error = (result or {}).get("error") or {}
        return "FAILURE ({})".format(error.get("code") or "unknown")
    return "UNKNOWN"


def _blank(lines: List[str]) -> None:
    """Exactly one blank line before the next block (stable output)."""
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")


def _find_result(
    results: List[Dict[str, Any]], call_id: Any
) -> Optional[Dict[str, Any]]:
    """First result recorded for *call_id* (fallback pairing only)."""
    for result in results:
        if result.get("tool_call_id") == call_id:
            return result
    return None


def _source_label(source: Any) -> Optional[str]:
    """Readable provenance of one ToolResult ("repo, snapshot …")."""
    if not isinstance(source, dict) or not source:
        return None
    bits: List[str] = []
    for key in ("repository", "artifact"):
        value = source.get(key)
        if value:
            bits.append(str(value))
    if source.get("snapshot_id"):
        bits.append("snapshot {}".format(source["snapshot_id"]))
    if source.get("commit_sha"):
        bits.append("commit {}".format(source["commit_sha"]))
    return ", ".join(bits) or None


def _derived_continuation(
    trace: "AgentTrace", position: int, iteration: AgentIteration
) -> Dict[str, Any]:
    """Best-effort continuation for a trace recorded before M8.7.1.

    Agent runs record ``continuation`` explicitly at every ``close``;
    a hand-built or legacy trace has none, so the renderer derives the
    same observable decision from the recorded decision type and run
    termination — never invents anything model-internal.
    """
    decision_type = (
        iteration.decision.get("type") or iteration.response_type or "UNKNOWN"
    )
    if decision_type == "FINAL_ANSWER":
        return {
            "decision": "FINAL_ANSWER",
            "reason": "LLM returned final answer",
            "termination_reason": "FINAL_ANSWER",
        }
    if decision_type == "STRUCTURED_OUTPUT":
        # M8.9: recorded runs state their own continuation; this is the
        # legacy/hand-built fallback — same observable shape.
        return {
            "decision": "FINAL_ANSWER",
            "reason": "LLM returned structured output",
            "termination_reason": trace.termination_reason or "STRUCTURED_OUTPUT",
        }
    if decision_type == "INVALID_RESPONSE":
        return {
            "decision": "TERMINATE",
            "reason": "LLM response contained no textual answer",
            "termination_reason": "INVALID_LLM_RESPONSE",
        }
    if decision_type == "LIMIT_REACHED":
        return {
            "decision": "TERMINATE",
            "reason": "max_tool_calls reached",
            "termination_reason": "EXECUTION_LIMIT_REACHED",
        }
    if decision_type == "ERROR":
        return {
            "decision": "TERMINATE",
            "reason": "run terminated with an error",
            "termination_reason": trace.termination_reason or "ERROR",
        }
    # TOOL_CALL (or unknown): a following iteration means CONTINUE.
    if position + 1 < len(trace.iterations):
        return {
            "decision": "CONTINUE",
            "reason": "tool result added to conversation state",
            "next_iteration": iteration.iteration + 1,
        }
    termination = trace.termination_reason
    if termination in (
        "TOOL_DISPATCH_ERROR",
        "EXECUTION_LIMIT_REACHED",
        "LLM_ERROR",
        "INVALID_LLM_RESPONSE",
        "ERROR",
    ):
        return {
            "decision": "TERMINATE",
            "reason": "run ended after this iteration",
            "termination_reason": termination,
        }
    return {
        "decision": "CONTINUE",
        "reason": "tool result added to conversation state",
    }


def render_markdown(trace: AgentTrace) -> str:
    """Human-readable debugging projection of ``trace.to_dict()``.

    M8.7.1 — one file reads top-to-bottom as the ordered interaction
    stages. Per iteration:

        LLM Request → LLM Response → Actions → Tool Executions →
        Tool Results → State Update → Continuation → Answer (final)

    then, at run level, **Plan (M10: objective, validation, step
    lifecycle, Architecture Context record) → Structured Claims →
    M8.8 Evidence Validation → Run Summary** — so one file answers
    "what went to
    the model, what came back, what was requested, what executed,
    what went back, and whether the run continues or ends".
    Observable inputs/outputs only; hidden chain of thought is never
    rendered because it is never recorded. The projection is derived
    from the trace — never a second source of truth — and tolerates
    legacy/hand-built traces without ``continuation``/``duration_ms``
    (derived from the recorded decision instead).
    """
    lines: List[str] = ["# Agent Execution Trace", ""]
    lines.append("* Run: `{}`".format(trace.run_id))
    lines.append("* Started: {}".format(trace.started_at))
    if trace.completed_at is not None:
        lines.append("* Completed: {}".format(trace.completed_at))
    lines.append(
        "* Agent: {} {}".format(
            trace.agent.get("name", AGENT_NAME),
            trace.agent.get("version", ""),
        ).rstrip()
    )
    if trace.repository:
        commit = trace.repository.get("commit_sha")
        snapshot = trace.repository.get("snapshot_id")
        if commit:
            lines.append("* Repository commit: `{}`".format(commit))
        if snapshot:
            lines.append("* Snapshot: `{}`".format(snapshot))
    _blank(lines)
    lines.extend(["Question:", "> {}".format(trace.user_question)])

    # M10 run-level plan section (only when recorded — non-plan
    # projections are byte-for-byte unchanged).
    if trace.plan is not None or trace.architecture_context is not None:
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Plan")
        lines.append("")
        if trace.architecture_context is not None:
            context = trace.architecture_context
            line = "* Architecture Context: `{}`".format(
                context.get("artifact") or "unknown"
            )
            details = []
            if context.get("schema_version"):
                details.append("`{}`".format(context["schema_version"]))
            if context.get("commit_sha"):
                details.append("commit `{}`".format(context["commit_sha"]))
            if context.get("snapshot_id"):
                details.append("snapshot `{}`".format(context["snapshot_id"]))
            if details:
                line += " ({})".format(", ".join(details))
            lines.append(line)
        plan = trace.plan or {}
        if plan:
            if plan.get("objective") is not None:
                lines.append("* Objective: {}".format(plan["objective"]))
            if plan.get("status") is not None:
                lines.append("* Status: `{}`".format(plan["status"]))
            validation_payload = plan.get("validation")
            if validation_payload:
                if validation_payload.get("status") == "SUCCESS":
                    lines.append("* Plan validation: SUCCESS")
                else:
                    failure = " ".join(
                        str(validation_payload.get("error") or "").split()
                    )
                    lines.append(
                        "* Plan validation: FAILED{}".format(
                            " \u2014 {}".format(failure) if failure else ""
                        )
                    )
            steps = plan.get("steps") or []
            if steps:
                _blank(lines)
                lines.append("### Plan Steps")
                lines.append("")
                for step in steps:
                    lines.append(
                        "* `{}` \u2014 **{}** \u2014 {}".format(
                            step.get("id") or "?",
                            step.get("status") or "pending",
                            step.get("goal") or "(no goal)",
                        )
                    )
                    if step.get("result"):
                        lines.append(
                            "  * result: {}".format(step["result"])
                        )

    # M14: one chronological Memory section (retrieval happens at the
    # execution boundary BEFORE planning/tool execution; store/verify
    # after validation — all events of this run in one place).
    if trace.memory_events:
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Memory")
        lines.append("")
        for event in trace.memory_events:
            lines.append("* {}".format(_render_memory_event(event)))

    # M15: one chronological Context section — the selection happened
    # right after retrieval at the execution boundary, so it reads
    # between the memory events and the first iteration.
    if trace.context_events:
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Context")
        lines.append("")
        for event in trace.context_events:
            lines.append("* {}".format(_render_context_event(event)))

    # M16: one chronological Collaboration section — which stages ran
    # and which typed messages crossed the boundary, in execution
    # order (records only; the runner owns the events).
    if trace.collaboration_events:
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Collaboration")
        lines.append("")
        for event in trace.collaboration_events:
            lines.append("* {}".format(_render_collaboration_event(event)))

    # M17: one chronological Orchestration section — which plan step
    # this agent executed, why it was assigned this role, and how the
    # step ended, in execution order (records only; the orchestrator
    # owns the events).
    if trace.orchestration_events:
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Orchestration")
        lines.append("")
        for event in trace.orchestration_events:
            lines.append("* {}".format(_render_orchestration_event(event)))

    validation = trace.evidence_validation

    for position, iteration in enumerate(trace.iterations):
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Iteration {}".format(iteration.iteration))
        lines.append("")
        if iteration.plan_step_id is not None:
            # M10 correlation: which plan step this iteration executed.
            lines.append("Plan step: `{}`".format(iteration.plan_step_id))
            lines.append("")
        decision_type = iteration.decision.get("type", "UNKNOWN")
        # This iteration's own executed results (call ids may repeat
        # across iterations in scripted runs — always resolve locally).
        local_results = {
            result.get("tool_call_id"): result
            for result in iteration.tool_results
        }
        # The results that produced THIS request's tool messages, in
        # chronological order (one ToolResult <-> one tool message).
        prior_results = [
            result
            for prior in trace.iterations[:position]
            for result in prior.tool_results
        ]

        # -- LLM Request: exactly what went OUT to the model ----------
        _blank(lines)
        lines.extend(["### LLM Request", ""])
        request = iteration.request or {}
        messages = request.get("messages")
        lines.append(
            "**Messages:** {}".format(
                request.get(
                    "message_count",
                    len(messages) if isinstance(messages, list) else 0,
                )
            )
        )
        if "tool_definition_count" in request:
            lines.append(
                "**Tools available:** {}".format(
                    request.get("tool_definition_count")
                )
            )
        tool_messages = [
            message
            for message in (messages or [])
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        if tool_messages:
            lines.append(
                "**Includes {} previous observation{}:**".format(
                    len(tool_messages),
                    "" if len(tool_messages) == 1 else "s",
                )
            )
            for index, message in enumerate(tool_messages):
                result = (
                    prior_results[index]
                    if index < len(prior_results)
                    else _find_result(prior_results, message.get("tool_call_id"))
                )
                tool_name = (result or {}).get("tool") or "tool"
                lines.append(
                    "* `{}` (`{}`) \u2192 {}".format(
                        tool_name,
                        message.get("tool_call_id") or "-",
                        _status_label(result),
                    )
                )
        if messages:
            lines.append("")
            for index, message in enumerate(messages, start=1):
                if not isinstance(message, dict):
                    continue
                role = message.get("role", "?")
                if message.get("tool_call_id") is not None:
                    line = "{}. **{}** (`{}`)".format(
                        index, role, message.get("tool_call_id")
                    )
                else:
                    line = "{}. **{}**".format(index, role)
                content = message.get("content")
                if content:
                    line += " \u2014 {}".format(_message_excerpt(content))
                lines.append(line)
                for call in message.get("tool_calls") or []:
                    lines.append(
                        "   * requested `{}` (`{}`)".format(
                            call.get("name"), call.get("id")
                        )
                    )

        # -- LLM Response: what came BACK from the model --------------
        _blank(lines)
        lines.extend(["### LLM Response", ""])
        lines.append("**Decision:** `{}`".format(decision_type))
        if "tool" in iteration.decision:
            _blank(lines)
            lines.append("**Tool:** `{}`".format(iteration.decision["tool"]))
        elif "tools" in iteration.decision:
            tools = ", ".join(
                "`{}`".format(name) for name in iteration.decision["tools"]
            )
            _blank(lines)
            lines.append("**Tools:** {}".format(tools))
        llm = iteration.llm_response or {}
        meta = iteration.response or {}
        response_type = (
            llm.get("type") or iteration.response_type or decision_type
        )
        detail = "Type: `{}`".format(response_type)
        if "finish_reason" in meta:
            detail += " \u00b7 Finish reason: `{}`".format(
                meta.get("finish_reason")
            )
        _blank(lines)
        lines.append(detail)
        # M8.9: application-side validation outcome of a structured
        # response — SUCCESS (typed result) or FAILED (explicit error).
        # (Named distinctly: ``validation`` below holds the run-level
        # evidence-validation block.)
        structured_validation = iteration.structured_validation
        if structured_validation is not None:
            if (structured_validation.get("status") or "") == "SUCCESS":
                lines.append("Validation status: SUCCESS")
            else:
                failure = " ".join(
                    str(structured_validation.get("error") or "").split()
                )
                lines.append(
                    "Validation status: FAILED \u2014 {}".format(failure)
                )
        # Correlation anchor: the exact ids the response requested —
        # the same ids rendered by the action and result stages.
        requested = llm.get("tool_calls") or iteration.tool_calls
        if requested:
            call_ids = ", ".join(
                "`{}`".format(call.get("id") or "-") for call in requested
            )
            lines.append("Tool calls requested: {}".format(call_ids))
        usage = meta.get("usage")
        if usage:
            lines.append(
                "Tokens: {} in \u00b7 {} out".format(
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
            )
        # Explicit text block: a TOOL_CALL without text is a complete
        # LLM response, not a missing one. The final response shows
        # its prose here, right where the learner reads the response.
        if iteration.response_type == "FINAL_ANSWER" and iteration.final_answer:
            _blank(lines)
            lines.append("Text:")
            lines.append(iteration.final_answer)
        elif iteration.response_type == "STRUCTURED_OUTPUT":
            # M8.9: the decoded JSON object the model returned (when
            # it arrived); a failed validation explains what did not.
            _blank(lines)
            structured_payload = llm.get("structured_output")
            if structured_payload is not None:
                lines.append("Structured output:")
                lines.extend(
                    ["```json", _render_arguments(structured_payload), "```"]
                )
            elif meta.get("has_text"):
                lines.append(
                    "Text: <present \u2014 not a structured payload>"
                )
            else:
                lines.append(
                    "Text: <none \u2014 structured output requested>"
                )
        elif iteration.llm_response is not None:
            _blank(lines)
            if meta.get("has_text"):
                lines.append(
                    "Text: <present \u2014 kept in the conversation state>"
                )
            elif response_type == "TOOL_CALL":
                lines.append(
                    "Text: <none \u2014 response requested tool execution>"
                )
            else:
                lines.append("Text: <none>")
        if iteration.response_type == "ERROR":
            _blank(lines)
            lines.append("**Error:** {}".format(iteration.error or ""))

        # -- ACTION: exactly what was requested, verbatim -------------
        if requested:
            _blank(lines)
            lines.append("### Actions")
            lines.append("")
            for index, call in enumerate(requested, start=1):
                if index > 1:
                    lines.append("")
                lines.append(
                    "**{}. `{}`** (`{}`)".format(
                        index, call.get("name"), call.get("id") or "-"
                    )
                )
                lines.extend(
                    [
                        "```json",
                        _render_arguments(call.get("arguments") or {}),
                        "```",
                    ]
                )

        # -- TOOL EXECUTION: dispatch status + duration, per call -----
        blocked_ids = {
            entry.get("tool_call_id")
            for entry in iteration.blocked_tool_calls
        }
        if iteration.tool_calls:
            _blank(lines)
            lines.append("### Tool Executions")
            lines.append("")
            for call in iteration.tool_calls:
                call_id = call.get("id")
                result = local_results.get(call_id)
                if result is None:
                    if call_id in blocked_ids:
                        status = "NOT EXECUTED (max_tool_calls reached)"
                    else:
                        status = "NOT EXECUTED"
                else:
                    status = _status_label(result)
                    duration = result.get("duration_ms")
                    if duration is not None:
                        status += " \u00b7 duration: {} ms".format(duration)
                lines.append(
                    "* `{}` (`{}`) \u2192 {}".format(
                        call_id or "-", call.get("name") or "tool", status
                    )
                )

        # -- TOOL RESULT: what actually executed ----------------------
        if iteration.tool_calls:
            _blank(lines)
            lines.append("### Tool Results")
            lines.append("")
            for call in iteration.tool_calls:
                call_id = call.get("id")
                _blank(lines)
                lines.append(
                    "**`{}` (`{}`)**".format(
                        call_id or "-", call.get("name") or "tool"
                    )
                )
                lines.append("")
                result = local_results.get(call_id)
                if result is None:
                    if call_id in blocked_ids:
                        lines.append(
                            "**Result:** NOT EXECUTED "
                            "(execution budget exhausted)"
                        )
                    else:
                        lines.append("**Result:** NOT EXECUTED")
                    continue
                if result.get("success"):
                    lines.append("**Result:** SUCCESS")
                else:
                    error = result.get("error") or {}
                    lines.append(
                        "**Result:** FAILURE ({})".format(
                            error.get("code") or "unknown"
                        )
                    )
                tool_name = result.get("tool") or call.get("name") or "tool"
                if result.get("success"):
                    observation = "{} result returned to the Agent.".format(
                        tool_name
                    )
                else:
                    error = result.get("error") or {}
                    observation = "{} failure ({}) returned to the Agent.".format(
                        tool_name, error.get("code") or "unknown"
                    )
                _blank(lines)
                lines.append("**Observation:** {}".format(observation))
                source_text = _source_label(result.get("source"))
                if source_text:
                    _blank(lines)
                    lines.append("**Source:** {}".format(source_text))

        # -- STATE UPDATE: what this iteration added to the
        # conversation, and what the next request will carry --------
        if iteration.tool_calls:
            _blank(lines)
            lines.append("### State Update")
            lines.append("")
            if "no ToolDispatcher is configured" in (
                iteration.reason_summary or ""
            ):
                lines.append(
                    "* no conversation messages added "
                    "(tool execution unavailable)"
                )
            else:
                call_ids = ", ".join(
                    "`{}`".format(call.get("id") or "-")
                    for call in iteration.tool_calls
                )
                lines.append(
                    "* assistant message added ({} requested tool "
                    "call{}: {})".format(
                        len(iteration.tool_calls),
                        "" if len(iteration.tool_calls) == 1 else "s",
                        call_ids,
                    )
                )
                if iteration.observations:
                    for observation in iteration.observations:
                        call_id = observation.get("tool_call_id")
                        tool_name = (
                            observation.get("tool")
                            or (local_results.get(call_id) or {}).get("tool")
                            or "tool"
                        )
                        lines.append(
                            "* `{}` (`{}`) \u2192 {}".format(
                                tool_name,
                                call_id or "-",
                                _status_label(
                                    local_results.get(call_id),
                                    observation.get("success"),
                                ),
                            )
                        )
                else:
                    lines.append(
                        "* no tool result message added (0 executed)"
                    )
                following = (
                    trace.iterations[position + 1]
                    if position + 1 < len(trace.iterations)
                    else None
                )
                if following is not None:
                    count = (following.request or {}).get("message_count")
                    if count is not None:
                        lines.append(
                            "Next LLM request: iteration {} "
                            "({} messages)".format(following.iteration, count)
                        )
                    else:
                        lines.append(
                            "Next LLM request: iteration {}".format(
                                following.iteration
                            )
                        )
                elif trace.termination_reason is not None:
                    lines.append(
                        "Next LLM request: none (run terminated: {})".format(
                            trace.termination_reason
                        )
                    )
                else:
                    lines.append("Next LLM request: none (trace ended)")

        if iteration.reason_summary:
            _blank(lines)
            lines.append("*Summary:* {}".format(iteration.reason_summary))

        # -- CONTINUATION / TERMINATION: the observable decision to
        # keep going or stop (recorded by the Agent loop; derived for
        # legacy traces). Terminal decisions render as Termination so
        # the control flow is unambiguous top-to-bottom. -------------
        continuation = iteration.continuation or _derived_continuation(
            trace, position, iteration
        )
        cont_decision = continuation.get("decision")
        _blank(lines)
        if cont_decision == "CONTINUE":
            lines.append("### Continuation")
            lines.append("")
            if "finalization" in (continuation.get("reason") or ""):
                # Execution-lifecycle boundary: no tool result follows —
                # the tool-free finalization pass is the next request.
                lines.append(
                    "Execution boundary reached: finalization pass "
                    "starts (tools disabled)."
                )
            elif (
                iteration.plan_step_id is not None
                and iteration.response_type == "FINAL_ANSWER"
            ):
                # M10 step boundary: this text is the step's result,
                # and the run continues with the next plan step.
                lines.append(
                    "Plan step `{}` result recorded.".format(
                        iteration.plan_step_id
                    )
                )
                lines.append("Continue plan execution.")
            else:
                lines.append("Tool result will be returned to the LLM.")
                lines.append("Continue investigation.")
        else:
            lines.append("### Termination")
            lines.append("")
            if cont_decision == "FINAL_ANSWER":
                lines.append("Final answer received from LLM.")
            elif (
                continuation.get("termination_reason")
                == "EXECUTION_LIMIT_REACHED"
            ):
                lines.append("Tool-call limit reached.")
            else:
                lines.append(
                    "Run terminated: {}.".format(
                        continuation.get("termination_reason")
                        or continuation.get("reason")
                        or "unknown"
                    )
                )
        _blank(lines)
        lines.append("Decision: {}".format(cont_decision))
        lines.append("Reason: {}".format(continuation.get("reason")))
        if continuation.get("next_iteration") is not None:
            lines.append(
                "Next iteration: {}".format(continuation["next_iteration"])
            )
        if continuation.get("termination_reason") is not None:
            lines.append(
                "Termination: `{}`".format(continuation["termination_reason"])
            )

        # -- FINAL ANSWER: the model's prose (final iteration only) --
        if iteration.response_type == "FINAL_ANSWER":
            _blank(lines)
            lines.extend(["### Answer", "", iteration.final_answer or ""])

    # Execution lifecycle (boundary → finalization): rendered after the
    # iterations because that is when it happened, and before the run
    # summary it explains. Recorded data only, like every other section.
    if trace.lifecycle_events:
        _blank(lines)
        lines.append("---")
        _blank(lines)
        lines.append("## Execution Lifecycle")
        lines.append("")
        for event in trace.lifecycle_events:
            lines.append("* {}".format(_render_lifecycle_event(event)))

    # M8.8 — run-level section after the investigation (recorded data
    # only): the model's claims, then the deterministic per-claim
    # outcome, then the run summary.
    if validation is not None:
        _blank(lines)
        lines.append("---")
        lines.extend(_render_validation(validation))

    _blank(lines)
    lines.append("---")
    _blank(lines)
    lines.append("## Run Summary")
    lines.append("")
    summary = trace.summary
    if summary is None:
        lines.append("* Trace not finalized.")
    else:
        lines.append("* Total iterations: {}".format(summary.total_iterations))
        lines.append("* Tool calls: {}".format(summary.total_tool_calls))
        if summary.max_tool_calls is not None:
            lines.append(
                "* Tool-call budget: {} max, {} executed, {} blocked".format(
                    summary.max_tool_calls,
                    summary.tool_calls_executed,
                    summary.tool_calls_blocked,
                )
            )
        lines.append(
            "* Tools used: {}".format(
                ", ".join(
                    "`{}`".format(name) for name in summary.tools_used
                )
                or "(none)"
            )
        )
        lines.append("* Termination: `{}`".format(summary.termination_reason))
        if trace.stop_reason is not None:
            lines.append(
                "* Stop reason: `{}`".format(trace.stop_reason)
            )
        if trace.context_restrictions:
            lines.append(
                "* Context restrictions: {}".format(
                    trace.context_restrictions
                )
            )
        if trace.context_compactions:
            lines.append(
                "* Context compactions: {}".format(
                    trace.context_compactions
                )
            )
        if trace.effort_threshold_reached:
            lines.append("* Soft effort target: reached")
        if summary.final_answer is not None:
            _blank(lines)
            lines.append("> {}".format(summary.final_answer))
    lines.append("")
    return "\n".join(lines)


def _render_memory_event(event: Dict[str, Any]) -> str:
    """One deterministic line per M14 memory event — recorded metadata
    only (operation, ids, scope, status), never model prose."""
    operation = event.get("operation") or "?"
    parts: List[str] = []
    if event.get("count") is not None:
        parts.append("{} retrieved".format(event["count"]))
    if event.get("memory_ids"):
        parts.append(
            ", ".join(
                "`{}`".format(memory_id)
                for memory_id in event["memory_ids"]
            )
        )
    if event.get("memory_id"):
        parts.append("`{}`".format(event["memory_id"]))
    if event.get("repository"):
        parts.append("repository `{}`".format(event["repository"]))
    if event.get("commit"):
        parts.append("commit `{}`".format(event["commit"]))
    if event.get("status"):
        parts.append("status `{}`".format(event["status"]))
    if event.get("verification_status"):
        # tri-state scope check (M14 §26): fresh / stale / unknown —
        # unknown is never rendered as fresh (or as "stale: no")
        parts.append(
            "verification `{}`".format(event["verification_status"])
        )
    elif event.get("stale") is not None:
        parts.append("stale: {}".format("yes" if event["stale"] else "no"))
    error = event.get("error")
    if isinstance(error, dict) and error.get("code"):
        parts.append(
            "ERROR {}: {}".format(
                error.get("code"), error.get("message") or ""
            )
        )
    if not parts:
        return "`{}`".format(operation)
    return "`{}` \u2014 {}".format(operation, ", ".join(parts))


def _render_context_event(event: Dict[str, Any]) -> str:
    """One deterministic line per M15 context event — recorded
    metadata only (operation, context/step ids, counts, budget,
    reason counts), never model prose or message content."""
    operation = event.get("operation") or "?"
    parts: List[str] = []
    if event.get("context_id"):
        parts.append("`{}`".format(event["context_id"]))
    if event.get("step_id"):
        parts.append("step `{}`".format(event["step_id"]))
    if event.get("candidate_count") is not None:
        parts.append(
            "{} candidates \u2192 {} selected / {} excluded".format(
                event.get("candidate_count", 0),
                event.get("selected_count", 0),
                event.get("excluded_count", 0),
            )
        )
    budget = event.get("budget")
    if isinstance(budget, dict) and budget:
        parts.append(
            "budget {}/{} chars, {}/{} items".format(
                budget.get("estimated_characters", 0),
                budget.get("max_characters", 0),
                budget.get("selected_items", 0),
                budget.get("max_items", 0),
            )
        )
    if event.get("characters") is not None:
        parts.append("{} characters".format(event["characters"]))
    if event.get("repository"):
        parts.append("repository `{}`".format(event["repository"]))
    if event.get("commit"):
        parts.append("commit `{}`".format(event["commit"]))
    reasons = event.get("reasons")
    if isinstance(reasons, dict) and reasons:
        parts.append(
            "reasons: {}".format(
                ", ".join(
                    "{}={}".format(reason, reasons[reason])
                    for reason in sorted(reasons)
                )
            )
        )
    error = event.get("error")
    if isinstance(error, dict) and error.get("code"):
        parts.append(
            "ERROR {}: {}".format(
                error.get("code"), error.get("message") or ""
            )
        )
    if not parts:
        return "`{}`".format(operation)
    return "`{}` \u2014 {}".format(operation, ", ".join(parts))


def _render_collaboration_event(event: Dict[str, Any]) -> str:
    """One deterministic line per M16 collaboration event — recorded
    metadata only (operation, sequence, actor, message type, status),
    never message prose or model reasoning."""
    operation = event.get("operation") or "?"
    parts: List[str] = []
    if event.get("sequence") is not None:
        parts.append("seq {}".format(event["sequence"]))
    if event.get("actor"):
        parts.append("actor `{}`".format(event["actor"]))
    if event.get("message_type"):
        parts.append("type `{}`".format(event["message_type"]))
    if event.get("status"):
        parts.append("status `{}`".format(event["status"]))
    if event.get("reason"):
        parts.append("reason `{}`".format(event["reason"]))
    if event.get("detail"):
        parts.append(str(event["detail"]))
    if not parts:
        return "`{}`".format(operation)
    return "`{}` \u2014 {}".format(operation, ", ".join(parts))


def _render_orchestration_event(event: Dict[str, Any]) -> str:
    """One deterministic line per M17 orchestration event — recorded
    metadata only (operation, sequence, step, role, assignment reason,
    message type, status), never model reasoning."""
    operation = event.get("operation") or "?"
    parts: List[str] = []
    if event.get("sequence") is not None:
        parts.append("seq {}".format(event["sequence"]))
    if event.get("step_id"):
        parts.append("step `{}`".format(event["step_id"]))
    if event.get("agent_role"):
        parts.append("role `{}`".format(event["agent_role"]))
    if event.get("attempt") is not None:
        parts.append("attempt {}".format(event["attempt"]))
    if event.get("message_type"):
        parts.append("type `{}`".format(event["message_type"]))
    if event.get("status"):
        parts.append("status `{}`".format(event["status"]))
    if event.get("assignment_reason"):
        parts.append("reason `{}`".format(event["assignment_reason"]))
    if event.get("reason"):
        parts.append("reason `{}`".format(event["reason"]))
    if event.get("termination_reason"):
        parts.append("termination `{}`".format(event["termination_reason"]))
    if event.get("detail"):
        parts.append(str(event["detail"]))
    if not parts:
        return "`{}`".format(operation)
    return "`{}` \u2014 {}".format(operation, ", ".join(parts))


def _render_lifecycle_event(event: Dict[str, Any]) -> str:
    """One deterministic line per execution-lifecycle event — recorded
    metadata only (operation, reason, budget counters, counts, tools
    disabled), never model reasoning or message content."""
    operation = event.get("operation") or "?"
    parts: List[str] = []
    if event.get("reason"):
        parts.append("reason `{}`".format(event["reason"]))
    if event.get("decision"):
        parts.append("decision `{}`".format(event["decision"]))
    if event.get("decision_point") is not None:
        parts.append("decision point {}".format(event["decision_point"]))
    if event.get("tool"):
        parts.append("tool `{}`".format(event["tool"]))
    if event.get("tool_calls_executed") is not None:
        parts.append(
            "{} of {} tool call(s) executed".format(
                event.get("tool_calls_executed", 0),
                event.get("max_tool_calls", 0),
            )
        )
    if event.get("soft_tool_call_target") is not None:
        parts.append(
            "soft target {} tool call(s)".format(
                event["soft_tool_call_target"]
            )
        )
    if event.get("tools_disabled") is not None:
        parts.append(
            "tools {}".format(
                "disabled" if event.get("tools_disabled") else "enabled"
            )
        )
    if event.get("evidence_count") is not None:
        parts.append("{} evidence ref(s)".format(event["evidence_count"]))
    if event.get("observation_count") is not None:
        parts.append(
            "{} observation(s)".format(event.get("observation_count"))
        )
    if event.get("before_characters") is not None:
        parts.append(
            "{} → {} character(s)".format(
                event.get("before_characters"),
                event.get("after_characters"),
            )
        )
    if event.get("max_context_characters") is not None:
        parts.append(
            "ceiling {} character(s)".format(
                event.get("max_context_characters")
            )
        )
    if event.get("messages_dropped") is not None:
        parts.append(
            "{} message(s) dropped ({} observation(s))".format(
                event.get("messages_dropped"),
                event.get("observations_dropped"),
            )
        )
    if event.get("evidence_recorded") is not None:
        parts.append(
            "{} tool result(s) kept on the trace".format(
                event.get("evidence_recorded")
            )
        )
    if event.get("before_context_estimate") is not None:
        if event.get("after_context_estimate") is not None:
            parts.append(
                "{} → {} character(s)".format(
                    event.get("before_context_estimate"),
                    event.get("after_context_estimate"),
                )
            )
        else:
            parts.append(
                "{} character(s)".format(
                    event.get("before_context_estimate")
                )
            )
    if event.get("messages_before") is not None:
        if event.get("messages_after") is not None:
            parts.append(
                "{} → {} message(s)".format(
                    event.get("messages_before"),
                    event.get("messages_after"),
                )
            )
        else:
            parts.append(
                "{} message(s)".format(event.get("messages_before"))
            )
    if event.get("evidence_count") is not None:
        parts.append(
            "{} evidence reference(s)".format(event.get("evidence_count"))
        )
    if event.get("failed_attempts_count") is not None:
        parts.append(
            "{} failed attempt(s)".format(
                event.get("failed_attempts_count")
            )
        )
    if event.get("termination_reason"):
        parts.append(
            "termination `{}`".format(event["termination_reason"])
        )
    error = event.get("error")
    if isinstance(error, str) and error:
        parts.append("error: {}".format(error))
    elif isinstance(error, dict) and error.get("code"):
        parts.append(
            "ERROR {}: {}".format(
                error.get("code"), error.get("message") or ""
            )
        )
    if not parts:
        return "`{}`".format(operation)
    return "`{}` \u2014 {}".format(operation, ", ".join(parts))


def _render_validation(validation: Dict[str, Any]) -> List[str]:
    """``### Structured Claims`` + ``### M8.8 Evidence Validation``
    blocks (M8.8 integration): the model's explicit claims, then the
    deterministic per-claim outcome — recorded data only, clipped,
    never model-authored prose beyond the claims themselves."""
    lines: List[str] = []
    claims = validation.get("claims") or []
    if claims:
        lines.extend(["", "### Structured Claims", ""])
        for index, claim in enumerate(claims, start=1):
            lines.append(
                "{}. {}".format(index, claim.get("statement") or "(no statement)")
            )
            for ref in claim.get("evidence_refs") or []:
                path = ref.get("path") or "(no path)"
                if (
                    ref.get("line_start") is not None
                    and ref.get("line_end") is not None
                ):
                    path = "{}:{}-{}".format(
                        path, ref.get("line_start"), ref.get("line_end")
                    )
                lines.append("   * evidence: `{}`".format(path))
    lines.extend(["", "### M8.8 Evidence Validation", ""])
    results = validation.get("results") or []
    if not results:
        lines.append("No structured claims were produced.")
    for entry in results:
        claim = entry.get("claim") or {}
        lines.append(
            "* `{}` — {}".format(
                entry.get("status", "UNKNOWN"),
                claim.get("statement") or "(no statement)",
            )
        )
        if entry.get("code"):
            # M13: recorded structured failure (e.g.
            # CLAIM_EVIDENCE_INVALID) — part of the trace, not prose.
            lines.append("  * code: {}".format(entry["code"]))
        if entry.get("reason"):
            lines.append("  * reason: {}".format(entry["reason"]))
    return lines
