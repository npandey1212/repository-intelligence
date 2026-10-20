"""Request-budget preflight at the model-call boundary (H1 + H2).

Every Agent model invocation (the main loop, the planner call, and the
tool-free finalization pass) sends one provider-neutral ``LLMRequest``.
This module is the single chokepoint that:

* **observes** the COMPLETE outgoing request — messages, tool-call
  argument JSON, tool definitions, and the structured-output
  (response-format) schema — and produces a structured, privacy-
  conscious diagnostic record (H1); and
* **enforces** a hard application request budget before the provider
  is ever invoked (H2): a request over budget has its tool
  observations deterministically clipped to bounded, valid-JSON
  excerpts (the full results already live on the execution trace); a
  request whose PROTECTED content (preamble, instructions, assistant
  turns, tool-call pairing) alone exceeds the budget is rejected —
  never sent, never retried unchanged.

Estimation policy (explicit, never presented as exact tokens): sizes
are measured in characters and converted with a fixed
``CHARS_PER_TOKEN`` ratio. No tokenizer is required, no provider SDK is
touched, and the record always labels the numbers as estimates
(``estimation.method``, ``estimation.is_estimate``). Provider-reported
usage arrives separately on the iteration's ``response.usage`` block —
it is never conflated with these preflight estimates.

Privacy: a diagnostic record contains sizes, roles, counts, tool
names, and dispositions ONLY. It never contains message content, tool
result payloads, repository file contents, or credentials. Instrumenta-
tion itself adds nothing to the outgoing request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm import LLMRequest, Message

__all__ = [
    "CHARS_PER_TOKEN",
    "DISPOSITION_ACCEPTED",
    "DISPOSITION_REDUCED",
    "DISPOSITION_REJECTED",
    "DEFAULT_CONTEXT_CAPACITY_TOKENS",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "MIN_TOOL_EXCERPT_CHARS",
    "ESTIMATION_METHOD",
    "RequestBudgetExceededError",
    "RequestBudgetPolicy",
    "build_request_budget_record",
    "enforce_request_budget",
    "enforce_request_budget_or_raise",
    "estimate_tokens",
    "note_provider_context_overflow",
]

# Deterministic character-to-token ratio used for every estimate.
# ~4 characters per token is the usual English/JSON ballpark for
# OpenAI-family tokenizers; it is an ESTIMATE, never a tokenizer.
CHARS_PER_TOKEN = 4

# Reserved capacity that is never spent on input: output headroom and
# a wide safety margin. Applied when a model context capacity is
# configured; irrelevant when an explicit input budget is given.
DEFAULT_RESERVED_OUTPUT_TOKENS = 16_384
DEFAULT_SAFETY_MARGIN_TOKENS = 16_384

# The assumed model context window when NO budget is configured — the
# observed 128k-token window of the default model (gpt-4o-mini), which
# is exactly the window the real overflow hit (a ~128,882-token
# request). The policy derives the effective input budget from this
# capacity MINUS the output reservation and safety margin above
# (→ 95,232 tokens), so the default path can never approach the
# window while a genuinely oversized request is still caught. Callers
# that know their model's window set ``context_capacity_tokens`` (or
# an explicit ``max_input_tokens``) instead.
DEFAULT_CONTEXT_CAPACITY_TOKENS = 128_000

# Smallest tool-observation excerpt kept by the reduction policy. The
# excerpt is a valid JSON envelope, never a raw mid-token cut.
MIN_TOOL_EXCERPT_CHARS = 400

ESTIMATION_METHOD = "chars_per_token_estimate"

DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_REDUCED = "reduced"
DISPOSITION_REJECTED = "rejected"

# Envelope written in place of an oversized tool observation. It is
# valid JSON, states exactly what happened, and points at the durable
# record (the execution trace) that still holds the full result.
_CLIP_NOTE = (
    "tool observation clipped for the application request budget; "
    "the full result is recorded on this run's execution trace"
)


def estimate_tokens(characters: int, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """Estimated tokens for *characters* (always an estimate).

    Ceil division so a non-empty payload is never estimated as zero.
    """
    if characters <= 0:
        return 0
    return -(-characters // chars_per_token)


@dataclass(frozen=True)
class RequestBudgetPolicy:
    """The documented request-budget policy for one Agent configuration.

    Resolution order for the effective INPUT budget (tokens):

    1. ``max_input_tokens`` — an explicit target input budget (the
       tightest, most direct control);
    2. ``context_capacity_tokens`` minus ``reserved_output_tokens``
       minus ``safety_margin_tokens`` — for a known model capacity the
       input target is NEVER the full window: output capacity and a
       wide margin are always reserved. This field DEFAULTS to the
       assumed window (``DEFAULT_CONTEXT_CAPACITY_TOKENS``, the
       observed 128k-token window of the default model), so a policy
       constructed without arguments is still bounded; pass
       ``context_capacity_tokens=None`` for a capacity-free policy;
    3. ``max_input_characters`` divided by ``chars_per_token`` — only
       reachable when the capacity is explicitly ``None``.

    The character ceiling (when set) is additionally enforced as a
    hard character bound on the total request, so a mismatched ratio
    can never silently double the effective budget.
    """

    max_input_tokens: Optional[int] = None
    max_input_characters: Optional[int] = None
    context_capacity_tokens: Optional[int] = DEFAULT_CONTEXT_CAPACITY_TOKENS
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS
    chars_per_token: int = CHARS_PER_TOKEN

    def __post_init__(self) -> None:
        for name in ("max_input_tokens", "max_input_characters",
                     "context_capacity_tokens"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(
                    "{} must be None or an integer >= 1, got {!r}".format(
                        name, value
                    )
                )
        for name in ("reserved_output_tokens", "safety_margin_tokens",
                     "chars_per_token"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    "{} must be an integer >= 0, got {!r}".format(name, value)
                )
        if self.chars_per_token < 1:
            raise ValueError(
                "chars_per_token must be >= 1, got {}".format(self.chars_per_token)
            )

    @property
    def effective_input_budget_tokens(self) -> Optional[int]:
        if self.max_input_tokens is not None:
            return self.max_input_tokens
        if self.context_capacity_tokens is not None:
            derived = (
                self.context_capacity_tokens
                - self.reserved_output_tokens
                - self.safety_margin_tokens
            )
            return max(derived, 1)
        if self.max_input_characters is not None:
            return estimate_tokens(
                self.max_input_characters, self.chars_per_token
            )
        return None

    @property
    def effective_input_budget_characters(self) -> Optional[int]:
        tokens = self.effective_input_budget_tokens
        characters = (
            tokens * self.chars_per_token if tokens is not None else None
        )
        if self.max_input_characters is not None:
            if characters is None:
                return self.max_input_characters
            return min(characters, self.max_input_characters)
        return characters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_input_characters": self.max_input_characters,
            "context_capacity_tokens": self.context_capacity_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "chars_per_token": self.chars_per_token,
            "effective_input_budget_tokens": self.effective_input_budget_tokens,
            "effective_input_budget_characters": (
                self.effective_input_budget_characters
            ),
        }


def _message_characters(message: Message) -> int:
    total = len(message.content or "")
    for call in message.tool_calls:
        total += len(call.name)
        total += len(json.dumps(call.arguments or {}))
    return total


def _request_characters(request: LLMRequest) -> int:
    total = sum(_message_characters(message) for message in request.messages)
    for tool in request.tools:
        total += len(tool.name)
        total += len(tool.description or "")
        total += len(json.dumps(tool.input_schema or {}))
    if request.structured_output is not None:
        total += len(request.structured_output.name)
        total += len(json.dumps(request.structured_output.schema or {}))
    return total


def _tool_schema_characters(request: LLMRequest) -> int:
    total = 0
    for tool in request.tools:
        total += len(tool.name)
        total += len(tool.description or "")
        total += len(json.dumps(tool.input_schema or {}))
    return total


def _response_format_characters(request: LLMRequest) -> int:
    if request.structured_output is None:
        return 0
    return len(request.structured_output.name) + len(
        json.dumps(request.structured_output.schema or {})
    )


def _tool_label(content: Optional[str]) -> Optional[str]:
    """Safe short label for a tool observation (the tool name only)."""
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    if isinstance(parsed, dict):
        name = parsed.get("tool")
        if isinstance(name, str) and name:
            return name[:100]
    return None


def build_request_budget_record(
    request: LLMRequest,
    policy: RequestBudgetPolicy,
    *,
    run_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """H1 diagnostic record for one outgoing request (sizes only).

    Captures the request AFTER every application-level transformation
    (compaction, restriction, tool/schema assembly) — the exact
    representation that would reach the provider. Contains no message
    content: roles, character counts, token ESTIMATES, tool names, and
    the configured budget only.
    """
    chars_per_token = policy.chars_per_token
    role_counts: Dict[str, int] = {}
    messages_meta: List[Dict[str, Any]] = []
    tool_call_count = 0
    tool_call_args_chars = 0
    messages_chars = 0
    for index, message in enumerate(request.messages):
        role_counts[message.role] = role_counts.get(message.role, 0) + 1
        content_chars = len(message.content or "")
        args_chars = sum(
            len(json.dumps(call.arguments or {})) for call in message.tool_calls
        )
        name_chars = sum(len(call.name) for call in message.tool_calls)
        entry_chars = content_chars + args_chars + name_chars
        messages_chars += entry_chars
        tool_call_count += len(message.tool_calls)
        tool_call_args_chars += args_chars + name_chars
        entry: Dict[str, Any] = {
            "index": index,
            "role": message.role,
            "content_characters": content_chars,
            "content_tokens_est": estimate_tokens(
                content_chars, chars_per_token
            ),
            "tool_call_count": len(message.tool_calls),
            "tool_call_arguments_characters": args_chars + name_chars,
        }
        if message.role == "tool":
            label = _tool_label(message.content)
            if label is not None:
                entry["tool"] = label
        messages_meta.append(entry)

    ordered = sorted(
        messages_meta,
        key=lambda entry: (
            entry["content_characters"]
            + entry["tool_call_arguments_characters"]
        ),
        reverse=True,
    )
    largest = [
        {
            "index": entry["index"],
            "role": entry["role"],
            "characters": entry["content_characters"]
            + entry["tool_call_arguments_characters"],
        }
        for entry in ordered[:3]
        if entry["content_characters"] + entry["tool_call_arguments_characters"] > 0
    ]

    tools_chars = _tool_schema_characters(request)
    response_chars = _response_format_characters(request)
    total_chars = messages_chars + tools_chars + response_chars
    budget_chars = policy.effective_input_budget_characters
    within = budget_chars is None or total_chars <= budget_chars

    rejection_reason: Optional[str] = None
    if not within:
        contract_chars = tools_chars + response_chars
        if contract_chars > (budget_chars or 0):
            rejection_reason = "tool_or_response_contract_exceeds_budget"

    record: Dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "message_count": len(request.messages),
        "role_counts": dict(sorted(role_counts.items())),
        "messages": messages_meta,
        "largest_messages": largest,
        "tool_call_count": tool_call_count,
        "tool_call_arguments_characters": tool_call_args_chars,
        "tool_definitions": {
            "count": len(request.tools),
            "characters": tools_chars,
            "tokens_est": estimate_tokens(tools_chars, chars_per_token),
        },
        "structured_output": (
            {
                "name": request.structured_output.name,
                "characters": response_chars,
                "tokens_est": estimate_tokens(response_chars, chars_per_token),
            }
            if request.structured_output is not None
            else None
        ),
        "totals": {
            "messages_characters": messages_chars,
            "messages_tokens_est": estimate_tokens(
                messages_chars, chars_per_token
            ),
            "tool_definitions_tokens_est": estimate_tokens(
                tools_chars, chars_per_token
            ),
            "structured_output_tokens_est": estimate_tokens(
                response_chars, chars_per_token
            ),
            "total_characters": total_chars,
            "total_tokens_est": estimate_tokens(total_chars, chars_per_token),
        },
        "estimation": {
            "method": ESTIMATION_METHOD,
            "chars_per_token": chars_per_token,
            "is_estimate": True,
        },
        "budget": policy.to_dict(),
        "within_budget": within,
        "disposition": (
            DISPOSITION_ACCEPTED if within else DISPOSITION_REJECTED
        ),
    }
    if rejection_reason is not None:
        record["rejection_reason"] = rejection_reason
    return record


def _envelope_overhead(label: Optional[str], original_characters: int) -> int:
    """Exact JSON size of a clip envelope EXCLUDING its excerpt.

    Measured with the same serialization the envelope itself uses, so
    the reduction loop can budget the excerpt precisely instead of
    overshooting by the envelope's own words.
    """
    probe = {
        "truncated": True,
        "tool": label,
        "original_characters": original_characters,
        "excerpt": "",
        "note": _CLIP_NOTE,
    }
    return len(json.dumps(probe, ensure_ascii=False, sort_keys=True))


def _clip_tool_message(
    message: Message,
    keep_characters: int,
    label: Optional[str] = None,
) -> Message:
    """Replace one oversized tool observation with a bounded envelope.

    The envelope is valid JSON (never a raw byte cut of the original
    payload), states the original size, and keeps a head excerpt of at
    least ``MIN_TOOL_EXCERPT_CHARS`` characters when the content is
    that long. The tool-call pairing (``tool_call_id``) is preserved
    exactly, so provider message ordering stays valid.
    """
    content = message.content or ""
    keep = max(0, keep_characters)
    excerpt = content[:keep]
    envelope = {
        "truncated": True,
        "tool": _tool_label(content) if label is None else label,
        "original_characters": len(content),
        "excerpt": excerpt,
        "note": _CLIP_NOTE,
    }
    return Message(
        role="tool",
        content=json.dumps(envelope, ensure_ascii=False, sort_keys=True),
        tool_call_id=message.tool_call_id,
    )


def enforce_request_budget(
    request: LLMRequest, policy: RequestBudgetPolicy
) -> Tuple[LLMRequest, Dict[str, Any]]:
    """H2 gate: reduce or reject an over-budget request BEFORE sending.

    Returns ``(final_request, record)``. The record's disposition:

    * ``accepted``  — at (or under) budget; ``final_request`` is the
      original request unchanged;
    * ``reduced``   — tool observations were clipped to valid-JSON
      excerpts until the TOTAL request fits; the record carries the
      before/after measurements and the clip count;
    * ``rejected``  — even after every reducible observation was
      clipped to the minimum excerpt, protected content (system/user/
      assistant messages, tool-call arguments, tool definitions,
      response-format schema) alone exceeds the budget. The caller
      must NOT send this request and must NOT retry it unchanged.

    Reduction order (deterministic): largest tool observation first,
    each clipped to whatever budget remains after everything else; the
    estimate is recomputed after every clip; non-tool messages,
    assistant tool-call turns, tool definitions, and the response
    format are never modified (they are contract, not observation).
    """
    record = build_request_budget_record(request, policy)
    if record["disposition"] == DISPOSITION_ACCEPTED:
        return request, record

    budget_chars = policy.effective_input_budget_characters
    if budget_chars is None:
        return request, record

    messages = list(request.messages)
    clip_order = sorted(
        (
            index
            for index, message in enumerate(messages)
            if message.role == "tool" and (message.content or "")
        ),
        key=lambda index: len(messages[index].content or ""),
        reverse=True,
    )
    clipped = 0
    before_chars = _request_characters(request)
    for index in clip_order:
        current_chars = _request_characters(
            LLMRequest(
                messages=messages,
                tools=request.tools,
                structured_output=request.structured_output,
            )
        )
        if current_chars <= budget_chars:
            break
        message = messages[index]
        content = message.content or ""
        # Room for THIS message's serialized content in the final
        # request (everything else is already measured). The room can
        # be negative — then only a best-effort minimum excerpt helps,
        # and only if it actually shrinks the observation.
        allowed = budget_chars - (current_chars - len(content))
        label = _tool_label(content)
        overhead = _envelope_overhead(label, len(content))
        keep = min(
            max(MIN_TOOL_EXCERPT_CHARS, allowed - overhead), len(content)
        )
        best: Optional[Message] = None
        # Exact-fit correction: the excerpt is embedded in a JSON
        # string, so quotes/backslashes inside it ESCAPE and expand
        # past the arithmetic estimate. Shrink the keep by the observed
        # overshoot until the serialized envelope fits the room (or the
        # policy excerpt floor is reached) — bounded attempts,
        # deterministic outcome.
        for _ in range(8):
            candidate = _clip_tool_message(message, keep, label=label)
            best = candidate
            size = len(candidate.content or "")
            if size <= allowed or keep <= MIN_TOOL_EXCERPT_CHARS:
                break
            keep = max(MIN_TOOL_EXCERPT_CHARS, keep - (size - allowed))
        if best is not None and len(best.content or "") < len(content):
            # Accept only a genuine shrink: a clip that would not make
            # this observation smaller (its envelope costing at least
            # as much as its content) is skipped.
            messages[index] = best
            clipped += 1

    clipped_request = LLMRequest(
        messages=messages,
        tools=list(request.tools),
        model=request.model,
        temperature=request.temperature,
        structured_output=request.structured_output,
    )
    after_chars = _request_characters(clipped_request)
    if after_chars <= budget_chars:
        record = build_request_budget_record(clipped_request, policy)
        record["disposition"] = DISPOSITION_REDUCED
        record["within_budget"] = True
        record.pop("rejection_reason", None)
        record["reduction"] = {
            "tool_messages_clipped": clipped,
            "before_characters": before_chars,
            "after_characters": after_chars,
            "before_tokens_est": estimate_tokens(
                before_chars, policy.chars_per_token
            ),
            "after_tokens_est": estimate_tokens(
                after_chars, policy.chars_per_token
            ),
            "method": "clip_tool_observations_to_excerpts",
        }
        return clipped_request, record

    # Rejected: even after every reducible observation reached the
    # excerpt floor (or had nothing to give), the request is over
    # budget. The ORIGINAL request is returned — nothing may be sent —
    # and the record describes THAT request plus what reduction
    # attempted.
    record = build_request_budget_record(request, policy)
    record["disposition"] = DISPOSITION_REJECTED
    record["within_budget"] = False
    record.setdefault(
        "rejection_reason", "protected_request_exceeds_budget"
    )
    record["reduction"] = {
        "tool_messages_clipped": clipped,
        "before_characters": before_chars,
        "after_characters": after_chars,
        "before_tokens_est": estimate_tokens(
            before_chars, policy.chars_per_token
        ),
        "after_tokens_est": estimate_tokens(
            after_chars, policy.chars_per_token
        ),
        "method": "clip_tool_observations_to_excerpts",
    }
    return request, record


class RequestBudgetExceededError(Exception):
    """A preflight-REJECTED request that must never reach the provider.

    Raised by :func:`enforce_request_budget_or_raise` for callers whose
    only safe outcome for an over-budget request is an explicit local
    failure (the planner, a composition-root CLI): they have no
    execution trace to attach the record to and no reducible content
    they are allowed to drop.

    The message and the attached ``record`` are privacy-safe — sizes,
    token ESTIMATES, the configured budget and the rejection reason
    only — so the failure can be surfaced without logging a prompt,
    repository text or credentials. ``exc.record`` is the same
    preflight record H1 produces, so the numbers are directly
    comparable with the specialist-agent diagnostics.
    """

    def __init__(self, message: str, record: Dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def enforce_request_budget_or_raise(
    request: LLMRequest,
    policy: Optional[RequestBudgetPolicy] = None,
    *,
    subject: str = "request",
) -> Tuple[LLMRequest, Dict[str, Any]]:
    """Preflight one request and REFUSE to send a rejected one.

    A thin composition over :func:`enforce_request_budget` for callers
    that cannot attach the record to a trace: the SAME
    ``RequestBudgetPolicy`` (``None`` applies the standard default —
    the observed model window minus the output reservation and safety
    margin), the SAME characters-per-token estimator, and the SAME
    accepted / reduced / rejected semantics, so there is no second
    budget implementation to drift.

    Returns ``(sendable_request, record)``:

    * ``accepted`` — the original request, byte-identical;
    * ``reduced``  — only TOOL observations were clipped to valid-JSON
      excerpts (instructions, user content, tool-call arguments, tool
      definitions and the response-format schema are never modified);
    * ``rejected`` — :class:`RequestBudgetExceededError` is raised and
      the request is never sent; it must never be retried unchanged.

    ``subject`` labels the failure (``"planner request"``, …) so a
    rejection is distinguishable from a provider-side ``LLMError``
    without inspecting request content.
    """
    effective = RequestBudgetPolicy() if policy is None else policy
    final_request, record = enforce_request_budget(request, effective)
    if record["disposition"] == DISPOSITION_REJECTED:
        totals = record.get("totals") or {}
        estimation = record.get("estimation") or {}
        budget = (record.get("budget") or {}).get(
            "effective_input_budget_tokens"
        )
        raise RequestBudgetExceededError(
            "{}: rejected by LOCAL context-budget preflight before the "
            "provider was called (total_tokens_est={}, budget_tokens_est={},"
            " method={}, reason={})".format(
                subject,
                totals.get("total_tokens_est"),
                budget,
                estimation.get("method"),
                record.get("rejection_reason"),
            ),
            record,
        )
    return final_request, record


def note_provider_context_overflow(
    record: Optional[Dict[str, Any]], error: BaseException
) -> Optional[Dict[str, Any]]:
    """H1.8: correlate a provider context-length failure with the most
    recent preflight record (a copy — the request record of the
    iteration is updated in place by the caller).

    The original exception is never swallowed or rewritten here; this
    only annotates the diagnostic record with the provider's own
    category and a bounded message excerpt.
    """
    if record is None:
        return None
    annotated = dict(record)
    message = " ".join(str(error).split())
    annotated["provider_error"] = {
        "category": "context_length_exceeded",
        "message": message[:500],
        "preflight_total_tokens_est": (
            record.get("totals") or {}
        ).get("total_tokens_est"),
        "preflight_within_budget": record.get("within_budget"),
    }
    return annotated
