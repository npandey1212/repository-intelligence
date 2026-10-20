"""Provider-neutral LLM request/response models.

Small frozen dataclasses shared by the application side (the future
Agent) and every provider adapter. Nothing in this module knows about a
specific LLM provider: no SDK imports, no provider field names, no
provider-specific parameter bags.

    Message / ToolDefinition / ToolCall / Usage / StructuredOutputSpec
    LLMRequest  →  (LLMClient)  →  LLMResponse

Tool calling is represented, never executed: a ToolCall in a response is
a *request from the model*, and deciding what to do with it belongs to
the Agent (next milestone).

Structured output (M8.9) is optional and symmetric: an
``LLMRequest.structured_output`` spec asks the model for one JSON
object matching a schema; an ``LLMResponse.structured_output`` payload
is the decoded JSON object the model returned (when any). Decoding is
provider-neutral; validating the payload against the schema is the
application's job (``llm.structured``) — never the provider adapter's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Roles supported by the abstraction (the OpenAI adapter maps these to
# its own wire format).
ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class Message:
    """One chat message.

    ``tool_calls`` is set on an *assistant* message when the model
    requested tools; ``tool_call_id`` is set on a *tool* message that
    carries a tool result back to the model. Both are optional so plain
    conversations stay one-liners.
    """

    role: str
    content: Optional[str] = None
    tool_calls: List["ToolCall"] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                "unsupported message role {!r}; expected one of {}".format(
                    self.role, ", ".join(ROLES)
                )
            )


@dataclass(frozen=True)
class ToolDefinition:
    """A tool the model may request, described with a JSON schema."""

    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A model-requested tool invocation (structured arguments)."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Minimal token accounting (any field may be unknown)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class StructuredOutputSpec:
    """Optional JSON contract for the response (M8.9).

    ``name`` is the contract's identifier (the provider uses it as the
    response-format name); ``schema`` is a JSON Schema describing the
    ONE JSON object the model should return. The spec travels on the
    request — a provider adapter may translate it (e.g. to its own
    response-format parameter), but it never interprets or validates
    the schema itself.

    ``None`` on a request means free-form text output (the established
    default — every existing request is unchanged).
    """

    name: str
    schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "structured output spec requires a non-empty name, "
                "got {!r}".format(self.name)
            )
        if not isinstance(self.schema, dict):
            raise ValueError(
                "structured output schema must be a JSON object, "
                "got {!r}".format(self.schema)
            )


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral completion request.

    ``structured_output`` (M8.9) optionally asks for the response as
    ONE JSON object matching the spec's schema; ``None`` (the default)
    keeps the established free-form text contract.  ``max_tokens``
    optionally caps the RESPONSE (output) length using the provider's
    own parameter (translated at the adapter boundary, e.g. OpenAI's
    ``max_tokens``); ``None`` keeps the provider default — the request
    contract gains no behavior of its own.
    """

    messages: List[Message] = field(default_factory=list)
    tools: List[ToolDefinition] = field(default_factory=list)
    model: Optional[str] = None
    temperature: Optional[float] = None
    structured_output: Optional[StructuredOutputSpec] = None
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral completion response.

    ``message`` is the assistant message (content and/or requested tool
    calls); ``tool_calls`` is a convenience view so the Agent can
    distinguish a normal response from a tool request with
    ``if response.tool_calls:``.

    ``structured_output`` (M8.9) is the decoded JSON object of a
    structured response — set only when the request carried a
    ``StructuredOutputSpec`` AND the response content decoded to a JSON
    object. It never holds prose, and ``None`` always means "no
    structured payload arrived" (the raw text, if any, stays in
    ``message.content``). Whether the payload matches the requested
    schema is decided later, application-side (``llm.structured``).
    """

    message: Message
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    structured_output: Optional[Dict[str, Any]] = None

    @property
    def tool_calls(self) -> List[ToolCall]:
        return list(self.message.tool_calls)
