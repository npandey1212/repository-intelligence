"""Application-facing LLM abstraction: ``LLMClient`` and ``LLMError``.

This is the seam every piece of application/Agent code depends on:

    Application / Agent
            |
            v
        LLMClient          ← this module (provider-neutral)
            |
            v
    OpenAILLMClient        ← llm/openai_client.py (provider-specific)
            |
            v
        provider API

Dependency rule: application code imports ``LLMClient``/``LLMError``
(and the models), never a provider SDK and never a concrete client. The
composition root (e.g. ``llm/__main__.py``) is the only place that
chooses the concrete implementation.

``LLMClient`` is a structural ``Protocol`` (typing_extensions: Python 3.7
has no ``typing.Protocol``): any object with a compatible ``complete``
method satisfies it, so fakes and future implementations need no base
class and no registration.

Model inference only — clients never execute tools, never loop, and
never keep agent state.
"""

from __future__ import annotations

from typing_extensions import Protocol, runtime_checkable

from .models import LLMRequest, LLMResponse

__all__ = ["LLMClient", "LLMError", "is_context_length_error"]


class LLMError(Exception):
    """Provider-neutral failure raised by any LLM client implementation.

    Concrete clients translate their provider's exceptions into this
    type at the provider boundary, so application code only ever deals
    with ``LLMError``.
    """


# Provider vocabulary for "the request does not fit the model's
# context window" — matched case-insensitively inside the translated
# message, so one classifier covers the providers seen so far without
# this layer ever importing an SDK or hard-coding a window size.
CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded",
    "context length",
    "maximum context length",
    "reduce the length of the messages",
    "too many tokens",
)


def is_context_length_error(error: BaseException) -> bool:
    """True when *error* reports CONTEXT EXHAUSTION, not a transient
    provider failure.

    The distinction is execution policy, not provider trivia: a
    context-length failure is an execution boundary (the SAME oversized
    request will fail again), so callers must not retry it blindly —
    they finalize from the state they already have. Rate limits,
    timeouts and 5xx failures stay ordinary retryable ``LLMError``s.
    """
    if not isinstance(error, LLMError):
        return False
    text = str(error).lower()
    return any(marker in text for marker in CONTEXT_LENGTH_MARKERS)


@runtime_checkable
class LLMClient(Protocol):
    """Provider-neutral completion contract.

    The application gives the client a provider-neutral ``LLMRequest``
    and receives a provider-neutral ``LLMResponse``; provider SDKs,
    credentials, wire formats, and exceptions never appear on this
    interface.
    """

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...
