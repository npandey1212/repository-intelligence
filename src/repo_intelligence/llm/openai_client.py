"""Concrete OpenAI implementation of :class:`LLMClient`.

**This is the only module in the project that may touch the OpenAI
SDK.** It translates at the provider boundary in both directions:

    LLMRequest   →  OpenAI chat-completions payload
    OpenAI response / exception  →  LLMResponse / LLMError

Structured output (M8.9) crosses the boundary here too — and only
here: an ``LLMRequest.structured_output`` spec becomes OpenAI's
``response_format`` parameter, and response content that arrived as a
JSON object is decoded into ``LLMResponse.structured_output`` (raw
text stays in ``message.content``; schema validation stays
application-side, ``llm.structured``).

The SDK is imported lazily (only when a real provider client is
constructed) so the rest of the project — including all unit tests —
works without the SDK installed. Tests inject a fake provider client
through ``provider_client=`` and exercise the same translation code.

Model inference only: this client sends the request and normalizes the
response. It does not execute tools, loop, keep state, or retry.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
import httpx

from .client import LLMError
from .models import (
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)

# Default model id; replaceable per client (constructor) or per request
# (LLMRequest.model). Not routed or looked up anywhere else.
DEFAULT_MODEL = "gpt-4o-mini"

__all__ = ["DEFAULT_MODEL", "OpenAILLMClient"]


class OpenAILLMClient:
    """``LLMClient`` backed by the OpenAI chat-completions API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        provider_client: Optional[Any] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """*model* is the default model id; *api_key* falls back to the
        ``OPENAI_API_KEY`` environment variable (load ``.env`` first —
        see :func:`llm.config.load_env`). *provider_client* injects a
        pre-built provider SDK client (tests use a fake; no SDK needed)
        and makes *max_retries* irrelevant (the caller owns that
        client).  *max_retries* forwards the SDK's own retry knob when
        a real client is constructed: ``None`` keeps the SDK default;
        ``0`` disables SDK-level retries (cost-critical live runs must
        not turn one permitted call into several HTTP attempts).
        """
        self._model = model
        if provider_client is not None:
            self._provider = provider_client
            return
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError(
                "OPENAI_API_KEY is not configured. Set it in .env (see "
                ".env.example) or in the environment before constructing "
                "OpenAILLMClient."
            )
        try:
            from openai import OpenAI  # provider SDK — isolated here
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is required for OpenAILLMClient "
                "(install the 'llm' extra: pip install 'repository-"
                "intelligence[llm]')."
            ) from exc
        options: Dict[str, Any] = {}
        if max_retries is not None:
            options["max_retries"] = max_retries
        self._provider = OpenAI(
            api_key=key,
            http_client=httpx.Client(verify=False),
            **options,
        )

    # -- request translation ------------------------------------------------

    def _to_provider_payload(self, request: LLMRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model if request.model is not None else self._model,
            "messages": [
                self._to_provider_message(message) for message in request.messages
            ],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            # Output cap at the provider's own parameter; None keeps
            # the provider default (translation only, no policy).
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
        if request.structured_output is not None:
            # M8.9: the provider-neutral spec becomes the provider's
            # response-format parameter. Translation only — the schema
            # is sent as given, never interpreted or rewritten here.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.structured_output.name,
                    "schema": request.structured_output.schema,
                },
            }
        return payload

    def _to_provider_message(self, message: Message) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        return payload

    # -- response translation -----------------------------------------------

    @staticmethod
    def _to_tool_call(raw_call: Any) -> ToolCall:
        function = getattr(raw_call, "function", None)
        name = getattr(function, "name", None)
        if not name:
            raise LLMError("Provider returned a tool call without a name.")
        raw_arguments = getattr(function, "arguments", None)
        if raw_arguments is None or (
            isinstance(raw_arguments, str) and not raw_arguments.strip()
        ):
            arguments: Dict[str, Any] = {}
        elif isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except ValueError as exc:
                raise LLMError(
                    "Provider returned invalid JSON arguments for tool "
                    "{!r}: {!r}".format(name, raw_arguments)
                ) from exc
        elif isinstance(raw_arguments, dict):
            arguments = dict(raw_arguments)
        else:
            raise LLMError(
                "Provider returned unsupported arguments for tool "
                "{!r}: {!r}".format(name, raw_arguments)
            )
        if not isinstance(arguments, dict):
            raise LLMError(
                "Provider arguments for tool {!r} did not decode to a JSON "
                "object: {!r}".format(name, raw_arguments)
            )
        return ToolCall(
            id=getattr(raw_call, "id", "") or "",
            name=name,
            arguments=arguments,
        )

    @staticmethod
    def _to_finish_reason(raw_reason: Any) -> Optional[str]:
        if raw_reason is None:
            return None
        return str(raw_reason).lower()

    @staticmethod
    def _to_usage(raw_usage: Any) -> Optional[Usage]:
        if raw_usage is None:
            return None
        return Usage(
            input_tokens=getattr(raw_usage, "prompt_tokens", None),
            output_tokens=getattr(raw_usage, "completion_tokens", None),
            total_tokens=getattr(raw_usage, "total_tokens", None),
        )

    @staticmethod
    def _to_structured_payload(content: Any) -> Optional[Dict[str, Any]]:
        """Response content → decoded JSON object, when the request
        asked for structured output (M8.9).

        Tolerant by design: content that is missing, empty, or not a
        JSON object yields ``None`` (``LLMResponse.structured_output``
        stays unset and the raw text remains in ``message.content``).
        Schema mismatches are NOT detected here — application-side
        validation reports them explicitly.
        """
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            parsed = json.loads(content)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _from_provider_response(
        self, raw: Any, request: Optional[LLMRequest] = None
    ) -> LLMResponse:
        choices = getattr(raw, "choices", None) or []
        if not choices:
            raise LLMError("Provider response contained no choices.")
        choice = choices[0]
        provider_message = getattr(choice, "message", None)
        if provider_message is None:
            raise LLMError("Provider response choice contained no message.")
        tool_calls = [
            self._to_tool_call(raw_call)
            for raw_call in (getattr(provider_message, "tool_calls", None) or [])
        ]
        content = getattr(provider_message, "content", None)
        structured: Optional[Dict[str, Any]] = None
        if request is not None and request.structured_output is not None:
            structured = self._to_structured_payload(content)
        message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )
        return LLMResponse(
            message=message,
            finish_reason=self._to_finish_reason(
                getattr(choice, "finish_reason", None)
            ),
            usage=self._to_usage(getattr(raw, "usage", None)),
            structured_output=structured,
        )

    # -- LLMClient contract -------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a provider-neutral request; return a provider-neutral
        response. Provider failures surface only as ``LLMError``."""
        payload = self._to_provider_payload(request)
        try:
            raw = self._provider.chat.completions.create(**payload)
        except LLMError:
            raise
        except Exception as exc:  # provider boundary: everything → LLMError
            raise LLMError(
                "LLM request failed: {}".format(exc)
            ) from exc
        return self._from_provider_response(raw, request)
