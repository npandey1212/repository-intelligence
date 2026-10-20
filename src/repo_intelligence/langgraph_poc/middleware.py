"""Middleware adapters binding ``langchain`` ``create_agent`` to
``react_safety``.

This package is the LangGraph/LangChain POC seam: the ONE place under
``src/`` where the ``langchain`` agent framework may be imported (the
provider-import guard exempts this package only; provider SDKs remain
confined to ``llm/openai_client.py``).

Three small middleware classes adapt the EXISTING
``react_safety`` controls to the real ``langchain`` middleware hooks
(no second safety implementation):

* :class:`RequestBudgetMiddleware` — ``wrap_model_call`` runs on
  EVERY model call in the agent loop (not just the first): the
  complete outgoing request (system prompt + messages + tool
  definitions + structured-output schema, to the extent the
  project's estimator supports) is measured and enforced with
  ``agent.request_budget.enforce_request_budget_or_raise`` before
  the provider is reached. An oversized request raises
  ``RequestBudgetExceededError`` locally — never sent, never retr
  unchanged. When the gate REDUCES a request (tool observations
  clipped to bounded excerpts), the reduced messages are written
  back through ``ModelRequest.override`` so the model sees the
  bounded history, not the original.
* :class:`ObservationBoundingMiddleware` — ``wrap_tool_call``
  bounds each tool observation with ``react_safety.bound_observation``
  BEFORE the ToolMessage joins the continuing message history, so
  the NEXT model call receives the bounded excerpt (with its
  truncation marker and continuation hint), never the unbounded
  original.
* :class:`ExecutionLimitMiddleware` — adapts
  ``react_safety.ReActLoopGuard`` (configured by
  ``ReActSafetyConfig``) to the same two hooks: every model call is
  checked/recorded via ``wrap_model_call`` and every tool call via
  ``wrap_tool_call``. On exhaustion the guard's terminal status is
  surfaced as a typed :class:`ReActExecutionLimitExceeded` — a safe,
  inspectable stop, never an unbounded retry.

Middleware order matters and is set by
``langgraph_poc.graph.build_create_agent_investigator``: the budget
middleware is listed FIRST (outermost), so a budget rejection raises
before the execution guard records a model call — mirroring the
``BudgetedLLMClient.calls_reached_provider`` rule that a rejection
does NOT consume a provider-call allowance.

Token figures come from the project's established chars-per-token
RATIO estimator (``agent.request_budget``) unless a provider-reported
usage is available — they are ESTIMATES, never exact counts, and
nothing here claims a monetary cap.
"""

from __future__ import annotations

import json
from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional, Sequence

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

from ..agent.request_budget import (
    RequestBudgetExceededError,
    RequestBudgetPolicy,
    enforce_request_budget_or_raise,
)
from ..llm.models import (
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    Usage,
)
from ..react_safety import (
    ReActLoopGuard,
    ReActSafetyConfig,
    ReActTermination,
    bound_observation,
)

__all__ = [
    "ExecutionLimitMiddleware",
    "ObservationBoundingMiddleware",
    "ReActExecutionLimitExceeded",
    "RequestBudgetMiddleware",
    "llm_request_from_model_request",
]


class ReActExecutionLimitExceeded(RuntimeError):
    """A configured execution limit was reached inside the
    ``create_agent`` loop; the run terminated safely.

    ``reason`` is the ``ReActTermination`` status (string constant);
    ``snapshot`` is the guard's privacy-safe counter snapshot.
    """

    def __init__(self, reason: str, snapshot: Dict[str, Any]) -> None:
        super().__init__(
            "create_agent execution limit reached: {}".format(reason)
        )
        self.reason = reason
        self.snapshot = snapshot


# -- provider-neutral message conversion --------------------------------------


def _text(content: Any) -> str:
    """Plain-text view of a langchain message content (str or a list
    of content blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except (TypeError, ValueError):
        return str(content)


def _llm_message(message: BaseMessage) -> Message:
    """langchain message -> the project's provider-neutral ``Message``."""
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = "assistant"
    calls: List[ToolCall] = []
    for raw in getattr(message, "tool_calls", None) or []:
        if isinstance(raw, dict):
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or ""),
                    name=str(raw.get("name") or ""),
                    arguments=dict(raw.get("args") or {}),
                )
            )
        else:
            calls.append(
                ToolCall(
                    id=str(getattr(raw, "id", "") or ""),
                    name=str(getattr(raw, "name", "") or ""),
                    arguments=dict(getattr(raw, "args", None) or {}),
                )
            )
    return Message(
        role=role,
        content=_text(message.content) or None,
        tool_calls=calls,
        tool_call_id=getattr(message, "tool_call_id", None),
    )


def _langchain_message(message: Message) -> BaseMessage:
    """project ``Message`` -> the matching langchain message."""
    if message.role == "system":
        return SystemMessage(content=message.content or "")
    if message.role == "user":
        return HumanMessage(content=message.content or "")
    if message.role == "tool":
        return ToolMessage(
            content=message.content or "",
            tool_call_id=message.tool_call_id or "",
        )
    tool_calls = [
        {"id": call.id, "name": call.name, "args": dict(call.arguments or {})}
        for call in message.tool_calls
    ]
    return AIMessage(content=message.content or "", tool_calls=tool_calls)


def _tool_definitions(tools: Sequence[Any]) -> List[ToolDefinition]:
    """langchain tools -> the project's ``ToolDefinition`` records
    (name + description + JSON schema), to the extent the tool
    exposes them — the same shapes the request budget measures on
    the production path."""
    definitions: List[ToolDefinition] = []
    for item in tools or ():
        name = getattr(item, "name", None) or ""
        description = getattr(item, "description", None) or ""
        schema: Dict[str, Any] = {}
        to_json = getattr(item, "to_json_schema", None)
        if callable(to_json):
            try:
                raw = to_json()
                if isinstance(raw, dict):
                    schema = raw
            except Exception:
                schema = {}
        if not schema:
            args_schema = getattr(item, "args_schema", None)
            if isinstance(args_schema, dict):
                # The POC's StructuredTools deliberately reuse the
                # project's existing JSON-schema tool contracts.
                schema = dict(args_schema)
            else:
                model_json = getattr(args_schema, "model_json_schema", None)
                if callable(model_json):
                    try:
                        schema = model_json()
                    except Exception:
                        schema = {}
        definitions.append(
            ToolDefinition(name=name, description=description,
                           input_schema=schema)
        )
    return definitions


def _structured_schema(response_format: Any) -> Optional[Dict[str, Any]]:
    """Best-effort JSON schema of the bound response format — a
    pydantic model (or a strategy object carrying one) contributes
    its schema to the measured request; anything else is skipped
    rather than guessed."""
    candidate = response_format
    inner = getattr(candidate, "schema", None)
    if inner is not None and not isinstance(inner, dict):
        candidate = inner
    if isinstance(candidate, dict):
        return candidate
    model_json = getattr(candidate, "model_json_schema", None)
    if callable(model_json):
        try:
            return model_json()
        except Exception:
            return None
    return None


def llm_request_from_model_request(request: ModelRequest) -> LLMRequest:
    """The provider-neutral projection of ONE outgoing model request:
    system message + conversation messages + tool definitions. This
    is the message/tool half of what the budget measures — the same
    request the provider would receive, in the project's own
    vocabulary."""
    messages: List[Message] = []
    if request.system_message is not None:
        messages.append(_llm_message(request.system_message))
    for message in request.messages or ():
        messages.append(_llm_message(message))
    return LLMRequest(
        messages=messages,
        tools=_tool_definitions(request.tools or ()),
    )


def _measure_request(request: ModelRequest) -> LLMRequest:
    """Complete measured request: the message/tool projection plus,
    when available, the structured-output schema as a real
    ``StructuredOutputSpec`` (so its size is part of the budget the
    same way the production planner's spec is)."""
    from ..llm.models import StructuredOutputSpec

    base = llm_request_from_model_request(request)
    schema = _structured_schema(request.response_format)
    if schema is None:
        return base
    return LLMRequest(
        messages=base.messages,
        tools=base.tools,
        structured_output=StructuredOutputSpec(
            name="response_format", schema=schema
        ),
    )


# -- A: input/token budget protection -----------------------------------------


class RequestBudgetMiddleware(AgentMiddleware):
    """``wrap_model_call``: enforce the complete-request budget on
    EVERY model call (first AND later ReAct iterations).

    Delegates entirely to ``agent.request_budget``'s
    ``enforce_request_budget_or_raise`` with a ``RequestBudgetPolicy``
    (``None`` = the project's standard default — the global H1
    ceiling, never weakened here). Dispositions:

    * accepted — the request passes through untouched (the model
      receives the exact messages it was going to receive);
    * reduced — tool observations were clipped; the reduced messages
      are written back via ``ModelRequest.override``;
    * rejected — ``RequestBudgetExceededError`` is raised BEFORE the
      handler runs: the provider is never reached, and the request is
      never retried unchanged.
    """

    def __init__(
        self,
        policy: Optional[RequestBudgetPolicy] = None,
        *,
        subject: str = "poc investigator request",
    ) -> None:
        self.policy = policy
        self.subject = subject
        self.records: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []
        # Context-local, rather than a shared mutable ``last_record``:
        # model calls from concurrent graph invocations must not borrow
        # another invocation's estimate for cumulative-budget accounting.
        self._active_record: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
            "langgraph_poc_active_request_budget_record_{}".format(id(self)),
            default=None,
        )

    @property
    def active_record(self) -> Optional[Dict[str, Any]]:
        """The preflight record for the currently nested model call."""
        return self._active_record.get()

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        measured = _measure_request(request)
        try:
            final_request, record = enforce_request_budget_or_raise(
                measured, self.policy, subject=self.subject
            )
        except RequestBudgetExceededError as exc:
            self.records.append(exc.record)
            self.rejected.append(exc.record)
            raise
        self.records.append(record)
        if record.get("disposition") != "accepted":
            updated = [_langchain_message(m) for m in final_request.messages]
            request = request.override(messages=updated)

        # The execution middleware is nested inside this handler. Keep
        # the exact preflight estimate available until that middleware
        # records the response, then reset it even if the provider fails.
        token: Token = self._active_record.set(record)
        try:
            return handler(request)
        finally:
            self._active_record.reset(token)


# -- C (observation half): bounded observations ------------------------------


class ObservationBoundingMiddleware(AgentMiddleware):
    """``wrap_tool_call``: bound ONE tool observation with
    ``react_safety.bound_observation`` before it joins the
    continuing message history.

    The returned ToolMessage carries the bounded text (whole-line
    truncation + the marker/continuation hint); the unbounded
    original is never what the next model call receives. Only
    string content is bounded — the project's own tools serialize
    to JSON strings, so that is the path exercised; a non-string
    content block list passes through unchanged (documented, and
    never produced by the project tools).
    """

    def __init__(self, max_chars: Optional[int] = None) -> None:
        from ..react_safety import MAX_TOOL_RESPONSE_CHARS

        self.max_chars = (
            MAX_TOOL_RESPONSE_CHARS if max_chars is None else max_chars
        )
        self.records: List[Dict[str, Any]] = []

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = handler(request)
        if not isinstance(result, ToolMessage):
            return result
        text = result.content if isinstance(result.content, str) else None
        if text is None:
            return result
        label = getattr(request.tool_call, "name", None)
        bounded, record = bound_observation(
            text, self.max_chars, label=label
        )
        self.records.append(record)
        if bounded == text:
            return result
        return result.model_copy(update={"content": bounded})


# -- C (loop half): model/tool/cumulative execution limits -------------------


def _input_tokens(message: BaseMessage) -> Optional[int]:
    usage = getattr(message, "usage_metadata", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("input_tokens")
    return getattr(usage, "input_tokens", None)


def _llm_response(response: Any) -> LLMResponse:
    """ModelResponse/AIMessage -> the project's ``LLMResponse`` for
    guard accounting (reported input tokens when the model carries
    usage metadata; otherwise ``None`` so the guard falls back to
    estimates)."""
    message: Any = response
    if isinstance(response, ModelResponse):
        message = response.result[-1] if response.result else None
    if message is None:
        message = AIMessage(content="")
    tokens = _input_tokens(message)
    return LLMResponse(
        message=Message(role="assistant", content=_text(message.content)),
        usage=Usage(
            input_tokens=tokens,
            output_tokens=None,
            total_tokens=None,
        ),
    )


class ExecutionLimitMiddleware(AgentMiddleware):
    """Adapt ``react_safety.ReActLoopGuard`` to the agent hooks.

    * ``wrap_model_call`` — ``allow_iteration`` and
      ``allow_model_call`` BEFORE the handler; ``record_model_response``
      AFTER (reported input tokens when the model provides usage
      metadata, else the budget estimate). A
      denied call raises :class:`ReActExecutionLimitExceeded` with
      the guard's terminal status (iteration, model-call, or
      cumulative-budget exhaustion).
    * ``wrap_tool_call`` — ``allow_tool_call`` BEFORE the handler;
      ``record_tool_call`` AFTER (a tool attempt that raises still
      consumed capacity — recorded in ``finally``). After the tool ceiling,
      exactly one finalization model turn is allowed with investigation
      tools removed; any further tool attempt is denied.

    ``REQUEST_BUDGET_EXCEEDED`` (the guard's budget-rejection
    status) is set by :class:`RequestBudgetMiddleware`'s callers via
    ``note_budget_rejection`` when a rejection should terminate the
    loop — this middleware never re-checks the request budget
    itself (one budget implementation, one authority).
    """

    def __init__(
        self,
        config: Optional[ReActSafetyConfig] = None,
        *,
        request_budget: Optional[RequestBudgetMiddleware] = None,
        investigation_tool_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.guard = ReActLoopGuard(config)
        self.request_budget = request_budget
        self.investigation_tool_names = set(investigation_tool_names or ())
        self._finalization_used = False

    def reset_for_invocation(self) -> None:
        """Start a fresh bounded budget for one planned investigation step.

        A multi-step orchestrator reuses the same compiled agent and
        middleware objects. Reset the execution guard between independent
        P1/P2/P3 tasks so one task cannot consume another task's loop budget.
        Request/observation audit records intentionally remain cumulative.
        """
        self.guard = ReActLoopGuard(self.guard.config)
        self._finalization_used = False

    @staticmethod
    def _tool_name(tool: Any) -> Optional[str]:
        if isinstance(tool, dict):
            name = tool.get("name")
            if name is None and isinstance(tool.get("function"), dict):
                name = tool["function"].get("name")
            return str(name) if name is not None else None
        name = getattr(tool, "name", None)
        return str(name) if name is not None else None

    def _without_investigation_tools(self, request: ModelRequest) -> ModelRequest:
        """Keep structured-output tools but disable repository tools.

        ``create_agent(response_format=...)`` may expose its schema as a
        tool. We remove only the explicitly configured investigation tools,
        so the final structured report can still be emitted.
        """
        tools = getattr(request, "tools", None)
        if tools is None or not self.investigation_tool_names:
            return request
        remaining = [
            tool for tool in tools
            if self._tool_name(tool) not in self.investigation_tool_names
        ]
        return request.override(tools=remaining)

    def note_budget_rejection(
        self, record: Dict[str, Any]
    ) -> None:
        """Terminate the loop after a budget rejection observed by
        the budget middleware (no retry, no further model calls)."""
        self.guard.record_budget_rejection(record)

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        # Once an investigation budget is exhausted, allow exactly one
        # synthesis turn with repository tools removed. This is not another
        # investigation iteration: it is a bounded finalization opportunity.
        finalization_reason: Optional[str] = None
        if (
            self.guard.tool_calls >= self.guard.config.max_tool_calls
            or self.guard.iterations >= self.guard.config.max_iterations
        ):
            if self._finalization_used:
                raise ReActExecutionLimitExceeded(
                    self.guard.terminal_status()
                    or ReActTermination.TOOL_CALL_LIMIT_REACHED,
                    self.guard.snapshot(),
                )
            if self.guard.model_calls >= self.guard.config.max_model_calls:
                self.guard.stop(ReActTermination.MODEL_CALL_LIMIT_REACHED)
                raise ReActExecutionLimitExceeded(
                    ReActTermination.MODEL_CALL_LIMIT_REACHED,
                    self.guard.snapshot(),
                )
            ceiling = self.guard.config.max_cumulative_input_tokens
            if (
                ceiling is not None
                and self.guard.cumulative_input_tokens >= ceiling
            ):
                self.guard.stop(ReActTermination.CUMULATIVE_BUDGET_EXCEEDED)
                raise ReActExecutionLimitExceeded(
                    ReActTermination.CUMULATIVE_BUDGET_EXCEEDED,
                    self.guard.snapshot(),
                )
            if self.guard.tool_calls >= self.guard.config.max_tool_calls:
                finalization_reason = ReActTermination.TOOL_CALL_LIMIT_REACHED
            else:
                finalization_reason = ReActTermination.ITERATION_LIMIT_REACHED
            self._finalization_used = True
            request = self._without_investigation_tools(request)
        else:
            if not self.guard.allow_iteration():
                raise ReActExecutionLimitExceeded(
                    self.guard.terminal_status()
                    or ReActTermination.ITERATION_LIMIT_REACHED,
                    self.guard.snapshot(),
                )
            self.guard.begin_iteration()
            if not self.guard.allow_model_call():
                raise ReActExecutionLimitExceeded(
                    self.guard.terminal_status()
                    or ReActTermination.MODEL_CALL_LIMIT_REACHED,
                    self.guard.snapshot(),
                )

        budget_record = (
            self.request_budget.active_record
            if self.request_budget is not None
            else None
        )
        # A cumulative ceiling must be checked against the CURRENT
        # request estimate before the provider is called. Per-request
        # budget enforcement alone cannot cap the sum of several valid
        # requests. Provider-reported usage is only available afterwards;
        # in that case the guard also records it after the response.
        ceiling = self.guard.config.max_cumulative_input_tokens
        estimated = None
        if budget_record is not None:
            estimated = (budget_record.get("totals") or {}).get(
                "total_tokens_est"
            )
        if (
            ceiling is not None
            and estimated is not None
            and self.guard.cumulative_input_tokens + estimated > ceiling
        ):
            self.guard.stop(ReActTermination.CUMULATIVE_BUDGET_EXCEEDED)
            raise ReActExecutionLimitExceeded(
                ReActTermination.CUMULATIVE_BUDGET_EXCEEDED,
                self.guard.snapshot(),
            )

        response = handler(request)
        self.guard.record_model_response(
            _llm_response(response), budget_record=budget_record
        )
        if finalization_reason is not None:
            # Preserve the reason the investigation stopped while allowing
            # create_agent to consume the structured final response.
            self.guard.stop(finalization_reason)
        return response

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        if not self.guard.allow_tool_call():
            raise ReActExecutionLimitExceeded(
                self.guard.terminal_status()
                or ReActTermination.TOOL_CALL_LIMIT_REACHED,
                self.guard.snapshot(),
            )
        try:
            return handler(request)
        finally:
            self.guard.record_tool_call()
