"""LLM abstraction layer — provider-neutral seam for the future Agent.

Public, provider-neutral API (safe for application/Agent code to
import):

    LLMClient            the completion Protocol (llm.client)
    LLMError             provider-neutral failure (llm.client)
    is_context_length_error  execution boundary vs transient failure
    LLMRequest           request models (llm.models)
    LLMResponse
    Message / ToolDefinition / ToolCall / Usage
    StructuredOutputSpec the optional JSON contract (llm.models, M8.9)

Application-side structured output (M8.9, llm.structured):

    spec_for_model               Pydantic model → StructuredOutputSpec
    normalize_structured_payload envelope/schema wrapper one level
                                 deeper → the level that validates
                                 (unchanged payload when nothing does)
    validate_structured          decoded payload → typed result or
                                 StructuredOutputValidationError

Concrete provider implementations are imported explicitly by the
composition root, never re-exported here, so this package stays
provider-neutral:

    from repo_intelligence.llm.openai_client import OpenAILLMClient

Configuration lives in llm.config (``load_env()`` reads ``.env``).
Standalone demonstration (no Agent): ``python -m repo_intelligence.llm``.
"""

from __future__ import annotations

from .client import LLMClient, LLMError, is_context_length_error
from .models import (
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    ToolCall,
    ToolDefinition,
    Usage,
)
from .structured import (
    StructuredOutputValidationError,
    normalize_structured_payload,
    spec_for_model,
    validate_structured,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "StructuredOutputSpec",
    "StructuredOutputValidationError",
    "ToolCall",
    "ToolDefinition",
    "Usage",
    "is_context_length_error",
    "normalize_structured_payload",
    "spec_for_model",
    "validate_structured",
]
