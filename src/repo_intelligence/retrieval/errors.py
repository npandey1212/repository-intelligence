"""Structured errors for repository retrieval (M18 §16).

Every failure mode a caller can hit is a ``RetrievalError`` with a
stable machine-readable ``code`` and an actionable message — the tool
layer maps them one-to-one onto ``ToolResult.error`` so agents and
humans see the same code, never a bare stack trace. Messages for
missing dependencies and missing indexes always carry the exact
command needed to fix them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "RAG_BUILD_FAILED",
    "RAG_CORRUPT_INDEX",
    "RAG_DEPENDENCIES_UNAVAILABLE",
    "RAG_EMBEDDINGS_UNAVAILABLE",
    "RAG_FAISS_UNAVAILABLE",
    "RAG_IDENTITY_MISMATCH",
    "RAG_IDENTITY_NOT_RESOLVED",
    "RAG_INDEX_AMBIGUOUS",
    "RAG_INDEX_NOT_FOUND",
    "RAG_INVALID_ARGUMENT",
    "RAG_REPOSITORY_NOT_FOUND",
    "RAG_UNKNOWN_EMBEDDER",
    "RetrievalError",
]

# Stable error codes (tool result errors use these verbatim).
RAG_INDEX_NOT_FOUND = "RAG_INDEX_NOT_FOUND"
RAG_INDEX_AMBIGUOUS = "RAG_INDEX_AMBIGUOUS"
RAG_CORRUPT_INDEX = "RAG_INDEX_CORRUPT"
RAG_IDENTITY_MISMATCH = "RAG_IDENTITY_MISMATCH"
RAG_IDENTITY_NOT_RESOLVED = "RAG_IDENTITY_NOT_RESOLVED"
RAG_REPOSITORY_NOT_FOUND = "RAG_REPOSITORY_NOT_FOUND"
RAG_DEPENDENCIES_UNAVAILABLE = "RAG_DEPENDENCIES_UNAVAILABLE"
RAG_EMBEDDINGS_UNAVAILABLE = "RAG_EMBEDDINGS_UNAVAILABLE"
RAG_FAISS_UNAVAILABLE = "RAG_FAISS_UNAVAILABLE"
RAG_INVALID_ARGUMENT = "RAG_INVALID_ARGUMENT"
RAG_UNKNOWN_EMBEDDER = "RAG_UNKNOWN_EMBEDDER"
RAG_BUILD_FAILED = "RAG_BUILD_FAILED"


class RetrievalError(Exception):
    """A retrieval failure with a stable code and actionable message."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: Dict[str, Any] = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload
