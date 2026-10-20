"""Repository retrieval (M18): RAG search over pinned checkouts.

Public surface::

    from repo_intelligence.retrieval import (
        BuildReport, HashingEmbedder, IndexIdentity, RetrievalService,
        RetrievalError, build_index, embedder_for, index_run,
        load_index, resolve_index_identity,
    )

Everything above is importable without the RAG extras (faiss,
torch, sentence-transformers) installed; those are needed only when
an index is actually built or searched, and their absence surfaces as
a structured ``RetrievalError`` with the install command — never as
an ImportError from a module import and never as a silent fallback
to a different embedder.
"""

from __future__ import annotations

from .chunking import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_CHUNK_OVERLAP,
    chunk_file,
    chunk_repository,
)
from .embeddings import (
    DEFAULT_EMBEDDING_IDENTITY,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    embedder_for,
)
from .errors import RetrievalError
from .identity import identity_for_snapshot, resolve_index_identity
from .index import (
    LoadedIndex,
    active_index_identity,
    build_index,
    delete_index,
    index_run,
    load_index,
)
from .models import BuildReport, Chunk, IndexIdentity, RetrievalResult
from .service import DEFAULT_TOP_K, RetrievalService, SearchResponse

__all__ = [
    "BuildReport",
    "Chunk",
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_EMBEDDING_IDENTITY",
    "DEFAULT_TOP_K",
    "HashingEmbedder",
    "IndexIdentity",
    "LoadedIndex",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalService",
    "SearchResponse",
    "SentenceTransformerEmbedder",
    "active_index_identity",
    "build_index",
    "chunk_file",
    "chunk_repository",
    "delete_index",
    "embedder_for",
    "identity_for_snapshot",
    "index_run",
    "load_index",
    "resolve_index_identity",
]
