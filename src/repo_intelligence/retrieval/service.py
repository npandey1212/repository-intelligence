"""Query-time retrieval over a built index (M18 §16, §26).

Read-only: searching loads the index, embeds the query, ranks chunks
by cosine similarity (inner product on L2-normalized vectors), and
returns evidence spans with line ranges and excerpts — nothing is
written, built, or refreshed here (spec §34: builds are always
explicit). A missing index is the caller's signal to run the build
command in the error, never an invitation to build implicitly.

Determinism: FAISS flat search over the fixed index is exact; ties
are broken by ``(file_path, line_start, chunk_id)`` after score, so
the same query against the same index always returns the same order.
Excerpts come from the chunk text stored at build time (the indexed
state), not from the working tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import embedder_for
from .errors import RAG_CORRUPT_INDEX, RAG_INVALID_ARGUMENT, RetrievalError
from .index import LoadedIndex, load_index
from .models import IndexIdentity, RetrievalResult

__all__ = [
    "DEFAULT_TOP_K",
    "EXCERPT_MAX_CHARS",
    "MAX_SCAN",
    "MAX_TOP_K",
    "RetrievalService",
    "SearchResponse",
]

DEFAULT_TOP_K = 5
MAX_TOP_K = 50
# Bounded candidate pool: enough to keep tie-breaking deterministic
# and to find path-filtered matches without unbounded memory, while
# still a small fraction of any real index.
MAX_SCAN = 200
EXCERPT_MAX_CHARS = 600


@dataclass
class SearchResponse:
    """Ranked results plus honest counts for the tool contract."""

    results: List[RetrievalResult]
    total_matches: int
    index_chunks: int
    scan_capped: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)


class RetrievalService:
    """Stateless-per-call retrieval (embedder model stays cached)."""

    def __init__(self, store_root: Optional[Path] = None) -> None:
        self.store_root = (
            Path(store_root) if store_root is not None else None
        )

    def search(
        self,
        query: str,
        identity: IndexIdentity,
        top_k: int = DEFAULT_TOP_K,
        path_filter: Optional[str] = None,
    ) -> SearchResponse:
        """Rank *identity*'s chunks for *query* (read-only)."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "query must be a non-empty string; got {!r}.".format(
                    query
                ),
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "top_k must be an integer; got {!r}.".format(top_k),
            )
        if top_k < 1 or top_k > MAX_TOP_K:
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "top_k must be between 1 and {}.".format(MAX_TOP_K),
            )
        if path_filter is not None and not isinstance(path_filter, str):
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "path_filter must be a string or None; got {!r}.".format(
                    type(path_filter).__name__
                ),
            )

        loaded = load_index(identity, store_root=self.store_root)
        if loaded.ntotal == 0:
            return SearchResponse(
                results=[], total_matches=0, index_chunks=0
            )

        embedder = embedder_for(identity.embedding_model)
        try:
            vector = embedder.encode([query.strip()])
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "embedding the query failed ({}).".format(exc),
            ) from exc
        shape = getattr(vector, "shape", None)
        if shape != (1, loaded.dimension):
            raise RetrievalError(
                RAG_CORRUPT_INDEX,
                "query vector shape {} does not match index dimension "
                "{}; the index was built with a different model."
                .format(shape, loaded.dimension),
            )

        scan_k = min(MAX_SCAN, loaded.ntotal)
        try:
            scores, positions = loaded.index.search(vector, scan_k)
        except Exception as exc:
            raise RetrievalError(
                RAG_CORRUPT_INDEX,
                "FAISS search failed over {} ({}).".format(
                    loaded.index_dir, exc
                ),
            ) from exc

        pool: List[tuple] = []
        for score, position in zip(scores[0], positions[0]):
            position = int(position)
            if position < 0 or position >= len(loaded.records):
                continue
            pool.append((float(score), position))

        if path_filter is not None:
            matched = [
                (score, position)
                for score, position in pool
                if _path_matches(
                    loaded.records[position]["file_path"], path_filter
                )
            ]
            scan_capped = len(matched) == scan_k and loaded.ntotal > scan_k
            total_matches = len(matched)
        else:
            matched = pool
            scan_capped = False
            total_matches = loaded.ntotal

        enriched = [
            _result(identity, score, loaded, position)
            for score, position in matched
        ]
        enriched.sort(
            key=lambda item: (
                -item.score,
                item.file_path,
                item.line_start,
                item.chunk_id,
            )
        )
        return SearchResponse(
            results=enriched[:top_k],
            total_matches=total_matches,
            index_chunks=loaded.ntotal,
            scan_capped=scan_capped,
        )


def _path_matches(file_path: str, path_filter: str) -> bool:
    """Prefix match on a repository-relative path (file or directory)."""
    normalized = path_filter.strip().strip("/")
    if not normalized:
        return True
    return file_path == normalized or file_path.startswith(
        normalized + "/"
    )


def _result(
    identity: IndexIdentity,
    score: float,
    loaded: LoadedIndex,
    position: int,
) -> RetrievalResult:
    record = loaded.records[position]
    content = str(record.get("content", ""))
    excerpt = content
    if len(excerpt) > EXCERPT_MAX_CHARS:
        excerpt = excerpt[:EXCERPT_MAX_CHARS] + "..."
    return RetrievalResult(
        file_path=str(record["file_path"]),
        line_start=int(record["line_start"]),
        line_end=int(record["line_end"]),
        chunk_id=str(record["chunk_id"]),
        score=score,
        language=str(record.get("language", "")),
        repository_name=identity.repository_name,
        commit_sha=identity.commit_sha,
        snapshot_id=identity.snapshot_id or str(
            record.get("snapshot_id", "")
        ),
        excerpt=excerpt,
    )
