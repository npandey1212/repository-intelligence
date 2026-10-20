"""Data model for repository retrieval indexes (M18).

Pure standard library — importable without the RAG extras installed.

One index = one *identity*: ``(repository_name, commit_sha,
embedding_model)``. The snapshot id and repository URL ride along for
provenance but never key the storage: the same commit of the same
repository is the same content regardless of which run pinned it, and
two embedding models over one commit must never share files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = [
    "BuildReport",
    "Chunk",
    "IndexIdentity",
    "RetrievalResult",
    "store_key",
]

# Directory-name safety: only these characters survive unmodified.
_KEY_SAFE = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789._-"
)


def store_key(value: str) -> str:
    """Stable, filesystem-safe directory token for *value*.

    Safe characters pass through; everything else becomes ``-``; when
    anything changed, a short digest of the original is appended so
    distinct inputs never collide after sanitizing (``a/b`` vs
    ``a_b`` stay distinct directories).
    """
    text = value or ""
    sanitized = "".join(
        ch if ch in _KEY_SAFE else "-" for ch in text
    )
    sanitized = _collapse_dashes(sanitized).strip("-.")
    if not sanitized:
        sanitized = "key"
    if sanitized != text:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        sanitized = "{}-{}".format(sanitized, digest)
    return sanitized


def _collapse_dashes(text: str) -> str:
    out = []
    previous_dash = False
    for ch in text:
        if ch == "-":
            if not previous_dash:
                out.append(ch)
            previous_dash = True
        else:
            out.append(ch)
            previous_dash = False
    return "".join(out)


@dataclass(frozen=True)
class IndexIdentity:
    """What an index belongs to — never inferred from file contents."""

    repository_name: str
    commit_sha: str
    embedding_model: str
    repository_url: str = ""
    snapshot_id: str = ""

    def repository_key(self) -> str:
        return store_key(self.repository_name)

    def embedding_key(self) -> str:
        return store_key(self.embedding_model)

    def logical_key(self) -> str:
        return "|".join(
            (
                self.repository_name,
                self.commit_sha,
                self.embedding_model,
            )
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "repository_name": self.repository_name,
            "repository_url": self.repository_url,
            "commit_sha": self.commit_sha,
            "snapshot_id": self.snapshot_id,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexIdentity":
        return cls(
            repository_name=str(data.get("repository_name", "")),
            commit_sha=str(data.get("commit_sha", "")),
            embedding_model=str(data.get("embedding_model", "")),
            repository_url=str(data.get("repository_url", "")),
            snapshot_id=str(data.get("snapshot_id", "")),
        )


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of one file, at index-build time."""

    file_path: str
    line_start: int
    line_end: int
    content: str
    language: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            self.content.encode("utf-8")
        ).hexdigest()

    @property
    def chunk_id(self) -> str:
        basis = "{}:{}:{}:{}".format(
            self.file_path,
            self.line_start,
            self.line_end,
            self.content_hash,
        )
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked answer: where the relevant evidence lives."""

    file_path: str
    line_start: int
    line_end: int
    chunk_id: str
    score: float
    language: str
    repository_name: str
    commit_sha: str
    snapshot_id: str
    excerpt: str

    def to_dict(self) -> Dict[str, Any]:
        # Field names follow the M18 evidence contract (§26):
        # similarity_score + repository_name; ``path`` is the
        # established alias every evidence consumer reads.
        return {
            "file_path": self.file_path,
            "path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_id": self.chunk_id,
            "similarity_score": round(float(self.score), 6),
            "language": self.language,
            "repository_name": self.repository_name,
            "commit_sha": self.commit_sha,
            "snapshot_id": self.snapshot_id,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class BuildReport:
    """What ``build_index`` did — written to JSON by the CLI."""

    identity: IndexIdentity
    index_dir: str
    files_indexed: int
    chunks_indexed: int
    reused: bool
    embedder: str
    dimension: int
    created_at: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "identity": self.identity.to_dict(),
            "index_dir": self.index_dir,
            "files_indexed": self.files_indexed,
            "chunks_indexed": self.chunks_indexed,
            "reused": self.reused,
            "embedder": self.embedder,
            "dimension": self.dimension,
        }
        if self.created_at:
            payload["created_at"] = self.created_at
        if self.notes:
            payload["notes"] = dict(self.notes)
        return payload
