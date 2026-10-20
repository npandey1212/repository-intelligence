"""Embedders for repository retrieval (M18 §16).

An embedder turns text into a fixed-width, L2-normalized ``float32``
vector; indexes are keyed by the embedder's *identity string* so two
different models never share a vector space:

* ``sentence-transformers://<model>`` — the production default,
  ``sentence-transformers://BAAI/bge-small-en-v1.5`` (384 dimensions,
  CPU). The heavy stack (torch / sentence-transformers) is imported
  only when this embedder is actually used, so the rest of the
  application (and the whole test suite) works without the RAG extras
  installed — a missing stack raises ``RAG_EMBEDDINGS_UNAVAILABLE``
  with the exact install command, never a silent downgrade.
* ``hashing://v1/<dim>`` — a deterministic, dependency-light built-in
  (word + bigram features, SHA-1 bucketing, signed accumulation). It
  exists for tests, fixtures, and offline demos; it is only ever used
  when explicitly selected, never as an automatic fallback.

Everything here is stdlib-only at import time; ``numpy`` is imported
inside the encode path (it ships with the RAG extras anyway).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from ..net import apply_insecure_ssl
from .errors import (
    RAG_EMBEDDINGS_UNAVAILABLE,
    RAG_INVALID_ARGUMENT,
    RAG_UNKNOWN_EMBEDDER,
    RetrievalError,
)

__all__ = [
    "DEFAULT_EMBEDDING_IDENTITY",
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "embedder_for",
    "reset_sentence_transformer_cache",
]

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_IDENTITY = (
    "sentence-transformers://" + DEFAULT_SENTENCE_TRANSFORMER_MODEL
)

_ST_PREFIX = "sentence-transformers://"
_HASH_PREFIX = "hashing://v1"
_DEFAULT_HASH_DIMENSION = 256

# Module-level model cache: loading a transformer once per process,
# not once per query (sentence-transformers caches weights on disk;
# this avoids re-instantiating the wrapper object every call).
_ST_CACHE: Dict[str, Any] = {}

_WORD_RE = re.compile(r"[a-z0-9_]+")


def reset_sentence_transformer_cache() -> None:
    """Drop cached models (used by tests that must observe reloads)."""
    _ST_CACHE.clear()


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no ML dependencies).

    Stable across processes and platforms: features are bucketed with
    SHA-1 (never Python's randomized ``hash()``), signs come from a
    second digest, and the vector is L2-normalized. Vectors are
    meaningful only within one ``hashing://v1/<dim>`` identity.
    """

    def __init__(self, dimension: int = _DEFAULT_HASH_DIMENSION) -> None:
        if dimension < 8 or dimension > 8192:
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "hashing embedder dimension must be between 8 and "
                "8192; got {}.".format(dimension),
            )
        self.dimension = int(dimension)
        self.name = "{}/{}".format(_HASH_PREFIX, self.dimension)

    def encode(self, texts: List[str]) -> Any:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - env without numpy
            raise RetrievalError(
                RAG_EMBEDDINGS_UNAVAILABLE,
                "numpy is required to embed text ({}). Install the "
                "project dependencies first.".format(exc),
            ) from exc
        matrix = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for feature, weight in _features(text or ""):
                bucket, sign = _bucket(feature, self.dimension)
                matrix[row, bucket] += sign * weight
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (matrix / norms).astype("float32")

    def encode_one(self, text: str) -> Any:
        return self.encode([text])[0]


def _features(text: str) -> List[tuple]:
    """Words + word bigrams with a mild positional weight."""
    words = _WORD_RE.findall(text.lower())
    features: List[tuple] = []
    for word in words:
        features.append((word, 1.0))
    for left, right in zip(words, words[1:]):
        features.append((left + "__" + right, 0.5))
    return features


def _bucket(feature: str, dimension: int):
    digest = hashlib.sha1(feature.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % dimension
    sign = 1.0 if digest[8] % 2 == 0 else -1.0
    return bucket, sign


class SentenceTransformerEmbedder:
    """Production embedder over ``sentence-transformers`` models.

    The model is loaded lazily on first encode (or first ``dimension``
    access) and then cached for the process. Query and document use
    the same symmetric encoding — this model is used without an
    asymmetric query prefix, and the choice is recorded in the index
    manifest with the identity string.
    """

    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        if not model_name:
            raise RetrievalError(
                RAG_INVALID_ARGUMENT,
                "sentence-transformers model name must not be empty.",
            )
        self.model_name = model_name
        self.name = _ST_PREFIX + model_name
        self._model: Optional[Any] = None
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    def _load(self) -> None:
        # before importing sentence_transformers: hub clients may be
        # constructed at import time (REPO_INTELLIGENCE_INSECURE_SSL
        # corporate-proxy bypass, see repo_intelligence.net)
        apply_insecure_ssl()
        if self._model is not None:
            return
        cached = _ST_CACHE.get(self.model_name)
        if cached is not None:
            self._model = cached["model"]
            self._dimension = int(cached["dimension"])
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrievalError(
                RAG_EMBEDDINGS_UNAVAILABLE,
                "sentence-transformers is not installed, so the default "
                "semantic embedder is unavailable. Install the RAG "
                "extras with: pip install '.[rag]'  (original error: "
                "{}). No fallback embedder is used automatically — an "
                "index must be built with the model it will be "
                "queried with.".format(exc),
            ) from exc
        try:
            model = SentenceTransformer(self.model_name)
            probe = model.encode(
                ["dimension probe"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # model fetch/decode failures
            raise RetrievalError(
                RAG_EMBEDDINGS_UNAVAILABLE,
                "failed to load embedding model {!r} ({}). Check "
                "network access to the model hub or pre-cache the "
                "model, then retry.".format(self.model_name, exc),
            ) from exc
        dimension = int(len(probe[0]))
        _ST_CACHE[self.model_name] = {"model": model, "dimension": dimension}
        self._model = model
        self._dimension = dimension

    def encode(self, texts: List[str]) -> Any:
        if not texts:
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover
                raise RetrievalError(
                    RAG_EMBEDDINGS_UNAVAILABLE,
                    "numpy is required to embed text ({}).".format(exc),
                ) from exc
            return np.zeros((0, self.dimension), dtype="float32")
        self._load()
        assert self._model is not None
        try:
            matrix = self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RetrievalError(
                RAG_EMBEDDINGS_UNAVAILABLE,
                "embedding model {!r} failed while encoding text ({})."
                .format(self.model_name, exc),
            ) from exc
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RetrievalError(
                RAG_EMBEDDINGS_UNAVAILABLE,
                "numpy is required to embed text ({}).".format(exc),
            ) from exc
        array = np.asarray(matrix, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def encode_one(self, text: str) -> Any:
        return self.encode([text])[0]


def embedder_for(identity: str) -> Any:
    """Construct the embedder registered for *identity*.

    Unknown schemes fail with ``RAG_UNKNOWN_EMBEDDER`` — there is no
    implicit default, because an index built with one model and
    queried with another is silently wrong, not "best effort".
    """
    if not isinstance(identity, str) or not identity:
        raise RetrievalError(
            RAG_INVALID_ARGUMENT,
            "embedding identity must be a non-empty string; got {!r}."
            .format(identity),
        )
    if identity.startswith(_ST_PREFIX):
        return SentenceTransformerEmbedder(identity[len(_ST_PREFIX):])
    if identity == _HASH_PREFIX or identity.startswith(_HASH_PREFIX + "/"):
        remainder = identity[len(_HASH_PREFIX):].lstrip("/")
        if not remainder:
            return HashingEmbedder(_DEFAULT_HASH_DIMENSION)
        try:
            dimension = int(remainder)
        except ValueError:
            raise RetrievalError(
                RAG_UNKNOWN_EMBEDDER,
                "invalid hashing embedder identity {!r} (expected "
                "hashing://v1/<integer dimension>).".format(identity),
            )
        return HashingEmbedder(dimension)
    raise RetrievalError(
        RAG_UNKNOWN_EMBEDDER,
        "unknown embedding identity {!r}; supported schemes: "
        "{}<model> and {}<dimension>.".format(
            identity, _ST_PREFIX, _HASH_PREFIX + "/"
        ),
    )
