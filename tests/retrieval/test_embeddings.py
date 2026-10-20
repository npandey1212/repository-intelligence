"""Spec §33 B — embedder determinism and identity routing.

``hashing://v1/<dim>`` is the offline, dependency-free embedder used
by every test and by the deterministic demo path: SHA-1 bucketing
(never Python's randomized ``hash()``), signed accumulation, L2
normalization. SentenceTransformer construction is checked without
loading a model (loading happens lazily on first encode).
"""

from __future__ import annotations

import hashlib
import sys

import pytest

from repo_intelligence.retrieval.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    embedder_for,
)
from repo_intelligence.retrieval.errors import RetrievalError

TEXTS = [
    "How do the agents, tools, and MCP servers interact?",
    "the agent uses the tool through this entry point",
    "completely unrelated gardening calendar placeholder",
]


def test_hashing_embedder_name_and_dimension():
    embedder = HashingEmbedder(128)
    assert embedder.name == "hashing://v1/128"
    assert embedder.dimension == 128
    default = HashingEmbedder()
    assert default.name == "hashing://v1/256"
    assert default.dimension == 256


def test_encode_shape_and_dtype():
    import numpy as np

    matrix = HashingEmbedder(64).encode(TEXTS)
    assert matrix.shape == (len(TEXTS), 64)
    assert matrix.dtype == np.float32


def test_vectors_are_l2_normalized():
    import numpy as np

    matrix = HashingEmbedder(128).encode(TEXTS)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encoding_is_deterministic_across_instances():
    import numpy as np

    first = HashingEmbedder(128).encode(TEXTS)
    second = HashingEmbedder(128).encode(TEXTS)
    assert np.array_equal(first, second)


def test_encoding_is_stable_across_processes():
    """The bucket comes from SHA-1, so two independent processes
    agree — no reliance on hash() randomization."""
    import subprocess
    import sys

    script = (
        "from repo_intelligence.retrieval.embeddings "
        "import HashingEmbedder\n"
        "import numpy as np\n"
        "m = HashingEmbedder(128).encode({!r})\n"
        "print(m.tobytes().hex())\n".format(TEXTS)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    here = HashingEmbedder(128).encode(TEXTS).tobytes().hex()
    assert proc.stdout.strip() == here


def test_similar_texts_score_higher_than_unrelated():
    import numpy as np

    embedder = HashingEmbedder(256)
    query = embedder.encode_one("agent uses the tool")
    related = embedder.encode_one("the agent uses the tool here")
    unrelated = embedder.encode_one("gardening calendar placeholder")
    dot = lambda a, b: float(np.dot(a, b))
    assert dot(query, related) > dot(query, unrelated)


def test_distinct_texts_produce_distinct_vectors():
    import numpy as np

    matrix = HashingEmbedder(128).encode(TEXTS)
    for i in range(len(TEXTS)):
        for j in range(i + 1, len(TEXTS)):
            assert not np.array_equal(matrix[i], matrix[j])


def test_dimension_bounds_are_enforced():
    for bad in (7, 8192 + 1, 0, -1):
        with pytest.raises(RetrievalError) as exc:
            HashingEmbedder(bad)
        assert exc.value.code == "RAG_INVALID_ARGUMENT"


def test_embedder_for_hashing_identities():
    default = embedder_for("hashing://v1")
    assert isinstance(default, HashingEmbedder)
    assert default.dimension == 256
    custom = embedder_for("hashing://v1/256")
    assert isinstance(custom, HashingEmbedder)
    assert custom.dimension == 256
    assert custom.name == "hashing://v1/256"
    small = embedder_for("hashing://v1/128")
    assert isinstance(small, HashingEmbedder)
    assert small.dimension == 128


def test_embedder_for_rejects_unknown_schemes():
    for identity in (
        "",
        None,
        7,
        "hashing://",
        "hashing://v1/abc",
        "word2vec://v1",
    ):
        with pytest.raises(RetrievalError) as exc:
            embedder_for(identity)
        assert exc.value.code in (
            "RAG_INVALID_ARGUMENT",
            "RAG_UNKNOWN_EMBEDDER",
        )


def test_sentence_transformer_embedder_constructs_lazily():
    embedder = embedder_for(
        "sentence-transformers://BAAI/bge-small-en-v1.5"
    )
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.name.startswith("sentence-transformers://")
    # no model is loaded at construction time (lazy on first encode)
    assert embedder._model is None


def test_hashing_vectors_are_cross_dimension_incompatible():
    import numpy as np

    small = HashingEmbedder(64).encode(TEXTS)
    large = HashingEmbedder(128).encode(TEXTS)
    assert small.shape[1] != large.shape[1]
    with pytest.raises(ValueError):
        np.dot(small, large.T)


def test_hashing_bucketing_is_sha1_based():
    embedder = HashingEmbedder(128)
    zero = embedder.encode([""])
    # empty text → no features → zero vector survives normalization
    import numpy as np

    assert np.allclose(zero[0], 0.0)
    digest = hashlib.sha1(b"agent").digest()
    assert int.from_bytes(digest[:8], "big") % 128 < 128


def test_model_load_applies_ssl_bypass_before_the_hub_import(
    monkeypatch
):
    """REPO_INTELLIGENCE_INSECURE_SSL must take effect BEFORE
    sentence_transformers (and its huggingface-hub httpx client) is
    imported — third-party hub clients can be constructed at import
    time, after which verify settings can no longer be changed."""
    import repo_intelligence.retrieval.embeddings as emb_mod

    calls = []
    monkeypatch.setattr(
        emb_mod, "apply_insecure_ssl", lambda: calls.append("apply")
    )
    # simulate "sentence-transformers not installed" (ImportError)
    # so _load fails fast without touching the network
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbedder("ssl-bypass-probe-model")
    with pytest.raises(RetrievalError):
        embedder._load()
    assert calls == ["apply"]
