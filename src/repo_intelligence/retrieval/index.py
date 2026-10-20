"""Build and load repository retrieval indexes (M18 §16, §34).

One index lives at
``<db_store>/rag/vector_store/<repo_key>/<commit_key>/<model_key>/``:

    index.faiss     vectors (one per chunk, L2-normalized, inner
                    product == cosine ranking)
    manifest.json   identity + embedder + chunking parameters +
                    counts — written LAST, so a crash mid-build
                    leaves no manifest and the next build simply
                    redoes the work (never a half-index that answers
                    queries)

Chunk metadata rows (what the vectors mean) live in the shared
``<db_store>/rag/metadata.db`` and are written replace-wise for the
identity. Building is explicit: query paths never build implicitly —
a missing index raises ``RAG_INDEX_NOT_FOUND`` carrying the exact
build command (spec §34).

CLI::

    python -m repo_intelligence.retrieval.index \
        --repository-root <checkout> [--force] ...

or through the project CLI: ``repo-intelligence retrieval-index ...``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..artifacts import repository_dir, resolve_output_root
from ..storage import rag_vector_store_path
from .chunking import DEFAULT_CHUNK_LINES, DEFAULT_CHUNK_OVERLAP
from .chunking import chunk_repository
from .embeddings import embedder_for
from .errors import (
    RAG_BUILD_FAILED,
    RAG_CORRUPT_INDEX,
    RAG_FAISS_UNAVAILABLE,
    RAG_IDENTITY_MISMATCH,
    RAG_INDEX_AMBIGUOUS,
    RAG_INDEX_NOT_FOUND,
    RAG_REPOSITORY_NOT_FOUND,
    RetrievalError,
)
from .identity import identity_for_snapshot, resolve_index_identity
from .metadata import (
    delete_identity_rows,
    metadata_db_path,
    read_chunks,
    write_chunks,
)
from .models import BuildReport, IndexIdentity

__all__ = [
    "INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "LoadedIndex",
    "active_index_identity",
    "build_index",
    "delete_index",
    "index_dir_for",
    "index_run",
    "load_index",
    "main",
]

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "index.faiss"


@dataclass
class LoadedIndex:
    """Everything a query needs for one identity."""

    index: Any
    records: List[Dict[str, Any]]
    manifest: Dict[str, Any]
    index_dir: Path
    dimension: int

    @property
    def ntotal(self) -> int:
        return int(getattr(self.index, "ntotal", 0))


def active_index_identity(
    identity: IndexIdentity, store_root: Optional[Path] = None
) -> IndexIdentity:
    """Identity pointing at the index that actually exists.

    The default model's index wins when present (production path);
    otherwise a single built index is selected by its manifest — the
    query then embeds with THAT index's model, so the vectors and the
    query always share one model (no cross-model fallback, ever).
    Several non-default indexes with no default →
    ``RAG_INDEX_AMBIGUOUS`` listing the candidates. No indexes → the
    identity stays on the default model, so the caller's
    ``RAG_INDEX_NOT_FOUND`` reports the default build command.
    """
    base = rag_vector_store_path(
        identity.repository_key(), identity.commit_sha, None, store_root
    )
    if not base.is_dir():
        return identity
    models: List[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        try:
            manifest = _read_manifest(child)
        except RetrievalError:
            continue  # a damaged sibling never blocks a good index
        if not manifest:
            continue
        stored = (manifest.get("identity") or {}).get("embedding_model")
        if isinstance(stored, str) and stored:
            models.append(stored)
    unique = sorted(set(models))
    if not unique:
        return identity
    if identity.embedding_model in unique:
        return identity
    if len(unique) == 1:
        return replace(identity, embedding_model=unique[0])
    raise RetrievalError(
        RAG_INDEX_AMBIGUOUS,
        "several retrieval indexes exist for repository {!r} at "
        "commit {!r} ({}) and none uses the default model {!r}; "
        "delete the stale ones or rebuild the one to query with "
        "python -m repo_intelligence.retrieval.index "
        "--repository-root <checkout> --model <identity>.".format(
            identity.repository_name,
            identity.commit_sha,
            ", ".join(unique),
            identity.embedding_model,
        ),
        details={
            "available_models": unique,
            "default_model": identity.embedding_model,
        },
    )


def index_dir_for(
    identity: IndexIdentity, store_root: Optional[Path] = None
) -> Path:
    """Directory of *identity*'s vectors + manifest."""
    return rag_vector_store_path(
        identity.repository_key(),
        identity.commit_sha,
        identity.embedding_key(),
        store_root=store_root,
    )


def _require_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RetrievalError(
            RAG_FAISS_UNAVAILABLE,
            "faiss is not installed, so retrieval indexes cannot be "
            "built or searched. Install the RAG extras with: "
            "pip install '.[rag]' (original error: {}).".format(exc),
        ) from exc
    return faiss


def _manifest_path(index_dir: Path) -> Path:
    return Path(index_dir) / MANIFEST_FILENAME


def _read_manifest(index_dir: Path) -> Optional[Dict[str, Any]]:
    path = _manifest_path(index_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RetrievalError(
            RAG_CORRUPT_INDEX,
            "index manifest at {} is unreadable ({}). Rebuild with: "
            "python -m repo_intelligence.retrieval.index "
            "--repository-root <checkout> --force".format(path, exc),
        )
    if not isinstance(data, dict):
        raise RetrievalError(
            RAG_CORRUPT_INDEX,
            "index manifest at {} is not a JSON object. Rebuild with: "
            "python -m repo_intelligence.retrieval.index "
            "--repository-root <checkout> --force".format(path),
        )
    return data


def _identity_matches(
    manifest: Dict[str, Any], identity: IndexIdentity
) -> bool:
    stored = manifest.get("identity")
    if not isinstance(stored, dict):
        return False
    return (
        stored.get("repository_name") == identity.repository_name
        and stored.get("commit_sha") == identity.commit_sha
        and stored.get("embedding_model") == identity.embedding_model
    )


def build_index(
    repository_root: Path,
    identity: IndexIdentity,
    store_root: Optional[Path] = None,
    embedder: Optional[Any] = None,
    chunk_lines: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    force: bool = False,
) -> BuildReport:
    """Chunk + embed + persist the index for *identity*.

    Returns a ``BuildReport`` with ``reused=True`` when a complete
    index for the same identity already exists and *force* is false —
    building is idempotent and cheap to ask for twice.
    """
    root = Path(repository_root)
    if not root.is_dir():
        raise RetrievalError(
            RAG_REPOSITORY_NOT_FOUND,
            "checkout not found: {}. Persist the pinned snapshot "
            "first (repo-intelligence snapshot) or pass an existing "
            "checkout.".format(root),
        )
    embedder = embedder or embedder_for(identity.embedding_model)
    if getattr(embedder, "name", None) != identity.embedding_model:
        raise RetrievalError(
            RAG_IDENTITY_MISMATCH,
            "embedder {!r} does not match the identity's embedding "
            "model {!r}; an index must be built with the model it "
            "will be queried with.".format(
                getattr(embedder, "name", None),
                identity.embedding_model,
            ),
        )
    lines = chunk_lines if chunk_lines is not None else DEFAULT_CHUNK_LINES
    overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else DEFAULT_CHUNK_OVERLAP
    )
    index_dir = index_dir_for(identity, store_root)
    existing = _read_manifest(index_dir) if index_dir.is_dir() else None
    if existing is not None and not force:
        if not _identity_matches(existing, identity):
            raise RetrievalError(
                RAG_IDENTITY_MISMATCH,
                "existing index at {} belongs to a different identity "
                "({}); refusing to reuse it.".format(
                    index_dir, existing.get("identity")
                ),
            )
        embedder_info = existing.get("embedder") or {}
        return BuildReport(
            identity=identity,
            index_dir=str(index_dir),
            files_indexed=int(existing.get("files_indexed", 0)),
            chunks_indexed=int(existing.get("chunks_indexed", 0)),
            reused=True,
            embedder=str(embedder_info.get("name", "")),
            dimension=int(embedder_info.get("dimension", 0)),
            created_at=existing.get("created_at"),
            notes={"chunking": existing.get("chunking", {})},
        )

    files_indexed, chunks = chunk_repository(
        root, chunk_lines=lines, chunk_overlap=overlap
    )
    try:
        vectors = embedder.encode([chunk.content for chunk in chunks])
        dimension = int(embedder.dimension)
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(
            RAG_BUILD_FAILED,
            "embedding the chunks of {} failed ({}).".format(root, exc),
        ) from exc
    if getattr(vectors, "shape", None) != (
        len(chunks),
        dimension,
    ):
        raise RetrievalError(
            RAG_BUILD_FAILED,
            "embedder {!r} produced shape {} for {} chunks "
            "(expected ({}, {})).".format(
                identity.embedding_model,
                getattr(vectors, "shape", None),
                len(chunks),
                len(chunks),
                dimension,
            ),
        )
    faiss = _require_faiss()
    faiss_index = faiss.IndexFlatIP(dimension)
    if len(chunks):
        faiss_index.add(vectors)

    try:
        if index_dir.exists():
            shutil.rmtree(str(index_dir))
        index_dir.mkdir(parents=True, exist_ok=True)
        write_chunks(
            identity, chunks, db_path=metadata_db_path(store_root)
        )
        faiss.write_index(faiss_index, str(index_dir / INDEX_FILENAME))
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "embedder": {
                "name": identity.embedding_model,
                "dimension": dimension,
            },
            "chunking": {
                "chunk_lines": lines,
                "chunk_overlap": overlap,
            },
            "files_indexed": files_indexed,
            "chunks_indexed": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _manifest_path(index_dir).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except RetrievalError:
        raise
    except Exception as exc:
        # roll back partial artifacts: no manifest means "not built".
        try:
            if index_dir.exists():
                shutil.rmtree(str(index_dir), ignore_errors=True)
        except OSError:
            pass
        delete_identity_rows(identity, db_path=metadata_db_path(store_root))
        raise RetrievalError(
            RAG_BUILD_FAILED,
            "writing the index at {} failed ({}); partial output was "
            "removed. Retry after checking disk space and "
            "permissions.".format(index_dir, exc),
        ) from exc

    return BuildReport(
        identity=identity,
        index_dir=str(index_dir),
        files_indexed=files_indexed,
        chunks_indexed=len(chunks),
        reused=False,
        embedder=identity.embedding_model,
        dimension=dimension,
        created_at=manifest["created_at"],
        notes={"chunking": {"chunk_lines": lines, "chunk_overlap": overlap}},
    )


def load_index(
    identity: IndexIdentity,
    store_root: Optional[Path] = None,
) -> LoadedIndex:
    """Load *identity*'s index, fully validated — or fail loudly."""
    index_dir = index_dir_for(identity, store_root)
    manifest = _read_manifest(index_dir)
    if manifest is None:
        raise RetrievalError(
            RAG_INDEX_NOT_FOUND,
            "no retrieval index for repository {!r} at commit {!r} "
            "(model {!r}); expected {}. Build it with: "
            "python -m repo_intelligence.retrieval.index "
            "--repository-root <checkout>".format(
                identity.repository_name,
                identity.commit_sha,
                identity.embedding_model,
                index_dir,
            ),
            details={
                "index_dir": str(index_dir),
                "identity": identity.to_dict(),
            },
        )
    if not _identity_matches(manifest, identity):
        raise RetrievalError(
            RAG_IDENTITY_MISMATCH,
            "index at {} belongs to {}, not {}.".format(
                index_dir,
                (manifest.get("identity") or {}),
                identity.to_dict(),
            ),
        )
    faiss = _require_faiss()
    vector_path = index_dir / INDEX_FILENAME
    if not vector_path.is_file():
        raise RetrievalError(
            RAG_CORRUPT_INDEX,
            "manifest exists but {} is missing. Rebuild with: "
            "python -m repo_intelligence.retrieval.index "
            "--repository-root <checkout> --force".format(vector_path),
        )
    try:
        faiss_index = faiss.read_index(str(vector_path))
    except Exception as exc:
        raise RetrievalError(
            RAG_CORRUPT_INDEX,
            "could not read the FAISS index at {} ({}). Rebuild with: "
            "python -m repo_intelligence.retrieval.index "
            "--repository-root <checkout> --force".format(
                vector_path, exc
            ),
        ) from exc
    records = read_chunks(identity, db_path=metadata_db_path(store_root))
    if int(faiss_index.ntotal) != len(records):
        raise RetrievalError(
            RAG_CORRUPT_INDEX,
            "index at {} has {} vectors but {} metadata rows; the two "
            "halves drifted apart. Rebuild with: "
            "python -m repo_intelligence.retrieval.index "
            "--repository-root <checkout> --force".format(
                index_dir, faiss_index.ntotal, len(records)
            ),
        )
    embedder_info = manifest.get("embedder") or {}
    dimension = int(embedder_info.get("dimension", 0)) or int(
        getattr(faiss_index, "d", 0)
    )
    return LoadedIndex(
        index=faiss_index,
        records=records,
        manifest=manifest,
        index_dir=index_dir,
        dimension=dimension,
    )


def index_run(
    output_root: Optional[Path],
    snapshot_id: str,
    store_root: Optional[Path] = None,
    embedder: Optional[Any] = None,
    chunk_lines: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    force: bool = False,
) -> BuildReport:
    """Build the index for an existing snapshot run (writer-side)."""
    root_out = resolve_output_root(output_root)
    checkout = repository_dir(root_out, snapshot_id)
    if not checkout.is_dir():
        raise RetrievalError(
            RAG_REPOSITORY_NOT_FOUND,
            "no persisted checkout at {} for run {!r}; run "
            "'repo-intelligence snapshot' first.".format(
                checkout, snapshot_id
            ),
        )
    identity = identity_for_snapshot(
        root_out,
        snapshot_id,
        embedding_model=(
            embedder.name if embedder is not None else None
        ),
    )
    return build_index(
        checkout,
        identity,
        store_root=store_root,
        embedder=embedder,
        chunk_lines=chunk_lines,
        chunk_overlap=chunk_overlap,
        force=force,
    )


def delete_index(
    identity: IndexIdentity, store_root: Optional[Path] = None
) -> bool:
    """Remove *identity*'s vectors, manifest, and metadata rows."""
    index_dir = index_dir_for(identity, store_root)
    removed = False
    if index_dir.exists():
        shutil.rmtree(str(index_dir), ignore_errors=True)
        removed = True
    rows = delete_identity_rows(identity, db_path=metadata_db_path(store_root))
    return removed or rows > 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: build a retrieval index, print the BuildReport as JSON."""
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.retrieval.index",
        description=(
            "Build the retrieval index (chunk + embed + persist) for "
            "a pinned repository checkout."
        ),
    )
    parser.add_argument(
        "--repository-root",
        dest="repository_root",
        default=None,
        help="local checkout to index (identity comes from the "
        "snapshot record beside it)",
    )
    parser.add_argument(
        "--snapshot-id",
        dest="snapshot_id",
        default=None,
        help="index this run's persisted checkout instead (uses "
        "--output-root to find it)",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="artifact output root (default: <project>/artifacts)",
    )
    parser.add_argument(
        "--store",
        dest="store",
        default=None,
        help="rag store root (default: <db_store>/rag)",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="override the embedding identity (default: "
        "sentence-transformers model configured for the project)",
    )
    parser.add_argument(
        "--chunk-lines",
        dest="chunk_lines",
        type=int,
        default=None,
        help="target lines per chunk (default: {})".format(
            DEFAULT_CHUNK_LINES
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        dest="chunk_overlap",
        type=int,
        default=None,
        help="overlapping lines between chunks (default: {})".format(
            DEFAULT_CHUNK_OVERLAP
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when a complete index exists",
    )
    args = parser.parse_args(argv)

    if not args.repository_root and not args.snapshot_id:
        parser.error(
            "one of --repository-root or --snapshot-id is required"
        )
    try:
        if args.snapshot_id:
            report = index_run(
                Path(args.output_root) if args.output_root else None,
                args.snapshot_id,
                store_root=Path(args.store) if args.store else None,
                embedder=(
                    embedder_for(args.model) if args.model else None
                ),
                chunk_lines=args.chunk_lines,
                chunk_overlap=args.chunk_overlap,
                force=args.force,
            )
        else:
            root = Path(args.repository_root or "")
            identity = resolve_index_identity(
                root,
                Path(args.output_root) if args.output_root else None,
                embedding_model=args.model,
            )
            report = build_index(
                root,
                identity,
                store_root=Path(args.store) if args.store else None,
                embedder=(
                    embedder_for(args.model) if args.model else None
                ),
                chunk_lines=args.chunk_lines,
                chunk_overlap=args.chunk_overlap,
                force=args.force,
            )
    except RetrievalError as exc:
        print(
            "retrieval index build failed [{}]: {}".format(
                exc.code, exc.message
            ),
            file=sys.stderr,
        )
        return 6
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
