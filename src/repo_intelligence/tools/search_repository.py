"""Tool 2 (read-only): search_repository.

Semantic retrieval over the pinned repository snapshot — the exact
checkout persisted by M1 (M18 replaces the former lexical walk).

    "Where in this repository should I look for evidence related to
    this query?"

Execution path (read-only; no index build, no LLM):

    search_repository({"query": "checkout flow"})
        → validate the input contract
        → resolve the repository root (application context: the pinned
          run snapshot at <output>/<run>/repository[/<name>] by default)
        → resolve the repository identity from the M1 snapshot record
          (repository name + pinned commit — never a guess)
        → semantic search over that identity's prebuilt retrieval
          index (chunk vectors ranked by cosine similarity)
        → return ToolResult with line-ranged, excerpted evidence

The index is built EXPLICITLY (never implicitly from a query):

    repo-intelligence retrieval-index --repository-root <checkout>

A missing index returns the structured error
``RAG_INDEX_NOT_FOUND`` carrying that exact command. Determinism: the
same query against the same pinned index always returns the same
results; ranking is exact inner-product search over the fixed vector
set with a stable tie-break, and excerpts come from the indexed text.

Response ceiling: one serialized ``ToolResult`` never exceeds
``MAX_TOOL_RESPONSE_CHARS``. Results are dropped WHOLE from the tail
(a result is never cut in half) until the payload — envelope
included — fits; ``omitted_results``, ``note`` and ``truncated`` say
how many were withheld and why. Ranking itself is never changed: the
delivered results are the head of the same ranked list.

Provenance (repository name, repository URL, commit_sha, snapshot id)
comes from the existing M1 snapshot artifact of the current run; the
tool never inspects Git state itself and never clones.

Shared helpers for the whole tool layer (reused by ``read_file``) live
here because the tool package adds no extra infrastructure modules:
``normalize_repository_relative_path``, ``resolve_repository_root``,
``source_metadata``, and the re-exported file-policy constants.

Standalone (JSON ToolResult on stdout):

    python -m repo_intelligence.tools.search_repository --query "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..artifacts import (
    repository_dir,
    resolve_output_root,
    snapshot_path,
)
from ..context.models import ArchitectureContextError, discover_snapshot_id
from ..facts.filesystem import (
    BINARY_SNIFF_BYTES,
    DEFAULT_IGNORED_DIRS,
    MAX_FILE_BYTES,
    SEARCHABLE_SUFFIXES,
)
from ..retrieval import RetrievalService, active_index_identity
from ..retrieval.errors import RetrievalError
from ..retrieval.identity import resolve_index_identity
from .protocol import (
    MAX_TOOL_RESPONSE_CHARS,
    TRUNCATION_NOTE_RESERVE,
    ToolError,
    ToolResult,
    response_chars,
)

TOOL_NAME = "search_repository"

__all__ = [
    "BINARY_SNIFF_BYTES",
    "DEFAULT_IGNORED_DIRS",
    "DEFAULT_MAX_RESULTS",
    "EVIDENCE_NOTE",
    "MAX_FILE_BYTES",
    "MAX_RESULTS_LIMIT",
    "SEARCHABLE_SUFFIXES",
    "TOOL_NAME",
    "main",
    "normalize_repository_relative_path",
    "resolve_repository_root",
    "search_repository",
    "source_metadata",
]

# Search limits (generic defaults; not repository-specific).
# M18: the default result count follows the retrieval top_k (5).
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 50

# Carried in every successful result so the caller's next context
# states the M13 rule where the candidates appear: a similarity hit
# names a candidate source; only a read_file of that range makes it
# inspectable evidence.
EVIDENCE_NOTE = (
    "results are discoveries, not evidence: each result names a "
    "candidate path and line range — call read_file on the exact "
    "range you intend to cite before putting it in evidence_refs"
)

# Shared file policy (SEARCHABLE_SUFFIXES, MAX_FILE_BYTES,
# BINARY_SNIFF_BYTES) comes from facts.filesystem — one definition
# for every reader of repository text (M18 §14).

# Conventional location of the pinned checkout inside a run directory
# (re-exported from artifacts): <output>/<run-id>/repository[/<name>]
# — application context, not a tool argument.

# Worst-case length of a truncation note while a bound is computed —
# the placeholder keeps the ceiling a guarantee, not an estimate.
_NOTE_PLACEHOLDER = "." * TRUNCATION_NOTE_RESERVE

_ALLOWED_ARGUMENTS = ("query", "path", "max_results")


def _failure(code: str, message: str) -> ToolResult:
    return ToolResult(
        success=False,
        tool=TOOL_NAME,
        error=ToolError(code=code, message=message),
    )


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def _validate_arguments(
    arguments: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
    """Validate and normalize the tool arguments."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        return None, _failure(
            "INVALID_ARGUMENT",
            "arguments must be a mapping containing a 'query' string.",
        )

    unknown = [str(key) for key in arguments if key not in _ALLOWED_ARGUMENTS]
    if unknown:
        return None, _failure(
            "INVALID_ARGUMENT",
            "unexpected argument(s): {}.".format(", ".join(sorted(unknown))),
        )

    query_value = arguments.get("query")
    if not isinstance(query_value, str):
        return None, _failure("INVALID_ARGUMENT", "query must be a string.")
    query = query_value.strip()
    if not query:
        return None, _failure("INVALID_ARGUMENT", "query must not be empty.")

    path_value = arguments.get("path", None)
    path_filter: Optional[str] = None
    if path_value is not None:
        if not isinstance(path_value, str):
            return None, _failure(
                "INVALID_ARGUMENT",
                "path must be a string or null.",
            )
        path_filter, path_error = normalize_repository_relative_path(
            path_value
        )
        if path_error is not None:
            return None, _failure("PATH_OUTSIDE_REPOSITORY", path_error)

    max_value = arguments.get("max_results", None)
    if max_value is None:
        max_results = DEFAULT_MAX_RESULTS
    elif isinstance(max_value, bool) or not isinstance(max_value, int):
        return None, _failure(
            "INVALID_ARGUMENT",
            "max_results must be an integer.",
        )
    elif max_value < 1 or max_value > MAX_RESULTS_LIMIT:
        return None, _failure(
            "INVALID_ARGUMENT",
            "max_results must be between 1 and {}.".format(MAX_RESULTS_LIMIT),
        )
    else:
        max_results = max_value

    return (
        {"query": query, "path": path_filter, "max_results": max_results},
        None,
    )


# ---------------------------------------------------------------------------
# Shared tool-layer helpers (also used by read_file)
# ---------------------------------------------------------------------------


def normalize_repository_relative_path(
    path_value: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize a repository-relative path.

    Returns ``(normalized, None)`` on success (``None`` means "no filter"
    for ``""``/``"."`` inputs) or ``(None, error_message)`` when the path
    is absolute or escapes the repository. The caller reports
    ``PATH_OUTSIDE_REPOSITORY`` for error messages.
    """
    text = path_value.strip().replace("\\", "/")
    if text in ("", "."):
        return None, None
    if text.startswith("/"):
        return None, (
            "path must be inside the repository; got an absolute path: "
            "{!r}".format(path_value)
        )
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return None, (
            "path must be inside the repository; got an absolute path: "
            "{!r}".format(path_value)
        )
    segments = [part for part in text.split("/") if part not in ("", ".")]
    if any(part == ".." for part in segments):
        return None, (
            "path must not escape the repository; got: {!r}".format(path_value)
        )
    return "/".join(segments) or None, None


# ---------------------------------------------------------------------------
# Repository selection (application context)
# ---------------------------------------------------------------------------


def resolve_repository_root(
    repository_root: Optional[Path], output_root: Optional[Path]
) -> Tuple[Optional[Path], Optional[ToolResult]]:
    """Resolve the local checkout to search (never clones, no Git).

    Without an explicit *repository_root* override, the pinned snapshot
    of the current run (``<output>/<run>/repository[/<name>]``,
    persisted by M1) is used. The explicit override exists for testing
    and standalone use only; agents never supply it.
    """
    root_out = resolve_output_root(output_root)
    if repository_root is not None:
        candidate = Path(repository_root)
        if candidate.is_dir():
            return candidate.resolve(), None
        return None, _failure(
            "REPOSITORY_NOT_FOUND",
            "Repository checkout not found: {}. Point repository_root at a "
            "local checkout of the analyzed repository.".format(candidate),
        )

    try:
        run_id = discover_snapshot_id(root_out)
    except ArchitectureContextError:
        return None, _failure(
            "REPOSITORY_NOT_FOUND",
            "No repository checkout resolved and no current run found under "
            "{}. Re-run M1 + M2 to persist the pinned snapshot, or pass "
            "repository_root (--repository-root) with a local checkout of "
            "the analyzed repository.".format(root_out),
        )
    candidate = repository_dir(root_out, run_id)
    if candidate.is_dir():
        return candidate.resolve(), None
    return None, _failure(
        "REPOSITORY_NOT_FOUND",
        "No repository checkout for run {!r} (expected {}). Re-run M1 to "
        "persist the pinned snapshot, or pass repository_root "
        "(--repository-root) with a local checkout of the analyzed "
        "repository.".format(run_id, candidate),
    )


def _resolve_search_target(
    repo_root: Path, path_filter: Optional[str]
) -> Tuple[Optional[Path], Optional[ToolResult]]:
    """Resolve the optional path filter inside the repository root."""
    if path_filter is None:
        return repo_root, None
    candidate = repo_root.joinpath(*path_filter.split("/"))
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        return None, _failure(
            "SEARCH_FAILED",
            "path could not be resolved: {!r} ({})".format(path_filter, exc),
        )
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return None, _failure(
            "PATH_OUTSIDE_REPOSITORY",
            "path escapes the repository: {!r}".format(path_filter),
        )
    if not resolved.exists():
        return None, _failure(
            "PATH_NOT_FOUND",
            "path does not exist in the repository: {!r}".format(path_filter),
        )
    if not (resolved.is_dir() or resolved.is_file()):
        return None, _failure(
            "PATH_NOT_FOUND",
            "path is not a file or directory: {!r}".format(path_filter),
        )
    return resolved, None


# ---------------------------------------------------------------------------
# Retrieval (read-only over the prebuilt index; never builds)
# ---------------------------------------------------------------------------


def source_metadata(output_root: Optional[Path]) -> Dict[str, Any]:
    """Provenance from the current run's M1 snapshot artifact (nulls if any
    part of the run context is unavailable — never an error).

    Keys: ``repository`` (name), ``commit_sha`` (pinned commit),
    ``snapshot_id`` (run/snapshot identifier), ``repository_url``.
    """
    source: Dict[str, Any] = {
        "repository": None,
        "commit_sha": None,
        "snapshot_id": None,
        "repository_url": None,
    }
    try:
        root_out = resolve_output_root(output_root)
        run_id = discover_snapshot_id(root_out)
        source["snapshot_id"] = run_id
        with snapshot_path(root_out, run_id).open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
        repository = snapshot["repository"]["name"]
        url = snapshot["repository"]["url"]
        commit = snapshot["snapshot"]["commit_sha"]
    except (
        ArchitectureContextError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ):
        return source
    source["repository"] = repository if isinstance(repository, str) else None
    source["commit_sha"] = commit if isinstance(commit, str) else None
    source["repository_url"] = url if isinstance(url, str) else None
    return source


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _bound_payload(
    data: Dict[str, Any], source: Dict[str, Any]
) -> ToolResult:
    """Cap one successful payload at ``MAX_TOOL_RESPONSE_CHARS``.

    Whole results are dropped from the TAIL until the SERIALIZED
    result (envelope included) fits — a result is never cut in half,
    ranking is never recomputed, so the delivered list stays the head
    of the ranked list. The bound is measured against a probe carrying
    every field the final payload will carry at its worst-case length
    (every result counted as omitted, the note reserved), so the final
    result is guaranteed to fit. ``truncated`` keeps its existing
    conditions and gains ``omitted_results > 0``; ``returned`` counts
    what was actually delivered.

    A payload whose envelope alone breaks the ceiling (an absurdly
    long query) is a structured ``QUERY_TOO_LONG`` failure — dropping
    results cannot help there.
    """
    results = list(data.get("results") or [])
    original_count = len(results)

    def _result(
        kept: List[Dict[str, Any]],
        truncated: bool,
        note: Optional[str],
        omitted: int,
    ) -> ToolResult:
        payload = dict(data)
        payload["results"] = kept
        payload["returned"] = len(kept)
        payload["truncated"] = truncated
        payload["omitted_results"] = omitted
        payload["note"] = note
        return ToolResult(
            success=True,
            tool=TOOL_NAME,
            data=payload,
            source=source,
            error=None,
        )

    full = _result(results, bool(data.get("truncated")), None, 0)
    if response_chars(full) <= MAX_TOOL_RESPONSE_CHARS:
        return full

    # Over the ceiling: probe with every result counted as omitted and
    # the note reserved (both are at their longest there — the final
    # payload can only be shorter), then drop tail results until the
    # probe fits.
    kept = list(results)
    while kept and response_chars(
        _result(kept, True, _NOTE_PLACEHOLDER, original_count)
    ) > MAX_TOOL_RESPONSE_CHARS:
        kept.pop()
    if not kept and response_chars(
        _result([], True, _NOTE_PLACEHOLDER, original_count)
    ) > MAX_TOOL_RESPONSE_CHARS:
        return _failure(
            "QUERY_TOO_LONG",
            "The query and response envelope exceed the "
            "{}-character response ceiling even without results; "
            "shorten the query.".format(MAX_TOOL_RESPONSE_CHARS),
        )

    omitted = original_count - len(kept)
    note = (
        "response truncated: {} of {} results omitted to fit the "
        "{}-character response ceiling; narrow the query or path "
        "filter to see them."
    ).format(omitted, original_count, MAX_TOOL_RESPONSE_CHARS)
    return _result(kept, True, note, omitted)


def search_repository(
    arguments: Optional[Mapping[str, Any]] = None,
    repository_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    store_root: Optional[Path] = None,
) -> ToolResult:
    """Semantic retrieval over the pinned repository snapshot's index,
    returned as the generic ToolResult.

    Input contract::

        {
            "query": "checkout flow",   # required, non-empty string
            "path": "docs",             # optional repository-relative filter
            "max_results": 5            # optional, 1..50
        }

    The same query against the same pinned index always returns the
    same results: exact vector ranking with a stable tie-break, and
    excerpts taken from the indexed text. The retrieval index must
    have been built explicitly (``repo-intelligence retrieval-index
    --repository-root <checkout>``); a missing index is reported as
    the structured error ``RAG_INDEX_NOT_FOUND`` carrying that
    command — searching never builds an index as a side effect.

    ``repository_root``, ``output_root``, and ``store_root`` are
    application context (the pinned checkout, the artifact output
    root, and the retrieval store — ``<db_store>/rag`` by default),
    not tool arguments.
    """
    normalized, error = _validate_arguments(arguments)
    if error is not None:
        return error
    assert normalized is not None
    query = normalized["query"]
    path_filter = normalized["path"]
    max_results = normalized["max_results"]

    repo_root, error = resolve_repository_root(repository_root, output_root)
    if error is not None:
        return error
    assert repo_root is not None

    # Validates path safety/existence against the checkout (the same
    # PATH_* contract as before M18); retrieval then filters by it.
    _target, error = _resolve_search_target(repo_root, path_filter)
    if error is not None:
        return error

    # Identity (repository + pinned commit) comes from the M1
    # snapshot record beside the checkout — never a guess, never
    # another repository's identity. The active model is whichever
    # index exists for that identity (the default model when built).
    try:
        identity = resolve_index_identity(repo_root, output_root)
        identity = active_index_identity(identity, store_root)
    except RetrievalError as exc:
        return _failure(exc.code, exc.message)

    service = RetrievalService(store_root=store_root)
    try:
        response = service.search(
            query,
            identity,
            top_k=max_results,
            path_filter=path_filter,
        )
    except RetrievalError as exc:
        return _failure(exc.code, exc.message)

    results = [item.to_dict() for item in response.results]
    total = int(response.total_matches)
    return _bound_payload(
        {
            "query": query,
            "results": results,
            "total_matches": total,
            "returned": len(results),
            "truncated": len(results) < total or response.scan_capped,
            "evidence_note": EVIDENCE_NOTE,
            "omitted_results": 0,
            "note": None,
            "retrieval": {
                "mode": "semantic",
                "embedding_model": identity.embedding_model,
                "repository": identity.repository_name,
                "commit_sha": identity.commit_sha,
                "snapshot_id": identity.snapshot_id,
                "index_chunks": int(response.index_chunks),
            },
        },
        source=source_metadata(output_root),
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Standalone entry point: print the ToolResult as JSON on stdout."""
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.tools.search_repository",
        description=(
            "Read-only tool: semantic retrieval over the pinned "
            "repository snapshot's index of the current run; prints "
            "the ToolResult as JSON."
        ),
    )
    parser.add_argument(
        "--query",
        required=True,
        help="natural-language or keyword query to rank evidence for",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="optional repository-relative path filter (file or directory)",
    )
    parser.add_argument(
        "--max-results",
        dest="max_results",
        type=int,
        default=None,
        help="maximum results to return, 1-50 (default: {})".format(
            DEFAULT_MAX_RESULTS
        ),
    )
    parser.add_argument(
        "--repository-root",
        dest="repository_root",
        default=None,
        help="optional override: local checkout to search (testing and "
        "standalone use only — agents never provide this; default: pinned "
        "snapshot at <output>/<run>/repository/<name>)",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="application context: artifact output root "
        "(default: <project>/artifacts)",
    )
    parser.add_argument(
        "--store",
        dest="store",
        default=None,
        help="retrieval store root (default: <db_store>/rag)",
    )
    args = parser.parse_args(argv)

    arguments: Dict[str, Any] = {"query": args.query}
    if args.path is not None:
        arguments["path"] = args.path
    if args.max_results is not None:
        arguments["max_results"] = args.max_results

    def _optional_path(value: Optional[str]) -> Optional[Path]:
        return Path(value) if value else None

    result = search_repository(
        arguments,
        repository_root=_optional_path(args.repository_root),
        output_root=_optional_path(args.output_root),
        store_root=_optional_path(args.store),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 6


if __name__ == "__main__":
    sys.exit(main())
