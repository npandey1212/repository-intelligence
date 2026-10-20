"""Tool 3 (read-only): read_file.

Exact text evidence retrieval from one file of the pinned repository
snapshot — the exact checkout persisted by M1 at
``<output>/<snapshot_id>/repository/`` — optionally restricted to a line
range. One question for a future agent runtime:

    "Show me the exact content of this file (optionally these lines)."

Execution path (no interpretation, no index, no LLM):

    read_file({"path": "docs/workflow.md", "line_start": 42,
               "line_end": 55})
        → validate the input contract
        → resolve the repository root (application context — shared
          helper from search_repository)
        → resolve the path safely inside the repository
        → stat/read bytes → size, binary, encoding policy
        → split into 1-based lines → slice the requested range
        → return ToolResult

Line numbering is 1-based and inclusive; full-file reads return the
file's exact bytes, range reads return the selected lines joined with
``\\n``.

Response ceiling: one serialized ``ToolResult`` never exceeds
``MAX_TOOL_RESPONSE_CHARS``. A selection that would break the ceiling
is cut at a whole-line boundary (only if not even one line fits: a
character prefix of the first selected line) and the payload says so —
``truncated``, ``returned_chars``/``total_chars``, ``partial_line``,
``next_line_start`` (where to resume) and a ``note``. Reported
``line_start``/``line_end`` always cover the DELIVERED lines, never the
requested range, so evidence coverage can never over-claim.

Provenance (repository name, repository URL, commit_sha, snapshot id)
comes from the existing M1 snapshot artifact of the current run (shared
helper); the tool never inspects Git state itself and never clones.

Standalone (JSON ToolResult on stdout):

    python -m repo_intelligence.tools.read_file --path docs/workflow.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .protocol import (
    MAX_TOOL_RESPONSE_CHARS,
    TRUNCATION_NOTE_RESERVE,
    ToolError,
    ToolResult,
    bound_result_text,
    response_chars,
)
from .search_repository import (
    BINARY_SNIFF_BYTES,
    MAX_FILE_BYTES,
    normalize_repository_relative_path,
    resolve_repository_root,
    source_metadata,
)

TOOL_NAME = "read_file"

_ALLOWED_ARGUMENTS = ("path", "line_start", "line_end")

# Worst-case length of a truncation note while a bound is computed —
# the placeholder keeps the ceiling a guarantee, not an estimate.
_NOTE_PLACEHOLDER = "." * TRUNCATION_NOTE_RESERVE


def _failure(code: str, message: str) -> ToolResult:
    return ToolResult(
        success=False, tool=TOOL_NAME, error=ToolError(code=code, message=message)
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
            "arguments must be a mapping containing a 'path' string.",
        )

    unknown = [str(key) for key in arguments if key not in _ALLOWED_ARGUMENTS]
    if unknown:
        return None, _failure(
            "INVALID_ARGUMENT",
            "unexpected argument(s): {}.".format(", ".join(sorted(unknown))),
        )

    path_value = arguments.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None, _failure(
            "INVALID_ARGUMENT", "path must be a non-empty string."
        )
    rel_path, path_error = normalize_repository_relative_path(path_value)
    if path_error is not None:
        return None, _failure("PATH_OUTSIDE_REPOSITORY", path_error)
    if rel_path is None:
        # path was "." — the repository root itself is not a file
        return None, _failure(
            "NOT_A_FILE",
            "path must name a file inside the repository; got: {!r}".format(
                path_value
            ),
        )

    line_bounds: Dict[str, Optional[int]] = {"line_start": None, "line_end": None}
    for name in ("line_start", "line_end"):
        value = arguments.get(name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            return None, _failure(
                "INVALID_ARGUMENT", "{} must be an integer.".format(name)
            )
        if value < 1:
            return None, _failure(
                "INVALID_ARGUMENT", "{} must be >= 1.".format(name)
            )
        line_bounds[name] = value
    if (
        line_bounds["line_start"] is not None
        and line_bounds["line_end"] is not None
        and line_bounds["line_end"] < line_bounds["line_start"]
    ):
        return None, _failure(
            "INVALID_ARGUMENT", "line_end must be >= line_start."
        )

    return (
        {
            "path": rel_path,
            "line_start": line_bounds["line_start"],
            "line_end": line_bounds["line_end"],
        },
        None,
    )


# ---------------------------------------------------------------------------
# Path safety: the target must be a regular file inside the repository
# ---------------------------------------------------------------------------


def _resolve_file(
    repo_root: Path, rel_path: str
) -> Tuple[Optional[Path], Optional[ToolResult]]:
    parts = rel_path.split("/")

    # every directory component must exist and be a directory
    current = repo_root
    for part in parts[:-1]:
        current = current / part
        if not current.is_dir():
            return None, _failure(
                "PATH_NOT_FOUND",
                "path component {!r} does not exist or is not a "
                "directory.".format(part),
            )

    candidate = repo_root.joinpath(*parts)
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        return None, _failure(
            "READ_FAILED",
            "File could not be read: {!r} ({})".format(rel_path, exc),
        )
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return None, _failure(
            "PATH_OUTSIDE_REPOSITORY",
            "path resolves outside the repository: {!r}".format(rel_path),
        )
    if not resolved.exists():
        return None, _failure(
            "FILE_NOT_FOUND",
            "Requested file was not found: {!r}".format(rel_path),
        )
    if resolved.is_dir():
        return None, _failure(
            "NOT_A_FILE",
            "Requested path is a directory, not a file: {!r}".format(rel_path),
        )
    if not resolved.is_file():
        return None, _failure(
            "NOT_A_FILE",
            "Requested path is not a regular file: {!r}".format(rel_path),
        )
    return resolved, None


# ---------------------------------------------------------------------------
# Reading: size, binary, encoding, line slicing
# ---------------------------------------------------------------------------


def _split_lines(text: str) -> List[str]:
    """Split on ``\\n``; a single trailing newline terminates the last
    line instead of creating an empty one. ``\\r`` stays attached to its
    line so CRLF content round-trips through range reads unchanged."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _read(
    resolved: Path, rel_path: str, line_start: Optional[int], line_end: Optional[int]
) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        return None, _failure(
            "READ_FAILED",
            "File could not be read: {!r} ({})".format(rel_path, exc),
        )
    if size > MAX_FILE_BYTES:
        return None, _failure(
            "FILE_TOO_LARGE",
            "File is too large to read: {} bytes exceeds the limit of "
            "{} bytes.".format(size, MAX_FILE_BYTES),
        )
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        return None, _failure(
            "READ_FAILED",
            "File could not be read: {!r} ({})".format(rel_path, exc),
        )
    if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
        return None, _failure(
            "BINARY_FILE",
            "Requested file is not a supported text file: {!r}".format(rel_path),
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, _failure(
            "UNSUPPORTED_ENCODING",
            "File is not valid UTF-8 text: {!r}".format(rel_path),
        )

    lines = _split_lines(text)
    total = len(lines)

    if line_start is None and line_end is None:
        # full-file read: byte-faithful content
        content = text
        start_report = 1
        end_report = total if total > 0 else 0
    else:
        start = line_start if line_start is not None else 1
        end = line_end if line_end is not None else total
        selected = lines[start - 1 : end]
        if selected:
            content = "\n".join(selected)
            start_report = start
            end_report = start + len(selected) - 1
        else:
            # requested range starts past the end of the file
            content = ""
            start_report = 1
            end_report = 0

    return (
        {
            "path": rel_path,
            "line_start": start_report,
            "line_end": end_report,
            "content": content,
        },
        None,
    )


# ---------------------------------------------------------------------------
# Response ceiling: worst-case probe values and truncation notes
# ---------------------------------------------------------------------------


def _probe_next_line(end_report: int) -> int:
    """Worst-case ``next_line_start`` for a bound probe.

    The final value is either ``None`` (serialized as ``null``, four
    characters) or an integer no larger than ``end_report + 1``. The
    probe must carry whichever representation is LONGER for every
    outcome, otherwise the bound could fit a payload the final result
    then exceeds.
    """
    digits = max(4, len(str(max(end_report, 0) + 1)))
    return int("9" * digits)


def _truncation_note(
    start: int,
    end: int,
    returned_chars: int,
    total_chars: int,
    next_line_start: Optional[int],
) -> str:
    """Plain-ASCII note (no quotes or backslashes: its serialized
    length is its own length + 2) well inside ``TRUNCATION_NOTE_RESERVE``."""
    if next_line_start is not None:
        return (
            "response truncated: lines {}-{} of the selected range were "
            "returned ({} of {} characters); resume at line {}."
        ).format(start, end, returned_chars, total_chars, next_line_start)
    return (
        "response truncated: lines {}-{} were returned ({} of {} "
        "characters); no further line remains in the requested range."
    ).format(start, end, returned_chars, total_chars)


def _partial_line_note(line: int, limit: int, returned_chars: int) -> str:
    """Plain-ASCII note for the prefix-of-one-line outcome: there is no
    resume point, because re-reading the line hits the same ceiling."""
    return (
        "response truncated: line {} alone exceeds the {}-character "
        "ceiling, so only its first {} characters were returned and no "
        "complete line was delivered."
    ).format(line, limit, returned_chars)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_file(
    arguments: Optional[Mapping[str, Any]] = None,
    repository_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> ToolResult:
    """Read one repository file (optionally a line range) as a ToolResult.

    Reads from the pinned repository snapshot of the current run by
    default (``<output>/<run>/repository``), so file content always
    matches the commit recorded by M1.

    Input contract::

        {
            "path": "docs/workflow.md",  # required, repository-relative
            "line_start": 42,            # optional, integer >= 1
            "line_end": 55               # optional, integer >= line_start
        }

    Both line bounds omitted → the complete file. ``repository_root`` and
    ``output_root`` are application context (the pinned checkout and the
    artifact output root), not tool arguments. Lines are 1-based and
    inclusive.

    Output contract (``data``)::

        path, line_start, line_end, content,   # the evidence itself
        truncated, returned_chars, total_chars,
        char_limit, next_line_start,
        partial_line, note

    ``line_start``/``line_end`` always describe the DELIVERED lines.
    When the ceiling cuts the selection at whole lines, they stop at the
    last complete line and ``next_line_start`` names the line to resume
    from. When not even one complete line fits, they report an empty
    delivered range (``line_end == line_start - 1``), ``partial_line``
    is true and ``next_line_start`` is ``None`` — a line that cannot be
    read once can never be read in full under this ceiling.
    """
    normalized, error = _validate_arguments(arguments)
    if error is not None:
        return error
    assert normalized is not None

    repo_root, error = resolve_repository_root(repository_root, output_root)
    if error is not None:
        return error
    assert repo_root is not None

    resolved, error = _resolve_file(repo_root, normalized["path"])
    if error is not None:
        return error
    assert resolved is not None

    payload, error = _read(
        resolved,
        normalized["path"],
        normalized["line_start"],
        normalized["line_end"],
    )
    if error is not None:
        return error
    assert payload is not None

    source = source_metadata(output_root)
    content = str(payload["content"])
    total_chars = len(content)
    start_report = int(payload["line_start"])
    end_report = int(payload["line_end"])

    data: Dict[str, Any] = {
        "path": payload["path"],
        "line_start": start_report,
        "line_end": end_report,
        "content": content,
        "truncated": False,
        "returned_chars": total_chars,
        "total_chars": total_chars,
        "char_limit": MAX_TOOL_RESPONSE_CHARS,
        "next_line_start": None,
        "partial_line": False,
        "note": None,
    }
    result = ToolResult(
        success=True, tool=TOOL_NAME, data=data, source=source, error=None
    )
    if response_chars(result) <= MAX_TOOL_RESPONSE_CHARS:
        return result

    # Over the ceiling. The bound is measured against a probe carrying
    # every field the final payload will carry, each at its worst-case
    # length (counts at their pre-bound maximum, the longest serialized
    # form of each flag, the reserved note) — so the rebound result is
    # guaranteed to fit. The full text stays on the probe: the helper
    # re-measures with it emptied.
    probe_data = dict(data)
    probe_data.update(
        truncated=True,
        next_line_start=_probe_next_line(end_report),
        partial_line=False,
        note=_NOTE_PLACEHOLDER,
    )
    probe = ToolResult(
        success=True,
        tool=TOOL_NAME,
        data=probe_data,
        source=source,
        error=None,
    )
    bounded = bound_result_text(probe, "content", MAX_TOOL_RESPONSE_CHARS)

    if bounded.partial_line:
        # Not one complete line fits: the content is a prefix of the
        # first selected line, so NO line was delivered — the reported
        # range is empty and evidence coverage can never over-claim.
        data.update(
            content=bounded.text,
            truncated=True,
            returned_chars=len(bounded.text),
            line_end=start_report - 1,
            partial_line=True,
            next_line_start=None,
            note=_partial_line_note(
                start_report, MAX_TOOL_RESPONSE_CHARS, len(bounded.text)
            ),
        )
        return ToolResult(
            success=True, tool=TOOL_NAME, data=data, source=source, error=None
        )

    delivered_end = start_report + bounded.whole_lines - 1
    if delivered_end > end_report:
        delivered_end = end_report
    next_start = (
        delivered_end + 1 if delivered_end < end_report else None
    )
    data.update(
        content=bounded.text,
        truncated=True,
        returned_chars=len(bounded.text),
        line_end=delivered_end,
        next_line_start=next_start,
        note=_truncation_note(
            start_report,
            delivered_end,
            len(bounded.text),
            total_chars,
            next_start,
        ),
    )
    return ToolResult(
        success=True, tool=TOOL_NAME, data=data, source=source, error=None
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Standalone entry point: print the ToolResult as JSON on stdout."""
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.tools.read_file",
        description=(
            "Read-only tool: return the content of one file from the "
            "pinned repository snapshot of the current run (optionally "
            "a 1-based inclusive line range) and print its ToolResult "
            "as JSON."
        ),
    )
    parser.add_argument(
        "--path",
        required=True,
        help="repository-relative file path",
    )
    parser.add_argument(
        "--line-start",
        dest="line_start",
        type=int,
        default=None,
        help="first line to return, 1-based (default: beginning of file)",
    )
    parser.add_argument(
        "--line-end",
        dest="line_end",
        type=int,
        default=None,
        help="last line to return, 1-based inclusive (default: end of file)",
    )
    parser.add_argument(
        "--repository-root",
        dest="repository_root",
        default=None,
        help="optional override: local checkout to read (testing and "
        "standalone use only — agents never provide this; default: pinned "
        "snapshot at <output>/<run>/repository)",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="application context: artifact output root "
        "(default: <project>/artifacts)",
    )
    args = parser.parse_args(argv)

    arguments: Dict[str, Any] = {"path": args.path}
    if args.line_start is not None:
        arguments["line_start"] = args.line_start
    if args.line_end is not None:
        arguments["line_end"] = args.line_end

    result = read_file(
        arguments,
        repository_root=Path(args.repository_root) if args.repository_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 6


if __name__ == "__main__":
    sys.exit(main())
