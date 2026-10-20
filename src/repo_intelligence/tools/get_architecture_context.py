"""Tool 1 (read-only): get_architecture_context.

Retrieves the already-generated architecture-context artifact:

    artifacts/<run-id>/architecture_context/architecture_context.md

and returns it inside the generic ToolResult protocol. The current run is
resolved through the existing artifact/run mechanism (never a hard-coded
run id).

Bounded, targeted output (this tool's contract for bounded tool
responses):

* the DEFAULT response is a compact heading index — the document's
  real headings, in document order, with their levels, plus a hint
  for requesting one section — never the whole document (headings
  inside fenced code blocks are code, never headings);
* ``section`` targets one Markdown heading block instead of the
  index (deterministic heading match, no retrieval), subject to the
  same response ceiling;
* no selection (index or section) is ever bigger than the shared
  response ceiling (``MAX_TOOL_RESPONSE_CHARS`` on the serialized
  ``ToolResult``) — a truncated response says so explicitly and lists
  the sections still available;
* ``max_chars`` lowers the ceiling for a call;
* ``section_heading``/``available_sections``/``note`` keep provenance
  and continuation visible to the caller.

The tool does NOT regenerate the context, parse the repository, analyze
source code, invoke pipeline analyzers, search, modify files, or call an
LLM. Execution path:

    get_architecture_context({})
        → resolve current architecture-context artifact
        → verify it exists
        → read Markdown → resolve source metadata
        → default: render the heading index / select the section
        → bound the response to the ceiling
        → return ToolResult

Standalone (JSON ToolResult on stdout):

    python -m repo_intelligence.tools.get_architecture_context
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..context.models import (
    ArchitectureContextError,
    default_context_markdown_path,
    display_path,
)
from .protocol import (
    MAX_TOOL_RESPONSE_CHARS,
    TRUNCATION_NOTE_RESERVE,
    ToolError,
    ToolResult,
    bound_result_text,
    response_chars,
)

TOOL_NAME = "get_architecture_context"

# Optional, model-supplied arguments (see _validate_arguments).
_ALLOWED_ARGUMENTS = ("section", "max_chars")

# Bound on how many section titles a failure/truncation may list — the
# hint stays useful without becoming its own oversized observation.
_MAX_SECTION_LIST = 60

# Smallest accepted ``max_chars``: the ceiling must still cover the
# ToolResult envelope plus the reserved truncation note, otherwise no
# context could ever be returned under it.
_MIN_CONTEXT_CHARS = 1_000

# Worst-case length of a truncation note while a bound is computed —
# the placeholder keeps the ceiling a guarantee, not an estimate.
_NOTE_PLACEHOLDER = "." * TRUNCATION_NOTE_RESERVE

# Source metadata parsed from the artifact itself (never re-derived).
_CONTRACT_LINE_RE = re.compile(r"^- Contract: `([^`]+)`\s*$", re.MULTILINE)
_COMMIT_LINE_RE = re.compile(r"^- Pinned commit: `([^`]+)`\s*$", re.MULTILINE)

# Markdown headings: 1-6 hashes, whitespace, title (the whole line is
# kept as the section heading so provenance is exact). Matched per
# line, outside fenced code blocks only.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")

# Fenced code blocks (CommonMark-style, deliberately small): a fence
# opens with 3+ backticks/tildes near a line start (at most three
# leading spaces) and closes on a line of the SAME character with at
# least as many characters and nothing else after it. A heading-
# looking line inside a fence is code, never a Markdown heading.
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")


def _failure(code: str, message: str) -> ToolResult:
    return ToolResult(
        success=False,
        tool=TOOL_NAME,
        error=ToolError(code=code, message=message),
    )


def _validate_arguments(
    arguments: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[ToolResult]]:
    """Validate the (entirely optional) input contract.

    ``{}`` / ``None`` keep the default heading-index behavior; the two
    accepted keys are targeting and sizing only — never a run id,
    repository, commit, query, or file path (the artifact location stays
    application context).
    """
    if arguments is None:
        return {"section": None, "max_chars": MAX_TOOL_RESPONSE_CHARS}, None
    if not isinstance(arguments, Mapping):
        return None, _failure(
            "INVALID_ARGUMENTS",
            "{} arguments must be a mapping; got {!r}.".format(
                TOOL_NAME, type(arguments).__name__
            ),
        )

    unknown = [str(key) for key in arguments if key not in _ALLOWED_ARGUMENTS]
    if unknown:
        return None, _failure(
            "INVALID_ARGUMENTS",
            "unexpected argument(s): {}.".format(", ".join(sorted(unknown))),
        )

    section = arguments.get("section", None)
    if section is not None:
        if not isinstance(section, str) or not section.strip():
            return None, _failure(
                "INVALID_ARGUMENTS", "section must be a non-empty string."
            )
        section = section.strip()

    max_chars = arguments.get("max_chars", None)
    if max_chars is None:
        limit = MAX_TOOL_RESPONSE_CHARS
    elif isinstance(max_chars, bool) or not isinstance(max_chars, int):
        return None, _failure(
            "INVALID_ARGUMENTS", "max_chars must be an integer."
        )
    elif max_chars < _MIN_CONTEXT_CHARS or max_chars > MAX_TOOL_RESPONSE_CHARS:
        return None, _failure(
            "INVALID_ARGUMENTS",
            "max_chars must be between {} and {}.".format(
                _MIN_CONTEXT_CHARS, MAX_TOOL_RESPONSE_CHARS
            ),
        )
    else:
        limit = max_chars

    return {"section": section, "max_chars": limit}, None


# ---------------------------------------------------------------------------
# Section selection (deterministic Markdown heading targeting)
# ---------------------------------------------------------------------------


def _heading_blocks(text: str) -> List[Tuple[int, str, str, int, int]]:
    """Every real heading as ``(level, title, line, start, end)``.

    A block runs from its heading line up to (but excluding) the next
    heading of the same or a higher level, or the end of the document.

    Fenced code (``` or ~~~) is scanned first: heading-looking lines
    inside a fence are code, so a ``# comment`` in a snippet can
    neither create a section nor terminate a real one. Offsets are
    positions in the ORIGINAL text, so section bodies stay exact
    slices.
    """
    spans: List[Tuple[int, str, str, int]] = []
    fence_char: Optional[str] = None
    fence_len = 0
    offset = 0
    for raw_line in text.split("\n"):
        body = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if fence_char is None:
            open_match = _FENCE_OPEN_RE.match(body)
            if open_match is not None:
                marker = open_match.group(1)
                info = open_match.group(2)
                # backtick fence info strings may not contain backticks
                if marker[0] != "`" or "`" not in info:
                    fence_char = marker[0]
                    fence_len = len(marker)
            else:
                heading = _HEADING_RE.match(body)
                if heading is not None:
                    spans.append(
                        (
                            len(heading.group(1)),
                            heading.group(2).strip(),
                            heading.group(0),
                            offset,
                        )
                    )
        else:
            close_match = _FENCE_CLOSE_RE.match(body)
            if (
                close_match is not None
                and close_match.group(1)[0] == fence_char
                and len(close_match.group(1)) >= fence_len
            ):
                fence_char = None
        offset += len(raw_line) + 1

    blocks: List[Tuple[int, str, str, int, int]] = []
    for index, (level, title, line, start) in enumerate(spans):
        end = len(text)
        for following in spans[index + 1 :]:
            if following[0] <= level:
                end = following[3]
                break
        blocks.append((level, title, line, start, end))
    return blocks


def _normalize_title(title: str) -> str:
    return title.strip().casefold()


def _section_titles(text: str) -> List[str]:
    titles: List[str] = []
    for _level, title, _line, _start, _end in _heading_blocks(text):
        titles.append(title)
    return titles


# Retrieval hint appended to the default heading index: tells the
# caller how to turn the index into one section without pre-emptively
# shipping every section body.
_INDEX_HINT = (
    'next: {"section": "<title from the list above>"} returns that '
    "section (case-insensitive exact heading match)."
)

# Replaces the heading list when the artifact has no real heading at
# all (still a valid, deterministic index).
_NO_HEADINGS_LINE = "(no Markdown headings found)"


def _render_index(text: str) -> str:
    """Compact heading index of ``text`` — the default response.

    One original heading line per real heading, in document order
    (the leading ``#``s carry the level), then the retrieval hint.
    No section bodies and no invented summaries: a deterministic
    orientation list the caller narrows with ``section``. Headings
    inside fenced code are already excluded by ``_heading_blocks``.
    """
    lines = [
        line
        for _level, _title, line, _start, _end in _heading_blocks(text)
    ]
    if not lines:
        lines.append(_NO_HEADINGS_LINE)
    lines.append("")
    lines.append(_INDEX_HINT)
    return "\n".join(lines)


def _select_section(
    text: str, section: str
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolve ``section`` against the document's headings.

    Returns ``(selection, heading_line, titles)``: ``selection`` and
    ``heading_line`` are ``None`` when nothing matched (the caller
    reports ``SECTION_NOT_FOUND`` with ``titles``). Matching is a case-
    insensitive exact comparison of the heading title after ``#``s and
    whitespace are trimmed — no fuzzy scoring, no ranking.
    """
    titles = _section_titles(text)
    requested = section.strip()
    if requested.startswith("#"):
        requested = requested.lstrip("#").strip()
    wanted = _normalize_title(requested)
    for _level, title, line, start, end in _heading_blocks(text):
        if _normalize_title(title) == wanted:
            return text[start:end], line, titles
    return None, None, titles


# ---------------------------------------------------------------------------
# Truncation note (explicit, actionable)
# ---------------------------------------------------------------------------


def _truncation_note(
    limit: int, bounded_text: str, partial: bool, total_chars: int
) -> str:
    """Plain-ASCII note (no quotes/backslashes) — its worst-case
    length is reserved while the bound is computed, so it never makes
    the payload exceed the ceiling."""
    if not bounded_text:
        return (
            "response truncated: a {}-character ceiling leaves no room "
            "for context.".format(limit)
        )
    if partial:
        return (
            "response truncated: only the first {} characters of this "
            "selection's first line fit under a {}-character ceiling."
            .format(len(bounded_text), limit)
        )
    return (
        "response truncated: {} of {} characters returned under a "
        "{}-character ceiling.".format(
            len(bounded_text), total_chars, limit
        )
    )


def get_architecture_context(
    arguments: Optional[Mapping[str, Any]] = None,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
) -> ToolResult:
    """Return the current run's architecture-context artifact.

    Input contract (all keys optional)::

        {}                        # compact heading index (default)
        {"section": "Technology"} # one Markdown heading block
        {"section": "Architecture", "max_chars": 4000}

    The default response is a heading index — the document's real
    headings in order plus a retrieval hint — not the whole document.
    ``section`` selects a heading block deterministically (case-
    insensitive exact heading match; an absent section is the structured
    ``SECTION_NOT_FOUND`` failure naming the available titles).
    ``max_chars`` may lower — never raise — the shared response ceiling
    ``MAX_TOOL_RESPONSE_CHARS``.

    ``output_root`` and ``snapshot_id`` are application context, not
    model-supplied tool arguments. Supplying the pinned snapshot avoids
    ambiguous discovery when several analysis runs exist under the
    artifact root.
    """
    normalized, error = _validate_arguments(arguments)
    if error is not None:
        return error
    assert normalized is not None
    section: Optional[str] = normalized["section"]
    limit: int = normalized["max_chars"]

    # --- resolve current architecture-context artifact (existing run
    # mechanism: <output>/<snapshot_id>/architecture_context/...) ---
    try:
        artifact_path = default_context_markdown_path(
            output_root, snapshot_id=snapshot_id
        )
    except ArchitectureContextError as exc:
        return _failure("CONTEXT_NOT_FOUND", str(exc))

    # --- verify the artifact exists (read-only; never created here) ---
    if not artifact_path.is_file():
        return _failure(
            "CONTEXT_NOT_FOUND",
            "Architecture context artifact was not found: {}".format(
                display_path(artifact_path)
            ),
        )

    # --- read Markdown ---
    try:
        context = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _failure(
            "CONTEXT_NOT_FOUND",
            "Architecture context artifact was not found or could not be "
            "read: {} ({})".format(display_path(artifact_path), exc),
        )
    if not context.strip():
        return _failure(
            "INVALID_CONTEXT",
            "Architecture context artifact is empty: {}".format(
                display_path(artifact_path)
            ),
        )

    # --- resolve source metadata from the WHOLE artifact (never from a
    # section: the contract line lives in the document's top matter) ---
    version_match = _CONTRACT_LINE_RE.search(context)
    if version_match is None:
        return _failure(
            "SOURCE_METADATA_UNAVAILABLE",
            "Architecture context contract line is missing; cannot report "
            "the schema version: {}".format(display_path(artifact_path)),
        )
    commit_match = _COMMIT_LINE_RE.search(context)
    source: Dict[str, Any] = {
        "artifact": display_path(artifact_path),
        "schema_version": version_match.group(1),
        "commit_sha": commit_match.group(1) if commit_match else None,
    }

    # --- selection: DEFAULT renders the compact heading index (targeted
    # orientation, never the whole document); ``section`` selects one
    # heading block (deterministic; absent → structured failure) ---
    heading: Optional[str] = None
    if section is not None:
        selected, heading, titles = _select_section(context, section)
        if selected is None:
            shown = titles[:_MAX_SECTION_LIST]
            if len(titles) > len(shown):
                shown.append("... (+{} more)".format(len(titles) - len(shown)))
            return _failure(
                "SECTION_NOT_FOUND",
                "Section {!r} does not exist in {}; available sections: "
                "{}.".format(
                    section,
                    display_path(artifact_path),
                    ", ".join(shown) if shown else "(none)",
                ),
            )
    else:
        selected = _render_index(context)

    # --- bound the response (index or section, never over the ceiling;
    # truncation is explicit, never silent) ---
    total_chars = len(selected)
    data: Dict[str, Any] = {
        "context": selected,
        "section": section,
        "section_heading": heading,
        "truncated": False,
        "returned_chars": total_chars,
        "total_chars": total_chars,
        "char_limit": limit,
        "available_sections": None,
        "note": None,
    }
    result = ToolResult(
        success=True, tool=TOOL_NAME, data=data, source=source, error=None
    )
    if response_chars(result) <= limit:
        return result

    # Over the ceiling. The bound is computed against a probe that
    # already carries every field the final payload will carry, each at
    # its worst-case length (section list present, note reserved, counts
    # still at their pre-bound maximum) — so the rebound result is
    # guaranteed to fit instead of merely being close. The full text
    # stays on the probe: the helper re-measures with it emptied.
    titles = _section_titles(context)[:_MAX_SECTION_LIST]
    probe: Dict[str, Any] = dict(data)
    probe.update(truncated=True, available_sections=titles,
                 note=_NOTE_PLACEHOLDER)

    def _probe_result(payload: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True, tool=TOOL_NAME, data=payload, source=source,
            error=None,
        )

    sizing = dict(probe)
    sizing["context"] = ""
    if response_chars(_probe_result(sizing)) > limit:
        # the section hint itself would break the ceiling — leave it out
        probe["available_sections"] = None
    bounded = bound_result_text(_probe_result(probe), "context", limit)

    data = dict(data)
    data.update(
        context=bounded.text,
        truncated=True,
        returned_chars=len(bounded.text),
        available_sections=probe["available_sections"],
        note=_truncation_note(
            limit, bounded.text, bounded.partial_line, total_chars
        ),
    )
    return ToolResult(
        success=True, tool=TOOL_NAME, data=data, source=source, error=None
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Standalone entry point: print the ToolResult as JSON on stdout."""
    parser = argparse.ArgumentParser(
        prog="python -m repo_intelligence.tools.get_architecture_context",
        description=(
            "Read-only tool: retrieve the current run's "
            "architecture_context.md (default: compact heading index; "
            "optionally one section — bounded to a response ceiling) "
            "and print its ToolResult as JSON."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Artifact output root (default: <project>/artifacts)",
    )
    parser.add_argument(
        "--section",
        default=None,
        help="optional Markdown heading to return instead of the default "
        "heading index (matched case-insensitively against heading titles)",
    )
    parser.add_argument(
        "--max-chars",
        dest="max_chars",
        type=int,
        default=None,
        help="lower the response ceiling for this call, 1-{}".format(
            MAX_TOOL_RESPONSE_CHARS
        ),
    )
    args = parser.parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    arguments: Dict[str, Any] = {}
    if args.section is not None:
        arguments["section"] = args.section
    if args.max_chars is not None:
        arguments["max_chars"] = args.max_chars
    result = get_architecture_context(arguments, output_root=output_root)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 6


if __name__ == "__main__":
    sys.exit(main())
