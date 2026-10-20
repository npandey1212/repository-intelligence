"""Focused tests for Tool 1: get_architecture_context (read-only).

Tool-layer tests are intentionally separate from the deterministic M1–M6
foundation tests: they exercise the generic ToolResult protocol, the
tool's retrieval/error contracts, run-independence, and read-only
behavior against temporary/synthetic artifact locations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo_intelligence.tools import ToolCall, ToolError, ToolResult
from repo_intelligence.tools.get_architecture_context import (
    TOOL_NAME,
    _MAX_SECTION_LIST,
    _MIN_CONTEXT_CHARS,
    get_architecture_context,
    main,
)
from repo_intelligence.tools.protocol import (
    MAX_TOOL_RESPONSE_CHARS,
    TRUNCATION_NOTE_RESERVE,
    response_chars,
)

TOOLS_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "tools"

# Repository/run-specific literals that must never appear in tool source.
FORBIDDEN_TOOL_SOURCE_TERMS = (
    "main_2026-09-22",
    "ef11dac",
    "PRIMARY_SNAPSHOT_ID",
    "ADOP",
    "adop",
    "sample-Agentic-Ai-Data-Operations",
    "data-onboarding",
    "aws-samples",
    "aws",
    "airflow",
    "glue",
    "mcp",
    ".claude",
    "s3",
)

COMMIT_SHA = "a" * 40

VALID_CONTEXT = """# Architecture Context

- Contract: `architecture_context.v1`

## Repository

- Name: `sample-repo-alpha`
- Pinned commit: `{commit}`
- Snapshot: `run_fixture_alpha`

## Evidence / Limitations

### Sources

- Architecture context template: contract/architecture_context_template.json
""".format(
    commit=COMMIT_SHA
)

NO_CONTRACT_CONTEXT = """# Something else

- Pinned commit: `{}`
""".format(
    COMMIT_SHA
)

NO_COMMIT_CONTEXT = """# Architecture Context

- Contract: `architecture_context.v1`

## Repository

- Name: `sample-repo-alpha`
"""

# The default response's retrieval hint — pinned verbatim as part of
# the index contract (the docs state the same sentence).
_INDEX_HINT_LINE = (
    'next: {"section": "<title from the list above>"} returns that '
    "section (case-insensitive exact heading match)."
)

# Heading lines of VALID_CONTEXT, in document order (levels intact).
_VALID_HEADINGS = (
    "# Architecture Context",
    "## Repository",
    "## Evidence / Limitations",
    "### Sources",
)


def _expected_index(*heading_lines: str) -> str:
    """The exact default response for the given heading lines."""
    return "\n".join(list(heading_lines) + ["", _INDEX_HINT_LINE])


def _write_run(
    root: Path,
    run_id: str,
    context_text: str = VALID_CONTEXT,
    write_context: bool = True,
) -> Path:
    """Create the minimal existing-run layout the artifact mechanism uses."""
    run_dir = root / run_id
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    if write_context:
        context_dir = run_dir / "architecture_context"
        context_dir.mkdir()
        (context_dir / "architecture_context.md").write_text(
            context_text, encoding="utf-8"
        )
    return run_dir


def _context_path(root: Path, run_id: str) -> Path:
    return (
        root
        / run_id
        / "architecture_context"
        / "architecture_context.md"
    )


# ---------------------------------------------------------------------------
# Generic ToolResult protocol (shared by all future tools)
# ---------------------------------------------------------------------------


def test_tool_result_protocol_is_generic():
    result = ToolResult(
        success=True,
        tool="some_future_tool",
        data={"anything": [1, 2, 3]},
        source={"artifact": "somewhere"},
        error=None,
    )
    assert list(result.__dataclass_fields__) == [
        "success",
        "tool",
        "data",
        "source",
        "error",
    ]
    assert result.to_dict() == {
        "success": True,
        "tool": "some_future_tool",
        "data": {"anything": [1, 2, 3]},
        "source": {"artifact": "somewhere"},
        "error": None,
    }


def test_tool_call_and_error_shapes():
    call = ToolCall(tool_name="get_architecture_context", arguments={})
    assert call.tool_name == "get_architecture_context"
    assert call.arguments == {}
    error = ToolError(code="CONTEXT_NOT_FOUND", message="gone")
    assert error.to_dict() == {
        "code": "CONTEXT_NOT_FOUND",
        "message": "gone",
    }


# ---------------------------------------------------------------------------
# Successful retrieval
# ---------------------------------------------------------------------------


def test_successful_retrieval(tmp_path):
    _write_run(tmp_path, "run_alpha")
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is True
    assert result.tool == TOOL_NAME == "get_architecture_context"
    assert result.error is None
    assert result.source is not None
    # default = compact heading index, not the document body
    assert result.data["context"] == _expected_index(*_VALID_HEADINGS)
    assert "- Name: `sample-repo-alpha`" not in result.data["context"]
    assert result.data["section"] is None
    assert result.data["section_heading"] is None


def test_source_identifies_run_specific_artifact(tmp_path):
    run_dir = _write_run(tmp_path, "run_alpha")
    result = get_architecture_context({}, output_root=tmp_path)
    artifact = result.source["artifact"]
    assert artifact.endswith(
        "run_alpha/architecture_context/architecture_context.md"
    )
    expected = run_dir / "architecture_context" / "architecture_context.md"
    assert Path(artifact) == expected
    assert Path(artifact).is_file()


def test_source_reports_schema_version(tmp_path):
    _write_run(tmp_path, "run_alpha")
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.source["schema_version"] == "architecture_context.v1"


def test_source_reports_commit_when_available(tmp_path):
    _write_run(tmp_path, "run_alpha")
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is True
    assert result.source["commit_sha"] == COMMIT_SHA


def test_source_commit_null_when_artifact_has_none(tmp_path):
    _write_run(tmp_path, "run_no_commit", context_text=NO_COMMIT_CONTEXT)
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is True
    assert result.source["commit_sha"] is None
    assert result.source["schema_version"] == "architecture_context.v1"


# ---------------------------------------------------------------------------
# Structured failures
# ---------------------------------------------------------------------------


def test_missing_context_artifact(tmp_path):
    _write_run(tmp_path, "run_alpha", write_context=False)
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is False
    assert result.tool == TOOL_NAME
    assert result.data is None
    assert result.source is None
    assert result.error.code == "CONTEXT_NOT_FOUND"
    assert "not found" in result.error.message
    # the tool never creates/regenerates the artifact
    assert not _context_path(tmp_path, "run_alpha").exists()


def test_unresolvable_run_context(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    result = get_architecture_context({}, output_root=empty_root)
    assert result.success is False
    assert result.error.code == "CONTEXT_NOT_FOUND"
    assert result.data is None
    assert result.source is None


def test_ambiguous_run_context(tmp_path):
    _write_run(tmp_path, "run_one")
    _write_run(tmp_path, "run_two")
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is False
    assert result.error.code == "CONTEXT_NOT_FOUND"
    assert "Multiple" in result.error.message


def test_empty_context_is_invalid(tmp_path):
    _write_run(tmp_path, "run_alpha", context_text="   \n")
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is False
    assert result.error.code == "INVALID_CONTEXT"
    assert result.data is None
    assert result.source is None


def test_missing_contract_line_reports_source_metadata_error(tmp_path):
    _write_run(tmp_path, "run_alpha", context_text=NO_CONTRACT_CONTEXT)
    result = get_architecture_context({}, output_root=tmp_path)
    assert result.success is False
    assert result.error.code == "SOURCE_METADATA_UNAVAILABLE"
    assert result.data is None
    assert result.source is None


def test_unknown_arguments_are_rejected(tmp_path):
    _write_run(tmp_path, "run_alpha")
    for arguments in (
        {"run_id": "run_alpha"},
        {"query": "agents"},
        {"file_path": "README.md"},
        {"section": "Architecture", "unexpected": 1},
    ):
        result = get_architecture_context(arguments, output_root=tmp_path)
        assert result.success is False
        assert result.error.code == "INVALID_ARGUMENTS"
        assert result.data is None

    # no-argument call (input {}) is the established contract
    assert get_architecture_context({}, output_root=tmp_path).success is True
    assert get_architecture_context(None, output_root=tmp_path).success is True
    # the two optional targeting/sizing keys are accepted
    targeted = get_architecture_context(
        {"section": "Repository"}, output_root=tmp_path
    )
    assert targeted.success is True
    assert targeted.data["section"] == "Repository"


# ---------------------------------------------------------------------------
# Repository/run independence + read-only behavior
# ---------------------------------------------------------------------------


def test_run_independence(tmp_path):
    root_one = tmp_path / "root_one"
    root_two = tmp_path / "root_two"
    _write_run(root_one, "first_run")
    other_context = VALID_CONTEXT.replace(
        "sample-repo-alpha", "sample-repo-beta"
    ).replace(COMMIT_SHA, "b" * 40)
    _write_run(root_two, "second_run", context_text=other_context)

    result_one = get_architecture_context({}, output_root=root_one)
    result_two = get_architecture_context({}, output_root=root_two)
    assert result_one.success and result_two.success
    # each run answers from ITS OWN artifact (provenance, not body
    # text: the index is heading-based and both fixtures share one
    # heading structure)
    assert "first_run" in result_one.source["artifact"]
    assert "second_run" in result_two.source["artifact"]
    assert result_one.source["commit_sha"] == COMMIT_SHA
    assert result_two.source["commit_sha"] == "b" * 40
    assert result_one.data["context"] == _expected_index(*_VALID_HEADINGS)


def test_tool_source_has_no_hardcoded_run_or_domain_terms():
    for path in sorted(TOOLS_SRC_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TOOL_SOURCE_TERMS:
            assert term not in text, (
                "hard-coded term {!r} in {}".format(term, path)
            )


def test_tool_is_read_only(tmp_path):
    run_dir = _write_run(tmp_path, "run_alpha")
    artifact = _context_path(tmp_path, "run_alpha")
    before_bytes = artifact.read_bytes()
    before_mtime = artifact.stat().st_mtime_ns
    before_listing = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*"))

    result = get_architecture_context({}, output_root=tmp_path)

    assert result.success is True
    assert artifact.read_bytes() == before_bytes
    assert artifact.stat().st_mtime_ns == before_mtime
    assert (
        sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*"))
        == before_listing
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def test_standalone_cli_success(tmp_path, capsys):
    _write_run(tmp_path, "run_alpha")
    exit_code = main(["--output-root", str(tmp_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["tool"] == "get_architecture_context"
    assert payload["data"]["context"] == _expected_index(*_VALID_HEADINGS)
    assert payload["source"]["schema_version"] == "architecture_context.v1"
    assert payload["error"] is None


def test_standalone_cli_failure_is_structured(tmp_path, capsys):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    exit_code = main(["--output-root", str(empty_root)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 6
    assert payload == {
        "success": False,
        "tool": "get_architecture_context",
        "data": None,
        "source": None,
        "error": {
            "code": "CONTEXT_NOT_FOUND",
            "message": payload["error"]["message"],
        },
    }
    assert payload["error"]["message"]


# ---------------------------------------------------------------------------
# Integration: default project run (skipped when absent)
# ---------------------------------------------------------------------------


def _default_run_available() -> bool:
    from repo_intelligence.context.models import (
        default_context_markdown_path,
    )

    try:
        return default_context_markdown_path().is_file()
    except Exception:
        return False


@pytest.mark.skipif(
    not _default_run_available(),
    reason="no default run with an architecture context artifact",
)
def test_default_artifact_root_retrieval():
    result = get_architecture_context({})
    assert result.success is True
    assert result.source["schema_version"] == "architecture_context.v1"
    assert result.source["artifact"].endswith(
        "/architecture_context/architecture_context.md"
    )
    assert result.source["commit_sha"]
    assert "# Architecture Context" in result.data["context"]


# ---------------------------------------------------------------------------
# Bounded output: response ceiling + section targeting
# ---------------------------------------------------------------------------


def _large_context(section_count: int = 24, filler_lines: int = 60) -> str:
    """A realistic artifact far over the response ceiling.

    ``filler_lines`` sizes each section: the default 60 lines keep one
    section well INSIDE the ceiling (default-index vs section tests);
    250 lines push a single section over it (truncation tests).
    """
    filler = (
        "deterministic filler describing one architecture slice of the "
        "repository under analysis; it exists only to make the artifact "
        "realistically large for ceiling tests."
    )
    lines = [
        "# Architecture Context",
        "",
        "- Contract: `architecture_context.v1`",
        "- Pinned commit: `{}`".format(COMMIT_SHA),
        "",
    ]
    for index in range(section_count):
        lines.append("## Section {:02d}".format(index))
        lines.append("")
        lines.extend("- {}".format(filler) for _ in range(filler_lines))
        lines.append("")
    return "\n".join(lines)


def _write_large_run(root: Path, run_id: str = "run_large") -> Path:
    return _write_run(root, run_id, context_text=_large_context())


def test_default_call_returns_a_compact_index_not_the_document(tmp_path):
    _write_run(tmp_path, "run_alpha")
    result = get_architecture_context({}, output_root=tmp_path)
    data = result.data

    assert result.success is True
    # the heading index, not the document body
    assert data["context"] == _expected_index(*_VALID_HEADINGS)
    assert data["context"].endswith(_INDEX_HINT_LINE)
    assert len(data["context"]) < len(VALID_CONTEXT)
    assert "- Name:" not in data["context"]
    assert data["section"] is None and data["section_heading"] is None
    assert data["truncated"] is False
    assert data["returned_chars"] == data["total_chars"] == len(
        data["context"]
    )
    assert data["char_limit"] == MAX_TOOL_RESPONSE_CHARS
    assert data["available_sections"] is None
    assert data["note"] is None
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    # provenance still identifies the source document
    assert result.source["schema_version"] == "architecture_context.v1"
    assert result.source["artifact"].endswith(
        "architecture_context/architecture_context.md"
    )
    assert result.source["commit_sha"] == COMMIT_SHA


def test_default_index_preserves_heading_levels_and_order(tmp_path):
    _write_run(tmp_path, "run_alpha")
    data = get_architecture_context({}, output_root=tmp_path).data
    lines = data["context"].split("\n")
    heading_lines = lines[:-2]  # drop the blank line + retrieval hint
    # document order, levels visible as the original leading hashes
    assert heading_lines == [
        "# Architecture Context",
        "## Repository",
        "## Evidence / Limitations",
        "### Sources",
    ]
    assert lines[-1] == _INDEX_HINT_LINE
    # hierarchy: major sections (##) distinguished from subsections
    # (###) and from the document title (#)
    assert heading_lines[0].startswith("# ")
    assert heading_lines[1].startswith("## ")
    assert heading_lines[3].startswith("### ")


def test_default_call_on_an_oversized_document_returns_only_the_index(
    tmp_path,
):
    document = _large_context(filler_lines=250)
    _write_run(tmp_path, "run_large", context_text=document)
    result = get_architecture_context({}, output_root=tmp_path)
    data = result.data
    headings = ["# Architecture Context"] + [
        "## Section {:02d}".format(index) for index in range(24)
    ]

    assert result.success is True
    assert len(document) > MAX_TOOL_RESPONSE_CHARS
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is False
    assert data["context"] == _expected_index(*headings)
    assert "deterministic filler" not in data["context"]
    assert data["section"] is None and data["section_heading"] is None
    assert data["available_sections"] is None
    assert data["note"] is None


def test_oversized_section_is_bounded_by_the_ceiling(tmp_path):
    document = _large_context(filler_lines=250)
    _write_run(tmp_path, "run_large", context_text=document)
    result = get_architecture_context(
        {"section": "Section 03"}, output_root=tmp_path
    )
    data = result.data
    section_text = document[
        document.index("## Section 03"): document.index("## Section 04")
    ]

    assert result.success is True
    assert len(section_text) > MAX_TOOL_RESPONSE_CHARS
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is True
    assert data["total_chars"] == len(section_text)
    assert data["returned_chars"] == len(data["context"])
    assert data["returned_chars"] < data["total_chars"]
    assert data["char_limit"] == MAX_TOOL_RESPONSE_CHARS
    assert data["note"] and "truncated" in data["note"]
    assert len(data["note"]) <= TRUNCATION_NOTE_RESERVE
    assert '"' not in data["note"] and "\\" not in data["note"]
    # what to target next is offered, bounded to a fixed number of titles
    assert data["available_sections"]
    assert len(data["available_sections"]) <= _MAX_SECTION_LIST
    # the returned text is an honest prefix of the section
    assert section_text.startswith(data["context"])


def test_truncation_returns_whole_lines_only(tmp_path):
    document = _large_context(filler_lines=250)
    _write_run(tmp_path, "run_large", context_text=document)
    data = get_architecture_context(
        {"section": "Section 03"}, output_root=tmp_path
    ).data
    section_text = document[
        document.index("## Section 03"): document.index("## Section 04")
    ]
    document_lines = section_text.split("\n")
    returned_lines = data["context"].split("\n")
    assert returned_lines == document_lines[: len(returned_lines)]
    assert len(returned_lines) < len(document_lines)


def test_section_selection_returns_only_that_block(tmp_path):
    document = _large_context()
    _write_run(tmp_path, "run_large", context_text=document)
    result = get_architecture_context(
        {"section": "Section 03"}, output_root=tmp_path
    )
    data = result.data

    assert result.success is True
    assert data["truncated"] is False  # one section stays well inside it
    assert data["section"] == "Section 03"
    assert data["section_heading"] == "## Section 03"
    assert data["context"] == document[
        document.index("## Section 03"): document.index("## Section 04")
    ]
    assert data["returned_chars"] == data["total_chars"] == len(
        data["context"]
    )
    assert data["available_sections"] is None
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS


def test_section_matching_is_case_insensitive_and_hash_tolerant(tmp_path):
    _write_large_run(tmp_path)
    baseline = get_architecture_context(
        {"section": "Section 05"}, output_root=tmp_path
    )
    assert baseline.success is True
    for variant in ("section 05", "SECTION 05", "## Section 05"):
        result = get_architecture_context(
            {"section": variant}, output_root=tmp_path
        )
        assert result.success is True, variant
        assert result.data["context"] == baseline.data["context"]
        assert result.data["section"] == variant
        assert result.data["section_heading"] == "## Section 05"


def test_section_stays_complete_even_when_the_document_overflows(tmp_path):
    _write_large_run(tmp_path)
    whole = get_architecture_context({}, output_root=tmp_path)
    targeted = get_architecture_context(
        {"section": "Section 00"}, output_root=tmp_path
    )

    # the default answers with the (small) index even though the
    # document itself is far over the ceiling
    assert len(_large_context()) > MAX_TOOL_RESPONSE_CHARS
    assert whole.data["truncated"] is False
    assert "deterministic filler" not in whole.data["context"]
    assert whole.data["context"].startswith("# Architecture Context")
    assert targeted.success is True
    assert targeted.data["truncated"] is False
    assert targeted.data["context"].startswith("## Section 00")
    assert targeted.data["returned_chars"] == targeted.data["total_chars"]
    assert response_chars(targeted) <= MAX_TOOL_RESPONSE_CHARS


def test_absent_section_lists_the_available_titles(tmp_path):
    _write_large_run(tmp_path)
    result = get_architecture_context(
        {"section": "Missing"}, output_root=tmp_path
    )
    assert result.success is False
    assert result.data is None and result.source is None
    assert result.error.code == "SECTION_NOT_FOUND"
    assert "'Missing'" in result.error.message
    assert "Architecture Context" in result.error.message
    assert "Section 00" in result.error.message
    assert len(result.error.message) < MAX_TOOL_RESPONSE_CHARS


def test_absent_section_hint_is_capped(tmp_path):
    section_count = 80
    _write_run(
        tmp_path, "run_many", context_text=_large_context(section_count)
    )
    result = get_architecture_context(
        {"section": "Missing"}, output_root=tmp_path
    )
    total_titles = section_count + 1  # the document title too
    shown = min(total_titles, _MAX_SECTION_LIST)
    assert result.error.code == "SECTION_NOT_FOUND"
    assert result.error.message.endswith(
        "... (+{} more).".format(total_titles - shown)
    )


def test_max_chars_lowers_the_ceiling(tmp_path):
    _write_run(
        tmp_path, "run_large", context_text=_large_context(filler_lines=250)
    )
    result = get_architecture_context(
        {"section": "Section 00", "max_chars": 2000}, output_root=tmp_path
    )
    data = result.data
    assert data["truncated"] is True
    assert data["char_limit"] == 2000
    assert response_chars(result) <= 2000
    assert data["total_chars"] > data["returned_chars"]


@pytest.mark.parametrize(
    "bad", [0, -1, _MIN_CONTEXT_CHARS - 1, MAX_TOOL_RESPONSE_CHARS + 1, True,
            "4000", 1.5]
)
def test_max_chars_out_of_range_is_rejected(tmp_path, bad):
    _write_run(tmp_path, "run_alpha")
    result = get_architecture_context({"max_chars": bad}, output_root=tmp_path)
    assert result.success is False
    assert result.error.code == "INVALID_ARGUMENTS"
    assert result.error.message.startswith("max_chars")


def test_non_mapping_arguments_are_rejected():
    for arguments in ("Architecture", 42, ["section"]):
        result = get_architecture_context(arguments)
        assert result.success is False
        assert result.error.code == "INVALID_ARGUMENTS"


def test_escaped_content_stays_within_the_ceiling(tmp_path):
    """JSON escaping (quotes, astral characters) must not let the
    serialized observation grow past the ceiling."""
    line = 'say "hello" to the demo \U0001F600 — escaping grows this. '
    lines = [
        "# Architecture Context",
        "",
        "- Contract: `architecture_context.v1`",
        "",
        "## Escaped",
        "",
    ]
    lines.extend("- {}".format(line) for _ in range(2000))
    document = "\n".join(lines)
    _write_run(tmp_path, "run_escaped", context_text=document)

    result = get_architecture_context(
        {"section": "Escaped"}, output_root=tmp_path
    )
    section_text = document[document.index("## Escaped"):]
    assert result.success is True
    assert len(document) > MAX_TOOL_RESPONSE_CHARS
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert result.data["truncated"] is True
    assert section_text.startswith(result.data["context"])


def test_single_line_bigger_than_the_ceiling_is_prefix_bounded(tmp_path):
    """When not even one line fits (a heading line alone exceeds the
    ceiling), the caller is told the line was cut instead of being
    handed a prefix presented as a whole line."""
    giant_line = "# " + "z" * (MAX_TOOL_RESPONSE_CHARS * 2)
    document = giant_line
    document += "\n\n- Contract: `architecture_context.v1`\n"
    _write_run(tmp_path, "run_huge_line", context_text=document)

    result = get_architecture_context({}, output_root=tmp_path)
    data = result.data
    assert result.success is True
    assert data["truncated"] is True
    assert "first line" in data["note"]
    assert data["returned_chars"] == len(data["context"])
    assert data["returned_chars"] < data["total_chars"]
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert document.startswith(data["context"])


def test_section_result_still_reports_document_source_metadata(tmp_path):
    _write_large_run(tmp_path)
    result = get_architecture_context(
        {"section": "Section 23"}, output_root=tmp_path
    )
    assert result.success is True
    assert result.data["section_heading"] == "## Section 23"
    # parsed from the whole artifact, not from the selected section
    assert result.source["schema_version"] == "architecture_context.v1"
    assert result.source["commit_sha"] == COMMIT_SHA


def test_standalone_cli_section_selection(tmp_path, capsys):
    _write_large_run(tmp_path)
    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--section",
            "Section 00",
            "--max-chars",
            "20000",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["data"]["section"] == "Section 00"
    assert payload["data"]["truncated"] is False
    assert payload["data"]["char_limit"] == 20000
    assert payload["data"]["context"].startswith("## Section 00")


# ---------------------------------------------------------------------------
# Targeted default: compact heading index + fence-aware heading parsing
# ---------------------------------------------------------------------------

FENCED_CONTEXT = (
    "# Architecture Context\n"
    "\n"
    "- Contract: `architecture_context.v1`\n"
    "- Pinned commit: `" + COMMIT_SHA + "`\n"
    "\n"
    "## Real Section\n"
    "\n"
    "Some real prose.\n"
    "\n"
    "```python\n"
    "# not a heading - a code comment\n"
    "## also not a heading\n"
    "```\n"
    "\n"
    "## After The Fence\n"
    "\n"
    "Done.\n"
)

TILDE_FENCED_CONTEXT = (
    "# Architecture Context\n"
    "\n"
    "- Contract: `architecture_context.v1`\n"
    "- Pinned commit: `" + COMMIT_SHA + "`\n"
    "\n"
    "~~~\n"
    "## Fake Tilde Section\n"
    "~~~~\n"
    "\n"
    "## Real Section\n"
    "\n"
    "- real body\n"
)


def test_headings_inside_backtick_fences_are_ignored(tmp_path):
    """A `# comment` or `## heading`-looking line inside a ``` fence is
    code: never an index row, never a selectable section, and never a
    terminator for the real section around it."""
    _write_run(tmp_path, "run_fenced", context_text=FENCED_CONTEXT)

    default = get_architecture_context({}, output_root=tmp_path)
    assert default.success is True
    assert default.data["context"] == _expected_index(
        "# Architecture Context",
        "## Real Section",
        "## After The Fence",
    )
    assert "not a heading" not in default.data["context"]

    missing = get_architecture_context(
        {"section": "also not a heading"}, output_root=tmp_path
    )
    assert missing.success is False
    assert missing.error.code == "SECTION_NOT_FOUND"

    section = get_architecture_context(
        {"section": "Real Section"}, output_root=tmp_path
    )
    body = section.data["context"]
    assert section.success is True
    assert body.startswith("## Real Section")
    assert "```python" in body
    assert "# not a heading - a code comment" in body
    assert "## also not a heading" in body
    # the block runs to the NEXT REAL heading, fence content included
    assert "## After The Fence" not in body
    assert "Done." not in body
    assert section.data["section_heading"] == "## Real Section"
    assert response_chars(section) <= MAX_TOOL_RESPONSE_CHARS


def test_headings_inside_tilde_fences_are_ignored(tmp_path):
    _write_run(tmp_path, "run_tilde", context_text=TILDE_FENCED_CONTEXT)

    data = get_architecture_context({}, output_root=tmp_path).data
    assert data["context"] == _expected_index(
        "# Architecture Context",
        "## Real Section",
    )
    assert "Fake Tilde Section" not in data["context"]

    missing = get_architecture_context(
        {"section": "Fake Tilde Section"}, output_root=tmp_path
    )
    assert missing.success is False
    assert missing.error.code == "SECTION_NOT_FOUND"


def test_oversized_index_is_bounded_visibly(tmp_path):
    """When the heading list itself exceeds the ceiling it is bounded
    deterministically — whole lines, explicit flag, honest counts —
    never silently cut."""
    heading_lines = ["# Architecture Context"]
    lines = [
        "# Architecture Context",
        "",
        "- Contract: `architecture_context.v1`",
        "",
    ]
    for index in range(2000):
        title = "Section {:04d}".format(index)
        heading_lines.append("## " + title)
        lines.append("## " + title)
        lines.append("")
    document = "\n".join(lines) + "\n"
    _write_run(tmp_path, "run_index", context_text=document)

    result = get_architecture_context({}, output_root=tmp_path)
    data = result.data
    full_index_lines = heading_lines + ["", _INDEX_HINT_LINE]
    full_index = "\n".join(full_index_lines)
    returned_lines = data["context"].split("\n")

    assert result.success is True
    assert len(full_index) > MAX_TOOL_RESPONSE_CHARS
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS
    assert data["truncated"] is True
    assert data["total_chars"] == len(full_index)
    assert data["returned_chars"] == len(data["context"])
    assert data["returned_chars"] < data["total_chars"]
    # exact whole-line prefix of the deterministic index (order kept)
    assert returned_lines == full_index_lines[: len(returned_lines)]
    assert len(returned_lines) < len(full_index_lines)
    assert data["note"] and "truncated" in data["note"]
    assert len(data["note"]) <= TRUNCATION_NOTE_RESERVE
    # narrowing offer stays bounded to a fixed number of titles
    assert data["available_sections"]
    assert len(data["available_sections"]) <= _MAX_SECTION_LIST
    assert data["section"] is None and data["section_heading"] is None
    assert result.source["schema_version"] == "architecture_context.v1"


def test_document_without_headings_returns_placeholder_index(tmp_path):
    context = (
        "- Contract: `architecture_context.v1`\n"
        "- Pinned commit: `" + COMMIT_SHA + "`\n"
        "\n"
        "No Markdown headings in this artifact.\n"
    )
    _write_run(tmp_path, "run_headless", context_text=context)

    result = get_architecture_context({}, output_root=tmp_path)
    data = result.data
    assert result.success is True
    assert data["context"] == _expected_index(
        "(no Markdown headings found)"
    )
    assert data["truncated"] is False
    assert data["available_sections"] is None
    assert response_chars(result) <= MAX_TOOL_RESPONSE_CHARS

    missing = get_architecture_context(
        {"section": "anything"}, output_root=tmp_path
    )
    assert missing.error.code == "SECTION_NOT_FOUND"
