# Tool 1 — `get_architecture_context`

Read-only tool layer above the deterministic M1–M6 pipeline. Part of the
project's learning architecture: establish the **tool boundary** before
introducing an LLM-driven agent loop.

```text
M1 Repository Snapshot → M2 Repository Facts → M3 Engineering Model
        → M4 Architecture Model → M5 Diagram Projection
        → M6 Architecture Context
                ↓
Tool 1: get_architecture_context()    high-level orientation   ← this tool
Tool 2: search_repository()           candidate source evidence
Tool 3: read_file()                   exact source evidence
                ↓
             ToolResult
                ↓
LLM layer (M7) + agent (M8) — one-shot tool dispatch: M8.3 (multi-step loop: M8.4 implemented)
```

---

## 12.1 Purpose

The deterministic pipeline already produced a repository-specific
architecture context:

```text
artifacts/<run-id>/architecture_context/architecture_context.md
```

A future agent needs that document as its structured "memory" of the
repository. `get_architecture_context` exposes it through a standardized
`ToolResult`, so an agent runtime can request it the same way it will
later request search results or file contents.

The tool solves one problem: **retrieve the already-generated context
(oriented by its heading index, narrowed by section) and describe where
it came from** — nothing more.

## 12.2 Why this is a tool

| | Normal application function | Agent-accessible tool |
| --- | --- | --- |
| Caller | application code | tool runtime (later: an LLM-driven loop) |
| Contract | arbitrary arguments/return values | fixed input contract + generic `ToolResult` |
| Reasoning | none (or embedded in the app) | none in the tool — reasoning belongs to the future agent |
| Independence | invoked as part of a flow | runnable and testable standalone |

The tool itself contains **no reasoning**: it does not decide *whether*
the context is relevant, does not summarize it, and does not answer
questions about it. It only retrieves (index or one section) and
reports provenance.

## 12.3 Input

```json
{}
```

`get_architecture_context` has **no required arguments** — the input
contract is `{}` plus two optional, model-supplied arguments:

| argument   | type   | default | meaning |
|------------|--------|---------|---------|
| `section`  | string | absent  | one Markdown heading block, matched case-insensitively as an exact heading title (leading `#`s stripped) |
| `max_chars`| int    | `24000` | response ceiling to apply, in characters; may LOWER — never raise — `MAX_TOOL_RESPONSE_CHARS` (bounds: 1000..24000) |

**Default = compact heading index (targeted, not the whole document).**
With no `section`, `data.context` is a deterministic heading index: the
document's real Markdown heading lines in document order (the leading
`#`s carry the level — `#`/`##`/`###` stay distinguishable), a blank
line, then the retrieval hint:

```text
next: {"section": "<title from the list above>"} returns that section (case-insensitive exact heading match).
```

Headings inside fenced code blocks (three-or-more backticks or tildes,
CommonMark rules: same fence character, closing run at least as long
as the opening run) are code, never index entries and never selectable
sections — a `# comment` in a snippet cannot create a section or
terminate a real one. A document with no real heading at all yields
`(no Markdown headings found)` in place of the heading list. The index
is bounded by the same response ceiling as any other response; if the
heading list alone exceeds it, it is cut at whole-line boundaries and
says so (`truncated` + `note`).

- No `repository`, `commit_sha`, `run_id`, `query`, `search_term`, or
  `file_path` arguments in this milestone.
- The current run is selected by the existing application/artifact
  execution context (the same mechanism the context generator uses).
- Supplying any other argument is a predictable failure
  (`INVALID_ARGUMENTS`); an unknown `section` is `SECTION_NOT_FOUND`
  (see 12.8).

Python API:

```python
from repo_intelligence.tools.get_architecture_context import (
    get_architecture_context,
)

result = get_architecture_context({})
# heading index of the artifact (default)
result = get_architecture_context({"section": "Technology"})
# one section, tighter budget:
result = get_architecture_context({"section": "Technology",
                                   "max_chars": 4000})
```

## 12.4 Output

Every tool returns the same generic `ToolResult`
(`src/repo_intelligence/tools/protocol.py`):

```text
ToolResult
├── success   bool
├── tool      str
├── data      dict | null
├── source    dict | null
└── error     {code, message} | null
```

### Success (default call — heading index)

```json
{
  "success": true,
  "tool": "get_architecture_context",
  "data": {
    "context": "# Architecture Context\n## Repository\n## Structure\n...\n\nnext: {\"section\": \"<title from the list above>\"} returns that section (case-insensitive exact heading match).",
    "section": null,
    "section_heading": null,
    "truncated": false,
    "returned_chars": 886,
    "total_chars": 886,
    "char_limit": 24000,
    "available_sections": null,
    "note": null
  },
  "source": {
    "artifact": "artifacts/<run-id>/architecture_context/architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "<repository snapshot commit>"
  },
  "error": null
}
```

The numbers are the pinned sample artifact: an 11,138-character
document answers as an 886-character index. With `section`, `context`
holds that heading block instead, `section` echoes the requested
title, `section_heading` carries the exact heading line
(e.g. `## Evidence / Limitations`), and `total_chars` measures the
selection (the index on a default call, the block on a section call).

`source.commit_sha` is `null` when the artifact does not record a commit
("when available").

**Response ceiling.** One serialized result never exceeds
`char_limit` (default `MAX_TOOL_RESPONSE_CHARS` = 24,000 characters,
envelope included). When the selection is larger, the payload is cut
at a whole-line boundary and says so: `truncated` is `true`,
`returned_chars` < `total_chars`, and `note` states what was kept and
that the selection is incomplete (a single line that cannot fit at
all is reported as a character prefix with a "first line" note —
never as a whole line). `available_sections` lists the heading titles
(capped) when truncation leaves room for the hint — the narrowing
offer for what to target next. The bound is measured against a probe
carrying every final field at its worst-case size, so
`response_chars(result) <= char_limit` is a guarantee, not an
estimate.

### Failure

```json
{
  "success": false,
  "tool": "get_architecture_context",
  "data": null,
  "source": null,
  "error": {
    "code": "CONTEXT_NOT_FOUND",
    "message": "Architecture context artifact was not found."
  }
}
```

`ToolResult` is deliberately **not** architecture-context-specific:
Tool 2 (`search_repository`) returns the same shape with different
`data`.

## 12.5 Execution flow

```text
Agent
  ↓
ToolCall          { tool_name, arguments: {} }   ← {} → heading index
  ↓
get_architecture_context        ← executed by the application/tool runtime
  ↓
resolve current run → verify artifact exists → read Markdown
  → parse source metadata
  → SELECT: no `section` → render the heading index (deterministic,
            fence-aware) | `section` → locate that heading block
  → bound the selection to the response ceiling (truncation explicit)
  ↓
architecture_context.md         (read-only, already on disk)
  ↓
ToolResult
  ↓
Agent
```

The LLM never executes the function itself; the
application/tool runtime executes the call and hands the `ToolResult`
back. The LLM layer (M7) and the agent (M8) exist above the tools; the
M8.3 dispatcher executes exactly one ToolCall per dispatch; the M8.4
Agent loop feeds each ToolResult back as an observation and may dispatch
again. The tool itself still runs standalone.

## 12.6 Source artifact

```text
artifacts/<run-id>/architecture_context/architecture_context.md
```

The tool retrieves an **already-generated** artifact produced by the M6
context generator from the contract
`contract/architecture_context_template.json`. The run id is resolved
through the existing artifact/run mechanism — never hard-coded — so the
tool keeps working when the run changes.

## 12.7 Evidence and provenance

The `source` block ties the result back to its origins:

| Field | Meaning |
| --- | --- |
| `artifact` | the exact run-specific Markdown path (project-relative when inside the project) |
| `schema_version` | parsed from the artifact's `- Contract: \`architecture_context.v1\`` line — identifies the context contract the document follows |
| `commit_sha` | parsed from the artifact's `- Pinned commit:` line — the repository snapshot commit the context describes |

If the contract line is missing, the tool refuses to guess the schema
version and returns `SOURCE_METADATA_UNAVAILABLE`.

## 12.8 Errors

Only these codes are implemented:

| Code | When |
| --- | --- |
| `INVALID_ARGUMENTS` | an argument outside the contract (`section`, `max_chars`) is supplied, `section` is not a non-empty string, or `max_chars` is not an integer in 1000..24000 |
| `SECTION_NOT_FOUND` | `section` matches no heading; the message lists the available heading titles (capped) |
| `CONTEXT_NOT_FOUND` | current run cannot be resolved (no/ambiguous run under the output root), or the artifact is missing/unreadable |
| `INVALID_CONTEXT` | artifact exists but is empty |
| `SOURCE_METADATA_UNAVAILABLE` | artifact has no contract line, so the schema version cannot be reported |

Errors are structured `{code, message}` values — raw stack traces are
never normal tool output.

## 12.9 What the tool does NOT do

- It does **not** search the repository (that is Tool 2:
  [search-repository.md](search-repository.md)).
- It does **not** read arbitrary files (that is Tool 3:
  [read-file.md](read-file.md)).
- It does **not** regenerate the architecture context or invoke any
  M1–M4 analyzer.
- It does **not** parse repository documents, source code, or Git
  history.
- It does **not** reason about architecture or answer questions.
- It does **not** summarize: the default index lists the document's
  real heading lines verbatim (plus one fixed hint line) — no
  generated prose or extracted claims.
- It does **not** call an LLM.
- It does **not** modify any file — strictly read-only.

```text
Context Generator   = creates architecture_context.md
get_architecture_context = retrieves architecture_context.md
```

## 12.10 Relationship to the other tools

The established convention (`src/repo_intelligence/tools/`,
`tests/tools/`, `docs/tools/`) is followed by:

```text
get_architecture_context   ← implemented (Tool 1, this document) — high-level orientation
search_repository          ← implemented (Tool 2, search-repository.md) — candidate source evidence
read_file                  ← implemented (Tool 3, read-file.md) — exact source evidence
        ↓
LLM / agent tool-selection loop   (one-shot dispatch: M8.3; loop: M8.4 implemented)
```

All of them reuse the same `ToolCall` / `ToolResult` protocol from
`src/repo_intelligence/tools/protocol.py`.

---

## Running the tool independently

Standalone command (prints the `ToolResult` as JSON on stdout; exit code
`0` on success, `6` on failure):

```bash
python -m repo_intelligence.tools.get_architecture_context
```

Optional application-context override (not a tool argument — it selects
the artifact output root):

```bash
python -m repo_intelligence.tools.get_architecture_context \
  --output-root <path-to-artifacts-root>
```

Optional tool arguments on the CLI (same contract as the tool):

```bash
python -m repo_intelligence.tools.get_architecture_context \
  --section Technology --max-chars 4000
```

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/tools/ -q
```
