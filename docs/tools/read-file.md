# Tool 3 — `read_file`

Read-only tool layer above the deterministic M1–M6 pipeline. Part of the
project's learning architecture: the third basic repository
evidence tool, completed before introducing an actual agent/tool-calling
loop.

```text
M1 Repository Snapshot → M2 Repository Facts → M3 Engineering Model
        → M4 Architecture Model → M5 Diagram Projection
        → M6 Architecture Context
                ↓
Tool 1: get_architecture_context()    high-level orientation
Tool 2: search_repository()           candidate source evidence
Tool 3: read_file()                   exact source evidence   ← this tool
                ↓
             ToolResult
                ↓
LLM layer (M7) + agent (M8) — one-shot tool dispatch: M8.3 (multi-step loop: M8.4 implemented)
```

---

## 22.1 Purpose

A future agent locates interesting evidence with `search_repository`
("docs/workflow-diagrams.md around lines 42–55"), but it still needs
the **exact text** to reason over.

`read_file` retrieves exactly that: given a repository-relative file
path, it returns the file's content from the **pinned repository
snapshot** of the current run — the exact checkout persisted by M1 at
`<output>/<snapshot_id>/repository/` and frozen at the recorded commit —
optionally restricted to a line range, inside the same generic
`ToolResult` every other tool returns.

The path resolves through a fixed chain, never through a caller-chosen
checkout:

```text
current run → pinned repository snapshot → requested repository path
```

It uses the **same repository resolution as `search_repository`** (the
current run's pinned `<output>/<run>/repository/`, application-owned —
not a tool argument), so a candidate location found by Tool 2 always
refers to the same frozen snapshot Tool 3 reads.

The tool solves one problem: **show the exact content of this file
(possibly these lines)** — nothing more.

## 22.2 Responsibility

> `read_file` retrieves exact text evidence from a repository file.

It is the last step of the evidence workflow:

```text
get_architecture_context()         high-level orientation
        ↓
search_repository()                candidate source evidence
        ↓
read_file()                        exact source evidence
        ↓
repository evidence
```

The tools only provide evidence. The (future) agent performs the
reasoning.

## 22.3 What it does not do

`read_file` does **not**:

- interpret the file (no relationships, classifications, or claims);
- summarize or condense content;
- search for related files or other evidence;
- infer architecture or answer questions;
- call an LLM, embeddings, a vector database, or RAG;
- call other tools (`read_file` never invokes `search_repository`, and
  vice versa — the future agent composes them);
- modify, create, or delete anything — strictly read-only;
- regenerate repository artifacts or execute repository code.

If a file contains `The onboarding agent triggers the DAG agent.`, the
tool returns that line as text — not a `TRIGGERS` relationship object.
Interpretation belongs to a reasoning layer, not to this tool.

## 22.4 Input

```json
{
  "path": "docs/workflow.md",
  "line_start": 42,
  "line_end": 55
}
```

| Argument | Required | Type | Meaning |
| --- | --- | --- | --- |
| `path` | yes | string | repository-relative file path, non-empty, no traversal |
| `line_start` | no | integer ≥ 1 | first line to return (1-based) |
| `line_end` | no | integer ≥ `line_start` | last line to return (1-based inclusive) |

- **Both line bounds omitted** → the complete file.
- **Only `line_start`** → from that line through the end of the file.
- **Only `line_end`** → from the beginning through that line.
- Unknown arguments are rejected (`INVALID_ARGUMENT`) — the contract is
  exactly these three keys.
- `repository_root` and `output_root` are application context (the
  pinned snapshot checkout and the artifact output root), not tool
  arguments. The default is always the pinned snapshot of the current
  run; the explicit `repository_root` exists only as a testing /
  standalone override (agents never provide it).

Python API:

```python
from repo_intelligence.tools.read_file import read_file

# normal usage: resolves the pinned snapshot of the current run
result = read_file(
    {"path": "docs/workflow.md", "line_start": 42, "line_end": 55}
)

# testing / standalone override only
result = read_file(
    {"path": "docs/workflow.md", "line_start": 42, "line_end": 55},
    repository_root=Path("/path/to/checkout"),
)
```

## 22.5 Output

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

### Success

```json
{
  "success": true,
  "tool": "read_file",
  "data": {
    "path": "docs/workflow.md",
    "line_start": 42,
    "line_end": 55,
    "content": "...",
    "truncated": false,
    "returned_chars": 1874,
    "total_chars": 1874,
    "char_limit": 24000,
    "next_line_start": null,
    "partial_line": false,
    "note": null
  },
  "source": {
    "repository": "<repository name>",
    "commit_sha": "<snapshot commit>",
    "snapshot_id": "<run/snapshot id>",
    "repository_url": "<repository url>"
  },
  "error": null
}
```

| Field | Meaning |
| --- | --- |
| `path` | the normalized repository-relative path that was read |
| `line_start` / `line_end` | the **actual** 1-based inclusive range of the returned lines — never what the caller requested, always what was delivered (see the ceiling rules below for truncation) |
| `content` | the exact returned text |
| `truncated` | `true` when the response ceiling cut the selection (see below) |
| `returned_chars` / `total_chars` | characters delivered vs. characters the selection had |
| `char_limit` | the response ceiling in characters (`MAX_TOOL_RESPONSE_CHARS`, 24000) |
| `next_line_start` | line to resume from after a whole-line cut (`null` when nothing more to read) |
| `partial_line` | `true` when the content is a PREFIX of one line (no complete line fit) |
| `note` | `null` normally; otherwise the plain-text truncation note |

Range reporting rules:

- full-file read → `line_start = 1`, `line_end = <final line number>`;
- a range that runs past the end of the file is **clamped** (the
  reported `line_end` is the file's last line);
- no lines returned (empty file, or `line_start` past the end) →
  `line_start = 1`, `line_end = 0`, `content = ""`.

**Response ceiling.** One serialized result never exceeds
`char_limit` = 24,000 characters (envelope included). When the
selection is larger:

- the cut is at a **whole-line boundary** whenever any line fits —
  `line_start`/`line_end` cover exactly the complete lines delivered,
  and `next_line_start` names the line to resume from (walking those
  windows reassembles the file, each line delivered once);
- when not even one complete line fits, `partial_line` is `true`,
  the reported range is **empty** (`line_end = line_start - 1`) so
  evidence coverage can never over-claim a line that was only partly
  read, and `next_line_start` is `null` (re-reading that line hits
  the same ceiling);
- `truncated`, `returned_chars`/`total_chars` and `note` always state
  what happened — truncation is explicit, never silent.

The caller never has to guess what content was returned.

### Failure

```json
{
  "success": false,
  "tool": "read_file",
  "data": null,
  "source": null,
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "Requested file was not found: 'docs/workflow.md'"
  }
}
```

## 22.6 Line numbers and newlines

The public contract is **1-based and inclusive**:

```text
1: alpha
2: beta
3: gamma
4: delta
5: epsilon
```

`{"path": "example.txt", "line_start": 2, "line_end": 4}` returns
exactly:

```text
beta
gamma
delta
```

with `line_start = 2`, `line_end = 4`. Python's zero-based indexing is
never exposed.

Newline policy (documented, deterministic):

- lines are split on `\n`; one trailing newline terminates the last
  line instead of creating an extra empty line; a `\r` stays attached
  to its line, so CRLF files round-trip unchanged;
- **full-file reads** return the file's exact bytes decoded as text
  (including a final newline, if present);
- **range reads** return exactly the selected lines joined with `\n`
  (no trailing newline is added);
- indentation, whitespace inside lines, and content are never
  reformatted, stripped, or normalized.

## 22.7 File policy: binary, encoding, size

The tool reuses the file policy established by Tool 2
(`search_repository`) — one policy for the whole tool layer:

| Policy | Rule | Error |
| --- | --- | --- |
| Size | files larger than `1,000,000` bytes are refused (checked via `stat`, before reading) | `FILE_TOO_LARGE` |
| Binary | a NUL byte in the first 8,192 bytes means "not a text file" | `BINARY_FILE` |
| Encoding | strict UTF-8 decode — invalid bytes are refused, never replaced or corrupted | `UNSUPPORTED_ENCODING` |

There is no encoding sniffing and no streaming: plain `stat` → `read`
→ decode. A file that cannot be opened or read (permissions, symlink
loops, …) returns `READ_FAILED`.

Unlike `search_repository` — whose walk indexes only the whitelisted
text extensions (M18 §14) — `read_file` has **no
extension whitelist**: an explicitly named file is read whatever its
extension (asking for `data/sample.csv` reads it).

## 22.8 Path safety

Only repository-contained files can be read:

- `path` must be repository-relative: absolute paths (`/etc/passwd`),
  drive-letter paths (`C:\secret.txt`), and `..` segments
  (`../../file`) are rejected as `PATH_OUTSIDE_REPOSITORY` before any
  filesystem access;
- separators are normalized (`\` → `/`);
- every directory component must exist and be a directory
  (`PATH_NOT_FOUND`);
- the target is **resolved** (symlinks followed) and must stay inside
  the resolved repository root — a symlink inside the checkout that
  points outside is rejected (`PATH_OUTSIDE_REPOSITORY`), and its
  content is never read;
- the target must be a regular file: directories and other non-regular
  files return `NOT_A_FILE` (device/fifo-style files are never opened).

Resolution is real path resolution, not a string-prefix check.

## 22.9 Errors

Only these codes are implemented:

| Code | When |
| --- | --- |
| `INVALID_ARGUMENT` | arguments are not a mapping; missing/non-string/empty `path`; `line_start`/`line_end` not integers ≥ 1; `line_end < line_start`; unknown argument keys |
| `PATH_OUTSIDE_REPOSITORY` | absolute path, drive letter, `..` segment, or a resolution that escapes the repository root |
| `PATH_NOT_FOUND` | a directory component of the path is missing or is not a directory |
| `FILE_NOT_FOUND` | the requested file does not exist |
| `NOT_A_FILE` | the path exists but is a directory (or not a regular file) |
| `BINARY_FILE` | NUL byte detected in the read window |
| `FILE_TOO_LARGE` | file exceeds the 1,000,000-byte limit |
| `UNSUPPORTED_ENCODING` | content is not valid UTF-8 |
| `READ_FAILED` | an OS error prevented reading (permissions, symlink loop, …) |
| `REPOSITORY_NOT_FOUND` | no checkout at the explicit `repository_root`, and no run-convention default available (same repository resolution as Tool 2) |

Errors are structured `{code, message}` values — raw stack traces are
never normal tool output. Argument validation happens before any file
access.

The response ceiling (22.5) is **not** an error: an oversized
selection is returned truncated with `truncated: true` and a `note`,
never rejected.

## 22.10 Determinism

For the same repository state and the same request
(`path`, `line_start`, `line_end`), the tool always returns the same
content:

- no LLM, no network, no randomness;
- no timestamps or nondeterministic fields in `data`;
- content depends only on the file's bytes and the requested range.

This is an evidence-retrieval primitive: repeated reads of the same
file are byte-identical and safe to cite.

## 22.11 Provenance

The `source` block ties content back to the analyzed snapshot:

| Field | Meaning |
| --- | --- |
| `repository` | repository name from the current run's M1 `repository_snapshot.json` |
| `commit_sha` | pinned commit from the same snapshot artifact — equals the git HEAD of the persisted `<run>/repository/` checkout |
| `snapshot_id` | run/snapshot identifier the content was resolved from |
| `repository_url` | repository URL recorded by M1 |

- Provenance comes only from existing artifacts — the tool never runs
  Git commands, never clones, and never derives a different commit
  (nothing is hard-coded; tests resolve it from the fixture snapshot).
- When the run snapshot is unavailable, `source` is
  `{"repository": null, "commit_sha": null, "snapshot_id": null,
  "repository_url": null}` and the read still succeeds ("when
  available", not an error).

This lets a future agent distinguish evidence from a specific
repository state rather than an unversioned working tree.

## 22.12 Running the tool independently

Standalone command (prints the `ToolResult` as JSON on stdout; exit code
`0` on success, `6` on failure):

```bash
python -m repo_intelligence.tools.read_file \
  --path "docs/workflow.md"
```

With a line range:

```bash
python -m repo_intelligence.tools.read_file \
  --path "docs/workflow.md" \
  --line-start 42 \
  --line-end 55
```

Application-context options follow the same convention as Tool 2
(`--output-root` selects the artifact root used for provenance and the
`<output>/<run>/repository` pinned-snapshot default; `--repository-root`
is a testing/standalone override of the checked-out files that agents
never provide):

```bash
python -m repo_intelligence.tools.read_file \
  --path "docs/workflow.md" \
  --output-root /path/to/artifacts

# testing / standalone override only
python -m repo_intelligence.tools.read_file \
  --path "docs/workflow.md" \
  --repository-root /path/to/local/checkout
```

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/tools/test_read_file.py -q
```

## 22.13 Relationship to the other tools

```text
get_architecture_context   ← implemented (Tool 1) — high-level orientation
search_repository          ← implemented (Tool 2) — candidate source evidence
read_file                  ← implemented (Tool 3, this document) — exact source evidence
query_engineering_graph    ← implemented (Tool 4, M11) — structured graph evidence
        ↓
LLM / agent tool-selection loop   (one-shot dispatch: M8.3; loop: M8.4 implemented)
```

The four tools have intentionally different responsibilities and are
never merged: `read_file` does not search, `search_repository` does not
read files on demand, `query_engineering_graph` returns relationship
evidence instead of text, and none of them calls an LLM. All four
reuse the same `ToolCall` / `ToolResult` protocol from
`src/repo_intelligence/tools/protocol.py`, so the M8.4 agent loop
treats them interchangeably as tool calls:

```text
get_architecture_context  → orient in the repository
search_repository         → locate candidate source evidence
read_file                 → retrieve exact source evidence
query_engineering_graph   → navigate structured graph evidence
```

## 22.14 What comes next

This tool completes the initial read-only repository evidence layer.
The LLM layer (M7) and the minimal agent with tool definitions (M8.1 /
M8.2) now exist above it, M8.3 adds the application-owned tool
dispatcher, M8.4 runs the multi-step loop, and M8.5 bounds it: the
agent can describe these tools to a model, recognize a returned
`ToolCall`, execute it behind the explicit execution budget
(`AgentExecutionConfig.max_tool_calls = 8` by default), append the
`ToolResult` to the conversation as a tool observation, and call the
LLM again until it gives a textual answer — or stop with
`EXECUTION_LIMIT_REACHED` when the budget is exhausted (a tool
requested after exhaustion is **not executed**).

What is **not** implemented yet: planner/memory state, retries, and
multi-agent orchestration — those belong to later learning steps,
after the tool contract, tool results, tool errors, and evidence
provenance are understood. (M8.5 is *bounded execution control*, not
a planner.)

---

All tools reuse the same `ToolCall` / `ToolResult` protocol from
`src/repo_intelligence/tools/protocol.py`.
