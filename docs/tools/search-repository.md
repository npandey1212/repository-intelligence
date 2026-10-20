# Tool 2 — `search_repository`

Read-only tool layer over the M18 retrieval stack (SentenceTransformers
/ FAISS / SQLite metadata) with the Engineering Graph as the
deterministic relationship source. Part of the project's learning
architecture: give a future agent runtime a **compact,
provenance-aware way to locate evidence** in the analyzed repository
before introducing an LLM-driven agent loop.

```text
M1 Repository Snapshot → M2 Repository Facts → M3 Engineering Model
        → M4 Architecture Model → M5 Diagram Projection
        → M6 Architecture Context
                ↓
Tool 1: get_architecture_context()    high-level orientation
Tool 2: search_repository()           ranked source evidence (RAG) ← this tool
Tool 3: read_file()                   exact source evidence
                ↓
             ToolResult
                ↓
LLM layer (M7) + agent (M8) — one-shot tool dispatch: M8.3
(multi-step loop: M8.4 implemented)
```

---

## 18.1 Purpose

The deterministic pipeline describes *what the repository contains*
(facts, models, context). An agent still needs to answer:

> "Where in this repository should I look for evidence related to this
> query?"

Since M18, `search_repository` answers that with **semantic retrieval**
over a prebuilt index of the **pinned repository snapshot** of the
current run — the exact checkout persisted by M1 at
`<output>/<run>/repository/<repository_name>/` and frozen at the
recorded commit. It returns ranked **candidates** — path, line range,
excerpt, similarity score, and full provenance — inside the same
generic `ToolResult` every other tool returns.

```text
search_repository → candidate chunk (path + range + score)
                 → read_file(path, line_start, line_end) → exact evidence
```

The tool never interprets, summarizes, or answers the question itself:
relevance judgment belongs to the caller.

## 18.2 Why this is a tool

| | Normal application function | Agent-accessible tool |
| --- | --- | --- |
| Caller | application code | tool runtime (later: an LLM-driven loop) |
| Contract | arbitrary arguments/return values | fixed input contract + generic `ToolResult` |
| Reasoning | none (or embedded in the app) | none in the tool — ranking is similarity, not interpretation |
| Independence | invoked as part of a flow | runnable and testable standalone |

## 18.3 Input

```json
{
  "query": "How do the agents, tools, and MCP servers interact?",
  "path": "docs",
  "max_results": 5
}
```

| Argument | Required | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `query` | yes | string | — | non-empty after trimming; embedded as the semantic query |
| `path` | no | string \| null | `null` | repository-relative file or directory filter |
| `max_results` | no | integer | `5` | `1`–`50`; other values rejected |

- Unknown arguments are rejected (`INVALID_ARGUMENT`) — the contract
  is exactly these three keys.
- `path` must stay inside the repository: absolute paths, drive-letter
  paths, and `..` segments are rejected
  (`PATH_OUTSIDE_REPOSITORY`). Separators are normalized
  (`\` → `/`); `""`, `"."` and `null` mean "no filter".
- `repository_root`, `output_root`, and `store_root` are
  **application context** (checkout, artifacts, retrieval store) —
  not tool arguments. The default is always the pinned snapshot of
  the current run; the explicit `repository_root` exists only as a
  testing / standalone override (agents never provide it).

Python API:

```python
from repo_intelligence.tools.search_repository import search_repository

# normal usage: resolves the pinned snapshot of the current run
result = search_repository(
    {"query": "How do agents use tools?", "path": "src"}
)

# testing / standalone override only
result = search_repository(
    {"query": "checkout flow"},
    repository_root=Path("/path/to/checkout"),
)
```

## 18.4 Output

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
  "tool": "search_repository",
  "data": {
    "query": "checkout flow",
    "results": [
      {
        "file_path": "docs/guide.md",
        "path": "docs/guide.md",
        "line_start": 3,
        "line_end": 7,
        "chunk_id": "bf2c04709e689b38",
        "similarity_score": 0.345857,
        "language": "markdown",
        "repository_name": "sample-repo-alpha",
        "commit_sha": "<snapshot commit>",
        "snapshot_id": "run_fixture_alpha",
        "excerpt": "The checkout flow starts here.\n..."
      }
    ],
    "total_matches": 7,
    "returned": 1,
    "truncated": true,
    "omitted_results": 0,
    "note": null,
    "retrieval": {
      "mode": "semantic",
      "embedding_model": "sentence-transformers://BAAI/bge-small-en-v1.5",
      "repository": "sample-repo-alpha",
      "commit_sha": "<snapshot commit>",
      "snapshot_id": "run_fixture_alpha",
      "index_chunks": 7
    }
  },
  "source": {
    "repository": "sample-repo-alpha",
    "commit_sha": "<snapshot commit>",
    "snapshot_id": "run_fixture_alpha",
    "repository_url": "https://example.invalid/sample.git"
  },
  "error": null
}
```

| Field | Meaning |
| --- | --- |
| `results[].path` / `file_path` | repository-relative POSIX path of the candidate's file (`path` is the established alias every evidence consumer reads) |
| `results[].line_start` / `line_end` | 1-based inclusive line range of the indexed chunk |
| `results[].chunk_id` | stable 16-hex id (path + range + content hash) |
| `results[].similarity_score` | cosine similarity of query vs. chunk (`0.0` = no overlap) |
| `results[].language` | classified language of the file |
| `results[].repository_name` / `commit_sha` / `snapshot_id` | provenance of the evidence (§26 contract) |
| `results[].excerpt` | the chunk's raw text — evidence, not a summary |
| `total_matches` | honest count of candidates considered in scope |
| `returned` | number of results actually DELIVERED (≤ `max_results`) |
| `truncated` | `true` when the result count, the bounded scan, or the response ceiling (see below) withheld candidates |
| `omitted_results` | results retrieved but withheld to fit the response ceiling (`0` normally) |
| `note` | `null` normally; otherwise the plain-text truncation note (how many results were omitted and why) |
| `retrieval.*` | mode, embedding model, and identity actually queried |

`source` values are `null` when the run snapshot is unavailable (see
18.12).

**Response ceiling.** One serialized result never exceeds
`MAX_TOOL_RESPONSE_CHARS` = 24,000 characters (envelope included).
When the ranked list would break the ceiling, whole results are
dropped from the TAIL until the payload fits — a result is never cut
in half, ranking is never recomputed, so the delivered list is always
the head of the same ranked list. `omitted_results` counts what was
withheld, `note` explains it, and `truncated` becomes `true`.

### Failure

```json
{
  "success": false,
  "tool": "search_repository",
  "data": null,
  "source": null,
  "error": {
    "code": "RAG_INDEX_NOT_FOUND",
    "message": "no retrieval index for repository ... Build it with: python -m repo_intelligence.retrieval.index --repository-root <checkout>"
  }
}
```

## 18.5 Execution flow

```text
Agent
  ↓
ToolCall   { tool_name, arguments: {query, path?, max_results?} }
  ↓
search_repository                ← executed by the application/tool runtime
  ↓
1. validate input                contract keys, query, path, max_results
2. resolve repository root       application context: pinned
                                 <output>/<run>/repository/<name>
3. resolve path filter           repository-relative scope (safe or rejected)
4. resolve identity              snapshot record beside the checkout
                                 (repository + commit + snapshot id)
5. resolve active index          adopt the built index's model
                                 (never guess between indexes)
6. embed query + FAISS search    bounded scan, same model as the index
7. filter + rank                 path filter, cosine score,
                                 tie-break (-score, path, line, chunk_id)
8. limit                         max_results (default 5, max 50)
  ↓
ToolResult (results + provenance)   (read-only; checkout + index untouched)
  ↓
Agent
```

The tool **never builds an index implicitly**. A missing index is a
structured, actionable error (18.11); building is an explicit
operator/CLI step (18.13).

## 18.6 Search semantics

- the query is embedded (sentence-transformers model, or the offline
  `hashing://v1/<dim>` embedder) with the **same** embedder that built
  the index — never a cross-model query;
- chunks are ranked by cosine similarity; ordering is deterministic
  with the stable tie-break `(-score, file_path, line_start,
  chunk_id)`;
- **no zero-score filtering**: a query with no overlap still returns
  bounded candidates with `similarity_score == 0.0` — retrieval is
  compact by construction (top-k), never a file dump;
- repeated runs over the same index return byte-identical results;
- the query is case-insensitive under the hashing embedder.

## 18.7 Search scope and exclusions

Index-time (shared file policy, `facts/filesystem.py`):

- searchable text/code extensions only;
- skips `node_modules/`, `.venv/`, `.git/`, caches and build output;
- skips binary files, files over 1 MB, and symlinks that resolve
  outside the checkout;
- only the pinned checkout is read — never run-level decoys, never
  other repositories, never the working tree outside the snapshot.

Query-time: the `path` filter restricts candidates to a
repository-relative file or directory inside the same checkout.

## 18.8 Ordering, limits, truncation

- `max_results` defaults to `5`, maximum `50`.
- `returned ≤ max_results`; `total_matches` reports the honest
  in-scope candidate count (with a bounded scan over very large
  indexes — `truncated` reflects both caps).
- the serialized payload is capped at 24,000 characters by dropping
  whole results from the tail (`omitted_results`, `note` — see 18.4);
  `truncated` reflects that cap too.
- `truncated: true` means "more evidence may exist; refine the query
  or raise `max_results`" — not an error.

## 18.9 Repository selection and path safety

- default: the current run's pinned checkout
  (`<output>/<run>/repository/<repository_name>/`, legacy flat
  `<output>/<run>/repository/` still resolves);
- explicit `repository_root` (testing/standalone): resolved against
  the run context, then accepted as an unpinned subject (M17-style
  `--repository <dir>` — identity scoped by a path digest);
- a checkout inside a `repository/` container without a readable
  snapshot record fails loudly (`RAG_IDENTITY_NOT_RESOLVED`) —
  identity is never guessed from file contents;
- path traversal (`..`), absolute paths, and symlink escapes are
  rejected before any read (`PATH_OUTSIDE_REPOSITORY`); unknown
  paths fail with `PATH_NOT_FOUND`.

## 18.10 Evidence, not interpretation

Results are candidate locations with excerpts and scores. The tool
does not answer the query, summarize files, or rank by "understanding"
— the caller (agent, flow diagnostic, human) decides what matters and
verifies it with `read_file`:

```text
RAG:            "This chunk is relevant."
Graph:          "This relationship has evidence at this location."
read_file:      "Here is exactly what those lines contain."
```

## 18.11 Errors

| Code | Cause | Fix |
| --- | --- | --- |
| `REPOSITORY_NOT_FOUND` | no checkout for the resolved context | run the snapshot pipeline or pass an existing root |
| `RAG_INDEX_NOT_FOUND` | no index for this identity | build it (message includes the exact command) |
| `RAG_INDEX_AMBIGUOUS` | several non-default indexes exist | rebuild/delete the stale ones |
| `RAG_INDEX_CORRUPT` | manifest/vector/metadata drift | rebuild with `--force` |
| `RAG_IDENTITY_MISMATCH` | index belongs to another identity/model | rebuild for this identity |
| `RAG_IDENTITY_NOT_RESOLVED` | no snapshot record beside the checkout | run `repo-intelligence snapshot` first |
| `RAG_REPOSITORY_NOT_FOUND` | explicit root does not exist | check the path |
| `RAG_INVALID_ARGUMENT` / `RAG_UNKNOWN_EMBEDDER` | bad argument or model identity | fix the input |
| `RAG_EMBEDDINGS_UNAVAILABLE` / `RAG_FAISS_UNAVAILABLE` | missing dependencies | `pip install '.[rag]'` |
| `PATH_OUTSIDE_REPOSITORY` | path escapes the checkout | use a repository-relative path |
| `PATH_NOT_FOUND` | path matches nothing | check the spelling |
| `QUERY_TOO_LONG` | query plus envelope exceed the 24,000-character response ceiling even with no results | shorten the query |
| `INVALID_ARGUMENT` | contract violation (keys, types, ranges) | fix the input |

Failures are always structured (`error.code` + `error.message`);
never a stack trace.

## 18.12 Provenance

- every result row carries `repository_name`, `commit_sha`,
  `snapshot_id`, `chunk_id`, `line_start`, `line_end`;
- `data.retrieval` reports the identity and embedding model actually
  queried;
- `source` describes the run snapshot (`repository`, `commit_sha`,
  `snapshot_id`, `repository_url`), or `null` fields when the run
  snapshot cannot be resolved for the supplied output root.

## 18.13 Running the tool independently

Build the index first (explicit — never a side effect of a query):

```bash
# module CLI
python -m repo_intelligence.retrieval.index --repository-root <checkout>

# project CLI
repo-intelligence retrieval-index --repository-root <checkout>

# isolated store (tests / experiments)
python -m repo_intelligence.retrieval.index \
    --repository-root <checkout> --store /tmp/rag-store
```

Query it:

```bash
# normal usage: searches the pinned snapshot of the current run
python -m repo_intelligence.tools.search_repository \
    --query "How do the agents, tools, and MCP servers interact?"

# limit scope + result count
python -m repo_intelligence.tools.search_repository \
    --query "checkout flow" --path docs --max-results 5

# isolated store for testing
python -m repo_intelligence.tools.search_repository \
    --query "checkout flow" --store /tmp/rag-store
```

Exit codes: `0` success (JSON `ToolResult` on stdout), `6` structured
failure. `--repository-root` / `--output-root` are testing and
standalone overrides only — agents never provide them.

Offline/deterministic embedding for tests and demos:

```bash
python -m repo_intelligence.retrieval.index \
    --repository-root <checkout> --model "hashing://v1/128"
```

## 18.14 Non-goals and relationship to other tools

- **No interpretation**: similarity ranking is not an answer (18.10).
- **No graph mutation**: RAG never writes nodes or edges; the
  Engineering Graph stays the deterministic relationship source and
  the persisted artifact stays byte-identical (§47 source-of-truth
  rule).
- **No implicit builds**: queries never trigger index builds.
- **No reranking/LLM in the tool** (§31): no cross-encoder, no LLM
  reranker — plain top-k over the index.
- **No enterprise RAG platform** (§48): FAISS + SQLite only — no
  cloud vector DBs, no Elasticsearch/OpenSearch/Redis/Kafka.
- `read_file` remains the **only** way to turn a candidate into
  exact source evidence; `query_engineering_graph` remains the way
  to walk relationships; `get_architecture_context` remains the
  high-level orientation.

See `docs/m18-repository-retrieval.md` for the full milestone
description and the flow diagnostic.
