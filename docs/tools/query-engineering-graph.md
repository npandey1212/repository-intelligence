# Tool 4 — `query_engineering_graph`

Read-only tool layer above the deterministic M1–M6 pipeline. Part of the
project's learning architecture: the graph-query capability over the
already-generated EngineeringModel — introduced after the agent loop
existed, so the LLM can navigate structured repository relationships
instead of only lexical evidence.

```text
M1 Repository Snapshot → M2 Repository Facts → M3 EngineeringModel
        → M4 ArchitectureModel → M5 Diagram Projection
        → M6 ArchitectureContext
        → M11 engineering_graph.json (persisted graph artifact,
          deterministic transformation of the M3 model)
                ↓
Tool 1: get_architecture_context()    high-level orientation
Tool 2: search_repository()           candidate source evidence
Tool 3: read_file()                   exact source evidence
Tool 4: query_engineering_graph()     structured graph evidence  ← this tool
                ↓
             ToolResult
                ↓
LLM layer (M7) + agent (M8) — tool dispatch through the application
registry; plan-and-execute over the same loop (M10)
```

---

## 23.1 Purpose

`search_repository` and `read_file` answer *what text exists*.
`query_engineering_graph` answers a different question:

> **Given this node id, which artifacts and code entities does it
> contain / depend on / reference — and with what evidence?**

It exposes the persisted engineering graph
(`<run>/engineering/engineering_graph.json` — the deterministic M11
transformation of `engineering_model.json`, projected into a read-only
NetworkX `MultiDiGraph`; that artifact is the only source the
agent-facing run-context path reads — a missing graph artifact is a
structured `ENGINEERING_GRAPH_NOT_FOUND` failure, never a model
projection — see §23.14) to a future agent as two
deterministic query modes:

- **node-kind summary** (discovery) — what kinds of nodes exist, how
  many, and a bounded sample of ids to start from;
- **bounded traversal** — the neighborhood of one node id, filtered by
  direction, relationship kinds, node kinds, and depth.

## 23.2 Responsibility

One structured, bounded, read-only query per call, returned inside the
generic `ToolResult`:

```text
ToolCall { name: "query_engineering_graph", arguments }
    ↓
resolve the current run's graph artifact (application context)
    engineering_graph.json  ← the only run-context source
    (absent → ENGINEERING_GRAPH_NOT_FOUND; engineering_model.json
      is never consulted on this path)
    ↓
load + validate the document
project_engineering_graph(document)   ← read-only projection
    (the explicit testing/standalone --engineering-model-path may
      project engineering_model.json directly — agents never
      supply application context)
    ↓
query_engineering_graph(graph, ...)   ← bounded walk or summary
    ↓
ToolResult { data: <query result>, source: <provenance> }
```

## 23.3 What it does not do

- **No LLM** — every step is deterministic; the same arguments over
  the same artifact return byte-identical JSON.
- **No graph/vector database, no index, no embeddings** — NetworkX
  projection built in-process from the artifact, per query.
- **No mutation, no repair** — the tool never writes the model, the
  graph artifact, the graph, or any file; it only reads. A corrupted
  `engineering_graph.json` is reported (`MODEL_INVALID`), never
  bypassed silently, and never repaired in place; an *absent* graph
  artifact is reported (`ENGINEERING_GRAPH_NOT_FOUND`) rather than
  replaced by a model projection.
- **No new relationship semantics** — queries only ever return the
  EngineeringModel vocabulary `CONTAINS`, `IMPORTS`, `DEPENDS_ON`,
  `TRIGGERS`, `DECLARES`, `EXPOSED_BY`, `CONFIGURED_BY` (M12 added the
  last three to the *model*; the query layer just reads them);
  unresolved claims never become edges (they are only *counted*).
- **No hard-coded repository knowledge** — no run id, repository name,
  agent name, or MCP config name appears in the implementation
  (enforced by source scans).

## 23.4 Input

All keys are optional; the contract is enforced by
`_ALLOWED_ARGUMENTS` (unknown keys are rejected):

```json
{
  "start_node_id": "artifact:file:pkg/mod.py",
  "direction": "out",
  "relationship_kinds": ["CONTAINS", "IMPORTS"],
  "node_kinds": ["artifact/file"],
  "max_depth": 2
}
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `start_node_id` | string / null | `null` | node id to traverse from; `null`/omitted → **summary mode** |
| `direction` | `"out"` \| `"in"` \| `"both"` | `"out"` | edge direction of the walk |
| `relationship_kinds` | list of the seven kinds / null | `null` | only follow/return these relationship kinds |
| `node_kinds` | list of `"kind"` or `"kind/subkind"` / null | `null` | only visit matching nodes (start node always included) |
| `max_depth` | integer `0..6` | `1` | traversal depth in hops (`0` = start node only) |

Validation never coerces: an unknown direction, a depth out of range, a
relationship kind outside the vocabulary, or an empty `start_node_id`
is a structured `INVALID_ARGUMENT` failure.

## 23.5 Output

`data` is the query result (same shape in both modes):

```json
{
  "mode": "traversal",
  "query": {"start_node_id": "…", "direction": "out", "relationship_kinds": null, "node_kinds": null, "max_depth": 1},
  "start_node": {"id": "…", "depth": 0, "kind": "artifact", "subkind": "file", "classification": "observed", "attributes": {}, "evidence": []},
  "nodes": [ … sorted by id, each with its BFS depth … ],
  "relationships": [ … sorted by (depth, id), each with kind, endpoints, classification, depth, evidence … ],
  "evidence": [ … flat view of every returned evidence entry, each carrying "origin": "node:<id>" | "relationship:<id>" … ],
  "kinds": [ … summary mode only: kind/subkind buckets with counts + bounded id samples … ],
  "counts": {"nodes": 7, "relationships": 6, "evidence": 19},
  "truncated": false
}
```

`source` carries provenance: artifact path, schema version, pinned
commit, snapshot id, and whole-model counts
(`nodes` / `relationships` / `unresolved`).

## 23.6 Traversal semantics

- **Breadth-first, bounded.** Frontier nodes are processed in sorted
  id order; the walk stops at `max_depth`, at `MAX_QUERY_NODES` (100)
  visited nodes, or at `MAX_QUERY_RELATIONSHIPS` (200) recorded
  relationships — hitting a cap sets `truncated: true` (never a silent
  partial answer without flag).
- **The start node is the anchor.** It is always returned and always
  expanded, whatever `node_kinds` says.
- **Filters restrict the walk.** A node that fails `node_kinds` is
  neither visited nor recorded, so results stay a connected subgraph;
  intermediate nodes of other kinds are *not* crossed (use a kind-level
  filter such as `"artifact"`, or re-query from the far node).
- **Relationship depth** = the deeper of its two endpoint depths.
- **Multi-edges are preserved** — the same (source, target) pair with
  two kinds (e.g. `CONTAINS` + `IMPORTS`) yields two distinct
  relationships, keyed by their canonical ids.

## 23.7 Summary mode (discovery)

`query_engineering_graph({})` answers "what is in this graph?":

```json
{"mode": "summary", "counts": {"nodes": 1113, "relationships": 1627, "unresolved": 449, "kinds": 14},
 "kinds": [{"kind": "artifact", "subkind": "agent_definition", "node_count": 11, "sample_node_ids": ["…"]},
           {"kind": "tool", "subkind": "mcp", "node_count": 66, "sample_node_ids": ["…"]}]}
```

Kinds are sorted; samples are capped (`SUMMARY_SAMPLE_LIMIT`) and
sorted. The summary accepts **no** traversal parameters — mixing
`start_node_id: null` with `direction`/`max_depth`/filters is a
structured `INVALID_ARGUMENT`, so an explicit query is never silently
ignored.

## 23.8 Determinism and safety

- Nodes sorted by id; relationships sorted by (depth, id); evidence
  copied (never aliased) from the model.
- The input document and graph are never mutated; classifications are
  returned exactly as recorded (`observed` stays `observed`, `derived`
  stays `derived` — evidence is never upgraded).
- Bounded by construction: depth ≤ 6 hops, ≤ 100 nodes, ≤ 200
  relationships per call.

## 23.9 Errors

| Code | When |
| --- | --- |
| `INVALID_ARGUMENT` | non-mapping arguments, unknown key, bad `direction`/`max_depth`/kind filter, summary-mode misuse |
| `NODE_NOT_FOUND` | `start_node_id` not in the graph (message points at the summary mode for id discovery) |
| `ENGINEERING_GRAPH_NOT_FOUND` | the run was resolved but `<run>/engineering/engineering_graph.json` does not exist — the agent-facing path never falls back to `engineering_model.json`; the message names the expected path and the regeneration command (`python -m repo_intelligence.graph.engineering_graph_persist`) |
| `MODEL_NOT_FOUND` | no usable run under the output root (discovery failed / multiple runs), or the explicit `--engineering-model-path` file is missing |
| `MODEL_INVALID` | the resolved artifact is unreadable, has the wrong schema version, or is structurally broken (a corrupted graph artifact fails here — no silent bypass, no repair) |

Failures are structured `ToolResult`s — no stack trace crosses the
agent boundary, and a failed query is an observation, not a crash.

## 23.10 Provenance

Every success reports where the data came from (never re-derived):

```json
{"artifact": "<output>/<snapshot>/engineering/engineering_graph.json",
 "derived_from": "engineering_graph.json",
 "schema_version": "engineering_graph.v1",
 "model_schema_version": "repository_knowledge_model.v1",
 "commit_sha": "…", "snapshot_id": "…",
 "counts": {"nodes": 1113, "relationships": 1627, "unresolved": 449}}
```

On the explicit testing/standalone path (`--engineering-model-path`),
`artifact` points at `engineering_model.json`, `derived_from` is
`"engineering_model.json"`, and `schema_version` equals
`model_schema_version`. The `counts` are the same either way — the two
projections are equivalent by contract (§23.14, tested) — but the
agent-facing run-context path always reports the graph artifact: it
never reads a model document, so `derived_from` is always
`"engineering_graph.json"` there.

Each node/relationship in `data` carries its own `evidence` entries
(path, artifact type, commit, line numbers/symbol where the model
recorded them), plus the flat annotated `evidence` view.

## 23.11 Running the tool independently

```bash
python -m repo_intelligence.tools.query_engineering_graph            # summary
python -m repo_intelligence.tools.query_engineering_graph \
  --start-node-id "artifact:directory:pkg" --direction out --max-depth 2
```

Prints the `ToolResult` as JSON; exit `0` = success, `6` = structured
failure. `--output-root` / `--engineering-model-path` are application
context for testing and standalone use — agents never supply them.

To (re)generate the persisted graph artifact the tool reads:

```bash
python -m repo_intelligence.graph.engineering_graph_persist \
  [--model PATH] [--output PATH] [--output-root DIR] [--snapshot-id ID]
```

## 23.12 Relationship to the other tools

```text
get_architecture_context   ← orientation (what the repo looks like)
search_repository          ← candidate text evidence (where to look)
read_file                  ← exact text evidence (what it says)
query_engineering_graph    ← structured relationship evidence
                             (what contains/references what, with ids)
        ↓
LLM / agent tool loop — all four share one ToolCall / ToolResult
protocol, one dispatcher registry, one tool-call budget
```

The tools are complementary: the graph says *which* node references
what (with canonical ids), `search_repository`/`read_file` retrieve the
*text* behind those ids (e.g. an `agent_definition` node id → read the
actual agent file).

## 23.13 What this graph can and cannot answer

**Can answer** (with evidence, from the EngineeringModel contract):

- containment: which directory/file contains which file or entity
  (`CONTAINS`);
- imports: which file imports which module (`IMPORTS`);
- cross-artifact references the construction rules derive (e.g. an
  `agent_command` **TRIGGERS** an `agent_definition`; documentation
  files referencing agents);
- the M12 knowledge slice: which agent **DECLARES** which tool, which
  tool is **EXPOSED_BY** which registered MCP server, and which server
  is **CONFIGURED_BY** which MCP configuration file — the forward
  `Agent → Tool → MCP Server → Configuration` path as evidence-backed
  edges (`direction: "out"`, `relationship_kinds` limited to those
  three kinds, `max_depth: 3`);
- code structure: modules/classes/functions with their evidence;
- counts: node/relationship/unresolved totals per kind.

**Cannot answer** (and must not be made to answer by inventing edges):

- **Runtime tool invocation**: *which tool an agent actually calls,
  and when* — M12 records explicit **declarations** (`DECLARES`),
  not executions; the executable mapping lives in the agent
  definitions and the application dispatcher, and nothing in this
  graph observes it at runtime.
- **Runtime call sequences, frequencies, or health** — this is a
  static, evidence-backed snapshot of one pinned commit, not a
  telemetry store.
- Anything the repository never states: candidates that do not resolve
  (unregistered server namespace), ambiguous bare tool names, and
  multi-configuration conflicts stay in `unresolved[]` /
  `classification: conflicting` — they are reported as such, never as
  edges (see [M12](../m12-agent-tool-mcp-knowledge.md)).

For runtime questions the honest path is still the evidence one:
locate the agent/MCP configuration files with `search_repository` and
read them with `read_file` — never synthesize edges inside a query.

## 23.14 Persisted engineering graph (M11.1 + M11.2)

Layout (beside the model, same run directory):

```text
artifacts/<snapshot_id>/engineering/
    ├── engineering_model.json   # source of truth (M3, frozen)
    └── engineering_graph.json   # persisted graph artifact (M11,
                                 # derived — never a second source
                                 # of truth)
```

The artifact is produced by a **deterministic transformation**
(`build_engineering_graph_document`):

- node ids, relationship ids/kinds/endpoints, classifications,
  attributes, evidence and unresolved claims are copied **verbatim**
  (deep copies — nothing invented, enriched, resolved, or dropped);
- only ordering is canonicalized: nodes by id, relationships by id;
  nested collections keep their model order;
- layout: `schema_version` (`engineering_graph.v1`), `provenance`
  (source artifact, model schema version, repository commit,
  snapshot id, generator), `metadata.counts`, `nodes`,
  `relationships`, `unresolved`;
- serialization matches every other artifact:
  `json.dumps(..., indent=2, ensure_ascii=False) + "\n"`;
- building the same model twice is **byte-identical**, regardless of
  input list order;
- unresolved claims are carried as data and **never** become edges.

Reading it back (`load_engineering_graph_document`) structurally
validates first (schema version, unique ids, endpoints among nodes,
metadata counts), then `project_engineering_graph` runs the exact
same projection as `build_engineering_graph` over a model document —
graph-artifact projection and model projection are equivalent by
contract (same node set, same keyed edge set with the same
attributes, same `graph.graph` metadata), proven by tests both on
synthetic fixtures and the pinned model.

Boundary: NetworkX remains a **runtime projection only** — never the
source of truth, never canonical, never a database. The tool never
generates the artifact (generation is an explicit pipeline step,
§23.11) and never substitutes the model for it: reading is one-way,
`engineering_model.json` → `engineering_graph.json` → query, and an
absent artifact is reported, not back-filled (§23.9).

## 23.15 Named query API (M11.3)

The same query layer exposes a small per-operation façade (library
API — the tool keeps using the single `query_engineering_graph`
entry point; both share one implementation):

```python
from repo_intelligence.graph import (
    get_node, neighbors, relationships,
    find_nodes, find_nodes_by_property, traverse,
)

get_node(graph, node_id)                    # → node record | None
neighbors(graph, node_id, direction, relationship_kinds)   # depth 1
relationships(graph, node_id, direction, relationship_kinds)
find_nodes(graph, kind)                     # "kind" | "kind/subkind"
find_nodes_by_property(graph, name, value)  # top-level field or
                                            # attributes key, exact match
traverse(graph, node_id, direction, relationship_kinds,
        node_kinds, max_depth)              # → the same payload as
                                            # query_engineering_graph(...)
```

Conventions: sorted outputs (nodes by id, relationships by id),
evidence copied, read-only, bounded by the same caps; `neighbors`,
`relationships` and `traverse` raise `GraphQueryError`
(`NODE_NOT_FOUND` / `INVALID_ARGUMENT`), while `get_node` returns
`None` for an unknown id and the `find_*` functions return a (possibly
empty) list — a plain lookup never throws for a missing node. The
property vocabulary is always the model's own (`kind`, `subkind`,
`classification`, `attributes.*`) — never hard-coded per repository.

## 23.16 Artifact validation (recorded results)

Approach — the artifact is validated against its source of truth, not
by inspection:

1. **structural problems** (`engineering_graph_document_problems`) —
   empty on the pinned artifact;
2. **content census** — node kind census (14 kind/subkind buckets,
   including the M12 `tool/mcp`, `tool/built_in`, `mcp_server/registered`)
   and relationship kind census (`CONTAINS` 1395, `IMPORTS` 98,
   `TRIGGERS` 15, `DEPENDS_ON` 3, `DECLARES` 37, `EXPOSED_BY` 66,
   `CONFIGURED_BY` 13) identical to the model; the 449 unresolved
   claims verbatim;
3. **byte-identical regeneration** — rebuilding from the model
   reproduces the committed file exactly (asserted by
   `test_adop_on_disk_artifact_matches_regeneration`);
4. **projection equivalence** — artifact projection == model
   projection (1113 nodes, 1627 edges, `unresolved_count` 449,
   invariant problems empty, duplicate `CONTAINS`+`IMPORTS` pair
   preserved);
5. **queries over the projection** — summary counts
   1113/1627/449/14 kinds; `find_nodes("artifact/mcp_config")` → the
   single `.mcp.json` node; `artifact/agent_definition` → 11;
   `tool/mcp` → 66; `mcp_server/registered` → 13.

Pinned artifact: `artifacts/main_2026-09-22_ef11dac/engineering/engineering_graph.json`
(schema `engineering_graph.v1`, 2 244 802 bytes,
sha256 `950956c763495839…`, commit `ef11dac…`).

## 23.17 Demonstration (spec §14)

```bash
python agentrun_m11_graph_demo.py            # default MCP objective
python agentrun_m11_graph_demo.py --question "…" --model gpt-4o-mini
```

The script (a composition root, no graph/agent logic of its own):

1. **artifact preflight** — loads `engineering_graph.json` and prints
   its provenance/counts *before* any LLM call (exit 6 if missing,
   with the exact regeneration command);
2. composes the existing Agent (4 read-only tools) and runs **one**
   investigation;
3. reads back the ONE recorded trace and projects it into the spec §14
   stages — **LLM reasoning** (each iteration), **graph retrieval**
   (executed `query_engineering_graph` calls), **source-file
   retrieval** (executed `search_repository`/`read_file` calls),
   **structured answer** (the recorded final answer) — nothing is
   re-derived;
4. prints the stage breakdown (each graph call with its
   `source.artifact`, proving the graph artifact was read), the
   answer, and the trace path.

Exit codes: `0` = run completed **and** at least one engineering graph
query executed; `6` = configuration/execution failure, or a completed
run that never queried the graph (reported explicitly, never counted
as success).

See also the M12 demonstration (`python agentrun_m12_knowledge_demo.py`
— same preflight/stages/exit-code contract plus the offline
`Agent → Tool → MCP → Configuration` slice walk and its structured
result; [M12 doc](../m12-agent-tool-mcp-knowledge.md)).

Recorded runs (all exit 0, `gpt-4o-mini`, default artifact root):

- primary: `runtime/agent_runs/a7fea0e651f34a61b95784903a465dde/trace.json`
  — 5 iterations, 4 graph queries (all reading
  `engineering_graph.json`), 1 `search_repository` call; the answer
  reports the MCP config/server-source nodes as relationship-less
  (true in the model) and a concrete `CONTAINS` chain from
  `artifact:file:mcp-servers/glue-athena-server/server.py`
  (**18 nodes / 31 relationships** — matching offline ground truth:
  directory → file → module/functions);
- honesty observation: an earlier run
  (`05ed8890cf764821b8e49e342e63abad`) *described* a traversal it
  never executed — the stage breakdown exposes this (3 graph calls,
  none from the node in question), which is exactly why the stages are
  recorded from the trace rather than trusted from the prose. Claim
  enforcement remains the M9 evaluator's job, not this demo's.
