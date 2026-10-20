# M12 — Agent → Tool → MCP Knowledge Slice

## What is M12?

M12 makes the **Agent → Tool → MCP Server → MCP Configuration** slice of
the repository queryable as deterministic, evidence-backed knowledge inside
the **existing** Engineering Graph — so an agent can retrieve *structural*
MCP knowledge through the existing bounded graph tool before any LLM
reasoning happens.

```text
M2 facts (tooling analyzer)  →  EngineeringModel (3 new node kinds,
                                 3 new relationship kinds)
                              →  engineering_graph.json (M11 persistence)
                              →  query_engineering_graph (existing Tool 4,
                                 bounded depth-3 forward path query)
                              →  agent observation + structured result
```

It answers questions like:

> How is this repository's agent connected to an MCP server, and what
> evidence supports the connection?

…deterministically, from the graph — the LLM **synthesizes** the
explanation; it does not discover the canonical relationship.

### M12 does

- Observes explicit tool/server/configuration statements in repository
  files (agent frontmatter, tool registry, MCP configuration) as facts.
- Builds canonical `tool` / `mcp_server` nodes and
  `DECLARES` / `EXPOSED_BY` / `CONFIGURED_BY` relationships with evidence.
- Preserves ambiguity and conflicts as unresolved claims — never edges.
- Extends the existing M3 model, M11 graph persistence, and Tool 4
  (no second graph, no new tool, no new runtime).

### M12 does not

- No MCP runtime execution, no dynamic server discovery, no autonomous
  registration — M12 is knowledge modeling, not MCP integration.
- No LLM-generated graph edges: canonical relationships come only from
  deterministic processing.
- No graph database, no GraphRAG, no new tracing, no workflow engine,
  no long-term memory, no multi-agent orchestration, no automatic
  replanning.

## Milestone boundary

```text
M2 (facts)  … + tooling analyzer + tool/mcp_server fact types
M3 (model)  … + 3 node kinds + 3 relationship kinds + rulebook rules
M11 (graph) … unchanged persistence/projection/query — now serving the
              M12 slice (no format or API change)
M12         = the slice itself + demonstration + tests + this document
```

M10 plan-and-execute semantics (one planner call, plan before first tool,
`plan_step_id`, one shared budget) are untouched. The M11 run-context
contract (reads **only** `engineering_graph.json`; `ENGINEERING_GRAPH_NOT_FOUND`
/ `MODEL_INVALID` / `MODEL_NOT_FOUND`; dispatcher passes only `arguments`)
is unchanged.

## Semantic model

The actual implemented model — relationship names exactly as the
implementation supports them:

```text
artifact/agent_definition          (existing M3.5 node)
        │
        │ DECLARES          (M2 interaction/declares, observed)
        ↓
tool/mcp  tool:mcp:<server>:<name>
tool/built_in  tool:builtin:<name>     (M2 tool/mcp_tool | built_in_tool)
        │
        │ EXPOSED_BY        (M2 interaction/exposes, observed)
        ↓
mcp_server/registered  mcp_server:<name>   (M2 mcp_server/registered)
        │
        │ CONFIGURED_BY     (M2 interaction/configures, observed;
        ↓                    classification conflicting on conflict)
artifact/mcp_config  (existing M2 artifact node, e.g. .mcp.json)
```

### Node kinds (M12 additions)

| Kind | Id template | Established by |
| --- | --- | --- |
| `tool` / `mcp` | `tool:mcp:{server}:{name}` | tool-registry entry, or an explicit namespaced agent declaration `mcp__{server}__{name}` |
| `tool` / `built_in` | `tool:builtin:{name}` | bare agent `tools:` declaration with **no** MCP-tool name collision |
| `mcp_server` / `registered` | `mcp_server:{name}` | explicit entry in a tool registry or MCP configuration file |
| `artifact` / `mcp_config` | `artifact:file:{path}` (existing) | existing M2 configuration fact (e.g. `.mcp.json`) |

Server-scoped identity (`tool:mcp:{server}:{name}`) is the MCP calling
identity: two servers may legitimately expose same-named tools without
ambiguity. `mcp_server/registered` is a *semantic* entity — distinct from
the existing `artifact/mcp_server_source` (a source-file classification).

### Relationship vocabulary (all seven kinds now in the model)

| Kind | M12 | Classification | Meaning |
| --- | --- | --- | --- |
| `CONTAINS` | existing | derived | structural containment |
| `IMPORTS` | existing | observed | module import |
| `DEPENDS_ON` | existing | observed | declared dependency |
| `TRIGGERS` | existing | observed | explicit M2 trigger statement |
| `DECLARES` | **new** | observed | agent → tool (explicit declaration) |
| `EXPOSED_BY` | **new** | observed | tool → MCP server (registry/namespaced reference) |
| `CONFIGURED_BY` | **new** | observed | MCP server → MCP configuration artifact (entry in a config file) |

The vocabulary is intentionally minimal: three forward kinds make the path
walk `direction="out"` from the agent, with no reverse-only edges and no
`USES_TOOL`-style inventions. All three are `CREATE_RELATIONSHIP_IF_RESOLVED`
rules in the rulebook (`rel.interaction.declares|exposes|configures`).

Relationship ids remain `rel:{kind}:{source_id}:{target_id}` — no random
UUIDs, no analyzer names.

## Evidence model

Each relationship edge answers "why does this exist?" with the existing
`EvidenceRef` envelope (`path`, `commit`, optional `line_start`,
`artifact_type`):

| Relationship | Evidence that establishes it |
| --- | --- |
| `DECLARES` | the agent artifact's `tools:` entry (frontmatter line for markdown agents) — `artifact_type: agent_definition` |
| `EXPOSED_BY` | the tool-registry entry for the server's tool — `artifact_type: configuration` (or a namespaced agent declaration) |
| `CONFIGURED_BY` | the server entry inside the MCP configuration file — `artifact_type: mcp_definition` |

Facts are produced by the `tooling` analyzer (source = `tooling`), which is
pure observation: a file only yields statements when its structure matches
the declared shape (frontmatter `tools:` lines; registry `servers:` /
`tools:` blocks; MCP configuration files by controlled name set). Endpoint
resolution into FactRecords happens after base artifact facts exist,
exactly like the existing `interactions` analyzer.

Classification discipline (never upgraded):

- `observed` — the repository directly states the relationship.
- `derived` — existing CONTAINS only (unchanged).
- `conflicting` — every `CONFIGURED_BY` edge for a server claimed by more
  than one configuration file is retained with this classification; no
  arbitrary pick is ever made.
- Unresolvable/ambiguous candidates stay in `engineering.unresolved[]`
  with retained evidence (below) — they are **not** edges.

## Deterministic boundary

Canonical relationships are created by deterministic processing, never by
the LLM:

1. `analyze_tooling` — regex/structure-driven statement extraction, sorted
   deterministically.
2. `build_tooling_facts` — endpoint resolution against existing artifact
   facts; identity from natural keys only.
3. `compose_facts` → `facts/repository_facts.json` — byte-stable for the
   same snapshot.
4. EngineeringModel construction — rulebook-driven
   (`CREATE_RELATIONSHIP_IF_RESOLVED`), evidence mandatory, ordering
   canonicalized.
5. `engineering_graph.json` — M11's deterministic transformation (M12 did
   not change the format; counts validate against the document).

No new fact types beyond `tool` / `mcp_server` (+ three `interaction`
kinds); no schema change was required beyond extending the controlled
enums (`FACT_TYPE_ORDER`, `counts()`, model/rulebook vocabularies, and the
two relationship-kind enums in `schemas/repository_knowledge_model.v1.json`).

## Graph integration

M12 extends the existing Engineering Graph — there is exactly one graph,
one persisted artifact, one projection, one query API:

- **Model**: 3 node kinds + 3 relationship kinds in the EngineeringModel
  (rulebook: 28 node rules / 14 relationship rules after M12).
- **Persistence**: unchanged `build_engineering_graph_document` — the
  slice nodes/edges flow through verbatim; `metadata.counts` still
  validates nodes/relationships/unresolved.
- **Projection**: unchanged `project_engineering_graph` (`MultiDiGraph`,
  edge key = relationship id).
- **Query**: unchanged Tool 4. The slice summary appears in the node-kind
  summary; the path query is an ordinary bounded traversal:

```json
{
  "start_node_id": "artifact:agent_definition:.claude/agents/ontology-agent.md",
  "direction": "out",
  "relationship_kinds": ["DECLARES", "EXPOSED_BY", "CONFIGURED_BY"],
  "max_depth": 3
}
```

Read-only, deterministic, bounded (≤ 6 hops, ≤ 100 nodes, ≤ 200
relationships per walk), **no LLM** — unresolved claims are counted, never
turned into edges.

### Pinned ADOP numbers (regenerated artifacts)

```text
facts:        1954 facts — tool: 72 (66 mcp_tool + 6 built_in),
              mcp_server: 13, interaction: 131 (37 declares + 66 exposes
              + 13 configures + 15 pre-existing triggers)
model:        1113 nodes, 1627 relationships, 449 unresolved
slice:        72 tool nodes, 13 mcp_server nodes,
              37 DECLARES, 66 EXPOSED_BY, 13 CONFIGURED_BY
```

## Agent integration

No new tool: the agent uses the existing `query_engineering_graph`
(Tool 4). The tool-definition enum and description now include the three
new kinds, so the LLM can discover and issue the bounded path query;
the dispatcher and run-context boundary are untouched (the `agent`
package still imports no tool/graph/engineering module).

Execution pattern:

```text
objective + Architecture Context
  → (M10 plan, unchanged)
  → ReAct: query_engineering_graph {}            ← node-kind summary
  → ReAct: bounded walk DECLARES → EXPOSED_BY → CONFIGURED_BY (depth 3)
  → optional read_file / search_repository (evidence pointed at by the graph)
  → LLM synthesizes the explanation
  → structured result (trace)
```

The trace records the graph retrieval with its evidence exactly as for the
other three tools (stage classification: LLM reasoning, graph retrieval,
source-file retrieval, structured answer).

## Demonstration

`python agentrun_m12_knowledge_demo.py` (manual, real model — costs
tokens; `tests/graph/test_m12_knowledge_demo.py` covers the offline parts):

1. **Offline preflight (no LLM)** — verifies the persisted
   `engineering_graph.json`, derives the slice counts from the projection,
   runs **one** bounded depth-3 path query, and prints the spec §21
   structured result (`finding` / `relationships` / `evidence_paths`).
   Verified on the pinned snapshot:

```text
finding: artifact:agent_definition:.claude/agents/ontology-agent.md
         --DECLARES--> tool:mcp:glue-athena:get_table
         --EXPOSED_BY--> mcp_server:glue-athena
         --CONFIGURED_BY--> artifact:mcp_config:.mcp.json
relationships:
  DECLARES     .../ontology-agent.md        -> tool:mcp:glue-athena:get_table
  EXPOSED_BY   tool:mcp:glue-athena:get_table -> mcp_server:glue-athena
  CONFIGURED_BY mcp_server:glue-athena      -> artifact:mcp_config:.mcp.json
evidence_paths:
  .claude/agents/ontology-agent.md:5
  tool-registry/servers.yaml:13
  .mcp.json:3
```

2. **Agent run** — the §20 objective drives the existing ReAct loop; the
   structured result the agent emits carries the same
   `finding`/`relationships`/`evidence_paths` shape.

**Exit codes**: `0` only when the run produced a slice path **and** an
answer **and** recorded at least one graph query; otherwise `6` (missing
graph artifact, empty slice, missing API key/context — each with a
how-to-fix hint printed to stderr). No other exit codes are used.

Recorded live run (`gpt-4o-mini`, default artifact root, **exit 0**):

- `runtime/agent_runs/b0db47635b94463c807d164b7547f05a/trace.json` —
  3 iterations, 2 `query_engineering_graph` calls (both reading
  `engineering_graph.json`), no source-file calls needed; the answer
  names the four nodes of the chain, the three relationship kinds, and
  quotes the three evidence paths (`.claude/agents/ontology-agent.md:5`,
  `tool-registry/servers.yaml:13`, `.mcp.json:3`) — matching the offline
  preflight result exactly.

## Limitations

What remains deliberately unknown / unresolved / ambiguous / conflicting
(this is preservation, not failure):

```text
unresolved    endpoints that do not resolve to engineering nodes —
              e.g. an agent declares mcp__<server>__<tool> where <server>
              is not registered (reason target_not_resolved), or a
              registered server has no entry in any MCP configuration
              file (reason target_not_resolved), or a source/target fact
              has no canonical node (reason missing_endpoint).
              Claims live in engineering.unresolved[] with evidence —
              never as edges.

ambiguous      a bare (namespace-free) agent tool declaration whose name
              also appears as an MCP tool name in the registry has two
              plausible scopes: no canonical tool/built_in node and no
              DECLARES edge are created; the candidate claim carries
              reason ambiguous_identity.

conflicting    the same server claimed by more than one configuration
              file: every CONFIGURED_BY edge is kept with classification
              conflicting — no arbitrary selection, no merge. (On the
              pinned snapshot: zero conflicts — one config file.)

unknown        anything the repository does not state at all: e.g. which
              runtime actually executes a server, whether an agent ever
              *calls* a declared tool at runtime, transport/auth details.
              M12 records explicit statements only; absence of evidence
              produces no node and no edge.
```

Other boundaries:

- The graph answers *knowledge* questions (who declares/exposes/configures
  what, with what evidence), not *runtime* questions (MCP process
  execution, live tool invocation).
- The bounded walk is ≤ 6 hops / ≤ 100 nodes / ≤ 200 relationships; deep
  or cross-repo questions are out of scope.
- Evidence line numbers are 1-based statements lines; markdown agents are
  parsed from YAML frontmatter only (prose/code mentions are not
  declarations).

## Non-goals (explicitly not introduced)

```text
❌ MCP runtime execution
❌ dynamic MCP server discovery
❌ autonomous MCP registration
❌ LLM-generated graph edges
❌ graph database
❌ GraphRAG
❌ multi-agent orchestration
❌ automatic replanning
```

## Code flow

```text
src/repo_intelligence/
├── facts/
│   ├── tooling.py      # analyze_tooling + build_tooling_facts (M12)
│   ├── records.py      # compute_fact_id: tool / mcp_server / interaction
│   ├── models.py       # FACT_TYPE_ORDER += tool, mcp_server; counts()
│   └── builder.py      # run_analyzers["tooling"] + composition pass
├── engineering/builder.py   # Phase 2: INTERACTION_REL_KINDS
│                            # (triggers→TRIGGERS, declares→DECLARES,
│                            #  exposes→EXPOSED_BY, configures→CONFIGURED_BY)
├── architecture/validator.py # relationship kinds from RELATIONSHIP_KINDS
└── agent/tool_definitions.py # enum + description (3 kinds)

artifacts/<snapshot_id>/
├── facts/repository_facts.json          # + tool/mcp_server/interaction facts
├── engineering/engineering_model.json   # + slice nodes/relationships
└── engineering/engineering_graph.json   # unchanged format, now carries slice

agentrun_m12_knowledge_demo.py           # offline preflight + agent demo
```

CLI regeneration (no new subcommands):

```bash
python -m repo_intelligence.cli facts --snapshot artifacts/<id>/snapshot/repository_snapshot.json
python -m repo_intelligence.cli model
python -m repo_intelligence.graph.engineering_graph_persist
```

## Testing

Focused M12 tests (all offline, **41 new** — full suite
**1048 passed, 2 skipped**):

| File | Tests | Covers |
| --- | --- | --- |
| `tests/test_tooling.py` | 11 | statement extraction (agents/registry/config), frontmatter requirement, shape-matching, determinism, fact identities, ambiguity, conflicts, full snapshot integration |
| `tests/test_m12_knowledge_slice.py` | 8 | model nodes/attributes, forward relationships, full path chain, unresolved stays unresolved, counts, conflicting edges, pinned ADOP numbers |
| `tests/graph/test_m12_knowledge_graph.py` | 9 | projection of the slice, depth-3 walk, depth bounds, kind filters, summary kinds, determinism, pinned counts + queryable path, all 13 servers → `.mcp.json` |
| `tests/graph/test_m12_knowledge_demo.py` | 13 | demo module contract, offline slice preflight, structured result, stage classification, exit-code gates, configuration failures |

Guard/updated tests: `tests/test_construction_rules.py` (vocabulary +
fact-combo pins), the M11 pinned-count tests (1028/1511 → 1113/1627),
`tests/test_architecture_model.py` (validator now shares
`RELATIONSHIP_KINDS`).

## Boundary

M12 is a **knowledge-modeling and retrieval milestone**. It does not make
the system an MCP client, does not change the agent loop, planner, or
evaluation semantics, and does not add an abstraction the existing model
could not already express: three fact types, three node kinds, three
relationship kinds, one analyzer, one demo, one document.

**Downstream (post-M17):** the canonical `RELATIONSHIP_KINDS` this
milestone's graph study runs on are exactly the edge vocabulary of the
Architecture Visualization Agent's diagram — the `agent → tool → mcp_
server` wiring proven here becomes diagram edges with no re-typing,
and neither collapsing a hop nor inventing one survives its
`relationship_validity` / `scope_validity`
(`docs/architecture-visualization-agent.md` §5, §7, §8).
