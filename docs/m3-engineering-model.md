# M3 — EngineeringModel Construction (`repository_knowledge_model.v1`)

## What EngineeringModel is

```text
EngineeringModel =
deterministic representation of repository implementation structure
```

It answers:

> **How is this repository actually structured and implemented?**

Built from:

```text
M2 Facts  →  M3 construction rules  →  EngineeringModel
```

Pinned commit for the primary artifact:

```text
ef11dac8618b6344d6a5e4add6ef51cd2811d7e0
```

Canonical output:

```text
artifacts/main_2026-09-22_ef11dac/engineering/engineering_model.json
```

## What it is not

It is **not**:

- ArchitectureModel / business ontology
- runtime topology or cloud inventory
- inferred architecture (“this is the orchestration service”)
- LLM-generated interpretation
- a graph-database export

Architecture remains empty in V1:

```text
architecture.nodes = []
architecture.relationships = []
architecture.unresolved = []
mappings = []
```

---

## Pipeline

```text
AST / Tree-sitter / YAML / JSON / TOML / Git / filesystem
              ↓
           ANALYSIS          (M2 analyzers — frozen)
              ↓
          M2 FACTS           repository_facts.v2
              ↓
      INTERPRETATION RULES   engineering_construction_rules.v1
              ↓
       ENGINEERING MODEL     repository_knowledge_model.v1
              ↓
          VALIDATION
```

Critical rule:

> **Analyzers discover repository facts. The construction engine builds the EngineeringModel.**

```text
                 ┌── Python AST analyzer
                 │
                 ├── Tree-sitter (future multi-language; optional)
                 │
Repository ──────┼── YAML / JSON / TOML parsers
                 │
                 ├── Git / pinned snapshot metadata
                 │
                 └── filesystem walk
                         │
                         ▼
                    M2 Facts
                         │
                         ▼
              M3 Construction Rules
                         │
                         ▼
                 EngineeringModel
```

**Never:**

```text
AST analyzer → AST graph
YAML analyzer → YAML graph     ← forbidden parallel graphs
```

---

## Why analyzers do not build graphs

If both AST and Tree-sitter emit nodes for `class Service`, two graphs would
disagree. The **central construction layer** owns:

```text
facts  →  identity  →  deduplication  →  evidence  →  validation
```

so every analyzer converges on the same canonical id:

```text
code_entity:src/service.py::Service
```

Analyzers stay interchangeable implementations of observation.

---

## Analyzer architecture (what is used)

| Analyzer | Method | Role |
| --- | --- | --- |
| Python | stdlib `ast` | modules, classes, functions, imports |
| YAML | PyYAML | structured configuration observations |
| JSON | stdlib `json` | structured configuration observations |
| TOML | `tomllib` / fallback | structured configuration observations |
| filesystem | path walk | files and directories |
| git | pinned M1 snapshot | commit provenance |
| repository-metadata | structured scan | agents, MCP, docs, tests, dependencies |
| Tree-sitter | optional | future multi-language; not required for V1 Python |

Recorded in `analysis.analyzers` with actual library versions where practical.
Analyzer identity is **never** part of node identity.

M2 remains the only source of facts. This milestone’s builder **consumes**
`repository_facts.v2`; it does not re-parse the repository unless you re-run M2.

---

## Identity

Canonical node ids (after normalization):

```text
artifact:directory:src
artifact:file:src/service.py
artifact:agent_definition:.claude/agents/dag-agent.md
code_entity:src/service.py                 # module
code_entity:src/service.py::Service         # class
code_entity:src/service.py::run             # function
external_package:jsonschema                 # dependency target
tool:mcp:glue-athena:get_table              # MCP tool, server-scoped (M12)
tool:builtin:list_dir                       # namespace-free declaration (M12)
mcp_server:glue-athena                      # registered server (M12)
```

M2 `qualified_name` already carries path context. M3 normalizes:

```text
path=src/service.py  qualified_name=src/service.py::Service
    → code_entity:src/service.py::Service

path=src/service.py  qualified_name=Service
    → code_entity:src/service.py::Service
```

**Not** `code_entity:src/service.py::src/service.py::Service`.
**Not** `python-class:…` / `ast-class:…`.

Relationship ids:

```text
rel:<kind>:<source_id>:<target_id>
```

No random UUIDs. No analyzer names.

---

## Relationships (V1)

| Kind | Classification | Source facts |
| --- | --- | --- |
| `CONTAINS` | derived | path structure from file/directory/entity facts |
| `IMPORTS` | observed | `import` / `import_from` |
| `DEPENDS_ON` | observed | `dependency` |
| `TRIGGERS` | observed | `interaction` / `triggers` (explicit M2 statements only) |
| `DECLARES` | observed | `interaction` / `declares` (explicit agent `tools:` declaration) |
| `EXPOSED_BY` | observed | `interaction` / `exposes` (tool-registry entry or namespaced agent reference) |
| `CONFIGURED_BY` | observed | `interaction` / `configures` (MCP configuration entry; `conflicting` when several configuration files claim the server) |

### CONTAINS examples

```text
directory CONTAINS file
file      CONTAINS module
file      CONTAINS class / function
module    CONTAINS class / function
class     CONTAINS nested function   (only with deterministic parent link)
```

Only when M2 attributes support a deterministic path link — never name guessing.

**Derived relationship evidence** merges material facts from **both** the
source and target endpoints (M2 evidence refs already attached to those nodes),
deduplicated deterministically — not only the target node’s evidence.

### IMPORTS resolution

```text
module string  →  known engineering modules/files  →  IMPORTS edge
               →  otherwise                         →  engineering.unresolved[]
```

Stdlib (`os`, `json`, …) stays **unresolved** unless a repository node exists.

### DEPENDS_ON

```text
dependency fact → external_package:<name> node (documented vocabulary)
                → DEPENDS_ON
                → or unresolved if name/source missing
```

### M12 slice — DECLARES / EXPOSED_BY / CONFIGURED_BY

The three forward kinds walk one path, always resolved through the same
`CREATE_RELATIONSHIP_IF_RESOLVED` rule shape as `TRIGGERS` (endpoints are
existing `fact_id`s; no name guessing, no inference):

```text
agent node --DECLARES--> tool node --EXPOSED_BY--> mcp_server node
                                      --CONFIGURED_BY--> artifact/mcp_config node
```

- endpoints resolve through the fact-to-node index built in Phase 1;
- a missing endpoint (or one that never became a node) is an unresolved
  claim with the reason carried from M2 (`missing_endpoint`,
  `target_not_resolved`, `ambiguous_identity`) — evidence retained,
  never an edge;
- `attributes.conflicting` on a `configures` fact marks every candidate
  `CONFIGURED_BY` edge `classification: conflicting` — all edges are
  kept, none is picked;
- relationship ids stay `rel:{kind}:{source}:{target}`.

See [M12](m12-agent-tool-mcp-knowledge.md) for the full slice (facts,
evidence, limits).

---

## Evidence

Mandatory:

```text
canonical node     ≥ 1 EvidenceRef
canonical relation ≥ 1 EvidenceRef
derived CONTAINS   evidence from contributing M2 facts
unresolved claim   retains evidence that caused failure
```

Evidence is copied from M2 (`path`, `commit`, optional `line_start`/`symbol`).
Every node/relationship can answer:

> Which deterministic fact(s) caused this object, and what evidence supports it?

---

## Unresolved claims

```json
{
  "source": "artifact:file:src/main.py",
  "target": "os",
  "kind": "IMPORTS",
  "reason": "target_not_resolved",
  "evidence": [{ "path": "src/main.py", "artifact_type": "python", "commit": "…" }]
}
```

Lives in `engineering.unresolved[]` — **not** `relationships[]`.

Reasons in use: `target_not_resolved`, `missing_endpoint`, and
`ambiguous_identity` (M12: a bare tool declaration whose name collides
with an MCP tool name has no canonical endpoint).

Why retain them:

- future investigator agents need leads
- “maybe imports X” is not canonical engineering truth
- prevents fake edges from string matching

---

## Determinism

Same M2 facts + same rules ⇒ byte-identical `engineering_model.json`:

- nodes sorted by `id`
- relationships sorted by `(kind, source, target, id)`
- unresolved sorted by `(kind, source, target, reason)` and de-duplicated
- evidence sorted by stable key
- no hash-order reliance, no UUIDs, no timestamps in the graph section

---

## Artifact layout

```text
artifacts/
└── main_2026-09-22_ef11dac/
    ├── snapshot/repository_snapshot.json
    ├── facts/repository_facts.json
    ├── schemas/…
    └── engineering/
        ├── engineering_construction_rules.v1.json
        ├── engineering_model.json
        └── engineering_graph.json     # derived (M11), see below
```

`engineering_model.json` is the **source of truth** for engineering
knowledge. `engineering_graph.json` (M11) is a *derived, persisted*
projection of it — produced by a deterministic transformation that
copies nodes, relationships, unresolved claims, and evidence verbatim
(only ordering canonicalized), never a second source of truth and
never NetworkX-as-canonical. The agent-facing query tool reads this
derived artifact and only it: when it is absent the query fails with
`ENGINEERING_GRAPH_NOT_FOUND` rather than silently projecting the
model. Generate or regenerate the artifact explicitly:

```bash
python -m repo_intelligence.graph.engineering_graph_persist
```

Details: [docs/tools/query-engineering-graph.md](docs/tools/query-engineering-graph.md)
§23.14–23.16 (persistence contract, validation, and the agent
demonstration).

Authoritative schema:

```text
schemas/repository_knowledge_model.v1.json
```

### CLI

```bash
python -m repo_intelligence model \
  --facts artifacts/main_2026-09-22_ef11dac/facts/repository_facts.json
```

### Python API

```python
from repo_intelligence.engineering import create_engineering_model

model, path = create_engineering_model()
print(model.counts())
```

---

## Validation (enforced)

```text
unique node IDs
unique relationship IDs
relationship source/target exist
canonical node/relationship has evidence
valid classifications (no inferred in V1 relationships)
unresolved not in canonical relationships
deterministic ordering
repository.commit_sha matches input snapshot
architecture empty; mappings empty
```

---

## What I should understand after EngineeringModel

1. **AST vs Tree-sitter** — AST is language-specific and deep for Python;
   Tree-sitter is multi-language and grammar-driven. Both observe structure;
   neither should own the graph.
2. **Why both are useful** — AST for precise Python semantics today;
   Tree-sitter for future languages without rewriting the construction layer.
3. **Analyzers → observations** — they emit facts with evidence, not edges.
4. **Facts → nodes** — construction rules map `(fact_type, kind)` to
   `CREATE_NODE` or `EVIDENCE_ONLY`, then fill identity templates.
5. **Relationships** — observed edges copy import/dependency facts;
   derived edges (CONTAINS) are computed from path structure with evidence.
6. **Identity prevents duplicates** — one `code_entity:path::Name` no matter
   which analyzer saw it first.
7. **Evidence follows the object** — M2 `EvidenceRef[]` is merged onto nodes
   and relationships; no evidence-free canonical objects.
8. **Unresolved is valuable** — open questions for later agents, not noise to
   delete and not fake edges to keep.
9. **Determinism before LLM reasoning** — reproducible structure is the
   substrate; probabilistic interpretation comes later (and only on top of it).
10. **EngineeringModel ≠ ArchitectureModel** — files/modules/imports are
    implementation structure; “agent platform / data product” is architecture.

---

## Boundary

```text
M1 snapshot          frozen
M2 facts             frozen
M3 construction rules frozen (v1)
M3 EngineeringModel  ← this milestone
M3+ builder extras   future
M4 ArchitectureModel / diagrams  future
```

No LLM. No RAG. No embeddings. No ReAct. No Neo4j/Neptune as canonical store.
No semantic architecture inference from `agent_definition` or class names.

---

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_engineering_model.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

| Level | Coverage |
| --- | --- |
| Unit | identity, resolver, evidence, unresolved, validation errors |
| Fixture | synthetic facts → nodes/CONTAINS/IMPORTS/TRIGGERS/unresolved |
| Integration | pinned ADOP facts → full model + determinism + commit check |

---

## Diagnostic coverage (not ontology)

`metadata.analysis_coverage` records fact totals by type/source for humans.
These metrics are **not** EngineeringModel nodes.

---

## §22 — TRIGGERS: observation vs canonicalization vs architecture

**M2 observes.** Explicit trigger statements become `interaction` facts whose
endpoints are existing `fact_id`s, with evidence at the statement location.
No ADOP-specific ontology is hard-coded.

**M3 canonicalizes.** Rule `rel.interaction.triggers` maps each interaction
fact to an observed `TRIGGERS` edge only when both endpoint facts produced
engineering nodes (`CREATE_NODE`). Failed endpoints stay in
`engineering.unresolved[]` with retained evidence. Relationship identity
remains `rel:TRIGGERS:<source>:<target>`.

**Architecture waits.** The architecture section stays empty on pure
engineering artifacts; `ArchitectureModel` is a later milestone and does not
populate from `TRIGGERS`.

**Why “Data Onboarding Agent → Metadata Agent” may not appear as that label:**
the source may resolve to the `onboard-workflow` **command** artifact (where
the self-identity statement lives), not a free-text agent title. Evidence
governs endpoints — desired semantics do not invent nodes.
