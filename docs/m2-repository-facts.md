# M2 — Deterministic Repository Facts (`repository_facts.v2`)

## What is M2?

M2 answers:

> **What objectively exists in the pinned repository snapshot, and where did we observe it?**

M1 tells us **exactly which repository version** we are studying (a commit SHA).

M2 creates a **structured inventory of what objectively exists** in that version, with **evidence** pointers back to source:

```text
RepositorySnapshot   →   which state?
RepositoryFacts      →   what exists in that state?
EvidenceRef          →   where was that observed?
```

M2 is deterministic. It uses the filesystem, Python AST, and simple parsers — never an LLM, never architecture guesses.

Principle:

> M2 records what was observed and where it came from. It does not explain the architecture.

### M2 does

- deterministic repository inspection
- entity / file / fact extraction under one **generic `facts[]` envelope**
- **EvidenceRef** on path-backed facts (path, optional lines/symbol, artifact kind, commit) as an **array**
- stable, deterministic **`fact_id`** per fact
- language-neutral entity identity (merged with language-specific detail)
- centralized generated artifacts under `artifacts/<snapshot_id>/`
- reproducible, byte-equivalent output for the same commit
- open contract: new analyzers add new `fact_type`/`kind` values **without schema edits**

### M2 does not

- build an architecture graph
- infer semantic relationships
- use an LLM
- infer architecture
- classify business capabilities
- perform deep semantic analysis

## Milestone boundary

```text
M1 → Repository Snapshot
M2 → Repository Facts + Evidence
M3 → Engineering construction rules → EngineeringModel builder
M3.5/M4 → Architecture knowledge
M5 → Diagram projection
```

M2 is frozen for M3 construction-rule work: do not redesign M1/M2 contracts
when adding `engineering_construction_rules.v1`.

## M2 Input

A valid M1 artifact (either layout is accepted):

```text
artifacts/main_2026-09-22_ef11dac/snapshot/repository_snapshot.json
# or legacy-style path still readable if present:
# snapshots/main_2026-09-22_ef11dac/repository_snapshot.json
```

M2 does **not** accept “whatever is on main now.” M2 analyzes only the
checkout M1 persisted beside the snapshot artifact:

```text
<run>/repository/              → the exact checkout (persisted by M1)
git HEAD == snapshot.commit_sha  → verified before analysis
missing or stale HEAD            → clear FactsError, never a silent re-clone
```

By reading exactly M1's persisted checkout, M2 never analyzes a
**separately materialized repository state** — the facts always describe
the same frozen source tree the snapshot recorded, and re-running M2
cannot drift onto a different checkout.

M2 is the **only stage** that reads `repository/` directly; M3 onward
consume M2's `facts/` artifact, and the evidence tools (Tool 2/3) read
the same pinned checkout for evidence.

## M2 Output — generic `repository_facts.v2`

Domain object: **`RepositoryFacts`**
Schema: **`repository_facts.v2`** (authoritative: `schemas/repository_facts.v2.json`)

The root envelope is **stable**:

```text
schema_version | snapshot | repository | facts[] | analyzers[]
```

### Single-source observation rule (M2 v2 freeze)

```text
repository  = repository identity / metadata only (name, url)
facts[]     = single canonical observation store
counts()    = derived reporting/CLI summaries computed from facts[]
```

M3 and future consumers must read repository observations **exclusively** from `facts[]`. Do not look for `repository.files` or `repository.directories` — those keys do not exist in the contract.

Analyzer-specific root sections (`entities`, `python`, `configuration`, `dependencies`, `agents`, `mcp`, `documentation`, `tests`) are **not** part of the published contract. All of that detail lives in the generic `facts[]` array.

```json
{
  "schema_version": "repository_facts.v2",
  "snapshot": {
    "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
    "snapshot_id": "main_2026-09-22_ef11dac",
    "repository_url": "https://github.com/aws-samples/..."
  },
  "repository": {
    "name": "sample-Agentic-Ai-Data-Operations",
    "url": "https://github.com/aws-samples/sample-Agentic-Ai-Data-Operations.git"
  },
  "facts": [
    {
      "fact_id": "fact:file:pyproject.toml",
      "fact_type": "file",
      "kind": "repository_file",
      "source": "filesystem",
      "attributes": { "path": "pyproject.toml", "type": "toml", "size": 1234 },
      "evidence": [
        {
          "path": "pyproject.toml",
          "artifact_type": "file",
          "commit": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
        }
      ]
    },
    {
      "fact_id": "fact:entity:python:function:.claude/hooks/enforce_template_codegen.py::extract_file_paths",
      "fact_type": "entity",
      "kind": "function",
      "source": "entities",
      "attributes": {
        "entity_kind": "function",
        "name": "extract_file_paths",
        "qualified_name": ".claude/hooks/enforce_template_codegen.py::extract_file_paths",
        "language": "python",
        "path": ".claude/hooks/enforce_template_codegen.py",
        "line_start": 27,
        "line": 27
      },
      "evidence": [
        {
          "path": ".claude/hooks/enforce_template_codegen.py",
          "artifact_type": "python",
          "commit": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
          "line_start": 27,
          "symbol": "extract_file_paths"
        }
      ]
    }
  ],
  "analyzers": [
    { "name": "repository-intelligence", "version": "0.2.0" }
  ]
}
```

### FactRecord envelope

| Field | Type | Rules |
| --- | --- | --- |
| `fact_id` | string | Deterministic stable identity: `fact:<fact_type>:<stable-key>`. Unique, no UUIDs, no timestamps, no array position. |
| `fact_type` | string | **Open** vocabulary (not an enum). Initial: `file`, `directory`, `entity`, `import`, `dependency`, `configuration`, `artifact`, `documentation`, `test`; open values added by later analyzers: `interaction` (post-v2), `tool`, `mcp_server` (M12). |
| `kind` | string | **Open** sub-kind within `fact_type` (e.g. `module`/`class`/`function`, `repository_file`/`repository_directory`, `json`/`yaml`, `agent_definition`). |
| `source` | string | Producing stage/analyzer: `filesystem`, `python`, `configuration`, `dependencies`, `agents`, `mcp`, `documentation`, `tests`, `entities`. |
| `attributes` | object | Analyzer-specific payload. **`additionalProperties: true`** — extension without schema change. |
| `evidence` | array | 0..n `EvidenceRef` objects. Path-backed facts must carry ≥1 (invariant). |

### Initial `fact_type` mapping (specialized analyzers → generic facts)

| Specialized analyzer output | v2 `fact_type` | v2 `kind` (typical) | `source` |
| --- | --- | --- | --- |
| filesystem files | `file` | `repository_file` | `filesystem` |
| filesystem directories | `directory` | `repository_directory` | `filesystem` |
| entities (+ python module/class/function detail) | `entity` | `module` \| `class` \| `function` | `entities` |
| `python.imports` | `import` | `import` \| `import_from` | `python` |
| configuration | `configuration` | `json` \| `yaml` \| … | `configuration` |
| dependencies | `dependency` | section name (e.g. `dependencies`) | `dependencies` |
| agents | `artifact` | `agent_definition` | `agents` |
| mcp | `artifact` | `mcp_config` \| `mcp_server_source` \| … | `mcp` |
| tooling (agent `tools:` declarations, tool registry, MCP configuration) | `tool` \| `mcp_server` \| `interaction` | `mcp_tool` \| `built_in_tool` \| `registered`; `declares` \| `exposes` \| `configures` | `tooling` |
| documentation | `documentation` | `markdown` \| … | `documentation` |
| tests | `test` | `test_file` \| `test_directory` | `tests` |

Filesystem facts keep `path`, `type`, and `size` (files) / `path` (directories) in `attributes`, plus file-level evidence — **without** inventing line numbers.

Specialized analyzers remain specialized internally:

```text
specialized analyzer  →  generic FactRecord  →  facts[]
```

They are **not** required to share identical internal code.

### `fact_id` rules

Conceptually `fact:<stable identity>`. Examples:

```text
fact:file:pyproject.toml
fact:directory:src
fact:entity:python:function:src/app/main.py::main
fact:import:src/app/main.py:3:import:os
fact:dependency:pyproject.toml:dependencies:jsonschema
fact:configuration:.github/dependabot.yml
fact:artifact:agents:agent_definition:.claude/agents/dag-agent.md
fact:documentation:README.md
fact:test:test_file:tests/test_app.py
fact:tool:mcp:glue-athena:get_table          # server-scoped MCP tool (M12)
fact:tool:builtin:Bash                       # namespace-free declaration (M12)
fact:mcp_server:server:glue-athena           # registered server (M12)
fact:interaction:declares:<source_fact_id>:<target_fact_id>   # (M12)
```

Derived only from natural keys (path, qualified_name, line+module, source+section+name). Never from insertion order, list index, timestamps, or random UUIDs. Two full pipeline runs over the same commit produce the **same `fact_id` sequence**.

### `analyzers` (array)

```json
"analyzers": [{ "name": "repository-intelligence", "version": "0.2.0" }]
```

A list so multiple suites can contribute later without breaking the envelope.

### Derived counts (`counts()`)

`counts()` is a **reporting/CLI convenience** only:

```text
facts[]  →  counts()  →  reporting metrics
```

It is **not**:

- part of the `repository_facts.v2` JSON contract
- canonical repository observations
- an alternate storage for facts
- an analyzer-specific schema section

Values such as `agents`, `mcp`, `documentation`, `tests`, `python_modules`, `python_symbols`, `tools`, `mcp_servers`, and `interactions` are labels for humans/CLI output, always recomputed from `facts[]`. Downstream consumers (M3) must query `facts[]` directly — never depend on `counts()` for observation data.

## Evidence (M2.2)

### What `EvidenceRef` is

A small, reusable first-class pointer that answers **where this fact was observed**. It is not an interpretation and not a graph edge.

| Field | Required | Meaning |
| --- | --- | --- |
| `path` | yes | Repository-relative source path (never absolute, never `..`) |
| `artifact_type` | yes | Controlled vocabulary of source kind |
| `commit` | yes | Full 40-char SHA; **must equal** `snapshot.commit_sha` |
| `line_start` | no | 1-based definition line when meaningful |
| `line_end` | no | 1-based end line when the runtime provides it |
| `symbol` | no | Simple symbol name when line-level evidence applies |

On each FactRecord, evidence is an **array** (`evidence: [EvidenceRef, ...]`).

### Controlled `artifact_type` vocabulary

```text
python | configuration | agent_definition | mcp_definition
documentation | test | file | directory
```

(`artifact_type` is a small controlled set. `fact_type` / `kind` are **open** strings.)

### Why evidence exists

M3 will claim relationships. Each claim must point at a fact. Each fact must point at a file (and optionally a line) **at a known commit**. Evidence makes that chain auditable without re-deriving architecture in M2.

Evidence ≠ interpretation:

```text
Evidence:  path=.claude/agents/dag-agent.md, artifact_type=agent_definition, commit=ef11dac…
Not evidence:  "dag-agent is the DAG orchestration component"
```

### Evidence attachment rules

- **Python modules / entities of kind module:** file-level only (`path`, `commit`, optional `symbol`); no forced lines.
- **Classes / functions / imports:** include `line_start` (and `line_end` when available) + `symbol`.
- **Configuration / agents / MCP / docs / tests:** file-level evidence (path + type + commit).
- **Dependencies:** file-level evidence pointing at the manifest path when known (`artifact_type: file`).
- **Files / directories:** self-evidence (`artifact_type` `file` / `directory`) at the snapshot commit.
- File-level facts do not invent fake line numbers.

### Evidence integrity (enforced)

- `commit` == `snapshot.commit_sha` (lowercase 40-hex) for **every** `EvidenceRef` in `facts[].evidence[]`
- `path` repository-relative, non-empty, normalized, not absolute, no `..`
- `line_start >= 1`, `line_end >= line_start` when present
- `artifact_type` in the controlled vocabulary
- Path-backed facts must have at least one evidence entry
- Deterministic ordering (evidence follows parent fact order)

## Centralized artifact layout (M2.2)

`artifacts/` is the **single generated-output root** for M1–M4.

**Default root** is resolved from the **project/repository root** (package location), not the process current working directory:

```text
<project-root>/artifacts/
```

Precedence:

```text
explicit --output / output_root=   → use exactly the requested override
otherwise                          → <project-root>/artifacts/
```

```text
artifacts/
└── <snapshot_id>/
    ├── schemas/
    │   ├── repository_facts.v2.json
    │   └── repository_snapshot.v1.json
    ├── facts/
    │   └── repository_facts.json
    ├── repository/
    │   └── <pinned checkout, exact M1 commit>   (persisted by M1; M2 reads it)
    └── snapshot/
        └── repository_snapshot.json
```

For the pinned ADOP snapshot:

```text
artifacts/
└── main_2026-09-22_ef11dac/
    ├── schemas/
    │   ├── repository_facts.v2.json
    │   └── repository_snapshot.v1.json
    ├── facts/
    │   └── repository_facts.json
    ├── repository/
    │   └── <pinned checkout, commit ef11dac…>
    └── snapshot/
        └── repository_snapshot.json
```

Do **not** create generated outputs under `facts/`, `snapshots/`, `output/`, `generated/`, `build/`, or `tmp/`.

### Schema ownership

| Location | Role |
| --- | --- |
| `schemas/repository_*.json` | **Authoritative** source definitions (versioned with the code) |
| `artifacts/<id>/schemas/` | **Generated copies** for that pinned snapshot (byte-identical when produced by the current pipeline) |

There is one authoritative schema definition; artifact copies are snapshots of it for the pinned snapshot id — not a second competing contract.

### Artifact ownership / future milestones

M2 owns:

```text
artifacts/<snapshot_id>/facts/
artifacts/<snapshot_id>/schemas/
artifacts/<snapshot_id>/snapshot/
```

`artifacts/<snapshot_id>/repository/` is created and HEAD-verified by
**M1** (the persisted pinned checkout); M2 only reads and re-verifies it.

M3/M4 may later add siblings under the same `<snapshot_id>/` (e.g. `graph/`). Those directories are **not** created now.

### Cross-artifact consistency (enforced)

```text
directory <snapshot_id>
        ↕
repository_facts.snapshot.snapshot_id
        ↕
repository_snapshot (directory under artifacts/)

repository_facts.snapshot.commit_sha
        =
repository_snapshot.snapshot.commit_sha
        =
every EvidenceRef.commit
```

Also: `repository_url` agrees between facts and snapshot.

## Entity model (M2.1)

Entities are **language-neutral observations**, not graph nodes. In v2 they appear as `facts[]` entries with `fact_type: "entity"`.

| Attribute / kind | Meaning |
| --- | --- |
| `kind` | `module` \| `class` \| `function` (small closed set for the language-neutral view) |
| `attributes.name` | Simple symbol name (e.g. `main`) |
| `attributes.qualified_name` | **Stable identity** — see rules below |
| `attributes.language` | Extractor language, e.g. `python` |
| `attributes.path` | Repository-relative source path (provenance) |
| `attributes.line_start` | Definition line when known (provenance) |
| `attributes.line_end` | End line when the runtime AST provides it (optional) |
| `evidence` | `EvidenceRef[]` for this entity (required) |

Language-specific detail from the Python analyzer (methods, parse flags, module member lists, etc.) is **merged into `attributes`** of the matching entity fact. Compose-time `validate_cross_representation()` still enforces that the dual views agree before publish; the published document only carries the unified facts envelope.

### Identity rules

```text
module:          qualified_name = <path>
class/function:  qualified_name = <path>::<name>
```

Uniqueness key for downstream use:

```text
(entity_kind, qualified_name, language)
```

Example — two functions both named `main` do **not** collide:

```text
src/foo.py::main
src/bar.py::main
```

M3 should derive node IDs from these fields without inventing a second identity scheme.

## Provenance

- **Evidence:** every path-backed fact carries `evidence[].path` + `evidence[].commit` (+ lines/symbol when known).
- **Snapshot:** `facts.snapshot.commit_sha` is the full 40-character M1 SHA; also `snapshot_id`, `repository_url`.
- Traceability: `commit_sha` + `path` (+ `line_start`) → exact source at that commit.

## Determinism / Reproducibility

Same commit + same analyzer version + same input ⇒ **byte-equivalent** facts JSON:

- facts sorted by (`fact_type` vocabulary order, `fact_type`, `fact_id`)
- within each source category, natural keys remain sorted (files/dirs by path; configuration/agents/mcp/docs/tests by path; dependencies by (source, section, name); entities by (kind, qualified_name); imports by (path, line, module))
- `fact_id` derived only from stable identity fields
- evidence follows parent ordering; no capture timestamps in facts JSON
- snapshot `captured_at` remains only in the snapshot artifact (M1 capture metadata), not in facts

## Extensibility (why v2)

**Core principle:**

> Adding a new analyzer adds facts; it does not add a new root-level RepositoryFacts field.

```text
NEW ANALYZER
     ↓
new fact records
     ↓
same RepositoryFacts schema
```

Not:

```text
NEW ANALYZER  →  new root-level JSON field  →  schema change  →  M3 change
```

A future analyzer (Terraform, social posts, …) emits additional FactRecords:

```json
{
  "fact_id": "fact:artifact:post-…",
  "fact_type": "artifact",
  "kind": "post",
  "source": "social-post-analyzer",
  "attributes": { "source": "linkedin" },
  "evidence": [
    {
      "path": "content/post.md",
      "artifact_type": "documentation",
      "commit": "…"
    }
  ]
}
```

This validates against `repository_facts.v2` **without changing the schema** because:

- `fact_type` and `kind` are plain strings (no enum)
- `attributes` allows additional properties
- `source` is a plain string
- the root envelope does not grow analyzer-specific keys

## Relationship boundary (M2 / M3)

```text
M2:  "What exists, and where did we observe it?"
M3:  "What relationships can we establish between those facts?"
```

M2 does **not** emit graph edges such as `HAS_FUNCTION`, `IMPORTS`, `CALLS`, `EXTENDS`.
M2 may record that module X contains function Y and that module X imports module Y as **facts with evidence**; wiring those into a graph is M3’s job.

### Example: fact → future relationship (not implemented in M2)

**M2 observes (with evidence):**

```text
fact_type=entity kind=function
  qualified_name = .claude/hooks/enforce_template_codegen.py::extract_file_paths
  evidence = path=…py, line_start=27, artifact_type=python, commit=ef11dac…

fact_type=entity kind=module
  qualified_name = .claude/hooks/enforce_template_codegen.py
  evidence = path=…py, artifact_type=python, commit=ef11dac…
```

**M3 may later derive (out of scope for M2):**

```text
module --HAS_FUNCTION--> function
```

Those edges are M3 concerns only. M3 should cite the M2 evidence when asserting an edge.

## How M2 Works

```text
M1 RepositorySnapshot
        ↓
Load + validate snapshot (commit_sha + repository.url required)
        ↓
Resolve the persisted pinned checkout beside the snapshot artifact
  (<run>/repository/, written and verified by M1)
        ↓
Verify its git HEAD == snapshot.commit_sha
  (missing or stale → FactsError; M2 never clones another copy)
        ↓
Filesystem / Python / Configuration / Dependency /
Agent / MCP / Documentation / Test analyzers
        ↓
Attach EvidenceRef to meaningful facts
        ↓
Build language-neutral entities; cross-check vs python detail
        ↓
Normalize all analyzer output → generic FactRecords (facts[])
        ↓
RepositoryFacts (schema_version/snapshot/repository/facts/analyzers)
        ↓
Schema validation (repository_facts.v2) + invariants
  (unique fact_ids, evidence commit/path, open types, root keys)
        ↓
artifacts/<snapshot_id>/facts/repository_facts.json
artifacts/<snapshot_id>/snapshot/repository_snapshot.json
artifacts/<snapshot_id>/schemas/*.json
        ↓
The pinned checkout stays in place for later reuse (M2 re-runs and the
evidence tools; M3+ consume facts, not the checkout)
```

## Code Flow

```text
cli.py::facts
   ↓
facts/builder.py::create_facts
   ↓ load_snapshot_file
facts/builder.py::_resolve_pinned_repository
   ↓ artifacts.repository_dir_for_snapshot (no clone)
   ↓ repository/git.py::get_head_sha (verify == snapshot.commit_sha)
facts/builder.py::run_analyzers
   ├─ facts/filesystem.py
   ├─ facts/python.py          (ast.parse)
   ├─ facts/configuration.py
   ├─ facts/dependencies.py
   ├─ facts/agents.py
   ├─ facts/mcp.py
   ├─ facts/documentation.py
   └─ facts/tests.py
   ↓
facts/builder.py::_attach_evidence → EvidenceRef
facts/builder.py::compose_facts
   ├─ normalize.build_entities + cross-representation check
   └─ records.build_fact_records → facts[]
   ↓
validate_facts_invariants + validate_facts_document
   ↓ schemas/repository_facts.v2.json
facts/builder.py::write_facts_artifact
   ↓
artifacts/<snapshot_id>/{facts,snapshot,schemas}/
```

Supporting modules:

| Path | Purpose |
| --- | --- |
| `src/repo_intelligence/facts/evidence.py` | `EvidenceRef`, path/line/commit validation |
| `src/repo_intelligence/artifacts.py` | Central `artifacts/<id>/` layout helpers |
| `src/repo_intelligence/facts/models.py` | `RepositoryFacts`, fact sorting, counts |
| `src/repo_intelligence/facts/records.py` | `FactRecord` build, `fact_id`, normalization |
| `src/repo_intelligence/facts/builder.py` | Pipeline, evidence attach, invariants, write layout |
| `src/repo_intelligence/facts/normalize.py` | Language-neutral entity identity + cross-check |
| `schemas/repository_facts.v2.json` | Authoritative facts contract (incl. EvidenceRef) |
| `schemas/repository_snapshot.v1.json` | Authoritative snapshot contract |
| `tests/test_facts.py` | M2 + evidence + v2 envelope/extensibility suite |
| `docs/m2-repository-facts.md` | This document |

## Fact vs Interpretation

### Valid M2 facts (observation)

```text
file fact: .claude/agents/iac-agent.md exists (type: agent_definition)
  evidence: path=.claude/agents/iac-agent.md, artifact_type=agent_definition, commit=…

entity fact (kind=function): src/pipeline.py::generate_pipeline at line 42
  evidence: path=src/pipeline.py, line_start=42, symbol=generate_pipeline, artifact_type=python, commit=…

dependency fact: pyproject.toml declares dependency jsonschema
  evidence: path=pyproject.toml, artifact_type=file, commit=…
```

### Not an M2 fact (interpretation)

```text
iac-agent is the infrastructure orchestration agent
generate_pipeline is the pipeline orchestrator for the core architecture
boto3 means the system uses S3
```

## Testing

| Kind | Coverage |
| --- | --- |
| Unit | filesystem, python, config, deps, agents, MCP, docs, tests, tooling |
| Evidence | required/optional fields, invalid paths/lines/commits, vocabulary, array form |
| Entity evidence | module/class/function/import representative checks via `facts[]` |
| File-level evidence | agent/MCP/doc/test/config/dependency/file/directory |
| Repository metadata | `repository.name` / `repository.url` only; no `files`/`directories` |
| File facts | `fact_type=file` kind=`repository_file` with path/type/size + evidence |
| Directory facts | `fact_type=directory` kind=`repository_directory` + evidence |
| No duplicate inventory | root has no `repository.files` / `repository.directories` |
| Counts derived | `counts()` recomputed only from `facts[]`; not in JSON contract |
| Generic root | root keys exactly `schema_version/snapshot/repository/facts/analyzers` |
| Fact preservation | representative file/directory/entity/import/config/dependency/artifact/doc/test |
| Stable `fact_id` | double compose → identical id sequence; no position dependence |
| Extensibility | synthetic `terraform` / `social-post-analyzer` facts validate **without schema edits** |
| Artifact layout | only `artifacts/<id>/{facts,snapshot,schemas}/`; v2 schema copy only |
| Cross-artifact | snapshot_id, commit_sha, repository_url, evidence commits |
| Determinism | full pipeline twice → byte-identical facts |
| Integration | local Git → M1 → M2; ADOP on-disk artifact when present |
| Schema | Draft 2020-12 validation of generated documents |

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

### CLI

```bash
python -m repo_intelligence facts \
  --snapshot artifacts/main_2026-09-22_ef11dac/snapshot/repository_snapshot.json
```

```text
Repository facts created

Snapshot : main_2026-09-22_ef11dac
Commit   : ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

Facts       : 1954
Files       : 322
Directories : 84
Entities    : 601
Python      : 115
Agents      : 15
MCP         : 13
Tests       : 39
Evidence    : 2012

Output:
artifacts/main_2026-09-22_ef11dac/facts/repository_facts.json
```

Exit codes: `0` success, `6` facts/snapshot errors, `8` schema validation failure, `3–7` Git/clone/ref.

## What M2 Does NOT Do

- build an architecture graph
- infer architectural relationships (“A calls B”)
- explain system behavior
- call an LLM
- use agents or agent frameworks
- use ReAct / planning / memory
- use MCP as an agent protocol (it only *sees* MCP files on disk)
- emit graph relationship types (`HAS_FUNCTION`, `IMPORTS`, `CALLS`, …)
- embeddings, RAG, Tree-sitter, type inference, data-flow analysis
- Mermaid/diagram generation

## Why M2 Exists

M3 will construct an architecture graph. Graphs need **edges backed by evidence**.

M2 is that substrate:

```text
Pinned repository
      ↓
Deterministic facts (generic facts[] + stable fact_id)
      ↓
Evidence
```

Every claimed relationship in M3 can point back to an M2 fact that points back to a file at a commit.

## M2 / M2.1 / M2.2 / v2 change notes

**M2:** eight specialized analyzers; initial aggregation contract.

**M2.1:** language-neutral entity identity (`qualified_name`), cross-representation validation.

**M2.2:**

- First-class `EvidenceRef` attached to meaningful facts
- Evidence integrity invariants + schema `$defs.EvidenceRef`
- Single generated-output root `artifacts/<snapshot_id>/` with `facts/`, `snapshot/`, `schemas/`
- Cross-artifact consistency tests (snapshot_id / commit / URL)
- Project-root-relative default output (not CWD)

**v2 (`repository_facts.v2`):**

- Root envelope only: `schema_version`, `snapshot`, `repository`, `facts[]`, `analyzers[]` (analyzers is now an **array**)
- `repository` = **identity/metadata only** (`name`, `url`); **no** `files` / `directories`
- All observations (including filesystem files/directories) live **only** in `facts[]`
  - file → `kind=repository_file`; directory → `kind=repository_directory`
- Deterministic unique `fact_id` (`fact:<type>:<stable-key>`)
- Open `fact_type` / `kind` / `source` / `attributes` for analyzer extensibility
- Evidence is always an **array** on each fact
- `counts()` = derived reporting convenience from `facts[]` (not in JSON contract)
- Specialized analyzers remain specialized internally; only the **published contract** is generic
- Still: no M3 graph, no LLM, no dual v1+v2 output

**M2 v2 freeze:** contract stable. New analyzers add facts, not root fields.

**Interaction facts (post-v2, same schema):**

- New open values only — **no `repository_facts.v2` schema bump** (`fact_type` / `kind` / `source` / `attributes` already open)
- `fact_type=interaction`, `kind=triggers`, `source=interactions`
- Endpoints are **existing `fact_id`s** (`attributes.source_fact_id` / `target_fact_id`)
- Emitted only from **explicit** statements (`Spawned by …`, `Spawn … Agent`, `agent(…)`) when both endpoints resolve uniquely — never from mentions, co-location, or name similarity
- `fact_id` = `fact:interaction:triggers:<source_fact_id>:<target_fact_id>` (deterministic)
- Observation only: M3 canonicalizes to `TRIGGERS` when both endpoints have engineering nodes

**Tooling facts (M12, same schema):**

- New open values only — still no `repository_facts.v2` schema bump
- `source=tooling` analyzer (`analyze_tooling`): pure structure-driven observation — agent frontmatter `tools:` lines (markdown parsed in frontmatter only), tool-registry `servers:`/`tools:` blocks, MCP configuration files (controlled name set); a file yields statements only when its structure matches
- Node facts: `fact_type=tool`, `kind=mcp_tool` (identity `tool:mcp:{server}:{name}`) or `built_in_tool` (identity `tool:builtin:{name}`); `fact_type=mcp_server`, `kind=registered` (identity `mcp_server:{name}`) — merged evidence from every declaring source
- Relationship facts: `fact_type=interaction`, `kind=declares|exposes|configures`, endpoints as existing `fact_id`s; unresolvable/ambiguous targets stay as `attributes.target_label` + `attributes.reason` (`ambiguous_identity` when a bare tool name collides with an MCP tool name; `target_not_resolved` for unregistered servers / unconfigured servers) so M3 publishes unresolved claims instead of edges; `attributes.conflicting` marks a server claimed by several configuration files
- `fact_id` = `fact:tool:mcp:{server}:{name}` / `fact:tool:builtin:{name}` / `fact:mcp_server:server:{name}` / `fact:interaction:{kind}:{source_fact_id}:{target_fact_id}` (deterministic)
- Observation only: M3 canonicalizes to `DECLARES` / `EXPOSED_BY` / `CONFIGURED_BY` when both endpoints have engineering nodes (see [M12](m12-agent-tool-mcp-knowledge.md))
