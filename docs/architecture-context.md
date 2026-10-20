# Architecture Context Contract (`architecture_context.v1`)

## What this artifact is

The Architecture Context is the first **agent-oriented context artifact**.
It is defined by a single repo-level template contract and generated as a
Markdown document:

```text
M1 snapshot → M2 facts → M3 engineering → M4 architecture
                                                  │
contract/architecture_context_template.json       │
        (exact section/field structure)           │
                    │                             │
                    └───────────┬─────────────────┘
                                ↓
                deterministic context generator   (no LLM)
                                ↓
        artifacts/<snapshot_id>/architecture_context/architecture_context.md
```

It answers:

> What is this repository/system about, what are its major architectural
> concepts and workflows, and what background should an investigator
> understand before investigating it?

## What this artifact is not

| Artifact | Role |
| --- | --- |
| `architecture_model.json` | Structured architectural **truth** (M4) |
| `contract/architecture_context_template.json` | Fixed **template contract** (section/field structure only, no content) |
| `architecture_context.md` | Investigator-oriented **context document** generated from the template + M1–M4 |

The context is not a replacement for the architecture model, not a second
evidence database, and not an expansion of M4. The Markdown is never
generated directly from M1–M4 — it is rendered **from the populated
template document**, which is itself validated against M1–M4.

## The template contract

`contract/architecture_context_template.json` is the only contract for this
artifact. It fixes the exact structure; every field starts empty
(`""` or `[]`):

| Section (`schema_version` first) | Fields |
| --- | --- |
| `repository` | `name`, `description`, `purpose`, `project_type`, `primary_domains` |
| `structure` | `overview`, `key_directories`, `key_files`, `important_artifacts` |
| `technology` | `languages`, `frameworks`, `libraries`, `infrastructure`, `external_services` |
| `architecture` | `overview`, `components`, `relationships`, `boundaries` |
| `workflows` | `overview`, `workflows` |
| `interfaces` | `overview`, `apis`, `commands`, `events`, `external_interfaces` |
| `operations` | `build`, `test`, `run`, `deploy` |
| `documentation` | `overview`, `key_documents` |
| `evidence` | `sources`, `known_unknowns`, `analysis_limitations` |

- `schema_version` is `architecture_context.v1`.
- The generator's document has **exactly** the template's sections and
  field order (validated); a custom template drives the output, and a
  missing template is a hard error.
- There is no JSON Schema file and no generated JSON artifact — the
  template itself is the contract.

## Information sources (allowed inputs)

1. `architecture_model.json` (M4 — repository/engineering/architecture/mappings envelope)
2. `repository_facts.json` (M2)
3. `contract/architecture_context_template.json` (structure + evidence source entry)

No file contents from the analyzed repository are read at generation time.
No LLM, embeddings, RAG, graph database, or agent framework is used.

### Section sources

| Section | Populated from |
| --- | --- |
| `repository` | M4 repository identity, then M2 snapshot, then M3 metadata; placeholder text otherwise |
| `structure` | M2 directory facts; documentation/config paths; M3 artifact subkind groups |
| `technology` | artifact types on `code_entity` evidence (languages); external package + dependency facts (libraries); everything else empty |
| `architecture` | exactly `architecture.nodes[]` (components) and `architecture.relationships[]` (relationships); boundaries always empty |
| `workflows` | architecture relationships grouped by source component (never invented narratives) |
| `interfaces` | always empty — no interface evidence is established |
| `operations` | always empty — build/test/run/deploy are not established |
| `documentation` | documentation paths from facts, README first |
| `evidence` | input artifacts + template as sources; count-derived known unknowns; analysis limitations |

## Physical representation

```text
artifacts/<snapshot_id>/architecture_context/architecture_context.md
```

For the current pinned run:

```text
artifacts/main_2026-09-22_ef11dac/architecture_context/architecture_context.md
```

- Markdown is the **only** generated context artifact; nothing else is
  written (no JSON, no schema copy).
- Output directory follows the run-specific artifact layout
  `<output>/<snapshot_id>/architecture_context/`. The snapshot/run id is
  never hard-coded: it comes from the explicit argument, the existing run
  directory under the output root, or the supplied input artifacts
  (document identity / `artifacts/<snapshot_id>/…` path).

## Markdown structure

```text
# Architecture Context

- Contract: `architecture_context.v1`

## Repository            ← 9 sections, template order
## Structure
## Technology
## Architecture
## Workflows
## Interfaces
## Operations
## Documentation
## Evidence / Limitations
```

Per-section rendering rules:

- Each template field is either an inline scalar line
  (`- <Label>: <value>`; empty value → `_not established_`) or, for lists,
  `_none recorded_` when empty and a `### <Label>` bullet list otherwise.
- `architecture.components` render as `### Component: <name>` blocks with
  `- Component id:`, `- Kind:`, `- Classification:`, `- Evidence:`.
- Relationships render under `### Relationships` as `- \`rel:id\`` with
  indented `Kind`/`Classification`/`Endpoints`/`Evidence` lines.
- Workflows render as `### Workflow: <source name>` with `- Source:`,
  `- Participants:`, `- Relationships:`, `- Evidence:`.
- `evidence.sources` render as `- <label>: <path>` (label derived from the
  path); `known_unknowns` and `analysis_limitations` as bullet lists.
- Long lists are capped (20 entries for key files/documents, 20 sample
  unresolved lines) with an explicit cap note in the same list.

## Absence semantics (no fabrication)

Dimensions without reliable evidence are never filled in:

```text
- Description: _not established_     scalar fields with no evidence
- Primary domains: _none recorded_   list fields with no evidence
```

`interfaces`, `operations`, `boundaries`, `frameworks`, `infrastructure`,
`external_services`, `description`, `purpose`, `project_type`, and
`primary_domains` are always empty for every repository — the generator
has no data source for them. Known unknowns and analysis limitations state
counts explicitly (for example `Test files observed as facts: 1.`,
`Engineering -> architecture mappings recorded in M4: 1.`), and unresolved
architecture claims are copied line by line — never dropped or reclassified.

## Claims the artifact is allowed to make

- Repository identity (name, commit, snapshot id) from M1/M2/M4.
- Component lists exactly equal to `architecture.nodes[]`.
- Relationship lists exactly equal to `architecture.relationships[]`.
- Evidence paths that exist on the referenced node/relationship.
- Observed fact summaries (directories, files, documents, languages,
  libraries, artifact groups).
- Explicit unknown/limitation statements with counts.

## Claims the artifact must not make

- New architecture nodes or relationships.
- Relationships inferred from proximity, naming, imports, or co-occurrence.
- Paraphrased purpose narratives from file contents (bodies are not read).
- Runtime/deployment/interface/operations claims (always `_not established_`).
- Upgraded classifications (`derived` must not be presented as `observed`).
- Any value not present in the inputs (the template fields that inputs
  cannot evidence stay empty).

## Evidence and truth rules

Provenance uses existing identifiers only:

- architecture node ids (`concept:…`)
- architecture relationship ids (`rel:…`)
- mapping ids (`map:…`)
- repository paths from `evidence[]`
- input artifact paths listed under `evidence.sources`

Each source line carries the pinned commit and snapshot id; the template
itself appears as `Architecture context template:
contract/architecture_context_template.json`. No second evidence-id system
is introduced.

## Repository agnosticism

The template and generator are repository-agnostic:

- No hard-coded component names, workflow names, directory names, agent
  names, cloud services, or domain terms in generator source.
- A different repository producing the same M1–M4 contracts must generate
  a valid context from its own data (tests prove this with two synthetic
  fixtures).

## Determinism

- Same inputs → byte-identical Markdown (no timestamp, no
  `generated_at` anywhere).

## Validation requirements (four layers)

Implemented in `src/repo_intelligence/context/validator.py`:

1. **Shape vs template** — `validate_context_document_shape`:
   sections/field keys and order equal the template; scalars are strings,
   lists are lists of strings; `schema_version` matches.
2. **Document vs M1–M4** — `validate_context_document`: identity matches
   M4/M2; component/relationship sets are exactly the M4 sets; workflows
   reference only validated relationships and cover each exactly once;
   structure/technology/documentation recompute to the same values;
   unevidenced fields are empty; sources have correct labels, commit, and
   snapshot; count/limitation/unresolved lines mirror the inputs.
3. **Markdown vs document** — `validate_markdown_context_consistency`:
   heading order, identity lines, list bullets, component/relationship/
   workflow blocks, source lines, and evidence all match the document (the
   Markdown is a faithful projection, not an independent source).
4. **Markdown vs M1–M4** — `validate_architecture_context_document`:
   the rendered Markdown checked directly against the architecture model
   and facts.

Additionally: generator source contains no repository-specific
hard-coding, and artifact paths follow
`<output>/<snapshot_id>/architecture_context/` (no top-level `artifact/`
output).

## Non-goals (this milestone)

- Agent loop, LLM, tool calling, UI. (Retrieval of this document now
  exists as read-only Tool 1:
  [tools/get-architecture-context.md](tools/get-architecture-context.md).)
- Architecture model changes (M4 semantics unchanged).
