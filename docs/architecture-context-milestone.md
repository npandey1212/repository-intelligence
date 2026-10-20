# Architecture Context Milestone (`architecture_context.v1`)

## What this milestone is

```text
M1 snapshot + M2 facts + M3 engineering + M4 architecture model
      +
contract/architecture_context_template.json   (fixed repo-level contract)
      ↓
deterministic context generator   (no LLM)
      ↓
artifacts/<snapshot_id>/architecture_context/architecture_context.md
```

The first **agent-oriented context artifact**, defined by a single
repo-level template contract and emitted as Markdown. It summarizes what
the deterministic pipeline already established so a future investigation
tool can answer:

> What is this repository/system, what are its major architectural concepts
> and workflows, and where should I investigate next?

Canonical output (run-specific artifact directory):

```text
artifacts/<snapshot_id>/architecture_context/architecture_context.md
```

For the current pinned run:

```text
artifacts/main_2026-09-22_ef11dac/architecture_context/architecture_context.md
```

There is exactly one contract — `contract/architecture_context_template.json`
(no JSON Schema file, no generated JSON document):

```bash
python -m repo_intelligence context
```

Generator: `src/repo_intelligence/context/`
(`builder.py` populates the template and renders the Markdown **from the
populated document**; `validator.py` validates all layers.)

Contract (repository-agnostic): [architecture-context.md](architecture-context.md)

---

## Position in the pipeline

```text
Any Repository
      ↓
M1 — Pinned Repository Snapshot
      ↓
M2 — Deterministic Repository Facts
      ↓
M3 — Engineering Knowledge Model
      ↓
M4 — Validated Architecture Knowledge
      ↓            └──→ M5 Diagram Projection (side branch)
      └── Architecture Context   ← this milestone
                ↑
    contract/architecture_context_template.json
                ↓
      architecture_context.md
                ↓
      Tool 1 get_architecture_context()   (implemented)
      Tool 2 search_repository()          (implemented)
      Tool 3 read_file()                  (implemented)
                ↓
      [future] agent / LLM tool loop()
```

The context sits **beside** M4, not instead of it:

```text
architecture_model.json        = structured architectural truth (M4)
contract/…_template.json       = fixed section/field structure (empty values)
architecture_context.md        = populated context document (generated)
```

M1–M5 are inputs/consumers only — this milestone does not redesign them.

---

## What it contains

### Template sections (M1–M4 sources)

| Section | Source |
| --- | --- |
| `repository` | M4 repository identity → M2 snapshot → M3 metadata; placeholders otherwise |
| `structure` | M2 directory facts + documentation/config paths + artifact subkind groups |
| `technology` | artifact types on code entities (languages) + external packages/dependency facts (libraries); frameworks/infrastructure/external services always empty |
| `architecture` | exactly `architecture.nodes[]` and `architecture.relationships[]`; boundaries always empty |
| `workflows` | architecture relationships grouped by source component |
| `interfaces` | always empty (no evidence source) |
| `operations` | always empty (build/test/run/deploy not established) |
| `documentation` | documentation paths from facts, README first |
| `evidence` | input artifacts + template sources; known unknowns; analysis limitations |

Absent fields use explicit markers — `- …: _not established_` (scalars)
and `- …: _none recorded_` (lists) — never fabricated content.

### Markdown sections (fixed heading order)

1. Repository
2. Structure
3. Technology
4. Architecture
5. Workflows
6. Interfaces
7. Operations
8. Documentation
9. Evidence / Limitations

The Markdown is rendered **only** from the populated template document; it
is never built directly from M1–M4.

---

## Generation

```text
architecture_model.json  ─┐
                          ├─→ build_architecture_context_document()
repository_facts.json    ─┤         (populates contract template in memory)
contract/architecture_context_template.json ─┘
                                   ↓
                   populated architecture context document
                                   ↓
               render_architecture_context_markdown()
                                   ↓
                 architecture_context.md   (written artifact)
                                   ↓
      validate: shape → document vs M1–M4 → Markdown vs document
                → Markdown vs M1–M4
                                   ↓
      write the Markdown under <output>/<snapshot_id>/architecture_context/
```

Rules:

1. Load existing deterministic M1–M4 artifacts only.
2. Populate the template contract (field by field, from evidenced data).
3. Render the Markdown strictly from the populated document.
4. Preserve repository name, commit, snapshot id (M1 semantics).
5. Validate all layers before writing; write Markdown only.
6. Output lands under `artifacts/<snapshot_id>/architecture_context/`.
7. The snapshot/run id is never hard-coded: it comes from the explicit
   argument, the existing run directory under the output root, or the
   supplied input artifacts.

- **No LLM**, no embeddings, no RAG, no graph database, no agent framework.
- The generator does **not** read analyzed-repository file bodies; it renders
  knowledge already recorded in M1–M4.
- Does **not** modify the architecture model or any M1–M4 artifact.

CLI:

```bash
python -m repo_intelligence context \
  [--architecture-model PATH] [--facts PATH] [--output PATH] [--output-file PATH]
```

(`--output-file` overrides the Markdown path.)

---

## Truth and provenance rules

```text
Established fact     → copied from M1–M4 artifacts (identity, counts, ids)
Derived knowledge    → architecture classifications (e.g. derived), groupings
Interpretation       → explicitly labeled (e.g. overviews = count summaries)
Unresolved question  → known_unknowns lines mirroring M4 unresolved claims
Not established      → empty template field rendered with the absence marker
```

Provenance uses **existing identifiers only** (`concept:…`, `rel:…`,
`map:…`, repository evidence paths, input artifact paths). Each
`evidence.sources` entry carries the pinned commit and snapshot id; the
template is listed as an explicit source. No second evidence-id system.

Forbidden:

- new architecture nodes or relationships
- relationships inferred from proximity, naming, imports, or co-occurrence
- paraphrased "purpose" claims from file contents
- presenting `derived` as `observed`
- presenting unresolved claims as validated relationships
- filling empty template fields with speculative content

A smaller truthful context is preferred over a larger speculative one.

---

## Repository-agnostic principle

The contract and generator are **repository-agnostic**:

```text
Any Repository → M1–M4 artifacts + template → architecture_context.md
```

- No hard-coded component, workflow, directory, agent, or service names in
  generator source.
- No `if repository == …` or domain term checks in the generator.
- Names and concepts in the output come from input artifact data only.
- Tests exercise two synthetic fixtures (Repo A: `ComponentA`/`WorkflowA`,
  Repo B: `ServiceX`/`WorkflowX`) and scan generator source for forbidden
  reference-repository terms to prove the machinery is generic.
- ADOP (`sample-Agentic-Ai-Data-Operations`) is the current
  **reference/test repository**, not part of the generic definition.
  Domain terms may appear in *generated output* when evidenced; they must
  never appear as assumptions in *implementation*.

---

## Validation

Implemented in `src/repo_intelligence/context/validator.py`:

| Layer | Check | Meaning |
| --- | --- | --- |
| 1. Shape | `validate_context_document_shape` | sections/field keys, order, and types equal `contract/architecture_context_template.json`; `schema_version` matches |
| 2. Document vs M1–M4 | `validate_context_document` | identity matches M4/M2; component/relationship sets == M4 sets; workflows reference only validated relationships (each exactly once); structure/technology/documentation recompute equal; unevidenced fields empty; source labels/commit/snapshot correct; count/limitation/unresolved lines mirror inputs |
| 3. Markdown vs document | `validate_markdown_context_consistency` | heading order, identity lines, list bullets, component/relationship/workflow blocks, sources, evidence, unresolved lines all match the document |
| 4. Markdown vs M1–M4 | `validate_architecture_context_document` | the rendered Markdown checked directly against the architecture model and facts |

Plus:

| Check | Meaning |
| --- | --- |
| No repository-specific hard-coding | generator source scanned for forbidden domain terms |
| Determinism | same inputs → byte-identical Markdown (no timestamp anywhere) |
| Artifact layout | Markdown under `<output>/<snapshot_id>/architecture_context/`; no top-level `artifact/` output |

---

## Tests

`tests/test_architecture_context.py` (pytest suite):

```text
Contract:
  test_contract_template_exists_and_is_valid_json
  test_contract_template_schema_version
  test_contract_template_structure_matches_spec
  test_contract_template_contains_no_domain_values
  test_contract_template_loads_through_generator_contract
Generator consumes the contract:
  test_generator_consumes_contract_template
  test_custom_template_drives_output
  test_render_rejects_documents_from_other_contracts
M1–M4 projection:
  test_sections_project_m1m4
  test_markdown_has_required_sections
  test_markdown_is_rendered_from_document_only
Absence semantics:
  test_missing_information_is_not_fabricated
  test_document_equality_checks_reject_fabrication
Consistency / tamper detection:
  test_context_document_identity
  test_context_document_evidence_integrity
  test_context_document_contains_no_unsupported_relationships
  test_unresolved_claims_are_counted_explicitly
  test_markdown_tampering_is_detected
  test_markdown_vs_m1m4_validation_detects_tampering
  test_shape_validation_detects_contract_drift
Determinism / rejection:
  test_architecture_context_determinism
  test_architecture_context_rejects_wrong_documents
Repository agnosticism:
  test_architecture_context_is_repository_agnostic
Artifact layout:
  test_create_writes_markdown_at_run_path
  test_default_paths_use_run_directory
  test_no_stale_artifact_context_references
  test_no_stale_json_contract_references
Integration (skipped when absent):
  test_pinned_run_generation
  test_pinned_run_artifact_location
  test_pinned_run_identity_and_evidence
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_architecture_context.py -q
```

---

## What is deliberately not included

Hard boundary for this milestone — **not** implemented:

```text
LLM / agent loop / tool calling / ReAct
planning, agent state, memory
RAG, embeddings, vector stores
multi-agent orchestration
UI / interactive visualization
changes to M1–M5 semantics
architecture model expansion
generated architecture_context.json / JSON Schema file
```

The Markdown document plus its template are the surface the tool layer
reads. Tool 1 (`get_architecture_context`) now retrieves it, Tool 2
(`search_repository`) locates textual evidence in the pinned repository
snapshot, and
Tool 3 (`read_file`) retrieves exact file content; the agent loop is
the next milestone.

---

## Summary

```text
M1 → M2 → M3 → M4
              │
              └── contract/architecture_context_template.json
                        ↓
              artifacts/<snapshot_id>/architecture_context/
                architecture_context.md
                        ↓
              Tool 1 get_architecture_context() → ToolResult
```

Tool 1 (read-only retrieval of this document) now lives in the tool
layer: [tools/get-architecture-context.md](tools/get-architecture-context.md).
Tool 2 extends the same layer with search — semantic retrieval since
M18 (this milestone originally shipped it as deterministic text
search):
[tools/search-repository.md](tools/search-repository.md), and Tool 3
with exact evidence retrieval:
[tools/read-file.md](tools/read-file.md).

Established here: a deterministic, template-contract-driven,
provenance-preserving, repository-agnostic **architecture context
document** — nothing more.

**Downstream (post-M17):** the M5 diagram projection this pipeline
positions as a side branch is now consumed end-to-end by the
Architecture Visualization Agent — it adapts a validated `DiagramSpec`
onto the same projection and lets `graph/exporters` serialize Mermaid /
DOT / SVG (`docs/architecture-visualization-agent.md` §10).
