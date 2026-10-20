# M3 — Engineering Construction Rules (`engineering_construction_rules.v1`)

## Why M3 exists

M1 and M2 are frozen:

```text
M1  Pinned repository snapshot     →  which commit?
M2  Deterministic repository facts →  what did we observe, and where?
```

M2 facts are **observations**, not graph nodes. A fact says a file exists at a
path, or that a function is defined at a line. It does **not** say how the
future EngineeringModel should identify that thing, how two observations
relate, or which claims are canonical.

M3 construction rules answer the next question **without building the graph
yet**:

```text
M2 Facts
    ↓
M3 Construction Rules     ← this milestone (rulebook only)
    ↓
Engineering nodes / relationships / unresolved candidates
    ↓
Validation
    ↓
EngineeringModel           ← later builder (not this milestone)
    ↓
ArchitectureModel          ← still later
```

Plain-language boundary:

| Layer | Question |
| --- | --- |
| M2 | What did we observe? |
| M3 construction rules | How do we deterministically interpret those observations as engineering knowledge? |
| EngineeringModel | What engineering structure can we establish from those rules? |
| ArchitectureModel | What higher-level architecture can be derived from the engineering structure? |

The deliverable is a **versioned rulebook**, not NetworkX/Neo4j/Neptune output.
A future builder may use any runtime store; the matrix describes domain
semantics.

---

## Matrix structure

Authoritative matrix (canonical location — not the project root):

```text
artifacts/main_2026-09-22_ef11dac/engineering/engineering_construction_rules.v1.json
```

Authoritative schema (unchanged by the artifact move):

```text
schemas/engineering_construction_rules.v1.json
```

Loader / validator helper (not a graph builder):

```text
src/repo_intelligence/construction/
```

Top-level sections:

| Section | Role |
| --- | --- |
| `input` | Binds the rulebook to `repository_facts.v2`. |
| `node_rules` | Which M2 facts create nodes (`CREATE_NODE`) or stay evidence-only (`EVIDENCE_ONLY`). |
| `relationship_rules` | Which M2 facts / derivations create `IMPORTS`, `DEPENDS_ON`, `CONTAINS`, `TRIGGERS`, and (M12) `DECLARES`, `EXPOSED_BY`, `CONFIGURED_BY`. |
| `identity_rules` | Deterministic ID templates; forbidden ID ingredients. |
| `evidence_rules` | Evidence is mandatory on canonical objects; lists evidence-only fact pairs. |
| `unresolved_rules` | Why/when claims go to `engineering.unresolved[]` instead of canonical edges. |
| `validation_rules` | Invariants a future EngineeringModel builder must enforce. |
| `deferred` | Explicitly out of scope for V1 (architecture, LLM, graph DBs, …). |

`node_rules` use a closed `action` vocabulary:

- `CREATE_NODE` — emit a canonical engineering node (requires evidence).
- `EVIDENCE_ONLY` — keep the fact as supporting evidence; do **not** emit a separate node.

---

## Fact-to-node examples

### File → artifact/file

```text
M2:  file / repository_file
       attributes.path = "src/main.py"
       evidence = [{ path: "src/main.py", artifact_type: "file", commit: … }]

Rule: node.file.repository_file
        action = CREATE_NODE
        node   = artifact / file
        identity = artifact:file:{path}
        classification = observed
        requires_evidence = true

M3:  node id = artifact:file:src/main.py
```

### Directory → artifact/directory

```text
M2:  directory / repository_directory  path="src"
M3:  artifact:directory:src
```

### entity/class → code_entity/class

```text
M2:  entity / class
       path           = src/service.py
       qualified_name = src/service.py::Service

Rule: node.entity.class
        identity = code_entity:{entity_key}
        entity_key = normalize(path, qualified_name)
        classification = observed

M3:  code_entity:src/service.py::Service
```

**Why normalization?** M2 `qualified_name` already carries path context.
Concatenating `{path}::{qualified_name}` blindly would produce the incorrect
`code_entity:src/service.py::src/service.py::Service`. M3 must not duplicate
the repository path.

If M2 ever supplies a bare symbol name:

```text
path = src/service.py
qualified_name = Service

→ code_entity:src/service.py::Service
```

### entity/function → code_entity/function

```text
M2:  path = src/service.py
     qualified_name = src/service.py::run

M3:  code_entity:src/service.py::run
```

(not `code_entity:src/service.py::src/service.py::run`)

### entity/module → code_entity/module

```text
identity = code_entity:{path}     # e.g. code_entity:src/service.py
```

**Not** `python-module:…`, `ast-module:…`, or `tree-sitter-module:…`.
Identity describes the **engineering entity**, not the analyzer mechanism.

### Specialized artifacts

These M2 kinds create `artifact / <kind>` nodes (still engineering layer, not
architecture):

```text
artifact / agent_definition   →  artifact:agent_definition:<path>
artifact / agent_command      →  artifact:agent_command:<path>
artifact / agent_config       →  artifact:agent_config:<path>
artifact / mcp_config         →  artifact:mcp_config:<path>
artifact / mcp_server_source  →  artifact:mcp_server_source:<path>
```

They do **not** create `concept:agent`, `service`, `application`, or
`concept:mcp-system`. Those would be architecture interpretation.

### Evidence-only facts (do not create separate canonical nodes)

```text
artifact / mcp_documentation | markdown | python | shell
configuration / file | json | toml | yaml
documentation / documentation | markdown | readme | runbook
test / test_directory | test_file
```

**Key modeling principle:** many of these facts describe *another aspect of a
path that already has a canonical node*.

```text
file / repository_file     path=config.yaml   →  artifact:file:config.yaml
configuration / yaml       path=config.yaml   →  (evidence only — same file)
documentation / readme     path=README.md     →  (evidence only)
test / test_file           path=…/test_x.py   →  (evidence only)
```

Do not create duplicate graph nodes merely because two M2 facts classify the
same physical artifact differently.

---

### Tool and MCP-server nodes (M12)

Two open fact types (added to M2 by the `tooling` analyzer, same envelope)
create **semantic** engineering nodes — not path classifications:

```text
tool / mcp_tool       →  tool:mcp:{server}:{name}       (server-scoped identity)
tool / built_in_tool  →  tool:builtin:{name}            (no server namespace)
mcp_server / registered →  mcp_server:{name}            (semantic entity, distinct
                                                          from artifact/mcp_server_source)
```

Evidence must cite an explicit registry entry or an explicit agent tool
declaration — never a Python function name or a prose mention. A bare
declaration whose name also appears as an MCP tool name stays ambiguous
(unresolved) instead of becoming a `tool/built_in` node.

---

## Fact-to-relationship examples

### import → IMPORTS (observed)

```text
M2:  import / import
       path   = src/main.py
       module = src.service

Rule: rel.import.import
        action = CREATE_RELATIONSHIP_IF_RESOLVED
        source = artifact:file:src/main.py   (path node)
        target = resolve imported module against known engineering nodes
        classification = observed
        requires_evidence = true

If resolution succeeds:
        IMPORTS  artifact:file:src/main.py  →  code_entity:src/service.py

If resolution fails (e.g. module=os, stdlib):
        UNRESOLVED  reason=target_not_resolved
        (kept in engineering.unresolved[], not in relationships[])
```

`import / import_from` uses the same `IMPORTS` kind with the same
resolution discipline (`from foo.bar import Baz` → resolve `foo.bar` only
against known entities/modules).

### dependency → DEPENDS_ON (observed)

```text
M2:  dependency / dependencies
       name   = jsonschema
       source = pyproject.toml

Rule: rel.dependency.dependencies
        target vocabulary = external_package:{name}   (documented, evolvable)
        on failure        = UNRESOLVED (never a fake target)

M3:  DEPENDS_ON  artifact:file:pyproject.toml  →  external_package:jsonschema
```

### interaction declares/exposes/configures → DECLARES / EXPOSED_BY / CONFIGURED_BY (observed, M12)

```text
M2:  interaction / declares            (source=tooling)
       attributes.source_fact_id = fact:artifact:agents:agent_definition:…
       attributes.target_fact_id = fact:tool:builtin:Bash
       evidence = path=…agent file, line_start=…

Rule: rel.interaction.declares
        action = CREATE_RELATIONSHIP_IF_RESOLVED
        source = node for source_fact_id   (on failure: UNRESOLVED missing_endpoint)
        target = node for target_fact_id   (on failure: UNRESOLVED with the
                                             M2-carried reason —
                                             ambiguous_identity / target_not_resolved)
        classification = observed
        requires_evidence = true

M3:  DECLARES     artifact:agent_definition:…  →  tool:builtin:Bash
     EXPOSED_BY   tool:mcp:glue-athena:get_table →  mcp_server:glue-athena
     CONFIGURED_BY mcp_server:glue-athena       →  artifact:mcp_config:.mcp.json
        (every CONFIGURED_BY edge for a server claimed by more than one
         configuration file is kept with classification = conflicting —
         no edge is ever picked)
```

Same discipline as `TRIGGERS`: endpoints are existing `fact_id`s, evidence
is mandatory, failed resolution is an unresolved claim — never a guess.

### class + containing file → CONTAINS (derived)

```text
M2:  entity / class   path=src/service.py
     file / repository_file  path=src/service.py

Rule: rel.contains.file_class
        action = CREATE_DERIVED_RELATIONSHIP
        derive.condition = class_path_equals_file_path
        classification = derived
        requires_evidence = true

M3:  CONTAINS  artifact:file:src/service.py  →  code_entity:src/service.py::…
```

**Why `derived`?** M2 records the underlying facts (the file exists; the class
exists at that path). M3 *derives* the containment edge from path equality.
The edge is still evidence-backed: evidence cites the contributing M2 fact(s)
from **all material endpoints** — source and target — not only the target
node’s evidence (see `evidence_rules.derived_relationship_requires_contributing_facts`).

Other V1 structural rules:

```text
directory → file          (parent path)
file → module             (module path == file path)
file → class / function   (entity path == file path)
module → class / function (same path module exists)
class → function/method   (only with a deterministic parent link in attributes)
```

Containment is **never** guessed from names alone. If M2 attributes cannot
support a deterministic parent link (common for bare method name lists), the
claim is unresolved — not invented.

---

## Evidence propagation

M2 `EvidenceRef` entries are **copied** onto engineering nodes and
relationships (`propagation = copy_m2_evidence_refs`).

```text
M2 fact.evidence[0]
  path=src/service.py, artifact_type=python, line_start=4,
  symbol=Service, commit=ef11dac…

        ↓ copy

M3 node.code_entity…evidence[0]     (same fields)
M3 relationship CONTAINS…evidence   (from contributing facts)
M3 unresolved candidate.evidence    (retains why it failed to resolve)
```

Invariants (matrix `evidence_rules` + `validation_rules`):

```text
canonical node requires ≥ 1 evidence
canonical relationship requires ≥ 1 evidence
derived relationship requires contributing M2 facts (source ∪ target endpoints)
unresolved candidates retain evidence
```

> A graph edge without evidence is not accepted as canonical engineering
> truth.

Later architecture diagrams must be traceable back to repository evidence at
a known commit.

---

## Unresolved handling

Reasons supported in V1:

```text
target_not_resolved
ambiguous_identity
missing_endpoint
unsupported_relationship
```

Shape (conceptual):

```json
{
  "source": "artifact:file:src/main.py",
  "target": "os",
  "kind": "IMPORTS",
  "reason": "target_not_resolved",
  "evidence": [ { "path": "src/main.py", "artifact_type": "python", "commit": "…" } ]
}
```

Rules:

- Container is `engineering.unresolved[]`, **not** `relationships[]`.
- Unresolved claims are excluded from canonical relationships
  (`unresolved_claims_excluded_from_canonical_relationships = true`).
- Do not invent `classification: unknown` canonical edges — that collapses
  “canonical knowledge” and “possible but unresolved knowledge”.
- Do not manufacture a graph target from a string merely because an imported
  name exists.

---

## Identity

### Nodes

| Kind | Template |
| --- | --- |
| artifact file | `artifact:file:{path}` |
| artifact directory | `artifact:directory:{path}` |
| specialized artifact | `artifact:{subkind}:{path}` |
| code module | `code_entity:{path}` |
| code class/function | `code_entity:{entity_key}` where `entity_key = normalize(path, qualified_name)` |
| external package | `external_package:{name}` |
| MCP tool (M12) | `tool:mcp:{server}:{name}` |
| built-in tool (M12) | `tool:builtin:{name}` |
| registered MCP server (M12) | `mcp_server:{name}` |

### Class/function identity normalization

M2 `qualified_name` already includes repository path context:

```text
path            = src/service.py
qualified_name  = src/service.py::Service

normalize  →  src/service.py::Service
M3 node   →  code_entity:src/service.py::Service
```

```text
path            = src/service.py
qualified_name  = src/service.py::run

M3 node   →  code_entity:src/service.py::run
```

```text
path            = src/service.py
qualified_name  = Service          # bare name, if ever supplied

M3 node   →  code_entity:src/service.py::Service
```

**Why normalize:** M2’s qualified-name representation already carries path
context, so M3 must avoid concatenating the same path twice
(`code_entity:src/service.py::src/service.py::Service` is incorrect).

Rule (`normalize_entity_key`):

1. If `qualified_name` equals or is already prefixed by `path` / `path::` → use it as-is.
2. Otherwise → `{path}::{qualified_name}`.

Deterministic and analyzer-independent (no `python:` / `ast:` / `tree-sitter:` prefixes).

### Relationships

```text
rel:{kind}:{source_id}:{target_id}
```

Stable for the same source + target + kind. No random UUIDs. No
analyzer-specific identifiers.

### Why identity must be deterministic

- Same commit + same rules ⇒ same EngineeringModel ids (reproducible tests).
- Evidence and diagrams can point at stable ids across runs.
- Merges and diffs of graphs become meaningful.

### Why analyzer names must not appear in identity

`python-module:src/service.py` and `ast-module:src/service.py` would claim
two different engineering entities for one module — identity would encode
*how we found it*, not *what it is*. Future analyzers must not change ids.

### Why random UUIDs are inappropriate

UUIDs are opaque and non-reproducible. Canonical engineering knowledge in this
project must be reconstructible from natural keys (path, qualified_name,
declared dependency name).

### Duplicate physical artifact protection

```text
file + configuration + documentation + test facts for one path
        → one artifact:file:<path> node
        → other classifications remain evidence-only
```

Specialized `artifact:agent_definition:<path>` nodes are **intentional**
separate engineering views (role = agent definition), not accidental file
duplicates. Validation flag:
`no_duplicate_physical_artifact_nodes`.

---

## Validation rules (future builder invariants)

Encoded as `true` in the matrix:

```text
unique_node_ids
unique_relationship_ids
relationship_source_must_exist
relationship_target_must_exist
canonical_node_requires_evidence
canonical_relationship_requires_evidence
deterministic_node_identity
deterministic_relationship_identity
deterministic_output_order
unresolved_claims_excluded_from_canonical_relationships
no_analyzer_name_in_entity_identity
no_random_ids_for_canonical_entities
no_duplicate_physical_artifact_nodes
```

---

## What is deliberately deferred (EngineeringModel ≠ ArchitectureModel)

V1 does **not** implement or fully specify:

```text
CALLS, CONSUMES, PRODUCES
runtime / cloud-resource relationships
data lineage, business-domain inference
architecture-level services and workflows
semantic agent interpretation
LLM inference, RAG, embeddings
graph databases (Neo4j, Neptune) as canonical store
MCP-based investigation, ReAct, multi-agent orchestration
ArchitectureModel / validated diagram generation
the full EngineeringModel builder implementation
```

Note: tool→server exposure arrives with M12 as the engineering relationship
`EXPOSED_BY` (fact-level `interaction/exposes`); runtime/API-style `EXPOSES`
semantics remain deferred.

Documented boundary:

```text
EngineeringModel   = deterministic engineering structure from M2 facts
ArchitectureModel  = higher-level architecture derived from that structure
```

This milestone must not imply that the entire architecture has been
reconstructed.

The matrix may be consumed later by Python objects, NetworkX, Neo4j, Neptune,
or other adapters — **without** coupling the rulebook to one database.

---

## M2 fact combinations covered

Verified against the regenerated ADOP artifact (1,954 facts). All 34
`(fact_type, kind)` pairs are either:

- `CREATE_NODE`
- `EVIDENCE_ONLY` (listed in `evidence_rules.evidence_only`), or
- relationship-driven (`import`, `import_from`, `dependency`, `triggers`,
  `declares`, `exposes`, `configures`)

The rulebook itself invented no fact types. M12 later added the open
`tool` / `mcp_server` fact types and three `interaction` kinds to M2's
analyzer set — **still no `repository_facts.v2` schema change**
(see [M12](m12-agent-tool-mcp-knowledge.md)).

---

## What I should understand after M3

Practical learning notes (not marketing):

1. **What a fact is** — a recorded observation with provenance (path/line +
   commit). Example: “this file exists,” “this function is defined here.”
2. **What a graph node is** — a named entity in a structure you can traverse
   (`artifact:file:src/main.py`).
3. **What a relationship is** — a typed, identified link between two nodes
   (`IMPORTS`, `CONTAINS`, `DEPENDS_ON`, `TRIGGERS`, plus the M12 slice
   `DECLARES`, `EXPOSED_BY`, `CONFIGURED_BY`).
4. **Why facts and nodes are different** — facts say what was seen; nodes are
   constructed identities under explicit rules. Collapsing them makes it
   impossible to change interpretation without rewriting observations.
5. **What deterministic derivation means** — same facts + same rules ⇒ same
   nodes, edges, and ids; no RNG, no timestamps in ids, no LLM guessing.
6. **Why evidence belongs on graph objects** — so every edge in a future
   diagram can be audited back to a file at a commit.
7. **Why unresolved relationships must not become canonical edges** —
   “maybe imports X” is not knowledge; parking it in `unresolved[]` keeps the
   canonical graph honest.
8. **Why identity must be deterministic** — stable ids make tests, diffs, and
   reproducible diagrams possible.
9. **Why the construction matrix exists separately from the graph** — the
   rulebook is reviewable and versioned; the graph is a build product. You
   can change the builder implementation without renegotiating semantics
   encoded in the matrix (and vice versa, under versioning).
10. **Why EngineeringModel is not yet ArchitectureModel** — modules, files,
    imports, and containment are engineering structure. “This is the
    orchestration service” is architecture and remains deferred.

---

## How to validate the matrix

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_construction_rules.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

Python API:

```python
from repo_intelligence.construction import load_construction_rules

rules = load_construction_rules()  # schema + invariants; raises on failure
```

Unit tests use small fixtures under `tests/fixtures/construction/`
(`src/main.py`, `src/service.py`, synthetic `repository_facts.json`). The real
ADOP artifact is used only as an optional integration benchmark when present.

---

## Milestone boundary

```text
M1 → Repository Snapshot
M2 → Repository Facts + Evidence
M3 → Engineering construction rules → EngineeringModel builder   ← builder: docs/m3-engineering-model.md
M4 → Validated Architecture Diagram
```

**Out of scope for this milestone:** building the EngineeringModel graph
builder, architecture inference, or diagram generation.
(See [m3-engineering-model.md](m3-engineering-model.md) for the builder that
consumes this rulebook.)
