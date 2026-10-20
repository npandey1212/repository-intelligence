# Flow / Dependency Investigator — a read-only specialist over M1–M17

The second post-M17 specialist capability: a read-only specialist that
investigates paths, dependencies, and flows through a repository and
produces a structured, evidence-backed `FlowDependencyReport` — composed
entirely from machinery that already existed after M17. It adds **no new
agent runtime, planner, orchestrator, memory, evidence, or tracing
system.**

This document is written for a reader who will maintain or extend the
capability (mirroring `docs/architecture-investigator.md`). Read §1–§3
for orientation, §4–§7 for the capability's own contracts, §8–§9 for the
two execution paths, and §10 for the boundary that keeps it from
growing into an orchestrator.

---

## 1. Why this capability exists

The Architecture Investigator answers *"what is the overall
architecture?"*. A different question recurs in every repository
review: *"how does X reach Y?"* — where does a request enter, what
intermediate components carry it, which edges are structural versus
runtime, and what the evidence does **not** establish.

One sentence (spec §1): **a read-only specialist that investigates
paths, dependencies, and flows through a repository and produces a
structured, evidence-backed `FlowDependencyReport`.**

Design posture:

- **Composition over inheritance** (§52): one class wrapping the
  existing `Agent` — no factory, no base class, no second agent loop,
  and explicitly **not** a subclass of `ArchitectureInvestigator`: the
  two specialists share the existing abstractions (tools, envelope,
  claim validation, wrap helper) instead of each other.
- **Domain behavior only** (§12): prompt, tool list, typed output,
  protocol projection, evidence check. The specialist owns no tool
  loop, no retry loop, no planner, no orchestration, no tracing, no
  memory/context engine — the injected `Agent` already runs all of
  those.
- **Read this file top to bottom** to understand the capability: the
  module docstring of `src/repo_intelligence/agents/flow_dependency_
  investigator.py` walks the same three paths documented in §8–§9.

**Downstream (post-M17):** the `FlowDependencyReport` produced here
is the second typed input of the Architecture Visualization Agent —
its ordered path supplies the diagram's flow edges while the
Architecture Investigator's report supplies the components, and both
reports' evidence aggregates define what the diagram may cite. Same
contract, second consumer: `docs/architecture-visualization-agent.md`.

## 2. What it does / does not do

Does:

- investigates any repository root through the four existing read-only
  tools, at a pinned commit label;
- answers the flow question — source, target, ordered path, and what
  the evidence supports — with a closed `path_status` vocabulary
  (§6) that distinguishes *not found* from *unknown* from
  *conflicting*;
- preserves every intermediate hop: `Agent → Tool → MCP Server` is
  reported as two evidenced steps, never collapsed to
  `Agent → MCP Server`, unless the repository itself supports the
  direct edge;
- produces one typed `FlowDependencyReport` — endpoints, ordered
  path, competing candidates, relationships, findings, aggregate
  evidence references, uncertainties, status;
- validates every statement's references against what the run
  actually inspected (M13 claim validation, deterministic);
- answers the M16/M17 investigator protocol (`run_result` +
  `last_trace`) so the same instance works as an orchestrator step;
- projects its report onto the M16 `InvestigationReport` handoff
  envelope so the Analyst loses no path step, alternative, or
  relationship.

Does not:

- does not modify, execute, install, or deploy anything (read-only
  tools only; a path outside the repository root is refused);
- does not plan, assign roles, retry, merge reports across agents, or
  decide pass/fail — that is the M10 planner and the M17
  orchestrator;
- does not invent a second evidence system, a second trace system, a
  second graph vocabulary, or a pathfinding framework;
- does not relabel structural edges as runtime calls: there is no
  `calls` kind in the vocabulary, and proximity never converts a
  `CONTAINS` edge into a call;
- does not contain repository-specific branches, product names, or
  assumptions about what a repository must contain;
- does not silently repair a wrong structured output — it raises
  (`FlowDependencyInvestigatorError`) or passes the failure through
  for the runner to record.

## 3. Architecture at a glance (§63)

```text
                 application / demo / orchestrator
        ┌───────────────────────────────────────────────────┐
        │                                                   │
        │   FlowDependencyInvestigator                      │
        │   ├── prompt: FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1
        │   ├── tools:  get_architecture_context,           │
        │   │           query_engineering_graph,            │
        │   │           read_file, search_repository        │
        │   ├── output: FlowDependencyReport (typed)        │
        │   └── wraps ──► Agent (M8 loop, one trace system) │
        │                    │                              │
        │                    │ restricted_dispatcher(       │
        │                    │   "investigator")            │
        │                    ▼                              │
        │             read-only tool inventory (M3–M6)      │
        └───────────────────────────────────────────────────┘

  DIRECT path        investigate()   → FlowDependencyReport → validate()
  ORCHESTRATED path  run_result()    → M16 envelope (projection)
                                     → Orchestrator (M17) → Analyst (M16)
```

Package layout (§9 — where the pieces live):

| File | Owns |
|---|---|
| `src/repo_intelligence/agents/flow_dependency_investigator.py` | the specialist class, `FLOW_DEPENDENCY_TOOLS`, the explicit error |
| `src/repo_intelligence/agents/prompts/flow_dependency_investigator.py` | `FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1`, `FLOW_PROMPT_VERSION`, the direct builder |
| `src/repo_intelligence/agents/models/flow_dependency_report.py` | `FlowDependencyReport`, `PATH_STATUSES`/`PathStatus`, the cross-field validators, `to_investigation_report()`, `derived_investigation_id()` |
| `src/repo_intelligence/agents/__init__.py`, `agents/{prompts,models}/__init__.py` | facades — the import surface |

## 4. The versioned prompt (§41 prompt version log)

One prompt, two transports, zero orchestration-awareness:

- `FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1` — `ROLE`,
  `RESPONSIBILITIES`, `INVESTIGATION METHOD`, `EVIDENCE RULES`,
  `FLOW / DEPENDENCY RULES`, `UNCERTAINTY RULES`, `OUTPUT`,
  `BOUNDARIES`. The relationship line is built from
  `RELATIONSHIP_KINDS` at import time and the `path_status` line from
  the closed `PATH_STATUSES` list — the prompt can never drift from
  the model's vocabulary.
- `build_flow_dependency_investigator_prompt(question,
  repository_scope, commit_scope, instructions)` — the direct-mode
  prompt: versioned instructions + explicit `TASK` block with scopes
  (the same transport shape the Architecture Investigator uses,
  because one shape for both specialists is one shape to maintain).
- `wrap_investigator_question(instructions, question)` — the
  orchestrated-mode wrapper, **reused from the architecture prompt
  module** (one shared wrap helper, not duplicated): versioned
  instructions around whatever question the runner sends. The wrap is
  unconditional — there is no branch that adapts the prompt when a
  runner is present, so the specialist cannot learn that orchestration
  exists.

Prompt rules the tests enforce: no `orchestrat*`/`planner`
vocabulary anywhere in the prompt; repository-agnostic ("never assume
product names"); static-evidence-only; the "never relabel a
structural edge" rule; and the intermediate-hop rule (`Agent → Tool →
MCP Server`, never collapsed).

Prompt version log (§41):

| Version | Status | Change |
|---|---|---|
| `V1` | current (`FLOW_PROMPT_VERSION = "V1"`) | initial: role, responsibilities, 10-step method, evidence rules (evidence order, observed-without-evidence forbidden, conflict preservation), flow/dependency rules (canonical kinds, no `calls`, no collapsing), uncertainty rules (four `path_status` values), typed `FlowDependencyReport` output block, boundaries (read-only, repository-agnostic, static evidence, no graph-only reasoning) |
| `V2` | current (`FLOW_PROMPT_VERSION = "V2"`) | open-ended-question rule (broad interaction questions report individually supported edges without inventing a single path; `status: "incomplete"` with `uncertainties[]` when not fully answered); amended in place, for the inspected-evidence acceptance fix, with the read-before-cite evidence rule and method step 5 (read the exact ranges you intend to cite — search results only name candidates; a never-read reference is rejected and its claim dropped to an uncertainty) — same version number, this log is the record |

Scope rule: `FLOW_PROMPT_VERSION` (prefixed with `FLOW_` so the facade
can export both specialists' versions without collision) and the prompt
constant are exported from the `repo_intelligence.agents` facade.

## 5. One tool list — the existing read-only inventory (§16/§17)

```python
FLOW_DEPENDENCY_TOOLS = ARCHITECTURE_TOOLS  # the same tuple object
```

- Exactly the four existing investigator-role tools; nothing new was
  added (§16: one obvious list; §17: the inventory is reused, never
  forked). The inventory belongs to the **role**, not to a specialist:
  `FLOW_DEPENDENCY_TOOLS is ARCHITECTURE_TOOLS` is asserted by test.
- Wiring is application-side: `restricted_dispatcher("investigator",
  bindings=...)` + `tool_definitions_for("investigator")`. The
  specialist receives an already-wired `Agent`; it never constructs a
  dispatcher.
- Foreign-root investigation uses the documented closure seam:
  `bindings = {"read_file": lambda a: read_file(a,
  repository_root=root), "search_repository": ...}`.
- `get_architecture_context` and `query_engineering_graph` are
  available (M4/M6 graph + M16 context) but never required — the
  offline stories drive `read_file` because evidence must be file-
  addressable for claim validation.

## 6. The typed output (§8–§9, §21)

`FlowDependencyReport` (pydantic) — fields: `question`,
`repository_scope`, `commit_scope`, `source`, `target`,
`path_status`, `path[]`, `alternative_paths[][]`, `relationships[]`,
`findings[]`, `evidence_refs[]`, `uncertainties[]`, `summary`,
`status ∈ complete|incomplete`.

**There is deliberately no `FlowStep` model**: one path step **is** an
`ArchitectureRelationship` (source → relationship_type → target, plus
classification, confidence, evidence refs) — the repository already
had the reusable relationship contract, and a flow is an ordered list
of them (§8: do not copy the conceptual structure when a reusable
contract exists).

Path status vocabulary (§9) — the one new closed vocabulary this
capability owns, and it is **not** a status machine; each report states
one terminal answer:

| Value | Meaning |
|---|---|
| `FOUND` | a supported path exists (path non-empty, every step established, no competing candidate) |
| `NOT_FOUND` | the investigation scope was sufficient to establish the requested path does not exist |
| `UNKNOWN` | the available evidence is insufficient either way — "I could not find it" is not "it does not exist" |
| `CONFLICTING` | repository evidence supports contradictory interpretations; both candidates preserved in `alternative_paths` |

Cross-field rules (all deterministic, all enforced by root validators
so a malformed flow never crosses a typed boundary):

- every reference used by a path step, alternative step, relationship,
  or finding must appear in the aggregate `evidence_refs` — a report
  can never cite something it did not also declare;
- every path step must carry evidence — a path is never returned
  unevidenced (§10);
- a path (or alternative) is a connected chain whose first source and
  last target match the declared `source`/`target` (§21 #6);
- `FOUND` requires a non-empty path whose steps are established
  (never unknown/conflicting), and no competing candidate;
- `NOT_FOUND`/`UNKNOWN` never carry a path or candidates — an
  unestablished path is never presented as found (§21 #7);
- competing candidate paths force `CONFLICTING` — contradictions are
  preserved, never silently chosen (§21 #8);
- an `observed` relationship without evidence is rejected —
  unsupported edges are never presented as observed (§21 #5).

Closed vocabularies (each has exactly one home):

| Vocabulary | Values | Source of truth |
|---|---|---|
| relationship types | `CONTAINS, IMPORTS, DEPENDS_ON, TRIGGERS, DECLARES, EXPOSED_BY, CONFIGURED_BY` | `engineering.models.RELATIONSHIP_KINDS` — reused via `Literal[RELATIONSHIP_KINDS]`, never re-typed |
| path status | `FOUND, NOT_FOUND, UNKNOWN, CONFLICTING` | this module (`PATH_STATUSES`) — the capability's one new list |
| confidence | `high, medium, low, unknown` | `architecture_report.Confidence` — reused |
| classification | `observed, derived, inferred, conflicting` (+ `unknown` at statement level) | `multiagent.contracts.CLASSIFICATIONS` — reused |
| support | `SUPPORTED, UNSUPPORTED, UNKNOWN` | `evidence.models` — reused |

`to_investigation_report()` (§17) projects onto the M16 envelope so
nothing established at a flow step is lost at the typed hop: path steps
first, then alternatives, then relationships, then the report's own
findings — each as an `InvestigationFinding` with its evidence
references, de-duplicated by statement (an edge appearing both in
`path` and `relationships` collapses to one) and by `(path, line_start,
line_end)` for refs; support is `SUPPORTED` when refs exist, else
`UNKNOWN`. `derived_investigation_id()` = `"inv-flow-" +
sha1(question ⟞ repository ⟞ commit)[:32]` — a deterministic
application correlation id, never model-authored prose.

## 7. Evidence rules — the ONE validator (§24)

`validate(report)` reuses the M13 claim machinery end to end — never a
second evidence system:

```text
report → to_investigation_report().findings (with refs)
       → evidence.models.Claim per statement
       → gather_evidence(specialist.last_trace)   # this run only
       → validate_claims(claims, evidence, require_inspected=True)
       → List[ClaimValidation> (SUPPORTED / UNSUPPORTED / UNKNOWN)
```

- `require_inspected=True`: a reference counts only if this run's
  `read_file` returned the file's content. A search hit (discovery)
  is never eligible claim evidence on its own (M13).
- `UNUPPORTED` carries `code = CLAIM_EVIDENCE_INVALID` and a reason —
  recorded, never repaired.
- Called with no prior run → `FlowDependencyInvestigatorError`
  ("call `investigate()` first"), because there is no trace to
  validate against.
- In orchestrated mode the same authority runs unchanged inside the
  M16 validator's `evidence_valid` rule over the merged traces — the
  specialist's `validate()` and the collaboration's validator are two
  entry points to the same code.

## 8. Direct execution path (§61-A)

```text
caller          FlowDependencyInvestigator      Agent            tools
  │  investigate(question, repo, commit)            │               │
  │───────────────────────────►│                    │               │
  │                            │ build prompt (V1 + │               │
  │                            │ TASK block)        │               │
  │                            │── run_result(prompt)►│             │
  │                            │                    │── ToolCall ──►│
  │                            │                    │◄── ToolResult─│
  │                            │                    │   (read only) │
  │                            │◄── AgentRunResult ─│               │
  │                            │   structured_output:│              │
  │                            │   FlowDependencyReport            │
  │◄── FlowDependencyReport ───│                    │               │
  │                            │                    │               │
  │  validate(report)          │                    │               │
  │───────────────────────────►│ gather_evidence(last_trace)        │
  │◄── List<ClaimValidation> ──│                                    │
```

1. `investigate()` builds the versioned prompt with explicit scopes.
2. The existing Agent loop runs (bounded tool calls, one trace
   system, memory/context if wired).
3. Structured output must be a `FlowDependencyReport` — anything else
   raises `FlowDependencyInvestigatorError` (explicit failure, no
   silent repair; a payload that fails pydantic validation never
   reaches this point — the Agent raises
   `StructuredOutputValidationError` first).
4. `validate()` performs the deterministic evidence check (§7).

Direct mode is what the demonstration's phases 2–4 and the direct
tests exercise.

## 9. Orchestrated execution path (§61-B)

The specialist is injected as the Investigator of the M17
`Orchestrator` — the runner never learns it is not the plain M16
investigator:

```text
Planner (M10)      Orchestrator (M17)       FlowDependencyInvestigator   Analyst (M16)
  │ plan (P1..Pn)       │                          │                         │
  │────────────────────►│ for each step:           │                         │
  │                     │── run_result(step question)►│                      │
  │                     │                          │ wrap: V1 + question     │
  │                     │                          │── Agent.run_result ──►… │
  │                     │                          │   (read-only tools)     │
  │                     │                          │ FlowDependencyReport    │
  │                     │                          │ to_investigation_report()│
  │                     │◄── InvestigationReport ──│ (typed projection)      │
  │                     │ merge step reports       │                         │
  │                     │── ANALYSIS_REQUEST ────────────────────────────────►│
  │                     │◄───────────────────────── ArchitectureFinding ─────│
  │                     │ validate_collaboration (M16, unchanged)            │
  │◄── OrchestrationRun ─│                                                   │
```

Protocol facts (§28/§31 — why the surface is exactly this):

- The M17 orchestrator calls `investigator.run_result(question)` and
  reads `investigator.last_trace`; `Orchestrator.__init__` performs
  no isinstance check — the specialist must provide **only** those
  two members, and does.
- Step outputs must be `InvestigationReport` instances (the
  orchestrator enforces this), hence the unconditional projection
  inside `run_result()` — the ONLY knowledge of the envelope, and it
  lives in the specialist's own adapter, not in a runner branch. The
  projection is the point where the typed handoff matters most: a flow
  report carries path steps and alternatives the envelope shape does
  not have, so every step crosses as its own evidence-backed
  statement.
- A wrong or missing structured output passes through untouched: the
  `Agent` raises / the runner records it as the failure it is; the
  specialist never substitutes a fabricated report.
- The derived id makes every step's report share one correlation id,
  so the merged report and the Analyst's echoed id are checkable
  without model cooperation.
- Role vocabulary: the M17 role classifier maps plan goals to
  `INVESTIGATOR` (identify/investigate/collect evidence) vs `ANALYST`
  (explain/interpret/synthesize); the specialist does not participate
  in role assignment.

## 10. Role boundary (§62)

| Concern | FlowDependencyInvestigator | Never (belongs to) |
|---|---|---|
| prompt, version, wrapping | **owns** | — |
| tool list + read-only discipline | **owns the list**; enforcement | dispatcher/tools (M3–M6, M16 permissions) |
| typed output + vocabularies | **owns** (`FlowDependencyReport`, `PATH_STATUSES`) | — |
| report ↔ envelope projection | **owns** (`to_investigation_report`) | M16 contracts own the envelope |
| evidence validation | entry point only | M13 `validate_claims` (authority) |
| tool loop, bounds, termination | — | Agent (M8) |
| tracing / events / markdown | — | Agent trace (M8.8/M17) |
| memory, context selection | — | M14/M15 (injected) |
| planning, plan gate | — | M10 |
| roles, steps, retries, budgets, merge, run status | — | M17 orchestrator |
| collaboration verdict (8 rules) | — | M16 validator |
| final answer text | reads `run.result` only | Agent |

The test suite greps the specialist source for orchestration/planner
vocabulary in the prompt, for product identifiers, and for
infrastructure the class must not own (own tool loop / own planner /
own trace system) — a boundary regression fails the build, not a code
review.

## 11. Demonstration

`python agentrun_flow_dependency_investigator_demo.py` — offline and
deterministic by default (§57–§59). Six phases, every one with an
explicit condition line:

| Phase | Story | Condition |
|---|---|---|
| 1 | question & scope: prompt version + length, tool list, target, MCP presence, task block | `V1`, 4 tools, prompt has no orchestration vocabulary |
| 2 | direct execution → typed `FlowDependencyReport` (path steps, findings, uncertainties printed) | refs + `status=complete` + `path_status ∈ PATH_STATUSES` + path/status consistency |
| 3 | deterministic evidence check (M13, no LLM) | ≥1 claim, 0 unsupported/unknown |
| 4 | the §24 human-readable `FLOW INVESTIGATION` view: `1. Investigator ✓ read …` / `2. Path` (ASCII chain with relationship kind + classification + evidence file) / `3. Evidence` / `4. Analyst — n/a in direct mode` / `5. Validator` / `Result: <path_status>` | inspected reads, 0 unsupported, `complete`, path/status consistency |
| 5 | orchestrated: plan gate → roles → 3 investigator steps → typed handoff → Analyst → 8 checks; both legs present as separate statements, no collapsed edge | `COMPLETED`/`SUCCESS`, roles `INVESTIGATOR×3 → ANALYST`, all steps completed, validation passed, finding echoes the derived id |
| 6 | trace summary: `trace.json`/`trace.md` for all three participants, `## Orchestration` mirrored, monotonic event sequences | disk artifacts + mirroring + ordering |

CLI (§57): `--question`, `--repository <root>` (default: the
capability's own `tests/fixtures/arch_mcp` fixture — the demo is
self-contained by design), `--commit`, `--live`, `--model`, `--db`,
`--snapshots`, `--traces`.

Exit codes (§58): `0` every condition holds · `1` a condition failed
(names listed on stderr) · `6` execution failed — including
`--live` without `OPENAI_API_KEY`, which prints copy-paste guidance
and never fabricates a success.

Modes: offline (scripted models; real tools, real validator) ·
`--live` (real model for planner + specialist + analyst).

Offline honesty: a root with no MCP configuration completes with
`path_status = UNKNOWN`, `target = null`, and an uncertainty
explaining that the destination could not be established from static
evidence — the destination is never invented.

## 12. Evaluation (§39 evaluation levels)

| Level | Question | Files | Tests |
|---|---|---|---|
| L2 — the capability | does the specialist itself behave: finds what fixtures contain, preserves intermediate hops, keeps relationship kinds, refuses malformed flows, prompt/version/vocab rules, evidence validation, projection purity, orchestrator wiring, repository-agnosticism, no-owned-infrastructure? | `tests/agents/test_flow_dependency_investigator.py` | 49 |
| L3 — end to end (inside the same file) | does it complete the real application path — planner → orchestrator → specialist steps → typed handoff → Analyst → validator — over fixtures? | same file (`test_flow_end_to_end_*`, `test_merged_report_*`, `test_roles_*`, `test_specialist_traces_*`, `test_trace_records_*`) | 5 of the 49 |
| demonstration | do the demo's phases and exit codes behave (offline, fixture, missing root, no-key live)? | `tests/agents/test_flow_dependency_investigator_demo.py` | 20 |

Shared scripted doubles live in `tests/agents/flow_doubles.py` (which
imports the shared `FakeLLMClient`/`fixture_bindings`/`structured`/
`tool_calls` from `architecture_doubles.py`; no `conftest.py`; test
directories carry no `__init__.py`; basenames unique across `tests/`).

## 13. Fixtures (§37)

Tiny by design — every fixture is small enough to read in one sitting
and each one exists to make exactly one evaluation claim pass or fail:

| Fixture | Contents | Proves |
|---|---|---|
| `tests/fixtures/flow_chain/` | `README.md`, `src/producer.py`, `src/transform.py`, `src/publisher.py` | a two-hop chain is reported with BOTH intermediate edges (`producer IMPORTS transform`, `transform IMPORTS publisher`) and the direct `producer → publisher` edge is never invented (the modules do not mention each other) |
| `tests/fixtures/flow_disconnected/` | `src/alpha.py`, `src/beta.py` | neither module mentions the other nor imports it → `NOT_FOUND` (sufficient evidence of absence) versus `UNKNOWN` (insufficient evidence) stay distinct answers |
| `tests/fixtures/flow_conflict/` | `docs/wiring.md` (claims `api → renderer`), `src/api.py` (imports `bridge`), `src/bridge.py`, `src/renderer.py` | documentation and code disagree → both candidates preserved as `alternative_paths` under `CONFLICTING`, both sides' evidence kept |
| `tests/fixtures/arch_mcp/` (reused) | `.mcp.json`, `agents/collector.py`, `tools/fetch_record.py` | the `Agent → Tool → MCP Server` chain (`collector IMPORTS fetch_record`, `fetch_record EXPOSED_BY records`) keeps both intermediates; also the demo's default root |

A fixture-size test asserts the byte budget; fixtures contain no
network access and no executables.

## 14. Repository agnosticism and read-only safety (§14/§15)

- Repository name and commit are **data** (scope labels on the
  prompt, task, and report) — never identifiers in code; the
  repository-agnostic tests grep the specialist source and prompt for
  product identifiers (spec §26: no ADOP, product, or agent-framework
  names) and for foreign framework imports.
- The default investigation root is an injected path; bindings close
  over `repository_root=`; a path that escapes the root is refused
  with `PATH_OUTSIDE_REPOSITORY` (a failed attempt, never evidence).
- Read-only is enforced twice: the role dispatcher exposes only the
  four read-only tools, and the prompt's `BOUNDARIES` restate the
  discipline for the model (defence in depth, different layers).
- A non-destructiveness test runs the specialist over a fixture and
  asserts the tree is byte-identical afterwards.

## 15. Errors, bounds, and failure semantics

| Failure | Behaviour |
|---|---|
| model returns non-`FlowDependencyReport` from `investigate()` | `FlowDependencyInvestigatorError` — explicit, typed, never a default report |
| payload fails pydantic validation | `StructuredOutputValidationError` from the Agent (pre-output) |
| reference outside aggregate | rejected by the report's root validator (payload never accepted) |
| path step without evidence | rejected (payload never accepted) |
| broken chain / endpoints mismatch | rejected (payload never accepted) |
| `FOUND` with empty path or with alternatives; `NOT_FOUND`/`UNKNOWN` with a path | rejected (payload never accepted) |
| unknown path-status/confidence/classification value | pydantic `Literal` rejection (no enum coercion) |
| `validate()` before any run | `FlowDependencyInvestigatorError` |
| claim without eligible evidence | `UNUPPORTED` + `CLAIM_EVIDENCE_INVALID` — recorded, never repaired |
| tool call limits / timeouts / retries | the injected Agent's bounds (M8.5); the specialist adds none |
| orchestrated step failure | the M17 failure mapping (INCOMPLETE/AGENT_FAILED, downstream BLOCKED) — untouched |

Bounds are inherited, never re-implemented: one tool loop (the
Agent's), one trace system (the Agent's), one validator (M13).

## 16. Observability — one trace system

- Direct mode: the specialist's trace **is** the inner Agent's trace
  (`last_trace` property delegates; there is no wrapper trace).
- Orchestrated mode: the M17 orchestrator mirrors step windows and
  orchestration events onto each participant's own trace — so the
  specialist's `trace.md` gains `## Orchestration` exactly like any
  other participant, and the demo's phase 6 verifies it on disk for
  all three participants.
- Evidence provenance is read from that one trace (`gather_evidence`)
  — the report's references resolve against tool results recorded
  there, closing the loop between claim and execution.

## 17. Concept table

| Term | Meaning here |
|---|---|
| specialist | `FlowDependencyInvestigator`: prompt + tool list + typed output + adapters over an injected `Agent` |
| direct path | `investigate()` → `FlowDependencyReport` → `validate()` |
| orchestrated path | `run_result(question)` → projected `InvestigationReport` inside the M17 run |
| projection | `to_investigation_report()` — path steps/alternatives/relationships become envelope findings; refs unioned; statements de-duplicated |
| derived id | `inv-flow-<sha1(question⟞repo⟞commit)[:32]>` — deterministic correlation id |
| path | an ordered chain of `ArchitectureRelationship` steps from source to target |
| alternative path | a competing candidate chain, only legal under `CONFLICTING` |
| path status | `FOUND / NOT_FOUND / UNKNOWN / CONFLICTING` — one terminal answer per report |
| relationship | an edge using the canonical `RELATIONSHIP_KINDS` vocabulary — the original kind is preserved on every step |
| classification | `observed / derived / inferred / conflicting / unknown` — how a statement was established |
| evidence ref | pointer into a repository artifact at the pinned commit; aggregate completeness enforced |
| inspection | a `read_file` result whose content this run actually received (M13 eligibility) |
| one obvious list | `FLOW_DEPENDENCY_TOOLS` = `ARCHITECTURE_TOOLS` = investigator role inventory = `tool_registry()` sorted |

## 18. Testing and regression

New tests (69): L2+L3 `49` + demo `20`.

Regression baselines (all re-verified):

- full suite: **1749 passed, 2 skipped** (was 1680/2 before this
  capability; the 2 skips are the unchanged Graphviz-dot and opt-in
  OpenAI smoke);
- frozen subsets `tests/agent tests/multiagent tests/evidence
  tests/planning` = **430 passed** untouched;
- harnesses: `agentrun_test.py` 6/6, `agentrun_react_test.py` 8/8,
  `agentrun_evidence_test.py` 10/10, `agentrun_evaluation_test.py`
  13/13;
- demonstrations: default fixture → exit 0, `--repository
  tests/fixtures/flow_chain` → 0, `--repository
  tests/fixtures/arch_simple` → 0, `--live` (no key) → 6.

Commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
python agentrun_flow_dependency_investigator_demo.py
python -m pyflakes src/repo_intelligence/agents/ tests/agents/ \
    agentrun_flow_dependency_investigator_demo.py
```

## 19. Risks and implementation notes

1. **Projection de-duplication** — a statement appearing both in
   `path` and `relationships` collapses to one finding; alternatives
   and relationships are added after the path. A live model may emit
   the same statement across orchestrated steps — harmless for
   validation (each claim validates independently) but visible in the
   merged report.
2. **Whole-file inspection semantics** — a path-only ref is supported
   by a whole-file `read_file` (M13 `_covers`); line-granular claims
   need line-precise refs, which the model may not emit. Accepted:
   traceability, not pinpoint citation.
3. **Envelope projection is lossy by necessity** — the Analyst sees
   findings, not the path array; classification and refs survive, the
   native chain shape does not. Consumers wanting the chain must read
   the investigator's own report (both exist in the run record).
4. **Unconditional wrapping (§29)** means orchestrated questions
   carry the full prompt every step — prompt-token overhead per step,
   accepted in exchange for zero orchestration-awareness.
5. **Duck-typed protocol** — the orchestrator does not type-check its
   investigator; a wrong collaborator would fail at `run_result`,
   not construction. Mitigated by the test asserting exactly the
   two-member surface.
6. **`PATH_STATUSES` is the capability's only new vocabulary** —
   justified because no existing project enum answers "did the
   investigation establish a path?". It is a closed four-value list,
   not a state machine.
7. **`Literal[tuple]` flattening** relies on `typing_extensions` +
   pydantic 1.10 behaviour (tuple expands into individual literals);
   pydantic's `ModelField.type_` for `List[X]` returns `X` directly —
   a pydantic upgrade must re-verify both lines.
8. **No `FlowStep` model** — a future reader may want a dedicated
   step type; the design deliberately reuses `ArchitectureRelationship`
   so the graph vocabulary has exactly one home.

## 20. Limitations, non-goals, and definition of done

Limitations / deferred:

- no runtime evidence: static inspection only (the prompt states the
  boundary; `uncertainties[]` carries what could not be established);
- no pathfinding search: the specialist constructs and checks a
  candidate path under the prompt's method — it does not run a graph
  algorithm over the M4/M6 graph;
- no diagram generation, no graph persistence, no cross-repository
  comparison — M4/M5/M11 remain separate capabilities;
- no parallel execution, no replanning — orchestration semantics stay
  exactly as M17 defined them;
- `get_architecture_context`/`query_engineering_graph` are wired but
  unused by the offline stories (they carry no file-addressable
  evidence for claim validation).

Non-goals: becoming an orchestrator, a planner, a validator, a memory
or context engine, a second agent framework, or a repository-specific
reasoner.

Definition of done (all verified in this change):

- [x] specialist composed over the existing `Agent` + tools — no new
      loop, factory, base class, or subclassing of the other
      specialist;
- [x] versioned prompt (`V1`) with the §29 unconditional wrap
      (reused helper) and no orchestration vocabulary;
- [x] typed output with closed vocabularies reusing the canonical
      sources (relationship kinds, classifications) plus the one new
      `PATH_STATUSES` list;
- [x] cross-field validators enforcing §21 #5–#8 (evidence, chain,
      endpoint, status rules);
- [x] M13 evidence validation as the only validator;
- [x] direct path (§8) and orchestrated path (§9) both demonstrated
      and tested;
- [x] three tiny fixtures (chain, disconnected, conflict) plus the
      reused MCP fixture;
- [x] evaluation at levels L2/L3 + demo tests (69 new);
- [x] full-suite, frozen-subset, and harness regressions green;
- [x] this document, README section, and plan-doc entry written;
- [x] repository-agnostic, read-only, hygiene-clean (no f-strings, no
      `.env`, caches removed, pyflakes clean).
