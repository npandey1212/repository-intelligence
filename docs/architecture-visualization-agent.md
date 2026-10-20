# Architecture Visualization Agent — a read-only specialist over M1–M17

The third and final post-M17 specialist capability: an agent that
transforms the two typed investigation results (`ArchitectureReport`,
`FlowDependencyReport`) into one typed, deterministic-validated
`DiagramSpec` and persists it through the EXISTING renderer — composed
entirely from machinery that already existed after M17. It adds **no new
agent runtime, planner, orchestrator, memory, evidence, graph, or
tracing system**, and it performs **zero repository reads**: its tool
list is empty by design.

One sentence (spec §1): **a specialist that transforms validated
repository investigation results into a structured architecture diagram
representation.**

This document is written for a reader who will maintain or extend the
capability (mirroring `docs/architecture-investigator.md` and
`docs/flow-dependency-investigator.md`). Read §1–§3 for orientation,
§4–§11 for the capability's own contracts, §12–§14 for the two
execution paths and the trace, and §17 for the boundary that keeps it
from growing into an orchestrator or a second renderer.

---

## 1. Why this capability exists (purpose)

The two investigators already answer *"what is the overall
architecture?"* (`ArchitectureInvestigator`) and *"how does X reach
Y?"* (`FlowDependencyInvestigator`), each producing an evidence-backed
typed report. What every repository review then needs is the
representation: **how should that knowledge be arranged — as one
diagram someone can read, whose every node and edge is backed by the
investigation that established it?**

Responsibility (spec §1): transform validated repository investigation
results into a structured architecture diagram representation.

The design posture is the same as both investigators:

- **Composition over inheritance** (§52): one class wrapping the
  existing `Agent` — no factory, no base class, no second agent loop.
- **Domain behavior only** (§12): prompt, typed input/output contracts,
  handoff projection, validation entry point, artifact persistence.
  The specialist owns no tool loop, no retry loop, no planner, no
  orchestration, no renderer, no memory/context engine — the injected
  `Agent` runs the loop, the orchestrator runs the plan, and
  `graph.exporters` renders.
- **A projection, not a model** (spec §6): `DiagramSpec` selects nodes
  and edges from the supplied reports; it never invents repository
  facts. The deterministic validator *proves* that selection.

## 2. What it does / does not do

Does:

- consumes the investigators' **typed outputs** (structured outputs,
  never raw repository content) through the typed `VisualizationRequest`
  (§4);
- produces one typed `DiagramSpec` (§5) — bounded diagram
  classification, stable node ids, canonical relationship edges,
  optional groups, a machine-checked legend, aggregate evidence refs,
  explicit uncertainties, terminal status;
- validates that spec **deterministically** through the six-rule
  diagram validator (§9) — the trust boundary between the model's
  proposal and anything that becomes a file;
- renders through the **existing** M5 renderer
  (`graph.exporters.render_mermaid` / `export_graphviz` /
  `render_svg`) — identical output for identical specs (§10);
- persists the four required artifacts of ONE validated spec
  (§11) and refuses to write anything when validation did not pass;
- participates in M17 as the analyst-slot specialist
  (`run_result()` → typed `ArchitectureFinding` handoff), so the ONE
  generic orchestrator coordinates investigator → investigator →
  visualizer without any orchestrator change (§13).

Does **not** (non-responsibilities):

- does not investigate, discover, or re-read the repository —
  `VISUALIZATION_TOOLS = ()`, no dispatcher calls, no evidence
  authority of its own (its own trace is deliberately never used for
  claim validation);
- does not validate its own diagram — it *calls* the deterministic
  validator; the verdict is a value, never a model opinion;
- does not render — Mermaid/DOT/SVG are produced by the existing
  renderer from the validated spec, never by the LLM;
- does not plan, schedule, retry, or orchestrate — it is a
  *participant*, never a scheduler;
- does not create a second graph model, a second evidence model, or a
  diagram-planning system; no `DATA_FLOW` type (the canonical
  relationship vocabulary has no data-flow kind — spec §5/§7);
- does not generate images, edit the repository, deploy, or modify
  infrastructure.

## 3. Architecture at a glance (spec §29.1)

The milestone progression this capability completes (spec §30):

```text
M16
Multi-agent collaboration
        ↓
M17
Multi-agent orchestration
        ↓
Architecture Investigator
What exists?
        ↓
Flow Investigator
How is it connected?
        ↓
Architecture Visualization Agent
How should it be represented?
        ↓
Validated Diagram
```

The runtime chain of one run:

```text
Repository
    ↓
Architecture Investigator
    ↓
ArchitectureReport
    ↓
Flow / Dependency Investigator
    ↓
FlowDependencyReport
    ↓
Architecture Visualization Agent
    ↓
DiagramSpec
    ↓
Validator  (deterministic, six rules, agents/diagram_validation.py)
    ↓
Renderer   (deterministic, graph/exporters.py)
    ↓
Diagram    (architecture.mmd / architecture.svg + diagram_spec.json
            + run_summary.md)
```

The most important design rule this capability must keep:

> Investigators produce evidence-backed knowledge. The Visualization
> Agent organizes that knowledge. The Validator establishes the trust
> boundary. The Renderer presents the validated result deterministically.

## 4. Inputs (spec §4 — the typed request)

`VisualizationRequest`
(`src/repo_intelligence/agents/models/architecture_visualization.py`)
is the one typed input. Implemented fields:

| field | meaning |
|---|---|
| `question` | the user question this diagram answers |
| `diagram_type` | bounded classification: `ARCHITECTURE`, `FLOW`, `COMPONENT`, `AGENT_TOOL_MCP` (§5) |
| `architecture_report` | the Architecture Investigator's typed output (optional) |
| `flow_report` | the Flow / Dependency Investigator's typed output (optional) |
| `additional_flow_reports` | further flow reports for the same question |
| `evidence_scope` | optional narrowing of which references the diagram may cite |

Root validators enforce the typed boundary (spec §20/§22):

- **at least one report** — a request without an investigation is
  rejected before the model is called (the agent never "fills the gap");
- **scopes agree** — every supplied report carries the same
  `repository_scope`/`commit_scope`; one visualization describes one
  pinned repository state;
- **diagram type needs its investigation** — a `FLOW` request requires
  a flow report, an `ARCHITECTURE` request requires an architecture
  report, so a failure at the typed boundary is observable instead of
  being papered over by a diagram from nothing.

The request carries no `rendering_preferences`: rendering is a
deterministic function of the validated spec through the ONE existing
renderer, so a preferences field would imply a second rendering policy.
The spec's conceptual input list is advisory ("information such as"),
and the project convention is to use actual existing models.

`DiagramScope.from_request(request)` derives the investigation scope
(node ids, edge triples, evidence keys, requested diagram type) — the
same object the validator judges against (§9).

## 5. Outputs — the typed `DiagramSpec` (spec §6–§8)

`DiagramSpec` (`schema_version = "diagram_spec.v1"`) contains:

```text
DiagramSpec
├── diagram_type        (DIAGRAM_TYPES — deliberately small)
├── title
├── question / repository_scope / commit_scope   (the envelope)
├── nodes   (DiagramNode: id, label, node_type, classification,
│            confidence, evidence_refs, …)
├── edges   (DiagramEdge: id, source, target, relationship_type,
│            classification, confidence, evidence_refs, …)
├── groups  (optional presentation grouping; node ids must exist)
├── legend  (machine-checked against the SAME vocabularies)
├── evidence_refs       (aggregate — every per-item ref ⊆ this)
├── uncertainties       (free-text, carried to open questions)
└── status              (complete | incomplete)
```

Structural rules are enforced at construction (so a *constructible*
spec is already structurally sound):

- `_ids_unique` — node/edge/group ids unique;
- `_edge_endpoints_exist` — every edge names real nodes;
- `_refs_within_aggregate` — every node/edge/group reference is inside
  the aggregate `evidence_refs`;
- `_observed_needs_evidence` — `observed` never appears without
  evidence references.

Every vocabulary has exactly one home: node types = `COMPONENT_TYPES`
(the Architecture Investigator's nine closed types), relationship types
= canonical `RELATIONSHIP_KINDS` (M11/M12), classifications =
`CLASSIFICATIONS` (M16), confidence = `CONFIDENCE_LEVELS`, status =
`ReportStatus`, evidence refs = `ReportEvidenceRef`, legend kinds =
`LEGEND_KINDS` checked against those same vocabularies.

Two projections keep it inside existing contracts:

- `to_diagram_projection()` → the M5 `DiagramProjection` the ONE
  renderer already serializes (node order follows the spec, so the
  same spec always yields byte-identical Mermaid/DOT);
- `to_finding()` → the M16 `ArchitectureFinding` the analyst slot is
  validated against; diagram statements become
  `Component <id> (<type>) [<classification>]` and
  `<source> <KIND> <target> [<classification>]`, uncertainties become
  `open_questions`, and the correlation id is a deterministic
  `inv-viz-<sha1[:32]>` derived from question + scopes.

`with_context(...)` applies the envelope parsed from the application's
question/request — model prose never overrides the application's scope.

## 6. Evidence — how references survive into the diagram

Evidence flows through the diagram unchanged, in three places:

1. **aggregate** — `DiagramSpec.evidence_refs` is the diagram's
   evidence scope;
2. **per-item** — every node and edge carries its own refs, and every
   per-item ref must be inside the aggregate (construction rule);
3. **validation** — `evidence_validity` (§9) proves every reference is
   a real investigation artifact — and, when the M13 `EvidenceIndex`
   is supplied, that the file content was actually inspected
   (`validate_claims(..., require_inspected=True)` over the statements
   projected by `spec.to_finding()`).

The evidence authority is always the **investigators'** execution:
direct mode gathers it from the specialists' traces
(`gather_evidence` over their iterations); orchestrated mode from the
merged M16 envelope. The visualizer performs no reads, so its own
trace is never consulted — an agent with no tool calls can never
attest that a file was inspected.

## 7. Relationship vocabulary — reused, never re-typed (spec §7)

Edges use the canonical engineering kinds — exactly the M11/M12
`RELATIONSHIP_KINDS`:

```text
CONTAINS, IMPORTS, DEPENDS_ON, TRIGGERS, DECLARES, EXPOSED_BY,
CONFIGURED_BY
```

`relationship_validity` rejects anything else, and the legend is
checked against the same vocabulary, so neither an edge nor a legend
entry can smuggle in new relationship semantics. There is deliberately
no visualization dialect (`CALLS`, `PRODUCES`, `DATA_FLOW` … do not
exist here); a relationship the investigation did not establish is
rejected by `scope_validity` rather than invented.

## 8. Uncertainty — the project's actual enums (spec §9)

The diagram speaks the existing vocabularies; nothing new:

| enum | values | where it appears |
|---|---|---|
| `CLASSIFICATIONS` (M16) | `observed`, `derived`, `inferred`, `conflicting`, `unknown` | node/edge `classification`; scope statements |
| `CONFIDENCE_LEVELS` | `high`, `medium`, `low`, `unknown` | node/edge `confidence` |
| `REPORT_STATUSES` | `complete`, `incomplete` | spec `status` (`status_validity`) |
| evidence claim status | `SUPPORTED`, `UNSUPPORTED`, `UNKNOWN` | M13 claim validation detail |
| `PATH_STATUSES` (flow) | `FOUND`, `NOT_FOUND`, `UNKNOWN`, `CONFLICTING` | the flow report feeding the diagram |

Rules that preserve uncertainty instead of hiding it:

- `observed` requires evidence (construction rule) — an observation
  without a reference can never be built;
- `classification_validity` demands canonical classifications;
- conflicting evidence is a first-class `conflicting` classification —
  the spec never silently picks an interpretation;
- open questions live in `spec.uncertainties` and are carried into the
  finding's `open_questions` (§5) and into `run_summary.md`'s
  `## Uncertainties` section (`- none recorded` when empty);
- a missing scope is reported honestly (`scope_validity` detail:
  *"no investigation scope supplied — structural rules only"*) instead
  of being treated as "everything is in scope".

## 9. Validation — the deterministic trust boundary (spec §14)

`agents/diagram_validation.py` — no model anywhere in the path —
returns the existing M16 `ValidationResult` (`status` + named checks),
so a diagram verdict looks exactly like every other deterministic
verdict in the system:

| rule | question it answers |
|---|---|
| `node_validity` | unique ids; every edge/group endpoint exists |
| `relationship_validity` | every edge type is a canonical kind |
| `evidence_validity` | every reference is real (and inspected, when the M13 index is supplied) |
| `classification_validity` | canonical classification; never `observed` without evidence |
| `status_validity` | a valid terminal report status |
| `scope_validity` | nothing outside the investigation scope (nodes, edge triples, evidence keys, requested diagram type) |

Scope semantics (the anti-invention core):

- **direct mode** — `DiagramScope.from_request(request)`: node ids
  from the reports' components/endpoints, edge triples from both
  reports' relationships/path steps, evidence keys from their
  aggregates;
- **orchestrated mode** — `DiagramScope.from_investigation_report(run
  .report)`: the M16 envelope's deterministic statements are read back
  (`Component <id> (<type>) …`, `<src> <KIND> <tgt> …`) — a parser,
  never a second evidence system;
- nodes are in scope when their id is known **or** all their element
  refs belong to the investigation's evidence keys; edges are strictly
  triple-checked (a hop the reports never established — e.g. collapsing
  the tool out of `agent → tool → server` — fails);
- the diagram type must answer the type the request asked for.

The verdict is a value: `ArchitectureVisualizationAgent.validate()`
returns the `ValidationResult`; a rejected spec never becomes an
artifact (`write_visualization_artifacts` raises
`VisualizationArtifactError` before touching the filesystem).

## 10. Rendering — LLM → DiagramSpec → Deterministic Renderer (spec §15)

Two stages, never mixed:

```text
LLM                     →  DiagramSpec        (proposal, typed)
Deterministic Renderer  →  Mermaid / DOT / SVG (presentation, reproducible)
```

`spec.to_diagram_projection()` adapts to the M5 renderer's expected
input (`graph.exporters`: `render_mermaid`, `export_mermaid`,
`export_graphviz`, `render_svg`, `graphviz_available`) — the renderer
and its serializers are **reused unchanged**, not forked. Identical
spec ⇒ byte-identical Mermaid (spec §19, proven by the demo test that
compares two runs' `architecture.mmd`).

SVG availability: `architecture.svg` is written only when Graphviz
(`dot`) is installed. Where it is not, the run is honest — no `.svg`
file, `VisualizationArtifacts.svg_available = False`, and
`SVG_UNAVAILABLE_NOTICE = "NOT AVAILABLE — existing renderer does not
support SVG"` surfaces in the demo output, `verify_artifacts`, and
`run_summary.md`. SVG is presentation-only, never a source of truth.

Known boundary: `groups` and `legend` are validated in the spec (they
are part of the representation contract and the legend is
vocabulary-checked), while the M5 renderer's projection currently
serializes nodes and edges. The groups/legend content is therefore
fully visible in `diagram_spec.json` and `run_summary.md` but not yet
drawn as swimlanes/styled keys — no renderer change was made to add
that (spec §7/§33: reuse, don't expand).

## 11. Artifacts (spec §17–§18, §25, §32)

```text
artifacts/
    architecture_visualization/
        diagram_spec.json     ← spec.to_json() (schema first, indent 2)
        architecture.mmd      ← renderer output for THIS spec
        architecture.svg      ← only where Graphviz is installed
                                (alongside architecture.dot, the DOT
                                intermediate; the .dot/.svg pair is
                                presentation-only)
        run_summary.md        ← observable facts of THIS run
```

- default directory: `<project-root>/artifacts/architecture_visualization`
  (`visualization_artifacts.DEFAULT_ARTIFACT_DIRECTORY`, resolved from
  the project root like every other generated-output root — never
  CWD-relative), overridable (`write_visualization_artifacts(…,
  directory=…)`, demo `--output`, which is used exactly as given);
- the **fourth artifact, `run_summary.md`**, has eleven fixed sections
  in deterministic order: `# Architecture Visualization Run`,
  `## User Question`, `## Architecture Investigator`,
  `## Flow / Dependency Investigator`,
  `## Architecture Visualization Agent`, `## Validation`,
  `## Rendering`, `## Evidence`, `## Uncertainties`,
  `## M17 Execution`, `## Trace` — execution facts only (structured
  outputs, validation results, locations); sections that were not part
  of the run say so instead of inventing content;
- `verify_artifacts()` reports every required artifact with
  `exists` + bytes (the §25 demo verification), including the honest
  SVG row when unsupported.

Repository hygiene (spec §32): the existing repository keeps
`artifacts/` output (snapshots and every milestone's generated output
are retained there) and ignores only `runtime/` (regenerable
telemetry). This capability follows that existing policy — the
`artifacts/architecture_visualization/` set is **generated
demonstration output**, not committed source, and it is reproducible:
running the demo again regenerates byte-identical
`diagram_spec.json` and `architecture.mmd` (spec §19). Generation:

```bash
python agentrun_architecture_visualization_demo.py
```

`runtime/` (agent traces, demo memory DB) stays ignored as before.

## 12. Direct execution path (spec §20)

One class, no orchestration-aware branch:

```python
from repo_intelligence.agents import (
    ArchitectureVisualizationAgent, VisualizationRequest,
)

request = VisualizationRequest(
    question="How does the collector agent reach the records MCP "
    "server?",
    diagram_type="AGENT_TOOL_MCP",
    architecture_report=architecture_report,   # typed, validated
    flow_report=flow_report,                   # typed, validated
)
spec = visualizer.visualize(request)           # → DiagramSpec
result = visualizer.validate(spec, request=request, evidence=index)  # → six rules
artifacts = write_visualization_artifacts(spec, result, context=…)
```

- `visualize(request)` builds the versioned prompt
  (`ARCHITECTURE_VISUALIZATION_PROMPT_V1` + the explicit TASK block —
  question, diagram type, scopes, and both reports as structured
  JSON), runs the existing agent loop with **no tools**, and applies
  the envelope from the request (`with_context`), never from model
  prose. A non-`DiagramSpec` output raises
  `ArchitectureVisualizationAgentError` — a wrong output is never
  silently adapted. The instructions carry an id-copying few-shot
  rule (node ids and edge endpoints copied byte-for-byte from the
  supplied reports, with right/wrong examples) added to stop the
  live-mode failure where a shortened id (`dag-agent` instead of the
  full `artifact:agent_definition:…` id) was rejected by
  `scope_validity`; the version constant stays `V1` — the pinned
  demo asserts `VISUALIZATION_PROMPT_VERSION == "V1"`, and this was
  a content amendment only (no vocabulary or structure change);
- `validate(spec, request=…, evidence=…)` is pure: scope from the
  request, optional M13 index from the investigators' traces.

Demo (offline by default):

```bash
python agentrun_architecture_visualization_demo.py
# options: --question --repository --commit --output --live --model
#          --db --snapshots --traces     exit 0 / 1 / 6
```

## 13. M17 execution path (spec §21/§24)

The same specialist implementation runs through the ONE generic
`Orchestrator` — **unmodified** (no orchestrator, planner, runner, or
role change):

```text
plan (M10):   S1 architecture goal   → role INVESTIGATOR
              S2 flow goal           → role INVESTIGATOR
              S3 "Interpret …" goal  → role ANALYST   (the visualizer)
        ↓
router (demo/test-owned adapter): one investigator slot → two
specialists, routed by the plan's own step label
("step: <id> — <goal>" parsed from the runner's question; step id
first, goal-content fallback for foreign ids)
        ↓
typed handoffs:  S1/S2 → M16 InvestigationReport (evidence statements)
                 S3    → M16 ArchitectureFinding (diagram statements
                         via spec.to_finding())
        ↓
validator:       the eight M16 rules over the merged collaboration
        ↓
diagram scope:   DiagramScope.from_investigation_report(run.report)
                 + gather_evidence over the investigators' traces
```

Role names are the repository's actual ones — `INVESTIGATOR`,
`INVESTIGATOR`, `ANALYST`, plus the deterministic `validator` handoff
and the deterministic `renderer` stage; there is no `VISUALIZER` or
`RENDERER` *role* because the renderer is a stage, not an agent.

Failure propagation (spec §22) is observable at every step:

- an investigator step fails → downstream steps are blocked, the
  visualizer is **never consulted** (`last_spec is None`, zero model
  requests), no finding, no validation;
- the model returns a wrong/missing output → the runner records the
  failed step (two attempts), no fabricated diagram;
- a constructible spec fails the deterministic rules → the run's
  finding exists but `validate_diagram_spec` rejects it — a rejected
  spec never persists (the persistence gate raises before `mkdir`);
- a request missing its investigation fails at the typed boundary
  (§4), before any model call.

## 14. Trace — what can be observed (spec §23)

One trace system (the existing `AgentTrace` + `JsonFileTraceSink`):

- every participant — both investigators, the visualizer, in both
  paths — writes `runtime/agent_runs/<run_id>/trace.json` +
  `trace.md` (`## Run Summary`, iterations, LLM request/response,
  termination);
- the orchestrator mirrors its step-window events onto each agent's
  own trace (`trace.record_orchestration_event` → re-written through
  the sink), so the orchestrated visualizer's markdown gains
  `## Orchestration` with `orchestration.step.assign`,
  `orchestration.handoff.send` (`VALIDATION_RESULT` from `validator`),
  `orchestration.complete`;
- the authoritative event log stays on `run.events` (monotonic,
  unique sequences) — `render_markdown()` renders either view;
- observable facts only: structured outputs, tool calls (the
  investigators'), evidence refs, validation checks, terminal state.
  No hidden chain-of-thought.

## 15. Demonstration (spec §24)

`agentrun_architecture_visualization_demo.py` — nine phases, offline
and deterministic by default (scripted models, the capability's own
`tests/fixtures/arch_mcp` fixture, real read-only tools, real
validator):

| phase | what it shows |
|---|---|
| 1 scope | question, repository/commit, story, prompt version, the empty visualizer tool list |
| 2 architecture | `ArchitectureInvestigator` → `ArchitectureReport` |
| 3 flow | `FlowDependencyInvestigator` → `FlowDependencyReport` |
| 4 visualization | `ArchitectureVisualizationAgent` → `DiagramSpec` |
| 5 validation | six deterministic rules over the investigators' evidence |
| 6 rendering | Mermaid from the existing renderer + honest SVG status |
| 7 artifacts | the four files, each printed `exists: yes (N bytes)` |
| 8 M17 | plan → routed specialists → visualizer → validator; `run_summary.md` refreshed with the M17 facts |
| 9 trace | actual role chain, disk traces, mirrored orchestration events |

followed by the §25 verification block (`ARTIFACTS (spec §25)` with one
`exists` row per file). Exit codes: `0` every condition holds, `1` a
demonstration condition failed, `6` execution failed (including
`--live` without `OPENAI_API_KEY`). `--repository <root>` investigates
any checkout; a root without MCP completes honestly as a one-node
`COMPONENT` diagram (`path_status = UNKNOWN`, destination never
invented).

## 16. Testing and fixtures

Three focused files (offline, never the network):

| file | what it proves |
|---|---|
| `tests/agents/test_architecture_visualization.py` (53) | models (construction rejections, envelope, vocabularies, projections), prompt (version, embedded reports, no orchestration vocabulary), agent protocol (`visualize`/`run_result`/no tools/wrong output), six-rule validation (out-of-scope node, invented edge, evidence scope, inspected evidence, viz trace not an authority), persistence (4 files, sections, determinism, invalid-never-persisted, SVG honesty, project-root default directory), round trip |
| `tests/agents/test_architecture_visualization_m17.py` (9) | the full application path: three specialists through ONE unmodified orchestrator, roles `[INVESTIGATOR, INVESTIGATOR, ANALYST]`, typed handoffs, all eight M16 rules, merged-evidence diagram scope, trace/role chain, and §22 failure propagation (investigator failure blocks visualization; no fabricated diagram; rejected spec fails validation, not the run) |
| `tests/agents/test_architecture_visualization_demo.py` (27) | demo surface + composition root (no second framework, no test-double imports), scripted models, story (MCP chain + honest no-MCP), read-target selection, router (step label, goal fallback, refusal), all nine phases end to end, §25 verification, byte-identical artifacts across runs, `main` exit codes 0/1/6, and the default `--output` path (regression: the phase-9 summary check reads the project-root artifact directory when no flag is given) |

Shared doubles live in `tests/agents/viz_doubles.py` (payloads that
match the fixture story plus payloads that deliberately do not),
importing the generic helpers from `architecture_doubles.py` /
`flow_doubles.py` rather than copying them.

Fixtures: `tests/fixtures/arch_mcp` (default — agent → tool → MCP
server wiring, three refs) and `tests/fixtures/flow_chain` (foreign
names, no MCP — proves repository agnosticism, spec §28: no
repository-specific identifiers anywhere in the agent).

## 17. Safety (spec §19/Safety)

- **read-only** — `VISUALIZATION_TOOLS = ()`; the analyst-role
  dispatcher exposes no read tools the agent could use; the demo's
  reads happen only inside the investigators;
- **no repository modification** — the capability writes only its own
  artifact directory and runtime telemetry;
- **no deployment** — nothing here deploys or publishes anything;
- **no autonomous infrastructure changes** — no planner/orchestrator
  edits, no new services, no PR generation;
- **no unsupported architecture claims** — a claim outside the
  investigation fails `scope_validity`; an `observed` claim without
  evidence fails construction/validation; a rejected spec never
  persists, so a written artifact set always means "validated";
- **no product/repository identifiers in code** — repository name and
  commit are data (scope labels), never conditions (spec §26/§28).

## 18. Concept table

| concept | module |
|---|---|
| `VisualizationRequest`, `DiagramSpec`, node/edge/group/legend | `agents/models/architecture_visualization.py` |
| `DiagramScope`, six-rule `validate_diagram_spec` | `agents/diagram_validation.py` |
| `ARCHITECTURE_VISUALIZATION_PROMPT_V1`, `PROMPT_VERSION`, `build_…_prompt` | `agents/prompts/architecture_visualization.py` |
| `ArchitectureVisualizationAgent`, `parse_handoff_context` | `agents/architecture_visualization.py` |
| artifacts (`write_visualization_artifacts`, `verify_artifacts`, `build_run_summary`) | `agents/visualization_artifacts.py` |
| renderer reuse (`render_mermaid`, `export_graphviz`, `render_svg`) | `graph/exporters.py` (unchanged) |
| M16 envelope projection (`to_finding`) | reuses `multiagent/contracts.py` |
| M13 evidence (`gather_evidence`, `validate_claims`) | `evidence/` (unchanged) |
| M17 orchestration (`Orchestrator`) | `orchestration/` (unchanged) |

## 19. Risks and implementation notes

- **envelope provenance** — question/repository/commit are parsed from
  the application-built request/question *before* the model is called;
  model prose can never override scope (same discipline as both
  investigators);
- **scope parsing is a parser, not a second evidence system** —
  statements that do not match the two deterministic forms contribute
  no scope; they are never re-validated as claims;
- **`model.copy(update=…)` does not coerce** (pydantic v1) — tests and
  callers construct `DiagramNode`/`DiagramEdge` objects for updates;
- **the orchestrator has ONE investigator slot** — the three-
  specialist chain is expressed with a demo/test-owned router adapter
  keyed by the runner's own step label; this is deliberately outside
  `src/` so the orchestration infrastructure stays frozen;
- **the visualizer's trace is not an evidence authority** — validating
  against it would let a no-read agent attest to file contents; the
  entry point takes the investigators' index instead;
- **persistence gating happens before `mkdir`** — a failed validation
  cannot leave a half-written artifact directory.

## 20. Limitations, non-goals, and definition of done

Limitations (documented, deliberate):

- `groups`/`legend` are represented and validated but the M5 renderer
  projects nodes/edges only (§10) — no renderer change was made;
- `architecture.svg` requires Graphviz; without it the artifact set is
  three files plus the honest notice (§11);
- no `rendering_preferences` field (§4) — rendering is fixed by the
  one renderer;
- the demo's live-mode router falls back to goal keywords for foreign
  planner step ids — a demonstration-owned heuristic, never product
  logic (the offline plan's `S1`/`S2` labels route exactly).

Non-goals: image generation, diagram AI, a visualization framework, a
graph database, a new graph abstraction, planner/orchestrator/runtime
changes (spec §33).

Definition of done:

- `VisualizationRequest` → `DiagramSpec` → six-rule validation →
  renderer → four artifacts, end to end, deterministic;
- the same specialist participates in an unmodified M17 run with the
  typed handoff and all eight M16 rules passing;
- failure propagation observable at every step (no fabricated diagram,
  no artifact of a rejected spec);
- demo exits 0 with the §25 verification printed; full suite green;
- this document describes exactly what exists — no future capability
  documented as implemented (spec §31).
