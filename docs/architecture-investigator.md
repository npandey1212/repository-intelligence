# Architecture Investigator — a read-only specialist over M1–M17

The capability this repository was named for: a read-only specialist
that investigates a repository to collect evidence about its
architecture and produces a structured, evidence-backed
`ArchitectureReport` — composed entirely from machinery that already
existed after M17.

This document is written for a reader who will maintain or extend the
capability (spec §51). Read §1–§3 for orientation, §4–§7 for the
capability's own contracts, §8–§9 for the two execution paths, and
§10 for the boundary that keeps it from growing into an orchestrator.

---

## 1. Why this capability exists

The ADOP learning plan opens with: *"Your first milestone is not an
autonomous coding agent. It is a read-only architecture investigator
that can prove every important conclusion with repository evidence."*
M1–M17 built the proof machinery (snapshot, facts, graph, tools, LLM
layer, agent loop, evidence validation, memory, context, typed
multi-agent contracts, orchestration); this capability assembles that
machinery into the investigator itself.

One sentence (spec §4): **a read-only specialist that investigates a
repository to collect evidence about its architecture and produces a
structured, evidence-backed architecture report.**

Design posture:

- **Composition over inheritance** (§52): one class wrapping the
  existing `Agent` — no factory, no base class, no second agent loop.
- **Domain behavior only** (§12): prompt, tool list, typed output,
  protocol projection, evidence check. The specialist owns no tool
  loop, no retry loop, no planner, no orchestration, no tracing, no
  memory/context engine — the injected `Agent` already runs all of
  those.
- **Read this file top to bottom** to understand the capability: the
  module docstring of `src/repo_intelligence/agents/architecture_
  investigator.py` walks the same three paths documented in §8–§9.

**Downstream (post-M17):** the `ArchitectureReport` produced here is
one of the two typed inputs of the Architecture Visualization Agent —
its components, relationships, and aggregate evidence refs become the
diagram scope that the deterministic six-rule validator judges
against. Same contract, second consumer:
`docs/architecture-visualization-agent.md`.

## 2. What it does / does not do

Does:

- investigates any repository root through the four existing
  read-only tools, at a pinned commit label;
- produces one typed `ArchitectureReport` — components,
  relationships, findings, aggregate evidence references,
  uncertainties, status — with a closed vocabulary (§6);
- validates every statement's references against what the run
  actually inspected (M13 claim validation, deterministic);
- answers the M16/M17 investigator protocol (`run_result` +
  `last_trace`) so the same instance works as an orchestrator step;
- projects its report onto the M16 `InvestigationReport` handoff
  envelope so the Analyst loses no component or relationship.

Does not:

- does not modify, execute, install, or deploy anything (read-only
  tools only; a path outside the repository root is refused);
- does not plan, assign roles, retry, merge reports across agents, or
  decide pass/fail — that is the M10 planner and the M17
  orchestrator;
- does not invent a second evidence system, a second trace system, or
  a parallel graph vocabulary;
- does not contain repository-specific branches, product names, or
  assumptions about what a repository must contain;
- does not silently repair a wrong structured output — it raises
  (`ArchitectureInvestigatorError`) or passes the failure through for
  the runner to record.

## 3. Architecture at a glance (§63)

```text
                 application / demo / orchestrator
        ┌───────────────────────────────────────────────────┐
        │                                                   │
        │   ArchitectureInvestigator                       │
        │   ├── prompt: ARCHITECTURE_INVESTIGATOR_PROMPT_V1 │
        │   ├── tools:  get_architecture_context,           │
        │   │           query_engineering_graph,            │
        │   │           read_file, search_repository        │
        │   ├── output: ArchitectureReport (typed)          │
        │   └── wraps ──► Agent (M8 loop, one trace system) │
        │                    │                              │
        │                    │ restricted_dispatcher(       │
        │                    │   "investigator")            │
        │                    ▼                              │
        │             read-only tool inventory (M3–M6)      │
        └───────────────────────────────────────────────────┘

  DIRECT path        investigate()   → ArchitectureReport → validate()
  ORCHESTRATED path  run_result()    → M16 envelope (projection)
                                     → Orchestrator (M17) → Analyst (M16)
```

Package layout (§9 — where the pieces live):

| File | Owns |
|---|---|
| `src/repo_intelligence/agents/architecture_investigator.py` | the specialist class, `ARCHITECTURE_TOOLS`, the explicit error |
| `src/repo_intelligence/agents/prompts/architecture_investigator.py` | `ARCHITECTURE_INVESTIGATOR_PROMPT_V1`, `PROMPT_VERSION`, both builders |
| `src/repo_intelligence/agents/models/architecture_report.py` | `ArchitectureComponent`/`ArchitectureRelationship`/`ArchitectureReport`, vocabularies, statement forms, `to_investigation_report()`, `derived_investigation_id()` |
| `src/repo_intelligence/agents/__init__.py`, `agents/{prompts,models}/__init__.py` | facades — the import surface |

## 4. The versioned prompt (§41 prompt version log)

One prompt, two transports, zero orchestration-awareness:

- `ARCHITECTURE_INVESTIGATOR_PROMPT_V1` — `ROLE`, `RESPONSIBILITIES`,
  `RULES`, `INVESTIGATION STRATEGY`, `OUTPUT` (spec §10). The
  relationship line is built from `RELATIONSHIP_KINDS` at import time —
  the prompt can never drift from the model's vocabulary.
- `build_architecture_investigator_prompt(question, repository_scope,
  commit_scope, instructions)` — the direct-mode prompt: versioned
  instructions + explicit `TASK` block with scopes.
- `wrap_investigator_question(instructions, question)` — the
  orchestrated-mode wrapper: versioned instructions around whatever
  question the runner sends (§29: the wrap is **unconditional** —
  there is no branch that adapts the prompt when a runner is present,
  so the specialist cannot learn that orchestration exists).

Prompt version log (§41):

| Version | Status | Change |
|---|---|---|
| `V1` | current (`PROMPT_VERSION = "V1"`) | initial: role, responsibilities, rules (read-only, evidence order, conflict preservation, repository-agnostic), typed `ArchitectureReport` output block; amended in place with `INVESTIGATION STRATEGY` (evidence sufficiency as the primary stopping condition, evidence plan, one broad search over many narrow ones, evidence reuse and no duplicate exploration, search-is-not-reasoning, reassess-after-every-result checklist, one targeted follow-up, budget as an emergency ceiling rather than a target) and, for the final M17 freeze, with the efficiency discipline (objectives derived from THIS question from the same reasoning turn — never a separate step needing its own model call; mark an objective satisfied then leave it; search only for a named unanswered question; uncertainty alone never justifies another call — evidence already answers / reportable as uncertainty / materially changes the answer; the one-time `INVESTIGATION_EFFORT_THRESHOLD_REACHED` notice anticipated as a reassessment, never a cutoff and never a licence for insufficient evidence) and, for the deterministic-finalization fix, with the notice's decision spelled out (when it arrives: finalize from the evidence already collected, or exactly ONE targeted tool call for a specific missing piece, then reassess once more and finalize — no open-ended investigation follows the notice, and a gap the evidence already answers is never a reason to call again) plus an `OUTPUT` bullet naming `question`, `repository_scope`, `commit_scope` as assignment facts echoed from the TASK block, never the model's own inference and, for the inspected-evidence acceptance fix, with the read-before-cite rule (a search result is a discovery, not evidence: a path/line range may appear in `evidence_refs` only after `read_file` returned that range — a never-read reference is rejected and its finding dropped to an uncertainty), the graph-id recovery rule (`NODE_NOT_FOUND` → discover real ids with the summary query, never invent prefixes), and the search-then-read strategy bullet (read the most relevant candidate ranges before issuing another similar search) — same version number, this log is the record |

The strategy section is the *stopping policy*: an investigator that
has enough evidence finalizes, so a healthy run ends voluntarily
(`FINAL_ANSWER`/`STRUCTURED_OUTPUT`, `stop_reason =
EVIDENCE_SUFFICIENT`) long before any tool-call budget, and nothing in
the section names a product, a runner, or an orchestration vocabulary
(§29). It is also where the soft effort target lands: the notice the
Agent appends at `SOFT_TOOL_CALL_TARGET` (16) is anticipated by name,
and the prompt spells the decision the enforced lifecycle expects —
finalize from the evidence already collected, or exactly one targeted
call, then reassess and finalize — so a model that sees it
reassesses sufficiency instead of reacting to a countdown
(`tests/agents/test_investigator_efficiency.py`,
`tests/agents/test_investigator_lifecycle.py`).

Scope rule: `PROMPT_VERSION` and the prompt constant are exported
from the `repo_intelligence.agents` facade; tests assert that the
prompt contains no `orchestrat*`/`planner` vocabulary (§29) and that
it matches the model field names exactly.

Runtime metadata: the three identity fields in `OUTPUT` are
assignment facts, and when a model omits them the Agent fills exactly
those keys from this run's own request (`agent/metadata.py`, the
line-start labels the TASK block, the orchestrated wrapper, and the
finalization pass all carry) — never rewriting what the model wrote,
never touching a substantive field, recorded as
`structured_validation.metadata_injected`. A payload that is wrong
*substantively* still fails validation exactly as before (§5 of
`docs/agent/structured-output.md`).

## 5. One tool list — the existing read-only inventory (§16/§17)

```python
ARCHITECTURE_TOOLS = (
    "get_architecture_context",
    "query_engineering_graph",
    "read_file",
    "search_repository",
)
```

- Exactly the four existing investigator-role tools; nothing new was
  added (§16: one obvious list; §17: the inventory is reused, never
  forked). `ARCHITECTURE_TOOLS` equals `INVESTIGATOR_TOOLS` equals
  `tuple(sorted(tool_registry()))` for that role — asserted by test.
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

## 6. The typed output (§20–§22)

`ArchitectureReport` (pydantic) — fields: `question`,
`repository_scope`, `commit_scope`, `summary`, `components[]`,
`relationships[]`, `findings[]`, `evidence_refs[]`, `uncertainties[]`,
`status ∈ complete|incomplete`.

- `ArchitectureComponent`: `id`, `name`, `type`, `responsibility`,
  `evidence_refs[]`, `classification`.
- `ArchitectureRelationship`: `source`, `target`, `relationship_type`,
  `evidence_refs[]`, `classification`, `confidence`.
- Root validator: every component/relationship/finding reference must
  appear in the aggregate `evidence_refs` — a report can never cite
  something it did not also declare (mirrors the M16 envelope rule).

Closed vocabularies (§21/§22 — each has exactly one home):

| Vocabulary | Values | Source of truth |
|---|---|---|
| component types | `agent, tool, mcp_server, workflow, module, service, configuration, data_store, unknown` | this module (§21 nine-value list) |
| relationship types | `CONTAINS, IMPORTS, DEPENDS_ON, TRIGGERS, DECLARES, EXPOSED_BY, CONFIGURED_BY` | `engineering.models.RELATIONSHIP_KINDS` — reused via `Literal[RELATIONSHIP_KINDS]`, never re-typed |
| confidence | `high, medium, low, unknown` (default `unknown`) | this module |
| classification | `observed, derived, inferred, conflicting` (+ `unknown` at statement level) | `multiagent.contracts.CLASSIFICATIONS` |

Statement forms — one each, used by BOTH the handoff projection and
the direct-mode evidence check so a component never means two
different things:

```text
Component {id} ({type}): {responsibility} [{classification}]
{source} {RELATIONSHIP_KIND} {target} [{classification}]
```

`to_investigation_report()` (§27) projects onto the M16 envelope:
components first, then relationships, then the report's own findings,
each as an `InvestigationFinding` with its evidence references;
aggregate refs unioned with de-duplication by `(path, line_start,
line_end)`; support is `SUPPORTED` when refs exist, else `UNKNOWN`.
`derived_investigation_id()` = `"inv-arch-" + sha1(question ⟞
repository ⟞ commit)[:32]` — a deterministic application correlation
id, never model-authored prose (§27).

## 7. Evidence rules — the ONE validator (§24)

`validate(report)` reuses the M13 claim machinery end to end —
never a second evidence system:

```text
report → to_investigation_report().findings (with refs)
       → evidence.models.Claim per statement
       → gather_evidence(specialist.last_trace)   # this run only
       → validate_claims(claims, evidence, require_inspected=True)
       → List[ClaimValidation] (SUPPORTED / UNSUPPORTED / UNKNOWN)
```

- `require_inspected=True`: a reference counts only if this run's
  `read_file` returned the file's content. A search hit (discovery)
  is never eligible claim evidence on its own (M13).
- `UNUPPORTED` carries `code = CLAIM_EVIDENCE_INVALID` and a reason —
  recorded, never repaired.
- Called with no prior run → `ArchitectureInvestigatorError` ("call
  `investigate()` first"), because there is no trace to validate
  against.
- In orchestrated mode the same authority runs unchanged inside the
  M16 validator's `evidence_valid` rule over the merged traces — the
  specialist's `validate()` and the collaboration's validator are
  two entry points to the same code.

## 8. Direct execution path (§61-A)

```text
caller            ArchitectureInvestigator        Agent            tools
  │  investigate(question, repo, commit)              │               │
  │────────────────────────────►│                     │               │
  │                             │ build prompt (V1 +  │               │
  │                             │ TASK block)         │               │
  │                             │── run_result(prompt)►│              │
  │                             │                     │── ToolCall ──►│
  │                             │                     │◄── ToolResult─│
  │                             │                     │   (read only) │
  │                             │◄── AgentRunResult ──│               │
  │                             │   structured_output:│               │
  │                             │   ArchitectureReport│               │
  │◄──── ArchitectureReport ────│                     │               │
  │                             │                     │               │
  │  validate(report)           │                     │               │
  │────────────────────────────►│ gather_evidence(last_trace)         │
  │◄── List[ClaimValidation] ───│                                     │
```

1. `investigate()` builds the versioned prompt with explicit scopes.
2. The existing Agent loop runs (bounded tool calls, one trace
   system, memory/context if wired).
3. Structured output must be an `ArchitectureReport` — anything else
   raises `ArchitectureInvestigatorError` (explicit failure, no
   silent repair; a payload that fails pydantic validation never
   reaches this point — the Agent raises
   `StructuredOutputValidationError` first).
4. `validate()` performs the deterministic evidence check (§7).

Direct mode is what the demonstration's phases 2–4 and the Level-2
tests exercise.

## 9. Orchestrated execution path (§61-B)

The specialist is injected as the Investigator of the M17
`Orchestrator` — the runner never learns it is not the plain M16
investigator:

```text
Planner (M10)      Orchestrator (M17)         ArchitectureInvestigator     Analyst (M16)
  │ plan (P1..Pn)       │                            │                         │
  │────────────────────►│ for each step:             │                         │
  │                     │── run_result(step question)►│                        │
  │                     │                            │ wrap: V1 + question     │
  │                     │                            │── Agent.run_result ──►… │
  │                     │                            │   (read-only tools)     │
  │                     │                            │ ArchitectureReport      │
  │                     │                            │ to_investigation_report()│
  │                     │◄── InvestigationReport ────│ (typed projection)      │
  │                     │ merge step reports         │                         │
  │                     │── ANALYSIS_REQUEST ─────────────────────────────────►│
  │                     │◄─────────────────────────── ArchitectureFinding ─────│
  │                     │ validate_collaboration (M16, unchanged)              │
  │◄── OrchestrationRun ─│                                                     │
```

Protocol facts (§28/§31 — why the surface is exactly this):

- The M17 orchestrator calls `investigator.run_result(question)` and
  reads `investigator.last_trace`; `Orchestrator.__init__` performs
  no isinstance check — the specialist must provide **only** those
  two members, and does.
- Step outputs must be `InvestigationReport` instances (the
  orchestrator enforces this), hence the unconditional projection
  inside `run_result()` — the ONLY knowledge of the envelope, and it
  lives in the specialist's own adapter, not in a runner branch.
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

| Concern | ArchitectureInvestigator | Never (belongs to) |
|---|---|---|
| prompt, version, wrapping | **owns** | — |
| tool list + read-only discipline | **owns the list**; enforcement | dispatcher/tools (M3–M6, M16 permissions) |
| typed output + vocabularies | **owns** (`ArchitectureReport`) | — |
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
vocabulary in the prompt and for infrastructure the class must not
own (own tool loop / own planner / own trace system) — a boundary
regression fails the build, not a code review.

## 11. Demonstration

`python agentrun_architecture_investigator_demo.py` — offline and
deterministic by default (§57–§59). Six phases, every one with an
explicit condition line:

| Phase | Story | Condition |
|---|---|---|
| 1 | question & scope: prompt version, tool list, target, MCP presence | `V1`, 4 tools, prompt has no orchestration vocabulary |
| 2 | direct execution → typed `ArchitectureReport` (components/relationships/findings printed) | components + refs + `status=complete` |
| 3 | deterministic evidence check (M13, no LLM) | ≥1 claim, 0 unsupported/unknown |
| 4 | the §33 human-readable view: `1. Investigator ✓ read …` / `2. Evidence` / `3. Analyst — n/a in direct mode` / `4. Validator` / `Result:` | inspected reads, printed counts match, all supported |
| 5 | orchestrated: plan gate → roles → 3 investigator steps → typed handoff → Analyst → 8 checks | `COMPLETED`/`SUCCESS`, roles `INVESTIGATOR×3 → ANALYST`, all steps completed, validation passed, finding echoes the derived id |
| 6 | trace summary: `trace.json`/`trace.md` for all three participants, `## Orchestration` mirrored, monotonic event sequences | disk artifacts + mirroring + ordering |

CLI (§57): `--question`, `--repository <root>` (default: the
capability's own `tests/fixtures/arch_simple` fixture — the demo is
self-contained by design), `--commit`, `--live`, `--model`, `--db`,
`--snapshots`, `--traces`.

Exit codes (§58): `0` every condition holds · `1` a condition failed
(names listed on stderr) · `6` execution failed — including
`--live` without `OPENAI_API_KEY`, which prints copy-paste guidance
and never fabricates a success.

Modes: offline (scripted models; real tools, real validator) ·
`--live` (real model for planner + specialist + analyst).

## 12. Evaluation (§39 evaluation levels)

| Level | Question | Files | Tests |
|---|---|---|---|
| L1 — tools | do the underlying read-only tools do their job over tiny fixtures (and refuse out-of-root paths)? | `tests/agents/test_architecture_investigator_tools.py` | 7 |
| L2 — the capability | does the specialist itself behave: finds what fixtures contain, prompt/version/vocab rules, evidence validation, projection purity, orchestrator wiring, repository-agnosticism, no-owned-infrastructure? | `tests/agents/test_architecture_investigator.py` | 29 |
| L3 — end to end | does it complete the real application path — planner → orchestrator → specialist steps → typed handoff → Analyst → validator — over fixtures? | `tests/agents/test_architecture_investigator_m17.py` | 6 |
| demonstration | do the demo's phases and exit codes behave (offline, fixture, missing root, no-key live)? | `tests/agents/test_architecture_investigator_demo.py` | 16 |

Shared scripted doubles live in `tests/agents/architecture_doubles.py`
(no `conftest.py`; test directories carry no `__init__.py`; basenames
unique across `tests/`).

## 13. Fixtures (§37)

Tiny by design — every fixture is small enough to read in one sitting
and each one exists to make exactly one evaluation claim pass or
fail:

| Fixture | Contents | Proves |
|---|---|---|
| `tests/fixtures/arch_simple/` | `README.md`, `agents/researcher.py`, `tools/search_notes.py`, `tools/fetch_note.py` — no MCP | a repository without MCP servers completes gracefully; agent/tool discovery from file layout |
| `tests/fixtures/arch_mcp/` | `.mcp.json`, `agents/collector.py`, `tools/fetch_record.py` | MCP server discovery + `DECLARES`/`EXPOSED_BY` wiring from configuration evidence |
| `tests/fixtures/arch_conflict/` | `README.md` (claims CSV) vs `src/report_generator.py` (uses JSON) | conflicting evidence is preserved with BOTH references, classified `conflicting` — never silently resolved |

A fixture-size test asserts the byte budget; fixtures contain no
network access and no executables.

## 14. Repository agnosticism and read-only safety (§14/§15)

- Repository name and commit are **data** (scope labels on the
  prompt, task, and report) — never identifiers in code; the
  repository-agnostic tests grep the specialist source for
  repository/product identifiers and for foreign framework imports.
- The default investigation root is an injected path; bindings close
  over `repository_root=`; a path that escapes the root is refused
  with `PATH_OUTSIDE_REPOSITORY` (a failed attempt, never evidence).
- Read-only is enforced twice: the role dispatcher exposes only the
  four read-only tools, and the prompt's RULES restate the
  discipline for the model (defence in depth, different layers).
- A non-destructiveness test runs the specialist over a fixture and
  asserts the tree is byte-identical afterwards.

## 15. Errors, bounds, and failure semantics

| Failure | Behaviour |
|---|---|
| model returns non-`ArchitectureReport` from `investigate()` | `ArchitectureInvestigatorError` — explicit, typed, never a default report |
| payload fails pydantic validation | `StructuredOutputValidationError` from the Agent (pre-output) |
| reference outside aggregate | rejected by the report's root validator (payload never accepted) |
| unknown component/relationship/confidence value | pydantic `Literal` rejection (no enum coercion) |
| `validate()` before any run | `ArchitectureInvestigatorError` |
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
| specialist | `ArchitectureInvestigator`: prompt + tool list + typed output + adapters over an injected `Agent` |
| direct path | `investigate()` → `ArchitectureReport` → `validate()` |
| orchestrated path | `run_result(question)` → projected `InvestigationReport` inside the M17 run |
| projection | `to_investigation_report()` — components/relationships become envelope findings; refs unioned |
| derived id | `inv-arch-<sha1(question⟞repo⟞commit)[:32]>` — deterministic correlation id |
| component | an architecture node of one of nine types, evidence-referenced |
| relationship | an edge using the canonical `RELATIONSHIP_KINDS` vocabulary |
| classification | `observed / derived / inferred / conflicting / unknown` — how a statement was established |
| evidence ref | pointer into a repository artifact at the pinned commit; aggregate completeness enforced |
| inspection | a `read_file` result whose content this run actually received (M13 eligibility) |
| one obvious list | `ARCHITECTURE_TOOLS` = investigator role inventory = `tool_registry()` sorted |

## 18. Testing and regression

New tests (58): L1 `7` + L2 `29` + L3 `6` + demo `16`. The
deterministic-finalization fix later added 25 more that exercise this
specialist's metadata and lifecycle through the shared Agent loop:
`tests/agent/test_execution_metadata.py` (12, helper unit),
`tests/agents/test_investigator_metadata.py` (4, direct +
orchestrated fill, no-overwrite, substantive failure), and
`tests/agents/test_investigator_lifecycle.py` (9, decision phase and
tool-free closure).

Regression baselines (all re-verified):

- full suite: **1986 passed, 2 skipped** (was 1622/2 before this
  capability; the 2 skips are the unchanged Graphviz-dot and opt-in
  OpenAI smoke);
- frozen subsets `tests/agent tests/multiagent tests/evidence
  tests/planning` = **486 passed** (the M15/M16 files themselves
  untouched; `tests/agent` also carries the post-freeze additions
  `test_structured_output_normalization.py` and
  `test_execution_metadata.py`);
- harnesses: `agentrun_test.py` 6/6, `agentrun_react_test.py` 8/8,
  `agentrun_evidence_test.py` 10/10, `agentrun_evaluation_test.py`
  13/13;
- demonstrations: default fixture → exit 0, `--repository
  tests/fixtures/arch_mcp` → 0, `--repository
  tests/fixtures/construction` → 0, `--live` (no key) → 6.

Commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
python agentrun_architecture_investigator_demo.py
python -m pyflakes src/repo_intelligence/agents/ tests/agents/ \
    agentrun_architecture_investigator_demo.py
```

## 19. Risks and implementation notes

1. **Projection duplication** — three orchestrated steps can emit the
   same statement; the merge de-duplicates refs but not statements.
   Demo/test payloads partition statements per step to keep the story
   clean; a live model may produce duplicates — harmless for
   validation (each claim validates independently) but visible in the
   merged report.
2. **Whole-file inspection semantics** — a path-only ref is supported
   by a whole-file `read_file` (M13 `_covers`); line-granular claims
   need line-precise refs, which the model may not emit. Accepted:
   traceability, not pinpoint citation.
3. **Envelope projection is lossy by necessity** — the Analyst sees
   findings, not components; classification and refs survive, native
   component shape does not. Consumers wanting components must read
   the investigator's own report (both exist in the run record).
4. **Unconditional wrapping (§29)** means orchestrated questions
   carry the full prompt every step — prompt-token overhead per step,
   accepted in exchange for zero orchestration-awareness.
5. **Duck-typed protocol** — the orchestrator does not type-check its
   investigator; a wrong collaborator would fail at `run_result`,
   not construction. Mitigated by the L3 test asserting exactly the
   two-member surface.
6. **Conflict evidence vs. validation** — a `conflicting`
   classification with both refs present is SUPPORTED (both were
   read); conflict is a reporting state, not a validation failure.
7. **Default demo root is a fixture** (differs from the M16/M17
   pinned-snapshot default): chosen so the demonstration is
   self-contained and honest about what offline mode inspects;
   `--repository` investigates anything else.
8. **`Literal[tuple]` flattening** relies on `typing_extensions`
   + pydantic 1.10 behaviour (tuple expands into individual literals;
   `spec_for_model` then emits a real enum in the JSON schema). A
   pydantic upgrade must re-verify this line.

## 20. Limitations, non-goals, and definition of done

Limitations / deferred:

- no runtime evidence: static inspection only (the prompt states the
  boundary; `uncertainties[]` carries what could not be established);
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
      loop, factory, or base class;
- [x] versioned prompt (`V1`) with the §29 unconditional wrap;
- [x] typed output with closed vocabularies reusing the canonical
      sources;
- [x] M13 evidence validation as the only validator;
- [x] direct path (§8) and orchestrated path (§9) both demonstrated
      and tested;
- [x] three tiny fixtures incl. the conflict case;
- [x] evaluation at levels L1/L2/L3 + demo tests (58 new);
- [x] full-suite, frozen-subset, and harness regressions green;
- [x] this document, README section, and plan-doc entry written;
- [x] repository-agnostic, read-only, hygiene-clean (no f-strings, no
      `.env`, caches removed, pyflakes clean).
