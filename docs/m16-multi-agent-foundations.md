# M16 — Multi-Agent Foundations

## 1. Why M16 exists

Up to M15 every capability ran inside **one** agent loop: one
`Agent`, one dispatcher, one trace, one final answer. Real
investigations want specialization — one participant that *inspects*
the repository and one that *interprets* — and the naive ways to get
that fail:

* **prompt-passing** ("you are now the analyst…") shares nothing but
  text: no type boundary, no evidence handoff, no way to prove the
  second participant even saw the first's output;
* **a second agent framework** (LangGraph/CrewAI/AutoGen) drags in
  its own loops, states, and vocabularies, duplicating the bounded
  execution, tracing, and evidence work M8–M15 already froze;
* **a message bus/broker** adds infrastructure the spec does not
  need for a fixed two-stage workflow.

M16 adds the smallest collaboration layer that is still *typed*,
*bounded*, *traceable*, and *deterministically validated*: two
EXISTING `Agent` instances, an explicit message contract, one
sequential runner, and a code-only validator that reuses the M13
evidence system. Everything else — orchestration across tasks,
analysis graphs, MCP delegation — is M17's job, built ON these
foundations.

## 2. What M16 does / does not do

Does:

* defines typed inter-agent contracts (`InvestigationReport`,
  `ArchitectureFinding`, `TaskPayload`, `ValidationRequest`,
  `ValidationResult`) as Pydantic models with non-empty scope
  fields — a hop that cannot say which repository state it
  describes is malformed, not defaulted;
* gives each role **bounded tool access on two enforced layers** —
  the definitions the model is offered and the registry that may
  execute (§4);
* executes the fixed sequence Investigator → Analyst → Validator
  through typed messages only, with explicit statuses and bounds
  (§6);
* validates the collaboration deterministically with eight named
  rules that reuse `gather_evidence` + `validate_claims`
  (require_inspected=True) against the **Investigator's** trace
  (§7–§8);
* records sequence-numbered `collaboration_events` on the task
  state AND on each agent's own execution trace — one trace system
  (§9).

Does NOT:

* no new agent framework, no parallel agents, no message broker or
  event bus, no retries (none exist in the project);
* no second trace system, no second evidence model, no second
  claim type — `ReportEvidenceRef` converts to
  `evidence.EvidenceRef`; conclusions become the existing `Claim`
  dataclass only inside the validator;
* no LLM anywhere in validation (a "judge model" would make the
  gate nondeterministic);
* no cross-task orchestration — M16 runs exactly one task through
  exactly two agents once (M17);
* no speculative tools, no dynamic tool loading — registries are
  filtered from the one application-owned `tool_registry()`.

Architecture (spec §28):

```text
                       ┌────────────────────────────────────────┐
                       │              MultiAgentRunner          │
                       │   (sequential, budgets, status gate)   │
                       └────────────────────────────────────────┘
   TaskPayload      message          message          validation
   + scopes  ───▶ INVESTIGATION_REQUEST ───▶ INVESTIGATION_RESULT
                       │                          │
        ┌──────────────▼───────────┐   ┌──────────▼──────────────┐
        │ Investigator             │   │ Analyst                 │
        │ existing Agent           │   │ existing Agent          │
        │ tools: 4 read-only       │   │ tools: NONE             │
        │ defs: 4  registry: 4     │   │ defs: 0  registry: 0    │
        │ structured:              │   │ structured:             │
        │   InvestigationReport    │   │   ArchitectureFinding   │
        │ trace: own AgentTrace    │   │ trace: own AgentTrace   │
        └──────────────┬───────────┘   └──────────┬──────────────┘
                       │  ANALYSIS_REQUEST / ANALYSIS_RESULT     │
                       └──────────────┬─────────────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ Deterministic Validator (no LLM)    │
                    │ 8 rules over recorded state;        │
                    │ evidence = gather_evidence(         │
                    │   INVESTIGATOR trace) + validate_   │
                    │   claims(.., require_inspected)     │
                    └─────────────────────────────────────┘
   shared: MemoryManager · MemoryContext · ContextEngine · trace sink
   message envelope on every hop: repository_scope + commit_scope
```

## 3. Participants and typed contracts

`repo_intelligence/multiagent/contracts.py`:

| Model | Producer | Required fields (highlights) |
|---|---|---|
| `TaskPayload` | task | question, repository_scope, commit_scope (all non-empty) |
| `InvestigationReport` | Investigator | investigation_id, question, scopes, findings, evidence_refs, status ∈ {complete, incomplete}; every finding's refs must be inside the aggregate |
| `ArchitectureFinding` | Analyst | investigation_id (must equal the report's — schema-required), scopes, conclusions, summary; every conclusion's refs inside the aggregate |
| `ValidationResult` | Validator | investigation_id, status ∈ {passed, failed}, checks[], tool_failures[] |

Design rules baked in:

* **Same evidence semantics, no parallel model.**
  `ReportEvidenceRef` is the Pydantic twin of
  `evidence.EvidenceRef` (identical fields and bounds: both line
  bounds or neither, start ≥ 1, start ≤ end) because
  structured-output contracts must be Pydantic models;
  `to_evidence_ref()` converts to the ONE reference type the rest
  of the system speaks.
* **Vocabulary reuse.** `support` reuses the M13
  `ValidationStatus` (`SUPPORTED/UNSUPPORTED/UNKNOWN`);
  `classification` reuses the artifact-model `CLASSIFICATIONS`
  (`observed/derived/inferred/conflicting`) plus `unknown` — a
  conclusion without valid evidence is `unknown`, never fabricated
  certainty.
* **No hidden reasoning.** No thought/scratchpad fields anywhere;
  a conclusion is statement + classification + evidence refs.
* **Scope travels as data.** Labels are demonstration data (§10
  pattern); the validator proves they stayed equal across every
  hop (`scope_match`).

## 4. Tool isolation — two enforced layers

`repo_intelligence/multiagent/permissions.py`:

| Role | Definitions offered | Registry allowed |
|---|---|---|
| investigator | the four (`AGENT_TOOL_DEFINITIONS`) | the four read-only tools |
| analyst | none (`[]`) | none (`()`) |
| validator | none | none (deterministic code, never a model) |

Layer 1 (`tool_definitions_for`) filters
`AGENT_TOOL_DEFINITIONS` by the role's allowlist — an omitted
definition never reaches the model. Layer 2
(`restricted_dispatcher`) builds the role's `ToolDispatcher` from
the same application-owned `tool_registry()` — a forbidden name
(including a hallucinated or hostile one) returns the existing
structured `TOOL_NOT_FOUND` failure; nothing outside the registry
executes. Neither layer alone is the story: the Agent's new
`tool_definitions` parameter only controls what the model may
*ask* for, and the dispatcher controls what may *run* — the demo
probe proves both (a scripted Analyst asking for `read_file`
anyway is refused).

Why no tools for the Analyst: the strongest isolation is also the
strongest provenance story — an `ArchitectureFinding` can only
come from the `InvestigationReport` it was handed, never from an
independent second investigation. (Graph/context access for the
Analyst is deliberately deferred to M17.)

`bindings` (name → callable) is the documented seam for foreign
repositories: `tool(arguments)` closures like
`lambda args: read_file(args, repository_root=fixture)` — the
demo's `--repository tests/fixtures/construction` uses exactly
this, and the package itself contains no repository identifiers
(grep-proven, §12).

## 5. The six typed messages

| # | Type | Sender → Recipient | Payload |
|---|---|---|---|
| 1 | INVESTIGATION_REQUEST | task → investigator | `TaskPayload` |
| 2 | INVESTIGATION_RESULT | investigator → analyst | `InvestigationReport` |
| 3 | ANALYSIS_REQUEST | task → analyst | `InvestigationReport` |
| 4 | ANALYSIS_RESULT | analyst → task | `ArchitectureFinding` |
| 5 | VALIDATION_REQUEST | task → validator | `ValidationRequest` |
| 6 | VALIDATION_RESULT | validator → task | `ValidationResult` |

`AgentMessage.payload` is declared `Any` so pydantic's Union
coercion can never mistype it; a root validator parses (or keeps)
the payload as EXACTLY the model the `message_type` declares —
at runtime the attribute is always the typed instance, never a
dict. Wrong payload type, unknown participant, empty scope, or an
envelope whose `evidence_refs` disagree with the payload's are
rejected as `ContractError` at the boundary (deterministic,
never coerced). The envelope's refs are *taken from* the payload,
so a message can neither invent nor drop provenance at the hop.

## 6. The runner — sequence, budgets, statuses

`repo_intelligence/multiagent/runner.py` — `MultiAgentRunner.run`
returns a terminal `MultiAgentTaskState` record (authoritative
`collaboration_events` with monotonic sequence numbers, the
messages, the typed outputs, the verdict, tool failures).

Status machine (`status.py`) — every move through one gate
(`advance`), terminal states sealed:

```text
PENDING → RUNNING → COMPLETED | INCOMPLETE | FAILED |
                    VALIDATION_FAILED | BUDGET_EXCEEDED | TIMEOUT
```

* `COMPLETED` — validator passed AND both outputs declared
  `complete`;
* `INCOMPLETE` — validator passed but a participant declared
  itself `incomplete` (never presented as a full success);
* `FAILED` — any agent exception (schema-invalid structured
  output included); no analyst run after a failed investigation,
  no fabricated answer, `state.error` carries the exception type
  and message;
* `BUDGET_EXCEEDED` — the message/handoff budget (checked BEFORE
  appending; exhaustion stops the workflow, never emits a partial
  final message) or an agent's `AgentExecutionConfig` tool budget
  (`termination_reason == EXECUTION_LIMIT_REACHED`);
* `TIMEOUT` — a stage exceeded `RunnerConfig.stage_deadline_seconds`
  (injectable clock; the runner never preempts a running Agent —
  it is single-threaded by design and checks after the stage).

Bounds that exist: `max_messages` (default 8; the happy path uses
6), per-agent `max_tool_calls`, the stage deadline, the M15
context budgets inside each agent. Bounds that deliberately do
NOT exist: retries (the project has no retry infrastructure —
one attempt per stage, then an explicit status).

Contract errors from message construction propagate as
`ContractError` — that is a runner bug, not a workflow outcome;
workflow failures are statuses.

## 7. Deterministic validation — the eight rules

`validate_collaboration(task, report, finding, investigator_trace,
messages, analyst_trace)` runs eight named checks, in order, and
never raises for a rule outcome:

1. `message_contract` — the four-stage handoff exists in order
   with the expected sender/recipient routing;
2. `scope_match` — task, report, and finding describe the same
   repository and commit (a hop can never quietly change what is
   under analysis);
3. `evidence_structural` — every ref on every model is a
   well-formed `EvidenceRef` (bounds rule, non-empty path);
4. `collaboration` — `finding.investigation_id ==
   report.investigation_id`, the ANALYSIS_REQUEST carried that
   report, and the INVESTIGATION_RESULT payload IS the validated
   report (consumption proof);
5. `evidence_handoff` — finding refs ⊆ report refs and every
   conclusion ref ⊆ report refs (nothing fabricated at either
   hop);
6. `evidence_valid` — every conclusion validates SUPPORTED
   against the Investigator's gathered evidence with
   `require_inspected=True` (search discoveries, graph knowledge,
   and context never satisfy a reference);
7. `classification` — `observed` needs ≥ 1 supported ref;
   `conflicting` needs ≥ 2 distinct supported paths;
8. `output_structure` — findings, conclusions, and evidence refs
   present (an empty workflow is never a success).

Overall `passed` ⇔ every check passed; the runner maps `failed`
→ `VALIDATION_FAILED`. Tool failures from either trace are
collected onto the result (`agent`, `tool`, `error_code`) —
visible in state AND `ValidationResult`, non-fatal by themselves
(spec §20).

## 8. Evidence reuse — one validator, one evidence model

There is exactly one evidence pipeline in the project:

```text
Agent trace → gather_evidence(trace) → EvidenceIndex
            → validate_claims(claims, index, require_inspected=True)
```

M16 does not fork it. The Validator converts each
`ArchitectureConclusion` into the existing `Claim` dataclass
(statement + `EvidenceRef`s) in memory, validates against
`gather_evidence(investigator_trace)` — the Analyst's own trace
has no tool evidence on purpose — and keeps the M13 semantics
byte-for-byte: `SUPPORTED` only for inspected matches, `UNKNOWN`
for no references (never silently converted to supported),
`UNSUPPORTED` with `CLAIM_EVIDENCE_INVALID` for references that
do not resolve.

Note: `claims_from_payload` (M8.9/M13) only reads payloads that
are mappings; typed M16 outputs are Pydantic models, so the
Agent's internal claim-validation step is skipped for them —
by design the M16 validator is the single gate for these two
contracts (documented risk, §14 #7).

## 9. Trace integration — one trace, collaboration as events

The ONE extension for M16: `AgentTrace.collaboration_events`
(additive field, omitted from serialization when empty — every
existing trace keeps its exact shape) plus the closed vocabulary
`COLLABORATION_OPERATIONS`:

```text
agent.start · agent.end · message.send · message.receive ·
analysis.start · analysis.complete · validator.start ·
validator.complete
```

The runner records every event on the task state (authoritative
sequence numbers; timestamps informational), then mirrors each
stage window onto that agent's OWN trace and re-invokes the
injected sink (`JsonFileTraceSink` overwrites by run id), so the
artifact a stakeholder reads (`trace.md`) gains a chronological
`## Collaboration` section between `## Context` and the
iterations. Unknown operations raise at the recording site —
only the runner records these, so an unknown one is a bug, not
model input. Validator events live only on the task state (the
validator has no Agent trace by design). A failed sink re-write
is observability-only (same policy as the Agent's own sink
errors) — never a run failure.

## 10. Demonstration — one story, six phases

```bash
python agentrun_m16_multi_agent_demo.py                        # offline
python agentrun_m16_multi_agent_demo.py \
    --repository tests/fixtures/construction                   # foreign root
python agentrun_m16_multi_agent_demo.py --live                 # real model
```

Phases: (1) permissions — both layers printed, the Analyst probe
refused; (2) the workflow run; (3) the typed report (id, run id,
findings, evidence refs); (4) the six-message handoff with scope
preserved on every hop and payload types listed; (5) the typed
finding (id echoed, conclusions + classifications); (6) the eight
checks; (7) status, the 16-event timeline, and the trace
artifacts (`collaboration_events` + markdown section per agent).
A summary block prints each condition yes/NO.

Offline is the default and fully deterministic: scripted model
responses over the REAL dispatcher (pinned M1 snapshot by
default; `--repository` binds `read_file`/`search_repository` to
a foreign root), the REAL MemoryManager / ContextEngine /
MultiAgentRunner. `--live` uses a real model for both agents and
exits 1 if deterministic validation rejects the answer — never a
fabricated success. Exit codes: 0 all conditions held · 1 a
condition failed · 6 configuration/execution failure (no
`OPENAI_API_KEY` under `--live`, ContractError, …).

Flags: `--live`, `--question`, `--model`, `--repository`,
`--db` (the demo's OWN `runtime/agent_memory/m16_demo.db`),
`--snapshots`, `--traces`.

## 11. Concept table

| Concept | One-line meaning | Home |
|---|---|---|
| `InvestigationReport` | evidence package produced by the Investigator | contracts.py |
| `ArchitectureFinding` | conclusions constrained to one report's evidence | contracts.py |
| `ReportEvidenceRef` | Pydantic twin of `EvidenceRef` (converts, never forks) | contracts.py |
| `AgentMessage` | one typed hop with scopes + envelope refs | contracts.py |
| `ContractError` | deterministic reject at the message boundary | contracts.py |
| allowlist | which tools a role may use (both layers) | permissions.py |
| `bindings` | documented seam to rebind tools to another root | permissions.py |
| `MultiAgentRunner` | the only orchestration: sequential, bounded | runner.py |
| `MultiAgentTaskState` | authoritative task record + collaboration timeline | runner.py |
| `RunnerConfig` | message budget + stage deadline | runner.py |
| status machine | 8 statuses, one gate, terminal sealed | status.py |
| `validate_collaboration` | the eight deterministic rules | validation.py |

## 12. Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/multiagent/ -q
# → 104 passed
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
# → 1524 passed, 2 skipped
```

(M16 adds 104 tests: contracts, permissions, validation, runner,
repository-agnosticism, trace/status, demo. The 2 skips are the
pre-existing environment-conditional ones — Graphviz `dot` and the
opt-in OpenAI smoke test — unchanged since M15.)

Spec §23 collaboration tests → files:

| §23 test | Where |
|---|---|
| 1–2 roles receive different tools; probe refused | `test_permissions.py`, `test_runner.py` |
| 3 malformed cross-agent contract rejected | `test_contracts.py` |
| 4 conclusions validated as evidence-backed | `test_validation.py`, `test_runner.py` |
| 5 scope preserved / mismatch rejected | `test_runner.py` |
| 6 each role consumes the previous typed output | `test_runner.py` |
| 7–8 missing / conflicting evidence rejected | `test_validation.py`, `test_runner.py` |
| 9–10 investigator / analyst failure leaves no fabricated output | `test_runner.py` |
| 11 tool failure visible, non-fatal | `test_runner.py` |
| 12 bounded (message/tool budgets, deadline) | `test_runner.py` |
| 13 not repository-specific + foreign fixture run | `test_repository_agnostic.py` |

Harnesses re-verified after the additive edits: `agentrun_test.py`
6/6, `agentrun_react_test.py` 8/8, `agentrun_evidence_test.py`
10/10, `agentrun_evaluation_test.py` 13/13.

## 13. Risks and implementation notes

1. **Payload dispatch trusts its construction paths.**
   `AgentMessage.payload` is `Any` + a root validator; anyone
   bypassing `make_message`/`message_from_dict` (e.g.
   `AgentMessage.construct`) skips the typing. Mitigation: the
   runner and tests only construct through the helpers, the
   validator re-checks envelope/payload ref equality, and
   `evidence_structural` re-validates refs (a construct-bypass
   test proves the rule still catches it).
2. **A no-tools Analyst is only as good as its report.** Garbage
   in → garbage out is possible — but rule 6 validates every
   conclusion against the Investigator's *inspected* evidence, so
   an unconstrained "second opinion" cannot pass; extending the
   Analyst with graph/context reads (M17) must keep the same
   one-directional evidence flow.
3. **File-addressable evidence only.** `gather_evidence` sees
   `read_file`/`search_repository` results; conclusions citing
   architecture-context or graph facts are `UNSUPPORTED` under
   `require_inspected=True`. Known limitation of M13 semantics,
   inherited deliberately — do not weaken the switch per-claim.
4. **The stage deadline cannot preempt.** The runner is
   single-threaded: a hung provider call overruns wall-clock
   until the provider/client timeout fires; the deadline is
   checked after the stage and yields `TIMEOUT`. Threading or
   async preemption is out of scope (no parallelism in M16).
5. **Tight message budget.** Default `max_messages=8` with a
   6-message happy path leaves two spares; any future extra hop
   must raise the default consciously — exhaustion is loud
   (`BUDGET_EXCEEDED`), never a silently dropped message.
6. **Id echo is necessary, not sufficient, as consumption
   proof.** A model can repeat `investigation_id` without
   "reading" the report; the real proof is rules 4–6 (payload
   identity + ref handoff + evidence validity), which bind
   *content*, not just the id.
7. **No agent-internal claim validation for typed M16 outputs.**
   `claims_from_payload` only reads mapping payloads, so
   `Agent.run_result` skips its M13 step for the report/finding
   models (no `claims` array in them). The M16 validator is the
   single gate — documented, tested, and intentional; if a future
   change makes the agents validate claims internally, the two
   gates must stay consistent rather than double-failing.
8. **Tool failures are visible, not classified.** A failed read
   surfaces on state + `ValidationResult` but never changes the
   verdict by itself; a critical failed read simply leaves its
   would-be evidence missing, which rule 6 then rejects. There is
   no severity policy — by design (mechanism, not judgment).
9. **`incomplete` is self-declared.** A model marking itself
   `complete` with thin evidence still faces rules 6–8; a model
   marking `complete` dishonestly with *fabricated* refs fails
   rule 6. The status field is an honest-signal aid, never the
   gate.
10. **Sink re-write semantics differ per sink.**
    `JsonFileTraceSink` overwrites by run id (the artifact
    converges to include collaboration events); an appending
    in-memory sink would hold both writes — tests read the JSON
    artifact or `agent.last_trace` directly.
11. **Timestamps are informational.** `sequence` on the task
    state is authoritative; event timestamps are `utcnow` at
    record time and are not used for ordering or decisions.
12. **Fixture binding covers the text tools only.**
    `--repository` rebinds `read_file`/`search_repository`;
    `get_architecture_context` and `query_engineering_graph`
    still resolve the pinned default artifacts (the offline
    story never calls them). Binding those would be a demo-only
    extension, not a package change.

Implementation notes:

* exactly three pre-existing files were touched, additively:
  `agent/agent.py` (`tool_definitions` parameter, default =
  `AGENT_TOOL_DEFINITIONS`), `tool_dispatcher.py`
  (`tool_registry()` export; `default_tool_dispatcher()` now
  consumes it — same four tools), `agent/trace.py`
  (`collaboration_events` + recorder + markdown section);
  every default is byte-identical (full suite 1420 → 1524 with
  the 104 new tests, skips unchanged);
* definitions travel in `AGENT_TOOL_DEFINITIONS` order while
  registries are sorted by the dispatcher — tests assert each
  order explicitly;
* role allowlists are closed: an unknown role raises instead of
  silently meaning "all tools";
* the Analyst's scripted probe in the demo/tests proves the
  refusal path end-to-end (`TOOL_NOT_FOUND` recorded on its
  trace).

## 14. Non-goals and boundary

* **No orchestration framework** — the runner is ~single-file
  deterministic code over two Agent instances; swapping it for
  LangGraph/CrewAI/AutoGen would contradict the frozen M8–M15
  execution model.
* **No parallelism, no broker, no retries** — one task, one
  sequence, one process; M17 may add scheduling ACROSS tasks on
  top, with M16 contracts as the hop language.
* **No second trace / evidence / claim system** — extension
  points used instead: the additive trace field, the
  `tool_registry()` export, the `tool_definitions` parameter, the
  `bindings` seam.
* **Validator stays code** — no LLM judge; a failed collaboration
  is a machine-readable `ValidationResult` plus an explicit task
  status, never an opinion.

## 15. Definition of done

Met: typed contracts with boundary rejection; two-layer role
isolation with a refused probe; six-message workflow with scope
preserved; eight deterministic rules incl. M13 evidence reuse
against the Investigator's trace; explicit failure/bound statuses
with no fabricated outputs; collaboration events on state and
agent traces with artifacts; 13 §23 collaboration tests; a
non-ADOP fixture run + package grep; offline-first demo exit 0 in
both modes; full suite and all four harnesses green; documentation
incl. these risks.

Learning progression: M8 built one bounded agent → M13 made its
conclusions evidence-backed → M14/M15 gave it memory and
context → M16 shows that a *second* participant can join without
any new framework — only types, permissions, and deterministic
validation → M17 can now orchestrate across tasks and analyses
using the same hop language → the post-M17 Architecture
Visualization Agent reuses this envelope directly: its DiagramSpec
projects onto the same ArchitectureFinding handoff.
