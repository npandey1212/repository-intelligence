# M17 — Multi-Agent Orchestration

> **Status:** implemented, validated, frozen. Every axis a run can
> grow on is bounded — steps, agent runs, messages, tool calls,
> retries, wall clock, and (added in the reliability cleanup) the
> normal loop's working conversation, which is *replaced* above the
> application working-context ceiling instead of growing forever,
> with evidence sufficiency — not the budget — as the Investigator's
> stopping policy. M18 (framework comparison) is the next milestone.

## 1. Why M17 exists

M16 proved that a *second* participant can join a workflow without any
new framework: typed messages, per-role permissions, one sequential
runner, one deterministic validator. What it deliberately did not do
was answer the orchestration question — **how, when, and under what
constraints should planned work be executed?** Up to M16:

* the plan (M10) was produced and validated, then executed by hand —
  nothing mapped plan steps to agents or recorded why a step went
  where it did;
* nothing bounded a *whole orchestrated run* (steps, agent
  invocations, typed handoffs, retries, wall clock) as one explicit
  object with one terminal state;
* no component could say what happens when a step fails halfway: the
  M16 runner stops a single collaboration, but a multi-step plan needs
  "downstream steps are blocked, the evidence gathered so far is
  preserved, the run ends in a named terminal state".

The naive fixes fail for the reasons M16 already documented: an LLM
scheduler loop reintroduces nondeterminism where a decision must be
auditable, a workflow framework (LangGraph/CrewAI/AutoGen) would carry
its own state/status/trace vocabularies over the ones M8–M16 froze,
and a DAG/parallel broker adds machinery the sequential spec does not
need. M17 therefore adds **one application-owned orchestrator**: a
deterministic control plane over the EXISTING plan, the EXISTING two
agents, and the EXISTING validator.

## 2. What M17 does / does not do

Does:

* executes **one validated M10 plan** through a sequential step loop
  the application owns (a `for` over `plan.steps` — no `while True`
  LLM loop, no repository-specific branch anywhere in the package);
* assigns every step a role (**INVESTIGATOR** / **ANALYST**) with a
  deterministic classifier plus explicit application overrides, and
  records a human-readable reason per step (§6);
* bounds the whole run explicitly: plan steps, agent invocations,
  typed messages, total tool calls, retries, and wall-clock runtime —
  each checked **before** the next unit of work (§9);
* allows **bounded retry** — a capability that exists only in M17,
  never on a validation failure, never past `max_retries`, never on an
  agent whose own tool budget is already exhausted (§10);
* merges every Investigator step's report into ONE evidence package
  deterministically and validates it with the **M16 eight-rule
  validator unchanged** — one verdict, reused (§11);
* records sequence-numbered orchestration events on the run and
  mirrors each step's events onto that agent's OWN trace — one trace
  system (§12);
* always returns a terminal-state record with a termination reason
  from a closed vocabulary (§7, §10).

Does NOT:

* no second planner — the plan comes from
  `build_planning_request` / `plan_from_response` and must pass
  `planning.validation.validate_plan` unchanged;
* no second agent framework, no parallel agents, no message broker,
  no DAG scheduler, no replanning, no LLM judge;
* no new validation rules, no second evidence model, no second trace
  system — validation, evidence, and tracing are all reused;
* no repository-specific knowledge in the package (a source scan
  proves no `adop` identifier, no foreign import);
* no restart after a terminal state: terminal states are sealed, and
  further agent execution after them is impossible by construction
  (§7);
* no hidden synthesis: a plan with no analysis step ends
  `INCOMPLETE / NO_ANALYSIS`, never a fabricated architectural answer.

## 3. Two-level architecture (spec §51)

```text
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1 — CONTROL PLANE  (repo_intelligence.orchestration)             │
│ the APPLICATION owns every decision; deterministic, no LLM here        │
│                                                                        │
│   plan gate                 role assignment                            │
│   validate_plan(plan)       assign_roles(steps, overrides)             │
│   + max_plan_steps          → RoleAssignment(step, role, reason)       │
│        │                          │                                    │
│        ▼                          ▼                                    │
│   ┌──────────────── sequential step loop (one plan, one pass) ───────┐  │
│   │  budgets checked first → step.start → step.assign → handoffs     │  │
│   │  → agent run (below) → step.complete | step.failed (+ retry)     │  │
│   │  → downstream steps BLOCKED on failure                           │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│        │                                              │                │
│        ▼                                              ▼                │
│   merge_investigation_reports()        orchestration.* events          │
│   → ONE evidence package                (sequence-stamped, run log)    │
│        │                                              │                │
│        ▼                                              ▼                │
│   validate_collaboration(...)          mirror onto the agent's OWN     │
│   = the M16 eight rules, UNCHANGED     trace (record_orchestration_    │
│   → ValidationResult + terminal state   event) → sink re-write         │
└───────────────┬────────────────────────────────────────────────────────┘
                │  one step at a time: a step-scoped question goes DOWN,
                │  ONE typed contract comes UP (never free text)
┌───────────────▼────────────────────────────────────────────────────────┐
│ LEVEL 2 — AGENT EXECUTION  (unchanged M8–M16: two Agent instances)     │
│                                                                        │
│   Investigator                          Analyst                         │
│   • 4 read-only tools (2 layers)        • NO tools (defs 0, registry 0)│
│   • structured: InvestigationReport     • structured: ArchitectureFinding│
│   • own AgentTrace + run id             • own AgentTrace + run id      │
│   • own bound: AgentExecutionConfig     • own bound: same              │
│                                                                        │
│   one run = LLM → tool → … → structured output (M8.5/M8.9)            │
│   evidence lives in the trace; the control plane never reads a file    │
└───────────────┬────────────────────────────────────────────────────────┘
                │  every attempt's trace feeds the evidence authority
                └────────▶ gather_evidence(merged investigator traces)
                           (M13 — inside the M16 validator)
```

Level 1 never executes a tool and Level 2 never decides a control-flow
question: the orchestrator does not know which tool a step will call,
and the agents do not know they are steps of a plan (each receives a
step-scoped question that carries the objective, the step, and the
scopes).

## 4. Responsibility matrix (spec §49)

| Concern | Owner (code) | Observable at | Never |
|---|---|---|---|
| Produce the plan | M10 `planning/planner.py` (`build_planning_request`, `plan_from_response`) | `orchestration.plan.created` | never inside `orchestration/` |
| Accept/reject the plan | `planning/validation.validate_plan` + `OrchestrationConfig.max_plan_steps` | `orchestration.plan.validated` (SUCCESS/FAILED) | never repaired, re-planned, or partially executed |
| Step lifecycle fields (`PlanStep.status`, `.result`) | the orchestrator as executor | `plan.steps[]` after the run | never model-authored |
| Step → role decision | `orchestration/roles.assign_roles` (classifier + post-conditions) | `orchestration.step.assign` (`agent_role`, `assignment_reason`) | never repository-specific, never LLM |
| Role overrides | the APPLICATION (caller of `execute`) | `role_overrides` argument | never silently coerced (unknown role/step ⇒ `ValueError`) |
| Step sequencing | `orchestration/runner.Orchestrator.execute` (a `for` over the plan) | event order on `run.events` | no scheduler, DAG, priority, or parallelism |
| Per-step question | the orchestrator (`_investigator_question` / `_analyst_question`) | agent trace's `user_question` | never a free-form model-authored prompt rewrite |
| Tool definitions + registry per role | `multiagent/permissions.py` (M16, frozen) | agent request `tools`, refused probe | never extended here |
| Tool execution & the agent's own budget | `Agent` + `AgentExecutionConfig` (M8.5, frozen) | `orchestration.budget.exceeded` (`agent_max_tool_calls`) | never re-bounded by the orchestrator |
| Typed message envelope/payload | `multiagent/contracts.make_message` (M16, frozen) | `orchestration.handoff.send` | never bypassed with raw dicts |
| Message budget | `OrchestrationConfig.max_messages` | `orchestration.budget.exceeded` (`max_messages`) | never a dropped or reordered message |
| Retry decision | the orchestrator (bounded, `max_retries`) | `orchestration.retry` | never on validation failure, never past the bound |
| Report merging | `runner.merge_investigation_reports` | merged `run.report` | never upgraded to `complete` when any step was incomplete |
| Evidence authority | M13 `gather_evidence` over the **merged** investigator traces | `evidence_valid` check | never a second evidence model |
| Validation verdict | M16 `validate_collaboration` (8 rules) | `run.validation`, `orchestration.validation.complete` | never an LLM judge, never re-implemented |
| Terminal status | `orchestration/status.advance` called only by the runner | `run.status`, `run.final` | never assigned directly, never re-opened |
| Termination reason | the orchestrator (closed vocabulary `TERMINATION_REASONS`) | terminal event | never free-form prose |
| Orchestration events | the orchestrator (run log, authoritative) | `run.events` with monotonic `sequence` | never model-authored, never a second bus |
| Event mirroring onto traces | the orchestrator + `AgentTrace.record_orchestration_event` | `## Orchestration` in `trace.md` | never a second trace system |
| Trace persistence | the injected `TraceSink` (M7, frozen) | `runtime/agent_runs/<run id>/` | never a new artifact format |
| Memory / context (M14/M15) | `MemoryManager`, `ContextEngine` on each agent | agent trace memory/context events | never read or written by the control plane |

## 5. The plan gate — M10 reused, never re-implemented

The gate runs before any agent exists in the loop:

1. `validate_plan(plan)` — the M10 deterministic checks (objective,
   ≥1 step, unique non-blank ids, goal-level goals, no unsupported
   execution directives);
2. `len(plan.steps) <= config.max_plan_steps` — a plan that is too
   long is a *rejected plan*, not a longer run;
3. failure ⇒ `orchestration.plan.validated` with `status=FAILED`,
   terminal **`PLAN_FAILED`**, plan status `failed`, and **zero** agent
   invocations (a test asserts the LLM clients were never called);
   success ⇒ `status=SUCCESS`, the plan moves `in_progress`, and the
   step loop starts.

The orchestrator is the *executor*: it alone moves `plan.status` and
each `PlanStep.status/result` (`pending → in_progress →
completed|failed|blocked`), so a model-supplied status can never make
a step look executed.

## 6. Role assignment — deterministic, observable

`assign_roles(steps, role_overrides)` decides in plan order:

1. an explicit override wins (reason: *application-assigned role for
   this step*); an unsupported role, an unknown step id, or an
   override that would put ANALYST before any INVESTIGATOR raises
   `ValueError` — a typo is a visible decision, never a silent fall
   through;
2. otherwise a short, repository-neutral keyword classifier over the
   goal: interpretation/synthesis vocabulary (`explain*`,
   `interpret*`, `synthesi*`, `summar*`, "what does … mean",
   "architectural …", "final conclusion") ⇒ ANALYST, everything else
   ⇒ INVESTIGATOR (reason: *step requires repository evidence
   collection*). Evidence vocabulary such as *identify*,
   *investigate*, *understand relationships* deliberately does NOT
   match — the classifier is a default, overrides are the precise
   instrument;
3. two post-conditions guarantee the workflow shape: if no step
   classified INVESTIGATOR the FIRST becomes it (*evidence collection
   required before interpretation*); if a plan of 2+ steps has no
   ANALYST the LAST becomes it (*final step synthesizes the
   architecture explanation*).

A single-step plan satisfies neither (one step cannot be both roles),
so it stays INVESTIGATOR and the run ends `INCOMPLETE / NO_ANALYSIS` —
documented, tested, never a fabricated answer.

## 7. Run status machine and step lifecycle

`PENDING → PLANNING → PLAN_VALIDATED → EXECUTING → VALIDATING`, or an
explicit exit:

| Status | Meaning (behavioral) | Terminal |
|---|---|---|
| `PENDING` | run constructed, gate not reached | no |
| `PLANNING` | at the deterministic plan gate | no |
| `PLAN_VALIDATED` | may execute; no step started | no |
| `EXECUTING` | steps running sequentially (or retrying) | no |
| `VALIDATING` | every step done; the validator deciding | no |
| `PLAN_FAILED` | invalid/over-long plan; **no agent executed** | yes |
| `AGENT_FAILED` | the Analyst produced no conclusion; no fabricated architectural answer exists | yes |
| `INCOMPLETE` | work stopped early but evidence may exist: failed Investigator step (downstream BLOCKED), declared-incomplete outputs, or no analysis step | yes |
| `TIMEOUT` | wall-clock bound hit **between** steps (never pre-empted mid-run) | yes |
| `BUDGET_EXCEEDED` | an orchestration bound (steps/runs/messages/tool calls) or the agent's own execution limit | yes |
| `VALIDATION_FAILED` | outputs exist, deterministic validation rejected them | yes |
| `COMPLETED` | validation passed and nothing declared the work incomplete | yes |

Every move goes through `status.advance`, which raises
`OrchestrationTransitionError` on an illegal pair — a double finish or
a restart after terminal fails loudly instead of corrupting the record.
The status vocabulary, the transition table, and
`TERMINATION_REASONS` (`SUCCESS`, `PLAN_FAILED`, `AGENT_FAILED`,
`VALIDATION_FAILED`, `TIMEOUT`, `BUDGET_EXCEEDED`,
`DECLARED_INCOMPLETE`, `NO_ANALYSIS`) are all closed sets with tests.

Step lifecycle (per `PlanStep`): `pending → in_progress → completed`
on success; `→ failed` when the step's agent could not produce its
typed contract (after the bounded retry); `→ blocked` for every
not-yet-executed step when an earlier one failed or a bound stopped
the run (reason `<step>_FAILED` / `ORCHESTRATION_STOPPED`).

## 8. The complete execution story (spec §50)

One offline run (the demonstration's), end to end — every line is a
recorded fact, not prose:

1. **Objective in.** `build_planning_request(objective)` → the
   planner answers one structured payload → `plan_from_response`
   yields the typed plan `P1 Identify … / P2 Investigate … / P3
   Collect … / P4 Explain …` (four steps, evidence before
   interpretation).
2. **Plan gate.** `validate_plan` passes and the step budget is
   satisfied → `orchestration.plan.created` then
   `orchestration.plan.validated status=SUCCESS`; the run moves
   `PLANNING → PLAN_VALIDATED → EXECUTING`.
3. **Roles.** `assign_roles` returns
   `P1,P2,P3 = INVESTIGATOR (evidence collection…)`,
   `P4 = ANALYST (architecture interpretation…)`; each is stamped on
   `orchestration.step.assign` with its reason.
4. **Step P1.** The run stamps `orchestration.step.start`, sends
   `INVESTIGATION_REQUEST` (task → investigator, scopes attached,
   `orchestration.handoff.send`), then `orchestration.agent.start`.
   The Investigator — an ordinary bounded M8 agent — calls
   `read_file` (its own tool budget, its own trace) and returns a
   typed `InvestigationReport inv-m17-demo-001`. On success:
   `orchestration.agent.complete run=<id>` + `orchestration.step.complete`,
   the window closes, and those five events are mirrored onto the
   investigator's OWN trace (and re-written through the sink). The
   step's `StepResult` records role, output, evidence refs, run id,
   attempts. `PlanStep.status = completed`.
5. **Steps P2 and P3** repeat with the same shape (`agent_runs` grows
   to 3, `tool_calls_total` to 3, zero retries, zero messages — the
   `INVESTIGATION_REQUEST` is only sent once).
6. **Step P4 (ANALYST).** Before the agent runs, the control plane
   merges the completed reports — findings concatenated in step
   order, evidence refs de-duplicated, open questions/uncertainties/
   sources unioned, `investigation_id` = the FIRST report's, status
   `complete` only if every report was — and sends
   `INVESTIGATION_RESULT` (payload **is** the merged report) plus
   `ANALYSIS_REQUEST`. The tool-less Analyst returns an
   `ArchitectureFinding` echoing `inv-m17-demo-001`; the
   `ANALYSIS_RESULT` completes the six-message sequence.
7. **Budgets before validation.** The pre-validation gate re-checks
   agent runs, tool calls, messages (needs 2), and wall clock; only
   then does the run move `EXECUTING → VALIDATING` and send
   `VALIDATION_REQUEST`.
8. **Validation (code, no LLM).** `validate_collaboration` runs the
   eight rules over the merged report, the finding, the six messages,
   and `gather_evidence` over the **merged** investigator traces (every
   attempt's evidence counts); `VALIDATION_RESULT` lands,
   `orchestration.validation.complete status=passed` is recorded.
9. **Terminal.** Nothing declared itself incomplete →
   `orchestration.complete status=COMPLETED termination_reason=SUCCESS`;
   `plan.status = completed`, all four steps `completed`,
   `run.current_step = None`, `completed_at` stamped. The run is
   sealed: no further agent execution is possible.
10. **Two failure stories, same machinery.**
    (a) a plan whose step IS a tool call fails the gate →
    `PLAN_FAILED`, `agent_runs == 0`, no `StepResult` at all;
    (b) `max_messages=2` on the same plan → P1–P3 execute, the
    analysis step trips the bound → `orchestration.budget.exceeded`,
    `P4` BLOCKED, the merged report **preserved**, `finding is None`,
    `validation is None`, terminal `BUDGET_EXCEEDED` — evidence
    survives a stopped run, a fabricated answer never appears.

## 9. Bounds — every dimension explicit

| Bound | Default | Checked | On exhaustion |
|---|---|---|---|
| `max_plan_steps` | 8 | plan gate, before execution | `PLAN_FAILED` (no agent ran) |
| `max_agent_runs` | 16 | before each step (every attempt counts) | `BUDGET_EXCEEDED`, remaining steps BLOCKED |
| `max_messages` | 8 | before each step / before validation (needs pre-computed) | `BUDGET_EXCEEDED`, report preserved |
| `max_total_tool_calls` | 32 | before each step (summed across attempts) | `BUDGET_EXCEEDED` |
| `max_retries` | 1 (0 disables) | after a retryable step failure | step `failed` → run terminal |
| `max_runtime_seconds` | unset | before each step / before validation (injectable `clock`) | `TIMEOUT` between steps, never pre-empted |
| agent's own `max_tool_calls` | 8 (M8.5) | inside the Agent (unchanged) | step `failed` + `BUDGET_EXCEEDED`, **no retry** |
| agent's working-context ceiling (`max_context_characters`) | 300 000 characters (M8.5) | every normal request, *before* it is sent (deterministic character estimate) | the working history is **replaced** (preamble + one compact state message + recent units) → `context.compaction.started/completed` (counts only); if it cannot shrink, oldest raw units dropped from **that request only** → `context.restricted`; preamble and newest interaction kept, trace and evidence untouched — both markers are continuations, never stops |
| agent's soft effort target (`soft_tool_call_target`) | 16 (`SOFT_TOOL_CALL_TARGET`) | once per run, when the count of executed tool calls reaches it; enforced in the M17 composition (`AgentExecutionConfig.enforce_soft_target`), signal-only under the default config | one execution-state notice (`INVESTIGATION_EFFORT_THRESHOLD_REACHED`, counters only) plus `execution.effort_threshold_reached` and `execution.investigation_reassessed` (point 1) are recorded, and the run enters the decision phase: each decision point emits exactly one `execution.investigation_decision` — **FINALIZE** (close from the evidence already collected) or **ONE_TARGETED_CALL** (exactly one chosen call dispatched under `execution.targeted_call_started`, the rest of that batch blocked with reason `SOFT_TOOL_CALL_TARGET_REACHED`) → `investigation_reassessed` (point 2) → FINALIZE; a tool-free response decides FINALIZE immediately. `* Soft effort target: reached` appears in the run summary. Enforcement never records `execution.boundary_reached` (`stop_reason` stays `EVIDENCE_SUFFICIENT`); without the flag nothing is blocked, and the 64-call ceiling is unchanged |

The working-context ceiling is what keeps a long run from dying the
way the original failure did (129 592 tokens on one request, mid-run).
Under the ceiling the request is byte-for-byte what it was. Over it,
the working conversation is **replaced** — persistently, not for one
request: the instruction side (system contract, question, current plan
step, newest context package) and the recent tool window survive, one
deterministic state message built from this run's own trace records
the objective, the repository scope, the EvidenceRefs it gathered,
failed attempts kept apart, and clipped snippets of the observations
that yield no reference — and `messages = compacted` means growth
becomes grow → threshold → compact → bounded again. No summarization
model, no token counter, no provider call to discover the limit: the
state message is assembled from the trace, twice, byte-identically.
Where compaction cannot shrink the conversation (one oversized
observation, or a preamble that already carries it), the SAME ceiling
still drops the oldest assistant-turn-plus-its-results unit from that
one request. The run's durable state (iterations, tool results,
evidence refs → artifact → pinned commit) lives on the trace, which
neither layer ever touches, so the model can still cite what it
gathered after its context has been compacted or bounded.
`AgentTrace.stop_reason` renders the outcome in the run summary:
`EVIDENCE_SUFFICIENT` (answered voluntarily), `BUDGET_EXCEEDED`,
`CONTEXT_LENGTH_EXCEEDED`, `ERROR` — and the two context markers are
reported alongside it, never as stop reasons: `* Context compactions:
n` and `* Context restrictions: n`.

Two different numbers, two different jobs. The **hard** ceiling
(`max_tool_calls`, 64 in the Repository Query composition) is the
emergency brake: checked before every execution, it blocks the next
dispatch and — with `finalize_on_boundary` — the run then closes from
the evidence it already collected. The **soft** target
(`soft_tool_call_target`, 16 by default) is an investigation
*lifecycle boundary* in the M17 composition (`enforce_soft_target`,
opt-in, default `False`, non-plan runs only — so every other
construction keeps the M8–M16 signal-only behavior): crossing it
changes *when exploration may continue*, not the evidence bar. At
most once per run the Agent appends ONE user-role execution-state
message (counters only — executed, target, ceiling — never message
content, never an answer) and records `execution.effort_threshold_
reached` plus `execution.investigation_reassessed(decision_point=1)`,
and the run stops exploring: a response that arrives with a decision
open is decided once (`execution.investigation_decision` with
`decision` = `FINALIZE` or `ONE_TARGETED_CALL`, plus `reason` and
`decision_point`) — a tool-free response finalizes on the spot; a
tool-bearing response in the first decision phase is cut down to
exactly ONE chosen call (the remainder of that batch is blocked with
reason `SOFT_TOOL_CALL_TARGET_REACHED` on the iteration and never
presented in the assistant turn, so the conversation only ever
contains actions that actually executed; the dispatch is recorded as
`execution.targeted_call_started`), then `investigation_reassessed`
(`decision_point=2`) and FINALIZE either way — a tool-bearing
response after the targeted call was used goes tool-free through the
same finalization machinery. `AgentTrace.stop_reason` never changes
for such a run (`EVIDENCE_SUFFICIENT`), enforcement never records
`execution.boundary_reached`, and `AgentTrace.effort_threshold_
reached` / `* Soft effort target: reached` remain read-only
efficiency views. The Architecture Investigator's prompt anticipates
the notice and spells the same decision (finalize, or exactly one
targeted call, then reassess and finalize), so the model treats it as
a reassessment rather than a countdown. A soft target at or above
the ceiling simply never fires.

Budgets are checked **before** the next unit of work — nothing is
pre-empted mid-run, and an exhausted bound is loud
(`orchestration.budget.exceeded` with `budget`, `limit`, `observed`,
`step_id`), never a silently dropped step. The two budget systems do
not overlap: orchestration bounds the *run*, each agent bounds its own
*tool execution*.

The Repository Query composition root (`repository_query`) sets four
of these values explicitly, with ONE number: `max_tool_calls = 64`,
`finalize_on_boundary = True`, and `enforce_soft_target = True` on
every specialist (`M17_EXECUTION_CONFIG`) and `max_total_tool_calls =
64` (`M17_ORCHESTRATION_CONFIG`). The shipped agent default (8) still
governs every other construction, and the shipped orchestration
default (32) would stop the run at the *next* step the moment one
investigator legitimately spent its own 64-call budget. With
`finalize_on_boundary`, an investigator that reaches its budget — or
meets a provider `context_length_exceeded` — stops taking tools and
closes from the evidence it already collected: one tool-free request
(`tools=[]`, bounded digest, full history never re-sent), the ordinary
terminal machinery, and four trace lifecycle events
(`execution.boundary_reached` or `context.length_exceeded`,
`finalization.started`, `tools.disabled`, `finalization.completed`).
The step then returns a structured report, so downstream steps run
instead of a workflow failure; a whole-run budget that the *run* has
truly spent still stops it, loudly, as in the table above.
`enforce_soft_target` is the same tool-free closure applied mid-run
by the decision phase in §9: it too returns a structured report,
never a workflow failure, and never replaces the budget check — the
64-call ceiling still wins first whenever it is the one that fired.

## 10. Retry and failure semantics

Retry exists **only in M17** (M16 explicitly had none) and is bounded
on every axis:

* retryable: an exception from the agent run, a missing/wrong typed
  output (`MISSING_STRUCTURED_OUTPUT`), a scope mismatch
  (`SCOPE_MISMATCH` — report scope ≠ task scope);
* never retried: a **validation failure** (the verdict is terminal),
  an agent whose own tool budget is exhausted
  (`EXECUTION_LIMIT_REACHED` — retrying only re-inflates a budget that
  is already spent), a `context_length_exceeded` boundary (the same
  oversized request cannot succeed — with `finalize_on_boundary` the
  agent finalizes from the evidence it already has instead of
  raising), a budget/timeout gate, or anything past
  `max_retries`;
* every attempt is visible: `orchestration.retry` events with
  `attempt`/`reason`, `StepResult.attempts`, `run.retries`,
  `run.agent_runs[attempt]`.

Failure outcomes (all terminal, all tested):

| Failure | Run status | Termination reason | Downstream |
|---|---|---|---|
| invalid / over-long plan | `PLAN_FAILED` | `PLAN_FAILED` | no step started, 0 agent runs |
| Investigator step fails (after retry) | `INCOMPLETE` | `AGENT_FAILED` | remaining steps BLOCKED (`<step>_FAILED`) |
| Analyst step fails (after retry) | `AGENT_FAILED` | `AGENT_FAILED` | report preserved, `finding is None` |
| agent's own tool budget exhausted | `BUDGET_EXCEEDED` | `BUDGET_EXCEEDED` | no retry, remaining steps BLOCKED |
| orchestration bound hit | `BUDGET_EXCEEDED` / `TIMEOUT` | same | remaining steps BLOCKED |
| validation rejects the outputs | `VALIDATION_FAILED` | `VALIDATION_FAILED` | outputs + verdict preserved |
| outputs declare themselves incomplete | `INCOMPLETE` | `DECLARED_INCOMPLETE` | validation still ran and passed |
| plan with no analysis step | `INCOMPLETE` | `NO_ANALYSIS` | evidence preserved, no fabricated answer |

A failed Investigator step yields `INCOMPLETE` (not `AGENT_FAILED` as
the status) because real evidence may already exist — the termination
reason still says `AGENT_FAILED`, and downstream steps are blocked
either way (spec §42).

## 11. Typed handoffs, merged evidence, one validation

The message sequence is exactly the six the M16 validator checks:
`INVESTIGATION_REQUEST → INVESTIGATION_RESULT → ANALYSIS_REQUEST →
ANALYSIS_RESULT → VALIDATION_REQUEST → VALIDATION_RESULT`, every hop
carrying `repository_scope` + `commit_scope`, every payload a typed
M16 contract built through `make_message`.

* **Merge** (`merge_investigation_reports`): findings concatenated in
  step order; evidence refs de-duplicated by `(path, line_start,
  line_end)`; open questions/uncertainties/investigated sources
  unioned (order-preserving); `question`/scopes taken from the shared
  `TaskPayload`; `investigation_id` = the FIRST report's; status
  `incomplete` if ANY step was incomplete (downgrade only).
* **Evidence authority**: `gather_evidence` over a mapping of the
  **merged** investigator traces — every attempt's inspected files
  count, because a tool that ran on a failed first attempt really was
  inspected; only the *successful* report decides which refs the
  package claims.
* **Validation**: `validate_collaboration(task, merged_report,
  finding, merged_trace, messages, analyst_trace)` — the M16 code,
  byte-for-byte, all eight rules; a failure maps to
  `VALIDATION_FAILED`, never to a second opinion.

## 12. Observability — one vocabulary, one trace system

Nineteen operations (`ORCHESTRATION_OPERATIONS`) with monotonic
`sequence` numbers on the authoritative run log:

```text
orchestration.start · plan.created · plan.validated
step.start · step.assign · agent.start · agent.complete
step.complete · step.failed · step.blocked
handoff.send · retry
validation.start · validation.complete
budget.exceeded · timeout
complete · failed · terminated        (exactly one terminal event)
```

Each step's events go into a *window* that is mirrored, at step end,
onto that agent's own `AgentTrace` via
`trace.record_orchestration_event` (unknown operations raise — only
the orchestrator records them) and re-written through the same
`TraceSink`. The trace gains an `orchestration_events` field (omitted
from `to_dict()` and from the markdown projection when empty, so every
pre-existing trace stays byte-identical) and a `## Orchestration`
section. Run-level events (start, plan gate, validation, termination)
belong to no single agent's execution and stay on the run only.

Two documented vocabulary deviations: there is no `handoff.receive`
(the sender-side `handoff.send` plus the message list already identify
both ends; a receive-side copy would add no information), and
`run_id` is stamped on `agent.complete`/`StepResult` rather than
`agent.start`, because the trace id does not exist before the agent
run begins.

## 13. Demonstration — one story, seven phases

`python agentrun_m17_orchestration_demo.py` (offline by default; no
API key):

| Phase | Shows |
|---|---|
| 1 plan gate | M10 planner → `validate_plan` → plan P1–P4 |
| 2 roles | step → role with a reason for every step |
| 3 execution | four sequential steps, run ids, attempts, terminal status |
| 4 handoffs | six messages, scope on every hop, budget counters |
| 5 validation | the eight checks over the merged report/finding |
| 6 failure | `PLAN_FAILED` with 0 agent runs; `BUDGET_EXCEEDED` with `P4` blocked |
| 7 status | sealed terminal state, event timeline, mirrored trace events, `## Orchestration` |

Flags: `--live`, `--question`, `--model`, `--repository`, `--db` (the
demo's OWN `runtime/agent_memory/m17_demo.db`), `--snapshots`,
`--traces`. Exit codes `0` (every condition held) / `1` (a condition
failed) / `6` (configuration/execution failure — e.g. `--live` without
`OPENAI_API_KEY`). `--repository tests/fixtures/construction` binds the
read-only tools to a foreign checkout through the dispatcher seam.

## 14. Concept table

| Concept | One-line meaning | Home |
|---|---|---|
| `Orchestrator` | the control plane: gate, roles, step loop, bounds, verdict dispatch | runner.py |
| `OrchestrationConfig` | every run bound in one explicit object | runner.py |
| `OrchestrationRun` | authoritative run record (status, steps, messages, events, verdict) | models.py |
| `StepResult` | one step's typed artifact + role + run id + attempts + error | models.py |
| `RoleAssignment` | step → role + application-generated reason | roles.py |
| `assign_roles` | deterministic classifier + overrides + post-conditions | roles.py |
| status machine | 12 statuses, one gate, terminal sealed, closed reason set | status.py |
| `merge_investigation_reports` | the ONE deterministic multi-report merge | runner.py |
| plan gate | M10 `validate_plan` + `max_plan_steps`, before any agent | runner.py |
| budget gate | `_pre_step_gate` — runs/messages/tool calls/clock before work | runner.py |
| step window | events of one step, mirrored onto that agent's trace | runner.py |
| `ORCHESTRATION_OPERATIONS` | the closed 19-operation vocabulary | agent/trace.py |
| `record_orchestration_event` | the additive trace recorder (raises on unknown ops) | agent/trace.py |

## 15. Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/orchestration/ -q
# → 102 passed   (98 when M17 first landed; +4 boundary/finalization)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
# → 1986 passed, 2 skipped   (1622 when M17 first landed)
```

(M17 adds 98 tests: status, roles, models/merge, runner, demo,
repository-agnosticism. The 2 skips are the pre-existing
environment-conditional ones — Graphviz `dot` and the opt-in OpenAI
smoke test — unchanged since M15; the frozen M15/M16 tests inside
`tests/agent`, `tests/multiagent`, `tests/evidence`, `tests/planning`
still pass (that subset runs 486 green today, including the
post-freeze additions), i.e. M17 touched no frozen behavior.)

The reliability cleanup added four test areas to that number:
boundary/finalization (`tests/agent/test_agent_finalization.py`,
`tests/orchestration/test_m17_finalization.py`), the working-context
ceiling (`tests/agent/test_working_context.py`), the stopping policy
(`tests/agents/test_investigator_stopping_policy.py`), and the
composition budget assertions in `tests/agents/test_repository_query.py`.
The final freeze added two more: envelope/level selection for the
structured path (`tests/llm/test_structured_output.py` +
`tests/agent/test_structured_output_normalization.py`) and the soft
efficiency target (`tests/agents/test_investigator_efficiency.py`).
The deterministic-finalization fix added three (25 tests):
execution-metadata extraction/injection as a unit
(`tests/agent/test_execution_metadata.py`), the same injection
observed through direct and orchestrated specialist runs plus the
substantive-failure guarantee (`tests/agents/test_investigator_metadata.py`),
and the investigation lifecycle end to end
(`tests/agents/test_investigator_lifecycle.py` — decision phase,
one targeted call, already-satisfied gaps, forced tool-free
finalization, flag-off regression, the 64 ceiling, bool validation).

| Spec §45 area | Where |
|---|---|
| status machine, sealed terminals, closed reason set | `test_m17_status.py` |
| role classifier, post-conditions, override rejection | `test_m17_roles.py` |
| run/step state, deterministic merge (dedup, union, downgrade) | `test_m17_models.py` |
| happy path: gate → roles → steps → six handoffs → verdict | `test_m17_runner.py` |
| plan-gate failures run no agent; tool-level goal rejected | `test_m17_runner.py` |
| budgets: steps, agent runs, messages, tool calls, clock | `test_m17_runner.py` |
| bounded retry (and what is never retried) | `test_m17_runner.py` |
| failure semantics: blocked downstream, no fabrication | `test_m17_runner.py` |
| multi-investigator merge feeding the evidence authority | `test_m17_runner.py` |
| events sequenced; mirroring onto both agents' traces | `test_m17_runner.py` |
| no repository identifiers / no framework imports; fixture run | `test_m17_repository_agnostic.py` |
| demonstration phases, `main`, all exit codes | `test_m17_demo.py` |

Harnesses re-verified: `agentrun_test.py` 6/6,
`agentrun_react_test.py` 8/8, `agentrun_evidence_test.py` 10/10,
`agentrun_evaluation_test.py` 13/13.

## 16. Risks and implementation notes

1. **The classifier is a default, not a policy.** Keyword matching
   can misread an unusual goal. Mitigation: the vocabulary is
   deliberately narrow (evidence words never match), every assignment
   carries a reason that a reader can challenge, and
   `role_overrides` is the precise instrument — validated so a typo
   raises instead of silently downgrading to the default.
2. **Overrides can break execution order.** An ANALYST step with no
   preceding INVESTIGATOR step could never receive a report. That is
   now rejected at assignment time (post-condition C), not discovered
   as a runtime failure inside the step loop.
3. **Merging keeps the FIRST investigation id.** The Analyst must
   echo that id, so the finding is tied to the first evidence
   package. Per-step reports remain visible on their `StepResult`.
   If a future milestone needs per-report identities in the finding,
   the merge rule — not the validator — must change.
4. **Evidence from failed attempts counts.** A tool that ran on a
   failed first attempt was really inspected, so every attempt's
   trace joins the evidence authority; only the successful report
   decides which refs the package claims. A scope-mismatched attempt
   therefore cannot inject refs (its report was rejected), but its
   *inspections* remain visible.
5. **The wall-clock bound cannot pre-empt.** The orchestrator is
   single-threaded: a hung provider call overruns until the
   client/provider timeout; the bound is checked between steps and
   yields `TIMEOUT`. Threading/async preemption stays out of scope.
6. **Budgets are checked before work, never during.** A step that is
   already running finishes (or fails) on its own; nothing is
   cancelled mid-flight, so `StepResult` is never half-populated.
7. **The message budget counts the validator's two hops.** The
   pre-validation gate pre-computes `needs_messages=2`; a configuration
   that cannot afford them terminates `BUDGET_EXCEEDED` with the
   outputs intact rather than sending an uncounted message.
8. **Terminal states are sealed by the transition table, not by
   convention.** Any code path that tried to continue would raise
   `OrchestrationTransitionError` — a test asserts the restart is
   impossible for every terminal status.
9. **Sink re-write semantics.** The orchestrator re-writes each
   agent's artifact once per step window through the injected sink
   (same policy as M16: a failing sink write is observability-only,
   never a run failure). `JsonFileTraceSink` overwrites by run id, so
   the artifact converges to include the mirrored events.
10. **`handoff.receive` is absent and `run_id` lands on
    `agent.complete`.** Both are deliberate (§12): the message list
    already identifies the receiving end, and a trace id does not
    exist before the agent run starts.
11. **A single-step plan can never complete.** One step cannot be
    both roles, so the run ends `INCOMPLETE / NO_ANALYSIS` with the
    evidence package preserved — no hidden synthesis run is added to
    make it "succeed".
12. **Shared agent instances across steps.** Each step overwrites
    `agent.last_trace`; the orchestrator keeps its own references to
    earlier traces (identity-checked per attempt), so the merged
    evidence authority never loses a step's trace.

Implementation notes:

* exactly one pre-existing file was touched, additively:
  `agent/trace.py` (`ORCHESTRATION_OPERATIONS`, the
  `orchestration_events` field + recorder + `## Orchestration`
  section); every default stays byte-identical (the field is omitted
  when empty, so all existing trace serializations are unchanged;
  full suite 1524 → 1622 with the 98 new tests, skips unchanged);
* the whole package is new code under `src/repo_intelligence/
  orchestration/` — no frozen M15/M16 file other than `trace.py`
  changed, which is why the frozen tests covering those milestones pass
  untouched;
* no f-strings (Python 3.7), no repository-specific identifier, no
  framework/broker import (both asserted by tests scanning the
  package and the demo source).

## 17. Limitations and deferred work

* **Sequential only.** Steps run one at a time in plan order;
  parallelism, DAG scheduling, and dependency inference are deferred
  (a failing step simply blocks what follows).
* **No replanning.** A rejected or failed plan ends the run; the
  planner is never re-invoked mid-run (M10's rule, unchanged).
* **One plan per run.** Cross-task/cross-analysis orchestration,
  analysis graphs, MCP delegation, and Analyst access to graph/context
  reads build ON this layer in later milestones — the hop language
  (typed messages, permissions, validation) stays as M16 defined it.
* **Role vocabulary is fixed** at INVESTIGATOR/ANALYST; there is no
  role marketplace, registration, or LLM-authored role.
* **No orchestrator-side concurrency, broker, or persistence** — the
  run record is an in-process object; durable queues/workflow engines
  are out of scope.
* **Tool-level evidence only** (inherited from M13): conclusions
  citing architecture-context or graph facts are unsupported under
  `require_inspected=True`.
* **The wall-clock bound is advisory between steps** (no pre-emption,
  §16.5).

## 18. Non-goals and boundary

* **No second planning/agent/validation framework** — M17 is
  application code over M10 + M16; replacing any of them would
  contradict the frozen M8–M16 model.
* **No LLM in a control-plane decision** — roles, budgets, retries,
  and terminal states are computed from contracts and configuration;
  the only model outputs involved are the ones M8.9 already typed.
* **No new evidence, claim, or trace system** — the additive trace
  field and the `gather_evidence` mapping-input are the only
  extension points used.
* **No hidden reasoning** — orchestration events carry application
  decisions (what, which step, which role, why in application terms),
  never model-internal scratchpads.

## 19. Definition of done

Met: one validated M10 plan executed through a sequential,
application-owned step loop; deterministic role assignment with a
reason per step and validated overrides; twelve-status machine with
sealed terminals and a closed termination-reason set; six explicit
bounds checked before every unit of work; bounded retry that never
touches validation failures or exhausted agent budgets; failed
Investigator steps blocking downstream work while preserving evidence;
deterministic report merging feeding the unchanged M16 eight-rule
validator; nineteen orchestration events mirrored onto both agents'
own traces; no repository identifier or framework import anywhere in
the package; a non-ADOP fixture run; an offline-first demonstration
exiting 0 in default and fixture mode (6 for `--live` without a key);
98 new tests, full suite 1622 passed / 2 skipped, all four harnesses
green; documentation including the responsibility matrix, the
two-level architecture, and the complete execution story.

Learning progression: M10 built the plan → M16 built a typed,
validated collaboration → **M17 makes the plan executable**: the
application decides *how, when, and under what constraints* the
planned work runs, with every decision observable, every bound
explicit, and every outcome terminal — the control plane on which
cross-task orchestration can later be built.

**Freeze (reliability cleanup).** Three gaps were closed before the
freeze. The Investigator's versioned prompt now states its stopping
policy — evidence sufficiency first, the tool-call limit as an
emergency ceiling, no duplicate exploration (§4 of
`docs/architecture-investigator.md`) — so a healthy run ends
voluntarily instead of by budget exhaustion; a run that still reaches
the 64-call boundary or a provider `context_length_exceeded` closes
tool-free from the evidence already collected (`finalize_on_boundary`,
`tools.disabled`, `finalization.*`), with `AgentTrace.stop_reason`
naming why execution stopped (`EVIDENCE_SUFFICIENT`,
`BUDGET_EXCEEDED`, `CONTEXT_LENGTH_EXCEEDED`, `ERROR`) in the run
summary. Third, the normal loop's working conversation is now
**replaced** rather than merely bounded: above the unchanged
application ceiling (`max_context_characters`, 300 000 characters,
`M17_EXECUTION_CONFIG`) the run keeps its instruction side, one
deterministic state message built from its own trace (objective,
scope, EvidenceRefs, failed attempts, clipped non-file observations)
and its recent tool window, drops the verbose history, and continues
from the compacted list — `context.compaction.started` →
`.completed`, counts only, no model call. Where compaction cannot
shrink the conversation, the same ceiling still bounds that one
request (`context.restricted`, the fallback). Neither layer ever
touches the trace, so `gather_evidence` stays intact, and neither is
a stop. The 64-call budget is unchanged: one number, both configs.

**Freeze (final M17 fix).** Two focused fixes closed the last gaps.
First, the structured path selects the payload *level* before it
validates: the live finalization response arrived one level deeper
than the contract asked for —
`{"structured_output": {"title": "ArchitectureReport", "type":
"object", "properties": {…}}}` — so validation ran against the
*wrapper* and reported the model's own required fields (`question`,
`repository_scope`, `commit_scope`) as missing. Which level the model
chose is provider behavior; choosing it is the application's job, so
`llm/structured.py` gained `normalize_structured_payload`: a
deterministic, provider-neutral descent (known envelope keys → an
echoed schema's `properties` → a single-key wrapper; never past
`NORMALIZATION_MAX_DEPTH = 3`; a key that is a field of the model is
never descended into) that returns the first level which **fully
validates**, and the original payload — with its original per-field
errors — when nothing does. No field is ever added, dropped, coerced,
or fabricated; `Agent._close_structured_output` applies it before
validation (and before the claims/answer fallbacks), recording
`structured_validation.normalized = true` only when a level actually
had to be chosen; `planning/planner.py` gets it for free through
`validate_structured`.

Second, execution gained a **soft** effort target that is not a
limit: `SOFT_TOOL_CALL_TARGET = 16`
(`AgentExecutionConfig.soft_tool_call_target`, shared by the M17
composition, while `max_tool_calls` stays 64). When a run's executed
tool calls reach it — at most once per run — the Agent appends ONE
user-role execution-state message (`INVESTIGATION_EFFORT_THRESHOLD_
REACHED`, counters only, never message content) before the next
request and records ONE `execution.effort_threshold_reached` event
(`reason`, counters, `signal`); `AgentTrace.effort_threshold_reached`
and `* Soft effort target: reached` are read-only efficiency views,
`stop_reason` never mentions it, and the Investigator's versioned
prompt anticipates the notice as a reassessment — insufficient
evidence stays insufficient (§4 of
`docs/architecture-investigator.md`). Nothing is blocked, nothing
terminates, and 64 remains the only ceiling that does either. New
tests: `tests/llm/test_structured_output.py` (level selection, unit),
`tests/agent/test_structured_output_normalization.py` (the observed
shape through `validate_structured`, the Agent path, the
finalization path, unwrapped unchanged, unfixable still fails), and
`tests/agents/test_investigator_efficiency.py` (sufficiency before
any signal, crossing recorded and not enforced, the notice sent once,
the hard ceiling unchanged, duplicate searches, prompt discipline).

**Freeze (deterministic finalization + hard investigation
lifecycle).** Two focused fixes closed the last gaps.

First (lifecycle), the soft target became a real boundary instead of
pure telemetry: `AgentExecutionConfig.enforce_soft_target` (default
`False`; enabled only by `M17_EXECUTION_CONFIG`, non-plan runs) turns
the crossing documented in §9 into the decision phase —
`execution.investigation_reassessed` (point 1), then exactly one
`execution.investigation_decision` per point (`FINALIZE`, or
`ONE_TARGETED_CALL` with the batch cut to one executed call and the
rest blocked `SOFT_TOOL_CALL_TARGET_REACHED`), `execution.
targeted_call_started` on the one allowed dispatch, reassessed
(point 2), then FINALIZE through the existing tool-free machinery.
The assistant turn carries only calls that actually executed, a
gap the evidence already answers never triggers a call, the budget
check still runs first so 64 remains the only hard stop, enforcement
never records `execution.boundary_reached` (`stop_reason` stays
`EVIDENCE_SUFFICIENT`), and the versioned prompt states the same
decision (§4 of `docs/architecture-investigator.md`).

Second (metadata), the run's identifying facts are now injected
deterministically instead of being model-authored: `agent/metadata.py`
derives `question`, `repository_scope`, `commit_scope` from the run's
own request text (`trace.user_question`, the line-start labels every
transport carries — direct TASK block, orchestrated wrapper,
finalization pass) and fills exactly those keys when the model
omitted them and the contract declares them. A value the model wrote
is never rewritten, no other field is ever touched (substantive
payloads still validate and fail exactly as before), and
`structured_validation.metadata_injected` records the filled keys on
success and on failure. The one label gap this exposed was fixed too:
the orchestrated wrapper now says `question:` (it said `objective:`),
and the prompt's OUTPUT block names the three fields as assignment
facts.

New tests (25): `tests/agent/test_execution_metadata.py` (whitelist,
extraction rules, fill/no-overwrite/contract-gate, round-trip),
`tests/agents/test_investigator_metadata.py` (direct + orchestrated
injection, no-overwrite, substantive failure never hidden), and
`tests/agents/test_investigator_lifecycle.py` (tests C–G: decision
phase, finalize with no extra call, one targeted call then reassess,
already-satisfied gaps, `M17_CONFIG` asserts 64 + enforcement,
forced finalization with `tools=[]`, flag-off regression, hard
boundary unchanged, bool validation).

One offline CLI run of
`python agentrun_repository_query.py "How do the agents, tools, and
MCP servers interact?"` against the pinned snapshot
(`sample-Agentic-Ai-Data-Operations@ef11dac…`) exits 0: three steps,
**16 tool calls** total (8 + 8 + 0 across the two Investigators and
the Analyst), **9 iterations** (4 + 4 + 1), **0 context compactions**
and 0 context restrictions (largest request 17 messages — nowhere near
the 300 000-character ceiling, so nothing to compact), 0
boundary/length/finalization events (**finalization reason: none**),
every step terminated on its own (`STRUCTURED_OUTPUT`,
`stop_reason = EVIDENCE_SUFFICIENT`), collaboration and diagram
validation passed, artifacts written, `Status: COMPLETED`,
`termination: SUCCESS`. The same run's metrics after every final fix:

| Metric | Offline run (this environment) |
|---|---|
| Total LLM iterations | **9** (4 + 4 + 1 across the three steps) |
| Investigator tool calls | **16** (S1 architecture 8, S2 flow 8) |
| Total tool calls | **16** (Analyst 0; 0 blocked) |
| Soft threshold reached | **no** — 0 `execution.effort_threshold_reached` events (the largest single agent run used 8 of the 16-target), `* Soft effort target: reached` absent |
| Context compactions | **0** (0 `context.restricted`; the largest request stays far below the 300 000-character ceiling) |
| Finalization reason | **none** — no `finalization.*`, boundary, or length events; every step terminated `STRUCTURED_OUTPUT` with `stop_reason = EVIDENCE_SUFFICIENT` |
| Final workflow status | `Status: COMPLETED`, `termination: SUCCESS` |
| Validation status | collaboration **passed** (M16 deterministic verdict), diagram **passed** (six rules), every terminal iteration `structured_validation.status = SUCCESS` |
| Structured-output normalization required | **no** — no `normalized` marker on any iteration; every payload validated at the level it arrived at |
| Execution metadata injected | **no** — the scripted payloads already carried `question`/`repository_scope`/`commit_scope`, so `metadata_injected` is absent everywhere (the fill is fill-missing-only) |

The scripted offline model cannot demonstrate model-driven stopping,
a ceiling crossing, the enforced decision phase, or the envelope
shape. Live provider behavior is attempted with the required command
and is unavailable **in this environment**:
`python agentrun_repository_query.py "How do the agents, tools, and
MCP servers interact?" --live --verbose` exits 1 with
`ERROR: Live mode unavailable. ... OPENAI_API_KEY is not configured
... No investigation was executed.` (no `.env`, no key in the
environment; the SDK itself imports). Each of those behaviors is
covered deterministically instead: compaction in
`tests/agent/test_context_compaction.py` (measured example: 10 492 →
5 919 characters, 6 → 5 messages) and the compacted handoff in
`tests/agents/test_architecture_investigator_m17.py`; the envelope
shape through `validate_structured`, the Agent path, and the
finalization path in `tests/agent/test_structured_output_normalization.py`;
the signal-only crossing and the unchanged 64 ceiling in
`tests/agents/test_investigator_efficiency.py`; the enforced
lifecycle (decision phase, one targeted call, forced tool-free
finalization, flag-off regression) in
`tests/agents/test_investigator_lifecycle.py`; and metadata
injection end to end in `tests/agent/test_execution_metadata.py` +
`tests/agents/test_investigator_metadata.py`. Suite: **1986 passed,
2 skipped**, pyflakes clean. Freeze assessment for this environment:
**M17 NOT READY TO FREEZE** — exact blocker: the mandated live E2E
cannot run (`OPENAI_API_KEY is not configured`), so the offline run
above cannot stand in for it; rerun the live command with a key to
complete the freeze. M18 (framework comparison) is next once the live
E2E passes.

## 20. Post-M17 specialists — the same orchestrator, three useful participants (addendum)

Added when the post-M17 specialist capabilities landed: M17's
orchestrator, planner, runner, roles, bounds, and validator are
**unchanged** — what changed is what a caller can inject into them.

```text
Architecture Investigator      → investigator slot (ArchitectureReport)
Flow / Dependency Investigator → investigator slot (FlowDependencyReport)
Architecture Visualization Agent → analyst slot  (DiagramSpec → ArchitectureFinding)
```

Because the orchestrator holds exactly **one** investigator slot, a
three-step plan routes that slot by the plan's own step label — the
`step: <id> — <goal>` question the runner already embeds — through a
demo/test-owned `RoutedInvestigator` adapter (never product code, never
an orchestrator edit). The role chain uses the repository's actual
role names:

```text
INVESTIGATOR   (architecture investigation)
    ↓
INVESTIGATOR   (flow / dependency investigation)
    ↓
ANALYST        (visualization handoff — ArchitectureVisualizationAgent)
    ↓
validator      (the unchanged deterministic M16 eight-rule verdict)
    ↓
renderer       (deterministic stage — Mermaid/DOT/SVG from graph/exporters)
```

Typed handoffs stay exactly as M16/M17 defined them: investigators
project onto `InvestigationReport`, the visualizer projects its
`DiagramSpec` onto `ArchitectureFinding` (deterministic `inv-viz-…`
correlation id), and failure propagation is the same — a failed
investigator step blocks the visualization (the visualizer is never
consulted, no diagram is fabricated), and a constructible-but-out-of-
scope spec is rejected by the deterministic diagram validator, never
persisted.

See `docs/architecture-visualization-agent.md` §13, the end-to-end
tests in `tests/agents/test_architecture_visualization_m17.py`, and
the nine-phase demonstration:

```bash
python agentrun_architecture_visualization_demo.py
```
