# M10 — Explicit Plan-and-Execute (plan → validate → execute)

## 1. Purpose

M10 makes the investigation **explicitly planned before it is
executed**: one provider-neutral LLM planner turns the objective and
the existing M6 Architecture Context into a structured,
goal-level `InvestigationPlan`; deterministic validation gates it;
the validated plan is recorded on the **existing** `AgentTrace`
before the first repository tool ever runs; and each plan step is
then executed **sequentially** by the existing ReAct loop under the
existing single `max_tool_calls` budget.

One end-to-end run, one trace, one budget, one result (spec §20):

```text
Objective + Architecture Context (passed directly, never a tool call)
        │
        ▼
 planner LLM call  (tools=[], M8.9 structured output: InvestigationPlan)
        │
        ▼
 deterministic validation  ── FAILED ──► plan recorded, iteration closed,
        │                                PLAN_VALIDATION_FAILED, no tool
        │                                ever executes, error re-raised
        ▼ SUCCESS
 PLAN RECORDED on this run's AgentTrace  (before the first tool)
        │
        ▼
 P1 ── existing ReAct loop ── ToolDispatcher ── repository evidence
        │  (iterations tagged plan_step_id="P1")
        ▼
 P2 ── existing ReAct loop ── ...
        ▼
 P3 ── final step ──► AgentRunResult  (M8.9 output, M8.8 claims,
                                        M9 evaluation — unchanged)
```

The learning goal: **an investigation can state what it will do
before doing it, and every step of execution stays accountable to
that recorded plan** — without replacing the adaptive ReAct loop
that decides *how* to achieve each goal.

## 2. Boundary — what this is (and is not)

| Concern | Owner | M10 change |
|---|---|---|
| Producing the plan | `repo_intelligence.planning` (one LLM call) | **new** |
| Validating the plan | `planning.validation` (deterministic) | **new** |
| Recording the plan | `agent.trace` (`plan`, `architecture_context`) | additive fields |
| Executing steps | existing ReAct loop (`_execute_loop`) | plan-aware, still one loop |
| Budget | existing `AgentExecutionConfig.max_tool_calls` | unchanged — plan steps **share** the one bound |
| Structured output / claims | M8.9 / M8.8 | unchanged (contract only on the final step) |
| Evaluation | M9 | unchanged — runs after, over the same run |

Explicitly **not** in M10: no new engine (`ReactEngine`/
`PlannerEngine`), no replanning, no plan repair, no parallel
execution, no DAG/scheduler, no sub-agents/multi-agent, no second
trace, no second budget, no incremental sink writes, no hidden
reasoning (banned fields stay banned), no tool-level plans (steps
state WHAT, never HOW — no tool names/arguments in goals).

Planner orientation comes from the **existing** Architecture Context
artifact, passed directly into the planning request (spec §8) — the
planner never calls `get_architecture_context`; that tool remains an
execution-time tool for steps that need it.

## 3. Data model — `repo_intelligence.planning.models`

```text
PlanStatus   = Literal[pending | in_progress | completed |
                       failed | invalidated | blocked]
PlanStep     { id, goal, status, result? }
InvestigationPlan { objective, steps, status }   # objective/steps required
plan_to_dict(plan, validation=None)              # trace payload shape
```

- `PLAN_STATUS_*` constants + `PLAN_STATUSES` frozenset (M8.8 style).
- Extra planner keys are ignored (tolerant input), missing required
  keys fail validation (strict output).
- `plan_to_dict` emits `{objective, status, [validation],
  steps: [{id, goal, status, [result]}]}` — the exact payload stored
  at `trace.plan`; step order in the list **is** execution order.

## 4. The planning contract

### 4.1 The one planner request (`planning.planner`)

`build_planning_request(objective, architecture_context=None)` →

- system message: `PLANNING_SYSTEM_INSTRUCTION` — goal-level
  sub-goals (typically 3–6, ids P1, P2, …), no tool calls, no
  execution, no private reasoning; the context is *orientation, not
  truth*;
- user message: the objective + the Architecture Context markdown
  (when supplied);
- `tools=[]` — the planner cannot execute anything;
- `structured_output = spec_for_model(InvestigationPlan)` — the M8.9
  provider-neutral contract (schema + name + strict validation).

`plan_from_response(response)` decodes via the existing M8.9
machinery; a missing/invalid payload raises
`StructuredOutputValidationError` (stage `schema`).

`load_architecture_context(output_root=None, snapshot_id=None)` →
`(markdown, record)` reads the existing M6 artifact through
`context.default_context_markdown_path` and parses the artifact's own
source-metadata lines (contract schema version, pinned commit,
snapshot id) — never re-derived. Missing/empty artifact raises
`ArchitectureContextError`.

### 4.2 Deterministic validation (`planning.validation`)

`validate_plan(plan)` raises `PlanValidationError` unless **all** of:

1. objective present and non-blank (the LLM's objective field, not
   re-used from the question);
2. at least one step;
3. every step id non-blank and **unique**; goal non-blank;
4. no **tool-level** goal — rejected when a goal matches
   `_TOOL_LEVEL_PATTERNS` (anchored `call|invoke|execute|run`
   followed by a tool-like name, `name(...)` invocation form, or
   `"arguments":`/`"tool_calls":` JSON fragments);
5. no **unsupported directive** — rejected when a goal matches
   `_UNSUPPORTED_DIRECTIVE_PATTERN` (`replan`/`re-plan`/`replanning`,
   "call the planner", "planner again", "parallel execution",
   "spawn" — case-insensitive).

Validation never executes anything and never repairs: failure is
recorded (stage `application`), the plan is marked `invalidated`,
the run terminates `PLAN_VALIDATION_FAILED`, and the error
re-raises — exactly one failure path, no retries (spec §18/§29).

## 5. Trace integration — one trace, chronologically honest

`agent.trace` gains only additive fields (M1–M9 projections
unchanged for runs without a plan):

- `AgentTrace.plan: Optional[dict]` — recorded payload (step 3 above)
  including the `validation` outcome; `record_plan()` deep-copies +
  clips like every other field; serialized in `to_dict` before
  `iterations` (chronology: plan recorded, then execution).
- `AgentTrace.architecture_context: Optional[dict]` — the `{artifact,
  schema_version, commit_sha, snapshot_id}` record, recorded **before**
  the planning call (spec §23/§26 ordering).
- `AgentIteration.plan_step_id: Optional[str]` — which plan step an
  iteration belongs to (planning iteration = `None`); serialized only
  when set, immediately after the `iteration` key.
- `TERMINATION_REASONS` += `PLAN_VALIDATION_FAILED`,
  `PLAN_INVALIDATED`.

Markdown projection (`render_markdown`):

- `## Plan` section (Objective / Status / `Plan validation:` /
  `### Plan Steps` + the Architecture Context artifact line) placed
  after *Question*, before *Iterations* — the plan existed before
  execution;
- `Plan step: \`P1\`` line after each execution iteration's heading;
- step-boundary iterations render "Plan step … result recorded." /
  "Continue plan execution." where ordinary runs render the existing
  tool-result continuation lines (non-plan projections byte-identical).

No reasoning fields: `plan`/`architecture_context` payloads contain
only plan and provenance data; `BANNED_REASONING_FIELDS` are asserted
absent in JSON and Markdown.

## 6. Execution semantics — `Agent.run_planned`

`run_planned(question, architecture_context=None,
architecture_context_record=None)` is a second entry point next to
`run_result` with the same result type (`AgentRunResult`) and the
same shell: input validation → trace start → … → `finalize()` +
sink write → return. The differences are the planner call and the
plan-aware loop:

1. **Record context first** (when supplied) →
   `_create_plan`: the planner request is itself recorded as an
   observable iteration (`response_type: STRUCTURED_OUTPUT`,
   `structured_validation: SUCCESS` on success); on failure the
   iteration closes with `structured_validation: FAILED`, the plan
   payload (schema stub or validated-but-failed plan) is recorded,
   and the run terminates `PLAN_VALIDATION_FAILED`.
2. **`_PlanExecution`** owns the plan lifecycle during execution and
   re-records the payload at every transition (`start_first`,
   `start_next`, `complete_current`, `invalidate_current`,
   `interrupt(reason)`, `finish(answer)`), so `trace.plan` always
   reflects reality. Construction normalizes a fresh plan's steps to
   `pending`.
3. **One `_execute_loop` invocation spans the whole plan** (the
   existing ReAct loop, made plan-aware): the local budget counter
   *is* the global bound — plan steps share the one
   `max_tool_calls` (spec §30), no per-step budgets, no scheduler.
   Per iteration:
   - `begin_iteration(..., plan_step_id=plan.current.id)`;
   - structured spec (`Finding`-style contract) is attached **only on
     the final step** — intermediate steps are free-form so the M8.9
     contract does not constrain step results (and never a second
     contract);
   - a text response on a **non-final step** is the step's result:
     `complete_current(text)` → `start_next()` → append the next
     step's instruction message (`PLAN_STEP_INSTRUCTION`: step id,
     index/total, goal, objective, ordered outline, plus
     `PLAN_FINAL_STEP_HINT` on the last step) → iteration closes
     `FINAL_ANSWER` with a plan-boundary continuation → `continue`;
   - **blank content** on a step → `invalidate_current()` →
     `PLAN_INVALIDATED`, the run *returns* (answer `""`, no replan,
     no further requests — spec §29);
   - an exception mid-step → `interrupt(trace.termination_reason)` —
     the step/plan become `failed`, or `blocked` when the reason is
     `EXECUTION_LIMIT_REACHED` — then the original error re-raises;
   - claims parsing/evidence validation (M8.8) and structured output
     (M8.9) happen only on the terminal path, exactly as before.
4. **Non-plan runs are untouched**: `run_result` → `_run_loop`
   wrapper → `_execute_loop(..., plan=None)`; every existing test
   (M1–M9) passes unchanged.

### 6.1 Failure semantics (spec §17/§18/§29)

| Failure | Effect | Replan? |
|---|---|---|
| planner `LLMError` | iteration closed by `record_llm_error`, plan never recorded, `LLM_ERROR`, error re-raised | no |
| malformed plan JSON / wrong shape | `StructuredOutputValidationError`, stage `schema`, `PLAN_VALIDATION_FAILED` | no |
| semantic validation failure | `PlanValidationError`, stage `application`, plan `invalidated`, `PLAN_VALIDATION_FAILED` | no |
| blank step result | run returns answer `""`, `PLAN_INVALIDATED`, 3 requests total (plan + one per started step) | no |
| step `LLMError` / dispatcher error | step + plan `failed`, original error re-raised (trace records why) | no |
| budget exhausted | current step + plan `blocked`, `EXECUTION_LIMIT_REACHED` re-raised | no |

In every case the trace records the reason; no failure is silently
converted into a success.

## 7. Demonstration (live — `agentrun_m10_plan_demo.py`)

```bash
python agentrun_m10_plan_demo.py
python agentrun_m10_plan_demo.py --model gpt-4o-mini
python agentrun_m10_plan_demo.py --question "Investigate …"
```

Composition-root style identical to the M9 demo: loads the existing
M6 artifact via `planning.load_architecture_context()`, composes the
existing `Agent` (OpenAI client + `default_tool_dispatcher` +
`JsonFileTraceSink`, one budget, M8.9 contract), runs
`agent.run_planned(objective, context, record)`, prints the recorded
plan (`validation`, per-step outcomes), the execution summary with
the plan-step timeline, then invokes the existing M9 evaluator once
and writes `trace.{json,md}` + `evaluation.{json,md}`.

Exit codes: `0` — completed and evaluation PASS/PARTIAL (honest
UNKNOWNs reported); `1` — evaluation FAIL with criterion detail;
`6` — planning/execution/configuration failure (missing
`OPENAI_API_KEY`, missing Architecture Context artifact,
`LLMError`, `AgentError`, `PlanValidationError`,
`StructuredOutputValidationError`) printed as
`Agent execution failed.` The script contains no planner prompt, no
validation rules, and no loop — it only supplies inputs and reads
back results.

Offline wiring: `tests/planning/test_plan_demo.py` (10 checks —
composition path, direct context (no `get_architecture_context`
call), one evaluator invocation, exit codes, missing key/context →
6 before client construction, full offline `main`).

## 8. Tests

| Suite | Tests | Covers |
|---|---|---|
| `tests/planning/test_plan_models.py` | 10 | model defaults/required fields, status vocabulary, extras ignored, `plan_to_dict` shape |
| `tests/planning/test_plan_validation.py` | 21 | ordering, uniqueness, tool-level rejection, directive rejection, negative (domain-word) cases |
| `tests/planning/test_planner.py` | 14 | request contract (tools=[], schema name, context in message), decode failures, context loader + provenance parsing |
| `tests/agent/test_agent_planned_trace.py` | 13 | additive trace fields, `plan_step_id` iteration tag, markdown plan section/step lines, banned fields, non-plan projections unchanged |
| `tests/agent/test_agent_planned.py` | 16 | `run_planned` acceptance: plan-before-first-tool (dispatch snapshot), sequential P2-after-P1, ReAct inside steps, shared bound (max=1), all failure semantics, structured contract only on final step, non-plan runs unchanged |
| `tests/agent/test_agent_planned_e2e.py` | 2 | the complete §39 path: objective + context → planner → validation → recorded plan → P1–P3 with dispatcher evidence → one result/trace; JSON + Markdown projections |
| `tests/planning/test_plan_demo.py` | 10 | demo wiring (see §7) |

Regression: full suite **855 passed, 2 skipped** (baseline 769 +
86 new); compileall clean; pyflakes clean over M10 scope
(`planning/`, `agent.py`, `trace.py`, demo, M10 tests); harnesses
8/8 (`agentrun_react_test.py`), 10/10 (`agentrun_evidence_test.py`),
13/13 (`agentrun_evaluation_test.py`) — M1–M9 behavior untouched.

## 9. Non-goals and freeze criteria

Non-goals (all explicit future milestones, never implicit): plan
repair/replanning, parallel or DAG step execution, scheduler,
sub-agents/multi-agent planning, per-step budgets, LLM-judged plan
quality, plan caching/versioning, enterprise workflow engines.

**Freeze criteria:** M10 is complete when the plan is recorded in the
existing trace before the first tool execution, steps execute
sequentially within the one existing bound, every failure mode above
is observable in JSON and Markdown, the demonstration runs end-to-end
against a real model, and the full M1–M9 regression stays green.
