# M9 — Investigation Evaluation (deterministic post-run evaluation)

> **Status:** COMPLETE / FROZEN · **Tests:** 86 (`tests/evaluation/`) ·
> **Harness:** `python agentrun_evaluation_test.py` (13 checks, offline) ·
> **Live demo:** `python agentrun_m9_evaluation_demo.py` (real model)

---

## 1. Purpose

M8 answered *"Can the agent investigate?"*. M9 asks the different,
deliberately harder question:

> **Did this completed investigation succeed — against an explicit
> definition of success?**

The answer is produced by an **evaluator that runs after the Agent
finished**, over the artifacts the run already left behind. It is not
another agent: it never reads the repository, never calls a model,
never dispatches a tool, never continues or repairs the run. It is a
pure function over existing runtime structures:

```text
Agent.run(...)  →  AgentRunResult  →  Evaluator.evaluate(...)  →  EvaluationResult
```

One meaningful concept is introduced: **evaluation is separate from
execution**. Everything else (score, judge, benchmark) is explicitly
left for future milestones.

---

## 2. Boundary — three questions, three milestones

These are different concepts and must not be merged:

| Milestone | Question | Output |
|---|---|---|
| **M8.8** evidence validation | *"Can this claim be traced to evidence gathered during this run?"* | `SUPPORTED / UNSUPPORTED / UNKNOWN` per claim |
| **M8.9** structured output | *"What typed result did the Agent produce?"* | a typed model instance or an explicit error |
| **M9** investigation evaluation | *"Did this completed investigation satisfy its explicit success criteria?"* | `PASS / FAIL / PARTIAL / UNKNOWN` per criterion + overall |

The evaluator **consumes** what M8.8 and M8.9 produced; it never
reimplements them (see §7). The trace remains *what happened*; the
evaluation is *how the completed run performed* — related artifacts,
deliberately separate (§8).

---

## 3. Data model — smallest useful evaluation model

`src/repo_intelligence/evaluation/models.py` — typed models, no
dictionary spelunking, **no numeric score anywhere**:

```text
EvaluationResult
├── status            PASS / FAIL / PARTIAL / UNKNOWN   (overall)
├── criteria[]        EvaluationCriterionResult, fixed CRITERIONS order
│   ├── criterion     final_result · evidence_gathered ·
│   │                 claim_traceability · execution_bound ·
│   │                 objective_coverage
│   ├── status        PASS / FAIL / PARTIAL / UNKNOWN
│   └── explanation   observable fact, never a hidden rationale
├── summary           deterministic counts: "4 PASS, 1 FAIL, …"
└── run_id            which run this evaluation belongs to

EvaluationCriteria                      (input — what success means)
├── objective            the question the run set out to answer
└── objective_criteria[] ObjectiveCriterion { key, description } —
                         the explicit definition of objective coverage
```

Status vocabulary (small, fixed, mirrors the M8.8 style):

```text
PASS      the criterion was evaluated and holds
FAIL      the criterion was evaluated and does not hold
PARTIAL   evaluated but incomplete (e.g. 2 of 3 claims supported)
UNKNOWN   cannot be evaluated from these inputs (no claims were
          produced, no expected criteria were defined, the run
          summary is missing) — NEVER silently upgraded to PASS
```

A single number (`score = 87`) would hide *why* a run succeeded or
failed; the learning objective is evaluation **mechanics**, not
numerical optimization. Every criterion carries its own short,
observable explanation instead.

---

## 4. The evaluation contract — five fixed criteria

```text
1. final result        Did the run produce a terminal result?
2. evidence gathered   Did the investigation actually gather
                       repository evidence (successful tools only)?
3. claim traceability  Were claims traceable to evidence? (M8.8
                       result consumed as-is)
4. execution bound     Did execution respect the configured
                       tool-call limit? (M8.5 facts, blocked ≠ executed)
5. objective coverage  Was the requested objective addressed?
                       (explicit expected fields on the structured
                       result — success must be defined, not assumed)
```

Deliberate rule: **evaluation requires an explicit definition of
what success means.** With no `objective_criteria` supplied, criterion
5 reports `UNKNOWN` — the evaluator never pretends generic text
quality is objectively measurable.

---

## 5. Aggregation and failure semantics

Deterministic overall rule (no weights, no hidden tuning):

```text
any FAIL                        → FAIL
else any PARTIAL / UNKNOWN      → PARTIAL
else (every criterion PASS)     → PASS
```

Selected failure semantics (spec §16):

* no final result → `final_result = FAIL`
* answer with zero successful tool calls → `evidence_gathered = FAIL`
  (a final textual answer alone is **not** evidence)
* claims with unresolved evidence → `claim_traceability = PARTIAL`
  or `FAIL` — the distinction is preserved, never flattened
  (M13: `supported > 0` → `PARTIAL`; `supported = 0` with claims
  recorded → `FAIL`, whether every claim is `UNKNOWN` — claims that
  reference no evaluable evidence — or `UNSUPPORTED` with
  `CLAIM_EVIDENCE_INVALID`)
* the run legitimately terminated at the configured limit →
  `execution_bound = PASS`, while `objective_coverage` may still
  `FAIL` → **overall FAIL**. Respecting the execution bound does not
  mean the investigation succeeded — that is the core M9 lesson
  (§9, Run C).

---

## 6. Architecture

```text
                        Agent Run
                           │
              +------------+------------+
              |                         │
              v                         v
           Trace                  Structured Result
              |                         |
              v                         v
      Evidence Validation           (typed, M8.9)
           (M8.8)
              |                         |
              +-------------+-----------+
                            |
                            v
                     Investigation
                       Evaluation
                         (M9)
                            |
                            v
                   EvaluationResult
                    |            |
                    v            v
             evaluation.json  evaluation.md
```

`src/repo_intelligence/evaluation/` contains exactly three modules:

```text
models.py      data model + status vocabulary
evaluator.py   Evaluator.evaluate — the only entry point (pure)
sink.py        JsonFileEvaluationSink — persistence, separable
```

---

## 7. Consumption, not reimplementation (M8.8 / M8.9)

The evaluator reads what earlier milestones already produced:

* **M8.8** — the recorded `AgentRunResult.evidence_validation`
  statuses are *interpreted* (`all supported → PASS`, `mixture →
  PARTIAL`, `none supported → FAIL`, `none recorded → UNKNOWN`;
  M13 clarifies `none supported` to include *claims recorded but none
  reference evaluable evidence* — an answerable "no", reported as
  `FAIL` with the count, never hidden behind `UNKNOWN`). Evidence
  matching, path resolution, line-range validation and claim
  extraction are never duplicated (a source-level test asserts
  `gather_evidence(` / `validate_claims(` do not appear in the
  evaluation package).
* **M8.9** — the validated `AgentRunResult.structured_output` (typed
  model or mapping) is consumed **as-is** for objective coverage; the
  evaluator never reparses raw JSON the Agent already validated.
* **M8.5** — the execution bound reads the recorded run summary
  (`max_tool_calls`, `tool_calls_executed`, `tool_calls_blocked`);
  blocked calls were never executed, so they are never counted as
  executions.
* The recorded evidence statuses and the run summary are inputs, not
  recomputation targets: evaluation performs zero additional
  validation work.

Cross-package layering: `evaluation` imports only stdlib +
`typing_extensions`, the evidence **vocabulary** (status constants),
and `default_trace_root` (sink only). No `..llm` import, no provider,
no tool API — enforced by a source-guarantee test.

---

## 8. Where evaluation artifacts live

The existing trace design already separates telemetry from durable
outputs, so M9 follows it (spec §14's second option) and adds
**nothing** to the trace itself:

```text
runtime/agent_runs/<run_id>/
    trace.json          what happened          (M8.x — untouched)
    trace.md
    evaluation.json     how it performed       (M9)
    evaluation.md
```

`trace = what happened`; `evaluation = how the completed run
performed against explicit criteria`. JSON is canonical; Markdown is
rendered from `EvaluationResult.to_dict()` — never an independent
source of truth. Embedding the full evaluation inside the trace was
rejected: it would blur the existing separation and turn the trace
into an evaluation database.

---

## 9. Demonstration (offline — `agentrun_evaluation_test.py`)

```text
Investigation Evaluation

Final result:          PASS
Evidence gathered:     PASS
Claim traceability:    PASS
Execution bound:       PASS
Objective coverage:    PASS

Overall:               PASS
```

and the deliberately incomplete run — **an Agent can execute
correctly and still fail the investigation objective**:

```text
Investigation Evaluation

Final result:          PASS
Evidence gathered:     PASS
Claim traceability:    PASS
Execution bound:       PASS        ← 5 of 5 executed, compliant
Objective coverage:    FAIL        ← loader field empty

Overall:               FAIL
```

The harness additionally evaluates the captured M8.7 trace fixture,
where unevaluable criteria honestly report `UNKNOWN` (overall
`PARTIAL`) — never a silent PASS because "the run looks fine".

One documented ceiling: in a single real Agent run today,
`structured_claims` (M8.8) and `structured_output_model` (M8.9) are
mutually exclusive, so one run supplies claims *or* a structured
result, not both. Real runs therefore cannot reach five `PASS` on
their own; the full-PASS demonstration assembles both artifact kinds
deterministically, and real-run integrations show the honest
`UNKNOWN` for the criterion their run mode cannot supply. This is
reported, not papered over.

---

## 10. Running the M9 demonstration (end-to-end, live)

`agentrun_m9_evaluation_demo.py` (project root — the same
`agentrun_*` runner convention as the M8.7 live runner) is a
human-runnable, end-to-end demonstration of the execution/evaluation
boundary over a REAL model investigation:

```bash
python agentrun_m9_evaluation_demo.py                 # needs OPENAI_API_KEY
python agentrun_m9_evaluation_demo.py --model gpt-4o-mini
python agentrun_m9_evaluation_demo.py --question "Investigate …"
```

Conceptual flow — every box is an **existing** component:

```text
Objective
   ↓
Agent              existing — ONE run_result(objective) call; M8.4
   ↓                loop, read-only tools, M8.9 structured contract,
   ↓                one tool-call budget; the script never calls the
   ↓                LLM or a tool itself
AgentRunResult     (+ agent.last_trace: iterations, tool results,
   ↓                summary)
M9 Evaluator       Evaluator.evaluate(result, trace, criteria) —
   ↓                invoked ONCE, strictly after the run
EvaluationResult   existing vocabulary: PASS / FAIL / PARTIAL / UNKNOWN
   ↓
evaluation.json / evaluation.md     JsonFileEvaluationSink — beside
                                     the run's trace
```

What the script does:

1. composes the existing `Agent` — `OpenAILLMClient` +
   `default_tool_dispatcher()` + `JsonFileTraceSink()`, one budget,
   `structured_output_model` contract; configuration follows the
   existing `.env` / `load_env()` convention (`OPENAI_API_KEY`,
   never hard-coded);
2. executes ONE real investigation (`agent.run_result(objective)`
   with a generic, repository-agnostic objective);
3. passes the completed `AgentRunResult`, the run's trace, and an
   `EvaluationCriteria` (objective + expected structured fields) to
   the **existing** evaluator;
4. prints the five criteria with the existing labels and status
   vocabulary, the overall verdict, and the summary counts;
5. persists the evaluation with the existing sink.

**The script does not implement evaluation.** It contains no
criteria logic, no status vocabulary, and no serialization of its
own — it invokes `repo_intelligence.evaluation.Evaluator.evaluate`
once and reads back the `EvaluationResult`. It introduces no second
loop, dispatcher, retry, or planner.

### Interpreting the output

* `PASS` — evaluated and holds; `FAIL` — evaluated and does not hold
  (each non-PASS row prints its observable explanation; any `FAIL`
  makes the overall verdict `FAIL`).
* `UNKNOWN` — the criterion cannot be evaluated from this run's
  information. In this demonstration **`claim_traceability =
  UNKNOWN`** is expected and honest: the M8.9 structured-output run
  mode produces no M8.8 structured claims. UNKNOWN is reported,
  never upgraded to `PASS`, and pulls the overall verdict to
  `PARTIAL`. **M13 changes what is measurable, not the
  vocabulary:** a claim-bearing run records claims, so
  `claim_traceability` becomes an answerable question — `PASS` when
  every claim's evidence is valid, `FAIL` when claims are recorded
  but none reference evaluable evidence (or cite uninspected paths
  with `CLAIM_EVIDENCE_INVALID`), `UNKNOWN` only when the run
  recorded no claims at all. The M13 experiment demonstrates the
  claim-bearing mode reaching overall `PASS` where the baseline
  stopped at `PARTIAL`.
* Objective coverage **is** evaluated in this mode: the expected
  `finding` / `evidence_paths` fields are checked on the typed
  result the run produced.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | run completed; overall `PASS` **or** `PARTIAL` — no criterion `FAIL`ed (UNKNOWNs are printed and recorded, not hidden) |
| `1` | run completed, but evaluation `FAIL`ed — prints `Investigation completed, but evaluation failed.` plus per-criterion detail |
| `6` | agent execution/configuration failure (missing `OPENAI_API_KEY`, `LLMError`, `AgentError`, structured-output validation failure) — prints `Agent execution failed.` |

Unexpected programming errors are never caught: they propagate with
a traceback rather than masquerading as an evaluation failure.

Artifacts written for the run:

```text
runtime/agent_runs/<run-id>/
    trace.json / trace.md            (Agent's sink, during the run)
    evaluation.json / evaluation.md  (M9 sink, after evaluation)
```

The offline wiring tests
(`tests/evaluation/test_evaluation_demo.py`) cover this entry point
with scripted doubles — no key, no network; the live run is the
script itself.

---

## 11. Reproducibility and repository-agnosticism

* `Evaluator.evaluate(result, execution, criteria)` is a pure
  function: no timestamps, no randomness, no model in the loop —
  same completed run + same criteria → **byte-identical
  `to_json()`**. Inputs are never mutated.
* The evaluator operates on Agent runtime concepts only. No ADOP,
  MCP, agent-name, tool-name, path or directory hard-coding; no
  `if tool == …`, no `if repository == …` (spec §25). Tests evaluate
  a generic *"Explain how configuration reaches the application."*
  fixture (`config.yaml`, `application.py`, `loader.py`) and assert
  the verdict payload is free of repository terminology.

---

## 12. Tests

`tests/evaluation/` — 86 tests, fully offline (no key, no network, no
repository checkout):

| File | Covers |
|---|---|
| `test_evaluation_models.py` | valid/invalid states, round-trip, fixed order, counts, input validation, no score, banned vocabulary, determinism |
| `test_evaluator.py` | the five criteria (incl. zero/below/at/over limit, blocked-not-counted, failed-tool-not-counted, M8.8 consumed not recomputed, M8.9 consumed not reparsed), overall aggregation, Run A/B/C, determinism, repo-agnostic fixture, source guarantees, input errors |
| `test_evaluation_sink.py` | artifact layout beside the trace, JSON round-trip, markdown projection, run-id requirement, no trace overwrite |
| `test_evaluation_integration.py` | real Agent runs (scripted offline doubles): claims run, structured run, `EXECUTION_LIMIT_REACHED` run, no-side-effects, determinism, repo-agnostic payload |
| `test_evaluation_demo.py` | demonstration entry-point wiring: import, existing Agent built through the demo path, existing Evaluator invoked exactly once (spied) with run+trace+criteria, artifact written, exit codes, missing-key failure, repo-agnostic source |

Regression: the full existing suite runs green alongside M9 — M8.1–M8.9
stay untouched.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
python agentrun_evaluation_test.py                    # offline harness
python agentrun_m9_evaluation_demo.py                 # live end-to-end demo
```

---

## 13. Non-goals, freeze criteria, learning answers

M9 does **not** introduce (spec §24 — future learning milestones):

* LLM-as-a-judge · evaluator/critic agents · reward models
* benchmark infrastructure · statistical quality scoring
* production monitoring · human feedback systems
* automated remediation · automatic retries · agent self-correction
* planning · persistent memory · multi-agent workflows
* a generic numeric `score` · a second ReAct loop

Freeze criteria — all verified for this milestone:

* implementation works · tests pass (86 M9 + full suite green)
* documentation accurate · repository-agnostic behavior demonstrated
* M8 regression suite passes (existing harnesses 8/8, 10/10)

Acceptance criteria (spec §27): **AC-001 – AC-018 all satisfied** —
independent post-run evaluation (AC-001), deterministic (AC-002),
repository-agnostic (AC-003), the five criteria evaluated
(AC-004…AC-005, AC-008, AC-009), M8.8/M8.9 consumed not reimplemented
(AC-006, AC-007), no tools / no LLM / no second loop (AC-010–AC-012),
compliance ≠ success (AC-013), successful + unsuccessful
investigations covered (AC-014), M8.x green (AC-015), boundary
documented (AC-016) and repo-agnostic docs (AC-017), no enterprise
evaluation abstractions (AC-018).

```text
M9 — Investigation Evaluation
STATUS: COMPLETE / FROZEN

Agent executes.
Evaluator evaluates.

M8.8 validates claim-to-evidence traceability.
M8.9 validates structured output.
M9 evaluates the completed investigation against explicit criteria.
```

LLM-based judging is deliberately **not** added after M9 — it should
become a separate future learning milestone only once this
deterministic evaluation boundary is understood.
