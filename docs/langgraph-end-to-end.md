# LangGraph end-to-end workflow — production opt-in + primary notebook

**Status:** implemented and verified (offline). **Date:** 2026-10-19.
**Interpreter:** `venv312/bin/python` (3.12.3), `langgraph==1.2.15`
(optional `graph` extra — unchanged from M21).

## 1. What this milestone is

Two deliverables on top of the M21 graph loop (`docs/m21-langgraph-
orchestration.md`):

1. **Production seam** — `agentrun_repository_query.py` / `run_
   repository_query` can now execute the task loop through the SAME
   compiled-state-graph machinery, as an explicit **opt-in**
   (`orchestration="langgraph"`, CLI `--orchestration langgraph`).
   The legacy `Orchestrator.execute` path stays the default; there is
   **no silent fallback** (an unavailable graph dependency is a typed
   structured failure, never a surprise switch of execution engine).
2. **One primary notebook** — `notebooks/langgraph_end_to_end_
   workflow.ipynb` demonstrates the complete multi-agent workflow
   (question → planner → graph topology → execution → evidence →
   validation → legacy-vs-graph parity → failure behaviour →
   safeguards) using the repository's **real** implementation, no
   re-implemented mini-rules.

Deferred deliberately (per the migration-audit addendum): parallel
execution, checkpointing, replanning, live-provider runs, flipping
the default to LangGraph, deleting the legacy loop.

## 2. Production seam

### 2.1 Entry contract

`run_repository_query(question, repository=None, *, mode="offline",
diagram=True, artifacts=True, db_path=None, trace_root=None,
artifact_directory=None, orchestration="legacy")` — new keyword-only
parameter, validated against `ORCHESTRATION_MODES = ("legacy",
"langgraph")`; an unknown value raises `RepositoryQueryError`
(same style as the existing `mode` validation).

CLI: `--orchestration {legacy,langgraph}` (default `legacy`).
When non-default, the banner prints one line:
`Orchestration: langgraph (M21 opt-in task loop)`.

### 2.2 The seam itself

`_run_with_langgraph(orchestrator, task, plan)` (module-private in
`repository_query.py`):

- imports `run_with_langgraph` from
  `orchestration/langgraph_runner` **lazily** (the core package still
  imports without the optional `graph` extra);
- preassigns roles with the **same** `assign_roles` deterministic
  assigner the legacy executor uses internally, so both paths get
  identical role validation, plan validation, budgets, and terminal
  validation; the graph then owns task selection, attempts, retry,
  commits, and assembly;
- the injected `Orchestrator` is a **compatibility runtime only** —
  `Orchestrator.execute` is never called on the graph path
  (test-pinned in both `tests/agents/test_repository_query_langgraph.
  py` and `tests/orchestration/test_langgraph_sequential.py`).

The planner (`_build_plan`) stays **off-graph** and is shared verbatim
by both backends (H1 request-budget preflight → `PLAN_FAILED`
before any agent runs).

### 2.3 Failure semantics (no silent fallback)

| Situation | Behaviour |
|---|---|
| `orchestration="legacy"` | exactly the pre-milestone path (default) |
| `orchestration="langgraph"`, graph available | full graph loop; `Orchestrator.execute` never called |
| `orchestration="langgraph"`, `ImportError` (missing extra) | structured `_failure(..., termination_reason=ORCHESTRATION_UNAVAILABLE)`, exit code `EXIT_INVESTIGATION_FAILED` — **no fallback** to legacy |
| graph node raises (e.g. budget stop) | propagates as `RuntimeError`/typed terminal status — never re-routed to legacy |
| planner over-budget (either backend) | `PLAN_FAILED` via H1 preflight, provider never called |

## 3. The notebook

`notebooks/langgraph_end_to_end_workflow.ipynb` — 10 sections, 30
cells (17 pure-Python code cells, no IPython magics, no hardcoded
machine paths, no secrets):

1. Learning objectives and architecture (Mermaid diagram).
2. Environment and setup (`find_repo_root()`, `_require_graph()`).
3. Inspect the real components (`run_repository_query` signature,
   `_run_with_langgraph`/`_build_plan` source).
4. Build (compile) the actual graph (state keys, nodes, edges).
5. Execute an end-to-end query (offline composition: planner +
   S1/S2 specialists + S3 visualizer through `run_with_langgraph`).
6. Inspect state transitions and evidence (task records, event
   timeline, trace attribution, findings).
7. Failure behaviour (invalid plan → `PLAN_FAILED`; no-analyst plan →
   `INCOMPLETE/NO_ANALYSIS` with no fabricated finding; broken
   structured output → one retry, `attempts=2`).
8. Legacy versus LangGraph (field-by-field parity; run ids disjoint).
9. Safeguards (H1 oversized-planner demo: `PLAN_FAILED`, provider
   never called; tool-response ceiling and graph projection noted).
10. Learning summary and next steps.

**Verification method (honest):** Jupyter is not installed in this
environment, so the notebook was verified by (a) nbformat-4 JSON
validation, (b) `ast.parse` of every code cell, and (c) sequential
`exec` of all 17 code cells in ONE fresh process at the repo root —
the same execution semantics a kernel would apply to these pure-Python
cells. All cells completed with the expected deterministic outputs
(graph COMPLETED/SUCCESS, M16 passed, diagram passed, parity EQUAL,
retry `attempts=2`, oversized planner → `PLAN_FAILED`).
`tests/agents/test_langgraph_notebook.py` (4 tests) pins the
structure and the referenced production symbols so a renamed seam
breaks the test, not a learner's kernel.

## 4. How to run

```bash
# production, legacy default (unchanged)
venv312/bin/python agentrun_repository_query.py "…question…"

# production, opt-in LangGraph task loop
venv312/bin/python agentrun_repository_query.py "…question…" \
    --orchestration langgraph

# the notebook (any Jupyter environment; offline, deterministic)
venv312/bin/python -m jupyter lab notebooks/langgraph_end_to_end_workflow.ipynb
```

## 5. Test gates

```bash
# production seam + notebook contract (expected: 15)
venv312/bin/python -m pytest tests/agents/test_repository_query_langgraph.py \
    tests/agents/test_langgraph_notebook.py -q

# M21 graph behavioural tests (expected: 34)
venv312/bin/python -m pytest tests/orchestration/test_langgraph_sequential.py -q

# full suite (expected: 2505 passed, 2 skipped)
venv312/bin/python -m pytest tests/ -q
```

## 6. Limitations

- Live-provider runs (`--live`, `OPENAI_API_KEY`) are **not** exercised
  in this environment (no key available) — offline mode is the
  verification path; live graph runs remain untested here.
- The notebook's sequential-exec verification is documented above
  (§3); running it in a real Jupyter kernel is recommended for
  learners but was not possible in this environment.
- Parallelism, checkpointing, replanning, and flipping the default to
  LangGraph are deferred; see `docs/langgraph-migration-audit.md`
  addendum for the full deferred list.

## 7. See also — restricted one-investigator smoke test

`notebooks/langgraph_one_investigator_smoke_test.ipynb` (2026-10-19)
is the restricted companion to the primary notebook above: fixture
plan `["P1"]`, ONE Investigator executed through
`OneTaskExecutionPolicy`, terminate `INCOMPLETE`/`NO_ANALYSIS` — no
Analyst, no retry, no second task, and **zero provider calls by
default** (a `ProviderCallGuard(limit=0)` class-patch, not just a
flag). The optional live cell (one REAL planner call + offline
investigator) is gated behind `LIVE_MODEL_TEST = False` +
`CONFIRM_LIVE_CALLS = False` and carries experiment-specific limits
SEPARATE from the global H1 budget: `max_tokens=2048` output (verified
to reach the provider payload's own `max_tokens` parameter) and an
ESTIMATED 8,000-token input ceiling enforced locally twice via the
SAME H1 machinery — a pre-call probe on the exact outgoing request +
`CallLimitedClient(input_budget=...)` re-enforcement at the client
boundary. Oversized requests are REJECTED before the provider (reason
+ est-vs-ceiling displayed); the estimate is a chars-per-token ratio,
not a tokenizer; no monetary cap is claimed; global H1–H4 defaults
(95,232 tokens) are unchanged. `max_calls=1` + `max_retries=0`; a
failed call consumes the one-call allowance, a budget rejection does
not. Tests: `tests/agents/test_langgraph_one_investigator_smoke.py`
(21) plus `tests/llm/test_openai_client.py` (21 + 1 skip, including
the additive `max_tokens`/`max_retries` contract). The live path was
not exercised in this environment (no `OPENAI_API_KEY`).

Section **11 — Shared ReAct context-safety demonstration (offline)**
of that notebook additionally demonstrates the REUSABLE,
graph-agnostic controls in `repo_intelligence.react_safety` using
fake clients and fake tools only (zero provider calls; the armed
class-level guard is never touched): `BudgetedLLMClient` measures
and enforces the COMPLETE outgoing request before the provider
boundary (an oversized request causes ZERO provider calls; a budget
rejection never consumes a call allowance while a provider failure
does), `bound_observation` bounds each tool observation before it
joins history, `ReActLoopGuard` enforces iteration / tool-call /
model-call / cumulative-input limits with clear terminal statuses,
and `restrict_history` performs per-request history restriction
without mutating the caller's transcript. Any hand-rolled ReAct loop
can opt into these controls directly — the module has no
agent-name/task-name branching and no second budget implementation
(it delegates to the existing `agent.request_budget` machinery and
reuses the tool-protocol response ceiling). Contract tests:
`tests/test_react_safety.py` (17). These controls are also
integrated into the real application path: the production `Agent`
loop bounds every tool observation before it joins the conversation
(`tests/agent/test_react_integration.py`, 4 tests) and the offline
graph workflow's agent traces each carry per-iteration
request-budget records
(`test_repository_query_langgraph.py::test_safety_records_ride_
every_model_call_of_the_graph_workflow`).
