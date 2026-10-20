# M15 — Context Engineering

> **Status:** implemented, validated, frozen. The deterministic,
> inspectable selection of *what information enters the model's
> context for one reasoning step* — candidates → explicit selection
> policy → `ContextPackage` → render → model → tool → new candidates →
> new snapshot — behind an injected `ContextEngine`, executed **per
> reasoning boundary inside the real Agent loop** (§10), with every
> decision recorded on the same trace and every M1–M14 test still
> green.

## 1. Why M15 exists

M14 made validated conclusions *survive* between executions: the agent
retrieves memory at its execution boundary and renders it into the
conversation. What it does not do is *decide* what enters context. The
memory text, the question, and whatever else the composition root
appends go into the model as an unordered bundle: nothing states which
candidates were considered, why each one was included or left out, how
much room it took, or what would have happened under a tighter budget.

That gap matters now that there are many candidate sources. The task
itself, repository scope facts, the engineering graph, current-run
evidence, persisted memory, raw tool output — each is useful, none is
universally useful, and their combined size is unbounded. Feeding all
of it forward is wasteful and sometimes harmful (stale memory reading
as current fact); feeding a hand-picked subset with no record is worse:
the model's context becomes an unaudited claim.

M15 closes exactly that gap, one selection capability:

> **Every reasoning step's context is assembled by a deterministic
> policy over an explicit candidate list: scope → freshness →
> relevance → dedup → budget, each candidate receiving a stable
> selected/excluded decision with a machine-readable reason; the
> resulting `ContextPackage` is what the model reads, the same
> selection is persisted as an inspectable JSON snapshot, and the
> trace records both what was *available* (memory) and what was
> *selected* (context) as separate events.**

It is a context-assembly milestone, not a retrieval milestone: no
embeddings, no vector store, no RAG, no LLM choosing its own context,
no second trace (§63 non-goals).

## 2. What M15 does / does not do

### Does

- Adds `src/repo_intelligence/context_engine/` — `errors.py`,
  `models.py`, `policy.py`, `engine.py`, `rendering.py`,
  `evaluation.py`, `adapters.py`, `__init__.py`. **The package never
  imports an LLM client, `sqlite3`, or a database** (source-scanned).
- Defines the **candidate contract** (`ContextCandidate`): one source's
  offer for one step — content, source identity, pinned
  repository/commit scope, authority, evidence references, freshness,
  priority, relevance tags — validated application-side before any
  policy runs.
- Runs the **documented selection pipeline** (`select_context`, M15 §7)
  over a `TaskContext` anchor: validate → scope filter → freshness →
  relevance → deterministic sort → exact-identity dedup → greedy
  budget fill → documented presentation order. Every candidate ends
  with exactly one `ContextSelectionDecision` — selected or excluded,
  with a stable reason from a fixed vocabulary (8 selection reasons,
  7 exclusion reasons). Nothing is dropped silently.
- Enforces **hard budgets** (§12): item count *and* characters, with
  the actual usage recorded on the package. A budget that cannot admit
  any eligible candidate raises `CONTEXT_BUDGET_EXCEEDED` — never an
  empty or over-budget package, never a silent truncation.
- Produces a **`ContextPackage`** — the exact, renderable content of
  one step — and `render_context_text(package)` the single projection
  the model receives (header + budget line + items in presentation
  order + open questions/constraints).
- Persists **`ContextSnapshot`** records as JSON under
  `runtime/context/<run_id>/context_<step_id>.json` (§42): package +
  every decision + excluded ids + run/step identity. The file *is*
  `snapshot.to_dict()` — what a reader inspects is the contract, not a
  view of it.
- Gives the `Agent` an **injected** `ContextEngine` (`context_engine=`
  ctor param). Selection runs **per reasoning boundary inside the
  existing loop** (§10): at the execution boundary the agent retrieves
  memory (M14), adapts task + memories to candidates, selects,
  renders, and records `context.select` / `context.render` events on
  the **same** trace (§27 — no second tracer); after every executed
  tool batch the observed `ToolResult`s are adapted
  (`candidates_from_tool_result`), selection re-runs for the next
  step, a new snapshot persists, and the new rendered package joins
  the conversation as that step's context message — every event
  carrying its `trigger` (`initial` / `tool_result`).
  A `ContextOperationError` is recorded as a structured error event +
  `agent.last_context_error` and produces **no context message at
  all** — a failed selection never degrades into a different,
  unrecorded fallback (§55).
- Keeps memory and context visibly separate (§56): `memory.retrieve`
  records what was *available*; `context.select` records what was
  *selected*. Available ≠ selected, proven numerically in the
  demonstration and in tests.
- Ships **deterministic evaluation** (§46/§47): `evaluate_context`
  computes counts, budget utilization, evidence coverage, freshness
  and isolation metrics, and `render_evaluation_report` renders the
  fixed report — no LLM judge, no prose scoring.
- Ships the **offline-first demonstration**
  (`agentrun_m15_context_demo.py`, §57): nine phases telling one
  story — inventory → snapshot → model step → tool step → snapshot
  evolution → memory interaction → budget → full agent run →
  evaluation report.

### Does not

- No embeddings, vector stores, similarity search, reranking, or RAG.
  Selection is deterministic filtering and sorting; "semantic
  relevance" is a future concern.
- No LLM in the selection path — the engine never sees a model, and
  the model never sees the policy's internals (only its output
  package).
- No second trace, no context-specific database, no context-specific
  framework. Context → JSON snapshots only; memory → the one SQLite
  store (§43).
- No change to the frozen M8–M14 stack: same planner, same ReAct loop,
  same tools, same claim validation, same memory lifecycle. A run
  without a context engine behaves byte-for-byte as M14 (asserted by
  tests).
- No policy of its own in the demonstration — the demo is a
  composition root; selection always happens inside
  `repo_intelligence.context_engine`.

## 3. Candidates and the task (§8/§9)

A **candidate** is what one source *offers* to one step. It never
selects itself; the policy decides, and the decision is recorded.

| Field | Meaning |
| --- | --- |
| `candidate_id` | Caller-assigned, stable across the select/snapshot pair (composition roots pass deterministic ids; the default generates an opaque `cand_<uuid>`). |
| `source_type` | One of eight: `task`, `repository`, `fact`, `graph`, `evidence`, `memory`, `tool_result`, `agent_state`. |
| `content` | The actual text that would enter context (non-empty). |
| `source_id` | The source record identity — feeds exact-identity dedup. |
| `repository_scope` / `commit_scope` | Where the claim is pinned. `None` = scope-neutral (eligible anywhere). |
| `authority` | Optional override of the source's §14 default. |
| `evidence[]` | `EvidenceRef{path,line_start,line_end}` — the frozen M13 reference type. |
| `memory_id` | Set by the memory adapter (M14 identity travels). |
| `freshness` | `fresh` / `stale` / `unknown` when declared; memory defaults to `unknown` when no comparison exists. |
| `priority` | Integer tie-breaker — higher first, only *within* equal authority. |
| `relevance[]` | Free-form tags matched against the task's tags. |

The **authority hierarchy** (§14) is data, not prose:

```text
task 600 > evidence 500 > repository 450 > graph_fact 400
     > memory 300 > tool_result 200 > agent_state 150 > inference 100
```

The **`TaskContext`** anchors every selection: `question`,
`repository`, `commit`, `open_questions`, `constraints`, `relevance`
tags. Its repository/commit define the scope filter; its tags feed
the relevance stage; its open questions/constraints travel into the
package untouched (never silently dropped).

Adapters (`adapters.py`) bridge existing contracts without new
formats: `candidate_from_task(task)` and
`candidate_from_memory(memory, current_commit, ...)` — the latter
computes freshness with the **M14 tri-state rule**
(`verification_status`): both commits known and equal → `fresh`,
both known and different → `stale`, either missing → `unknown`.
`current_commit=None` therefore always adapts as unknown: only a
positively established comparison makes memory eligible as current
context.

## 4. The selection policy — seven documented stages (§7)

`select_context(task, candidates, budget, step_id)` runs, in order:

```text
1  validate          every candidate satisfies the contract;
                     candidate_ids unique within one selection
2  scope             repository/commit filter — scope-neutral passes,
                     pinned candidates require a matching task;
                     non-memory commit mismatch → OUT_OF_SCOPE;
                     (memory commit differences are the NEXT stage's
                     job so the reason is STALE_MEMORY, not a blunt
                     OUT_OF_SCOPE; repository mismatch still excludes
                     a memory first — isolation beats freshness)
3  freshness         memory: fresh proceeds; stale → STALE_MEMORY;
                     unknown → UNKNOWN_FRESHNESS (never silently fresh)
4  relevance         a candidate whose tags are KNOWN and entirely
                     disjoint with the task's tags → IRRELEVANT;
                     untagged on either side is neutral
5  sort              total deterministic order: scope match, then
                     §14 authority, then relevance match, then
                     priority, then source_type + (source_id,
                     content) — the random candidate_id NEVER
                     decides order (§24: equal candidates stay in
                     the same order across runs)
6  dedup             exact identity only (§22): same source record,
                     else identical content within the same source
                     type → DUPLICATE (the first in sort order wins)
7  budget            greedy fill in sort order; a candidate that does
                     not fit is recorded BUDGET_EXCEEDED and the
                     greedy fill CONTINUES (smaller later candidates
                     may still fit); eligible candidates exist but
                     nothing fits → CONTEXT_BUDGET_EXCEEDED raised
8  presentation      package items re-ordered into the documented
                     section order (§23): task 10, repository 20,
                     agent_state 30, evidence 40, fact 50, graph 60,
                     memory 70, tool_result 80 — then positional
                     ids ctxitem_001… assigned, so the id encodes
                     exactly the order the model reads
```

Every stage's outcome lands in the decision list. The reason
vocabulary is fixed (`SELECTION_REASONS` / `EXCLUSION_REASONS`):

| Selected because | Excluded because |
| --- | --- |
| `TASK_DIRECT` (the step's question) | `OUT_OF_SCOPE` (repository/commit mismatch) |
| `REPOSITORY_SCOPE` (same-repo scope fact) | `STALE_MEMORY` (declared/derived stale) |
| `REPOSITORY_FACT` (fact source) | `UNKNOWN_FRESHNESS` (no comparison — never fresh) |
| `CURRENT_EVIDENCE` (this run's evidence) | `DUPLICATE` (exact identity) |
| `GRAPH_NEIGHBOR` (graph query result) | `BUDGET_EXCEEDED` (did not fit) |
| `VERIFIED_MEMORY` (fresh memory) | `LOW_PRIORITY` (item-count cut) |
| `RECENT_TOOL_RESULT` (this run's tool output) | `IRRELEVANT` (tags known and disjoint) |
| `AGENT_STATE` (execution bookkeeping) | |

Determinism (§30/§24): the same inputs produce the same items, the
same order, the same decisions, and the same reasons. Only ids
(`cand_*`, `ctx_*`, snapshot ids) and timestamps vary between runs —
and opaque ids never *decide* anything: the final tie-break is
`(source_id, content)`, so two runs with disjoint uuid id sets select
identically (asserted by `test_agent_context_selection_is_deterministic_across_runs`).

## 5. Budget — hard ceilings (§12)

`ContextBudget(max_items, max_characters, selected_items,
estimated_characters)` — maxima *and* observed usage in one record.
The `ContextPackage` constructor enforces the invariant: recorded
usage == the actual items (count and summed character length), and
never above the maxima. A budget is only trustworthy when the
consumption travels with it.

- Defaults: `20` items / `12000` characters
  (`DEFAULT_CONTEXT_MAX_ITEMS` / `…_CHARACTERS`); callers pass their
  own (the engine holds a default, `select` accepts an override).
- Greedy fill in sort order, skip-and-continue: under a tight budget
  a large candidate is recorded `BUDGET_EXCEEDED` and smaller later
  candidates are still considered — the ceiling is never breached
  silently, and the fill never stops earlier than it must.
- **Nothing fits** (eligible candidates exist, budget admits none) →
  `ContextOperationError(CONTEXT_BUDGET_EXCEEDED)`. The caller sees a
  structured failure; it never receives an empty package pretending
  to be a selection, and never an over-budget one.
- Usage (`selected_items`, `estimated_characters`, remaining) is part
  of every snapshot and every evaluation report.

## 6. The package and its single rendering (§10/§23)

`ContextPackage` = what the model sees for one step: `context_id`,
`task`, `step_id`, scope, `selected_items` (presentation order),
`open_questions`, `constraints`, `selection_summary` (counts by
reason/source), `budget` (with usage), `created_at`.

`render_context_text(package)` is the **only** projection into model
context (§23):

```text
Context selected for this reasoning step (deterministic policy — …)
ctx_… | step 001 | scope: ADOP@cccccccccccc | budget: 762/6000 chars, 5/64 items
Task: How are the ADOP agents, tools, and MCP servers connected? …

[ctxitem_001] task | TASK_DIRECT | authority task | scope ADOP@…
How are the ADOP agents, tools, and MCP servers connected? …

[ctxitem_002] repository | REPOSITORY_SCOPE | authority repository | priority 5 | …
Repository ADOP at commit cccccccccccc: …

[ctxitem_003] evidence | CURRENT_EVIDENCE | authority evidence | …
Inspected README.md:1-1125 — current-run evidence for the investigation.
…
```

Each item line carries its selection reason — the model reads *why*
it is included (§55: no hidden context; the rendering is the whole
context, nothing else is smuggled in alongside it). A non-package
argument raises `CONTEXT_RENDER_FAILURE` — a render failure is
explicit, never an empty string that would silently become an empty
context.

## 7. Snapshots — the inspectable record (§11/§42)

`ContextSnapshot` binds one selection to its run and step:
`snapshot_id`, `run_id`, `step_id`, `created_at`, the package, every
`selection_decisions` row, and `excluded_candidates`. `snapshot()`
refuses a step id that does not match the package's own step (a
package can never be relabeled into another step's record).

`ContextEngine.persist(snapshot)` writes:

```text
runtime/context/<run_id>/context_<step_id>.json
```

The file is `snapshot.to_dict()` — a flat JSON record (`snapshot_id`,
`run_id`, `step_id`, `created_at` + the package's fields +
`selection_decisions` + `excluded_candidates`). Runtime telemetry,
deliberately outside `artifacts/`, gitignored, CWD-independent
(default root from `default_context_root()`). An I/O failure raises
`CONTEXT_SNAPSHOT_FAILURE` — never a silently skipped artifact.

## 8. The engine and its isolation guard (§5)

```python
engine = ContextEngine(
    budget=ContextBudget(max_items=64, max_characters=6000),
    repository="ADOP",          # optional binding — the isolation guard
    snapshot_root=None,         # default: runtime/context
)
selection = engine.select(task, candidates, budget=None, step_id="001")
snapshot  = engine.snapshot(selection, run_id, step_id)   # or select_snapshot(...)
path      = engine.persist(snapshot)
```

The constructor takes only policy configuration — no LLM client, no
dispatcher, no store: composition roots inject whatever produced the
candidates. When bound to a repository, the engine **refuses**
(`CONTEXT_SCOPE_MISMATCH`) a task scoped to a different repository,
and refuses to snapshot or persist a package scoped elsewhere —
cross-repository context is impossible through this API. Candidates
from other repositories still flow *through* the policy so their
exclusion stays observable (`OUT_OF_SCOPE` decisions) rather than
being filtered away before anyone can see them.

Structured errors (all raised as `ContextOperationError` with a
stable code): `CONTEXT_CONTRACT_VIOLATION`, `CONTEXT_INVALID_CANDIDATE`,
`CONTEXT_BUDGET_EXCEEDED`, `CONTEXT_SCOPE_MISMATCH`,
`CONTEXT_RENDER_FAILURE`, `CONTEXT_SNAPSHOT_FAILURE`.

## 9. Memory interplay — available ≠ selected (§55/§56)

M14 retrieves; M15 decides. At the agent's execution boundary:

```text
_retrieve_memories(trace)                → memory.retrieve — N AVAILABLE
candidate_from_task / candidate_from_memory   (explicit adaptation)
engine.select(task, candidates)           → context.select — M selected
render_context_text(package)              → context.render — what the
                                             model actually receives
```

The two responsibilities stay visibly separate on the trace (§56):
retrieval may return three memories while the package contains one —
the other two are present as *decisions* (`STALE_MEMORY`,
`UNKNOWN_FRESHNESS`), not as silent omissions. In the demonstration
the numbers are literal: `memory.retrieve — 3 retrieved` next to
`context.select — 4 candidates → 2 selected / 2 excluded`.

Freshness is the same tri-state as M14 (§24): a memory pinned to an
earlier commit is excluded `STALE_MEMORY`; a memory whose commit is
unknown is excluded `UNKNOWN_FRESHNESS` — unknown is never silently
treated as fresh, and "available" is never conflated with "current".

On any `ContextOperationError` the agent records
`context.select` (or `context.render`) with `error{code,message}`,
mirrors it on `agent.last_context_error`, and produces **no context
message** — the investigation continues exactly as a pre-M15 run
would (legacy `render_memory_context` path), never with a different,
unrecorded context (§55).

## 10. Context evolution during agent execution (§2/§5/§17/§33)

Context is not constructed once for an entire execution. It is
reconstructed at meaningful reasoning boundaries — and that
reconstruction happens inside the **real Agent loop**, not only in the
standalone demonstration. New tool results, evidence, state, or other
candidates can change what should enter the next model context.

```text
Step 1 (execution boundary)
Task + Memory + Graph        ← candidates generated once, here
        ↓
Context selection #1  (context.select step 001)
Context package #1    (context.render step 001)
        ↓
LLM
        ↓
Tool call → ToolResult       ← observed something new
        ↓
Step 2 (post-tool boundary)
Task + Memory + Graph + Tool Result + Evidence   ← accumulated
        ↓
Context selection #2  (context.select step 002, trigger tool_result)
Context package #2    (context.render step 002)
        ↓
LLM → tool call / answer → … (every tool batch repeats the cycle)
```

Planned runs (M10 `run_planned`, the demonstration's phase 8) use
the SAME lifecycle — the planner is simply the first model
invocation after step `001`:

```text
Context selection #1 → render → PLANNER (iteration 1, no tools)
        → validated plan (P1, P2, … recorded on the trace)
        → planned steps execute → tool → selection #2 → …
```

so context #1 feeds the planner request AND the first execution
message, the plan executes inside the same run, and every
`runtime/context/<run_id>/context_<step>.json` snapshot carries the
SAME run id as `runtime/agent_runs/<run_id>/trace.json` — snapshot ↔
trace ownership is one run, not one phase (the demonstration prints
it explicitly: `snapshot ownership: … belong to run <run_id>`).

Mechanics in the Agent (no second agent, no second loop — the same
`_execute_loop`):

- **`_ContextRunState`** carries the run's task, accumulated
  candidates, trace, scope, and the per-boundary step counter
  (`001`, `002`, …) across the whole execution. Created once at the
  boundary by `_begin_context_state`, threaded into the loop.
- **`_prepare_context_for_step(state, new_candidates, trigger)`** is
  one reasoning step's complete lifecycle: extend candidates →
  `engine.select` → snapshot + persist → `context.select` event →
  `render_context_text` → `context.render` event → returned text.
- **Triggers** record *why* a boundary ran: `initial` at the
  execution boundary (task + retrieved memory), `tool_result` after
  an executed tool batch. The trace shows the evolution's cause, not
  just its occurrence.
- **Tool results become candidates, never memory** (§7): each
  observed `ToolResult` is adapted through
  `candidates_from_tool_result` — one `tool_result` candidate
  (authority 200) plus, when the payload carries
  `path`/`line_start`/`line_end`, one `evidence` candidate (authority
  500) carrying the existing `EvidenceRef`. The deterministic policy
  still decides; memory formation remains M14's controlled path.
- **Candidates accumulate**; memory is retrieved once at the
  boundary (never re-retrieved per step). Selection re-runs over the
  full, honest set — a repeated observation stays a `DUPLICATE`
  instead of growing the context.
- **Snapshots represent real reasoning steps**: every selection the
  agent performs persists
  `runtime/context/<run_id>/context_<step_id>.json`, so an
  execution with two boundaries has `context_001.json` and
  `context_002.json` — demonstration-independent (§42).
- **Failure semantics stay §55**: a select/render error records a
  structured event, sets `last_context_error`, and produces no
  context text for that step (the run continues); a snapshot-write
  error is recorded as `context.snapshot` while the already-selected
  context still renders — a selection success is never discarded
  because an artifact write failed, and no failure is silent.
- **No new information → no re-selection**: a tool-less execution
  selects exactly once (step `001`). The capability is per
  *meaningful* boundary, not per request.

The boundary this preserves:

```text
Working memory:
    what the Agent has in its execution state/messages —
    the chronological `messages` list, unchanged as the mechanism;
    a rendered context package is appended as an ordinary user
    message at each boundary (Option C: never mutate history)

Persistent memory:
    information intentionally retained across executions (M14,
    MemoryManager, one SQLite store)

Context:
    information SELECTED for the model at this reasoning step —
    the ContextPackage, not the conversation

Evidence:
    provenance supporting repository-derived claims (M13,
    EvidenceRef — reused, never redefined)

Graph:
    structured engineering relationships (M5 — consumed through
    candidate adapters, never imported by the engine)

Tool result:
    newly observed information from a tool — a candidate source,
    not memory and not working memory
```

`Working memory ≠ persistent memory ≠ context` therefore remains
literally true in the code: the loop owns messages, M14 owns
formation, M15 owns selection.

## 11. Trace integration — one trace, context as events (§27)

`AgentTrace` gains exactly one additive field (`context_events:
List[Dict]`, omitted from `to_dict()` when empty so legacy trace
shapes stay intact) plus `record_context_event()` (deep-clipped like
every other trace payload). Two operations are recorded by the
agent — **once per reasoning boundary**, so a run that executed one
tool records each operation twice (steps `001` and `002`):

- `context.select` — context/step ids, candidate/selected/excluded
  counts, the full budget record, per-candidate decisions, reason
  counts, `trigger` (`initial` / `tool_result`), repository/commit
  (or `error{code,message}`);
- `context.render` — context/step ids, the rendered character count,
  and the same `trigger` (or the render error).

The markdown trace renders a `## Context` section after `## Memory`
and before the iterations:

```text
## Context

* `context.select` — `ctx_b20d8ff…`, step `001`, 4 candidates → 2 selected / 2 excluded, budget 264/6000 chars, 2/64 items, repository `ADOP`, commit `cccc…`, reasons: STALE_MEMORY=1, TASK_DIRECT=1, UNKNOWN_FRESHNESS=1, VERIFIED_MEMORY=1
* `context.render` — `ctx_b20d8ff…`, step `001`, 930 characters
* `context.select` — `ctx_84880ca…`, step `002`, 6 candidates → 4 selected / 2 excluded, budget 524/6000 chars, 4/64 items, repository `ADOP`, commit `cccc…`, reasons: CURRENT_EVIDENCE=1, GRAPH_NEIGHBOR=…, RECENT_TOOL_RESULT=1, …
* `context.render` — `ctx_84880ca…`, step `002`, 1509 characters
```

No second trace, no hidden reasoning field, no context-specific
markdown file — context is events on the same `AgentTrace` the M8.8
rendering already consumes, and never carries model-internal
reasoning (same observability policy as every other recorded block).

## 12. Evaluation — deterministic metrics (§46/§47)

`evaluate_context(selection_or_snapshot)` computes, from the recorded
decisions alone:

| Metric | Meaning |
| --- | --- |
| `considered` / `selected` / `excluded` | decision counts |
| `context_size` / `budget_max` / `budget_utilization` | what entered vs. the ceiling (percent, 2 decimals) |
| `duplicate_count` / `budget_blocked_count` | what the dedup and budget stages actually did |
| `evidence_coverage` | share of selected items carrying ≥1 `EvidenceRef` |
| `memory_items_selected` | memory that made it through |
| `stale_items_excluded` / `unknown_items_excluded` | freshness behaviour, numeric |
| `out_of_scope_excluded` / `irrelevant_excluded` | isolation and relevance, numeric |
| `isolation_ok` | `True` iff every *selected* decision was scope-matched (a boolean assertion, not a score) |
| `by_source` / `by_reason` | full count maps |

`render_evaluation_report(metrics)` renders the fixed §47 report —
lines, no prose scoring, no LLM judge. Two reports over two snapshots
are byte-comparable across runs given equal inputs.

Over a whole agent run, `evaluate_context_run(snapshots)` (fed
`agent.last_context_snapshots`) composes the same metrics across
every boundary: selection/snapshot counts, **`context_evolution_count`
= `len − 1`** (how many times a NEW context was created after the
first), summed decision counts, last-step size/utilization/coverage
(what the final reasoning step actually read), `isolation_ok` over
every step, and a per-step `steps` list. An empty sequence returns
zero counts — never an error, never a fabricated step.

## 13. Worked example (§50)

From the real offline demonstration (`python
agentrun_m15_context_demo.py`, scope `ADOP@cccccccccccc`, real
engineering-graph artifact, real `read_file` over `README.md`):

**Seven candidates are offered** (inventory, §33): the task, one
repository scope fact, three memories (fresh @ current commit, stale
@ earlier commit, unknown commit), and two graph candidates (kind
discovery summary + one bounded neighborhood query).

**Step 001 selects five, excludes two** (snapshot #1):

```text
* cand_task           -> SELECTED | TASK_DIRECT
* cand_repo           -> SELECTED | REPOSITORY_SCOPE
* cand_mem_fresh      -> SELECTED | VERIFIED_MEMORY | freshness fresh
* cand_mem_stale      -> excluded | STALE_MEMORY | freshness stale
* cand_mem_unknown    -> excluded | UNKNOWN_FRESHNESS | freshness unknown
* cand_graph_summary  -> SELECTED | GRAPH_NEIGHBOR
* cand_graph_neigh…   -> SELECTED | GRAPH_NEIGHBOR
budget: 5/64 items, 762/6000 characters
```

**The model step** receives exactly `render_context_text(package)` —
1774 characters, header + five items in section order, each line
annotated with its reason.

**The tool step** reads `README.md` (lines 1-1125). **Step 002**
re-selects over nine candidates: two new ones arrive
(`cand_tool_read` → `RECENT_TOOL_RESULT`,
`cand_evidence_read` → `CURRENT_EVIDENCE` with an `EvidenceRef`),
seven are selected, and the diff is:

```text
added: cand_evidence_read, cand_tool_read
removed: (none)
```

**The tight budget** (max 400 chars) selects 3 items / 316 chars and
records `BUDGET_EXCEEDED` for four candidates; the impossible budget
(5 chars) raises `CONTEXT_BUDGET_EXCEEDED` — caught, printed, never a
silent empty package.

**The full agent run** — the real loop, two boundaries:
`memory.retrieve — 3 retrieved`, then `context.select step 001 — 4
candidates → 2 selected / 2 excluded` (task + fresh memory in; stale
+ unknown out), `context.render — 930 characters`, one `read_file`,
then the observed tool result is adapted and **re-selected**:
`context.select step 002 — 6 candidates → 4 selected / 2 excluded`
(`RECENT_TOOL_RESULT` + `CURRENT_EVIDENCE` enter),
`context.render — 1509 characters`, one claim → `SUPPORTED`, memory
formation stores the validated conclusion. Snapshots
`context_001.json` and `context_002.json` persist under the run's
id; `evaluate_context_run` reports
`context_evolution_count = 1`.

**The reports** (§47) close the story:

```text
* Candidates considered: 7 → 9
* Selected: 5 → 7
* Stale excluded: 1
* Unknown-freshness excluded: 1
* Isolation: PASS
```

## 14. Concept table (§51)

| Concept | Type | One-line meaning |
| --- | --- | --- |
| `TaskContext` | anchor | the step's question + scope + tags every selection is made *for* |
| `ContextCandidate` | input | one source's offer; considered, never self-selected |
| `ContextSelectionDecision` | audit row | exactly one per candidate: selected/excluded + stable reason |
| `ContextBudget` | ceiling | max items/chars **with** observed usage attached |
| `ContextItem` | unit | one selected candidate in presentation order (`ctxitem_<NNN>`) |
| `ContextPackage` | output | what the model sees for one step; usage invariant enforced |
| `ContextSelection` | verdict | package + full decision audit of one `select` |
| `ContextSnapshot` | record | one selection bound to run/step, persisted as JSON |
| `ContextEngine` | authority | deterministic selection API; optional repository binding |
| reason vocabulary | contract | 8 selection reasons + 7 exclusion reasons (stable strings) |
| authority hierarchy | contract | task 600 > evidence 500 > repository 450 > graph_fact 400 > memory 300 > tool_result 200 > agent_state 150 > inference 100 |
| freshness (tri-state) | contract | `fresh` / `stale` / `unknown` — unknown never silently fresh |
| `render_context_text` | projection | the single rendering that becomes model input |
| `evaluate_context` | metrics | deterministic §47 metric set over a selection/snapshot |
| `context.select` / `context.render` | trace events | what was selected / what was rendered (§27) |
| snapshot path | artifact | `runtime/context/<run_id>/context_<step_id>.json` |

## 15. Demonstration — one story, nine phases (§57)

`python agentrun_m15_context_demo.py` — **offline by default**
(deterministic: scripted model responses, real tool dispatcher, real
graph artifact, real engine — no API key, no network); `--live`
switches the same story to a real model (requires `OPENAI_API_KEY`
and the M6 Architecture Context).

There is deliberately **no `--offline` flag**: offline is the
default behavior of the plain invocation, and `--live` is the only
mode switch (same convention as the M9/M10/M13/M14 demos). The
documented command *is* `python agentrun_m15_context_demo.py`;
`python agentrun_m15_context_demo.py --live` for the real model.

| Phase | § | Proves |
| --- | --- | --- |
| 1 inventory | §33 | every source offers a candidate (7 across 4 source types) |
| 2 snapshot #1 | §34 | `=== STEP 1 ===`: policy decides; JSON persisted |
| 3 model step | §35 | the model receives exactly the rendered package |
| 4 tool step | §36 | real `read_file` over a real file |
| 5 snapshot #2 | §37 | `=== STEP 2 ===`: new candidates enter; diff +2/−0 |
| 6 memory interaction | §38 | fresh SELECTED, stale excluded, unknown excluded |
| 7 budget | §39 | tight budget blocks within limits; impossible budget raises |
| 8 planner + full agent run | §40 | **the real Agent path, end to end**: planner iteration → validated plan (`=== PLANNER ===`, `plan validated: status completed`) → retrieve → select (001) → tool → re-select (002) → claims → memory formation → per-step snapshots + context events, printed as `=== STEP 1 ===` / `=== STEP 2 ===` with trigger, counts, rendered context, and snapshot path per boundary, plus `snapshot ownership: … belong to run <run_id>` |
| 9 evaluation | §41/§47 | deterministic reports, isolation PASS |

Phase 8 uses the injected engine through the ordinary
`agent.run_planned(...)` call — the demonstration never re-implements
the lifecycle; its `tool_candidates` helper composes the shared
`candidates_from_tool_result` adapter (stable ids `cand_tool_read` /
`cand_evidence_read` for fixture readability). The offline model
supplies **five** responses in consumption order: the §35 model
step, the planner payload, the P1 tool call, P1's step result, and
the final claim payload — so even offline the run is plan-driven
(the plan is printed by `print_plan(trace)`, read from recorded
trace state). Its condition requires the plan (`status completed`,
at least two steps), the per-step lifecycle: steps `001 → 002`, one
persisted snapshot per boundary whose `run_id` equals the trace's
`run_id`, and no `last_context_error`.

Exit codes: `0` every condition held; `1` a demonstration condition
failed; `6` configuration/execution failure (missing key in
`--live`, bad graph path, `LLMError`, `AgentError`, …). Snapshots
land under `runtime/context/m15_demo/` (fixed run id — deterministic
artifact paths) and the agent run's per-step snapshots under
`runtime/context/<run_id>/context_001.json|context_002.json`; the
demonstration keeps its **own** memory database
(`runtime/agent_memory/m15_demo.db`, `--db` to override), never the
canonical store. Options: `--question`, `--model`, `--db`,
`--snapshots`, `--graph`, `--traces`.

## 16. Testing (§60)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/context_engine/ -q
# → 184 passed
# full suite: 1421 passed, 1 skipped
# (M15 = 184 tests: 151 engine/agent/trace tests + 33 demo tests;
#  the 2 skips are environment-conditional and outside M15 —
#  Graphviz `dot` not installed here + the opt-in OpenAI smoke test.
#  The frozen M14 baseline remains 1237 passed / 1 skipped as
#  recorded in docs/m14 — that environment had `dot` installed,
#  which accounts for the one-test delta.)
```

| File | Tests | Covers |
| --- | --- | --- |
| `tests/context_engine/test_context_contract.py` | 66 | candidate/task/budget/package/snapshot field-by-field validation, reason vocabularies, authority ranks, section order, id shapes, `to_dict` projections, source scan (no llm/sqlite3 in the package), **§45 negatives** (stale never current, unknown never fresh, budget invariant) |
| `tests/context_engine/test_context_selection.py` | 22 | every pipeline stage: scope (neutral/pinned/mismatch), freshness tri-state, relevance (disjoint/neutral), deterministic sort, exact-identity dedup, greedy budget fill incl. skip-and-continue, `CONTEXT_BUDGET_EXCEEDED`, presentation order, **§45 negatives** (duplicate items, nondeterministic selection, silent overflow) |
| `tests/context_engine/test_context_isolation.py` | 16 | repository isolation: foreign task refused by engine, foreign snapshot refused, cross-repo candidates excluded as `OUT_OF_SCOPE` yet observable, **§45 negative** (repo A's context into repo B) |
| `tests/context_engine/test_context_snapshots.py` | 13 | persist paths, file == `to_dict()`, step-id binding, overwrite semantics, `CONTEXT_SNAPSHOT_FAILURE`, runtime-root default |
| `tests/context_engine/test_context_rendering.py` | 10 | single-render contract, section order in output, reason lines, budget header, empty-package render, `CONTEXT_RENDER_FAILURE` |
| `tests/context_engine/test_context_trace.py` | 8 | `context_events` field shape, clipping, `to_dict` omission when empty, `## Context` markdown (after Memory, before Iterations), select/render event payloads, error-event shape |
| `tests/context_engine/test_context_agent_integration.py` | 16 | no-engine byte-identical M14 behavior, injected-engine select/render events, **per-step re-selection after a tool batch — the six §8 proofs from one real execution (§14/§8)**: tool executed → result in trace + working history → tool candidate selected in step 002 → 4 ordered events with triggers → request #2 carries context #2 → next model invocation received it, **snapshot ↔ run ownership (§7)** (`run_id` equality on disk and in memory), **meaningful evolution (§15)**: selected-ids change with recorded reasons + tight-budget opposite (new candidates offered → excluded `LOW_PRIORITY`/`BUDGET_EXCEEDED`), no-tool run selects once, available≠selected counts, failure → recorded error per step + no context message (never a fallback), `last_context_error`, run/run_planned parity, **tool result ≠ memory (§25)**, **engine ≠ working memory (§25)**, **full offline planned investigation plan → tool → context → claims → validation → memory formation (§16)**, **cross-run determinism with uuid candidate ids (§24)** |
| `tests/context_engine/test_m15_context_demo.py` | 33 | offline demo: all nine phases, whole-story `run_demo`, cross-run determinism, per-step agent phase (`001 → 002`, snapshots, STEP banners), **planner + ReAct in one linked run (§40/§7)** (planner iteration first with no tool results, validated `P1 → P2` plan recorded, snapshots carry the trace's `run_id`, `=== PLANNER ===` / `plan validated` / `plan executed` output), `main` offline success **without** `OPENAI_API_KEY`, `--live` without key → 6, bad graph → 6, snapshot JSON artifacts, no database written for context, composition-root source scan |

LLM-free throughout (scripted clients); the only real I/O exercised
is the local `read_file` over the repository and the JSON snapshot
writes (temporary paths). Full regression: `pytest tests/`,
`compileall`, `pyflakes` on touched files, harnesses 6/6, 8/8, 10/10,
13/13.

## 17. Manual analysis guide (§59)

How to audit context by hand — no special tooling:

1. **What did the model actually receive?** Open the trace
   (`runtime/agent_runs/<run_id>/trace.md`), read `## Context`: the
   `context.select` line gives counts, budget, and reason map;
   `context.render` gives the rendered character count. The JSON
   `trace.json` carries the full per-candidate `decisions` array.
2. **Why was this item in (or out)?** Find the candidate's row in
   `selection_decisions` of the snapshot
   (`runtime/context/<run_id>/context_<step>.json`): `selected` +
   `reason` are stable strings; `duplicate_of` names the surviving
   duplicate; `budget_cost` shows what it would have consumed.
   Cross-check: every id in `excluded_candidates` appears exactly
   once with `selected: false`.
3. **Was anything dropped silently?** `len(selection_decisions)` must
   equal the number of candidates offered (the trace event's
   `candidate_count` for agent runs). Selected items must equal the
   budget's `selected_items`, and `estimated_characters` must equal
   the sum of item content lengths — both enforced at package
   construction, so a mismatch means a corrupted file.
4. **Was isolation respected?** Every selected decision must have
   `scope_match: true`; any selected row whose
   `repository_scope` differs from the task's is a bug. The §47
   report's `Isolation: PASS` is the same check over the whole
   selection.
5. **Freshness honesty:** a memory row with `freshness: "stale"` or
   `"unknown"` must have `selected: false` with reason
   `STALE_MEMORY` / `UNKNOWN_FRESHNESS`. "Available" (memory
   retrieve count) and "selected" (memory items in the package) are
   different numbers — compare them directly.
6. **Budget honesty:** `budget.estimated_characters ≤
   budget.max_characters` and `selected_items ≤ max_items` on every
   package; `BUDGET_EXCEEDED` rows in the decisions explain the gap
   between considered and selected under tight budgets.
7. **Re-run and diff:** the offline demonstration is deterministic —
   run it twice and compare the `by_reason` maps and selected item
   lists (ids/timestamps may differ; decisions and order must not).

## 18. Non-goals (§63) and boundary

- **No vector database, no embeddings, no RAG** — selection is
  deterministic filtering/sorting over explicit candidates; semantic
  similarity is a different mechanism with different failure modes.
- **No LLM selector** — the model never chooses or prunes its own
  context; the policy is code, and its decisions are inspectable
  strings.
- **No second trace / second framework** — context events live on the
  existing `AgentTrace`; snapshots are JSON files, not a store.
- **No context database** (§43) — memory keeps the only SQLite file;
  context is files under `runtime/context/`.
- **No automatic candidate discovery** — a composition root lists its
  candidates; sources that were not asked contribute nothing (and
  nothing is invented on their behalf).
- **Boundary and philosophy:**
  - **Context ≠ memory ≠ evidence.** Evidence is what this run
    inspected (M13); memory is what a previous run concluded (M14);
    context is the *selection decision* about what this step reads.
  - **Available ≠ selected** (§56) — retrieval counts and package
    contents are separate, separately recorded facts.
  - **The application owns selection; the model only receives the
    result.** Every decision is a deterministic string a test can
    assert and a human can read.
  - **Failures are loud, and never change meaning** — a failed
    selection records an error and produces *no* context message,
    never a different one (§55).
  - **One trace, one loop, one store** — M8.8/M8.9 exclusivity, the
    M10 plan-first invariant, the M13 validator, and the M14 memory
    boundary are untouched; M15 adds a selection boundary, not an
    architecture.

## 19. Definition of done (§62) and learning progression

- [x] `src/repo_intelligence/context_engine/` — contracts, seven-stage
      policy, engine with isolation guard, single renderer,
      deterministic evaluation, adapters; no LLM/`sqlite3` imports
      (source-scanned).
- [x] Fixed reason vocabulary (8 + 7), authority hierarchy (§14),
      freshness tri-state (§24), exact-identity dedup (§22), documented
      presentation order (§23).
- [x] Hard budgets with recorded usage (§12) — invariant enforced at
      package construction; nothing-fits → `CONTEXT_BUDGET_EXCEEDED`,
      never silent overflow or empty selection.
- [x] Snapshots persisted as JSON at
      `runtime/context/<run_id>/context_<step_id>.json` (§42); file
      == `to_dict()`; no context database.
- [x] Agent integration via injected `context_engine` only — and
      **per reasoning boundary inside the loop** (§10): step `001`
      at the execution boundary (feeding the planner request in a
      planned run), re-selection after every executed
      tool batch (`trigger: tool_result`), one snapshot per step,
      context message appended chronologically; snapshot `run_id`
      == trace `run_id` (one run owns its snapshots); byte-identical
      M14 behavior without the engine; failure recorded per step, no
      fallback context (§55); tool results → candidates, never
      memory (§7); no new information → no re-selection.
- [x] Trace: `context_events` + `## Context` (after Memory);
      `memory.retrieve` vs `context.select` separation (§56).
- [x] `evaluate_context` / `render_evaluation_report` — §47 report,
      `isolation_ok` assertion, `evaluate_context_run` run-level
      metrics incl. `context_evolution_count` — no LLM judge (§46).
- [x] `agentrun_m15_context_demo.py` — nine phases, offline default
      (no key; no `--offline` flag — offline IS the default),
      `--live` optional, exit 0/1/6, real graph artifact +
      real `read_file`, composition-root candidates via the shared
      tool adapter, **planner → validated plan → planned ReAct with
      per-step context (phase 8 prints `=== PLANNER ===`,
      `plan validated`, `plan executed: P1 -> P2`)**, per-step
      STEP 1/STEP 2 sections in the real agent phase, snapshots
      under `runtime/context/m15_demo/` +
      `runtime/context/<run_id>/`.
- [x] `tests/context_engine/` — 184 tests (151 + 33 demo), LLM-free,
      §45 negative cases covered, mandatory per-step critical tests
      (§14/§15/§16/§24/§25), planner↔context run linkage (§7/§40);
      full suite 1421 passed / 1 skipped (environment-conditional
      skips outside M15; M14 frozen baseline remains 1237/1);
      harnesses 6/6, 8/8, 10/10, 13/13.
- [x] **Runtime hygiene** — demo database, snapshots, and traces
      written only under gitignored `runtime/` (tests use temporary
      paths).
- [x] Docs: this file, README narrative + milestones row, plan doc
      §40 — counts verified against the actual suite run.

```text
M14  validated conclusions → SQLite → next run retrieves     (across runs)
M15  what enters THIS step → explicit policy → package →
      snapshot → trace      (within one step, every step)
M16+ consolidation / similarity / learned selection (deferred, §17)
```

The stack stays honest at every step: M14 proved a conclusion
survives and must still be re-checked; M15 proves the *input* to every
reasoning step is a decision — explicit, budgeted, scoped, fresh, and
auditable — rather than an accident of accumulation.
