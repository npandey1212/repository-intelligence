# AGENTS.md — repository-intelligence (opencode session context)

**Read this file first in every new session.** It is the single source of
truth for environment, conventions, verification commands, and open items.
`SESSION_CONTEXT.md` (older) is superseded by this file where they disagree.

**Last updated:** 2026-10-17 (H1–H4 context-overflow boundary work:
every model call in `agent/agent.py` (loop + planner + finalization)
now passes ONE preflight seam — `agent/request_budget.py` measures the
COMPLETE request (messages + tool args + tool definitions +
structured-output schema) into a privacy-safe `AgentIteration.
request_budget` record (H1), and `enforce_request_budget` hard-enforces
a token budget BEFORE any provider call: over-budget tool
observations clip to deterministic valid-JSON envelopes, protected
content over budget is rejected (never sent, never retried unchanged)
with typed `CONTEXT_BUDGET_EXCEEDED`/`FINALIZATION_FAILED` failures
(H2); `tools/graph_projection.py` compacts canonical
`query_engineering_graph` payloads to `model_graph_compact.v1`
(deduplicated evidence table, attributes dropped, deepest-node
truncation, ≤24,000 B) wired into `tool_registry()` +
`live_bindings` (H3); run-wide lifecycle regression tests (H4).
35 + 6 agent tests + 22 projection tests; default budget = 95,232
tokens (128k observed window − 16,384 output − 16,384 margin),
deliberately NOT coupled to `max_context_characters`.
**Last updated:** 2026-10-18 (context-overflow Step 1 — the SAME
`request_budget` preflight is now shared by every non-Agent provider
call: `enforce_request_budget_or_raise` (new, in `agent/request_budget.py`)
raises `RequestBudgetExceededError` instead of returning a rejected
record. Gated: `repository_query._build_plan` (the production planner
bypass), `langgraph/planner.py::get_plan`, `llm/__main__.py`. Planner
requests carry no tool observations, so they are ACCEPTED byte-identical
or REJECTED — never truncated. 11 tests in
`tests/agents/test_planner_request_budget.py` + 1 in
`tests/llm/test_llm_cli.py`.)
**Last updated:** 2026-10-18 (bounded tool outputs — Steps 1A–1D:
`tools/protocol.py` now owns a shared response ceiling —
`MAX_TOOL_RESPONSE_CHARS = 24_000` chars on ONE serialized
`ToolResult` + `response_chars()` + line-aligned `bound_result_text()`
(whole-line first, character prefix only when no line fits, measured
against the FINAL envelope so the bound is a guarantee). Tool 1
`get_architecture_context` gained optional `{section, max_chars}`
(case-insensitive exact heading, `SECTION_NOT_FOUND`, CLI
`--section/--max-chars`); Tool 3 `read_file` bounds every response with
`truncated/returned_chars/total_chars/char_limit/next_line_start/
partial_line/note` — delivered ranges only, partial-line reads report an
EMPTY range (`line_end = line_start - 1`) so evidence coverage cannot
over-claim; Tool 2 `search_repository` drops WHOLE trailing results with
`omitted_results`/`note` (head of the ranked list preserved, `QUERY_TOO_LONG`
for an envelope-only overflow). Step 1D pins the three boundaries as
separate knobs (tools 24,000 chars / graph projection 24,000 bytes /
request budget 95,232 tokens) in
`tests/tools/test_response_ceiling_boundaries.py`; `graph_projection.py`
and `request_budget.py` untouched. 43 new tests.)
**Last updated:** 2026-10-18 (Tool 1 targeted default — `{}` now returns
a compact, FENCE-AWARE Markdown heading index (`_render_index`: the
document's real heading lines in document order + a fixed
"next: {"section": …}" retrieval hint; `#`/`##`/`###` levels intact;
`(no Markdown headings found)` placeholder), never the whole document;
`{"section": …}` is the retrieval path — `_heading_blocks` is now
per-line CommonMark fence-aware (`_FENCE_OPEN_RE`/`_FENCE_CLOSE_RE`:
backtick/tilde, close ≥ open, unclosed → EOF) so fenced `#`/`##` lines
can neither index, select, nor terminate a section; block-end rule,
11-key shape, ceiling, provenance, and all error codes unchanged
(`total_chars` = the selection). `tool_definitions` description, CLI
help, tool doc (§12.3/12.4/12.5), README Tool 1 updated; ReAct
scripted orient step now requests `{"section": "Mechanism"}` (its
derivation still feeds from the observation). +6 tests, gac 40 → 46.)
**Last updated:** 2026-10-19 (LangGraph end-to-end — production opt-in +
primary notebook: `run_repository_query(..., orchestration="langgraph")`
+ CLI `--orchestration langgraph` route the task loop through
`run_with_langgraph` (roles preassigned via the SAME `assign_roles`;
planner off-graph behind the H1 preflight; `_assemble` unchanged);
default stays `legacy`; missing graph extra → typed
`ORCHESTRATION_UNAVAILABLE` structured failure, **no silent fallback**;
`Orchestrator.execute` still never on the graph path (test-pinned at
the entry level). One primary notebook
`notebooks/langgraph_end_to_end_workflow.ipynb` (10 sections, 17
offline code cells; verified by sequential single-process exec —
Jupyter not installed here). 11 entry tests + 4 notebook tests;
offline field-by-field parity with legacy (run ids disjoint).)
**Last updated:** 2026-10-19 (one-investigator LangGraph smoke test —
restricted orchestration flow, zero provider calls by default:
`notebooks/langgraph_one_investigator_smoke_test.ipynb` (11 sections,
9 code cells) runs fixture plan `["P1"]` → QueryRouter selects Plan 1
→ ONE Investigator executes via `OneTaskExecutionPolicy`
(`tests/agents/one_investigator_smoke_policy.py` — policy + Analyst
double + `ProviderCallGuard(limit=0)` class-patch + live
`CallLimitedClient(max_calls=1, max_tokens=512)`) → terminate
`INCOMPLETE/NO_ANALYSIS`, no Analyst, no retry (`max_retries=0`),
attempts 1. Live cell gated behind `LIVE_MODEL_TEST = False` +
`CONFIRM_LIVE_CALLS = False` (one REAL planner call + offline
investigator, unexercised — no `OPENAI_API_KEY`). Contract additions:
`LLMRequest.max_tokens` (None default, translated only when set) +
`OpenAILLMClient(max_retries=None)` (SDK default preserved when
unset). 13 smoke tests + 3 OpenAI-client tests; `test_models`
defaults test updated for the additive field.)
**Last updated:** 2026-10-19 (smoke-test safety adjustment — the live
Planner experiment now has a REALISTIC output allowance (2,048) and a
bounded, EXPERIMENT-SPECIFIC estimated input ceiling (8,000) enforced
locally twice via the SAME H1 machinery: a pre-call probe
(`build_planning_request` + `enforce_request_budget_or_raise` with a
tighter `RequestBudgetPolicy`) and `CallLimitedClient(input_budget=)`
at the client boundary; oversized requests REJECTED before the
provider (reason + est-vs-ceiling displayed), output cap verified to
reach the provider payload's `max_tokens`, a failed call consumes the
one-call allowance, a budget rejection does not; global H1–H4
defaults UNCHANGED (95,232-token default still test-pinned). +6
focused tests, smoke file 13 → 19.)
**Last updated:** 2026-10-19 (Generic ReAct context-budget safety —
REUSABLE, graph-agnostic controls in
`src/repo_intelligence/react_safety.py` (package ROOT, next to the
`tool_dispatcher.py` agent↔tools bridge — `agent/` may describe
tools but never import the `tools` layer, both layering guards stay
untouched): `BudgetedLLMClient` wraps ANY `LLMClient` and routes
EVERY outgoing request through the existing
`enforce_request_budget_or_raise` (complete request = messages +
tool defs + response schema; oversized → `RequestBudgetExceededError`
locally, ZERO provider calls, rejection never consumes the
provider-call allowance while a provider failure does — tracked via
`calls_reached_provider`); `bound_observation` bounds one tool
observation (default = the tool-protocol `MAX_TOOL_RESPONSE_CHARS`,
real import, no duplicate constant) with line-aligned truncation
metadata + continuation hint BEFORE it joins history;
`ReActLoopGuard` enforces iteration / tool-call / model-call /
cumulative-input limits with terminal statuses
(`ITERATION_LIMIT_REACHED`/`TOOL_CALL_LIMIT_REACHED`/
`MODEL_CALL_LIMIT_REACHED`/`CUMULATIVE_BUDGET_EXCEEDED`/
`REQUEST_BUDGET_EXCEEDED`) and stop-on-exhaustion (all actions
refused once terminal); `restrict_history` drops OLDEST
non-protected units per-REQUEST only (system/developer kept, tool
call/result pairs atomic, caller transcript NEVER mutated);
`ReActSafetyConfig.request_budget_policy()` returns the untouched
global default or a tighter LOCAL policy (no global default change).
Smoke notebook grew a §11 offline demo (12 sections / 10 code cells;
8 demos D1–D8 with fake clients/tools, `PROVIDER_GUARD.calls == 0`)
and §12 became "Results, limitations, and next steps" (Planner-only
limits are not the fix; estimate honesty; no monetary cap). +17
  `tests/test_react_safety.py` + 2 smoke tests (19 → 21).)
**Last updated:** 2026-10-19 (ReAct safety integrated into the REAL
application path + workflow validation — call-path audit: all six
src `.complete(` sites are gated (agent.py loop/planner/finalization
via `_gate_model_request`, `repository_query._build_plan`,
`langgraph/planner.py`, `llm/__main__.py`); graph nodes make NO
direct model calls (dispatch → executor → participants → the Agent
loop), so H1/H2 + `max_tool_calls` were REUSED, never re-wrapped
(a reject-only BudgetedLLMClient around the shared client would
pre-empt H2's reduce path — deliberately not done). The one real
gap — tool observations entered the conversation unbounded at the
append seam — now passes through `bound_observation`
(`agent/agent.py::_execute_loop`, LAZY import so agent/ textual
imports stay tool-free and both layering guards stay green):
identity byte-identical for tool-protocol-compliant results,
line-aligned excerpt + continuation note otherwise, trace evidence
copy untouched. Cumulative-input enforcement is NOT supported by the
Agent and was NOT added (new budget feature = out of scope;
`ReActLoopGuard` carries it for new loops). +5 tests:
`tests/agent/test_react_integration.py` (4 — oversized observation
bounded before the next real request with trace evidence preserved,
independent per-request budget records across 3 model calls,
byte-identical compliant pass-through, EXECUTION_LIMIT_REACHED
terminal intact) + graph-workflow budget-record test in
`test_repository_query_langgraph.py` (every iteration of all 3
agent traces gated @ 95,232). Notebook §11/§12 amended (integration
paragraph; "a live Planner call does NOT validate the workflow"
limitation); gates intact. Full suite 2524 → 2529.)
**Last updated:** 2026-10-19 (Visualizer id-copy prompt hardening —
entry Q, §7 open item addressed at the PROMPT level: RULES gained a
copy-verbatim bullet + OUT gained an `IDs are copied verbatim, never
shortened` paragraph with right/wrong few-shot examples (node id +
edge endpoint; hypothetical paths, repository-agnosticism checks
intact). Version stays `V1` DELIBERATELY — the protected demo
asserts `VISUALIZATION_PROMPT_VERSION == "V1"` in its Phase-1 ok
condition, so an in-place content amendment preserves the pin (a
V1→V2 bump would break the protected file). +1 test; 2529 → 2530.)
**Last updated:** 2026-10-19 (LangGraph create_agent POC —
entry R, new optional package `langgraph_poc/` (middleware.py:
RequestBudget/ExecutionLimit/ObservationBounding middleware adapting
the EXISTING `react_safety` controls to real langchain
`AgentMiddleware` hooks, budget outermost; tools.py: project tool
bindings as StructuredTools; graph.py: StateGraph
planner→router→select→create_agent investigator→receive→final with
explicit failure routing, POC_PLAN_STEP_IDS P1-P4, roles via the
SAME assign_roles with P1-P3=INVESTIGATOR/P4=ANALYST). Guard policy:
langchain/langchain_core confined to langgraph_poc/ (M21-style
comment, provider-adapter langchain_openai added to
PROVIDER_IMPORT_RE). venv openai 1.99.1 → 3.35.1 (langchain-openai
requirement; tests/llm green on both). +25 tests (23 POC + 2 guard);
2530 → 2555.)
**Status:** All work green. Full suite 2555 passed / 2 skipped. Offline CLI
exit 0 on BOTH backends (`--orchestration langgraph` banner + COMPLETED);
offline LangGraph workflow end-to-end validated (planner → dispatched
Investigators → Analyst → SUCCESS + per-iteration budget records in
every agent trace); protected viz demo exit 0 with `scope condition:
yes`. POC demo `agentrun_langgraph_poc.py` exit 0 offline (budget
records accepted ×2, SUCCESS, P1 completed). One open item (visualizer
node-id compliance — prompt hardened, STILL LIVE-UNVERIFIED). Live
`--live` runs blocked in this environment (no `OPENAI_API_KEY`), POC
live path implemented + gated but unexercised.

---

## 1. What this project is

A deterministic, evidence-backed multi-agent system that investigates a
pinned git repository snapshot and produces typed architecture findings and
diagrams. The end-to-end offline flow (no network, scripted models):

1. **S1 Architecture Investigator** — components/relationships → typed
   `ArchitectureReport` + projected `InvestigationReport`.
2. **S2 Flow / Dependency Investigator** — path tracing → `FlowDependencyReport`
   + projected `InvestigationReport`.
3. **S3 Architecture Visualization (Analyst)** — both reports in, one typed
   `DiagramSpec` out (Mermaid/DOT via the existing renderer).
4. **Deterministic validation** — M16 collaboration verdict + six diagram
   rules (incl. `scope_validity`: every node/edge id must already exist in
   the supplied reports; ids are never invented or shortened).

Entry points:

- Offline CLI (default mode — offline unless `--live`):
  `venv312/bin/python agentrun_repository_query.py "<question>"`
  Flags: `--live`, `--verbose`, `--repository`, `--no-artifacts`,
  `--no-diagram`, `--orchestration {legacy,langgraph}` (default
  `legacy`; langgraph = M21 graph task loop, opt-in, loud failure if
  the optional extra is missing). There is **no** `--offline` flag
  (offline is the default).
- Exit codes: `0` success; `1` investigation incomplete; `3` validation
  failed (e.g. diagram scope_validity).
- Milestone demos `agentrun_m9..m17_*.py`, `agentrun_react_test.py`, etc.

Docs per milestone live in `docs/` (`m1`–`m18`, plus investigator/agent
specs). Architecture spec: `docs/architecture-context.md`.

---

## 2. Environment (CRITICAL — two venvs, do not confuse)

| Env | Python | Purpose | Status |
|---|---|---|---|
| `venv312/` | 3.12.3 (`/usr/bin/python3.12`) | **THIS project** — run everything with it | Active; pydantic 2.10.6 |
| `.venv/` | 3.7.10 (pyenv) | **OTHER projects on this machine** — required to stay untouched | Frozen; pydantic 1.10.26 |

Rules:

- **Always use `venv312/bin/python`** for tests, demos, and CLI runs in this
  repo. Never install into or modify `.venv/` — the user's other projects
  depend on system/pyenv Python 3.7 remaining untouched.
- `venv312` already has: pydantic 2.10.6, openai 1.99.1, jsonschema,
  networkx, typing_extensions, PyYAML, toml, pytest 8.4.2, numpy, faiss-cpu,
  scikit-learn, scipy, requests (recreated 2026-10-14 from
  `requirements.txt` pins + AGENTS.md §2 set; `sentence-transformers` was
  NOT reinstalled — suite unaffected, same 2 skips; `pip install
  sentence-transformers` if M18 embedding tests need it — pulls torch).
- The package is installed editable (`pip install -e .`) into `venv312`.
- If `venv312/` is missing, recreate:
  `python3.12 -m venv venv312 && venv312/bin/pip install -e . pytest numpy faiss-cpu scikit-learn requests openai`
- **Partial-wipe repair (hit once on 2026-10-15):** `venv312/bin/` was
  deleted and some packages gutted (`pydantic_core`, `numpy`,
  `scipy.libs`). Fix WITHOUT recreating from scratch (site-packages
  survives): `/usr/bin/python3.12 -m venv venv312` (regenerates
  `bin/`), then if imports fail:
  `venv312/bin/python -m pip install --force-reinstall pydantic==2.10.6 numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1`
  (re-pin `typing_extensions==4.15.0` afterwards if upgraded), then
  `venv312/bin/python -m pip check` + a pytest run.
- `pyproject.toml`: `requires-python = ">=3.10"`, `pydantic>=2.10,<3`.
  `requirements.txt` pins the venv312 set.
- **`langgraph` (M21):** `langgraph==1.2.15` installed 2026-10-16 and
  declared — optional extra `graph = ["langgraph>=1.0"]` in
  `pyproject.toml`, pinned in `requirements.txt`. NOTE: the
  `venv312/bin/pip` script is MISSING — install with
  `venv312/bin/python -m pip install ...`. Core `repo_intelligence`
  imports without langgraph; only `orchestration/langgraph_runner.py`
  needs it (clear ImportError + test importorskip otherwise).
- **`langchain` (create_agent POC, 2026-10-19):** `langchain==1.4.6`,
  `langchain-core==1.6.11`, `langchain-openai==1.7.2` installed and
  declared — optional extra `poc` in `pyproject.toml`, pinned in
  `requirements.txt`. langchain-openai 1.7.2 requires `openai>=2.45`
  → venv openai upgraded 1.99.1 → **3.35.1** (plus transitive
  `tiktoken==0.14.0`, `regex==2026.9.29`); `tests/llm/` green on
  3.35.1 (82 passed/1 skipped) — the provider-neutral contract does
  not pin the SDK major. Core imports + `langgraph_runner` never
  need langchain; only `langgraph_poc/` does (importorskip in tests;
  import guards confine langchain/langchain_core to that package).
- No `OPENAI_API_KEY` in this environment — live runs impossible here (user
  ran one on Windows separately). Offline mode is the verification path.

---

## 3. Code conventions

- **Pydantic v2** (2.10.x). Migrated 2026-10-12. Use:
  - `@model_validator(mode="after")` → method takes `self`, returns `self`
    (replaces `@root_validator(skip_on_failure=True)`)
  - `@model_validator(mode="before")` + `@classmethod` → returns the dict
    (replaces `@root_validator(pre=True)`)
  - `@field_validator("x")` + `@classmethod` (replaces `@validator`)
  - `model_validate` (not `parse_obj`), `model_dump()` (not `.dict()`),
    `model_copy(update=...)` (not `.copy(update=...)`),
    `model_json_schema()` (not `.schema()`), `model_fields` (not `__fields__`),
    `model_construct` (not `construct`)
  - v2 revalidates nested model instances at **constructors**
    (`Model(...)`), but `model_validate(instance)` does not revalidate.
    `mode="before"` validators receive raw dicts for nested refs.
- Style: `from __future__ import annotations`; `.format()` (f-strings are
  fine on 3.12 but match surrounding code); lines ≤79 chars in `src/`.
- Every vocabulary term (relationship kinds, component types,
  classifications, confidence, statuses) is reused from the canonical
  homes in `engineering/models.py`, `multiagent/contracts.py`,
  `architecture_report.py` — never fork a new vocabulary.
- Evidence discipline (enforced by code, not just prompts): a finding may
  only cite refs that were actually inspected (`read_file`) in the trace —
  `agents/evidence_acceptance.py::enforce_inspected_evidence` strips
  search-only citations at `run_result` for both investigators.
- Deterministic validators own boundary correctness; LLM output is never
  trusted. Diagram elements must reuse exact ids from the reports.

---

## 4. Verification (run after ANY change)

```bash
# full suite — expected: 2555 passed, 2 skipped (~43s)
venv312/bin/python -m pytest tests/ -q

# LangGraph create_agent POC (entry R — middleware safety behavior;
# importorskip skips if the optional poc extra is absent) — 23 passed
venv312/bin/python -m pytest tests/test_langgraph_poc.py -q

# POC + shared ReAct controls together (expected: 40 passed)
venv312/bin/python -m pytest tests/test_react_safety.py \
    tests/test_langgraph_poc.py -q

# ReAct safety shared controls + REAL-loop integration + graph
# workflow budget records (expected: 17 + 4 + 12 = 33 passed)
venv312/bin/python -m pytest tests/test_react_safety.py \
    tests/agent/test_react_integration.py \
    tests/agents/test_repository_query_langgraph.py -q

# one-investigator smoke notebook + OpenAI client contract
# (expected: 42 passed, 1 skipped — 21 + 21 + 1 real-API skip)
venv312/bin/python -m pytest \
    tests/agents/test_langgraph_one_investigator_smoke.py \
    tests/llm/test_openai_client.py -q

# shared ReAct context-safety controls (expected: 17 passed)
venv312/bin/python -m pytest tests/test_react_safety.py -q

# agent layering guards (agent describes, never imports tools/graph/
# engineering) — must stay green alongside react_safety at the root
venv312/bin/python -m pytest tests/agent/test_agent.py \
    tests/agent/test_agent_graph_tool.py \
    tests/agent/test_tool_definitions.py -q

# bounded tool output + Tool 1 targeted default — expected: 285
venv312/bin/python -m pytest tests/tools/ tests/test_tool_dispatcher.py \
    tests/agent/test_tool_definitions.py -q

# the three context-overflow boundaries together — expected: 70
venv312/bin/python -m pytest tests/agent/test_request_budget.py \
    tests/agent/test_context_budget_lifecycle.py \
    tests/tools/test_graph_projection.py \
    tests/tools/test_response_ceiling_boundaries.py -q

# request-budget preflight + run lifecycle (H1/H2: 35, H4: 6 — 41)
venv312/bin/python -m pytest tests/agent/test_request_budget.py \
    tests/agent/test_context_budget_lifecycle.py -q

# planner / LangGraph-planner / tool-result preflight boundaries (11)
venv312/bin/python -m pytest \
    tests/agents/test_planner_request_budget.py -q

# model-facing graph projection (H3: 22)
venv312/bin/python -m pytest tests/tools/test_graph_projection.py -q

# orchestration package (M17 doubles + M19 + M20 + M21 + demo tests) — expected: 301
venv312/bin/python -m pytest tests/orchestration/ -q

# M21 LangGraph behavioral tests (expected: 34 passed; needs the optional
# langgraph install — importorskip guards skip the file if absent)
venv312/bin/python -m pytest tests/orchestration/test_langgraph_sequential.py -q

# budget-boundary wrapper tests (expected: 51 passed)
venv312/bin/python -m pytest tests/agents/test_repository_query.py -q

# LangGraph end-to-end: production seam + notebook contract + graph
# behaviour (expected: 11 + 4 + 34 = 49 passed; importorskip skips the
# graph files if the optional langgraph extra is absent)
venv312/bin/python -m pytest tests/agents/test_repository_query_langgraph.py \
    tests/agents/test_langgraph_notebook.py \
    tests/orchestration/test_langgraph_sequential.py -q

# evidence acceptance regression tests (9 tests)
venv312/bin/python -m pytest tests/agents/test_investigator_evidence_acceptance.py -q

# M19 coordinator tests + demonstration (expected: 103 passed; demo exit 0)
venv312/bin/python -m pytest tests/orchestration/test_m19_*.py -q
venv312/bin/python agentrun_m19_orchestrator_agents_demo.py

# M20 budget/validation/lifecycle tests (expected: 45 passed)
venv312/bin/python -m pytest tests/orchestration/test_m20_*.py -q

# agent per-call budget override (expected: 22 passed)
venv312/bin/python -m pytest tests/agent/test_agent_bounded.py -q

# offline CLI end-to-end — expected: exit 0, Status COMPLETED
# (add --orchestration langgraph for the same on the graph backend)
venv312/bin/python agentrun_repository_query.py "How do the agents, tools, and MCP servers interact?" --verbose
```

Baselines (do not regress):

- Full suite: **2555 passed, 2 skipped** (2344 M1–M21 chain, see §6
  history + 63 H1–H4 tests: 35 `test_request_budget.py`, 6
  `test_context_budget_lifecycle.py`, 22 `test_graph_projection.py`
  + 12 planner/CLI preflight tests: 11 `tests/agents/
  test_planner_request_budget.py`, 1 `tests/llm/test_llm_cli.py`
  + 43 bounded-tool-output tests: 40 get_architecture_context, 73
  read_file, 66 search_repository, 7 `test_response_ceiling_boundaries.py`
  — those four files against pre-step counts 19/64/60/0
  + 6 Tool-1 targeted-default tests (heading index, fence-aware
  parsing, oversized index — gac file 40 → 46)
  + 15 LangGraph end-to-end tests: 11 production-seam
  [tests/agents/test_repository_query_langgraph.py], 4 notebook
  contract [tests/agents/test_langgraph_notebook.py]
  + 25 create_agent POC tests [tests/test_langgraph_poc.py — 23
  graph+middleware behavioral tests + 2 import-guard tests in
  tests/llm/test_client_contract.py]
  + 24 one-investigator smoke tests: 21
  [tests/agents/test_langgraph_one_investigator_smoke.py] (13 base
  + 6 safety-adjustment: experiment ceiling, oversized rejection,
  output-payload reach, live-cell pins, offline roles, H1–H4 intact
  — 3 renumbered/extended, + 2 shared-§11 demo/offline pins), 3
  OpenAI-client contract
  [tests/llm/test_openai_client.py: max_tokens translation, unset,
  max_retries]
  + 17 shared ReAct safety tests [tests/test_react_safety.py]
  + 5 ReAct-application integration tests: 4
  [tests/agent/test_react_integration.py — observation bound on the
  real loop, independent per-request budget records, compliant
  identity pass-through, EXECUTION_LIMIT_REACHED intact] + 1
  graph-workflow budget-record test in
  [tests/agents/test_repository_query_langgraph.py]
  + 1 visualizer id-copy prompt test
  [tests/agents/test_architecture_visualization.py::
  test_prompt_forbids_shortened_ids_with_a_few_shot_example]
- Offline demo: **COMPLETED, exit 0**; collaboration + diagram validation
  passed; artifacts in `artifacts/architecture_visualization/`
  (`diagram_spec.json`, `architecture.mmd`, `run_summary.md`; SVG
  unavailable — renderer limitation).
- ~9k `PydanticDeprecatedSince20` warnings are from test files still using
  deprecated v2 aliases (`parse_obj`, `.schema()` in ~17 test files) —
  cosmetic; optional cleanup.

---

## 5. Key files map

| File / dir | Role |
|---|---|
| `agentrun_repository_query.py` | Main CLI (offline default, `--live` opt-in; `--orchestration {legacy,langgraph}` selects the task-loop backend, default `legacy`) |
| `src/repo_intelligence/repository_query.py` | Offline orchestrated pipeline; LangGraph opt-in seam `ORCHESTRATION_*` constants + `_run_with_langgraph` (lazy graph import, `assign_roles` preassignment, `ORCHESTRATION_UNAVAILABLE` failure — never a fallback) |
| `src/repo_intelligence/agents/architecture_investigator.py` | S1 specialist (`run_result` wires evidence acceptance) |
| `src/repo_intelligence/agents/flow_dependency_investigator.py` | S2 specialist (same wiring) |
| `src/repo_intelligence/agents/architecture_visualization.py` | S3 analyst |
| `src/repo_intelligence/agents/evidence_acceptance.py` | Inspected-evidence gate (removal-only) |
| `src/repo_intelligence/agents/diagram_validation.py` | Six deterministic diagram rules |
| `src/repo_intelligence/agents/models/` | Typed contracts: `architecture_report.py`, `flow_dependency_report.py`, `architecture_visualization.py` |
| `src/repo_intelligence/agents/prompts/` | Prompt texts + version constants (V1/V2) |
| `src/repo_intelligence/multiagent/contracts.py` | M16 message/payload contracts (`AgentMessage`, `InvestigationReport`, …) |
| `src/repo_intelligence/orchestration/contracts.py` | M19 task boundary: `AgentTaskRequest`/`AgentTaskResult`, `TaskFailure` taxonomy, `TaskUsage`; M20: `max_tool_calls` validation, `_validate_task_usage`/`_validate_task_failure` runtime rules, `OrchestrationConfig.max_task_tool_calls` |
| `src/repo_intelligence/orchestration/executor.py` | `AgentExecutor` port + `AgentTaskExecutor` adapter + `ParticipantRegistry`; M20: per-attempt kwarg pass-through, counts EXECUTED `tool_results` |
| `src/repo_intelligence/orchestration/scheduling.py` | Pure chain dependencies + `ready_tasks` |
| `src/repo_intelligence/orchestration/assembly.py` | `merge_investigation_reports` (relocated) + `RunAssembler` |
| `src/repo_intelligence/orchestration/runner.py` | `Orchestrator` (dispatches tasks through the executor; `TaskRecord` commits); M20: retry-decision run gate, budget event `limit`/`observed`, defensive validator wrap |
| `src/repo_intelligence/orchestration/langgraph_runner.py` | M21: compiled StateGraph task loop over unchanged M19/M20 contracts (`initialize/select/execute/commit/retry/assemble`); `run_with_langgraph`; NEVER calls `Orchestrator.execute`; clear ImportError without the optional langgraph install |
| `src/repo_intelligence/langgraph_poc/` | Entry R POC package (optional `poc` extra): `middleware.py` (RequestBudget/ExecutionLimit/ObservationBounding `AgentMiddleware`s over the EXISTING `react_safety` controls, budget outermost), `tools.py` (existing tool bindings as StructuredTools), `graph.py` (`LangGraphPocState`, `build_create_agent_investigator`, `build_poc_graph`/`run_poc` — planner→router→select→create_agent investigator→receive→final, explicit failure routing) |
| `agentrun_langgraph_poc.py` | POC demo — offline scripted default (exit 0), `--live` via `ChatOpenAI`/`OpenAILLMClient`/`live_bindings` (needs `OPENAI_API_KEY`) |
| `src/repo_intelligence/agent/agent.py` | `run_result(question, *, max_tool_calls=None)` per-call frozen budget (M20) + bounded-loop internals |
| `src/repo_intelligence/agent/request_budget.py` | H1/H2: `RequestBudgetPolicy`, complete-request record builder, `enforce_request_budget` (clip envelopes / return original on rejection), `note_provider_context_overflow` |
| `src/repo_intelligence/react_safety.py` | Generic ReAct context safety (package ROOT — the `agent/` layering guard forbids agent sources importing `tools`): `BudgetedLLMClient` (every request → `enforce_request_budget_or_raise`; rejection ≠ consumed call allowance, provider failure = consumed), `bound_observation` (line-aligned, default = tool-protocol `MAX_TOOL_RESPONSE_CHARS`), `ReActLoopGuard` (iteration/tool/model/cumulative limits + terminal statuses, stop-on-exhaustion), `restrict_history` (per-request, pairs atomic, transcript never mutated), `ReActSafetyConfig.request_budget_policy()` (global default or tighter local — no global change) |
| `src/repo_intelligence/tools/graph_projection.py` | H3: `project_graph_result_for_model` → `model_graph_compact.v1` + `model_facing_graph_tool`/`model_facing_graph_result` wiring seams |
| `src/repo_intelligence/llm/structured.py` | Structured-output validation + payload normalization |
| `src/repo_intelligence/tools/protocol.py` | Shared tool protocol: `ToolCall`/`ToolError`/`ToolResult` + 1A–1C response ceiling — `MAX_TOOL_RESPONSE_CHARS = 24_000` (chars, envelope-inclusive via `response_chars()`), `TRUNCATION_NOTE_RESERVE = 400`, line-aligned `bound_result_text()` |
| `src/repo_intelligence/tools/get_architecture_context.py` | Tool 1 orientation — default heading index (fence-aware `_render_index`), optional `{section, max_chars}` targeted retrieval (1A), 11-key bounded payload |
| `src/repo_intelligence/tools/read_file.py` | Tool 3 exact evidence; always-bounded response (1B) with `truncated/next_line_start/partial_line/note` |
| `src/repo_intelligence/tools/search_repository.py` | Search tool (`EVIDENCE_NOTE` in payloads); response ceiling drops whole trailing results (`omitted_results`/`note`) |
| `src/repo_intelligence/retrieval/` | M18 RAG (index/embeddings/faiss) |
| `agentrun_m19_orchestrator_agents_demo.py` | M19 offline coordinator demonstration (seven phases; exit 0/1/6) |
| `src/repo_intelligence/langgraph/langgraph_orchestrator_integrated_M20.ipynb` | M21 notebook: planner → roles → `build_langgraph_orchestration_graph` as ONE nested node → summary/trace (12 code cells, all compile; no `Orchestrator.execute`) |
| `docs/` | Milestone specs m1–m21 + `langgraph-end-to-end.md` (production opt-in + notebook) + agent specs; version logs in investigator docs |
| `notebooks/langgraph_end_to_end_workflow.ipynb` | PRIMARY LangGraph notebook: 10 sections / 17 offline code cells (topology, execution, failure/parity/safeguards) on the REAL composition root; contract pinned by `tests/agents/test_langgraph_notebook.py` |
| `notebooks/langgraph_one_investigator_smoke_test.ipynb` | Restricted one-investigator smoke notebook (12 sections / 10 code cells): fixture plan `["P1"]` → one Investigator → terminate `NO_ANALYSIS`; provider guard limit 0; live cell gated `LIVE_MODEL_TEST`/`CONFIRM_LIVE_CALLS` (both False); live caps `max_tokens=2048` output + `8,000`-token ESTIMATED input ceiling (probe + client-boundary enforcement) + `max_calls=1`/`max_retries=0`; §11 offline demo of the shared `react_safety` controls (fake clients/tools, 0 provider calls) |
| `tests/agents/one_investigator_smoke_policy.py` | Test-side policy module: `OneTaskExecutionPolicy` (blocks a second task), `AnalystForbidden`, `ProviderCallGuard(limit=0)`, live `CallLimitedClient(max_calls=1, max_tokens=512)` + optional `input_budget` (experiment 8,000-token ceiling re-enforced at the client boundary) |
| `tests/` | ~2524 tests; `conftest.py` has shared fixtures (`enable_search`, …) |
| `runtime/agent_runs/<id>/` | Per-run traces |
| `artifacts/` | Demo outputs (pinned sample repo under `artifacts/main_2026-09-22_ef11dac/`) |

Protected files (do not modify unless explicitly asked):
`agentrun_m17_orchestration_demo.py`, `agentrun_architecture_visualization_demo.py`.

**Not a git repo** — never commit/push; no `.git`.

---

## 6. Recent work (completed)

### A. Evidence lifecycle fix (M13 acceptance at run_result)
- Root cause: investigators returned projections without the
  inspected-evidence gate; search hits (path+line, never read) laundered
  into `SUPPORTED` findings.
- Fix: `enforce_inspected_evidence(report, trace)` — deterministic,
  removal-only; unread-backed findings stripped → uncertainty; status →
  `incomplete` if anything stripped; aggregate rebuilt. Wired into both
  specialists' `run_result` via `dataclasses.replace`.
- Prompt amendments (read-before-cite, graph-id recovery, search-then-read);
  `EVIDENCE_NOTE` in search payloads.
- 9 regression tests: `tests/agents/test_investigator_evidence_acceptance.py`.

### B. Pydantic v2 + Python 3.12 migration (isolated env)
- Created `venv312/` (Python 3.12.3) — `.venv` (3.7) and pyenv untouched
  (verified byte-identical against `/tmp/old_venv_freeze.txt` snapshot).
- Migrated 7 source files (validators, `parse_obj`, `.dict()`, `.copy()`,
  `.schema()`, `__fields__`). Remaining v1-style calls in ~17 TEST files
  use v2's still-supported deprecated aliases — pass, optional cleanup.
- Behavioral fix: `_ref_key` in `multiagent/contracts.py` accepts raw dicts
  (v2 `mode="before"` sees dicts in `evidence_refs` on round-trip).
- Test updates: `FieldInfo.type_` → `model_fields[...].annotation`;
  `construct()` → `model_construct()`; evidence-structural test injects the
  bad ref into the already-built message (v2 revalidates nested instances at
  constructors, v1 did not).

### C. Attempted fixes REVERTED (2026-10-14) — do not re-apply blindly
- OpenAI `strict: true` structured-output enforcement
  (`llm/openai_client.py` + strict-schema translation) and the M17
  demo `--live` fixes (Phase 6 `getattr` on `_llm_client.requests`,
  `LIVE_MAX_TOOL_CALLS=64` in `build_agents`) were implemented and
  then REVERTED: the user reported they broke live runs. All source,
  tests, and this file are back to the pre-change baseline (2135).
- Known live-mode issues that remain BY CHOICE (accepted for now):
  * M10 `--live`: gpt-4o-mini may return wrong field names (e.g.
    `description` instead of `finding`) because the schema is sent
    without `strict` — application validation fails explicitly; the
    user accepts these failures.
  * M17 `--live`: Phase 6 crashes (`OpenAILLMClient` has no
    `.requests`) and the default 8-call budget can exhaust mid-step.
- Any future fix for these must first reproduce against
  `agentrun_repository_query.py --live` (core path), not only the
  demo scripts.

### D. Preassigned step roles in `Orchestrator.execute` (2026-10-14)
- `execute(task, plan, role_overrides=None, role_assignments=None)`:
  callers may pass a fully preassigned `Sequence[RoleAssignment]`
  (additive 4th param; signature back-compat kept). Both inputs
  together → `ValueError` at entry.
- Preassigned path (`orchestration/runner.py::_preassigned_role_
  assignments`) bypasses `role_assigner` entirely; validated before
  any agent runs: items are `RoleAssignment`s, ids unique, coverage
  EXACT vs the plan (unknown/missing ids named), roles in
  `AGENT_ROLES`, ANALYST must follow an INVESTIGATOR in plan order,
  then normalized to plan order (caller may list any order). No
  silent fallback, no post-condition rewriting (all-INVESTIGATOR
  plans end `INCOMPLETE/NO_ANALYSIS`, same as the injectable
  assigner path). Reasons are preserved verbatim in
  `orchestration.step.assign` events.
- Only files touched: `src/repo_intelligence/orchestration/runner.py`
  + `tests/orchestration/test_m17_runner.py` (17 new tests, baseline
  2135 → 2152). Roles/post-conditions in `roles.py` unchanged.

### E. Contract-driven agent coordinator (M19, 2026-10-15)

- New modules under `orchestration/`: `contracts.py` (frozen
  `AgentTaskRequest`/`AgentTaskResult`, `TaskFailure` +
  `classify_exception` — the exact M17 predicates, one taxonomy,
  `TaskUsage` honest accounting), `executor.py` (`AgentExecutor`
  Protocol, `AgentTaskExecutor` adapter with per-call `last_trace`
  snapshot, `ParticipantRegistry` never falls back),
  `scheduling.py` (pure `sequential_dependencies` + `ready_tasks`),
  `assembly.py` (`merge_investigation_reports` relocated VERBATIM +
  `RunAssembler`/`AssemblyOutcome`).
- `models.py`: `TaskRecord` (authoritative task state — status,
  attempts, failure_category, usage, trace id, output) + additive
  `OrchestrationRun.tasks`.
- `runner.py`: the retry loop now builds an `AgentTaskRequest` per
  attempt and dispatches through `self._executor.execute(...)`
  (constructor gained optional `executor:` — injected there, NOT on
  `execute()`); private helpers `_trace_run_id`, `_count_tool_calls`,
  `_failure_reason`, `_is_agent_budget_*`, `_is_execution_boundary_*`
  removed (executor owns attribution/classification); task records
  committed only by the coordinator (completed / failed+category /
  blocked+reason); events BYTE-unchanged (task lifecycle surfaces via
  `run.tasks`, never new event ops — `ORCHESTRATION_OPERATIONS`
  tests must stay green); default adapter keeps its OWN monotonic
  clock (do not thread the orchestrator clock in).
- New demo `agentrun_m19_orchestrator_agents_demo.py` — six phases
  (contract → dispatch → coordinated → readiness → failure →
  assembly), offline, byte-deterministic, exit 0/1/6,
  `--repository` fixture mode; deliberately NO `--live` flag (live
  goes through `agentrun_repository_query.py --live`).
- Tests: 69 new (`tests/orchestration/test_m19_*.py`), baseline
  2152 → 2221. Docs: `docs/m19-contract-driven-coordinator.md`,
  README M19 paragraph + milestone line, this work log.

### F. M19.1 orchestrator contract hardening (2026-10-15)

- **Result validation at the boundary** (`contracts.py`):
  `validate_task_result(request, result)` + `TaskContractError` +
  `FAILURE_CONTRACT_VIOLATION` (non-retryable). Checks: result type,
  correlation fields echo the request (`run_id`/`task_id`/`attempt`/
  `role`), `usage`/`evidence_refs` declared types, status ∈
  {completed, failed} (`incomplete`/`cancelled` REJECTED as result
  statuses — coordinator-side record states only), completed-with-
  failure / free-form-dict output rejected, failed-without-failure /
  failed-with-output rejected; `completed + output=None` deliberately
  allowed (runner's retryable `MISSING_STRUCTURED_OUTPUT` policy).
  Runner calls it immediately after `executor.execute` and, on
  violation, commits NOTHING (no output/usage/trace/agent_run_id/
  agent_runs/tool calls) and fails the task via the existing
  `fail_step` path (`INCOMPLETE`/`AGENT_FAILED`, successors blocked,
  no retry, events unchanged). `ScriptedExecutor` test doubles echo
  `request.run_id` via `dataclasses.replace`.
- **Trace attribution** (`runner.py`): `_close_window(window, trace)`
  and `fail_step(..., trace, ...)` now take the RESULT's own
  `result.trace` — the participant `last_trace` is never read (an
  injected executor's participants may never run or expose stale
  traces); untrusted traces (contract violation) get `None`;
  objects lacking `record_orchestration_event` are skipped
  (observability only, run log stays authoritative).
- **Usage semantics** (`contracts.py`, `models.py`, `runner.py`):
  `combine_usage(*usages)` — counters/elapsed always sum, token
  fields sum ONLY reported values (stay `None` when unreported, never
  fabricated 0). `TaskRecord.usage_total` + `OrchestrationRun.
  usage_total` (additive fields) accumulate per committed attempt;
  `TaskRecord.usage` stays the MOST RECENT attempt. Budgets keep
  using `tool_calls_total` (unchanged).
- **`max_tool_calls` documented + test-pinned as RESERVED** — the
  runner never sets it; the executor never enforces it; enforcement
  remains agent-level `AgentExecutionConfig.max_tool_calls` +
  `OrchestrationConfig.max_total_tool_calls`.
- **Sequential dispatch test-pinned** at runner level:
  `test_m19_sequential_dispatch.py` (plan-order requests, event
  sequences prove P1 complete before P2 start, successor gets
  exactly the merged report). Docs already stated this seam.
- **Demo**: 7th phase `run_recovery_phase` — the SAME transient
  `structured_output` failure under the same bound where attempt 2
  COMMITS (attempts 2, retries 1, recent 2 vs cumulative 3 model
  iterations); assembly renumbered to Phase 7; byte-deterministic
  (two-run test), default + fixture exit 0.
- **Docs**: `docs/m19-contract-driven-coordinator.md` — 3 Mermaid
  diagrams (boundary flow, task lifecycle state machine, retry
  decision), validation rule table, usage/trace/reserved-budget
  sections, updated counts; README M19 paragraph; this work log.
- **Environment note**: `venv312` was found partially wiped today
  (§2 repair recipe) — repaired in place, `pip check` clean.
- Tests: 34 new (validation 21 + trace 4 + usage 6 + sequential 2 +
  demo recovery 1), baseline 2221 → **2255 passed, 2 skipped**;
  `test_m19_*.py` 69 → 103; `tests/orchestration/` 188 → 222.
  Full suite, M19 demo (both modes), evidence tests green.

### G. M20 orchestrator contract completion + task-budget
### enforcement (2026-10-16)

- **Per-attempt budget enforced** (`AgentTaskRequest.max_tool_calls`
  no longer reserved): `Agent.run_result(question, *,
  max_tool_calls=None)` validates the value (int ≥ 0, bool/str
  rejected → `ValueError`), builds a FROZEN per-call config via
  `dataclasses.replace` (no shared mutation), threads it as
  `execution_config` through `_run_loop` → `_execute_loop` →
  `_finalize_run` — used for `AgentTrace.start` and every in-loop
  budget read (context-compaction helpers still read the shared
  config; unrelated fields). Three specialists gained the same
  kwarg with None-guarded pass-through (one-arg calls byte-identical
  M19). Executor passes the kwarg ONLY when the request set it.
- **Runner**: `OrchestrationConfig.max_task_tool_calls` (validated,
  `None` default) rides every request; the run-level
  `max_total_tool_calls` gate was moved to the RETRY DECISION
  (inside `if retryable and attempts <= max_retries:` BEFORE
  `run.retries += 1`) — a budget stop never counts a consumed
  retry, blocks remaining steps, and returns before validation
  (terminal status never overwritten); budget event gained additive
  `limit`/`observed` fields (M17 event name/ops unchanged);
  `validate_task_result` wrapped with `except TaskContractError` +
  `except Exception` → same `contract_violation` path, never leaks.
- **Result validation hardened** (`contracts.py`): new
  `_validate_task_usage` (non-negative int counters, bool rejected;
  elapsed finite ≥ 0 via `math.isfinite`; tokens None|non-negative
  int) + `_validate_task_failure` (isinstance FIRST — a str/dict
  failure raises `TaskContractError`, never `AttributeError`
  mid-read — then category/error_type/error_message str, retryable
  bool, termination_reason None|str); `agent_run_id` None|str.
  `AgentTaskRequest.__post_init__` validates `max_tool_calls`.
- **Executor defect fixed**: `_count_tool_calls` sums EXECUTED
  `tool_results`, not requested `tool_calls` (blocked calls were
  over-counting usage).
- **Docs**: `docs/m20-orchestrator-contract-completion.md` (4
  Mermaid diagrams: component/budget enforcement, success sequence
  w/ contract-violation branch, failure/retry budget flow, +
  sections 1–8 incl. future-LangGraph boundary); M19 doc "reserved"
  wording cross-referenced to M20; README M20 paragraph + milestone
  line; this work log.
- Tests: 52 new (45 orchestration `test_m20_*` [budget 9 +
  validation 33 + lifecycle 3] + 7 agent-boundary), M19
  usage-semantics reserved tests rewritten 2→2 (net 0), M19
  executor double updated to real `tool_results` shape. Baseline
  2255 → **2307 passed, 2 skipped**; `tests/orchestration/` 222 →
  267. Verification: full suite, orchestration, M19 tests (103),
  evidence (9), agent budget (22), M19 demo default+fixture exit 0,
  M17 demo exit 0, offline CLI exit 0/COMPLETED.

### H. M21 LangGraph orchestration (2026-10-16)

- **Scope honored**: task-level control flow (selection → execute →
  commit → retry/budget-gate/block/terminate → assemble/validate) runs
  as ONE compiled StateGraph in
  `orchestration/langgraph_runner.py` over unchanged M19/M20
  contracts; the plan is data (2-step and 4-step plans on the same
  compiled instance, test-proven); `Orchestrator.execute` never on the
  graph path (monkeypatch + source-scan tests). Planner/role
  assignment stay at the driver/notebook boundary; `validate_plan`
  runs inside `initialize`. Real specialists reach the graph through
  the same `AgentTaskExecutor` boundary (`QueryRouter.run_result`
  tested both repeat-routing and per-goal routing).
- **Environment**: `langgraph==1.2.15` installed into `venv312` (the
  `venv312/bin/pip` script is missing — use
  `venv312/bin/python -m pip`); declared as optional `graph` extra +
  `requirements.txt` pin (dependency declaration policy).
- **Guard-test policy updates (deliberate, documented — not silent
  weakening)**: `tests/llm/test_client_contract.py::PROVIDER_IMPORT_RE`
  dropped `langgraph` (orchestration graph lib, not a provider SDK);
  `test_m17_repository_agnostic.py` now uses two patterns — full
  framework/broker ban everywhere EXCEPT `langgraph_runner.py` may
  import `langgraph` only (crewai/autogen/sqlalchemy/redis/pika/
  celery/requests/httpx still banned everywhere).
- **Defect found + fixed**: graph `terminate()`'s 3rd param was named
  `reason`, so the PLAN_FAILED/VALIDATION_FAILED branches
  (`terminate(..., reason=...)` — the runner's event field) raised
  `TypeError: got multiple values for 'reason'` instead of committing
  terminal events. Renamed to `termination_reason` (runner
  `_terminate` signature); `reason` now flows as an event field.
- **4 runner-parity fixes (from the prior audit session, kept)**:
  `commit_result` RETURNS `analyst_trace` (LangGraph discards
  in-place key writes); terminal failure records +
  closes `orchestration.step.failed`; `orchestration.retry` records
  INTO the per-step window which `retry_task` REUSES; failed
  `agent.complete` reason = `failure.error_type`.
- **Shared-layer budget defect found + fixed (post-milestone notebook
  run)**: the notebook's failing P1 (2 attempts, `tool_calls=0`,
  `trace_id=None`, P2 blocked) was root-caused to the executor's
  loud-boundary kwarg `run_result(question, max_tool_calls=...)`
  missing from the TWO participant wrappers in `repository_query.py`
  — `QueryRouter.run_result` and `ObservedAgent.run_result` →
  `TypeError` before ANY execution → classified `unexpected`/
  retryable → retried once → `INCOMPLETE/AGENT_FAILED`; no trace was
  ever created (correctly not fabricated). Reproduced IDENTICALLY on
  both paths (shared layer, not the graph adapter); the CLI escaped
  it only because `M17_ORCHESTRATION_CONFIG` leaves the budget
  `None`. Fix: both wrappers accept `max_tool_calls=None` and
  forward it verbatim; `None` keeps the exact pre-M20 one-argument
  call (legacy participants unchanged); a participant that rejects a
  set budget still fails loudly at the call (policy pinned).
  Regression tests: 2 wrapper-unit + 1 specialist-chain
  (`tests/agents/`) + 5 graph-level (budgeted-router e2e, both-paths
  compat, terminal-failure-after-start keeps trace, retry keeps both
  attempts' attribution, pre-trace failure records honest `None`).
- **Tests**: `tests/orchestration/test_langgraph_sequential.py` — 34
  deterministic behavioral tests (order/roles/status, dynamic plan
  sizes, router dispatch, assembled-output handoff, 5 invalid-role
  ValueError-before-execution variants, dependency timeline + blocked
  successors, retry/cumulative usage/non-retryable/exhaustion,
  correlation/missing/dict-output policies, agent + run budgets with
  `limit`/`observed`, trace identity, NO_ANALYSIS /
  DECLARED_INCOMPLETE / missing-analyst, never-`Orchestrator.execute`,
  + budget-boundary/tracing regressions).
  Scripted boundary fakes cover failure paths; tests that must reach
  `COMPLETED` use real scripted agents (validation needs gathered
  evidence in traces).
- **Notebook**: `langgraph_orchestrator_integrated_M20.ipynb` — dead
  "routing preview" print removed, `Validation status:` line added; 12
  code cells all `ast.parse` clean.
- **Docs**: `docs/m21-langgraph-orchestration.md` (state/nodes/
  transitions, parity rules, scenario→test table, guard policy,
  boundary-defect section, verification); README M21 paragraph +
  amended "no agent frameworks" claim + milestone line; this work log.
- Baseline 2307 → **2344 passed, 2 skipped** (+29 initial M21 +
  3 wrapper/chain + 5 boundary/tracing);
  `tests/orchestration/` 267 → **301**. Verification: focused 34,
  orchestration 301, full suite, evidence+budget 31, repository_query
  51, M19 demo default+fixture exit 0, M17 demo exit 0, offline CLI
  exit 0/COMPLETED, four-step scenario (router + budget 12)
  COMPLETED/SUCCESS with trace ids + 3 measured tool calls. Live
  `--live` NOT run here (no `OPENAI_API_KEY`,
  no `.env` — confirmed) and not claimed.

---

### I. H1–H4 context-overflow boundary work (2026-10-17)

- **Incident root cause (boundary map)**: exactly three
  `self._llm_client.complete(request)` sites in `agent/agent.py`
  (loop `_execute_loop`, planner `_create_plan`, finalization
  `_finalize_run`); the only provider adapter is `OpenAILLMClient`
  (`gpt-4o-mini`, the observed 128k window). Nothing measured a
  request before sending: `estimate_context_characters` covered
  message content only — blind to the ~2,719-char tool-definition
  block and the structured-output schema — and graph observations
  triple-copied their evidence (hub depth-2 traversal measured
  138,999 B ≈ 35k tokens). Result: a 128,882-token request reached
  the provider → `context_length_exceeded`.
- **H1 diagnostics**: `agent/request_budget.py` builds a
  privacy-safe record (per-message chars/tokens-est, role counts,
  largest messages, tool-def + `structured_output` sizes, totals,
  budget, disposition) with the explicit
  `CHARS_PER_TOKEN=4`/`chars_per_token_estimate` method;
  `AgentIteration.request_budget` carries it (omitted when `None`,
  so hand-built trace shapes stay valid) at ALL THREE sites;
  provider context-length `LLMError`s are correlated onto the record
  via `note_provider_context_overflow` (message clipped to 500,
  `preflight_within_budget`/`preflight_total_tokens_est`).
  `stop_reason` maps `CONTEXT_BUDGET_EXCEEDED`/`context.budget_
  exceeded` → `CONTEXT_LENGTH_EXCEEDED` (the pinned 4-entry
  `STOP_REASONS` unchanged); `trace.TERMINATION_REASONS` += `CONTEXT_
  BUDGET_EXCEEDED` — deliberately reusing the established
  context-exhaustion term (the `context_engine.ContextOperationError.
  code` of the same name lives in a separate namespace; no consumer
  conflates them); lifecycle ops += `context.request_reduced`,
  `context.budget_exceeded` (count-only payloads; ACCEPTED runs
  still emit no budget events at all).
- **H2 enforcement**: `enforce_request_budget` returns
  ACCEPTED (original identity) / REDUCED (largest tool messages
  clipped largest-first to valid-JSON envelopes
  `{truncated, tool, original_characters, excerpt, note}`,
  `MIN_TOOL_EXCERPT_CHARS=400`, deterministic exact-fit correction,
  paired with its own record) / REJECTED (returns the ORIGINAL
  request — never sent, never retried unchanged; reason
  `tool_or_response_contract_exceeds_budget` wins when the fixed
  contract alone is over, else `protected_request_exceeds_budget`).
  `AgentExecutionConfig.max_request_input_tokens` (validated) feeds
  `_gate_model_request`, which runs BEFORE `_lifecycle_event` at each
  site; typed failures: loop/planner → `CONTEXT_BUDGET_EXCEEDED`
  (+ `context.budget_exceeded`), boundary finalization →
  `FINALIZATION_FAILED`, finalize-gate rejection →
  `finalization.failed` + `FINALIZATION_FAILED`;
  `FINALIZATION_BOUNDARY_NOTICES["context.budget_exceeded"]` added.
- **Design decisions**: default budget = 128,000 capacity − 16,384
  output − 16,384 margin = **95,232 tokens (380,928 chars)** from the
  observed window; deliberately NOT derived from
  `max_context_characters` (that ceiling keeps its history-only
  exemption semantics — coupling broke small-ceiling tests and
  mis-attributed failures; `_request_budget_policy` documents the
  split); record key `structured_output` (provider-neutrality test
  bans `response_format`/`json_schema`/`openai`/`gpt-` in trace
  payloads); estimates always labeled, provider `usage` kept separate.
- **H3 compact graph projection**: `tools/graph_projection.py`
  (`model_graph_compact.v1`) — drop `attributes`, one deduplicated
  evidence table (`evidence_id = "ev:" + sha256(canonical)[:12]`,
  `origin`/`index` stripped), nodes/relationships reference
  `evidence_ids`; `mode`/`query`/`kinds`/`counts`/`truncated` pass
  through verbatim; `payload{projection, max_bytes, body_bytes,
  truncated[, original_body_bytes, kept]}` + `refinement_hint`;
  over-budget → drop DEEPEST nodes one at a time (start node never),
  then dangling relationships and unreferenced evidence; irreducible
  → structured `MODEL_PAYLOAD_TOO_LARGE` failure; non-canonical
  payloads (no `mode` — the fixture seam) pass through; re-projecting
  a projected payload resolves carried ids (no dangling refs).
  Wired: `tool_dispatcher.tool_registry()` →
  `model_facing_graph_tool`; `repository_query.live_bindings`
  `graph()` → `model_facing_graph_result(...)` (the
  `rq.query_engineering_graph_tool` monkeypatch seam preserved).
  Measured on the pinned artifact: 138,999 → 23,200 B,
  94,480 → 23,246 B, 64,900 → 22,930 B (all ≤ 24,000, truncated);
  summary 4,158 → 4,271 B (untruncated). The canonical tool and its
  suite are byte-untouched.
- **H4 lifecycle tests** (6): planned run — all five model calls
  carry accepted records on the default budget; 400,000-char
  incident-shaped question → zero provider calls, typed failure,
  effective budget 95,232 asserted; character-ceiling vs token-budget
  independence in BOTH directions (ceiling op-prefixes present with
  no budget events; armed budget rejects with `effective=100` while
  the armed ceiling stays silent); provider-overflow record embeds no
  request content (sentinel absent, message ≤ 500);
  armed-but-generous budget sends byte-identical transcripts.
- **Baselines**: 2344 → **2407 passed, 2 skipped** (+35 request-
  budget + 22 graph-projection + 6 lifecycle). Focused gates:
  orchestration 301, M21 34, evidence 9, repository_query 51,
  bounded 22; M19 demo default+fixture exit 0; M17 demo exit 0;
  offline CLI exit 0 / COMPLETED with collaboration+diagram
  validation passed; a real offline run's `trace.json` carries
  per-iteration `request_budget` records. Live `--live` NOT run in
  this environment (no `OPENAI_API_KEY`) and not claimed.

### J. Bounded tool outputs — Steps 1A–1D (2026-10-18)

- **Why**: H1–H4 capped the WHOLE request, but individual tool
  observations could still be arbitrarily large (the incident's hub
  graph payload was ~139 KB); per-tool ceilings make each observation
  self-bounding and resumable, so the H2 clip path becomes a safety
  net rather than the primary defence.
- **Shared contract** (`tools/protocol.py`): `MAX_TOOL_RESPONSE_CHARS
  = 24_000` measured on ONE serialized `ToolResult`
  (`response_chars()` = `len(json.dumps(result.to_dict()))` — exactly
  what `agent/agent.py` embeds), `TRUNCATION_NOTE_RESERVE = 400`,
  `BoundedText`, line-aligned `bound_result_text()`. **Defect found &
  fixed while writing 1C**: the first `bound_result_text` measured the
  DATA dict only (undershot the envelope by ~300 chars); it now
  empties the text on a COPY of the final `to_dict()` so the bound is
  measured against what the model actually receives.
- **1A `get_architecture_context`**: optional `{section?, max_chars?}`
  — case-insensitive exact heading match (`SECTION_NOT_FOUND` lists
  available), probe-bounded `context` + `truncated/returned_chars/
  total_chars/char_limit/available_sections/note`; CLI
  `--section/--max-chars`; tool schema in
  `agent/tool_definitions.py` updated (unknown keys still rejected).
- **1B `read_file`**: every success bounded; new keys
  `truncated/returned_chars/total_chars/char_limit/next_line_start/
  partial_line/note`; whole-line cut first, character prefix only
  when ONE line exceeds the remaining budget (evidence-coverage
  honesty: a partial-line read reports an EMPTY delivered range
  `line_end = line_start - 1` — same shape as the existing (1,0)
  convention — so `enforce_inspected_evidence` can never attribute a
  claim to a line that was only partly read; re-reading hits the same
  ceiling deterministically).
- **1C `search_repository`**: `_bound_payload` drops WHOLE trailing
  results (head of the ranked list preserved, ranking never
  recomputed) with `omitted_results`/`note`; `truncated` ORs the new
  condition; an envelope-only overflow (no result fits — absurd
  query) is a structured `QUERY_TOO_LONG` failure.
- **1D three-boundary regression** (`tests/tools/test_response_
  ceiling_boundaries.py`, 7 tests): pins the three boundaries as
  SEPARATE knobs (tools 24,000 chars / projection 24,000 bytes /
  request budget 95,232 tokens — 128k − 16,384 − 16,384), namespace
  independence, projection truncation + `MODEL_PAYLOAD_TOO_LARGE`,
  and preflight composition (ceiling-sized observation → ACCEPTED;
  500 KB observation → REDUCED to a valid-JSON envelope then stable
  under re-enforce; huge system prompt → REJECTED, request identity
  preserved, `RequestBudgetExceededError` surfaces the record).
  `graph_projection.py` and `request_budget.py` byte-untouched.
- **Test-side contract updates (deliberate, compatible)**:
  `test_read_file.py` exact-key-set assertion + CLI full-file exact
  dict expanded to the 11-key shape; `test_standalone_cli_section_
  selection` max-chars 8000 → 20000 (old value would truncate the
  section it was asserting on). Tool-definition tests updated for
  the 1A schema; 2 renamed, none removed.
- **Docs**: the three tool docs (new input/output tables, ceiling and
  continuation semantics, new failure rows), README Tool 1–3 input
  contracts + ceiling clauses, `tools/__init__.py` package docstring,
  AGENTS.md header/§4/§5 baselines.
- **Baselines**: 2419 → **2462 passed, 2 skipped** (+43: gac 40
  [was 19], read_file 73 [was 64], search 66 [was 60], 1D +7).
  Focused gates: tools trio 279 (was 236), tools-only 243, three
  boundaries 70 (41 H1/H2 + 22 projection + 7), orchestration 301,
  M21 34, evidence 9, repository_query 51, bounded 22; M19 demo
  default+fixture exit 0; M17 exit 0; offline CLI exit 0 / COMPLETED
  (collaboration + diagram validation passed). Real-run proof:
  the CLI trace shows `read_file` truncations of `README.md`
  (21,954 / 66,276 chars, `next_line_start: 363`) and
  `MCP_GUARDRAILS.md` (20,930 / 27,802, `next_line_start: 350`) with
  the run still completing — note `trace.json` stores a separately
  capped (2,000-char) copy per observation (`TRACE_DATA_CHAR_LIMIT`);
  the model sees the full ≤24,000-char result. Live `--live` NOT run
  (no `OPENAI_API_KEY`) and not claimed.

### K. Tool 1 targeted default — heading index (2026-10-18)

- **Why**: H3/J bounded tool outputs, but Tool 1's default `{}` still
  handed the WHOLE architecture-context document (up to 24,000 chars)
  as orientation — pay-for-everything before the agent knows which
  section it needs; the opposite of targeted.
- **Default = compact heading index** (`_render_index`): the
  document's real Markdown heading lines in document order (leading
  `#`s carry the level — `#`/`##`/`###` stay distinguishable) + a
  blank line + the fixed hint line
  `next: {"section": "<title from the list above>"} returns that
  section (case-insensitive exact heading match).`; no headings →
  `(no Markdown headings found)` placeholder (still a deterministic
  index). The index is rendered INTO `data.context` — reusing the
  existing bounding machinery; `total_chars` = the selection. Measured
  on the pinned artifact: 11,138-char document → 886-char index (39
  headings), untruncated, provenance intact.
- **Fence-aware heading parsing** (`_heading_blocks` rewrite):
  per-line scan (`text.split("\n")`, one trailing `\r` stripped) with
  `_FENCE_OPEN_RE`/`_FENCE_CLOSE_RE` (CommonMark-style small regexes,
  no new dependency): open = 3+ backticks/tildes, ≤3 leading spaces
  (backtick info strings may not contain backticks); close = same
  character, run ≥ open, whitespace only; unclosed fence extends to
  EOF. A fenced `# comment` / `## heading` can neither enter the
  index, become a selectable section, nor terminate the real section
  around it (block-end rule unchanged for everything else). Offsets
  stay positions in the ORIGINAL text (section bodies remain exact
  slices). No `~~~` fences exist in repo markdown today; tildes
  supported anyway (CommonMark parity, cheap regex, tested).
- **Unchanged, verified**: the 11-key shape,
  `SECTION_NOT_FOUND`/`INVALID_ARGUMENTS`/`CONTEXT_NOT_FOUND`/
  `INVALID_CONTEXT`/`SOURCE_METADATA_UNAVAILABLE`, the case-
  insensitive exact selector, `max_chars` lowering the ceiling only,
  provenance from the WHOLE artifact, the 1D three-boundary set, and
  `MAX_TOOL_RESPONSE_CHARS` (no competing size constant; oversized
  indexes cut at whole lines with `truncated` + `note` + bounded
  `available_sections`; a single line that cannot fit at all is a
  character prefix with the "first line" note). No new arguments
  (`_ALLOWED_ARGUMENTS` still pinned by test).
- **Consumers**: no prompt/production code pinned the old whole-
  document default (the evidence validator treats gac as not-file-
  addressable; `live_bindings` is a passthrough). The 7
  `tests/agent/test_agent_react.py` ReAct tests scripted the orient
  step as `{}` then derived later actions from the observation body —
  the scripted model now asks `{"section": "Mechanism"}` (derivation
  still feeds from observation 1; tool-call counts unchanged).
- **Docs**: tool doc §12.1–12.5 (index semantics, fence rule,
  updated success example, selection step in the flow diagram,
  "does not summarize" bullet), README Tool 1 input contract + orient
  bullet, `tool_definitions` description (no schema-shape change),
  CLI help.
- **Tests** (gac 40 → 46): default-index-not-document, exact index
  equality + hint pin, heading levels/order, oversized-document
  default returns only the index, oversized-index visibly bounded
  (2,000 headings; exact whole-line prefix of the deterministic
  index), placeholder index for a heading-less document, backtick
  fence (ignored in the index; fake section → `SECTION_NOT_FOUND`;
  the real block keeps the fence verbatim and runs to the NEXT real
  heading), tilde fence (longer closer closes); whole-document
  pinning tests moved to section mode with an XL fixture
  (`filler_lines=250`).
- **Baselines**: 2462 → **2468 passed, 2 skipped** (+6). Focused
  gates: tools trio 285 (was 279), three boundaries 70,
  orchestration 301, react 7; offline CLI exit 0 / COMPLETED.
  Real-artifact smoke: `{}` → 886-char index of the pinned doc;
  `--section "Evidence / Limitations" --max-chars 3000` → complete
  1,635-char section. Live `--live` NOT run (no `OPENAI_API_KEY`)
  and not claimed.

### L. LangGraph end-to-end — production opt-in + primary notebook (2026-10-19)

- **Goal**: close the migration audit's F1 (the production CLI never
  used LangGraph; graph owns none of the production workflow's task
  sequencing) without flipping any default, plus ONE educational
  notebook showing the complete workflow on the real implementation.
- **Production seam** (`repository_query.py`): `ORCHESTRATION_LEGACY/
  LANGGRAPH/MODES/UNAVAILABLE` constants; `run_repository_query(...,
  orchestration="legacy")` keyword-only param validated via
  `RepositoryQueryError` (same style as `mode`); `_run_with_langgraph`
  helper — lazy graph import, roles preassigned with the SAME
  `assign_roles` assigner legacy `execute` uses internally (role
  parity by construction); planner (`_build_plan`) off-graph, shared
  verbatim; `_assemble` unchanged on the graph's `OrchestrationRun`.
  `ImportError` (missing optional extra) → typed structured failure
  `ORCHESTRATION_UNAVAILABLE` (exit `EXIT_INVESTIGATION_FAILED`) —
  **never a silent fallback**; other graph exceptions propagate.
  CLI: `--orchestration {legacy,langgraph}` + one-line banner only
  when non-default.
- **Notebook** (`notebooks/langgraph_end_to_end_workflow.ipynb`, the
  PRIMARY one): 10 sections / 30 cells (17 pure-Python code, no
  magics, no hardcoded paths, no secrets) — objectives + Mermaid
  diagram; setup with `_require_graph()`; real-component inspection
  (source of `_run_with_langgraph`/`_build_plan` printed); compiled
  graph topology; end-to-end offline execution; task records + event
  timeline + trace attribution; failure demos (invalid plan →
  `PLAN_FAILED`; no-analyst → `INCOMPLETE/NO_ANALYSIS`, no fabricated
  finding; broken structured output → one retry, `attempts=2`);
  legacy-vs-graph parity; H1 oversized-planner gate (`PLAN_FAILED`,
  provider never called); summary. Verified honestly: Jupyter is NOT
  installed here, so verification = nbformat-4 JSON validity +
  `ast.parse` of every code cell + sequential `exec` of all 17 cells
  in ONE fresh process (identical semantics to a kernel for pure
  Python) — all expected outputs observed.
- **Tests** (15 new, importorskip-guards on the graph files):
  `tests/agents/test_repository_query_langgraph.py` (11 — unsupported
  value; default→legacy with `execute` spy + graph untouched;
  opt-in→graph with `execute` never called; failure propagates;
  ImportError→typed failure; H1 oversized→`PLAN_FAILED` before graph;
  e2e COMPLETED; field-by-field parity + disjoint run ids; CLI
  forwards flag + banner; CLI rejects unknown) +
  `tests/agents/test_langgraph_notebook.py` (4 — nbformat/cell
  parse/no secrets-or-machine paths; 10 section headings + mermaid +
  `orchestration="langgraph"`; referenced production symbols exist —
  rq seam symbols, `run_with_langgraph`/`build_langgraph_orchestration_
  graph`/state keys, m17_doubles path).
- **Docs**: `docs/langgraph-end-to-end.md` (seam contract, failure
  table, notebook map + honest verification method, how to run, test
  gates, limitations); `docs/m21-langgraph-orchestration.md` §11
  amendment (F1 closed, pointers); `docs/langgraph-migration-audit.md`
  §11 addendum (Phase 1 table: F1/F4 closed, Phase 2/3 already
  satisfied, Phase 5 still deferred); README end-to-end paragraph +
  milestone index line; this work log.
- **Baselines**: 2468 → **2483 passed, 2 skipped** (+15). Focused
  gates: 49 (11+4+34), orchestration 301, three boundaries 70, tools
  trio 285; offline CLI exit 0 / COMPLETED on BOTH backends (graph
  run prints `Orchestration: langgraph (M21 opt-in task loop)`).
  Offline field-by-field parity: `status, termination_reason,
  collaboration_status, diagram_status, answer, conclusions,
  evidence, steps shape` all EQUAL, run ids disjoint. Live graph runs
  NOT exercised (no `OPENAI_API_KEY`) and not claimed. Deferred by
  design: parallelism, checkpointing, replanning, default flip,
  legacy retirement (audit Phase 5).

### M. One-investigator LangGraph smoke test (2026-10-19)

- **Goal**: restricted orchestration flow proving the graph loop runs
  ONE Investigator and stops — no Analyst, no retry, no second task,
  ZERO provider calls by default (the MVP smoke brief); complements
  entry L's full-workflow notebook rather than replacing it.
- **Notebook** (`notebooks/langgraph_one_investigator_smoke_test.ipynb`,
  11 sections / 9 pure-Python code cells, nbformat 4): fixture plan
  `["P1"]` (single architecture-classifiable step) → `assign_roles`
  → `OneTaskExecutionPolicy` wrapping `AgentTaskExecutor.for_agents`
  (registry-level block: a second selected task raises
  `OneTaskPolicyViolation` BEFORE its participant is reached —
  asserted with a real 2-step-plan composition) → `run_with_langgraph`
  with `OrchestrationConfig(max_retries=0)` → `INCOMPLETE` /
  `NO_ANALYSIS`, attempts 1, no `orchestration.retry` op. §5/§9 print
  Planner request, Query Router label, task record, event ops,
  `TaskUsage(tool_calls=8, model_iterations=4, tokens=None)`,
  evidence refs, files read. §10 live cell is a no-op behind
  `LIVE_MODEL_TEST = False` + `CONFIRM_LIVE_CALLS = False` (one REAL
  planner call through `rq._build_plan` H1 gate + offline
  investigator; `CallLimitedClient(max_calls=1, max_tokens=512)` +
  `OpenAILLMClient(max_retries=0)`; cost = NOT CALCULATED). Verified
  by nbformat/ast + sequential single-process exec (Jupyter absent).
- **Policy module** (`tests/agents/one_investigator_smoke_policy.py`,
  test-side per the brief): `OneTaskExecutionPolicy` (counts attempts,
  records `executed`/`blocked`, blocks non-allowed task ids),
  `AnalystForbidden` double, `ProviderCallGuard(limit=0)` class-patch
  of `OpenAILLMClient.complete` + `ProviderCallAttempted`,
  `FailingInvestigator`, one-task plan fixture, and the live
  `CallLimitedClient` (`max_calls`/`max_tokens` > 0 validation,
  request copy stamped `max_tokens`, 2nd call →
  `LiveCallLimitExceeded`). Notebook + focused tests each load their
  OWN module copy via importlib (fresh guard state); both reset the
  class patch in `finally`/fixture-teardown — zero leakage into other
  test modules (proved: `tests/llm` green after the smoke module).
- **Contract additions (additive, provider-neutral)**:
  `LLMRequest.max_tokens: Optional[int] = None` (translated to
  `payload["max_tokens"]` only when set) in `llm/models.py` +
  `llm/openai_client.py`; `OpenAILLMClient(max_retries: Optional[int]
  = None)` → `OpenAI(**options)` forwarded only when set (SDK default
  preserved). `tests/llm/test_models.py::test_request_defaults_are_
  minimal` updated for the additive field (count unchanged).
- **Tests** (16 new): 13 in
  `tests/agents/test_langgraph_one_investigator_smoke.py` (zero-calls
  + guard mechanism proof, fixture plan + real `validate_plan`, router
  selection, one task / attempts=1, `NO_ANALYSIS` termination + ops
  counts, no-second-Investigator, no-retry + plan identity, Analyst
  untouched, failure composition `AGENT_FAILED`/attempts 1/retries 0,
  2-step-plan policy block, H1 `_build_plan` 400k-char preflight
  (0 client calls) + live-cell source pins, `CallLimitedClient`
  unit limits, notebook JSON/defaults/headings/secrets) + 3 in
  `tests/llm/test_openai_client.py` (max_tokens translated / unset /
  max_retries forwarded).
- **Docs**: this entry, §1/§4/§5 command + baseline updates, §5 table
  rows, README milestone line + pointer paragraph,
  `docs/langgraph-end-to-end.md` §7 pointer.
- **Baselines**: 2483 → **2499 passed, 2 skipped** (+13 smoke + 3
  client). Focused gates: smoke+client 34/1sk, LangGraph+llm 131/1sk,
  three boundaries 70; offline CLI exit 0 / COMPLETED. Live §10 path
  implemented + gated but unexercised (no `OPENAI_API_KEY`) — next
  step when a key is available: enable the two gates, run once, keep
  `max_calls=1` as the ceiling.

### N. Smoke-test safety adjustment — realistic Planner output + bounded input (2026-10-19)

- **Why**: entry M's live §10 cell inherited the application's global
  H1 request budget (95,232 tokens — far too large for one Planner
  call) and capped output at only 512 tokens (too small for a
  realistic structured plan). The brief: make ONE meaningful Planner
  call possible without inheriting the general budget or sending a
  large context.
- **Experiment knobs (notebook cell 2, separate from global H1–H4)**:
  `LIVE_MAX_OUTPUT_TOKENS = 2048` (was 512 — realistic Planner
  output), `LIVE_MAX_INPUT_TOKENS = 8000` (NEW — experiment-specific
  ESTIMATED input ceiling), `LIVE_MAX_PROVIDER_CALLS = 1`,
  `max_retries=0`, `LIVE_MODEL_TEST`/`CONFIRM_LIVE_CALLS` still both
  False. Global defaults untouched (`RequestBudgetPolicy()` still
  resolves to 95,232 tokens; test-pinned both before and after).
- **Input ceiling — two local enforcement points, SAME machinery**:
  (1) a PRE-CALL probe in §10 builds the EXACT outgoing Planner
  request via `build_planning_request(QUESTION)` and measures it with
  `enforce_request_budget_or_raise(..., RequestBudgetPolicy(
  max_input_tokens=8000, context_capacity_tokens=None), subject="live
  planner request")` — the same builder + estimator + record the
  application uses, a tighter policy object only. An over-ceiling
  request raises `RequestBudgetExceededError` LOCALLY: provider never
  invoked, nothing silently truncated (planner requests carry no tool
  observations, so the clip path has nothing to reduce — rejection is
  the only outcome). (2) `CallLimitedClient` gained an optional
  `input_budget` parameter that re-enforces the same policy at the
  CLIENT BOUNDARY on the exact instance about to be sent (defense in
  depth behind the probe; `rejected_inputs`/`input_records` kept for
  honest reporting). Measured on the real request: 2,138 chars →
  ~535 estimated tokens (well under 8,000); a 40k-char question →
  42,139 chars → ~10,535 → `LOCAL REJECTION`, reason
  `protected_request_exceeds_budget`, 0 client calls (all verified
  offline with a fake key + CONFIRM off, and with an oversized
  question — never a real provider call).
- **Estimate honesty (documented limitation)**: the ceiling is an
  ESTIMATE from the H1 chars-per-token RATIO (4 chars ≈ 1 token), NOT
  a tokenizer and NOT a provider-side count — it bounds the request
  conservatively but does not guarantee exact token usage; the
  provider-reported usage is displayed AFTER the call for comparison.
  §10 prints the estimated input size, the ceiling, the estimate
  caveat, the output limit, and the remaining call allowance BEFORE
  any call; a rejection prints the reason + the est-vs-ceiling
  comparison + `provider calls made: 0`. No monetary cap is claimed.
- **Output limit reaches the provider (verified, not assumed)**:
  `CallLimitedClient(max_tokens=2048)` stamps the value onto the
  frozen request via `dataclasses.replace`, and the EXISTING
  `OpenAILLMClient._to_provider_payload` translates it into the
  provider payload's own `max_tokens` parameter — test-pinned end to
  end with a recording fake provider (`provider.calls[0]["max_tokens"]
  == 2048`); no competing provider-specific call path added.
- **One-call / no-retry hardening**: a FAILED provider call now
  explicitly CONSUMES the one permitted call (it reached the
  provider); a budget REJECTION does not (the provider was never
  touched) — both pinned by tests. `OpenAILLMClient(max_retries=0)`
  still forwards `max_retries=0` to the SDK constructor (test-pinned);
  the live cell still constructs the client with `max_retries=0`.
  The blanket provider guard is relaxed 0→1 ONLY around the single
  `_build_plan` call and re-armed to 0 in a `finally`; the cell stops
  after the Planner call (no graph run → routing can never invoke a
  second live model); Investigator stays offline, Analyst disabled.
- **Tests** (+6, smoke file 13 → 19; extended 3): `experiment_input_
  ceiling_configured_and_tighter_than_global` (constants + policy
  separation + global default still 95,232 + live-cell source pins),
  `oversized_planner_request_rejected_before_provider` (probe +
  client-boundary rejection, provider untouched, rejection doesn't
  burn the allowance, normal request well under ceiling),
  `planner_output_limit_reaches_provider_payload` (end-to-end through
  the real adapter translation), `live_cell_pins_one_call_no_retry_
  and_offline_roles`, `investigator_and_analyst_remain_offline_in_
  experiment`, `h1_h4_global_safeguards_remain_active` (default
  policy unchanged, H2 clip path intact, §9 bomb block still present,
  experiment policy is a separate object). Extended: live-guard unit
  now uses 2048 + failed-call-burns-allowance; notebook-shape test
  pins the three new constants + a no-credential-print guard
  (`print(os.environ`/`print(key|api_key|OPENAI_API_KEY)` banned in
  code cells).
- **Baselines**: 2499 → **2505 passed, 2 skipped** (+6). Focused
  gates: smoke+client 40/1sk (was 34/1sk), three boundaries 70;
  offline CLI exit 0 / COMPLETED. Live path implemented + gated but
  unexercised (no `OPENAI_API_KEY`) — the 2,048/8,000 controls are
  verified against fakes and the real request builder, never a real
  provider call.

### O. Generic ReAct context-budget safety (2026-10-19)

- **Audit (why)**: the production `Agent` loop already defends itself
  (`_gate_model_request` on every model call, working-context
  compaction, the tool-protocol response ceiling) — but those defenses
  live inside that one heavyweight class. Any hand-rolled ReAct loop
  calling `LLMClient.complete()` directly (a different graph node, a
  future agent runtime, the smoke notebook's fakes) inherited NONE of
  them: requests were unmeasured until the provider rejected them,
  observations were unbounded before joining history, and there was no
  configurable stop condition. The smoke milestone's Planner-only
  caps (`2048`/`8000`) are an EXPERIMENT boundary, not a fix — the
  fixes had to be reusable and graph-agnostic.
- **Placement decision (layering, deliberate)**: the module lives at
  the package ROOT — `src/repo_intelligence/react_safety.py`, beside
  the `tool_dispatcher.py` agent↔tools bridge — because
  `tests/agent/test_agent.py::test_agent_sources_do_not_import_
  repository_tools` + `tests/agent/test_agent_graph_tool.py::
  test_agent_sources_import_no_tools_graph_or_engineering_layers`
  pin "the agent package may DESCRIBE tools but never IMPORT the
  tools/graph/engineering layers". It imports `tools.protocol`'s
  `MAX_TOOL_RESPONSE_CHARS` directly (single source of truth, no
  duplicate size constant — unlike a hard-coded literal) while the
  `agent/` sources stay layer-clean. Both guard tests + the provider
  import guard stay untouched (no silent weakening; verified green).
- **Four composable pieces, one budget system**:
  * `BudgetedLLMClient(inner, policy=, subject=)` — wraps ANY
    `LLMClient`; EVERY `complete()` routes through the existing
    `enforce_request_budget_or_raise` (complete request = messages +
    tool definitions + structured-output schema, same H1 estimator).
    Oversized → `RequestBudgetExceededError` locally with the record
    (`.record`), ZERO provider calls, never sent, never retried
    unchanged. `.records`/`.sent`/`.rejected` lists for inspection;
    `.calls_reached_provider` counts only requests actually handed to
    the inner client — a budget rejection does NOT consume a
    provider-call allowance, a provider failure DOES.
  * `bound_observation(text, max_chars=MAX_TOOL_RESPONSE_CHARS,
    label=)` → `(bounded, record)` — line-aligned truncation BEFORE
    the observation joins history; record keys
    `label/original_chars/returned_chars/char_limit/truncated/note`;
    note carries the continuation hint ("re-read with a narrower
    range…"); deterministic; whole-line cut first, character prefix
    only when one line cannot fit; `max_chars <= 0` → `ValueError`.
    Per-observation bounds do NOT guarantee the total request fits —
    the client boundary remains the authority.
  * `ReActLoopGuard(config)` — `allow_iteration()/allow_tool_call()/
    allow_model_call()` (all refuse once a terminal status is set —
    stop-on-exhaustion), `begin_iteration()`, `record_tool_call()`,
    `record_model_response(response, budget_record=)` (reported
    `Usage.input_tokens` preferred, else the record's estimate —
    tracked SEPARATELY as `cumulative_reported_input_tokens` vs
    `cumulative_estimated_input_tokens`; `cumulative_input_tokens`
    sums both bases against `max_cumulative_input_tokens`),
    `record_budget_rejection(record)` → `REQUEST_BUDGET_EXCEEDED`,
    `stop(reason)`, `terminal_status()`, `snapshot()`. Terminal
    statuses: `ITERATION_LIMIT_REACHED` / `TOOL_CALL_LIMIT_REACHED`
    / `MODEL_CALL_LIMIT_REACHED` / `CUMULATIVE_BUDGET_EXCEEDED` /
    `REQUEST_BUDGET_EXCEEDED` (constant home: `ReActTermination`).
  * `restrict_history(messages, max_chars, protected_roles=("system",
    "developer"), keep_recent=2)` → `(msgs, record)` — the per-request
    safety net: drop OLDEST non-protected UNITS (tool call + result
    travel whole as one atomic pair), protected messages kept,
    the caller's transcript NEVER mutated (returns a new list);
    record keys `dropped_units/dropped_chars/original_chars/
    returned_chars/char_limit/within_limit`. Per-request only — it
    does NOT accumulate across turns; the cumulative allowance in the
    guard covers run-wide growth.
  * `ReActSafetyConfig` (frozen dataclass) validates every limit
    (int ≥ 1, bool rejected) and `request_budget_policy()` returns
    the UNTOUCHED global `RequestBudgetPolicy()` when no input budget
    is set, else a tighter LOCAL `RequestBudgetPolicy(max_input_
    tokens=..., context_capacity_tokens=None)` — constructing configs
    never mutates global defaults (test-pinned: default still 95,232).
  * Provider neutrality: no `openai`/`gpt-`/`response_format` tokens
    in the module (checked by test); no second tokenizer or
    `CHARS_PER_TOKEN` definition — `enforce_request_budget_or_raise`
    + `RequestBudgetPolicy` only.
- **Notebook §11/§12** (`notebooks/langgraph_one_investigator_smoke_
  test.ipynb` now 12 sections / 10 code cells): §11 "Shared ReAct
  context-safety demonstration (offline)" defines fake clients + a
  fake tool and runs 8 demos: D1+2 — 4-iteration loop, history
  growth `[48, 1848, 3648, 5464]`, COMPLETED, per-call records;
  D3 — 50,000-char observation → 3,894 (limit 4000); D4 —
  per-observation cap 2000 vs request budget 4000 chars, dispositions
  `accepted×3, reduced×2` (H2 reduces before it rejects), combined
  history exceeds the budget; D5 — complete-request rejection
  (`tool_or_response_contract_exceeds_budget`, est 3,768 > 1,000
  tokens, 0 provider calls, `sent == []`); D6 — `restrict_history`
  drops 10 units / 20,155 chars, system kept, pairs whole,
  transcript unmutated; D7 — iteration AND tool-call exhaustion →
  terminal statuses, every later action refused; D8 — a SECOND dummy
  agent runs the identical classes (source-scan test: no agent/task
  name branching). Section ends `assert PROVIDER_GUARD.calls == 0`
  (the armed class guard is never touched — fakes are not
  `OpenAILLMClient`). §12 is "Results, limitations, and next steps":
  Planner-only limits are NOT the fix (the shared controls are),
  estimate honesty (chars-per-token ratio, not a tokenizer), per-
  request vs cumulative distinction, no monetary cap claimed.
- **Tests** (+19): `tests/test_react_safety.py` (17 — policy
  boundary: config validation, local-vs-global policy separation,
  global default still 95,232; provider boundary: every outgoing
  request checked (records == complete attempts across iterations),
  complete-request measurement (messages + tool defs + schema all
  counted), oversized → 0 provider calls + `.sent == []`, rejection
  does not consume the allowance while a provider failure does,
  accepted request sent byte-identical; observation bounding before
  history, metadata/determinism, many individually-acceptable
  observations still hit the combined boundary (reduced/rejected,
  nothing over-budget sent); `restrict_history` provenance +
  no-mutation; iteration/tool/model/cumulative exhaustion → terminal
  statuses; two dummy agents share the same abstractions + source
  has no agent/task names + reuses project machinery (no competing
  budget, real `MAX_TOOL_RESPONSE_CHARS` identity)) + 2 smoke tests
  (§11 demos ran offline with the notebook namespace — guard_a
  COMPLETED, `budgeted_c` rejected/zero provider calls, identity of
  shared classes, `PROVIDER_GUARD.calls == 0`; §11 source is offline
  (no `OpenAILLMClient(` construction) and §12 points at the shared
  controls / "not by Planner-only limits").
- **Guard interaction, resolved deliberately**: the module was
  ORIGINALLY created under `agent/` and the two layering guards
  failed it (it imports `tools.protocol`). Rather than carve an
  exception (weakening a pinned boundary), the module moved to the
  package root where the `tool_dispatcher.py` precedent lives; the
  test moved with it (`tests/test_react_safety.py`, root tests like
  `test_tool_dispatcher.py`). All notebook/README/AGENTS references
  updated to `repo_intelligence.react_safety`; generator script
  `/tmp/opencode/insert_react_section.py` kept in sync.
- **Baselines**: 2505 → **2524 passed, 2 skipped** (+17 react_safety
  + 2 smoke §11). Focused gates: react 17, smoke+client 42/1sk (was
  40/1sk), layering guards (test_agent + graph_tool +
  tool_definitions) green, three boundaries 70, orchestration 301;
  offline CLI exit 0 / COMPLETED on BOTH backends; notebook 22 cells
  sequential-exec green (0 provider calls). Honest limits: token
  figures are ESTIMATES (4 chars ≈ 1 token ratio, no tokenizer, no
  provider-side count); `restrict_history` is per-REQUEST (cumulative
  control is the guard's separate allowance); no monetary cap;
  `LLMClient`-level allowances in any real deployment must still be
  set by the caller (the notebook pins them; the module does not
  invent defaults beyond validated 8/8/16). Live `--live` NOT run
  (no `OPENAI_API_KEY`) and not claimed.

### P. ReAct safety integrated into the application path + workflow validation (2026-10-19)

- **Call-path audit (what actually runs)**: exactly six `.complete(`
  sites exist in `src/` — the three `agent/agent.py` calls (loop
  `_execute_loop`, planner `_create_plan`, boundary finalization),
  all behind `_gate_model_request`; `repository_query._build_plan`
  (planner preflight → `RequestBudgetExceededError` → `PLAN_FAILED`,
  never reaches the graph); `langgraph/planner.py::get_plan`; and
  `llm/__main__.py`. Graph nodes make NO direct model calls (the
  graph selects/commits/retries at task level; execution dispatches
  through `AgentTaskExecutor` → participants → the same Agent loop).
  There is therefore NO ungated provider call in the application,
  and every execution path with equivalent protection was REUSED:
  deliberately did NOT wrap the shared client with
  `BudgetedLLMClient` (a reject-only wrapper would pre-empt H2's
  reduce path and turn REDUCED runs into failures — competing
  enforcement), and did NOT add `ReActLoopGuard` beside the Agent's
  own `max_tool_calls` (competing policy object).
- **The one real gap, fixed minimally**: tool observations entered
  the working conversation UNBOUNDED at the append seam in
  `_execute_loop` — the tool layer bounds its own results, but the
  seam guaranteed nothing for any non-compliant/custom result. The
  seam now runs `react_safety.bound_observation` (LAZY import —
  agent/ textual imports stay free of the `tools` layer, both
  layering guard tests untouched/green): identity byte-identical
  for tool-protocol-compliant results (same serialized string the
  ceiling already measures), line-aligned excerpt + continuation
  note otherwise; the TRACE's evidence copy (success/source, its own
  `TRACE_DATA_CHAR_LIMIT`) is untouched. New lifecycle ops: none
  (identity in every existing run — no pinned-set churn).
- **Limits mapping (no new budget)**: tool-call exhaustion keeps
  its established terminal `EXECUTION_LIMIT_REACHED` (re-asserted
  after the seam change); iterations are structurally bounded (a
  response without tool calls always terminates, and dispatch is
  budget-blocked) — no separate iteration config added; the
  cumulative input allowance is NOT supported by the Agent and was
  NOT added (that would be a new context-budget feature — out of
  scope; `ReActLoopGuard` carries cumulative control for NEW loops,
  as §11 demonstrates). Global defaults unchanged (95,232 still
  test-pinned inside the new integration tests).
- **Tests (+5)**: `tests/agent/test_react_integration.py` (4 —
  real `Agent` loop with fake client/dispatcher: oversized
  observation bounded before the NEXT request (≤ 24,000, truncation
  marker + continuation note) while trace evidence fields stay
  intact and the follow-up preflight accepts; three model calls →
  three independent budget records over the grown history
  (message_count strictly grows, all accepted @ 95,232); compliant
  observation byte-identical through the seam; tool-call exhaustion
  still raises `AgentError(EXECUTION_LIMIT_REACHED)` with the
  bounded observation present) + `test_repository_query_langgraph.
  py::test_safety_records_ride_every_model_call_of_the_graph_
  workflow` (offline graph e2e: all 3 agent trace files, every
  iteration carries an accepted `request_budget` @ 95,232, ≥ 4
  iterations, exit 0). Brief items 1/3/4/6/7 already covered by
  existing suites (H1 rejected-request test, bounded/trace
  exhaustion tests, smoke 21, H1–H4 gates) — not duplicated.
- **Workflow validation (§5)**: offline CLI exit 0 / COMPLETED /
  SUCCESS on BOTH backends (graph banner + validation passed);
  production-seam file 12 passed (incl. oversized-planner pre-graph
  rejection, e2e COMPLETED, legacy↔graph field parity);
  orchestration 301; notebook 22 cells sequential-exec offline with
  `PROVIDER_GUARD.calls == 0`. Real graph-run artifacts: the three
  newest `trace.json` files (2 investigators × 4 iterations + 1
  analyst × 1) — every iteration gated, `accepted`, 95,232.
- **Notebook (minimal corrections only)**: §11 gained the
  integration paragraph (same `bound_observation` in the production
  Agent loop; H1/H2 + `max_tool_calls` reused, not re-wrapped;
  pointed at `test_react_integration.py`); §12 gained a
  "production-path integration proven outside this notebook" bullet
  and an explicit limitation that **the optional live Planner call
  does NOT validate the multi-agent workflow** (workflow proof =
  offline graph execution + integration tests). Gates untouched:
  `LIVE_MODEL_TEST`/`CONFIRM_LIVE_CALLS` both False, 8,000/2,048/
  `max_calls=1`/`max_retries=0` pinned by their existing tests.
- **Baselines**: 2524 → **2529 passed, 2 skipped** (+4 integration
  + 1 graph-workflow). Focused gates in brief order: integration
  group 33, react 17, three-boundaries + llm 152/1sk, orchestration
  + notebook + smoke 326, full 2529/2; CLI both backends exit 0.
  Honest limits: everything above is OFFLINE (no `OPENAI_API_KEY`);
  token figures remain ratio-based estimates; cumulative-run budget
  enforcement does not exist in the Agent (documented, not
  implemented); no monetary cap.

### Q. Visualizer id-copy prompt hardening (2026-10-19)

- **Open item §7 addressed at the prompt level**: live runs
  sometimes shortened report ids (`dag-agent` instead of the full
  artifact id) → `scope_validity` rejected the diagram → exit 3.
  The validator working correctly and was NOT weakened.
- **Prompt change** (`prompts/architecture_visualization.py`, V1
  text amended in place): RULES gained a "never shorten, rename, or
  'clean up' an id taken from the supplied reports — copy it
  byte-for-byte" bullet; OUT gained the `IDs are copied verbatim,
  never shortened` paragraph with right/wrong few-shot examples
  (node id `artifact:agent_definition:internal/agents/dag-agent.md`
  vs `dag-agent`, and an edge endpoint) using HYPOTHETICAL paths —
  repository-agnosticism checks intact (no ADOP/orchestrat/planner;
  headers and vocabularies unchanged).
- **Version stays V1, deliberately**: the protected demo
  (`agentrun_architecture_visualization_demo.py:982`) asserts
  `VISUALIZATION_PROMPT_VERSION == "V1"` inside its Phase-1 `ok`
  condition — an in-place content amendment preserves the pin; a
  V1→V2 bump (flow-investigator style) would make the protected
  demo report `scope condition: NO` without the file being modified
  (forbidden). Documented in
  `docs/architecture-visualization-agent.md` §12.
- **Tests (+1)**: `test_prompt_forbids_shortened_ids_with_a_few_shot_
  example` (rule bullet, example pins, banned-words intact).
  Baselines: 2529 → **2530 passed, 2 skipped**. Focused: viz trio
  (visualization + demo + m17) 91 passed; offline CLI exit 0 /
  COMPLETED; protected viz demo exit 0 with `scope condition: yes`
  (prompt length now 5,267 chars — printed dynamically, not pinned).
  Honest limit: the shortening failure was live-model behavior — the
  prompt hardening is verified offline only; next live run is the
  real confirmation (no `OPENAI_API_KEY` here).

### R. LangGraph create_agent POC with react_safety middleware (2026-10-19)

- **Scope**: an optional, fully-tested POC package
  (`langgraph_poc/`, extra `poc`) proving the generic `react_safety`
  controls adapt as REAL langchain `AgentMiddleware` around a REAL
  `create_agent` investigator inside a LangGraph `StateGraph`
  (planner → query router → orchestrator select → investigator →
  orchestrator receive → final response), reusing the existing
  contracts. NOT on the production path (legacy orchestrator + M21
  opt-in task loop remain the production paths).
- **Environment**: `langchain==1.4.6`, `langchain-core==1.6.11`,
  `langchain-openai==1.7.2` installed; langchain-openai requires
  `openai>=2.45` → venv openai 1.99.1 → **3.35.1** (+ `tiktoken
  0.14.0`, `regex 2026.9.29`); `pip check` clean; `tests/llm/`
  green on 3.35.1 (the provider-neutral contract does not pin the
  SDK major). Declared as optional extra `poc` + requirements pins.
- **Guard policy update (deliberate, documented)**:
  `tests/llm/test_client_contract.py::PROVIDER_IMPORT_RE` dropped
  `langchain` (framework, not provider SDK — M21 precedent for
  `langgraph`) and ADDED `langchain_openai` (provider adapter,
  banned everywhere). New tests: langchain/langchain_core imports
  confined to `langgraph_poc/`; the POC package never imports any
  provider SDK. Orchestration framework-ban test untouched.
- **Middleware** (`langgraph_poc/middleware.py`): three
  `AgentMiddleware` classes over the EXISTING controls —
  `RequestBudgetMiddleware` (`wrap_model_call`, every call;
  `enforce_request_budget_or_raise`; reduced messages written back
  via `ModelRequest.override`; rejection raises
  `RequestBudgetExceededError`), `ExecutionLimitMiddleware`
  (`ReActLoopGuard` on both hooks; typed
  `ReActExecutionLimitExceeded` with the guard's terminal status;
  `note_budget_rejection` marks the guard terminal WITHOUT
  consuming a model call), `ObservationBoundingMiddleware`
  (`bound_observation` before the ToolMessage joins history).
  Order pinned: budget outermost (rejection beats the model-call
  limit on a shared boundary call — test-proven).
- **Tool seam** (`tools.py`): existing bound tool callables +
  `AGENT_TOOL_DEFINITIONS` schemas as `StructuredTool`s; unbound
  names skipped, never silently wrapped.
- **Graph** (`graph.py`): `LangGraphPocState` (TypedDict,
  total=False); planner uses `build_planning_request`/
  `plan_from_response`/`validate_plan` via injected `LLMClient`
  + the POC four-step shape (P1-P4, model-authored goals);
  router uses `assign_roles` with overrides P1-P3=INVESTIGATOR,
  P4=ANALYST (metadata only — no Analyst runs); select takes the
  FIRST step + `TaskPayload`; investigator runs the
  `create_agent` agent (structured response from the agent
  state's `structured_response`, type-checked); receive
  re-validates the report; every failure routes to
  `final_response` with an EXPLICIT category
  (PLANNER_FAILED/ROUTER_FAILED/SELECTION_FAILED/
  REQUEST_BUDGET_EXCEEDED/MODEL_CALL_LIMIT_REACHED/MISSING_
  STRUCTURED_OUTPUT/RECEIVE_FAILED/INVESTIGATOR_FAILED) —
  `report=None`, empty completed steps. Two independent limits:
  `GRAPH_RECURSION_LIMIT=25` (outer) vs
  `AGENT_RECURSION_LIMIT=12` (agent-internal), both separate
  from the middleware ceilings.
- **Tests** (25 new): 23 in `tests/test_langgraph_poc.py`
  (importorskip) — happy path with validated report; planner
  contract + four-step shape; router coverage; only P1 executes
  (single TASK human message ever seen by the model); real
  create_agent path (spy observes the ToolMessage before the
  structured turn); analyst metadata-only; invalid payload /
  three-step plan / tool-level goal / empty scopes all explicit;
  budget rejects the initial request BEFORE the model (spy.calls
  == 0, guard model_calls == 0); budget rejects the SECOND
  request after a tool observation (probe-determined ceiling;
  dispositions [accepted, rejected]; guard terminal
  REQUEST_BUDGET_EXCEEDED); budget OUTERMOST of the execution
  guard (max_model_calls=1 + oversized second → budget wins);
  oversized observation (50k) bounded to 4k before the next model
  call; execution limit stops after exactly 2 model calls;
  budget failure = one attempt, zero model calls, no retry;
  safety failure → FAILED final with no report; provider-neutral
  request projection; source-scan reuse + default policy still
  95,232; only bound tools wrapped — + 2 guard tests in
  `tests/llm/test_client_contract.py`.
- **Demo** (`agentrun_langgraph_poc.py`): offline scripted
  default exit 0 (budget records accepted ×2 ~2.9k tokens each,
  SUCCESS, P1 completed, verbose prints the guard snapshot);
  `--live` builds `ChatOpenAI` + `OpenAILLMClient` +
  `live_bindings` from the same `OPENAI_API_KEY` seam (gated,
  unexercised here).
- **Docs**: `docs/langgraph-poc-create-agent.md` (Mermaid of the
  ACTUAL graph, middleware order, failure taxonomy table, honest
  limitations); this entry; §1/§2/§4/§5 updates; README pointer.
- **Baselines**: 2530 → **2555 passed, 2 skipped** (+23 POC +2
  guard). Focused: POC 23, POC+react 40, llm+orchestration
  385/1sk, layering guards 44; offline CLI exit 0/COMPLETED;
  POC demo exit 0. Live `--live` NOT run (no `OPENAI_API_KEY`)
  and not claimed.

---

## 7. Open item

**Visualizer node-id compliance (live runs only) — prompt hardened
2026-10-19 (entry Q), STILL live-unverified.** Live model sometimes
shortens report ids (`dag-agent` instead of the full artifact id) →
diagram `scope_validity` rejects → exit 3. The validator working is
by design and was NOT weakened. The prompt
(`src/repo_intelligence/agents/prompts/architecture_visualization.py`)
now carries an explicit copy-verbatim RULES bullet plus a few-shot
right/wrong example in OUT (node id + edge endpoint, hypothetical
paths). Version deliberately stays `V1`: the protected demo asserts
`VISUALIZATION_PROMPT_VERSION == "V1"` in its Phase-1 ok condition,
so the fix was an in-place CONTENT amendment (a V1→V2 bump would
break the protected file without modifying it). Pinned by
`test_prompt_forbids_shortened_ids_with_a_few_shot_example`.
Cannot be confirmed in this environment (no `OPENAI_API_KEY`) — next
live run should watch for uncropped ids and exit-0 diagrams.

---

## 8. Session safety rules

1. Use `venv312/bin/python` for everything in this repo. Never touch
   `.venv/` or pyenv 3.7.
2. No git operations (not a repo; no commit/push/init).
3. Do not modify the two protected demo files unless explicitly asked.
4. After any code change: run the three verification commands in §4 and
   compare against the baselines.
5. Keep `.format()` / ≤79-char style when editing `src/`; match the
   surrounding code.
6. Never invent vocabularies, node ids, or evidence — reuse canonical
   contracts; validators must keep rejecting non-compliant model output.
7. Live runs need `OPENAI_API_KEY` (not available here) — verify with
   offline mode.
