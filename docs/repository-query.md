# Repository Query — one natural-language question, one evidence-backed answer

The user-facing entry point of the whole stack: a thin CLI that asks
one repository one question and reports what the existing machinery
established — the answer, its evidence, the collaboration and diagram
verdicts, the investigation/run identifiers, where the trace lives,
which artifacts were written, and the exact exit code that encodes all
of it.

The repository is this project's **pinned ADOP snapshot**
(`artifacts/main_2026-09-22_ef11dac/repository/sample-Agentic-Ai-Data-Operations`,
commit `ef11dac…`), resolved from the script's own project root and
verified before anything runs; `--repository PATH` points the same
pipeline at a different, unpinned checkout.

It composes **no new framework**: no second agent runtime, no second
planner, no second orchestrator, no memory, no context engine, no new
validator, no renderer, no clone of the subject. The one piece written
for this entry point is the bounded, evidence-driven investigation
script (`repo_intelligence/offline_investigation.py`) that derives the
offline answer from files it actually read (§3); everything else
already existed after M17 and the post-M17 specialist capabilities,
and this document describes only how they are wired together and
rendered.

One sentence: **ask a repository a question, get one answer you can
trace back to files, with a trust verdict attached.**

Written for whoever will run, extend or diagnose the query. §1–§4 are
what a user needs; §5–§9 are the contracts; §10–§13 are the
implementation, failures, and tests.

---

## 1. Why it exists

After M17 the project can already do every step individually: plan a
multi-step investigation, run an Architecture Investigator, run a Flow /
Dependency Investigator, turn both into a validated diagram, render
artifacts, record an M16 collaboration verdict, and persist a trace.
What was missing is the front door — one command that takes a sentence
and returns an answer a person can read, in a format a script can gate
on (exit code), without knowing any of those steps exist.

Responsibility:

- take one question and a repository root;
- run the existing pipeline once;
- render the observable result (§5);
- map the result onto the documented exit codes (§4).

Non-responsibilities: it invents no facts, builds no new plan, writes
its own nothing, and never converts a failure into a success.

## 2. Usage

```bash
python agentrun_repository_query.py \
    "Explain how the agents, tools, and MCP servers in this \
repository interact."
```

The default subject is **not** this analyzer's own tree: without
`--repository` the query analyzes this project's pinned ADOP snapshot —
`artifacts/main_2026-09-22_ef11dac/repository/sample-Agentic-Ai-Data-Operations`
at commit `ef11dac8618b6344d6a5e4add6ef51cd2811d7e0` — and the commit
is verified *before* anything runs (§5, §9). An explicit
`--repository PATH` is a different subject and is not pinned to a
commit.

Every flag:

| Flag | Meaning |
| --- | --- |
| `QUERY` (positional) | the question; may be omitted to start the interactive session |
| `--query QUERY` | the same question in flag form (never both) |
| `--repository PATH` | root to investigate (default: the pinned subject snapshot) |
| `--live` | use the configured live LLM provider instead of the deterministic offline model (§3) |
| `--verbose` | print the per-step progress lines of the run |
| `--no-artifacts` | write nothing (the diagram is still validated) |
| `--no-diagram` | diagram out of scope: no diagram validation, no artifacts (§8) |

Interactive — run it with no query:

```bash
$ python agentrun_repository_query.py
Repository Intelligence — interactive (type exit or quit to leave, end-of-input to finish)

query> how do the tools work
...
exit code: 0

query> exit
```

`exit` / `quit` (case-insensitive) leave; end-of-input (Ctrl-D) leaves;
an empty line re-prompts; **one failed query never ends the session** —
the loop, not the query, decides when the session is over.

The demonstration targets remain available through `--repository`:
`tests/fixtures/arch_mcp` (three files — an agent, a tool, an MCP
configuration) and `tests/fixtures/query_evidence` (README with a
numbered agent heading, an MCP configuration, an agent definition, a
tool). Both are answered from their own files, and the pinned default
already carries an MCP configuration and a tool registry, so the
default run establishes the agent → tool → MCP server chain from the
subject's own evidence (§11).

## 3. Two modes — the caller's decision, never the environment's

| Mode | Selected by | What runs |
| --- | --- | --- |
| **offline** (the default) | no flag | the evidence-driven investigation script (`repo_intelligence/offline_investigation.py`): a bounded search → read → derive loop over the repository's real read-only tools, fully deterministic, no network, no key |
| **live** | `--live` | the *same* composition talking to the configured provider (`OpenAILLMClient` over the project's provider-neutral LLM layer) |

Rules the mode keeps:

- **offline ignores a provider key that happens to be set.** The
  default is the default; nothing about the environment silently
  changes what a plain command does.
- **`--live` never falls back.** When the provider cannot be
  constructed — no `OPENAI_API_KEY` after `load_env()`, or the
  `openai` package not installed — the run stops with its own block
  (§9) and exit `1`. It never degrades to the offline script and
  never reports a failure as a success.
- **The banner always says which mode ran.**

```text
Mode:       offline (deterministic, scripted models — no network)
```

`--live` without a usable provider is a whole block of its own, and no
investigation is executed:

```text
Repository Intelligence
────────────────────────────────────────

ERROR: Live mode unavailable.

Live mode requested but the configured LLM provider is unavailable.
Set the required configuration and retry.

Reason:
  OPENAI_API_KEY is not configured. Set it in .env (see .env.example) or in the environment before constructing OpenAILLMClient.

No investigation was executed.
exit code: 1
```

## 4. Exit codes (the contract)

| Code | Meaning |
| --- | --- |
| `0` | answered and everything in scope validated |
| `1` | the investigation failed (a requested artifact could not be written, the subject snapshot could not be verified (§5), or `--live` had no usable provider (§3)) |
| `2` | invalid usage — empty query, both query forms, `--repository` that is not a directory, unknown flag |
| `3` | validation failed — the M16 collaboration verdict, or the six-rule diagram verdict while the diagram is in scope |
| `130` | interrupted (Ctrl-C) |

`exit_code(result)` in `src/repo_intelligence/repository_query.py` is
the single mapping; the CLI never recomputes it. The last line of a run
prints it, so a captured transcript carries its own verdict:

```text
exit code: 0
```

## 5. Output (the §12 template)

Sections, in order, all from recorded execution facts — no invented
narrative, no raw JSON, no event dump:

```text
Repository Intelligence
────────────────────────────────────────

Subject repository:
artifacts/main_2026-09-22_ef11dac/repository/sample-Agentic-Ai-Data-Operations

Pinned commit:
ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

Question:
How do the agents, tools, and MCP servers interact?

Repository: sample-Agentic-Ai-Data-Operations   target: README.md
Mode:       offline (deterministic, scripted models — no network)

Status:     COMPLETED   termination: SUCCESS

Final answer
────────────────────────────────────────
<the one answer>
  - <conclusion statements>

Evidence
────────────────────────────────────────
  - [classification] path[:start[-end]]

Uncertainties
────────────────────────────────────────
  - ...

Agent execution summary
────────────────────────────────────────
  ✓ S1   INVESTIGATOR  architecture   <goal>
  ✓ S2   INVESTIGATOR  flow           <goal>
  ✓ S3   ANALYST       visualization  <goal>

Investigation ID: inv-arch-<hex>

Runs:
  Architecture Investigator: <run id>
  Flow / Dependency Investigator: <run id>
  Architecture Visualization: <run id>

Validation
────────────────────────────────────────
  collaboration: passed (M16 deterministic verdict)
  diagram:       passed (six deterministic rules)

Artifacts
────────────────────────────────────────
  ✓ diagram_spec.json  artifacts/architecture_visualization/diagram_spec.json
  ○ architecture.svg — unavailable — existing renderer does not support SVG
  ...

Trace
────────────────────────────────────────
  runtime/agent_runs/<run id>
  ...

exit code: 0
```

Subject identity — `Subject repository:` is where the query actually
read (project-relative whenever the snapshot lives under this project);
`Pinned commit:` is the commit that was **verified before the run**, or
`-` when an explicit, unpinned `--repository` carries no provenance.
`Repository:` names the M1 snapshot record (name + the target file), so
the answer always says which repository and which commit it is about.

### Verification comes first

The default subject is analyzed only if its commit is the pinned
`ef11dac8618b6344d6a5e4add6ef51cd2811d7e0`. Provenance is read from
Git when the checkout has `.git`, otherwise from the M1 record beside it
(`…/main_2026-09-22_ef11dac/snapshot/repository_snapshot.json`). If the
directory is missing, the commit mismatches, or neither source can
prove it, **no investigation is executed** and this block is the whole
output (exit `1`):

```text
Repository Intelligence
────────────────────────────────────────

ERROR: Subject repository snapshot verification failed.

Expected commit:
  ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

Resolved repository:
  artifacts/main_2026-09-22_ef11dac/repository/sample-Agentic-Ai-Data-Operations

Reason:
  commit mismatch (expected ef11dac…, found 0000000…)

No investigation was executed.
exit code: 1
```

Rules the renderer keeps:

- **Paths are relative.** Everything printed is project-relative
  (`artifacts/…`, `runtime/…`); an absolute machine path never appears
  in a default run.
- **The SVG tick is earned.** `✓ architecture.svg` is printed only when
  `verify_artifacts` says the file exists on disk; otherwise
  `○ architecture.svg — unavailable — <renderer's own notice>`. An
  artifact that was never produced is reported, never claimed.
- **Failures keep their message.** `Failures` lists the orchestrator's
  own messages; an empty answer reads `(no answer — the investigation
  did not complete)`.
- **Progress is post-hoc.** `--verbose` (and every interactive query)
  prints the per-step lines from the run's own records — there is no
  live event stream to stream from, and none is invented.

## 6. Architecture at a glance

The chain one query executes (all existing components):

```text
question + subject root
        ↓
default subject           artifacts.py.repository_dir (pinned snapshot)
        ↓  only when the subject is the pinned snapshot
verify commit             repository_query.verify_subject_snapshot
        ├─ mismatch / missing / unprovable → §5 error block, exit 1
        ↓ verified (or an explicit, unpinned --repository)
resolve target            repository_query.resolve_read_target
        ↓ target file
model build              repository_query.build_offline_models   (default)
                         repository_query.build_live_models      (--live)
        │                 offline: a bounded read-only loop → evidence →
        │                 InvestigationModels; live: the same roles
        │                 filled from the configured provider
        ↓
M10 plan                  planning.planner + planning.validation
        ↓ validated plan
M17 Orchestrator          orchestration.Orchestrator
        ├─ S1 INVESTIGATOR  architecture  → ArchitectureInvestigator → ArchitectureReport
        ├─ S2 INVESTIGATOR  flow          → FlowInvestigator         → FlowDependencyReport
        └─ S3 ANALYST       visualization → ArchitectureVisualizationAgent → DiagramSpec
        ↓
M16 collaboration verdict (deterministic)      → status
        ↓
six-rule diagram validation  agents.diagram_validation → diagram verdict
        ↓ only when the verdict passes
renderer + writer          agents.visualization_artifacts → 4 artifacts
        ↓
one answer + evidence + uncertainties  (repository_query._assemble)
        ↓
exit_code(result)          0 / 1 / 3
```

Routing (which investigator step goes to which specialist) is the
application's own `QueryRouter`: plan step ids first (`S1`, `S2`), then
a memo of already-routed ids (retry-stable), then the goal's content,
then order assignment. A step none of those can route raises
`RepositoryQueryRouterError` — an unroutable step is an explicit
failure, never a guess. Neither the orchestrator nor the runner knows a
router exists, so orchestration stays unmodified.

## 7. What is reused (composition table)

| Step | Existing component | Not touched |
| --- | --- | --- |
| subject default | `artifacts.PRIMARY_SNAPSHOT_ID`, `repository_dir`, `snapshot_path`, `default_artifacts_root` | M1 snapshot layout |
| subject verification | `repository.git.get_head_sha` / `GitError` + the M1 record | Git tooling, snapshot format |
| target + investigation | `resolve_read_target` (the entry target) + `offline_investigation.InvestigatorModel` / `AnalystModel` (bounded search → read → derive) | tool layer, report models |
| models | `build_offline_models` / `build_live_models` over `llm` | model layer |
| plan | `build_planning_request`, `plan_from_response`, `validate_plan` | planner |
| memory | `MemoryManager`, `MemoryContext`, `MemoryOwner`, `MemoryScope` | memory schema |
| context | `ContextEngine`, `ContextBudget` | selection rules |
| trace | `agent.trace_sink.JsonFileTraceSink` | trace format |
| tools | `tools.read_file`, `tools.search_repository`, `fixture_bindings` | tool layer |
| specialists | `ArchitectureInvestigator`, `FlowInvestigator`, `ArchitectureVisualizationAgent` wrapped by `ObservedAgent` | the agents |
| orchestration | `orchestration.Orchestrator` + `restricted_dispatcher` / `tool_definitions_for` | roles, post-conditions |
| verdict | M16 deterministic collaboration verdict | evidence validator |
| diagram trust | `DiagramScope`, `agents.diagram_validation.validate_diagram_spec` | six rules |
| artifacts | `agents.visualization_artifacts.write_visualization_artifacts`, `verify_artifacts` | renderer, `RunSummaryContext` |
| rendering of the run | `repository_query` (dataclasses) + this CLI | — |

The only new code is this glue: `src/repo_intelligence/repository_query.py`
(composition + rendering contracts), `src/repo_intelligence/offline_investigation.py`
(the bounded offline investigation and its answer synthesis), and
`agentrun_repository_query.py`.

`tools.search_repository` is the M18 semantic retrieval tool: it reads
a per-repository FAISS index under
`db_store/rag/vector_store/<name>/<commit>/` (built with
`repo-intelligence retrieval-index` — see
[storage-layout.md](storage-layout.md) and
[m18-repository-retrieval.md](m18-repository-retrieval.md)). Without
that index every `search_repository` turn returns
`RAG_INDEX_NOT_FOUND` — a failed observation, never a failed run: the
loop keeps the candidates it discovers from files it has already read
(quoted paths, imports, YAML scalars) and its answer is still derived
only from evidence it actually read. Building the index once per
repository/commit upgrades those same turns to ranked semantic results:

```bash
python -m repo_intelligence.retrieval.index \
    --repository-root artifacts/main_2026-09-22_ef11dac/repository/sample-Agentic-Ai-Data-Operations
```

## 8. The two scope flags

Both are honored by the module; the CLI only renders the result.

**`--no-artifacts`** — keep the diagram in scope. The six-rule
validation still runs and still decides the exit code; nothing is
written.

```text
Validation
────────────────────────────────────────
  collaboration: passed (M16 deterministic verdict)
  diagram:       passed (six deterministic rules)

Artifacts
────────────────────────────────────────
  (not written — --no-artifacts; validation still ran)
```

**`--no-diagram`** — put the diagram out of scope. No six-rule
validation, no writer. The answer does not change shape: it is always
rendered from the merged investigation reports (§5), never from the
diagram finding, so an unvalidated diagram can never become the
answer. The analyst step still runs (role post-conditions cannot be
skipped), and the answer keeps its deterministic lead line:

```text
Repository: sample-Agentic-Ai-Data-Operations   target: README.md
Mode:       offline (deterministic, scripted models — no network)

Status:     COMPLETED   termination: SUCCESS

Final answer
────────────────────────────────────────
158 evidence-backed statement(s) established by the investigation.
components: 6 agent(s), 16 mcp_server(s), 66 tool(s) — e.g. add-lf-tags-to-resource, alarms, apply-column-security, athena-query, attach-group-policy, attach-user-policy
relationships: 1 DECLARES, 66 EXPOSED_BY
established path: ontology-agent -> DECLARES -> get-table -> EXPOSED_BY -> glue-athena (2 hop(s)).
  - ... (all 158 statements, unchanged — the diagram is
    simply not in scope)

Validation
────────────────────────────────────────
  collaboration: passed (M16 deterministic verdict)
  diagram:       skipped (--no-diagram)

Artifacts
────────────────────────────────────────
  (skipped — --no-diagram puts the diagram out of scope)

exit code: 0
``````

## 9. Failure modes

Every failure carries a reason and an exit code computed by
`exit_code(result)` — never by the CLI. No failure is rendered as a
success, and no failure ever switches mode behind the caller's back.

| When | What the run prints | `termination` | exit |
| --- | --- | --- | --- |
| the pinned subject is missing, the commit mismatches, or the checkout has no provenance | the §5 `ERROR: Subject repository snapshot verification failed.` block — the pipeline never starts | `SNAPSHOT_VERIFICATION_FAILED` | `1` |
| the entry target cannot be resolved | `Failures` plus `(no answer — the investigation did not complete)` | `NO_TARGET` | `1` |
| `--live` with an unusable provider (§3) | the `ERROR: Live mode unavailable.` block — the pipeline never starts | `LIVE_CONFIGURATION_UNAVAILABLE` | `1` |
| the model set could not be built | `Failures` | `MODEL_SET_FAILED` | `1` |
| `validate_plan` rejected the plan | `Failures` | `PLAN_FAILED` | `1` |
| a specialist step failed in the orchestrator | a `✗` row in the agent execution summary + `Failures` | `EXECUTION_FAILED` | `1` |
| the M16 collaboration verdict failed | `collaboration: failed (M16 deterministic verdict)` | `VALIDATION_FAILED` | `3` |
| the six-rule verdict failed while the diagram is in scope | `diagram: failed (six deterministic rules)` | `VALIDATION_FAILED` | `3` |
| a requested artifact could not be written | `✗ <file> write failed — <reason>` under `Artifacts` | the run's own reason | `1` |
| the investigation never completed | `Final answer` reads `(no answer — the investigation did not complete)` | — | — |

Rules these keep:

- `Failures` repeats the orchestrator's own messages verbatim
  (clipped, never rephrased into something friendlier);
- a run that never completed prints no invented conclusion — the
  answer section says exactly that it has none;
- `--live` failure is a block of its own (§3): no `Status:` line at
  all, because nothing ran;
- invalid usage never reaches the pipeline — argparse's own message
  on stderr and exit `2` (empty query, both query forms, a
  `--repository` that is not a directory, an unknown flag). Anything
  the CLI did not expect prints `error: <Type>: <message>` and exits
  `1`.

## 10. Artifacts and trace

Artifacts are written only while the diagram is in scope and the
six-rule verdict passed, into this project's artifact directory:

```text
artifacts/architecture_visualization/
  diagram_spec.json     the merged DiagramSpec (the projection)
  architecture.mmd      the Mermaid text
  architecture.svg      optional — only when the renderer supports it
  run_summary.md        the renderer's own run summary
```

Every line is earned: `✓ <file> <project-relative path>` only when
`verify_artifacts` confirms the file exists on disk, otherwise
`○ <file> — unavailable — <renderer's own notice>`; a write failure is
`✗ <file> write failed — <reason>`. `--no-artifacts` writes nothing
(validation still runs); `--no-diagram` collapses the section to one
`skipped` line (§8).

The trace lives outside `artifacts/`: one directory per specialist run
under `runtime/agent_runs/<run id>/`, listed under `Trace` exactly as
recorded. Together with the existing memory database that is the only
state a query leaves behind — no per-query store, no extra file.

## 11. What a default run looks like

The pinned ADOP snapshot, commit verified first, read offline in
about three seconds (measured: 2.7s end to end, index built). Complete
output, nothing elided:

```text
Repository Intelligence
────────────────────────────────────────

Subject repository:
artifacts/main_2026-09-22_ef11dac/repository/sample-Agentic-Ai-Data-Operations

Pinned commit:
ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

Question:
How do the agents, tools, and MCP servers interact?

Repository: sample-Agentic-Ai-Data-Operations   target: README.md
Mode:       offline (deterministic, scripted models — no network)

Status:     COMPLETED   termination: SUCCESS

Final answer
────────────────────────────────────────
158 evidence-backed statement(s) established by the investigation.
components: 6 agent(s), 16 mcp_server(s), 66 tool(s) — e.g. add-lf-tags-to-resource, alarms, apply-column-security, athena-query, attach-group-policy, attach-user-policy
relationships: 1 DECLARES, 66 EXPOSED_BY
established path: ontology-agent -> DECLARES -> get-table -> EXPOSED_BY -> glue-athena (2 hop(s)).
  - Component data-onboarding-agent (agent): Data Onboarding Agent [observed]
  - Component ontology-staging-agent (agent): Ontology Staging Agent (AWS Semantic Layer Handoff) [observed]
  - Component environment-setup-agent (agent): Environment Setup Agent [observed]
  - Component prompt-intelligence-agent (agent): Prompt Intelligence Agent (Self-Healing) [observed]
  - Component devops-agent (agent): DevOps Agent [observed]
  - Component glue-athena (mcp_server): MCP server registered in servers.yaml [observed]
  - Component create-database (tool): tool served by the glue-athena MCP server [observed]
  - Component get-table (tool): tool served by the glue-athena MCP server [observed]
  - Component get-tables (tool): tool served by the glue-athena MCP server [observed]
  - Component create-crawler (tool): tool served by the glue-athena MCP server [observed]

Evidence
────────────────────────────────────────
  - [observed, unknown] README.md
  - [observed, derived, unknown] tool-registry/servers.yaml
  - [observed, unknown] .mcp.json
  - [observed, derived, unknown] tool-registry/invariants.yaml
  - [observed, derived, unknown] .claude/rules/02-architecture.md
  - [unknown] runbooks/environment-setup-agent/agentcore/gateway/GATEWAY_ARCHITECTURE.md

Uncertainties
────────────────────────────────────────
  (none recorded)

Agent execution summary
────────────────────────────────────────
  ✓ S1   INVESTIGATOR  architecture   Document the architecture components and their rela...
  ✓ S2   INVESTIGATOR  flow           Trace the dependency path the question asks about t...
  ✓ S3   ANALYST       visualization  Interpret the investigated findings as one diagram ...

Investigation ID: inv-arch-5771f6c794dde8f051623a07f6b8751f

Runs:
  Architecture Investigator: 7eb986bb5f204470928450264340eb85
  Flow / Dependency Investigator: 2790d9075e61457b8845c3cabd0d7a98
  Architecture Visualization: 041f80664ec54e1a94a7d6aaac336815

Validation
────────────────────────────────────────
  collaboration: passed (M16 deterministic verdict)
  diagram:       passed (six deterministic rules)

Artifacts
────────────────────────────────────────
  ✓ diagram_spec.json  artifacts/architecture_visualization/diagram_spec.json
  ✓ architecture.mmd  artifacts/architecture_visualization/architecture.mmd
  ○ architecture.svg — unavailable — existing renderer does not support SVG
  ✓ run_summary.md  artifacts/architecture_visualization/run_summary.md

Trace
────────────────────────────────────────
  runtime/agent_runs/7eb986bb5f204470928450264340eb85
  runtime/agent_runs/2790d9075e61457b8845c3cabd0d7a98
  runtime/agent_runs/041f80664ec54e1a94a7d6aaac336815

exit code: 0
```

Every line above was derived from files the investigation read: the
agent components come from the numbered `## N. … Agent` headings in
`README.md`, the 16 MCP servers and 66 tools from
`tool-registry/servers.yaml` (corroborated by `.mcp.json`), the
`ontology-agent DECLARES get-table` edge from
`.claude/rules/02-architecture.md`, and the path is the walk the flow
investigator established over those same statements. The reads are
chosen by a documented, repository-agnostic score (registry and
architecture files first, breadth before depth), the budget is fixed
(six files, eight tool calls), and nothing about this subject is
written into the code.

Semantic discovery participates exactly as §7 describes: each
investigator issues its `search_repository` turns against the built
retrieval index (0 `RAG_INDEX_NOT_FOUND` in all three traces above).
When no index exists for the repository, those turns fail with
`RAG_INDEX_NOT_FOUND` in the recorded trace while the loop keeps the
candidates it derives from files it has already read — the run still
completes honestly, with fewer statements, and building the index
turns the same searches into ranked semantic results without changing
anything else.

When verification fails (wrong commit, missing snapshot, or a checkout
with no provenance) the run stops before the pipeline — see §5 for the
block and `exit code: 1`.

The chain is visible against a three-file fixture, where the same
script derives the whole story from an agent file, a tool file and an
MCP configuration. An unpinned checkout has no retrieval index of its
own, so build one first (once) to let its searches participate:

```bash
python -m repo_intelligence.retrieval.index \
    --repository-root tests/fixtures/arch_mcp

python agentrun_repository_query.py --repository tests/fixtures/arch_mcp \
    "Explain how the agents, tools, and MCP servers in this repository interact."
```

```text
Repository Intelligence
────────────────────────────────────────

Subject repository:
tests/fixtures/arch_mcp

Pinned commit:
-

Question:
Explain how the agents, tools, and MCP servers in this repository interact.

Repository: arch_mcp   target: agents/collector.py
Mode:       offline (deterministic, scripted models — no network)

Status:     COMPLETED   termination: SUCCESS

Final answer
────────────────────────────────────────
8 evidence-backed statement(s) established by the investigation.
components: 1 agent(s), 1 mcp_server(s), 1 tool(s) — e.g. collector, fetch-record, records
relationships: 1 EXPOSED_BY, 1 IMPORTS
established path: collector -> IMPORTS -> fetch-record -> EXPOSED_BY -> records (2 hop(s)).
The evidence does not establish: the wire-level protocol between these tools and the 1 MCP server(s) is not established by the inspected files
  - Component collector (agent): agent implemented by collector.py [observed]
  - Component fetch-record (tool): tool implemented by fetch_record.py [observed]
  - Component records (mcp_server): MCP server declared in .mcp.json [observed]
  - collector IMPORTS fetch-record [observed]
  - fetch-record EXPOSED_BY records [derived]
  - the inspected evidence establishes the path collector -> IMPORTS -> fetch-record -> EXPOSED_BY -> records (2 hop(s))

Evidence
────────────────────────────────────────
  - [observed, unknown] agents/collector.py
  - [observed, derived, unknown] .mcp.json
  - [observed, unknown] tools/fetch_record.py

Uncertainties
────────────────────────────────────────
  - the wire-level protocol between these tools and the 1 MCP server(s) is not established by the ins...

Agent execution summary
────────────────────────────────────────
  ✓ S1   INVESTIGATOR  architecture   Document the architecture components and their rela...
  ✓ S2   INVESTIGATOR  flow           Trace the dependency path the question asks about t...
  ✓ S3   ANALYST       visualization  Interpret the investigated findings as one diagram ...

Investigation ID: inv-arch-be17d1a9eb24317f2ad14d7e85f6a21e

Runs:
  Architecture Investigator: 0d1c09e1ac34438a82ec755b355c1a95
  Flow / Dependency Investigator: 2382044f3b2842bbbf276c27475d348b
  Architecture Visualization: 265cc8fe435d40df8f60a598e65c6809

Validation
────────────────────────────────────────
  collaboration: passed (M16 deterministic verdict)
  diagram:       passed (six deterministic rules)

Artifacts
────────────────────────────────────────
  ✓ diagram_spec.json  artifacts/architecture_visualization/diagram_spec.json
  ✓ architecture.mmd  artifacts/architecture_visualization/architecture.mmd
  ○ architecture.svg — unavailable — existing renderer does not support SVG
  ✓ run_summary.md  artifacts/architecture_visualization/run_summary.md

Trace
────────────────────────────────────────
  runtime/agent_runs/0d1c09e1ac34438a82ec755b355c1a95
  runtime/agent_runs/2382044f3b2842bbbf276c27475d348b
  runtime/agent_runs/265cc8fe435d40df8f60a598e65c6809

exit code: 0
```

`Pinned commit: -` marks it as an explicit, unpinned
`--repository`; the run still exits `0`. The wire-protocol
uncertainty is recorded because no inspected file names a transport —
on the ADOP snapshot it is absent because its own files do say so.
Without the fixture index the same command also exits `0`, with the
discovery limited to references inside the files it reads (5
statements — no `.mcp.json` hit); building the index is what makes
the search turn find it.

## 12. Testing

`tests/agents/test_repository_query.py` — offline, never the network.
Two fixtures redirect every run:

- `sandbox` — artifact, trace, memory and context roots under `tmp_path`
  (`visualization_artifacts.default_artifacts_root`,
  `trace_sink.default_trace_root`,
  `context_engine.engine.default_context_root`, `QUERY_DB_PATH`) with
  `OPENAI_API_KEY` removed;
- `relative` — `sandbox` plus the display root, so printed paths are
  relative exactly as they are for a default project run.

The CLI itself is loaded by path, exactly like the other
demonstrations.

Documented cases — §15 of the specification (the pinned subject):

| Spec case | Test |
| --- | --- |
| 1 — the default subject is the pinned snapshot | `test_default_subject_is_the_pinned_adop_snapshot` |
| 2 — an explicit `--repository` overrides it | `test_explicit_repository_overrides_the_default` |
| 3 — a missing default snapshot fails without a run | `test_missing_default_snapshot_fails_without_running` |
| 4 — a wrong commit stops before any investigation | `test_wrong_commit_stops_before_any_investigation` (+ `test_unverifiable_subject_is_refused`) |
| 5 — the correct commit proceeds to M17 | `test_correct_commit_proceeds_to_the_m17_workflow` |
| 6 — the subject reaches the agents | `test_subject_repository_reaches_the_agents` |
| 7 — regression: never the analyzer's README | `test_default_never_analyzes_the_analyzer_repository` (+ `test_success_produces_relative_paths_only`) |
| 8 — the final answer is in the output | `test_final_answer_is_displayed` |
| 9 — SVG is optional, Graphviz not a dependency | `test_svg_is_optional_and_graphviz_is_not_required` |

Plus the earlier contract cases: a positional query renders every
section, an empty query is invalid usage, a missing target exits 1, a
missing SVG is never claimed, interactive mode (exit, EOF,
query-then-quit, Ctrl-C → 130), `test_exit_code_contract`,
`test_display_path_is_project_relative`,
`test_resolve_read_target_reports_missing_targets`, the three
`QueryRouter` tests (static ids → goal → memo → order → refusal), the
two scope flags, `--verbose` progress, and the snapshot helpers
(`test_snapshot_metadata_is_read_from_the_m1_record`).

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
    tests/agents/test_repository_query.py -q
```

`tests/agents/test_repository_query_evidence.py` — the
evidence-driven run and the two modes, over two synthetic fixtures
(`tests/fixtures/query_evidence` and `…/query_evidence_alt`: a README
with a numbered agent heading, an MCP configuration, an agent
definition, a tool — the second fixture differs only in the server it
registers):

| Case | Test |
| --- | --- |
| components, relationships and evidence are derived from the fixture's own files | `test_offline_run_derives_components_and_relationships` |
| the answer states the established path | `test_answer_reports_the_path_the_evidence_established` |
| an unknown wire protocol stays an uncertainty | `test_uncertainty_is_stated_when_the_protocol_is_unknown` |
| the investigation sources name no subject's node | `test_no_hard_coded_story_in_the_investigation_sources` |
| the answer is not the diagram finding | `test_answer_is_not_the_diagram_summary` |
| reads and tool calls stay inside the budget | `test_investigation_reads_a_bounded_number_of_files` |
| offline ignores a provider key that happens to be set | `test_offline_mode_ignores_a_provider_key` |
| `--live` without a provider fails with exactly its two lines | `test_live_mode_without_a_provider_fails_clearly` |
| an unknown mode is rejected | `test_unknown_mode_is_rejected` |
| `--help` documents `--live` | `test_cli_help_documents_the_live_flag` |
| the CLI's live block prints the two lines and exits non-zero | `test_cli_live_failure_prints_the_two_lines_and_exits_non_zero` |
| two fixtures give two different answers (no fixed story survives) | `test_two_repositories_produce_two_different_answers` |

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
    tests/agents/test_repository_query.py \
    tests/agents/test_repository_query_evidence.py -q
```

## 13. Boundary, risks, non-goals

Does:

- one composition, one render, one exit code per question;
- verify the pinned subject's commit before the first agent runs;
- honest mode banner, honest verdict, honest artifact marks;
- reuse of every existing component listed in §7.

Does not:

- add a planner, agent, orchestrator, memory, context engine,
  validator, renderer, or a `RepositoryQueryEngine` of any kind;
- clone, fetch, or modify the subject snapshot (read-only, and the
  Git probe is `git rev-parse` on the existing checkout);
- persist query state beyond the existing memory database and trace;
- retry, refine, or chain queries (one question, one run);
- invent a destination, an edge, or a conclusion the evidence does not
  carry;
- hard-code any subject's node: `README`, `cloudtrail_tool`,
  `cloudtrail` and every other name come from files the run read —
  the investigation script itself names none of them.

Known limits:

- progress lines are printed after the run, not during it;
- `run_summary.md` records the artifact paths it was handed (the
  renderer's own behaviour), while the CLI's own output stays relative;
- the offline answer is only as good as the files the bounded budget
  reaches (six files, eight tool calls): a fact outside them stays an
  uncertainty instead of a guess. Live mode widens the *reader*, never
  the evidence policy.

Definition of done: the acceptance command exits 0, names the pinned
subject and `ef11dac8618b6344d6a5e4add6ef51cd2811d7e0`, prints every
section of §5, and says `Mode: offline`; a wrong or unprovable commit
exits 1 with the §5 block and no investigation; `--live` without a
usable provider prints the §3 block, executes nothing, exits 1, and
never falls back; the usage/failure paths of §9 return the documented
codes; no absolute path appears in the output; the full suite still
passes with no regression to M1–M17 or the specialist capabilities.
