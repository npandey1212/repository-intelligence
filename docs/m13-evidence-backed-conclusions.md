# M13 — Evidence-Backed Agent Conclusions

> **Status:** implemented, validated, frozen. The smallest change that
> makes an agent's final conclusions *explicitly* traceable to
> repository evidence it actually inspected — contract, deterministic
> validation, evaluator measurement, controlled experiment — with
> every M8–M12 test still green.

## 1. Why M13 exists

The M10 baseline run (`runtime/agent_runs/10f19b4b…`) did its job: a
plan was produced before investigation, 18 tool calls executed inside
the bound, a structured final answer landed, and the M9 evaluator
reported four `PASS`es. But one criterion read:

```text
claim_traceability  UNKNOWN
run recorded no claims — traceability not applicable (structured
claims not requested or none produced)
```

The run's final answer asserted plenty — "specific MCP servers such
as `glue-athena`, `lakeformation`, and `iam` are essential, as the
system blocks deployment if they are not properly functioning" — and
**none of it was checked against anything**. The evaluator could only
say "not applicable": the structured contract had no claims to
validate. Honest, but weak. The answer's *shape* was verified (M8.9)
while its *grounding* was not (M8.8 never ran).

M13 closes exactly that gap, one agentic capability:

> **The agent investigates → observes evidence → forms claims →
> explicitly grounds those claims → the application deterministically
> verifies the grounding.**

It is an agentic reasoning experiment, not a new repository-
intelligence milestone: no new analyzers, no new graph kinds, no new
tools, no new agents (§11 non-goals).

## 2. What M13 does / does not do

### Does

- Extends the **typed** M8.9 structured contract with `claims`
  (ordinary fields — `finding`, `evidence_paths`, `claims`), keeping
  the established exclusivity of M8.8/M8.9 at most one contract per
  run.
- Extracts the claims application-side after the loop
  (`claims_from_payload` — identical malformed-entry rules to M8.8;
  nothing invented).
- Validates every evidence reference **deterministically** against
  this run's trace with a stricter eligibility rule: only evidence
  the run actually **inspected** (`read_file`) can make a claim
  `SUPPORTED` — a search match, plan mention, graph node, or
  architecture-context line is a *discovery*, not evidence.
- Records structured failure `CLAIM_EVIDENCE_INVALID` on
  `UNSUPPORTED` validations (trace + markdown render `code:`).
- Makes M9's `claim_traceability` measurable: claims are now always
  present, `PASS`/`FAIL`/`PARTIAL` answer real questions, `UNKNOWN`
  survives only for runs that recorded no claims at all.
- Reruns the §16 controlled investigation against the same pinned
  snapshot and reports the §17 comparison.

### Does not

- No second tracer, no `AgentTrace` redesign — provenance rides the
  existing `ToolResult` → iteration → `tool_call_id` chain plus one
  boolean (`GatheredEvidence.inspected`) set at gather time.
- No LLM-as-judge, no critic agent, no self-certification: the model
  *declares* claims; only the application *verifies* them.
- No retry/repair of the structured contract (M8.9 failure semantics
  unchanged: explicit failure, traced).
- No change to the planner, the ReAct loop, the four tools, the
  evaluator's five criteria vocabulary, or any other milestone's
  behavior: a typed model that declares no `claims` field is
  byte-for-byte the pre-M13 run.

## 3. The claim/evidence contract

The M13 demonstration contract (the model's own typed fields — exact
classes follow the repository's existing structured-output
conventions):

```json
{
  "finding": "A self-contained explanation of the investigation result.",
  "evidence_paths": ["README.md", "MCP_GUARDRAILS.md"],
  "claims": [
    {
      "statement": "MCP servers operate in either Local or Gateway mode.",
      "evidence_refs": [
        {"path": "MCP_GUARDRAILS.md", "line_start": 59, "line_end": 62}
      ]
    }
  ]
}
```

Contract rules (schema-level, so a nonconforming response fails the
run before any claim logic sees it):

- every field required; `evidence_paths` and `claims` non-empty;
- statements/paths reject empty and whitespace-only strings;
- each claim cites **at least one** evidence reference; line ranges
  require both bounds.
- Field descriptions *are* the model-facing instruction: conclusions
  and evidence references only, claims may not exceed the inspected
  evidence, uncertainty is represented rather than invented away —
  no chain-of-thought request, no hidden fields anywhere (asserted
  by the same `BANNED_REASONING_FIELDS` vocabulary).

§14 semantics — success means more than "the schema is valid":

```text
schema valid  AND  claims exist  AND  claims reference evidence
              AND  referenced evidence was actually inspected
```

…without bureaucratic annotation: nothing requires every sentence of
`finding` to become a claim.

## 4. Evidence eligibility — the important distinction

The four-tool environment, by role:

| Tool | Role | Evidence eligibility |
| --- | --- | --- |
| `search_repository` | **discovery** — names a candidate file for the next step | gathered, `inspected=False` — never sufficient alone |
| `read_file` | **inspection** — returns actual content + line range | `inspected=True` — the only source-evidence eligibility |
| `query_engineering_graph` | **structured graph observation** (relationships/knowledge) | not file-addressable — contributes no claim evidence |
| `get_architecture_context` | **orientation/context** | not file-addressable — contributes no claim evidence |

```text
search_repository(...)   →  candidate file discovered
graph query / context    →  structured knowledge / orientation mentioned
read_file(...)           →  content actually inspected ⇒ ELIGIBLE
```

Discovery followed by inspection works the obvious way: a search hit
plus a later `read_file` of the same path leaves one *inspected*
evidence item — eligibility comes from the read, not the search.

> Traceability validation verifies that claims reference evidence
> actually obtained during the run. It does not independently prove
> the semantic truth of the claim.

`gather_evidence(trace)` sets one field at gather time — `read_file`
success ⇒ `inspected=True`, a search match ⇒ `inspected=False` — and
`EvidenceIndex.inspected` exposes the eligible subset. Serialized in
the trace (`to_dict`), `from_dict` defaults `True` for legacy
payloads so pre-M13 traces still validate exactly as before.

The validator honors it through `require_inspected` (default
`False`, preserving every established direct-call contract; the
Agent passes `True` after every run). Three negative reasons stay
distinct — never merged:

```text
…, but the path was never gathered during this run
…, but the path was found during the run (search results are
    discoveries, not evidence) and never inspected
…, but the inspected line range does not cover the cited range
```

Graph knowledge stays graph knowledge: nothing in the evidence layer
reads the graph artifact — a claim citing a path the run never
opened is `UNSUPPORTED`, whatever else the repository knows about it.

## 5. Deterministic validation (application-side, post-loop)

The gate in `Agent.run_result` / `run_planned` is
`if structured_claims or claims:` — one post-loop step, pure
functions over the finished trace, no tool dispatch, no LLM call:

```text
structured payload → claims_from_payload(payload)        (extract)
→ validate_claims(claims, gather_evidence(trace),
                  require_inspected=True)                (verify)
→ ClaimValidation(status, reason, code?, matched evidence,
                   tool_call_id provenance)              (record)
→ trace.evidence_validation + AgentRunResult.evidence_validation
```

Statuses keep the established vocabulary; M13 only sharpens meaning:

| Status | Meaning (M13) |
| --- | --- |
| `SUPPORTED` | ≥ 1 reference resolved to **inspected** evidence (matched items carry tool, lines, iteration, `tool_call_id`) |
| `UNSUPPORTED` | valid references exist, none resolved — `code: CLAIM_EVIDENCE_INVALID` |
| `UNKNOWN` | no references / structurally insufficient / out-of-repository — cannot be answered deterministically; flag-only, no code |

Empty claim entries are rejected at the schema (`min_length`);
malformed extras inside a raw payload are dropped by the extractor
(same M8.8 rule — never invented). Duplicate references resolve
deterministically (first match wins, order preserved).

`CLAIM_EVIDENCE_INVALID` is a flag on the recorded validation, not a
raised exception: the run's answer still exists (the model did the
investigation), but the evaluation can prove it does not trace back —
the honest split between *execution outcome* and *evaluation
outcome*.

## 6. Trace integration (smallest possible change)

- `GatheredEvidence.inspected` (new field, serialized) and
  `ClaimValidation.code` (new optional field, serialized only when
  set) — nothing else in `AgentTrace` moves.
- The markdown projection renders `* code: CLAIM_EVIDENCE_INVALID`
  under an unsupported claim's validation block; everything else is
  the existing `### M8.8 Evidence Validation` section.
- Provenance question from §5 answers directly from the trace:

```text
claim → evidence_refs → GatheredEvidence → tool_call_id
      → iteration (which tool, when) → ToolResult source
                                            (repository, commit, snapshot)
```

- The demo prints the complete chain per claim (§8), and both
  `trace.json` / `trace.md` carry it — one run, one trace, no second
  system.

## 7. M9 evaluation now *measures* claim traceability

The evaluator is unchanged in structure — same five criteria, same
`PASS/PARTIAL/FAIL/UNKNOWN` vocabulary, still consuming recorded
statuses without recomputation (`gather_evidence(` /
`validate_claims(` still appear nowhere in the evaluation package).
What changed is that the question is now answerable, and one branch
was sharpened:

```text
supported > 0 and all supported    → PASS
supported > 0, some unsupported    → PARTIAL
supported = 0, claims recorded     → FAIL   (was: could hide in UNKNOWN)
   (all UNKNOWN — claims reference no evaluable evidence — or
    UNSUPPORTED with CLAIM_EVIDENCE_INVALID)
no claims recorded                 → UNKNOWN (nothing to measure)
```

M13's experiment therefore turns the baseline's `UNKNOWN` into
measured truth either way: a run whose claims do not trace back
**fails** the criterion (an answerable "no"), and one that does
passes it.

On partial support: the project's existing terminology is used as-is
— no parallel `PARTIALLY_SUPPORTED` status was invented. Per claim,
validation stays `SUPPORTED / UNSUPPORTED / UNKNOWN` (with the
established rule that *one* resolving reference suffices for
`SUPPORTED`, while the result records exactly which references
matched); **partial support surfaces where it is already measurable,
at the criterion level**: mixed claim outcomes aggregate to
`claim_traceability = PARTIAL`. Broad claims are not upgraded by
unrelated inspections — sufficiency of evidence for a statement's
strength remains a semantic question, explicitly out of scope
(traceability, not truth).

## 8. Demonstration

```bash
python agentrun_m13_evidence_demo.py            # real model, exit 0/1/6
```

The demo is a composition root over the existing pieces — no loop,
validator, evaluator, or planner of its own — and prints the §19
chain from recorded data only:

```text
Objective → Plan (validation, steps) → Agent execution (iterations,
tool calls, plan-step timeline) → Claims and evidence (each claim →
refs → validation status/code/reason → matched evidence with
iteration/tool) → Evaluation (five criteria + overall) → artifacts
```

Live excerpt (run `8675c7a7…`, §16):

```text
Claims and evidence
1. MCP servers operate in either Local mode or Gateway mode, …
   evidence: MCP_GUARDRAILS.md:59-62
   validation: SUPPORTED
     reason: matched gathered evidence: read_file MCP_GUARDRAILS.md:59-62
     inspected: MCP_GUARDRAILS.md:59-62 (iteration 15 via read_file)
claims: 3 — 3 supported, 0 unsupported, 0 unevaluable

Evaluation
Final result: PASS · Evidence gathered: PASS · Claim traceability: PASS
Execution bound: PASS · Objective coverage: PASS
Overall: PASS — 5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL
```

No private model reasoning is printed or persisted anywhere: claims,
evidence references, statuses, reasons — observable events only.

## 9. The experiment (§16/§17)

Same investigation question, same pinned snapshot
(`main_2026-09-22_ef11dac`, commit `ef11dac…`), same model
(`gpt-4o-mini`), same planner/loop/evaluator composition:

| | Baseline `10f19b4b…` | M13 `8675c7a7…` |
| --- | --- | --- |
| Plan | produced before investigation, 5/5 steps completed, `validation: SUCCESS` | same (5/5, `SUCCESS`) |
| Iterations / tool calls | 15 / 18 (6 searches, 12 reads, 0 failed) | 17 / 19 (7 searches, 12 reads, 0 failed) |
| Files inspected | 5 unique | 7 unique |
| Budget | 18 of 20 executed, 0 blocked | 19 of 60 executed, 0 blocked |
| Claims produced | **0** (contract had no claims) | **3**, each with line-ranged refs |
| Claims with valid evidence | not applicable | **3 / 3** (`SUPPORTED`) |
| Claims unsupported | not applicable | 0 (`CLAIM_EVIDENCE_INVALID`: never fired) |
| `claim_traceability` | `UNKNOWN` | `PASS` — "3 of 3 claim(s) supported by gathered evidence" |
| Objective coverage | `PASS` (2 expected fields) | `PASS` (3 expected fields, incl. `claims`) |
| Overall | **PARTIAL** (4 PASS, 1 UNKNOWN) | **PASS** (5 PASS) |

**Final answer, compared against the evidence.** The baseline's
1109-character `finding` was fluent and uncited — "specific MCP
servers such as `glue-athena`, `lakeformation`, and `iam` are
essential" had no recorded claim to trace (its own `evidence_paths`
were never validated). The M13 `finding` is shorter — and every
factual assertion it rests on appears as a claim with a verified
pointer: README.md:255-258 (agents/MCP configurations),
MCP_GUARDRAILS.md:59-62 (Local **or** Gateway — "either … or", not
"all run in Gateway"), MCP_GUARDRAILS.md:396-399 (each Gateway
server runs with **its own** least-privilege IAM policy). The
distinctions §17 asks about — *some* vs *all*, *server exists* vs
*this agent uses it* — are exactly the ones deterministic validation
can hold: precision where the inspected evidence supports it, and
`UNSUPPORTED` rather than silent confidence where it does not.

Environment note: same model/snapshot/question; the recorded
baseline ran under a 20-call bound, the M13 run under the script's
60 — neither hit its bound, and executed tool counts differ by one
(18 vs 19).

**Confirmation rerun** (`6dd0fb76…`, same objective/snapshot/model/
environment): plan 5/5 `SUCCESS`, 13 iterations / 27 calls (12
searches, 15 reads over 13 unique files, 1 architecture-context
call, 0 failed), 2 claims → **2/2 `SUPPORTED`**, `claim_traceability
PASS`, overall **PASS** again (exit 0). It also shows the validator
being conservative rather than generous: claim 1 cited three
references, only two matched — `.claude/agents/transformation-agent.md`
was never `read_file`'d in that run — so the claim passed on its two
genuinely inspected references (the established one-reference rule),
and the trace prints the matched subset separately from the cited
ones: a learner can see *cited ≠ inspected* without any code change.

**Final wiring run** (`3d5a373d…`, objective extended to request
claims explicitly): plan 5/5 `SUCCESS`, 19 iterations / 24 calls
(7 searches, 15 reads over 10 unique files, plus
`query_engineering_graph` observations, 0 failed), **4 claims →
4/4 `SUPPORTED`**, every cited path a real `read_file` result,
`claim_traceability PASS (4 of 4)`, overall **PASS** — the graph
was used during the run yet no claim cites graph output as source
evidence, the eligibility rule holding in a live trace.

## 10. Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
# → 1103 passed, 2 skipped   (M12 baseline 1048 + 55 new, none modified)
```

| File | Tests | Covers |
| --- | --- | --- |
| `tests/evidence/test_m13_claim_evidence.py` | 19 | inspected vs discovered provenance (+ serialization, legacy default), valid inspected path accepted, search-only rejected with distinct reason, `CLAIM_EVIDENCE_INVALID` code + JSON round-trip, out-of-repo → `UNKNOWN`, normalization, coverage misses, deterministic duplicates, no-refs → `UNKNOWN`, `validate_claims` forwarding, cross-run association, tool_call_id provenance, **architecture context is not source inspection, graph observation is not source inspection, discovery-then-read becomes eligible** (§13 Tests 4–6), mixed refs within one claim → one-reference rule with exact matched subset |
| `tests/agent/test_m13_evidence_claims.py` | 8 | typed-with-claims → claims + validation + trace block; typed-without-claims unchanged; invalid claim → code + markdown `code: CLAIM_EVIDENCE_INVALID`; search discovery rejected at agent level; mixed order; no hidden reasoning; determinism; config default |
| `tests/agent/test_m13_evidence_demo.py` | 21 | demo API exposure, controlled objective + budget vs the M10 baseline, no second implementation (source scan), agent construction, structured schema (valid/multiple/empty/missing/requires) + model-facing descriptions, offline planned run → claims + `SUPPORTED`, spied single evaluator invocation → traceability + overall `PASS`, chain printing, exit codes, no-key/no-context → 6, full offline `main` → 0, invalid claim → 1 with `CLAIM_EVIDENCE_INVALID` |
| `tests/evaluation/test_m13_claim_traceability.py` | 7 | PASS when evidence valid, FAIL when invalid (`0 of 1`), FAIL when claims recorded but none evaluable, PARTIAL on mixture, no claims → still `UNKNOWN`, claim-bearing typed run reaches **overall PASS** (baseline ceiling gone), evaluator consumes statuses without revalidating |

Regression: full suite green (1103/2), `compileall` clean, `pyflakes`
clean on every changed/new file, harnesses `6/6`, `8/8`, `10/10`,
`13/13`, and the frozen M8–M12 suites untouched — including the
M8.8/M8.9 exclusivity contract, the `REASON_PREFIXES` wording pins,
the direct `validate_claim` default-mode semantics, and the
one-reference-suffices rule (`test_multiple_references_one_valid_is_supported`).

## 11. Non-goals (§21 — explicitly not introduced)

New repository analyzers · new graph relationships · MCP discovery or
MCP runtime execution · graph/vector database · embeddings ·
GraphRAG · semantic search · another agent · critic agent · evaluator
agent · LLM-as-judge · reflection loop · autonomous replanning ·
multi-agent orchestration · workflow engine · memory system ·
knowledge-base redesign · new agent-facing tool · replacement of
ReAct · replacement of M10 plan-and-execute. If an implementation
idea requires one of these, the design is wrong for M13.

## 12. Boundary and philosophy

M13 is a **claim-grounding milestone inside the existing agent**:
one contract extension, one eligibility boolean, one structured
failure code, one evaluator branch, one demonstration. Everything
the run uses was already there — planner, loop, tools, trace,
evidence index, evaluator.

> Repository intelligence exists to provide a meaningful environment
> for learning agentic AI. It is not the product goal.

Nothing here makes the repository "smarter"; it makes the agent's
*conclusions* answerable to what the agent actually saw — the
smallest honest version of that question.

## 13. Definition of done (§22)

- [x] Existing four tools unchanged in purpose.
- [x] Existing M10 planner remains in place.
- [x] Existing ReAct execution remains in place.
- [x] Final structured output contains evidence-backed claims.
- [x] Evidence paths correspond to files actually inspected during
      the run.
- [x] Application-level validation verifies this deterministically.
- [x] No LLM-as-judge is introduced.
- [x] M9 evaluator measures claim traceability.
- [x] The controlled investigation is rerun (`8675c7a7…`, PASS).
- [x] The new trace can be inspected end-to-end (`trace.json`/`md`,
      `evaluation.json`/`md` under `runtime/agent_runs/`).
- [x] The final finding is more precise than the baseline where
      evidence supports that precision (§9).
- [x] Existing M8–M12 tests remain green.
- [x] Focused M13 tests are green (51).
- [x] Documentation is updated (this file +
      [evidence-traceability.md](agent/evidence-traceability.md),
      [structured-output.md](agent/structured-output.md),
      [investigation-evaluation.md](agent/investigation-evaluation.md),
      README).
- [x] The milestone is frozen after validation.

## 14. Learning progression

```text
M8.8   "Can structured claims be traced to gathered evidence?"
            ↓
M9     "Did the investigation satisfy its success criteria?"
            ↓
M10    "Does the investigation plan before it acts?"
            ↓
M13    "Can the agent's CONCLUSIONS be traced to evidence it
        actually INSPECTED — and does the evaluator say so
        instead of shrugging at UNKNOWN?"
            ↓
Later  "What should the agent do when its evidence does not
        support the conclusion it wants to reach?"
```

Each question stays separate so the abstractions remain
understandable. M13 establishes traceability of conclusions to
inspected evidence — not answer correctness: `UNSUPPORTED` means
*this run did not establish the referenced evidence*, never that the
statement is false.
