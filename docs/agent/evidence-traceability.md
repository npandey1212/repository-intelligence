# M8.8 — Evidence Traceability (deterministic validation)

> **Status:** implemented. M8.6 proved the adaptive mechanics with a
> scripted model; M8.7 observed a real model performing the same
> investigation; M8.8 asks the follow-up question the M8.7 trace
> exposed — *are the claims in an answer actually backed by evidence
> this run gathered?* — and answers it **deterministically**, offline,
> with no model in the loop. **Integrated end-to-end:** the M8.7
> runner now requests the model's explicit structured claims and the
> run validates them right after the investigation (§4–§5, §8–§9).
> **M13 extends the eligibility rule:** *gathered* alone is no longer
> enough — only evidence the run actually **inspected** (`read_file`)
> can make a claim `SUPPORTED`; search matches are discoveries, not
> evidence (§6, §7, and [m13-evidence-backed-conclusions.md](../m13-evidence-backed-conclusions.md)).

## 1. Purpose

Validate whether **structured claims** point to evidence actually
gathered during an Agent run.

Given:

```text
claims  +  evidence references  +  execution/tool evidence (the trace)
```

produce, for each claim, one of:

```text
SUPPORTED      the referenced evidence was gathered successfully
               (M13: and actually inspected — see §6)
UNSUPPORTED    references exist (and are valid) but none were gathered
               (M13: recorded with code CLAIM_EVIDENCE_INVALID when
               the reference points outside this run's inspections)
UNKNOWN        no references / insufficient metadata — cannot be
               evaluated deterministically
```

The objective is deliberately narrow: **traceability**, not truth.
M8.8 does not decide whether an answer is correct — only whether each
claim's evidence references resolve to successful tool results the
run actually produced.

Integrated (end-to-end), the run itself now produces those claims: the
model's FINAL response is ONE explicit JSON object
`{"answer": ..., "claims": [...]}` — an output contract, not prose
extraction — and the existing validator runs the moment the
investigation finishes:

```text
investigation loop → trace holds the ToolResults
→ model's structured final response → answer + claims
→ gather_evidence(trace) + validate_claims(...)   (deterministic)
→ recorded in the trace, returned with the answer
```

## 2. Boundary — traceability, not truth

```text
Claim
   ↓
Evidence reference
   ↓
Was that evidence actually gathered?
   ↓
YES / NO / CANNOT TELL
```

There is **no** `Claim → LLM → "is this true?"` anywhere: adding one
would reintroduce a probabilistic reasoning layer exactly where
determinism matters most. Concretely:

- a claim with **no** evidence references is `UNKNOWN` — *not*
  "unsupported = false" and *not* silently "supported";
- a failed tool result (`FILE_NOT_FOUND`) can **never** count as
  positive evidence;
- `UNSUPPORTED` means *this run did not establish the referenced
  evidence* — not that the statement is necessarily false;
- an exhaustive-sounding statement (e.g. "the repository contains
  exactly 11 agents") is neither confirmed nor refuted here: the
  validator only verifies the references it was given. Whether a set
  of references is *sufficient* for the strength of a statement is a
  semantic question — explicitly out of scope (§10).

M8.8 checks one property: **an agentic answer is more trustworthy
when its claims can be explicitly traced to evidence the Agent
actually gathered.**

## 3. Data model

Smallest useful representation (frozen value objects, project
convention — `dataclass`, no framework):

```python
@dataclass(frozen=True)
class EvidenceRef:
    path: str                       # repository-relative, normalized
    line_start: Optional[int] = None
    line_end: Optional[int] = None

@dataclass(frozen=True)
class Claim:
    statement: str
    evidence_refs: Tuple[EvidenceRef, ...] = ()

@dataclass(frozen=True)
class ClaimValidation:
    claim: Claim
    status: Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]
    matched_evidence: Tuple[GatheredEvidence, ...] = ()
    reason: str
    code: Optional[str] = None      # M13: CLAIM_EVIDENCE_INVALID on
                                    # UNSUPPORTED (None otherwise)
```

Two supporting records complete the provenance chain:

```text
GatheredEvidence     what the run observed: path + lines + which tool
                     + tool_call_id + ToolResult source (repository,
                     commit_sha, snapshot_id) + inspected flag (M13:
                     read_file results are inspected; search matches
                     are discovered only) — evidence is REFERENCED,
                     file contents are never copied into a claim
FailedEvidenceAttempt  a failed tool call for a path (reason text only)
EvidenceIndex        gathered + failed for one run (+ the inspected
                     subset for one run, M13)
```

No field named `thought`, `chain_of_thought`, `private_reasoning`,
`internal_monologue`, or `reasoning_trace` exists anywhere in this
model — asserted by tests with the same `BANNED_REASONING_FIELDS`
vocabulary the execution trace rejects.

Naming note: `repo_intelligence.facts.evidence.EvidenceRef` is the
M2 fact-provenance pointer (path + artifact type + commit + symbol).
This `EvidenceRef` is the M8.8 claim-to-run pointer (path + lines).
Different concerns, one word — each is documented where it lives.

## 4. Input / Output

Input (structured — claims are *given*, never extracted from
free-form answer text). Standalone, they are constructed directly; in
the integrated run they come verbatim from the model's explicit
`{answer, claims}` final response (decoded by
`repo_intelligence.agent.claims.parse_structured_answer`):

```python
from repo_intelligence.evidence import Claim, EvidenceRef

claims = [
    Claim(
        statement="dag-agent is defined in .claude/agents/dag-agent.md",
        evidence_refs=[
            EvidenceRef(path=".claude/agents/dag-agent.md",
                        line_start=1, line_end=20),
        ],
    ),
]
```

plus the run's execution evidence — an `AgentTrace` or its JSON
mapping (`runtime/agent_runs/<run_id>/trace.json`).

Output: one `ClaimValidation` per claim (status, matched evidence,
reason), in input order. A run returns them as:

```text
AgentRunResult
├── answer                human-readable (never raw claim JSON)
├── claims[]              the model's explicit structured claims
└── evidence_validation[] deterministic result per claim
```

and records the same information on the trace as an optional
`evidence_validation` block (`{"claims": [...], "results": [...]}`)
before the sink persists it.

## 5. Architecture

```text
                  Agent  (run_result)
                    │
                    ↓
       investigation loop (M8.4, single tool-call limit)
                    │  every ToolResult recorded
                    ↓
             Execution Trace
                    │
                    ↓  after the loop, still inside run_result
       gather_evidence(trace) ──► EvidenceIndex
                    │
       validate_claims(claims, evidence)
                    │
             ┌──────┴──────┐
             ↓             ↓
          Claims       Evidence refs
             │             │
             └──────┬──────┘
                    ↓
          Deterministic result
             │
             ├─► AgentRunResult (answer, claims, validation)
             └─► trace.evidence_validation ─► TraceSink
                    → runtime/agent_runs/<run_id>/trace.json
```

This is a **validation/provenance layer**, not another reasoning
component. It runs strictly *after* the investigation, consumes data
the run already produced, and changes nothing about how the run
behaved:

```text
investigation loop / tool-call limit ← unchanged (M8.4–M8.5 semantics)
ToolDispatcher.dispatch              ← unchanged; validation never dispatches
TraceSink → runtime/agent_runs/<run_id>/trace.json  ← unchanged location
                                     (+ one optional evidence_validation block)
gather_evidence(trace)               ← reads the trace, writes nothing
validate_claim(claim, index)         ← pure function, no I/O
```

The one-line integration: `AgentExecutionConfig.structured_claims`
opt-in asks the model for that explicit final JSON (an output
contract — `agent/claims.py`), and `Agent.run_result` adds the
post-loop validation step above. Runs without the flag behave exactly
as before — same loop, same answer, no new trace field. There is no
second Agent, no critic, no second LLM call after the answer, and no
second loop.

## 6. Evidence source (reused, never rebuilt)

`gather_evidence` derives the gathered-evidence set from
**successful** tool results already recorded in the execution trace:

| Tool result | Contributes |
| --- | --- |
| successful `read_file` (`data.path` + reported lines) | one evidence item: path + line range — **inspected** (M13: evidence-eligible) |
| successful `search_repository` (`data.results[]` of matches) | one evidence item per match: path + excerpt range — **discovered only** (M13: never sufficient on its own for `SUPPORTED`) |
| successful `get_architecture_context` (`data.context`, no file address) | nothing file-addressable — stays in the trace, skipped here |
| **failed** result (any tool) | **never evidence** — recorded only as a failed attempt (path + error code) so reasons can say `FILE_NOT_FOUND` |

The M13 eligibility rule is one field, set at gather time: `read_file`
success ⇒ `inspected=True`; a search match ⇒ `inspected=False`.
Discovery (a path *appearing* in a search hit, the plan, the graph, or
the architecture context) is not inspection; only a subsequent
`read_file` of that path makes the run's evidence eligible. This is
why the M13 validator distinguishes "found during the run but never
inspected" from "never gathered" in its `UNSUPPORTED` reason.

So the M8.7 trace yields exactly:

```text
gathered: read_file .claude/agents/dag-agent.md:1-20
          read_file tool-registry/servers.yaml:1-30
failed:   read_file tool-registry/README.md → FILE_NOT_FOUND
```

The provenance chain stays intact:

```text
Claim → EvidenceRef → GatheredEvidence → ToolResult → source
                                                 (repository, commit,
                                                  snapshot_id)
```

No second repository evidence system, no file contents duplicated —
reference, don't copy.

## 7. Matching rules (explicit, deterministic)

1. **Path normalization / repository boundary.** Claim-reference and
   gathered paths are normalized with the existing
   `normalize_repository_relative_path` helper (the same path-safety
   the tools use): backslashes → `/`, `./x` ≡ `x`, empty segments
   dropped. Absolute paths (POSIX or `C:\`) and `..` escapes are
   rejected — an arbitrary absolute filesystem path can never bypass
   the repository boundary, and a rejected reference makes the claim
   `UNKNOWN` (insufficient metadata), never `SUPPORTED`.

2. **Reference structure.** A line range requires **both** bounds,
   each ≥ 1, with `line_start ≤ line_end`. A single bound is
   insufficient metadata → `UNKNOWN`.

3. **Line coverage.** With claim lines and gathered lines present,
   gathered evidence must *cover* the claim's range:
   `gathered.line_start ≤ claim.line_start` and
   `claim.line_end ≤ gathered.line_end`.

   | gathered | claim reference | result |
   | --- | --- | --- |
   | `1-50` | `1-20` | covers → may be `SUPPORTED` |
   | `1-20` | `40-50` | does not cover → not `SUPPORTED` |
   | `1-20` | `10-20` | covers → `SUPPORTED` |
   | no lines (whole file) | any range | covers |
   | any | no lines (path-only) | matches on path |

4. **One reference is enough.** A claim is `SUPPORTED` when at least
   one reference resolves to gathered evidence; the result records
   exactly which gathered items matched. Several references where
   none match → `UNSUPPORTED`.

5. **Inspected-only eligibility (M13).** `validate_claim` takes
   `require_inspected` (default `False`, so the established direct-call
   contract and M8.8 integrations are unchanged; the Agent passes
   `True` after every M13 run). In that mode a match must resolve to
   evidence with `inspected=True` — a search-discovered path that was
   never `read_file`'d cannot make a claim `SUPPORTED`, and an
   `UNSUPPORTED` result carries `code=CLAIM_EVIDENCE_INVALID` so the
   failure is structured (trace + markdown render `code:`) rather than
   buried in prose. The three negative reasons stay distinct: the
   path was never gathered / was found during the run but never
   inspected / was inspected but does not cover the cited range.

Status semantics:

```text
SUPPORTED    ≥ 1 reference → successfully gathered (and, under
             require_inspected, inspected) evidence
UNSUPPORTED  valid references exist, none matched (failed attempts
             are called out in the reason, e.g. FILE_NOT_FOUND;
             M13: code = CLAIM_EVIDENCE_INVALID when the reference
             could not resolve to this run's evidence)
UNKNOWN      no references, or all references structurally
             insufficient — never silently converted to SUPPORTED
```

## 8. Demonstration

```bash
python agentrun_evidence_test.py
```

Fully offline: it loads the M8.7-style fixture
(`tests/fixtures/m87_react_trace.json` — real tool results captured
over the pinned checkout: architecture context, a successful
`read_file` of `.claude/agents/dag-agent.md:1-20`, the failed
`read_file` of `tool-registry/README.md` → `FILE_NOT_FOUND`, and a
successful `read_file` of `tool-registry/servers.yaml:1-30`), prints
the gathered evidence with provenance, then each claim with its
status and reason, and finishes with 10 PASS/FAIL checks (exit 0 / 6):

```text
Claim validation
------------------------------------------------------------------------
[1] dag-agent is defined in .claude/agents/dag-agent.md ...
    ref: .claude/agents/dag-agent.md:1-20
    status: SUPPORTED
    reason: matched gathered evidence: read_file .claude/agents/dag-agent.md:1-20
[2] tool-registry/README.md documents the tool registry.
    status: UNSUPPORTED
    reason: ... (attempted during the run and failed: FILE_NOT_FOUND)
[3] The repository contains an agent called X.
    ref: (none)
    status: UNKNOWN
    reason: no evidence references supplied; ...
```

The core demo claim from the milestone definition — `read_file`
success on `.claude/agents/dag-agent.md:1-20` supporting the matching
claim — and the counter-example — `tool-registry/README.md` failing
with `FILE_NOT_FOUND`, therefore never `SUPPORTED` — are checks 1 and
2.

End-to-end with a real model (M8.7 runner + M8.8 integration):

```bash
python agentrun_react_live_test.py     # requires OPENAI_API_KEY
```

same composition as before (single `run_result` call, real dispatcher,
`max_tool_calls=8`) with `structured_claims=True`: the printed run
ends with an *Evidence validation (M8.8)* block — one status + reason
per claim — before the human-readable answer, and the persisted
runtime trace carries the same `evidence_validation` block, with the
Markdown projection rendering it as a run-level section after the
iterations (`### Structured Claims` followed by
`### M8.8 Evidence Validation`, before `## Run Summary` — M8.7.1
stage order; see [execution-trace.md](execution-trace.md) §7). The
offline equivalent (scripted model, real dispatcher, real pinned
checkout, including the failed `tool-registry/README.md` read) is
`tests/evidence/test_evidence_integration.py`.

## 9. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/evidence/ -q
```

`tests/evidence/test_evidence_traceability.py` covers the ten required
cases: supported claim / failed evidence never positive / missing
references → `UNKNOWN` / wrong path / wrong line range / containing
range / multiple references / all failed / the M8.7 fixture /
no hidden reasoning fields — plus normalization + repository-boundary
cases, single-bound references, search-match evidence, path-only
references, the non-file-addressable architecture context, an
`AgentTrace` object as input (not just its JSON), order/vocabulary
stability, and `to_dict`/`from_dict` round-trips.

`tests/evidence/test_evidence_integration.py` covers the integration:
supported claim / failed evidence → `UNSUPPORTED` / multiple claims
validated independently / the trace (JSON **and** the file written
before persistence) carrying `evidence_validation` / validation
dispatching **no** tool / no hidden reasoning in the structured output
or trace / tolerant degradation when the model ignores the contract /
`run` delegating to `run_result` / the config contract.

M13 (`tests/evidence/test_m13_claim_evidence.py`) covers the
eligibility rule: inspected vs discovered provenance (including
serialization round-trips and the legacy `inspected=True` default),
search-only references rejected with the distinct reason and
`CLAIM_EVIDENCE_INVALID`, coverage misses, out-of-repository paths,
deterministic duplicate handling, and cross-run association.

Regression (whole milestone):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agent/ tests/evidence/ -q
```

The integration is one opt-in flag plus one post-loop step: default
runs keep their exact previous behavior (no new trace field), and the
runtime trace location stays `runtime/agent_runs/<run_id>/`.

## 10. Explicit non-goals (and what M8.8 does not do)

- **no semantic truth verification** — statements are never parsed or
  judged; sufficiency/exhaustiveness of references is out of scope;
- **no LLM call, critic agent, evaluator agent, or model-based claim
  verification** — the layer is pure functions over the trace;
- **no automatic claim extraction from prose** — claims come only
  from the model's explicit `{answer, claims}` output contract (or
  are constructed directly); the application never infers them from
  free-form text;
- **no RAG, embeddings, vector search, memory, knowledge graph, web
  search**;
- **no second Agent, no orchestration loop, no planner, no
  workflow engine, no MCP** — the integration is one opt-in,
  post-loop step inside `run_result`; there is no critic/evaluator
  agent, no second LLM call after the answer, and the validator
  remains pure functions over the finished trace;
- **no new evidence system** — evidence comes from the existing
  tools' results inside the existing trace;
- **no runtime/artifacts changes** — traces stay under
  `runtime/agent_runs/<run_id>/`; the validator writes nothing.

## 11. Learning progression

```text
M8.7   "Can the agent investigate adaptively?"
            ↓
M8.8   "Can we deterministically verify that its structured claims
        are grounded in evidence actually gathered during that
        investigation?"   (deterministic, offline; integrated
        end-to-end so the run produces those claims itself)
            ↓
M13    "...and is 'gathered' enough? Search results are gathered
        too. Only evidence the run actually INSPECTED can make a
        claim supported — discoveries are not evidence."
            ↓
Later  "How should an agent produce reliable evidence-backed
        structured claims?"
```

These questions stay separate so each abstraction remains
understandable. M8.8 does **not** establish answer correctness — only
traceability: `UNSUPPORTED` means the run did not establish the
referenced evidence, not that the claim is necessarily false.
