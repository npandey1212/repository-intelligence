# M8.9 — Provider-Neutral Structured Output (typed LLM responses)

> **Status:** implemented. M8.7 observed a real model investigating;
> M8.8 verified its claims against the trace; M8.9 makes the *shape*
> of a model's final response a first-class, provider-neutral
> contract: ask for one JSON object matching a schema, receive a
> **typed result** — or an **explicit failure**. Same single loop,
> same tools, same trace. **M13** shows the contract carrying the
> investigation's *claims* as ordinary typed fields: the model
> declares `finding` / `evidence_paths` / `claims`, the application
> extracts the claims and validates their evidence after the loop —
> still one contract per run, still no second loop
> ([m13-evidence-backed-conclusions.md](../m13-evidence-backed-conclusions.md)).

## 1. Purpose

Free-form prose is fine for answers people read. It is a poor
interface for code. M8.9 lets application code say:

> "For this run, the model's response must be ONE JSON object matching
> *this* Pydantic model. Validate it application-side and hand me the
> typed instance — or fail loudly."

Given:

```text
a Pydantic model (application type)
  → its JSON Schema travels on every LLMRequest (StructuredOutputSpec)
  → the provider is asked for JSON matching that schema
  → the decoded payload is validated against the model
```

produce:

```text
typed result    the Pydantic instance (AgentRunResult.structured_output)
explicit fail   StructuredOutputValidationError — never a default,
                never a repair, never a silently dropped payload
```

The example model used throughout this document is domain-neutral on
purpose (a support-ticket summary): the contract is a general
mechanism, not tied to any particular application domain.

```python
from pydantic import BaseModel

class TicketSummary(BaseModel):
    title: str
    severity: int
```

## 2. The three validation layers (and where M8.9 sits)

A structured response can be checked at three distinct layers. They
answer different questions, never overlap, and M8.9 implements
exactly the middle one:

```text
┌─────────────────────────────────────────────────────────────────┐
│ 1. PROVIDER LEVEL        "the provider was ASKED for JSON       │
│    (translated in the      matching this schema"                │
│     adapter)               StructuredOutputSpec → response_format│
├─────────────────────────────────────────────────────────────────┤
│ 2. APPLICATION LEVEL  ◄── M8.9 lives here                      │
│    (this milestone)       decoded payload vs the Pydantic model │
│                            → typed result  OR  explicit failure │
├─────────────────────────────────────────────────────────────────┤
│ 3. DOMAIN LEVEL           "is this content right for OUR        │
│    (application data)      domain?" — outside this package      │
└─────────────────────────────────────────────────────────────────┘
```

Layer 1 is *intent* (a request parameter), not a guarantee — which is
why layer 2 exists. Layer 3 is business logic the LLM abstraction
must never absorb (M8.8's evidence validator is such a layer and
stays separate).

## 3. Data model — one spec on the way out, one payload on the way back

```text
LLMRequest
├── messages / tools / model / temperature      (unchanged)
└── structured_output: StructuredOutputSpec?    (M8.9, default None)
        ├── name: str          contract identifier
        └── schema: {...}      JSON Schema of the ONE object wanted

LLMResponse
├── message / finish_reason / usage             (unchanged)
└── structured_output: {…}?                     (M8.9, default None)
        decoded JSON object — only set when the request carried a
        spec AND the content decoded to a JSON object
```

Both fields default to `None`: **every existing request and response
is byte-for-byte unchanged** (the compatibility contract asserted by
`tests/llm/test_models.py`).

Response classification gains a fourth type — and the rule is
explicit:

```text
response has tool_calls?   → TOOL_CALL        (wins — dispatched as ever)
else payload decoded?      → STRUCTURED_OUTPUT (M8.9 terminal)
else text present?         → FINAL_ANSWER
else                       → INVALID_RESPONSE
```

`STRUCTURED_OUTPUT` is a distinct response type, not a flavor of
text: a structured run's terminal response is classified by the
contract it agreed to, whether or not the payload turned out valid.

## 4. Provider translation (the adapter's only new job)

`llm/openai_client.py` remains the single module that touches a
provider SDK. Structured output crosses the boundary in both
directions there — translation only, never interpretation:

```text
request  StructuredOutputSpec ──► response_format = { type:
                                   "json_schema", json_schema:
                                   { name, schema } }      (as given)

response content (raw text)  ──► json.loads → dict? →
                                 LLMResponse.structured_output
                                 (else None — tolerant, never raises)
```

Deliberate tolerances at this layer:

- non-JSON / broken JSON / JSON-array content under a spec →
  `structured_output` stays `None` (the raw text remains in
  `message.content`); reporting that as a *validation* failure is
  layer 2's job, with a precise message;
- no spec → no `response_format` key at all (existing payloads
  untouched — asserted by `test_no_response_format_without_a_spec`).

A provider-neutral consequence: any `LLMClient` implementation (a
test fake, a future second provider) feeds the same application
validation simply by setting `LLMResponse.structured_output` — the
Agent never learns which provider produced it (spec Test 8).

## 5. Application validation — Pydantic in, typed or explicit out

`llm/structured.py` is the entire layer:

```python
spec = spec_for_model(TicketSummary)          # model → request spec
typed = validate_structured(TicketSummary, payload)   # → instance
# mismatch → raise StructuredOutputValidationError (message + errors)
```

Rules:

- **The schema is derived from the application's own type** — never
  hand-assembled at the call site; a non-model argument raises
  `TypeError` at construction time.
- **Failures are exceptions, not fallbacks.** Missing field, wrong
  type, non-object payload — all raise
  `StructuredOutputValidationError` carrying the per-field Pydantic
  error dicts (`exc.errors`). Nothing is defaulted, coerced beyond
  the model's own declared behavior, or silently dropped.
- **The model's own Pydantic config is the authority on extras.**
  Pydantic v1's default (`Extra.ignore`) *drops* unknown keys while
  still requiring every declared field — that is the model's
  documented behavior, exercised as-is (a model with
  `extra = Extra.forbid` rejects them through the same explicit
  failure path). Validation code here never adds its own policy.

**Level selection in front of validation (added at the M17 freeze).**
One real failure put the payload one level deeper than the contract:
the model answered `{"structured_output": {"title": …, "type":
"object", "properties": {…}}}`, validation ran against the *wrapper*
object, and the model's own required fields were reported missing.
Which level the model chose is provider behavior the application does
not control; *selecting* the level that validates is the
application's job:

```python
normalize_structured_payload(model_type, payload)   # → the level
validate_structured(model_type, payload)            # selects, then validates
```

Rules, all deterministic and provider-neutral (`llm/structured.py`):

1. a payload that already validates is returned **unchanged** (the
   overwhelmingly common case — byte-for-byte the pre-fix path);
2. otherwise a breadth-first descent, never deeper than
   `NORMALIZATION_MAX_DEPTH` (3), tries known envelope keys
   (`structured_output`, `output`, `result`, `payload`, `data` —
   skipped when the key is itself a field of the model), the
   `properties` object of an echoed JSON-Schema wrapper, and a
   single-key wrapper;
3. the first child that **fully validates** wins — descending is
   choosing which object the model sent, never editing what it wrote;
4. if nothing validates, the ORIGINAL payload comes back untouched,
   so the explicit failure still reports the payload the model sent
   with its original per-field errors.

No field is ever added, removed, renamed, coerced, or fabricated, and
a payload that never matched keeps failing exactly as it did before.
`Agent._close_structured_output` applies the same selection before
validation (and before the claims/answer fallbacks), recording
`structured_validation.normalized = true` on the iteration only when
a level actually had to be chosen — every other record is unchanged.
`planning/planner.py` reaches the same behavior for free, since it
calls `validate_structured`.

**Execution metadata in front of validation (added with the final
M17 fix).** `question`, `repository_scope`, and `commit_scope` are
assignment facts the runtime already holds — the request the run is
answering — so a model that omits them is reporting *where the
report belongs*, not doing repository work:

```python
payload, injected = inject_execution_metadata(payload, metadata, model)
```

`agent/metadata.py` derives the three values from the run's own
request text (`trace.user_question`, the line-start `label: value`
lines the direct TASK block, the orchestrated wrapper, and the
finalization pass all carry) and:

- fills a key only when the model omitted it AND the contract
  declares the field — a value the model wrote is never rewritten,
  a present non-string value is left for Pydantic to judge, and the
  order of `injected` follows the fixed whitelist
  (`EXECUTION_METADATA_FIELDS`: `question`, `repository_scope`,
  `commit_scope`);
- touches nothing else: substantive fields are never added, dropped,
  or repaired here, so a substantively invalid payload still fails
  with its original per-field errors, injection markers or not;
- records `structured_validation.metadata_injected = [...]` (the
  keys it filled, in whitelist order) on success **and** on failure,
  and is a no-op — same object, `()` — when there is nothing to fill
  or the payload is not a dict.

Injection runs between level selection and validation inside
`Agent._close_structured_output` — one validation pass, unweakened —
for direct runs, orchestrated runs, and the tool-free finalization
pass alike (all three carry the same labeled request text).

## 6. Agent integration — one loop, one new terminal path

```python
agent = Agent(llm_client, structured_output_model=TicketSummary)
result = agent.run_result(question)
result.structured_output   # → TicketSummary(...)  or the run raised
result.answer              # → the raw JSON text (never replaced)
```

- The constructor derives the spec once (`spec_for_model`) and fails
  fast on wrong types; combining `structured_output_model` with
  `AgentExecutionConfig.structured_claims` raises `ValueError` (two
  different structured contracts — at most one per run). M13 keeps
  that exclusivity: claims ride *inside* the typed contract as
  ordinary fields (`claims_from_payload` reads them from the
  payload), and the Agent's post-loop validation gate is
  `structured_claims or claims` — a typed model that declares no
  `claims` field behaves byte-for-byte as before.
- **Every** request of the run carries the spec — including the ones
  before tool calls — so the contract holds for the whole
  conversation, not just the final turn.
- The loop's terminal branch: no tool calls + structured contract →
  `_close_structured_output` (validate → close iteration → terminate
  → return or raise). Tool-call responses take the exact same
  dispatch path as before; a payload is never dispatched.
- Success terminates the run with `STRUCTURED_OUTPUT`; failure raises
  `StructuredOutputValidationError` with trace termination
  `STRUCTURED_OUTPUT_VALIDATION_FAILED` — both recorded *before*
  anything is returned or raised. No second loop, no retry, no
  repair framework.
- `Agent.run(...)` keeps its `str` contract (the raw JSON text);
  the typed instance is exposed only via `run_result(...)` — matching
  how structured claims were introduced in M8.8.

## 7. Trace integration (small, additive, same trace)

New vocabulary: response/decision type `STRUCTURED_OUTPUT`,
termination reasons `STRUCTURED_OUTPUT` / `STRUCTURED_OUTPUT_VALIDATION_FAILED`.
New conditional records (absent everywhere else, so legacy payloads
are unchanged):

```text
iteration.request.structured_output        { name, schema }   (what was asked)
iteration.llm_response.structured_output   { …decoded payload… }  (what came)
iteration.structured_validation            { status: SUCCESS | FAILED,
                                             model | error,
                                             normalized?: true,
                                             metadata_injected?: [ … ] }
                                             (what validation said;
                                             `normalized` present only
                                             when level selection ran;
                                             `metadata_injected` present
                                             on success and on failure,
                                             only when keys were filled)
```

Markdown (same section as every response):

```text
### LLM Response

**Decision:** `STRUCTURED_OUTPUT`
Type: `STRUCTURED_OUTPUT` · Finish reason: `stop`
Validation status: SUCCESS

Structured output:
```json
{
  "severity": 2,
  "title": "login fails on SSO"
}
```
```

A failed run shows `Validation status: FAILED — <message>` with the
same honest text the exception carried; a spec-requested response
that contained no payload shows `Text: <present — not a structured
payload>` instead of pretending it was an answer. The M8.8
evidence-validation section is untouched and stays a separate,
run-level block.

## 8. Demonstration (offline — fakes, no provider)

```python
from repo_intelligence.agent import Agent
from repo_intelligence.llm import LLMResponse, Message

response = LLMResponse(
    message=Message(role="assistant",
                    content='{"title": "login fails on SSO", "severity": 2}'),
    finish_reason="stop",
    structured_output={"title": "login fails on SSO", "severity": 2},
)
agent = Agent(FakeLLMClient(response),
              structured_output_model=TicketSummary)
result = agent.run_result("Summarize the failure reports.")

result.structured_output.title   # "login fails on SSO"  (typed)
result.answer                    # the raw JSON text
agent.last_trace.summary.termination_reason   # "STRUCTURED_OUTPUT"
```

Failure shape (payload `{"severity": 2}` — `title` missing):

```text
StructuredOutputValidationError: structured output does not match
model TicketSummary: 1 validation error for TicketSummary
title
  field required (type=value_error.missing)
trace summary termination: STRUCTURED_OUTPUT_VALIDATION_FAILED
```

## 9. Tests (spec items 1–10)

| # | Spec test | Where |
|---|-----------|-------|
| 1 | valid output → typed result (+ full trace records) | `test_agent_structured_output.py::test_valid_structured_output_returns_the_typed_result` |
| 2 | missing field → explicit failure, no default | `test_missing_field_fails_explicitly_without_a_default` (+ llm-layer twin) |
| 3 | wrong type → explicit failure | `test_wrong_field_type_fails_explicitly` (+ llm-layer twin) |
| 4 | extra field → documented Pydantic config behavior | `test_structured_output.py::test_extra_fields_follow_the_models_own_pydantic_config` |
| 5 | text responses still work | `test_plain_agent_still_returns_free_form_text` (default `None` paths asserted) |
| 6 | tool-calls still work under the contract | `test_tool_calls_are_dispatched_and_the_loop_continues` |
| 7 | `STRUCTURED_OUTPUT` ≠ `TOOL_CALL`, payload never dispatched | `test_structured_payload_response_is_never_dispatched` + classification-order test |
| 8 | provider neutrality of the Agent | `test_agent_and_trace_are_provider_neutral` (no provider string in payload/markdown) |
| 9 | spec reaches the provider adapter | `test_structured_output.py::test_spec_translates_to_provider_response_format` |
| 10 | full M8.x regression | full suite run (reported below) |

Plus: no-payload failure, constructor `TypeError`/`ValueError`,
spec absent by default, model-contract field lists, and the
banned-reasoning sweep over a structured trace. The final M17 fix
added the execution-metadata unit tests
(`tests/agent/test_execution_metadata.py`: whitelist, extraction
rules, fill/no-overwrite/contract-gate, round-trip through
`ArchitectureReport.derived_investigation_id()`) plus the specialist
observations in `tests/agents/test_investigator_metadata.py`.

## 10. Non-goals, freeze criteria, learning answers

**Explicitly not built** (each would be a milestone of its own):

- no `StructuredOutputEngine`/`ValidationEngine`/`SchemaEngine` — one
  helper module, one terminal path;
- no second provider (the seam is tested, a second adapter is not
  M8.9's job);
- no schema registry, no output repair/retry loop, no LLM-as-judge,
  no semantic validation or scoring;
- no LangChain/LangGraph/CrewAI/AutoGen, no second loop, no new
  agent;
- no domain concepts in this layer — nothing here knows about
  repositories, claims, or evidence (M8.8 stays a separate,
  domain-level validator; M13 likewise extracts and validates claims
  in the agent/evidence layers — this layer only carries whatever
  fields the application's model declares).

**Freeze criteria (all met):**

1. `StructuredOutputSpec` optional on `LLMRequest`, payload optional
   on `LLMResponse`, both default `None` — existing contracts
   asserted unchanged (model field lists, exact payload tests).
2. `STRUCTURED_OUTPUT` is a distinct response type; tool-call
   classification and dispatch unchanged.
3. Application validation is Pydantic-based with explicit failures
   (typed result or `StructuredOutputValidationError` — no silent
   behavior of any kind), extras documented as the model's own
   config.
4. Provider translation lives only in `openai_client.py`; the Agent
   and trace contain no provider-specific code or strings.
5. Trace: conditional additive records + `Validation status:
   SUCCESS/FAILED` markdown line; banned-reasoning sweep clean; M8.8
   evidence section untouched.
6. Tests 1–10 present; full suite green; compileall + pyflakes clean
   (M8 scope); harnesses 8/8 and 10/10.

**Learning answers:**

> *Why is the spec on the request and the payload on the response
> (not one symmetric field)?* — They are different things: intent vs
> outcome. Symmetry would invite "the response validates itself",
> which is exactly the provider-level illusion layer 2 exists to
> prevent.

> *Why validate in the application at all, if the provider was given
> the schema?* — The provider is told, not trusted: response formats
> are best-effort (models can emit prose, arrays, or malformed JSON —
> documented behavior of every current provider). The application
> owns its types, so it must own the last check — and the explicit
> failure that comes with it.

> *Why an exception instead of a result object with `ok: false`?* —
> Because a structured run has exactly one success path; a typed
> result is either there or it is not. An exception cannot be
> ignored by accident the way a `None`/empty struct can, and it
> carries the per-field errors to the same place every other terminal
> failure goes: recorded on the trace, visible to the caller.
