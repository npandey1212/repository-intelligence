# Agent Execution Trace (observability MVP)

> **Status:** implemented alongside the M8.4 multi-step loop and
> extended by the M8.5 bounded agent (budget, blocked tools,
> termination); storage relocated to `runtime/agent_runs/` for M8.7
> (runtime telemetry vs durable artifacts); made readable as an
> explicit conversation loop by the M8.8 readability follow-up; and
> restructured into the ordered interaction stages (LLM Request →
> LLM Response → Actions → Tool Executions → Tool Results → State
> Update → Continuation/Termination → Answer, with the M8.8 validation
> as a run-level section) by **M8.7.1 — observable interaction trace
> clarity**; the **M8.8 closeout** then made each response and control
> decision self-explanatory — every response stage states its type,
> the requested tool-call ids and the model's exact text (`Text:
> <none — response requested tool execution>` for a tool-call
> response), and a terminal decision renders as **Termination** with
> one-line learner prose instead of **Continuation**. This is a
> deliberately small, local observability capability — not a
> production tracing platform.

Every `Agent.run` automatically records an **execution trace**: a
structured, provider-neutral record of what the run *observably* did —
the question, each numbered LLM decision cycle, the exact tools the
model requested, what the dispatcher returned (including how long each
execution took), which observations went back into the conversation,
whether the run continues or terminates after each step, how the run
terminated, and the totals.

The point is developer understanding: one trace should let you follow
the Agent loop step by step and debug it without reading the model's
mind — because the trace never contains the model's private
chain-of-thought. The trace exposes observable model inputs/outputs
and tool activity for debugging and learning. It does not store
hidden chain-of-thought or private reasoning.

## 1. What the trace represents

```text
Agent run
│
├── Run metadata        run_id, timestamps, agent id/version,
│                       repository provenance (when observed)
│
├── Iteration 1         (numbering starts at 1, monotonic)
│   ├── request          message_count, tool_definition_count, model,
│   │                    messages[] — roles, order, clipped content
│   ├── response         finish_reason, tool_call_count, usage
│   ├── llm_response     type, tool_call_count, has_text, exact
│   │                    requested tool_calls (readability follow-up)
│   ├── decision         TOOL_CALL / FINAL_ANSWER / ERROR / …
│   ├── tool_calls       id + name + EXACT arguments
│   ├── tool_results     success/failure, error, source, data
│   │                    (clipped), duration_ms (M8.7.1, executed)
│   ├── observations     TOOL_RESULT identity (+ tool_call_id)
│   └── continuation     CONTINUE + next_iteration, FINAL_ANSWER or
│                        TERMINATE + termination_reason (M8.7.1)
│
├── Iteration 2         … same shape …
│
├── Iteration N         FINAL_ANSWER (or an error/limit decision)
│
├── Evidence validation  structured-claims runs only (M8.8):
│                        claims[] + per-claim results[]
│                        (SUPPORTED / UNSUPPORTED / UNKNOWN),
│                        recorded AFTER the loop, before persistence —
│                        the Markdown projection renders it as a
│                        run-level section after the iterations
│                        (Answer → Structured Claims → M8.8
│                        Evidence Validation)
│
└── Summary             total_iterations, total_tool_calls,
                        tools_used, termination_reason, final_answer,
                        max_tool_calls, tool_calls_executed,
                        tool_calls_blocked (M8.5 budget)
```

Each iteration is one LLM decision cycle: *model responded* →
*observable decision* → (tool call → tool result → observation) *or*
final answer. The sequence inside an iteration is never reordered
during serialization, and the Markdown projection renders the M8.7.1
stage order — **LLM Request → LLM Response → Actions → Tool
Executions → Tool Results → State Update → Continuation** per
iteration (with **Answer** on the final one), followed by the
run-level **Structured Claims → M8.8 Evidence Validation** section
and the **Run Summary** — one file reads top-to-bottom as one
investigation.

Since M8.8 the trace also serves as the deterministic evidence source
for claim validation: `gather_evidence()` reads a run's trace (an
`AgentTrace` or this JSON) to verify that structured claims reference
evidence actually gathered — see
[evidence-traceability.md](evidence-traceability.md). Structured-
claims runs additionally record their outcome as an optional
`evidence_validation` block (`{"claims": [...], "results": [...]}`),
written after the investigation loop and before the sink persists the
trace; runs without structured claims gain no new field.

## 2. What is intentionally NOT captured

- **No hidden chain-of-thought.** The trace has no `thought`,
  `chain_of_thought`, `private_reasoning`, `hidden_reasoning`,
  `internal_monologue`, `reasoning`, or `scratchpad` field (enforced
  in code as `BANNED_REASONING_FIELDS` and asserted by tests). Nothing
  asks the model to reveal hidden reasoning.
- **No fabricated reasoning text.** `reason_summary` is a concise,
  developer-facing description of the *observable* action ("Model
  requested search_repository(...)"), never "the model thought X
  because…".
- **No model-authored validation prose.** The optional M8.8
  `evidence_validation` block carries the model's *structured* claims
  (its explicit output contract) plus the validator's deterministic
  reasons ("matched gathered evidence: ...") — the same ban list
  applies to it.
- **Observable conversation, clipped.** The trace records the
  messages actually *sent* (roles, chronological order, contents
  clipped to `TRACE_DATA_CHAR_LIMIT`) as the model's observable
  inputs, plus per-iteration request/response *metadata* (message
  counts, tool definition count, finish reason, token usage when
  present) and the explicit `llm_response` record of the output
  (type, tool-call count, exact requested tool calls). What went out
  and what came back — never what the model "thought" in between.
- **No secrets.** API keys, authorization headers, credentials, and
  environment variables are never read by the trace.
- **No bounded-agent execution state, memory, or multi-agent data.**
  The trace *records* the budget the run was configured with and any
  budget-blocked tool calls (observable execution facts, M8.5) — but
  it carries no planner state, memory, or coordination data; memory
  and multi-agent concerns are later milestones.

Because tool results may contain source code, recorded `data`/`source`
strings are clipped (`TRACE_DATA_CHAR_LIMIT`) so traces cannot grow
without limit; short values (arguments, ids, errors, provenance) are
kept verbatim.

## 3. Where it is stored

| Producer | Location |
| --- | --- |
| Every run | in-memory on `Agent.last_trace` (an `AgentTrace`) — always |
| Standalone CLI | `runtime/agent_runs/<run_id>/trace.json` (canonical) + `trace.md` (projection) |
| Tests | `InMemoryTraceSink` (list of traces) or `JsonFileTraceSink(tmp_path)` |

Traces are **execution telemetry**, not repository-intelligence
outputs, so the default root is deliberately outside `artifacts/`:

```text
runtime/agent_runs/<run_id>/     trace.json + trace.md  (this document)
artifacts/<run_id>/              facts, architecture, context, snapshots, …
```

Since M9 (investigation evaluation) the same run directory may also
contain `evaluation.json` + `evaluation.md` — a **separate artifact**
written after the run by `JsonFileEvaluationSink`, never by the Agent
and never part of the trace itself (`trace = what happened`,
`evaluation = how the completed run performed`; see
[investigation-evaluation.md](investigation-evaluation.md)).

Both are created on demand under the project root; `runtime/` is
gitignored (regenerable telemetry). The separation keeps the durable
evidence outputs (`artifacts/`) distinct from volatile run logs
(`runtime/`), and neither location is ever read by the Agent itself —
they are written by the composition root after a run.

The Agent only *builds* the trace; persistence is a separable
**trace sink** injected at the composition root:

```text
Agent ──► AgentTrace ──► TraceSink ──► local runtime file
                 (always)      (optional)
```

```python
from repo_intelligence.agent import Agent, JsonFileTraceSink

agent = Agent(
    llm_client=client,
    tool_dispatcher=dispatcher,
    trace_sink=JsonFileTraceSink(),   # optional; default root: runtime/agent_runs/
)
answer = agent.run(question)
print(agent.last_trace.summary.termination_reason)
```

The CLI prints `Trace: <path>` after the answer (and on the error
stream) so the trace file is easy to find. Two sinks exist for the MVP:
`InMemoryTraceSink` (tests/inspection) and `JsonFileTraceSink` (JSON +
Markdown files). No OpenTelemetry, no databases, no external services.

## 4. How to read an iteration

One iteration is one pass through the loop. Read it in this order —
the M8.7.1 stage order the Markdown projection uses:

1. **LLM Request** (`request`) — Stage *LLM REQUEST*: what went OUT:
   the messages actually sent (roles, chronological order, clipped
   contents) plus counts.
2. **LLM Response** (`llm_response`, `response`, `decision`) — Stage
   *LLM RESPONSE*: what came BACK: the response type, finish reason,
   the exact tool-call ids requested (`Tool calls requested:`), the
   model's text (`Text:` — or `Text: <none — response requested tool
   execution>` for a tool-call response), token usage, and the
   observable decision. A TOOL_CALL is itself a complete response —
   an absent text is never a missing response.
3. **Actions** (`llm_response.tool_calls` / `tool_calls`) — Stage
   *ACTION*: the exact requested calls (ids + arguments verbatim),
   before anything ran.
4. **Tool Executions** (`tool_calls` → `tool_results`) — Stage *TOOL
   EXECUTION*: the dispatch status of each requested call —
   `SUCCESS` / `FAILURE (code)` / `NOT EXECUTED (max_tool_calls
   reached)` — plus the observed `duration_ms` of executions that
   happened.
5. **Tool Results** (`tool_results`) — Stage *TOOL RESULT*: the
   recorded result per call — outcome, the observation returned to
   the Agent, and `source` provenance when the tool reported it.
6. **State Update** (`observations`) — Stage *STATE UPDATE*: what this
   iteration added to the conversation (the assistant message and the
   tool result messages, one bullet per observation) and which LLM
   request follows ("Next LLM request: iteration N (M messages)" —
   or "none (run terminated: …)").
7. **Continuation / Termination** (`continuation`) — Stage
   *CONTINUATION / TERMINATION*: the explicit loop decision recorded
   when the iteration closed — rendered as **Continuation** with
   prose ("Tool result will be returned to the LLM. Continue
   investigation.") for `CONTINUE` + `next_iteration`, or as
   **Termination** with prose ("Final answer received from LLM." /
   "Tool-call limit reached." / run-terminated reason) for
   `FINAL_ANSWER` / `TERMINATE` + `termination_reason` — always
   followed by the machine-readable decision lines.
8. **Answer** (`final_answer`) — Stage *FINAL ANSWER*, on the final
   iteration only.

After all iterations, structured-claims runs render the run-level
**Structured Claims → M8.8 Evidence Validation** section, then the
**Run Summary**.

Example iteration (tool step):

```json
{
  "iteration": 1,
  "response_type": "TOOL_CALL",
  "decision": { "type": "TOOL_CALL", "tool": "search_repository" },
  "request": {
    "message_count": 2, "tool_definition_count": 3, "model": "gpt-4o-mini",
    "messages": [
      { "role": "system", "content": "You are a repository intelligence assistant. …" },
      { "role": "user", "content": "How does the data onboarding workflow work?" }
    ]
  },
  "response": { "finish_reason": "tool_calls", "tool_call_count": 1,
                "has_text": false, "usage": { "input_tokens": 120, "output_tokens": 18, "total_tokens": 138 } },
  "llm_response": {
    "type": "TOOL_CALL", "tool_call_count": 1, "has_text": false,
    "tool_calls": [
      { "id": "call_1", "name": "search_repository",
        "arguments": { "query": "data onboarding", "max_results": 5 } }
    ]
  },
  "tool_calls": [
    { "id": "call_1", "name": "search_repository",
      "arguments": { "query": "data onboarding", "max_results": 5 } }
  ],
  "tool_results": [
    { "tool": "search_repository", "tool_call_id": "call_1", "success": true,
      "data": { "…": "clipped payload" }, "source": { "…": "provenance" },
      "error": null, "duration_ms": 3 }
  ],
  "observations": [
    { "type": "TOOL_RESULT", "tool": "search_repository",
      "success": true, "tool_call_id": "call_1" }
  ],
  "reason_summary": "Model requested search_repository({\"max_results\": 5, \"query\": \"data onboarding\"}).",
  "continuation": {
    "decision": "CONTINUE",
    "reason": "tool result added to conversation state",
    "next_iteration": 2
  }
}
```

Reading rules:

- **`request.messages`** — the observable conversation state the
  provider received: `role`, chronological order, and (clipped)
  `content`; assistant messages may carry `tool_calls`, tool messages
  carry `tool_call_id`. Counts stay in `message_count` for a quick
  glance; the growth 2 → 4 → 6 … IS the loop visible.
- **`llm_response`** — one explicit record of this LLM turn:
  `type` (`TOOL_CALL`, `FINAL_ANSWER`, `INVALID_RESPONSE` — the same
  classification the loop applies), `tool_call_count`, `has_text`,
  and, for a tool call, the verbatim `tool_calls` requested. Present
  only when a response actually arrived (an LLM failure records none).
- **`response_type`** — what the loop concluded from the LLM
  response: `TOOL_CALL`, `FINAL_ANSWER`, `ERROR` (the LLM call
  failed), `INVALID_RESPONSE` (neither text nor tool calls).
- **`decision.type`** — the observable decision: the response types
  plus `LIMIT_REACHED` (the tool-call limit was exhausted — the
  model asked for another tool after `max_tool_calls` was consumed)
  — always with the blocked tool name(s), never a rationale.
- **`tool_calls` vs `tool_results`** — requested (ids and arguments
  verbatim; never rewritten or inferred) vs executed (`success`, the
  structured `error`, `source` provenance, the size-clipped `data`,
  and `duration_ms` — the observed dispatch duration in whole
  milliseconds, recorded only for executions that actually ran).
  Same ids, two different questions: *asked* vs *happened*.
- **`continuation`** (M8.7.1) — the observable continue/terminate
  decision recorded when the iteration closed: `CONTINUE` with
  `next_iteration` (the number of the next LLM request), or
  `FINAL_ANSWER` / `TERMINATE` with the run's `termination_reason`
  vocabulary. `reason` is a one-line description of the observable
  action ("tool result added to conversation state"), never a
  rationale behind the model's decision. A trace recorded before
  M8.7.1 (or a hand-built one) has no `continuation` block — the
  Markdown projection derives the same decision from
  `decision.type` + the run's `termination_reason` instead.
- **`blocked_tool_calls`** (M8.5, present only when non-empty) — tool
  calls the model requested but the tool-call limit blocked: they
  were **not executed**, so there is deliberately no `success` flag
  and no fabricated `ToolResult` (a skipped tool ≠ a failed tool).
  Each entry carries `reason`, `tool_calls_executed`, and
  `max_tool_calls`.
- **`observations`** — the identity of what was appended to the
  conversation (`type: TOOL_RESULT` + the same `tool_call_id`), so an
  observation is traceable to the call that produced it. The payload
  itself is the `tool_result` above — it is not duplicated.
- **`repository`** (run-level, optional) — provenance taken from an
  actual ToolResult `source`; a text-only run has no `repository`
  block because nothing was analyzed.

There is deliberately no synthetic `request_id`: one numbered
iteration IS one LLM interaction, and ids exist only where the
provider protocol requires them (tool-call ids).

### Termination reasons

| Reason | Meaning |
| --- | --- |
| `FINAL_ANSWER` | the model returned text; the answer is returned |
| `LLM_ERROR` | the provider/client raised `LLMError` |
| `INVALID_LLM_RESPONSE` | response had neither text nor tool calls |
| `TOOL_DISPATCH_ERROR` | no dispatcher injected, or dispatch raised |
| `EXECUTION_LIMIT_REACHED` | the `max_tool_calls` budget was exhausted; the blocked tool was NOT executed (M8.5) |
| `ERROR` | safety net for any other unexpected exception |

## 5. `total_iterations` vs `total_tool_calls` (and the M8.5 budget)

They count different things — the final-answer cycle is an iteration
too:

```text
Iteration 1 → ToolCall (search_repository)   [executed 1]
Iteration 2 → ToolCall (read_file)           [executed 2]
Iteration 3 → Final Answer

total_iterations     = 3   # every LLM decision cycle
total_tool_calls     = 2   # ToolCalls the model requested
tool_calls_executed  = 2   # dispatches performed (budget consumed)
tool_calls_blocked   = 0   # requested but never executed
max_tool_calls       = 2   # configured budget (when known)
```

On a budget termination the requested count exceeds the executed one —
the difference is exactly `tool_calls_blocked` (e.g. requested 3,
executed 2, blocked 1 with `max_tool_calls = 2`).

`tools_used` lists the tools that actually executed, in order,
deduplicated (blocked tools never appear — they never ran). All of the
above appear in `summary`.

## 6. Example — complete multi-step run (JSON, canonical)

Real output of a deterministic offline end-to-end run (scripted
model, real dispatcher, real pinned checkout): `get_architecture_context`
→ 3× `read_file` (one `FILE_NOT_FOUND`) → structured final answer with
4 claims. Abridged in message contents, tool payloads, and validation
reasons (`"…"` placeholders) — field names and shapes are verbatim:

```json
{
  "run_id": "5944d3f0dc1e4b20b100a6b02b958b0e",
  "started_at": "2026-09-29T15:45:14.310Z",
  "completed_at": "2026-09-29T15:45:14.312Z",
  "agent": { "name": "repo_intelligence.agent", "version": "0.3.0" },
  "user_question": "Investigate which agents, tools, and MCP (Model Context Protocol) connections are defined in this repository …",
  "repository": {
    "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
  },
  "iterations": [
    {
      "iteration": 1,
      "response_type": "TOOL_CALL",
      "decision": { "type": "TOOL_CALL", "tool": "get_architecture_context" },
      "request": {
        "message_count": 2, "tool_definition_count": 3, "model": null,
        "messages": [
          { "role": "system", "content": "You are a repository intelligence assistant. Answer the user's question using only the information supplied to you. …" },
          { "role": "user", "content": "Investigate which agents, tools, and MCP (Model Context Protocol) connections are defined in this repository …" }
        ]
      },
      "response": { "finish_reason": "tool_calls", "tool_call_count": 1,
                    "has_text": false, "usage": null },
      "llm_response": {
        "type": "TOOL_CALL", "tool_call_count": 1, "has_text": false,
        "tool_calls": [
          { "id": "c1", "name": "get_architecture_context", "arguments": {} }
        ]
      },
      "tool_calls": [
        { "id": "c1", "name": "get_architecture_context", "arguments": {} }
      ],
      "tool_results": [
        { "tool": "get_architecture_context", "tool_call_id": "c1", "success": true,
          "data": { "context": "# Architecture Context\n\n- Contract: `architecture_context.v1`\n\n## Repo…" },
          "source": { "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/architecture_context.md",
                      "schema_version": "architecture_context.v1",
                      "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0" },
          "error": null,
          "duration_ms": 0 }
      ],
      "observations": [
        { "type": "TOOL_RESULT", "tool": "get_architecture_context",
          "success": true, "tool_call_id": "c1" }
      ],
      "reason_summary": "Model requested get_architecture_context({}).",
      "continuation": { "decision": "CONTINUE",
                        "reason": "tool result added to conversation state",
                        "next_iteration": 2 }
    },
    {
      "iteration": 2,
      "response_type": "TOOL_CALL",
      "decision": { "type": "TOOL_CALL", "tools": ["read_file", "read_file", "read_file"] },
      "request": {
        "message_count": 4, "tool_definition_count": 3, "model": null,
        "messages": [
          { "…": "system, user, assistant(tool_calls c1), tool(c1)" }
        ]
      },
      "response": { "finish_reason": "tool_calls", "tool_call_count": 3,
                    "has_text": false, "usage": null },
      "llm_response": {
        "type": "TOOL_CALL", "tool_call_count": 3, "has_text": false,
        "tool_calls": [
          { "id": "c2", "name": "read_file",
            "arguments": { "path": ".claude/agents/dag-agent.md", "line_start": 1, "line_end": 20 } },
          { "id": "c3", "name": "read_file",
            "arguments": { "path": "tool-registry/README.md", "line_start": 1, "line_end": 50 } },
          { "id": "c4", "name": "read_file",
            "arguments": { "path": "tool-registry/servers.yaml", "line_start": 1, "line_end": 30 } }
        ]
      },
      "tool_calls": [ { "…": "same three calls as llm_response.tool_calls" } ],
      "tool_results": [
        { "tool": "read_file", "tool_call_id": "c2", "success": true,
          "data": { "path": ".claude/agents/dag-agent.md", "line_start": 1, "line_end": 20,
                    "content": "---\nname: dag-agent\ndescription: Builds the DAG spec and renders the w…" },
          "source": { "repository": "sample-Agentic-Ai-Data-Operations",
                      "commit_sha": "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
                      "snapshot_id": "main_2026-09-22_ef11dac",
                      "repository_url": "https://github.com/aws-samples/sample-Agentic-Ai-Data-Operations.git" },
          "error": null,
          "duration_ms": 0 },
        { "tool": "read_file", "tool_call_id": "c3", "success": false,
          "data": null, "source": null,
          "error": { "code": "FILE_NOT_FOUND",
                     "message": "Requested file was not found: 'tool-registry/README.md'" },
          "duration_ms": 0 },
        { "tool": "read_file", "tool_call_id": "c4", "success": true,
          "data": { "path": "tool-registry/servers.yaml", "line_start": 1, "line_end": 30,
                    "content": "# tool-registry/servers.yaml\n# Single source of truth for MCP server m…" },
          "source": { "…": "provenance (same pinned snapshot)" },
          "error": null,
          "duration_ms": 0 }
      ],
      "observations": [
        { "type": "TOOL_RESULT", "tool": "read_file", "success": true,  "tool_call_id": "c2" },
        { "type": "TOOL_RESULT", "tool": "read_file", "success": false, "tool_call_id": "c3" },
        { "type": "TOOL_RESULT", "tool": "read_file", "success": true,  "tool_call_id": "c4" }
      ],
      "reason_summary": "Model requested 3 tool calls: read_file, read_file, read_file.",
      "continuation": { "decision": "CONTINUE",
                        "reason": "tool result added to conversation state",
                        "next_iteration": 3 }
    },
    {
      "iteration": 3,
      "response_type": "FINAL_ANSWER",
      "decision": { "type": "FINAL_ANSWER" },
      "request": {
        "message_count": 8, "tool_definition_count": 3, "model": null,
        "messages": [
          { "…": "system, user, assistant(c1), tool(c1), assistant(c2,c3,c4), tool(c2), tool(c3), tool(c4)" }
        ]
      },
      "response": { "finish_reason": "stop", "tool_call_count": 0,
                    "has_text": true, "usage": null },
      "llm_response": {
        "type": "FINAL_ANSWER", "tool_call_count": 0, "has_text": true
      },
      "tool_calls": [],
      "tool_results": [],
      "observations": [],
      "final_answer": "The repository (ADOP) defines six agent definitions under .claude/agents/ (dag-agent, iac-agent…",
      "reason_summary": "Model returned a final textual answer (295 characters).",
      "continuation": { "decision": "FINAL_ANSWER",
                        "reason": "LLM returned final answer",
                        "termination_reason": "FINAL_ANSWER" }
    }
  ],
  "summary": {
    "total_iterations": 3,
    "total_tool_calls": 4,
    "tools_used": [ "get_architecture_context", "read_file" ],
    "termination_reason": "FINAL_ANSWER",
    "final_answer": "The repository (ADOP) defines six agent definitions under .claude/agents/ (dag-agent, iac-agent…",
    "max_tool_calls": 8,
    "tool_calls_executed": 4,
    "tool_calls_blocked": 0
  },
  "evidence_validation": {
    "claims": [
      { "statement": "The repository defines a DAG Agent (.claude/agents/dag-agent.md).",
        "evidence_refs": [ { "path": ".claude/agents/dag-agent.md", "line_start": 1, "line_end": 20 } ] },
      { "statement": "MCP servers are configured in tool-registry/servers.yaml.",
        "evidence_refs": [ { "path": "tool-registry/servers.yaml", "line_start": 1, "line_end": 30 } ] },
      { "statement": "tool-registry/README.md documents the tool registry.",
        "evidence_refs": [ { "path": "tool-registry/README.md", "line_start": 1, "line_end": 50 } ] },
      { "statement": "The investigation stayed within the tool-call limit.",
        "evidence_refs": [] }
    ],
    "results": [
      { "claim": { "…": "same claim objects as above" }, "status": "SUPPORTED",
        "matched_evidence": [ { "path": ".claude/agents/dag-agent.md", "line_start": 1, "line_end": 20,
                                "tool": "read_file", "tool_call_id": "c2", "source": { "…": "provenance" } } ],
        "reason": "matched gathered evidence: read_file .claude/agents/dag-agent.md:1-20" },
      { "claim": { "…" }, "status": "SUPPORTED",
        "matched_evidence": [ { "path": "tool-registry/servers.yaml", "line_start": 1, "line_end": 30,
                                "tool": "read_file", "tool_call_id": "c4", "source": { "…": "provenance" } } ],
        "reason": "matched gathered evidence: read_file tool-registry/servers.yaml:1-30" },
      { "claim": { "…" }, "status": "UNSUPPORTED", "matched_evidence": [],
        "reason": "no gathered evidence matches: tool-registry/README.md:1-50 (attempted during the run and failed: FILE_NOT_FOUND)" },
      { "claim": { "…" }, "status": "UNKNOWN", "matched_evidence": [],
        "reason": "no evidence references supplied; traceability cannot be evaluated (not counted as supported)" }
    ]
  }
}
```

No `thought` field exists anywhere in this document — the flow is
fully described by observable inputs (`request.messages`), outputs
(`llm_response`), decisions, actions, per-execution durations,
observations, and each iteration's explicit `continuation`.

## 7. Example — Markdown projection

`trace.md` is rendered from the JSON above (never maintained
separately) and reads top-to-bottom as the M8.7.1 stage sequence:
per iteration **LLM Request → LLM Response → Actions → Tool
Executions → Tool Results → State Update → Continuation/Termination**
(plus **Answer** on the final one), then the run-level **Structured
Claims → M8.8 Evidence Validation** section and the **Run
Summary**. Real output of the same offline run as §6 (long message
contents are clipped by the trace itself with `[trace clipped:
…]`; message excerpts are shortened with `…`):

````markdown
# Agent Execution Trace

* Run: `5944d3f0dc1e4b20b100a6b02b958b0e`
* Started: 2026-09-29T15:45:14.310Z
* Completed: 2026-09-29T15:45:14.312Z
* Agent: repo_intelligence.agent 0.3.0
* Repository commit: `ef11dac8618b6344d6a5e4add6ef51cd2811d7e0`

Question:
> Investigate which agents, tools, and MCP (Model Context Protocol) connections are defined in this repository (the ADOP / Agentic-AI Data-Operations project). Start from the architecture context, but do not rely solely on it: use the repository tools to gather supporting evidence from files in the repository itself, and write your final answer with specific references to the files you actually inspected (citing paths from the evidence), naming the agent definitions, tool registry entries, and MCP server configuration you found.

---

## Iteration 1

### LLM Request

**Messages:** 2
**Tools available:** 3

1. **system** — You are a repository intelligence assistant. Answer the user's question using only the information supplied to you. When the investigation is complete, your FIN…
2. **user** — Investigate which agents, tools, and MCP (Model Context Protocol) connections are defined in this repository (the ADOP / Agentic-AI Data-Operations project). St…

### LLM Response

**Decision:** `TOOL_CALL`

**Tool:** `get_architecture_context`

Type: `TOOL_CALL` · Finish reason: `tool_calls`
Tool calls requested: `c1`

Text: <none — response requested tool execution>

### Actions

**1. `get_architecture_context`** (`c1`)
```json
{}
```

### Tool Executions

* `c1` (`get_architecture_context`) → SUCCESS · duration: 0 ms

### Tool Results

**`c1` (`get_architecture_context`)**

**Result:** SUCCESS

**Observation:** get_architecture_context result returned to the Agent.

**Source:** artifacts/main_2026-09-22_ef11dac/architecture_context/architecture_context.md, commit ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

### State Update

* assistant message added (1 requested tool call: `c1`)
* `get_architecture_context` (`c1`) → SUCCESS
Next LLM request: iteration 2 (4 messages)

*Summary:* Model requested get_architecture_context({}).

### Continuation

Tool result will be returned to the LLM.
Continue investigation.

Decision: CONTINUE
Reason: tool result added to conversation state
Next iteration: 2

---

## Iteration 2

### LLM Request

**Messages:** 4
**Tools available:** 3
**Includes 1 previous observation:**
* `get_architecture_context` (`c1`) → SUCCESS

1. **system** — You are a repository intelligence assistant. Answer the user's question using only the information supplied to you. When the investigation is complete, your FIN…
2. **user** — Investigate which agents, tools, and MCP (Model Context Protocol) connections are defined in this repository (the ADOP / Agentic-AI Data-Operations project). St…
3. **assistant**
   * requested `get_architecture_context` (`c1`)
4. **tool** (`c1`) — {"success": true, "tool": "get_architecture_context", "data": {"context": "# Architecture Context\n\n- Contract: `architecture_context.v1`\n\n## Repository\n\n-…

### LLM Response

**Decision:** `TOOL_CALL`

**Tools:** `read_file`, `read_file`, `read_file`

Type: `TOOL_CALL` · Finish reason: `tool_calls`
Tool calls requested: `c2`, `c3`, `c4`

Text: <none — response requested tool execution>

### Actions

**1. `read_file`** (`c2`)
```json
{
  "line_end": 20,
  "line_start": 1,
  "path": ".claude/agents/dag-agent.md"
}
```

**2. `read_file`** (`c3`)
```json
{
  "line_end": 50,
  "line_start": 1,
  "path": "tool-registry/README.md"
}
```

**3. `read_file`** (`c4`)
```json
{
  "line_end": 30,
  "line_start": 1,
  "path": "tool-registry/servers.yaml"
}
```

### Tool Executions

* `c2` (`read_file`) → SUCCESS · duration: 0 ms
* `c3` (`read_file`) → FAILURE (FILE_NOT_FOUND) · duration: 0 ms
* `c4` (`read_file`) → SUCCESS · duration: 0 ms

### Tool Results

**`c2` (`read_file`)**

**Result:** SUCCESS

**Observation:** read_file result returned to the Agent.

**Source:** sample-Agentic-Ai-Data-Operations, snapshot main_2026-09-22_ef11dac, commit ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

**`c3` (`read_file`)**

**Result:** FAILURE (FILE_NOT_FOUND)

**Observation:** read_file failure (FILE_NOT_FOUND) returned to the Agent.

**`c4` (`read_file`)**

**Result:** SUCCESS

**Observation:** read_file result returned to the Agent.

**Source:** sample-Agentic-Ai-Data-Operations, snapshot main_2026-09-22_ef11dac, commit ef11dac8618b6344d6a5e4add6ef51cd2811d7e0

### State Update

* assistant message added (3 requested tool calls: `c2`, `c3`, `c4`)
* `read_file` (`c2`) → SUCCESS
* `read_file` (`c3`) → FAILURE (FILE_NOT_FOUND)
* `read_file` (`c4`) → SUCCESS
Next LLM request: iteration 3 (8 messages)

*Summary:* Model requested 3 tool calls: read_file, read_file, read_file.

### Continuation

Tool result will be returned to the LLM.
Continue investigation.

Decision: CONTINUE
Reason: tool result added to conversation state
Next iteration: 3

---

## Iteration 3

### LLM Request

**Messages:** 8
**Tools available:** 3
**Includes 4 previous observations:**
* `get_architecture_context` (`c1`) → SUCCESS
* `read_file` (`c2`) → SUCCESS
* `read_file` (`c3`) → FAILURE (FILE_NOT_FOUND)
* `read_file` (`c4`) → SUCCESS

1. **system** — You are a repository intelligence assistant. Answer the user's question using only the information supplied to you. When the investigation is complete, your FIN…
2. **user** — Investigate which agents, tools, and MCP (Model Context Protocol) connections are defined in this repository (the ADOP / Agentic-AI Data-Operations project). St…
3. **assistant**
   * requested `get_architecture_context` (`c1`)
4. **tool** (`c1`) — {"success": true, "tool": "get_architecture_context", "data": {"context": "# Architecture Context\n\n- Contract: `architecture_context.v1`\n\n## Repository\n\n-…
5. **assistant**
   * requested `read_file` (`c2`)
   * requested `read_file` (`c3`)
   * requested `read_file` (`c4`)
6. **tool** (`c2`) — {"success": true, "tool": "read_file", "data": {"path": ".claude/agents/dag-agent.md", "line_start": 1, "line_end": 20, "content": "---\nname: dag-agent\ndescri…
7. **tool** (`c3`) — {"success": false, "tool": "read_file", "data": null, "source": null, "error": {"code": "FILE_NOT_FOUND", "message": "Requested file was not found: 'tool-regist…
8. **tool** (`c4`) — {"success": true, "tool": "read_file", "data": {"path": "tool-registry/servers.yaml", "line_start": 1, "line_end": 30, "content": "# tool-registry/servers.yaml\…

### LLM Response

**Decision:** `FINAL_ANSWER`

Type: `FINAL_ANSWER` · Finish reason: `stop`

Text:
The repository (ADOP) defines six agent definitions under .claude/agents/ (dag-agent, iac-agent, metadata-agent, ontology-agent, quality-agent, transformation-agent). MCP servers are configured in tool-registry/servers.yaml. The tool registry's own README could not be inspected during this run.

*Summary:* Model returned a final textual answer (295 characters).

### Termination

Final answer received from LLM.

Decision: FINAL_ANSWER
Reason: LLM returned final answer
Termination: `FINAL_ANSWER`

### Answer

The repository (ADOP) defines six agent definitions under .claude/agents/ (dag-agent, iac-agent, metadata-agent, ontology-agent, quality-agent, transformation-agent). MCP servers are configured in tool-registry/servers.yaml. The tool registry's own README could not be inspected during this run.

---

### Structured Claims

1. The repository defines a DAG Agent (.claude/agents/dag-agent.md).
   * evidence: `.claude/agents/dag-agent.md:1-20`
2. MCP servers are configured in tool-registry/servers.yaml.
   * evidence: `tool-registry/servers.yaml:1-30`
3. tool-registry/README.md documents the tool registry.
   * evidence: `tool-registry/README.md:1-50`
4. The investigation stayed within the tool-call limit.

### M8.8 Evidence Validation

* `SUPPORTED` — The repository defines a DAG Agent (.claude/agents/dag-agent.md).
  * reason: matched gathered evidence: read_file .claude/agents/dag-agent.md:1-20
* `SUPPORTED` — MCP servers are configured in tool-registry/servers.yaml.
  * reason: matched gathered evidence: read_file tool-registry/servers.yaml:1-30
* `UNSUPPORTED` — tool-registry/README.md documents the tool registry.
  * reason: no gathered evidence matches: tool-registry/README.md:1-50 (attempted during the run and failed: FILE_NOT_FOUND)
* `UNKNOWN` — The investigation stayed within the tool-call limit.
  * reason: no evidence references supplied; traceability cannot be evaluated (not counted as supported)

---

## Run Summary

* Total iterations: 3
* Tool calls: 4
* Tool-call budget: 8 max, 4 executed, 0 blocked
* Tools used: `get_architecture_context`, `read_file`
* Termination: `FINAL_ANSWER`

> The repository (ADOP) defines six agent definitions under .claude/agents/ (dag-agent, iac-agent, metadata-agent, ontology-agent, quality-agent, transformation-agent). MCP servers are configured in tool-registry/servers.yaml. The tool registry's own README could not be inspected during this run.
````

## 8. Trace failures do not break the Agent

Observability is secondary to execution. If the sink fails
(filesystem, serialization, anything):

- the Agent result (answer or raised error) is unchanged;
- the failure is recorded as error metadata —
  `Agent.last_trace_error` and `AgentTrace.sink_error`;
- it is never silently ignored as success and never re-raised into
  the run.

## 9. Relationship to ReAct

The trace makes the ReAct-style *action/observation* pattern visible:

```text
decision → action (tool call) → observation → decision → … → answer
```

but it records only the observable boundary — and since M8.7.1 the
Markdown projection spells that boundary out per iteration as the
ordered stages **LLM Request → LLM Response → Actions → Tool
Executions → Tool Results → State Update → Continuation/Termination**
(what went out, what came back and its exact text, what was
requested, what executed and how long it took, what went back, and
whether the run continues or ends).
There is no fake `thought` field: the goal is to understand the
mechanism, not to manufacture hidden reasoning. This is also why the
milestone calls it a *plain tool-calling loop with observability* —
not a ReAct reasoning implementation.

The same trace backs both ReAct demonstrations: M8.6
(`agentrun_react_test.py`, deterministic scripted model, fully
offline) and M8.7 (`agentrun_react_live_test.py`, real
`OpenAILLMClient` round trips). Only the model source differs — the
loop, dispatcher, tools, and trace schema are identical.

## 10. Explicitly out of scope

- OpenTelemetry / Jaeger / Datadog / LangSmith / any external
  tracing SaaS, databases, queues, distributed trace IDs;
- memory, multi-agent coordination, RAG (not introduced);
- bounded-agent *control* lives in the Agent, not the trace — the
  trace only **records** what M8.5 did: the configured
  `max_tool_calls`, any budget-blocked tool calls, and the
  `EXECUTION_LIMIT_REACHED` termination (see
  [bounded-agent.md](bounded-agent.md)).

Traces stay local: `Agent → structured execution trace → local
runtime file`. Tool results can contain repository content, so treat
trace files as potentially sensitive project data — they live under
`runtime/agent_runs/` (execution telemetry), deliberately separated
from the durable `artifacts/` outputs.

## 11. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agent/test_agent_trace.py -q
```

covers: final answer / one tool / multiple tools (exact ordering) /
tool failure / LLM failure / tool-call limit + blocked-tool records /
exact arguments / no hidden reasoning fields / pinned-snapshot
provenance / iteration numbering / sink behavior (in-memory, file,
failing) / JSON determinism / Markdown projection / data clipping /
runtime relocation (`runtime/agent_runs/<run_id>/trace.json`, never
under `artifacts/`). The readability follow-up (recorded messages,
the explicit `llm_response` record, requested-vs-executed clarity,
run-level claims + M8.8 validation section) is covered by
`tests/agent/test_agent_trace_readability.py`. The M8.7.1 stage
sequence (stage order, `continuation`, per-execution `duration_ms`,
blocked-never-rendered-as-executed, validation placement) is covered
by `tests/agent/test_agent_trace_sequence.py`. The M8.8
`evidence_validation` block (structured runs only) is covered by
`tests/evidence/test_evidence_integration.py` (including the closeout
regression: a structured payload with claims but an unusable `answer`
field still reaches the validator).
Budget-specific assertions live in
`tests/agent/test_agent_bounded.py`.
