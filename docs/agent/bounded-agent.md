# M8.5 — Bounded Agent (explicit execution control)

> **Status:** implemented. M8.4 taught the Agent to *continue* after a
> `ToolResult`; M8.5 teaches it when it must **stop**. One explicit
> budget, one explicit termination vocabulary, no new frameworks.

## 1. Why an Agent needs a bound

A tool-calling loop with no bound can run forever — a runaway model, a
buggy adapter, or a tool the model keeps re-requesting. "It stopped
eventually" is not a design; a controlled, testable stop is.

The central learning point of this milestone:

> **An Agent is not just a loop. An Agent is a loop with explicit
> state, actions, observations, and termination conditions.**

M8.4 gave us the action/observation cycle:

```text
action → observation → action → observation
```

M8.5 adds the control boundary around it:

```text
action → observation → action → observation
                         ↑
                    bounded by policy
```

The budget is **execution control**, never conversation state:

```text
Conversation State    = what the LLM has observed
Execution Budget      = what the Agent is still allowed to do
```

## 2. The contract — `AgentExecutionConfig`

```python
@dataclass(frozen=True)
class AgentExecutionConfig:
    max_tool_calls: int = 8
    structured_claims: bool = False   # M8.8 integration, opt-in
```

That is the whole configuration model: one frozen value object,
injected at construction (dependency injection — not a scattered
module-level constant):

```python
from repo_intelligence.agent import Agent, AgentExecutionConfig

agent = Agent(
    llm_client=client,
    tool_dispatcher=dispatcher,
    execution_config=AgentExecutionConfig(max_tool_calls=3),
)
answer = agent.run(question)
```

`Agent()` without a config uses the finite, safe default
(**`max_tool_calls = 8`**). A negative value is rejected at
construction (`ValueError`); `0` is legal and means *no tool may
execute*.

`structured_claims` (M8.8 integration, default `False`) does not
change execution at all: `True` only asks the model for its final
response as the explicit `{answer, claims}` JSON contract, after
which the run validates those claims deterministically against its
own trace (see [evidence-traceability.md](evidence-traceability.md)).
Every existing free-form run keeps it `False` — same loop, same
answer, no new trace field.

**Exact meaning of `max_tool_calls`:** the maximum number of
**individual tool executions** (dispatches) one `run` may perform.
The check happens **before** every execution — never after:

```text
check budget
    ↓
allowed? ── yes ──► execute tool ──► increment counter
    ↓
  block: the requested tool is NOT executed
         (recorded as blocked, never a fake ToolResult)
         terminate: EXECUTION_LIMIT_REACHED
```

This is the single limit. The M8.4 `MAX_TOOL_CALL_STEPS` guard was
**refactored into** this mechanism — there is no second competing
limit anywhere.

CLI support is the minimum necessary:

```bash
python -m repo_intelligence.agent --max-tool-calls 3   # budget for one run
python -m repo_intelligence.agent --max-tool-calls 0   # tools disabled
```

(default: 8; a negative value is an argparse usage error.)

## 3. Iterations vs tool calls

Keep the two concepts distinct — the budget counts the second one:

| Concept | What it counts | Example |
| --- | --- | --- |
| **iteration** | one Agent ↔ LLM interaction cycle | request → response |
| **tool call** | one individual tool execution | one dispatch, success *or* failure |

```text
iteration 1:  LLM → ToolCall (search)        [tool call 1]
iteration 2:  LLM → ToolCall (read_file)     [tool call 2]
iteration 3:  LLM → Final Answer             (no tool call)
```

→ 3 iterations, 2 tool calls, budget consumed = 2.

If one LLM response contains multiple tool calls, each is an
individual execution consuming one unit:

```text
max_tool_calls = 2
LLM → Tool A, Tool B, Tool C
A → executed [1]   B → executed [2]   C → NOT executed → terminate
```

One subtlety worth understanding: the budget gates **executions**,
not LLM calls. After the budget is exactly consumed, the model still
gets its next turn — it may return a final answer (success!) or
request another tool (blocked → `EXECUTION_LIMIT_REACHED`). The loop
below shows this in §9.

## 4. What happens when the budget is exhausted

- the requested tool is **not executed** (checked before dispatch);
- the call is recorded as `blocked_tool_calls` on the iteration —
  explicitly *not* a `ToolResult` (a skipped tool is not a failed
  tool: there is no `success` flag, no data, no fabricated result);
- the run terminates with `EXECUTION_LIMIT_REACHED`;
- `Agent.run` raises a structured `AgentError` whose
  `termination_reason` is `"EXECUTION_LIMIT_REACHED"` (the CLI exits
  `6`) — distinguishable from a successful final answer without
  reading any text;
- **no** fabricated final answer, **no** silent truncation, **no**
  hidden retry, **no** provider/model switch.

## 5. Tool failures and the budget

Unchanged M8.4 semantics — a failed `ToolResult` is an *observation*
the model may react to — but every attempt consumes one budget unit:

```text
max_tool_calls = 3

search → failure   [1]   (observed by the LLM, still spent a unit)
search → success   [2]
read_file → ok     [3]
(no fourth tool call may ever execute)
```

There is no application-level retry: only a *new model-requested*
ToolCall can start another execution, and only while budget remains.

## 6. LLM failures terminate execution

```text
Agent → LLM → failure → terminate (LLM_ERROR)
```

The `LLMError` propagates unchanged, the trace records `LLM_ERROR`,
and the run ends. No `retry 3 times`, no exponential backoff, no
model fallback, no provider switching — retry policy is a future
milestone, not this one.

## 7. How the trace exposes termination

The existing M8.4 execution trace (one system, no second
observability stack) now carries the budget:

```json
"summary": {
  "total_iterations": 3,
  "total_tool_calls": 3,
  "tools_used": ["search_repository", "read_file"],
  "termination_reason": "EXECUTION_LIMIT_REACHED",
  "final_answer": null,
  "max_tool_calls": 2,
  "tool_calls_executed": 2,
  "tool_calls_blocked": 1
}
```

and the blocked iteration records the never-executed request:

```json
{
  "iteration": 3,
  "decision": { "type": "LIMIT_REACHED", "tool": "search_repository" },
  "tool_calls": [ { "id": "call_3", "name": "search_repository", "…": "…" } ],
  "tool_results": [],
  "observations": [],
  "blocked_tool_calls": [
    { "tool": "search_repository", "tool_call_id": "call_3",
      "reason": "EXECUTION_LIMIT_REACHED",
      "tool_calls_executed": 2, "max_tool_calls": 2 }
  ]
}
```

Reading rules: `total_tool_calls` = what the model **requested**
(= executed + blocked on a budget termination); `tool_calls_executed`
= budget consumed; `tool_calls_blocked` = requested but never
executed. The Markdown projection keeps the two visibly distinct:
the `Tool Executions` stage shows
`→ NOT EXECUTED (max_tool_calls reached)` for the blocked call (no
`duration`, never `SUCCESS`), the `Tool Results` stage shows
`**Result:** NOT EXECUTED (execution budget exhausted)`, and the run
ends with a `* Tool-call budget: N max, E executed, B blocked`
summary line plus a `TERMINATE` / `max_tool_calls reached`
continuation (M8.7.1).
No hidden reasoning fields are added — observable budget state only
(`BANNED_REASONING_FIELDS` still enforced).

## 8. Why there is no automatic retry

M8.5 answers *"how far is the Agent allowed to continue acting?"* —
not *"how do we recover from failure?"*. Retrying silently would
hide both budget consumption and failure, making termination
unobservable. Failures terminate; the caller decides what next.

## 9. Execution example (`max_tool_calls = 2`)

```text
User
 ↓
LLM → search_repository
 ↓
ToolResult                [1 executed]
 ↓
LLM → read_file
 ↓
ToolResult                [2 executed — budget now exhausted]
 ↓
LLM → search_repository
 ↓
BUDGET CHECK: 2 of 2 executed → BLOCKED
  search_repository was **NOT executed**
  (no ToolResult, no observation, no fabricated answer)
 ↓
Agent terminates: EXECUTION_LIMIT_REACHED
```

The third tool was **not executed**. Had the model returned a final
answer after the second ToolResult instead, the run would have ended
`FINAL_ANSWER` — the budget gates executions, not answers.

## 10. Termination reasons (explicit, machine-readable)

| Reason | Meaning |
| --- | --- |
| `FINAL_ANSWER` | model returned text; the answer is returned (str) |
| `EXECUTION_LIMIT_REACHED` | tool-call budget exhausted; blocked tool NOT executed (`AgentError.termination_reason`) |
| `LLM_ERROR` | provider/client raised `LLMError` (no retry) |
| `INVALID_LLM_RESPONSE` | response had neither text nor tool calls |
| `TOOL_DISPATCH_ERROR` | no dispatcher injected, or dispatch raised |
| `ERROR` | safety net for any other unexpected exception |

## 11. What M8.5 deliberately does NOT introduce

No ReAct framework, planning module, chain-of-thought capture, memory,
embeddings/vector stores/RAG, multi-agent or supervisor execution,
workflow engine, LangGraph/CrewAI/AutoGen, OpenTelemetry/LangSmith,
retry framework, model fallback, token/cost optimization, concurrency,
or parallel tool execution. The Agent remains a single class owning a
single bounded loop:

```text
Agent
 ├── LLMClient
 ├── ToolDispatcher
 ├── ExecutionConfig        ← M8.5 (this milestone)
 └── TraceSink
```

## 12. Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agent/test_agent_bounded.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agent -q
```

`tests/agent/test_agent_bounded.py` covers: final answer before
exhaustion / one tool then final / multiple tools within budget /
budget exhaustion (third call blocked) / `max_tool_calls = 0` /
`max_tool_calls = 1` (no off-by-one) / tool failure consuming budget /
multiple tool calls in one response (A, B executed; C blocked) / LLM
failure without retry / final answer terminal / trace budget fields
distinguishing executed from blocked / configuration validation. The
CLI flag is covered in `tests/agent/test_agent_cli.py`; all M8.1–M8.4
regression tests remain unchanged and passing.
