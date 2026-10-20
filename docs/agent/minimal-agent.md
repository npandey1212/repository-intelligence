# M8.1 — Minimal Agent (`Agent`)

The first Agent object in the repository intelligence project: one
question in, one textual answer out. Deliberately minimal — no tool
execution, no loop, no state.

> **Status:** this document describes the M8.1 baseline (and the M8.2
> extension: provider-neutral tool definitions sent with the request).
> M8.3 now executes exactly one requested tool call through the
> application-owned dispatcher, and M8.4 feeds each `ToolResult` back
> as a tool observation and calls the LLM again until a textual answer
> — see [tool-definitions.md](tool-definitions.md) §9.4.

```text
User Question
      ↓
    Agent
      ↓
  LLMClient
      ↓
OpenAI implementation
      ↓
    OpenAI
```

---

## 13.1 Purpose

The previous milestones built everything the Agent will eventually use:

```text
Pinned snapshot → facts → knowledge → architecture context
        → read-only tools → LLM abstraction
                ↓
        NOW: Minimal Agent
```

The key concept this milestone teaches:

> **An Agent is an application component that uses an LLM to accomplish
> a task.**

The `Agent` object exists to own the application-level interaction with
the LLM: it takes a user question, formulates the request, sends it
through the provider-neutral `LLMClient`, and returns the answer. That
is the entire component — small enough to inspect completely before
tool calling makes it interesting.

## 13.2 Current Scope

This is the **minimal Agent**. It only:

1. accepts a question,
2. constructs an `LLMRequest` (system + user message),
3. calls `LLMClient.complete()`,
4. inspects the `LLMResponse`,
5. returns the textual answer.

It explicitly does **not**: execute tools, register/dispatch tools,
loop, plan, retry, remember, or hold state. Those belong to later
milestones (13.9) — this milestone stops before any of them.

## 13.3 Architecture

```text
src/repo_intelligence/
│
├── agent/
│   ├── __init__.py      # exports Agent, AgentError, SYSTEM_PROMPT
│   ├── agent.py         # the Agent — depends only on repo_intelligence.llm
│   └── __main__.py      # composition root: python -m repo_intelligence.agent
│
├── llm/                 # existing provider-neutral abstraction
├── tools/               # existing repository tools (not connected yet)
└── ...
```

| Piece | Role |
| --- | --- |
| `Agent` | `__init__(llm_client)` + `run(question) -> str` |
| `AgentError` | project-level failure: invalid question, tool-call response, missing text |
| `SYSTEM_PROMPT` | one minimal system instruction owned by the Agent |
| `agent/__main__.py` | standalone execution — the only place that picks `OpenAILLMClient` |

## 13.4 Dependency Inversion

The Agent receives its `LLMClient` through dependency injection:

```python
llm_client = OpenAILLMClient(...)   # composition boundary (outside)
agent = Agent(llm_client=llm_client)
answer = agent.run(question)
```

It must never do this:

```python
class Agent:
    def __init__(self):
        self.llm = OpenAILLMClient(...)   # ✗ couples the Agent to a provider
```

Why: the Agent should know only the abstraction `LLMClient`. Provider
selection (OpenAI today, others later) happens at the application /
composition boundary, which keeps the Agent testable with a fake,
independent of any SDK, and free to swap providers without changes.
`agent/agent.py` imports exclusively from `repo_intelligence.llm`; a
test scans the package sources to enforce that no provider SDK import
ever appears.

```text
Agent → LLMClient → OpenAILLMClient → OpenAI
  ▲        (neutral)   (composition root)   (SDK)
```

## 13.5 Request Flow

```text
Agent.run(question)
    ↓  validate question (string, non-empty)
    ↓  Message(role="system", SYSTEM_PROMPT)
    ↓  Message(role="user", question)
    ↓
LLMRequest(messages=[system, user], tools=…)   # + tool definitions since M8.2; client-default model
    ↓
LLMClient.complete(request)
    ↓
LLMResponse
    ↓  tool_calls?  ──► AgentError (unsupported, see 13.6)
    ↓  no text?     ──► AgentError
    ↓
answer = response.message.content  ──► returned to the caller
```

## 13.6 Tool Boundary

> **The M8.1/M8.2 Agent does not execute tools** (M8.3 adds one-shot
> execution through an injected dispatcher; M8.4 appends each
> `ToolResult` as a tool observation and loops back to the LLM — see
> [tool-definitions.md](tool-definitions.md) §9.4).

If the LLM unexpectedly returns tool calls, the Agent raises
`AgentError` — it does not execute them, silently ignore them, or
retry:

```text
M8.1 → M8.2

LLMResponse
    │
    ├── text response ──────► return answer
    │
    └── tool call ───────────► AgentError: tool call recognized, but
                               tool execution is not supported yet
```

No tool registry, no tool dispatcher, no agent loop exists in this
milestone — even though the repository already ships three independent
tools (`get_architecture_context`, `search_repository`, `read_file`).
They remain standalone application tools. (M8.2 now *describes* them
to the LLM via `ToolDefinition` and carries the structured request on
`AgentError.tool_calls` — still without executing anything; execution
is introduced in M8.3.)

## 13.7 Testing

All Agent tests run offline — no API key, no network, no model:

```python
class FakeLLMClient:                    # satisfies LLMClient (Protocol)
    def __init__(self, response):
        self.response = response
        self.requests = []

    def complete(self, request):
        self.requests.append(request)   # records what the Agent sent
        return self.response
```

The fake lets tests assert both directions of the contract: the exact
request the Agent constructed, and the answer it returned. Covered:

| Area | Test file |
| --- | --- |
| answer passthrough, request construction, DI, provider/tool import scans, tool-call boundary, question validation | `tests/agent/test_agent.py` |
| standalone command output + exit codes (client monkeypatched at the composition root) | `tests/agent/test_agent_cli.py` |
| M8.2 tool definitions: exposure, neutrality, schema correctness, recognition, no-execution spies | `tests/agent/test_tool_definitions.py` |
| M8.5 bounded execution: budget boundaries, blocked tools, termination, trace budget fields | `tests/agent/test_agent_bounded.py` |
| M8.6 adaptive investigation: observation-derived actions, bounded investigation, failure adaptation, no hidden reasoning | `tests/agent/test_agent_react.py` |
| execution trace: iterations, decisions, exact arguments, terminations, sinks, runtime relocation to `runtime/agent_runs/` (M8.7), no hidden reasoning | `tests/agent/test_agent_trace.py` |

```console
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agent -q
115 passed
```

## 13.8 Standalone Execution

Real LLM smoke test through the full dependency chain (never
`Agent → OpenAI` directly):

```console
$ cp .env.example .env        # add OPENAI_API_KEY
$ python -m repo_intelligence.agent
Question:
What is the purpose of this repository?

Answer:
...
```

Options: `--question "..."` (ask something else), `--model ID`,
`--max-tool-calls N` (M8.5 tool-call limit, default 8). Exit codes:
`0` success, `6` structured failure (`LLMError` / `AgentError`,
including the budget exhaustion guard — see
[bounded-agent.md](bounded-agent.md)). A failed
tool execution is not a CLI failure — it is observed by the loop like
any other `ToolResult`.

Each run writes its execution trace to
`runtime/agent_runs/<run_id>/trace.json` (plus `trace.md` — runtime
telemetry, separate from `artifacts/`) and prints `Trace: <path>`.
The live adaptive demonstration wires the same `Agent` through a real
`OpenAILLMClient`: `python agentrun_react_live_test.py` (M8.7; needs
`.env`/`OPENAI_API_KEY`, fails clearly without it) — contrast with the
offline `python agentrun_react_test.py` (M8.6, scripted model).

## 13.9 Future Evolution

Intended progression:

```text
M8.1 Minimal Agent               question → answer        ✓ implemented
       ↓
M8.2 Agent + Tool Definition     tools described to the model  ✓ implemented
       ↓
M8.3 Tool Execution Boundary     one ToolCall → one ToolResult ✓ implemented
       ↓
M8.4 Multi-Step Tool Loop        LLM → tool → LLM → …         ✓ implemented
       ↓
M8.5 Bounded Agent               limits, termination, control  ✓ implemented
       ↓
M8.6 Bounded ReAct Investigation adaptive, evidence-backed     ✓ implemented
       ↓
M8.7 Live ReAct Investigation   real model + runtime traces    ✓ implemented
       ↓
M8.8 Evidence Traceability      claims → gathered evidence     ✓ implemented
       ↓
M8.8 integration                structured claims validated     ✓ implemented ← current
                                 post-run, in the trace
```
