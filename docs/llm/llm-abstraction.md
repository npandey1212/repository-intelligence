# LLM Abstraction Layer — `LLMClient`

Provider-neutral seam between the application (and, next milestone, the
Agent) and concrete LLM providers. Milestone **M7** of the repository
intelligence project.

```text
Application / Agent (future)
        ↓  LLMRequest
   LLMClient                 ← llm/client.py (provider-neutral)
        ↓
   OpenAILLMClient           ← llm/openai_client.py (only provider module)
        ↓
   provider API
        ↓  provider response / exception
   LLMResponse / LLMError    ← normalized back to the application
```

---

## 14.1 Purpose

The deterministic pipeline (M1–M6) and the tool layer
(`get_architecture_context`, `search_repository`, `read_file`) produce
evidence. Answering questions *with* that evidence requires a model —
but the project must not couple its application code to any one model
vendor or SDK.

The LLM abstraction layer solves exactly that seam:

- **One contract** (`LLMClient.complete(LLMRequest) → LLMResponse`) that
  every piece of application code depends on.
- **One place** where provider details live (request/response wire
  formats, credentials, SDK exceptions).
- **Provider-neutral types** (`Message`, `ToolDefinition`, `ToolCall`,
  `Usage`, `LLMRequest`, `LLMResponse`) shared by both sides.

This layer performs **model inference only**. It does not execute
tools, does not loop, does not hold agent state, and does not rank or
route — those are Agent-layer concerns (next milestone).

## 14.2 Architecture

Package layout:

| Module | Role |
| --- | --- |
| `llm/models.py` | Frozen dataclasses: `Message`, `ToolDefinition`, `ToolCall`, `Usage`, `LLMRequest`, `LLMResponse` |
| `llm/client.py` | `LLMClient` protocol (`typing_extensions.Protocol`, runtime-checked) and `LLMError` |
| `llm/config.py` | Dependency-free `.env` loader (`load_env`, `get_openai_api_key`) |
| `llm/openai_client.py` | The **only** module that imports a provider SDK; translates both directions |
| `llm/__init__.py` | Re-exports the neutral API only — never a concrete client |
| `llm/__main__.py` | Standalone demonstration CLI (composition root) |

`LLMClient` is a structural `Protocol`, so any object with a compatible
`complete` method satisfies it — fakes, mocks, and future
implementations need no base class and no registration
(`isinstance(fake, LLMClient)` works at runtime).

Design rules:

- Frozen dataclasses everywhere — requests and responses are values.
- `LLMRequest`/`LLMResponse` carry optional `model`, `temperature`,
  `finish_reason`, `usage`; unknown information is `None`, not a guess.
- `Response.tool_calls` is a convenience view of `message.tool_calls`
  so the Agent can branch with a simple truthiness check.

## 14.3 Dependency Rule

```text
Application / Agent  →  llm.client, llm.models   (provider-neutral)
Composition root     →  llm.openai_client         (chooses the provider)
Tests                →  fakes injected via provider_client=
```

- No module outside `llm/openai_client.py` may import a provider SDK
  (`openai`, …). Enforced by a contract test that scans every file
  under `src/repo_intelligence/`.
- `llm/__init__.py` re-exports **only** the neutral API; the concrete
  `OpenAILLMClient` is imported explicitly where a provider is
  actually chosen.
- The provider SDK is imported **lazily** inside `OpenAILLMClient`'s
  constructor, so the whole project (including the test suite) works
  without the SDK installed; absence surfaces as `LLMError`.
- No provider-specific field names appear in `client.py` or
  `models.py` — the SDK's wire format exists only in
  `openai_client.py`.

## 14.4 Request / Response Contract

```python
from repo_intelligence.llm import (
    LLMClient, LLMError, LLMRequest, LLMResponse,
    Message, ToolCall, ToolDefinition, Usage,
)

request = LLMRequest(
    messages=[Message(role="system", content="..."),
              Message(role="user", content="...")],
    tools=[ToolDefinition(name="search_repository", ...)],
    model=None,        # None → client default
    temperature=None,  # None → omitted from the payload
)
response: LLMResponse = client.complete(request)
```

| Type | Fields | Notes |
| --- | --- | --- |
| `Message` | `role`, `content`, `tool_calls`, `tool_call_id` | roles: `system` / `user` / `assistant` / `tool`; others rejected |
| `ToolDefinition` | `name`, `description`, `input_schema` | `input_schema` is a generic JSON-Schema dict |
| `ToolCall` | `id`, `name`, `arguments` | `arguments` is a **dict** (parsed JSON), never a raw string |
| `Usage` | `input_tokens`, `output_tokens`, `total_tokens` | every field optional |
| `LLMRequest` | `messages`, `tools`, `model`, `temperature` | provider-neutral input |
| `LLMResponse` | `message`, `finish_reason`, `usage` | `finish_reason` lowercased pass-through; `None` when absent; `.tool_calls` property |

`complete()` either returns an `LLMResponse` or raises `LLMError` —
provider exceptions, malformed responses, and missing configuration are
all translated at the provider boundary. There is exactly one error
type on the application side.

## 14.5 Tool Calling Boundary

The layer **represents** tool calls; it never executes them.

- Request side: `tools=[...]` tells the model which tools exist
  (translated to the provider's function-calling format).
- Response side: a model-requested tool call arrives as
  `response.tool_calls → [ToolCall(id, name, arguments-dict)]`,
  with `finish_reason == "tool_calls"`.
- Round-trip: an `assistant` message carrying `tool_calls` and a `tool`
  message carrying `tool_call_id` + result translate cleanly back into
  the next request — the Agent (next milestone) owns that loop.

Executing a requested tool is the Agent's job, not this layer's. The
standalone CLI demonstrates the boundary by *displaying* a requested
call and never running it.

## 14.6 Configuration

- Credential: `OPENAI_API_KEY`, read from the environment by
  `OpenAILLMClient` (explicit `api_key=` argument also accepted).
- `.env` support is hand-rolled in `llm/config.py` — `load_env()`
  parses `KEY=VALUE` lines (comments, quoted values, never overriding
  existing environment variables). No third-party configuration
  dependency.
- Loading is **explicit at the composition root** (`load_env()` in
  `__main__.py` or application startup); constructing a client never
  reads files implicitly.
- `.env.example` documents the one required variable; `.gitignore`
  excludes `.env` and `.env.*` so credentials cannot be committed.
- Missing key → `LLMError` at construction (structured, actionable).

## 14.7 Standalone Test

With `OPENAI_API_KEY` in `.env` (copy `.env.example`):

```console
$ python -m repo_intelligence.llm
LLM Response
------------
Finish reason: stop

Message:
Hello! How can I help you today?

Usage:
Input tokens: ...
Output tokens: ...
Total tokens: ...
```

Options:

| Flag | Meaning |
| --- | --- |
| `--prompt TEXT` | user prompt (default: a one-sentence hello) |
| `--model ID` | override the default model id |
| `--temperature N` | sampling temperature |
| `--tools` | include the sample `search_repository` tool definition so the model may **request** it (displayed, never executed) |

Exit codes: `0` success, `6` structured failure (`LLMError` — e.g. no
API key — printed to stderr). The command exercises configuration →
composition root → neutral request → `complete()` → normalized response
with **no Agent and no tool execution** involved.

## 14.8 Unit Testing

Tests never call a real API:

- **Models** (`tests/llm/test_models.py`) — construction, role
  validation, field contracts, tool-call/usage representation,
  read-only values.
- **Contract** (`tests/llm/test_client_contract.py`) — a fake client
  satisfies `isinstance(x, LLMClient)`; the public package surface is
  neutral; a source scan proves no provider SDK import exists outside
  `llm/openai_client.py`; `.env` loading rules.
- **Adapter** (`tests/llm/test_openai_client.py`) — a fake provider
  client (`provider_client=`) exercises every translation: payload
  building, tool-definition and tool-history messages, response
  normalization (content, tool calls, finish reason, usage), and every
  failure path (provider exception, no choices, invalid tool-argument
  JSON, non-object arguments, missing key, missing SDK) — all as
  `LLMError`.
- **CLI** (`tests/llm/test_llm_cli.py`) — the composition root is
  monkeypatched; output format and exit codes (`0` / `6`) are asserted
  via `capsys`.

Run:

```console
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/llm -q
50 passed, 1 skipped
```

The single skip is an **opt-in real-API smoke test**
(`RUN_OPENAI_SMOKE=1` + API key); the default suite is fully offline.

## 14.9 Future Integration

The seam is designed so the Agent milestone plugs in without touching
this layer:

```text
Agent (future)
    │  builds LLMRequest from conversation history + available tools
    ▼
LLMClient ── complete() ──► LLMResponse
    │                              │
    │  response.tool_calls == []   │  response.tool_calls present
    ▼                              ▼
 return text              Agent executes the named tool (ToolResult)
                                  │
                                  ▼
                    append tool message, call complete() again
```

- The Agent depends only on the neutral API in `llm/__init__.py`.
- Adding another provider means adding one module beside
  `openai_client.py` and choosing it in the composition root — no
  application code changes (`OpenAILLMClient` is referenced only by
  `llm/__main__.py` and tests).
- Future work stays out of this layer by design: tool execution,
  multi-step loops, prompt management, retries/routing, tracing, and
  embedding providers all belong to the Agent / platform milestones —
  the current client performs a single model inference per call.
