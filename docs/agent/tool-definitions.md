# M8.2 — Agent Tool Definitions

The Agent now *describes* its four read-only tools to the LLM.
(M8.3 adds the execution boundary: exactly one requested tool call is
executed by the application-owned dispatcher — see §9.4. M8.4 then
feeds the `ToolResult` back as a tool observation and calls the LLM
again until it returns a textual answer. M11 appended the fourth
definition, `query_engineering_graph`, without changing any of these
rules.)

```text
User Question
      ↓
    Agent
      ↓
  LLMClient  (+ ToolDefinition × 4)
      ↓
    LLM
      │
      └── receives definitions of:
            ├── get_architecture_context
            ├── search_repository
            ├── read_file
            └── query_engineering_graph
```

---

## 9.1 Purpose

M8.1 showed that an Agent uses an LLM to accomplish a task — but it
could only answer from what the prompt contained. The repository
already ships four read-only evidence tools; the model should at
least be *told* they exist.

This milestone teaches one thing: **tool definitions**. The Agent adds
provider-neutral `ToolDefinition` values to its `LLMRequest`, so the
model can decide *which* capability a question calls for — and, if it
chooses one, return a structured `ToolCall`. The Agent observes that
request and stops. Definitions only: no execution, no loop, no
dispatcher.

## 9.2 Tool Definition

A **tool definition** and an **executable tool** are different things:

```text
ToolDefinition                    actual executable tool
(name, description,               (function in tools/ that
 input_schema — a JSON             reads the repository and
 description for the LLM)          returns a ToolResult)
```

| Concern | Location | Role |
| --- | --- | --- |
| Tool implementation | `src/repo_intelligence/tools/*.py` | executes; returns `ToolResult` (source of truth for behavior) |
| LLM tool definition | `src/repo_intelligence/agent/tool_definitions.py` | describes the capability to the model |
| Execution contract | tool `_validate_arguments` + `docs/tools/*.md` | argument rules the schemas are pinned against |

Rules that keep a single source of truth:

- Definitions use the existing provider-neutral `ToolDefinition` from
  `repo_intelligence.llm` — no second definition abstraction, no
  provider tool types.
- The Agent package **never imports** the executable tools (enforced by
  a source scan) — it only carries descriptions.
- Schemas are *pinned by tests* to the actual tool contracts
  (`TOOL_NAME`, `_ALLOWED_ARGUMENTS`, `DEFAULT_MAX_RESULTS`,
  `MAX_RESULTS_LIMIT`, and behavioral acceptance via
  `_validate_arguments`), so descriptions cannot silently drift from
  behavior.

> The module is named `tool_definitions.py` (not `tools.py`) because a
> same-package `tools` submodule would collide with the existing
> agent source scan that forbids tool-package imports.

## 9.3 Available Tools

Exactly the four existing read-only tools — nothing else is exposed.
Together they form the evidence ladder, each with one role:

```text
get_architecture_context   high-level orientation
search_repository          candidate source evidence
read_file                  exact source evidence
query_engineering_graph    structured graph evidence (bounded queries,
                           incl. the M12 Agent → Tool → MCP →
                           Configuration knowledge slice)
```

**Repository location is application-owned.** None of the definitions
accepts a path to a checkout: `get_architecture_context` takes no
arguments, `search_repository` takes only search terms plus an optional
repository-relative filter, and `read_file` takes only a
repository-relative path with optional line bounds (unknown keys are
rejected by each tool). The application resolves both file tools
against the current run's pinned repository snapshot
(`<output>/<run>/repository/`), so the model chooses *what* to search
or read inside that checkout — never *which* checkout. The same rule
applies to `query_engineering_graph`: its arguments are only graph
query parameters (node id, direction, filters, depth) — the
EngineeringModel artifact location is application-resolved, never a
tool argument.

### `get_architecture_context`

> Retrieve the already-generated high-level architecture context for
> the repository.

```json
{
  "type": "object",
  "properties": {},
  "additionalProperties": false
}
```

No arguments (`{}`); it never regenerates or analyzes anything.

### `search_repository`

> Search the pinned repository snapshot for relevant evidence (semantic retrieval over the repository's prebuilt index; read-only, never builds an index).

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Free-text question or keyword query; ranked by semantic similarity against the indexed repository chunks."
    },
    "path": {
      "type": ["string", "null"],
      "description": "Optional repository-relative file or directory filter; null means no filter."
    },
    "max_results": {
      "type": "integer",
      "minimum": 1,
      "maximum": 50,
      "default": 5,
      "description": "Maximum matches to return (1-50)."
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

Every result carries provenance (`similarity_score`,
`repository_name`, `commit_sha`, `snapshot_id`, `chunk_id`, line
range, excerpt). A missing index is an actionable structured error —
the tool never builds one implicitly (see
`docs/m18-repository-retrieval.md`).

### `read_file`

> Retrieve exact text evidence from a file of the pinned repository snapshot.

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "minLength": 1,
      "description": "Non-empty repository-relative file path."
    },
    "line_start": {
      "type": "integer",
      "minimum": 1,
      "description": "First line (1-based, inclusive)."
    },
    "line_end": {
      "type": "integer",
      "minimum": 1,
      "description": "Last line (1-based, inclusive); must be >= line_start."
    }
  },
  "required": ["path"],
  "additionalProperties": false
}
```

### `query_engineering_graph`

> Query the repository's engineering graph (structured code/artifact
> relationships: CONTAINS, IMPORTS, DEPENDS_ON, TRIGGERS, and the M12
> knowledge slice DECLARES, EXPOSED_BY, CONFIGURED_BY — the forward
> Agent → Tool → MCP Server → Configuration path) with evidence.
> Call with {} for the node-kind summary (discover node
> kinds and ids), or provide a start_node_id to traverse its bounded
> neighborhood.

```json
{
  "type": "object",
  "properties": {
    "start_node_id": {
      "type": ["string", "null"],
      "minLength": 1,
      "description": "Node id to traverse from; omit or null for the node-kind summary."
    },
    "direction": {
      "type": "string",
      "enum": ["out", "in", "both"],
      "default": "out",
      "description": "Edge direction for the traversal."
    },
    "relationship_kinds": {
      "type": ["array", "null"],
      "items": {"type": "string", "enum": ["CONTAINS", "IMPORTS", "DEPENDS_ON", "TRIGGERS", "DECLARES", "EXPOSED_BY", "CONFIGURED_BY"]},
      "description": "Only follow/return these relationship kinds; null means all."
    },
    "node_kinds": {
      "type": ["array", "null"],
      "items": {"type": "string", "minLength": 1},
      "description": "Only visit nodes matching 'kind' or 'kind/subkind' (the start node is always included); null means all."
    },
    "max_depth": {
      "type": "integer",
      "minimum": 0,
      "maximum": 6,
      "default": 1,
      "description": "Traversal depth in hops (0 = start node only)."
    }
  },
  "additionalProperties": false
}
```

No `required` key: the empty object `{}` is the node-kind summary
(discovery mode). The relationship-kind enum and direction enum are
written as literals here (the agent package imports neither the tool
nor the engineering packages — see the source scan in §9.5) and are
pinned to the real `RELATIONSHIP_KINDS` / `QUERY_DIRECTIONS` constants
by tests.

## 9.4 Execution Boundary

This section is the milestone's most important rule:

```text
LLM
 ↓
ToolCall
 ↓
Agent
 ↓
Tool Dispatcher        ← application-owned (M8.3): resolves the name
 ↓                        against the explicit registry, validates,
 ↓                        executes exactly one registered tool
Application Tool
 ↓
ToolResult             ← appended to the conversation as a tool
                            observation (M8.4), then the LLM is
                            called again
```

If the model requests a tool, the Agent:

1. **recognizes** it — the structured `ToolCall` (id, name, arguments)
   is passed to the injected `ToolDispatcher`;
2. **executes exactly one** — the dispatcher resolves the name against
   the application-owned registry (unknown names fail safely), the
   registered tool applies its own argument contract, and the generic
   `ToolResult` is returned;
3. **loops** — the result is appended as a tool observation (same
   `tool_call_id`) and the LLM is called again with the full history;
   each tool step costs exactly one extra LLM invocation, bounded by
   the M8.5 tool-call limit `AgentExecutionConfig.max_tool_calls`
   (default 8):

```python
# M8.4/M8.5 core (see agent/agent.py): the assistant tool-call turn is
# preserved first, every ToolCall is executed in order behind the
# budget check, and each ToolResult becomes a tool observation before
# looping back.
messages.append(Message(role="assistant", tool_calls=list(response.tool_calls)))
for tool_call in response.tool_calls:
    if tool_calls_executed >= max_tool_calls:
        # M8.5: budget exhausted → this call is blocked, NOT executed
        trace.terminate("EXECUTION_LIMIT_REACHED")
        raise AgentError("execution budget exhausted: ...")
    result = self._tool_dispatcher.dispatch(tool_call)
    tool_calls_executed += 1
    messages.append(Message(
        role="tool",
        content=json.dumps(result.to_dict()),
        tool_call_id=getattr(tool_call, "id", None),
    ))
# loop continues: next LLM call
```

Without an injected dispatcher the M8.2 behavior remains: the tool
call is recognized and reported unexecuted (the `AgentError` above).
The boundary is explicit and observable in code, tests, and the CLI
output (`Tool steps: 1. search_repository -> success=True …`).

## 9.5 Why this separation matters

- **The LLM only decides/requests; the application executes.** The
  model proposes *what* to do; deterministic application code decides
  *whether and how* it happens.
- Safety: a model response reaches an executable tool only through the
  explicit application registry — the Agent package does not import
  the tools, nothing resolves dynamically, and only the four read-only
  tools are registered.
- Auditability: resolution and validation happen in the dispatcher
  before any execution; every execution returns an observable
  `ToolResult`, and calls are observable via `AgentError.tool_calls`
  when no dispatcher is injected.
- Testability: the boundary is provable offline with a fake client, a
  fake dispatcher, and spies on the tools.
- Swappability: definitions are provider-neutral, so models/providers
  change without touching application tool code.

## 9.6 Testing

Offline — fake `LLMClient`, no key, no network (`tests/agent/`):

| Requirement | Test |
| --- | --- |
| All four tools exposed, nothing else | `test_agent_request_exposes_exactly_the_four_tools`, `test_definition_list_contains_no_unrelated_tools` |
| Provider-neutral `ToolDefinition`, no provider types | `test_definitions_are_llm_abstraction_tool_definitions`, `test_agent_sources_import_no_provider_tool_types` |
| Schemas match real tool contracts | `test_*_schema_matches_tool_contract`, `test_*_schema_examples_validate`, `test_*_schema_agrees_with_tool_validation` (incl. `jsonschema` acceptance/rejection) |
| Graph-tool schema enums pinned to the real contracts | `test_relationship_kind_enum_is_pinned_to_the_tool_contract` |
| ToolCall recognized (structured) | `test_tool_call_is_recognized_with_structured_details` |
| Tools not executed without a dispatcher (spy) | `test_application_tools_are_not_executed` |
| One-shot execution / exactly one dispatch per step (M8.3) | `tests/agent/test_agent_execution.py`, `tests/test_tool_dispatcher.py` |
| Multi-step loop: observation → next LLM call, budget guard (M8.4/M8.5) | `tests/agent/test_agent_loop.py`, `tests/agent/test_agent_execution.py`, `tests/agent/test_agent_bounded.py` |
| Adaptive ReAct investigation: later action derived from earlier observation (M8.6) | `tests/agent/test_agent_react.py` |
| Graph tool described, dispatched, traced with plan correlation (M11) | `tests/agent/test_agent_graph_tool.py` |
| Graph-tool enums/description include the M12 slice kinds (pinned to the real contracts) | `tests/agent/test_tool_definitions.py` (enum pin), `tests/test_tooling.py` / `tests/test_m12_knowledge_slice.py` (M12 facts/model) |
| Simple answers unchanged | `test_plain_answer_behavior_remains_intact` |

```console
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/agent -q
191 passed
```

## 9.7 Standalone demonstration

```console
$ python -m repo_intelligence.agent --question "What components participate in the onboarding workflow?"
```

If the model answers directly in text (no tool request):

```text
Question:
What components participate in the onboarding workflow?

Answer:
...
```

If the model requests a tool, the M8.4 loop executes it (behind the
M8.5 budget), feeds the `ToolResult` back as an observation, and calls
the LLM again until a textual answer (budget:
`AgentExecutionConfig.max_tool_calls = 8` by default,
`--max-tool-calls` to change):

```text
Question:
What components participate in the onboarding workflow?

Tool steps:
1. search_repository -> success=True

Answer:
...
```

Exit `0` = the loop returned a textual answer (a failed tool is just
an observation, not a CLI failure); `6` = structured failure
(`LLMError`, `AgentError`, including the step-limit guard). If the
model chooses not to request a tool, the normal answer path is used —
the demonstration never forces tool use.
