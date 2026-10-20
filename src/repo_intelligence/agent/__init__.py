"""Minimal Agent — question → final answer over the LLM abstraction.

Public surface:

    Agent, AgentError, SYSTEM_PROMPT   (M8.1: the question/answer flow)
    AGENT_TOOL_DEFINITIONS             (M8.2: descriptions sent to the LLM)
    ToolDispatcher                     (M8.3: one execution per ToolCall)
    AgentExecutionConfig               (M8.5: bounded execution — the
                                        tool-call budget; the M8.4
                                        MAX_TOOL_CALL_STEPS guard was
                                        refactored into it)
    AgentRunResult                     (M8.8 integration: answer +
                                        structured claims + post-run
                                        deterministic evidence
                                        validation)
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    DEFAULT_COMPACTION_RECENT_MESSAGES,
    COMPACT_STATE_HEADER,
    estimate_context_characters,
    compact_working_context,
    restrict_working_context           (bounded normal execution: the
                                        application working-context
                                        ceiling — persistent
                                        compaction of the working
                                        conversation plus a
                                        deterministic per-request
                                        bound, both model-free)
    SOFT_TOOL_CALL_TARGET              (the SOFT investigation effort
                                        target: one execution-state
                                        notice at the crossing,
                                        recorded on the trace — never
                                        a limit, never a stop)
    EXECUTION_METADATA_FIELDS,
    execution_metadata_from_request,
    inject_execution_metadata          (deterministic metadata on the
                                        structured path: question/
                                        repository_scope/commit_scope
                                        read from the run's own
                                        request, filled only where the
                                        contract declares them and the
                                        payload left them missing)
    AgentTrace, AgentIteration,
    TraceSummary, render_markdown      (execution trace: observable run log)
    TraceSink, InMemoryTraceSink,
    JsonFileTraceSink, default_trace_root
                                        (trace persistence, separable;
                                        default root runtime/agent_runs)

Two run entry points exist: ``Agent.run_result`` (M8.x — direct
investigation) and ``Agent.run_planned`` (M10 — explicit
plan-and-execute: one planner LLM call → deterministic validation →
plan recorded on the trace before the first tool → the SAME loop
executes the steps sequentially under the SAME single budget). Both
return ``AgentRunResult`` and record one ``AgentTrace``.

Concrete provider clients are never part of this package's API —
composition roots (``agent/__main__.py`` or the future application)
choose them outside the Agent. Executable repository tools are not
part of this package either; the Agent only *describes* capabilities
and — when a dispatcher is injected — delegates each requested
execution to the application-owned boundary (the concrete
implementation lives in ``repo_intelligence.tool_dispatcher``) while
the multi-step loop itself stays inside ``Agent.run``. The M10
planning *contract* (plan model, validation rules, planner request
builder, Architecture Context loader) lives in
``repo_intelligence.planning`` — one-way dependency, agent → planning
— and is only *consumed* here; no planner prompt, validation rule, or
second loop exists in this package.

Every run automatically records an ``AgentTrace`` (observable events
only — no hidden chain of thought, including the configured budget
and any budget-blocked tool calls); persistence via ``TraceSink`` is
optional and never affects execution. A planned run additionally
records ``trace.plan`` (the validated plan + its per-step outcomes)
and ``trace.architecture_context`` (artifact provenance), and tags
execution iterations with ``plan_step_id``.
"""

from __future__ import annotations

from .agent import (
    COMPACT_STATE_HEADER,
    DEFAULT_COMPACTION_RECENT_MESSAGES,
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    SOFT_TOOL_CALL_TARGET,
    SYSTEM_PROMPT,
    Agent,
    AgentError,
    AgentExecutionConfig,
    AgentRunResult,
    compact_working_context,
    estimate_context_characters,
    restrict_working_context,
)
from .dispatch import ToolDispatcher
from .metadata import (
    EXECUTION_METADATA_FIELDS,
    execution_metadata_from_request,
    inject_execution_metadata,
)
from .request_budget import (
    RequestBudgetExceededError,
    RequestBudgetPolicy,
    enforce_request_budget,
    enforce_request_budget_or_raise,
)
from .tool_definitions import AGENT_TOOL_DEFINITIONS
from .trace import (
    STOP_REASONS,
    AgentIteration,
    AgentTrace,
    TraceSummary,
    render_markdown,
)
from .trace_sink import (
    InMemoryTraceSink,
    JsonFileTraceSink,
    TraceSink,
    default_trace_root,
)

__all__ = [
    "AGENT_TOOL_DEFINITIONS",
    "COMPACT_STATE_HEADER",
    "DEFAULT_COMPACTION_RECENT_MESSAGES",
    "DEFAULT_MAX_CONTEXT_CHARACTERS",
    "EXECUTION_METADATA_FIELDS",
    "Agent",
    "AgentError",
    "AgentExecutionConfig",
    "AgentIteration",
    "AgentRunResult",
    "AgentTrace",
    "InMemoryTraceSink",
    "JsonFileTraceSink",
    "RequestBudgetExceededError",
    "RequestBudgetPolicy",
    "STOP_REASONS",
    "SOFT_TOOL_CALL_TARGET",
    "SYSTEM_PROMPT",
    "ToolDispatcher",
    "TraceSink",
    "TraceSummary",
    "compact_working_context",
    "default_trace_root",
    "enforce_request_budget",
    "enforce_request_budget_or_raise",
    "estimate_context_characters",
    "execution_metadata_from_request",
    "inject_execution_metadata",
    "render_markdown",
    "restrict_working_context",
]
