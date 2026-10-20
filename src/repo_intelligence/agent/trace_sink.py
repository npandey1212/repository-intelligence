"""Trace sinks — where an ``AgentTrace`` is persisted (MVP).

The Agent builds the trace; a sink decides where it goes. This is the
whole observability output stack for now — local files, nothing else:

    Agent ──► AgentTrace ──► TraceSink ──► local runtime file

Two implementations:

    InMemoryTraceSink   keeps traces in a list (tests / inspection)
    JsonFileTraceSink   ``<root>/<run_id>/trace.json`` plus a
                        ``trace.md`` projection — a rendering of the
                        same JSON, never an independent source of truth

Storage location (deliberate conceptual separation):

    runtime/agent_runs/<run_id>/     Agent execution telemetry
                                      (this module's default root)
    artifacts/<run_id>/...           durable repository-intelligence
                                      outputs (the artifacts module)

Traces are operational telemetry — LLM calls, tool calls, timings,
iterations, termination — so they live under ``runtime/``, NOT under
``artifacts/``. The default root is
``<project-root>/runtime/agent_runs`` (``default_trace_root()``);
callers may inject any other root.

Writing is best-effort: ``Agent.run`` catches sink failures so
observability can never corrupt an Agent result (the failure is
surfaced as ``Agent.last_trace_error`` / ``AgentTrace.sink_error``).

Explicitly out of scope: OpenTelemetry, external tracing services,
databases, message queues.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from typing_extensions import Protocol, runtime_checkable

from ..artifacts import PROJECT_ROOT
from .trace import AgentTrace, render_markdown

__all__ = [
    "InMemoryTraceSink",
    "JsonFileTraceSink",
    "TraceSink",
    "default_trace_root",
]

RUNTIME_DIR_NAME = "runtime"
AGENT_RUNS_DIR_NAME = "agent_runs"


def default_trace_root() -> Path:
    """Default trace root: ``<project-root>/runtime/agent_runs``.

    Absolute and independent of the process current working directory,
    so every run lands in the same runtime telemetry tree regardless
    of where the command was started. Each trace is written to
    ``<root>/<run_id>/trace.json`` (+ ``trace.md``).
    """
    return PROJECT_ROOT / RUNTIME_DIR_NAME / AGENT_RUNS_DIR_NAME


@runtime_checkable
class TraceSink(Protocol):
    """Receives every finalized trace — one ``write`` call per run."""

    def write(self, trace: AgentTrace) -> None:
        ...


class InMemoryTraceSink:
    """Test/inspection sink: keeps traces in insertion order."""

    def __init__(self) -> None:
        self.traces: List[AgentTrace] = []

    def write(self, trace: AgentTrace) -> None:
        self.traces.append(trace)


class JsonFileTraceSink:
    """File sink: JSON is canonical, Markdown is a rendered projection.

    Layout (runtime telemetry — deliberately outside ``artifacts/``)::

        <root>/<run_id>/trace.json
        <root>/<run_id>/trace.md

    ``root`` defaults to ``default_trace_root()``
    (``<project-root>/runtime/agent_runs``).
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        markdown: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else default_trace_root()
        self.markdown = markdown

    def trace_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def json_path(self, run_id: str) -> Path:
        return self.trace_dir(run_id) / "trace.json"

    def markdown_path(self, run_id: str) -> Path:
        return self.trace_dir(run_id) / "trace.md"

    def write(self, trace: AgentTrace) -> None:
        directory = self.trace_dir(trace.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.json_path(trace.run_id).write_text(
            trace.to_json() + "\n", encoding="utf-8"
        )
        if self.markdown:
            self.markdown_path(trace.run_id).write_text(
                render_markdown(trace), encoding="utf-8"
            )
