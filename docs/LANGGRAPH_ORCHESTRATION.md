# LangGraph-owned sequential orchestration

## Purpose

`repo_intelligence.orchestration.langgraph_runner.build_langgraph_orchestration_graph`
creates a LangGraph that owns task selection, per-attempt dispatch, result commit,
retry routing, result assembly, and final validation. It does **not** call
`Orchestrator.execute(...)`.

The configured `Orchestrator` instance is supplied as a compatibility runtime for
the already-tested executor/configuration and the shared event, handoff, trace-window,
and budget-gate primitives. This keeps the graph from duplicating those policies
while moving the actual task loop into graph transitions.

## Install

LangGraph is a project dependency. In an environment with package-index access:

```bash
python -m pip install -e .
```

The repository's offline test environment may not have network access, so the
LangGraph runtime integration test is skipped when the package is unavailable.

## Use

```python
from repo_intelligence.orchestration.langgraph_runner import (
    build_langgraph_orchestration_graph,
)

# `orchestrator` is constructed with the existing configured executor,
# QueryRouter-backed investigator participant, analyst participant, trace sink,
# and M20 OrchestrationConfig.
graph = build_langgraph_orchestration_graph(orchestrator)
run_state = graph.invoke({
    "task": task_payload,
    "plan": validated_investigation_plan,
    "role_assignments": prevalidated_role_assignments,
})
run = run_state["run"]
```

Role assignments must cover the plan exactly and preserve the investigator-before-
analyst constraint. The graph validates the plan and the supplied role assignments
before task execution. Investigator requests reach the injected investigator
participant, which in the notebook is the actual `QueryRouter`; analyst requests
reach the configured visualization/analyst participant.

## Scope and current limitation

This first version is sequential. It does not implement `Send`, parallel workers,
checkpointing, resume/recovery, or replanning. Existing task contracts, measured
usage, per-attempt task limits, run-level budget gates, evidence merging, and M16
collaboration validation are reused. Since this environment could not install
LangGraph from the package index, the graph has not yet been runtime-compiled or
exercised here; run the graph-specific integration tests in an environment with
the dependency installed before treating this milestone as complete.
