# Investigator loop termination fix

## Root cause addressed

The LangGraph `create_agent` Investigator has two different loop controls:

- `ReActSafetyConfig` middleware limits model iterations and repository-tool calls.
- LangGraph's `recursion_limit` counts internal graph transitions (model/tool nodes), not just tool calls.

With `max_tool_calls=12` and `AGENT_RECURSION_LIMIT=20`, the framework limit can interrupt the run after roughly ten model/tool cycles, before the middleware gets a chance to execute its finalization turn. Also, `ReActLoopGuard.record_tool_call()` previously marked the run terminal immediately when the last allowed tool call was recorded, which could prevent a final structured report from being synthesized.

## Changes

1. `react_safety.py`: recording the final allowed tool call no longer marks the whole run terminal immediately. A subsequent tool attempt is still denied by `allow_tool_call()`; the middleware can first allow one final model turn with investigation tools removed.
2. `langgraph_poc/middleware.py`: retain the bounded finalization behavior and remove only configured investigation tools from the final model request, leaving the structured-output path available.
3. `langgraph_poc/graph.py`: pass the exact investigation tool names to the middleware and set `AGENT_RECURSION_LIMIT=32`, above the 12-tool-call finalization path. The outer graph limit remains separate.
4. `tests/test_react_safety.py`: add a regression test that the last allowed tool call does not preempt finalization.

No request-token budget, observation-size ceiling, role assignment, or multi-agent architecture was broadened as part of this fix.

## Validation performed

Command:

```bash
PYTHONPATH=src pytest -q tests/test_react_safety.py tests/test_langgraph_poc.py tests/orchestration/test_langgraph_runner.py tests/multiagent/test_validation.py
```

Result: **40 passed, 2 skipped**. The `create_agent` integration module is skipped in this environment because its optional LangChain/LangGraph dependencies are unavailable, so a live provider-backed Investigator run has not been verified here.


## M5.1 — sequential Investigator steps

- `run_poc(..., multi_step=True)` dispatches P1, P2, and P3 sequentially; P4 remains Analyst metadata only.
- Each Investigator prompt includes its own validated plan-step goal.
- The final result includes `reports`, `completed_step_ids`, and `step_statuses`; `report` remains as a backward-compatible alias for the most recent successful report.
- If a later step fails after earlier steps completed, status is `PARTIAL` and earlier reports are preserved. A failure before any completed step remains `FAILED`.
- The command-line demo runs P1-P3 by default. Pass `--single-step` to run P1 only.
- The execution middleware guard resets per planned step; audit records remain available across the whole run. This prevents one step's loop budget from being consumed by another step.
