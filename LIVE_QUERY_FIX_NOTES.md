# Repository Query Live-Run Fix

## Root cause from `runtime_agent_run2.zip`

The run completed P1-P4 but stopped before P5 because the orchestration budget was exhausted.

- Each investigator used 17 tool calls after crossing the 16-call soft target and performing one targeted follow-up.
- Four investigator steps therefore consumed 68 calls.
- The whole-run orchestration limit was 64 calls, so P5 and validation never ran.
- More importantly, `run_repository_query()` used `fixture_bindings(root)` for live mode too. Those bindings return canned empty results for `get_architecture_context` and `query_engineering_graph`. The traces show repeated empty graph results, which explains wasted calls and weak relationship evidence.

## Changes in this patch

1. **Real live tool bindings**
   - Added `live_bindings()` for actual context and engineering-graph tools.
   - Live mode now uses those bindings; deterministic offline mode keeps `fixture_bindings()`.
   - `read_file` and `search_repository` remain constrained to the resolved repository root.
   - Context and graph calls are pinned to the snapshot ID derived from the repository's M1 artifact layout.
   - If a custom repository has no matching snapshot artifacts, the tools return an explicit unavailable result rather than borrowing another repository's graph/context.

2. **Explicit snapshot selection**
   - `get_architecture_context()` and `query_engineering_graph_tool()` accept an application-supplied `snapshot_id`.
   - The graph tool can select a pinned run even when multiple valid artifact runs exist.

3. **Budget alignment**
   - Kept the hard whole-run ceiling at 64 calls.
   - Changed the M17 composition's soft investigation target from 16 to 8. With the existing one-targeted-call policy, a healthy investigator normally closes after at most 9 calls.
   - This leaves room for five investigator steps, one 9-call retry, and validation (56 calls), within the 64-call hard ceiling.

4. **More focused plans**
   - The planner is instructed to prefer 3-4 distinct, complementary steps and avoid overlapping sub-goals.

## Verification performed here

- Query/planner/orchestration focused tests: **109 passed, 2 deselected**.
  - The two deselected tests require FAISS, which is not installed in this test environment.
- Graph/context tool tests: **54 passed**.
- Direct pinned-artifact smoke test: `get_architecture_context` returned real context, and `query_engineering_graph` returned the real graph summary with 14 node-kind groups.
- Syntax compilation passed for changed Python files.

The live OpenAI query itself was not run here. The patch addresses the observed budget exhaustion and the confirmed empty-tool-binding defect, but a successful and semantically complete live result must still be confirmed in the configured environment.


## Follow-up fix from `runtime-multiagent-run.zip` (2026-10-11)

The next live run reached P4 (Architecture Visualization) but failed typed output validation because an edge cited `.claude/commands/onboard-workflow.md:1-20` while the top-level `evidence_refs` aggregate contained only `:2-2` for that file. The element-level citation was present; the redundant aggregate omitted the exact line range.

Changes in this follow-up:

- `DiagramSpec` now deterministically builds the top-level `evidence_refs` aggregate as the deduplicated union of all node and edge evidence references before typed validation. It preserves the exact path/line bounds already supplied by each element and does not create new evidence.
- The visualization prompt now explicitly says the aggregate must include every node/edge reference exactly, with no omissions.
- Added regression coverage for both node and edge references omitted from the aggregate, including the same line-range mismatch shape as the observed failure.

Verification: `tests/agents/test_architecture_visualization.py` and `tests/agents/test_architecture_visualization_m17.py`: **63 passed** in this environment.

The live OpenAI query was not rerun here. This specifically fixes the observed P4 schema failure; it does not claim that later semantic/scope validation will necessarily pass on the same model-generated diagram.
