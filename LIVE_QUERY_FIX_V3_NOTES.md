# Live Multi-Agent Query Fix v3

## Why the 2026-10-11 run failed

The attached runtime trace for P2 (`d6ff5c30cc3f40fd94a7beb5c1c1459a`) records two structured-output failures:

1. Retry attempt 1: `path does not run from source 'data-onboarding-agent' to target 'metadata-agent'`.
2. Retry attempt 2: `path_status FOUND requires a non-empty path`.

The trace also shows a guessed graph node ID, `concept:agent:data-onboarding-agent`, rejected with `NODE_NOT_FOUND`. The investigator then continued using documentation evidence and attempted to force a broad inventory question into a single source-to-target path. This is a contract/prompt mismatch, not the earlier global tool-call budget issue.

## Changes in v3

- Versioned the flow-investigator prompt as V2, while retaining the old V1 symbol as a backwards-compatible alias.
- Added an explicit open-ended-question rule: broad inventory/explanation questions must use `source=null`, `target=null`, `path_status=UNKNOWN`, and empty `path` / `alternative_paths`; individually evidenced relationships and findings remain reportable.
- Explicitly prohibits `FOUND` with an empty path and forbids inventing endpoints from unrelated discovered relationships.
- Requires graph component IDs to come from tool results; guessed IDs and prefixes are not permitted.
- Added a regression test for a broad interaction question with no single source/target path.

## Verification performed here

- `tests/agents/test_flow_dependency_investigator.py`: **48 passed, 2 deselected**.
- The two deselected tests use Pydantic v1 internals (`FieldInfo.type_`) and fail under the Pydantic 2.13 environment available here; they were not changed as part of this fix.
- Python syntax compilation passed for all changed Python files.
- The live OpenAI query was not executed in this environment. This patch addresses the exact failure pattern in the supplied trace but cannot honestly be called a verified live end-to-end success until the user runs it.
