# M5.2 P4 provenance contract fix

## Root cause
The P4 deterministic validator reads `ArchitectureFinding.source_investigation_ids`,
and the Analyst prompt requests that field, but the `ArchitectureFinding` Pydantic
model did not declare it. Pydantic therefore returned an object without that
attribute, and deterministic validation raised `AttributeError` instead of
returning a validation result.

## Changes
- Add `source_investigation_ids: List[str]` to `ArchitectureFinding`.
- Use an empty-list default for compatibility with existing non-P4 producers
  such as architecture diagram conversion. The P4 deterministic validator still
  requires the ordered IDs to exactly match the three P1-P3 report IDs.
- Add contract tests for the compatibility default and round-trip serialization.

## Apply
Copy these two files into the matching paths in your repository:
- `src/repo_intelligence/multiagent/contracts.py`
- `tests/multiagent/test_contracts.py`

Restart the notebook kernel before rerunning.

## Verification
Python compilation passed for the changed files and the current M5.2 graph.py.
The live P4 workflow has not been run here. After applying, run the multi-agent
contract tests and the P4 notebook workflow.
