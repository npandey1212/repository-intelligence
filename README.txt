M5.2 corrected contract patch

This patch corrects two issues in the previous patch:
1. source_investigation_ids belongs on ArchitectureFinding (the P4 output), not InvestigationReport.
2. InvestigationReport.evidence_refs is a deterministic aggregate of finding-level evidence refs. The model now unions exact finding refs into that aggregate before validation, so an LLM cannot fail merely because it omitted a redundant aggregate entry or used a different line range in the aggregate. It does not create new refs; it only copies refs already present on findings.

Apply these two files over the matching paths in the active repository:
- src/repo_intelligence/multiagent/contracts.py
- tests/multiagent/test_contracts.py

Then restart the notebook kernel and run from the beginning. Confirm the active import with:
from repo_intelligence.multiagent.contracts import ArchitectureFinding, InvestigationReport
print('P4 provenance field:', 'source_investigation_ids' in ArchitectureFinding.model_fields)
print('P4 provenance on wrong model:', 'source_investigation_ids' in InvestigationReport.model_fields)

Expected: True, False.

Validation performed here: py_compile passed for both Python files. Full pytest and live graph execution were not run in this patch-only environment.
