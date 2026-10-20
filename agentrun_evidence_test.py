"""M8.8 — evidence traceability demonstration (deterministic, offline).

Visibly demonstrates one flow — no LLM, no network, no second agent:

    Claim
      ↓
    Evidence reference (path + optional lines)
      ↓  was that evidence gathered during the run?
    Gathered evidence (successful ToolResults from the M8.7 trace)
      ↓
    SUPPORTED / UNSUPPORTED / UNKNOWN

The script loads the captured M8.7-style execution trace fixture
(tests/fixtures/m87_react_trace.json — real tool results over the
pinned checkout: get_architecture_context, a successful read_file of
.claude/agents/dag-agent.md, a FAILED read_file of
tool-registry/README.md → FILE_NOT_FOUND, and a successful read_file
of tool-registry/servers.yaml), derives the gathered-evidence index
from it, validates explicit structured claims, and prints every
result with its reason.

Structured claims are INPUTS (as in M8.8): nothing here extracts
claims from free-form answer text, decides whether a statement is
semantically true, or evaluates answer correctness. This script
checks *traceability* only:

    evidence-backed        → SUPPORTED
    refs never gathered    → UNSUPPORTED  (including failed tools)
    no/insufficient refs   → UNKNOWN      (never silently supported)

Checks (all required):

    1  SUPPORTED claim (successful read_file, matching reference)
    2  failed FILE_NOT_FOUND result is never positive evidence
    3  claim without evidence references → UNKNOWN
    4  reference to a wrong path → UNSUPPORTED
    5  reference outside the gathered line range → not SUPPORTED
    6  reference inside a larger gathered range → SUPPORTED
    7  multiple references, one valid → SUPPORTED (matched identified)
    8  every referenced attempt failed → UNSUPPORTED
    9  path normalization resolves; absolute/escape refs → UNKNOWN
    10 no hidden reasoning fields in any output payload

Usage (from the project root; fully offline):

    python agentrun_evidence_test.py

Exit codes: 0 = all checks passed, 6 = failure.

See also: agentrun_react_test.py (M8.6 adaptive mechanics),
agentrun_react_live_test.py (M8.7 real-model run),
agentrun_test.py (M8.4/M8.5 harness, retained). This script
validates evidence AFTER a run — it never calls a model and never
touches Agent.run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent.trace import (  # noqa: E402
    BANNED_REASONING_FIELDS,
)
from repo_intelligence.evidence import (  # noqa: E402
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    Claim,
    EvidenceRef,
    gather_evidence,
    validate_claim,
    validate_claims,
)

FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "m87_react_trace.json"
)
RULE = "-" * 72


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


def _claim(statement: str, *refs) -> Claim:
    return Claim(
        statement=statement,
        evidence_refs=tuple(
            ref if isinstance(ref, EvidenceRef) else EvidenceRef(*ref)
            for ref in refs
        ),
    )


def _build_claims() -> List[Claim]:
    return [
        _claim(
            "dag-agent is defined in .claude/agents/dag-agent.md "
            "(the DAG-generation agent).",
            (".claude/agents/dag-agent.md", 1, 20),
        ),
        _claim(
            "tool-registry/README.md documents the tool registry.",
            ("tool-registry/README.md", 1, 20),
        ),
        _claim(
            "The repository contains an agent called X.",  # no refs
        ),
        _claim(
            "The workflow is defined in never-gathered.md.",
            ("never-gathered.md", 1, 20),
        ),
        _claim(
            "dag-agent.md lines 40-50 define the DAG codegen contract.",
            (".claude/agents/dag-agent.md", 40, 50),
        ),
        _claim(
            "dag-agent.md lines 5-10 describe the agent's name.",
            (".claude/agents/dag-agent.md", 5, 10),
        ),
        _claim(
            "tool-registry/servers.yaml was inspected (an unrelated "
            "reference was not).",
            ("tool-registry/missing.yaml", 1, 10),
            ("tool-registry/servers.yaml", 1, 30),
        ),
        _claim(
            "tool-registry/README.md and tool-registry/CHANGELOG.md "
            "both document the registry.",
            ("tool-registry/README.md", 1, 20),
            ("tool-registry/CHANGELOG.md", 1, 10),
        ),
    ]


def main() -> int:
    started = time.time()
    print("=" * 72)
    print("M8.8 — Evidence Traceability (deterministic, offline)")
    print("=" * 72)
    print("fixture: {}".format(FIXTURE_PATH.relative_to(PROJECT_ROOT)))
    print()

    if not FIXTURE_PATH.is_file():
        print("fixture not found: {}".format(FIXTURE_PATH), file=sys.stderr)
        return 6
    payload: Dict[str, Any] = json.loads(
        FIXTURE_PATH.read_text(encoding="utf-8")
    )
    index = gather_evidence(payload)

    print("Evidence gathered during the run (successful tools only)")
    print(RULE)
    if not index.gathered:
        print("  (none)")
    for number, item in enumerate(index.gathered, start=1):
        provenance = ""
        if item.source and item.source.get("commit_sha"):
            provenance = "  [commit {}]".format(
                str(item.source["commit_sha"])[:12]
            )
        print("  {}. {}{}".format(number, item.render(), provenance))
    print("failed attempts (never positive evidence):")
    if not index.failed:
        print("  (none)")
    for attempt in index.failed:
        print(
            "  - {} {} -> {}".format(
                attempt.tool,
                attempt.path or "(path unresolved)",
                attempt.error_code or "tool failure",
            )
        )
    print()

    claims = _build_claims()
    results = validate_claims(claims, index)

    print("Claim validation")
    print(RULE)
    for number, result in enumerate(results, start=1):
        print("[{}] {}".format(number, result.claim.statement))
        if result.claim.evidence_refs:
            for ref in result.claim.evidence_refs:
                print("    ref: {}".format(ref.render()))
        else:
            print("    ref: (none)")
        print("    status: {}".format(result.status))
        print("    reason: {}".format(result.reason))
        if result.matched_evidence:
            for item in result.matched_evidence:
                print("    matched: {}".format(item.render()))
        print()
    print(RULE)

    by_number = {index_: r for index_, r in enumerate(results, start=1)}
    checks = [
        (
            "SUPPORTED claim (successful read_file reference)",
            by_number[1].status == STATUS_SUPPORTED,
            by_number[1].reason,
        ),
        (
            "failed FILE_NOT_FOUND result is never positive evidence",
            by_number[2].status == STATUS_UNSUPPORTED
            and "FILE_NOT_FOUND" in by_number[2].reason,
            by_number[2].reason,
        ),
        (
            "claim without evidence references -> UNKNOWN",
            by_number[3].status == STATUS_UNKNOWN,
            by_number[3].reason,
        ),
        (
            "reference to a wrong path -> UNSUPPORTED",
            by_number[4].status == STATUS_UNSUPPORTED,
            by_number[4].reason,
        ),
        (
            "reference outside gathered line range -> not SUPPORTED",
            by_number[5].status != STATUS_SUPPORTED,
            by_number[5].reason,
        ),
        (
            "reference inside larger gathered range -> SUPPORTED",
            by_number[6].status == STATUS_SUPPORTED,
            by_number[6].reason,
        ),
        (
            "multiple references (one invalid, one valid) -> SUPPORTED",
            by_number[7].status == STATUS_SUPPORTED
            and len(by_number[7].matched_evidence) == 1
            and by_number[7].matched_evidence[0].path
            == "tool-registry/servers.yaml",
            by_number[7].reason,
        ),
        (
            "failed-attempt reference never supported -> UNSUPPORTED",
            by_number[8].status == STATUS_UNSUPPORTED
            and "FILE_NOT_FOUND" in by_number[8].reason,
            by_number[8].reason,
        ),
        (
            "path normalization resolves; absolute/escape -> UNKNOWN",
            _checks_normalization(index),
            "normalized spellings match; absolute/escape -> UNKNOWN",
        ),
        (
            "no hidden reasoning fields in any output payload",
            _checks_no_hidden_reasoning(results, index),
            "BANNED_REASONING_FIELDS absent from every payload",
        ),
    ]

    print("Checks")
    failures = 0
    for number, (label, ok, detail) in enumerate(checks, start=1):
        verdict = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        dots = "." * max(3, 62 - len(label))
        print(
            "[{}/{}] {} {}  {}".format(
                number, len(checks), label, dots, verdict
            )
        )
        if not ok:
            print("         {}".format(detail))
    print(RULE)
    elapsed = time.time() - started
    if failures:
        print(
            "Result: FAIL - {}/{} checks failed ({:.2f}s)".format(
                failures, len(checks), elapsed
            )
        )
        return 6
    print(
        "Result: PASS - {}/{} checks ({:.2f}s)".format(
            len(checks), len(checks), elapsed
        )
    )
    return 0


def _checks_normalization(index) -> bool:
    for spelling in (
        ".claude/agents/dag-agent.md",
        "./.claude/agents/dag-agent.md",
        ".claude\\agents\\dag-agent.md",
    ):
        result = validate_claim(
            _claim("normalized", (spelling, 1, 20)), index
        )
        if result.status != STATUS_SUPPORTED:
            return False
    for outside in ("/etc/passwd", "../outside.md"):
        result = validate_claim(
            _claim("outside", (outside, 1, 20)), index
        )
        if result.status != STATUS_UNKNOWN:
            return False
    return True


def _checks_no_hidden_reasoning(results, index) -> bool:
    payloads = [result.to_dict() for result in results]
    payloads.append(index.to_dict())
    for payload in payloads:
        for banned in BANNED_REASONING_FIELDS:
            if banned in _walk_keys(payload):
                print(
                    "         banned field present: {}".format(banned),
                    file=sys.stderr,
                )
                return False
    return True


if __name__ == "__main__":
    sys.exit(main())
