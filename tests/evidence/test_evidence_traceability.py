"""M8.8 evidence-traceability tests (deterministic, offline).

Proves the validator answers exactly one question — "do these
structured claims point to evidence actually gathered during the
run?" — with SUPPORTED / UNSUPPORTED / UNKNOWN, no LLM anywhere:

 1  supported claim (successful read_file + matching reference)
 2  failed tool result is never positive evidence (FILE_NOT_FOUND)
 3  no evidence references → UNKNOWN (not false, not supported)
 4  wrong path → UNSUPPORTED
 5  wrong line range → not SUPPORTED
 6  containing line range → SUPPORTED
 7  multiple references: one valid → SUPPORTED, matched identified
 8  all references failed → UNSUPPORTED
 9  M8.7 trace fixture (real tool results over the pinned checkout)
10  no hidden reasoning fields anywhere in the data model
    (+ normalization, single-bound, search-evidence, AgentTrace-object,
     architecture-context, roundtrip extras)

Loop mechanics live in ``tests/agent/test_agent_loop.py``; the trace
itself in ``tests/agent/test_agent_trace.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.evidence import (
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    VALIDATION_STATUSES,
    Claim,
    ClaimValidation,
    EvidenceIndex,
    EvidenceRef,
    FailedEvidenceAttempt,
    GatheredEvidence,
    gather_evidence,
    validate_claim,
    validate_claims,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "m87_react_trace.json"
)

SOURCE = {
    "repository": "sample-repo-alpha",
    "commit_sha": "b" * 40,
    "snapshot_id": "fixture",
    "repository_url": "https://example.invalid/sample.git",
}


# ---------------------------------------------------------------------------
# Trace-shaped execution data (same shapes AgentTrace.to_dict() emits)
# ---------------------------------------------------------------------------


def _step(
    name: str,
    arguments: Dict[str, Any],
    success: bool,
    data: Any = None,
    error: Any = None,
    call_id: str = None,
):
    step_id = call_id or "call_{}".format(name)
    return {
        "id": step_id,
        "name": name,
        "arguments": arguments,
        "result": {
            "tool": name,
            "success": success,
            "data": data,
            "source": dict(SOURCE) if success else None,
            "error": error,
        },
    }


def _read_success(path, line_start, line_end, call_id=None):
    return _step(
        "read_file",
        {
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
        },
        success=True,
        data={
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
            "content": "# excerpt\n",
        },
        call_id=call_id,
    )


def _read_failure(path, code="FILE_NOT_FOUND", call_id=None):
    return _step(
        "read_file",
        {"path": path},
        success=False,
        error={"code": code, "message": "Requested file was not found."},
        call_id=call_id,
    )


def _search_success(query, results, call_id=None):
    return _step(
        "search_repository",
        {"query": query},
        success=True,
        data={
            "query": query,
            "results": results,
            "returned": len(results),
            "total_matches": len(results),
            "truncated": False,
        },
        call_id=call_id,
    )


def _arch_context(call_id=None):
    return _step(
        "get_architecture_context",
        {},
        success=True,
        data={"context": "# Architecture Context\n"},
        call_id=call_id,
    )


def _execution(*steps) -> Dict[str, Any]:
    """One trace-shaped mapping: each tool step in its own iteration
    (tool call + its ToolResult share an iteration, as the Agent does)."""
    iterations: List[Dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        result = dict(step["result"])
        result["tool_call_id"] = step["id"]
        iterations.append(
            {
                "iteration": index,
                "tool_calls": [
                    {
                        "id": step["id"],
                        "name": step["name"],
                        "arguments": step["arguments"],
                    }
                ],
                "tool_results": [result],
            }
        )
    return {"run_id": "fixture_run", "iterations": iterations}


def _claim(statement, *refs):
    return Claim(
        statement=statement,
        evidence_refs=tuple(
            ref if isinstance(ref, EvidenceRef) else EvidenceRef(*ref)
            for ref in refs
        ),
    )


# ---------------------------------------------------------------------------
# 1 — supported claim
# ---------------------------------------------------------------------------


def test_supported_claim_points_to_gathered_read_file_evidence():
    execution = _execution(_read_success("foo.md", 1, 20))
    index = gather_evidence(execution)
    claim = _claim(
        "foo.md defines the workflow", ("foo.md", 1, 20)
    )

    result = validate_claim(claim, index)

    assert result.status == STATUS_SUPPORTED
    assert len(result.matched_evidence) == 1
    matched = result.matched_evidence[0]
    assert matched.path == "foo.md"
    assert matched.line_start == 1 and matched.line_end == 20
    assert matched.tool == "read_file"
    assert "read_file foo.md:1-20" in result.reason
    # provenance chain: GatheredEvidence → ToolResult source
    assert matched.source == SOURCE


# ---------------------------------------------------------------------------
# 2 — failed evidence never counts as positive
# ---------------------------------------------------------------------------


def test_failed_tool_result_is_never_positive_evidence():
    execution = _execution(
        _read_failure("tool-registry/README.md", code="FILE_NOT_FOUND")
    )
    index = gather_evidence(execution)
    # the failed attempt is recorded only as a failure
    assert index.gathered == ()
    assert len(index.failed) == 1
    assert index.failed[0].path == "tool-registry/README.md"
    assert index.failed[0].error_code == "FILE_NOT_FOUND"

    claim = _claim(
        "tool-registry/README.md exists",
        ("tool-registry/README.md", 1, 20),
    )
    result = validate_claim(claim, index)

    assert result.status == STATUS_UNSUPPORTED
    assert result.matched_evidence == ()
    assert "FILE_NOT_FOUND" in result.reason


# ---------------------------------------------------------------------------
# 3 — missing evidence references → UNKNOWN
# ---------------------------------------------------------------------------


def test_claim_without_evidence_references_is_unknown():
    index = gather_evidence(
        _execution(_read_success("foo.md", 1, 20))
    )

    result = validate_claim(
        _claim("The repository contains an agent called X."), index
    )

    assert result.status == STATUS_UNKNOWN
    assert result.matched_evidence == ()
    assert "no evidence references" in result.reason
    # never silently converted to supported (or to "false")
    assert result.status != STATUS_SUPPORTED


# ---------------------------------------------------------------------------
# 4 — wrong path
# ---------------------------------------------------------------------------


def test_reference_to_wrong_path_is_unsupported():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 20)))

    result = validate_claim(
        _claim("bar.md defines the workflow", ("bar.md", 1, 20)), index
    )

    assert result.status == STATUS_UNSUPPORTED
    assert result.matched_evidence == ()
    assert "bar.md" in result.reason


# ---------------------------------------------------------------------------
# 5 — wrong line range
# ---------------------------------------------------------------------------


def test_reference_outside_gathered_line_range_is_not_supported():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 20)))

    result = validate_claim(
        _claim("foo.md:40-50 says X", ("foo.md", 40, 50)), index
    )

    assert result.status == STATUS_UNSUPPORTED
    assert result.status != STATUS_SUPPORTED


# ---------------------------------------------------------------------------
# 6 — containing line range
# ---------------------------------------------------------------------------


def test_reference_inside_gathered_line_range_is_supported():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 50)))

    result = validate_claim(
        _claim("foo.md:10-20 says X", ("foo.md", 10, 20)), index
    )

    assert result.status == STATUS_SUPPORTED
    assert len(result.matched_evidence) == 1


# ---------------------------------------------------------------------------
# 7 — multiple references, one valid
# ---------------------------------------------------------------------------


def test_multiple_references_one_valid_is_supported():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 20)))

    result = validate_claim(
        _claim(
            "foo.md says X (and bar.md was mentioned)",
            ("bar.md", 1, 20),
            ("foo.md", 1, 20),
        ),
        index,
    )

    assert result.status == STATUS_SUPPORTED
    assert len(result.matched_evidence) == 1
    assert result.matched_evidence[0].path == "foo.md"


# ---------------------------------------------------------------------------
# 8 — all evidence failed
# ---------------------------------------------------------------------------


def test_all_referenced_evidence_failed_is_unsupported():
    execution = _execution(
        _read_failure("a.md", code="FILE_NOT_FOUND"),
        _read_failure("b.md", code="READ_FAILED", call_id="call_b"),
    )
    index = gather_evidence(execution)

    result = validate_claim(
        _claim(
            "a.md and b.md both exist",
            ("a.md", 1, 10),
            ("b.md", 1, 10),
        ),
        index,
    )

    assert result.status == STATUS_UNSUPPORTED
    assert result.matched_evidence == ()
    assert "FILE_NOT_FOUND" in result.reason
    assert "READ_FAILED" in result.reason


# ---------------------------------------------------------------------------
# 9 — M8.7 trace fixture (real tool results, pinned checkout)
# ---------------------------------------------------------------------------


def _m87_index():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return payload, gather_evidence(payload)


def test_m87_fixture_gathers_real_evidence_and_failures():
    payload, index = _m87_index()

    assert payload["summary"]["termination_reason"] == "FINAL_ANSWER"
    paths = {item.path for item in index.gathered}
    assert ".claude/agents/dag-agent.md" in paths
    assert "tool-registry/servers.yaml" in paths
    # architecture context is not file-addressable → never gathered
    assert not any(
        item.path.endswith("architecture_context.md")
        for item in index.gathered
    )
    # the failed read_file is recorded as a failure, never as evidence
    assert not any(
        item.path == "tool-registry/README.md" for item in index.gathered
    )
    assert any(
        attempt.path == "tool-registry/README.md"
        and attempt.error_code == "FILE_NOT_FOUND"
        for attempt in index.failed
    )
    # provenance preserved (pinned commit of the M8.7 run)
    dag = next(
        item
        for item in index.gathered
        if item.path == ".claude/agents/dag-agent.md"
    )
    assert dag.source["commit_sha"] == (
        "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
    )


def test_m87_fixture_dag_agent_claim_supported_and_missing_file_not():
    _, index = _m87_index()

    supported = validate_claim(
        _claim(
            "dag-agent is defined in .claude/agents/dag-agent.md",
            (".claude/agents/dag-agent.md", 1, 20),
        ),
        index,
    )
    failed = validate_claim(
        _claim(
            "tool-registry/README.md exists",
            ("tool-registry/README.md", 1, 20),
        ),
        index,
    )

    assert supported.status == STATUS_SUPPORTED
    assert supported.matched_evidence[0].tool == "read_file"
    # the failed tool call must NOT be treated as positive evidence
    assert failed.status == STATUS_UNSUPPORTED
    assert "FILE_NOT_FOUND" in failed.reason


# ---------------------------------------------------------------------------
# 10 — no hidden reasoning anywhere in the data model
# ---------------------------------------------------------------------------


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


def test_evidence_model_contains_no_hidden_reasoning_fields():
    index = gather_evidence(
        _execution(_read_success("foo.md", 1, 20), _read_failure("b.md"))
    )
    claim = _claim("foo.md says X", ("foo.md", 1, 20))
    result = validate_claim(claim, index)

    payload = {
        "claim": claim.to_dict(),
        "validation": result.to_dict(),
        "index": index.to_dict(),
        "ref": EvidenceRef("foo.md", 1, 2).to_dict(),
        "gathered": GatheredEvidence(
            path="foo.md", tool="read_file"
        ).to_dict(),
        "failed": FailedEvidenceAttempt(
            path="b.md", tool="read_file", error_code="FILE_NOT_FOUND"
        ).to_dict(),
    }
    keys = list(_walk_keys(payload))
    assert keys, "payload must not be empty"
    for banned in BANNED_REASONING_FIELDS:
        assert banned not in keys, banned


# ---------------------------------------------------------------------------
# Extras — normalization, structure, search evidence, input forms
# ---------------------------------------------------------------------------


def test_path_normalization_and_repository_boundary():
    index = gather_evidence(_execution(_read_success("docs/x.md", 1, 5)))

    # equivalent representations resolve to the same gathered evidence
    for spelling in ("docs/x.md", "./docs/x.md", "docs//x.md", "docs\\x.md"):
        result = validate_claim(
            _claim("normalized", (spelling, 1, 5)), index
        )
        assert result.status == STATUS_SUPPORTED, spelling

    # absolute paths and escapes cannot bypass the repository boundary
    for outside in ("/etc/passwd", "C:\\secrets.txt", "../outside.md"):
        result = validate_claim(
            _claim("outside", (outside, 1, 5)), index
        )
        assert result.status == STATUS_UNKNOWN, outside
        assert "insufficient" in result.reason


def test_single_line_bound_is_insufficient_metadata():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 20)))

    half = EvidenceRef(path="foo.md", line_start=5, line_end=None)
    result = validate_claim(_claim("half-bounded", half), index)

    assert result.status == STATUS_UNKNOWN
    assert "both bounds" in result.reason


def test_search_match_evidence_supports_covered_references_only():
    index = gather_evidence(
        _execution(
            _search_success(
                "dag-agent",
                [
                    {
                        "path": "foo.md",
                        "line_start": 5,
                        "line_end": 9,
                        "excerpt": "name: dag-agent",
                    }
                ],
            )
        )
    )
    assert len(index.gathered) == 1
    assert index.gathered[0].tool == "search_repository"

    covered = validate_claim(
        _claim("foo.md:6-8 mentions it", ("foo.md", 6, 8)), index
    )
    outside = validate_claim(
        _claim("foo.md:20-30 mentions it", ("foo.md", 20, 30)), index
    )

    assert covered.status == STATUS_SUPPORTED
    assert outside.status == STATUS_UNSUPPORTED


def test_path_only_reference_matches_any_gathered_evidence_on_path():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 20)))

    result = validate_claim(
        _claim("foo.md was inspected", ("foo.md",)), index
    )

    assert result.status == STATUS_SUPPORTED


def test_architecture_context_is_not_file_addressable_evidence():
    index = gather_evidence(_execution(_arch_context()))

    assert index.gathered == ()
    result = validate_claim(
        _claim("context says X", ("architecture_context.md", 1, 5)),
        index,
    )
    assert result.status == STATUS_UNSUPPORTED


def test_gather_evidence_accepts_a_real_agent_trace_object():
    from repo_intelligence.agent import AgentIteration, AgentTrace

    trace = AgentTrace(
        run_id="real-object",
        started_at="2026-09-29T00:00:00.000Z",
        user_question="question",
    )
    step = _read_success("foo.md", 1, 20, call_id="call_real")
    iteration = AgentIteration(iteration=1, request={"message_count": 2})
    iteration.tool_calls = [
        {
            "id": step["id"],
            "name": step["name"],
            "arguments": step["arguments"],
        }
    ]
    result = dict(step["result"])
    result["tool_call_id"] = step["id"]
    iteration.tool_results = [result]
    trace.iterations.append(iteration)

    index = gather_evidence(trace)

    assert len(index.gathered) == 1
    assert index.gathered[0].path == "foo.md"
    assert validate_claim(
        _claim("foo.md says X", ("foo.md", 1, 20)), index
    ).status == STATUS_SUPPORTED


def test_validate_claims_preserves_order_and_vocabulary():
    index = gather_evidence(_execution(_read_success("foo.md", 1, 20)))
    claims = [
        _claim("supported", ("foo.md", 1, 20)),
        _claim("unsupported", ("bar.md", 1, 20)),
        _claim("unknown"),
    ]

    results = validate_claims(claims, index)

    assert [r.status for r in results] == [
        STATUS_SUPPORTED,
        STATUS_UNSUPPORTED,
        STATUS_UNKNOWN,
    ]
    assert [r.claim.statement for r in results] == [
        c.statement for c in claims
    ]


def test_claim_dict_roundtrip():
    claim = _claim(
        "dag-agent is defined in .claude/agents/dag-agent.md",
        (".claude/agents/dag-agent.md", 1, 20),
    )

    restored = Claim.from_dict(json.loads(json.dumps(claim.to_dict())))

    assert restored == claim


def test_validation_statuses_are_the_smallest_vocabulary():
    assert VALIDATION_STATUSES == frozenset(
        {"SUPPORTED", "UNSUPPORTED", "UNKNOWN"}
    )
    index = EvidenceIndex()
    # a result object is always one of the three — never "FALSE"
    result = ClaimValidation(
        claim=_claim("x"), status=STATUS_UNKNOWN, reason="no refs"
    )
    assert result.status in VALIDATION_STATUSES
    assert validate_claim(_claim("x"), index).status == STATUS_UNKNOWN
