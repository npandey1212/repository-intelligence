"""M13 — evidence eligibility: inspected vs. discovered (spec §18).

Focused tests for the smallest M13 behavior in the evidence layer:

 * structured schema helpers feed claims whose refs name files
 * file inspection (read_file-style) becomes evidence-eligible
 * search-only discovery does NOT automatically become inspected
   evidence (§4) — the two validation modes differ deterministically
 * uninspected / nonexistent / out-of-repository paths are rejected
   with CLAIM_EVIDENCE_INVALID (or insufficient-metadata UNKNOWN)
 * repository-relative path handling and duplicate references stay
   deterministic
 * evidence remains associated with the run that produced it (§5)
 * the recorded dictionaries carry the inspected flag (trace §19)
 * tool-role semantics (§3): architecture context and graph
   observations are never source inspection; discovery followed by
   an actual read becomes eligible

Frozen M8–M12 behavior stays intact: the default validation mode
(`require_inspected=False`) is covered by
``tests/evidence/test_evidence_traceability.py`` (still green).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from repo_intelligence.evidence import (
    CLAIM_EVIDENCE_INVALID,
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    Claim,
    EvidenceIndex,
    EvidenceRef,
    GatheredEvidence,
    gather_evidence,
    validate_claim,
    validate_claims,
)

SOURCE = {
    "repository": "sample-repo-alpha",
    "commit_sha": "b" * 40,
    "snapshot_id": "fixture",
    "repository_url": "https://example.invalid/sample.git",
}


# -- per-file helpers (no conftest in this repo) ---------------------------


def _result(
    success=True,
    tool="read_file",
    data=None,
    call_id="c1",
    error=None,
) -> Dict[str, Any]:
    return {
        "id": call_id,
        "name": tool,
        "arguments": {"path": "x.md"} if data is None else {},
        "result": {
            "tool_call_id": call_id,
            "tool": tool,
            "success": success,
            "data": {} if data is None else data,
            "source": dict(SOURCE),
            **({"error": error} if error else {}),
        },
    }


def _read_success(path, line_start, line_end, call_id="c_read"):
    return _result(
        tool="read_file",
        call_id=call_id,
        data={
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
            "content": "# inspected",
            "total_lines": 100,
        },
    )


def _search_success(query, matches, call_id="c_search"):
    return _result(
        tool="search_repository",
        call_id=call_id,
        data={"query": query, "results": matches},
    )


def _execution(*steps) -> Dict[str, Any]:
    tool_calls = []
    tool_results = []
    for step in steps:
        tool_calls.append(
            {
                "id": step["id"],
                "name": step["name"],
                "arguments": step["arguments"],
            }
        )
        tool_results.append(step["result"])
    return {
        "run_id": "run-m13",
        "iterations": [
            {
                "iteration": 1,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            }
        ],
    }


def _claim(statement, *refs) -> Claim:
    return Claim(
        statement=statement,
        evidence_refs=tuple(
            ref
            if isinstance(ref, EvidenceRef)
            else EvidenceRef(
                path=ref[0],
                line_start=ref[1] if len(ref) > 1 else None,
                line_end=ref[2] if len(ref) > 2 else None,
            )
            for ref in refs
        ),
    )


# ---------------------------------------------------------------------------
# 1 — inspection vs. discovery (gather_evidence provenance, §4)
# ---------------------------------------------------------------------------


def test_read_file_result_is_inspected_evidence():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )

    assert len(index.gathered) == 1
    assert index.gathered[0].inspected is True
    assert len(index.inspected) == 1
    assert index.inspected == index.gathered


def test_search_match_is_gathered_but_not_inspected():
    index = gather_evidence(
        _execution(
            _search_success(
                "dag-agent",
                [
                    {
                        "path": "docs/x.md",
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
    assert index.gathered[0].inspected is False
    # discovery is never eligible claim evidence (§4)
    assert index.inspected == ()


def test_inspected_flag_survives_serialization_round_trip():
    index = gather_evidence(
        _execution(
            _read_success("a.md", 1, 5, call_id="c1"),
            _search_success(
                "q", [{"path": "b.md", "line_start": 1, "line_end": 2}]
            ),
        )
    )
    payload = json.loads(json.dumps(index.to_dict()))
    flags = [item["inspected"] for item in payload["gathered"]]
    assert flags == [True, False]

    restored = EvidenceIndex.from_dict(payload)
    assert [item.inspected for item in restored.gathered] == [True, False]
    assert len(restored.inspected) == 1

    # legacy payloads (pre-M13) carry no flag and are never downgraded
    legacy = {"gathered": [{"path": "c.md", "tool": "search_repository"}]}
    assert EvidenceIndex.from_dict(legacy).gathered[0].inspected is True


# ---------------------------------------------------------------------------
# 2 — inspected-only validation accepts / rejects (§18 evidence validation)
# ---------------------------------------------------------------------------


def test_valid_inspected_path_accepted():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    result = validate_claim(
        _claim("docs/x.md defines X", ("docs/x.md", 1, 20)),
        index,
        require_inspected=True,
    )

    assert result.status == STATUS_SUPPORTED
    assert result.code is None
    assert result.matched_evidence[0].tool == "read_file"


def test_search_only_discovery_is_rejected_under_inspected_only():
    index = gather_evidence(
        _execution(
            _search_success(
                "dag-agent",
                [{"path": "docs/x.md", "line_start": 5, "line_end": 9}],
            )
        )
    )
    claim = _claim("docs/x.md mentions it", ("docs/x.md", 5, 9))

    # M13 mode: discovery never satisfies the reference
    rejected = validate_claim(claim, index, require_inspected=True)
    assert rejected.status == STATUS_UNSUPPORTED
    assert rejected.code == CLAIM_EVIDENCE_INVALID
    assert "never inspected" in rejected.reason
    assert "search results are discoveries" in rejected.reason

    # frozen default mode keeps the established M8.8 semantics
    established = validate_claim(claim, index)
    assert established.status == STATUS_SUPPORTED


def test_uninspected_path_rejected_with_claim_evidence_invalid():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    result = validate_claim(
        _claim("never-read.md says Y", ("never-read.md",)),
        index,
        require_inspected=True,
    )

    assert result.status == STATUS_UNSUPPORTED
    assert result.code == CLAIM_EVIDENCE_INVALID
    assert "never gathered" in result.reason
    # the structured failure is recorded, JSON-ready, deterministic
    payload = result.to_dict()
    assert payload["code"] == CLAIM_EVIDENCE_INVALID
    assert json.loads(json.dumps(payload))["code"] == (
        CLAIM_EVIDENCE_INVALID
    )


def test_nonexistent_path_is_rejected_not_supported():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    for outside in ("/etc/passwd", "C:\\secrets.txt", "../outside.md"):
        result = validate_claim(
            _claim("escape", (outside, 1, 5)),
            index,
            require_inspected=True,
        )
        assert result.status == STATUS_UNKNOWN, outside
        assert "insufficient" in result.reason
        assert result.code is None


def test_repository_relative_path_handling_under_inspected_only():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    for spelling in ("docs/x.md", "./docs/x.md", "docs//x.md", "docs\\x.md"):
        result = validate_claim(
            _claim("normalized", (spelling, 1, 20)),
            index,
            require_inspected=True,
        )
        assert result.status == STATUS_SUPPORTED, spelling


def test_inspected_but_not_covering_the_range_is_unsupported():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    result = validate_claim(
        _claim("out of range", ("docs/x.md", 40, 50)),
        index,
        require_inspected=True,
    )

    assert result.status == STATUS_UNSUPPORTED
    assert result.code == CLAIM_EVIDENCE_INVALID
    assert "no observed line range covers" in result.reason


def test_duplicate_evidence_paths_validate_deterministically():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    claim = _claim(
        "duplicated reference",
        ("docs/x.md", 1, 5),
        ("./docs/x.md", 1, 5),
        ("docs/x.md", None, None),
    )

    first = validate_claim(claim, index, require_inspected=True)
    second = validate_claim(claim, index, require_inspected=True)

    assert first.status == STATUS_SUPPORTED
    # deduplicated by (path, lines, tool, call id) — same every time
    assert len(first.matched_evidence) == 1
    assert first == second


def test_claim_without_evidence_paths_stays_unknown_flagged():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20))
    )
    result = validate_claim(
        _claim("no references"), index, require_inspected=True
    )

    assert result.status == STATUS_UNKNOWN
    assert "no evidence references" in result.reason
    assert result.code is None


def test_validate_claims_forwards_the_eligibility_switch():
    index = gather_evidence(
        _execution(
            _read_success("inspected.md", 1, 20),
            _search_success(
                "q", [{"path": "found.md", "line_start": 1, "line_end": 2}]
            ),
        )
    )
    claims = [
        _claim("inspected one", ("inspected.md",)),
        _claim("discovered only", ("found.md",)),
    ]

    results = validate_claims(claims, index, require_inspected=True)

    assert [item.status for item in results] == [
        STATUS_SUPPORTED,
        STATUS_UNSUPPORTED,
    ]
    assert results[1].code == CLAIM_EVIDENCE_INVALID


# ---------------------------------------------------------------------------
# 3 — evidence stays with the run that produced it (§5 trace provenance)
# ---------------------------------------------------------------------------


def test_evidence_remains_associated_with_the_correct_run():
    run_a = gather_evidence(
        _execution(_read_success("only-in-a.md", 1, 10))
    )
    run_b = gather_evidence(
        _execution(_read_success("only-in-b.md", 1, 10))
    )

    claim_on_a = _claim("A file", ("only-in-a.md",))
    claim_on_b = _claim("B file", ("only-in-b.md",))

    assert (
        validate_claim(claim_on_a, run_a, require_inspected=True).status
        == STATUS_SUPPORTED
    )
    assert (
        validate_claim(claim_on_b, run_b, require_inspected=True).status
        == STATUS_SUPPORTED
    )
    # run B's evidence can never validate a run-A claim, and vice versa
    cross_a = validate_claim(claim_on_b, run_a, require_inspected=True)
    cross_b = validate_claim(claim_on_a, run_b, require_inspected=True)
    assert cross_a.status == STATUS_UNSUPPORTED
    assert cross_b.status == STATUS_UNSUPPORTED
    assert cross_a.code == CLAIM_EVIDENCE_INVALID
    assert cross_b.code == CLAIM_EVIDENCE_INVALID


def test_matched_evidence_records_the_inspecting_tool_call():
    index = gather_evidence(
        _execution(_read_success("docs/x.md", 1, 20, call_id="c7"))
    )
    result = validate_claim(
        _claim("provenance", ("docs/x.md", 1, 20)),
        index,
        require_inspected=True,
    )

    matched = result.matched_evidence[0]
    assert matched.tool_call_id == "c7"
    assert matched.tool == "read_file"
    assert matched.inspected is True
    assert matched.to_dict()["inspected"] is True


def test_gathered_evidence_defaults_preserve_legacy_construction():
    legacy = GatheredEvidence(path="old.md", tool="read_file")
    assert legacy.inspected is True
    assert legacy.to_dict()["inspected"] is True


# ---------------------------------------------------------------------------
# tool-role semantics (§3/§13 — context and graph are not source
# inspection; only read_file establishes eligibility)
# ---------------------------------------------------------------------------


def test_architecture_context_is_not_source_inspection():
    """§13 Test 5: the run obtained the Architecture Context but never
    opened the claimed repository file — context is orientation, not
    inspected source evidence."""
    index = gather_evidence(
        _execution(
            _result(
                tool="get_architecture_context",
                call_id="c_ctx",
                data={
                    "context": "# Architecture Context\n"
                    "- see docs/agents.md for the agent roster\n"
                },
            )
        )
    )
    # the context result contributes nothing file-addressable
    assert index.gathered == ()
    assert index.inspected == ()

    result = validate_claim(
        _claim(
            "docs/agents.md defines the agents",
            ("docs/agents.md",),
        ),
        index,
        require_inspected=True,
    )
    assert result.status == STATUS_UNSUPPORTED
    assert result.code == CLAIM_EVIDENCE_INVALID
    assert "never gathered" in result.reason

    # control: the SAME path cited after an actual read is supported —
    # eligibility comes from inspection, not from the claim's wording
    read_index = gather_evidence(
        _execution(_read_success("docs/agents.md", 1, 10))
    )
    control = validate_claim(
        _claim(
            "docs/agents.md defines the agents",
            ("docs/agents.md",),
        ),
        read_index,
        require_inspected=True,
    )
    assert control.status == STATUS_SUPPORTED


def test_graph_observation_is_not_source_inspection():
    """§13 Test 6: a successful query_engineering_graph observation is
    structured knowledge — it never becomes source-file inspection,
    even when its evidence entries name repository files."""
    graph_step = _result(
        tool="query_engineering_graph",
        call_id="c_graph",
        data={
            "mode": "walk",
            "relationships": [
                {
                    "kind": "DECLARES",
                    "source": "agent:ontology-agent",
                    "target": "tool:mcp:glue-athena:get_table",
                    "evidence": [
                        {
                            "path": ".claude/agents/ontology-agent.md",
                            "line": 5,
                        }
                    ],
                }
            ],
        },
    )
    index = gather_evidence(_execution(graph_step))
    assert index.gathered == ()
    assert index.inspected == ()

    result = validate_claim(
        _claim(
            "ontology-agent declares the glue-athena tool",
            (".claude/agents/ontology-agent.md",),
        ),
        index,
        require_inspected=True,
    )
    assert result.status == STATUS_UNSUPPORTED
    assert result.code == CLAIM_EVIDENCE_INVALID
    assert "never gathered" in result.reason

    # control: reading that same file during the run supports the claim
    read_index = gather_evidence(
        _execution(
            _read_success(".claude/agents/ontology-agent.md", 1, 10)
        )
    )
    control = validate_claim(
        _claim(
            "ontology-agent declares the glue-athena tool",
            (".claude/agents/ontology-agent.md",),
        ),
        read_index,
        require_inspected=True,
    )
    assert control.status == STATUS_SUPPORTED


def test_search_then_read_makes_the_discovery_eligible():
    """Discovery → inspection: the search found the path (not evidence),
    the subsequent read makes it eligible — both in one trace."""
    index = gather_evidence(
        _execution(
            _search_success(
                "ontology",
                [
                    {
                        "path": ".claude/agents/ontology-agent.md",
                        "line_start": 1,
                        "line_end": 3,
                    }
                ],
            ),
            _read_success(".claude/agents/ontology-agent.md", 1, 30),
        )
    )
    assert len(index.inspected) == 1
    assert index.inspected[0].path == ".claude/agents/ontology-agent.md"

    result = validate_claim(
        _claim(
            "ontology-agent.md names the agent",
            (".claude/agents/ontology-agent.md", 1, 3),
        ),
        index,
        require_inspected=True,
    )
    assert result.status == STATUS_SUPPORTED
    assert result.matched_evidence[0].tool == "read_file"


def test_mixed_references_within_a_claim_follow_one_reference_rule():
    """§8 Case C: a claim citing [inspected, uninspected] keeps the
    established defined behavior — one resolving reference suffices
    for SUPPORTED, matched evidence records exactly what resolved,
    and the uninspected citation is simply absent from the match
    (never silently credited). The PARTIAL outcome for a result with
    a supported and an unsupported claim is the evaluator's
    aggregate (test_m13_claim_traceability.py)."""
    index = gather_evidence(
        _execution(
            _read_success("inspected.md", 1, 20),
            _search_success(
                "q", [{"path": "found.md", "line_start": 1, "line_end": 2}]
            ),
        )
    )

    result = validate_claim(
        _claim(
            "mixed citations",
            ("inspected.md", 1, 5),
            ("found.md", 1, 2),
        ),
        index,
        require_inspected=True,
    )

    assert result.status == STATUS_SUPPORTED
    assert len(result.matched_evidence) == 1
    assert result.matched_evidence[0].path == "inspected.md"
    assert result.matched_evidence[0].inspected is True
