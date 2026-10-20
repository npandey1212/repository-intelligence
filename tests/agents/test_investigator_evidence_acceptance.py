"""Inspected-evidence acceptance at the orchestrated run_result
boundary (defect: search results cited as evidence, never read).

The M13 authority — ``gather_evidence`` + ``validate_claims`` with
``require_inspected`` — must strip any finding whose evidence the run
never actually read (a search hit is a discovery, never an
inspection), record it as an actionable uncertainty, force the
report's status to ``incomplete``, and rebuild the aggregate from
kept-finding refs plus individually-inspected originals.

Covered by the 9 required tests:
  1. unread-only finding → stripped + uncertainty + incomplete
  2. inspected-and-covering finding → kept whole
  3. empty-refs finding → kept untouched (honest negative)
  4. mixed: some kept, some stripped
  5. all inspected → report returned unchanged (same object)
  6. ArchitectureInvestigator.run_result enforces
  7. FlowDependencyInvestigator.run_result enforces
  8. aggregate rebuilt from kept refs + individually-inspected originals
  9. end-to-end: search-only run (no read_file) → everything stripped
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from architecture_doubles import (  # noqa: E402
    MCP_COLLECTOR_PATH,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    RESEARCHER_PATH,
    build_investigator,
    fixture_bindings,
    mcp_report_payload,
    read_then_report,
    report_payload,
    structured,
    tool_calls,
    call,
)
from flow_doubles import (  # noqa: E402
    COLLECTOR_PATH,
    MCP_CONFIG_PATH as FLOW_MCP_CONFIG,
    FLOW_COMMIT,
    FLOW_QUESTION,
    FLOW_REPOSITORY,
    mcp_payload,
    ref,
)
from viz_doubles import (  # noqa: E402
    FETCH_RECORD_PATH,
    read_paths,
    build_flow,
)

from repo_intelligence.agents.evidence_acceptance import (  # noqa: E402
    enforce_inspected_evidence,
)
from repo_intelligence.multiagent.contracts import (  # noqa: E402
    InvestigationFinding,
    InvestigationReport,
    ReportEvidenceRef,
)

Q = "How do the agents, tools, and MCP servers interact?"
REPO = "example-repository"
COMMIT = "c" * 40


def _ref(path, start=None, end=None):
    return ReportEvidenceRef(
        path=path, line_start=start, line_end=end
    )


def _finding(statement, refs, support="SUPPORTED", uncertainty=None):
    return InvestigationFinding(
        statement=statement,
        evidence_refs=list(refs),
        support=support,
        uncertainty=uncertainty,
    )


def _report(findings, evidence_refs=None, uncertainties=None,
            status="complete"):
    return InvestigationReport(
        investigation_id="inv-test",
        question=Q,
        repository_scope=REPO,
        commit_scope=COMMIT,
        findings=findings,
        evidence_refs=list(evidence_refs or []),
        open_questions=[],
        uncertainties=list(uncertainties or []),
        investigated_sources=[],
        status=status,
    )


def _fake_trace(read_paths_list):
    """A minimal trace shape that gather_evidence understands: one
    iteration whose successful read_file results set inspected=True."""
    tool_calls_list = []
    tool_results = []
    for idx, path in enumerate(read_paths_list):
        call_id = "call_{}".format(idx)
        tool_calls_list.append({
            "id": call_id,
            "name": "read_file",
            "arguments": {"path": path},
        })
        tool_results.append({
            "tool_call_id": call_id,
            "tool": "read_file",
            "success": True,
            "data": {
                "path": path,
                "content": "x",
                "line_start": 1,
                "line_end": 1,
            },
            "source": None,
        })
    return {"iterations": [{
        "tool_calls": tool_calls_list,
        "tool_results": tool_results,
    }]}


# -- direct unit tests on enforce_inspected_evidence -----------------------

def test_unread_finding_is_stripped_and_recorded():
    """Test 1: a finding citing only never-read files is removed,
    its statement appears in uncertainties with the next action,
    and status is forced to incomplete."""
    trace = _fake_trace(["agents/collector.py"])
    report = _report(
        findings=[
            _finding(
                "records is exposed over MCP",
                [_ref(MCP_CONFIG_PATH)],
            ),
        ],
        evidence_refs=[_ref(MCP_CONFIG_PATH)],
    )

    cleaned, stripped = enforce_inspected_evidence(report, trace)

    assert len(stripped) == 1
    assert stripped[0].statement == "records is exposed over MCP"
    assert [f.statement for f in cleaned.findings] == []
    assert cleaned.status == "incomplete"
    assert any(
        "unresolved (evidence never inspected)" in note
        and "records is exposed over MCP" in note
        and ".mcp.json" in note
        and "read_file" in note
        for note in cleaned.uncertainties
    )


def test_inspected_finding_is_kept_whole():
    """Test 2: a finding whose refs resolve to inspected, covering
    evidence is kept unchanged (including mixed refs: any one
    inspected ref is enough)."""
    trace = _fake_trace([
        "agents/collector.py", "tools/fetch_record.py",
    ])
    report = _report(
        findings=[
            _finding(
                "collector DECLARES fetch_record",
                [
                    _ref(MCP_COLLECTOR_PATH),
                    _ref(FETCH_RECORD_PATH),
                ],
            ),
        ],
        evidence_refs=[
            _ref(MCP_COLLECTOR_PATH), _ref(FETCH_RECORD_PATH),
        ],
    )

    cleaned, stripped = enforce_inspected_evidence(report, trace)

    assert stripped == []
    assert len(cleaned.findings) == 1
    assert cleaned.findings[0].statement == (
        "collector DECLARES fetch_record"
    )
    assert cleaned.status == "complete"


def test_empty_refs_finding_is_kept():
    """Test 3: a finding with no evidence refs (honest statement of
    absence) is kept untouched; the report is returned unchanged."""
    report = _report(
        findings=[_finding("no MCP configuration exists", [])],
    )

    cleaned, stripped = enforce_inspected_evidence(report, _fake_trace([]))

    assert stripped == []
    assert cleaned is report  # no change at all


def test_mixed_findings_some_kept_some_stripped():
    """Test 4: inspected-backed findings survive; unread-backed ones
    are stripped; empty-refs findings survive; status incomplete."""
    trace = _fake_trace(["agents/collector.py"])
    report = _report(
        findings=[
            _finding(
                "collector exists",
                [_ref(MCP_COLLECTOR_PATH)],
            ),
            _finding(
                "records is exposed over MCP",
                [_ref(MCP_CONFIG_PATH)],
            ),
            _finding("no runtime evidence", []),
        ],
        evidence_refs=[
            _ref(MCP_COLLECTOR_PATH), _ref(MCP_CONFIG_PATH),
        ],
    )

    cleaned, stripped = enforce_inspected_evidence(report, trace)

    assert [f.statement for f in cleaned.findings] == [
        "collector exists", "no runtime evidence",
    ]
    assert [f.statement for f in stripped] == [
        "records is exposed over MCP",
    ]
    assert cleaned.status == "incomplete"


def test_all_inspected_returns_same_report_object():
    """Test 5: when every evidenced finding validates, the report is
    returned unchanged (same object — no copy, no rebuild)."""
    trace = _fake_trace(["agents/researcher.py"])
    report = _report(
        findings=[
            _finding(
                "researcher imports tools directly",
                [_ref(RESEARCHER_PATH)],
            ),
        ],
        evidence_refs=[_ref(RESEARCHER_PATH)],
    )

    cleaned, stripped = enforce_inspected_evidence(report, trace)

    assert cleaned is report
    assert stripped == []


def test_aggregate_rebuilt_from_kept_and_inspected():
    """Test 8: the rebuilt aggregate contains refs from kept findings
    plus any original aggregate ref that individually validates as
    inspected — never a reference this run did not read."""
    trace = _fake_trace([
        "agents/collector.py", "tools/fetch_record.py",
    ])
    report = _report(
        findings=[
            _finding(
                "collector DECLARES fetch_record",
                [_ref(MCP_COLLECTOR_PATH)],
            ),
            _finding(
                "records is exposed over MCP",
                [_ref(MCP_CONFIG_PATH)],
            ),
        ],
        evidence_refs=[
            _ref(MCP_COLLECTOR_PATH),
            _ref(FETCH_RECORD_PATH),  # inspected but no kept finding cites it
            _ref(MCP_CONFIG_PATH),    # never inspected
        ],
    )

    cleaned, stripped = enforce_inspected_evidence(report, trace)

    assert len(stripped) == 1
    kept_paths = {ref.path for ref in cleaned.evidence_refs}
    # kept finding's ref + individually-inspected aggregate ref
    assert MCP_COLLECTOR_PATH in kept_paths
    assert FETCH_RECORD_PATH in kept_paths
    assert MCP_CONFIG_PATH not in kept_paths


# -- specialist run_result enforcement (Tests 6 and 7) --------------------

def test_architect_run_result_enforces_inspected_evidence():
    """Test 6: ArchitectureInvestigator.run_result strips findings
    whose refs were never read, records uncertainty, forces
    incomplete."""
    specialist = build_investigator(
        read_paths(MCP_COLLECTOR_PATH),  # only this file is read
        structured(mcp_report_payload()),  # cites .mcp.json too
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )

    result = specialist.run_result("Investigate step P1")

    envelope = result.structured_output
    statements = [f.statement for f in envelope.findings]
    # collector findings survive (agents/collector.py was read)
    assert any(
        s.startswith("Component collector (agent)")
        for s in statements
    )
    assert any(
        "collector DECLARES fetch_record" in s for s in statements
    )
    # fetch_record / records / MCP-config findings are stripped
    # (tools/fetch_record.py and .mcp.json were never read)
    assert not any(
        s.startswith("Component fetch_record")
        for s in statements
    )
    assert not any(
        s.startswith("Component records")
        for s in statements
    )
    assert not any(
        "EXPOSED_BY" in s for s in statements
    )
    assert envelope.status == "incomplete"
    assert any(
        "unresolved (evidence never inspected)" in note
        for note in envelope.uncertainties
    )


def test_flow_run_result_enforces_inspected_evidence():
    """Test 7: FlowDependencyInvestigator.run_result strips findings
    whose refs were never read, records uncertainty, forces
    incomplete."""
    specialist = build_flow(
        read_paths(FLOW_MCP_CONFIG),  # only .mcp.json is read
        structured(mcp_payload()),    # cites agents/collector.py too
    )

    result = specialist.run_result("Investigate step P2")

    envelope = result.structured_output
    statements = [f.statement for f in envelope.findings]
    # .mcp.json-backed findings survive
    assert any("MCP" in s for s in statements)
    # agents/collector.py-backed findings are stripped
    assert not any(
        "IMPORTS" in s and "collector" in s for s in statements
    )
    assert envelope.status == "incomplete"
    assert any(
        "unresolved (evidence never inspected)" in note
        for note in envelope.uncertainties
    )


# -- end-to-end: the exact investigation question (Test 9) -----------------

def test_search_only_run_yields_no_supported_relationships():
    """Test 9: the investigation's failure mode — search_repository
    only, no read_file — must produce zero established findings and
    an incomplete status, not a confident wrong answer."""
    from repo_intelligence.evidence.validator import (
        gather_evidence, validate_claims,
    )

    # A trace that only ever searched (no read_file): the exact
    # pattern from the live run's P3 attempt 2.
    search_trace = {
        "iterations": [{
            "tool_calls": [{
                "id": "call_s1",
                "name": "search_repository",
                "arguments": {"query": "agent tool MCP interaction"},
            }],
            "tool_results": [{
                "tool_call_id": "call_s1",
                "tool": "search_repository",
                "success": True,
                "data": {
                    "results": [
                        {
                            "path": "agents/collector.py",
                            "line_start": 1,
                            "line_end": 40,
                        },
                        {
                            "path": ".mcp.json",
                            "line_start": 1,
                            "line_end": 20,
                        },
                    ],
                },
                "source": None,
            }],
        }],
    }
    # A report that cites those search hits as if they were evidence
    # (the exact pattern from the live run).
    report = _report(
        findings=[
            _finding(
                "collector DECLARES fetch_record",
                [_ref("agents/collector.py", 1, 40)],
            ),
            _finding(
                "fetch_record EXPOSED_BY records",
                [_ref(".mcp.json", 1, 20)],
            ),
        ],
        evidence_refs=[
            _ref("agents/collector.py", 1, 40),
            _ref(".mcp.json", 1, 20),
        ],
    )

    # gather_evidence correctly marks search results inspected=False
    evidence = gather_evidence(search_trace)
    assert len(evidence.inspected) == 0
    assert len(evidence.gathered) == 2  # named but never read

    cleaned, stripped = enforce_inspected_evidence(report, search_trace)

    assert len(stripped) == 2
    assert cleaned.findings == []
    assert cleaned.status == "incomplete"
    assert len(cleaned.uncertainties) == 2
    # no supported claim can be manufactured from search hits alone
    validations = validate_claims(
        [
            type("C", (), {
                "statement": f.statement,
                "evidence_refs": tuple(
                    r.to_evidence_ref() for r in f.evidence_refs
                ),
            })()
            for f in report.findings
        ],
        evidence,
        require_inspected=True,
    )
    assert all(v.status != "SUPPORTED" for v in validations)
