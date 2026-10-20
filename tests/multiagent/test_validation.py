"""M16 deterministic workflow validation (spec §22 / §23 tests 4–8).

Eight rules, no LLM: evidence handoff integrity, scope agreement,
message routing, collaboration proof, M13 evidence validity against
the Investigator's own trace (inspected-only), classification honesty,
and output structure — plus visible, non-fatal tool-failure
collection. Inputs are typed contracts + synthetic trace mappings.
"""

from __future__ import annotations

from repo_intelligence.multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    ReportEvidenceRef,
    TaskPayload,
    make_message,
)
from repo_intelligence.multiagent.validation import (
    RULE_ORDER,
    validate_collaboration,
)

COMMIT = "c" * 40
REPOSITORY = "example-repository"


def _ref(path: str = "README.md") -> ReportEvidenceRef:
    return ReportEvidenceRef(path=path)


def _task(**overrides):
    data = {
        "question": "Explain how the parts interact.",
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }
    data.update(overrides)
    return TaskPayload(**data)


def _report(refs=("README.md",), **overrides):
    refs = tuple(refs)
    data = {
        "investigation_id": "inv-1",
        "question": "Explain how the parts interact.",
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "evidence_refs": [_ref(path).dict() for path in refs],
        "findings": [
            {
                "statement": "the dispatcher owns the boundary",
                "evidence_refs": [_ref(path).dict() for path in refs],
                "support": "SUPPORTED",
            }
        ],
    }
    data.update(overrides)
    return InvestigationReport(**data)


def _finding(
    report,
    extra_refs=(),
    conclusion_refs=None,
    classification="observed",
    conclusion_statement="the boundary is application-owned",
    **overrides,
):
    all_refs = [r.path for r in report.evidence_refs] + list(extra_refs)
    if conclusion_refs is None:
        conclusion_refs = [r.path for r in report.evidence_refs]
    data = {
        "investigation_id": report.investigation_id,
        "question": report.question,
        "repository_scope": report.repository_scope,
        "commit_scope": report.commit_scope,
        "evidence_refs": [_ref(path).dict() for path in all_refs],
        "conclusions": [
            {
                "statement": conclusion_statement,
                "classification": classification,
                "evidence_refs": [_ref(path).dict() for path in conclusion_refs],
            }
        ],
        "summary": "One dispatcher, one boundary.",
    }
    data.update(overrides)
    return ArchitectureFinding(**data)


def _read_trace(*paths):
    """Synthetic read_file-style trace: each path inspected (content
    returned), so references to it are eligible under
    ``require_inspected=True``."""
    iterations = []
    for index, path in enumerate(paths, start=1):
        call_id = "c{}".format(index)
        iterations.append(
            {
                "tool_calls": [
                    {
                        "id": call_id,
                        "name": "read_file",
                        "arguments": {"path": path},
                    }
                ],
                "tool_results": [
                    {
                        "tool_call_id": call_id,
                        "tool": "read_file",
                        "success": True,
                        "data": {"path": path, "content": "…"},
                    }
                ],
            }
        )
    return {"iterations": iterations}


def _search_trace(path="README.md"):
    """Search-style trace: the path was DISCOVERED, never read."""
    return {
        "iterations": [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_repository",
                        "arguments": {"query": "boundary"},
                    }
                ],
                "tool_results": [
                    {
                        "tool_call_id": "c1",
                        "tool": "search_repository",
                        "success": True,
                        "data": {"results": [{"path": path}]},
                    }
                ],
            }
        ]
    }


def _failed_tool_trace():
    return {
        "iterations": [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "read_file",
                        "arguments": {"path": "missing.py"},
                    }
                ],
                "tool_results": [
                    {
                        "tool_call_id": "c1",
                        "tool": "read_file",
                        "success": False,
                        "error": {"code": "FILE_NOT_FOUND"},
                    }
                ],
            }
        ]
    }


def _messages(task, report, finding):
    return [
        make_message(
            "INVESTIGATION_REQUEST",
            "task",
            "investigator",
            task,
            task.repository_scope,
            task.commit_scope,
        ),
        make_message(
            "INVESTIGATION_RESULT",
            "investigator",
            "analyst",
            report,
            report.repository_scope,
            report.commit_scope,
        ),
        make_message(
            "ANALYSIS_REQUEST",
            "task",
            "analyst",
            report,
            report.repository_scope,
            report.commit_scope,
        ),
        make_message(
            "ANALYSIS_RESULT",
            "analyst",
            "task",
            finding,
            finding.repository_scope,
            finding.commit_scope,
        ),
    ]


def _validated(task=None, report=None, finding=None, trace=None, messages=None):
    task = task or _task()
    report = report or _report()
    finding = finding if finding is not None else _finding(report)
    trace = trace if trace is not None else _read_trace("README.md")
    if messages is None:
        messages = _messages(task, report, finding)
    return validate_collaboration(
        task=task,
        report=report,
        finding=finding,
        investigator_trace=trace,
        messages=messages,
    )


def _check_by_rule(result, rule):
    matches = [check for check in result.checks if check.rule == rule]
    assert len(matches) == 1, rule
    return matches[0]


class TestHappyPath:
    def test_all_eight_rules_pass(self):
        result = _validated()
        assert result.status == "passed"
        assert [check.rule for check in result.checks] == list(RULE_ORDER)
        assert all(check.ok for check in result.checks)
        assert result.tool_failures == []
        assert result.investigation_id == "inv-1"

    def test_conflicting_with_two_supported_paths_passes(self):
        report = _report(refs=("README.md", "agent.py"))
        finding = _finding(
            report,
            conclusion_refs=["README.md", "agent.py"],
            classification="conflicting",
        )
        result = _validated(
            report=report,
            finding=finding,
            trace=_read_trace("README.md", "agent.py"),
        )
        assert _check_by_rule(result, "classification").ok
        assert result.status == "passed"


class TestRuleScopeMatch:
    def test_commit_mismatch_fails(self):
        report = _report()
        finding = _finding(report, commit_scope="f" * 40)
        result = _validated(report=report, finding=finding)
        check = _check_by_rule(result, "scope_match")
        assert not check.ok
        assert "scope mismatch" in check.detail
        assert result.status == "failed"

    def test_repository_mismatch_fails(self):
        report = _report()
        finding = _finding(report, repository_scope="other-repository")
        result = _validated(report=report, finding=finding)
        assert not _check_by_rule(result, "scope_match").ok


class TestRuleEvidenceHandoff:
    def test_finding_ref_outside_report_aggregate_fails(self):
        report = _report(refs=("README.md",))
        finding = _finding(report, extra_refs=("src/mystery.py",))
        result = _validated(report=report, finding=finding)
        check = _check_by_rule(result, "evidence_handoff")
        assert not check.ok
        assert "missing from report" in check.detail

    def test_conclusion_ref_outside_report_fails(self):
        report = _report(refs=("README.md",))
        # Legal at the message boundary only if the finding aggregate
        # carries it — but the aggregate is outside the report, so the
        # HANDOFF rule is the one that must catch it.
        finding = _finding(
            report,
            extra_refs=("src/mystery.py",),
            conclusion_refs=["src/mystery.py"],
        )
        result = _validated(report=report, finding=finding)
        assert not _check_by_rule(result, "evidence_handoff").ok


class TestRuleEvidenceValid:
    def test_unobserved_evidence_fails(self):
        report = _report(refs=("never-read.py",))
        finding = _finding(report)
        result = _validated(
            report=report, finding=finding, trace=_read_trace("README.md")
        )
        check = _check_by_rule(result, "evidence_valid")
        assert not check.ok
        assert "UNSUPPORTED" in check.detail
        assert result.status == "failed"

    def test_search_discovery_is_not_inspected_evidence(self):
        report = _report(refs=("README.md",))
        finding = _finding(report)
        result = _validated(
            report=report, finding=finding, trace=_search_trace("README.md")
        )
        check = _check_by_rule(result, "evidence_valid")
        assert not check.ok

    def test_covered_line_range_stays_supported(self):
        report = _report(refs=("README.md",))
        finding = _finding(report)
        # Whole-file gathered evidence covers any declared range —
        # bounds must stay consistent across every model so the
        # handoff rules stay satisfied.
        touched = list(report.evidence_refs)
        touched.extend(
            ref for item in report.findings for ref in item.evidence_refs
        )
        touched.extend(finding.evidence_refs)
        for conclusion in finding.conclusions:
            touched.extend(conclusion.evidence_refs)
        for ref in touched:
            ref.line_start = 1
            ref.line_end = 5
        result = _validated(report=report, finding=finding)
        assert _check_by_rule(result, "evidence_valid").ok
        assert result.status == "passed"


class TestRuleClassification:
    def test_observed_without_supported_evidence_fails(self):
        report = _report(refs=("never-read.py",))
        finding = _finding(report, classification="observed")
        result = _validated(
            report=report, finding=finding, trace=_read_trace("README.md")
        )
        check = _check_by_rule(result, "classification")
        assert not check.ok
        assert "observed without supported evidence" in check.detail

    def test_conflicting_with_one_path_fails(self):
        report = _report(refs=("README.md",))
        finding = _finding(report, classification="conflicting")
        result = _validated(report=report, finding=finding)
        check = _check_by_rule(result, "classification")
        assert not check.ok
        assert ">= 2 distinct supported paths" in check.detail

    def test_unknown_classification_needs_no_refs(self):
        report = _report(refs=("never-read.py",))
        finding = _finding(report, classification="unknown")
        result = _validated(
            report=report, finding=finding, trace=_read_trace("README.md")
        )
        # evidence_valid still fails (a conclusion must resolve), but
        # the unknown classification itself is not overclaimed.
        assert _check_by_rule(result, "classification").ok


class TestRuleMessageContract:
    def test_missing_handoff_fails(self):
        task, report = _task(), _report()
        finding = _finding(report)
        messages = _messages(task, report, finding)[:2]
        result = _validated(task=task, report=report, finding=finding,
                            messages=messages)
        check = _check_by_rule(result, "message_contract")
        assert not check.ok
        assert "expected handoffs" in check.detail

    def test_swapped_participants_fail(self):
        task, report = _task(), _report()
        finding = _finding(report)
        messages = _messages(task, report, finding)
        messages[1] = make_message(
            "INVESTIGATION_RESULT",
            "analyst",
            "investigator",
            report,
            report.repository_scope,
            report.commit_scope,
        )
        result = _validated(task=task, report=report, finding=finding,
                            messages=messages)
        assert not _check_by_rule(result, "message_contract").ok


class TestRuleCollaboration:
    def test_investigation_id_mismatch_fails(self):
        report = _report()
        finding = _finding(report, investigation_id="inv-OTHER")
        result = _validated(report=report, finding=finding)
        check = _check_by_rule(result, "collaboration")
        assert not check.ok
        assert "investigation_id" in check.detail

    def test_analysis_request_with_wrong_report_fails(self):
        task, report = _task(), _report()
        other_report = _report(investigation_id="inv-2")
        finding = _finding(report)
        messages = _messages(task, report, finding)
        messages[2] = make_message(
            "ANALYSIS_REQUEST",
            "task",
            "analyst",
            other_report,
            other_report.repository_scope,
            other_report.commit_scope,
        )
        result = _validated(task=task, report=report, finding=finding,
                            messages=messages)
        assert not _check_by_rule(result, "collaboration").ok


class TestRuleOutputStructure:
    def test_empty_conclusions_fail(self):
        report = _report()
        finding = _finding(report, conclusions=[])
        result = _validated(report=report, finding=finding)
        check = _check_by_rule(result, "output_structure")
        assert not check.ok
        assert "no conclusions" in check.detail

    def test_empty_findings_fail(self):
        report = _report(findings=[])
        finding = _finding(report, conclusions=[])
        result = _validated(report=report, finding=finding)
        assert not _check_by_rule(result, "output_structure").ok


class TestRuleEvidenceStructural:
    def test_invalid_ref_catches_via_unvalidated_construction(self):
        bad = ReportEvidenceRef.model_construct(
            path="", line_start=9, line_end=2
        )
        report = _report()
        finding = _finding(report, conclusions=[])
        messages = _messages(_task(), report, finding)
        # Bypass the message boundary on purpose: the STRUCTURAL rule
        # must still catch it (defense in depth, never a crash).
        # Pydantic v2 revalidates nested model instances at the
        # AgentMessage constructor (v1 did not), so build the message
        # with the clean report first, then inject the bad ref into
        # the already-built message without re-running construction.
        report.__dict__["evidence_refs"] = [bad]
        messages[1].__dict__["evidence_refs"] = [bad]
        result = _validated(
            report=report,
            finding=finding,
            messages=messages,
        )
        check = _check_by_rule(result, "evidence_structural")
        assert not check.ok


class TestToolFailureCollection:
    def test_failures_visible_but_not_fatal(self):
        report = _report()
        finding = _finding(report)
        trace = _read_trace("README.md")
        trace["iterations"].extend(_failed_tool_trace()["iterations"])
        result = _validated(report=report, finding=finding, trace=trace)
        assert result.status == "passed"
        assert len(result.tool_failures) == 1
        failure = result.tool_failures[0]
        assert failure.agent == "investigator"
        assert failure.tool == "read_file"
        assert failure.error_code == "FILE_NOT_FOUND"
