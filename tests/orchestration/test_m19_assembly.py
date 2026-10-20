"""M19 assembly — explicit merge + outcome pairing (moved, unchanged).

``merge_investigation_reports`` is the M17/M16 merge relocated from
runner to assembly with byte-identical semantics (still re-exported
from both the package and the runner); ``RunAssembler`` makes the
report/finding pairing an explicit, testable step of the coordinator.
"""

from __future__ import annotations

import pytest

from repo_intelligence.multiagent import (
    ArchitectureFinding,
    InvestigationReport,
)
from repo_intelligence.orchestration import (
    ANALYST,
    INVESTIGATOR,
    STEP_COMPLETED,
    STEP_FAILED,
    AssemblyOutcome,
    RunAssembler,
    StepResult,
    merge_investigation_reports,
)
from repo_intelligence.orchestration.assembly import (
    merge_investigation_reports as from_assembly,
)
from repo_intelligence.orchestration.runner import (
    merge_investigation_reports as from_runner,
)

from m17_doubles import (
    COMMIT,
    REPOSITORY,
    finding_payload,
    report_payload,
    task,
)


def _report(**overrides):
    return InvestigationReport.model_validate(report_payload(**overrides))


def _result(step_id, role, status, output):
    return StepResult(
        step_id=step_id,
        agent_role=role,
        status=status,
        output=output,
        evidence_refs=list(getattr(output, "evidence_refs", ()) or ()),
    )


def test_merge_is_re_exported_from_all_three_homes():
    assert from_assembly is merge_investigation_reports
    assert from_runner is merge_investigation_reports


def test_merge_requires_at_least_one_report():
    with pytest.raises(ValueError, match="requires a report"):
        merge_investigation_reports([], task())


def test_merge_concatenates_findings_and_deduplicates_refs():
    first = _report()
    second = _report(investigation_id="inv-2")
    merged = merge_investigation_reports([first, second], task())
    assert merged.investigation_id == "inv-1"
    assert merged.question == task().question
    assert merged.repository_scope == REPOSITORY
    assert merged.commit_scope == COMMIT
    assert len(merged.findings) == 2
    assert len(merged.evidence_refs) == 1  # README ref seen twice
    assert merged.investigated_sources == ["README.md"]


def test_merge_never_upgrades_an_incomplete_package():
    merged = merge_investigation_reports(
        [_report(), _report(status="incomplete")], task()
    )
    assert merged.status == "incomplete"


def test_assembler_collects_only_completed_investigator_reports():
    assembler = RunAssembler()
    completed = _report()
    results = [
        _result("P1", INVESTIGATOR, STEP_COMPLETED, completed),
        _result(
            "P2",
            INVESTIGATOR,
            STEP_FAILED,
            _report(investigation_id="inv-failed"),
        ),
        _result(
            "P3",
            ANALYST,
            STEP_COMPLETED,
            ArchitectureFinding.model_validate(finding_payload()),
        ),
    ]
    reports = assembler.collect_investigator_reports(results)
    assert reports == [completed]


def test_assembler_pairs_merged_report_with_the_finding():
    assembler = RunAssembler()
    report = _report()
    outcome = assembler.assemble(
        [_result("P1", INVESTIGATOR, STEP_COMPLETED, report)],
        task(),
        None,
    )
    assert isinstance(outcome, AssemblyOutcome)
    assert outcome.report is not None
    assert outcome.finding is None
    assert outcome.has_analysis is False


def test_assembler_has_analysis_only_with_a_finding():
    assembler = RunAssembler()
    report = _report()
    outcome = assembler.assemble(
        [_result("P1", INVESTIGATOR, STEP_COMPLETED, report)],
        task(),
        ArchitectureFinding.model_validate(finding_payload()),
    )
    assert outcome.has_analysis is True
    assert outcome.finding is not None


def test_assembler_assembles_nothing_from_an_empty_run():
    assembler = RunAssembler()
    outcome = assembler.assemble([], task(), None)
    assert outcome.report is None
    assert outcome.finding is None
    assert outcome.has_analysis is False
