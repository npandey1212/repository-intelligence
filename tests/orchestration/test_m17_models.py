"""M17 run state + deterministic report merge (spec §12/§13/§31/§32).

``OrchestrationRun`` is the authoritative record of one orchestrated
workflow; ``merge_investigation_reports`` is the ONE deterministic way
several Investigator steps become one evidence package for the M16
validator — findings concatenated in step order, refs de-duplicated,
open questions/uncertainties/sources unioned, status downgraded (never
upgraded), scope and question taken from the shared task.
"""

from __future__ import annotations

import pytest

from repo_intelligence.orchestration import (
    PENDING,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_STATUSES,
    OrchestrationRun,
    StepResult,
    TERMINATION_REASONS,
    is_terminal,
    merge_investigation_reports,
    utcnow,
)
from repo_intelligence.planning.models import InvestigationPlan, PlanStep

from m17_doubles import (
    COMMIT,
    QUESTION,
    REPOSITORY,
    report_payload,
    task,
    two_step_plan,
)


def _report(**overrides):
    from repo_intelligence.multiagent import InvestigationReport

    return InvestigationReport.parse_obj(report_payload(**overrides))


class TestRunState:
    def test_defaults_are_pending_and_not_final(self):
        run = OrchestrationRun(
            run_id="r1", task=task(), plan=two_step_plan(), started_at=utcnow()
        )
        assert run.status == PENDING
        assert run.final is False
        assert run.current_step is None
        assert run.step_results == []
        assert run.messages == []
        assert run.events == []
        assert run.report is None and run.finding is None
        assert run.validation is None
        assert run.termination_reason is None
        assert run.retries == 0 and run.tool_calls_total == 0

    def test_step_result_lookup_and_investigation_id(self):
        run = OrchestrationRun(
            run_id="r1", task=task(), plan=two_step_plan(), started_at=utcnow()
        )
        report = _report()
        run.step_results.append(
            StepResult(
                step_id="P1",
                agent_role="INVESTIGATOR",
                status=STEP_COMPLETED,
                output=report,
            )
        )
        assert run.step_result("P1").investigation_id == "inv-1"
        assert run.step_result("P2") is None

    def test_step_result_without_typed_output_has_no_investigation_id(self):
        result = StepResult(
            step_id="P1", agent_role="INVESTIGATOR", status=STEP_FAILED
        )
        assert result.investigation_id is None

    def test_event_operations_are_ordered(self):
        run = OrchestrationRun(
            run_id="r1", task=task(), plan=two_step_plan(), started_at=utcnow()
        )
        run.events.extend(
            [
                {"operation": "orchestration.start", "sequence": 1},
                {"operation": "orchestration.complete", "sequence": 2},
            ]
        )
        assert run.event_operations() == [
            "orchestration.start",
            "orchestration.complete",
        ]

    def test_step_status_vocabulary_is_closed(self):
        assert STEP_STATUSES == ("completed", "failed", "blocked")

    def test_plan_status_vocabulary_is_not_reused_by_steps(self):
        plan = InvestigationPlan(
            objective=QUESTION,
            steps=[PlanStep(id="P1", goal="Identify the boundary")],
        )
        assert plan.steps[0].status == "pending"
        assert StepResult(
            step_id="P1", agent_role="INVESTIGATOR", status=STEP_COMPLETED
        ).status == "completed"

    def test_utcnow_is_iso_and_trailing_z(self):
        stamp = utcnow()
        assert stamp.endswith("Z")
        assert "T" in stamp


class TestMergeInvestigationReports:
    def test_requires_at_least_one_report(self):
        with pytest.raises(ValueError):
            merge_investigation_reports([], task())

    def test_findings_concatenated_in_step_order(self):
        first = _report(
            investigation_id="inv-1",
            findings=[
                {
                    "statement": "first step finding",
                    "evidence_refs": [{"path": "README.md"}],
                    "support": "SUPPORTED",
                }
            ],
            evidence_refs=[{"path": "README.md"}],
        )
        second = _report(
            investigation_id="inv-2",
            findings=[
                {
                    "statement": "second step finding",
                    "evidence_refs": [{"path": "src/service.py"}],
                    "support": "SUPPORTED",
                }
            ],
            evidence_refs=[{"path": "src/service.py"}],
            open_questions=["who calls it"],
            uncertainties=["boundary may move"],
            investigated_sources=["src/service.py"],
        )
        merged = merge_investigation_reports([first, second], task())
        assert merged.investigation_id == "inv-1"
        assert [item.statement for item in merged.findings] == [
            "first step finding",
            "second step finding",
        ]
        assert [ref.path for ref in merged.evidence_refs] == [
            "README.md",
            "src/service.py",
        ]
        assert merged.open_questions == ["who calls it"]
        assert merged.uncertainties == ["boundary may move"]
        assert merged.investigated_sources == [
            "README.md",
            "src/service.py",
        ]

    def test_duplicate_evidence_refs_are_deduplicated_by_identity(self):
        first = _report()
        second = _report(investigation_id="inv-2")
        merged = merge_investigation_reports([first, second], task())
        assert len(merged.evidence_refs) == 1
        assert merged.evidence_refs[0].path == "README.md"

    def test_scope_and_question_come_from_the_shared_task(self):
        first = _report(repository_scope="other-repo", question="old question")
        merged = merge_investigation_reports(
            [first], task(repository_scope=REPOSITORY, question=QUESTION)
        )
        assert merged.repository_scope == REPOSITORY
        assert merged.commit_scope == COMMIT
        assert merged.question == QUESTION

    def test_status_downgrades_but_never_upgrades(self):
        complete = _report()
        incomplete = _report(investigation_id="inv-2", status="incomplete")
        assert merge_investigation_reports(
            [complete, incomplete], task()
        ).status == "incomplete"
        assert merge_investigation_reports(
            [incomplete, incomplete], task()
        ).status == "incomplete"
        assert merge_investigation_reports(
            [complete, complete], task()
        ).status == "complete"

    def test_union_deduplicates_identical_entries(self):
        first = _report(
            open_questions=["q1"],
            uncertainties=["u1"],
            investigated_sources=["README.md"],
        )
        second = _report(
            investigation_id="inv-2",
            open_questions=["q1", "q2"],
            uncertainties=["u1", "u2"],
            investigated_sources=["README.md", "src/main.py"],
        )
        merged = merge_investigation_reports([first, second], task())
        assert merged.open_questions == ["q1", "q2"]
        assert merged.uncertainties == ["u1", "u2"]
        assert merged.investigated_sources == ["README.md", "src/main.py"]


def test_termination_reason_vocabulary_is_reachable_from_the_state_names():
    # the run's terminal reason is one of the eight explicit values —
    # never free-form prose written by the orchestrator at a call site.
    assert "SUCCESS" in TERMINATION_REASONS
    assert all(
        isinstance(reason, str) and reason for reason in TERMINATION_REASONS
    )
    assert is_terminal("INCOMPLETE")
