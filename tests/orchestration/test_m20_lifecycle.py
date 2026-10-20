"""M20 sequential lifecycle — plan-order dispatch, plan-order merge,
and terminal-state consistency (no budget termination overwritten by
a later validation status).

Sequential dispatch and failure/blocking semantics are pinned by
``test_m19_sequential_dispatch.py`` / ``test_m19_runner_tasks.py``
(retained unchanged); this file adds the M20 merge-order and
terminal-consistency proofs.
"""

from __future__ import annotations

from repo_intelligence.orchestration import (
    BUDGET_EXCEEDED,
    COMPLETED,
    TASK_COMPLETED,
    OrchestrationConfig,
    RoleAssignment,
)

from m17_doubles import (
    COMMIT,
    REPOSITORY,
    build,
    call,
    finding_payload,
    report_payload,
    structured,
    task,
    tool_calls,
    two_step_plan,
)

from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.planning.models import PlanStep


def _payload(investigation_id, statement):
    return report_payload(
        investigation_id=investigation_id,
        findings=[
            {
                "statement": statement,
                "evidence_refs": [{"path": "README.md"}],
                "support": "SUPPORTED",
            }
        ],
        evidence_refs=[{"path": "README.md"}],
        investigated_sources=["README.md"],
    )


def test_two_investigator_steps_dispatch_and_merge_in_plan_order():
    # Plan P1(INV), P2(INV), P3(ANA): dispatch is strictly plan
    # order, and the merged report concatenates findings in PLAN
    # order (the first report keeps its investigation_id).
    first = _payload("inv-first", "README documents the boundary")
    second = _payload("inv-second", "service.py enforces the boundary")
    orch, investigator, _ = build(
        inv_responses=[
            tool_calls(call("read_file", {"path": "README.md"})),
            structured(first),
            structured(second),
        ],
        ana_responses=[
            structured(finding_payload(investigation_id="inv-first"))
        ],
    )
    plan = two_step_plan(
        steps=[
            PlanStep(id="P1", goal="Identify the boundary"),
            PlanStep(id="P2", goal="Locate enforcement"),
            PlanStep(id="P3", goal="Explain the enforcement"),
        ]
    )
    assignments = [
        RoleAssignment(step_id="P1", role="INVESTIGATOR", reason="evidence"),
        RoleAssignment(step_id="P2", role="INVESTIGATOR", reason="evidence"),
        RoleAssignment(step_id="P3", role="ANALYST", reason="analysis"),
    ]
    run = orch.execute(task(), plan, role_assignments=assignments)

    assert run.status == COMPLETED
    # Dispatch order: P1 → P2 → P3, never interleaved.
    starts = [
        event["step_id"]
        for event in run.events
        if event["operation"] == "orchestration.step.start"
    ]
    assert starts == ["P1", "P2", "P3"]
    assert [record.task_id for record in run.tasks] == ["P1", "P2", "P3"]
    assert all(
        record.status == TASK_COMPLETED for record in run.tasks
    )
    # Merge order is PLAN order (P1's finding first), and the
    # aggregate keeps the FIRST report's investigation_id.
    assert run.report is not None
    assert run.report.investigation_id == "inv-first"
    assert [finding.statement for finding in run.report.findings] == [
        "README documents the boundary",
        "service.py enforces the boundary",
    ]


def test_budget_terminal_status_survives_to_the_end_of_the_run():
    # A run terminated by the budget never reaches (or is overwritten
    # by) the deterministic validation stage.
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", {"path": "README.md"})),
            structured({"findings": []}),  # broken → retryable
        ],
        config=OrchestrationConfig(max_total_tool_calls=1),
    )
    run = orch.execute(task(), two_step_plan())

    assert run.status == BUDGET_EXCEEDED
    assert run.termination_reason == "BUDGET_EXCEEDED"
    assert run.validation is None
    assert run.validation_status is None
    assert run.finding is None
    # The partial evidence package (if any) is preserved, never
    # re-validated into a different terminal status.
    assert "orchestration.validation.start" not in run.event_operations()
    assert "orchestration.complete" not in run.event_operations()


def test_happy_path_validation_still_reaches_success():
    orch, _, _ = build()
    run = orch.execute(task(), two_step_plan())
    assert run.status == COMPLETED
    assert run.validation is not None
    assert run.validation_status == "passed"
    assert isinstance(run.report, InvestigationReport)
