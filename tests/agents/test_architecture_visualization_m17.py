"""Level 3 — the capability end to end through M10 → M17 (spec §24
Phase 8, §21, §22, §26).

The SAME specialist implementation, driven by the planner, orchestrated
step by step (Architecture Investigator → Flow / Dependency
Investigator → Architecture Visualization Agent), handed to the M16
validator through the typed envelope, and judged by the deterministic
validator — over the foreign arch_mcp fixture with scripted models.

One generic orchestrator, unmodified: the investigator slot routes the
two specialists by the plan's own step labels, the analyst slot holds
the visualizer.
"""

from __future__ import annotations

from typing import Dict, List

from architecture_doubles import (
    MCP_COLLECTOR_PATH,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    FakeLLMClient,
    build_investigator,
    fixture_bindings,
    rule,
    structured,
)
from viz_doubles import (
    FETCH_RECORD_PATH,
    RoutedInvestigator,
    VIZ_QUESTION,
    build_flow,
    build_visualization_agent,
    out_of_scope_spec_payload,
    read_paths,
    spec_payload,
    viz_arch_payload,
    viz_flow_payload,
    viz_plan_payload,
    viz_task,
)

from repo_intelligence.agent.trace import render_markdown
from repo_intelligence.agents import (
    ArchitectureVisualizationAgent,
    DiagramScope,
    DiagramSpec,
    validate_diagram_spec,
)
from repo_intelligence.evidence.validator import gather_evidence
from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.multiagent.contracts import ArchitectureFinding
from repo_intelligence.orchestration import (
    AGENT_FAILED,
    COMPLETED,
    INCOMPLETE,
    Orchestrator,
)
from repo_intelligence.planning.planner import (
    build_planning_request,
    plan_from_response,
)
from repo_intelligence.planning.validation import validate_plan


def _execute(
    arch_responses=None, flow_responses=None, viz_responses=None
):
    """Run the full application path and return the run plus its
    participants."""
    planner = FakeLLMClient(structured(viz_plan_payload()))
    response = planner.complete(build_planning_request(VIZ_QUESTION))
    investigation_plan = plan_from_response(response)
    validate_plan(investigation_plan)

    arch = build_investigator(
        *(arch_responses or [
            read_paths(
                MCP_COLLECTOR_PATH,
                FETCH_RECORD_PATH,
                MCP_CONFIG_PATH,
            ),
            structured(viz_arch_payload()),
        ]),
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )
    flow = build_flow(
        *(flow_responses or [
            read_paths(MCP_COLLECTOR_PATH, MCP_CONFIG_PATH),
            structured(viz_flow_payload()),
        ])
    )
    router = RoutedInvestigator({"S1": arch, "S2": flow})
    visualizer = build_visualization_agent(
        *(viz_responses or [structured(spec_payload())])
    )
    orchestrator = Orchestrator(router, visualizer)
    run = orchestrator.execute(viz_task(), investigation_plan)
    return run, arch, flow, visualizer, investigation_plan


def _assignments(run) -> List[Dict]:
    return [
        event
        for event in run.events
        if event["operation"] == "orchestration.step.assign"
    ]


def test_three_specialists_run_through_one_generic_orchestrator():
    run, arch, flow, visualizer, plan = _execute()

    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert [event["agent_role"] for event in _assignments(run)] == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    # one investigator slot, two specialists, routed by the plan's
    # own step labels — the orchestrator itself never changed: S1's
    # envelope speaks the architecture report, S2's the flow report
    first = run.step_result("S1").output
    second = run.step_result("S2").output
    assert any(
        finding.statement.startswith("Component collector")
        for finding in first.findings
    )
    assert any(
        "IMPORTS" in finding.statement for finding in second.findings
    )
    assert not any(
        finding.statement.startswith("Component")
        for finding in second.findings
    )
    assert [step.status for step in plan.steps] == ["completed"] * 3
    assert [type(result.output).__name__ for result in run.step_results] == [
        "InvestigationReport",
        "InvestigationReport",
        "ArchitectureFinding",
    ]
    assert isinstance(visualizer, ArchitectureVisualizationAgent)
    assert visualizer.tools == []
    assert visualizer.last_spec is not None
    assert arch.last_trace is not None
    assert flow.last_trace is not None


def test_the_diagram_handoff_survives_the_orchestrated_run():
    run, _, _, visualizer, _ = _execute()
    finding = run.finding
    spec = visualizer.last_spec

    assert finding is not None
    assert finding.investigation_id == run.report.investigation_id
    assert finding.repository_scope == run.task.repository_scope
    assert finding.commit_scope == run.task.commit_scope
    assert finding.question == VIZ_QUESTION

    statements = [conclusion.statement for conclusion in finding.conclusions]
    assert any(
        statement.startswith("Component collector (agent)")
        for statement in statements
    )
    assert any(
        statement.startswith("Component records (mcp_server)")
        for statement in statements
    )
    assert "collector DECLARES fetch_record [observed]" in statements
    assert "fetch_record EXPOSED_BY records [derived]" in statements
    assert finding.open_questions == spec.uncertainties

    # evidence scope survives the handoff: every diagram reference is
    # carried by the merged report aggregate
    report_paths = {ref.path for ref in run.report.evidence_refs}
    spec_paths = {ref.path for ref in spec.evidence_refs}
    assert spec_paths <= report_paths
    assert spec_paths == {
        MCP_COLLECTOR_PATH,
        FETCH_RECORD_PATH,
        MCP_CONFIG_PATH,
    }


def test_the_orchestrated_spec_validates_against_the_merged_evidence():
    run, arch, flow, visualizer, _ = _execute()

    scope = DiagramScope.from_investigation_report(run.report)
    merged_trace = {
        "iterations": [
            iteration.to_dict()
            for trace in (arch.last_trace, flow.last_trace)
            for iteration in trace.iterations
        ]
    }
    result = validate_diagram_spec(
        visualizer.last_spec,
        scope=scope,
        evidence=gather_evidence(merged_trace),
    )

    assert result.status == "passed"
    assert all(check.ok for check in result.checks)
    assert set(scope.node_ids) == {
        "collector",
        "fetch_record",
        "records",
    }


def test_every_m16_rule_passed_for_the_diagram_finding():
    run, _, _, _, _ = _execute()

    for name in (
        "message_contract",
        "scope_match",
        "evidence_structural",
        "collaboration",
        "evidence_valid",
        "evidence_handoff",
        "classification",
        "output_structure",
    ):
        assert rule(run.validation, name).ok is True


def test_the_trace_shows_the_actual_role_chain():
    run, _, _, visualizer, _ = _execute()

    roles = [event["agent_role"] for event in _assignments(run)]
    assert roles == ["INVESTIGATOR", "INVESTIGATOR", "ANALYST"]

    handoffs = [
        event
        for event in run.events
        if event["operation"] == "orchestration.handoff.send"
    ]
    validation_handoff = [
        event
        for event in handoffs
        if event["message_type"] == "VALIDATION_RESULT"
    ]
    assert validation_handoff
    assert validation_handoff[0]["sender"] == "validator"

    markdown = render_markdown(visualizer.last_trace)
    assert "## Orchestration" in markdown
    assert "orchestration.step.assign" in markdown
    assert run.event_operations()[-1] == "orchestration.complete"
    # the visualizer's own runs are ordinary agent runs in the ONE
    # trace system — no visualization-specific trace exists
    assert any(
        entry["agent_role"] == "ANALYST" and entry["run_id"]
        for entry in run.agent_runs
    )


# -- failure propagation (spec §22) ---------------------------------------


def test_investigator_failure_blocks_the_visualization():
    # Runtime metadata (question/scopes) is filled from the request,
    # so the way to fail S1 here is SUBSTANTIVE invalidity — which is
    # exactly what must still block the visualization.
    broken = viz_arch_payload()
    broken["status"] = "unreachable"
    run, _, _, visualizer, _ = _execute(
        arch_responses=[structured(broken), structured(broken)]
    )

    assert run.termination_reason == "AGENT_FAILED"
    assert run.status in (INCOMPLETE, AGENT_FAILED)
    assert run.finding is None
    assert run.validation is None
    assert run.report is None
    assert run.failed_steps == ["S1"]
    assert run.blocked_steps == ["S2", "S3"]
    # the visualizer was never consulted — no fabricated diagram
    assert visualizer.last_spec is None
    assert visualizer.agent._llm_client.requests == []


def test_visualization_failure_yields_no_fabricated_finding():
    broken = spec_payload()
    del broken["title"]
    run, _, _, visualizer, _ = _execute(
        viz_responses=[structured(broken), structured(broken)]
    )

    assert run.termination_reason == "AGENT_FAILED"
    assert run.status == AGENT_FAILED
    assert run.report is not None
    assert run.finding is None
    assert run.validation is None
    assert run.failed_steps == ["S3"]
    assert run.step_result("S3").status == "failed"
    assert run.step_result("S3").attempts == 2
    assert visualizer.last_spec is None


def test_a_rejected_spec_fails_the_diagram_validation_not_the_run():
    """The model produced a constructible DiagramSpec; the
    deterministic validator is the one that rejects it (spec §22)."""
    run, _, _, visualizer, _ = _execute()
    scope = DiagramScope.from_investigation_report(run.report)
    rejected = DiagramSpec(**out_of_scope_spec_payload())

    result = validate_diagram_spec(rejected, scope=scope)

    assert result.status == "failed"
    assert rule(result, "scope_validity").ok is False
    assert "notes_store" in rule(result, "scope_validity").detail
    # ...while the actual orchestrated spec still stands
    assert (
        validate_diagram_spec(visualizer.last_spec, scope=scope).status
        == "passed"
    )


def test_the_specialist_is_just_an_injected_participant():
    run, _, _, visualizer, _ = _execute()

    assert isinstance(visualizer, ArchitectureVisualizationAgent)
    assert visualizer.agent is not None
    investigator_results = [
        result
        for result in run.step_results
        if result.agent_role == "INVESTIGATOR"
    ]
    assert all(
        isinstance(result.output, InvestigationReport)
        for result in investigator_results
    )
    analyst_results = [
        result
        for result in run.step_results
        if result.agent_role == "ANALYST"
    ]
    assert all(
        isinstance(result.output, ArchitectureFinding)
        for result in analyst_results
    )
