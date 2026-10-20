"""Levels 2 + 3 — the Flow / Dependency Investigator itself (§22, §39).

Example-style scenarios a developer can read as the capability's
specification: direct flows, multi-hop paths, agent→tool→MCP chains,
NOT_FOUND vs UNKNOWN, relationship-kind preservation, conflicting
evidence, evidence rules, repository-agnosticism, the versioned
prompt, the M16/M17 handoff protocol, and the full orchestrated run —
all over scripted doubles and tiny fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from architecture_doubles import build_analyst, rule
from flow_doubles import (
    ALPHA_PATH,
    API_PATH,
    BRIDGE_PATH,
    COLLECTOR_PATH,
    FLOW_CHAIN_ROOT,
    FLOW_COMMIT,
    FLOW_CONFLICT_ROOT,
    FLOW_DISCONNECTED_ROOT,
    FLOW_QUESTION,
    FLOW_REPOSITORY,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    MCP_QUESTION,
    PRODUCER_PATH,
    TRANSFORM_PATH,
    WIRING_DOC_PATH,
    FakeLLMClient,
    build_flow_investigator,
    conflicting_payload,
    contains_payload,
    direct_payload,
    fixture_bindings,
    flow_payload,
    flow_plan,
    flow_plan_payload,
    flow_task,
    mcp_endpoint_payload,
    mcp_finding_payload,
    mcp_leg_one_payload,
    mcp_leg_two_payload,
    mcp_payload,
    no_path_payload,
    read_then_report,
    structured,
)

from repo_intelligence.agent import Agent
from repo_intelligence.agent.trace import render_markdown
from repo_intelligence.agents import (
    ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    ARCHITECTURE_TOOLS,
    FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1,
    FLOW_DEPENDENCY_TOOLS,
    FLOW_PROMPT_VERSION,
    PATH_STATUSES,
    RELATIONSHIP_TYPES,
    FlowDependencyInvestigator,
    FlowDependencyInvestigatorError,
    FlowDependencyReport,
    build_flow_dependency_investigator_prompt,
)
from repo_intelligence.agents import __file__ as agents_init_file
from repo_intelligence.engineering.models import RELATIONSHIP_KINDS
from repo_intelligence.evidence.models import (
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
)
from repo_intelligence.llm import (
    LLMResponse,
    Message,
    StructuredOutputValidationError,
)
from repo_intelligence.multiagent import (
    COMPLETED,
    InvestigationReport,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import Orchestrator
from repo_intelligence.planning.planner import (
    build_planning_request,
    plan_from_response,
)
from repo_intelligence.planning.validation import validate_plan

AGENTS_PACKAGE = Path(agents_init_file).parent

FLOW_SOURCES = [
    AGENTS_PACKAGE / "flow_dependency_investigator.py",
    AGENTS_PACKAGE / "models" / "flow_dependency_report.py",
    AGENTS_PACKAGE / "prompts" / "flow_dependency_investigator.py",
]

READ_ONLY_ANSWER = LLMResponse(
    message=Message(role="assistant", content="not a typed report"),
    finish_reason="stop",
)


def _user_text(client) -> str:
    """All user-message text the scripted client was asked to answer."""
    texts = []
    for request in client.requests:
        for message in request.messages:
            if message.role == "user":
                texts.append(message.content or "")
    return "\n---\n".join(texts)


def _specialist(client, bindings=None) -> FlowDependencyInvestigator:
    """The composition under test: specialist over a scripted Agent."""
    agent = Agent(
        client,
        tool_dispatcher=restricted_dispatcher(
            "investigator",
            bindings=bindings
            if bindings is not None
            else fixture_bindings(FLOW_CHAIN_ROOT),
        ),
        structured_output_model=FlowDependencyReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    return FlowDependencyInvestigator(
        agent=agent, tools=FLOW_DEPENDENCY_TOOLS
    )


def plain_agent() -> Agent:
    """An Agent with no structured model (output is plain text)."""
    return Agent(FakeLLMClient(READ_ONLY_ANSWER))


# -- direct execution: flows, paths, intermediates (§22 tests 1-3, 5-6) -


def test_direct_flow_with_evidence_is_reported_as_found():
    specialist = build_flow_investigator(
        *read_then_report([PRODUCER_PATH], direct_payload())
    )

    report = specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert isinstance(report, FlowDependencyReport)
    assert report.path_status == "FOUND"
    assert report.source == "producer"
    assert report.target == "transform"
    assert len(report.path) == 1
    assert report.path[0].relationship_type == "IMPORTS"
    assert report.path[0].classification == "observed"
    assert report.status == "complete"

    validations = specialist.validate(report)
    assert validations
    assert {item.status for item in validations} == {STATUS_SUPPORTED}


def test_multi_hop_flow_preserves_the_intermediate_component():
    specialist = build_flow_investigator(
        *read_then_report(
            [PRODUCER_PATH, TRANSFORM_PATH], flow_payload()
        )
    )

    report = specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert report.path_status == "FOUND"
    assert [step.source for step in report.path] == [
        "producer",
        "transform",
    ]
    assert [step.target for step in report.path] == [
        "transform",
        "publisher",
    ]
    envelope = report.to_investigation_report()
    statements = [finding.statement for finding in envelope.findings]
    assert "producer IMPORTS transform [observed]" in statements
    assert "transform IMPORTS publisher [observed]" in statements
    assert not any(
        "producer IMPORTS publisher" in statement
        for statement in statements
    )


def test_agent_to_tool_to_mcp_path_keeps_both_intermediates():
    specialist = build_flow_investigator(
        *read_then_report(
            [COLLECTOR_PATH, MCP_CONFIG_PATH], mcp_payload()
        ),
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )

    report = specialist.investigate(
        MCP_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert report.source == "collector"
    assert report.target == "records"
    assert [step.target for step in report.path] == [
        "fetch_record",
        "records",
    ]
    assert report.path[1].relationship_type == "EXPOSED_BY"
    statements = [
        finding.statement
        for finding in report.to_investigation_report().findings
    ]
    assert "collector IMPORTS fetch_record [observed]" in statements
    assert "fetch_record EXPOSED_BY records [observed]" in statements
    assert not any(
        "collector EXPOSED_BY records" in statement
        for statement in statements
    )


def test_containment_edge_stays_a_containment_edge():
    specialist = build_flow_investigator(
        *read_then_report([PRODUCER_PATH], contains_payload())
    )

    report = specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert report.path[0].relationship_type == "CONTAINS"
    statements = [
        finding.statement
        for finding in report.to_investigation_report().findings
    ]
    assert "producer CONTAINS transform [derived]" in statements


def test_structural_edge_cannot_be_relabelled_as_a_call():
    payload = direct_payload()
    payload["path"][0]["relationship_type"] = "calls"
    payload["relationships"][0]["relationship_type"] = "calls"

    with pytest.raises(Exception):
        FlowDependencyReport(**payload)


def test_conflicting_evidence_preserves_both_candidate_paths():
    specialist = build_flow_investigator(
        *read_then_report(
            [WIRING_DOC_PATH, API_PATH, BRIDGE_PATH],
            conflicting_payload(),
        ),
        bindings=fixture_bindings(FLOW_CONFLICT_ROOT),
    )

    report = specialist.investigate(
        "How does the api reach the renderer?",
        FLOW_REPOSITORY,
        FLOW_COMMIT,
    )

    assert report.path_status == "CONFLICTING"
    assert report.path[0].relationship_type == "DEPENDS_ON"
    assert len(report.alternative_paths) == 1
    alternative = report.alternative_paths[0]
    assert [step.source for step in alternative] == ["api", "bridge"]
    assert [step.target for step in alternative] == [
        "bridge",
        "renderer",
    ]
    paths = {ref.path for ref in report.evidence_refs}
    assert {WIRING_DOC_PATH, API_PATH, BRIDGE_PATH} <= paths
    statements = [
        finding.statement
        for finding in report.to_investigation_report().findings
    ]
    assert "api DEPENDS_ON renderer [conflicting]" in statements
    assert "api IMPORTS bridge [observed]" in statements
    assert "bridge IMPORTS renderer [observed]" in statements

    validations = specialist.validate(report)
    assert {item.status for item in validations} == {STATUS_SUPPORTED}


def test_conflict_fixture_documents_a_direct_edge_the_code_routes():
    docs = (FLOW_CONFLICT_ROOT / "docs" / "wiring.md").read_text()
    api = (FLOW_CONFLICT_ROOT / "src" / "api.py").read_text()

    assert "api → renderer" in docs
    assert "src.bridge" in api
    assert "src.renderer" not in api


# -- no-path semantics: NOT_FOUND vs UNKNOWN (§22 test 4) ---------------


def test_insufficient_evidence_reports_unknown_not_not_found():
    specialist = build_flow_investigator(
        *read_then_report(
            [ALPHA_PATH], no_path_payload(path_status="UNKNOWN")
        ),
        bindings=fixture_bindings(FLOW_DISCONNECTED_ROOT),
    )

    report = specialist.investigate(
        "How does alpha reach beta?", FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert report.path_status == "UNKNOWN"
    assert report.path == []
    assert report.alternative_paths == []
    assert report.uncertainties


def test_open_ended_question_does_not_require_source_target_path():
    payload = no_path_payload(
        path_status="UNKNOWN",
        source=None,
        target=None,
    )
    payload["question"] = "How do agents, tools, and MCP servers interact?"
    payload["summary"] = (
        "The evidence supports individual relationships, but not one "
        "specific end-to-end path."
    )
    report = FlowDependencyReport(**payload)

    assert report.source is None
    assert report.target is None
    assert report.path_status == "UNKNOWN"
    assert report.path == []
    assert report.alternative_paths == []


def test_sufficient_evidence_reports_not_found():
    specialist = build_flow_investigator(
        *read_then_report(
            [ALPHA_PATH], no_path_payload(path_status="NOT_FOUND")
        ),
        bindings=fixture_bindings(FLOW_DISCONNECTED_ROOT),
    )

    report = specialist.investigate(
        "How does alpha reach beta?", FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert report.path_status == "NOT_FOUND"
    assert report.path == []
    assert report.uncertainties
    validations = specialist.validate(report)
    assert {item.status for item in validations} == {STATUS_SUPPORTED}


def test_not_found_and_unknown_are_distinct_answers():
    not_found = FlowDependencyReport(
        **no_path_payload(path_status="NOT_FOUND")
    )
    unknown = FlowDependencyReport(
        **no_path_payload(path_status="UNKNOWN")
    )

    assert not_found.summary != unknown.summary
    assert not_found.uncertainties != unknown.uncertainties
    assert "no path between alpha and beta is established" in (
        not_found.summary
    )


def test_unestablished_path_is_never_presented_as_found():
    with pytest.raises(Exception):
        FlowDependencyReport(
            **no_path_payload(
                path_status="NOT_FOUND",
                path=[
                    {
                        "source": "alpha",
                        "target": "beta",
                        "relationship_type": "IMPORTS",
                        "evidence_refs": [{"path": ALPHA_PATH}],
                        "classification": "observed",
                        "confidence": "high",
                    }
                ],
            )
        )
    with pytest.raises(Exception):
        FlowDependencyReport(
            **no_path_payload(
                path_status="UNKNOWN",
                path=[
                    {
                        "source": "alpha",
                        "target": "beta",
                        "relationship_type": "IMPORTS",
                        "evidence_refs": [{"path": ALPHA_PATH}],
                        "classification": "observed",
                        "confidence": "high",
                    }
                ],
            )
        )


# -- evidence rules on every path step (§22 test 7) ---------------------


def test_path_step_without_evidence_is_rejected():
    payload = direct_payload()
    payload["path"][0]["evidence_refs"] = []

    with pytest.raises(Exception):
        FlowDependencyReport(**payload)


def test_evidence_ref_outside_the_aggregate_is_rejected():
    payload = direct_payload()
    payload["path"][0]["evidence_refs"] = [{"path": "missing.py"}]

    with pytest.raises(Exception):
        FlowDependencyReport(**payload)


def test_observed_relationship_without_evidence_is_rejected():
    payload = direct_payload(
        relationships=[
            {
                "source": "producer",
                "target": "transform",
                "relationship_type": "IMPORTS",
                "evidence_refs": [],
                "classification": "observed",
                "confidence": "high",
            }
        ]
    )

    with pytest.raises(Exception):
        FlowDependencyReport(**payload)


def test_found_requires_a_non_empty_path():
    with pytest.raises(Exception):
        FlowDependencyReport(**direct_payload(path=[]))


def test_found_cannot_carry_competing_candidates():
    payload = flow_payload(
        alternative_paths=[
            [
                {
                    "source": "producer",
                    "target": "transform",
                    "relationship_type": "IMPORTS",
                    "evidence_refs": [{"path": PRODUCER_PATH}],
                    "classification": "observed",
                    "confidence": "high",
                },
                {
                    "source": "transform",
                    "target": "publisher",
                    "relationship_type": "IMPORTS",
                    "evidence_refs": [{"path": TRANSFORM_PATH}],
                    "classification": "observed",
                    "confidence": "high",
                },
            ]
        ]
    )

    with pytest.raises(Exception):
        FlowDependencyReport(**payload)


def test_broken_chain_is_rejected():
    payload = direct_payload(target="publisher")
    payload["source"] = "producer"
    payload["path"] = [
        {
            "source": "producer",
            "target": "transform",
            "relationship_type": "IMPORTS",
            "evidence_refs": [{"path": PRODUCER_PATH}],
            "classification": "observed",
            "confidence": "high",
        },
        {
            "source": "producer",
            "target": "publisher",
            "relationship_type": "IMPORTS",
            "evidence_refs": [{"path": TRANSFORM_PATH}],
            "classification": "observed",
            "confidence": "high",
        },
    ]
    payload["relationships"] = list(payload["path"])
    payload["evidence_refs"] = [
        {"path": PRODUCER_PATH},
        {"path": TRANSFORM_PATH},
    ]

    with pytest.raises(Exception) as excinfo:
        FlowDependencyReport(**payload)
    assert "connected chain" in str(excinfo.value)


def test_path_must_run_between_the_declared_endpoints():
    payload = direct_payload()
    payload["target"] = "publisher"

    with pytest.raises(Exception) as excinfo:
        FlowDependencyReport(**payload)
    assert "does not run from source" in str(excinfo.value)


def test_path_status_vocabulary_is_closed():
    with pytest.raises(Exception):
        FlowDependencyReport(**direct_payload(path_status="MAYBE"))


# -- repository-agnosticism (§22 test 8, §26) ---------------------------


def test_no_product_identifiers_in_specialist_source_or_prompt():
    joined = "\n".join(
        path.read_text() for path in FLOW_SOURCES
    ).lower()

    for identifier in (
        "adop",
        "devops agent",
        "prompt intelligence",
        "data onboarding",
    ):
        assert identifier not in joined


def test_prompt_declares_repository_agnosticism():
    prompt = FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1

    assert "repository-agnostic" in prompt
    assert "assume product names" in prompt
    assert "static evidence only" in prompt


# -- explicit failure semantics (§22 tests 9-10) ------------------------


def test_invalid_structured_output_fails_at_the_agent():
    # Runtime metadata is filled from the request before validation;
    # a substantively invalid payload is still rejected on its own
    # content (nothing is rescued, nothing is invented).
    specialist = build_flow_investigator(
        structured({"summary": "malformed", "status": "unreachable"})
    )

    with pytest.raises(StructuredOutputValidationError):
        specialist.investigate(
            FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
        )


def test_missing_structured_output_raises_the_specialist_error():
    specialist = FlowDependencyInvestigator(agent=plain_agent())

    with pytest.raises(FlowDependencyInvestigatorError):
        specialist.investigate(
            FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
        )


# -- read-only boundary (§22 test 11) -----------------------------------


def test_investigation_does_not_modify_the_repository():
    def digest(root: Path):
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = digest(FLOW_CHAIN_ROOT)
    specialist = build_flow_investigator(
        *read_then_report(
            [PRODUCER_PATH, TRANSFORM_PATH], flow_payload()
        )
    )

    specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    assert digest(FLOW_CHAIN_ROOT) == before


# -- M16/M17 handoff protocol (§22 test 12) -----------------------------


def test_run_result_wraps_the_runner_question_under_the_flow_prompt():
    client = FakeLLMClient(
        *read_then_report(
            [COLLECTOR_PATH, MCP_CONFIG_PATH], mcp_payload()
        )
    )
    specialist = _specialist(
        client, bindings=fixture_bindings(MCP_FIXTURE_ROOT)
    )
    runner_question = (
        "Investigate this plan step and produce your structured "
        "InvestigationReport.\nstep: P1 — Identify the endpoints"
    )

    result = specialist.run_result(runner_question)

    text = _user_text(client)
    assert text.startswith("ROLE")
    assert "QUESTION\n{}".format(runner_question) in text
    assert isinstance(result.structured_output, InvestigationReport)


def test_run_result_projects_path_steps_and_findings():
    specialist = build_flow_investigator(
        *read_then_report(
            [COLLECTOR_PATH, MCP_CONFIG_PATH], mcp_payload()
        ),
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )

    result = specialist.run_result("Investigate step P2")

    envelope = result.structured_output
    assert isinstance(envelope, InvestigationReport)
    statements = [finding.statement for finding in envelope.findings]
    assert "collector IMPORTS fetch_record [observed]" in statements
    assert "fetch_record EXPOSED_BY records [observed]" in statements
    expected = FlowDependencyReport(
        **mcp_payload()
    ).derived_investigation_id()
    assert envelope.investigation_id == expected
    paths = {ref.path for ref in envelope.evidence_refs}
    assert {COLLECTOR_PATH, MCP_CONFIG_PATH} <= paths


def test_run_result_passes_failed_output_through_untouched():
    specialist = FlowDependencyInvestigator(agent=plain_agent())

    result = specialist.run_result("Investigate step P1")

    assert result.structured_output is None  # the runner records it


def test_run_result_conversion_is_pure():
    def run():
        return build_flow_investigator(
            *read_then_report([PRODUCER_PATH], direct_payload())
        ).run_result("Investigate step P1")

    first = run()
    second = run()

    assert (
        first.structured_output.investigation_id
        == second.structured_output.investigation_id
    )
    assert first.structured_output is not second.structured_output


# -- L3: the full M10 → M17 orchestrated run (§22 test 13) --------------


def _execute_flow():
    """Run the whole application path for the agent → tool → MCP flow
    question: planner, three Investigator steps, Analyst, validator."""
    planner = FakeLLMClient(structured(flow_plan_payload()))
    response = planner.complete(build_planning_request(MCP_QUESTION))
    investigation_plan = plan_from_response(response)
    validate_plan(investigation_plan)

    script = (
        read_then_report(
            [COLLECTOR_PATH, MCP_CONFIG_PATH], mcp_endpoint_payload()
        )
        + read_then_report([COLLECTOR_PATH], mcp_leg_one_payload())
        + read_then_report([MCP_CONFIG_PATH], mcp_leg_two_payload())
    )
    specialist = build_flow_investigator(
        *script, bindings=fixture_bindings(MCP_FIXTURE_ROOT)
    )
    analyst = build_analyst(structured(mcp_finding_payload()))
    orchestrator = Orchestrator(specialist, analyst)
    run = orchestrator.execute(flow_task(), investigation_plan)
    return run, specialist, analyst, investigation_plan


def test_flow_end_to_end_completes():
    run, _, _, _ = _execute_flow()

    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert rule(run.validation, "evidence_valid").ok is True
    assert rule(run.validation, "collaboration").ok is True
    assert rule(run.validation, "classification").ok is True
    assert rule(run.validation, "evidence_handoff").ok is True


def test_merged_report_keeps_both_legs_without_collapsing_the_edge():
    run, _, _, _ = _execute_flow()

    statements = [
        finding.statement for finding in run.report.findings
    ]
    assert "collector IMPORTS fetch_record [observed]" in statements
    assert "fetch_record EXPOSED_BY records [observed]" in statements
    assert "collector is the flow's agent endpoint" in statements
    assert not any(
        "collector EXPOSED_BY records" in statement
        for statement in statements
    )
    paths = {ref.path for ref in run.report.evidence_refs}
    assert {COLLECTOR_PATH, MCP_CONFIG_PATH} <= paths
    expected = FlowDependencyReport(
        **mcp_endpoint_payload()
    ).derived_investigation_id()
    assert run.report.investigation_id == expected
    assert run.finding is not None
    assert run.finding.investigation_id == expected


def test_roles_and_steps_follow_the_plan():
    run, _, _, investigation_plan = _execute_flow()

    assigns = [
        event
        for event in run.events
        if event["operation"] == "orchestration.step.assign"
    ]
    roles = [event["agent_role"] for event in assigns]
    assert roles == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    assert all(event["assignment_reason"] for event in assigns)
    assert [step.status for step in investigation_plan.steps] == [
        "completed",
    ] * 4
    assert len(run.agent_runs) == 4
    investigator_results = [
        result
        for result in run.step_results
        if result.agent_role == "INVESTIGATOR"
    ]
    assert all(
        type(result.output).__name__ == "InvestigationReport"
        for result in investigator_results
    )


def test_specialist_traces_carry_the_orchestration_events():
    run, specialist, analyst, _ = _execute_flow()

    trace = specialist.last_trace
    assert trace is specialist.agent.last_trace
    assert trace.orchestration_events
    operations = {
        event["operation"] for event in trace.orchestration_events
    }
    assert "orchestration.step.assign" in operations
    assert "orchestration.agent.start" in operations
    markdown = render_markdown(trace)
    assert "## Orchestration" in markdown
    analyst_ops = {
        event["operation"]
        for event in analyst.last_trace.orchestration_events
    }
    assert "orchestration.step.assign" in analyst_ops


# -- observable execution trace (§22 test 14) ---------------------------


def test_trace_records_the_reads_that_back_the_report():
    specialist = build_flow_investigator(
        *read_then_report(
            [PRODUCER_PATH, TRANSFORM_PATH], flow_payload()
        )
    )

    report = specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    trace = specialist.last_trace
    assert trace is not None
    read_paths = [
        result.get("data", {}).get("path")
        for iteration in trace.iterations
        for result in iteration.tool_results
    ]
    assert PRODUCER_PATH in read_paths
    assert TRANSFORM_PATH in read_paths
    contents = [
        result.get("data", {}).get("content") or ""
        for iteration in trace.iterations
        for result in iteration.tool_results
    ]
    assert any("src.transform" in content for content in contents)
    markdown = render_markdown(trace)
    assert "## Run Summary" in markdown
    assert report.status == "complete"


# -- versioned prompt & vocabulary reuse (§13-§15) ----------------------


def test_investigate_runs_the_versioned_flow_prompt():
    client = FakeLLMClient(
        *read_then_report([PRODUCER_PATH], direct_payload())
    )
    specialist = _specialist(client)

    specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )

    text = _user_text(client)
    assert FLOW_PROMPT_VERSION == "V2"
    assert text.startswith("ROLE")
    assert FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1 in text
    assert "question: {}".format(FLOW_QUESTION) in text
    assert "repository_scope: {}".format(FLOW_REPOSITORY) in text
    assert "commit_scope: {}".format(FLOW_COMMIT) in text
    assert (
        "You investigate repository evidence about flows and "
        "dependencies."
    ) in text


def test_prompt_is_versioned_and_generated_from_the_models():
    prompt = FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1
    flat = " ".join(prompt.split())

    assert "{relationship_kinds}" not in prompt
    assert "{" not in prompt
    for kind in RELATIONSHIP_KINDS:
        assert kind in prompt
    for status in PATH_STATUSES:
        assert status in prompt
    for level in ("observed", "derived", "inferred", "conflicting"):
        assert level in prompt
    for section in (
        "ROLE",
        "RESPONSIBILITIES",
        "INVESTIGATION METHOD",
        "EVIDENCE RULES",
        "FLOW / DEPENDENCY RULES",
        "UNCERTAINTY RULES",
        "OUTPUT",
        "BOUNDARIES",
    ):
        assert section in prompt
    assert "never relabel a structural edge" in flat
    assert (
        "an unestablished path is never presented as a found path"
        in flat
    )
    assert "OPEN-ENDED QUESTION RULE" in prompt
    assert 'path_status="UNKNOWN"' in flat
    assert 'Never set path_status="FOUND" with an empty path' in flat
    assert prompt != ARCHITECTURE_INVESTIGATOR_PROMPT_V1


def test_direct_prompt_builder_adds_the_task_block():
    prompt = build_flow_dependency_investigator_prompt(
        question=FLOW_QUESTION,
        repository_scope=FLOW_REPOSITORY,
        commit_scope=FLOW_COMMIT,
    )

    assert prompt.startswith("ROLE")
    assert "TASK" in prompt
    assert "question: {}".format(FLOW_QUESTION) in prompt


def test_tool_inventory_is_the_existing_read_only_list():
    assert FLOW_DEPENDENCY_TOOLS is ARCHITECTURE_TOOLS
    assert len(FLOW_DEPENDENCY_TOOLS) == 4
    assert tuple(sorted(FLOW_DEPENDENCY_TOOLS)) == (
        FLOW_DEPENDENCY_TOOLS
    )
    specialist = build_flow_investigator(*[])
    assert specialist.tools == list(FLOW_DEPENDENCY_TOOLS)


def test_path_status_vocabulary_is_reused_not_forked():
    literal_args = FlowDependencyReport.model_fields[
        "path_status"
    ].annotation.__args__

    assert tuple(literal_args) == PATH_STATUSES
    assert PATH_STATUSES == (
        "FOUND",
        "NOT_FOUND",
        "UNKNOWN",
        "CONFLICTING",
    )


def test_relationship_vocabulary_is_the_engineering_graph():
    import typing

    step_model = typing.get_args(
        FlowDependencyReport.model_fields["path"].annotation
    )[0]
    literal_args = step_model.model_fields[
        "relationship_type"
    ].annotation.__args__

    assert RELATIONSHIP_TYPES is RELATIONSHIP_KINDS
    assert tuple(literal_args) == RELATIONSHIP_KINDS


def test_statements_and_id_are_deterministic():
    first = FlowDependencyReport(**direct_payload())
    second = FlowDependencyReport(**direct_payload())

    assert first.derived_investigation_id().startswith("inv-flow-")
    assert (
        first.derived_investigation_id()
        == second.derived_investigation_id()
    )
    assert (
        first.to_investigation_report().investigation_id
        == first.derived_investigation_id()
    )


# -- evidence validation for direct mode (§24) --------------------------


def test_validate_accepts_evidence_this_run_read():
    specialist = build_flow_investigator(
        *read_then_report([PRODUCER_PATH], direct_payload())
    )

    report = specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )
    results = specialist.validate(report)

    assert results
    assert {item.status for item in results} == {STATUS_SUPPORTED}


def test_validate_rejects_evidence_never_read():
    specialist = build_flow_investigator(
        *read_then_report([PRODUCER_PATH], direct_payload())
    )
    report = specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )
    report.findings[0].evidence_refs[0].path = "never/read.py"

    results = specialist.validate(report)

    unsupported = [
        item for item in results if item.status == STATUS_UNSUPPORTED
    ]
    assert [item.claim.statement for item in unsupported] == [
        "producer reaches transform directly"
    ]
    assert unsupported[0].code == "CLAIM_EVIDENCE_INVALID"


def test_validate_needs_a_run_to_validate_against():
    specialist = build_flow_investigator(*[])
    report = FlowDependencyReport(**direct_payload())

    with pytest.raises(FlowDependencyInvestigatorError):
        specialist.validate(report)


# -- boundaries: composition, no second infrastructure (§12, §50) -------


def test_last_trace_delegates_to_the_agent():
    specialist = build_flow_investigator(
        *read_then_report([PRODUCER_PATH], direct_payload())
    )

    assert specialist.last_trace is None
    specialist.investigate(
        FLOW_QUESTION, FLOW_REPOSITORY, FLOW_COMMIT
    )
    assert specialist.last_trace is specialist.agent.last_trace


def test_orchestrator_wiring_is_one_obvious_argument():
    specialist = build_flow_investigator(*[])
    orchestrator = Orchestrator(specialist, specialist)  # type smoke

    assert orchestrator._investigator is specialist


def test_specialist_is_not_an_architecture_subclass():
    from repo_intelligence.agents import ArchitectureInvestigator

    assert not issubclass(
        FlowDependencyInvestigator, ArchitectureInvestigator
    )


def test_specialist_owns_no_infrastructure():
    import repo_intelligence.agents.flow_dependency_investigator as module

    text = Path(module.__file__).read_text().lower()
    for banned in (
        "def _tool_loop",
        "while true",
        "plan_from_response",
        "memorymanager",
        "contextengine",
        "repo_intelligence.orchestration",
    ):
        assert banned not in text


def test_models_refuse_unknown_values():
    with pytest.raises(Exception):
        FlowDependencyReport(
            **direct_payload(path_status="PARTIAL")
        )
    with pytest.raises(Exception):
        FlowDependencyReport(
            **direct_payload(
                path=[
                    {
                        "source": "producer",
                        "target": "transform",
                        "relationship_type": "USES",
                        "evidence_refs": [{"path": PRODUCER_PATH}],
                        "classification": "observed",
                        "confidence": "high",
                    }
                ]
            )
        )


def test_plan_shape_matches_the_flow_question():
    plan = flow_plan()

    assert [step.id for step in plan.steps] == [
        "P1",
        "P2",
        "P3",
        "P4",
    ]
    assert plan.objective == MCP_QUESTION
