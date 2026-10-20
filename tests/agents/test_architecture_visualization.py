"""Levels 1 + 2 — the Architecture Visualization Agent itself (spec §26).

Example-style scenarios a developer can read as the capability's
specification: bounded diagram types, consumption of both typed
reports, the agent/tool/MCP chain, the no-path case, unsupported and
conflicting inputs, the M16 handoff protocol, deterministic six-rule
validation, evidence scope, deterministic rendering, and artifact
persistence — all over scripted doubles and the foreign arch_mcp
fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from architecture_doubles import (
    COMMIT,
    REPOSITORY,
    FakeLLMClient,
    build_investigator,
    fixture_bindings,
    rule,
    structured,
)
from pydantic import ValidationError

from repo_intelligence.agent import Agent
from repo_intelligence.agents import (
    ARCHITECTURE_VISUALIZATION_PROMPT_V1,
    DIAGRAM_CHECKS,
    DIAGRAM_TYPES,
    PROMPT_VERSION,
    RUN_SUMMARY_FILENAME,
    SPEC_FILENAME,
    SVG_UNAVAILABLE_NOTICE,
    VISUALIZATION_PROMPT_VERSION,
    VISUALIZATION_TOOLS,
    ArchitectureVisualizationAgent,
    ArchitectureVisualizationAgentError,
    DiagramScope,
    DiagramSpec,
    RunSummaryContext,
    VisualizationArtifactError,
    VisualizationRequest,
    build_architecture_visualization_prompt,
    validate_diagram_spec,
    verify_artifacts,
    write_visualization_artifacts,
)
from repo_intelligence.agents.models.architecture_report import (
    ArchitectureReport,
    component_statement,
    relationship_statement,
)
from repo_intelligence.agents.models.flow_dependency_report import (
    FlowDependencyReport,
)
from repo_intelligence.evidence.validator import gather_evidence
from repo_intelligence.graph.exporters import graphviz_available, render_mermaid
from repo_intelligence.multiagent.contracts import (
    ArchitectureFinding,
    ValidationResult,
    report_json,
)
from repo_intelligence.multiagent.permissions import restricted_dispatcher

from flow_doubles import ALPHA_PATH, no_path_payload

from viz_doubles import (
    FETCH_RECORD_PATH,
    MCP_COLLECTOR_PATH,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    VIZ_QUESTION,
    build_visualization_agent,
    invalid_edge_spec_payload,
    out_of_scope_spec_payload,
    read_paths,
    spec_payload,
    viz_arch_payload,
    viz_flow_payload,
    viz_request,
)

FLOW_REPOSITORY = "example-flow-repository"


def _visualize(payload=None, request=None):
    agent = build_visualization_agent(
        structured(payload if payload is not None else spec_payload())
    )
    return agent, agent.visualize(request or viz_request())


def _analyst_question(envelope) -> str:
    """The label block the runner builds for an analyst step (the
    contract ``parse_handoff_context`` reads), with the typed report
    embedded exactly as the runner embeds it."""
    return (
        "Analyze the Investigator's typed report and produce your "
        "structured ArchitectureFinding.\n"
        "question: {}\n"
        "repository_scope: {}\n"
        "commit_scope: {}\n"
        "investigation_id: {}\n"
        "Use ONLY this report's evidence (no independent "
        "investigation):\n{}".format(
            VIZ_QUESTION,
            REPOSITORY,
            COMMIT,
            envelope.investigation_id,
            report_json(envelope),
        )
    )


# -- basic visualization (spec §26: architecture, flow) -----------------


def test_architecture_diagram_is_produced():
    request = viz_request(diagram_type="ARCHITECTURE")
    agent, spec = _visualize(
        spec_payload(diagram_type="ARCHITECTURE"), request
    )

    assert spec.diagram_type == "ARCHITECTURE"
    assert spec.title
    assert [node.id for node in spec.nodes] == [
        "collector",
        "fetch_record",
        "records",
    ]
    assert [(edge.source, edge.relationship_type, edge.target) for edge in spec.edges] == [
        ("collector", "DECLARES", "fetch_record"),
        ("fetch_record", "EXPOSED_BY", "records"),
    ]
    assert agent.last_spec is spec
    assert agent.validate(spec, request).status == "passed"


def test_flow_diagram_is_produced():
    request = VisualizationRequest(
        question=VIZ_QUESTION,
        diagram_type="FLOW",
        flow_report=FlowDependencyReport(**viz_flow_payload()),
    )
    payload = spec_payload(
        diagram_type="FLOW",
        edges=[
            {
                "id": "collector-imports-fetch_record",
                "source": "collector",
                "target": "fetch_record",
                "relationship_type": "IMPORTS",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
            },
            {
                "id": "fetch_record-exposed_by-records",
                "source": "fetch_record",
                "target": "records",
                "relationship_type": "EXPOSED_BY",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": MCP_CONFIG_PATH}],
            },
        ],
        evidence_refs=[
            {"path": MCP_COLLECTOR_PATH},
            {"path": MCP_CONFIG_PATH},
        ],
    )
    # The flow report evidences the hops, not the tool file itself —
    # an element it never evidenced stays honest (unknown, no refs).
    payload["nodes"][1]["classification"] = "unknown"
    payload["nodes"][1]["confidence"] = "unknown"
    payload["nodes"][1]["evidence_refs"] = []

    agent, spec = _visualize(payload, request)

    assert spec.diagram_type == "FLOW"
    assert spec.nodes[1].evidence_refs == []
    assert spec.nodes[1].classification == "unknown"
    assert agent.validate(spec, request).status == "passed"


def test_not_found_flow_yields_an_edge_free_diagram():
    report = FlowDependencyReport(**no_path_payload())
    request = VisualizationRequest(
        question=report.question,
        diagram_type="FLOW",
        flow_report=report,
    )
    payload = spec_payload(
        diagram_type="FLOW",
        title="alpha and beta share no established path",
        question=report.question,
        nodes=[
            {
                "id": "alpha",
                "label": "alpha",
                "node_type": "module",
                "classification": "observed",
                "confidence": "high",
                "evidence_refs": [{"path": ALPHA_PATH}],
            },
            {
                "id": "beta",
                "label": "beta",
                "node_type": "module",
                "classification": "unknown",
                "confidence": "unknown",
            },
        ],
        edges=[],
        groups=[],
        evidence_refs=[{"path": ALPHA_PATH}],
        uncertainties=list(report.uncertainties),
        status="complete",
    )

    agent, spec = _visualize(payload, request)

    assert spec.edges == []
    assert [node.id for node in spec.nodes] == ["alpha", "beta"]
    assert spec.uncertainties == report.uncertainties
    assert agent.validate(spec, request).status == "passed"


# -- input consumption (spec §26) ---------------------------------------


def test_request_consumes_both_investigator_reports():
    request = viz_request()

    assert request.scope_label() == (REPOSITORY, COMMIT)
    assert [type(report).__name__ for report in request.flow_reports()] == [
        "FlowDependencyReport"
    ]

    scope = DiagramScope.from_request(request)
    assert set(scope.node_ids) == {"collector", "fetch_record", "records"}
    assert ( "collector", "fetch_record", "DECLARES") in scope.edges
    assert ("collector", "fetch_record", "IMPORTS") in scope.edges
    assert (
        "fetch_record",
        "records",
        "EXPOSED_BY",
    ) in scope.edges
    assert scope.requested_diagram_type == "AGENT_TOOL_MCP"
    assert len(scope.evidence_keys) == 3


def test_additional_flow_reports_are_consumed():
    extra = FlowDependencyReport(
        **viz_flow_payload(question="Which component exposes the records server?")
    )
    request = viz_request(additional_flow_reports=[extra])

    assert len(request.flow_reports()) == 2
    prompt = build_architecture_visualization_prompt(request)
    assert "additional flow reports:" in prompt
    # one pinned state: the extra report shares it, or the request
    # would never be constructed
    assert request.scope_label() == (REPOSITORY, COMMIT)


def test_request_rejects_reports_from_different_scopes():
    with pytest.raises(ValidationError, match="disagree on scope"):
        VisualizationRequest(
            question=VIZ_QUESTION,
            architecture_report=ArchitectureReport(**viz_arch_payload()),
            flow_report=FlowDependencyReport(**viz_flow_payload(
                repository_scope=FLOW_REPOSITORY
            )),
        )


def test_request_needs_at_least_one_report():
    with pytest.raises(ValidationError, match="at least one"):
        VisualizationRequest(question=VIZ_QUESTION)


def test_flow_diagram_without_a_flow_report_cannot_proceed():
    with pytest.raises(ValidationError, match="FLOW diagram needs"):
        VisualizationRequest(
            question=VIZ_QUESTION,
            diagram_type="FLOW",
            architecture_report=ArchitectureReport(**viz_arch_payload()),
        )


def test_architecture_diagram_without_an_architecture_report_cannot_proceed():
    with pytest.raises(ValidationError, match="ARCHITECTURE diagram needs"):
        VisualizationRequest(
            question=VIZ_QUESTION,
            diagram_type="ARCHITECTURE",
            flow_report=FlowDependencyReport(**viz_flow_payload()),
        )


def test_prompt_embeds_the_reports_and_scopes():
    prompt = build_architecture_visualization_prompt(viz_request())

    assert prompt.startswith(ARCHITECTURE_VISUALIZATION_PROMPT_V1)
    assert "TASK\nquestion: {}".format(VIZ_QUESTION) in prompt
    assert "diagram_type: AGENT_TOOL_MCP" in prompt
    assert "repository_scope: {}".format(REPOSITORY) in prompt
    assert "commit_scope: {}".format(COMMIT) in prompt
    assert "ARCHITECTURE INVESTIGATION" in prompt
    assert "FLOW INVESTIGATION" in prompt
    assert '"id": "collector"' in prompt
    assert '"path_status": "FOUND"' in prompt


def test_prompt_is_versioned_and_repository_agnostic():
    assert VISUALIZATION_PROMPT_VERSION == "V1"
    assert PROMPT_VERSION == "V1"  # the architecture prompt's own alias
    prompt = ARCHITECTURE_VISUALIZATION_PROMPT_V1

    for header in ("ROLE", "RESPONSIBILITIES", "RULES", "OUT", "BOUNDARIES"):
        assert header in prompt
    assert "ARCHITECTURE, FLOW, COMPONENT, AGENT_TOOL_MCP" in prompt
    assert "CONTAINS, IMPORTS, DEPENDS_ON" in prompt
    assert "observed, derived, inferred, conflicting, unknown" in prompt
    assert "ADOP" not in prompt
    # every vocabulary placeholder was substituted — .format() would
    # have raised otherwise, but the assertion documents the contract
    assert "{diagram_types}" not in prompt


def test_prompt_forbids_shortened_ids_with_a_few_shot_example():
    # live runs sometimes shortened report ids (bare name instead of
    # the full artifact path) and scope_validity rejected the diagram;
    # the prompt now carries an explicit copy-verbatim rule + example
    prompt = ARCHITECTURE_VISUALIZATION_PROMPT_V1

    assert (
        "never shorten, rename, or \"clean up\" an id taken from the"
        in prompt
    )
    assert "IDs are copied verbatim, never shortened" in prompt
    assert "byte-identical" in prompt
    assert (
        '"artifact:agent_definition:internal/agents/dag-agent.md"'
        in prompt
    )
    assert '"dag-agent", "dag-agent.md"' in prompt
    assert '"artifact:file:src/payments/ledger_client.py"' in prompt
    assert '"ledger_client.py"' in prompt
    # still repository-agnostic: the examples are hypothetical and the
    # demo's banned-word checks stay satisfied
    assert "ADOP" not in prompt
    assert "orchestrat" not in prompt.lower()
    assert "planner" not in prompt.lower()


def test_the_diagram_taxonomy_is_small_and_bounded():
    # spec §5: deliberately small; DATA_FLOW is absent on purpose —
    # the canonical relationship vocabulary has no data-flow kind
    assert DIAGRAM_TYPES == (
        "ARCHITECTURE",
        "FLOW",
        "COMPONENT",
        "AGENT_TOOL_MCP",
    )
    assert "DATA_FLOW" not in DIAGRAM_TYPES


def test_the_agent_has_no_repository_tools():
    assert VISUALIZATION_TOOLS == ()
    agent = build_visualization_agent(structured(spec_payload()))
    assert agent.tools == []


# -- agent behavior ------------------------------------------------------


def test_the_envelope_comes_from_the_request_not_the_model():
    payload = spec_payload(
        question="a question the model wrote",
        repository_scope="model-invented-repository",
        commit_scope="f" * 40,
    )
    agent, spec = _visualize(payload, viz_request())

    assert spec.question == VIZ_QUESTION
    assert spec.repository_scope == REPOSITORY
    assert spec.commit_scope == COMMIT


def test_wrong_structured_output_is_an_explicit_failure():
    plain = Agent(
        FakeLLMClient(structured({"not": "a diagram"})),
        tool_dispatcher=restricted_dispatcher("analyst"),
    )
    specialist = ArchitectureVisualizationAgent(agent=plain)

    with pytest.raises(
        ArchitectureVisualizationAgentError, match="DiagramSpec"
    ):
        specialist.visualize(viz_request())


def test_agent_tool_mcp_chain_is_represented():
    _, spec = _visualize()
    mermaid = render_mermaid(spec.to_diagram_projection())

    assert [node.node_type for node in spec.nodes] == [
        "agent",
        "tool",
        "mcp_server",
    ]
    assert [edge.relationship_type for edge in spec.edges] == [
        "DECLARES",
        "EXPOSED_BY",
    ]
    assert 'node_001["Collector"]' in mermaid
    assert "node_001 -->|DECLARES| node_002" in mermaid
    assert "node_002 -->|EXPOSED_BY| node_003" in mermaid
    assert [entry.term for entry in spec.legend] == [
        "agent",
        "EXPOSED_BY",
    ]


def test_uncertainties_and_status_survive_visualization():
    _, spec = _visualize()

    assert spec.status == "complete"
    assert spec.uncertainties == [
        "the wire format between the tool and the MCP server is not "
        "established"
    ]


# -- the M16/M17 handoff protocol (spec §20) -----------------------------


def test_run_result_projects_the_spec_onto_the_m16_finding():
    envelope = ArchitectureReport(**viz_arch_payload()).to_investigation_report()
    agent = build_visualization_agent(structured(spec_payload()))

    result = agent.run_result(_analyst_question(envelope))
    finding = result.structured_output

    assert isinstance(finding, ArchitectureFinding)
    assert finding.investigation_id == envelope.investigation_id
    assert finding.repository_scope == REPOSITORY
    assert finding.commit_scope == COMMIT
    assert finding.question == VIZ_QUESTION
    statements = [conclusion.statement for conclusion in finding.conclusions]
    assert "Component collector (agent) [observed]" in statements
    assert "Component records (mcp_server) [observed]" in statements
    assert (
        relationship_statement(
            ArchitectureReport(**viz_arch_payload()).relationships[0]
        )
        in statements
    )
    assert finding.open_questions == spec_payload()["uncertainties"]
    assert agent.last_spec is not None
    assert agent.last_spec.diagram_type == "AGENT_TOOL_MCP"


def test_the_envelope_comes_from_the_question_not_the_model():
    envelope = ArchitectureReport(**viz_arch_payload()).to_investigation_report()
    payload = spec_payload(
        question="model text",
        repository_scope="model-scope",
        commit_scope="0" * 40,
    )
    agent = build_visualization_agent(structured(payload))

    finding = agent.run_result(_analyst_question(envelope)).structured_output

    assert finding.question == VIZ_QUESTION
    assert finding.repository_scope == REPOSITORY
    assert finding.commit_scope == COMMIT


def test_missing_handoff_labels_fail_explicitly():
    agent = build_visualization_agent(structured(spec_payload()))

    with pytest.raises(
        ArchitectureVisualizationAgentError, match="missing label"
    ):
        agent.run_result("a question with no envelope labels")


def test_wrong_output_passes_through_untouched():
    plain = Agent(
        FakeLLMClient(structured({"not": "a diagram"})),
        tool_dispatcher=restricted_dispatcher("analyst"),
    )
    specialist = ArchitectureVisualizationAgent(agent=plain)

    result = specialist.run_result(
        _analyst_question(
            ArchitectureReport(**viz_arch_payload()).to_investigation_report()
        )
    )

    assert result.structured_output is None
    assert specialist.last_spec is None


# -- deterministic validation (spec §14) ---------------------------------


def test_scope_from_reports_covers_components_and_paths():
    scope = DiagramScope.from_reports(
        architecture_report=ArchitectureReport(**viz_arch_payload()),
        flow_reports=[FlowDependencyReport(**viz_flow_payload())],
    )

    assert set(scope.node_ids) == {"collector", "fetch_record", "records"}
    assert len(scope.edges) == 3
    assert {key[0] for key in scope.evidence_keys} <= {
        MCP_COLLECTOR_PATH,
        FETCH_RECORD_PATH,
        MCP_CONFIG_PATH,
    }


def test_scope_from_investigation_report_reads_statement_forms():
    envelope = ArchitectureReport(**viz_arch_payload()).to_investigation_report()

    scope = DiagramScope.from_investigation_report(envelope)

    assert set(scope.node_ids) == {"collector", "fetch_record", "records"}
    assert (
        "collector",
        "fetch_record",
        "DECLARES",
    ) in scope.edges
    assert (
        "fetch_record",
        "records",
        "EXPOSED_BY",
    ) in scope.edges
    assert len(scope.evidence_keys) == 3


def test_validate_passes_for_a_report_backed_diagram():
    request = viz_request()
    spec = DiagramSpec(**spec_payload())

    result = validate_diagram_spec(spec, scope=DiagramScope.from_request(request))

    assert isinstance(result, ValidationResult)
    assert [check.rule for check in result.checks] == list(DIAGRAM_CHECKS)
    assert result.status == "passed"
    assert all(check.ok for check in result.checks)


def test_out_of_scope_component_is_rejected():
    request = viz_request()
    spec = DiagramSpec(**out_of_scope_spec_payload())

    result = validate_diagram_spec(spec, scope=DiagramScope.from_request(request))

    check = rule(result, "scope_validity")
    assert result.status == "failed"
    assert check.ok is False
    assert "notes_store" in check.detail


def test_invented_relationship_is_rejected():
    request = viz_request()
    spec = DiagramSpec(**invalid_edge_spec_payload())

    result = validate_diagram_spec(spec, scope=DiagramScope.from_request(request))

    check = rule(result, "scope_validity")
    assert result.status == "failed"
    assert "not established by the investigation" in check.detail


def test_missing_scope_reports_structural_rules_only():
    result = validate_diagram_spec(DiagramSpec(**spec_payload()))

    check = rule(result, "scope_validity")
    assert check.ok is True
    assert "no investigation scope supplied" in check.detail


def test_evidence_scope_narrows_what_the_diagram_may_cite():
    request = viz_request(
        evidence_scope=[{"path": MCP_COLLECTOR_PATH}]
    )
    spec = DiagramSpec(**spec_payload())

    result = validate_diagram_spec(spec, scope=DiagramScope.from_request(request))

    check = rule(result, "evidence_validity")
    assert result.status == "failed"
    assert check.ok is False
    assert "not carried by the investigation" in check.detail


def test_inspected_evidence_from_the_investigator_trace_is_required():
    request = viz_request()
    scope = DiagramScope.from_request(request)
    spec = DiagramSpec(**spec_payload())

    complete = build_investigator(
        read_paths(MCP_COLLECTOR_PATH, FETCH_RECORD_PATH, MCP_CONFIG_PATH),
        structured(viz_arch_payload()),
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )
    complete.investigate(VIZ_QUESTION, REPOSITORY, COMMIT)
    passed = validate_diagram_spec(
        spec, scope=scope, evidence=gather_evidence(complete.last_trace)
    )
    assert passed.status == "passed"

    partial = build_investigator(
        read_paths(MCP_COLLECTOR_PATH, MCP_CONFIG_PATH),
        structured(viz_arch_payload()),
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )
    partial.investigate(VIZ_QUESTION, REPOSITORY, COMMIT)
    failed = validate_diagram_spec(
        spec, scope=scope, evidence=gather_evidence(partial.last_trace)
    )
    check = rule(failed, "evidence_validity")
    assert failed.status == "failed"
    assert "not supported by inspected evidence" in check.detail
    assert FETCH_RECORD_PATH in check.detail


def test_the_visualization_agents_own_trace_is_never_an_evidence_authority():
    agent, spec = _visualize()
    own_evidence = gather_evidence(agent.last_trace)

    result = agent.validate(
        spec, viz_request(), evidence=own_evidence
    )

    assert result.status == "failed"
    assert rule(result, "evidence_validity").ok is False


# -- unsupported and malformed input (spec §26) ---------------------------


def test_unsupported_diagram_type_is_rejected():
    with pytest.raises(ValidationError):
        DiagramSpec(**spec_payload(diagram_type="DATA_FLOW"))


def test_unknown_relationship_type_is_rejected():
    payload = spec_payload()
    payload["edges"][0]["relationship_type"] = "FLOWS_TO"
    with pytest.raises(ValidationError, match="FLOWS_TO"):
        DiagramSpec(**payload)


def test_duplicate_node_ids_are_rejected():
    payload = spec_payload()
    payload["nodes"].append(dict(payload["nodes"][0]))
    with pytest.raises(ValidationError, match="duplicate node id"):
        DiagramSpec(**payload)


def test_observed_element_without_evidence_is_rejected():
    payload = spec_payload()
    payload["nodes"][0]["evidence_refs"] = []
    with pytest.raises(ValidationError, match="observed node"):
        DiagramSpec(**payload)


def test_edge_endpoint_must_be_a_node():
    payload = spec_payload()
    payload["nodes"] = [
        node for node in payload["nodes"] if node["id"] != "fetch_record"
    ]
    with pytest.raises(ValidationError, match="has no node"):
        DiagramSpec(**payload)


def test_group_member_must_be_a_node():
    payload = spec_payload()
    payload["groups"][0]["node_ids"] = ["collector", "ghost"]
    with pytest.raises(ValidationError, match="has no node"):
        DiagramSpec(**payload)


def test_element_evidence_is_added_to_the_aggregate_deterministically():
    payload = spec_payload()
    missing_ref = {
        "path": MCP_COLLECTOR_PATH,
        "line_start": 1,
        "line_end": 4,
    }
    payload["nodes"][0]["evidence_refs"] = [missing_ref]
    payload["evidence_refs"] = [
        ref for ref in payload["evidence_refs"]
        if not (
            ref["path"] == missing_ref["path"]
            and ref.get("line_start") == missing_ref["line_start"]
            and ref.get("line_end") == missing_ref["line_end"]
        )
    ]

    spec = DiagramSpec(**payload)

    assert any(
        (ref.path, ref.line_start, ref.line_end)
        == (MCP_COLLECTOR_PATH, 1, 4)
        for ref in spec.evidence_refs
    )


def test_edge_evidence_is_added_to_the_aggregate_deterministically():
    payload = spec_payload()
    missing_ref = {
        "path": ".claude/commands/onboard-workflow.md",
        "line_start": 1,
        "line_end": 20,
    }
    payload["edges"][0]["evidence_refs"] = [missing_ref]
    payload["evidence_refs"] = [
        ref for ref in payload["evidence_refs"]
        if not (
            ref["path"] == missing_ref["path"]
            and ref.get("line_start") == missing_ref["line_start"]
            and ref.get("line_end") == missing_ref["line_end"]
        )
    ]

    spec = DiagramSpec(**payload)

    assert any(
        (ref.path, ref.line_start, ref.line_end)
        == (missing_ref["path"], 1, 20)
        for ref in spec.evidence_refs
    )


def test_legend_term_must_be_a_canonical_vocabulary_member():
    payload = spec_payload()
    payload["legend"][0]["term"] = "CONNECTS"
    with pytest.raises(ValidationError, match="not a canonical"):
        DiagramSpec(**payload)


def test_unsupported_report_status_is_rejected():
    with pytest.raises(ValidationError):
        DiagramSpec(**spec_payload(status="verified"))


# -- conflicting and unknown input (spec §26) -----------------------------


def test_conflicting_classification_is_carried_not_flattened():
    payload = spec_payload()
    payload["nodes"][2]["classification"] = "conflicting"
    payload["nodes"][2]["evidence_refs"] = [
        {"path": MCP_CONFIG_PATH},
        {"path": MCP_COLLECTOR_PATH},
    ]
    spec = DiagramSpec(**payload)

    result = validate_diagram_spec(
        spec, scope=DiagramScope.from_request(viz_request())
    )
    assert result.status == "passed"
    assert rule(result, "classification_validity").ok is True
    assert spec.nodes[2].classification == "conflicting"


def test_unknown_elements_are_allowed_and_honest():
    payload = spec_payload(
        status="incomplete",
        title="partial visualization",
        uncertainties=["the MCP server's transport is unestablished"],
    )
    spec = DiagramSpec(**payload)

    assert spec.status == "incomplete"
    assert spec.nodes[0].confidence in ("high", "unknown")
    result = validate_diagram_spec(
        spec, scope=DiagramScope.from_request(viz_request())
    )
    assert result.status == "passed"


def test_ref_less_elements_become_open_questions_not_conclusions():
    payload = spec_payload()
    payload["nodes"][1]["evidence_refs"] = []
    payload["nodes"][1]["classification"] = "unknown"
    spec = DiagramSpec(**payload)

    finding = spec.to_finding()
    statements = [conclusion.statement for conclusion in finding.conclusions]

    assert not any(
        statement.startswith("Component fetch_record")
        for statement in statements
    )
    assert any(
        "diagram node 'fetch_record' carries no evidence reference"
        in question
        for question in finding.open_questions
    )
    assert finding.open_questions[: len(payload["uncertainties"])] == (
        payload["uncertainties"]
    )


def test_finding_statements_reuse_the_investigator_statement_forms():
    report = ArchitectureReport(**viz_arch_payload())
    spec = DiagramSpec(**spec_payload())

    finding = spec.to_finding()
    statements = [conclusion.statement for conclusion in finding.conclusions]

    # same statement form, less detail: a diagram node carries a label,
    # not the report's responsibility prose
    assert component_statement(report.components[0].copy(update={"responsibility": ""})) in statements
    assert relationship_statement(report.relationships[0]) in statements
    assert all(" [" in statement for statement in statements)


# -- deterministic rendering (spec §19) -----------------------------------


def test_same_spec_renders_byte_identical_mermaid():
    spec = DiagramSpec(**spec_payload())
    other = DiagramSpec(**spec_payload())

    first = render_mermaid(spec.to_diagram_projection())
    second = render_mermaid(other.to_diagram_projection())

    assert first == second
    assert first.startswith("flowchart TD\n")


def test_mermaid_comes_from_the_spec_not_the_model():
    spec = DiagramSpec(**spec_payload())

    mermaid = render_mermaid(spec.to_diagram_projection())

    assert "|DECLARES|" in mermaid
    assert "|EXPOSED_BY|" in mermaid
    # no model-authored rendering: only the projection's own content
    assert "graph TD" not in mermaid


# -- artifact persistence (spec §17, §18, §25) -----------------------------


def _write(tmp_path, spec=None, validation=None, context=None, directory=None):
    agent, produced = _visualize()
    spec = spec if spec is not None else produced
    request = viz_request()
    validation = validation or agent.validate(spec, request)
    return write_visualization_artifacts(
        spec,
        validation,
        context=context
        or RunSummaryContext(
            architecture_report=ArchitectureReport(**viz_arch_payload()),
            flow_report=FlowDependencyReport(**viz_flow_payload()),
        ),
        directory=str(directory or (tmp_path / "architecture_visualization")),
    )


def test_four_artifacts_are_persisted(tmp_path):
    artifacts = _write(tmp_path)

    statuses = {status.name: status for status in verify_artifacts(artifacts)}
    assert statuses[SPEC_FILENAME].exists and statuses[SPEC_FILENAME].bytes > 0
    assert statuses["architecture.mmd"].exists
    assert statuses[RUN_SUMMARY_FILENAME].exists
    if graphviz_available():
        assert statuses["architecture.svg"].exists
    else:
        assert not statuses["architecture.svg"].exists
        assert statuses["architecture.svg"].note == SVG_UNAVAILABLE_NOTICE

    document = json.loads(Path(artifacts.spec).read_text(encoding="utf-8"))
    assert list(document)[0] == "schema_version"
    assert document["schema_version"] == "diagram_spec.v1"
    assert DiagramSpec.from_document(document) == DiagramSpec(
        **spec_payload()
    )

    mermaid = Path(artifacts.mermaid).read_text(encoding="utf-8")
    assert mermaid == render_mermaid(
        DiagramSpec(**spec_payload()).to_diagram_projection()
    )


def test_mermaid_and_spec_agree_from_the_same_execution(tmp_path):
    artifacts = _write(tmp_path)
    document = json.loads(Path(artifacts.spec).read_text(encoding="utf-8"))
    spec = DiagramSpec.from_document(document)
    mermaid = Path(artifacts.mermaid).read_text(encoding="utf-8")

    assert mermaid == render_mermaid(spec.to_diagram_projection())


def test_run_summary_describes_this_run(tmp_path):
    artifacts = _write(
        tmp_path,
        context=RunSummaryContext(
            architecture_report=ArchitectureReport(**viz_arch_payload()),
            flow_report=FlowDependencyReport(**viz_flow_payload()),
            evidence_notes=["investigator traces: arch + flow runs"],
            m17_agents=["investigator", "analyst"],
            m17_status="passed",
            trace_location=str(tmp_path / "runs"),
        ),
    )
    summary = Path(artifacts.run_summary).read_text(encoding="utf-8")

    headings = [line for line in summary.splitlines() if line.startswith("#")]
    assert headings == [
        "# Architecture Visualization Run",
        "## User Question",
        "## Architecture Investigator",
        "## Flow / Dependency Investigator",
        "## Architecture Visualization Agent",
        "## Validation",
        "## Rendering",
        "## Evidence",
        "## Uncertainties",
        "## M17 Execution",
        "## Trace",
    ]
    assert VIZ_QUESTION in summary
    assert "Flow: collector -> fetch_record -> records" in summary
    assert "Status: passed" in summary
    assert "node_validity: PASS" in summary
    assert "investigator traces: arch + flow runs" in summary
    assert "Trace/artifact location: {}".format(tmp_path / "runs") in summary
    if not artifacts.svg_available:
        assert summary.count(SVG_UNAVAILABLE_NOTICE) >= 1


def test_artifacts_are_reproducible_from_the_same_spec(tmp_path):
    first = _write(tmp_path, directory=tmp_path / "first")
    second = _write(tmp_path, directory=tmp_path / "second")

    assert (
        Path(first.mermaid).read_text(encoding="utf-8")
        == Path(second.mermaid).read_text(encoding="utf-8")
    )
    assert (
        Path(first.spec).read_text(encoding="utf-8")
        == Path(second.spec).read_text(encoding="utf-8")
    )


def test_default_artifact_directory_is_project_rooted(
    tmp_path, monkeypatch
):
    """The default artifact location resolves from the project root
    (never the process CWD); an explicit directory is used as given."""
    from repo_intelligence.agents import visualization_artifacts as module

    monkeypatch.setattr(
        module, "default_artifacts_root", lambda: tmp_path / "artifacts"
    )
    resolved = module.resolve_artifact_directory(None)
    assert resolved == tmp_path / "artifacts" / (
        "architecture_visualization"
    )
    assert resolved.is_dir()
    explicit = module.resolve_artifact_directory(
        str(tmp_path / "elsewhere")
    )
    assert explicit == tmp_path / "elsewhere"
    assert explicit.is_dir()
    assert module.DEFAULT_ARTIFACT_DIRECTORY.endswith(
        "architecture_visualization"
    )


def test_invalid_spec_is_never_persisted(tmp_path):
    agent, spec = _visualize()
    rejected = validate_diagram_spec(
        DiagramSpec(**out_of_scope_spec_payload()),
        scope=DiagramScope.from_request(viz_request()),
    )
    destination = tmp_path / "rejected"

    with pytest.raises(VisualizationArtifactError, match="refusing"):
        write_visualization_artifacts(
            spec, rejected, directory=str(destination)
        )

    assert not destination.exists()


def test_svg_is_reported_honestly(tmp_path):
    artifacts = _write(tmp_path)
    statuses = {status.name: status for status in verify_artifacts(artifacts)}

    if graphviz_available():
        assert artifacts.svg_available is True
        assert Path(artifacts.svg).is_file()
        assert statuses["architecture.svg"].note == ""
    else:
        assert artifacts.svg_available is False
        assert artifacts.svg is None
        assert SVG_UNAVAILABLE_NOTICE in statuses["architecture.svg"].path
        summary = Path(artifacts.run_summary).read_text(encoding="utf-8")
        assert "SVG:\n{}".format(SVG_UNAVAILABLE_NOTICE) in summary


def test_diagram_spec_document_round_trip():
    spec = DiagramSpec(**spec_payload())

    restored = DiagramSpec.from_document(spec.to_document())

    assert restored == spec
    assert spec.to_json().endswith("\n")
    assert json.loads(spec.to_json())["title"] == spec.title
