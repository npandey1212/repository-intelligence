"""Level 3 — the capability end to end through M10 → M17 (§39, §59).

The SAME specialist, driven by the planner, orchestrated step by step,
handed to the M16 Analyst through the typed envelope, and judged by the
deterministic validator — over tiny fixtures with scripted models.
"""

from __future__ import annotations

from typing import Dict, List

from architecture_doubles import (
    FIXTURE_ROOT,
    MCP_COLLECTOR_PATH,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    QUESTION,
    RESEARCHER_PATH,
    FakeLLMClient,
    build_analyst,
    build_investigator,
    derived_id,
    finding_payload,
    fixture_bindings,
    mcp_report_payload,
    plan_payload,
    report_payload,
    rule,
    structured,
    task,
    tool_calls,
    call,
)

from repo_intelligence.agent import AgentExecutionConfig
from repo_intelligence.agent.trace import render_markdown
from repo_intelligence.agents import ARCHITECTURE_TOOLS, ArchitectureInvestigator
from repo_intelligence.evidence import gather_evidence
from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.tools import ToolResult
from repo_intelligence.orchestration import Orchestrator
from repo_intelligence.planning.planner import (
    build_planning_request,
    plan_from_response,
)
from repo_intelligence.planning.validation import validate_plan

FETCH_RECORD_PATH = "tools/fetch_record.py"


# -- per-step scripts (each step reads what it claims) -----------------


def _mcp_step_payloads() -> List[Dict]:
    return [
        mcp_report_payload(
            components=[
                {
                    "id": "collector",
                    "name": "Collector",
                    "type": "agent",
                    "responsibility": "collects records",
                    "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                    "classification": "observed",
                },
                {
                    "id": "fetch_record",
                    "name": "fetch_record",
                    "type": "tool",
                    "responsibility": "reads one record",
                    "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                    "classification": "observed",
                },
            ],
            relationships=[
                {
                    "source": "collector",
                    "target": "fetch_record",
                    "relationship_type": "DECLARES",
                    "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                    "classification": "observed",
                    "confidence": "high",
                }
            ],
            findings=[
                {
                    "statement": "the collector imports the record tool",
                    "evidence_refs": [{"path": MCP_COLLECTOR_PATH}],
                    "support": "SUPPORTED",
                }
            ],
            evidence_refs=[{"path": MCP_COLLECTOR_PATH}],
        ),
        mcp_report_payload(
            components=[
                {
                    "id": "records",
                    "name": "records",
                    "type": "mcp_server",
                    "responsibility": "serves fetch_record over MCP",
                    "evidence_refs": [{"path": MCP_CONFIG_PATH}],
                    "classification": "observed",
                }
            ],
            relationships=[
                {
                    "source": "fetch_record",
                    "target": "records",
                    "relationship_type": "EXPOSED_BY",
                    "evidence_refs": [{"path": MCP_CONFIG_PATH}],
                    "classification": "observed",
                    "confidence": "high",
                }
            ],
            findings=[],
            evidence_refs=[{"path": MCP_CONFIG_PATH}],
        ),
        mcp_report_payload(
            components=[],
            relationships=[],
            findings=[
                {
                    "statement": "the tool reads records by key",
                    "evidence_refs": [{"path": FETCH_RECORD_PATH}],
                    "support": "SUPPORTED",
                }
            ],
            evidence_refs=[{"path": FETCH_RECORD_PATH}],
            uncertainties=[
                "no runtime invocation evidence exists in the repository"
            ],
            status="complete",
        ),
    ]


def _simple_step_payloads() -> List[Dict]:
    return [
        report_payload(
            components=[
                {
                    "id": "researcher",
                    "name": "Researcher",
                    "type": "agent",
                    "responsibility": "answers questions from notes",
                    "evidence_refs": [{"path": RESEARCHER_PATH}],
                    "classification": "observed",
                }
            ],
            relationships=[],
            findings=[],
            evidence_refs=[{"path": RESEARCHER_PATH}],
        ),
        report_payload(
            components=[
                {
                    "id": "search_notes",
                    "name": "search_notes",
                    "type": "tool",
                    "responsibility": "lexical note search",
                    "evidence_refs": [
                        {"path": "tools/search_notes.py"}
                    ],
                    "classification": "observed",
                }
            ],
            relationships=[
                {
                    "source": "researcher",
                    "target": "search_notes",
                    "relationship_type": "DECLARES",
                    "evidence_refs": [
                        {"path": "tools/search_notes.py"}
                    ],
                    "classification": "derived",
                    "confidence": "medium",
                }
            ],
            findings=[],
            evidence_refs=[{"path": "tools/search_notes.py"}],
        ),
        report_payload(
            components=[],
            relationships=[],
            findings=[
                {
                    "statement": "notes are read from disk by name",
                    "evidence_refs": [{"path": "tools/fetch_note.py"}],
                    "support": "SUPPORTED",
                }
            ],
            evidence_refs=[{"path": "tools/fetch_note.py"}],
            uncertainties=["no MCP configuration exists in this repository"],
        ),
    ]


def _execute(root, payloads, reads, conclusion_ref):
    """Run the full application path over a fixture and return the
    run plus its participants."""
    planner = FakeLLMClient(structured(plan_payload()))
    response = planner.complete(build_planning_request(QUESTION))
    investigation_plan = plan_from_response(response)
    validate_plan(investigation_plan)

    investigator_script = []
    for payload, path in zip(payloads, reads):
        investigator_script.append(tool_calls(call("read_file", {"path": path})))
        investigator_script.append(structured(payload))

    specialist = build_investigator(
        *investigator_script, bindings=fixture_bindings(root)
    )
    analyst = build_analyst(
        structured(
            finding_payload(
                investigation_id=derived_id(payloads[0]),
                conclusions=[
                    {
                        "statement": "the agents reach their tools directly",
                        "classification": "observed",
                        "evidence_refs": [{"path": conclusion_ref}],
                    }
                ],
                evidence_refs=[{"path": conclusion_ref}],
                summary="Evidence-backed architecture summary.",
            )
        )
    )
    orchestrator = Orchestrator(specialist, analyst)
    run = orchestrator.execute(task(), investigation_plan)
    return run, specialist, analyst, investigation_plan


def test_mcp_repository_end_to_end():
    payloads = _mcp_step_payloads()
    run, specialist, analyst, investigation_plan = _execute(
        MCP_FIXTURE_ROOT,
        payloads,
        [MCP_COLLECTOR_PATH, MCP_CONFIG_PATH, FETCH_RECORD_PATH],
        MCP_COLLECTOR_PATH,
    )

    assert run.status == "COMPLETED"
    assert run.termination_reason == "SUCCESS"
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert rule(run.validation, "evidence_valid").ok is True
    assert rule(run.validation, "collaboration").ok is True
    assert rule(run.validation, "classification").ok is True


def test_components_survive_the_typed_handoff():
    payloads = _mcp_step_payloads()
    run, _, _, _ = _execute(
        MCP_FIXTURE_ROOT,
        payloads,
        [MCP_COLLECTOR_PATH, MCP_CONFIG_PATH, FETCH_RECORD_PATH],
        MCP_COLLECTOR_PATH,
    )

    statements = [f.statement for f in run.report.findings]
    assert any(
        s.startswith("Component collector (agent)") for s in statements
    )
    assert any(
        s.startswith("Component records (mcp_server)") for s in statements
    )
    assert any(
        "fetch_record EXPOSED_BY records [observed]" in s
        for s in statements
    )
    paths = {ref.path for ref in run.report.evidence_refs}
    assert {MCP_COLLECTOR_PATH, MCP_CONFIG_PATH, FETCH_RECORD_PATH} <= paths


def test_roles_and_steps_follow_the_plan():
    payloads = _mcp_step_payloads()
    run, specialist, _, investigation_plan = _execute(
        MCP_FIXTURE_ROOT,
        payloads,
        [MCP_COLLECTOR_PATH, MCP_CONFIG_PATH, FETCH_RECORD_PATH],
        MCP_COLLECTOR_PATH,
    )

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
    assert all(entry["run_id"] for entry in run.agent_runs)


def test_specialist_traces_carry_the_orchestration_events():
    payloads = _mcp_step_payloads()
    run, specialist, analyst, _ = _execute(
        MCP_FIXTURE_ROOT,
        payloads,
        [MCP_COLLECTOR_PATH, MCP_CONFIG_PATH, FETCH_RECORD_PATH],
        MCP_COLLECTOR_PATH,
    )

    trace = specialist.last_trace
    assert trace is specialist.agent.last_trace
    assert trace.orchestration_events
    operations = {e["operation"] for e in trace.orchestration_events}
    assert "orchestration.step.assign" in operations
    assert "orchestration.agent.start" in operations
    markdown = render_markdown(trace)
    assert "## Orchestration" in markdown
    analyst_ops = {e["operation"] for e in analyst.last_trace.orchestration_events}
    assert "orchestration.step.assign" in analyst_ops


def test_repository_without_mcp_still_completes():
    payloads = _simple_step_payloads()
    run, specialist, _, _ = _execute(
        FIXTURE_ROOT,
        payloads,
        [
            RESEARCHER_PATH,
            "tools/search_notes.py",
            "tools/fetch_note.py",
        ],
        RESEARCHER_PATH,
    )

    assert run.status == "COMPLETED"
    assert run.validation.status == "passed"
    statements = [f.statement for f in run.report.findings]
    assert any(
        s.startswith("Component researcher (agent)") for s in statements
    )
    assert not any("mcp_server" in s for s in statements)


def test_the_specialist_is_just_an_injected_participant():
    payloads = _mcp_step_payloads()
    run, specialist, _, _ = _execute(
        MCP_FIXTURE_ROOT,
        payloads,
        [MCP_COLLECTOR_PATH, MCP_CONFIG_PATH, FETCH_RECORD_PATH],
        MCP_COLLECTOR_PATH,
    )

    assert isinstance(specialist, ArchitectureInvestigator)
    assert specialist.tools == list(ARCHITECTURE_TOOLS)
    # the specialist produced M16 envelopes for the orchestrator
    investigator_results = [
        result
        for result in run.step_results
        if result.agent_role == "INVESTIGATOR"
    ]
    assert all(
        type(result.output).__name__ == "InvestigationReport"
        for result in investigator_results
    )


# -- working-context compaction inside one specialist step -------------


def _padded_read_bindings(root):
    """Real fixture reads whose content is padded with filler, so a
    single specialist step crosses the working-context ceiling."""
    real = fixture_bindings(root)

    def read(arguments):
        result = real["read_file"](arguments)
        if not result.success:
            return result
        data = dict(result.data or {})
        data["content"] = "{} {}".format(
            data.get("content", ""), "filler " * 3000
        )
        return ToolResult(
            success=True,
            tool="read_file",
            data=data,
            source=getattr(result, "source", None),
        )

    bindings = dict(real)
    bindings["read_file"] = read
    return bindings


def test_compacted_specialist_still_hands_over_a_typed_report():
    """Working-context compaction inside a specialist step must not
    disturb the M16/M17 handoff: the Analyst still receives the typed
    envelope, over evidence this run actually read."""
    specialist = build_investigator(
        tool_calls(call("read_file", {"path": RESEARCHER_PATH})),
        tool_calls(
            call("read_file", {"path": "tools/search_notes.py"})
        ),
        structured(report_payload()),
        bindings=_padded_read_bindings(FIXTURE_ROOT),
        execution_config=AgentExecutionConfig(
            max_context_characters=30_000
        ),
    )

    result = specialist.run_result(QUESTION)

    trace = specialist.agent.last_trace
    # The two padded reads crossed the ceiling inside this one step:
    # the working history was replaced, with no extra model call for
    # it and no restriction left behind.
    assert trace.context_compactions == 1
    assert trace.context_restrictions == 0
    assert len(trace.iterations) == 3

    # The typed handoff itself: components and relationships projected
    # to findings, refs aggregated — nothing lost at the compacted hop.
    envelope = result.structured_output
    assert isinstance(envelope, InvestigationReport)
    statements = [f.statement for f in envelope.findings]
    assert any(
        s.startswith("Component researcher (agent)") for s in statements
    )
    assert any(
        "researcher DECLARES search_notes [observed]" in s
        for s in statements
    )
    paths = {ref.path for ref in envelope.evidence_refs}
    assert {RESEARCHER_PATH, "tools/search_notes.py"} <= paths

    # ...and the step's own evidence record survived the replacement.
    assert {ref.path for ref in gather_evidence(trace).gathered} == {
        RESEARCHER_PATH,
        "tools/search_notes.py",
    }
