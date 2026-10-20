"""Level 2 — the Architecture Investigator itself (spec §35-§36, §64).

Example-style scenarios a developer can read as the capability's
specification: components, relationships, evidence, classification,
scope, repository-agnosticism, the prompt, and the M16/M17 handoff
protocol — all over scripted doubles and tiny fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from architecture_doubles import (
    COMMIT,
    CONFLICT_FIXTURE_ROOT,
    FIXTURE_ROOT,
    MCP_COLLECTOR_PATH,
    MCP_CONFIG_PATH,
    MCP_FIXTURE_ROOT,
    QUESTION,
    RESEARCHER_PATH,
    FakeLLMClient,
    build_investigator,
    canned_bindings,
    derived_id,
    fixture_bindings,
    mcp_report_payload,
    read_then_report,
    report_payload,
    structured,
    tool_calls,
    call,
)

from repo_intelligence.agent import Agent
from repo_intelligence.agents import (
    ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    COMPONENT_TYPES,
    CONFIDENCE_LEVELS,
    PROMPT_VERSION,
    RELATIONSHIP_TYPES,
    ArchitectureComponent,
    ArchitectureInvestigator,
    ArchitectureInvestigatorError,
    ArchitectureRelationship,
    ArchitectureReport,
    component_statement,
    relationship_statement,
)
from repo_intelligence.agents import __file__ as agents_init_file
from repo_intelligence.engineering.models import RELATIONSHIP_KINDS
from repo_intelligence.evidence.models import (
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
)
from repo_intelligence.llm import LLMResponse, Message, StructuredOutputValidationError
from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)

AGENTS_PACKAGE = Path(agents_init_file).parent

READ_ONLY_ANSWER = LLMResponse(
    message=Message(role="assistant", content="not a typed report"),
    finish_reason="stop",
)


def _specialist(client) -> ArchitectureInvestigator:
    """The composition under test: specialist over a scripted Agent."""
    agent = Agent(
        client,
        tool_dispatcher=restricted_dispatcher(
            "investigator", bindings=canned_bindings()
        ),
        structured_output_model=ArchitectureReport,
        tool_definitions=tool_definitions_for("investigator"),
    )
    return ArchitectureInvestigator(agent=agent)


def _user_text(client) -> str:
    """All user-message text the scripted client was asked to answer."""
    texts = []
    for request in client.requests:
        for message in request.messages:
            if message.role == "user":
                texts.append(message.content or "")
    return "\n---\n".join(texts)


def plain_agent() -> Agent:
    """An Agent with no structured model (output is plain text)."""
    return Agent(FakeLLMClient(READ_ONLY_ANSWER))


# -- discovery: components, relationships, MCP (§36 categories) --------


def test_finds_repository_agents():
    specialist = build_investigator(*read_then_report())

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    agents = [c for c in report.components if c.type == "agent"]
    assert [c.id for c in agents] == ["researcher"]
    assert agents[0].classification == "observed"
    assert agents[0].evidence_refs[0].path == RESEARCHER_PATH


def test_finds_tools_used_by_agents():
    specialist = build_investigator(*read_then_report())

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    tool_ids = [c.id for c in report.components if c.type == "tool"]
    assert "search_notes" in tool_ids
    declares = [
        r for r in report.relationships if r.relationship_type == "DECLARES"
    ]
    assert declares
    assert declares[0].source == "researcher"
    assert declares[0].target == "search_notes"


def test_finds_mcp_servers_and_their_wiring():
    specialist = build_investigator(
        tool_calls(call("read_file", {"path": MCP_COLLECTOR_PATH})),
        tool_calls(call("read_file", {"path": MCP_CONFIG_PATH})),
        structured(mcp_report_payload()),
    )

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    servers = [c for c in report.components if c.type == "mcp_server"]
    assert [s.id for s in servers] == ["records"]
    exposed = [
        r
        for r in report.relationships
        if r.relationship_type == "EXPOSED_BY"
    ]
    assert exposed and exposed[0].target == "records"
    assert report.summary  # human-readable explanation present (§46)


def test_works_with_repository_without_mcp():
    specialist = build_investigator(*read_then_report())

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    assert [c.type for c in report.components].count("mcp_server") == 0
    assert report.status == "complete"


def test_works_with_different_repository_structure():
    specialist = build_investigator(
        tool_calls(call("read_file", {"path": MCP_COLLECTOR_PATH})),
        structured(mcp_report_payload()),
        bindings=fixture_bindings(MCP_FIXTURE_ROOT),
    )

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    read_paths = [
        result.get("data", {}).get("path")
        for iteration in specialist.last_trace.iterations
        for result in iteration.tool_results
    ]
    assert MCP_COLLECTOR_PATH in read_paths  # the fixture file, really read
    assert any(c.type == "agent" for c in report.components)


# -- evidence & classification (§23, §24, §26) -------------------------


def test_preserves_evidence_references():
    specialist = build_investigator(*read_then_report())

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    aggregate = {
        (ref.path, ref.line_start, ref.line_end)
        for ref in report.evidence_refs
    }
    for item in report.components + report.relationships + report.findings:
        for ref in item.evidence_refs:
            assert (ref.path, ref.line_start, ref.line_end) in aggregate


def test_report_refuses_refs_outside_the_aggregate():
    payload = report_payload(
        components=[
            {
                "id": "ghost",
                "name": "ghost",
                "type": "module",
                "evidence_refs": [{"path": "missing.py"}],
                "classification": "inferred",
            }
        ],
        evidence_refs=[{"path": RESEARCHER_PATH}],
    )

    with pytest.raises(Exception):
        ArchitectureReport(**payload)


def test_invalid_payload_fails_explicitly_at_the_agent():
    # Runtime metadata (question/repository_scope/commit_scope) is
    # filled from the request before validation; a SUBSTANTIVELY
    # invalid payload is never rescued by that fill — it still fails
    # here, explicitly, on its own content.
    specialist = build_investigator(
        structured({"summary": "malformed", "status": "unreachable"})
    )

    with pytest.raises(StructuredOutputValidationError):
        specialist.investigate(QUESTION, "example-repository", COMMIT)


def test_missing_structured_output_raises_a_specialist_error():
    specialist = ArchitectureInvestigator(agent=plain_agent())

    with pytest.raises(ArchitectureInvestigatorError):
        specialist.investigate(QUESTION, "example-repository", COMMIT)


def test_marks_unsupported_relationship_as_unknown():
    payload = report_payload(
        relationships=[
            {
                "source": "researcher",
                "target": "notes",
                "relationship_type": "DEPENDS_ON",
                "evidence_refs": [],
            }
        ],
        findings=[],
        evidence_refs=[
            {"path": RESEARCHER_PATH},
            {"path": "tools/search_notes.py"},
        ],
    )
    specialist = build_investigator(structured(payload))

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    relationship = report.relationships[0]
    assert relationship.classification == "unknown"
    assert relationship.confidence == "unknown"


def test_preserves_conflicting_evidence():
    payload = report_payload(
        relationships=[
            {
                "source": "researcher",
                "target": "csv_parser",
                "relationship_type": "DEPENDS_ON",
                "evidence_refs": [
                    {"path": "README.md"},
                    {"path": RESEARCHER_PATH},
                ],
                "classification": "conflicting",
                "confidence": "low",
            }
        ],
        evidence_refs=[
            {"path": "README.md"},
            {"path": RESEARCHER_PATH},
            {"path": "tools/search_notes.py"},
        ],
    )
    specialist = build_investigator(structured(payload))

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)

    assert report.relationships[0].classification == "conflicting"
    paths = {ref.path for ref in report.evidence_refs}
    assert {"README.md", RESEARCHER_PATH} <= paths


def test_conflict_fixture_is_documented_vs_implemented():
    readme = (CONFLICT_FIXTURE_ROOT / "README.md").read_text()
    implementation = (
        CONFLICT_FIXTURE_ROOT / "src" / "report_generator.py"
    ).read_text()

    assert "CSV parser" in readme
    assert "json" in implementation
    assert "csv" not in implementation


# -- direct execution (§30) & prompt (§13-§15) -------------------------


def test_investigate_runs_the_versioned_prompt():
    client = FakeLLMClient(*read_then_report())
    specialist = _specialist(client)

    specialist.investigate(QUESTION, "example-repository", COMMIT)

    text = _user_text(client)
    assert text.startswith("ROLE")
    assert PROMPT_VERSION == "V1"
    assert ARCHITECTURE_INVESTIGATOR_PROMPT_V1 in text
    assert "question: {}".format(QUESTION) in text
    assert "repository_scope: example-repository" in text
    assert "commit_scope: {}".format(COMMIT) in text
    assert "You investigate. You do not modify." in text


def test_prompt_is_repository_agnostic_and_matches_the_models():
    prompt = ARCHITECTURE_INVESTIGATOR_PROMPT_V1

    assert "adop" not in prompt.lower()
    for kind in RELATIONSHIP_KINDS:
        assert kind in prompt
    for component_type in COMPONENT_TYPES:
        assert component_type in prompt
    for level in CONFIDENCE_LEVELS:
        assert level in prompt


def test_vocabulary_is_reused_not_forked():
    literal_args = ArchitectureRelationship.model_fields[
        "relationship_type"
    ].annotation.__args__

    assert RELATIONSHIP_TYPES == RELATIONSHIP_KINDS
    assert tuple(literal_args) == RELATIONSHIP_KINDS
    assert set(COMPONENT_TYPES) >= {"agent", "tool", "mcp_server", "unknown"}


def test_statements_are_deterministic():
    component = ArchitectureComponent(
        id="researcher",
        name="Researcher",
        type="agent",
        responsibility="answers questions",
        classification="observed",
    )
    relationship = ArchitectureRelationship(
        source="researcher",
        target="search_notes",
        relationship_type="DECLARES",
        classification="derived",
    )

    assert component_statement(component) == (
        "Component researcher (agent): answers questions [observed]"
    )
    assert (
        relationship_statement(relationship)
        == "researcher DECLARES search_notes [derived]"
    )


# -- evidence validation for direct mode (§24) -------------------------


def test_validate_accepts_evidence_this_run_read():
    specialist = build_investigator(*read_then_report())

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)
    results = specialist.validate(report)

    assert results
    assert {r.status for r in results} == {STATUS_SUPPORTED}


def test_validate_rejects_evidence_never_read():
    specialist = build_investigator(*read_then_report())

    report = specialist.investigate(QUESTION, "example-repository", COMMIT)
    report.findings[0].evidence_refs[0].path = "never/read.py"
    report.evidence_refs[0].path = "never/read.py"

    results = specialist.validate(report)

    unsupported = [
        r for r in results if r.status == STATUS_UNSUPPORTED
    ]
    assert [r.claim.statement for r in unsupported] == [
        "the researcher imports its tools directly"
    ]
    assert unsupported[0].code == "CLAIM_EVIDENCE_INVALID"


def test_validate_needs_a_run_to_validate_against():
    specialist = build_investigator(*read_then_report())
    report = ArchitectureReport(
        question=QUESTION,
        repository_scope="example-repository",
        commit_scope=COMMIT,
    )

    with pytest.raises(ArchitectureInvestigatorError):
        specialist.validate(report)


# -- M16/M17 handoff protocol (§27, §29, §31) --------------------------


def test_run_result_wraps_the_runner_question():
    client = FakeLLMClient(*read_then_report())
    specialist = _specialist(client)
    runner_question = (
        "Investigate this plan step and produce your structured "
        "InvestigationReport.\nstep: P1 — Identify the agents"
    )

    result = specialist.run_result(runner_question)

    text = _user_text(client)
    assert text.startswith("ROLE")
    assert "QUESTION\n{}".format(runner_question) in text
    assert isinstance(result.structured_output, InvestigationReport)


def test_run_result_projects_components_and_relationships():
    specialist = build_investigator(*read_then_report())

    result = specialist.run_result("Investigate step P1")

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
    assert envelope.investigation_id == derived_id(report_payload())
    paths = {ref.path for ref in envelope.evidence_refs}
    assert {RESEARCHER_PATH, "tools/search_notes.py"} <= paths


def test_run_result_passes_failed_output_through_untouched():
    specialist = ArchitectureInvestigator(agent=plain_agent())

    result = specialist.run_result("Investigate step P1")

    assert result.structured_output is None  # the runner records failure


def test_run_result_conversion_is_pure():
    first = build_investigator(*read_then_report()).run_result(
        "Investigate step P1"
    )
    second = build_investigator(*read_then_report()).run_result(
        "Investigate step P1"
    )

    assert (
        first.structured_output.investigation_id
        == second.structured_output.investigation_id
    )
    assert first.structured_output is not second.structured_output


def test_last_trace_delegates_to_the_agent():
    specialist = build_investigator(*read_then_report())

    assert specialist.last_trace is None
    specialist.investigate(QUESTION, "example-repository", COMMIT)
    assert specialist.last_trace is specialist.agent.last_trace


def test_orchestrator_wiring_is_one_obvious_argument():
    from repo_intelligence.orchestration import Orchestrator

    specialist = build_investigator(*read_then_report())
    orchestrator = Orchestrator(specialist, specialist)  # type smoke only

    assert orchestrator._investigator is specialist


# -- boundaries: read-only, no repository logic (§12, §50, §66) --------


def test_does_not_modify_repository():
    def digest(root: Path):
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = digest(FIXTURE_ROOT)
    specialist = build_investigator(
        *read_then_report(),
        bindings=fixture_bindings(FIXTURE_ROOT),
    )

    specialist.investigate(QUESTION, "example-repository", COMMIT)

    assert digest(FIXTURE_ROOT) == before


def test_specialist_source_has_no_repository_specific_orchestration():
    sources = [
        path.read_text()
        for path in sorted(AGENTS_PACKAGE.rglob("*.py"))
    ]
    joined = "\n".join(sources).lower()

    assert "adop" not in joined
    assert "repo_intelligence.orchestration" not in joined
    assert "if orchestrated" not in joined
    assert "while true" not in joined


def test_specialist_owns_no_infrastructure():
    import repo_intelligence.agents.architecture_investigator as module

    text = Path(module.__file__).read_text().lower()
    for banned in (
        "def _tool_loop",
        "while true",
        "plan_from_response",
        "memorymanager",
        "contextengine",
    ):
        assert banned not in text


def test_component_and_relationship_models_refuse_unknown_values():
    with pytest.raises(Exception):
        ArchitectureComponent(id="x", name="x", type="microservice")
    with pytest.raises(Exception):
        ArchitectureRelationship(
            source="a", target="b", relationship_type="USES"
        )


def test_specialist_forwards_the_budget_through_the_observer_chain():
    """M20 participant contract through the notebook's exact chain:
    ``ArchitectureInvestigator -> ObservedAgent -> agent``. Every
    link must accept AND forward ``max_tool_calls`` — the missing
    forwards (router + observer wrappers) were the root cause of the
    LangGraph notebook's P1 ``TypeError`` failure: zero tool calls,
    no trace, all downstream steps blocked."""
    from repo_intelligence.repository_query import ObservedAgent

    class _RecordingAgent:
        last_trace = None

        def __init__(self):
            self.calls = []

        def run_result(self, question, *, max_tool_calls=None):
            self.calls.append((question, max_tool_calls))
            return "inner"

    inner = _RecordingAgent()
    specialist = ArchitectureInvestigator(agent=ObservedAgent(inner))

    assert specialist.run_result("q", max_tool_calls=9) == "inner"
    assert len(inner.calls) == 1
    question, budget = inner.calls[0]
    assert budget == 9
    assert question  # wrapped with the specialist's prompt

    # Budget unset -> the exact pre-M20 one-argument call, so a
    # legacy inner participant keeps working unchanged.
    class _LegacyAgent:
        last_trace = None

        def run_result(self, question):
            return "legacy-ok"

    legacy = ArchitectureInvestigator(
        agent=ObservedAgent(_LegacyAgent())
    )
    assert legacy.run_result("q") == "legacy-ok"
