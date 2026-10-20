"""M16 multi-agent runner — spec §23 collaboration tests 1–12 and
runner-level bounds/status/tracing (test 13 lives in
``test_repository_agnostic.py`` with the non-ADOP fixture).

Every scenario is scripted and LLM-free: queued ``LLMResponse``s, a
canned tool dispatcher, one ``TaskPayload``. What is proven here:

* role isolation end-to-end (definitions + registry, probe refused)
* typed consumption (the Analyst's request embeds the Investigator's
  exact report; ids must echo)
* deterministic validation outcomes (evidence, scope, classification)
* explicit failure statuses with NO fabricated outputs
* bounded execution (message cap, tool budget, stage deadline)
* trace mirroring: collaboration events land on each agent's own
  execution trace and the shared sink artifact.
"""

from __future__ import annotations

import json
import time

from repo_intelligence.agent import Agent, AgentExecutionConfig
from repo_intelligence.agent.trace_sink import JsonFileTraceSink
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.multiagent import (
    BUDGET_EXCEEDED,
    COMPLETED,
    FAILED,
    INCOMPLETE,
    RUNNING,
    TIMEOUT,
    VALIDATION_FAILED,
    ContractError,
    InvestigationReport,
    ArchitectureFinding,
    MultiAgentRunner,
    RunnerConfig,
    TaskPayload,
    make_message,
    is_terminal,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.tools import ToolError, ToolResult

REPOSITORY = "example-repository"
COMMIT = "c" * 40
REPORT_ID = "inv-1"
QUESTION = "Where is the application boundary enforced?"
README_REF = {"path": "README.md"}


# -- payloads (schema-valid, deterministic) -------------------------------

def _task(**overrides):
    data = {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }
    data.update(overrides)
    return TaskPayload(**data)


def _report_payload(**overrides):
    data = {
        "investigation_id": REPORT_ID,
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "findings": [
            {
                "statement": "README.md documents the dispatcher boundary",
                "evidence_refs": [dict(README_REF)],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [dict(README_REF)],
        "investigated_sources": ["README.md"],
        "status": "complete",
    }
    data.update(overrides)
    return data


def _finding_payload(**overrides):
    data = {
        "investigation_id": REPORT_ID,
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "conclusions": [
            {
                "statement": "the boundary is application-owned",
                "classification": "observed",
                "evidence_refs": [dict(README_REF)],
            }
        ],
        "evidence_refs": [dict(README_REF)],
        "summary": "One dispatcher, one boundary.",
        "status": "complete",
    }
    data.update(overrides)
    return data


# -- test doubles ---------------------------------------------------------


class FakeLLMClient:
    """Queued responses; refuses to be over-invoked."""

    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


def _structured(payload) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(*calls) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name, arguments):
    return ToolCall(id="call_1", name=name, arguments=dict(arguments))


class QueuedDispatcher:
    """Canned results in order, then generic per-name success."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call) -> ToolResult:
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(
            success=True,
            tool=tool_call.name,
            data={"path": "README.md", "content": "boundary"},
        )


def _investigator_bindings(failure_paths=()):
    """Hermetic read-only tools: ``read_file`` serves canned content
    for any path except the ones scripted to fail."""

    def read_file(arguments):
        path = arguments.get("path") or "README.md"
        if path in failure_paths:
            return ToolResult(
                success=False,
                tool="read_file",
                error=ToolError(code="FILE_NOT_FOUND", message="no file"),
            )
        return ToolResult(
            success=True,
            tool="read_file",
            data={"path": path, "content": "the dispatcher owns the boundary"},
        )

    def generic(name):
        def tool(arguments):
            return ToolResult(
                success=True,
                tool=name,
                data={"path": arguments.get("path") or "README.md",
                      "content": "x"},
            )

        return tool

    return {
        "read_file": read_file,
        "search_repository": generic("search_repository"),
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def _build(
    inv_responses=None,
    ana_responses=None,
    inv_bindings=None,
    ana_dispatcher=None,
    inv_kwargs=None,
    config=None,
    clock=None,
    trace_sink=None,
):
    if inv_responses is None:
        inv_responses = [
            _tool_calls(_call("read_file", {"path": "README.md"})),
            _structured(_report_payload()),
        ]
    if ana_responses is None:
        ana_responses = [_structured(_finding_payload())]
    inv_kwargs = dict(inv_kwargs or {})
    investigator = Agent(
        FakeLLMClient(*inv_responses),
        tool_dispatcher=restricted_dispatcher(
            "investigator",
            bindings=inv_bindings or _investigator_bindings(),
        ),
        trace_sink=trace_sink,
        structured_output_model=InvestigationReport,
        tool_definitions=tool_definitions_for("investigator"),
        **inv_kwargs,
    )
    analyst = Agent(
        FakeLLMClient(*ana_responses),
        tool_dispatcher=(
            ana_dispatcher
            if ana_dispatcher is not None
            else restricted_dispatcher("analyst")
        ),
        trace_sink=trace_sink,
        structured_output_model=ArchitectureFinding,
        tool_definitions=tool_definitions_for("analyst"),
    )
    runner = MultiAgentRunner(
        investigator,
        analyst,
        trace_sink=trace_sink,
        config=config,
        clock=clock or time.monotonic,
    )
    return runner, investigator, analyst


def _rules(result, rule):
    matches = [check for check in result.checks if check.rule == rule]
    assert len(matches) == 1, rule
    return matches[0]


# -- §23 tests 1–2: role isolation ---------------------------------------


def test_roles_receive_isolated_tools():
    runner, investigator, analyst = _build()
    state = runner.run(_task())
    assert state.status == COMPLETED
    # Investigator offered/executed its allowlist; the Analyst was
    # offered nothing and its registry refused everything.
    inv_requests = investigator._llm_client.requests
    assert all(len(request.tools) == 4 for request in inv_requests)
    ana_requests = analyst._llm_client.requests
    assert all(request.tools == [] for request in ana_requests)
    assert analyst._tool_dispatcher.tool_names == ()


def test_analyst_probe_tool_refused():
    probe = _tool_calls(_call("read_file", {"path": "README.md"}))
    runner, _, analyst = _build(ana_responses=[probe, _structured(_finding_payload())])
    state = runner.run(_task())
    assert state.status == COMPLETED
    refused = [
        result
        for iteration in analyst.last_trace.iterations
        for result in iteration.tool_results
        if not result.get("success")
    ]
    assert refused, "the probe must surface as a recorded refusal"
    error = refused[0].get("error") or {}
    assert error.get("code") == "TOOL_NOT_FOUND"
    # ...and the unregistered tool never executed (empty registry).
    assert analyst._tool_dispatcher.tool_names == ()


# -- §23 tests 4, 6: evidence + typed consumption -------------------------


def test_evidence_validated_by_validator():
    runner, _, _ = _build()
    state = runner.run(_task())
    assert state.status == COMPLETED
    assert state.validation is not None
    assert state.validation.status == "passed"
    assert len(state.validation.checks) == 8
    assert all(check.ok for check in state.validation.checks)
    assert state.tool_failures == []


def test_typed_output_consumed_by_next_stage():
    runner, _, analyst = _build()
    state = runner.run(_task())
    assert state.status == COMPLETED
    user_texts = [
        message.content or ""
        for request in analyst._llm_client.requests
        for message in request.messages
    ]
    embedded = [text for text in user_texts if REPORT_ID in text]
    assert embedded, "the analyst's request must embed the report id"
    assert any(
        "README.md" in text and "evidence_refs" in text
        for text in embedded
    ), "the analyst must receive the report's typed evidence package"
    # The ANALYSIS_RESULT echoed the investigation id (collaboration
    # proof checked deterministically by the validator).
    assert state.finding.investigation_id == REPORT_ID


# -- §23 test 5: scope preservation --------------------------------------


def test_scope_preserved_on_every_hop():
    runner, _, _ = _build()
    state = runner.run(_task())
    assert state.status == COMPLETED
    for message in state.messages:
        assert message.repository_scope == REPOSITORY
        assert message.commit_scope == COMMIT


def test_scope_mismatch_rejected_by_validator():
    runner, _, _ = _build(
        inv_responses=[
            _tool_calls(_call("read_file", {"path": "README.md"})),
            _structured(_report_payload(commit_scope="f" * 40)),
        ]
    )
    state = runner.run(_task())
    assert state.status == VALIDATION_FAILED
    assert _rules(state.validation, "scope_match").ok is False
    assert state.report is not None
    assert state.finding is not None


# -- §23 tests 7–8: missing / conflicting evidence ------------------------


def test_missing_evidence_rejected():
    runner, _, _ = _build(
        inv_responses=[
            _tool_calls(_call("read_file", {"path": "README.md"})),
            _structured(
                _report_payload(
                    findings=[
                        {
                            "statement": "claims a file never read",
                            "evidence_refs": [{"path": "never-read.py"}],
                            "support": "SUPPORTED",
                        }
                    ],
                    evidence_refs=[{"path": "never-read.py"}],
                )
            ),
        ],
        ana_responses=[
            _structured(
                _finding_payload(
                    conclusions=[
                        {
                            "statement": "claims a file never read",
                            "classification": "observed",
                            "evidence_refs": [{"path": "never-read.py"}],
                        }
                    ],
                    evidence_refs=[{"path": "never-read.py"}],
                )
            )
        ],
    )
    state = runner.run(_task())
    assert state.status == VALIDATION_FAILED
    assert _rules(state.validation, "evidence_valid").ok is False


def test_conflicting_evidence_rejected_when_single_path():
    runner, _, _ = _build(
        ana_responses=[
            _structured(
                _finding_payload(
                    conclusions=[
                        {
                            "statement": "one path cannot be conflicting",
                            "classification": "conflicting",
                            "evidence_refs": [dict(README_REF)],
                        }
                    ]
                )
            )
        ]
    )
    state = runner.run(_task())
    assert state.status == VALIDATION_FAILED
    assert _rules(state.validation, "classification").ok is False


# -- §23 tests 9–10: failures leave no fabricated output ------------------


def test_investigator_failure_no_fabrication():
    broken = _report_payload()
    del broken["findings"]
    runner, _, analyst = _build(
        inv_responses=[
            _tool_calls(_call("read_file", {"path": "README.md"})),
            _structured(broken),
        ]
    )
    state = runner.run(_task())
    assert state.status == FAILED
    assert state.report is None
    assert state.finding is None
    assert state.validation is None
    assert state.error is not None
    assert analyst._llm_client.requests == []
    assert is_terminal(state.status)


def test_analyst_failure_no_fabrication():
    broken = _finding_payload()
    del broken["summary"]
    runner, _, _ = _build(
        ana_responses=[_structured(broken)],
    )
    state = runner.run(_task())
    assert state.status == FAILED
    assert state.report is not None
    assert state.finding is None
    assert state.validation is None
    assert state.error is not None


# -- §23 test 11: tool failures visible, never hidden --------------------


def test_tool_failure_visible_but_not_fatal():
    failed_first = QueuedDispatcher(
        ToolResult(
            success=False,
            tool="read_file",
            error=ToolError(code="FILE_NOT_FOUND", message="missing"),
        ),
        ToolResult(
            success=True,
            tool="read_file",
            data={"path": "README.md", "content": "boundary"},
        ),
    )
    runner, investigator, _ = _build(
        inv_responses=[
            _tool_calls(
                _call("read_file", {"path": "missing.py"}),
                _call("read_file", {"path": "README.md"}),
            ),
            _structured(_report_payload()),
        ],
        inv_bindings=None,
    )
    # swap in the queued dispatcher after construction
    investigator._tool_dispatcher = failed_first
    state = runner.run(_task())
    assert state.status == COMPLETED
    assert len(state.tool_failures) == 1
    failure = state.tool_failures[0]
    assert failure.agent == "investigator"
    assert failure.error_code == "FILE_NOT_FOUND"
    assert state.validation.tool_failures == state.tool_failures


# -- §23 test 12: bounded execution --------------------------------------


def test_message_budget_exhaustion_is_explicit():
    runner, _, analyst = _build(config=RunnerConfig(max_messages=2))
    state = runner.run(_task())
    assert state.status == BUDGET_EXCEEDED
    assert state.report is not None          # investigation finished
    assert state.finding is None             # analysis never started
    assert state.validation is None
    assert analyst._llm_client.requests == []
    assert is_terminal(state.status)


def test_tool_budget_maps_to_budget_exceeded():
    runner, _, _ = _build(
        inv_kwargs={
            "execution_config": AgentExecutionConfig(max_tool_calls=0)
        }
    )
    state = runner.run(_task())
    assert state.status == BUDGET_EXCEEDED
    assert state.finding is None
    assert "budget exhausted" in (state.error or "")
    assert "blocked and NOT executed" in (state.error or "")


def test_stage_deadline_maps_to_timeout():
    class SteppingClock:
        def __init__(self, step):
            self.value = 0.0
            self.step = step

        def __call__(self):
            current = self.value
            self.value += self.step
            return current

    runner, _, analyst = _build(
        clock=SteppingClock(10.0),
        config=RunnerConfig(stage_deadline_seconds=5.0),
    )
    state = runner.run(_task())
    assert state.status == TIMEOUT
    assert state.finding is None
    assert analyst._llm_client.requests == []


# -- runner behaviour: statuses, events, trace mirroring ------------------


def test_happy_path_message_sequence_and_status():
    runner, _, _ = _build()
    state = runner.run(_task())
    assert state.status == COMPLETED
    assert [message.message_type for message in state.messages] == [
        "INVESTIGATION_REQUEST",
        "INVESTIGATION_RESULT",
        "ANALYSIS_REQUEST",
        "ANALYSIS_RESULT",
        "VALIDATION_REQUEST",
        "VALIDATION_RESULT",
    ]
    assert state.completed_at
    assert state.status not in (RUNNING,)


def test_collaboration_events_are_sequenced():
    runner, investigator, analyst = _build()
    state = runner.run(_task())
    sequences = [event["sequence"] for event in state.collaboration_events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))
    operations = [event["operation"] for event in state.collaboration_events]
    assert operations[0] == "message.send"
    assert operations[-1] == "message.send"  # VALIDATION_RESULT
    assert "agent.start" in operations
    assert "validator.complete" in operations


def test_events_mirrored_onto_agent_traces(tmp_path):
    sink = JsonFileTraceSink(root=str(tmp_path / "traces"), markdown=True)
    runner, investigator, analyst = _build(trace_sink=sink)
    state = runner.run(_task())
    inv_ops = [
        event["operation"]
        for event in investigator.last_trace.collaboration_events
    ]
    assert "agent.start" in inv_ops
    assert "message.receive" in inv_ops
    assert "agent.end" in inv_ops
    ana_ops = [
        event["operation"]
        for event in analyst.last_trace.collaboration_events
    ]
    assert "agent.start" in ana_ops
    assert "agent.end" in ana_ops
    # The runner re-wrote the sink artifact with the mirrored events.
    trace_json = (
        tmp_path / "traces" / state.investigator_run_id / "trace.json"
    )
    payload = json.loads(trace_json.read_text(encoding="utf-8"))
    assert payload.get("collaboration_events")
    assert "## Collaboration" in (
        tmp_path / "traces" / state.investigator_run_id / "trace.md"
    ).read_text(encoding="utf-8")


def test_incomplete_declaration_is_not_a_success():
    runner, _, _ = _build(
        inv_responses=[
            _tool_calls(_call("read_file", {"path": "README.md"})),
            _structured(_report_payload(status="incomplete")),
        ],
    )
    state = runner.run(_task())
    assert state.status == INCOMPLETE
    assert state.validation.status == "passed"
    assert is_terminal(state.status)


def test_contract_error_propagates_from_malformed_handoff():
    report = InvestigationReport.parse_obj(_report_payload())
    try:
        make_message(
            "INVESTIGATION_RESULT",
            "investigator",
            "analyst",
            {"not": "a report"},
            REPOSITORY,
            COMMIT,
        )
    except ContractError:
        pass
    else:
        raise AssertionError("malformed handoff must raise ContractError")
    # the typed path stays valid
    message = make_message(
        "INVESTIGATION_RESULT",
        "investigator",
        "analyst",
        report,
        REPOSITORY,
        COMMIT,
    )
    assert isinstance(message.payload, InvestigationReport)
