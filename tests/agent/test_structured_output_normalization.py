"""Envelope selection on the structured path (M17 finalization fix).

The live failure this file pins down: a finalization response shaped

    {"structured_output": {"title": "ArchitectureReport",
                            "type": "object", "properties": {...}}}

never reached ``ArchitectureReport`` — validation ran against the
WRAPPER object and reported the model's own required fields
(``question`` / ``repository_scope`` / ``commit_scope``) as missing.
Which level the model chose is provider behavior; selecting the level
that validates is the application's job, so:

1. unit shape        the observed envelope — schema echo alongside,
                     or instance data under ``properties`` — validates
                     to a real ``ArchitectureReport`` instance
2. agent path        one scripted run returns the typed report, and
                     the iteration records ``normalized: true`` so the
                     level selection is observable on the trace
3. finalization path the same holds on the tool-free finalization
                     pass, where the failure was actually seen
4. no change when
   nothing wraps     an unwrapped payload is byte-for-byte the
                     pre-fix behavior (no ``normalized`` marker, same
                     validation dict)
5. nothing to take   an envelope with no valid instance inside still
                     fails explicitly — no field is invented, no
                     answer is fabricated

Scripted and LLM-free: queued responses, a recording dispatcher, one
question — no network, no API key.
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.agent import (
    Agent,
    AgentExecutionConfig,
)
from repo_intelligence.agents.models.architecture_report import (
    ArchitectureReport,
)
from repo_intelligence.llm import (
    LLMResponse,
    Message,
    StructuredOutputValidationError,
    ToolCall,
    normalize_structured_payload,
    validate_structured,
)
from repo_intelligence.tools import ToolResult

QUESTION = "How do the modules interact?"
REPOSITORY = "example-repository"
COMMIT = "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"


def _instance():
    """A valid ArchitectureReport payload — the instance to recover."""
    return {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "summary": "observed interaction",
        "components": [],
        "relationships": [],
        "findings": [],
        "evidence_refs": [],
        "uncertainties": [],
        "status": "complete",
    }


def _wrapped_schema_echo():
    """Level 1: the echoed schema object with the instance written
    alongside its ``title``/``type``/``properties`` keys."""
    envelope = dict(ArchitectureReport.schema())
    envelope.update(_instance())
    return {"structured_output": envelope}


def _wrapped_properties():
    """Level 1: the echoed schema object with the instance written
    under its ``properties`` key."""
    return {
        "structured_output": {
            "title": "ArchitectureReport",
            "type": "object",
            "properties": _instance(),
        }
    }


class FakeLLMClient:
    def __init__(self, *responses):
        self._queue = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than queued")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    def __init__(self, *results):
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _structured(payload):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_response(*calls):
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def _read(path, call_id="call_1"):
    return ToolCall(id=call_id, name="read_file", arguments={"path": path})


def _read_result(path):
    return ToolResult(
        success=True,
        tool="read_file",
        data={"path": path, "content": "content of {}".format(path)},
        source={"artifact": path},
    )


def _agent(responses, dispatcher, max_tool_calls):
    return Agent(
        FakeLLMClient(*responses),
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(
            max_tool_calls=max_tool_calls,
            finalize_on_boundary=True,
        ),
        structured_output_model=ArchitectureReport,
    )


# ---------------------------------------------------------------------------
# 1 — the observed shapes validate to the real instance
# ---------------------------------------------------------------------------


def test_the_observed_envelope_validates_to_a_typed_report():
    for payload in (_wrapped_schema_echo(), _wrapped_properties()):
        typed = validate_structured(ArchitectureReport, payload)
        assert isinstance(typed, ArchitectureReport)
        assert typed.question == QUESTION
        assert typed.repository_scope == REPOSITORY
        assert typed.commit_scope == COMMIT
        assert typed.status == "complete"


def test_the_envelope_level_is_selected_not_repaired():
    payload = _wrapped_schema_echo()
    selected = normalize_structured_payload(ArchitectureReport, payload)
    assert selected is payload["structured_output"]
    # the selected object is the model's own — no field was added
    assert selected["title"] == "ArchitectureReport"


# ---------------------------------------------------------------------------
# 2 — one scripted run returns the typed report
# ---------------------------------------------------------------------------


def test_agent_run_normalizes_the_envelope_and_returns_the_report():
    dispatcher = FakeDispatcher(_read_result("src/alpha.py"))
    agent = _agent(
        [
            _tool_response(_read("src/alpha.py")),
            _structured(_wrapped_schema_echo()),
        ],
        dispatcher,
        max_tool_calls=4,
    )

    result = agent.run_result(QUESTION)

    assert isinstance(result.structured_output, ArchitectureReport)
    assert result.structured_output.question == QUESTION
    trace = agent.last_trace
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    iteration = trace.iterations[-1]
    assert iteration.structured_validation == {
        "status": "SUCCESS",
        "model": "ArchitectureReport",
        "normalized": True,
    }


# ---------------------------------------------------------------------------
# 3 — the finalization pass (where the live failure happened)
# ---------------------------------------------------------------------------


def test_finalization_normalizes_the_envelope():
    dispatcher = FakeDispatcher(_read_result("src/alpha.py"))
    agent = _agent(
        [
            _tool_response(_read("src/alpha.py")),
            _tool_response(_read("src/beta.py", "call_2")),  # blocked
            _structured(_wrapped_properties()),
        ],
        dispatcher,
        max_tool_calls=1,  # boundary: the second tool request is blocked
    )

    result = agent.run_result(QUESTION)

    assert len(dispatcher.calls) == 1  # the blocked call never ran
    assert isinstance(result.structured_output, ArchitectureReport)
    operations = [
        event["operation"] for event in agent.last_trace.lifecycle_events
    ]
    assert "finalization.started" in operations
    assert operations[-1] == "finalization.completed"
    final_iteration = agent.last_trace.iterations[-1]
    assert final_iteration.structured_validation["normalized"] is True
    # the finalization request itself stays tool-free
    assert agent._llm_client.requests[-1].tools == []


# ---------------------------------------------------------------------------
# 4 — an unwrapped payload is unchanged behavior
# ---------------------------------------------------------------------------


def test_unwrapped_payload_keeps_the_pre_fix_record():
    payload = _instance()
    assert normalize_structured_payload(ArchitectureReport, payload) is payload

    dispatcher = FakeDispatcher(_read_result("src/alpha.py"))
    agent = _agent(
        [_tool_response(_read("src/alpha.py")), _structured(payload)],
        dispatcher,
        max_tool_calls=4,
    )

    result = agent.run_result(QUESTION)

    assert isinstance(result.structured_output, ArchitectureReport)
    assert agent.last_trace.iterations[-1].structured_validation == {
        "status": "SUCCESS",
        "model": "ArchitectureReport",
    }


# ---------------------------------------------------------------------------
# 5 — an envelope with nothing valid inside still fails explicitly
# ---------------------------------------------------------------------------


def test_envelope_without_a_valid_instance_fails_explicitly():
    payload = {"structured_output": {"question": QUESTION}}

    with pytest.raises(StructuredOutputValidationError) as excinfo:
        validate_structured(ArchitectureReport, payload)

    assert "ArchitectureReport" in str(excinfo.value)
    assert excinfo.value.errors  # the field-level detail travels

    dispatcher = FakeDispatcher(_read_result("src/alpha.py"))
    agent = _agent(
        [_tool_response(_read("src/alpha.py")), _structured(payload)],
        dispatcher,
        max_tool_calls=4,
    )
    with pytest.raises(StructuredOutputValidationError):
        agent.run_result(QUESTION)

    trace = agent.last_trace
    assert trace.termination_reason == (
        "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    )
    iteration = trace.iterations[-1]
    assert iteration.structured_validation["status"] == "FAILED"
    # nothing was invented: the failure still names the missing fields
    assert "question" in iteration.structured_validation["error"] or (
        "repository_scope" in iteration.structured_validation["error"]
    )
