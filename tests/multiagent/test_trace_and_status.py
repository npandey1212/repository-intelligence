"""M16 task status machine + trace collaboration section
(spec §21, §16 collaboration).

Status: explicit, minimal, terminal states sealed — an illegal
transition raises instead of corrupting the task record.

Trace: the ONE extension for M16 — ``collaboration_events`` on the
same execution trace (no second trace system). Existing traces are
byte-identical: the field defaults empty and is omitted from
serialization and from the markdown projection when unused.
"""

from __future__ import annotations

import pytest

from repo_intelligence.agent.trace import (
    COLLABORATION_OPERATIONS,
    AgentTrace,
    render_markdown,
)
from repo_intelligence.multiagent.status import (
    BUDGET_EXCEEDED,
    COMPLETED,
    FAILED,
    INCOMPLETE,
    PENDING,
    RUNNING,
    TIMEOUT,
    VALIDATION_FAILED,
    StatusTransitionError,
    advance,
    can_advance,
    is_terminal,
)


class TestStatusMachine:
    def test_start_run_end(self):
        assert advance(PENDING, RUNNING) == RUNNING
        assert advance(RUNNING, COMPLETED) == COMPLETED

    def test_every_terminal_outcome_reachable_from_running(self):
        for terminal in (
            COMPLETED,
            INCOMPLETE,
            FAILED,
            VALIDATION_FAILED,
            BUDGET_EXCEEDED,
            TIMEOUT,
        ):
            assert can_advance(RUNNING, terminal)

    def test_terminal_states_are_sealed(self):
        for terminal in (
            COMPLETED,
            INCOMPLETE,
            FAILED,
            VALIDATION_FAILED,
            BUDGET_EXCEEDED,
            TIMEOUT,
        ):
            assert is_terminal(terminal)
            assert not can_advance(terminal, RUNNING)
            assert not can_advance(terminal, PENDING)
            with pytest.raises(StatusTransitionError):
                advance(terminal, RUNNING)

    def test_no_skip_from_pending(self):
        assert not can_advance(PENDING, COMPLETED)
        with pytest.raises(StatusTransitionError):
            advance(PENDING, COMPLETED)

    def test_unknown_status_rejected(self):
        with pytest.raises(StatusTransitionError):
            advance("EXHAUSTED", COMPLETED)


class TestCollaborationTrace:
    def test_vocabulary_is_closed(self):
        assert COLLABORATION_OPERATIONS == (
            "agent.start",
            "agent.end",
            "message.send",
            "message.receive",
            "analysis.start",
            "analysis.complete",
            "validator.start",
            "validator.complete",
        )

    def test_default_trace_has_no_collaboration_section(self):
        trace = AgentTrace.start("question")
        assert trace.collaboration_events == []
        payload = trace.to_dict()
        assert "collaboration_events" not in payload
        assert "## Collaboration" not in render_markdown(trace)

    def test_events_appended_in_order(self):
        trace = AgentTrace.start("question")
        trace.record_collaboration_event(
            {"operation": "agent.start", "sequence": 1, "actor": "investigator"}
        )
        trace.record_collaboration_event(
            {
                "operation": "message.send",
                "sequence": 2,
                "actor": "investigator",
                "message_type": "INVESTIGATION_RESULT",
            }
        )
        operations = [event["operation"] for event in trace.collaboration_events]
        assert operations == ["agent.start", "message.send"]
        payload = trace.to_dict()
        assert payload["collaboration_events"][0]["sequence"] == 1

    def test_unknown_operation_rejected(self):
        trace = AgentTrace.start("question")
        with pytest.raises(ValueError):
            trace.record_collaboration_event({"operation": "stage.skip"})

    def test_markdown_renders_metadata_only(self):
        trace = AgentTrace.start("question")
        trace.record_collaboration_event(
            {
                "operation": "message.receive",
                "sequence": 3,
                "actor": "analyst",
                "message_type": "ANALYSIS_RESULT",
                "status": "COMPLETED",
            }
        )
        markdown = render_markdown(trace)
        assert "## Collaboration" in markdown
        assert "`message.receive`" in markdown
        assert "seq 3" in markdown
        assert "actor `analyst`" in markdown
        assert "type `ANALYSIS_RESULT`" in markdown
