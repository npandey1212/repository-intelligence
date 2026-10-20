"""M16 typed inter-agent contracts (spec §7/§22/§23).

Contract validity is deterministic: malformed payloads, wrong payload
types for a message type, scope-field defaults, envelope/provenance
disagreement, and out-of-aggregate evidence refs are all rejected at
the boundary — never coerced, never defaulted, never accepted as raw
dictionaries. Everything here is LLM-free.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from repo_intelligence.evidence.models import EvidenceRef
from repo_intelligence.multiagent.contracts import (
    CLASSIFICATIONS,
    MESSAGE_TYPES,
    ArchitectureFinding,
    ContractError,
    InvestigationReport,
    ReportEvidenceRef,
    TaskPayload,
    make_message,
    message_from_dict,
    message_to_dict,
    report_json,
)

COMMIT = "c" * 40


def _ref(path: str = "README.md", start: int = 1, end: int = 9):
    return ReportEvidenceRef(path=path, line_start=start, line_end=end)


def _report(**overrides):
    data = {
        "investigation_id": "inv-1",
        "question": "How do the agents interact?",
        "repository_scope": "example-repository",
        "commit_scope": COMMIT,
        "findings": [
            {
                "statement": "agent.py wires the dispatcher",
                "evidence_refs": [_ref().dict()],
                "support": "SUPPORTED",
            }
        ],
        "evidence_refs": [_ref().dict()],
        "investigated_sources": ["agent.py"],
    }
    data.update(overrides)
    return InvestigationReport(**data)


def _finding(report: InvestigationReport, **overrides):
    data = {
        "investigation_id": report.investigation_id,
        "question": report.question,
        "repository_scope": report.repository_scope,
        "commit_scope": report.commit_scope,
        "conclusions": [
            {
                "statement": "the boundary is application-owned",
                "classification": "observed",
                "evidence_refs": [r.dict() for r in report.evidence_refs],
            }
        ],
        "evidence_refs": [r.dict() for r in report.evidence_refs],
        "summary": "One dispatcher, one boundary.",
    }
    data.update(overrides)
    return ArchitectureFinding(**data)


class TestArchitectureFindingProvenance:
    def test_source_investigation_ids_default_for_legacy_producers(self):
        report = _report()
        finding = _finding(report)
        assert finding.source_investigation_ids == []

    def test_source_investigation_ids_round_trip(self):
        report = _report()
        expected = [report.investigation_id, "p2-report", "p3-report"]
        finding = _finding(
            report,
            source_investigation_ids=expected,
        )
        assert finding.source_investigation_ids == expected
        assert finding.model_dump()["source_investigation_ids"] == expected



class TestEvidenceRefParity:
    def test_single_line_bound_rejected(self):
        with pytest.raises(ValidationError):
            ReportEvidenceRef(path="README.md", line_start=1)

    def test_start_after_end_rejected(self):
        with pytest.raises(ValidationError):
            ReportEvidenceRef(path="README.md", line_start=9, line_end=2)

    def test_both_bounds_or_neither(self):
        neither = ReportEvidenceRef(path="README.md")
        assert neither.line_start is None and neither.line_end is None
        both = ReportEvidenceRef(path="README.md", line_start=3, line_end=5)
        assert (both.line_start, both.line_end) == (3, 5)

    def test_converts_to_evidence_ref_type(self):
        ref = _ref("src/agent.py", 10, 20)
        evidence_ref = ref.to_evidence_ref()
        assert isinstance(evidence_ref, EvidenceRef)
        assert (
            evidence_ref.path,
            evidence_ref.line_start,
            evidence_ref.line_end,
        ) == ("src/agent.py", 10, 20)
        assert (
            ReportEvidenceRef.from_evidence_ref(evidence_ref).dict()
            == ref.dict()
        )


class TestInvestigationReport:
    def test_valid_report_is_typed(self):
        report = _report()
        assert isinstance(report, InvestigationReport)
        assert report.status == "complete"
        assert report.findings[0].support == "SUPPORTED"
        assert isinstance(report.findings[0].evidence_refs[0], ReportEvidenceRef)

    def test_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            InvestigationReport(question="q", findings=[])

    def test_empty_investigation_id_rejected(self):
        with pytest.raises(ValidationError):
            _report(investigation_id="")

    def test_empty_scope_rejected(self):
        with pytest.raises(ValidationError):
            _report(repository_scope="")

    def test_unknown_support_rejected(self):
        with pytest.raises(ValidationError):
            _report(
                findings=[
                    {
                        "statement": "s",
                        "evidence_refs": [_ref().dict()],
                        "support": "PROBABLY",
                    }
                ]
            )

    def test_finding_refs_are_added_to_derived_aggregate(self):
        report = _report(
            evidence_refs=[],
            findings=[
                {
                    "statement": "s",
                    "evidence_refs": [_ref("src/missing.py", 1, 2).model_dump()],
                }
            ],
        )
        assert _ref("src/missing.py", 1, 2) in report.evidence_refs

    def test_aggregate_keeps_existing_refs_and_adds_exact_finding_refs(self):
        broad = _ref("SKILLS.md", 1, 40)
        narrow = _ref("SKILLS.md", 1, 2)
        report = _report(
            evidence_refs=[broad.model_dump()],
            findings=[{"statement": "s", "evidence_refs": [narrow.model_dump()]}],
        )
        assert broad in report.evidence_refs
        assert narrow in report.evidence_refs

    def test_report_json_is_deterministic(self):
        report = _report()
        assert report_json(report) == report_json(_report())


class TestArchitectureFinding:
    def test_requires_matching_investigation_id(self):
        report = _report()
        finding = _finding(report)
        assert finding.investigation_id == report.investigation_id

    def test_unknown_classification_rejected(self):
        report = _report()
        with pytest.raises(ValidationError):
            _finding(
                report,
                conclusions=[
                    {
                        "statement": "s",
                        "classification": "vibes",
                        "evidence_refs": [r.dict() for r in report.evidence_refs],
                    }
                ],
            )

    def test_conclusion_ref_outside_aggregate_rejected(self):
        report = _report()
        with pytest.raises(ValidationError):
            _finding(
                report,
                conclusions=[
                    {
                        "statement": "s",
                        "classification": "observed",
                        "evidence_refs": [_ref("nope.py", 1, 2).dict()],
                    }
                ]
            )

    def test_classification_vocabulary(self):
        assert CLASSIFICATIONS == (
            "observed",
            "derived",
            "inferred",
            "conflicting",
            "unknown",
        )


class TestMessageContract:
    def test_six_message_types(self):
        assert MESSAGE_TYPES == (
            "INVESTIGATION_REQUEST",
            "INVESTIGATION_RESULT",
            "ANALYSIS_REQUEST",
            "ANALYSIS_RESULT",
            "VALIDATION_REQUEST",
            "VALIDATION_RESULT",
        )

    def test_payload_dict_is_parsed_to_typed_instance(self):
        report = _report()
        message = make_message(
            "INVESTIGATION_RESULT",
            "investigator",
            "analyst",
            report.dict(),
            "example-repository",
            COMMIT,
        )
        assert isinstance(message.payload, InvestigationReport)
        assert message.evidence_refs == report.evidence_refs

    def test_envelope_refs_are_taken_from_payload(self):
        report = _report()
        message = make_message(
            "ANALYSIS_REQUEST",
            "task",
            "analyst",
            report,
            "example-repository",
            COMMIT,
        )
        assert len(message.evidence_refs) == len(report.evidence_refs)

    def test_wrong_payload_type_rejected(self):
        payload = TaskPayload(
            question="q", repository_scope="example-repository", commit_scope=COMMIT
        )
        with pytest.raises(ContractError):
            make_message(
                "INVESTIGATION_RESULT",
                "investigator",
                "analyst",
                payload,
                "example-repository",
                COMMIT,
            )

    def test_unknown_message_type_rejected(self):
        with pytest.raises(ContractError):
            make_message(
                "SMOKE_SIGNAL",
                "task",
                "investigator",
                {},
                "example-repository",
                COMMIT,
            )

    def test_envelope_refs_disagreeing_with_payload_rejected(self):
        report = _report()
        message = make_message(
            "INVESTIGATION_RESULT",
            "investigator",
            "analyst",
            report,
            "example-repository",
            COMMIT,
        )
        data = message_to_dict(message)
        data["evidence_refs"] = [_ref("forged.py", 1, 2).dict()]
        with pytest.raises(ContractError):
            message_from_dict(data)

    def test_malformed_message_rejected_deterministically(self):
        with pytest.raises(ContractError):
            message_from_dict(
                {
                    "message_id": "m1",
                    "sender_agent": "task",
                    "recipient_agent": "analyst",
                    "message_type": "ANALYSIS_RESULT",
                    "payload": {"not": "a finding"},
                    "repository_scope": "example-repository",
                    "commit_scope": COMMIT,
                }
            )

    def test_unknown_participant_rejected(self):
        with pytest.raises(ContractError):
            make_message(
                "INVESTIGATION_REQUEST",
                "boss",
                "investigator",
                TaskPayload(
                    question="q",
                    repository_scope="example-repository",
                    commit_scope=COMMIT,
                ),
                "example-repository",
                COMMIT,
            )

    def test_empty_scope_hop_rejected(self):
        report = _report()
        with pytest.raises(ContractError):
            make_message(
                "INVESTIGATION_RESULT",
                "investigator",
                "analyst",
                report,
                "",
                COMMIT,
            )

    def test_round_trip_preserves_typed_payload(self):
        report = _report()
        message = make_message(
            "INVESTIGATION_RESULT",
            "investigator",
            "analyst",
            report,
            "example-repository",
            COMMIT,
        )
        parsed = message_from_dict(message_to_dict(message))
        assert parsed.payload == report
        assert parsed.message_id == message.message_id
