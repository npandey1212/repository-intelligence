"""M16 deterministic collaboration validation (spec §22 — no LLM).

Runs AFTER both agents produced their typed outputs and BEFORE the
task is marked ``COMPLETED``: eight deterministic rules over the
already-recorded state, reusing the ONE existing evidence system
(M13 ``gather_evidence`` + ``validate_claims``) instead of a second
implementation.

The eight rules (order = report order):

1. ``message_contract``      — the four-stage typed handoff exists in
                               order with the expected sender/recipient
                               routing and payload types.
2. ``scope_match``           — task, report, and finding describe the
                               SAME repository and commit (a hop can
                               never quietly change what is under
                               analysis).
3. ``evidence_structural``   — every evidence ref on every model
                               converts to a well-formed
                               ``evidence.EvidenceRef`` (bounds rule,
                               non-empty path).
4. ``collaboration``         — the finding carries the report's
                               ``investigation_id`` (proof the Analyst
                               consumed the Investigator's contract)
                               and the ANALYSIS_REQUEST carried that
                               same report.
5. ``evidence_handoff``      — finding refs ⊆ report refs and every
                               conclusion ref ⊆ report refs (nothing
                               fabricated at either hop).
6. ``evidence_valid``        — every Analyst conclusion validates as
                               SUPPORTED against the INVESTIGATOR's
                               gathered evidence with
                               ``require_inspected=True`` (the
                               Analyst produced no tool evidence of
                               its own — that is the point).
7. ``classification``        — ``observed`` needs ≥1 supported ref;
                               ``conflicting`` needs ≥2 distinct
                               supported refs (evidence vocabulary
                               never claims more than it has).
8. ``output_structure``      — the report carries findings and the
                               finding carries conclusions (an empty
                               workflow is never a success).

Tool failures observed in either trace are collected onto the result
(visible, non-fatal per spec §20) — a failed optional lookup does not
invalidate an otherwise proven collaboration. Overall status is
``passed`` only when every check passes; the runner maps ``failed``
to ``VALIDATION_FAILED`` (and an ``incomplete`` declared status to
``INCOMPLETE`` — statuses are the runner's job, not this module's).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ..evidence.models import (
    STATUS_SUPPORTED,
    Claim,
    EvidenceRef,
)
from ..evidence.validator import gather_evidence, validate_claims
from .contracts import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
    ToolFailure,
    ValidationCheck,
    ValidationResult,
)

__all__ = [
    "CLASSIFICATION_RULE",
    "COLLABORATION_RULE",
    "EVIDENCE_HANDOFF_RULE",
    "EVIDENCE_STRUCTURAL_RULE",
    "EVIDENCE_VALID_RULE",
    "MESSAGE_CONTRACT_RULE",
    "OUTPUT_STRUCTURE_RULE",
    "RULE_ORDER",
    "SCOPE_MATCH_RULE",
    "collect_tool_failures",
    "validate_collaboration",
]

MESSAGE_CONTRACT_RULE = "message_contract"
SCOPE_MATCH_RULE = "scope_match"
EVIDENCE_STRUCTURAL_RULE = "evidence_structural"
COLLABORATION_RULE = "collaboration"
EVIDENCE_HANDOFF_RULE = "evidence_handoff"
EVIDENCE_VALID_RULE = "evidence_valid"
CLASSIFICATION_RULE = "classification"
OUTPUT_STRUCTURE_RULE = "output_structure"

RULE_ORDER: Tuple[str, ...] = (
    MESSAGE_CONTRACT_RULE,
    SCOPE_MATCH_RULE,
    EVIDENCE_STRUCTURAL_RULE,
    COLLABORATION_RULE,
    EVIDENCE_HANDOFF_RULE,
    EVIDENCE_VALID_RULE,
    CLASSIFICATION_RULE,
    OUTPUT_STRUCTURE_RULE,
)

# (message_type, sender, recipient) the runner must have produced, in
# order — the machine-checkable shape of the M16 workflow.
_EXPECTED_HANDOFFS: Tuple[Tuple[str, str, str], ...] = (
    ("INVESTIGATION_REQUEST", "task", "investigator"),
    ("INVESTIGATION_RESULT", "investigator", "analyst"),
    ("ANALYSIS_REQUEST", "task", "analyst"),
    ("ANALYSIS_RESULT", "analyst", "task"),
)

_MAX_DETAIL = 400


def _clip(text: str, limit: int = _MAX_DETAIL) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "\u2026"


def _check(rule: str, ok: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(rule=rule, ok=ok, detail="" if ok else _clip(detail))


def _ref_keys(refs: Sequence[Any]) -> set:
    return {
        (ref.path, ref.line_start, ref.line_end)
        for ref in refs
    }


def _all_refs(
    report: InvestigationReport, finding: ArchitectureFinding
) -> List[Tuple[str, EvidenceRef]]:
    """(where, evidence_ref) for every ref on both models."""
    out: List[Tuple[str, EvidenceRef]] = []
    for ref in report.evidence_refs:
        out.append(("report.evidence_refs", ref.to_evidence_ref()))
    for index, item in enumerate(report.findings):
        for ref in item.evidence_refs:
            out.append(
                ("report.findings[{}].evidence_refs".format(index),
                 ref.to_evidence_ref())
            )
    for ref in finding.evidence_refs:
        out.append(("finding.evidence_refs", ref.to_evidence_ref()))
    for index, conclusion in enumerate(finding.conclusions):
        for ref in conclusion.evidence_refs:
            out.append(
                ("finding.conclusions[{}].evidence_refs".format(index),
                 ref.to_evidence_ref())
            )
    return out


def _claims_of(finding: ArchitectureFinding) -> List[Claim]:
    """Analyst conclusions as the ONE existing claim type (statement +
    refs) — no parallel claim model, nothing extracted from prose."""
    return [
        Claim(
            statement=conclusion.statement,
            evidence_refs=tuple(
                ref.to_evidence_ref() for ref in conclusion.evidence_refs
            ),
        )
        for conclusion in finding.conclusions
    ]


def collect_tool_failures(
    traces: Sequence[Tuple[str, Any]],
) -> List[ToolFailure]:
    """Failed tool attempts from the given (role, trace) pairs —
    surfaced on the validation result (spec §20: visible in trace AND
    result; never hidden, never fatal by itself)."""
    failures: List[ToolFailure] = []
    for role, trace in traces:
        if trace is None:
            continue
        index = gather_evidence(trace)
        for attempt in index.failed:
            failures.append(
                ToolFailure(
                    agent=role,
                    tool=attempt.tool,
                    error_code=attempt.error_code or "TOOL_FAILED",
                )
            )
    return failures


def _rule_message_contract(
    task: TaskPayload, messages: Sequence[Any]
) -> ValidationCheck:
    observed = [
        (message.message_type, message.sender_agent, message.recipient_agent)
        for message in messages
        if getattr(message, "message_type", None) in {h[0] for h in _EXPECTED_HANDOFFS}
    ]
    expected = list(_EXPECTED_HANDOFFS)
    if observed[: len(expected)] == expected:
        return _check(MESSAGE_CONTRACT_RULE, True, "")
    return _check(
        MESSAGE_CONTRACT_RULE,
        False,
        "expected handoffs {} but observed {}".format(
            ["{}/{}/{}".format(*h) for h in expected],
            ["{}/{}/{}".format(*h) for h in observed],
        ),
    )


def _rule_scope_match(
    task: TaskPayload,
    report: InvestigationReport,
    finding: ArchitectureFinding,
) -> ValidationCheck:
    scopes = {
        "task": (task.repository_scope, task.commit_scope),
        "report": (report.repository_scope, report.commit_scope),
        "finding": (finding.repository_scope, finding.commit_scope),
    }
    if len(set(scopes.values())) == 1:
        return _check(SCOPE_MATCH_RULE, True, "")
    return _check(
        SCOPE_MATCH_RULE,
        False,
        "scope mismatch: {}".format(
            "; ".join(
                "{}=repository {} commit {}".format(
                    name, value[0], value[1]
                )
                for name, value in sorted(scopes.items())
            )
        ),
    )


def _rule_evidence_structural(
    report: InvestigationReport, finding: ArchitectureFinding
) -> ValidationCheck:
    errors: List[str] = []
    for where, ref in _all_refs(report, finding):
        if not isinstance(ref.path, str) or not ref.path:
            errors.append("{}: empty path".format(where))
            continue
        if ref.line_start is not None and ref.line_end is not None:
            if ref.line_start > ref.line_end:
                errors.append(
                    "{}: line_start > line_end ({})".format(where, ref.render())
                )
    if not errors:
        return _check(EVIDENCE_STRUCTURAL_RULE, True, "")
    return _check(EVIDENCE_STRUCTURAL_RULE, False, "; ".join(errors))


def _rule_collaboration(
    report: InvestigationReport,
    finding: ArchitectureFinding,
    messages: Sequence[Any],
) -> ValidationCheck:
    errors: List[str] = []
    if finding.investigation_id != report.investigation_id:
        errors.append(
            "finding.investigation_id {!r} != report.investigation_id "
            "{!r}".format(finding.investigation_id, report.investigation_id)
        )
    for message in messages:
        if (
            message.message_type == "ANALYSIS_REQUEST"
            and message.payload.investigation_id != report.investigation_id
        ):
            errors.append(
                "ANALYSIS_REQUEST payload id {!r} != report id {!r}".format(
                    message.payload.investigation_id, report.investigation_id
                )
            )
        if (
            message.message_type == "INVESTIGATION_RESULT"
            and message.payload != report
        ):
            errors.append(
                "INVESTIGATION_RESULT payload differs from the validated "
                "report"
            )
    if not errors:
        return _check(COLLABORATION_RULE, True, "")
    return _check(COLLABORATION_RULE, False, "; ".join(errors))


def _rule_evidence_handoff(
    report: InvestigationReport, finding: ArchitectureFinding
) -> ValidationCheck:
    report_keys = _ref_keys(report.evidence_refs)
    finding_keys = _ref_keys(finding.evidence_refs)
    errors: List[str] = []
    missing_from_report = finding_keys - report_keys
    if missing_from_report:
        errors.append(
            "{} finding ref(s) missing from report aggregate".format(
                len(missing_from_report)
            )
        )
    for index, conclusion in enumerate(finding.conclusions):
        for key in _ref_keys(conclusion.evidence_refs):
            if key not in report_keys:
                errors.append(
                    "conclusion[{}] ref {} not in report aggregate".format(
                        index, key
                    )
                )
    if not errors:
        return _check(EVIDENCE_HANDOFF_RULE, True, "")
    return _check(EVIDENCE_HANDOFF_RULE, False, "; ".join(errors))


def _rule_evidence_valid(
    finding: ArchitectureFinding,
    investigator_trace: Any,
) -> ValidationCheck:
    evidence = gather_evidence(investigator_trace)
    validations = validate_claims(
        _claims_of(finding), evidence, require_inspected=True
    )
    failures = [
        "{} -> {} ({})".format(
            _clip(validation.claim.statement, 80),
            validation.status,
            validation.reason,
        )
        for validation in validations
        if validation.status != STATUS_SUPPORTED
    ]
    if not failures:
        return _check(EVIDENCE_VALID_RULE, True, "")
    return _check(
        EVIDENCE_VALID_RULE,
        False,
        "{} of {} conclusion(s) not supported: {}".format(
            len(failures), len(validations), "; ".join(failures)
        ),
    )


def _rule_classification(
    finding: ArchitectureFinding,
    investigator_trace: Any,
) -> ValidationCheck:
    evidence = gather_evidence(investigator_trace)
    validations = validate_claims(
        _claims_of(finding), evidence, require_inspected=True
    )
    errors: List[str] = []
    for conclusion, validation in zip(finding.conclusions, validations):
        distinct_paths = {
            item.path for item in validation.matched_evidence
        }
        if conclusion.classification == "observed" and not validation.matched_evidence:
            errors.append(
                "observed without supported evidence: {}".format(
                    _clip(conclusion.statement, 80)
                )
            )
        if conclusion.classification == "conflicting" and len(distinct_paths) < 2:
            errors.append(
                "conflicting needs >= 2 distinct supported paths, got {}: "
                "{}".format(
                    len(distinct_paths), _clip(conclusion.statement, 80)
                )
            )
    if not errors:
        return _check(CLASSIFICATION_RULE, True, "")
    return _check(CLASSIFICATION_RULE, False, "; ".join(errors))


def _rule_output_structure(
    report: InvestigationReport, finding: ArchitectureFinding
) -> ValidationCheck:
    errors: List[str] = []
    if not report.findings:
        errors.append("report carries no findings")
    if not finding.conclusions:
        errors.append("finding carries no conclusions")
    if not report.evidence_refs:
        errors.append("report carries no evidence_refs")
    if not errors:
        return _check(OUTPUT_STRUCTURE_RULE, True, "")
    return _check(OUTPUT_STRUCTURE_RULE, False, "; ".join(errors))


def validate_collaboration(
    task: TaskPayload,
    report: InvestigationReport,
    finding: ArchitectureFinding,
    investigator_trace: Any,
    messages: Sequence[Any] = (),
    analyst_trace: Any = None,
) -> ValidationResult:
    """Run all eight rules; never raises for a rule outcome — a failed
    collaboration is a ``ValidationResult(status="failed")``, not an
    exception (protocol construction errors raised earlier at the
    message boundary).

    ``investigator_trace`` is the evidence authority: it is the trace
    the Analyst's conclusions must resolve against.
    """
    checks = [
        _rule_message_contract(task, messages),
        _rule_scope_match(task, report, finding),
        _rule_evidence_structural(report, finding),
        _rule_collaboration(report, finding, messages),
        _rule_evidence_handoff(report, finding),
        _rule_evidence_valid(finding, investigator_trace),
        _rule_classification(finding, investigator_trace),
        _rule_output_structure(report, finding),
    ]
    tool_failures = collect_tool_failures(
        [
            ("investigator", investigator_trace),
            ("analyst", analyst_trace),
        ]
    )
    return ValidationResult(
        investigation_id=report.investigation_id,
        status="passed" if all(check.ok for check in checks) else "failed",
        checks=checks,
        tool_failures=tool_failures,
    )
