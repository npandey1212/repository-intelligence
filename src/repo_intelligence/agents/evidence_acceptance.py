"""Inspected-evidence acceptance for orchestrated investigator reports.

The M13 authority (``gather_evidence`` + ``validate_claims`` with
``require_inspected``) already decides whether a claim's references
resolve to evidence this run actually *read*.  Direct-mode callers get
that verdict from each specialist's ``validate``; diagram validation
applies it to the Analyst's spec.  This module applies the SAME
authority at the one place it was never wired: the orchestrated
``run_result`` path, where a schema-valid report used to cross the
handoff boundary with findings whose evidence the run never inspected
(a search hit's path/line range is a discovery, never an inspection).

Policy — deterministic, removal-only, trace-only:

* a finding with no evidence references is an honest statement of
  absence and is kept untouched;
* a finding whose references do not validate as inspected-and-covering
  cannot stand as an established finding: it is removed and recorded
  as an actionable uncertainty naming the cited source and the next
  action (``read_file`` on that range);
* a finding with at least one inspected-and-covering reference is
  kept whole (mixed references are the model's own "any of these
  back it" set — the validator already resolved it);
* if anything was removed the report's ``status`` is forced to
  ``incomplete`` (a report that lost findings is not complete);
* the aggregate ``evidence_refs`` is rebuilt from the kept findings
  plus any original aggregate reference that individually validates
  as inspected — never a reference this run did not read.

Never dispatches a tool, never calls a model, never adds a claim.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from ..evidence.models import (
    STATUS_SUPPORTED,
    Claim,
    EvidenceIndex,
)
from ..evidence.validator import gather_evidence, validate_claims
from ..multiagent.contracts import (
    InvestigationFinding,
    InvestigationReport,
    ReportEvidenceRef,
)

__all__ = ["enforce_inspected_evidence"]

_UNRESOLVED_PREFIX = "unresolved (evidence never inspected): "
_NEXT_ACTION = (
    " — inspect the cited source with read_file before establishing "
    "this claim"
)


def _render_ref(ref: ReportEvidenceRef) -> str:
    if ref.line_start is None:
        return ref.path
    return "{}:{}-{}".format(ref.path, ref.line_start, ref.line_end)


def _claim_for(finding: InvestigationFinding) -> Claim:
    return Claim(
        statement=finding.statement,
        evidence_refs=tuple(
            ref.to_evidence_ref() for ref in finding.evidence_refs
        ),
    )


def _ref_is_inspected(
    ref: ReportEvidenceRef, evidence: EvidenceIndex
) -> bool:
    validation = validate_claims(
        [_claim_for(InvestigationFinding(
            statement="ref", evidence_refs=[ref]
        ))],
        evidence,
        require_inspected=True,
    )[0]
    return validation.status == STATUS_SUPPORTED


def enforce_inspected_evidence(
    report: InvestigationReport, trace: Any
) -> Tuple[InvestigationReport, List[InvestigationFinding]]:
    """Return ``(accepted_report, stripped_findings)``.

    ``accepted_report`` is the input unchanged when every evidenced
    finding validates against the run's inspected evidence; otherwise
    it is a new report with the unestablished findings removed, their
    statements recorded as actionable uncertainties, the status
    forced to ``incomplete``, and the aggregate rebuilt.  The second
    element lists exactly what was removed (empty when unchanged) so
    callers can observe the decision without re-deriving it.
    """
    if trace is None:
        return report, []
    evidenced = [
        finding for finding in report.findings if finding.evidence_refs
    ]
    if not evidenced:
        return report, []
    evidence = gather_evidence(trace)
    validations = validate_claims(
        [_claim_for(finding) for finding in evidenced],
        evidence,
        require_inspected=True,
    )
    stripped_keys = {
        (validation.claim.statement, index)
        for index, validation in enumerate(validations)
        if validation.status != STATUS_SUPPORTED
    }
    if not stripped_keys:
        return report, []

    kept: List[InvestigationFinding] = []
    stripped: List[InvestigationFinding] = []
    evidenced_index = 0
    for finding in report.findings:
        if not finding.evidence_refs:
            kept.append(finding)
            continue
        if (
            finding.statement,
            evidenced_index,
        ) in stripped_keys:
            stripped.append(finding)
        else:
            kept.append(finding)
        evidenced_index += 1

    notes = [
        _UNRESOLVED_PREFIX
        + finding.statement
        + " (cited: "
        + ", ".join(_render_ref(ref) for ref in finding.evidence_refs)
        + ")"
        + _NEXT_ACTION
        for finding in stripped
    ]
    kept_keys = {
        (ref.path, ref.line_start, ref.line_end)
        for finding in kept
        for ref in finding.evidence_refs
    }
    aggregate: List[ReportEvidenceRef] = []
    seen = set()
    for finding in kept:
        for ref in finding.evidence_refs:
            key = (ref.path, ref.line_start, ref.line_end)
            if key not in seen:
                seen.add(key)
                aggregate.append(ref)
    for ref in report.evidence_refs:
        key = (ref.path, ref.line_start, ref.line_end)
        if key in kept_keys:
            continue
        if _ref_is_inspected(ref, evidence):
            kept_keys.add(key)
            aggregate.append(ref)

    cleaned = InvestigationReport(
        investigation_id=report.investigation_id,
        question=report.question,
        repository_scope=report.repository_scope,
        commit_scope=report.commit_scope,
        findings=kept,
        evidence_refs=aggregate,
        open_questions=list(report.open_questions),
        uncertainties=list(report.uncertainties) + notes,
        investigated_sources=list(report.investigated_sources),
        status="incomplete",
    )
    return cleaned, stripped
