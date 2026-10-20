"""M19 result collection and answer assembly (deterministic, no LLM).

The explicit assembly stage of the run lifecycle — a distinct,
independently testable component between "typed task results
collected" and "deterministic validation":

    typed task results (plan order)
        → collect completed INVESTIGATOR outputs   (RunAssembler)
        → merge into ONE aggregate InvestigationReport
          (merge_investigation_reports — the ONE merge, moved here
          from runner.py byte-for-byte, still re-exported everywhere)
        → pair with the analyst/specialist finding (AssemblyOutcome)
        → orchestrator runs validate_collaboration (unchanged)

The assembler NEVER invents: no analysis result means
``has_analysis`` is ``False`` and the orchestrator keeps the
documented ``INCOMPLETE / NO_ANALYSIS`` behavior instead of a
fabricated architectural answer. Evidence refs, uncertainties, open
questions and sources all survive the merge (they are the merge's
union/dedup rules — same code, same tests). No model call is added
anywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..multiagent.contracts import (
    ArchitectureFinding,
    InvestigationReport,
    TaskPayload,
)
from .models import STEP_COMPLETED, StepResult
from .roles import INVESTIGATOR

__all__ = [
    "AssemblyOutcome",
    "RunAssembler",
    "merge_investigation_reports",
]


def merge_investigation_reports(
    reports: Sequence[InvestigationReport], task: TaskPayload
) -> InvestigationReport:
    """One evidence package from every Investigator step's report.

    Deterministic merge (spec §31/§32): findings concatenated in step
    order, evidence refs de-duplicated by identity, open questions/
    uncertainties/sources unioned, scope and question taken from the
    SHARED task, and the aggregate ``status`` downgraded to
    ``incomplete`` if any step declared itself incomplete (never
    upgraded). ``investigation_id`` is the FIRST report's id — the
    aggregate keeps the id the first evidence package was produced
    under; per-step reports remain visible on their ``StepResult``.

    Callers must verify each report's scope against the task BEFORE
    merging (the orchestrator does, at step completion) so the shared
    scope can never mask a mis-scoped package.
    """
    if not reports:
        raise ValueError("merge_investigation_reports requires a report")
    seen_refs = set()
    evidence_refs = []
    for report in reports:
        for ref in report.evidence_refs:
            key = (ref.path, ref.line_start, ref.line_end)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            evidence_refs.append(ref)

    def _union(attr: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for report in reports:
            for value in getattr(report, attr):
                if value in seen:
                    continue
                seen.add(value)
                out.append(value)
        return out

    status = "complete"
    if any(report.status == "incomplete" for report in reports):
        status = "incomplete"
    findings = [finding for report in reports for finding in report.findings]
    return InvestigationReport(
        investigation_id=reports[0].investigation_id,
        question=task.question,
        repository_scope=task.repository_scope,
        commit_scope=task.commit_scope,
        findings=findings,
        evidence_refs=evidence_refs,
        open_questions=_union("open_questions"),
        uncertainties=_union("uncertainties"),
        investigated_sources=_union("investigated_sources"),
        status=status,
    )


@dataclass(frozen=True)
class AssemblyOutcome:
    """The assembled (not yet validated) artifacts of one run.

    ``report`` is the aggregated evidence package (``None`` when no
    Investigator task completed); ``finding`` is the analyst's typed
    output (``None`` means no analysis ran — the orchestrator decides
    ``INCOMPLETE / NO_ANALYSIS`` from ``has_analysis``, never from a
    guess).
    """

    report: Optional[InvestigationReport]
    finding: Optional[ArchitectureFinding]

    @property
    def has_analysis(self) -> bool:
        return self.finding is not None


class RunAssembler:
    """Collects typed task outputs in PLAN order and assembles the
    final (pre-validation) artifacts.

    Deterministic and pure over its inputs: given the same step
    results, the same outcome — independent of when (or, in the future,
    in what completion order) the tasks finished, because collection
    follows the step-result list the coordinator commits in plan order.
    """

    def collect_investigator_reports(
        self, step_results: Sequence[StepResult]
    ) -> List[InvestigationReport]:
        """Completed Investigator outputs, in the list's (plan) order.

        Non-Investigator roles, non-completed steps, and non-report
        outputs are excluded — the same selection the M17 merge used.
        """
        return [
            result.output
            for result in step_results
            if result.agent_role == INVESTIGATOR
            and result.status == STEP_COMPLETED
            and isinstance(result.output, InvestigationReport)
        ]

    def merge_reports(
        self, reports: Sequence[InvestigationReport], task: TaskPayload
    ) -> InvestigationReport:
        """The ONE deterministic merge (raises on an empty list —
        callers decide whether absence of reports is an error or a
        ``None`` outcome, via ``assemble``)."""
        return merge_investigation_reports(reports, task)

    def assemble(
        self,
        step_results: Sequence[StepResult],
        task: TaskPayload,
        finding: Optional[ArchitectureFinding],
    ) -> AssemblyOutcome:
        """Pair the aggregated report with the finding (plan order)."""
        reports = self.collect_investigator_reports(step_results)
        report = (
            merge_investigation_reports(reports, task) if reports else None
        )
        return AssemblyOutcome(report=report, finding=finding)
