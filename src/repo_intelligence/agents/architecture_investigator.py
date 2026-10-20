"""Architecture Investigator — a read-only architecture specialist.

One sentence (spec §4): a read-only specialist that investigates a
repository to collect evidence about its architecture and produces a
structured, evidence-backed architecture report.

Read this file top to bottom to understand the capability (spec §51):

* ``__init__``  — injects the existing ``Agent`` and the visible tool
  list (composition over inheritance, §52; no factory, no base class).
* ``investigate`` — DIRECT mode (§30): versioned prompt + question →
  existing Agent runtime → typed ``ArchitectureReport``.
* ``run_result`` — the M16/M17 investigator protocol (§28/§31): the
  same prompt wrapped around the runner's question, with the structured
  output projected onto the M16 handoff envelope.  The specialist never
  imports or references the runner — no orchestration-aware branch.
* ``validate`` — the deterministic evidence check for a direct-mode
  report, reusing the M13 claim machinery (§24) — never a second
  evidence system.

The specialist owns domain behavior only (§12): no tool loop, no retry
loop, no planner, no orchestration, no tracing, no memory/context
engine — the injected ``Agent`` already runs all of those.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence, Tuple

from ..evidence.models import Claim, ClaimValidation
from ..evidence.validator import gather_evidence, validate_claims
from .evidence_acceptance import enforce_inspected_evidence
from .models.architecture_report import ArchitectureReport
from .prompts.architecture_investigator import (
    ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    build_architecture_investigator_prompt,
    wrap_investigator_question,
)

__all__ = [
    "ARCHITECTURE_TOOLS",
    "ArchitectureInvestigator",
    "ArchitectureInvestigatorError",
]

# The visible tool list (§51: "What tools does it use? One obvious
# list"): the EXISTING read-only inventory — nothing new is added
# (§16/§17).  The application wires the matching dispatcher through
# multiagent.permissions (the investigator role's four tools).
ARCHITECTURE_TOOLS: Tuple[str, ...] = (
    "get_architecture_context",
    "query_engineering_graph",
    "read_file",
    "search_repository",
)


class ArchitectureInvestigatorError(Exception):
    """Explicit specialist failure — a wrong or missing structured
    output, never a silently repaired report."""


class ArchitectureInvestigator:
    """Investigates repository architecture using read-only evidence
    tools (spec §10/§11)."""

    def __init__(
        self,
        agent: Any,
        tools: Optional[Sequence[str]] = None,
        prompt: str = ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    ) -> None:
        self.agent = agent
        self.tools: List[str] = list(
            ARCHITECTURE_TOOLS if tools is None else tools
        )
        self.prompt = prompt

    # -- M16/M17 runner protocol (delegation only — the specialist
    #    never learns that a runner exists) --------------------------

    @property
    def last_trace(self) -> Any:
        """The underlying Agent's own trace — the ONE trace system."""
        return getattr(self.agent, "last_trace", None)

    def run_result(
        self,
        question: str,
        *,
        max_tool_calls: Optional[int] = None,
    ) -> Any:
        """Run the runner's question under the specialist's
        instructions and return its run result.

        ``max_tool_calls`` (M20, default ``None``) is the per-call
        tool-budget override passed straight through to the underlying
        Agent — the specialist never reinterprets it and never mutates
        the agent's configured budget.

        The structured output is projected onto the M16
        ``InvestigationReport`` envelope (spec §27) so the typed handoff
        to the Analyst keeps every component and relationship as an
        evidence-backed statement.  A wrong or missing output passes
        through untouched — the runner records it as the failure it is.
        """
        if max_tool_calls is None:
            result = self.agent.run_result(
                wrap_investigator_question(self.prompt, question)
            )
        else:
            result = self.agent.run_result(
                wrap_investigator_question(self.prompt, question),
                max_tool_calls=max_tool_calls,
            )
        output = getattr(result, "structured_output", None)
        if isinstance(output, ArchitectureReport):
            projected = output.to_investigation_report()
            # M13 acceptance at the handoff boundary: a finding whose
            # evidence this run never inspected cannot stand (a search
            # discovery is not an inspection) — removed and recorded
            # as an actionable uncertainty, status forced incomplete.
            cleaned, _stripped = enforce_inspected_evidence(
                projected, self.last_trace
            )
            result = dataclasses.replace(
                result, structured_output=cleaned
            )
        return result

    # -- direct execution (§30) ---------------------------------------

    def investigate(
        self, question: str, repository_scope: str, commit_scope: str
    ) -> ArchitectureReport:
        """Answer one architecture question: versioned prompt + task →
        existing Agent runtime → typed ``ArchitectureReport``."""
        prompt = build_architecture_investigator_prompt(
            question=question,
            repository_scope=repository_scope,
            commit_scope=commit_scope,
            instructions=self.prompt,
        )
        result = self.agent.run_result(prompt)
        output = getattr(result, "structured_output", None)
        if not isinstance(output, ArchitectureReport):
            raise ArchitectureInvestigatorError(
                "expected ArchitectureReport structured output, got "
                "{!r}".format(type(output).__name__)
            )
        return output

    def validate(self, report: ArchitectureReport) -> List[ClaimValidation]:
        """Deterministic evidence check for a direct-mode report: every
        statement's references must be artifacts this run actually
        inspected (M13 ``validate_claims`` with ``require_inspected``,
        over the SAME handoff projection the Analyst would receive)."""
        trace = self.last_trace
        if trace is None:
            raise ArchitectureInvestigatorError(
                "no investigation trace to validate against — call "
                "investigate() first"
            )
        evidence = gather_evidence(trace)
        claims: List[Claim] = [
            Claim(
                statement=finding.statement,
                evidence_refs=tuple(
                    ref.to_evidence_ref()
                    for ref in finding.evidence_refs
                ),
            )
            for finding in report.to_investigation_report().findings
            if finding.evidence_refs
        ]
        return validate_claims(claims, evidence, require_inspected=True)
