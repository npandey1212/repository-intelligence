"""FlowDependencyInvestigator — a read-only flow/dependency specialist.

One sentence (spec §1): a read-only specialist that investigates
paths, dependencies, and flows through a repository and produces a
structured, evidence-backed ``FlowDependencyReport``.

Read this file top to bottom to understand the capability:

* ``__init__``  — injects the existing ``Agent`` and the visible tool
  list (composition over inheritance; no factory, no base class).
* ``investigate`` — DIRECT mode: versioned prompt + question →
  existing Agent runtime → typed ``FlowDependencyReport``.
* ``run_result`` — the M16/M17 investigator protocol: the same prompt
  wrapped around the runner's question, with the structured output
  projected onto the M16 handoff envelope.  The specialist never
  imports or references the runner — no orchestration-aware branch.
* ``validate`` — the deterministic evidence check for a direct-mode
  report, reusing the M13 claim machinery — never a second evidence
  system.

Domain boundary (spec §3): this specialist answers "what path,
dependency, or flow is supported by the repository evidence?" —
NOT "what is the overall architecture?" (that remains the
Architecture Investigator's question).

The specialist owns domain behavior only: prompt, tool list, typed
output, protocol projection, evidence check.  It owns no tool loop,
no retry loop, no planner, no orchestration, no tracing, no
memory/context engine — the injected ``Agent`` already runs all of
those.  It does not subclass the Architecture Investigator: the two
specialists share the existing abstractions (tools, envelope, claim
validation, wrap helper) instead of each other.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence, Tuple

from ..evidence.models import Claim, ClaimValidation
from ..evidence.validator import gather_evidence, validate_claims
from .architecture_investigator import ARCHITECTURE_TOOLS
from .evidence_acceptance import enforce_inspected_evidence
from .models.flow_dependency_report import FlowDependencyReport
from .prompts.architecture_investigator import (
    wrap_investigator_question,
)
from .prompts.flow_dependency_investigator import (
    FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V2,
    build_flow_dependency_investigator_prompt,
)

__all__ = [
    "FLOW_DEPENDENCY_TOOLS",
    "FlowDependencyInvestigator",
    "FlowDependencyInvestigatorError",
]

# The visible tool list (spec §12): the SAME existing read-only
# investigator inventory the Architecture Investigator uses — one
# list, never forked (the inventory belongs to the role, not to a
# specialist; ``FLOW_DEPENDENCY_TOOLS is ARCHITECTURE_TOOLS`` is
# asserted by test).  The application wires the matching dispatcher
# through multiagent.permissions (the investigator role's four tools).
FLOW_DEPENDENCY_TOOLS: Tuple[str, ...] = ARCHITECTURE_TOOLS


class FlowDependencyInvestigatorError(Exception):
    """Explicit specialist failure — a wrong or missing structured
    output, never a silently repaired report."""


class FlowDependencyInvestigator:
    """Investigates paths, dependencies, and flows using read-only
    evidence tools (spec §1)."""

    def __init__(
        self,
        agent: Any,
        tools: Optional[Sequence[str]] = None,
        prompt: str = FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V2,
    ) -> None:
        self.agent = agent
        self.tools: List[str] = list(
            FLOW_DEPENDENCY_TOOLS if tools is None else tools
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
        ``InvestigationReport`` envelope so the typed handoff to the
        Analyst keeps every path step and relationship as an
        evidence-backed statement.  A wrong or missing output passes
        through untouched — the runner records it as the failure it
        is.
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
        if isinstance(output, FlowDependencyReport):
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

    # -- direct execution ---------------------------------------------

    def investigate(
        self, question: str, repository_scope: str, commit_scope: str
    ) -> FlowDependencyReport:
        """Answer one flow/dependency question: versioned prompt +
        task → existing Agent runtime → typed ``FlowDependencyReport``."""
        prompt = build_flow_dependency_investigator_prompt(
            question=question,
            repository_scope=repository_scope,
            commit_scope=commit_scope,
            instructions=self.prompt,
        )
        result = self.agent.run_result(prompt)
        output = getattr(result, "structured_output", None)
        if not isinstance(output, FlowDependencyReport):
            raise FlowDependencyInvestigatorError(
                "expected FlowDependencyReport structured output, got "
                "{!r}".format(type(output).__name__)
            )
        return output

    def validate(
        self, report: FlowDependencyReport
    ) -> List[ClaimValidation]:
        """Deterministic evidence check for a direct-mode report:
        every statement's references must be artifacts this run
        actually inspected (M13 ``validate_claims`` with
        ``require_inspected``, over the SAME handoff projection the
        Analyst would receive)."""
        trace = self.last_trace
        if trace is None:
            raise FlowDependencyInvestigatorError(
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
