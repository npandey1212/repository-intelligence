"""M17 agent role assignment (spec §14/§15/§36/§37 — smallest thing).

The deterministic mapping from a goal-level ``PlanStep`` to the M16
agent role that executes it. Application-owned: no dynamic discovery,
no marketplace, no runtime registration, no LLM-authored roles.

Two inputs decide a role:

1. **Explicit application overrides** — ``role_overrides`` maps a step
   id to a role; an unsupported role is REJECTED (``ValueError``), so
   a typo can never silently fall through to a default.
2. **The default classifier** — a short, generic, repository-neutral
   keyword pattern over the step goal. Goals that ask for
   interpretation/synthesis ("explain", "architecture explanation",
   "synthesize", "summarize", "what does … mean") go to the ANALYST;
   everything else goes to the INVESTIGATOR, because evidence
   collection is the safe default before interpretation.

Two post-conditions guarantee a workable workflow shape (spec §10/
§33 — evidence first, interpretation last):

* if no step classified INVESTIGATOR, the FIRST step becomes the
  INVESTIGATOR ("evidence collection required before interpretation");
* if a plan of two or more steps has no ANALYST step, the LAST step
  becomes the ANALYST ("final step synthesizes the architecture
  explanation").

A single-step plan can satisfy neither (one step cannot be both
roles): it stays INVESTIGATOR and the run ends ``INCOMPLETE /
NO_ANALYSIS`` — documented, never a fabricated architectural answer.

Every assignment carries a short application-generated reason so the
trace answers "why did this plan step go to this agent?" (spec §15).
The reason is decision metadata, never model reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from ..planning.models import PlanStep

__all__ = [
    "AGENT_ROLES",
    "ANALYST",
    "INVESTIGATOR",
    "AgentRole",
    "RoleAssignment",
    "assign_roles",
]

AgentRole = str

INVESTIGATOR: AgentRole = "INVESTIGATOR"
ANALYST: AgentRole = "ANALYST"

AGENT_ROLES = (INVESTIGATOR, ANALYST)

# Application-generated decision metadata (spec §15 — the two example
# reasons, verbatim in spirit).
REASON_INVESTIGATOR = "step requires repository evidence collection"
REASON_ANALYST = "architecture interpretation requires Analyst role"
REASON_FIRST_INVESTIGATOR = (
    "evidence collection required before interpretation"
)
REASON_LAST_ANALYST = (
    "final step synthesizes the architecture explanation"
)

# Generic, repository-neutral synthesis/interpretation signals. Kept
# deliberately short: ordinary evidence goals ("identify", "investigate",
# "understand relationships") must NOT match — the classifier is a
# default, and explicit overrides are the precise instrument.
_ANALYST_GOAL_PATTERN = re.compile(
    r"\bexplan(?:ation|ations)\b"
    r"|\bexplain(?:s|ed|ing)?\b"
    r"|\binterpret(?:s|ed|ing|ation|ations)?\b"
    r"|\bsynthesi(?:s|ze|zes|zing|sed|zed)\b"
    r"|\bsummar(?:y|ies|ize|izes|izing|ized)\b"
    r"|\bwhat does\b.{0,60}\bmean\b"
    r"|\barchitectural (?:meaning|story|narrative)\b"
    r"|\bfinal conclusions?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoleAssignment:
    """One plan step's deterministic role decision (traceable)."""

    step_id: str
    role: str
    reason: str


def _classify_goal(goal: str) -> str:
    if _ANALYST_GOAL_PATTERN.search(goal or ""):
        return ANALYST
    return INVESTIGATOR


def assign_roles(
    steps: Sequence[PlanStep],
    role_overrides: Optional[Mapping[str, str]] = None,
) -> List[RoleAssignment]:
    """Assign every step a role, in plan order.

    *role_overrides* is application metadata (step id → role); an
    unknown role raises ``ValueError`` — unsupported roles are
    rejected, never coerced (spec §37/§45).
    """
    overrides: Dict[str, str] = dict(role_overrides or {})
    for step_id, role in overrides.items():
        if role not in AGENT_ROLES:
            raise ValueError(
                "unsupported role {!r} for step {!r} (expected one of "
                "{})".format(role, step_id, ", ".join(AGENT_ROLES))
            )
    known_ids = {step.id for step in steps}
    unknown = sorted(set(overrides) - known_ids)
    if unknown:
        raise ValueError(
            "role_overrides name step id(s) absent from the plan: "
            "{}".format(", ".join(unknown))
        )

    assignments: List[RoleAssignment] = []
    for step in steps:
        override = overrides.get(step.id)
        if override is not None:
            assignments.append(
                RoleAssignment(
                    step_id=step.id,
                    role=override,
                    reason="application-assigned role for this step",
                )
            )
            continue
        role = _classify_goal(step.goal)
        assignments.append(
            RoleAssignment(
                step_id=step.id,
                role=role,
                reason=(
                    REASON_ANALYST if role == ANALYST else REASON_INVESTIGATOR
                ),
            )
        )

    if not assignments:
        return assignments

    # Post-condition A — evidence first (even a single-step plan).
    if not any(item.role == INVESTIGATOR for item in assignments):
        first = assignments[0]
        assignments[0] = RoleAssignment(
            step_id=first.step_id, role=INVESTIGATOR, reason=REASON_FIRST_INVESTIGATOR
        )

    # Post-condition B — interpretation last, plans of 2+ steps only
    # (a lone step cannot also be the Investigator).
    if len(assignments) > 1 and not any(
        item.role == ANALYST for item in assignments
    ):
        last = assignments[-1]
        assignments[-1] = RoleAssignment(
            step_id=last.step_id, role=ANALYST, reason=REASON_LAST_ANALYST
        )

    # Post-condition C — execution order must be evidence before
    # interpretation: an ANALYST step that no INVESTIGATOR step
    # precedes could never receive a report (an override typo, caught
    # here as a decision the application can see, never as a runtime
    # failure inside the step loop).
    seen_investigator = False
    for item in assignments:
        if item.role == INVESTIGATOR:
            seen_investigator = True
        elif item.role == ANALYST and not seen_investigator:
            raise ValueError(
                "step {!r} is assigned ANALYST before any INVESTIGATOR "
                "step in plan order; analysis requires a preceding "
                "evidence step".format(item.step_id)
            )

    return assignments
