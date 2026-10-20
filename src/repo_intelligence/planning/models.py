"""M10 plan model — the explicit structured plan (spec §13/§28).

The smallest useful plan the LLM planner returns through the existing
M8.9 structured-output machinery:

    InvestigationPlan
    ├── objective      what the investigation set out to answer
    ├── status         application-managed lifecycle (six values)
    ├── validation     (only in the trace payload — see plan_to_dict)
    └── steps[]        PlanStep { id, goal, status, result }

    PlanStep
    ├── id             stable step identifier ("P1", "P2", …)
    ├── goal           GOAL-LEVEL sub-goal, never a tool instruction
    ├── status         pending · in_progress · completed · failed ·
    │                  invalidated · blocked (application-managed)
    └── result         concise evidence/result summary recorded when
                       the step completes (executor-managed)

Boundaries:

* **Goal-level, not tool-level** — a step states a sub-goal; which
  tools execute inside the step stays adaptive ReAct territory. The
  deterministic check lives in ``planning.validation``.
* **Lifecycle is application-managed** — the planner payload only
  needs ``objective`` and ``steps[].{id,goal}``; lifecycle fields
  default to ``pending``/``None`` and are driven by the executor, so
  a model-supplied status can never make a step look completed.
* **No workflow-engine fields** — no scheduling, dependencies,
  concurrency, priority, or optimization (spec §16/§42).
* **No hidden reasoning** — unknown planner keys (notes, "reasoning",
  tool hints) are dropped by parsing; ``plan_to_dict`` emits an
  explicit key set only (spec §24).

``plan_to_dict`` is the canonical trace projection of a plan —
JSON-ready, with the deterministic validation outcome nestable beside
the lifecycle (spec §21/§40).
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional

from pydantic import BaseModel
from typing_extensions import Literal

__all__ = [
    "PLAN_STATUSES",
    "PLAN_STATUS_BLOCKED",
    "PLAN_STATUS_COMPLETED",
    "PLAN_STATUS_FAILED",
    "PLAN_STATUS_IN_PROGRESS",
    "PLAN_STATUS_INVALIDATED",
    "PLAN_STATUS_PENDING",
    "InvestigationPlan",
    "PlanStatus",
    "PlanStep",
    "plan_to_dict",
]

# Application-managed lifecycle vocabulary — the six statuses spec §13
# suggests, shared by the plan and its steps (small, explicit, never
# free-form model output).
PlanStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "invalidated",
    "blocked",
]

PLAN_STATUS_PENDING: PlanStatus = "pending"
PLAN_STATUS_IN_PROGRESS: PlanStatus = "in_progress"
PLAN_STATUS_COMPLETED: PlanStatus = "completed"
PLAN_STATUS_FAILED: PlanStatus = "failed"
PLAN_STATUS_INVALIDATED: PlanStatus = "invalidated"
PLAN_STATUS_BLOCKED: PlanStatus = "blocked"
PLAN_STATUSES: FrozenSet[str] = frozenset(
    {
        PLAN_STATUS_PENDING,
        PLAN_STATUS_IN_PROGRESS,
        PLAN_STATUS_COMPLETED,
        PLAN_STATUS_FAILED,
        PLAN_STATUS_INVALIDATED,
        PLAN_STATUS_BLOCKED,
    }
)


class PlanStep(BaseModel):
    """One goal-level step of the plan.

    ``id``/``goal`` come from the planner; ``status``/``result`` are
    owned by the executor (defaults above).
    """

    id: str
    goal: str
    status: PlanStatus = PLAN_STATUS_PENDING
    result: Optional[str] = None


class InvestigationPlan(BaseModel):
    """The structured plan the LLM planner produces.

    ``objective`` and ``steps`` are required: a planner payload that
    omits them fails M8.9 schema validation explicitly — never a
    defaulted plan.
    """

    objective: str
    steps: List[PlanStep]
    status: PlanStatus = PLAN_STATUS_PENDING


def plan_to_dict(
    plan: InvestigationPlan, validation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Canonical JSON-ready trace payload of a plan (spec §21/§40).

    *validation* is the deterministic application-side outcome
    (``{"status": "SUCCESS"}`` or ``{"status": "FAILED", "error": ...}``)
    recorded beside the lifecycle — never model-authored. Long values
    are clipped by the trace's own recording policy, not here.
    """
    payload: Dict[str, Any] = {
        "objective": plan.objective,
        "status": plan.status,
    }
    if validation is not None:
        payload["validation"] = dict(validation)
    payload["steps"] = []
    for step in plan.steps:
        entry: Dict[str, Any] = {
            "id": step.id,
            "goal": step.goal,
            "status": step.status,
        }
        if step.result is not None:
            entry["result"] = step.result
        payload["steps"].append(entry)
    return payload
