"""M10 — Explicit Plan-and-Execute: the planning layer.

Public surface:

    InvestigationPlan, PlanStep, PlanStatus, PLAN_STATUSES, …
                                    the smallest structured plan
                                    (goal-level steps, application-
                                    managed lifecycle)
    plan_to_dict                    canonical trace payload
    PlanValidationError, validate_plan
                                    deterministic application gate
                                    (no repair, no planner retry)
    PLANNING_SYSTEM_INSTRUCTION,
    build_planning_request,
    plan_from_response              provider-neutral planning LLM
                                    contract (M8.9 structured output)
    load_architecture_context       existing M6 artifact → planner
                                    input + compact trace record

Responsibility split (spec §12):

    LLM planner      creates the structured InvestigationPlan
    THIS package     provides the request/response contract and the
                     deterministic validation
    existing Agent   records the plan in its AgentTrace, then executes
                     each step sequentially with the existing
                     ReAct/tool loop under the one M8.5 budget

Boundaries: no tool execution, no replanning, no parallel/DAG
scheduling, no sub-agents, no second trace, no second budget — the
planner receives the M6 Architecture Context directly as input
(never as a tool call before the plan exists, spec §8) and this
package never imports a provider SDK.
"""

from __future__ import annotations

from .models import (
    PLAN_STATUSES,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_COMPLETED,
    PLAN_STATUS_FAILED,
    PLAN_STATUS_IN_PROGRESS,
    PLAN_STATUS_INVALIDATED,
    PLAN_STATUS_PENDING,
    InvestigationPlan,
    PlanStatus,
    PlanStep,
    plan_to_dict,
)
from .planner import (
    PLANNING_SYSTEM_INSTRUCTION,
    build_planning_request,
    load_architecture_context,
    plan_from_response,
)
from .validation import PlanValidationError, validate_plan

__all__ = [
    "PLAN_STATUSES",
    "PLAN_STATUS_BLOCKED",
    "PLAN_STATUS_COMPLETED",
    "PLAN_STATUS_FAILED",
    "PLAN_STATUS_IN_PROGRESS",
    "PLAN_STATUS_INVALIDATED",
    "PLAN_STATUS_PENDING",
    "PLANNING_SYSTEM_INSTRUCTION",
    "InvestigationPlan",
    "PlanStatus",
    "PlanStep",
    "PlanValidationError",
    "build_planning_request",
    "load_architecture_context",
    "plan_from_response",
    "plan_to_dict",
    "validate_plan",
]
