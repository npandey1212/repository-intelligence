"""Deterministic application-side plan validation (M10, spec §18).

The gate between "the LLM produced a structured plan" and "the Agent
may execute it". Pure functions over the typed plan — no model call,
no repair, no planner retry (the M8.9 rule: explicit failure, no
second loop):

    validate_plan(plan)
        ├── passes          → execution may start
        └── fails           → PlanValidationError
                              → NO repository tool execution
                              → the run terminates

Checks (each is observable and deterministic):

    objective exists                  non-empty after stripping
    at least one step                 an empty plan is not executable
    step IDs non-empty and unique     duplicate/blank IDs are ambiguous
    every step has a meaningful goal  non-empty after stripping
    goal-level, not tool-level        a step must not BE a tool call
                                      (``search_repository(...)``,
                                      ``call read_file(...)``, raw
                                      tool-call JSON) — which tools run
                                      inside a step stays adaptive
    no unsupported execution          no replanning, planner
    directives                        re-invocation, parallel-execution
                                      or spawn directives (spec §16/§17/
                                      §42 — future capabilities, not
                                      M10)

Step ordering: the list order IS the sequential execution order
(spec §16) — there is no scheduler, dependency graph, or priority to
validate.
"""

from __future__ import annotations

import re

from .models import InvestigationPlan

__all__ = ["PlanValidationError", "validate_plan"]


class PlanValidationError(Exception):
    """The produced plan did not pass deterministic validation.

    Raised instead of repairing, dropping, or re-planning: an invalid
    plan is a failure the caller must see, and no repository tool may
    execute after it (spec §18).
    """


# A step goal that IS a direct tool invocation — whole-goal tool-call
# syntax, an imperative invocation of an underscore tool name, or raw
# tool-call JSON. Goals that merely MENTION a function inside prose
# ("Explain how load_config() works") are not tool instructions.
_TOOL_LEVEL_PATTERNS = (
    re.compile(r"^(?:call\s+)?[A-Za-z_][\w]*\s*\(.*\)\s*[.;]?$"),
    re.compile(r"^(?:call|invoke|execute|run)\s+[A-Za-z_][\w]*_[\w]+"),
    re.compile(r'"(?:arguments|tool_calls?)"\s*:'),
)

# Execution directives that belong to future milestones, not M10
# (spec §16 parallel execution, §17 replanning, §42 forbidden list).
# Narrow phrases/words chosen so ordinary investigation goals that
# merely describe the repository still validate.
_UNSUPPORTED_DIRECTIVE_PATTERN = re.compile(
    r"\bre-?plan(?:ning)?\b"
    r"|\bcall the planner\b"
    r"|\bplanner again\b"
    r"|\bparallel execution\b"
    r"|\bspawn\b",
    re.IGNORECASE,
)


def _invalid(message: str) -> PlanValidationError:
    return PlanValidationError(message)


def validate_plan(plan: InvestigationPlan) -> None:
    """Validate *plan* deterministically; raise ``PlanValidationError``.

    Returns ``None`` when the plan may be executed. The typed plan is
    never modified — validation observes, the executor drives the
    lifecycle.
    """
    if not isinstance(plan, InvestigationPlan):
        raise _invalid(
            "plan must be an InvestigationPlan, got {}".format(
                type(plan).__name__
            )
        )

    # -- objective exists ------------------------------------------------
    if not isinstance(plan.objective, str) or not plan.objective.strip():
        raise _invalid("plan objective is missing or empty")

    # -- at least one step ----------------------------------------------
    if not plan.steps:
        raise _invalid(
            "plan has no steps; at least one goal-level step is required"
        )

    # -- step identity, goals, goal-level-ness, directives ---------------
    seen = set()
    for position, step in enumerate(plan.steps):
        label = "step {}".format(position + 1)
        step_id = step.id.strip() if isinstance(step.id, str) else ""
        if not step_id:
            raise _invalid("{} has a blank step id".format(label))
        if step_id in seen:
            raise _invalid(
                "duplicate step id {!r}; step ids must be unique".format(
                    step_id
                )
            )
        seen.add(step_id)
        label = "step {!r}".format(step_id)

        goal = step.goal.strip() if isinstance(step.goal, str) else ""
        if not goal:
            raise _invalid("{} has no goal".format(label))

        for pattern in _TOOL_LEVEL_PATTERNS:
            if pattern.search(goal):
                raise _invalid(
                    "{} goal is a direct tool-call instruction, not a "
                    "goal-level sub-goal: {!r}".format(label, goal)
                )

        directive = _UNSUPPORTED_DIRECTIVE_PATTERN.search(goal)
        if directive is not None:
            raise _invalid(
                "{} goal contains the unsupported execution directive "
                "{!r} (M10: one plan, sequential execution, no "
                "replanning): {!r}".format(
                    label, directive.group(0), goal
                )
            )

    # -- ordering --------------------------------------------------------
    # The list order is the sequential execution order (spec §16); with
    # unique, non-blank ids above there is nothing further to check —
    # no scheduler, dependency graph, or priority exists to validate.
