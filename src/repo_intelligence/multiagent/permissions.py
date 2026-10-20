"""M16 tool permission layers — bounded access per agent role.

Two independent layers, both enforced, neither sufficient alone
(defense in depth):

1. **Definitions** — WHICH tools the model is offered
   (``tool_definitions_for`` → the Agent's ``tool_definitions``
   parameter). An omitted definition never reaches the model, so a
   forbidden name cannot even be requested in a well-formed run.
2. **Registry** — WHAT may actually execute (``restricted_dispatcher``
   → the injected ``ToolDispatcher``). Even if a forbidden name
   arrives (a hallucinated or hostile call), the dispatcher refuses it
   with the existing structured ``TOOL_NOT_FOUND`` failure — no code
   outside the registry ever runs.

Role policy (M16 §16):

* ``investigator`` — all four read-only tools (it MUST inspect the
  repository to produce evidence).
* ``analyst`` — **no tools**: its input is the typed
  ``InvestigationReport``; the strongest isolation is also the
  strongest provenance story (the ArchitectureFinding can only come
  from the report's evidence package, never from an independent
  second investigation).
* ``validator`` — no tools: it is deterministic code over recorded
  state, never a model.

The registries are BUILT from the one application-owned
``tool_registry()`` — nothing here defines, discovers, or loads a
tool; the bound execution remains the M8.3 boundary unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

from ..agent.tool_definitions import AGENT_TOOL_DEFINITIONS
from ..llm import ToolDefinition
from ..tool_dispatcher import ToolDispatcher, ToolFunction, tool_registry

__all__ = [
    "ANALYST_TOOLS",
    "INVESTIGATOR_TOOLS",
    "VALIDATOR_TOOLS",
    "restricted_dispatcher",
    "tool_definitions_for",
    "tool_allowlist",
]

# The investigator may use every registered read-only tool (the four).
INVESTIGATOR_TOOLS: Tuple[str, ...] = tuple(sorted(tool_registry()))
# The analyst and validator get NO tools (decision, M16 §16 — see
# module docstring).
ANALYST_TOOLS: Tuple[str, ...] = ()
VALIDATOR_TOOLS: Tuple[str, ...] = ()

ROLES = ("investigator", "analyst", "validator")


def tool_allowlist(role: str) -> Tuple[str, ...]:
    """The tool names a role may use — deterministic, explicit.

    An unknown role raises: composition must name a real role so a
    typo can never silently become "all tools".
    """
    if role == "investigator":
        return INVESTIGATOR_TOOLS
    if role == "analyst":
        return ANALYST_TOOLS
    if role == "validator":
        return VALIDATOR_TOOLS
    raise ValueError(
        "unknown agent role {!r} (expected one of {})".format(
            role, ", ".join(ROLES)
        )
    )


def tool_definitions_for(
    role: str,
    bindings: Optional[Mapping[str, ToolFunction]] = None,
) -> List[ToolDefinition]:
    """Layer 1 — the tool definitions a role's model is offered.

    Filtered from the single source of truth
    ``AGENT_TOOL_DEFINITIONS`` by the role's allowlist; ``bindings``
    (name → callable) only decides WHICH callable the dispatcher will
    later use, never which definitions exist. The analyst's list is
    empty: its request carries no tool schema at all.
    """
    allowed = set(tool_allowlist(role))
    if bindings is not None:
        allowed &= set(bindings)
    return [tool for tool in AGENT_TOOL_DEFINITIONS if tool.name in allowed]


def restricted_dispatcher(
    role: str,
    bindings: Optional[Mapping[str, ToolFunction]] = None,
) -> ToolDispatcher:
    """Layer 2 — the dispatcher a role's tools actually execute
    through.

    Built from ``tool_registry()`` filtered by the role's allowlist.
    ``bindings`` maps a tool name to an alternative callable (the
    argument-taking ``tool(arguments)`` convention) — e.g. binding
    ``read_file``/``search_repository`` to a foreign repository root
    without touching the pinned default. Unknown bindings names raise
    rather than being silently dropped.
    """
    registry = tool_registry()
    allowed = tool_allowlist(role)
    if bindings is not None:
        unknown = sorted(set(bindings) - set(registry))
        if unknown:
            raise ValueError(
                "bindings reference unregistered tool(s): {}".format(
                    ", ".join(unknown)
                )
            )
    selected: Dict[str, ToolFunction] = {}
    for name in allowed:
        if bindings is not None and name in bindings:
            selected[name] = bindings[name]
        else:
            selected[name] = registry[name]
    return ToolDispatcher(selected)
