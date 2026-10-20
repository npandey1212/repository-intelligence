"""M10 planner — provider-neutral planning LLM contract (spec §11/§27).

One request builder, one response decoder, one artifact resolver. The
planner CREATES a plan; it never executes anything (spec §12) and is
coupled to no provider SDK — only the existing provider-neutral
``llm`` models and the M8.9 structured-output machinery:

    build_planning_request(objective, architecture_context)
        → LLMRequest(messages=[planning system instruction,
                                user objective + architecture context],
                      tools=[],                      ← nothing to execute
                      structured_output=spec derived from
                                              InvestigationPlan)

    plan_from_response(response)
        → typed InvestigationPlan, or an EXPLICIT
          StructuredOutputValidationError (no defaults, no repair)

    load_architecture_context(output_root=None, snapshot_id=None)
        → (architecture_context_text, compact_record)

Critical boundary (spec §8): the ALREADY-GENERATED M6 Architecture
Context is passed directly to the planner as input. The planner never
retrieves it through the Agent's tool loop — a tool call before the
plan exists would blur the planning/execution boundary M10 teaches.
The existing ``get_architecture_context`` tool remains available to
the executor during normal ReAct investigation.

``load_architecture_context`` resolves the artifact through the
existing ``context.models`` mechanism (never a hard-coded snapshot
directory) and returns the compact trace record spec §26 asks for:
artifact path, schema version, pinned commit, snapshot id.

Application-side plan validation (``planning.validation``) happens
AFTER this contract — schema validity here, semantic validity there.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..context.models import (
    ArchitectureContextError,
    default_context_markdown_path,
    display_path,
)
from ..llm import (
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputValidationError,
    spec_for_model,
    validate_structured,
)
from .models import InvestigationPlan

__all__ = [
    "PLANNING_SYSTEM_INSTRUCTION",
    "build_planning_request",
    "load_architecture_context",
    "plan_from_response",
]

# Application-level planning instruction. The planner produces ONE
# structured plan: goal-level sub-goals, sequential order, no tool
# instructions, no execution, no hidden reasoning fields — the
# Architecture Context is orientation, not truth (spec §9).
PLANNING_SYSTEM_INSTRUCTION = (
    "You are the planning component of a repository intelligence "
    "assistant. Given the user's objective and the already-generated "
    "Architecture Context, produce ONE explicit investigation plan: a "
    "short ordered list of goal-level sub-goals (typically 3-4 steps, "
    "ids P1, P2, ...) that together accomplish the objective. Prefer "
    "distinct, complementary evidence objectives over splitting one "
    "investigation into several overlapping searches. Avoid repeating "
    "the same objective in different words. Leave the final step for "
    "synthesis or visualization when the task requires it. Each step "
    "states WHAT to accomplish, never HOW: no direct tool calls, "
    "no tool names, no arguments — which tools execute inside a step "
    "is decided adaptively later. Do not execute anything, do not "
    "answer the objective, and do not include private reasoning: "
    "return only the structured plan."
)

# Source-metadata lines parsed from the artifact itself (never
# re-derived; mirrors the labels the M6 generator writes).
_CONTRACT_LINE_RE = re.compile(r"^- Contract: `([^`]+)`\s*$", re.MULTILINE)
_COMMIT_LINE_RE = re.compile(r"^- Pinned commit: `([^`]+)`\s*$", re.MULTILINE)
_SNAPSHOT_LINE_RE = re.compile(r"^- Snapshot: `([^`]+)`\s*$", re.MULTILINE)


def build_planning_request(
    objective: str,
    architecture_context: Optional[str] = None,
    memory_context: Optional[str] = None,
) -> LLMRequest:
    """Provider-neutral planning request (spec §27).

    system planning instruction + user objective (+ the M6
    Architecture Context, passed directly — spec §8) (+ optional M14
    retrieved persistent memory, appended as context — prior
    conclusions, not evidence). ``tools`` is empty: the planner has
    nothing to execute. The response contract is the
    ``InvestigationPlan`` model's own JSON schema (spec §28 — the
    existing M8.9 mechanism, no second schema framework).
    ``memory_context=None`` keeps the request byte-for-byte identical
    to pre-M14.
    """
    user_content = objective
    if architecture_context:
        user_content = "{}\n\nArchitecture Context:\n{}".format(
            user_content, architecture_context
        )
    if memory_context:
        user_content = "{}\n\n{}".format(user_content, memory_context)
    return LLMRequest(
        messages=[
            Message(role="system", content=PLANNING_SYSTEM_INSTRUCTION),
            Message(role="user", content=user_content),
        ],
        tools=[],
        structured_output=spec_for_model(InvestigationPlan),
    )


def plan_from_response(response: LLMResponse) -> InvestigationPlan:
    """Decode + schema-validate the planner's structured response.

    Exactly two outcomes (the M8.9 contract): a payload matching the
    ``InvestigationPlan`` schema → typed instance; anything else —
    no payload, not a JSON object, missing/invalid fields — an
    explicit ``StructuredOutputValidationError``. Semantic plan
    validation (goal-level steps, unique ids, directives) is the
    application's separate deterministic gate (``validate_plan``).
    """
    payload = response.structured_output
    if payload is None:
        raise StructuredOutputValidationError(
            "planner response contained no JSON object"
        )
    return validate_structured(InvestigationPlan, payload)


def load_architecture_context(
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve the current M6 Architecture Context for the planner.

    Uses the existing artifact/context mechanism
    (``default_context_markdown_path`` — snapshot discovery included),
    never a hard-coded run directory (spec §36). Returns:

        (architecture_context_markdown, compact_record)

    where *compact_record* is the trace-ready provenance spec §26
    asks for::

        {"artifact": ..., "schema_version": ..., "commit_sha": ...,
         "snapshot_id": ...}

    Raises ``ArchitectureContextError`` when the artifact cannot be
    resolved, read, or is empty — the planner is never fed a missing
    or blank context silently.
    """
    artifact_path = default_context_markdown_path(output_root, snapshot_id)
    if not artifact_path.is_file():
        raise ArchitectureContextError(
            "Architecture context artifact was not found: {}".format(
                display_path(artifact_path)
            )
        )
    try:
        context = artifact_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArchitectureContextError(
            "Architecture context artifact could not be read: {} ({})".format(
                display_path(artifact_path), exc
            )
        )
    if not context.strip():
        raise ArchitectureContextError(
            "Architecture context artifact is empty: {}".format(
                display_path(artifact_path)
            )
        )

    contract_match = _CONTRACT_LINE_RE.search(context)
    commit_match = _COMMIT_LINE_RE.search(context)
    snapshot_match = _SNAPSHOT_LINE_RE.search(context)
    if snapshot_match is not None:
        snapshot = snapshot_match.group(1)
    else:
        # <output>/<snapshot_id>/architecture_context/<file>
        snapshot = artifact_path.parent.parent.name
    record: Dict[str, Any] = {
        "artifact": display_path(artifact_path),
        "schema_version": contract_match.group(1) if contract_match else None,
        "commit_sha": commit_match.group(1) if commit_match else None,
        "snapshot_id": snapshot,
    }
    return context, record
