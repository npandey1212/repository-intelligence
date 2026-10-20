"""ArchitectureVisualizationAgent — the third specialist (spec §1, §3, §15).

One sentence (spec §1): a specialist that transforms the two
investigators' validated outputs into one typed, evidence-backed
``DiagramSpec`` for deterministic rendering.

Read this file top to bottom to understand the capability:

* ``__init__``  — injects the existing ``Agent`` and the visible tool
  list (composition over inheritance; no factory, no base class).
* ``visualize`` — DIRECT mode: versioned prompt + ``VisualizationRequest``
  → existing Agent runtime → typed ``DiagramSpec`` whose envelope comes
  from the request, never from model-authored prose.
* ``run_result`` — the M16/M17 analyst protocol: the same prompt wrapped
  around the runner's analysis question, with the structured output
  projected onto the M16 ``ArchitectureFinding`` envelope so the typed
  handoff keeps every diagram statement evidence-backed.  The specialist
  never imports or references the runner — no orchestration-aware branch.
* ``validate`` — the deterministic six-rule verdict (``agents.
  diagram_validation``), over the M16 ``ValidationResult`` contract.
* ``last_spec`` / ``last_trace`` — observability: the spec this agent
  produced and the ONE underlying Agent trace.

Domain boundary (spec §3): this specialist answers "given those
validated findings, what is the clearest useful visual
representation?" — NOT "what exists?" (Architecture Investigator) and
NOT "what path connects them?" (Flow / Dependency Investigator).
It performs no repository reads of its own: its tools are empty
(spec §13), so evidence authority always stays with the
investigators' traces.  It never renders: the deterministic renderer
in ``agents.visualization_artifacts`` owns Mermaid/DOT/SVG.

The specialist owns domain behavior only: prompt, tool list, typed
output, protocol projection, validation entry point.  It owns no tool
loop, no retry loop, no planner, no orchestration, no tracing, no
memory/context engine — the injected ``Agent`` already runs all of
those.  It does not subclass either investigator: the three
specialists share the existing abstractions (wrap helper, envelope
contracts, claim validation) instead of each other.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ..multiagent.contracts import ValidationResult
from .diagram_validation import DiagramScope, validate_diagram_spec
from .models.architecture_visualization import DiagramSpec, VisualizationRequest
from .prompts.architecture_investigator import wrap_investigator_question
from .prompts.architecture_visualization import (
    ARCHITECTURE_VISUALIZATION_PROMPT_V1,
    build_architecture_visualization_prompt,
)

__all__ = [
    "ArchitectureVisualizationAgent",
    "ArchitectureVisualizationAgentError",
    "HandoffContext",
    "VISUALIZATION_TOOLS",
    "parse_handoff_context",
]

# The visible tool list (spec §13): deliberately EMPTY.  The
# specialist reads typed investigator outputs, not the repository —
# no repository read means no evidence of its own, so the evidence
# authority can never drift away from the two investigators (spec §20).
VISUALIZATION_TOOLS: Tuple[str, ...] = ()


class ArchitectureVisualizationAgentError(Exception):
    """Explicit specialist failure — a wrong or missing structured
    output, or a question that carries no handoff envelope, never a
    silently repaired diagram."""


@dataclass(frozen=True)
class HandoffContext:
    """The envelope the analyst-slot question carries (question, the
    one pinned scope, and the correlation id of the merged report)."""

    question: str
    repository_scope: str
    commit_scope: str
    investigation_id: str


_HANDOFF_LABELS: Tuple[str, ...] = (
    "question",
    "repository_scope",
    "commit_scope",
    "investigation_id",
)


def parse_handoff_context(question: str) -> HandoffContext:
    """Read the handoff labels the application (runner) built into the
    question — deterministic parsing of application-authored text,
    first occurrence wins so an embedded report payload can never
    override the envelope.  Missing labels are an explicit failure
    (spec §20), not a guess."""
    found: dict = {}
    for line in question.splitlines():
        for label in _HANDOFF_LABELS:
            if label in found:
                continue
            prefix = "{}: ".format(label)
            if line.startswith(prefix):
                found[label] = line[len(prefix):].strip()
    missing = [
        label for label in _HANDOFF_LABELS if not found.get(label)
    ]
    if missing:
        raise ArchitectureVisualizationAgentError(
            "question carries no handoff envelope — missing "
            "label(s): {}".format(", ".join(missing))
        )
    return HandoffContext(
        question=found["question"],
        repository_scope=found["repository_scope"],
        commit_scope=found["commit_scope"],
        investigation_id=found["investigation_id"],
    )


class ArchitectureVisualizationAgent:
    """Transforms validated investigation results into one typed
    ``DiagramSpec`` (spec §1)."""

    def __init__(
        self,
        agent: Any,
        tools: Optional[Sequence[str]] = None,
        prompt: str = ARCHITECTURE_VISUALIZATION_PROMPT_V1,
    ) -> None:
        self.agent = agent
        self.tools: List[str] = list(
            VISUALIZATION_TOOLS if tools is None else tools
        )
        self.prompt = prompt
        self._last_spec: Optional[DiagramSpec] = None

    # -- observability ---------------------------------------------------

    @property
    def last_trace(self) -> Any:
        """The underlying Agent's own trace — the ONE trace system."""
        return getattr(self.agent, "last_trace", None)

    @property
    def last_spec(self) -> Optional[DiagramSpec]:
        """The DiagramSpec produced by the most recent successful
        ``visualize`` / ``run_result`` (None before the first)."""
        return self._last_spec

    # -- M16/M17 runner protocol (delegation only — the specialist
    #    never learns that a runner exists) --------------------------

    def run_result(
        self,
        question: str,
        *,
        max_tool_calls: Optional[int] = None,
    ) -> Any:
        """Run the runner's analysis question under the specialist's
        instructions and return its run result.

        ``max_tool_calls`` (M20, default ``None``) is the per-call
        tool-budget override passed straight through to the underlying
        Agent — the specialist never reinterprets it and never mutates
        the agent's configured budget.

        The envelope (question, scope, correlation id) is parsed from
        the application-built question BEFORE the model is called — a
        malformed question never spends a run.  A ``DiagramSpec``
        structured output is projected onto the M16
        ``ArchitectureFinding`` envelope so the typed handoff keeps
        every diagram statement as an evidence-backed conclusion.  A
        wrong or missing output passes through untouched — the runner
        records it as the failure it is.
        """
        parsed = parse_handoff_context(question)
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
        if isinstance(output, DiagramSpec):
            spec = output.with_context(
                parsed.question,
                parsed.repository_scope,
                parsed.commit_scope,
            )
            self._last_spec = spec
            result = dataclasses.replace(
                result,
                structured_output=spec.to_finding(
                    investigation_id=parsed.investigation_id
                ),
            )
        return result

    # -- direct execution ---------------------------------------------

    def visualize(self, request: VisualizationRequest) -> DiagramSpec:
        """Answer one visualization request: versioned prompt + typed
        request → existing Agent runtime → typed ``DiagramSpec``.

        The envelope always comes from the request (the application's
        typed input), never from what the model wrote — the same
        provenance discipline both investigators follow.  Structural
        diagram rules are already enforced by the model contract;
        scope/evidence validation is a separate, explicit step
        (``validate``), so a verdict is always a value, never an
        exception.
        """
        prompt = build_architecture_visualization_prompt(
            request, instructions=self.prompt
        )
        result = self.agent.run_result(prompt)
        output = getattr(result, "structured_output", None)
        if not isinstance(output, DiagramSpec):
            raise ArchitectureVisualizationAgentError(
                "expected DiagramSpec structured output, got "
                "{!r}".format(type(output).__name__)
            )
        repository, commit = request.scope_label()
        spec = output.with_context(
            request.question, repository, commit
        )
        self._last_spec = spec
        return spec

    def validate(
        self,
        spec: DiagramSpec,
        request: Optional[VisualizationRequest] = None,
        evidence: Any = None,
    ) -> ValidationResult:
        """The deterministic verdict (no model involved).

        ``request`` supplies the investigation scope in direct mode
        (``DiagramScope.from_request``); orchestrated callers pass a
        scope built from the merged ``InvestigationReport`` instead.
        ``evidence`` is an M13 ``EvidenceIndex`` gathered from the
        INVESTIGATORS' traces (``gather_evidence``) — this agent
        performs no repository reads, so its own trace is never an
        evidence authority and is deliberately never used here.
        """
        scope = (
            DiagramScope.from_request(request)
            if request is not None
            else None
        )
        return validate_diagram_spec(spec, scope=scope, evidence=evidence)
