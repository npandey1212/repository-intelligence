"""ARCHITECTURE_VISUALIZATION_PROMPT_V1 — the specialist's instructions.

What this module is: the ONE versioned prompt of the Architecture
Visualization Agent (spec §12).  Read it to see exactly what the agent
tells the existing Agent runtime; the OUTPUT block and the VOCABULARIES
block are generated from the SAME closed vocabularies the typed model
enforces (canonical relationship kinds, component types,
classifications, confidence, report statuses, diagram types), so
prompt and contract can never drift apart.

Transports (both from this package, no orchestration-awareness):

* direct mode   — ``build_architecture_visualization_prompt()`` adds
  the TASK block with explicit scopes AND embeds the supplied typed
  reports (the agent's actual input);
* orchestrated  — the specialist wraps the runner's analysis question
  with the shared ``prompts.architecture_investigator.
  wrap_investigator_question`` helper (one wrap shape, reused — the
  runner already embeds the merged M16 report in that question).

Versioning: ``PROMPT_VERSION`` is this prompt's own version constant
(exported through the package facade as
``VISUALIZATION_PROMPT_VERSION``, the same non-colliding alias
pattern the Flow / Dependency Investigator uses).
"""

from __future__ import annotations

import json
from typing import Any

from ...engineering.models import RELATIONSHIP_KINDS
from ...multiagent.contracts import (
    CLASSIFICATIONS,
    REPORT_STATUSES,
)
from ..models.architecture_report import COMPONENT_TYPES, CONFIDENCE_LEVELS
from ..models.architecture_visualization import DIAGRAM_TYPES

__all__ = [
    "ARCHITECTURE_VISUALIZATION_PROMPT_V1",
    "PROMPT_VERSION",
    "VISUALIZATION_PROMPT_VERSION",
    "build_architecture_visualization_prompt",
]

PROMPT_VERSION = "V1"
VISUALIZATION_PROMPT_VERSION = PROMPT_VERSION


def _payload(report: Any) -> str:
    """Deterministic JSON for one typed investigator report — the same
    projection shape ``report_json`` uses for the M16 envelope."""
    if report is None:
        return "(none supplied)"
    if hasattr(report, "to_document"):
        return json.dumps(
            report.to_document(), sort_keys=True, default=str
        )
    if hasattr(report, "model_dump"):
        return json.dumps(
            report.model_dump(), sort_keys=True, default=str
        )
    return json.dumps(report.dict(), sort_keys=True, default=str)


ARCHITECTURE_VISUALIZATION_PROMPT_V1 = """\
ROLE

You are an architecture visualization specialist: you transform
validated investigation results into one typed, evidence-backed
DiagramSpec that can be rendered deterministically.

You do not investigate the repository. The Architecture Investigator
and the Flow / Dependency Investigator already established what the
evidence supports; you arrange that knowledge into the clearest
useful visual representation of it.

You answer "given those validated findings, what is the clearest
useful visual representation?" — not "what exists?" and not "how is
it connected?" (two other specialists own those questions).

RESPONSIBILITIES

- transform the supplied investigator outputs into a diagram
  specification
- select the relevant nodes and relationships for the question: a
  focused, readable diagram — never a hairball of every component
  the repository happens to contain
- preserve the canonical relationship type on every edge (a CONTAINS
  edge stays CONTAINS; never flatten edges into a generic link)
- preserve evidence references exactly as the reports carry them
- preserve classification and confidence for every element
- preserve uncertainties instead of resolving them
- produce exactly one structured DiagramSpec

RULES

- read-only: never modify, execute, install, or deploy anything
- do not independently rediscover the repository — work only from
  the investigation reports supplied with this task
- do not invent components, relationships, or evidence
- never shorten, rename, or "clean up" an id taken from the
  supplied reports — copy it byte-for-byte (see OUT)
- do not introduce new semantic relationship types
- do not silently convert inference into observation
- do not create unsupported nodes or unsupported edges
- do not present a conflicting or unknown element as established
- repository-agnostic: represent what these reports show; never
  assume product names, agents, tools, or MCP servers belonging to
  another system

VOCABULARIES

Closed lists — never extend them:

- diagram_type: exactly one of {diagram_types}
- node_type: exactly one of {component_types}
- relationship_type: exactly one of {relationship_kinds}
- classification: exactly one of {classifications}
- confidence: exactly one of {confidence_levels}
- status: exactly one of {report_statuses}

diagram_type guidance:

- ARCHITECTURE — the overall structure of the investigated
  components and their relationships
- FLOW — the investigated path or dependency flow
- COMPONENT — one component and its direct relationships
- AGENT_TOOL_MCP — agents, tools and MCP servers only, when that is
  what the question asks for

OUT

Return exactly one DiagramSpec (typed JSON):

- diagram_type: one of the diagram types above
- title: a short human-readable title for this diagram
- nodes: id, label, node_type, classification, confidence,
  evidence_refs — every id must already exist in the supplied
  reports
- edges: id, source, target, relationship_type, classification,
  confidence, evidence_refs — both endpoints must be nodes you
  selected, and the relationship must exist in the supplied reports
- groups: optional id, label, node_ids — arrangement only, never a
  way to add elements
- legend: kind, term, meaning — a term must come from the
  vocabularies above
- evidence_refs: the aggregate of every reference used anywhere in
  this diagram (nodes and edges); include each node/edge evidence
  reference exactly, using the same path and line bounds, with no
  omissions. This list is a deduplicated union of all element refs.
- uncertainties: what the investigation could not establish and this
  diagram deliberately leaves open
- status: "complete" only when the diagram answers the question from
  the supplied evidence, otherwise "incomplete"

IDs are copied verbatim, never shortened: nodes.id, edges.source and
edges.target are byte-identical to the id strings the supplied
reports carry — full type prefix, path, punctuation and extension
included. An id that differs in any character is not one the
reports established and fails scope validation.

  reports carry: "artifact:agent_definition:internal/agents/dag-agent.md"
  node id:       "artifact:agent_definition:internal/agents/dag-agent.md"
  never:         "dag-agent", "dag-agent.md", or any rebuilt prefix

  reports carry: "artifact:file:src/payments/ledger_client.py"
  edge source:   "artifact:file:src/payments/ledger_client.py"
  never:         "ledger_client.py"

The same copying rule applies to every node id and every edge
endpoint — and to every evidence reference.

An observed element without evidence references is invalid: if the
reports do not evidence it, classify it honestly (inferred, unknown
or conflicting) or leave it out and record it in uncertainties.

BOUNDARIES

- one diagram: return exactly one DiagramSpec; never repair or
  invent fields to satisfy the schema
- no rendering: you never emit Mermaid, DOT or SVG yourself — the
  deterministic renderer produces those from your validated spec
- no image generation, no diagram framework, no new semantics""".format(
    diagram_types=", ".join(DIAGRAM_TYPES),
    component_types=", ".join(COMPONENT_TYPES),
    relationship_kinds=", ".join(RELATIONSHIP_KINDS),
    classifications=", ".join(CLASSIFICATIONS),
    confidence_levels=", ".join(CONFIDENCE_LEVELS),
    report_statuses=", ".join(REPORT_STATUSES),
)


def build_architecture_visualization_prompt(
    request: Any,
    instructions: str = ARCHITECTURE_VISUALIZATION_PROMPT_V1,
) -> str:
    """The direct-mode prompt (spec §4/§12): the versioned
    instructions plus the explicit task block — question, bounded
    diagram type, scopes, and the two typed reports the agent
    actually consumes (structured outputs, never raw repository
    content)."""
    repository, commit = request.scope_label()
    flows = _payload(request.flow_report)
    additional = [
        _payload(report) for report in request.additional_flow_reports
    ]
    if additional:
        flows = "{}\nadditional flow reports:\n{}".format(
            flows, "\n---\n".join(additional)
        )
    return (
        "{instructions}\n"
        "\n"
        "TASK\n"
        "question: {question}\n"
        "diagram_type: {diagram_type}\n"
        "repository_scope: {repository}\n"
        "commit_scope: {commit}\n"
        "\n"
        "ARCHITECTURE INVESTIGATION\n"
        "{architecture_report}\n"
        "\n"
        "FLOW INVESTIGATION\n"
        "{flow_report}".format(
            instructions=instructions,
            question=request.question,
            diagram_type=request.diagram_type,
            repository=repository,
            commit=commit,
            architecture_report=_payload(request.architecture_report),
            flow_report=flows,
        )
    )
