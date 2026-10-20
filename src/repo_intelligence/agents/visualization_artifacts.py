"""Deterministic rendering + persistence for one ``DiagramSpec``
(spec §15, §16, §17, §18, §19).

What this module is: the LAST stage of the pipeline —
``DiagramSpec → validator verdict → renderer → artifact files``.  It
adapts the spec to the ONE existing renderer's expected input (the M5
``DiagramProjection`` built by ``DiagramSpec.to_diagram_projection``)
and lets ``graph.exporters`` do every serialization byte:

* Mermaid — ``export_mermaid`` (always written; deterministic)
* Graphviz DOT — ``export_graphviz`` (written only when Graphviz can
  actually render it, so no half-artifact is left behind)
* SVG — ``render_svg`` when the ``dot`` binary exists; otherwise
  honestly reported as unavailable (spec §17.3: do not build a new
  rendering framework, do not fabricate an SVG)

What it is NOT: a second renderer, an inference step, or a place
where a model is consulted.  The same validated spec always yields
byte-identical Mermaid (spec §19) — there is no randomness here by
construction.

Gate (spec §17.1): the artifacts are written only from a spec whose
deterministic verdict ``status == "passed"``.  An unvalidated or
rejected ``DiagramSpec`` is never persisted as the run's artifact —
``write_visualization_artifacts`` raises instead of writing a file
that would misrepresent the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..artifacts import default_artifacts_root
from ..graph.exporters import (
    export_graphviz,
    export_mermaid,
    graphviz_available,
    render_svg,
)
from ..graph.models import (
    GRAPHVIZ_FILENAME,
    MERMAID_FILENAME,
    SVG_FILENAME,
)
from ..multiagent.contracts import ValidationResult
from .models.architecture_report import component_statement
from .models.architecture_visualization import DiagramSpec
from .models.flow_dependency_report import FlowDependencyReport

SPEC_FILENAME = "diagram_spec.json"
RUN_SUMMARY_FILENAME = "run_summary.md"

DEFAULT_ARTIFACT_DIRECTORY = "artifacts/architecture_visualization"

RENDERER_NAME = (
    "M5 deterministic renderer (repo_intelligence.graph.exporters)"
)

# The exact wording the spec requires when SVG cannot be produced
# (spec §25) — printed and persisted verbatim, never silently skipped.
SVG_UNAVAILABLE_NOTICE = (
    "NOT AVAILABLE — existing renderer does not support SVG"
)

REQUIRED_ARTIFACTS: tuple = (
    SPEC_FILENAME,
    MERMAID_FILENAME,
    SVG_FILENAME,
    RUN_SUMMARY_FILENAME,
)

__all__ = [
    "DEFAULT_ARTIFACT_DIRECTORY",
    "RENDERER_NAME",
    "REQUIRED_ARTIFACTS",
    "RUN_SUMMARY_FILENAME",
    "SPEC_FILENAME",
    "SVG_UNAVAILABLE_NOTICE",
    "ArtifactStatus",
    "RunSummaryContext",
    "VisualizationArtifactError",
    "VisualizationArtifacts",
    "build_run_summary",
    "resolve_artifact_directory",
    "verify_artifacts",
    "write_visualization_artifacts",
]


class VisualizationArtifactError(Exception):
    """Persistence refused — an invalid spec never becomes an
    artifact, and a renderer failure is never hidden."""


class VisualizationArtifacts(BaseModel):
    """The persistent output of one run (spec §17) — paths only, so
    the bundle itself is JSON-serializable and inspectable."""

    directory: str
    spec: str
    mermaid: str
    run_summary: str
    svg: Optional[str] = None
    svg_available: bool = False

    def paths(self) -> Dict[str, Optional[str]]:
        return {
            "diagram_spec.json": self.spec,
            "architecture.mmd": self.mermaid,
            "architecture.svg": self.svg,
            "run_summary.md": self.run_summary,
        }


class ArtifactStatus(BaseModel):
    """One row of the demo's artifact verification (spec §25)."""

    name: str
    path: str
    exists: bool
    bytes: int = 0
    note: str = ""


class RunSummaryContext(BaseModel):
    """Observable facts for ``run_summary.md`` (spec §17.4).

    Every field is an execution fact the caller already holds —
    structured outputs, validation results, trace/artifact locations.
    Nothing here is model-authored narrative (no hidden
    chain-of-thought).  Reports are optional: a section that was not
    part of this run says so instead of inventing content.
    """

    architecture_report: Optional[Any] = None
    flow_report: Optional[FlowDependencyReport] = None
    evidence_notes: List[str] = Field(default_factory=list)
    m17_agents: List[str] = Field(default_factory=list)
    m17_status: str = "not run"
    trace_location: str = ""
    renderer: str = RENDERER_NAME


def resolve_artifact_directory(directory: Optional[str] = None) -> Path:
    """The spec's artifact location (spec §17), overridable for tests.

    The default resolves under the PROJECT ROOT (independent of the
    process current working directory) like every other generated-
    output root; an explicit ``directory`` is used exactly as given.
    """
    path = (
        Path(directory)
        if directory is not None
        else default_artifacts_root() / Path(DEFAULT_ARTIFACT_DIRECTORY).name
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _line_range(ref: object) -> str:
    start = getattr(ref, "line_start", None)
    end = getattr(ref, "line_end", None)
    if start is None:
        return str(getattr(ref, "path", ""))
    if end is None or end == start:
        return "{}:{}".format(ref.path, start)
    return "{}:{}-{}".format(ref.path, start, end)


def _status_line(status: str, summary: str = "") -> List[str]:
    lines = ["Status: {}".format(status)]
    if summary:
        lines.append("Summary: {}".format(summary))
    return lines


def _architecture_section(report: Optional[Any]) -> List[str]:
    if report is None:
        return [
            "Status: not supplied for this run",
            "Key findings: none (this run did not include the "
            "Architecture Investigator)",
        ]
    lines = _status_line(report.status, report.summary)
    lines.append("Key findings:")
    findings = [
        finding.statement
        for finding in report.findings
        if getattr(finding, "statement", "")
    ]
    if not findings:
        findings = [
            component_statement(component)
            for component in report.components
        ][:10]
    if findings:
        lines.extend("- {}".format(statement) for statement in findings)
    else:
        lines.append("- none recorded")
    return lines


def _flow_section(report: Optional[FlowDependencyReport]) -> List[str]:
    if report is None:
        return [
            "Status: not supplied for this run",
            "Flow: none (this run did not include the Flow / "
            "Dependency Investigator)",
        ]
    lines = _status_line(report.status, report.summary)
    lines.append("path_status: {}".format(report.path_status))
    hops: List[str] = []
    if report.source:
        hops.append(report.source)
    for step in report.path:
        hops.append(step.target)
    if hops:
        lines.append("Flow: {}".format(" -> ".join(hops)))
    else:
        lines.append(
            "Flow: no established path ({} -> {})".format(
                report.source or "?", report.target or "?"
            )
        )
    if report.uncertainties:
        lines.append("Uncertainties:")
        lines.extend(
            "- {}".format(item) for item in report.uncertainties
        )
    return lines


def _diagram_section(spec: DiagramSpec) -> List[str]:
    lines = [
        "Diagram type: {}".format(spec.diagram_type),
        "Title: {}".format(spec.title),
        "Nodes: {}".format(len(spec.nodes)),
        "Edges: {}".format(len(spec.edges)),
    ]
    if spec.nodes:
        lines.append("Node list:")
        lines.extend(
            "- {} ({}, {}, confidence {}): {}".format(
                node.id,
                node.node_type,
                node.classification,
                node.confidence,
                node.label,
            )
            for node in spec.nodes
        )
    if spec.edges:
        lines.append("Edge list:")
        lines.extend(
            "- {} -{}-> {} ({})".format(
                edge.source,
                edge.relationship_type,
                edge.target,
                edge.classification,
            )
            for edge in spec.edges
        )
    return lines


def _validation_section(validation: ValidationResult) -> List[str]:
    lines = [
        "Status: {}".format(validation.status),
        "Validation checks:",
    ]
    if validation.checks:
        lines.extend(
            "- {}: {}".format(
                check.rule, "PASS" if check.ok else "FAIL"
            )
            + (" — {}".format(check.detail) if check.detail else "")
            for check in validation.checks
        )
    else:
        lines.append("- no checks recorded")
    return lines


def _rendering_section(
    artifacts: VisualizationArtifacts, renderer: str
) -> List[str]:
    lines = ["Renderer: {}".format(renderer), "Artifacts:"]
    for name, path in artifacts.paths().items():
        if path is None:
            if name == "architecture.svg":
                lines.append(
                    "- architecture.svg: {}".format(
                        SVG_UNAVAILABLE_NOTICE
                    )
                )
            else:
                lines.append("- {}: not produced".format(name))
        else:
            lines.append("- {}: {}".format(name, path))
    if not artifacts.svg_available:
        lines.append("SVG:")
        lines.append(SVG_UNAVAILABLE_NOTICE)
    return lines


def _evidence_section(spec: DiagramSpec, notes: List[str]) -> List[str]:
    lines = [
        "{} reference(s) carried by this diagram:".format(
            len(spec.evidence_refs)
        )
    ]
    lines.extend(
        "- {}".format(_line_range(ref))
        for ref in spec.evidence_refs
    )
    if notes:
        lines.append("Evidence notes:")
        lines.extend("- {}".format(note) for note in notes)
    return lines


def build_run_summary(
    spec: DiagramSpec,
    validation: ValidationResult,
    artifacts: VisualizationArtifacts,
    context: Optional[RunSummaryContext] = None,
) -> str:
    """The ``run_summary.md`` content (spec §17.4) — observable facts
    only, one section per contract section, deterministic ordering."""
    context = context or RunSummaryContext()
    sections: List[str] = ["# Architecture Visualization Run"]

    sections.append(
        "## User Question\n\n{}\n\nrepository_scope: {}\n"
        "commit_scope: {}".format(
            spec.question, spec.repository_scope, spec.commit_scope
        )
    )
    sections.append(
        "## Architecture Investigator\n\n{}".format(
            "\n".join(_architecture_section(context.architecture_report))
        )
    )
    sections.append(
        "## Flow / Dependency Investigator\n\n{}".format(
            "\n".join(_flow_section(context.flow_report))
        )
    )
    sections.append(
        "## Architecture Visualization Agent\n\n{}".format(
            "\n".join(_diagram_section(spec))
        )
    )
    sections.append(
        "## Validation\n\n{}".format(
            "\n".join(_validation_section(validation))
        )
    )
    sections.append(
        "## Rendering\n\n{}".format(
            "\n".join(
                _rendering_section(artifacts, context.renderer)
            )
        )
    )
    sections.append(
        "## Evidence\n\n{}".format(
            "\n".join(_evidence_section(spec, context.evidence_notes))
        )
    )
    uncertainties = list(spec.uncertainties)
    sections.append(
        "## Uncertainties\n\n{}".format(
            "\n".join(
                ["- {}".format(item) for item in uncertainties]
                or ["- none recorded"]
            )
        )
    )
    m17 = ["Agents:"]
    if context.m17_agents:
        m17.extend("- {}".format(agent) for agent in context.m17_agents)
    else:
        m17.append("- none recorded")
    m17.append("Status: {}".format(context.m17_status))
    sections.append("## M17 Execution\n\n{}".format("\n".join(m17)))
    sections.append(
        "## Trace\n\nTrace/artifact location: {}".format(
            context.trace_location or "not recorded for this run"
        )
    )
    return "\n\n".join(sections) + "\n"


def verify_artifacts(
    artifacts: VisualizationArtifacts,
) -> List[ArtifactStatus]:
    """The demo's file-set verification (spec §25): every required
    artifact reported with exists/bytes — an unsupported SVG reports
    the notice instead of being silently skipped."""
    statuses: List[ArtifactStatus] = []
    entries = [
        (SPEC_FILENAME, artifacts.spec, ""),
        (MERMAID_FILENAME, artifacts.mermaid, ""),
        (
            SVG_FILENAME,
            artifacts.svg,
            "" if artifacts.svg_available else SVG_UNAVAILABLE_NOTICE,
        ),
        (RUN_SUMMARY_FILENAME, artifacts.run_summary, ""),
    ]
    for name, path, note in entries:
        resolved = Path(path) if path else None
        exists = bool(resolved is not None and resolved.is_file())
        statuses.append(
            ArtifactStatus(
                name=name,
                path=path or "{} ({})".format(name, SVG_UNAVAILABLE_NOTICE),
                exists=exists,
                bytes=resolved.stat().st_size if exists else 0,
                note=note,
            )
        )
    return statuses


def write_visualization_artifacts(
    spec: DiagramSpec,
    validation: ValidationResult,
    context: Optional[RunSummaryContext] = None,
    directory: Optional[str] = None,
) -> VisualizationArtifacts:
    """Persist the four required artifacts of ONE validated spec
    (spec §17, §18): the spec first, then Mermaid/DOT/SVG from that
    spec through the existing renderer, then the summary describing
    exactly this execution.

    Raises ``VisualizationArtifactError`` when the deterministic
    verdict did not pass — the artifacts of a rejected diagram are
    never written, so a written artifact set always means "validated".
    """
    if validation.status != "passed":
        raise VisualizationArtifactError(
            "refusing to persist a DiagramSpec whose validation "
            "status is {!r} — failing rule(s): {}".format(
                validation.status,
                ", ".join(
                    check.rule
                    for check in validation.checks
                    if not check.ok
                )
                or "none recorded",
            )
        )

    dest = resolve_artifact_directory(directory)

    spec_path = dest / SPEC_FILENAME
    spec_path.write_text(spec.to_json(), encoding="utf-8")

    projection = spec.to_diagram_projection()
    mermaid_path = dest / MERMAID_FILENAME
    export_mermaid(projection, mermaid_path)

    svg_path: Optional[Path] = None
    svg_ok = graphviz_available()
    if svg_ok:
        dot_path = dest / GRAPHVIZ_FILENAME
        export_graphviz(projection, dot_path)
        svg_path = render_svg(dot_path, dest / SVG_FILENAME)
        if svg_path is None:
            svg_ok = False

    artifacts = VisualizationArtifacts(
        directory=str(dest),
        spec=str(spec_path),
        mermaid=str(mermaid_path),
        run_summary=str(dest / RUN_SUMMARY_FILENAME),
        svg=str(svg_path) if svg_path is not None else None,
        svg_available=svg_ok,
    )

    summary = build_run_summary(spec, validation, artifacts, context)
    Path(artifacts.run_summary).write_text(summary, encoding="utf-8")
    return artifacts
