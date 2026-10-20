"""Specialized domain agents — useful capabilities over M1-M17.

This package holds *specialists* (focused domain roles composed from
the existing runtime), NOT new infrastructure: no factories, no base
classes, no orchestration awareness.  Three specialists ship today:

* ``ArchitectureInvestigator`` — what exists, and how it is related
  (``docs/architecture-investigator.md``);
* ``FlowDependencyInvestigator`` — what path, dependency, or flow the
  evidence supports (``docs/flow-dependency-investigator.md``);
* ``ArchitectureVisualizationAgent`` — given those validated findings,
  what is the clearest useful visual representation
  (``docs/architecture-visualization-agent.md``).

Each owns a versioned prompt, a typed output, and the same two
adapters over the injected ``Agent`` (direct execution + M16/M17
``run_result()``).  They share — never fork — the investigator-role
tool inventory (the visualization agent's is empty by design), the M16
envelope, the M13 claim validator, the question-wrap helper, and the
M5 deterministic renderer behind ``visualization_artifacts``.
"""

from __future__ import annotations

from .architecture_investigator import (
    ARCHITECTURE_TOOLS,
    ArchitectureInvestigator,
    ArchitectureInvestigatorError,
)
from .architecture_visualization import (
    VISUALIZATION_TOOLS,
    ArchitectureVisualizationAgent,
    ArchitectureVisualizationAgentError,
)
from .diagram_validation import (
    DIAGRAM_CHECKS,
    DiagramScope,
    validate_diagram_spec,
)
from .flow_dependency_investigator import (
    FLOW_DEPENDENCY_TOOLS,
    FlowDependencyInvestigator,
    FlowDependencyInvestigatorError,
)
from .models.architecture_report import (
    CLASSIFICATIONS,
    COMPONENT_TYPES,
    CONFIDENCE_LEVELS,
    RELATIONSHIP_TYPES,
    ArchitectureComponent,
    ArchitectureRelationship,
    ArchitectureReport,
    component_statement,
    relationship_statement,
)
from .models.architecture_visualization import (
    DIAGRAM_TYPES,
    DiagramEdge,
    DiagramNode,
    DiagramSpec,
    VisualizationRequest,
)
from .models.flow_dependency_report import (
    PATH_STATUSES,
    FlowDependencyReport,
    PathStatus,
)
from .prompts.architecture_investigator import (
    ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    PROMPT_VERSION,
    build_architecture_investigator_prompt,
    wrap_investigator_question,
)
from .prompts.architecture_visualization import (
    ARCHITECTURE_VISUALIZATION_PROMPT_V1,
    VISUALIZATION_PROMPT_VERSION,
    build_architecture_visualization_prompt,
)
from .prompts.flow_dependency_investigator import (
    FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1,
    FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V2,
    FLOW_PROMPT_VERSION,
    build_flow_dependency_investigator_prompt,
)
from .visualization_artifacts import (
    DEFAULT_ARTIFACT_DIRECTORY,
    RENDERER_NAME,
    RUN_SUMMARY_FILENAME,
    SPEC_FILENAME,
    SVG_UNAVAILABLE_NOTICE,
    RunSummaryContext,
    VisualizationArtifactError,
    VisualizationArtifacts,
    build_run_summary,
    verify_artifacts,
    write_visualization_artifacts,
)

__all__ = [
    "ARCHITECTURE_INVESTIGATOR_PROMPT_V1",
    "ARCHITECTURE_TOOLS",
    "ARCHITECTURE_VISUALIZATION_PROMPT_V1",
    "CLASSIFICATIONS",
    "COMPONENT_TYPES",
    "CONFIDENCE_LEVELS",
    "DEFAULT_ARTIFACT_DIRECTORY",
    "DIAGRAM_CHECKS",
    "DIAGRAM_TYPES",
    "FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1",
    "FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V2",
    "FLOW_DEPENDENCY_TOOLS",
    "FLOW_PROMPT_VERSION",
    "PATH_STATUSES",
    "PROMPT_VERSION",
    "RELATIONSHIP_TYPES",
    "RENDERER_NAME",
    "RUN_SUMMARY_FILENAME",
    "SPEC_FILENAME",
    "SVG_UNAVAILABLE_NOTICE",
    "VISUALIZATION_PROMPT_VERSION",
    "VISUALIZATION_TOOLS",
    "ArchitectureComponent",
    "ArchitectureInvestigator",
    "ArchitectureInvestigatorError",
    "ArchitectureRelationship",
    "ArchitectureReport",
    "ArchitectureVisualizationAgent",
    "ArchitectureVisualizationAgentError",
    "DiagramEdge",
    "DiagramNode",
    "DiagramScope",
    "DiagramSpec",
    "FlowDependencyInvestigator",
    "FlowDependencyInvestigatorError",
    "FlowDependencyReport",
    "PathStatus",
    "RunSummaryContext",
    "VisualizationArtifactError",
    "VisualizationArtifacts",
    "VisualizationRequest",
    "build_architecture_investigator_prompt",
    "build_architecture_visualization_prompt",
    "build_flow_dependency_investigator_prompt",
    "build_run_summary",
    "component_statement",
    "relationship_statement",
    "validate_diagram_spec",
    "verify_artifacts",
    "wrap_investigator_question",
    "write_visualization_artifacts",
]
