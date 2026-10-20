"""M ArchitectureContext — agent-oriented context artifact (architecture_context.v1).

Pipeline:

    M1 snapshot + M2 facts + M3 engineering model + M4 architecture model
      + contract/architecture_context_template.json   (fixed repo-level template)
        → architecture_context.md    (the generated artifact)

The template is the contract (created once, never per run); the generator
populates a copy of it in memory from M1–M4 and projects it to Markdown.
The read-only tool get_architecture_context
(src/repo_intelligence/tools/) retrieves the generated Markdown artifact
for a future agent loop.

Repository-agnostic: no LLM, no domain hard-coding, no architecture expansion.
"""

from __future__ import annotations

from .builder import (
    ARCHITECTURE_CONTEXT_SCHEMA_VERSION,
    ArchitectureContextError,
    ArchitectureContextValidationError,
    ContextBuildResult,
    build_architecture_context,
    build_architecture_context_document,
    create_architecture_context,
    render_architecture_context_markdown,
)
from .models import (
    ARCHITECTURE_CONTEXT_FILENAME,
    COMMIT_LABEL,
    CONTRACT_LABEL,
    CONTEXT_SECTION_KEYS,
    CONTEXT_TEMPLATE_PATH,
    GENERATOR_NAME,
    SECTION_HEADINGS,
    default_context_markdown_path,
    default_context_path,
    format_source_line,
    load_context_template,
    source_identity,
)
from .validator import (
    validate_architecture_context_document,
    validate_architecture_context_or_raise,
    validate_context_document,
    validate_context_document_shape,
    validate_context_or_raise,
    validate_markdown_context_consistency,
    validate_markdown_context_or_raise,
)

__all__ = [
    "ARCHITECTURE_CONTEXT_FILENAME",
    "ARCHITECTURE_CONTEXT_SCHEMA_VERSION",
    "COMMIT_LABEL",
    "CONTRACT_LABEL",
    "CONTEXT_SECTION_KEYS",
    "CONTEXT_TEMPLATE_PATH",
    "ArchitectureContextError",
    "ArchitectureContextValidationError",
    "ContextBuildResult",
    "GENERATOR_NAME",
    "SECTION_HEADINGS",
    "build_architecture_context",
    "build_architecture_context_document",
    "create_architecture_context",
    "default_context_markdown_path",
    "default_context_path",
    "format_source_line",
    "load_context_template",
    "render_architecture_context_markdown",
    "source_identity",
    "validate_architecture_context_document",
    "validate_architecture_context_or_raise",
    "validate_context_document",
    "validate_context_document_shape",
    "validate_context_or_raise",
    "validate_markdown_context_consistency",
    "validate_markdown_context_or_raise",
]
