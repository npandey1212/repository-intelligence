"""M3 EngineeringModel construction (repository_knowledge_model.v1).

Pipeline:

    M2 repository_facts.v2 + engineering_construction_rules.v1
        → construction rules evaluation
        → identity / dedup / evidence
        → validation
        → EngineeringModel

This package does not parse source trees; M2 analyzers already produced facts.
No LLM, no graph database as canonical storage.
"""

from __future__ import annotations

from .builder import (
    ENGINEERING_MODEL_FILENAME,
    build_engineering_model,
    create_engineering_model,
    engineering_model_path,
    load_facts_document,
    write_engineering_model_artifact,
)
from .models import (
    CLASSIFICATIONS,
    RELATIONSHIP_KINDS,
    SCHEMA_VERSION,
    UNRESOLVED_REASONS,
    EngineeringModel,
    empty_graph_section,
)
from .resolver import (
    NodeIndex,
    module_to_path_candidates,
    parent_directory,
    resolve_dependency_target,
)
from .validator import (
    EngineeringModelError,
    EngineeringModelValidationError,
    MODEL_SCHEMA_PATH,
    load_model_schema,
    validate_engineering_model_document,
    validate_engineering_model_or_raise,
    validate_model,
)

__all__ = [
    "CLASSIFICATIONS",
    "ENGINEERING_MODEL_FILENAME",
    "EngineeringModel",
    "EngineeringModelError",
    "EngineeringModelValidationError",
    "MODEL_SCHEMA_PATH",
    "NodeIndex",
    "RELATIONSHIP_KINDS",
    "SCHEMA_VERSION",
    "UNRESOLVED_REASONS",
    "build_engineering_model",
    "create_engineering_model",
    "empty_graph_section",
    "engineering_model_path",
    "load_facts_document",
    "load_model_schema",
    "module_to_path_candidates",
    "parent_directory",
    "resolve_dependency_target",
    "validate_engineering_model_document",
    "validate_engineering_model_or_raise",
    "validate_model",
    "write_engineering_model_artifact",
]
