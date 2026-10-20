"""M3.5 ArchitectureModel construction (architecture_construction_rules.v1).

Pipeline:

    EngineeringModel (repository_knowledge_model.v1)
        + architecture_construction_rules.v1
        → deterministic rule evaluation
        → identity / dedup / evidence
        → validation
        → ArchitectureModel (architecture section + mappings)

Consumes the existing EngineeringModel only. Never re-parses the repository,
re-runs analyzers, or introduces LLM/graph-database inference.
"""

from __future__ import annotations

from .builder import (
    ARCHITECTURE_MODEL_FILENAME,
    build_architecture_model,
    create_architecture_model,
    load_engineering_model_document,
    write_architecture_model_artifact,
)
from .models import (
    ARCHITECTURE_RELATIONSHIP_KINDS,
    ARCHITECTURE_RULES_SCHEMA_VERSION,
    MAPPING_KINDS,
    architecture_counts,
    architecture_relationship_identity,
    mapping_identity,
)
from .resolver import (
    agent_name_from_engineering_node,
    architecture_agent_node_id,
    slugify,
)
from .rules import (
    ArchitectureRulesError,
    ArchitectureRulesSchemaError,
    default_architecture_rules_path,
    load_architecture_rules,
    relationship_rules_by_action,
    validate_architecture_rules_document,
    validate_architecture_rules_document_or_raise,
)
from .validator import (
    ArchitectureModelError,
    ArchitectureModelValidationError,
    validate_knowledge_model_document,
    validate_knowledge_model_or_raise,
)

__all__ = [
    "ARCHITECTURE_MODEL_FILENAME",
    "ARCHITECTURE_RELATIONSHIP_KINDS",
    "ARCHITECTURE_RULES_SCHEMA_VERSION",
    "ArchitectureModelError",
    "ArchitectureModelValidationError",
    "ArchitectureRulesError",
    "ArchitectureRulesSchemaError",
    "MAPPING_KINDS",
    "agent_name_from_engineering_node",
    "architecture_agent_node_id",
    "architecture_counts",
    "architecture_relationship_identity",
    "build_architecture_model",
    "create_architecture_model",
    "default_architecture_rules_path",
    "load_architecture_rules",
    "load_engineering_model_document",
    "mapping_identity",
    "relationship_rules_by_action",
    "slugify",
    "validate_architecture_rules_document",
    "validate_architecture_rules_document_or_raise",
    "validate_knowledge_model_document",
    "validate_knowledge_model_or_raise",
    "write_architecture_model_artifact",
]
