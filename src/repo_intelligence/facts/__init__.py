"""Deterministic repository facts (M2)."""

from .builder import (
    FactsError,
    FactsSchemaValidationError,
    compose_facts,
    create_facts,
    load_snapshot_file,
    run_analyzers,
    validate_facts_document,
    validate_facts_invariants,
    write_facts_artifact,
)
from .evidence import (
    ARTIFACT_TYPES,
    EvidenceRef,
    is_repository_relative_path,
    make_evidence,
    validate_evidence_list,
    validate_evidence_ref,
)
from .interactions import (
    analyze_interactions,
    build_endpoint_registry,
    build_interaction_facts,
)
from .models import (
    FACT_TYPE_ORDER,
    AnalyzersInfo,
    RepositoryFacts,
    SCHEMA_VERSION,
    sort_facts,
)
from .normalize import (
    build_entities,
    validate_cross_representation,
    validate_entity_identity,
)
from .records import build_fact_records, compute_fact_id, make_fact
from .tooling import analyze_tooling, build_tooling_facts

__all__ = [
    "ARTIFACT_TYPES",
    "AnalyzersInfo",
    "EvidenceRef",
    "FACT_TYPE_ORDER",
    "RepositoryFacts",
    "SCHEMA_VERSION",
    "FactsError",
    "FactsSchemaValidationError",
    "analyze_interactions",
    "analyze_tooling",
    "build_entities",
    "build_endpoint_registry",
    "build_fact_records",
    "build_interaction_facts",
    "build_tooling_facts",
    "compose_facts",
    "compute_fact_id",
    "create_facts",
    "is_repository_relative_path",
    "load_snapshot_file",
    "make_evidence",
    "make_fact",
    "run_analyzers",
    "sort_facts",
    "validate_cross_representation",
    "validate_entity_identity",
    "validate_evidence_list",
    "validate_evidence_ref",
    "validate_facts_document",
    "validate_facts_invariants",
    "write_facts_artifact",
]
