"""ArchitectureContext validation (architecture_context.v1).

Three layers, all repository-agnostic:

1. Populated document vs the repo-level template
   (contract/architecture_context_template.json): section/field identity,
   field order, and value types.
2. Populated document vs M1–M4 artifacts: repository identity, exact M4
   component/relationship projection, observed structure/technology/
   documentation, explicit unresolved counts, and no fabricated content.
3. Markdown vs the populated document: the Markdown is a faithful
   projection of the document, never an independent source.

Markdown-vs-M1–M4 validation is kept for callers that validate a Markdown
string directly; it is transitively covered by layers 2 + 3 for generated
artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ARCHITECTURE_CONTEXT_SCHEMA_VERSION,
    COMMIT_LABEL,
    CONTRACT_LABEL,
    CONTEXT_SECTION_KEYS,
    KEY_DOCUMENT_LIMIT,
    KEY_FILE_LIMIT,
    SECTION_HEADINGS,
    SNAPSHOT_LABEL,
    SPECIAL_LIST_FIELDS,
    ArchitectureContextError,
    ArchitectureContextValidationError,
    empty_list_line,
    field_label,
    format_line,
    format_source_line,
    limitation_mappings_line,
    load_context_template,
    readme_first_key,
    source_identity,
    source_label,
    unknown_test_files_line,
    unknown_unresolved_line,
    validate_context_template_structure,
)

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_REL_TOKEN_RE = re.compile(r"`(rel:[^`]+)`")
_CONCEPT_TOKEN_RE = re.compile(r"`(concept:[^`]+)`")
_COMPONENT_ID_RE = re.compile(r"- Component id: `([^`]+)`")
_TOP_REL_RE = re.compile(r"^- `(rel:[^`]+)`$", re.MULTILINE)

_INTERFACES_FIELDS = ("apis", "commands", "events", "external_interfaces")
_OPERATIONS_FIELDS = ("build", "test", "run", "deploy")
_TECHNOLOGY_EMPTY_FIELDS = (
    "frameworks",
    "infrastructure",
    "external_services",
)
_REPOSITORY_EMPTY_FIELDS = ("description", "purpose", "project_type")


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------


def _section_map(markdown: str) -> Dict[str, str]:
    """Split Markdown into ``{section heading: section text}``."""
    sections: Dict[str, str] = {}
    current: Optional[str] = None
    buffer: List[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(buffer)
            current = line[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer)
    return sections


def _section_headings(markdown: str) -> List[str]:
    return [
        line[3:].strip()
        for line in markdown.splitlines()
        if line.startswith("## ")
    ]


def _subhead_bullets(section_text: str, label: str) -> Tuple[int, List[str]]:
    """Return (heading occurrences, bullet items) for ``### <label>``."""
    occurrences = 0
    bullets: List[str] = []
    collecting = False
    for line in section_text.splitlines():
        if line.startswith("### "):
            matches = line[4:].strip() == label
            if matches:
                occurrences += 1
            collecting = matches
            continue
        if not collecting:
            continue
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.strip() == "" and bullets:
            collecting = False
    return occurrences, bullets


def _extract_entity_evidence(markdown: str) -> List[Tuple[str, List[str]]]:
    """Collect (entity id, evidence paths) from component/relationship blocks.

    A relationship block is only recognized when the bullet is exactly the
    backticked id. Workflow blocks list relationships inline and must not
    re-bind the following Evidence line.
    """
    results: List[Tuple[str, List[str]]] = []
    current: Optional[str] = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Component id:"):
            tokens = _BACKTICK_RE.findall(stripped)
            current = tokens[0] if tokens else None
        else:
            match = _TOP_REL_RE.match(stripped)
            if match:
                current = match.group(1)
        if stripped.startswith("- Evidence:") and current:
            paths = _BACKTICK_RE.findall(stripped[len("- Evidence:"):])
            results.append((current, paths))
            current = None
    return results


def _relationship_blocks(markdown: str) -> Dict[str, str]:
    """Map relationship id → indented detail text of its block."""
    blocks: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in markdown.splitlines():
        match = _TOP_REL_RE.match(line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, [])
            continue
        if current is None:
            continue
        if line.startswith("### ") or (
            line.startswith("- ") and not line.startswith("  - ")
        ):
            current = None
            continue
        blocks[current].append(line)
    return {key: "\n".join(lines) for key, lines in blocks.items()}


def _workflow_blocks(section_text: str) -> List[Tuple[str, str]]:
    """Return (workflow title, block text) in document order."""
    blocks: List[Tuple[str, str]] = []
    title: Optional[str] = None
    buffer: List[str] = []
    for line in section_text.splitlines():
        if line.startswith("### "):
            if title is not None:
                blocks.append((title, "\n".join(buffer)))
            title = line[4:].strip()
            buffer = []
            continue
        if title is not None:
            buffer.append(line)
    if title is not None:
        blocks.append((title, "\n".join(buffer)))
    return blocks


def _line_value(block: str, prefix: str) -> Optional[str]:
    for line in block.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()[len(prefix):].strip()
    return None


def _doc_repository(document: Dict[str, Any]) -> Dict[str, Any]:
    return dict(document.get("repository") or {})


# ---------------------------------------------------------------------------
# Layer 1: populated document vs the repo-level template
# ---------------------------------------------------------------------------


def validate_context_document_shape(
    context: Any, template: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Return problems when a document does not match the template."""
    if not isinstance(context, dict) or not context:
        return ["empty_context_document"]
    if template is None:
        try:
            template = load_context_template()
        except ArchitectureContextError as exc:
            return ["cannot_load_context_template: {}".format(exc)]

    problems: List[str] = []
    if list(context.keys()) != list(CONTEXT_SECTION_KEYS):
        problems.append(
            "document_keys_mismatch: expected {} got {}".format(
                list(CONTEXT_SECTION_KEYS), list(context.keys())
            )
        )
        return problems

    if context.get("schema_version") != ARCHITECTURE_CONTEXT_SCHEMA_VERSION:
        problems.append(
            "schema_version_mismatch: expected {!r} got {!r}".format(
                ARCHITECTURE_CONTEXT_SCHEMA_VERSION,
                context.get("schema_version"),
            )
        )

    template_problems = validate_context_template_structure(template)
    problems.extend(template_problems)
    if template_problems:
        return problems

    for key, template_section in template.items():
        section = context.get(key)
        if key == "schema_version":
            continue
        if not isinstance(section, dict):  # pragma: no cover - keys matched
            problems.append("section_not_an_object: {!r}".format(key))
            continue
        if list(section.keys()) != list(template_section.keys()):
            problems.append(
                "section_keys_mismatch: {} expected {} got {}".format(
                    key,
                    list(template_section.keys()),
                    list(section.keys()),
                )
            )
            continue
        for field, template_value in template_section.items():
            value = section[field]
            if isinstance(template_value, str):
                if not isinstance(value, str):
                    problems.append(
                        "field_type_mismatch: {}.{} expected str got "
                        "{!r}".format(key, field, type(value).__name__)
                    )
            elif isinstance(template_value, list):
                if not isinstance(value, list):
                    problems.append(
                        "field_type_mismatch: {}.{} expected list got "
                        "{!r}".format(key, field, type(value).__name__)
                    )
                elif field in SPECIAL_LIST_FIELDS:
                    if not all(isinstance(item, dict) for item in value):
                        problems.append(
                            "field_item_type_mismatch: {}.{} expected "
                            "objects".format(key, field)
                        )
                elif not all(isinstance(item, str) for item in value):
                    problems.append(
                        "field_item_type_mismatch: {}.{} expected "
                        "strings".format(key, field)
                    )
    return problems


# ---------------------------------------------------------------------------
# Layer 2: populated document vs M1–M4 artifacts
# ---------------------------------------------------------------------------


def _top_level_directories(facts_document: Dict[str, Any]) -> List[str]:
    dirs = set()
    for fact in facts_document.get("facts") or []:
        if fact.get("fact_type") != "directory":
            continue
        path = str((fact.get("attributes") or {}).get("path") or "")
        if path and "/" not in path:
            dirs.add(path)
    return sorted(dirs)


def _fact_paths(facts_document: Dict[str, Any], fact_type: str) -> List[str]:
    paths = set()
    for fact in facts_document.get("facts") or []:
        if fact.get("fact_type") != fact_type:
            continue
        path = str((fact.get("attributes") or {}).get("path") or "")
        if path:
            paths.add(path)
    return sorted(paths)


def _languages(engineering: Dict[str, Any]) -> List[str]:
    types = set()
    for node in engineering.get("nodes") or []:
        if node.get("kind") != "code_entity":
            continue
        for item in node.get("evidence") or []:
            artifact_type = item.get("artifact_type")
            if artifact_type:
                types.add(str(artifact_type))
    return sorted(value.title() for value in types)


def _external_packages(engineering: Dict[str, Any]) -> List[str]:
    names = set()
    for node in engineering.get("nodes") or []:
        if node.get("kind") != "external_package":
            continue
        name = (node.get("attributes") or {}).get("name") or node.get("id")
        if name:
            names.add(str(name))
    return sorted(names)


def _dependency_fact_names(facts_document: Dict[str, Any]) -> List[str]:
    names = set()
    for fact in facts_document.get("facts") or []:
        if fact.get("fact_type") != "dependency":
            continue
        name = (fact.get("attributes") or {}).get("name")
        if name:
            names.add(str(name))
    return sorted(names)


def _important_artifacts(engineering: Dict[str, Any]) -> List[str]:
    grouped: Dict[str, set] = {}
    for node in engineering.get("nodes") or []:
        if node.get("kind") != "artifact":
            continue
        subkind = str(node.get("subkind") or "artifact")
        paths = {
            item.get("path")
            for item in node.get("evidence") or []
            if item.get("path")
        }
        if not paths:
            path = (node.get("attributes") or {}).get("path")
            if path:
                paths = {str(path)}
        if not paths:
            continue
        grouped.setdefault(subkind, set()).update(paths)
    return [
        "{}: {} item(s)".format(group, len(grouped[group]))
        for group in sorted(grouped)
    ]


def _engineering_unresolved_count(architecture_document: Dict[str, Any]) -> int:
    metadata = architecture_document.get("metadata") or {}
    counts = metadata.get("counts") or {}
    if counts.get("unresolved") is not None:
        try:
            return int(counts["unresolved"])
        except (TypeError, ValueError):
            pass
    engineering = architecture_document.get("engineering") or {}
    return len(engineering.get("unresolved") or [])


def validate_context_document(
    context: Dict[str, Any],
    architecture_document: Optional[Dict[str, Any]] = None,
    facts_document: Optional[Dict[str, Any]] = None,
    template: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return problems proving the context document mirrors M1–M4."""
    problems = validate_context_document_shape(context, template)
    if not isinstance(context, dict) or not context:
        return problems
    if architecture_document is None:
        return problems

    architecture = architecture_document.get("architecture") or {}
    arch_nodes = list(architecture.get("nodes") or [])
    arch_rels = list(architecture.get("relationships") or [])
    mappings = list(architecture_document.get("mappings") or [])
    arch_unresolved = list(architecture.get("unresolved") or [])
    node_ids = {node.get("id") for node in arch_nodes}
    rel_ids = {rel.get("id") for rel in arch_rels}
    engineering = architecture_document.get("engineering") or {}

    # --- repository identity (M1 semantics) ---
    arch_repo = _doc_repository(architecture_document)
    facts_repo = _doc_repository(facts_document or {})
    snapshot = (facts_document or {}).get("snapshot") or {}
    ctx_repo = context.get("repository") or {}

    expected_name = arch_repo.get("name") or facts_repo.get("name")
    expected_commit = arch_repo.get("commit_sha") or snapshot.get("commit_sha")
    expected_snapshot = (
        snapshot.get("snapshot_id")
        or (engineering.get("metadata") or {}).get("snapshot_id")
    )

    if expected_name and ctx_repo.get("name") != str(expected_name):
        problems.append(
            "repository_name_mismatch: expected={!r} got={!r}".format(
                expected_name, ctx_repo.get("name")
            )
        )
    if not expected_name and ctx_repo.get("name"):
        problems.append(
            "repository_name_fabricated: got={!r}".format(ctx_repo.get("name"))
        )
    if arch_repo.get("name") and facts_repo.get("name"):
        if arch_repo.get("name") != facts_repo.get("name"):
            problems.append(
                "source_repository_identity_conflict: facts={!r} "
                "architecture={!r}".format(
                    facts_repo.get("name"), arch_repo.get("name")
                )
            )
    if expected_commit and arch_repo.get("commit_sha") and snapshot.get(
        "commit_sha"
    ):
        if snapshot.get("commit_sha") != arch_repo.get("commit_sha"):
            problems.append(
                "source_commit_identity_conflict: facts={!r} "
                "architecture={!r}".format(
                    snapshot.get("commit_sha"), arch_repo.get("commit_sha")
                )
            )

    for field in _REPOSITORY_EMPTY_FIELDS:
        if ctx_repo.get(field) not in ("", None):
            problems.append(
                "unevidenced_repository_field_populated: {}={!r}".format(
                    field, ctx_repo.get(field)
                )
            )
    if ctx_repo.get("primary_domains") not in ([], None):
        problems.append(
            "unevidenced_repository_field_populated: primary_domains={!r}".format(
                ctx_repo.get("primary_domains")
            )
        )

    # --- components: exactly the M4 node set ---
    components = list((context.get("architecture") or {}).get("components") or [])
    component_ids = {component.get("id") for component in components}
    for node_id in node_ids:
        if node_id not in component_ids:
            problems.append(
                "architecture_node_missing_from_context: {!r}".format(node_id)
            )
    for component_id in component_ids:
        if component_id not in node_ids:
            problems.append(
                "invented_architecture_node_reference: {!r}".format(component_id)
            )

    node_by_id = {node.get("id"): node for node in arch_nodes}
    for component in components:
        node = node_by_id.get(component.get("id"))
        if node is None:
            continue
        expected_component_name = (
            (node.get("attributes") or {}).get("name") or node.get("id")
        )
        if component.get("name") != expected_component_name:
            problems.append(
                "component_name_mismatch: {!r} expected={!r} got={!r}".format(
                    component.get("id"),
                    expected_component_name,
                    component.get("name"),
                )
            )
        if str(node.get("kind") or "unknown") != component.get("kind"):
            problems.append(
                "component_kind_mismatch: {!r}".format(component.get("id"))
            )
        if str(node.get("classification") or "unknown") != component.get(
            "classification"
        ):
            problems.append(
                "component_classification_mismatch: {!r}".format(
                    component.get("id")
                )
            )
        allowed = {e.get("path") for e in node.get("evidence") or []}
        for item in component.get("evidence") or []:
            if item.get("path") not in allowed:
                problems.append(
                    "evidence_path_not_in_model: {!r} path={!r}".format(
                        component.get("id"), item.get("path")
                    )
                )

    # --- relationships: exactly the M4 relationship set ---
    relationships = list(
        (context.get("architecture") or {}).get("relationships") or []
    )
    context_rel_ids = {rel.get("id") for rel in relationships}
    for rel_id in rel_ids:
        if rel_id not in context_rel_ids:
            problems.append(
                "architecture_relationship_missing_from_context: {!r}".format(
                    rel_id
                )
            )
    for rel_id in context_rel_ids:
        if rel_id not in rel_ids:
            problems.append(
                "invented_architecture_relationship: {!r}".format(rel_id)
            )

    rel_by_id = {rel.get("id"): rel for rel in arch_rels}
    for rel in relationships:
        model = rel_by_id.get(rel.get("id"))
        if model is None:
            continue
        if rel.get("source") != model.get("source") or rel.get(
            "target"
        ) != model.get("target"):
            problems.append(
                "relationship_endpoint_mismatch: {!r}".format(rel.get("id"))
            )
        if rel.get("kind") != model.get("kind"):
            problems.append(
                "relationship_kind_mismatch: {!r}".format(rel.get("id"))
            )
        allowed = {e.get("path") for e in model.get("evidence") or []}
        for item in rel.get("evidence") or []:
            if item.get("path") not in allowed:
                problems.append(
                    "evidence_path_not_in_model: {!r} path={!r}".format(
                        rel.get("id"), item.get("path")
                    )
                )

    unresolved_keys = {
        (u.get("kind"), u.get("source"), u.get("target"))
        for u in arch_unresolved
    }
    for rel in relationships:
        key = (rel.get("kind"), rel.get("source"), rel.get("target"))
        if key in unresolved_keys:
            problems.append(
                "unresolved_claim_presented_as_validated: {!r}".format(
                    rel.get("id")
                )
            )

    # --- boundaries are never established without evidence ---
    if (context.get("architecture") or {}).get("boundaries") not in ([], None):
        problems.append("unevidenced_boundaries_populated")

    # --- workflows cover exactly the validated relationship set ---
    workflows = list((context.get("workflows") or {}).get("workflows") or [])
    workflow_rel_ids: List[Any] = []
    for workflow in workflows:
        workflow_rel_ids.extend(workflow.get("relationship_ids") or [])
        for rel_id in workflow.get("relationship_ids") or []:
            if rel_id not in rel_ids:
                problems.append(
                    "workflow_references_unsupported_relationship: "
                    "{!r}".format(rel_id)
                )
        if workflow.get("source_component_id") not in node_ids:
            problems.append(
                "workflow_references_unknown_component: {!r}".format(
                    workflow.get("source_component_id")
                )
            )
        for participant in workflow.get("participant_ids") or []:
            if participant not in node_ids:
                problems.append(
                    "workflow_references_unknown_participant: {!r}".format(
                        participant
                    )
                )
    if rel_ids:
        if len(workflow_rel_ids) != len(set(workflow_rel_ids)):
            problems.append("workflow_relationship_assignment_not_unique")
        if set(workflow_rel_ids) != rel_ids:
            problems.append(
                "workflow_relationship_coverage_mismatch: workflows={} "
                "model={}".format(
                    sorted(set(workflow_rel_ids)), sorted(rel_ids)
                )
            )
    elif workflows:
        problems.append("workflow_present_without_relationships")

    # --- structure ---
    structure = context.get("structure") or {}
    if facts_document is not None:
        expected_dirs = _top_level_directories(facts_document)
        if list(structure.get("key_directories") or []) != expected_dirs:
            problems.append(
                "structure_directories_mismatch: expected {} got {}".format(
                    expected_dirs, structure.get("key_directories")
                )
            )
        doc_paths = _fact_paths(facts_document, "documentation")
        cfg_paths = _fact_paths(facts_document, "configuration")
        expected_key_files = sorted(set(doc_paths) | set(cfg_paths))[
            :KEY_FILE_LIMIT
        ]
        if list(structure.get("key_files") or []) != expected_key_files:
            problems.append("structure_key_files_mismatch")
    expected_artifacts = _important_artifacts(engineering)
    if list(structure.get("important_artifacts") or []) != expected_artifacts:
        problems.append("structure_important_artifacts_mismatch")

    # --- technology ---
    technology = context.get("technology") or {}
    if list(technology.get("languages") or []) != _languages(engineering):
        problems.append("technology_languages_mismatch")
    expected_libraries = _external_packages(engineering)
    if facts_document is not None:
        expected_libraries = sorted(
            set(expected_libraries)
            | set(_dependency_fact_names(facts_document))
        )
    if list(technology.get("libraries") or []) != expected_libraries:
        problems.append("technology_libraries_mismatch")
    for field in _TECHNOLOGY_EMPTY_FIELDS:
        if technology.get(field) not in ([], None):
            problems.append(
                "unevidenced_technology_field_populated: {}".format(field)
            )

    # --- interfaces / operations stay empty (no reliable evidence) ---
    interfaces = context.get("interfaces") or {}
    for field in _INTERFACES_FIELDS:
        if interfaces.get(field) not in ([], None):
            problems.append(
                "unevidenced_interfaces_field_populated: {}".format(field)
            )
    operations = context.get("operations") or {}
    for field in _OPERATIONS_FIELDS:
        if operations.get(field) not in ([], None):
            problems.append(
                "unevidenced_operations_field_populated: {}".format(field)
            )

    # --- documentation ---
    documentation = context.get("documentation") or {}
    if facts_document is not None:
        doc_paths = _fact_paths(facts_document, "documentation")
        expected_documents = sorted(doc_paths, key=readme_first_key)[
            :KEY_DOCUMENT_LIMIT
        ]
        if list(documentation.get("key_documents") or []) != (
            expected_documents
        ):
            problems.append("documentation_key_documents_mismatch")

    # --- evidence ---
    evidence = context.get("evidence") or {}
    sources = list(evidence.get("sources") or [])
    if not sources:
        problems.append("evidence_sources_empty")
    template_source_found = False
    for entry in sources:
        if not isinstance(entry, dict) or not entry.get("path"):
            problems.append("evidence_source_entry_invalid: {!r}".format(entry))
            continue
        if entry.get("label") != source_label(str(entry.get("path"))):
            problems.append(
                "evidence_source_label_mismatch: {!r}".format(entry)
            )
        if Path(str(entry.get("path"))).name == (
            "architecture_context_template.json"
        ):
            template_source_found = True
    if not template_source_found:
        problems.append("evidence_sources_missing_template")

    source_commit, source_snapshot = source_identity(sources)
    if expected_commit and source_commit != str(expected_commit):
        problems.append(
            "source_commit_mismatch: expected={!r} got={!r}".format(
                expected_commit, source_commit
            )
        )
    if expected_snapshot and source_snapshot != str(expected_snapshot):
        problems.append(
            "snapshot_id_mismatch: expected={!r} got={!r}".format(
                expected_snapshot, source_snapshot
            )
        )

    known_unknowns = list(evidence.get("known_unknowns") or [])
    test_count = (
        len(_fact_paths(facts_document or {}, "test"))
        if facts_document is not None
        else None
    )
    if test_count is not None:
        required_test_line = unknown_test_files_line(test_count)
        if required_test_line not in known_unknowns:
            problems.append(
                "known_unknown_missing: {!r}".format(required_test_line)
            )
    required_unresolved_line = unknown_unresolved_line(len(arch_unresolved))
    if required_unresolved_line not in known_unknowns:
        problems.append(
            "known_unknown_missing: {!r}".format(required_unresolved_line)
        )
    model_unresolved_keys = {
        (
            str(u.get("kind")),
            str(u.get("source") or ""),
            str(u.get("target") or ""),
            str(u.get("reason")),
        )
        for u in arch_unresolved
    }
    for entry in known_unknowns:
        if not entry.startswith("Unresolved claim: "):
            continue
        if "(+" in entry:
            continue  # truncation note, not a claim
        if not model_unresolved_keys:
            problems.append(
                "invented_unresolved_claim: {!r}".format(entry)
            )
            continue
        if not any(
            "{kind} {source} -> {target} ({reason})".format(
                kind=kind, source=source, target=target, reason=reason
            )
            in entry
            for kind, source, target, reason in model_unresolved_keys
        ):
            problems.append(
                "invented_unresolved_claim: {!r}".format(entry)
            )

    limitations = list(evidence.get("analysis_limitations") or [])
    if limitation_mappings_line(len(mappings)) not in limitations:
        problems.append(
            "analysis_limitation_missing: {!r}".format(
                limitation_mappings_line(len(mappings))
            )
        )
    return problems


def validate_context_or_raise(
    context: Dict[str, Any],
    architecture_document: Optional[Dict[str, Any]] = None,
    facts_document: Optional[Dict[str, Any]] = None,
    template: Optional[Dict[str, Any]] = None,
) -> None:
    problems = validate_context_document(
        context, architecture_document, facts_document, template=template
    )
    if problems:
        raise ArchitectureContextValidationError(
            "Architecture context document failed validation:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Layer 3: Markdown projection vs the populated document
# ---------------------------------------------------------------------------


def validate_markdown_context_consistency(
    markdown: str, context: Dict[str, Any]
) -> List[str]:
    """Return problems when architecture_context.md diverges from the doc."""
    if not markdown or not markdown.strip():
        return ["empty_context_document"]
    if not isinstance(context, dict) or not context:
        return ["empty_context_document"]

    problems: List[str] = []
    lines = markdown.splitlines()
    line_counts: Dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1

    def require_unique(expected: str, label: str) -> None:
        if line_counts.get(expected) != 1:
            problems.append(
                "{}_line_mismatch: expected exactly one {!r}".format(
                    label, expected
                )
            )

    # --- contract line ---
    require_unique(
        format_line(CONTRACT_LABEL, context.get("schema_version"), code=True),
        "contract",
    )

    # --- section headings (order, each exactly once) ---
    headings = _section_headings(markdown)
    if headings != list(SECTION_HEADINGS):
        problems.append(
            "section_headings_mismatch: expected {} got {}".format(
                list(SECTION_HEADINGS), headings
            )
        )
    sections = _section_map(markdown)

    # --- repository identity lines ---
    repo = context.get("repository") or {}
    for field in ("name",) + _REPOSITORY_EMPTY_FIELDS:
        require_unique(
            format_line(
                field_label(field), repo.get(field), code=(field == "name")
            ),
            "repository_{}".format(field),
        )
    commit_sha, snapshot_id = source_identity(
        (context.get("evidence") or {}).get("sources") or []
    )
    require_unique(format_line(COMMIT_LABEL, commit_sha, code=True), "commit")
    require_unique(
        format_line(SNAPSHOT_LABEL, snapshot_id, code=True), "snapshot"
    )

    # --- overviews ---
    for key in CONTEXT_SECTION_KEYS[1:]:
        section = context.get(key) or {}
        if not isinstance(section, dict):
            continue
        overview = section.get("overview")
        if overview and str(overview) not in lines:
            problems.append("overview_missing: {}".format(key))

    # --- plain string lists ---
    heading_by_key = dict(
        zip(CONTEXT_SECTION_KEYS[1:], SECTION_HEADINGS)
    )
    for key in CONTEXT_SECTION_KEYS[1:]:
        section = context.get(key) or {}
        if not isinstance(section, dict):
            continue
        section_text = sections.get(heading_by_key[key], "")
        for field, value in section.items():
            if field == "overview" or field in SPECIAL_LIST_FIELDS:
                continue
            if not isinstance(value, list):
                continue
            if not value:
                if empty_list_line(field) not in section_text:
                    problems.append(
                        "empty_list_line_missing: {}.{}".format(key, field)
                    )
                continue
            occurrences, bullets = _subhead_bullets(
                section_text, field_label(field)
            )
            if occurrences != 1:
                problems.append(
                    "list_heading_missing_or_ambiguous: {}.{} ({!r} "
                    "occurrences={})".format(
                        key, field, field_label(field), occurrences
                    )
                )
            elif bullets != list(value):
                problems.append(
                    "list_projection_mismatch: {}.{} expected {} got "
                    "{}".format(key, field, list(value), bullets)
                )

    # --- components ---
    architecture = context.get("architecture") or {}
    components = list(architecture.get("components") or [])
    relationships = list(architecture.get("relationships") or [])
    component_ids = [str(component.get("id")) for component in components]
    md_component_ids = _COMPONENT_ID_RE.findall(markdown)
    if md_component_ids != component_ids:
        problems.append(
            "component_set_mismatch: markdown={} document={}".format(
                md_component_ids, component_ids
            )
        )
    for component in components:
        expected_heading = "### Component: {}".format(component.get("name"))
        pattern = r"### Component: {}\n\n- Component id: `{}`".format(
            re.escape(str(component.get("name"))),
            re.escape(str(component.get("id"))),
        )
        if expected_heading not in lines or not re.search(pattern, markdown):
            problems.append(
                "component_label_mismatch: {!r} expected {!r}".format(
                    component.get("id"), component.get("name")
                )
            )
        kind = str(component.get("kind") or "unknown")
        if component.get("subkind") is not None:
            kind = "{}/{}".format(kind, component.get("subkind"))
        if "- Kind: {}".format(kind) not in lines:
            problems.append(
                "component_kind_missing: {!r}".format(component.get("id"))
            )
        if "- Classification: {}".format(
            component.get("classification")
        ) not in lines:
            problems.append(
                "component_classification_missing: {!r}".format(
                    component.get("id")
                )
            )

    # --- relationships ---
    rel_ids = [str(rel.get("id")) for rel in relationships]
    md_rel_ids = _TOP_REL_RE.findall(markdown)
    if md_rel_ids != rel_ids:
        problems.append(
            "relationship_set_mismatch: markdown={} document={}".format(
                md_rel_ids, rel_ids
            )
        )
    all_rel_tokens = set(_REL_TOKEN_RE.findall(markdown))
    if all_rel_tokens != set(rel_ids):
        problems.append(
            "relationship_token_mismatch: markdown={} document={}".format(
                sorted(all_rel_tokens), sorted(rel_ids)
            )
        )
    blocks = _relationship_blocks(markdown)
    for rel in relationships:
        rel_id = str(rel.get("id"))
        block = blocks.get(rel_id)
        if block is None:
            problems.append("relationship_block_missing: {!r}".format(rel_id))
            continue
        if "  - Kind: {}".format(rel.get("kind")) not in block:
            problems.append(
                "relationship_kind_missing: {!r}".format(rel_id)
            )
        if "  - Classification: {}".format(
            rel.get("classification")
        ) not in block:
            problems.append(
                "relationship_classification_missing: {!r}".format(rel_id)
            )
        if "  - Endpoints: `{}` → `{}`".format(
            rel.get("source"), rel.get("target")
        ) not in block:
            problems.append(
                "relationship_endpoint_mismatch: {!r}".format(rel_id)
            )

    # --- workflows ---
    workflows = list((context.get("workflows") or {}).get("workflows") or [])
    workflow_text = sections.get(
        heading_by_key["workflows"], ""
    )
    blocks_found = _workflow_blocks(workflow_text)
    if not workflows:
        if empty_list_line("workflows") not in workflow_text:
            problems.append("empty_list_line_missing: workflows.workflows")
    else:
        expected_titles = [
            "Workflow: {}".format(workflow.get("source_name"))
            for workflow in workflows
        ]
        titles = [title for title, _ in blocks_found]
        if titles != expected_titles:
            problems.append(
                "workflow_set_mismatch: markdown={} document={}".format(
                    titles, expected_titles
                )
            )
        by_title = dict(blocks_found)
        for workflow in workflows:
            title = "Workflow: {}".format(workflow.get("source_name"))
            block = by_title.get(title, "")
            if _line_value(block, "- Source:") != "`{}`".format(
                workflow.get("source_component_id")
            ):
                problems.append(
                    "workflow_source_mismatch: {!r}".format(title)
                )
            expected_participants = sorted(
                str(name) for name in workflow.get("participant_names") or []
            )
            found_participants = sorted(
                _BACKTICK_RE.findall(
                    _line_value(block, "- Participants:") or ""
                )
            )
            if found_participants != expected_participants:
                problems.append(
                    "workflow_participants_mismatch: {!r}".format(title)
                )
            expected_rels = sorted(
                str(rel_id) for rel_id in workflow.get("relationship_ids") or []
            )
            found_rels = sorted(
                _BACKTICK_RE.findall(
                    _line_value(block, "- Relationships:") or ""
                )
            )
            if found_rels != expected_rels:
                problems.append(
                    "workflow_relationships_mismatch: {!r}".format(title)
                )

    # --- evidence sources ---
    evidence = context.get("evidence") or {}
    sources = list(evidence.get("sources") or [])
    evidence_text = sections.get(heading_by_key["evidence"], "")
    if not sources:
        if empty_list_line("sources") not in evidence_text:
            problems.append("empty_list_line_missing: evidence.sources")
    else:
        occurrences, bullets = _subhead_bullets(evidence_text, "Sources")
        expected_sources = [format_source_line(entry) for entry in sources]
        if occurrences != 1:
            problems.append(
                "sources_heading_missing_or_ambiguous: occurrences={}".format(
                    occurrences
                )
            )
        elif bullets != expected_sources:
            problems.append(
                "sources_projection_mismatch: expected {} got {}".format(
                    expected_sources, bullets
                )
            )

    # --- per-entity evidence paths ---
    evidence_by_id: Dict[str, set] = {}
    for component in components:
        evidence_by_id[str(component.get("id"))] = {
            item.get("path") for item in component.get("evidence") or []
        }
    for rel in relationships:
        evidence_by_id[str(rel.get("id"))] = {
            item.get("path") for item in rel.get("evidence") or []
        }
    for entity_id, paths in _extract_entity_evidence(markdown):
        allowed = evidence_by_id.get(str(entity_id))
        if allowed is None:
            problems.append(
                "evidence_for_unknown_entity: {!r}".format(entity_id)
            )
            continue
        for path in paths:
            if "(+" in path:
                continue
            if path not in allowed:
                problems.append(
                    "evidence_path_not_in_model: {!r} path={!r}".format(
                        entity_id, path
                    )
                )

    # --- unresolved claims must stay explicit in the document ---
    unresolved_lines = [
        entry
        for entry in evidence.get("known_unknowns") or []
        if str(entry).startswith("Unresolved claim: ")
    ]
    md_unresolved_lines = [
        line[2:].strip()
        for line in lines
        if line.startswith("- Unresolved claim: ")
    ]
    if md_unresolved_lines != unresolved_lines:
        problems.append(
            "unresolved_claims_projection_mismatch: markdown={} "
            "document={}".format(md_unresolved_lines, unresolved_lines)
        )

    return problems


def validate_markdown_context_or_raise(
    markdown: str, context: Dict[str, Any]
) -> None:
    problems = validate_markdown_context_consistency(markdown, context)
    if problems:
        raise ArchitectureContextValidationError(
            "architecture_context.md failed validation against the "
            "populated context document:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Markdown vs M1–M4 (direct validation of a Markdown string)
# ---------------------------------------------------------------------------


def validate_architecture_context_document(
    markdown: str,
    architecture_document: Dict[str, Any],
    facts_document: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return problems proving architecture_context.md mirrors its inputs."""
    if not markdown or not markdown.strip():
        return ["empty_context_document"]

    problems: List[str] = []
    lines = markdown.splitlines()
    line_set = set(lines)

    architecture = architecture_document.get("architecture") or {}
    arch_nodes = list(architecture.get("nodes") or [])
    arch_rels = list(architecture.get("relationships") or [])
    arch_unresolved = list(architecture.get("unresolved") or [])
    node_ids = [node.get("id") for node in arch_nodes]
    rel_ids = [rel.get("id") for rel in arch_rels]

    # --- contract + section structure ---
    if (
        format_line(
            CONTRACT_LABEL, ARCHITECTURE_CONTEXT_SCHEMA_VERSION, code=True
        )
        not in line_set
    ):
        problems.append(
            "schema_version_mismatch: expected {!r}".format(
                ARCHITECTURE_CONTEXT_SCHEMA_VERSION
            )
        )
    if _section_headings(markdown) != list(SECTION_HEADINGS):
        problems.append("section_headings_mismatch")

    arch_repo = _doc_repository(architecture_document)
    if arch_repo.get("name"):
        expected_name_line = format_line(
            "Name", str(arch_repo.get("name")), code=True
        )
        if expected_name_line not in line_set:
            problems.append(
                "repository_name_mismatch: expected={!r}".format(
                    arch_repo.get("name")
                )
            )
    if arch_repo.get("commit_sha"):
        expected_commit_line = format_line(
            COMMIT_LABEL, str(arch_repo.get("commit_sha")), code=True
        )
        if expected_commit_line not in line_set:
            problems.append(
                "repository_commit_mismatch: expected={!r}".format(
                    arch_repo.get("commit_sha")
                )
            )
    if facts_document is not None:
        facts_repo = _doc_repository(facts_document)
        snapshot = facts_document.get("snapshot") or {}
        if arch_repo.get("name") and facts_repo.get("name"):
            if facts_repo.get("name") != arch_repo.get("name"):
                problems.append(
                    "source_repository_identity_conflict: facts={!r} "
                    "architecture={!r}".format(
                        facts_repo.get("name"), arch_repo.get("name")
                    )
                )
        if arch_repo.get("commit_sha") and snapshot.get("commit_sha"):
            if snapshot.get("commit_sha") != arch_repo.get("commit_sha"):
                problems.append(
                    "source_commit_identity_conflict: facts={!r} "
                    "architecture={!r}".format(
                        snapshot.get("commit_sha"), arch_repo.get("commit_sha")
                    )
                )

    # --- component ids and labels ---
    md_component_ids = _COMPONENT_ID_RE.findall(markdown)
    if sorted(md_component_ids) != sorted(node_ids):
        problems.append(
            "component_set_mismatch: markdown={} model={}".format(
                sorted(md_component_ids), sorted(node_ids)
            )
        )
    name_by_id = {
        node.get("id"): (node.get("attributes") or {}).get("name")
        or node.get("id")
        for node in arch_nodes
    }
    for node in arch_nodes:
        node_id = node.get("id")
        expected = name_by_id.get(node_id)
        if not expected:
            continue
        pattern = r"### Component: {}\n\n- Component id: `{}`".format(
            re.escape(str(expected)), re.escape(str(node_id))
        )
        if not re.search(pattern, markdown):
            problems.append(
                "component_label_mismatch: {!r} expected {!r}".format(
                    node_id, expected
                )
            )

    # --- relationships ---
    md_rel_ids = set(_REL_TOKEN_RE.findall(markdown))
    for rel_id in rel_ids:
        if rel_id not in md_rel_ids:
            problems.append(
                "architecture_relationship_missing_from_context: {!r}".format(
                    rel_id
                )
            )
    for rel_id in md_rel_ids:
        if rel_id not in set(rel_ids):
            problems.append(
                "invented_architecture_relationship: {!r}".format(rel_id)
            )
    if md_rel_ids and md_rel_ids != set(rel_ids):
        problems.append(
            "relationship_cardinality_mismatch: context={} model={}".format(
                len(md_rel_ids), len(rel_ids)
            )
        )
    rel_by_id = {rel.get("id"): rel for rel in arch_rels}
    for rel in arch_rels:
        if "  - Endpoints: `{}` → `{}`".format(
            rel.get("source"), rel.get("target")
        ) not in markdown:
            problems.append(
                "relationship_endpoint_mismatch: {!r}".format(rel.get("id"))
            )
        if "  - Kind: {}".format(rel.get("kind")) not in markdown:
            problems.append(
                "relationship_kind_missing: {!r}".format(rel.get("id"))
            )

    unresolved_keys = {
        (u.get("kind"), u.get("source"), u.get("target"))
        for u in arch_unresolved
    }
    for rel in arch_rels:
        key = (rel.get("kind"), rel.get("source"), rel.get("target"))
        if key in unresolved_keys and rel.get("id") in md_rel_ids:
            problems.append(
                "unresolved_claim_presented_as_validated: {!r}".format(
                    rel.get("id")
                )
            )

    # --- concept tokens must be known architecture nodes ---
    for concept_id in set(_CONCEPT_TOKEN_RE.findall(markdown)):
        if concept_id not in set(node_ids):
            problems.append(
                "unknown_concept_reference: {!r}".format(concept_id)
            )

    # --- evidence paths must belong to their referenced entity ---
    for entity_id, paths in _extract_entity_evidence(markdown):
        if entity_id in rel_by_id:
            allowed = {
                item.get("path")
                for item in rel_by_id[entity_id].get("evidence") or []
            }
        elif entity_id in set(node_ids):
            node = next(
                n for n in arch_nodes if n.get("id") == entity_id
            )
            allowed = {
                item.get("path") for item in node.get("evidence") or []
            }
        else:
            problems.append(
                "evidence_for_unknown_entity: {!r}".format(entity_id)
            )
            continue
        for path in paths:
            if "(+" in path:
                continue
            if path not in allowed:
                problems.append(
                    "evidence_path_not_in_model: {!r} path={!r}".format(
                        entity_id, path
                    )
                )

    return problems


def validate_architecture_context_or_raise(
    markdown: str,
    architecture_document: Dict[str, Any],
    facts_document: Optional[Dict[str, Any]] = None,
) -> None:
    problems = validate_architecture_context_document(
        markdown, architecture_document, facts_document
    )
    if problems:
        raise ArchitectureContextValidationError(
            "architecture_context.md failed validation against M1–M4 "
            "artifacts:\n  - " + "\n  - ".join(problems)
        )


__all__ = [
    "ArchitectureContextError",
    "ArchitectureContextValidationError",
    "validate_architecture_context_document",
    "validate_architecture_context_or_raise",
    "validate_context_document",
    "validate_context_document_shape",
    "validate_context_or_raise",
    "validate_markdown_context_consistency",
    "validate_markdown_context_or_raise",
]
