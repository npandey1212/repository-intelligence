"""Deterministic ArchitectureContext generator (architecture_context.v1).

Canonical flow:

    M1–M4 artifacts (architecture model, repository facts)
      + contract/architecture_context_template.json   (fixed repo-level template)
        → populated context document   (in-memory, template-shaped)
        → architecture_context.md      (the generated artifact)

Repository-agnostic: every name, component, relationship, path, and
technology string in the output is derived from input artifact data.
No LLM. No domain hard-coding. Does not expand the architecture model.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from ..architecture.validator import ArchitectureModelError
from ..artifacts import (
    architecture_model_path as artifact_architecture_model_path,
    facts_path as artifact_facts_path,
    resolve_output_root,
)
from .models import (
    ARCHITECTURE_CONTEXT_SCHEMA_VERSION,
    COMMIT_LABEL,
    CONTRACT_LABEL,
    CONTEXT_SECTION_KEYS,
    CONTEXT_TEMPLATE_PATH,
    KEY_DOCUMENT_LIMIT,
    KEY_FILE_LIMIT,
    SECTION_HEADINGS,
    SNAPSHOT_LABEL,
    SPECIAL_LIST_FIELDS,
    TITLE,
    UNRESOLVED_SAMPLE_LIMIT,
    ArchitectureContextError,
    _DEFAULT_SOURCE_PATHS,
    default_context_markdown_path,
    discover_snapshot_id,
    display_path,
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
)
from .validator import (
    ArchitectureContextValidationError,
    validate_architecture_context_or_raise,
    validate_context_or_raise,
    validate_markdown_context_or_raise,
)

_PATH_EVIDENCE_KEYS = ("path", "artifact_type", "commit", "line_start", "line_end")

_INTERFACES_OVERVIEW = (
    "No reliable evidence in the available artifacts establishes interface "
    "contracts. Every Interfaces list below is intentionally empty."
)


class ContextBuildResult(NamedTuple):
    """Result of create_architecture_context: document + Markdown + path."""

    document: Dict[str, Any]
    markdown: str
    markdown_path: Path


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_json_document(path: Path, label: str) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ArchitectureModelError(
            "{} not found: {}".format(label, source)
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArchitectureModelError(
            "Unable to read {} {}: {}".format(label, source, exc)
        )
    if not isinstance(document, dict):
        raise ArchitectureModelError(
            "{} must be a JSON object: {}".format(label, source)
        )
    return document


# ---------------------------------------------------------------------------
# Generic input extractors (data-driven, repository-agnostic)
# ---------------------------------------------------------------------------


def _node_name(node: Dict[str, Any]) -> str:
    attributes = node.get("attributes") or {}
    return str(attributes.get("name") or node.get("id") or "")


def _evidence_paths(items: List[Dict[str, Any]]) -> List[str]:
    paths = []
    for item in items or []:
        path = item.get("path")
        if path and path not in paths:
            paths.append(path)
    return paths


def _copy_path_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy model evidence restricted to path-evidence keys."""
    copied: List[Dict[str, Any]] = []
    for item in items or []:
        entry = {
            key: item[key]
            for key in _PATH_EVIDENCE_KEYS
            if key in item and item[key] is not None
        }
        if entry.get("path"):
            copied.append(entry)
    return copied


def _format_evidence(paths: List[str]) -> str:
    if not paths:
        return "_none recorded_"
    return ", ".join("`{}`".format(p) for p in sorted(paths))


def _fact_counts(facts_document: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fact in facts_document.get("facts") or []:
        fact_type = str(fact.get("fact_type") or "unknown")
        counts[fact_type] = counts.get(fact_type, 0) + 1
    return counts


def _fact_paths(facts_document: Dict[str, Any], fact_type: str) -> List[str]:
    paths = []
    for fact in facts_document.get("facts") or []:
        if fact.get("fact_type") != fact_type:
            continue
        path = str((fact.get("attributes") or {}).get("path") or "")
        if path and path not in paths:
            paths.append(path)
    return sorted(paths)


def _top_level_directories(facts_document: Dict[str, Any]) -> List[str]:
    dirs = set()
    for fact in facts_document.get("facts") or []:
        if fact.get("fact_type") != "directory":
            continue
        path = str((fact.get("attributes") or {}).get("path") or "")
        if path and "/" not in path:
            dirs.add(path)
    return sorted(dirs)


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


def _languages(engineering: Dict[str, Any]) -> List[str]:
    """Languages observed as evidence artifact types on code entities."""
    types = set()
    for node in engineering.get("nodes") or []:
        if node.get("kind") != "code_entity":
            continue
        for item in node.get("evidence") or []:
            artifact_type = item.get("artifact_type")
            if artifact_type:
                types.add(str(artifact_type))
    return sorted(value.title() for value in types)


def _engineering_artifact_groups(
    engineering: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Group engineering artifact paths by subkind (discovered, not fixed)."""
    grouped: Dict[str, List[str]] = {}
    for node in engineering.get("nodes") or []:
        if node.get("kind") != "artifact":
            continue
        subkind = str(node.get("subkind") or "artifact")
        paths = _evidence_paths(node.get("evidence") or [])
        if not paths:
            path = (node.get("attributes") or {}).get("path")
            if path:
                paths = [str(path)]
        if not paths:
            continue
        bucket = grouped.setdefault(subkind, [])
        for path in paths:
            if path not in bucket:
                bucket.append(path)
    return [
        {"group": group, "paths": sorted(grouped[group])}
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


def _identity(
    architecture_document: Dict[str, Any], facts_document: Dict[str, Any]
) -> Dict[str, str]:
    arch_repo = architecture_document.get("repository") or {}
    facts_repo = facts_document.get("repository") or {}
    snapshot = facts_document.get("snapshot") or {}
    engineering = architecture_document.get("engineering") or {}
    eng_metadata = engineering.get("metadata") or {}
    return {
        "name": str(
            arch_repo.get("name") or facts_repo.get("name") or ""
        ),
        "commit_sha": str(
            arch_repo.get("commit_sha") or snapshot.get("commit_sha") or ""
        ),
        "snapshot_id": str(
            snapshot.get("snapshot_id")
            or eng_metadata.get("snapshot_id")
            or ""
        ),
    }


# ---------------------------------------------------------------------------
# Template population
# ---------------------------------------------------------------------------


def _build_architecture_section(
    architecture_document: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    architecture = architecture_document.get("architecture") or {}
    nodes = sorted(
        architecture.get("nodes") or [], key=lambda n: str(n.get("id"))
    )
    rels = sorted(
        architecture.get("relationships") or [],
        key=lambda r: str(r.get("id")),
    )

    components: List[Dict[str, Any]] = []
    for node in nodes:
        component: Dict[str, Any] = {
            "id": str(node.get("id")),
            "name": _node_name(node),
            "kind": str(node.get("kind") or "unknown"),
            "classification": str(node.get("classification") or "unknown"),
            "evidence": _copy_path_evidence(node.get("evidence") or []),
        }
        if node.get("subkind") is not None:
            component["subkind"] = str(node.get("subkind"))
        components.append(component)

    relationships: List[Dict[str, Any]] = []
    for rel in rels:
        relationships.append(
            {
                "id": str(rel.get("id")),
                "kind": str(rel.get("kind") or "unknown"),
                "source": str(rel.get("source")),
                "target": str(rel.get("target")),
                "classification": str(rel.get("classification") or "unknown"),
                "evidence": _copy_path_evidence(rel.get("evidence") or []),
            }
        )

    return {"components": components, "relationships": relationships}


def _build_workflows(
    architecture_section: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    components_by_id = {
        component["id"]: component
        for component in architecture_section["components"]
    }
    outbound: Dict[str, List[Dict[str, Any]]] = {}
    for rel in architecture_section["relationships"]:
        outbound.setdefault(rel["source"], []).append(rel)

    workflows: List[Dict[str, Any]] = []
    for source_id in sorted(outbound):
        group = sorted(
            outbound[source_id],
            key=lambda r: (r["target"], r["kind"], r["id"]),
        )
        source = components_by_id.get(source_id) or {}
        evidence: List[Dict[str, Any]] = []
        seen_paths = set()
        for rel in group:
            for item in rel["evidence"]:
                path = item.get("path")
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    evidence.append(item)
        workflows.append(
            {
                "source_component_id": source_id,
                "source_name": source.get("name") or source_id,
                "participant_ids": [rel["target"] for rel in group],
                "participant_names": [
                    (components_by_id.get(rel["target"]) or {}).get("name")
                    or rel["target"]
                    for rel in group
                ],
                "relationship_ids": [rel["id"] for rel in group],
                "evidence": evidence,
            }
        )
    return workflows


def _build_sources(
    identity: Dict[str, str], source_paths: Optional[List[str]]
) -> List[Dict[str, Any]]:
    paths = list(source_paths) if source_paths else list(_DEFAULT_SOURCE_PATHS)
    sources: List[Dict[str, Any]] = []
    for path in paths:
        entry: Dict[str, Any] = {
            "label": source_label(str(path)),
            "path": str(path),
        }
        if identity["commit_sha"]:
            entry["commit_sha"] = identity["commit_sha"]
        if identity["snapshot_id"]:
            entry["snapshot_id"] = identity["snapshot_id"]
        sources.append(entry)
    sources.append(
        {
            "label": source_label("architecture_context_template.json"),
            "path": display_path(CONTEXT_TEMPLATE_PATH),
        }
    )
    return sources


def _known_unknowns(
    test_count: int, unresolved: List[Dict[str, Any]]
) -> List[str]:
    entries = [
        "Repository description, purpose, project type, and primary "
        "domains are not established by the available artifacts; those "
        "template fields stay empty.",
        "Interface contracts (APIs, commands, events, external "
        "interfaces) are not established; every Interfaces list is empty.",
        "Build, test, run, and deploy commands are not established; "
        "every Operations list is empty.",
        unknown_test_files_line(test_count),
        unknown_unresolved_line(len(unresolved)),
    ]
    for claim in unresolved[:UNRESOLVED_SAMPLE_LIMIT]:
        entries.append(
            "Unresolved claim: {} {} -> {} ({})".format(
                claim.get("kind") or "unknown",
                claim.get("source") or "",
                claim.get("target") or "",
                claim.get("reason") or "unknown",
            )
        )
    if len(unresolved) > UNRESOLVED_SAMPLE_LIMIT:
        entries.append(
            "Unresolved claim: (+{} more; see the architecture model)".format(
                len(unresolved) - UNRESOLVED_SAMPLE_LIMIT
            )
        )
    return entries


def _analysis_limitations(
    architecture_document: Dict[str, Any],
    facts_document: Dict[str, Any],
    component_count: int,
    relationship_count: int,
    key_files_total: int,
    key_files_shown: int,
    document_total: int,
    document_shown: int,
) -> List[str]:
    mappings = architecture_document.get("mappings") or []
    engineering = architecture_document.get("engineering") or {}
    fact_counts = _fact_counts(facts_document)
    entries = [
        limitation_mappings_line(len(mappings)),
        "Validated M4 slice: {nodes} components / {rels} relationships; "
        "the underlying engineering model reports {eng_nodes} nodes, "
        "{eng_rels} relationships, and {eng_unresolved} unresolved "
        "claims.".format(
            nodes=component_count,
            rels=relationship_count,
            eng_nodes=len(engineering.get("nodes") or []),
            eng_rels=len(engineering.get("relationships") or []),
            eng_unresolved=_engineering_unresolved_count(
                architecture_document
            ),
        ),
        "Relationships come only from the architecture model; nothing is "
        "inferred from proximity, naming, or imports.",
        "Document bodies are not read at generation time; no purpose or "
        "narrative claim is paraphrased from file contents.",
        "M2 recorded {facts} facts across {types} fact types; this "
        "context summarizes them and does not duplicate the fact "
        "store.".format(
            facts=sum(fact_counts.values()), types=len(fact_counts)
        ),
        "Runtime behavior, performance, and deployment state are not "
        "established by the available evidence.",
    ]
    if key_files_total > key_files_shown:
        entries.append(
            "Structure key files list the first {shown} of {total} "
            "observed documentation and configuration paths.".format(
                shown=key_files_shown, total=key_files_total
            )
        )
    if document_total > document_shown:
        entries.append(
            "Documentation key documents list the first {shown} of "
            "{total} observed documentation paths.".format(
                shown=document_shown, total=document_total
            )
        )
    return entries


def build_architecture_context_document(
    architecture_document: Dict[str, Any],
    facts_document: Dict[str, Any],
    source_paths: Optional[List[str]] = None,
    template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Populate the architecture context template from M1–M4 artifacts.

    Returns an in-memory document whose keys are exactly the template keys.
    Deterministic given identical inputs; does not modify input documents
    and does not write files.
    """
    if "architecture" not in architecture_document:
        raise ArchitectureContextError(
            "Expected document with architecture section "
            "(architecture_model.json)"
        )
    if "facts" not in facts_document:
        raise ArchitectureContextError(
            "Expected repository_facts document with facts[]"
        )
    if template is None:
        template = load_context_template()

    context: Dict[str, Any] = copy.deepcopy(template)

    engineering = architecture_document.get("engineering") or {}
    identity = _identity(architecture_document, facts_document)
    architecture_section = _build_architecture_section(architecture_document)
    components = architecture_section["components"]
    relationships = architecture_section["relationships"]
    workflows = _build_workflows(architecture_section)
    unresolved = list(
        (architecture_document.get("architecture") or {}).get("unresolved")
        or []
    )
    mappings = architecture_document.get("mappings") or []

    doc_paths = _fact_paths(facts_document, "documentation")
    cfg_paths = _fact_paths(facts_document, "configuration")
    test_paths = _fact_paths(facts_document, "test")
    top_dirs = _top_level_directories(facts_document)
    packages = _external_packages(engineering)
    dependency_names = _dependency_fact_names(facts_document)
    languages = _languages(engineering)
    artifact_groups = _engineering_artifact_groups(engineering)
    important_artifacts = [
        "{}: {} item(s)".format(group["group"], len(group["paths"]))
        for group in artifact_groups
    ]
    key_files_total = len(set(doc_paths) | set(cfg_paths))
    key_files = sorted(set(doc_paths) | set(cfg_paths))[:KEY_FILE_LIMIT]
    key_documents = sorted(doc_paths, key=readme_first_key)[
        :KEY_DOCUMENT_LIMIT
    ]

    # --- repository (identity only; other fields stay as the template) ---
    context["repository"]["name"] = identity["name"]

    # --- structure ---
    context["structure"]["overview"] = (
        "{dirs} top-level directories and {groups} engineering artifact "
        "groupings observed in the input artifacts. Key files are "
        "documentation and configuration paths ({total} observed, "
        "{shown} listed); all paths come from facts and engineering "
        "nodes, not from directory traversal at generation time.".format(
            dirs=len(top_dirs),
            groups=len(artifact_groups),
            total=key_files_total,
            shown=len(key_files),
        )
    )
    context["structure"]["key_directories"] = top_dirs
    context["structure"]["key_files"] = key_files
    context["structure"]["important_artifacts"] = important_artifacts

    # --- technology (template has no overview field) ---
    context["technology"]["languages"] = languages
    context["technology"]["libraries"] = sorted(
        set(packages) | set(dependency_names)
    )

    # --- architecture (components/relationships are exactly M4's) ---
    context["architecture"]["overview"] = (
        "Validated M4 slice: {nodes} components and {rels} relationships, "
        "alongside {maps} engineering -> architecture mappings and "
        "{unresolved} unresolved claims. Relationships are only those "
        "recorded in the architecture model (never inferred from "
        "proximity, naming, or imports); boundaries are not established "
        "by the available evidence.".format(
            nodes=len(components),
            rels=len(relationships),
            maps=len(mappings),
            unresolved=len(unresolved),
        )
    )
    context["architecture"]["components"] = components
    context["architecture"]["relationships"] = relationships

    # --- workflows ---
    context["workflows"]["overview"] = (
        "{count} workflow group(s): validated architecture relationships "
        "grouped by their source component. No ordering and no "
        "intermediate step is invented.".format(count=len(workflows))
    )
    context["workflows"]["workflows"] = workflows

    # --- interfaces: no reliable evidence -> stays empty ---
    context["interfaces"]["overview"] = _INTERFACES_OVERVIEW

    # --- documentation ---
    context["documentation"]["overview"] = (
        "{total} documentation file(s) observed in facts ({shown} listed "
        "as key documents). Document bodies are not embedded in this "
        "context.".format(total=len(doc_paths), shown=len(key_documents))
    )
    context["documentation"]["key_documents"] = key_documents

    # --- evidence ---
    context["evidence"]["sources"] = _build_sources(identity, source_paths)
    context["evidence"]["known_unknowns"] = _known_unknowns(
        len(test_paths), unresolved
    )
    context["evidence"]["analysis_limitations"] = _analysis_limitations(
        architecture_document,
        facts_document,
        component_count=len(components),
        relationship_count=len(relationships),
        key_files_total=key_files_total,
        key_files_shown=len(key_files),
        document_total=len(doc_paths),
        document_shown=len(key_documents),
    )
    return context


def build_architecture_context(
    architecture_document: Dict[str, Any],
    facts_document: Dict[str, Any],
    source_paths: Optional[List[str]] = None,
    template: Optional[Dict[str, Any]] = None,
) -> str:
    """Build architecture_context.md from M1–M4 artifacts + the template."""
    context = build_architecture_context_document(
        architecture_document,
        facts_document,
        source_paths=source_paths,
        template=template,
    )
    return render_architecture_context_markdown(context)


# ---------------------------------------------------------------------------
# Markdown projection (reads ONLY the populated context document)
# ---------------------------------------------------------------------------


def _document_problems(context: Any) -> List[str]:
    if not isinstance(context, dict):
        return ["document_must_be_a_json_object"]
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
    for key in CONTEXT_SECTION_KEYS[1:]:
        if not isinstance(context.get(key), dict):
            problems.append("section_not_an_object: {!r}".format(key))
    return problems


def _render_string_list(field: str, values: List[str]) -> Tuple[str, List[str]]:
    if not values:
        return "inline", [empty_list_line(field)]
    lines = ["### {}".format(field_label(field)), ""]
    lines.extend("- {}".format(value) for value in values)
    return "block", lines


def _render_evidence_value(values: List[str]) -> str:
    if not values:
        return "_none recorded_"
    return ", ".join("`{}`".format(value) for value in sorted(values))


def _render_components(
    components: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    if not components:
        return "inline", [empty_list_line("components")]
    lines: List[str] = []
    for index, component in enumerate(components):
        if index:
            lines.append("")
        kind = str(component.get("kind") or "unknown")
        if component.get("subkind") is not None:
            kind = "{}/{}".format(kind, component.get("subkind"))
        lines.extend(
            [
                "### Component: {}".format(component.get("name")),
                "",
                "- Component id: `{}`".format(component.get("id")),
                "- Kind: {}".format(kind),
                "- Classification: {}".format(
                    component.get("classification")
                ),
                "- Evidence: {}".format(
                    _render_evidence_value(
                        _evidence_paths(component.get("evidence") or [])
                    )
                ),
            ]
        )
    return "block", lines


def _render_relationships(
    relationships: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    if not relationships:
        return "inline", [empty_list_line("relationships")]
    lines = ["### Relationships", ""]
    for index, rel in enumerate(relationships):
        if index:
            lines.append("")
        lines.extend(
            [
                "- `{}`".format(rel.get("id")),
                "  - Kind: {}".format(rel.get("kind")),
                "  - Classification: {}".format(rel.get("classification")),
                "  - Endpoints: `{}` → `{}`".format(
                    rel.get("source"), rel.get("target")
                ),
                "  - Evidence: {}".format(
                    _render_evidence_value(
                        _evidence_paths(rel.get("evidence") or [])
                    )
                ),
            ]
        )
    return "block", lines


def _render_workflows(
    workflows: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    if not workflows:
        return "inline", [empty_list_line("workflows")]
    lines: List[str] = []
    for index, workflow in enumerate(workflows):
        if index:
            lines.append("")
        participant_names = [
            str(name) for name in workflow.get("participant_names") or []
        ]
        relationship_ids = [
            str(rel_id) for rel_id in workflow.get("relationship_ids") or []
        ]
        lines.extend(
            [
                "### Workflow: {}".format(workflow.get("source_name")),
                "",
                "- Source: `{}`".format(workflow.get("source_component_id")),
                "- Participants: {}".format(
                    _render_evidence_value(participant_names)
                ),
                "- Relationships: {}".format(
                    _render_evidence_value(relationship_ids)
                ),
                "- Evidence: {}".format(
                    _render_evidence_value(
                        _evidence_paths(workflow.get("evidence") or [])
                    )
                ),
            ]
        )
    return "block", lines


def _render_sources(sources: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not sources:
        return "inline", [empty_list_line("sources")]
    lines = ["### Sources", ""]
    lines.extend(
        "- {}".format(format_source_line(entry)) for entry in sources
    )
    return "block", lines


def _render_special(field: str, value: List[Any]) -> Tuple[str, List[str]]:
    if field == "components":
        return _render_components(value)
    if field == "relationships":
        return _render_relationships(value)
    if field == "workflows":
        return _render_workflows(value)
    if field == "sources":
        return _render_sources(value)
    return "inline", [empty_list_line(field)]


def _render_repository_identity(context: Dict[str, Any]) -> List[str]:
    sources = (context.get("evidence") or {}).get("sources") or []
    commit_sha, snapshot_id = source_identity(sources)
    return [
        format_line(COMMIT_LABEL, commit_sha, code=True),
        format_line(SNAPSHOT_LABEL, snapshot_id, code=True),
    ]


def _render_section(
    context: Dict[str, Any], key: str, heading: str
) -> List[str]:
    section = context.get(key) or {}
    lines: List[str] = ["## {}".format(heading), ""]
    overview = section.get("overview")
    if overview:
        lines.extend([str(overview), ""])
    for field, value in section.items():
        if field == "overview":
            continue
        if field in SPECIAL_LIST_FIELDS:
            kind, chunk = _render_special(field, value)
        elif isinstance(value, list):
            kind, chunk = _render_string_list(field, value)
        else:
            kind = "inline"
            chunk = [
                format_line(
                    field_label(field), value, code=(field == "name")
                )
            ]
        if key == "repository" and field == "name":
            kind = "block"
            chunk = chunk + _render_repository_identity(context)
        if kind == "block":
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend(chunk)
            lines.append("")
        else:
            lines.extend(chunk)
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")
    return lines


def render_architecture_context_markdown(context: Dict[str, Any]) -> str:
    """Render architecture_context.md strictly from a populated document."""
    problems = _document_problems(context)
    if problems:
        raise ArchitectureContextError(
            "Expected a populated architecture context document "
            "(contract/architecture_context_template.json):\n  - "
            + "\n  - ".join(problems)
        )

    lines: List[str] = ["# {}".format(TITLE), ""]
    lines.append(
        format_line(CONTRACT_LABEL, context["schema_version"], code=True)
    )
    lines.append("")
    for key, heading in zip(CONTEXT_SECTION_KEYS[1:], SECTION_HEADINGS):
        lines.extend(_render_section(context, key, heading))

    markdown = "\n".join(lines)
    if not markdown.endswith("\n"):
        markdown += "\n"
    return markdown


# ---------------------------------------------------------------------------
# Creation: write architecture_context.md
# ---------------------------------------------------------------------------


def _snapshot_id_from_path(path: Path) -> str:
    """Run id when the path follows <...>/artifacts/<snapshot_id>/<kind>/<file>."""
    parts = Path(path).parts
    for index in range(len(parts) - 2):
        if parts[index] == "artifacts":
            return str(parts[index + 1])
    return ""


def create_architecture_context(
    architecture_model_path: Optional[Path] = None,
    facts_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
    template_path: Optional[Path] = None,
    validate: bool = True,
) -> ContextBuildResult:
    """Load M1–M4 artifacts + the template, validate, write the Markdown.

    Writes architecture_context.md under
    <output>/<snapshot_id>/architecture_context/ unless overridden.

    The snapshot/run id is never hard-coded: it comes from the explicit
    argument, the existing run context under ``output_root``, or the
    supplied input artifacts (document identity / artifact path).
    """
    root = resolve_output_root(output_root)

    # --- resolve input paths (explicit > selected run > discovered run) ---
    discovered: Optional[str] = None
    if architecture_model_path is None or facts_path is None:
        discovered = (
            snapshot_id
            if snapshot_id is not None
            else discover_snapshot_id(root)
        )
    if architecture_model_path is None:
        arch_path = artifact_architecture_model_path(root, discovered)
    else:
        arch_path = Path(architecture_model_path)

    if facts_path is None:
        facts_file = artifact_facts_path(root, discovered)
    else:
        facts_file = Path(facts_path)

    architecture_document = _load_json_document(
        arch_path, "ArchitectureModel"
    )
    facts_document = _load_json_document(facts_file, "RepositoryFacts")
    template = load_context_template(template_path)

    # --- run id from the run context or the supplied input artifacts ---
    if snapshot_id is None:
        snapshot_id = (
            discovered
            or _identity(architecture_document, facts_document)["snapshot_id"]
            or _snapshot_id_from_path(arch_path)
            or _snapshot_id_from_path(facts_file)
        )
    if snapshot_id is None and output_path is None:
        raise ArchitectureContextError(
            "Unable to determine the snapshot/run id from the supplied "
            "inputs; pass snapshot_id or input artifacts under "
            "<output>/<snapshot_id>/."
        )

    source_paths = [display_path(arch_path), display_path(facts_file)]
    context = build_architecture_context_document(
        architecture_document,
        facts_document,
        source_paths=source_paths,
        template=template,
    )
    markdown = render_architecture_context_markdown(context)

    if validate:
        validate_context_or_raise(
            context, architecture_document, facts_document
        )
        validate_markdown_context_or_raise(markdown, context)
        validate_architecture_context_or_raise(
            markdown, architecture_document, facts_document
        )

    if output_path is None:
        markdown_dest = default_context_markdown_path(root, snapshot_id)
    else:
        markdown_dest = Path(output_path)

    markdown_dest.parent.mkdir(parents=True, exist_ok=True)
    markdown_dest.write_text(markdown, encoding="utf-8")
    return ContextBuildResult(
        document=context, markdown=markdown, markdown_path=markdown_dest
    )


__all__ = [
    "ARCHITECTURE_CONTEXT_SCHEMA_VERSION",
    "ArchitectureContextError",
    "ArchitectureContextValidationError",
    "ContextBuildResult",
    "build_architecture_context",
    "build_architecture_context_document",
    "create_architecture_context",
    "render_architecture_context_markdown",
]
