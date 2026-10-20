"""EngineeringModel construction pipeline.

    M2 repository_facts.v2
        + engineering_construction_rules.v1
        → rule evaluation
        → identity resolution / deduplication
        → evidence attachment
        → validation
        → EngineeringModel (repository_knowledge_model.v1)

Analyzers discover facts (M2). This module does **not** parse source or
build parallel graphs — it only transforms existing facts under the rulebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

from ..artifacts import (
    CONSTRUCTION_RULES_FILENAME,
    PRIMARY_SNAPSHOT_ID,
    construction_rules_path,
    default_artifacts_root,
    engineering_dir,
    facts_path as artifact_facts_path,
    resolve_output_root,
)
from ..construction import (
    ConstructionRulesError,
    evidence_only_matches,
    format_identity,
    load_construction_rules,
    node_rules_by_action,
    relationship_identity,
)
from .models import SCHEMA_VERSION, EngineeringModel
from .resolver import NodeIndex, parent_directory, resolve_dependency_target
from .validator import (
    EngineeringModelError,
    EngineeringModelValidationError,
    validate_engineering_model_or_raise,
)

ENGINEERING_MODEL_FILENAME = "engineering_model.json"
FACTS_SCHEMA = "repository_facts.v2"

# M2 interaction fact kind → EngineeringModel relationship kind.
# declares/exposes/configures walk the forward Agent → Tool → MCP Server →
# MCP Configuration path (M12).
INTERACTION_REL_KINDS = {
    "triggers": "TRIGGERS",
    "declares": "DECLARES",
    "exposes": "EXPOSED_BY",
    "configures": "CONFIGURED_BY",
}


def engineering_model_path(output_root: Path, snapshot_id: str) -> Path:
    return engineering_dir(output_root, snapshot_id) / ENGINEERING_MODEL_FILENAME


def load_facts_document(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise EngineeringModelError("M2 facts not found: {}".format(path))
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise EngineeringModelError(
            "Cannot read M2 facts {}: {}".format(path, exc)
        )
    if not isinstance(document, dict):
        raise EngineeringModelError("M2 facts root must be an object")
    if document.get("schema_version") != FACTS_SCHEMA:
        raise EngineeringModelError(
            "Expected {}, got {}".format(
                FACTS_SCHEMA, document.get("schema_version")
            )
        )
    return document


def _evidence_sort_key(item: Dict[str, Any]) -> Tuple:
    return (
        str(item.get("path") or ""),
        int(item.get("line_start") or 0),
        str(item.get("symbol") or ""),
        str(item.get("artifact_type") or ""),
        str(item.get("commit") or ""),
    )


def _merge_evidence(target: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> None:
    for item in incoming or []:
        if item not in target:
            target.append(dict(item))


def _node_rule_index(rules: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rule in rules.get("node_rules") or []:
        match = rule.get("match") or {}
        index[(match.get("fact_type"), match.get("kind"))] = rule
    return index


def _relationship_rule_index(
    rules: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for rule in rules.get("relationship_rules") or []:
        index[rule["rule_id"]] = rule
    return index


def _upsert_node(
    nodes: Dict[str, Dict[str, Any]],
    node_id: str,
    kind: str,
    subkind: str,
    classification: str,
    evidence: List[Dict[str, Any]],
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Create or merge a canonical node (construction-layer deduplication)."""
    if node_id not in nodes:
        nodes[node_id] = {
            "id": node_id,
            "kind": kind,
            "subkind": subkind,
            "classification": classification,
            "evidence": [],
        }
        if attributes:
            nodes[node_id]["attributes"] = dict(attributes)
    else:
        # first writer wins for kind/subkind; merge attributes if present
        if attributes:
            existing = nodes[node_id].setdefault("attributes", {})
            for key, value in attributes.items():
                existing.setdefault(key, value)
    _merge_evidence(nodes[node_id]["evidence"], evidence)


def _analysis_section(facts_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Record deterministic analyzers that contributed M2 facts."""
    sources = sorted(
        {
            f.get("source")
            for f in facts_doc.get("facts") or []
            if f.get("source")
        }
    )
    analyzers: List[Dict[str, Any]] = []

    # Map M2 sources → analyzer technology metadata (no identity impact).
    if "entities" in sources or "python" in sources:
        import sys

        analyzers.append(
            {
                "analyzer": "python",
                "method": "ast",
                "library": "python-stdlib",
                "version": "{}.{}".format(*sys.version_info[:2]),
            }
        )
    if "configuration" in sources:
        yaml_version = "unavailable"
        try:
            import yaml

            yaml_version = getattr(yaml, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            pass
        analyzers.append(
            {
                "analyzer": "yaml",
                "method": "structured-parser",
                "library": "PyYAML",
                "version": yaml_version,
            }
        )
        analyzers.append(
            {
                "analyzer": "json",
                "method": "structured-parser",
                "library": "python-stdlib",
                "version": "stdlib",
            }
        )
        toml_version = "stdlib-tomllib-or-fallback"
        try:
            import tomllib  # noqa: F401

            toml_version = "python-tomllib"
        except Exception:  # noqa: BLE001
            try:
                import toml

                toml_version = getattr(toml, "__version__", "toml")
            except Exception:  # noqa: BLE001
                toml_version = "path-classification-only"
        analyzers.append(
            {
                "analyzer": "toml",
                "method": "structured-parser",
                "library": "python",
                "version": toml_version,
            }
        )
    if "filesystem" in sources:
        analyzers.append(
            {
                "analyzer": "filesystem",
                "method": "walk",
                "library": "python-stdlib",
                "version": "stdlib",
            }
        )
    if any(s in sources for s in ("agents", "mcp", "documentation", "tests", "dependencies")):
        analyzers.append(
            {
                "analyzer": "repository-metadata",
                "method": "structured-scan",
                "library": "repository-intelligence",
                "version": str(
                    (facts_doc.get("analyzers") or [{}])[0].get("version") or "0.0.0"
                ),
            }
        )
    # Git/pinned snapshot provenance always present via M1/M2 envelope.
    analyzers.append(
        {
            "analyzer": "git",
            "method": "pinned-snapshot",
            "library": "repository-intelligence",
            "version": str(
                (facts_doc.get("analyzers") or [{}])[0].get("version") or "0.0.0"
            ),
        }
    )
    # Tree-sitter is the future multi-language foundation; not used for V1
    # EngineeringModel identity (Python semantic extraction uses stdlib ast).
    # Recorded only when the library is actually installed.
    try:
        import tree_sitter

        analyzers.append(
            {
                "analyzer": "source-code",
                "method": "tree-sitter",
                "library": "tree-sitter",
                "version": getattr(tree_sitter, "__version__", "unknown"),
            }
        )
    except Exception:  # noqa: BLE001
        pass

    # stable order
    analyzers = sorted(analyzers, key=lambda a: (a.get("analyzer"), a.get("method")))
    return {"strategy": "deterministic", "analyzers": analyzers}


def _coverage_metadata(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Diagnostic coverage metrics (not ontology)."""
    from collections import Counter

    by_type = Counter(f.get("fact_type") for f in facts)
    by_source = Counter(f.get("source") for f in facts)
    return {
        "facts_total": len(facts),
        "facts_by_type": dict(sorted(by_type.items())),
        "facts_by_source": dict(sorted(by_source.items())),
    }


def build_engineering_model(
    facts_doc: Dict[str, Any],
    rules: Dict[str, Any],
) -> EngineeringModel:
    """Apply construction rules to M2 facts and return a validated model."""
    facts = list(facts_doc.get("facts") or [])
    commit = (facts_doc.get("snapshot") or {}).get("commit_sha") or ""
    repository = {
        "name": (facts_doc.get("repository") or {}).get("name") or "",
        "url": (facts_doc.get("repository") or {}).get("url") or "",
        "commit_sha": commit,
    }
    if len(commit) != 40:
        raise EngineeringModelError(
            "facts snapshot.commit_sha must be a 40-char SHA"
        )

    nodes: Dict[str, Dict[str, Any]] = {}
    relationships: Dict[str, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    # fact_id → canonical node id for CREATE_NODE facts (interaction endpoints).
    fact_id_to_node: Dict[str, str] = {}
    fact_by_id: Dict[str, Dict[str, Any]] = {
        f.get("fact_id"): f for f in facts if f.get("fact_id")
    }

    node_rule_index = _node_rule_index(rules)
    evidence_only = set(evidence_only_matches(rules))
    create_rules = node_rules_by_action(rules, "CREATE_NODE")
    rel_index = _relationship_rule_index(rules)

    # ------------------------------------------------------------------
    # Phase 1 — nodes (CREATE_NODE only; evidence-only facts skipped)
    # ------------------------------------------------------------------
    for fact in facts:
        fact_type = fact.get("fact_type")
        kind = fact.get("kind")
        key = (fact_type, kind)
        if key in evidence_only:
            # still allow CREATE_NODE if also listed (should not happen for EO)
            rule = node_rule_index.get(key)
            if rule is None or rule.get("action") != "CREATE_NODE":
                continue
        rule = node_rule_index.get(key)
        if rule is None or rule.get("action") != "CREATE_NODE":
            continue
        attrs = fact.get("attributes") or {}
        evidence = list(fact.get("evidence") or [])
        if not evidence:
            # canonical nodes require evidence — skip non-evidenced facts
            continue
        try:
            node_id = format_identity(rule["identity"]["template"], attrs)
        except Exception as exc:  # noqa: BLE001
            raise EngineeringModelError(
                "identity failed for fact {}: {}".format(
                    fact.get("fact_id"), exc
                )
            )
        node_attributes: Dict[str, Any] = {
            "path": attrs.get("path"),
            "qualified_name": attrs.get("qualified_name"),
            "name": attrs.get("name") or attrs.get("symbol"),
            "fact_type": fact_type,
            "fact_kind": kind,
        }
        # M12: server scope only on MCP tool facts (absent elsewhere).
        if attrs.get("server") is not None:
            node_attributes["server"] = attrs.get("server")
        _upsert_node(
            nodes,
            node_id,
            rule["node"]["kind"],
            rule["node"]["subkind"],
            rule.get("classification") or "observed",
            evidence,
            attributes=node_attributes,
        )
        fact_id = fact.get("fact_id")
        if fact_id:
            # Prefer CREATE_NODE identity when multiple facts share a path.
            fact_id_to_node.setdefault(fact_id, node_id)

    index = NodeIndex(nodes.values())

    def add_relationship(
        kind: str,
        source_id: str,
        target_id: str,
        classification: str,
        evidence: List[Dict[str, Any]],
    ) -> None:
        if not source_id or not target_id:
            return
        if source_id not in nodes or target_id not in nodes:
            return
        rel_id = relationship_identity(kind, source_id, target_id)
        if rel_id in relationships:
            _merge_evidence(relationships[rel_id]["evidence"], evidence)
            return
        relationships[rel_id] = {
            "id": rel_id,
            "kind": kind,
            "source": source_id,
            "target": target_id,
            "classification": classification,
            "evidence": [],
        }
        _merge_evidence(relationships[rel_id]["evidence"], evidence)

    def add_unresolved(
        source_id: str,
        target_label: str,
        kind: str,
        reason: str,
        evidence: List[Dict[str, Any]],
    ) -> None:
        unresolved.append(
            {
                "source": source_id or "",
                "target": target_label or "",
                "kind": kind,
                "reason": reason,
                "evidence": [dict(e) for e in evidence or []],
            }
        )

    # ------------------------------------------------------------------
    # Phase 2 — observed relationships (IMPORTS, DEPENDS_ON, and the
    # interaction family: triggers + M12 declares/exposes/configures)
    # ------------------------------------------------------------------
    for fact in facts:
        fact_type = fact.get("fact_type")
        kind = fact.get("kind")
        attrs = fact.get("attributes") or {}
        evidence = list(fact.get("evidence") or [])
        if not evidence:
            continue

        if fact_type == "import" and kind in ("import", "import_from"):
            rule_id = (
                "rel.import.import" if kind == "import" else "rel.import.import_from"
            )
            rule = rel_index.get(rule_id)
            if rule is None:
                continue
            source_path = attrs.get("path") or ""
            source_id = index.resolve_path_node(source_path, prefer="file")
            module = attrs.get("module") or ""
            target_id = index.resolve_imported_module(module)
            if source_id is None:
                add_unresolved(
                    source_path or module,
                    module,
                    "IMPORTS",
                    "missing_endpoint",
                    evidence,
                )
                continue
            if target_id is None:
                add_unresolved(
                    source_id,
                    module,
                    "IMPORTS",
                    "target_not_resolved",
                    evidence,
                )
                continue
            # refresh index if target somehow missing (should exist)
            if target_id not in nodes:
                add_unresolved(
                    source_id, module, "IMPORTS", "target_not_resolved", evidence
                )
                continue
            add_relationship(
                "IMPORTS",
                source_id,
                target_id,
                rule.get("classification") or "observed",
                evidence,
            )

        elif fact_type == "dependency" and kind == "dependencies":
            rule = rel_index.get("rel.dependency.dependencies")
            if rule is None:
                continue
            source_path = attrs.get("source") or ""
            source_id = index.resolve_path_node(source_path, prefer="file")
            name = attrs.get("name") or ""
            target_label = resolve_dependency_target(name)
            if source_id is None:
                add_unresolved(
                    source_path or name, name, "DEPENDS_ON", "missing_endpoint", evidence
                )
                continue
            if target_label is None:
                add_unresolved(
                    source_id, name, "DEPENDS_ON", "target_not_resolved", evidence
                )
                continue
            if target_label not in nodes:
                _upsert_node(
                    nodes,
                    target_label,
                    "external_package",
                    "package",
                    "observed",
                    evidence,
                    attributes={"name": name},
                )
                index = NodeIndex(nodes.values())
            add_relationship(
                "DEPENDS_ON",
                source_id,
                target_label,
                rule.get("classification") or "observed",
                evidence,
            )

        elif fact_type == "interaction" and kind in INTERACTION_REL_KINDS:
            rel_kind = INTERACTION_REL_KINDS[kind]
            rule = rel_index.get("rel.interaction.{}".format(kind))
            if rule is None:
                continue
            source_fact_id = attrs.get("source_fact_id") or ""
            target_fact_id = attrs.get("target_fact_id") or ""
            target_label = (
                attrs.get("target_path")
                or attrs.get("target_label")
                or target_fact_id
                or ""
            )
            source_node = fact_id_to_node.get(source_fact_id)
            target_node = fact_id_to_node.get(target_fact_id)
            if not source_fact_id or source_node is None:
                add_unresolved(
                    source_node or source_fact_id or attrs.get("source_path") or "",
                    target_label,
                    rel_kind,
                    "missing_endpoint",
                    evidence,
                )
                continue
            if not target_fact_id or target_node is None:
                # M2 carries the declared reason for candidate claims
                # (ambiguous_identity / target_not_resolved).
                add_unresolved(
                    source_node,
                    target_label,
                    rel_kind,
                    attrs.get("reason") or "target_not_resolved",
                    evidence,
                )
                continue
            if source_node not in nodes or target_node not in nodes:
                add_unresolved(
                    source_node,
                    target_label,
                    rel_kind,
                    "target_not_resolved",
                    evidence,
                )
                continue
            # M12: multiple configuration sources for one server keep every
            # candidate edge marked conflicting — never an arbitrary pick.
            classification = (
                "conflicting"
                if attrs.get("conflicting")
                else (rule.get("classification") or "observed")
            )
            add_relationship(
                rel_kind,
                source_node,
                target_node,
                classification,
                evidence,
            )

    # refresh after possible external package inserts
    index = NodeIndex(nodes.values())

    # ------------------------------------------------------------------
    # Phase 3 — derived CONTAINS relationships
    # ------------------------------------------------------------------
    def contains(source_id: str, target_id: str) -> None:
        if not source_id or not target_id:
            return
        if source_id not in nodes or target_id not in nodes:
            return
        # Provenance: derived edges retain evidence from all material
        # contributing facts — both source and target nodes.
        evidence: List[Dict[str, Any]] = []
        _merge_evidence(evidence, nodes[source_id].get("evidence") or [])
        _merge_evidence(evidence, nodes[target_id].get("evidence") or [])
        add_relationship("CONTAINS", source_id, target_id, "derived", evidence)

    # directory → file (parent path)
    for node in list(nodes.values()):
        if node["kind"] != "artifact" or node["subkind"] != "file":
            continue
        path = (node.get("attributes") or {}).get("path")
        if not path:
            # parse from id artifact:file:<path>
            path = node["id"].split(":", 2)[2] if node["id"].count(":") >= 2 else ""
        parent = parent_directory(path or "")
        if not parent:
            continue
        dir_id = "artifact:directory:{}".format(parent)
        contains(dir_id, node["id"])

    # file → module / class / function (path equality)
    file_ids = {
        (n.get("attributes") or {}).get("path")
        or (n["id"].split(":", 2)[2] if n["id"].count(":") >= 2 else ""): n["id"]
        for n in nodes.values()
        if n.get("kind") == "artifact" and n.get("subkind") == "file"
    }
    module_ids_by_path = {
        (n["id"].split(":", 1)[1] if ":" in n["id"] else ""): n["id"]
        for n in nodes.values()
        if n.get("kind") == "code_entity" and n.get("subkind") == "module"
    }

    for node in list(nodes.values()):
        if node.get("kind") != "code_entity":
            continue
        if node.get("subkind") == "module":
            path = node["id"].split(":", 1)[1] if ":" in node["id"] else ""
            file_id = file_ids.get(path)
            if file_id:
                contains(file_id, node["id"])
            continue
        # class / function
        rest = node["id"].split(":", 1)[1] if ":" in node["id"] else ""
        path = rest.split("::", 1)[0] if rest else ""
        file_id = file_ids.get(path)
        if file_id:
            contains(file_id, node["id"])
        module_id = module_ids_by_path.get(path)
        if module_id:
            contains(module_id, node["id"])

    # class → nested function when qualified_name is nested under class
    class_ids = [
        n
        for n in nodes.values()
        if n.get("kind") == "code_entity" and n.get("subkind") == "class"
    ]
    function_nodes = [
        n
        for n in nodes.values()
        if n.get("kind") == "code_entity" and n.get("subkind") == "function"
    ]
    for fn in function_nodes:
        rest = fn["id"].split(":", 1)[1] if ":" in fn["id"] else ""
        # rest = path::qualified_tail
        if "::" not in rest:
            continue
        path, tail = rest.split("::", 1)
        for cls in class_ids:
            cls_rest = cls["id"].split(":", 1)[1] if ":" in cls["id"] else ""
            if "::" not in cls_rest:
                continue
            cls_path, cls_tail = cls_rest.split("::", 1)
            if cls_path != path:
                continue
            # nested: tail starts with Class. or Class::
            if tail.startswith(cls_tail + ".") or tail.startswith(cls_tail + "::"):
                contains(cls["id"], fn["id"])

    # ------------------------------------------------------------------
    # Phase 4 — sort deterministically
    # ------------------------------------------------------------------
    ordered_nodes = [nodes[key] for key in sorted(nodes)]
    for node in ordered_nodes:
        node["evidence"] = sorted(
            node.get("evidence") or [], key=_evidence_sort_key
        )
        if "attributes" in node and node["attributes"] is not None:
            # drop null-only attributes for compactness? keep as-is if any value
            attrs = node["attributes"]
            if attrs and all(v is None for v in attrs.values()):
                node.pop("attributes", None)

    ordered_rels = sorted(
        relationships.values(),
        key=lambda r: (
            r.get("kind") or "",
            r.get("source") or "",
            r.get("target") or "",
            r.get("id") or "",
        ),
    )
    for rel in ordered_rels:
        rel["evidence"] = sorted(rel.get("evidence") or [], key=_evidence_sort_key)

    # de-duplicate unresolved claims
    seen_unresolved = set()
    deduped_unresolved = []
    for claim in sorted(
        unresolved,
        key=lambda u: (
            u.get("kind") or "",
            str(u.get("source") or ""),
            str(u.get("target") or ""),
            u.get("reason") or "",
        ),
    ):
        key = (
            claim.get("kind"),
            claim.get("source"),
            claim.get("target"),
            claim.get("reason"),
        )
        if key in seen_unresolved:
            # merge evidence into existing
            for existing in deduped_unresolved:
                if (
                    existing.get("kind"),
                    existing.get("source"),
                    existing.get("target"),
                    existing.get("reason"),
                ) == key:
                    _merge_evidence(existing["evidence"], claim.get("evidence") or [])
                    existing["evidence"] = sorted(
                        existing["evidence"], key=_evidence_sort_key
                    )
                    break
            continue
        seen_unresolved.add(key)
        claim["evidence"] = sorted(
            claim.get("evidence") or [], key=_evidence_sort_key
        )
        deduped_unresolved.append(claim)

    model = EngineeringModel(
        schema_version=SCHEMA_VERSION,
        repository=repository,
        analysis=_analysis_section(facts_doc),
        engineering={
            "nodes": ordered_nodes,
            "relationships": ordered_rels,
            "unresolved": deduped_unresolved,
            "metadata": {
                "construction_rules": rules.get("schema_version"),
                "fact_schema": facts_doc.get("schema_version"),
                "snapshot_id": (facts_doc.get("snapshot") or {}).get("snapshot_id"),
            },
        },
        architecture={
            "nodes": [],
            "relationships": [],
            "unresolved": [],
            "metadata": {"status": "not_implemented"},
        },
        mappings=[],
        metadata={
            "analysis_coverage": _coverage_metadata(facts),
            "counts": {},  # filled after validation from counts()
        },
    )

    document = model.to_dict()
    validate_engineering_model_or_raise(document, expected_commit_sha=commit)
    model.metadata["counts"] = model.counts()
    # re-serialize with counts
    document = model.to_dict()
    validate_engineering_model_or_raise(document, expected_commit_sha=commit)
    return model


def write_engineering_model_artifact(
    model: EngineeringModel, output_root: Path, snapshot_id: str
) -> Path:
    dest = engineering_model_path(output_root, snapshot_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    document = model.to_dict()
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    dest.write_text(text, encoding="utf-8")
    return dest


def create_engineering_model(
    facts_path: Optional[Path] = None,
    rules_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
) -> Tuple[EngineeringModel, Path]:
    """Build EngineeringModel from M2 facts + M3 rules and write the artifact."""
    root = resolve_output_root(output_root)

    if facts_path is None:
        if snapshot_id is None:
            snapshot_id = PRIMARY_SNAPSHOT_ID
        facts_path = artifact_facts_path(root, snapshot_id)
    facts_doc = load_facts_document(Path(facts_path))

    if snapshot_id is None:
        snapshot_id = (facts_doc.get("snapshot") or {}).get("snapshot_id") or (
            PRIMARY_SNAPSHOT_ID
        )

    if rules_path is None:
        rules_path = construction_rules_path(root, snapshot_id)
        if not Path(rules_path).is_file():
            # fall back to primary snapshot matrix
            rules_path = construction_rules_path(
                root, PRIMARY_SNAPSHOT_ID
            )
        if not Path(rules_path).is_file():
            # last resort: project primary location
            rules_path = construction_rules_path(
                default_artifacts_root(), PRIMARY_SNAPSHOT_ID
            )

    try:
        rules = load_construction_rules(Path(rules_path))
    except ConstructionRulesError as exc:
        raise EngineeringModelError(str(exc))

    model = build_engineering_model(facts_doc, rules)
    path = write_engineering_model_artifact(model, root, snapshot_id)
    return model, path


__all__ = [
    "ENGINEERING_MODEL_FILENAME",
    "EngineeringModelError",
    "EngineeringModelValidationError",
    "build_engineering_model",
    "create_engineering_model",
    "engineering_model_path",
    "load_facts_document",
    "write_engineering_model_artifact",
]
