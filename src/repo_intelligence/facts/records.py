"""Generic FactRecord construction for repository_facts.v2.

Each analyzer's specialized output is normalized into a FactRecord:

    {
      "fact_id":    "fact:<type>:<stable-key>",
      "fact_type":  open string (file, directory, entity, ...),
      "kind":       open sub-kind string,
      "source":     producing analyzer/stage,
      "attributes": open analyzer-specific object,
      "evidence":   [EvidenceRef, ...]
    }

fact_id is derived only from stable identity fields (never order, timestamps,
or array position). New analyzers can introduce new fact_type/kind/source
values without schema changes.

Filesystem observations use::

    fact_type=file      kind=repository_file     source=filesystem
    fact_type=directory kind=repository_directory source=filesystem

and are the **only** canonical representation of repository files/directories
(not mirrored under repository.files / repository.directories).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import make_evidence
from .models import sort_facts


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def compute_fact_id(
    fact_type: str,
    kind: str,
    source: str,
    attributes: Dict[str, Any],
) -> str:
    """Deterministic stable identity for a fact.

    Conceptually ``fact:<stable identity>``. Uses only natural keys:
    path, qualified_name, (path, line, module), (source, section, name), etc.
    """
    path = _text(attributes.get("path"))
    if fact_type in ("file", "directory", "configuration", "documentation", "test"):
        identity = path
    elif fact_type == "entity":
        identity = ":".join(
            (
                _text(attributes.get("language")),
                _text(kind),
                _text(attributes.get("qualified_name")),
            )
        )
        # Redefined symbols (same path::name at different lines) stay unique.
        line = attributes.get("line_start")
        if line is not None:
            identity = "{}:{}".format(identity, line)
    elif fact_type == "import":
        identity = ":".join(
            (
                path,
                _text(attributes.get("line")),
                _text(kind),
                _text(attributes.get("module")),
            )
        )
    elif fact_type == "dependency":
        identity = ":".join(
            (
                _text(attributes.get("source")),
                _text(attributes.get("section")),
                _text(attributes.get("name")),
            )
        )
    elif fact_type == "artifact":
        identity = ":".join((source, _text(kind), path))
    elif fact_type == "interaction":
        # Stable identity: explicit endpoints only (no timestamps/order).
        # Unresolved candidates carry a target_label instead of a
        # target_fact_id (same deterministic slot per source).
        identity = "{}:{}:{}".format(
            _text(kind),
            _text(attributes.get("source_fact_id")),
            _text(attributes.get("target_fact_id"))
            or _text(attributes.get("target_label")),
        )
    elif fact_type == "tool":
        # MCP tools are server-scoped (the MCP calling identity);
        # namespace-free declarations are builtin-scoped.
        server = _text(attributes.get("server"))
        name = _text(attributes.get("name"))
        if server:
            identity = "mcp:{}:{}".format(server, name)
        else:
            identity = "builtin:{}".format(name)
    elif fact_type == "mcp_server":
        identity = "server:{}".format(_text(attributes.get("name")))
    else:
        # Open extension path: prefer path / qualified_name, else canonical JSON.
        identity = path or _text(attributes.get("qualified_name"))
        if not identity:
            identity = json.dumps(attributes, sort_keys=True, separators=(",", ":"))
    return "fact:{}:{}".format(fact_type, identity)


def make_fact(
    fact_type: str,
    kind: str,
    source: str,
    attributes: Dict[str, Any],
    evidence: Optional[List[Dict[str, Any]]] = None,
    fact_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one FactRecord envelope."""
    attrs = dict(attributes)
    if fact_id is None:
        fact_id = compute_fact_id(fact_type, kind, source, attrs)
    return {
        "fact_id": fact_id,
        "fact_type": fact_type,
        "kind": kind,
        "source": source,
        "attributes": attrs,
        "evidence": list(evidence or []),
    }


def _evidence_list(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lift a single attached evidence object into array form."""
    evidence = record.get("evidence")
    if evidence is None:
        return []
    if isinstance(evidence, list):
        return list(evidence)
    return [evidence]


def _record_attributes(record: Dict[str, Any]) -> Dict[str, Any]:
    attrs = dict(record)
    attrs.pop("evidence", None)
    return attrs


def _index_python_records(python: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """Index python module/class/function records by qualified identity.

    Key is ``qualified_name`` for modules (file-level). For classes/functions
    the key is ``(qualified_name, line_start)`` so redefined symbols at
    different lines do not overwrite each other.
    """
    index: Dict[Any, Dict[str, Any]] = {}
    for key in ("modules", "classes", "functions"):
        for record in python.get(key) or []:
            qname = record.get("qualified_name")
            path = record.get("path") or ""
            name = record.get("name") or record.get("symbol") or ""
            if not qname:
                qname = "{}::{}".format(path, name) if name else path
            if not qname:
                continue
            if key == "modules":
                index[qname] = record
            else:
                line = record.get("line_start", record.get("line"))
                index[(qname, line)] = record
    return index


def _lookup_python_record(
    index: Dict[Any, Dict[str, Any]], entity: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    qname = entity.get("qualified_name")
    if entity.get("entity_kind") == "module":
        return index.get(qname)
    line = entity.get("line_start")
    if line is None:
        line = entity.get("line")
    # Prefer exact (qname, line); fall back to any record with that qname.
    if (qname, line) in index:
        return index[(qname, line)]
    for key, record in index.items():
        if isinstance(key, tuple) and key[0] == qname:
            return record
    return None


def _merge_entity_attributes(
    entity: Dict[str, Any], python_record: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge language-neutral identity with language-specific detail."""
    attrs = _record_attributes(entity)
    if not python_record:
        return attrs
    py = _record_attributes(python_record)
    identity_keys = {
        "entity_kind",
        "name",
        "qualified_name",
        "language",
        "path",
    }
    for key, value in py.items():
        if key in identity_keys:
            continue
        attrs[key] = value
    # Prefer entity line fields when both define them.
    if "line_start" in _record_attributes(entity):
        attrs["line_start"] = entity.get("line_start")
        if "line" in py and "line" not in attrs:
            attrs["line"] = py.get("line")
    return attrs


def build_fact_records(
    analyzer_results: Dict[str, Any],
    entities: List[Dict[str, Any]],
    commit: str,
) -> List[Dict[str, Any]]:
    """Normalize all analyzer outputs + entities into generic FactRecords.

    *analyzer_results* is the run_analyzers shape (with evidence already
    attached where applicable). *entities* is the language-neutral list.
    """
    facts: List[Dict[str, Any]] = []
    repository = analyzer_results.get("repository") or {}
    python = analyzer_results.get("python") or {}

    for record in repository.get("files") or []:
        item = dict(record)
        path = item.get("path") or ""
        if not path:
            continue
        facts.append(
            make_fact(
                fact_type="file",
                kind="repository_file",
                source="filesystem",
                attributes=_record_attributes(item),
                evidence=[make_evidence(path=path, artifact_type="file", commit=commit)],
            )
        )

    for record in repository.get("directories") or []:
        item = dict(record)
        path = item.get("path") or ""
        if not path:
            continue
        facts.append(
            make_fact(
                fact_type="directory",
                kind="repository_directory",
                source="filesystem",
                attributes=_record_attributes(item),
                evidence=[
                    make_evidence(path=path, artifact_type="directory", commit=commit)
                ],
            )
        )

    py_index = _index_python_records(python)
    for entity in entities:
        item = dict(entity)
        evidence = _evidence_list(item)
        attrs = _merge_entity_attributes(
            item, _lookup_python_record(py_index, entity)
        )
        if not evidence:
            path = attrs.get("path")
            if path:
                evidence = [
                    make_evidence(
                        path=path,
                        artifact_type="python",
                        commit=commit,
                        line_start=attrs.get("line_start"),
                        line_end=attrs.get("line_end"),
                        symbol=_text(attrs.get("name")) or None,
                    )
                ]
        facts.append(
            make_fact(
                fact_type="entity",
                kind=_text(attrs.get("entity_kind")) or "entity",
                source="entities",
                attributes=attrs,
                evidence=evidence,
            )
        )

    for record in python.get("imports") or []:
        item = dict(record)
        evidence = _evidence_list(item)
        attrs = _record_attributes(item)
        facts.append(
            make_fact(
                fact_type="import",
                kind=_text(attrs.get("kind")) or "import",
                source="python",
                attributes=attrs,
                evidence=evidence,
            )
        )

    category_specs = (
        ("configuration", "configuration", "configuration"),
        ("dependencies", "dependency", "dependencies"),
        ("agents", "artifact", "agents"),
        ("mcp", "artifact", "mcp"),
        ("documentation", "documentation", "documentation"),
        ("tests", "test", "tests"),
    )
    for result_key, fact_type, source in category_specs:
        for record in analyzer_results.get(result_key) or []:
            item = dict(record)
            evidence = _evidence_list(item)
            attrs = _record_attributes(item)
            # Prefer the analyzer's type field as kind; fall through to fact_type.
            kind = _text(attrs.get("type")) or fact_type
            if fact_type == "dependency":
                kind = _text(attrs.get("section")) or "dependency"
            facts.append(
                make_fact(
                    fact_type=fact_type,
                    kind=kind,
                    source=source,
                    attributes=attrs,
                    evidence=evidence,
                )
            )

    return sort_facts(facts)
