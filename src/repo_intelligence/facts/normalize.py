"""Language-neutral entity normalization for RepositoryFacts.

Builds a flat ``entities`` list from language-specific extractor output.
M2 only records observations (kind, name, qualified identity, provenance).
Graph relationships remain an M3 concern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def build_entities(python: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Python extractor output into language-neutral entities.

    Identity rule (stable, documented for M3):
      - module:  qualified_name = path
      - class/function: qualified_name = path::name

    Same simple names in different paths do not collide.
    """
    entities: List[Dict[str, Any]] = []

    for module in python.get("modules") or []:
        path = module.get("path") or ""
        if not path:
            continue
        entities.append(
            {
                "entity_kind": "module",
                "name": module.get("name") or Path(path).stem,
                "qualified_name": module.get("qualified_name") or path,
                "language": module.get("language") or "python",
                "path": path,
            }
        )

    for cls in python.get("classes") or []:
        path = cls.get("path") or ""
        name = cls.get("name") or cls.get("symbol") or ""
        if not path or not name:
            continue
        entry: Dict[str, Any] = {
            "entity_kind": cls.get("entity_kind") or "class",
            "name": name,
            "qualified_name": cls.get("qualified_name")
            or "{}::{}".format(path, name),
            "language": cls.get("language") or "python",
            "path": path,
        }
        if "line" in cls:
            entry["line_start"] = cls.get("line_start", cls["line"])
        if "line_end" in cls:
            entry["line_end"] = cls["line_end"]
        entities.append(entry)

    for fn in python.get("functions") or []:
        path = fn.get("path") or ""
        name = fn.get("name") or fn.get("symbol") or ""
        if not path or not name:
            continue
        entry = {
            "entity_kind": fn.get("entity_kind") or "function",
            "name": name,
            "qualified_name": fn.get("qualified_name")
            or "{}::{}".format(path, name),
            "language": fn.get("language") or "python",
            "path": path,
        }
        if "line" in fn:
            entry["line_start"] = fn.get("line_start", fn["line"])
        if "line_end" in fn:
            entry["line_end"] = fn["line_end"]
        entities.append(entry)

    entities.sort(
        key=lambda e: (
            e.get("entity_kind") or "",
            e.get("qualified_name") or "",
            e.get("line_start") or 0,
            e.get("path") or "",
        )
    )
    return entities


def validate_entity_identity(entities: Iterable[Dict[str, Any]]) -> List[str]:
    """Return identity problems (empty list = OK).

    Uniqueness key: (entity_kind, qualified_name, language, line_start|0).
    ``line_start`` is included so legal redefined symbols (same ``path::name``
    at different lines in one file) remain distinct observations.
    path must be present; qualified_name must be non-empty.
    """
    problems: List[str] = []
    seen = set()
    for index, entity in enumerate(entities):
        kind = entity.get("entity_kind")
        qname = entity.get("qualified_name")
        language = entity.get("language")
        path = entity.get("path")
        name = entity.get("name")
        line = entity.get("line_start")
        if not kind:
            problems.append("entities[{}]: missing entity_kind".format(index))
        if not name:
            problems.append("entities[{}]: missing name".format(index))
        if not qname:
            problems.append("entities[{}]: missing qualified_name".format(index))
        if not language:
            problems.append("entities[{}]: missing language".format(index))
        if not path:
            problems.append("entities[{}]: missing path".format(index))
        key = (kind, qname, language, line if isinstance(line, int) else 0)
        if key in seen:
            problems.append(
                "entities[{}]: duplicate identity {} / {} (line={})".format(
                    index, kind, qname, key[3]
                )
            )
        seen.add(key)
    return problems


def _entity_key(kind: str, record: Dict[str, Any]) -> Any:
    """Identity key for cross-representation comparison.

    Classes/functions include line so redefined symbols at different lines
    are compared as distinct observations. Modules stay path-based.
    """
    qname = record.get("qualified_name")
    if not qname:
        path = record.get("path") or ""
        name = record.get("name") or record.get("symbol") or ""
        qname = "{}::{}".format(path, name) if name else path
    if kind == "module":
        return qname
    line = record.get("line_start")
    if line is None:
        line = record.get("line")
    return (qname, line)


def validate_cross_representation(
    entities: List[Dict[str, Any]],
    python: Dict[str, Any],
) -> List[str]:
    """Ensure entities[] agrees with python.modules/classes/functions.

    Dual representation is intentional:
      - entities[] → language-neutral identity for M3
      - python[]   → language-specific detail (imports, methods, parse errors)

    They must not contradict each other for overlapping symbols.
    """
    problems: List[str] = []

    def index_entities(kind: str) -> Dict[Any, Dict[str, Any]]:
        out: Dict[Any, Dict[str, Any]] = {}
        for e in entities:
            if e.get("entity_kind") != kind or not e.get("qualified_name"):
                continue
            out[_entity_key(kind, e)] = e
        return out

    def name_of(record: Dict[str, Any]) -> str:
        return record.get("name") or record.get("symbol") or ""

    def line_of(record: Dict[str, Any]) -> Any:
        if "line_start" in record:
            return record["line_start"]
        return record.get("line")

    ent_modules = index_entities("module")
    ent_classes = index_entities("class")
    ent_functions = index_entities("function")

    def index_python(kind: str, records: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
        out: Dict[Any, Dict[str, Any]] = {}
        for rec in records:
            if not rec.get("path"):
                continue
            if not rec.get("qualified_name"):
                rec = dict(rec)
                rec["qualified_name"] = "{}::{}".format(
                    rec.get("path") or "", name_of(rec)
                )
            out[_entity_key(kind, rec)] = rec
        return out

    py_modules = index_python("module", python.get("modules") or [])
    py_classes = index_python("class", python.get("classes") or [])
    py_functions = index_python("function", python.get("functions") or [])

    if set(ent_modules) != set(py_modules):
        problems.append(
            "entities modules != python.modules ({} vs {})".format(
                len(ent_modules), len(py_modules)
            )
        )
    if set(ent_classes) != set(py_classes):
        problems.append(
            "entities classes != python.classes ({} vs {})".format(
                len(ent_classes), len(py_classes)
            )
        )
    if set(ent_functions) != set(py_functions):
        problems.append(
            "entities functions != python.functions ({} vs {})".format(
                len(ent_functions), len(py_functions)
            )
        )

    # Detect line drift even when identity keys include line (set mismatch above
    # would otherwise hide a pure line corruption on a single record).
    def qname_of_key(key: Any) -> str:
        return key[0] if isinstance(key, tuple) else str(key)

    for kind, ent_map, py_map in (
        ("class", ent_classes, py_classes),
        ("function", ent_functions, py_functions),
    ):
        ent_lines: Dict[str, List[Any]] = {}
        py_lines: Dict[str, List[Any]] = {}
        for key, entity in ent_map.items():
            ent_lines.setdefault(qname_of_key(key), []).append(
                entity.get("line_start")
            )
        for key, record in py_map.items():
            py_lines.setdefault(qname_of_key(key), []).append(line_of(record))
        for qname in sorted(set(ent_lines) & set(py_lines)):
            if sorted(ent_lines[qname], key=lambda x: (x is None, x)) != sorted(
                py_lines[qname], key=lambda x: (x is None, x)
            ):
                problems.append(
                    "{} {}: line mismatch entity={} python={}".format(
                        kind,
                        qname,
                        sorted(ent_lines[qname], key=lambda x: (x is None, x)),
                        sorted(py_lines[qname], key=lambda x: (x is None, x)),
                    )
                )

    def compare(
        kind: str,
        ent_map: Dict[Any, Dict[str, Any]],
        py_map: Dict[Any, Dict[str, Any]],
    ) -> None:
        for key, entity in ent_map.items():
            record = py_map.get(key)
            if record is None:
                continue
            if entity.get("path") != record.get("path"):
                problems.append(
                    "{} {}: path mismatch entity={!r} python={!r}".format(
                        kind, key, entity.get("path"), record.get("path")
                    )
                )
            if entity.get("name") != name_of(record):
                problems.append(
                    "{} {}: name mismatch entity={!r} python={!r}".format(
                        kind, key, entity.get("name"), name_of(record)
                    )
                )
            if entity.get("language") and record.get("language"):
                if entity["language"] != record["language"]:
                    problems.append(
                        "{} {}: language mismatch".format(kind, key)
                    )
            entity_line = entity.get("line_start")
            python_line = line_of(record)
            if entity_line is not None and python_line is not None:
                if entity_line != python_line:
                    problems.append(
                        "{} {}: line mismatch entity={} python={}".format(
                            kind, key, entity_line, python_line
                        )
                    )

    compare("module", ent_modules, py_modules)
    compare("class", ent_classes, py_classes)
    compare("function", ent_functions, py_functions)
    return problems
