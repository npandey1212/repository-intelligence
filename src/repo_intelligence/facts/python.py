"""Python source facts via deterministic AST parsing (no LLM).

Extraction is language-specific; emitted symbol records carry
language-neutral identity fields (entity_kind, name, qualified_name,
language) so M2 output can support cross-language normalization.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from .filesystem import DEFAULT_IGNORED_DIRS, iter_repository_paths

LANGUAGE = "python"


def module_qualified_name(rel_path: str) -> str:
    """Stable module identity: repository-relative path."""
    return rel_path


def symbol_qualified_name(rel_path: str, name: str) -> str:
    """Stable symbol identity: ``path::name`` (disambiguates same simple names)."""
    return "{}::{}".format(rel_path, name)


def _module_path(rel_path: str) -> str:
    """Logical module path (POSIX-style) for provenance."""
    return rel_path


def _symbol_line(node: ast.AST, name: str, source_lines: List[str]) -> int:
    """Best-effort line where *name* appears for a definition node.

    Decorated classes/functions may have ``lineno`` on the first decorator
    (Python 3.8+). Prefer the line that actually contains the symbol name
    within a small window after ``node.lineno``.
    """
    start = getattr(node, "lineno", 1)
    window = 1 + len(getattr(node, "decorator_list", []) or []) + 2
    end = min(len(source_lines), start + window)
    for index in range(start - 1, end):
        if name in source_lines[index]:
            return index + 1
    return start


def _class_record(rel_path: str, node: ast.ClassDef, line: int) -> Dict[str, Any]:
    name = node.name
    record: Dict[str, Any] = {
        "path": rel_path,
        "symbol": name,
        "name": name,
        "kind": "class",
        "entity_kind": "class",
        "qualified_name": symbol_qualified_name(rel_path, name),
        "language": LANGUAGE,
        "line": line,
        "line_start": line,
        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
    }
    end_line = getattr(node, "end_lineno", None)
    if isinstance(end_line, int) and end_line >= line:
        record["line_end"] = end_line
    return record


def _function_record(
    rel_path: str, node: ast.AST, name: str, line: int
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": rel_path,
        "symbol": name,
        "name": name,
        "kind": "function",
        "entity_kind": "function",
        "qualified_name": symbol_qualified_name(rel_path, name),
        "language": LANGUAGE,
        "line": line,
        "line_start": line,
    }
    end_line = getattr(node, "end_lineno", None)
    if isinstance(end_line, int) and end_line >= line:
        record["line_end"] = end_line
    return record


def _module_record(rel_path: str, parse_error: bool) -> Dict[str, Any]:
    return {
        "path": _module_path(rel_path),
        "name": Path(rel_path).stem,
        "entity_kind": "module",
        "qualified_name": module_qualified_name(rel_path),
        "language": LANGUAGE,
        "classes": [],
        "functions": [],
        "imports": [],
        "parse_error": parse_error,
    }


def _collect_from_module(
    rel_path: str,
    tree: ast.AST,
    classes: List[Dict[str, Any]],
    functions: List[Dict[str, Any]],
    imports: List[Dict[str, Any]],
    source_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    module_classes: List[str] = []
    module_functions: List[str] = []
    module_imports: List[str] = []
    lines = source_lines or []

    def line_for(node: ast.AST, name: str) -> int:
        if lines:
            return _symbol_line(node, name, lines)
        return getattr(node, "lineno", 1)

    info = _module_record(rel_path, parse_error=False)

    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.ClassDef):
            module_classes.append(node.name)
            record = _class_record(rel_path, node, line_for(node, node.name))
            classes.append(record)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.append(node.name)
            record = _function_record(
                rel_path, node, node.name, line_for(node, node.name)
            )
            functions.append(record)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                module_imports.append(name)
                imports.append(
                    {
                        "path": rel_path,
                        "module": name,
                        "kind": "import",
                        "language": LANGUAGE,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level and not name:
                name = "." * node.level
            elif node.level:
                name = "." * node.level + name
            module_imports.append(name)
            imports.append(
                {
                    "path": rel_path,
                    "module": name,
                    "kind": "import_from",
                    "language": LANGUAGE,
                    "line": node.lineno,
                    "names": [a.name for a in node.names],
                }
            )
        elif isinstance(node, ast.If):
            for sub in list(node.body) + list(node.orelse):
                if isinstance(
                    sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if isinstance(sub, ast.ClassDef):
                        module_classes.append(sub.name)
                        classes.append(
                            _class_record(rel_path, sub, line_for(sub, sub.name))
                        )
                    else:
                        module_functions.append(sub.name)
                        functions.append(
                            _function_record(
                                rel_path, sub, sub.name, line_for(sub, sub.name)
                            )
                        )

    module_imports = sorted(set(module_imports))
    info["classes"] = sorted(module_classes)
    info["functions"] = sorted(module_functions)
    info["imports"] = module_imports
    return info


def analyze_python(root: Path) -> Dict[str, Any]:
    """Extract modules / classes / functions / imports from Python files."""
    root = Path(root)
    files, _ = iter_repository_paths(root, ignored_dirs=DEFAULT_IGNORED_DIRS)

    modules: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    imports: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for rel in files:
        if not rel.endswith(".py"):
            continue
        path = root / rel
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            source = None
        if source is None:
            modules.append(_module_record(rel, parse_error=True))
            errors.append(
                {
                    "path": rel,
                    "kind": "read_error",
                    "detail": "Python file could not be read",
                }
            )
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            modules.append(_module_record(rel, parse_error=True))
            errors.append(
                {
                    "path": rel,
                    "kind": "syntax_error",
                    "detail": "Python file could not be parsed",
                }
            )
            continue
        info = _collect_from_module(
            rel,
            tree,
            classes,
            functions,
            imports,
            source_lines=source.splitlines(),
        )
        modules.append(info)

    modules.sort(key=lambda m: m["path"])
    classes.sort(key=lambda c: (c["path"], c["line"], c["symbol"]))
    functions.sort(key=lambda f: (f["path"], f["line"], f["symbol"]))
    imports.sort(key=lambda i: (i["path"], i["line"], i["module"]))

    return {
        "modules": modules,
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "parse_errors": errors,
    }
