"""Test artifact facts (existence + symbols; no purpose claims)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

from .filesystem import DEFAULT_IGNORED_DIRS, iter_repository_paths


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _is_test_path(rel: str) -> bool:
    posix = _posix(rel)
    name = Path(rel).name
    parts = posix.split("/")
    if "test" in parts or "tests" in parts:
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    if name.endswith("_test.go"):
        return True
    if name.startswith("test") and name.endswith((".js", ".ts")):
        return True
    return False


def _is_test_directory(rel: str) -> bool:
    posix = _posix(rel)
    return posix in ("test", "tests", "spec", "specs") or posix.endswith(
        ("/test", "/tests", "/spec", "/specs")
    )


def _python_test_symbols(path: Path) -> Dict[str, List[str]]:
    functions: List[str] = []
    classes: List[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return {"functions": functions, "classes": classes}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            # pytest-style Test* classes
            if node.name.startswith("Test"):
                methods = [
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name.startswith("test_")
                ]
                if methods or node.name.startswith("Test"):
                    classes.append(node.name)
    return {"functions": sorted(functions), "classes": sorted(classes)}


def analyze_tests(root: Path) -> List[Dict[str, Any]]:
    """Identify test files/directories and Python test symbols."""
    root = Path(root)
    files, directories = iter_repository_paths(root, ignored_dirs=DEFAULT_IGNORED_DIRS)
    results: List[Dict[str, Any]] = []
    seen_dirs = set()

    for rel in directories:
        if _is_test_directory(rel):
            results.append({"path": rel, "type": "test_directory"})
            seen_dirs.add(rel)

    for rel in files:
        if not _is_test_path(rel):
            continue
        entry: Dict[str, Any] = {"path": rel, "type": "test_file"}
        if rel.endswith(".py"):
            symbols = _python_test_symbols(root / rel)
            if symbols["functions"]:
                entry["functions"] = symbols["functions"]
            if symbols["classes"]:
                entry["classes"] = symbols["classes"]
        results.append(entry)

    # de-dupe path (a file under tests/ listed once)
    results.sort(key=lambda e: (e["path"], e["type"]))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for entry in results:
        key = entry["path"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped
