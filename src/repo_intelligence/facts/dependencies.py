"""Dependency manifest facts (declared packages only)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .filesystem import iter_repository_paths

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_TOML_DEP_TABLES = (
    "dependencies",
    "dev-dependencies",
    "dev_dependencies",
    "optional-dependencies",
    "optional_dependencies",
    "build-dependencies",
    "build_dependencies",
)


def _strip_requirement(line: str) -> str:
    line = line.strip().strip(",").strip()
    line = line.strip("\"'")
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-"):
        return ""
    # drop environment markers and extras for a simple name
    line = line.split(";", 1)[0].strip()
    match = _REQ_LINE.match(line)
    if match:
        return match.group(1)
    return ""


def _parse_requirements_txt(path: Path, rel: str) -> List[Dict[str, Any]]:
    deps: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return deps
    for raw in text.splitlines():
        name = _strip_requirement(raw)
        if name:
            deps.append({"name": name, "source": rel})
    return deps


def _parse_package_json(path: Path, rel: str) -> List[Dict[str, Any]]:
    deps: List[Dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return deps
    if not isinstance(data, dict):
        return deps
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        block = data.get(section)
        if isinstance(block, dict):
            for name in sorted(block.keys()):
                deps.append({"name": name, "source": rel, "section": section})
    return deps


def _iter_toml_dependency_names(data: Any, prefix: str = "") -> List[str]:
    """Yield dependency names from nested TOML structures (best-effort)."""
    names: List[str] = []
    if not isinstance(data, dict):
        return names
    for key, value in data.items():
        key_l = str(key).lower().replace("_", "-")
        if key_l in _TOML_DEP_TABLES and isinstance(value, dict):
            names.extend(str(k) for k in value.keys())
        elif isinstance(value, dict):
            # e.g. [project.optional-dependencies]
            names.extend(_iter_toml_dependency_names(value, prefix + str(key) + "."))
    return names


def _parse_pyproject(path: Path, rel: str) -> List[Dict[str, Any]]:
    deps: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return deps

    data: Any = None
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        tomllib = None  # type: ignore

    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = None

    if data is None:
        # Minimal fallback: collect quoted names inside dependencies = [ ... ]
        # lines under common keys — still deterministic, not full TOML.
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            # lines that look like list items: "pkg>=1.0",
            if (
                stripped.startswith('"')
                or stripped.startswith("'")
            ) and any(ch in stripped for ch in "><=!~ "):
                name = _strip_requirement(stripped.strip(","))
                if name:
                    deps.append({"name": name, "source": rel, "section": "dependencies"})
            elif stripped.startswith("dependencies") and "=" in stripped:
                # single-line array: dependencies = ["a", "b"]
                if "[" in stripped and "]" in stripped:
                    inner = stripped.split("[", 1)[1].split("]", 1)[0]
                    for part in inner.split(","):
                        name = _strip_requirement(part)
                        if name:
                            deps.append(
                                {"name": name, "source": rel, "section": "dependencies"}
                            )
        # de-dupe while preserving order
        seen = set()
        unique: List[Dict[str, Any]] = []
        for d in deps:
            key = (d["name"], d.get("section", ""), d["source"])
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    # tomllib path
    project = data.get("project") if isinstance(data, dict) else None
    if isinstance(project, dict):
        for item in project.get("dependencies") or []:
            name = _strip_requirement(str(item))
            if name:
                deps.append({"name": name, "source": rel, "section": "dependencies"})
        optional = project.get("optional-dependencies") or {}
        if isinstance(optional, dict):
            for section in sorted(optional.keys()):
                for item in optional[section] or []:
                    name = _strip_requirement(str(item))
                    if name:
                        deps.append(
                            {
                                "name": name,
                                "source": rel,
                                "section": "optional-dependencies:{}".format(section),
                            }
                        )
    for name in sorted(set(_iter_toml_dependency_names(data))):
        entry = {"name": name, "source": rel, "section": "dependencies"}
        if entry not in deps:
            # avoid duplicating project.dependencies already captured
            if not any(d["name"] == name and d["source"] == rel for d in deps):
                deps.append(entry)
    return deps


def _parse_setup_py(path: Path, rel: str) -> List[Dict[str, Any]]:
    """Extract install_requires strings from setup.py without executing it."""
    deps: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return deps
    # very small deterministic scan for quoted requirement-like strings
    for match in re.finditer(
        r"install_requires\s*=\s*\[(.*?)\]", text, flags=re.DOTALL
    ):
        block = match.group(1)
        for item in re.finditer(r"['\"]([^'\"]+)['\"]", block):
            name = _strip_requirement(item.group(1))
            if name:
                deps.append({"name": name, "source": rel, "section": "install_requires"})
    return deps


def analyze_dependencies(root: Path) -> List[Dict[str, Any]]:
    """Record declared dependencies from known manifest files."""
    root = Path(root)
    files, _ = iter_repository_paths(root)
    results: List[Dict[str, Any]] = []

    for rel in files:
        name = Path(rel).name.lower()
        path = root / rel
        if name == "requirements.txt" or (
            name.startswith("requirements") and name.endswith(".txt")
        ):
            results.extend(_parse_requirements_txt(path, rel))
        elif name == "package.json":
            results.extend(_parse_package_json(path, rel))
        elif name == "pyproject.toml":
            results.extend(_parse_pyproject(path, rel))
        elif name == "setup.py":
            results.extend(_parse_setup_py(path, rel))
        elif name == "setup.cfg":
            # record manifest presence; parsing optional for M2
            results.append(
                {"name": "(manifest)", "source": rel, "section": "setup.cfg"}
            )

    results.sort(key=lambda d: (d["source"], d.get("section", ""), d["name"]))
    return results
