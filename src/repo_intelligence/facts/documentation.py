"""Documentation artifact facts (paths/types only; no summarization)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .filesystem import iter_repository_paths

DOC_DIR_HINTS = (
    "/docs/",
    "/doc/",
    "/documentation/",
    "/runbooks/",
    "/.github/",
)

DOC_NAMES = {
    "readme",
    "readme.md",
    "readme.rst",
    "readme.txt",
    "changelog",
    "changelog.md",
    "contributing",
    "contributing.md",
    "code_of_conduct.md",
    "architecture.md",
    "license",
    "license.md",
    "license.txt",
    "notice",
}

DOC_SUFFIXES = (".md", ".markdown", ".rst", ".adoc")


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _is_documentation(rel: str) -> bool:
    posix = "/" + _posix(rel)
    name = Path(rel).name.lower()

    if name in DOC_NAMES:
        return True
    if name.startswith("readme"):
        return True
    if name.endswith(DOC_SUFFIXES):
        for hint in DOC_DIR_HINTS:
            if posix.startswith(hint) or hint in posix:
                return True
        # top-level markdown often docs; nested markdown under docs counted;
        # also count runbooks-like paths
        if "/runbook" in posix:
            return True
    return False


def _doc_type(rel: str) -> str:
    name = Path(rel).name.lower()
    posix = "/" + _posix(rel)
    if name.startswith("readme"):
        return "readme"
    if "/runbook" in posix:
        return "runbook"
    if name.startswith("changelog"):
        return "changelog"
    if name.endswith((".md", ".markdown")):
        return "markdown"
    if name.endswith(".rst"):
        return "restructuredtext"
    if name.endswith(".adoc"):
        return "asciidoc"
    return "documentation"


def analyze_documentation(root: Path) -> List[Dict[str, Any]]:
    """Identify documentation artifacts without summarizing content."""
    root = Path(root)
    files, _ = iter_repository_paths(root)
    results: List[Dict[str, Any]] = []

    for rel in files:
        if _is_documentation(rel):
            results.append({"path": rel, "type": _doc_type(rel)})

    results.sort(key=lambda e: e["path"])
    return results
