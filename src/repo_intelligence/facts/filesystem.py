"""Filesystem structure facts (observation only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Directories never walked into (tool caches / VCS internals).
DEFAULT_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        ".eggs",
        "egg-info",
    }
)

# Shared searchable-file policy (M18 §14): ONE definition used by the
# ``search_repository`` tool walk AND by the retrieval chunker, so a
# file indexed is a file searchable and vice versa. Text files of
# these extensions are eligible for reading during a directory walk
# (an explicitly named file path is always read by the tool layer).
SEARCHABLE_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".md",
        ".rst",
        ".txt",
        ".sh",
    }
)

# Read policy for walked files: size cap + binary sniff.
MAX_FILE_BYTES = 1_000_000
BINARY_SNIFF_BYTES = 8192


def is_probably_binary(raw: bytes) -> bool:
    """True when *raw* looks binary (NUL byte in the sniff window)."""
    return b"\x00" in raw[:BINARY_SNIFF_BYTES]

# Extension → simple type label (no semantic meaning).
EXTENSION_TYPES = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "cfg",
    ".cfgx": "cfg",
    ".env": "environment",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".dockerfile": "dockerfile",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
    ".sql": "sql",
    ".xml": "xml",
    ".csv": "csv",
    ".lock": "lock",
    ".cfg.example": "example",
}

SPECIAL_FILE_TYPES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "license": "license",
    "licence": "license",
    "notice": "notice",
}


def classify_path(rel_path: str) -> str:
    """Return a simple deterministic type label for *rel_path*."""
    name = Path(rel_path).name
    lower = name.lower()

    if lower in SPECIAL_FILE_TYPES:
        return SPECIAL_FILE_TYPES[lower]
    if lower.startswith(".env"):
        return "environment"

    suffix = Path(name).suffix.lower()
    if suffix in EXTENSION_TYPES:
        return EXTENSION_TYPES[suffix]

    # multi-dot names like setup.cfg already handled by suffix;
    # fall back to final suffix or "file"
    return suffix.lstrip(".") or "file"


def iter_repository_paths(
    root: Path,
    ignored_dirs: Iterable[str] = DEFAULT_IGNORED_DIRS,
) -> Tuple[List[str], List[str]]:
    """Walk *root* and return (relative file paths, relative directory paths).

    Both lists are sorted. ``.git`` and other ignored directories are skipped.
    """
    ignored = set(ignored_dirs)
    files: List[str] = []
    directories: List[str] = []

    root = Path(root)
    for path in sorted(root.rglob("*")):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        parts = Path(rel).parts
        if any(part in ignored for part in parts):
            continue
        if path.is_dir():
            directories.append(rel)
        elif path.is_file():
            files.append(rel)

    files.sort()
    directories.sort()
    return files, directories


def analyze_filesystem(root: Path) -> Dict[str, Any]:
    """Produce structure facts for the repository rooted at *root*."""
    files, directories = iter_repository_paths(root)
    file_entries = []
    for rel in files:
        entry: Dict[str, Any] = {
            "path": rel,
            "type": classify_path(rel),
        }
        try:
            entry["size"] = (Path(root) / rel).stat().st_size
        except OSError:
            entry["size"] = None
        file_entries.append(entry)

    dir_entries = [{"path": rel} for rel in directories]
    return {"files": file_entries, "directories": dir_entries}
