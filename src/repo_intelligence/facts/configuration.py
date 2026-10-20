"""Configuration artifact facts (paths/types only; no secret values)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .filesystem import classify_path, iter_repository_paths

CONFIG_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "package.json",
    "tsconfig.json",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "makefile",
    ".editorconfig",
    ".gitignore",
    ".pre-commit-config.yaml",
}

CONFIG_SUFFIXES = (
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
    ".env",
)

# Names that often hold credentials — never copy values.
SENSITIVE_NAME_MARKERS = (
    ".env",
    "credential",
    "secret",
    "password",
    "passwd",
    "token",
    "keystore",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".key",
)

# Non-sensitive config paths we still classify as configuration.
CONFIG_DIR_HINTS = (
    ".github/",
    ".claude/",
    ".circleci/",
    ".vscode/",
)


def _is_sensitive_name(rel_path: str) -> bool:
    lower = rel_path.lower()
    name = Path(lower).name
    if name == ".env" or name.startswith(".env."):
        return True
    return any(marker in lower for marker in SENSITIVE_NAME_MARKERS)


def _looks_like_config(rel_path: str) -> bool:
    name = Path(rel_path).name.lower()
    if name in CONFIG_NAMES:
        return True
    if name.endswith(CONFIG_SUFFIXES) or name == ".env":
        return True
    # files named exactly like env variants
    if name.startswith(".env"):
        return True
    # path-based config locations (json/yaml inside them)
    posix = rel_path.replace("\\", "/")
    for hint in CONFIG_DIR_HINTS:
        if "/{0}".format(hint) in "/{0}".format(posix) or posix.startswith(hint):
            if name.endswith((".yml", ".yaml", ".json", ".toml", ".ini")):
                return True
    return False


def _config_type(rel_path: str) -> str:
    if Path(rel_path).name.lower().startswith(".env"):
        return "environment"
    return classify_path(rel_path)


def _structural_keys(rel_path: str, root: Path) -> Optional[List[str]]:
    """Return sorted top-level keys for JSON configs when safe."""
    if not rel_path.lower().endswith(".json"):
        return None
    if _is_sensitive_name(rel_path):
        return None
    try:
        data = json.loads((root / rel_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return sorted(str(k) for k in data.keys())
    return None


def analyze_configuration(root: Path) -> List[Dict[str, Any]]:
    """Identify configuration artifacts without exposing secret values."""
    root = Path(root)
    files, _ = iter_repository_paths(root)
    results: List[Dict[str, Any]] = []

    for rel in files:
        if not _looks_like_config(rel):
            continue
        entry: Dict[str, Any] = {
            "path": rel,
            "type": _config_type(rel),
        }
        sensitive = _is_sensitive_name(rel)
        if sensitive:
            entry["contains_sensitive_values"] = True
        else:
            keys = _structural_keys(rel, root)
            if keys is not None:
                entry["top_level_keys"] = keys
            # malformed JSON still recorded as config; flag parse status for .json
            if rel.lower().endswith(".json"):
                try:
                    json.loads((root / rel).read_text(encoding="utf-8"))
                    entry["parse_status"] = "ok"
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    entry["parse_status"] = "error"
        results.append(entry)

    results.sort(key=lambda e: e["path"])
    return results
