"""MCP artifact facts (existence only; no relationship inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .filesystem import classify_path, iter_repository_paths

MCP_DIR_HINTS = (
    "/mcp/",
    "/mcp-servers/",
    "/mcp_servers/",
    "/.mcp/",
    "/.mcp.json",
)

MCP_FILE_MARKERS = (
    "mcp",
    "model_context_protocol",
    "model-context-protocol",
)

MCP_CONFIG_NAMES = {
    ".mcp.json",
    "mcp.json",
    "mcp.yaml",
    "mcp.yml",
    "mcp-config.json",
    "mcp_settings.json",
}

SOURCE_SUFFIXES = (".py", ".ts", ".js", ".go", ".json", ".yaml", ".yml", ".toml", ".md")


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _name_and_stem(rel: str):
    name = Path(rel).name.lower()
    stem = Path(name).stem
    return name, stem


def _is_mcp_artifact(rel: str) -> bool:
    posix = "/" + _posix(rel)
    name, stem = _name_and_stem(rel)

    if name in MCP_CONFIG_NAMES or name == ".mcp.json":
        return True

    for hint in MCP_DIR_HINTS:
        if hint.rstrip("/") in posix:
            return True

    if "mcp" in stem and name.endswith(SOURCE_SUFFIXES):
        # e.g. server.py under mcp/, mcp_server.py, foo-mcp.yaml
        if "mcp" in stem.replace("-", "_").split("_") or "mcp" in stem:
            return True

    # directory path contains /mcp/
    if "/mcp/" in posix or posix.rstrip("/").endswith("/mcp"):
        return True

    return False


def _mcp_type(rel: str) -> str:
    name = Path(rel).name.lower()
    posix = "/" + _posix(rel)
    if name in MCP_CONFIG_NAMES:
        return "mcp_config"
    if name.endswith((".py", ".ts", ".js", ".go")) and (
        "mcp" in name or "/mcp/" in posix
    ):
        return "mcp_server_source"
    if "/mcp/" in posix or "mcp" in name:
        if name.endswith((".md", ".markdown")):
            return "mcp_documentation"
        if name.endswith((".yaml", ".yml", ".json", ".toml")):
            return "mcp_config"
        if name.endswith((".py", ".ts", ".js", ".go")):
            return "mcp_server_source"
    return classify_path(rel)


def analyze_mcp(root: Path) -> List[Dict[str, Any]]:
    """Find clearly identifiable MCP-related artifacts."""
    root = Path(root)
    files, _ = iter_repository_paths(root)
    results: List[Dict[str, Any]] = []

    for rel in files:
        if _is_mcp_artifact(rel):
            results.append(
                {
                    "path": rel,
                    "type": _mcp_type(rel),
                }
            )

    results.sort(key=lambda e: e["path"])
    return results
