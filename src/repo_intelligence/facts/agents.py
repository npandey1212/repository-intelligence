"""Agent artifact facts (existence + path only; no role inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .filesystem import classify_path, iter_repository_paths

# Explicit, well-known agent artifact locations / name patterns.
AGENT_DIR_HINTS = (
    "/.claude/agents/",
    "/.claude/commands/",
    "/.cursor/rules/",
    "/.github/agents/",
    "/agents/",
    "/.codex/",
)

AGENT_FILE_NAMES = {
    "agent.md",
    "agent.yaml",
    "agent.yml",
    "agent.json",
    "agents.md",
}

AGENT_NAME_MARKERS = (
    "agent",
    "subagent",
    "sub-agent",
    "assistant_definition",
)

# Files that are clearly agent skill/prompt packs when under known dirs.
SKILL_DIR_HINTS = (
    "/.claude/skills/",
    "/skills/",
)


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _is_agent_artifact(rel: str) -> bool:
    posix = "/" + _posix(rel)
    name = Path(rel).name.lower()

    for hint in AGENT_DIR_HINTS:
        if posix.startswith(hint) or hint in posix:
            # any file under an agents directory counts
            return True

    if name in AGENT_FILE_NAMES:
        return True

    # explicit agent definition file names e.g. iac-agent.md, foo_agent.yaml
    stem = Path(name).stem
    if any(marker in stem for marker in AGENT_NAME_MARKERS):
        if name.endswith((".md", ".yaml", ".yml", ".json", ".toml")):
            # avoid matching random docs like "agents-overview" only when
            # under docs? Spec wants known patterns — keep filename marker.
            return True
    return False


def _agent_type(rel: str) -> str:
    posix = "/" + _posix(rel)
    name = Path(rel).name.lower()
    if "/agents/" in posix or name in AGENT_FILE_NAMES or "agent" in name:
        if name.endswith((".md", ".markdown")):
            return "agent_definition"
        if name.endswith((".yaml", ".yml")):
            return "agent_config"
        if name.endswith(".json"):
            return "agent_config"
    for hint in SKILL_DIR_HINTS:
        if posix.startswith(hint) or hint in posix:
            return "agent_skill"
    if "/commands/" in posix:
        return "agent_command"
    return classify_path(rel)


def analyze_agents(root: Path) -> List[Dict[str, Any]]:
    """Find explicitly identifiable agent-related artifacts."""
    root = Path(root)
    files, _ = iter_repository_paths(root)
    results: List[Dict[str, Any]] = []

    for rel in files:
        if _is_agent_artifact(rel):
            results.append(
                {
                    "path": rel,
                    "type": _agent_type(rel),
                }
            )

    results.sort(key=lambda e: e["path"])
    return results
