"""Deterministic architecture identity resolution.

Architecture identity describes the architectural concept, not the
implementation file path.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Dict, Optional

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_INVALID_ID_CHARS = re.compile(r"[^a-z0-9._-]")


def slugify(value: str) -> str:
    """Lowercase alphanumeric slug with hyphens; empty when unusable."""
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return ""
    slug = _SLUG_RE.sub("-", cleaned).strip("-")
    return slug


def agent_name_from_engineering_node(node: Dict[str, Any]) -> Optional[str]:
    """Canonical architecture agent name for an engineering agent_definition node.

    Deterministic and analyzer-independent:

    1. ``attributes.name`` when present (already a stable label)
    2. else filename stem of ``attributes.path`` / node id path

    Returns None when identity cannot be established confidently.
    """
    if not node:
        return None
    if node.get("kind") != "artifact" or node.get("subkind") != "agent_definition":
        return None

    attrs = node.get("attributes") or {}
    name = attrs.get("name")
    if isinstance(name, str) and name.strip():
        slug = slugify(name)
        if slug:
            return slug

    path = attrs.get("path")
    if not isinstance(path, str) or not path:
        node_id = node.get("id") or ""
        if node_id.startswith("artifact:agent_definition:"):
            path = node_id.split(":", 2)[2]
        else:
            path = ""
    if not path:
        return None

    stem = PurePosixPath(str(path)).stem
    slug = slugify(stem)
    if not slug:
        return None
    return slug


def architecture_agent_node_id(agent_name: str) -> str:
    """Canonical architecture node id: concept:agent:<agent_name>."""
    return "concept:agent:{}".format(agent_name)


def sanitize_agent_name(agent_name: str) -> str:
    slug = slugify(agent_name)
    if not slug:
        return ""
    return _INVALID_ID_CHARS.sub("-", slug).strip("-")


__all__ = [
    "agent_name_from_engineering_node",
    "architecture_agent_node_id",
    "sanitize_agent_name",
    "slugify",
]
