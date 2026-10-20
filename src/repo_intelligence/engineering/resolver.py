"""Deterministic endpoint resolution for EngineeringModel relationships.

Analyzers/facts only observe. The construction layer resolves targets
against already-built canonical nodes — never invents targets from free text.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set


def parent_directory(path: str) -> Optional[str]:
    """Repository-relative parent directory, or None at repo root."""
    if not path or "/" not in path:
        return None
    parent = path.rsplit("/", 1)[0]
    return parent or None


def module_to_path_candidates(module: str) -> list:
    """Candidate repository paths for a dotted import module name."""
    if not module:
        return []
    candidates = []
    dotted = module.replace(".", "/")
    for suffix in (".py", "/__init__.py"):
        candidates.append(dotted + suffix if not dotted.endswith(".py") else dotted)
    if not module.endswith(".py"):
        candidates.append(dotted + ".py")
        candidates.append(dotted + "/__init__.py")
    # unique, preserve order
    seen = set()
    ordered = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class NodeIndex:
    """Lookup helpers over canonical engineering node ids."""

    def __init__(self, nodes: Iterable[Dict[str, Any]]):
        self.file_paths: Set[str] = set()
        self.directory_paths: Set[str] = set()
        self.module_paths: Set[str] = set()
        self.code_entity_ids_by_path: Dict[str, list] = {}
        self.all_ids: Set[str] = set()
        for node in nodes:
            node_id = node.get("id") or ""
            self.all_ids.add(node_id)
            kind = node.get("kind")
            subkind = node.get("subkind")
            if kind == "artifact" and subkind == "file":
                # artifact:file:<path>
                path = node_id.split(":", 2)[2] if node_id.count(":") >= 2 else ""
                if path:
                    self.file_paths.add(path)
            elif kind == "artifact" and subkind == "directory":
                path = node_id.split(":", 2)[2] if node_id.count(":") >= 2 else ""
                if path:
                    self.directory_paths.add(path)
            elif kind == "code_entity" and subkind == "module":
                path = node_id.split(":", 1)[1] if ":" in node_id else ""
                if path:
                    self.module_paths.add(path)
                    self.code_entity_ids_by_path.setdefault(path, []).append(node_id)
            elif kind == "code_entity":
                rest = node_id.split(":", 1)[1] if ":" in node_id else ""
                path = rest.split("::", 1)[0] if rest else ""
                if path:
                    self.code_entity_ids_by_path.setdefault(path, []).append(node_id)

    def resolve_imported_module(self, module: str) -> Optional[str]:
        """Deterministic IMPORTS target among known nodes.

        Strategies (matrix order): exact_module_path, dotted_module_to_path,
        suffix_path_match. Returns None when not confidently resolvable.
        """
        if not module:
            return None
        # 1) exact path-like module (rare) or module path already a file path
        if module in self.module_paths:
            return "code_entity:{}".format(module)
        if module in self.file_paths:
            return "artifact:file:{}".format(module)

        # 2) dotted → repository path
        for candidate in module_to_path_candidates(module):
            if candidate in self.module_paths:
                return "code_entity:{}".format(candidate)
            if candidate in self.file_paths:
                return "artifact:file:{}".format(candidate)

        # 3) suffix match against known module paths (unique only)
        dotted = module.replace(".", "/")
        matches = []
        for path in sorted(self.module_paths):
            if path == dotted or path.endswith("/" + dotted) or path.endswith(
                "/" + dotted + ".py"
            ):
                matches.append(path)
            elif path.endswith(dotted + ".py") or path == dotted + ".py":
                matches.append(path)
        # also try with .py
        if not matches:
            for path in sorted(self.module_paths):
                if path.endswith("/" + dotted + ".py") or path == dotted + ".py":
                    matches.append(path)
        unique = sorted(set(matches))
        if len(unique) == 1:
            return "code_entity:{}".format(unique[0])
        # prefer module over bare file when both somehow match
        file_matches = [p for p in unique if p in self.file_paths]
        if len(file_matches) == 1 and len(unique) > 1:
            return "artifact:file:{}".format(file_matches[0])
        if len(unique) == 1:
            path = unique[0]
            if path in self.module_paths:
                return "code_entity:{}".format(path)
            return "artifact:file:{}".format(path)
        return None

    def resolve_path_node(
        self, path: str, prefer: str = "file"
    ) -> Optional[str]:
        """Resolve a repository path to an existing file (or directory/module) node."""
        if not path:
            return None
        if prefer == "file" and path in self.file_paths:
            return "artifact:file:{}".format(path)
        if path in self.directory_paths:
            return "artifact:directory:{}".format(path)
        if path in self.module_paths:
            return "code_entity:{}".format(path)
        if path in self.file_paths:
            return "artifact:file:{}".format(path)
        return None


def resolve_dependency_target(name: str) -> Optional[str]:
    """Documented external package vocabulary: external_package:<name>."""
    cleaned = (name or "").strip()
    if not cleaned:
        return None
    return "external_package:{}".format(cleaned)


__all__ = [
    "NodeIndex",
    "module_to_path_candidates",
    "parent_directory",
    "resolve_dependency_target",
]
