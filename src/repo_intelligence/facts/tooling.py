"""Tool / MCP server / MCP configuration declarations (M12 knowledge slice).

Extracts only explicit, structured statements - never free-text mentions:

* **agent tool declarations** - a ``tools:`` entry in an agent artifact's
  frontmatter, either bare (``Read``) or namespaced
  (``mcp__<server>__<tool>``);
* **tool registry entries** - a repository YAML document whose structure is
  ``servers: [ {name: ..., tools: [ ... ]}, ... ]`` (shape-discovered, no
  repository-specific path);
* **MCP configuration entries** - keys under ``mcpServers`` in a
  configuration file identified by the existing MCP config-name rules.

``analyze_tooling`` is pure observation: statements carry repository paths
and 1-based line evidence. Endpoint resolution into FactRecords happens in
``build_tooling_facts`` after the base artifact facts exist, following the
same pattern as :mod:`repo_intelligence.facts.interactions`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evidence import make_evidence
from .filesystem import iter_repository_paths
from .mcp import MCP_CONFIG_NAMES
from .records import make_fact

# Agent artifacts that may declare tools (same classifier as M2 agents).
from .agents import _is_agent_artifact  # noqa: PLC2701  (single source of truth)

_SOURCE = "tooling"

_AGENT_TEXT_SUFFIXES = (".md", ".markdown", ".yaml", ".yml", ".json", ".toml")
_MAX_READ_BYTES = 1_000_000

# Evidence artifact_type per statement type (controlled vocabulary).
_ARTIFACT_TYPE = {
    "agent_tool": "agent_definition",
    "server_tool": "configuration",
    "registry_server": "configuration",
    "config_server": "mcp_definition",
}

# Agent artifact kinds that can be the source of a DECLARES claim.
_AGENT_FACT_KINDS = frozenset(
    {"agent_definition", "agent_command", "agent_config", "agent_skill"}
)

_TOOLS_LINE = re.compile(r"^(\s*)tools:\s*(.*?)\s*$")
_BLOCK_ITEM = re.compile(r"^(\s*)-\s+(.+?)\s*$")
_NAMESPACED = re.compile(r"^mcp__(.+)$")

_SERVERS_ANCHOR = re.compile(r"^servers:\s*(?:#.*)?$")
_SERVER_ITEM = re.compile(r"^(\s*)-\s+name:\s*(.+?)\s*(?:#.*)?$")
_REGISTRY_TOOLS_KEY = re.compile(r"^(\s*)tools:\s*(?:#.*)?$")
_LIST_ITEM = re.compile(r"^(\s*)-\s+(.+?)\s*(?:#.*)?$")
_TOP_LEVEL_KEY = re.compile(r"^[A-Za-z_]")

_JSON_KEY = re.compile(r'^(\s*)"((?:[^"\\]|\\.)*)"\s*:')
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _posix(rel: str) -> str:
    return rel.replace("\\", "/")


def _read_text(path: Path) -> Optional[str]:
    try:
        if not path.is_file():
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > _MAX_READ_BYTES:
        return None
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1].strip()
    return value


def _split_tool_entries(value: str) -> List[str]:
    """Split one inline ``tools:`` value into normalized entries."""
    text = (value or "").strip()
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return []
    entries: List[str] = []
    for part in text.split(","):
        entry = _unquote(part)
        if entry:
            entries.append(entry)
    return entries


def _parse_tool_entry(entry: str) -> Tuple[Optional[str], str]:
    """Split a declaration entry into ``(server, name)``.

    Namespaced entries use the ``mcp__<server>__<tool>`` calling identity;
    everything else is a bare (namespace-free) declaration. Entries that
    start with the namespace marker but do not carry two segments stay bare.
    """
    match = _NAMESPACED.match(entry)
    if match:
        pieces = match.group(1).split("__")
        if len(pieces) >= 2 and pieces[0] and all(pieces[1:]):
            return pieces[0], "__".join(pieces[1:])
    return None, entry


def _frontmatter_window(
    lines: List[str], suffix: str
) -> Optional[Tuple[int, int]]:
    """Parseable region ``(start, end_exclusive)`` of an agent artifact.

    Markdown agents declare tools in YAML frontmatter only (prose and code
    samples are never declarations); YAML/JSON agent configs are structural
    documents, so their whole body is parseable.
    """
    if suffix in (".md", ".markdown"):
        if not lines or lines[0].strip() != "---":
            return None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return 1, index
        return None
    return 0, len(lines)


def _agent_tool_statements(root: Path) -> List[Dict[str, Any]]:
    """Explicit ``tools:`` declarations in agent artifacts (inline or block)."""
    statements: List[Dict[str, Any]] = []
    files, _ = iter_repository_paths(root)
    for rel in files:
        posix = _posix(rel)
        suffix = Path(posix).suffix.lower()
        if suffix not in _AGENT_TEXT_SUFFIXES:
            continue
        if not _is_agent_artifact(rel):
            continue
        text = _read_text(root / rel)
        if not text:
            continue
        lines = text.splitlines()
        window = _frontmatter_window(lines, suffix)
        if window is None:
            continue
        start, end = window
        index = start
        while index < end:
            line = lines[index]
            tools_line = _TOOLS_LINE.match(line)
            if tools_line is None:
                index += 1
                continue
            key_indent = tools_line.group(1)
            value = tools_line.group(2)
            line_no = index + 1
            entries: List[str] = []
            if value and not value.startswith("#"):
                entries = _split_tool_entries(value)
            else:
                cursor = index + 1
                while cursor < end:
                    candidate = lines[cursor]
                    if not candidate.strip() or candidate.lstrip().startswith("#"):
                        cursor += 1
                        continue
                    item = _BLOCK_ITEM.match(candidate)
                    if item and len(item.group(1)) > len(key_indent):
                        raw_entry = _unquote(item.group(2))
                        if "#" in raw_entry:
                            raw_entry = raw_entry.split("#", 1)[0].strip()
                        if raw_entry:
                            entries.append(raw_entry)
                        cursor += 1
                        continue
                    break
            for entry in entries:
                server, name = _parse_tool_entry(entry)
                if not name:
                    continue
                statements.append(
                    {
                        "type": "agent_tool",
                        "path": posix,
                        "line": line_no,
                        "entry": entry,
                        "server": server,
                        "name": name,
                    }
                )
            index += 1
    statements.sort(key=_statement_key)
    return statements


def _registry_statements(root: Path) -> List[Dict[str, Any]]:
    """Shape-discovered tool registry documents (``servers:`` list)."""
    statements: List[Dict[str, Any]] = []
    files, _ = iter_repository_paths(root)
    for rel in files:
        posix = _posix(rel)
        if not posix.endswith((".yaml", ".yml")):
            continue
        text = _read_text(root / rel)
        if not text:
            continue
        if not any(_SERVERS_ANCHOR.match(line) for line in text.splitlines()):
            continue
        file_statements: List[Dict[str, Any]] = []
        in_servers = False
        current_server: Optional[str] = None
        tools_indent: Optional[int] = None
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if _SERVERS_ANCHOR.match(line):
                in_servers = True
                current_server = None
                tools_indent = None
                continue
            if not in_servers:
                continue
            if _TOP_LEVEL_KEY.match(line):
                in_servers = False
                current_server = None
                tools_indent = None
                continue
            item = _SERVER_ITEM.match(line)
            if item:
                current_server = _unquote(item.group(2))
                tools_indent = None
                if current_server:
                    file_statements.append(
                        {
                            "type": "registry_server",
                            "path": posix,
                            "line": line_no,
                            "server": current_server,
                        }
                    )
                continue
            tools_key = _REGISTRY_TOOLS_KEY.match(line)
            if tools_key:
                if current_server is not None:
                    tools_indent = len(tools_key.group(1))
                continue
            listed = _LIST_ITEM.match(line)
            if tools_indent is not None:
                if listed and len(listed.group(1)) > tools_indent:
                    tool_name = _unquote(listed.group(2))
                    if tool_name and current_server:
                        file_statements.append(
                            {
                                "type": "server_tool",
                                "path": posix,
                                "line": line_no,
                                "server": current_server,
                                "name": tool_name,
                            }
                        )
                    continue
                # any deeper non-list-item line closes the tools block
                tools_indent = None
        if any(s["type"] == "registry_server" for s in file_statements):
            statements.extend(file_statements)
    statements.sort(key=_statement_key)
    return statements


def _strip_json_strings(line: str) -> str:
    return _JSON_STRING.sub('""', line)


def _config_statements(root: Path) -> List[Dict[str, Any]]:
    """``mcpServers`` entries in configuration files (with line numbers)."""
    statements: List[Dict[str, Any]] = []
    files, _ = iter_repository_paths(root)
    for rel in files:
        name = Path(rel).name
        if name not in MCP_CONFIG_NAMES:
            continue
        text = _read_text(root / rel)
        if not text:
            continue
        try:
            document = json.loads(text)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        mcp_servers = document.get("mcpServers")
        if not isinstance(mcp_servers, dict):
            continue
        posix = _posix(rel)
        depth = 0
        in_mcp_servers = False
        found: Dict[str, int] = {}
        for line_no, raw in enumerate(text.splitlines(), start=1):
            key_match = _JSON_KEY.match(raw)
            if key_match and depth == 1 and key_match.group(2) == "mcpServers":
                in_mcp_servers = True
            elif key_match and in_mcp_servers and depth == 2:
                name = key_match.group(2)
                if name in mcp_servers and name not in found:
                    found[name] = line_no
            stripped = _strip_json_strings(raw)
            depth += stripped.count("{") - stripped.count("}")
            if in_mcp_servers and depth <= 1:
                in_mcp_servers = False
        # Compact (single-line) JSON defeats depth tracking: fall back to
        # the first line carrying each known server key.
        if len(found) < len(mcp_servers):
            lines = text.splitlines()
            for name in mcp_servers:
                if name in found:
                    continue
                needle = '"{}"'.format(name)
                for line_no, raw in enumerate(lines, start=1):
                    if needle in raw and ":" in raw.split(needle, 1)[1]:
                        found[name] = line_no
                        break
        for name in sorted(found):
            statements.append(
                {
                    "type": "config_server",
                    "path": posix,
                    "line": found[name],
                    "server": name,
                }
            )
    statements.sort(key=_statement_key)
    return statements


def _statement_key(statement: Dict[str, Any]) -> Tuple:
    return (
        str(statement.get("type") or ""),
        str(statement.get("path") or ""),
        int(statement.get("line") or 0),
        str(statement.get("server") or ""),
        str(statement.get("name") or ""),
    )


def analyze_tooling(root: Path) -> Dict[str, Any]:
    """Scan the repository for explicit tool/server/configuration statements.

    Returns ``{"statements": [...]}`` sorted deterministically. Pure
    observation - no endpoint resolution, no relationship guesses: a file
    only yields statements when its structure matches the declared shape.
    """
    root = Path(root)
    statements: List[Dict[str, Any]] = []
    statements.extend(_agent_tool_statements(root))
    statements.extend(_registry_statements(root))
    statements.extend(_config_statements(root))
    statements.sort(key=_statement_key)
    return {"statements": statements}


# ---------------------------------------------------------------------------
# FactRecords (post-pass; endpoints resolved against existing facts)
# ---------------------------------------------------------------------------


def _sorted_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[Tuple, Dict[str, Any]] = {}
    for item in items:
        key = (
            item.get("path"),
            item.get("artifact_type"),
            item.get("commit"),
            item.get("line_start"),
            item.get("line_end"),
            item.get("symbol"),
        )
        unique.setdefault(key, item)
    return [
        unique[key]
        for key in sorted(
            unique,
            key=lambda k: (
                str(k[0] or ""),
                int(k[3] or 0),
                str(k[1] or ""),
                str(k[2] or ""),
            ),
        )
    ]


def _statement_evidence(statement: Dict[str, Any], commit: str) -> Dict[str, Any]:
    line = int(statement.get("line") or 0) or None
    return make_evidence(
        path=str(statement.get("path") or ""),
        artifact_type=_ARTIFACT_TYPE[str(statement.get("type"))],
        commit=commit,
        line_start=line,
        line_end=line,
        symbol=None,
    )


def build_tooling_facts(
    facts: List[Dict[str, Any]],
    tooling_analysis: Dict[str, Any],
    commit: str,
) -> List[Dict[str, Any]]:
    """Resolve tooling statements into M12 FactRecords.

    Emits node facts (``tool``/``mcp_tool``, ``tool``/``built_in_tool``,
    ``mcp_server``/``registered`` - one per identity with merged evidence
    from every declaring source) and relationship facts
    (``interaction``/``declares|exposes|configures`` - resolved endpoints
    as ``target_fact_id``; unresolvable or ambiguous targets as
    ``target_label`` + ``reason`` so M3 can publish unresolved claims
    instead of inventing edges).
    """
    statements = list((tooling_analysis or {}).get("statements") or [])
    if not statements:
        return []

    agent_fact_by_path: Dict[str, str] = {}
    config_fact_by_path: Dict[str, str] = {}
    for fact in facts:
        if fact.get("fact_type") != "artifact":
            continue
        fact_id = fact.get("fact_id")
        attrs = fact.get("attributes") or {}
        path = attrs.get("path")
        if not path or not fact_id:
            continue
        kind = fact.get("kind")
        if kind in _AGENT_FACT_KINDS:
            agent_fact_by_path.setdefault(str(path), str(fact_id))
        elif kind == "mcp_config":
            config_fact_by_path.setdefault(str(path), str(fact_id))

    statements = sorted(statements, key=_statement_key)

    mcp_tool_evidence: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    builtin_evidence: Dict[str, List[Dict[str, Any]]] = {}
    server_evidence: Dict[str, List[Dict[str, Any]]] = {}
    registry_server_evidence: Dict[str, List[Dict[str, Any]]] = {}
    registry_tool_names = set()
    registry_servers = set()
    config_servers = set()
    config_file_paths = set()

    for statement in statements:
        evidence = _statement_evidence(statement, commit)
        statement_type = statement["type"]
        if statement_type == "server_tool":
            registry_tool_names.add(str(statement["name"]))
            mcp_tool_evidence.setdefault(
                (str(statement["server"]), str(statement["name"])), []
            ).append(evidence)
        elif statement_type == "agent_tool":
            if statement.get("server"):
                mcp_tool_evidence.setdefault(
                    (str(statement["server"]), str(statement["name"])), []
                ).append(evidence)
            else:
                builtin_evidence.setdefault(str(statement["name"]), []).append(
                    evidence
                )
        elif statement_type == "registry_server":
            server = str(statement["server"])
            registry_servers.add(server)
            server_evidence.setdefault(server, []).append(evidence)
            registry_server_evidence.setdefault(server, []).append(evidence)
        elif statement_type == "config_server":
            server = str(statement["server"])
            config_servers.add(server)
            config_file_paths.add(str(statement["path"]))
            server_evidence.setdefault(server, []).append(evidence)

    out: List[Dict[str, Any]] = []
    node_fact_id: Dict[Tuple, str] = {}

    for (server, name) in sorted(mcp_tool_evidence):
        fact = make_fact(
            fact_type="tool",
            kind="mcp_tool",
            source=_SOURCE,
            attributes={"name": name, "server": server},
            evidence=_sorted_evidence(mcp_tool_evidence[(server, name)]),
        )
        out.append(fact)
        node_fact_id[("mcp_tool", server, name)] = fact["fact_id"]

    # A bare declaration whose name is also an MCP tool name has two
    # plausible scopes; no canonical builtin node is created for it.
    for name in sorted(builtin_evidence):
        if name in registry_tool_names:
            continue
        fact = make_fact(
            fact_type="tool",
            kind="built_in_tool",
            source=_SOURCE,
            attributes={"name": name},
            evidence=_sorted_evidence(builtin_evidence[name]),
        )
        out.append(fact)
        node_fact_id[("built_in_tool", "", name)] = fact["fact_id"]

    for server in sorted(server_evidence):
        fact = make_fact(
            fact_type="mcp_server",
            kind="registered",
            source=_SOURCE,
            attributes={"name": server},
            evidence=_sorted_evidence(server_evidence[server]),
        )
        out.append(fact)
        node_fact_id[("mcp_server", server)] = fact["fact_id"]

    # ------------------------------------------------------------------
    # Relationship statements
    # ------------------------------------------------------------------
    Spec = Dict[str, Any]
    specs: Dict[Tuple, Spec] = {}

    def add_spec(
        kind: str,
        source_fact_id: str,
        target_fact_id: str,
        target_label: str,
        reason: str,
        evidence: List[Dict[str, Any]],
    ) -> None:
        key = (kind, source_fact_id, target_fact_id, target_label)
        spec = specs.get(key)
        if spec is None:
            specs[key] = {
                "kind": kind,
                "source_fact_id": source_fact_id,
                "target_fact_id": target_fact_id,
                "target_label": target_label,
                "reason": reason,
                "evidence": list(evidence),
            }
            return
        for item in evidence:
            if item not in spec["evidence"]:
                spec["evidence"].append(dict(item))

    registered = registry_servers | config_servers

    for statement in statements:
        evidence = [_statement_evidence(statement, commit)]
        statement_type = statement["type"]
        if statement_type == "agent_tool":
            source = agent_fact_by_path.get(str(statement["path"]))
            if not source:
                continue
            if statement.get("server"):
                server = str(statement["server"])
                tool = node_fact_id.get(
                    ("mcp_tool", server, str(statement["name"]))
                )
                if tool:
                    add_spec("declares", source, tool, "", "", evidence)
                if server in registered:
                    server_fact = node_fact_id.get(("mcp_server", server))
                    if tool and server_fact:
                        add_spec("exposes", tool, server_fact, "", "", evidence)
                elif tool:
                    add_spec(
                        "exposes",
                        tool,
                        "",
                        server,
                        "target_not_resolved",
                        evidence,
                    )
            elif str(statement["name"]) in registry_tool_names:
                add_spec(
                    "declares",
                    source,
                    "",
                    str(statement["name"]),
                    "ambiguous_identity",
                    evidence,
                )
            else:
                tool = node_fact_id.get(
                    ("built_in_tool", "", str(statement["name"]))
                )
                if tool:
                    add_spec("declares", source, tool, "", "", evidence)
        elif statement_type == "server_tool":
            tool = node_fact_id.get(
                ("mcp_tool", str(statement["server"]), str(statement["name"]))
            )
            server_fact = node_fact_id.get(("mcp_server", str(statement["server"])))
            if tool and server_fact:
                add_spec("exposes", tool, server_fact, "", "", evidence)
        elif statement_type == "config_server":
            server_fact = node_fact_id.get(("mcp_server", str(statement["server"])))
            config = config_fact_by_path.get(str(statement["path"]))
            if server_fact and config:
                add_spec("configures", server_fact, config, "", "", evidence)
            elif server_fact:
                add_spec(
                    "configures",
                    server_fact,
                    "",
                    str(statement["path"]),
                    "target_not_resolved",
                    evidence,
                )

    # Registered servers with no configuration entry stay unresolved.
    label_paths = sorted(config_file_paths)
    for server in sorted(registry_servers - config_servers):
        server_fact = node_fact_id.get(("mcp_server", server))
        if not server_fact:
            continue
        add_spec(
            "configures",
            server_fact,
            "",
            label_paths[0] if label_paths else "configuration",
            "target_not_resolved",
            _sorted_evidence(registry_server_evidence.get(server, [])),
        )

    # Conflicting configuration: one server mapped by multiple configuration
    # files - every candidate edge is kept and marked, none is selected.
    config_targets: Dict[str, set] = {}
    for spec in specs.values():
        if spec["kind"] == "configures" and spec["target_fact_id"]:
            config_targets.setdefault(spec["source_fact_id"], set()).add(
                spec["target_fact_id"]
            )
    for spec in specs.values():
        if spec["kind"] != "configures" or not spec["target_fact_id"]:
            continue
        if len(config_targets.get(spec["source_fact_id"], ())) > 1:
            spec["conflicting"] = True

    for key in sorted(specs):
        spec = specs[key]
        attributes: Dict[str, Any] = {
            "source_fact_id": spec["source_fact_id"],
            "relation": spec["kind"],
        }
        if spec["target_fact_id"]:
            attributes["target_fact_id"] = spec["target_fact_id"]
        if spec["target_label"]:
            attributes["target_label"] = spec["target_label"]
        if spec["reason"]:
            attributes["reason"] = spec["reason"]
        if spec.get("conflicting"):
            attributes["conflicting"] = True
        out.append(
            make_fact(
                fact_type="interaction",
                kind=spec["kind"],
                source=_SOURCE,
                attributes=attributes,
                evidence=_sorted_evidence(spec["evidence"]),
            )
        )

    return out


__all__ = [
    "analyze_tooling",
    "build_tooling_facts",
]
