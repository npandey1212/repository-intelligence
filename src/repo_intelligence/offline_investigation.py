"""Evidence-driven offline investigation — search, read, derive.

Offline mode answers from the repository itself.  One bounded,
deterministic script drives the EXISTING ``read_file`` and
``search_repository`` tools, parses what came back, and derives every
component, relationship, path, uncertainty and answer line from that
evidence.  Nothing here knows a subject-specific server name, a
snapshot id or a fixture path, and nothing here authors an answer in
advance: the same script answers a synthetic fixture, the ADOP
snapshot, or any other checkout.

The script (one :class:`InvestigatorModel` per investigator role):

    turn 1   read the entry target + the two discovery searches
    turn 2   read the three highest-scoring unread candidates
    turn 3   read two more candidates (the tool budget ends here)
    turn 4   emit the structured report built from the reads

Bounds are constants of this module: ``MAX_READS`` files,
``MAX_TOOL_CALLS`` tool calls (the M8.5 budget of the agent loop —
the model never requests a call the loop cannot execute), and
``SEARCH_MAX_RESULTS`` results per search.  A candidate that cannot
be read is an observation, not a run failure.

Extraction rules — format-driven, never subject-driven:

    R1  JSON ``mcpServers`` map        -> servers, their tools
    R2  YAML ``- name:`` + ``tools:``  -> servers, their tools
    R3  numbered Markdown ``… Agent``  -> an agent component
    R4  Markdown frontmatter ``name:`` -> an agent component
        (+ ``mcp__`` tokens on its ``tools:`` line -> DECLARES)
    R5  ``mcp__<server>__<tool>`` token-> tool + server + EXPOSED_BY,
        plus DECLARES when an ``*-agent`` token shares the line
    R6  a read code file               -> one component (type from
        its path) + IMPORTS to module paths this run also read

Candidates for the next read come from the reads themselves: paths
named in the content, plus — when a file names exactly ONE directory
and some config files beside it — that directory joined to those
config files.  Everything is scored by a static, documented path
heuristic and read in rank order, so two different repositories read
two different files and produce two different answers.

The vocabularies used here are never redefined: node types come from
the investigator's ``COMPONENT_TYPES``, relationship kinds from the
engineering ``RELATIONSHIP_KINDS``, and the legend meanings below are
presentation prose over those same terms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .agents.models.architecture_report import COMPONENT_TYPES
from .engineering.models import RELATIONSHIP_KINDS
from .llm import LLMResponse, Message, ToolCall

__all__ = [
    "MAX_READS",
    "MAX_COMPONENTS",
    "MAX_RELATIONSHIPS",
    "MAX_TOOL_CALLS",
    "SEARCH_MAX_RESULTS",
    "SEARCH_QUERIES",
    "AnalystModel",
    "Component",
    "Extraction",
    "InvestigatorModel",
    "Relationship",
    "build_architecture_payload",
    "build_chain",
    "build_diagram_payload",
    "build_flow_payload",
    "candidate_score",
    "discover_candidates",
    "extract_read_files",
    "observations_from_messages",
    "rank_candidates",
    "slug",
    "structured_response",
    "synthesize_answer",
    "tool_calls_response",
]

# -- bounds (the entire "budget framework" this capability owns) -----------

# AgentExecutionConfig.max_tool_calls — the agent loop's own ceiling.
# The plan is counted from the observed tool turns, so the budget is
# spent, never crossed: the last turn that fits is the last that runs.
MAX_TOOL_CALLS = 8
# Files one investigator may read (the entry target included).
MAX_READS = 6
# Search results one query may contribute to the candidate pool.
SEARCH_MAX_RESULTS = 50
# Discovery queries: one hits MCP configuration, one hits tool lists.
SEARCH_QUERIES = ("mcpServers", "tools")
# Structural caps — a large repository is truncated, said out loud.
MAX_COMPONENTS = 96
MAX_RELATIONSHIPS = 128
MAX_CANDIDATES = 48
MAX_HINTS = 6
MAX_BARE_NAMES = 6
MAX_UNCERTAINTIES = 6
MAX_CHAIN_HOPS = 8

# -- presentation (meanings are prose, terms are canonical) ----------------

_NODE_MEANINGS: Dict[str, str] = {
    "agent": "autonomous actor that drives the workflow",
    "tool": "operation exposed to an agent",
    "mcp_server": "server exposing tools over MCP",
    "workflow": "ordered sequence of work",
    "module": "code unit of the repository",
    "service": "running capability",
    "configuration": "configuration artifact",
    "data_store": "stored data",
    "unknown": "type not established by evidence",
}

_RELATIONSHIP_MEANINGS: Dict[str, str] = {
    "CONTAINS": "holds inside",
    "IMPORTS": "imports from",
    "DEPENDS_ON": "depends on",
    "TRIGGERS": "triggers",
    "DECLARES": "declares as its tooling",
    "EXPOSED_BY": "exposed through a server",
    "CONFIGURED_BY": "configured by",
}

# Transport vocabulary: when none of these appears in the inspected
# files, the wire protocol between a tool and its server stays an
# uncertainty instead of being named from thin air.
_TRANSPORT_TERMS = (
    "stdio",
    "json-rpc",
    "jsonrpc",
    "websocket",
    "grpc",
    "server-sent events",
)
_TRANSPORT_RE = re.compile(
    r"(?<![a-z])(" + "|".join(_TRANSPORT_TERMS) + r")(?![a-z])"
)

# -- identifier slug (one component id, however it is spelled) -------------

_NON_ID_RE = re.compile("[^a-z0-9-]+")
_SPLIT_ID_RE = re.compile(r"[\s_/]+")


def slug(text: Any) -> str:
    """A stable, space-free component id from any human spelling.

    ``Data Onboarding Agent`` → ``data-onboarding-agent`` and
    ``get_table`` → ``get-table``: the same component named two ways
    in two files collapses to one id instead of two nodes.
    """
    lowered = str(text).strip().lower()
    split = _SPLIT_ID_RE.sub("-", lowered)
    cleaned = _NON_ID_RE.sub("-", split)
    return re.sub(r"-{2,}", "-", cleaned).strip("-")


# -- observations (what the tools actually returned) -----------------------


@dataclass
class ReadObservation:
    """One successful ``read_file``: the content this run inspected."""

    path: str
    content: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class SearchObservation:
    """One successful ``search_repository`` — discovery, not inspection."""

    query: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


@dataclass
class ToolObservation:
    """One tool turn of the conversation, success or failure."""

    tool: str
    success: bool
    error: str = ""
    read: Optional[ReadObservation] = None
    search: Optional[SearchObservation] = None


def _observation(payload: Dict[str, Any]) -> ToolObservation:
    error = payload.get("error")
    code = ""
    if isinstance(error, dict):
        code = str(error.get("code") or "")
    elif isinstance(error, str):
        code = error
    observation = ToolObservation(
        tool=str(payload.get("tool") or ""),
        success=bool(payload.get("success")),
        error=code,
    )
    data = payload.get("data")
    if not observation.success or not isinstance(data, dict):
        return observation
    if observation.tool == "read_file":
        path = data.get("path")
        content = data.get("content")
        if isinstance(path, str) and isinstance(content, str):
            observation.read = ReadObservation(
                path=path,
                content=content,
                line_start=data.get("line_start"),
                line_end=data.get("line_end"),
            )
        return observation
    if observation.tool == "search_repository":
        raw = data.get("results")
        results: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            results = [item for item in raw if isinstance(item, dict)]
        observation.search = SearchObservation(
            query=str(data.get("query") or ""),
            results=results,
            truncated=bool(data.get("truncated")),
        )
    return observation


def observations_from_messages(
    messages: Sequence[Any],
) -> List[ToolObservation]:
    """Every tool turn of one conversation, in order.

    Only ``role == "tool"`` messages are parsed: the M15 context
    engine injects ``role == "user"`` messages that may quote any
    text at all, and those are never mistaken for tool output.
    """
    observations: List[ToolObservation] = []
    for message in messages:
        if getattr(message, "role", "") != "tool":
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if isinstance(payload, dict):
            observations.append(_observation(payload))
    return observations


# -- candidates (what to read next) ----------------------------------------

_STRUCTURED_SUFFIXES = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")
_CODE_SUFFIXES = (".py", ".js", ".ts", ".java", ".go", ".rs", ".sh")

# Entry-point file names: the module says nothing about identity, the
# directory that holds it does.
_GENERIC_STEMS = frozenset(
    ("server", "main", "index", "app", "cli", "run", "__init__")
)
_PROSE_SUFFIXES = (".md", ".rst", ".txt")
_ALL_SUFFIXES = (
    _STRUCTURED_SUFFIXES + _CODE_SUFFIXES + _PROSE_SUFFIXES
)

# Static, documented path heuristic.  "architecture" leads because a
# file that says so in its own path is the strongest static signal
# that it DESCRIBES the architecture rather than merely naming one
# tool; structured registries follow, then code, then prose.
_PATH_TERMS: Tuple[Tuple[str, int], ...] = (
    ("architecture", 5),
    ("registry", 3),
    ("mcp", 3),
    ("server", 2),
    ("tool", 2),
    ("agent", 2),
    ("config", 2),
    ("rules", 2),
)


def candidate_score(path: str) -> int:
    """Score one candidate path (higher = read sooner).

    Deliberately path-only: a file cannot be judged by content that
    has not been read yet.
    """
    lowered = path.lower()
    name = lowered.rsplit("/", 1)[-1]
    score = 0
    if lowered.endswith(_STRUCTURED_SUFFIXES):
        score += 3
    elif lowered.endswith(_CODE_SUFFIXES):
        score += 2
    elif lowered.endswith(_PROSE_SUFFIXES):
        score += 1
    for term, value in _PATH_TERMS:
        if term in lowered:
            score += value
    if name.startswith("."):
        score += 2
    if "/" not in lowered:
        score += 1
    return score


def rank_candidates(candidates: Dict[str, int]) -> List[str]:
    """Candidate paths in read order.

    Score first, then depth: a file that ties with a deeply nested
    one but sits near the repository root is the more central of the
    two, so ties break toward breadth.  Alphabet last, for stability.
    """
    return sorted(
        candidates,
        key=lambda path: (
            -(candidate_score(path) + candidates[path]),
            path.count("/"),
            path,
        ),
    )


_QUOTED_RE = re.compile(r"[\"']([^\"'\n]{1,160})[\"']")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,160})`")
_YAML_SCALAR_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.-]*[ \t]*:[ \t]*([^#\"'\n]{1,160}?)"
    r"[ \t]*$"
)
_PY_FROM_RE = re.compile(
    r"^[ \t]*from[ \t]+([A-Za-z_][A-Za-z0-9_.]*)[ \t]+import", re.M
)
_PY_REL_RE = re.compile(
    r"^[ \t]*from[ \t]+(\.+)([A-Za-z0-9_.]*)", re.M
)
_PY_IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+([A-Za-z_][A-Za-z0-9_.]*)", re.M
)
_JS_MODULE_RE = re.compile(
    r"""(?:from|import)[ \t]+['"](\.[^'"]+)['"]|"""
    r"""require\([ \t]*['"](\.[^'"]+)['"][ \t]*\)"""
)


def _module_key(path: str) -> Optional[str]:
    """The module a ``.py`` path names — ``pkg.py`` IS
    ``pkg/__init__.py``, so reading one settles the other."""
    if not path.endswith(".py"):
        return None
    if path.endswith("/__init__.py"):
        return path[: -len("/__init__.py")]
    return path[: -len(".py")]


def _has_known_suffix(token: str) -> bool:
    return token.lower().endswith(_ALL_SUFFIXES)


def _is_path_token(token: str) -> bool:
    """True for tokens that name a repository file, never prose."""
    if not token or len(token) > 160:
        return False
    if any(ch in token for ch in " \t{}()*$@&?|<>\\"):
        return False
    if "://" in token or token.startswith("/") or ".." in token:
        return False
    if token.startswith("-"):
        return False
    if "/" in token:
        return True
    return _has_known_suffix(token)


def _module_candidates(module: str, importer: str) -> List[str]:
    """``pkg.mod`` → the file paths that module could name."""
    del importer
    segments = [segment for segment in module.split(".") if segment]
    if len(segments) < 2:
        return []
    base = "/".join(segments)
    return [base + ".py", base + "/__init__.py"]


def _relative_candidates(reference: str, importer: str) -> List[str]:
    """``./x`` from one importer → the resolved repository path."""
    directory = importer.rsplit("/", 1)[0] if "/" in importer else ""
    joined = "{}/{}".format(directory, reference) if directory else reference
    if not _has_known_suffix(joined):
        joined = joined + ".py"
    parts: List[str] = []
    for segment in joined.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if parts:
                parts.pop()
            continue
        parts.append(segment)
    return ["/".join(parts)] if parts else []


def _directory_hints(text: str) -> List[str]:
    """Directory-looking tokens (no suffix, no separator) in content."""
    hints: List[str] = []
    tokens = list(_QUOTED_RE.findall(text)) + list(_BACKTICK_RE.findall(text))
    tokens.extend(_YAML_SCALAR_RE.findall(text))
    for token in tokens:
        token = token.strip().strip("`").rstrip("/")
        if "/" in token or _has_known_suffix(token):
            continue
        if len(token) < 3 or len(token) > 40:
            continue
        if not re.match(r"^[A-Za-z0-9._-]+$", token):
            continue
        if token.lower() in ("none", "null", "true", "false"):
            continue
        if token not in hints:
            hints.append(token)
    return sorted(hints)[:MAX_HINTS]


def discover_candidates(path: str, text: str) -> Dict[str, int]:
    """Candidate paths named inside one read file (value = bonus).

    Three sources, each one an observation of a path and nothing
    more: tokens that name a file (quoted, backticked or a YAML
    scalar), module imports resolved the way Python resolves them,
    and — only when a file names exactly ONE directory — that
    directory joined to the config files named beside it.  A file
    naming several directories is too ambiguous to join at all.
    """
    found: Dict[str, int] = {}
    bare: List[str] = []
    seen: List[str] = []

    tokens: List[str] = list(_QUOTED_RE.findall(text))
    tokens.extend(_BACKTICK_RE.findall(text))
    if path.lower().endswith((".yaml", ".yml")):
        tokens.extend(_YAML_SCALAR_RE.findall(text))
    for token in tokens:
        token = token.strip().strip("`").rstrip(",")
        if token in seen:
            continue
        seen.append(token)
        if not _is_path_token(token):
            continue
        if "/" in token or token.startswith("."):
            # a root-level or nested file is known outright; a
            # dotfile is never joined to a directory hint.
            found[token] = 0
        else:
            bare.append(token)

    if path.lower().endswith(".py"):
        for match in _PY_FROM_RE.finditer(text):
            for candidate in _module_candidates(match.group(1), path):
                found[candidate] = 0
        for match in _PY_IMPORT_RE.finditer(text):
            for candidate in _module_candidates(match.group(1), path):
                found[candidate] = 0
        for match in _PY_REL_RE.finditer(text):
            reference = match.group(1) + (match.group(2) or "")
            for candidate in _relative_candidates(reference, path):
                found[candidate] = 0
    else:
        for match in _JS_MODULE_RE.finditer(text):
            reference = match.group(1) or match.group(2) or ""
            if reference:
                for candidate in _relative_candidates(reference, path):
                    found[candidate] = 0

    hints = _directory_hints(text)
    if len(hints) == 1:
        config_bare = [
            name
            for name in bare
            if name.lower().endswith(_STRUCTURED_SUFFIXES)
        ]
        for name in sorted(set(config_bare))[:MAX_BARE_NAMES]:
            found["{}/{}".format(hints[0], name)] = 0
    for name in sorted(set(bare))[:MAX_BARE_NAMES]:
        found[name] = 0
    return found


# -- extraction (what the reads establish) ---------------------------------

_SERVER_ITEM_RE = re.compile(
    r"^[ \t]*-[ \t]+(?:name|id|server)[ \t]*:[ \t]*(\S+)[ \t]*$"
)
_TOOLS_KEY_RE = re.compile(r"^[ \t]+tools[ \t]*:[ \t]*$")
_LIST_ITEM_RE = re.compile(r"^[ \t]+-[ \t]+(\S+)[ \t]*$")
_NUMBERED_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\d+\.[ \t]+(.+)$", re.M)
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.S)
_FM_NAME_RE = re.compile(r"^name[ \t]*:[ \t]*(.+)$", re.M)
_FM_TOOLS_RE = re.compile(r"^tools[ \t]*:[ \t]*(.+)$", re.M)
_MCP_TOKEN_RE = re.compile(r"mcp__([A-Za-z0-9_-]+)__([A-Za-z0-9_-]+)")
_AGENT_TOKEN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]*-agent)\b")
_AGENT_WORD_RE = re.compile(r"agent", re.I)


@dataclass
class Component:
    """One component the inspected files establish."""

    id: str
    name: str
    type: str
    responsibility: str
    paths: List[str] = field(default_factory=list)
    classification: str = "observed"

    def refs(self) -> List[Dict[str, str]]:
        return [{"path": path} for path in self.paths]


@dataclass
class Relationship:
    """One relationship the inspected files establish."""

    source: str
    target: str
    relationship_type: str
    paths: List[str] = field(default_factory=list)
    classification: str = "derived"
    confidence: str = "medium"

    def refs(self) -> List[Dict[str, str]]:
        return [{"path": path} for path in self.paths]


class Extraction:
    """Accumulates components, relationships and uncertainties."""

    def __init__(self, read_files: Sequence[Tuple[str, str]]) -> None:
        self.read_files: List[Tuple[str, str]] = list(read_files)
        self.read_paths: List[str] = [path for path, _ in self.read_files]
        self.components: Dict[str, Component] = {}
        self.relationships: List[Relationship] = []
        self.uncertainties: List[str] = []
        self.truncated = False

    # -- accumulation ----------------------------------------------------

    def component(
        self,
        component_id: str,
        name: str,
        component_type: str,
        responsibility: str,
        path: str,
        classification: str,
    ) -> None:
        identifier = slug(component_id)
        if not identifier:
            return
        if component_type not in COMPONENT_TYPES:
            component_type = "unknown"
        existing = self.components.get(identifier)
        if existing is None:
            if len(self.components) >= MAX_COMPONENTS:
                self.truncated = True
                return
            self.components[identifier] = Component(
                id=identifier,
                name=str(name).strip() or identifier,
                type=component_type,
                responsibility=str(responsibility).strip(),
                paths=[path],
                classification=classification,
            )
            return
        if path not in existing.paths:
            existing.paths.append(path)
        if classification == "observed":
            existing.classification = "observed"
        if component_type != "unknown" and existing.type == "unknown":
            existing.type = component_type

    def relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        path: str,
        classification: str = "derived",
        confidence: str = "medium",
    ) -> None:
        source_id = slug(source)
        target_id = slug(target)
        if not source_id or not target_id or source_id == target_id:
            return
        if relationship_type not in RELATIONSHIP_KINDS:
            return
        for existing in self.relationships:
            if (
                existing.source == source_id
                and existing.target == target_id
                and existing.relationship_type == relationship_type
            ):
                if path not in existing.paths:
                    existing.paths.append(path)
                if classification == "observed":
                    existing.classification = "observed"
                return
        if len(self.relationships) >= MAX_RELATIONSHIPS:
            self.truncated = True
            return
        self.relationships.append(
            Relationship(
                source=source_id,
                target=target_id,
                relationship_type=relationship_type,
                paths=[path],
                classification=classification,
                confidence=confidence,
            )
        )

    def uncertainty(self, text: str) -> None:
        text = " ".join(str(text).split())
        if text and text not in self.uncertainties:
            self.uncertainties.append(text)

    # -- derived views ---------------------------------------------------

    def known_components(self) -> List[Component]:
        return list(self.components.values())

    def known_relationships(self) -> List[Relationship]:
        known = set(self.components)
        return [
            relationship
            for relationship in self.relationships
            if relationship.source in known
            and relationship.target in known
        ]

    def evidence(self) -> List[Dict[str, str]]:
        """The aggregate: every file this run actually read."""
        return [{"path": path} for path in self.read_paths]

    def protocol_established(self) -> bool:
        for _, content in self.read_files:
            if _TRANSPORT_RE.search(content.lower()):
                return True
        return False

    def finalize(self) -> "Extraction":
        """Bound the inventory, then say what the evidence does not."""
        if self.truncated:
            self.uncertainty(
                "the component/relationship inventory is capped at "
                "{} component(s) and {} relationship(s); further "
                "established elements are not listed".format(
                    MAX_COMPONENTS, MAX_RELATIONSHIPS
                )
            )
        relationships = self.known_relationships()
        if not relationships and self.components:
            self.uncertainty(
                "no relationship between the identified components is "
                "established by the inspected files"
            )
        servers = [
            component
            for component in self.components.values()
            if component.type == "mcp_server"
        ]
        if servers and not self.protocol_established():
            self.uncertainty(
                "the wire-level protocol between these tools and the "
                "{} MCP server(s) is not established by the inspected "
                "files".format(len(servers))
            )
        self.uncertainties = self.uncertainties[:MAX_UNCERTAINTIES]
        return self


# -- extraction rules (R1-R6) ----------------------------------------------


def _rule_mcp_json(path: str, text: str, extraction: Extraction) -> None:
    """R1 — a JSON document's ``mcpServers`` map."""
    if not path.lower().endswith(".json"):
        return
    try:
        data = json.loads(text)
    except ValueError:
        return
    if not isinstance(data, dict):
        return
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return
    for name in sorted(servers):
        entry = servers[name]
        server = slug(name)
        if not server:
            continue
        extraction.component(
            name,
            name,
            "mcp_server",
            "MCP server declared in {}".format(Path(path).name),
            path,
            "observed",
        )
        tools = entry.get("tools") if isinstance(entry, dict) else None
        for tool_name in tools if isinstance(tools, list) else []:
            if not isinstance(tool_name, str):
                continue
            tool = slug(tool_name)
            if not tool:
                continue
            extraction.component(
                tool_name,
                tool_name,
                "tool",
                "tool served by the {} MCP server".format(server),
                path,
                "observed",
            )
            extraction.relationship(
                tool, server, "EXPOSED_BY", path, "derived", "medium"
            )


def _rule_yaml_registry(
    path: str, text: str, extraction: Extraction
) -> None:
    """R2 — a YAML registry of ``- name:`` servers with ``tools:`` lists."""
    if not path.lower().endswith((".yaml", ".yml")):
        return
    names: List[str] = []
    tools: Dict[int, List[str]] = {}
    current = -1
    in_tools = -1
    for raw in text.splitlines():
        stripped = raw.strip()
        item = _SERVER_ITEM_RE.match(raw)
        if item is not None:
            names.append(item.group(1))
            current = len(names) - 1
            in_tools = -1
            continue
        if _TOOLS_KEY_RE.match(raw) is not None and current >= 0:
            in_tools = current
            continue
        if in_tools >= 0:
            entry = _LIST_ITEM_RE.match(raw)
            if entry is not None:
                tools.setdefault(in_tools, []).append(entry.group(1))
                continue
            if stripped:
                in_tools = -1
    if not names or not any(tools.values()):
        return
    for index, name in enumerate(names):
        server = slug(name)
        if not server:
            continue
        extraction.component(
            name,
            name,
            "mcp_server",
            "MCP server registered in {}".format(Path(path).name),
            path,
            "observed",
        )
        for tool_name in tools.get(index, []):
            tool = slug(tool_name)
            if not tool:
                continue
            extraction.component(
                tool_name,
                tool_name,
                "tool",
                "tool served by the {} MCP server".format(server),
                path,
                "observed",
            )
            extraction.relationship(
                tool, server, "EXPOSED_BY", path, "derived", "medium"
            )


def _rule_markdown_agents(
    path: str, text: str, extraction: Extraction
) -> None:
    """R3 — numbered ``## 7. DevOps Agent`` section headings."""
    if not path.lower().endswith(".md"):
        return
    for match in _NUMBERED_HEADING_RE.finditer(text):
        title = " ".join(match.group(1).split())
        base = title.split(" (")[0].strip()
        if not base or not _AGENT_WORD_RE.search(base):
            continue
        extraction.component(base, base, "agent", title, path, "observed")


def _rule_frontmatter_agent(
    path: str, text: str, extraction: Extraction
) -> None:
    """R4 — Markdown frontmatter ``name:`` and its ``tools:`` line."""
    if not path.lower().endswith(".md"):
        return
    frontmatter = _FRONTMATTER_RE.match(text)
    if frontmatter is None:
        return
    body = frontmatter.group(1)
    name = _FM_NAME_RE.search(body)
    tools_line = _FM_TOOLS_RE.search(body)
    if name is None or tools_line is None:
        return
    agent = slug(name.group(1))
    if not agent:
        return
    extraction.component(
        name.group(1).strip(),
        name.group(1).strip(),
        "agent",
        "agent defined by {} (frontmatter name)".format(Path(path).name),
        path,
        "observed",
    )
    for token in _MCP_TOKEN_RE.finditer(tools_line.group(1)):
        server = slug(token.group(1))
        tool = slug(token.group(2))
        if not server or not tool:
            continue
        extraction.component(
            token.group(2),
            token.group(2),
            "tool",
            "tool exposed by the {} MCP server".format(server),
            path,
            "observed",
        )
        extraction.relationship(
            agent, tool, "DECLARES", path, "observed", "high"
        )


def _rule_mcp_tokens(path: str, text: str, extraction: Extraction) -> None:
    """R5 — ``mcp__<server>__<tool>`` tokens, line by line."""
    for line in text.splitlines():
        tokens = list(_MCP_TOKEN_RE.finditer(line))
        if not tokens:
            continue
        agent = _AGENT_TOKEN_RE.search(line)
        for token in tokens:
            server = slug(token.group(1))
            tool = slug(token.group(2))
            if not server or not tool:
                continue
            extraction.component(
                token.group(1),
                token.group(1),
                "mcp_server",
                "MCP server named by an mcp__ token",
                path,
                "observed",
            )
            extraction.component(
                token.group(2),
                token.group(2),
                "tool",
                "tool exposed by the {} MCP server".format(server),
                path,
                "observed",
            )
            extraction.relationship(
                tool, server, "EXPOSED_BY", path, "derived", "medium"
            )
            if agent is not None:
                extraction.component(
                    agent.group(1),
                    agent.group(1),
                    "agent",
                    "agent holding the {} tool".format(tool),
                    path,
                    "observed",
                )
                extraction.relationship(
                    agent.group(1),
                    tool,
                    "DECLARES",
                    path,
                    "derived",
                    "medium",
                )


def _component_type_for(path: str) -> str:
    segments = path.split("/")
    directories = "/".join(segments[:-1]).lower()
    name = segments[-1].lower()
    if re.search(r"(^|/)agents?(/|$)", directories) or name.endswith(
        ("-agent.py", "-agent.js", "-agent.ts")
    ):
        return "agent"
    if re.search(r"(^|/)tools?(/|$)", directories):
        return "tool"
    if "mcp" in directories or "server" in directories:
        return "mcp_server"
    if "config" in directories:
        return "configuration"
    return "module"


def _generic_stem(stem: str) -> bool:
    """True for names that describe a file's ROLE, not its identity."""
    return stem in _GENERIC_STEMS


def _rule_code_file(path: str, text: str, extraction: Extraction) -> None:
    """R6 — a read code file: one component, then its imports."""
    if not path.lower().endswith(_CODE_SUFFIXES):
        return
    stem = Path(path).stem
    if _generic_stem(stem):
        # mcp-servers/glue-athena-server/server.py is glue-athena's
        # server, not a shared component named "server"
        segments = path.split("/")
        stem = segments[-2] if len(segments) > 1 else stem
    component_type = _component_type_for(path)
    extraction.component(
        stem,
        stem,
        component_type,
        "{} implemented by {}".format(component_type, Path(path).name),
        path,
        "observed",
    )
    targets: List[str] = []
    for match in _PY_FROM_RE.finditer(text):
        targets.extend(_module_candidates(match.group(1), path))
    for match in _PY_IMPORT_RE.finditer(text):
        targets.extend(_module_candidates(match.group(1), path))
    for match in _PY_REL_RE.finditer(text):
        reference = match.group(1) + (match.group(2) or "")
        targets.extend(_relative_candidates(reference, path))
    for match in _JS_MODULE_RE.finditer(text):
        reference = match.group(1) or match.group(2) or ""
        if reference:
            targets.extend(_relative_candidates(reference, path))
    for target in targets:
        if target not in extraction.read_paths:
            continue
        target_stem = Path(target).stem
        target_type = _component_type_for(target)
        if not slug(target_stem):
            continue
        extraction.component(
            target_stem,
            target_stem,
            target_type,
            "{} implemented by {}".format(target_type, Path(target).name),
            target,
            "observed",
        )
        extraction.relationship(
            stem, target_stem, "IMPORTS", path, "observed", "high"
        )


_RULES = (
    _rule_mcp_json,
    _rule_yaml_registry,
    _rule_markdown_agents,
    _rule_frontmatter_agent,
    _rule_mcp_tokens,
    _rule_code_file,
)


def extract_read_files(read_files: Sequence[Tuple[str, str]]) -> Extraction:
    """Run every rule over every read file — one bounded pass."""
    extraction = Extraction(read_files)
    for path, text in read_files:
        for rule in _RULES:
            rule(path, text, extraction)
    return extraction.finalize()


# -- the flow chain (one deterministic walk) -------------------------------


def build_chain(
    components: Sequence[Component],
    relationships: Sequence[Relationship],
) -> List[Relationship]:
    """The one path this report reports: a deterministic walk.

    Seeds are agent components first (the question is about how the
    work starts) and then every component id in sorted order;
    neighbours are followed in sorted order, so the same evidence
    always yields the same chain.  No chain → an empty list, never a
    guess.
    """
    adjacency: Dict[str, List[Relationship]] = {}
    for relationship in relationships:
        adjacency.setdefault(relationship.source, []).append(relationship)
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (edge.target, edge.relationship_type))
    seeds: List[str] = [
        component.id for component in components if component.type == "agent"
    ]
    seeds.extend(sorted(component.id for component in components))
    best: List[Relationship] = []
    for seed in seeds:
        chain = _walk(seed, adjacency, set())
        if len(chain) > len(best):
            best = chain
        if len(best) >= 2:
            break
    return best


def _walk(
    start: str,
    adjacency: Dict[str, List[Relationship]],
    visited: Set[str],
) -> List[Relationship]:
    """Depth-first longest simple path from one seed (bounded)."""
    if start in visited:
        return []
    visited.add(start)
    best: List[Relationship] = []
    for edge in adjacency.get(start, []):
        if len(visited) > MAX_CHAIN_HOPS:
            break
        tail = _walk(edge.target, adjacency, visited)
        candidate = [edge] + tail
        if len(candidate) > len(best):
            best = candidate
    visited.discard(start)
    return best


# -- payload builders (the reports the specialists emit) -------------------


def _summary(extraction: Extraction) -> str:
    components = extraction.known_components()
    relationships = extraction.known_relationships()
    return (
        "Read-only inspection of {} file(s) established {} "
        "component(s) and {} relationship(s).".format(
            len(extraction.read_files), len(components), len(relationships)
        )
    )


def build_architecture_payload(
    question: str,
    repository: str,
    commit: str,
    extraction: Extraction,
) -> Dict[str, Any]:
    """The ArchitectureReport over what this run read."""
    components = [
        {
            "id": component.id,
            "name": component.name,
            "type": component.type,
            "responsibility": component.responsibility,
            "evidence_refs": component.refs(),
            "classification": component.classification,
        }
        for component in extraction.known_components()
    ]
    relationships = [
        {
            "source": relationship.source,
            "target": relationship.target,
            "relationship_type": relationship.relationship_type,
            "evidence_refs": relationship.refs(),
            "classification": relationship.classification,
            "confidence": relationship.confidence,
        }
        for relationship in extraction.known_relationships()
    ]
    return {
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "summary": _summary(extraction),
        "components": components,
        "relationships": relationships,
        "findings": [],
        "evidence_refs": extraction.evidence(),
        "uncertainties": list(extraction.uncertainties),
        "status": "complete",
    }


def build_flow_payload(
    question: str,
    repository: str,
    commit: str,
    extraction: Extraction,
) -> Dict[str, Any]:
    """The FlowDependencyReport: the chain this evidence establishes."""
    components = extraction.known_components()
    chain = build_chain(components, extraction.known_relationships())
    path = [
        {
            "source": relationship.source,
            "target": relationship.target,
            "relationship_type": relationship.relationship_type,
            "evidence_refs": relationship.refs(),
            "classification": relationship.classification,
            "confidence": relationship.confidence,
        }
        for relationship in chain
    ]
    source = path[0]["source"] if path else None
    target = path[-1]["target"] if path else None
    if path:
        summary = (
            "Read-only inspection of {} file(s) established the path "
            "{} over {} hop(s).".format(
                len(extraction.read_files), _path_label(path), len(path)
            )
        )
        finding_statement = (
            "the inspected evidence establishes the path {} "
            "({} hop(s))".format(_path_label(path), len(path))
        )
    else:
        summary = (
            "Read-only inspection of {} file(s) identified {} "
            "component(s); no destination hop is established.".format(
                len(extraction.read_files), len(components)
            )
        )
        finding_statement = (
            "the inspected evidence establishes no path between the "
            "identified components"
        )
    return {
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "source": source,
        "target": target,
        "path_status": "FOUND" if path else "UNKNOWN",
        "summary": summary,
        "path": path,
        "alternative_paths": [],
        "relationships": list(path),
        "findings": [
            {
                "statement": finding_statement,
                "evidence_refs": extraction.evidence(),
                "support": "SUPPORTED" if path else "UNKNOWN",
            }
        ],
        "evidence_refs": extraction.evidence(),
        "uncertainties": list(extraction.uncertainties),
        "status": "complete",
    }


def _path_label(path: Sequence[Dict[str, Any]]) -> str:
    if not path:
        return ""
    parts = [str(path[0]["source"])]
    for step in path:
        parts.append(str(step["relationship_type"]))
        parts.append(str(step["target"]))
    return " -> ".join(parts)


# -- the diagram (a projection of the merged report) -----------------------


def _statement_node(statement: str) -> Optional[Dict[str, Any]]:
    match = re.match(
        r"^Component (\S+) \(([^)]+)\)(?:: (.*))? \[([a-z]+)\]$", statement
    )
    if match is None:
        return None
    return {
        "id": match.group(1),
        "label": match.group(1),
        "node_type": match.group(2),
        "classification": match.group(4),
    }


def _statement_edge(statement: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"^(\S+) ([A-Z_]+) (\S+) \[([a-z]+)\]$", statement)
    if match is None:
        return None
    if match.group(2) not in RELATIONSHIP_KINDS:
        return None
    return {
        "source": match.group(1),
        "target": match.group(3),
        "relationship_type": match.group(2),
        "classification": match.group(4),
    }


def build_diagram_payload(
    question: str,
    repository: str,
    commit: str,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """The DiagramSpec built ONLY from the merged report's statements.

    The analyst performs no investigation: every node, edge and
    reference below is read back from a statement the investigators
    already backed with evidence, so the diagram stays a projection.
    """
    findings = report.get("findings") or []
    evidence_refs = report.get("evidence_refs") or []
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes: Set[str] = set()
    seen_edges: Set[str] = set()
    # pass 1 — every component statement becomes a node
    for finding in findings:
        statement = str(finding.get("statement") or "")
        refs = finding.get("evidence_refs") or []
        node = _statement_node(statement)
        if node is None:
            continue
        if node["id"] in seen_nodes:
            continue
        seen_nodes.add(node["id"])
        node["evidence_refs"] = list(refs)
        node["confidence"] = "high" if refs else "unknown"
        nodes.append(node)
    # pass 2 — every relationship statement whose endpoints the same
    # report established becomes an edge (order in the report can
    # never drop one)
    for finding in findings:
        statement = str(finding.get("statement") or "")
        refs = finding.get("evidence_refs") or []
        edge = _statement_edge(statement)
        if edge is None:
            continue
        endpoints = (edge["source"], edge["target"])
        if endpoints[0] not in seen_nodes or endpoints[1] not in seen_nodes:
            continue
        edge_id = "{}-{}-{}".format(
            edge["source"], edge["relationship_type"], edge["target"]
        )
        if edge_id in seen_edges:
            continue
        seen_edges.add(edge_id)
        edge["id"] = edge_id
        edge["evidence_refs"] = list(refs)
        edge["confidence"] = "high" if refs else "unknown"
        edges.append(edge)

    node_types = sorted({node["node_type"] for node in nodes})
    kinds = sorted({edge["relationship_type"] for edge in edges})
    groups = []
    for node_type in node_types:
        member_ids = [
            node["id"] for node in nodes if node["node_type"] == node_type
        ]
        if len(member_ids) >= 2:
            groups.append(
                {
                    "id": "group-{}".format(node_type),
                    "label": "{}(s)".format(node_type),
                    "node_ids": member_ids,
                }
            )
    legend = [
        {
            "kind": "node_type",
            "term": node_type,
            "meaning": _NODE_MEANINGS.get(node_type, node_type),
        }
        for node_type in node_types
    ]
    legend.extend(
        {
            "kind": "relationship_type",
            "term": kind,
            "meaning": _RELATIONSHIP_MEANINGS.get(kind, kind),
        }
        for kind in kinds
    )
    if "mcp_server" in node_types and "tool" in node_types:
        diagram_type = "AGENT_TOOL_MCP"
    elif "agent" in node_types:
        diagram_type = "ARCHITECTURE"
    else:
        diagram_type = "COMPONENT"
    uncertainties = [
        str(item)
        for item in (
            report.get("uncertainties") or []
        ) + (report.get("open_questions") or [])
    ]
    return {
        "diagram_type": diagram_type,
        "title": "{} ({} component(s))".format(
            " ".join(question.split())[:160] or "repository", len(nodes)
        ),
        "question": question,
        "repository_scope": repository,
        "commit_scope": commit,
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "legend": legend,
        "evidence_refs": list(evidence_refs),
        "uncertainties": uncertainties,
        "status": "complete",
    }


# -- the answer (synthesized from the investigation, never from a diagram) --

_COMPONENT_STATEMENT_RE = re.compile(
    r"^Component (\S+) \(([^)]+)\)(?:: (.*))? \[([a-z]+)\]$"
)
_RELATIONSHIP_STATEMENT_RE = re.compile(
    r"^(\S+) ([A-Z_]+) (\S+) \[([a-z]+)\]$"
)
_ANSWER_SAMPLE = 6


def synthesize_answer(report: Any) -> str:
    """One evidence-backed answer, read back from the merged report.

    Line one states how many statements the investigation
    established; the following lines say what those statements
    contain (components by type, relationships by kind, the path the
    walk found) and finish with what the evidence does NOT
    establish.  A diagram is never consulted: the answer is the
    investigation's, in the investigation's own words.
    """
    findings = list(getattr(report, "findings", None) or [])
    if not findings:
        # a run whose merged report never came together still carries
        # the specialist's own conclusions — the same statement shapes
        findings = list(getattr(report, "conclusions", None) or [])
    statements = [finding.statement for finding in findings]
    nodes: List[Component] = []
    edges: List[Dict[str, Any]] = []
    seen_edges: Set[str] = set()
    for statement in statements:
        match = _COMPONENT_STATEMENT_RE.match(statement)
        if match is not None:
            nodes.append(
                Component(
                    id=match.group(1),
                    name=match.group(1),
                    type=match.group(2),
                    responsibility=match.group(3) or "",
                    classification=match.group(4),
                )
            )
            continue
        match = _RELATIONSHIP_STATEMENT_RE.match(statement)
        if match is not None and match.group(2) in RELATIONSHIP_KINDS:
            edge_id = "|".join(
                (match.group(1), match.group(2), match.group(3))
            )
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            edges.append(
                {
                    "source": match.group(1),
                    "target": match.group(3),
                    "relationship_type": match.group(2),
                }
            )

    lines = [
        "{} evidence-backed statement(s) established by the "
        "investigation.".format(len(statements))
    ]
    if nodes:
        counts: Dict[str, List[str]] = {}
        for node in nodes:
            counts.setdefault(node.type, []).append(node.id)
        counts_text = ", ".join(
            "{} {}(s)".format(len(counts[node_type]), node_type)
            for node_type in sorted(counts)
        )
        sample = sorted(node.id for node in nodes)[:_ANSWER_SAMPLE]
        lines.append(
            "components: {} — e.g. {}".format(
                counts_text, ", ".join(sample)
            )
        )
    if edges:
        kinds: Dict[str, int] = {}
        for edge in edges:
            kind = edge["relationship_type"]
            kinds[kind] = kinds.get(kind, 0) + 1
        lines.append(
            "relationships: {}".format(
                ", ".join(
                    "{} {}".format(kinds[kind], kind)
                    for kind in sorted(kinds)
                )
            )
        )
    relationships = [
        Relationship(
            source=edge["source"],
            target=edge["target"],
            relationship_type=edge["relationship_type"],
        )
        for edge in edges
    ]
    chain = build_chain(nodes, relationships)
    if chain:
        label_parts = [chain[0].source]
        for step in chain:
            label_parts.append(step.relationship_type)
            label_parts.append(step.target)
        lines.append(
            "established path: {} ({} hop(s)).".format(
                " -> ".join(label_parts), len(chain)
            )
        )
    else:
        lines.append(
            "established path: none in the inspected evidence."
        )
    uncertainties = [
        str(item)
        for item in list(getattr(report, "uncertainties", None) or [])
        + list(getattr(report, "open_questions", None) or [])
    ]
    if uncertainties:
        lines.append(
            "The evidence does not establish: {}".format(
                uncertainties[0]
            )
        )
    return "\n".join(lines)


# -- the scripted models (the offline participants) ------------------------


def structured_response(payload: Dict[str, Any]) -> LLMResponse:
    """One structured assistant turn (the report the agent parses)."""
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def tool_calls_response(
    calls: Sequence[ToolCall],
) -> LLMResponse:
    """One tool-requesting assistant turn."""
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


class InvestigatorModel:
    """The scripted investigator: search → read → derive, bounded.

    Each ``complete`` call is one turn of the loop: it reads the
    observations the tools produced, applies the bounded plan, and
    either asks for the next batch of reads or emits the structured
    report derived from everything read so far.  No network, no
    model, no answer authored ahead of time.
    """

    def __init__(
        self,
        role: str,
        question: str,
        repository: str,
        commit: str,
        entry_target: str,
    ) -> None:
        self.role = role
        self.question = question
        self.repository = repository
        self.commit = commit
        self.entry_target = entry_target
        self.requests: List[Any] = []
        self.model = "offline-scripted"
        self.read_files: List[Tuple[str, str]] = []
        self.attempted: List[str] = []
        self.candidates: Dict[str, int] = {}
        self.searched: List[str] = []

    # -- one turn --------------------------------------------------------

    def complete(self, request: Any) -> LLMResponse:
        self.requests.append(request)
        observations = observations_from_messages(
            getattr(request, "messages", None) or []
        )
        self._absorb(observations)
        calls = self._plan(observations)
        if calls is None:
            return structured_response(self._payload())
        return tool_calls_response(calls)

    def _absorb(self, observations: Sequence[ToolObservation]) -> None:
        """Fold this turn's observations into the working set."""
        for observation in observations:
            if observation.read is not None:
                read = observation.read
                if (read.path, read.content) not in self.read_files:
                    self.read_files.append((read.path, read.content))
                if read.path not in self.attempted:
                    self.attempted.append(read.path)
                # the same module under its package spelling: one
                # read of pkg.py settles pkg/__init__.py too
                if read.path.endswith(".py"):
                    sibling = read.path[:-3] + "/__init__.py"
                    if sibling in self.candidates:
                        del self.candidates[sibling]
                for path, bonus in discover_candidates(
                    read.path, read.content
                ).items():
                    if path in self.attempted:
                        continue
                    self.candidates[path] = max(
                        self.candidates.get(path, 0), bonus
                    )
            elif observation.tool == "read_file":
                # a failed read is still an attempt: never repeated
                pass
            if observation.search is not None:
                search = observation.search
                if search.query not in self.searched:
                    self.searched.append(search.query)
                for item in search.results:
                    path = item.get("path")
                    if not isinstance(path, str) or not path:
                        continue
                    if path in self.attempted:
                        continue
                    self.candidates[path] = max(
                        self.candidates.get(path, 0), 0
                    )

    def _plan(
        self, observations: Sequence[ToolObservation]
    ) -> Optional[List[ToolCall]]:
        executed = len(observations)
        reads = [
            observation
            for observation in observations
            if observation.read is not None
        ]
        if executed >= MAX_TOOL_CALLS:
            return None
        if not reads and self.entry_target not in self.attempted:
            plan = [
                ToolCall(
                    id="call_offline_{}_entry".format(self.role),
                    name="read_file",
                    arguments={"path": self.entry_target},
                )
            ]
            self.attempted.append(self.entry_target)
            for query in SEARCH_QUERIES:
                if query in self.searched:
                    continue
                if len(plan) >= 3:
                    break
                if executed + len(plan) >= MAX_TOOL_CALLS:
                    break
                plan.append(
                    ToolCall(
                        id="call_offline_{}_search_{}".format(
                            self.role, len(plan)
                        ),
                        name="search_repository",
                        arguments={
                            "query": query,
                            "max_results": SEARCH_MAX_RESULTS,
                        },
                    )
                )
            return plan
        ranked = [
            path
            for path in rank_candidates(self.candidates)
            if path not in self.attempted
        ]
        if not ranked:
            return None
        room = min(
            3, MAX_READS - len(reads), MAX_TOOL_CALLS - executed
        )
        if room <= 0:
            return None
        # one call per module: pkg.py and pkg/__init__.py never both
        # spend a slot of the budget
        settled = set()
        for path, _content in self.read_files:
            key = _module_key(path)
            if key is not None:
                settled.add(key)
        selected: List[str] = []
        for path in ranked:
            if len(selected) >= room:
                break
            key = _module_key(path)
            if key is not None:
                if key in settled:
                    continue
                settled.add(key)
            selected.append(path)
        if not selected:
            return None
        # recorded as attempted the moment they are requested: a
        # candidate that could not be read is never asked for twice
        for path in selected:
            self.attempted.append(path)
        return [
            ToolCall(
                id="call_offline_{}_read_{}".format(self.role, index),
                name="read_file",
                arguments={"path": path},
            )
            for index, path in enumerate(selected)
        ]

    def _payload(self) -> Dict[str, Any]:
        extraction = extract_read_files(self.read_files)
        if not extraction.components:
            # Nothing structured was found in this repository.  The
            # one thing the evidence DOES establish is that the entry
            # target was read, so that is the one component — a single
            # honest node, never an invented path.  It also keeps the
            # report's findings non-empty, which the validators
            # require of every investigation.
            name = Path(self.entry_target).stem or self.entry_target
            extraction.component(
                name,
                name,
                _component_type_for(self.entry_target),
                "the investigation's entry point, read in full",
                self.entry_target,
                "observed",
            )
        if self.role == "flow":
            return build_flow_payload(
                self.question, self.repository, self.commit, extraction
            )
        return build_architecture_payload(
            self.question, self.repository, self.commit, extraction
        )


class AnalystModel:
    """The scripted analyst: project the merged report into a diagram.

    It receives the merged ``InvestigationReport`` as JSON (the same
    message every analyst receives) and answers with the DiagramSpec
    over THOSE statements — no second investigation, no repository
    access, no diagram invented from a question.
    """

    def __init__(
        self, question: str, repository: str, commit: str
    ) -> None:
        self.question = question
        self.repository = repository
        self.commit = commit
        self.requests: List[Any] = []
        self.model = "offline-scripted"

    def complete(self, request: Any) -> LLMResponse:
        self.requests.append(request)
        report = self._report(getattr(request, "messages", None) or [])
        return structured_response(
            build_diagram_payload(
                self.question, self.repository, self.commit, report
            )
        )

    @staticmethod
    def _report(messages: Sequence[Any]) -> Dict[str, Any]:
        for message in reversed(list(messages)):
            if getattr(message, "role", "") != "user":
                continue
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                continue
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped.startswith("{"):
                    continue
                try:
                    payload = json.loads(stripped)
                except ValueError:
                    continue
                if isinstance(payload, dict) and "findings" in payload:
                    return payload
        raise ValueError(
            "the analyst request carries no merged investigation "
            "report to project"
        )
