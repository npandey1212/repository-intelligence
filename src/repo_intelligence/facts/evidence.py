"""Evidence references for M2 facts (where a fact was observed).

Evidence is provenance only: path, optional lines/symbol, artifact kind,
and the pinned commit. Evidence does not interpret what a fact means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

# Controlled vocabulary — keep small and stable.
ARTIFACT_TYPES = frozenset(
    {
        "python",
        "configuration",
        "agent_definition",
        "mcp_definition",
        "documentation",
        "test",
        "file",
        "directory",
    }
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class EvidenceRef:
    """Pointer from a fact back to observed source material."""

    path: str
    artifact_type: str
    commit: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "path": self.path,
            "artifact_type": self.artifact_type,
            "commit": self.commit,
        }
        if self.line_start is not None:
            out["line_start"] = self.line_start
        if self.line_end is not None:
            out["line_end"] = self.line_end
        if self.symbol is not None:
            out["symbol"] = self.symbol
        # Stable key order for byte-equivalent JSON.
        ordered: Dict[str, Any] = {
            "path": out["path"],
            "artifact_type": out["artifact_type"],
            "commit": out["commit"],
        }
        if "line_start" in out:
            ordered["line_start"] = out["line_start"]
        if "line_end" in out:
            ordered["line_end"] = out["line_end"]
        if "symbol" in out:
            ordered["symbol"] = out["symbol"]
        return ordered

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef":
        return cls(
            path=data.get("path", ""),
            artifact_type=data.get("artifact_type", ""),
            commit=data.get("commit", ""),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            symbol=data.get("symbol"),
        )


def make_evidence(
    path: str,
    artifact_type: str,
    commit: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a deterministic evidence dict for a fact."""
    return EvidenceRef(
        path=path,
        artifact_type=artifact_type,
        commit=commit,
        line_start=line_start,
        line_end=line_end,
        symbol=symbol,
    ).to_dict()


def is_repository_relative_path(path: str) -> bool:
    """True when *path* is non-empty, normalized, repo-relative, not escaping."""
    if not path or not isinstance(path, str):
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    if len(path) >= 2 and path[1] == ":":
        return False  # Windows drive
    if "\\" in path:
        return False
    if path != path.strip():
        return False
    posix = PurePosixPath(path)
    if posix.is_absolute():
        return False
    parts = posix.parts
    if not parts:
        return False
    if any(part == ".." for part in parts):
        return False
    if parts[0] == "." and len(parts) > 1:
        return False
    return True


def validate_evidence_ref(
    evidence: Dict[str, Any],
    snapshot_commit: str,
    label: str = "evidence",
) -> List[str]:
    """Integrity checks for a single evidence reference."""
    problems: List[str] = []
    path = evidence.get("path")
    artifact_type = evidence.get("artifact_type")
    commit = evidence.get("commit")
    line_start = evidence.get("line_start")
    line_end = evidence.get("line_end")
    symbol = evidence.get("symbol")

    if not path:
        problems.append("{}: missing path".format(label))
    elif not is_repository_relative_path(str(path)):
        problems.append(
            "{}: path must be repository-relative and non-escaping: {!r}".format(
                label, path
            )
        )

    if not artifact_type:
        problems.append("{}: missing artifact_type".format(label))
    elif artifact_type not in ARTIFACT_TYPES:
        problems.append(
            "{}: artifact_type {!r} not in controlled vocabulary".format(
                label, artifact_type
            )
        )

    if not commit:
        problems.append("{}: missing commit".format(label))
    elif not _COMMIT_RE.match(str(commit)):
        problems.append("{}: commit must be 40-char lowercase hex".format(label))
    elif snapshot_commit and commit != snapshot_commit:
        problems.append(
            "{}: commit {!r} != snapshot.commit_sha".format(label, commit)
        )

    if line_start is not None:
        if not isinstance(line_start, int) or isinstance(line_start, bool):
            problems.append("{}: line_start must be an integer".format(label))
        elif line_start < 1:
            problems.append("{}: line_start must be >= 1".format(label))
    if line_end is not None:
        if not isinstance(line_end, int) or isinstance(line_end, bool):
            problems.append("{}: line_end must be an integer".format(label))
        elif line_end < 1:
            problems.append("{}: line_end must be >= 1".format(label))
        elif line_start is not None and isinstance(line_start, int):
            if line_end < line_start:
                problems.append(
                    "{}: line_end ({}) < line_start ({})".format(
                        label, line_end, line_start
                    )
                )

    if symbol is not None and not str(symbol).strip():
        problems.append("{}: symbol must be non-empty when present".format(label))

    return problems


def validate_evidence_list(
    items: Iterable[Dict[str, Any]],
    snapshot_commit: str,
    section: str,
) -> List[str]:
    """Validate evidence attached to every item in *items* (must all have evidence)."""
    problems: List[str] = []
    for index, item in enumerate(items):
        evidence = item.get("evidence")
        label = "{}[{}].evidence".format(section, index)
        if evidence is None:
            problems.append("{}: missing evidence".format(label))
            continue
        if not isinstance(evidence, dict):
            problems.append("{}: evidence must be an object".format(label))
            continue
        problems.extend(
            validate_evidence_ref(
                evidence, snapshot_commit, label=label
            )
        )
    return problems


def attach_evidence_to_record(
    record: Dict[str, Any],
    commit: str,
    artifact_type: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    symbol: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Set ``evidence`` on *record* (in place) and return it."""
    record_path = path if path is not None else record.get("path")
    if line_start is None:
        line_start = record.get("line_start", record.get("line"))
    if line_end is None:
        line_end = record.get("line_end")
    if symbol is None:
        # Only invent a symbol from the record when we also have a line
        # (definition-level provenance). File-level facts keep symbol unset
        # unless the caller passed one explicitly via path/record symbol.
        if line_start is not None:
            symbol = record.get("symbol") or record.get("name") or record.get("module")
        else:
            symbol = record.get("symbol")
    record["evidence"] = make_evidence(
        path=record_path or "",
        artifact_type=artifact_type,
        commit=commit,
        line_start=line_start if isinstance(line_start, int) else None,
        line_end=line_end if isinstance(line_end, int) else None,
        symbol=symbol if symbol else None,
    )
    return record
