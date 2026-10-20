"""Domain model for repository_snapshot.v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SCHEMA_VERSION = "repository_snapshot.v1"


@dataclass
class RepositoryInfo:
    """Identity of the source repository."""

    name: str
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "url": self.url}


@dataclass
class SnapshotInfo:
    """Exact Git state being identified."""

    ref: str
    commit_sha: str
    commit_short_sha: str
    captured_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "commit_sha": self.commit_sha,
            "commit_short_sha": self.commit_short_sha,
            "captured_at": self.captured_at,
        }


@dataclass
class WorkspaceInfo:
    """State of the temporary/local Git workspace."""

    working_tree_clean: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"working_tree_clean": self.working_tree_clean}


@dataclass
class AnalyzerInfo:
    """Tool that produced this snapshot."""

    name: str
    version: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass
class RepositorySnapshot:
    """A reproducible identification of one repository state."""

    schema_version: str = SCHEMA_VERSION
    repository: RepositoryInfo = field(
        default_factory=lambda: RepositoryInfo(name="", url="")
    )
    snapshot: SnapshotInfo = field(
        default_factory=lambda: SnapshotInfo(
            ref="", commit_sha="", commit_short_sha="", captured_at=""
        )
    )
    workspace: WorkspaceInfo = field(
        default_factory=lambda: WorkspaceInfo(working_tree_clean=True)
    )
    analyzer: AnalyzerInfo = field(
        default_factory=lambda: AnalyzerInfo(name="", version="")
    )

    @property
    def snapshot_id(self) -> str:
        """Human-friendly id: ``<ref>_<date>_<short-sha>``."""
        date = _date_from_timestamp(self.snapshot.captured_at)
        ref = sanitize_ref_for_id(self.snapshot.ref)
        return "{}_{}_{}".format(ref, date, self.snapshot.commit_short_sha)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "workspace": self.workspace.to_dict(),
            "analyzer": self.analyzer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositorySnapshot":
        repository = data.get("repository", {}) or {}
        snapshot = data.get("snapshot", {}) or {}
        workspace = data.get("workspace", {}) or {}
        analyzer = data.get("analyzer", {}) or {}
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            repository=RepositoryInfo(
                name=repository.get("name", ""),
                url=repository.get("url", ""),
            ),
            snapshot=SnapshotInfo(
                ref=snapshot.get("ref", ""),
                commit_sha=snapshot.get("commit_sha", ""),
                commit_short_sha=snapshot.get("commit_short_sha", ""),
                captured_at=snapshot.get("captured_at", ""),
            ),
            workspace=WorkspaceInfo(
                working_tree_clean=bool(
                    workspace.get("working_tree_clean", True)
                )
            ),
            analyzer=AnalyzerInfo(
                name=analyzer.get("name", ""),
                version=analyzer.get("version", ""),
            ),
        )


def sanitize_ref_for_id(ref: str) -> str:
    """Make *ref* safe for use as a directory name.

    ``feature/foo`` becomes ``feature-foo``; other characters are kept
    unless they are problematic on common filesystems.
    """
    cleaned = ref.strip().replace("/", "-")
    for ch in '\\:*?"<>| ':
        cleaned = cleaned.replace(ch, "-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "ref"


def _date_from_timestamp(timestamp: str) -> str:
    """Extract ``YYYY-MM-DD`` from an ISO-8601 timestamp (UTC date part)."""
    if not timestamp:
        return "unknown-date"
    # captured_at is ISO-8601; date is everything before 'T'
    date_part = timestamp.split("T", 1)[0]
    return date_part if date_part else "unknown-date"
