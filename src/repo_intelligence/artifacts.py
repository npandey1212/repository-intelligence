"""Centralized generated-artifact layout under artifacts/<snapshot_id>/.

Single generated-output root for M1–M4:

    <repository/project-root>/artifacts/
    └── <snapshot_id>/
        ├── schemas/
        │   ├── repository_facts.v2.json
        │   └── repository_snapshot.v1.json
        ├── facts/
        │   └── repository_facts.json
        ├── snapshot/
        │   └── repository_snapshot.json
        ├── repository/
        │   └── <repository_name>/    exact pinned checkout (M1, M18
        │                             names the subdir from the
        │                             snapshot's repository.name)
        └── engineering/
            ├── engineering_construction_rules.v1.json
            ├── engineering_model.json
            └── engineering_graph.json          (M11 persisted graph)
        └── architecture/
            ├── architecture_construction_rules.v1.json
            └── architecture_model.json

The default artifact root is resolved from the installed package location
(project root), not the process current working directory. Explicit output
overrides (CLI ``--output`` / API ``output_root=``) take precedence.

Authoritative schemas remain under schemas/ in source; copies under
artifacts/<id>/schemas/ are generated for the pinned snapshot only.

The canonical M3 construction-rule matrix lives under
artifacts/<snapshot_id>/engineering/ (not the project root).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

# Package lives at <project-root>/src/repo_intelligence/artifacts.py
# so parents[2] is the repository/project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR_NAME = "artifacts"

# Primary pinned ADOP snapshot id used for the canonical M3 matrix artifact.
PRIMARY_SNAPSHOT_ID = "main_2026-09-22_ef11dac"
CONSTRUCTION_RULES_FILENAME = "engineering_construction_rules.v1.json"
ENGINEERING_MODEL_FILENAME = "engineering_model.json"
ENGINEERING_GRAPH_FILENAME = "engineering_graph.json"


def default_artifacts_root() -> Path:
    """Default generated-output root: ``<project-root>/artifacts``.

    Absolute and independent of the process current working directory.
    """
    return PROJECT_ROOT / ARTIFACTS_DIR_NAME


# Absolute default (not CWD-relative). Prefer default_artifacts_root() at
# call time when re-resolution is desired.
ARTIFACTS_ROOT = default_artifacts_root()

# Source schema locations (authoritative definitions).
FACTS_SCHEMA_SOURCE = PROJECT_ROOT / "schemas" / "repository_facts.v2.json"
SNAPSHOT_SCHEMA_SOURCE = PROJECT_ROOT / "schemas" / "repository_snapshot.v1.json"


def resolve_output_root(output_root: Optional[Path]) -> Path:
    """Return explicit override as-is, or the default project artifact root."""
    if output_root is None:
        return default_artifacts_root()
    return Path(output_root)


def snapshot_artifact_dir(output_root: Path, snapshot_id: str) -> Path:
    return Path(output_root) / snapshot_id


def facts_path(output_root: Path, snapshot_id: str) -> Path:
    return snapshot_artifact_dir(output_root, snapshot_id) / "facts" / "repository_facts.json"


def snapshot_path(output_root: Path, snapshot_id: str) -> Path:
    return (
        snapshot_artifact_dir(output_root, snapshot_id)
        / "snapshot"
        / "repository_snapshot.json"
    )


REPOSITORY_DIR_NAME = "repository"
SNAPSHOT_FILENAME = "repository_snapshot.json"


def _safe_repository_subdir(name: Any) -> Optional[str]:
    """Return *name* as a single safe path segment, else ``None``."""
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped or stripped in {".", ".."}:
        return None
    if "/" in stripped or "\\" in stripped or "\x00" in stripped:
        return None
    return stripped


def _snapshot_repository_name(snapshot_json: Path) -> Optional[str]:
    """``repository.name`` from a snapshot JSON, sanitized, or ``None``.

    Missing, unreadable, or malformed documents yield ``None`` — the
    resolver then falls back to the legacy flat layout instead of
    guessing.
    """
    try:
        with Path(snapshot_json).open("r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    repository = data.get("repository")
    if not isinstance(repository, dict):
        return None
    return _safe_repository_subdir(repository.get("name"))


def _resolve_repository_container(
    container: Path, name: Optional[str]
) -> Path:
    """Named checkout when present, legacy flat layout as fallback.

    Resolution order (M18):

    1. ``<container>/<name>/`` when the named directory exists — the
       layout written from M18 on;
    2. ``<container>/`` when it still holds entries — pre-M18 flat
       artifacts (and their fixtures) keep resolving exactly as before;
    3. ``<container>/<name>/`` prospectively when *name* is known but
       no checkout exists yet (readers treat a non-existent path as
       "not found"; the writer uses ``repository_dir_for_name``);
    4. ``<container>/`` when no snapshot name is available at all.
    """
    if name is not None:
        named = container / name
        if named.is_dir():
            return named
        try:
            has_entries = container.is_dir() and any(container.iterdir())
        except OSError:
            has_entries = True
        if has_entries:
            return container
        return named
    return container


def repository_dir_for_name(
    output_root: Path, snapshot_id: str, repository_name: str
) -> Path:
    """Writer-side target: ``<output>/<id>/repository/<repository_name>/``.

    The snapshot's own ``repository.name`` decides the subdirectory (no
    hard-coded project names). An unusable name falls back to the
    legacy flat directory rather than failing the snapshot.
    """
    container = (
        snapshot_artifact_dir(output_root, snapshot_id)
        / REPOSITORY_DIR_NAME
    )
    subdir = _safe_repository_subdir(repository_name)
    if subdir is None:
        return container
    return container / subdir


def repository_dir(output_root: Path, snapshot_id: str) -> Path:
    """Pinned checkout persisted by M1: ``.../repository/<name>/``.

    Smart resolver (see ``_resolve_repository_container``): named
    layout when written, legacy flat layout when that is what the run
    directory holds.
    """
    container = (
        snapshot_artifact_dir(output_root, snapshot_id)
        / REPOSITORY_DIR_NAME
    )
    name = _snapshot_repository_name(snapshot_path(output_root, snapshot_id))
    return _resolve_repository_container(container, name)


def repository_dir_for_snapshot(snapshot_artifact: Path) -> Path:
    """Pinned repository directory that accompanies an M1 snapshot artifact.

    The snapshot artifact lives at ``<run>/snapshot/repository_snapshot.json``
    and M1 persists the exact checkout at ``<run>/repository/<name>/``
    beside it, so the run directory is the artifact path's grandparent.
    The repository name comes from that same snapshot document, with
    the legacy flat layout as fallback (see
    ``_resolve_repository_container``).
    """
    artifact = Path(snapshot_artifact)
    parent = artifact.resolve().parent
    run_dir = parent.parent if parent.name == "snapshot" else parent
    container = run_dir / REPOSITORY_DIR_NAME
    name = _snapshot_repository_name(artifact)
    return _resolve_repository_container(container, name)


def schemas_dir(output_root: Path, snapshot_id: str) -> Path:
    return snapshot_artifact_dir(output_root, snapshot_id) / "schemas"


def engineering_dir(output_root: Path, snapshot_id: str) -> Path:
    return snapshot_artifact_dir(output_root, snapshot_id) / "engineering"


def construction_rules_path(output_root: Path, snapshot_id: str) -> Path:
    return (
        engineering_dir(output_root, snapshot_id) / CONSTRUCTION_RULES_FILENAME
    )


def engineering_model_path(output_root: Path, snapshot_id: str) -> Path:
    return engineering_dir(output_root, snapshot_id) / ENGINEERING_MODEL_FILENAME


def engineering_graph_path(output_root: Path, snapshot_id: str) -> Path:
    """M11 persisted graph artifact beside the model.

    Derived from ``engineering_model.json`` by a deterministic
    transformation — a projection, never a second source of truth.
    """
    return engineering_dir(output_root, snapshot_id) / ENGINEERING_GRAPH_FILENAME


def architecture_dir(output_root: Path, snapshot_id: str) -> Path:
    return snapshot_artifact_dir(output_root, snapshot_id) / "architecture"


ARCHITECTURE_RULES_FILENAME = "architecture_construction_rules.v1.json"


def architecture_rules_path(output_root: Path, snapshot_id: str) -> Path:
    return (
        architecture_dir(output_root, snapshot_id)
        / ARCHITECTURE_RULES_FILENAME
    )


def architecture_model_path(output_root: Path, snapshot_id: str) -> Path:
    return architecture_dir(output_root, snapshot_id) / "architecture_model.json"


ARCHITECTURE_CONTEXT_DIR_NAME = "architecture_context"
ARCHITECTURE_CONTEXT_FILENAME = "architecture_context.md"


def architecture_context_dir(output_root: Path, snapshot_id: str) -> Path:
    """Run-specific context dir: <output>/<snapshot_id>/architecture_context."""
    return (
        snapshot_artifact_dir(output_root, snapshot_id)
        / ARCHITECTURE_CONTEXT_DIR_NAME
    )


def architecture_context_path(output_root: Path, snapshot_id: str) -> Path:
    return (
        architecture_context_dir(output_root, snapshot_id)
        / ARCHITECTURE_CONTEXT_FILENAME
    )


def diagram_dir(output_root: Path, snapshot_id: str) -> Path:
    """M5 diagram exports: Mermaid, Graphviz DOT, optional SVG, validation."""
    return snapshot_artifact_dir(output_root, snapshot_id) / "diagram"


def copy_schema_artifacts(output_root: Path, snapshot_id: str) -> Path:
    """Copy authoritative source schemas into artifacts/<id>/schemas/."""
    dest = schemas_dir(output_root, snapshot_id)
    dest.mkdir(parents=True, exist_ok=True)
    for source in (FACTS_SCHEMA_SOURCE, SNAPSHOT_SCHEMA_SOURCE):
        if not source.is_file():
            continue
        shutil.copy2(str(source), str(dest / source.name))
    return dest


def copy_snapshot_artifact(
    source_snapshot: Path, output_root: Path, snapshot_id: str
) -> Path:
    """Place a snapshot JSON under artifacts/<id>/snapshot/."""
    dest = snapshot_path(output_root, snapshot_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source_snapshot)
    if source.resolve() != dest.resolve():
        shutil.copy2(str(source), str(dest))
    return dest
