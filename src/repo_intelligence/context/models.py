"""ArchitectureContext contract (architecture_context.v1).

Repository-agnostic. The contract defines section/field identity and the
Markdown projection rules only — no domain concepts from any particular
analyzed repository.

Repo-level template (the contract, created once):

    contract/architecture_context_template.json

Generated artifact (per run):

    artifacts/<snapshot_id>/architecture_context/architecture_context.md

Flow:

    M1–M4 artifacts + contract/architecture_context_template.json
        → architecture_context.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..artifacts import (
    ARCHITECTURE_CONTEXT_FILENAME,
    PROJECT_ROOT,
    architecture_context_dir,
    resolve_output_root,
)

ARCHITECTURE_CONTEXT_SCHEMA_VERSION = "architecture_context.v1"
GENERATOR_NAME = "repository-intelligence architecture-context"
CONTEXT_TEMPLATE_PATH = (
    PROJECT_ROOT / "contract" / "architecture_context_template.json"
)

# Top-level template keys, in template/Markdown order.
CONTEXT_SECTION_KEYS = (
    "schema_version",
    "repository",
    "structure",
    "technology",
    "architecture",
    "workflows",
    "interfaces",
    "operations",
    "documentation",
    "evidence",
)

# Markdown sections (template order; schema_version has no section).
SECTION_HEADINGS = (
    "Repository",
    "Structure",
    "Technology",
    "Architecture",
    "Workflows",
    "Interfaces",
    "Operations",
    "Documentation",
    "Evidence / Limitations",
)

# List fields rendered as structured blocks instead of plain string lists.
SPECIAL_LIST_FIELDS = ("components", "relationships", "workflows", "sources")

# Deterministic caps for long observed path lists.
KEY_FILE_LIMIT = 20
KEY_DOCUMENT_LIMIT = 20
UNRESOLVED_SAMPLE_LIMIT = 20

EMPTY_LIST_MARKER = "_none recorded_"
EMPTY_VALUE_MARKER = "_not established_"

TITLE = "Architecture Context"
CONTRACT_LABEL = "Contract"
COMMIT_LABEL = "Pinned commit"
SNAPSHOT_LABEL = "Snapshot"

_LABEL_OVERRIDES = {"apis": "APIs"}

_SOURCE_LABELS = (
    ("architecture_model.json", "M4 architecture model"),
    ("repository_facts.json", "M2 repository facts"),
    ("repository_snapshot.json", "M1 repository snapshot"),
    ("architecture_context_template.json", "Architecture context template"),
)

_DEFAULT_SOURCE_PATHS = (
    "artifacts/<snapshot_id>/architecture/architecture_model.json",
    "artifacts/<snapshot_id>/facts/repository_facts.json",
)


class ArchitectureContextError(Exception):
    """Raised when ArchitectureContext construction fails."""


class ArchitectureContextValidationError(ArchitectureContextError):
    """Raised when architecture context artifacts fail validation."""


def field_label(key: str) -> str:
    """Human label for a template field (``key_files`` → ``Key files``)."""
    override = _LABEL_OVERRIDES.get(key)
    if override is not None:
        return override
    text = key.replace("_", " ")
    if not text:
        return text
    return text[:1].upper() + text[1:]


def format_line(label: str, value: Any = None, code: bool = False) -> str:
    """Render one ``- <label>: <value>`` line (empty value → marker)."""
    if value is None or value == "":
        text = EMPTY_VALUE_MARKER
    elif code:
        text = "`{}`".format(value)
    else:
        text = str(value)
    return "- {}: {}".format(label, text)


def empty_list_line(key: str) -> str:
    """Render the inline line used for an empty list field."""
    return "- {}: {}".format(field_label(key), EMPTY_LIST_MARKER)


def format_source_line(entry: Dict[str, Any]) -> str:
    """Render one evidence source entry (``label: path``)."""
    path = str(entry.get("path") or "")
    label = str(entry.get("label") or source_label(path))
    return "{}: {}".format(label, path)


def source_label(path: str) -> str:
    """Role label for a known pipeline artifact path, else its file name."""
    name = Path(str(path)).name
    for suffix, label in _SOURCE_LABELS:
        if name == suffix:
            return label
    return name


def source_identity(sources: Optional[List[Dict[str, Any]]]) -> Tuple[str, str]:
    """First commit/snapshot identity recorded on evidence sources."""
    commit_sha = ""
    snapshot_id = ""
    for entry in sources or []:
        if not commit_sha and entry.get("commit_sha"):
            commit_sha = str(entry["commit_sha"])
        if not snapshot_id and entry.get("snapshot_id"):
            snapshot_id = str(entry["snapshot_id"])
    return commit_sha, snapshot_id


def readme_first_key(path: str):
    """Sort key that lists README-style files before other documents."""
    name = Path(str(path)).name.lower()
    return (0 if name.startswith("readme") else 1, str(path))


def display_path(path: Any) -> str:
    """Project-relative path when possible (machine-independent output)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def unknown_test_files_line(count: int) -> str:
    return "Test files observed as facts: {}.".format(count)


def unknown_unresolved_line(count: int) -> str:
    return "Unresolved architecture claims reported by M4: {}.".format(count)


def limitation_mappings_line(count: int) -> str:
    return "Engineering -> architecture mappings recorded in M4: {}.".format(
        count
    )


def _template_contract_problems(template: Any) -> List[str]:
    """Structural problems in the template itself (contract identity)."""
    if not isinstance(template, dict):
        return ["template_must_be_a_json_object"]
    problems: List[str] = []
    if list(template.keys()) != list(CONTEXT_SECTION_KEYS):
        problems.append(
            "template_keys_mismatch: expected {} got {}".format(
                list(CONTEXT_SECTION_KEYS), list(template.keys())
            )
        )
    if template.get("schema_version") != ARCHITECTURE_CONTEXT_SCHEMA_VERSION:
        problems.append(
            "template_schema_version_mismatch: expected {!r} got {!r}".format(
                ARCHITECTURE_CONTEXT_SCHEMA_VERSION,
                template.get("schema_version"),
            )
        )
    for key, section in template.items():
        if key == "schema_version":
            continue
        if not isinstance(section, dict):
            problems.append("template_section_not_an_object: {!r}".format(key))
            continue
        for field, value in section.items():
            if isinstance(value, str):
                continue
            if isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                continue
            problems.append(
                "template_field_has_unexpected_type: {}.{} -> {!r}".format(
                    key, field, type(value).__name__
                )
            )
    return problems


def validate_context_template_structure(template: Any) -> List[str]:
    """Return problems when a template does not match the fixed contract."""
    return _template_contract_problems(template)


def load_context_template(template_path: Optional[Any] = None) -> Dict[str, Any]:
    """Load the repo-level architecture context template.

    Raises ArchitectureContextError when the template is missing, unreadable,
    or does not match the fixed contract identity.
    """
    path = Path(template_path) if template_path else CONTEXT_TEMPLATE_PATH
    if not path.is_file():
        raise ArchitectureContextError(
            "Architecture context template not found: {}".format(path)
        )
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArchitectureContextError(
            "Unable to read architecture context template {}: {}".format(
                path, exc
            )
        )
    problems = validate_context_template_structure(template)
    if problems:
        raise ArchitectureContextError(
            "Invalid architecture context template {}:\n  - ".format(path)
            + "\n  - ".join(problems)
        )
    return template


def default_context_markdown_path(
    output_root: Optional[Path] = None, snapshot_id: Optional[str] = None
) -> Path:
    """Markdown path: <output>/<snapshot_id>/architecture_context/architecture_context.md."""
    return _context_dir(output_root, snapshot_id) / ARCHITECTURE_CONTEXT_FILENAME


def default_context_path(
    output_root: Optional[Path] = None, snapshot_id: Optional[str] = None
) -> Path:
    """Alias for the generated architecture_context.md path."""
    return default_context_markdown_path(output_root, snapshot_id)


def discover_snapshot_id(output_root: Optional[Path] = None) -> str:
    """Run id of the single run under ``<output>`` usable as context input.

    A usable run has both M2 facts and M4 architecture artifacts::

        <output>/<snapshot_id>/facts/repository_facts.json
        <output>/<snapshot_id>/architecture/architecture_model.json

    The run id always comes from the existing artifact/run context — never
    from a repository-specific constant.
    """
    root = resolve_output_root(output_root)
    candidates: List[str] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "facts" / "repository_facts.json").is_file() and (
                child / "architecture" / "architecture_model.json"
            ).is_file():
                candidates.append(child.name)
    if not candidates:
        raise ArchitectureContextError(
            "No architecture context run found under {}: expected "
            "<output>/<snapshot_id>/{{facts,architecture}} artifacts. "
            "Supply snapshot_id (or explicit input artifact paths).".format(
                root
            )
        )
    if len(candidates) > 1:
        raise ArchitectureContextError(
            "Multiple architecture context runs found under {}: {}. "
            "Supply snapshot_id (or explicit input artifact paths).".format(
                root, ", ".join(candidates)
            )
        )
    return candidates[0]


def _context_dir(output_root: Optional[Path], snapshot_id: Optional[str]) -> Path:
    root = resolve_output_root(output_root)
    if snapshot_id is None:
        snapshot_id = discover_snapshot_id(root)
    return architecture_context_dir(root, snapshot_id)


__all__ = [
    "ARCHITECTURE_CONTEXT_FILENAME",
    "ARCHITECTURE_CONTEXT_SCHEMA_VERSION",
    "ArchitectureContextError",
    "ArchitectureContextValidationError",
    "COMMIT_LABEL",
    "CONTRACT_LABEL",
    "CONTEXT_SECTION_KEYS",
    "CONTEXT_TEMPLATE_PATH",
    "EMPTY_LIST_MARKER",
    "EMPTY_VALUE_MARKER",
    "GENERATOR_NAME",
    "KEY_DOCUMENT_LIMIT",
    "KEY_FILE_LIMIT",
    "SECTION_HEADINGS",
    "SNAPSHOT_LABEL",
    "SPECIAL_LIST_FIELDS",
    "TITLE",
    "UNRESOLVED_SAMPLE_LIMIT",
    "_DEFAULT_SOURCE_PATHS",
    "default_context_markdown_path",
    "default_context_path",
    "discover_snapshot_id",
    "display_path",
    "empty_list_line",
    "field_label",
    "format_line",
    "format_source_line",
    "limitation_mappings_line",
    "load_context_template",
    "readme_first_key",
    "source_identity",
    "source_label",
    "unknown_test_files_line",
    "unknown_unresolved_line",
    "validate_context_template_structure",
]
