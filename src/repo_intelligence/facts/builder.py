"""Compose RepositoryFacts (repository_facts.v2) from deterministic analyzers.

Pipeline:

    M1 RepositorySnapshot
        → reuse the persisted pinned checkout (artifacts/<id>/repository/)
          written by M1, verified against snapshot.commit_sha
        → run analyzers
        → attach EvidenceRef provenance
        → normalize analyzer output into generic FactRecords
        → RepositoryFacts
        → schema validation
        → artifacts/<snapshot_id>/facts/repository_facts.json
        → artifacts/<snapshot_id>/snapshot/ + schemas/ copies
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

from .. import __version__
from ..artifacts import (
    ARTIFACTS_ROOT,
    copy_schema_artifacts,
    copy_snapshot_artifact,
    default_artifacts_root,
    facts_path as artifact_facts_path,
    repository_dir_for_snapshot,
    resolve_output_root,
)
from ..repository import git as git_adapter
from ..snapshot.models import RepositorySnapshot
from .agents import analyze_agents
from .configuration import analyze_configuration
from .dependencies import analyze_dependencies
from .documentation import analyze_documentation
from .evidence import (
    attach_evidence_to_record,
    validate_evidence_ref,
)
from .filesystem import analyze_filesystem
from .interactions import analyze_interactions, build_interaction_facts
from .mcp import analyze_mcp
from .models import (
    AnalyzersInfo,
    RepositoryFacts,
    SCHEMA_VERSION,
    SnapshotRef,
    sort_facts,
)
from .normalize import (
    build_entities,
    validate_cross_representation,
    validate_entity_identity,
)
from .python import analyze_python
from .records import build_fact_records
from .tests import analyze_tests
from .tooling import analyze_tooling, build_tooling_facts

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "repository_facts.v2.json"
)
# Absolute default: <project-root>/artifacts (not CWD-relative).
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT


class FactsError(Exception):
    """Raised when facts cannot be built or the snapshot is unusable."""


class FactsSchemaValidationError(FactsError):
    """Raised when facts fail schema or invariant checks."""


def load_snapshot_file(snapshot_path: Path) -> RepositorySnapshot:
    """Load and minimally validate an M1 RepositorySnapshot artifact."""
    path = Path(snapshot_path)
    if not path.is_file():
        raise FactsError("M1 snapshot not found: {}".format(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactsError("Cannot read M1 snapshot {}: {}".format(path, exc))

    if not isinstance(data, dict):
        raise FactsError("M1 snapshot must be a JSON object: {}".format(path))
    if data.get("schema_version") != "repository_snapshot.v1":
        raise FactsError(
            "Expected schema_version 'repository_snapshot.v1', got {!r}".format(
                data.get("schema_version")
            )
        )
    commit = (data.get("snapshot") or {}).get("commit_sha")
    if not commit or len(commit) != 40:
        raise FactsError(
            "M1 snapshot is missing a full commit_sha (provenance required)"
        )
    url = (data.get("repository") or {}).get("url")
    if not url:
        raise FactsError("M1 snapshot is missing repository.url")
    return RepositorySnapshot.from_dict(data)


def _snapshot_id_from_path(snapshot_path: Path, snapshot: RepositorySnapshot) -> str:
    """Prefer the M1 artifact directory name when it looks like a snapshot id.

    Supports both layouts:
      - artifacts/<snapshot_id>/snapshot/repository_snapshot.json
      - snapshots/<snapshot_id>/repository_snapshot.json
    """
    parent = Path(snapshot_path).resolve().parent
    if parent.name == "snapshot" and parent.parent.name not in ("", "artifacts", "snapshots"):
        return parent.parent.name
    name = parent.name
    # e.g. main_2026-09-22_ef11dac
    if "_" in name and name not in ("snapshots", "snapshot", "artifacts"):
        return name
    return snapshot.snapshot_id


def run_analyzers(root: Path) -> Dict[str, Any]:
    """Run all deterministic analyzers against a workspace root."""
    return {
        "repository": analyze_filesystem(root),
        "python": analyze_python(root),
        "configuration": analyze_configuration(root),
        "dependencies": analyze_dependencies(root),
        "agents": analyze_agents(root),
        "mcp": analyze_mcp(root),
        "documentation": analyze_documentation(root),
        "tests": analyze_tests(root),
        "interactions": analyze_interactions(root),
        "tooling": analyze_tooling(root),
    }


def _attach_evidence(
    analyzer_results: Dict[str, Any],
    commit: str,
) -> Dict[str, Any]:
    """Attach EvidenceRef to meaningful facts (where the fact came from).

    Does not invent line numbers for file-level-only facts.
    """
    python = dict(analyzer_results.get("python") or {})
    for key, artifact_type in (
        ("modules", "python"),
        ("classes", "python"),
        ("functions", "python"),
        ("imports", "python"),
    ):
        records = []
        for record in python.get(key) or []:
            item = dict(record)
            path = item.get("path")
            if not path:
                records.append(item)
                continue
            # Modules: file-level only (no forced line).
            if key == "modules":
                attach_evidence_to_record(
                    item,
                    commit=commit,
                    artifact_type="python",
                    path=path,
                    line_start=None,
                    line_end=None,
                    symbol=item.get("name") or Path(path).stem,
                )
            else:
                attach_evidence_to_record(
                    item,
                    commit=commit,
                    artifact_type="python",
                    path=path,
                    line_start=item.get("line_start", item.get("line")),
                    line_end=item.get("line_end"),
                    symbol=item.get("symbol") or item.get("name") or item.get("module"),
                )
            records.append(item)
        python[key] = records

    def _file_level(items: Any, artifact_type: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for record in items or []:
            item = dict(record)
            path = item.get("path") or item.get("source")
            if path:
                attach_evidence_to_record(
                    item,
                    commit=commit,
                    artifact_type=artifact_type,
                    path=path,
                    line_start=None,
                    line_end=None,
                    symbol=None,
                )
            out.append(item)
        return out

    entities_out: Optional[List[Dict[str, Any]]] = None
    if analyzer_results.get("entities") is not None:
        entities_out = []
        for record in analyzer_results.get("entities") or []:
            item = dict(record)
            path = item.get("path")
            if path:
                attach_evidence_to_record(
                    item,
                    commit=commit,
                    artifact_type="python",
                    path=path,
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    symbol=item.get("name"),
                )
            entities_out.append(item)

    out: Dict[str, Any] = {
        "repository": analyzer_results.get("repository") or {},
        "python": python,
        "configuration": _file_level(
            analyzer_results.get("configuration"), "configuration"
        ),
        "dependencies": _file_level(
            analyzer_results.get("dependencies"), "file"
        ),
        "agents": _file_level(
            analyzer_results.get("agents"), "agent_definition"
        ),
        "mcp": _file_level(analyzer_results.get("mcp"), "mcp_definition"),
        "documentation": _file_level(
            analyzer_results.get("documentation"), "documentation"
        ),
        "tests": _file_level(analyzer_results.get("tests"), "test"),
    }
    interactions = analyzer_results.get("interactions")
    if isinstance(interactions, dict):
        out["interactions"] = interactions
    # Tooling statements are already evidence-bearing (no line inference).
    tooling = analyzer_results.get("tooling")
    if isinstance(tooling, dict):
        out["tooling"] = tooling
    if entities_out is not None:
        out["entities"] = entities_out
    return out


def compose_facts(
    snapshot: RepositorySnapshot,
    analyzer_results: Dict[str, Any],
    snapshot_id: str,
) -> RepositoryFacts:
    """Combine analyzer outputs into a RepositoryFacts domain object (v2)."""
    commit = snapshot.snapshot.commit_sha
    results = _attach_evidence(analyzer_results, commit)
    python = results.get("python") or {}

    # Language-neutral entity view for stable downstream identity (M2.1).
    entities = results.get("entities")
    if entities is None:
        entities = build_entities(python)
        for item in entities:
            path = item.get("path")
            if path:
                attach_evidence_to_record(
                    item,
                    commit=commit,
                    artifact_type="python",
                    path=path,
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    symbol=item.get("name"),
                )

    # Dual-representation consistency is an internal compose-time concern;
    # the published document only carries the unified facts[] envelope.
    problems = validate_entity_identity(entities)
    problems.extend(validate_cross_representation(entities, python))
    if problems:
        raise FactsSchemaValidationError(
            "Facts failed entity consistency checks:\n  - " + "\n  - ".join(problems)
        )

    fact_records = build_fact_records(results, entities=entities, commit=commit)
    # Post-pass: resolve explicit interaction statements to existing fact_ids.
    fact_records = fact_records + build_interaction_facts(
        fact_records,
        results.get("interactions") or {},
        commit=commit,
    )
    # Post-pass (M12): tool/server/configuration statements resolved against
    # the artifact facts that already exist (agent + MCP config identities).
    fact_records = fact_records + build_tooling_facts(
        fact_records,
        results.get("tooling") or {},
        commit=commit,
    )
    fact_records = sort_facts(fact_records)

    return RepositoryFacts(
        snapshot=SnapshotRef(
            commit_sha=snapshot.snapshot.commit_sha,
            snapshot_id=snapshot_id,
            repository_url=snapshot.repository.url,
        ),
        # Identity/metadata only — observations live in facts[].
        repository={
            "name": snapshot.repository.name,
            "url": snapshot.repository.url,
        },
        facts=fact_records,
        analyzers=[
            AnalyzersInfo(name="repository-intelligence", version=__version__)
        ],
    )


def load_facts_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(schema_path) if schema_path else SCHEMA_PATH
    if not path.is_file():
        raise FactsError("Schema not found: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def validate_facts_document(
    document: Dict[str, Any], schema_path: Optional[Path] = None
) -> None:
    schema = load_facts_schema(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    if errors:
        messages = [
            "{}: {}".format(
                "/".join(str(p) for p in error.path) or "<root>",
                error.message,
            )
            for error in errors
        ]
        raise FactsSchemaValidationError(
            "Facts failed schema validation:\n  - " + "\n  - ".join(messages)
        )


def validate_fact_envelope(facts: List[Dict[str, Any]]) -> List[str]:
    """Envelope rules for each FactRecord (beyond JSON Schema)."""
    problems: List[str] = []
    seen_ids = set()
    for index, fact in enumerate(facts):
        label = "facts[{}]".format(index)
        if not isinstance(fact, dict):
            problems.append("{}: must be an object".format(label))
            continue
        fact_id = fact.get("fact_id")
        fact_type = fact.get("fact_type")
        kind = fact.get("kind")
        source = fact.get("source")
        attributes = fact.get("attributes")
        evidence = fact.get("evidence")

        if not fact_id or not isinstance(fact_id, str):
            problems.append("{}: missing fact_id".format(label))
        elif not fact_id.startswith("fact:"):
            problems.append("{}: fact_id must start with 'fact:'".format(label))
        elif fact_id in seen_ids:
            problems.append("{}: duplicate fact_id {!r}".format(label, fact_id))
        else:
            seen_ids.add(fact_id)

        if not fact_type or not isinstance(fact_type, str):
            problems.append("{}: missing fact_type".format(label))
        elif not isinstance(kind, str):
            problems.append("{}: kind must be a string".format(label))
        if not source or not isinstance(source, str):
            problems.append("{}: missing source".format(label))
        if not isinstance(attributes, dict):
            problems.append("{}: attributes must be an object".format(label))
        if not isinstance(evidence, list):
            problems.append("{}: evidence must be an array".format(label))
        elif evidence:
            if len(evidence) != 1 or not isinstance(evidence[0], dict):
                # Multi-evidence is allowed by schema; validate each ref below.
                pass
    return problems


def validate_fact_evidence(
    facts: List[Dict[str, Any]], snapshot_commit: str
) -> List[str]:
    """Evidence integrity on every EvidenceRef inside facts[]."""
    problems: List[str] = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("fact_id") or "facts[{}]".format(index)
        evidence = fact.get("evidence")
        if not isinstance(evidence, list):
            continue
        # Path-backed facts must carry at least one evidence pointer.
        attributes = fact.get("attributes") or {}
        path = attributes.get("path") or attributes.get("source")
        if path and not evidence:
            problems.append("{}: missing evidence".format(fact_id))
            continue
        for e_index, ref in enumerate(evidence):
            label = "{}.evidence[{}]".format(fact_id, e_index)
            if not isinstance(ref, dict):
                problems.append("{}: must be an object".format(label))
                continue
            problems.extend(
                validate_evidence_ref(ref, snapshot_commit, label=label)
            )
    return problems


def validate_facts_invariants(facts: RepositoryFacts) -> None:
    """Business rules beyond the JSON Schema."""
    problems = []
    commit = facts.snapshot.commit_sha
    if not commit or len(commit) != 40:
        problems.append("snapshot.commit_sha must be a full 40-char SHA")
    if not facts.snapshot.snapshot_id:
        problems.append("snapshot.snapshot_id is missing")
    if not facts.snapshot.repository_url:
        problems.append("snapshot.repository_url is missing")
    if facts.schema_version != SCHEMA_VERSION:
        problems.append(
            "schema_version must be {}, got {!r}".format(
                SCHEMA_VERSION, facts.schema_version
            )
        )
    # repository = identity/metadata only (single-source observation rule)
    if not facts.repository.get("name"):
        problems.append("repository.name is missing")
    if not facts.repository.get("url"):
        problems.append("repository.url is missing")
    for banned in ("files", "directories"):
        if banned in facts.repository:
            problems.append(
                "repository.{} is not allowed; observations belong in facts[]".format(
                    banned
                )
            )
    if not facts.analyzers:
        problems.append("analyzers must be a non-empty array")
    else:
        for index, analyzer in enumerate(facts.analyzers):
            if not analyzer.version:
                problems.append("analyzers[{}].version is missing".format(index))
            if not analyzer.name:
                problems.append("analyzers[{}].name is missing".format(index))

    # provenance: python symbol facts must carry path + line where applicable
    for fact in facts.facts:
        if fact.get("fact_type") != "entity" or fact.get("kind") != "function":
            continue
        attributes = fact.get("attributes") or {}
        if "path" not in attributes or "line" not in attributes:
            problems.append("python function facts must include path and line")
            break

    # stable language-neutral identity for graph construction (M2.1)
    entity_records = []
    for fact in facts.facts:
        if fact.get("fact_type") != "entity":
            continue
        attributes = dict(fact.get("attributes") or {})
        attributes.setdefault("entity_kind", fact.get("kind"))
        entity_records.append(attributes)
    problems.extend(validate_entity_identity(entity_records))

    # snapshot provenance must remain bound to M1 commit (full SHA only)
    if len(facts.snapshot.commit_sha) != 40:
        problems.append("snapshot.commit_sha must remain a full SHA")
    if facts.snapshot.commit_sha != facts.snapshot.commit_sha.lower():
        problems.append("snapshot.commit_sha must be lowercase hex")

    # Generic envelope integrity: unique stable fact_ids, open types
    problems.extend(validate_fact_envelope(facts.facts))

    # Evidence integrity (M2.2/v2): every EvidenceRef commit/path/line rules
    problems.extend(validate_fact_evidence(facts.facts, commit))

    # Root must not reintroduce analyzer-specific top-level sections.
    allowed_root = {"schema_version", "snapshot", "repository", "facts", "analyzers"}
    for key in facts.to_dict():
        if key not in allowed_root:
            problems.append(
                "root section {!r} is not allowed under repository_facts.v2".format(
                    key
                )
            )
    # repository must stay metadata-only (no inventory arrays)
    for key in facts.repository:
        if key not in ("name", "url"):
            problems.append(
                "repository.{!r} is not allowed; use facts[]".format(key)
            )

    if problems:
        raise FactsSchemaValidationError(
            "Facts failed invariant checks:\n  - " + "\n  - ".join(problems)
        )


def write_facts_artifact(
    facts: RepositoryFacts,
    output_root: Optional[Path] = None,
    snapshot_source: Optional[Path] = None,
) -> Path:
    """Write facts + snapshot copy + schema copies under artifacts/<id>/.

    When *output_root* is None, uses ``<project-root>/artifacts`` (not CWD).
    Explicit *output_root* is used as given (CLI/API override).
    """
    out_root = resolve_output_root(output_root)
    snapshot_id = facts.snapshot.snapshot_id
    artifact = artifact_facts_path(out_root, snapshot_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w", encoding="utf-8") as fh:
        json.dump(facts.to_dict(), fh, indent=2, sort_keys=False)
        fh.write("\n")
    copy_schema_artifacts(out_root, snapshot_id)
    if snapshot_source is not None and Path(snapshot_source).is_file():
        copy_snapshot_artifact(Path(snapshot_source), out_root, snapshot_id)
    return artifact


def _resolve_pinned_repository(
    snapshot: RepositorySnapshot,
    snapshot_artifact: Path,
) -> Path:
    """Return the persisted pinned checkout for *snapshot* (never re-clone).

    M1 persists the exact checkout beside the snapshot artifact at
    ``<run>/repository/``. The directory must exist and its git HEAD must
    equal ``snapshot.commit_sha``; otherwise fail clearly instead of
    silently cloning another copy.
    """
    git_adapter.ensure_git_available()
    pinned = repository_dir_for_snapshot(snapshot_artifact)
    if not pinned.is_dir():
        raise FactsError(
            "Persisted repository snapshot not found at {}. Re-run M1 "
            "create_snapshot for {} to persist the exact checkout; "
            "M2 never clones another copy.".format(
                pinned, snapshot.snapshot_id
            )
        )
    expected = snapshot.snapshot.commit_sha
    actual = git_adapter.get_head_sha(pinned)
    if actual != expected:
        raise FactsError(
            "Persisted repository HEAD {} does not match snapshot commit "
            "{} ({}); the checkout at {} is stale or corrupted. Re-run M1 "
            "create_snapshot; M2 never clones another copy.".format(
                actual, expected, snapshot.snapshot_id, pinned
            )
        )
    return pinned


def create_facts(
    snapshot_path: Path,
    output_root: Optional[Path] = None,
) -> Tuple[RepositoryFacts, Path]:
    """Full M2 pipeline from an M1 snapshot artifact to a facts artifact.

    Reuses the pinned checkout persisted by M1 at
    ``<run>/repository/`` (verified against ``snapshot.commit_sha``)
    instead of cloning again. *output_root* None → ``<project-root>/artifacts``;
    otherwise the explicit override is used exactly as given.
    """
    out_root = resolve_output_root(output_root)
    snapshot = load_snapshot_file(snapshot_path)
    snapshot_id = _snapshot_id_from_path(Path(snapshot_path), snapshot)

    root = _resolve_pinned_repository(snapshot, Path(snapshot_path))
    results = run_analyzers(root)

    facts = compose_facts(snapshot, results, snapshot_id)
    validate_facts_invariants(facts)
    validate_facts_document(facts.to_dict())
    artifact = write_facts_artifact(
        facts, out_root, snapshot_source=Path(snapshot_path)
    )
    return facts, artifact
