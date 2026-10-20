"""Command-line interface for repository-intelligence (M1–M5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .architecture.builder import create_architecture_model
from .architecture.models import architecture_counts
from .architecture.validator import (
    ArchitectureModelError,
    ArchitectureModelValidationError,
)
from .artifacts import default_artifacts_root, repository_dir
from .engineering.builder import (
    EngineeringModelError,
    EngineeringModelValidationError,
    create_engineering_model,
)
from .context.builder import (
    ArchitectureContextError,
    create_architecture_context,
)
from .context.models import CONTEXT_TEMPLATE_PATH, display_path
from .context.validator import ArchitectureContextValidationError
from .facts.builder import (
    FactsError,
    FactsSchemaValidationError,
    create_facts,
)
from .graph.builder import (
    DiagramError,
    DiagramValidationError,
    create_diagram_artifacts,
)
from .repository.git import (
    CloneError,
    GitError,
    GitNotFoundError,
    RefResolutionError,
)
from .snapshot.builder import (
    SchemaValidationError,
    SnapshotError,
    create_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repo-intelligence",
        description="Repository Intelligence and Architecture Investigator "
        "(M1 snapshot, M2 facts)",
    )
    subparsers = parser.add_subparsers(dest="command")

    snap = subparsers.add_parser(
        "snapshot",
        help="M1: create a repository_snapshot.v1 artifact and persist "
        "the exact checkout under <output>/<snapshot_id>/repository/"
        "<name>/",
    )
    snap.add_argument(
        "--repo",
        required=True,
        help="Repository URL (or local path) to clone",
    )
    snap.add_argument(
        "--ref",
        required=True,
        help="Git ref to resolve: tag, branch, or commit SHA",
    )
    snap.add_argument(
        "--output",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts "
        "(not CWD). Writes <output>/<snapshot_id>/snapshot/repository_snapshot.json",
    )
    snap.add_argument(
        "--on-dirty",
        choices=("warn", "fail", "ignore"),
        default="warn",
        help="Behavior when the working tree is not clean "
        "(default: warn — record working_tree_clean=false and continue)",
    )

    facts = subparsers.add_parser(
        "facts",
        help="M2: create repository_facts.v2 from an M1 snapshot "
        "(reuses the persisted pinned checkout; never re-clones)",
    )
    facts.add_argument(
        "--snapshot",
        required=True,
        help="Path to an M1 repository_snapshot.json artifact",
    )
    facts.add_argument(
        "--output",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts "
        "(not CWD). Writes <output>/<snapshot_id>/facts/repository_facts.json",
    )

    model = subparsers.add_parser(
        "model",
        help="M3: build EngineeringModel from M2 facts + construction rules",
    )
    model.add_argument(
        "--facts",
        default=None,
        help="Path to M2 repository_facts.json "
        "(default: <output>/main_2026-09-22_ef11dac/facts/repository_facts.json)",
    )
    model.add_argument(
        "--rules",
        default=None,
        help="Path to engineering_construction_rules.v1.json "
        "(default: artifacts under the facts snapshot)",
    )
    model.add_argument(
        "--output",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts. "
        "Writes <output>/<snapshot_id>/engineering/engineering_model.json",
    )

    arch = subparsers.add_parser(
        "architecture",
        help="M3.5: build ArchitectureModel section from EngineeringModel",
    )
    arch.add_argument(
        "--engineering-model",
        default=None,
        help="Path to EngineeringModel JSON "
        "(default: <output>/main_2026-09-22_ef11dac/engineering/engineering_model.json)",
    )
    arch.add_argument(
        "--rules",
        default=None,
        help="Path to architecture_construction_rules.v1.json",
    )
    arch.add_argument(
        "--output",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts. "
        "Writes <output>/<snapshot_id>/architecture/architecture_model.json",
    )

    diagram = subparsers.add_parser(
        "diagram",
        help="M5: export NetworkX architecture graph to Mermaid/DOT "
        "(consumes existing M4 ArchitectureModel)",
    )
    diagram.add_argument(
        "--architecture-model",
        default=None,
        help="Path to M4 architecture_model.json "
        "(default: <output>/main_2026-09-22_ef11dac/architecture/"
        "architecture_model.json)",
    )
    diagram.add_argument(
        "--output",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts. "
        "Writes <output>/<snapshot_id>/diagram/{architecture.mmd,"
        "architecture.dot,validation.json}",
    )
    diagram.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip optional Graphviz SVG rendering even if dot is installed",
    )

    context = subparsers.add_parser(
        "context",
        help="Architecture context: write "
        "<output>/<snapshot_id>/architecture_context/"
        "architecture_context.md "
        "(architecture_context.v1, consumes existing M2/M4 artifacts + "
        "contract/architecture_context_template.json)",
    )
    context.add_argument(
        "--architecture-model",
        default=None,
        help="Path to M4 architecture_model.json "
        "(default: <output>/main_2026-09-22_ef11dac/architecture/"
        "architecture_model.json)",
    )
    context.add_argument(
        "--facts",
        default=None,
        help="Path to M2 repository_facts.json "
        "(default: <output>/main_2026-09-22_ef11dac/facts/"
        "repository_facts.json)",
    )
    context.add_argument(
        "--output",
        default=None,
        help="Artifact root override. Default: <project-root>/artifacts. "
        "Writes <output>/<snapshot_id>/architecture_context/"
        "architecture_context.md",
    )
    context.add_argument(
        "--output-file",
        default=None,
        help="Markdown output file override. "
        "Default: <output>/<snapshot_id>/architecture_context/"
        "architecture_context.md",
    )

    ridx = subparsers.add_parser(
        "retrieval-index",
        help="M18: build the repository retrieval index (chunk + "
        "embed + FAISS + SQLite metadata) for a pinned checkout or "
        "a snapshot run",
    )
    ridx.add_argument(
        "--repository-root",
        default=None,
        help="local checkout to index (identity comes from the "
        "snapshot record beside it)",
    )
    ridx.add_argument(
        "--snapshot-id",
        default=None,
        help="index this run's persisted checkout instead (uses "
        "--output-root to find it)",
    )
    ridx.add_argument(
        "--output-root",
        default=None,
        help="artifact output root (default: <project>/artifacts)",
    )
    ridx.add_argument(
        "--store",
        default=None,
        help="rag store root (default: <db_store>/rag)",
    )
    ridx.add_argument(
        "--model",
        default=None,
        help="override the embedding identity (default: the "
        "project's sentence-transformers model)",
    )
    ridx.add_argument(
        "--chunk-lines",
        type=int,
        default=None,
        help="target lines per chunk",
    )
    ridx.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="overlapping lines between chunks",
    )
    ridx.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when a complete index exists",
    )
    return parser


def _cmd_snapshot(args: argparse.Namespace) -> int:
    output_root = (
        Path(args.output) if args.output else default_artifacts_root()
    )
    snapshot, artifact_path = create_snapshot(
        repository_url=args.repo,
        ref=args.ref,
        output_root=output_root,
        on_dirty=args.on_dirty,
    )

    print("Snapshot created")
    print()
    print("Repository : {}".format(snapshot.repository.name))
    print("Ref        : {}".format(snapshot.snapshot.ref))
    print("Commit     : {}".format(snapshot.snapshot.commit_sha))
    print(
        "Clean      : {}".format(
            str(snapshot.workspace.working_tree_clean).lower()
        )
    )
    print()
    print("Output:")
    print(artifact_path)
    print()
    print("Pinned repository:")
    print(repository_dir(output_root, snapshot.snapshot_id))
    return 0


def _cmd_facts(args: argparse.Namespace) -> int:
    output_root = (
        Path(args.output) if args.output else default_artifacts_root()
    )
    facts, artifact_path = create_facts(
        snapshot_path=Path(args.snapshot),
        output_root=output_root,
    )
    counts = facts.counts()

    print("Repository facts created")
    print()
    print("Snapshot : {}".format(facts.snapshot.snapshot_id))
    print("Commit   : {}".format(facts.snapshot.commit_sha))
    print()
    print("Facts       : {}".format(counts["facts"]))
    print("Files       : {}".format(counts["files"]))
    print("Directories : {}".format(counts["directories"]))
    print("Entities    : {}".format(counts["entities"]))
    print("Python      : {}".format(counts["python_modules"]))
    print("Agents      : {}".format(counts["agents"]))
    print("MCP         : {}".format(counts["mcp"]))
    print("Tests       : {}".format(counts["tests"]))
    print("Evidence    : {}".format(counts.get("evidence", 0)))
    print()
    print("Output:")
    print(artifact_path)
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    output_root = (
        Path(args.output) if args.output else default_artifacts_root()
    )
    facts_path = Path(args.facts) if args.facts else None
    rules_path = Path(args.rules) if args.rules else None
    model, artifact_path = create_engineering_model(
        facts_path=facts_path,
        rules_path=rules_path,
        output_root=output_root,
    )
    counts = model.counts()

    print("EngineeringModel created")
    print()
    print("Repository  : {}".format(model.repository.get("name")))
    print("Commit      : {}".format(model.repository.get("commit_sha")))
    print("Strategy    : {}".format(model.analysis.get("strategy")))
    print("Analyzers   : {}".format(len(model.analysis.get("analyzers") or [])))
    print()
    print("Nodes         : {}".format(counts["nodes"]))
    print("  artifact    : {}".format(counts["nodes_artifact"]))
    print("  code_entity : {}".format(counts["nodes_code_entity"]))
    print("  external    : {}".format(counts["nodes_external_package"]))
    print("  tool        : {}".format(counts.get("nodes_tool", 0)))
    print("  mcp_server  : {}".format(counts.get("nodes_mcp_server", 0)))
    print("Relationships : {}".format(counts["relationships"]))
    print("  CONTAINS    : {}".format(counts["rel_contains"]))
    print("  IMPORTS     : {}".format(counts["rel_imports"]))
    print("  DEPENDS_ON  : {}".format(counts["rel_depends_on"]))
    print("  TRIGGERS    : {}".format(counts.get("rel_triggers", 0)))
    print("  DECLARES    : {}".format(counts.get("rel_declares", 0)))
    print("  EXPOSED_BY  : {}".format(counts.get("rel_exposed_by", 0)))
    print("  CONFIGURED_BY: {}".format(counts.get("rel_configured_by", 0)))
    print("Unresolved    : {}".format(counts["unresolved"]))
    print("Evidence      : {}".format(counts["evidence"]))
    print()
    print("Output:")
    print(artifact_path)
    return 0


def _cmd_architecture(args: argparse.Namespace) -> int:
    output_root = (
        Path(args.output) if args.output else default_artifacts_root()
    )
    eng_path = (
        Path(args.engineering_model) if args.engineering_model else None
    )
    rules_path = Path(args.rules) if args.rules else None
    model, artifact_path = create_architecture_model(
        engineering_model_path=eng_path,
        rules_path=rules_path,
        output_root=output_root,
    )
    counts = architecture_counts(model.architecture, model.mappings)

    print("ArchitectureModel created")
    print()
    print("Repository : {}".format(model.repository.get("name")))
    print("Commit     : {}".format(model.repository.get("commit_sha")))
    print()
    print("Architecture nodes      : {}".format(counts["architecture_nodes"]))
    print("  concept/agent         : {}".format(
        counts["architecture_nodes_concept_agent"]
    ))
    print("Architecture relations  : {}".format(
        counts["architecture_relationships"]
    ))
    print("  TRIGGERS              : {}".format(
        counts.get("architecture_rel_triggers", 0)
    ))
    print("Mappings                : {}".format(counts["mappings"]))
    print("  CONTRIBUTES_TO        : {}".format(
        counts["mappings_contributes_to"]
    ))
    print("Architecture unresolved : {}".format(
        counts["architecture_unresolved"]
    ))
    print("  TRIGGERS              : {}".format(
        counts.get("architecture_unresolved_triggers", 0)
    ))
    print("Architecture evidence   : {}".format(
        counts["architecture_evidence"]
    ))
    print()
    sample = [n["id"] for n in (model.architecture.get("nodes") or [])[:5]]
    if sample:
        print("Sample architecture ids:")
        for item in sample:
            print("  {}".format(item))
        print()
    print("Output:")
    print(artifact_path)
    return 0


def _cmd_diagram(args: argparse.Namespace) -> int:
    output_root = (
        Path(args.output) if args.output else default_artifacts_root()
    )
    model_path = (
        Path(args.architecture_model) if args.architecture_model else None
    )
    projection, paths = create_diagram_artifacts(
        architecture_model_path=model_path,
        output_root=output_root,
        render_svg_output=not args.no_svg,
    )
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    counts = validation.get("counts") or {}

    print("Diagram artifacts created")
    print()
    print("Validation           : {}".format(validation.get("status")))
    print("Architecture nodes   : {}".format(counts.get("architecture_nodes")))
    print("Diagram nodes        : {}".format(counts.get("diagram_nodes")))
    print(
        "Architecture rels    : {}".format(
            counts.get("architecture_relationships")
        )
    )
    print(
        "Diagram relationships: {}".format(
            counts.get("diagram_relationships")
        )
    )
    print(
        "Unresolved (not exported): {}".format(
            counts.get("architecture_unresolved")
        )
    )
    print()
    print("Output:")
    for key in ("mermaid", "graphviz", "svg", "validation"):
        if key in paths:
            print("  {}: {}".format(key, paths[key]))
    return 0 if validation.get("status") == "PASS" else 6


def _cmd_context(args: argparse.Namespace) -> int:
    output_root = (
        Path(args.output) if args.output else default_artifacts_root()
    )
    arch_path = (
        Path(args.architecture_model) if args.architecture_model else None
    )
    facts_path = Path(args.facts) if args.facts else None
    output_file = Path(args.output_file) if args.output_file else None
    result = create_architecture_context(
        architecture_model_path=arch_path,
        facts_path=facts_path,
        output_path=output_file,
        output_root=output_root,
    )
    markdown = result.markdown
    section_count = markdown.count("\n## ")
    print("Architecture context created")
    print()
    print("Contract : architecture_context.v1")
    print("Template : {}".format(display_path(CONTEXT_TEMPLATE_PATH)))
    print("Sections : {}".format(section_count))
    print("Bytes    : {}".format(len(markdown.encode("utf-8"))))
    print()
    print("Output:")
    print(str(result.markdown_path))
    return 0


def _cmd_retrieval_index(args: argparse.Namespace) -> int:
    """M18: delegate to the retrieval index builder's own CLI."""
    from .retrieval.index import main as retrieval_index_main

    argv: List[str] = []
    for flag, value in (
        ("--repository-root", args.repository_root),
        ("--snapshot-id", args.snapshot_id),
        ("--output-root", args.output_root),
        ("--store", args.store),
        ("--model", args.model),
        ("--chunk-lines", args.chunk_lines),
        ("--chunk-overlap", args.chunk_overlap),
    ):
        if value is None:
            continue
        argv.extend([flag, str(value)])
    if args.force:
        argv.append("--force")
    return retrieval_index_main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    try:
        if args.command == "snapshot":
            return _cmd_snapshot(args)
        if args.command == "facts":
            return _cmd_facts(args)
        if args.command == "model":
            return _cmd_model(args)
        if args.command == "architecture":
            return _cmd_architecture(args)
        if args.command == "diagram":
            return _cmd_diagram(args)
        if args.command == "context":
            return _cmd_context(args)
        if args.command == "retrieval-index":
            return _cmd_retrieval_index(args)
        parser.print_help()
        return 2
    except GitNotFoundError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 3
    except CloneError as exc:
        print(
            "error: invalid or unreachable repository\n  {}".format(exc),
            file=sys.stderr,
        )
        return 4
    except RefResolutionError as exc:
        print("error: invalid ref\n  {}".format(exc), file=sys.stderr)
        return 5
    except (SchemaValidationError, FactsSchemaValidationError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 8
    except (
        EngineeringModelValidationError,
        ArchitectureModelValidationError,
        SnapshotError,
        FactsError,
        EngineeringModelError,
        ArchitectureModelError,
        DiagramError,
        DiagramValidationError,
        ArchitectureContextError,
        ArchitectureContextValidationError,
    ) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 6
    except GitError as exc:
        print(
            "error: git operation failed\n  {}".format(exc),
            file=sys.stderr,
        )
        return 7
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
