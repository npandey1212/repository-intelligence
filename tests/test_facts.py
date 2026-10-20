"""M2 contract tests: analyzers, aggregation, schema, provenance paths.

Uses small local fixtures (no GitHub). Integration path reuses M1
snapshot creation against a local Git repository when possible.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from repo_intelligence.facts import (
    ARTIFACT_TYPES,
    EvidenceRef,
    FactsError,
    FactsSchemaValidationError,
    RepositoryFacts,
    build_entities,
    compose_facts,
    compute_fact_id,
    create_facts,
    is_repository_relative_path,
    load_snapshot_file,
    make_evidence,
    make_fact,
    run_analyzers,
    validate_cross_representation,
    validate_entity_identity,
    validate_evidence_list,
    validate_evidence_ref,
    validate_facts_document,
    validate_facts_invariants,
)
from repo_intelligence.facts.agents import analyze_agents
from repo_intelligence.facts.configuration import analyze_configuration
from repo_intelligence.facts.dependencies import analyze_dependencies
from repo_intelligence.facts.documentation import analyze_documentation
from repo_intelligence.facts.filesystem import (
    analyze_filesystem,
    classify_path,
    iter_repository_paths,
)
from repo_intelligence.facts.mcp import analyze_mcp
from repo_intelligence.facts.python import analyze_python
from repo_intelligence.facts.tests import analyze_tests
from repo_intelligence.snapshot import create_snapshot


# ---------------------------------------------------------------------------
# Fixture tree
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "docs").mkdir()
    (root / ".claude" / "agents").mkdir(parents=True)
    (root / "mcp").mkdir()

    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "sample"\ndependencies = ["boto3>=1.0", "jsonschema"]\n',
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text(
        "requests>=2.0\n# comment\nflask\n", encoding="utf-8"
    )
    (root / ".env").write_text("API_KEY=super-secret\n", encoding="utf-8")
    (root / "config.yaml").write_text("debug: true\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    (root / ".claude" / "agents" / "iac-agent.md").write_text(
        "---\nname: iac-agent\n---\n", encoding="utf-8"
    )
    (root / "mcp" / "server.py").write_text(
        "def create_server():\n    return None\n", encoding="utf-8"
    )
    (root / "tests" / "test_sample.py").write_text(
        "import os\n\n"
        "class TestSample:\n"
        "    def test_ok(self):\n"
        "        assert True\n\n"
        "def test_helper():\n"
        "    assert 1 == 1\n",
        encoding="utf-8",
    )
    (root / "src" / "app" / "main.py").write_text(
        "import os\n"
        "from pathlib import Path\n\n"
        "class Engine:\n"
        "    def run(self):\n"
        "        pass\n\n"
        "def generate_pipeline():\n"
        "    return Path('.')\n",
        encoding="utf-8",
    )
    (root / "src" / "app" / "broken.py").write_text(
        "def oops(:\n", encoding="utf-8"
    )
    # Same simple name in two files — identity must not collide.
    (root / "src" / "foo.py").write_text(
        "def main():\n    return 'foo'\n", encoding="utf-8"
    )
    (root / "src" / "bar.py").write_text(
        "def main():\n    return 'bar'\n", encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# Filesystem analyzer
# ---------------------------------------------------------------------------

def test_filesystem_files_and_directories(sample_repo: Path):
    facts = analyze_filesystem(sample_repo)
    paths = {f["path"] for f in facts["files"]}
    dirs = {d["path"] for d in facts["directories"]}
    assert "README.md" in paths
    assert "src/app/main.py" in paths
    assert "docs" in dirs
    assert "src/app" in dirs
    # sorted deterministic order (analyzer returns lists, not sets)
    file_list = [f["path"] for f in facts["files"]]
    dir_list = [d["path"] for d in facts["directories"]]
    assert file_list == sorted(file_list)
    assert dir_list == sorted(dir_list)


def test_filesystem_file_types(sample_repo: Path):
    facts = analyze_filesystem(sample_repo)
    by_path = {f["path"]: f for f in facts["files"]}
    assert by_path["README.md"]["type"] == "markdown"
    assert by_path["pyproject.toml"]["type"] == "toml"
    assert by_path["src/app/main.py"]["type"] == "python"
    assert by_path["config.yaml"]["type"] == "yaml"


def test_classify_path_special_names():
    assert classify_path("Dockerfile") == "dockerfile"
    assert classify_path("Makefile") == "makefile"
    assert classify_path(".env") == "environment"
    assert classify_path("setup.cfg") == "cfg"


def test_iter_skips_git_and_venv(tmp_path: Path):
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    files, dirs = iter_repository_paths(tmp_path)
    assert files == ["keep.txt"]
    assert not any(p.startswith(".git") or p.startswith(".venv") for p in files + dirs)


# ---------------------------------------------------------------------------
# Python analyzer
# ---------------------------------------------------------------------------

def test_python_functions_classes_methods_imports(sample_repo: Path):
    facts = analyze_python(sample_repo)
    functions = {(f["path"], f["symbol"]) for f in facts["functions"]}
    classes = {(c["path"], c["symbol"]) for c in facts["classes"]}
    imports = {(i["path"], i["module"]) for i in facts["imports"]}

    assert ("src/app/main.py", "generate_pipeline") in functions
    assert ("mcp/server.py", "create_server") in functions
    assert ("src/app/main.py", "Engine") in classes
    assert ("src/app/main.py", "os") in imports
    assert ("src/app/main.py", "pathlib") in imports

    engine = next(c for c in facts["classes"] if c["symbol"] == "Engine")
    assert engine["methods"] == ["run"]
    assert engine["line"] >= 1


def test_python_syntax_error_handled(sample_repo: Path):
    facts = analyze_python(sample_repo)
    broken = next(m for m in facts["modules"] if m["path"] == "src/app/broken.py")
    assert broken["parse_error"] is True
    assert any(e["path"] == "src/app/broken.py" for e in facts["parse_errors"])


def test_python_symbols_have_line_provenance(sample_repo: Path):
    facts = analyze_python(sample_repo)
    main = (sample_repo / "src" / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main)
    expected_line = next(
        n.lineno for n in tree.body if getattr(n, "name", None) == "generate_pipeline"
    )
    fact = next(
        f
        for f in facts["functions"]
        if f["path"] == "src/app/main.py" and f["symbol"] == "generate_pipeline"
    )
    assert fact["line"] == expected_line


# ---------------------------------------------------------------------------
# Configuration analyzer
# ---------------------------------------------------------------------------

def test_configuration_detects_known_files(sample_repo: Path):
    results = analyze_configuration(sample_repo)
    paths = {r["path"] for r in results}
    assert "pyproject.toml" in paths
    assert "config.yaml" in paths
    assert ".env" in paths


def test_configuration_masks_secrets(sample_repo: Path):
    results = analyze_configuration(sample_repo)
    env = next(r for r in results if r["path"] == ".env")
    assert env["contains_sensitive_values"] is True
    blob = json.dumps(results)
    assert "super-secret" not in blob


def test_configuration_malformed_json(tmp_path: Path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    results = analyze_configuration(tmp_path)
    entry = next(r for r in results if r["path"] == "bad.json")
    assert entry["parse_status"] == "error"


# ---------------------------------------------------------------------------
# Dependency analyzer
# ---------------------------------------------------------------------------

def test_dependencies_requirements_and_pyproject(sample_repo: Path):
    results = analyze_dependencies(sample_repo)
    names = {r["name"] for r in results}
    assert "requests" in names
    assert "flask" in names
    assert "boto3" in names
    sources = {r["source"] for r in results}
    assert "requirements.txt" in sources
    assert "pyproject.toml" in sources


def test_dependencies_sorted(sample_repo: Path):
    results = analyze_dependencies(sample_repo)
    keys = [(r["source"], r.get("section", ""), r["name"]) for r in results]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Agent analyzer
# ---------------------------------------------------------------------------

def test_agents_detect_claude_agent_definition(sample_repo: Path):
    results = analyze_agents(sample_repo)
    paths = {r["path"] for r in results}
    assert ".claude/agents/iac-agent.md" in paths
    entry = next(r for r in results if r["path"] == ".claude/agents/iac-agent.md")
    assert entry["type"] == "agent_definition"
    # no role inference
    assert "infrastructure" not in json.dumps(results).lower()


def test_agents_filename_marker(tmp_path: Path):
    (tmp_path / "ops-agent.yaml").write_text("name: ops\n", encoding="utf-8")
    results = analyze_agents(tmp_path)
    assert any(r["path"] == "ops-agent.yaml" for r in results)


# ---------------------------------------------------------------------------
# MCP analyzer
# ---------------------------------------------------------------------------

def test_mcp_detects_server_under_mcp_dir(sample_repo: Path):
    results = analyze_mcp(sample_repo)
    paths = {r["path"] for r in results}
    assert "mcp/server.py" in paths
    entry = next(r for r in results if r["path"] == "mcp/server.py")
    assert entry["type"] == "mcp_server_source"


def test_mcp_detects_config_name(tmp_path: Path):
    (tmp_path / ".mcp.json").write_text("{}", encoding="utf-8")
    results = analyze_mcp(tmp_path)
    assert any(r["path"] == ".mcp.json" and r["type"] == "mcp_config" for r in results)


def test_mcp_does_not_claim_relationships(sample_repo: Path):
    results = analyze_mcp(sample_repo)
    assert "uses" not in json.dumps(results).lower()


# ---------------------------------------------------------------------------
# Documentation analyzer
# ---------------------------------------------------------------------------

def test_documentation_detects_readme_and_docs(sample_repo: Path):
    results = analyze_documentation(sample_repo)
    by_path = {r["path"]: r for r in results}
    assert "README.md" in by_path
    assert by_path["README.md"]["type"] == "readme"
    assert "docs/guide.md" in by_path


# ---------------------------------------------------------------------------
# Test analyzer
# ---------------------------------------------------------------------------

def test_tests_detect_files_and_symbols(sample_repo: Path):
    results = analyze_tests(sample_repo)
    by_path = {r["path"]: r for r in results}
    assert "tests" in by_path
    assert by_path["tests"]["type"] == "test_directory"
    entry = by_path["tests/test_sample.py"]
    assert entry["type"] == "test_file"
    assert "test_helper" in entry.get("functions", [])
    assert "TestSample" in entry.get("classes", [])


# ---------------------------------------------------------------------------
# Aggregation + schema
# ---------------------------------------------------------------------------

def test_run_analyzers_and_compose(sample_repo: Path):
    results = run_analyzers(sample_repo)
    assert set(results) >= {
        "repository",
        "python",
        "configuration",
        "dependencies",
        "agents",
        "mcp",
        "documentation",
        "tests",
    }

    class FakeSnapshot(object):
        pass

    # compose via RepositoryFacts path
    snap_repo_url = "https://example.com/x/y.git"
    commit = "e" * 40
    snapshot_id = "main_2026-09-22_eeeeeee"

    # build a minimal snapshot-like object
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url=snap_repo_url),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha=commit,
            commit_short_sha="e" * 7,
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(snapshot, results, snapshot_id)
    assert facts.snapshot.commit_sha == commit
    assert facts.snapshot.snapshot_id == snapshot_id
    assert facts.snapshot.repository_url == snap_repo_url
    validate_facts_invariants(facts)
    validate_facts_document(facts.to_dict())
    assert facts.analyzers
    assert facts.analyzers[0].version


def test_schema_rejects_bad_commit():
    from repo_intelligence.facts.models import AnalyzersInfo

    facts = RepositoryFacts()
    facts.snapshot.commit_sha = "nope"
    facts.snapshot.snapshot_id = "x"
    facts.snapshot.repository_url = "https://example.com/r.git"
    facts.analyzers = [AnalyzersInfo(version="0.1.0")]
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(facts.to_dict())


def test_schema_rejects_wrong_schema_version():
    from repo_intelligence.facts.models import AnalyzersInfo

    facts = RepositoryFacts()
    facts.snapshot.commit_sha = "a" * 40
    facts.snapshot.snapshot_id = "main_x_aaaaaaa"
    facts.snapshot.repository_url = "https://example.com/r.git"
    facts.analyzers = [AnalyzersInfo(version="0.1.0")]
    doc = facts.to_dict()
    doc["schema_version"] = "repository_facts.v0"
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(doc)


def test_schema_file_exists():
    from repo_intelligence.facts.builder import SCHEMA_PATH

    assert SCHEMA_PATH.is_file()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "repository_facts.v2"


# ---------------------------------------------------------------------------
# Snapshot loading / provenance requirements
# ---------------------------------------------------------------------------

def test_load_snapshot_missing_file(tmp_path: Path):
    with pytest.raises(FactsError):
        load_snapshot_file(tmp_path / "missing.json")


def test_load_snapshot_wrong_schema(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v0",
                "repository": {"url": "https://x/y.git"},
                "snapshot": {"commit_sha": "a" * 40},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FactsError):
        load_snapshot_file(p)


def test_load_snapshot_requires_commit(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {"url": "https://x/y.git"},
                "snapshot": {"commit_sha": "short"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FactsError):
        load_snapshot_file(p)


# ---------------------------------------------------------------------------
# Integration: M1 local snapshot → M2 facts
# ---------------------------------------------------------------------------

def test_integration_local_snapshot_to_facts(tmp_path: Path):
    # local origin repo
    origin = tmp_path / "origin"
    origin.mkdir()
    import subprocess

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "README.md").write_text("# hi\n", encoding="utf-8")
    (origin / "app.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    (origin / "tests").mkdir()
    (origin / "tests" / "test_app.py").write_text(
        "def test_hello():\n    assert True\n", encoding="utf-8"
    )
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)
    git(["tag", "v1.0.0"], origin)

    snap, snap_path = create_snapshot(
        repository_url=str(origin),
        ref="v1.0.0",
        output_root=tmp_path / "snapshots",
    )
    assert snap_path.is_file()

    facts, facts_path = create_facts(
        snapshot_path=snap_path,
        output_root=tmp_path / "facts",
    )

    # provenance: same commit as M1
    assert facts.snapshot.commit_sha == snap.snapshot.commit_sha
    assert facts_path.is_file()
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    assert data["snapshot"]["commit_sha"] == snap.snapshot.commit_sha
    assert data["schema_version"] == "repository_facts.v2"
    validate_facts_document(data)

    # repository = identity/metadata only
    assert set(data["repository"]) == {"name", "url"}
    assert data["repository"]["url"]
    assert "files" not in data["repository"]
    assert "directories" not in data["repository"]

    # file observations live only in facts[]
    file_paths = {
        f["attributes"]["path"]
        for f in data["facts"]
        if f["fact_type"] == "file"
    }
    assert "README.md" in file_paths
    assert "app.py" in file_paths
    symbols = {
        (f["attributes"]["path"], f["attributes"].get("symbol"))
        for f in data["facts"]
        if f["fact_type"] == "entity" and f["kind"] == "function"
    }
    assert ("app.py", "hello") in symbols

    # artifact layout: <root>/<snapshot_id>/facts/repository_facts.json
    assert facts_path.name == "repository_facts.json"
    assert facts_path.parent.name == "facts"
    assert facts_path.parent.parent.name == facts.snapshot.snapshot_id


def test_facts_output_only_json(tmp_path: Path):
    origin = tmp_path / "origin"
    origin.mkdir()
    import subprocess

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "a.txt").write_text("a", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)

    _, snap_path = create_snapshot(
        repository_url=str(origin),
        ref="main",
        output_root=tmp_path / "snapshots",
    )
    _, facts_path = create_facts(
        snapshot_path=snap_path,
        output_root=tmp_path / "facts",
    )
    entries = {p.name for p in facts_path.parent.iterdir()}
    assert entries == {"repository_facts.json"}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_fixture_twice(sample_repo: Path):
    a = run_analyzers(sample_repo)
    b = run_analyzers(sample_repo)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_determinism_full_pipeline_local(tmp_path: Path):
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "lib.py").write_text("def f():\n    pass\n", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)
    git(["tag", "v1"], origin)

    _, snap_path = create_snapshot(
        repository_url=str(origin),
        ref="v1",
        output_root=tmp_path / "snapshots",
    )
    f1, p1 = create_facts(snapshot_path=snap_path, output_root=tmp_path / "o1")
    f2, p2 = create_facts(snapshot_path=snap_path, output_root=tmp_path / "o2")

    d1 = f1.to_dict()
    d2 = f2.to_dict()
    # identity + semantic facts identical (no timestamps in facts)
    assert d1 == d2
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_facts_roundtrip(sample_repo: Path):
    results = run_analyzers(sample_repo)
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha="b" * 40,
            commit_short_sha="bbbbbbb",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(snapshot, results, "main_2026-09-22_bbbbbbb")
    restored = RepositoryFacts.from_dict(facts.to_dict())
    assert restored.to_dict() == facts.to_dict()


# ---------------------------------------------------------------------------
# M2.1 — Stable identity, language neutrality, provenance
# ---------------------------------------------------------------------------

def test_same_function_name_distinct_qualified_names(sample_repo: Path):
    results = run_analyzers(sample_repo)
    fns = [
        f
        for f in results["python"]["functions"]
        if f["symbol"] == "main" and f["path"] in ("src/foo.py", "src/bar.py")
    ]
    assert len(fns) == 2
    qnames = {f["qualified_name"] for f in fns}
    assert qnames == {"src/foo.py::main", "src/bar.py::main"}
    assert len(qnames) == len(fns)


def test_entities_language_neutral_identity(sample_repo: Path):
    results = run_analyzers(sample_repo)
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha="c" * 40,
            commit_short_sha="ccccccc",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(snapshot, results, "main_2026-09-22_ccccccc")
    entity_facts = facts.facts_of_type("entity")
    assert entity_facts

    kinds = {e["kind"] for e in entity_facts}
    assert kinds <= {"module", "class", "function"}
    for fact in entity_facts:
        entity = fact["attributes"]
        assert entity["language"] == "python"
        assert entity["name"]
        assert entity["qualified_name"]
        assert entity["path"]
        if fact["kind"] == "module":
            assert entity["qualified_name"] == entity["path"]
        else:
            assert entity["qualified_name"].startswith(entity["path"] + "::")

    # collision check on graph-ready identity key
    keys = [
        (e["kind"], e["attributes"]["qualified_name"], e["attributes"]["language"])
        for e in entity_facts
    ]
    assert len(keys) == len(set(keys))
    validate_facts_invariants(facts)
    validate_facts_document(facts.to_dict())


def test_entities_include_both_mains(sample_repo: Path):
    results = run_analyzers(sample_repo)
    entities = build_entities(results["python"])
    mains = [e for e in entities if e["name"] == "main"]
    paths = {e["path"] for e in mains}
    assert "src/foo.py" in paths
    assert "src/bar.py" in paths
    assert len({e["qualified_name"] for e in mains}) == len(mains)


def test_entity_kind_and_language_on_python_records(sample_repo: Path):
    results = run_analyzers(sample_repo)
    for fn in results["python"]["functions"]:
        assert fn["entity_kind"] == "function"
        assert fn["language"] == "python"
        assert fn["name"] == fn["symbol"]
        assert fn["qualified_name"] == "{}::{}".format(fn["path"], fn["symbol"])
        assert fn["line_start"] == fn["line"]
    for cls in results["python"]["classes"]:
        assert cls["entity_kind"] == "class"
        assert cls["language"] == "python"
        assert "::" in cls["qualified_name"]
    for mod in results["python"]["modules"]:
        if not mod.get("parse_error"):
            assert mod["entity_kind"] == "module"
            assert mod["language"] == "python"
            assert mod["qualified_name"] == mod["path"]


def test_entity_provenance_line_points_to_symbol(sample_repo: Path):
    results = run_analyzers(sample_repo)
    entity = next(
        e
        for e in build_entities(results["python"])
        if e["qualified_name"] == "src/foo.py::main"
    )
    assert entity["path"] == "src/foo.py"
    assert entity.get("line_start") is not None
    lines = (sample_repo / "src" / "foo.py").read_text(encoding="utf-8").splitlines()
    assert "main" in lines[entity["line_start"] - 1]


def test_snapshot_provenance_preserved_in_facts(sample_repo: Path):
    results = run_analyzers(sample_repo)
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    commit = "d" * 40
    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha=commit,
            commit_short_sha="ddddddd",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(snapshot, results, "main_2026-09-22_ddddddd")
    assert facts.snapshot.commit_sha == commit
    doc = facts.to_dict()
    assert doc["snapshot"]["commit_sha"] == commit
    validate_facts_document(doc)


def test_schema_requires_entities_and_identity_fields():
    schema = json.loads(
        Path("schemas/repository_facts.v2.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["schema_version"]["const"] == "repository_facts.v2"
    # Generic envelope: facts[] + open fact_type/kind, analyzer array
    assert "facts" in schema["required"]
    assert "analyzers" in schema["required"]
    for forbidden in (
        "entities",
        "python",
        "configuration",
        "dependencies",
        "agents",
        "mcp",
        "documentation",
        "tests",
    ):
        assert forbidden not in schema["required"]
        assert forbidden not in schema["properties"]
    # repository = identity/metadata only — no inventory arrays
    repo = schema["properties"]["repository"]
    assert set(repo["required"]) == {"name", "url"}
    assert set(repo["properties"]) == {"name", "url"}
    assert "files" not in repo["properties"]
    assert "directories" not in repo["properties"]
    assert schema["properties"]["facts"]["description"]
    fact = schema["$defs"]["FactRecord"]
    assert set(fact["required"]) == {
        "fact_id",
        "fact_type",
        "kind",
        "source",
        "attributes",
        "evidence",
    }
    # open vocabulary — no enum on fact_type or kind
    assert "enum" not in fact["properties"]["fact_type"]
    assert "enum" not in fact["properties"]["kind"]
    assert fact["properties"]["attributes"].get("additionalProperties") is True
    assert schema["properties"]["analyzers"]["type"] == "array"


def test_validate_entity_identity_rejects_duplicates():
    entities = [
        {
            "entity_kind": "function",
            "name": "main",
            "qualified_name": "a.py::main",
            "language": "python",
            "path": "a.py",
        },
        {
            "entity_kind": "function",
            "name": "main",
            "qualified_name": "a.py::main",
            "language": "python",
            "path": "a.py",
        },
    ]
    problems = validate_entity_identity(entities)
    assert any("duplicate" in p for p in problems)


def test_normalize_entities_deterministic(sample_repo: Path):
    results = run_analyzers(sample_repo)
    a = build_entities(results["python"])
    b = build_entities(results["python"])
    assert a == b
    # sorted by kind then qualified_name
    keys = [(e["entity_kind"], e["qualified_name"]) for e in a]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# M2 finalization — cross-representation, schema of real output, ordering
# ---------------------------------------------------------------------------

def test_cross_representation_consistency(sample_repo: Path):
    results = run_analyzers(sample_repo)
    entities = build_entities(results["python"])
    problems = validate_cross_representation(entities, results["python"])
    assert problems == []

    # every python function has a matching entity with same identity fields
    ent_fn = {
        e["qualified_name"]: e
        for e in entities
        if e["entity_kind"] == "function"
    }
    for fn in results["python"]["functions"]:
        e = ent_fn[fn["qualified_name"]]
        assert e["path"] == fn["path"]
        assert e["name"] == fn["name"] == fn["symbol"]
        assert e["language"] == fn["language"] == "python"
        assert e["line_start"] == fn["line_start"] == fn["line"]


def test_cross_representation_detects_mismatch():
    python = {
        "modules": [
            {
                "path": "a.py",
                "name": "a",
                "entity_kind": "module",
                "qualified_name": "a.py",
                "language": "python",
            }
        ],
        "classes": [],
        "functions": [
            {
                "path": "a.py",
                "symbol": "main",
                "name": "main",
                "kind": "function",
                "entity_kind": "function",
                "qualified_name": "a.py::main",
                "language": "python",
                "line": 1,
                "line_start": 1,
            }
        ],
        "imports": [],
    }
    entities = build_entities(python)
    # corrupt entity line
    for e in entities:
        if e["entity_kind"] == "function":
            e["line_start"] = 99
    problems = validate_cross_representation(entities, python)
    assert any("line mismatch" in p for p in problems)


def test_malformed_facts_fail_schema():
    facts_doc = {
        "schema_version": "repository_facts.v2",
        "snapshot": {
            "commit_sha": "a" * 40,
            "snapshot_id": "x",
            "repository_url": "https://example.com/r.git",
        },
        "repository": {"name": "r", "url": "https://example.com/r.git"},
        "facts": [],
        "analyzers": [{"name": "repository-intelligence", "version": "0.1.0"}],
    }
    validate_facts_document(facts_doc)  # baseline valid

    broken = json.loads(json.dumps(facts_doc))
    broken["facts"] = [
        {
            "fact_id": "fact:entity:python:function:x",
            "fact_type": "entity",
            "kind": "function",
            "source": "entities",
            # missing attributes + evidence
        }
    ]
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken)

    broken2 = json.loads(json.dumps(facts_doc))
    broken2["snapshot"]["commit_sha"] = "abc"
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken2)

    # analyzer-specific root sections are rejected under the generic envelope
    broken3 = json.loads(json.dumps(facts_doc))
    broken3["python"] = {"modules": [], "classes": [], "functions": [], "imports": []}
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken3)

    # repository.files / repository.directories are not part of the contract
    broken4 = json.loads(json.dumps(facts_doc))
    broken4["repository"]["files"] = []
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken4)
    broken5 = json.loads(json.dumps(facts_doc))
    broken5["repository"]["directories"] = []
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken5)


def test_generated_artifact_validates_against_schema(sample_repo: Path, tmp_path: Path):
    """Real pipeline output → repository_facts.v2 validation."""
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)

    _, snap_path = create_snapshot(
        repository_url=str(origin),
        ref="main",
        output_root=tmp_path / "snapshots",
    )
    _, facts_path = create_facts(
        snapshot_path=snap_path, output_root=tmp_path / "facts"
    )
    doc = json.loads(facts_path.read_text(encoding="utf-8"))
    validate_facts_document(doc)  # must not raise
    validate_facts_invariants(RepositoryFacts.from_dict(doc))


def test_category_ordering_deterministic(sample_repo: Path):
    results = run_analyzers(sample_repo)
    for key in (
        "configuration",
        "dependencies",
        "agents",
        "mcp",
        "documentation",
        "tests",
    ):
        seq = results[key]
        if key == "dependencies":
            keys = [(x["source"], x.get("section", ""), x["name"]) for x in seq]
            assert keys == sorted(keys)
        else:
            paths = [x["path"] for x in seq]
            assert paths == sorted(paths)

    files = [f["path"] for f in results["repository"]["files"]]
    dirs = [d["path"] for d in results["repository"]["directories"]]
    assert files == sorted(files)
    assert dirs == sorted(dirs)


def test_entities_python_modules_agree(sample_repo: Path):
    results = run_analyzers(sample_repo)
    entities = build_entities(results["python"])
    ent_mods = {
        e["qualified_name"] for e in entities if e["entity_kind"] == "module"
    }
    py_mods = {m["qualified_name"] for m in results["python"]["modules"]}
    assert ent_mods == py_mods


def test_on_disk_facts_if_present_schema_and_provenance():
    """Validate the checked-in ADOP facts artifact when it exists."""
    path = Path(
        "artifacts/main_2026-09-22_ef11dac/facts/repository_facts.json"
    )
    if not path.is_file():
        pytest.skip("ADOP facts artifact not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_facts_document(doc)
    facts = RepositoryFacts.from_dict(doc)
    validate_facts_invariants(facts)
    # full 40-char SHA bound to M1
    assert len(facts.snapshot.commit_sha) == 40
    assert facts.snapshot.commit_sha == "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"
    # entity facts carry stable identity
    entity_facts = facts.facts_of_type("entity")
    assert entity_facts
    mains = [
        e
        for e in entity_facts
        if e["kind"] == "function" and e["attributes"]["name"] == "main"
    ]
    if len(mains) >= 2:
        assert len({e["attributes"]["qualified_name"] for e in mains}) == len(mains)
    # evidence present and commit-consistent
    for fact in facts.facts:
        assert fact["evidence"]
        for ref in fact["evidence"]:
            assert ref["commit"] == facts.snapshot.commit_sha
    ids = [fact["fact_id"] for fact in facts.facts]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# M2.2 — EvidenceRef, integrity, centralized artifact layout
# ---------------------------------------------------------------------------

def test_evidence_ref_required_and_optional_fields():
    commit = "a" * 40
    ev = make_evidence(path="src/foo.py", artifact_type="python", commit=commit)
    assert set(ev) == {"path", "artifact_type", "commit"}
    assert is_repository_relative_path(ev["path"])

    ev2 = make_evidence(
        path="src/foo.py",
        artifact_type="python",
        commit=commit,
        line_start=3,
        line_end=10,
        symbol="main",
    )
    assert ev2["line_start"] == 3
    assert ev2["line_end"] == 10
    assert ev2["symbol"] == "main"
    assert validate_evidence_ref(ev2, commit) == []


def test_evidence_invalid_paths_and_lines():
    commit = "a" * 40
    assert not is_repository_relative_path("/abs/path.py")
    assert not is_repository_relative_path("../escape.py")
    assert not is_repository_relative_path("C:/windows/path.py")
    assert not is_repository_relative_path("")
    assert not is_repository_relative_path("a\\b.py")
    assert is_repository_relative_path("src/foo.py")
    assert is_repository_relative_path(".claude/agents/iac-agent.md")

    bad_path = make_evidence("/etc/passwd", "file", commit)
    assert any("repository-relative" in p for p in validate_evidence_ref(bad_path, commit))

    bad_lines = make_evidence("a.py", "python", commit, line_start=5, line_end=2)
    assert any("line_end" in p for p in validate_evidence_ref(bad_lines, commit))

    zero = make_evidence("a.py", "python", commit, line_start=0)
    assert any("line_start" in p for p in validate_evidence_ref(zero, commit))

    bad_commit = make_evidence("a.py", "python", "b" * 40)
    assert any("snapshot.commit_sha" in p for p in validate_evidence_ref(bad_commit, commit))

    bad_type = make_evidence("a.py", "architecture_role", commit)
    assert any("vocabulary" in p for p in validate_evidence_ref(bad_type, commit))


def test_evidence_commit_format_rejected():
    commit = "a" * 40
    ev = make_evidence("a.py", "python", "not-a-sha")
    problems = validate_evidence_ref(ev, commit)
    assert any("40-char" in p for p in problems)


def test_compose_attaches_evidence_to_all_meaningful_sections(sample_repo: Path):
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    commit = "f" * 40
    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha=commit,
            commit_short_sha="fffffff",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    results = run_analyzers(sample_repo)
    facts = compose_facts(snapshot, results, "main_2026-09-22_fffffff")

    # every path-backed fact carries at least one evidence ref bound to commit
    for fact in facts.facts:
        attributes = fact["attributes"]
        path = attributes.get("path") or attributes.get("source")
        assert path, "expected path-backed fact: {}".format(fact["fact_id"])
        assert fact["evidence"], "missing evidence on {}".format(fact["fact_id"])
        for ref in fact["evidence"]:
            assert ref["commit"] == commit
            assert is_repository_relative_path(ref["path"])
            assert ref["artifact_type"] in ARTIFACT_TYPES

    # representative evidence quality by fact_type/kind
    fn = next(
        f
        for f in facts.facts
        if f["fact_type"] == "entity"
        and f["kind"] == "function"
        and f["attributes"]["name"] == "main"
    )
    ref = fn["evidence"][0]
    assert ref["line_start"] == fn["attributes"]["line_start"]
    assert ref["symbol"] == "main"
    assert ref["artifact_type"] == "python"

    mod = next(f for f in facts.facts if f["fact_type"] == "entity" and f["kind"] == "module")
    # modules are file-level — no forced lines
    assert "line_start" not in mod["evidence"][0]

    agent = next(
        f
        for f in facts.facts
        if f["source"] == "agents"
        and f["attributes"]["path"].endswith("iac-agent.md")
    )
    assert agent["evidence"][0]["artifact_type"] == "agent_definition"
    assert "line_start" not in agent["evidence"][0]

    mcp = next(
        f
        for f in facts.facts
        if f["source"] == "mcp" and "server" in f["attributes"]["path"]
    )
    assert mcp["evidence"][0]["artifact_type"] == "mcp_definition"

    doc = next(
        f
        for f in facts.facts
        if f["fact_type"] == "documentation"
        and f["attributes"]["path"] == "README.md"
    )
    assert doc["evidence"][0]["artifact_type"] == "documentation"

    test_art = next(
        f
        for f in facts.facts
        if f["fact_type"] == "test" and f["kind"] == "test_file"
    )
    assert test_art["evidence"][0]["artifact_type"] == "test"

    cfg = next(
        f
        for f in facts.facts
        if f["fact_type"] == "configuration"
        and f["attributes"]["path"] == "pyproject.toml"
    )
    assert cfg["evidence"][0]["artifact_type"] == "configuration"

    dep = next(
        f
        for f in facts.facts
        if f["fact_type"] == "dependency"
        and f["attributes"]["source"] == "pyproject.toml"
    )
    assert dep["evidence"][0]["path"] == "pyproject.toml"
    assert dep["evidence"][0]["artifact_type"] == "file"

    # import evidence has line
    imp = next(
        f
        for f in facts.facts
        if f["fact_type"] == "import" and f["attributes"].get("module") == "os"
    )
    assert imp["evidence"][0]["line_start"] == imp["attributes"]["line"]
    assert imp["evidence"][0]["artifact_type"] == "python"

    validate_facts_invariants(facts)
    validate_facts_document(facts.to_dict())


def test_evidence_ordering_deterministic(sample_repo: Path):
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    def build():
        snapshot = RepositorySnapshot(
            repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
            snapshot=SnapshotInfo(
                ref="main",
                commit_sha="a" * 40,
                commit_short_sha="aaaaaaa",
                captured_at="2026-09-22T00:00:00+00:00",
            ),
            workspace=WorkspaceInfo(working_tree_clean=True),
            analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
        )
        return compose_facts(snapshot, run_analyzers(sample_repo), "x").to_dict()

    assert build() == build()


def test_validate_evidence_list_requires_evidence_and_matching_commit():
    commit = "a" * 40
    ok = [{"evidence": make_evidence("a.py", "python", commit)}]
    assert validate_evidence_list(ok, commit, "entities") == []

    missing = [{"path": "a.py"}]
    problems = validate_evidence_list(missing, commit, "entities")
    assert any("missing evidence" in p for p in problems)

    wrong = [{"evidence": make_evidence("a.py", "python", "b" * 40)}]
    problems = validate_evidence_list(wrong, commit, "entities")
    assert any("snapshot.commit_sha" in p for p in problems)


def test_artifact_layout_under_artifacts_root(tmp_path: Path):
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (origin / "README.md").write_text("# t\n", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)

    out = tmp_path / "artifacts"
    snap, snap_path = create_snapshot(
        repository_url=str(origin), ref="main", output_root=out
    )
    facts, facts_path = create_facts(snapshot_path=snap_path, output_root=out)

    sid = facts.snapshot.snapshot_id
    root = out / sid
    assert facts_path == root / "facts" / "repository_facts.json"
    assert snap_path == root / "snapshot" / "repository_snapshot.json"
    assert (root / "schemas" / "repository_facts.v2.json").is_file()
    assert (root / "schemas" / "repository_snapshot.v1.json").is_file()

    # no generated files directly under <id>/ (only the pinned checkout
    # persisted by M1 plus the three artifact subdirs)
    entries = {p.name for p in root.iterdir()}
    assert entries == {"schemas", "facts", "snapshot", "repository"}
    # M18: the checkout itself sits in repository/<name>/ beside the
    # container, not directly inside it.
    from repo_intelligence.artifacts import repository_dir

    assert (repository_dir(out, sid) / ".git").exists()


# ---------------------------------------------------------------------------
# Pinned snapshot reuse (M1 persists, M2 reuses — never re-clones)
# ---------------------------------------------------------------------------

def _make_pinned_snapshot(tmp_path: Path):
    """Create a local origin repo and an M1 snapshot persisted under out."""
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)

    out = tmp_path / "artifacts"
    snapshot, snap_path = create_snapshot(
        repository_url=str(origin), ref="main", output_root=out
    )
    return snapshot, snap_path, out


def _forbid_clone(*args, **kwargs):
    raise AssertionError("M2 must never clone; it reuses the persisted snapshot")


def test_create_facts_reuses_persisted_repository_without_cloning(
    tmp_path: Path, monkeypatch
):
    from repo_intelligence.facts import builder as facts_builder

    snapshot, snap_path, out = _make_pinned_snapshot(tmp_path)
    monkeypatch.setattr(facts_builder.git_adapter, "clone_repository", _forbid_clone)

    facts, facts_path = create_facts(snapshot_path=snap_path, output_root=out)

    assert facts_path.is_file()
    assert facts.snapshot.commit_sha == snapshot.snapshot.commit_sha
    assert facts.snapshot.snapshot_id == snapshot.snapshot_id


def test_create_facts_fails_clearly_when_persisted_repository_missing(
    tmp_path: Path, monkeypatch
):
    import shutil

    from repo_intelligence.facts import builder as facts_builder

    snapshot, snap_path, out = _make_pinned_snapshot(tmp_path)
    shutil.rmtree(out / snapshot.snapshot_id / "repository")
    monkeypatch.setattr(facts_builder.git_adapter, "clone_repository", _forbid_clone)

    with pytest.raises(FactsError, match="Persisted repository snapshot not found"):
        create_facts(snapshot_path=snap_path, output_root=out)


def test_create_facts_detects_stale_persisted_head_without_recloning(
    tmp_path: Path, monkeypatch
):
    import subprocess

    from repo_intelligence.facts import builder as facts_builder

    snapshot, snap_path, out = _make_pinned_snapshot(tmp_path)
    from repo_intelligence.artifacts import repository_dir

    pinned = repository_dir(out, snapshot.snapshot_id)
    # Corrupt the persisted checkout for real: advance its detached HEAD
    # to a new commit that no longer equals the recorded snapshot commit.
    subprocess.run(
        [
            "git",
            "-C",
            str(pinned),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "corruption",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setattr(facts_builder.git_adapter, "clone_repository", _forbid_clone)

    with pytest.raises(FactsError, match="does not match snapshot commit"):
        create_facts(snapshot_path=snap_path, output_root=out)


def test_cross_artifact_consistency(tmp_path: Path):
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)

    out = tmp_path / "artifacts"
    snap, snap_path = create_snapshot(
        repository_url=str(origin), ref="main", output_root=out
    )
    facts, facts_path = create_facts(snapshot_path=snap_path, output_root=out)

    snap_doc = json.loads(snap_path.read_text(encoding="utf-8"))
    facts_doc = json.loads(facts_path.read_text(encoding="utf-8"))

    sid = facts.snapshot.snapshot_id
    assert snap_path.parent.parent.name == sid
    assert facts_path.parent.parent.name == sid
    assert facts_doc["snapshot"]["snapshot_id"] == sid
    assert facts_doc["snapshot"]["commit_sha"] == snap_doc["snapshot"]["commit_sha"]
    assert facts_doc["snapshot"]["repository_url"] == snap_doc["repository"]["url"]
    assert snap.snapshot.commit_sha == facts_doc["snapshot"]["commit_sha"]

    # every evidence commit matches
    for fact in facts_doc["facts"]:
        for ref in fact["evidence"]:
            assert ref["commit"] == snap_doc["snapshot"]["commit_sha"]


def test_schema_copies_match_authoritative_sources():
    from repo_intelligence.artifacts import FACTS_SCHEMA_SOURCE, SNAPSHOT_SCHEMA_SOURCE

    src_facts = json.loads(FACTS_SCHEMA_SOURCE.read_text(encoding="utf-8"))
    src_snap = json.loads(SNAPSHOT_SCHEMA_SOURCE.read_text(encoding="utf-8"))
    assert src_facts["title"] == "repository_facts.v2"
    assert src_snap["title"] == "repository_snapshot.v1"
    # single authoritative definition: copies are byte-identical when present
    copy = Path(
        "artifacts/main_2026-09-22_ef11dac/schemas/repository_facts.v2.json"
    )
    if copy.is_file():
        assert json.loads(copy.read_text(encoding="utf-8")) == src_facts
    stale = Path(
        "artifacts/main_2026-09-22_ef11dac/schemas/repository_facts.v1.json"
    )
    assert not stale.exists()


def test_evidence_ref_dataclass_roundtrip():
    commit = "d" * 40
    ev = EvidenceRef(
        path="x.py",
        artifact_type="python",
        commit=commit,
        line_start=1,
        line_end=2,
        symbol="f",
    )
    restored = EvidenceRef.from_dict(ev.to_dict())
    assert restored == ev


# ---------------------------------------------------------------------------
# M2 freeze — default artifact root is project-root-relative, not CWD
# ---------------------------------------------------------------------------

def test_default_artifacts_root_is_project_absolute_not_cwd(tmp_path: Path, monkeypatch):
    from repo_intelligence.artifacts import (
        ARTIFACTS_ROOT,
        PROJECT_ROOT,
        default_artifacts_root,
    )

    assert ARTIFACTS_ROOT.is_absolute()
    assert ARTIFACTS_ROOT == PROJECT_ROOT / "artifacts"
    assert PROJECT_ROOT.name == "repository-intelligence" or (
        (PROJECT_ROOT / "schemas").is_dir()
        and (PROJECT_ROOT / "src").is_dir()
    )

    nested = tmp_path / "nested" / "working-directory"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    resolved = default_artifacts_root()
    assert resolved == PROJECT_ROOT / "artifacts"
    assert resolved.is_absolute()
    assert not str(resolved).startswith(str(nested.resolve()))
    assert Path.cwd() == nested.resolve() or Path.cwd() == nested


def test_default_generation_from_nested_cwd_uses_project_artifacts(
    tmp_path: Path, monkeypatch
):
    """From a nested CWD, default output_root writes under project artifacts/,
    not <cwd>/artifacts/."""
    import subprocess

    origin = tmp_path / "origin"
    origin.mkdir()

    def git(args, cwd):
        subprocess.run(
            ["git"] + args, cwd=str(cwd), check=True, capture_output=True, text=True
        )

    git(["init", "-q", "-b", "main"], origin)
    git(["config", "user.email", "t@example.com"], origin)
    git(["config", "user.name", "T"], origin)
    (origin / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    git(["add", "."], origin)
    git(["commit", "-q", "-m", "init"], origin)

    # Sandbox the default root so the test does not write into the real
    # project artifacts/ tree, while still proving CWD-independence.
    sandbox_root = tmp_path / "repository-root" / "artifacts"
    nested = tmp_path / "repository-root" / "nested" / "working-directory"
    nested.mkdir(parents=True)

    import repo_intelligence.artifacts as art_mod
    import repo_intelligence.facts.builder as facts_mod
    import repo_intelligence.snapshot.builder as snap_mod

    monkeypatch.setattr(art_mod, "default_artifacts_root", lambda: sandbox_root)
    monkeypatch.setattr(facts_mod, "resolve_output_root", lambda p: Path(p) if p is not None else sandbox_root)
    monkeypatch.setattr(snap_mod, "resolve_output_root", lambda p: Path(p) if p is not None else sandbox_root)

    monkeypatch.chdir(nested)
    assert Path.cwd().resolve() == nested.resolve()

    snap, snap_path = create_snapshot(
        repository_url=str(origin), ref="main"
    )  # no output_root → default
    facts, facts_path = create_facts(snapshot_path=snap_path)

    sid = facts.snapshot.snapshot_id
    assert facts_path == sandbox_root / sid / "facts" / "repository_facts.json"
    assert snap_path == sandbox_root / sid / "snapshot" / "repository_snapshot.json"
    # not under nested CWD
    assert not (nested / "artifacts").exists()
    assert not (nested / sid).exists()

    # explicit override still wins over default
    override = tmp_path / "explicit-output"
    snap2, snap_path2 = create_snapshot(
        repository_url=str(origin), ref="main", output_root=override
    )
    assert str(snap_path2).startswith(str(override.resolve())) or str(
        snap_path2
    ).startswith(str(override))
    assert snap_path2 == override / snap2.snapshot_id / "snapshot" / "repository_snapshot.json"


def test_resolve_output_root_precedence(tmp_path: Path):
    from repo_intelligence.artifacts import (
        PROJECT_ROOT,
        resolve_output_root,
    )

    override = tmp_path / "custom"
    assert resolve_output_root(override) == override
    assert resolve_output_root(None) == PROJECT_ROOT / "artifacts"
    # relative explicit path preserved as given (user override)
    assert resolve_output_root(Path("relative-out")) == Path("relative-out")


# ---------------------------------------------------------------------------
# repository_facts.v2 — generic envelope, fact preservation, extensibility
# ---------------------------------------------------------------------------


def _minimal_facts_doc(facts_list=None, analyzers=None):
    return {
        "schema_version": "repository_facts.v2",
        "snapshot": {
            "commit_sha": "a" * 40,
            "snapshot_id": "main_2026-09-22_aaaaaaa",
            "repository_url": "https://example.com/r.git",
        },
        "repository": {"name": "r", "url": "https://example.com/r.git"},
        "facts": facts_list if facts_list is not None else [],
        "analyzers": analyzers
        if analyzers is not None
        else [{"name": "repository-intelligence", "version": "0.2.0"}],
    }


def test_root_contract_is_generic_only(sample_repo: Path):
    """Root keys are exactly the stable envelope — no analyzer sections."""
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha="e" * 40,
            commit_short_sha="eeeeeee",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(
        snapshot, run_analyzers(sample_repo), "main_2026-09-22_eeeeeee"
    )
    doc = facts.to_dict()
    assert set(doc) == {
        "schema_version",
        "snapshot",
        "repository",
        "facts",
        "analyzers",
    }
    assert doc["schema_version"] == "repository_facts.v2"
    # repository = identity/metadata only
    assert set(doc["repository"]) == {"name", "url"}
    assert doc["repository"]["name"] == "y"
    assert doc["repository"]["url"] == "https://example.com/y.git"
    assert isinstance(doc["analyzers"], list)
    assert doc["analyzers"] and set(doc["analyzers"][0]) == {"name", "version"}
    assert isinstance(doc["facts"], list)
    for fact in doc["facts"]:
        assert set(fact) == {
            "fact_id",
            "fact_type",
            "kind",
            "source",
            "attributes",
            "evidence",
        }
        assert fact["fact_id"].startswith("fact:")
        assert isinstance(fact["fact_type"], str) and fact["fact_type"]
        assert isinstance(fact["kind"], str)
        assert isinstance(fact["source"], str) and fact["source"]
        assert isinstance(fact["attributes"], dict)
        assert isinstance(fact["evidence"], list)
    validate_facts_document(doc)
    validate_facts_invariants(facts)


def test_fact_preservation_representatives(sample_repo: Path):
    """All v1 categories survive as facts[] representatives under v2."""
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha="a" * 40,
            commit_short_sha="aaaaaaa",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(snapshot, run_analyzers(sample_repo), "x")

    def paths(fact_type):
        return {
            f["attributes"].get("path") or f["attributes"].get("source")
            for f in facts.facts_of_type(fact_type)
        }

    assert "src/app/main.py" in paths("file")
    assert "src" in paths("directory") or any(
        p and p.startswith("src") for p in paths("directory")
    )

    entity_facts = facts.facts_of_type("entity")
    assert {f["kind"] for f in entity_facts} >= {"module", "class", "function"}
    assert any(
        f["kind"] == "function" and f["attributes"]["name"] == "main"
        for f in entity_facts
    )
    assert any(f["fact_type"] == "import" for f in facts.facts)
    assert any(f["fact_type"] == "configuration" for f in facts.facts)
    assert any(f["fact_type"] == "dependency" for f in facts.facts)
    assert any(
        f["fact_type"] == "artifact" and f["source"] == "agents" for f in facts.facts
    )
    assert any(
        f["fact_type"] == "artifact" and f["source"] == "mcp" for f in facts.facts
    )
    assert any(f["fact_type"] == "documentation" for f in facts.facts)
    assert any(f["fact_type"] == "test" for f in facts.facts)

    counts = facts.counts()
    assert counts["files"] == len(paths("file"))
    assert counts["entities"] == len(entity_facts)
    assert counts["agents"] > 0
    assert counts["mcp"] > 0
    assert counts["evidence"] >= len(facts.facts)


def test_fact_ids_stable_across_double_compose(sample_repo: Path):
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    def build():
        snapshot = RepositorySnapshot(
            repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
            snapshot=SnapshotInfo(
                ref="main",
                commit_sha="b" * 40,
                commit_short_sha="bbbbbbb",
                captured_at="2026-09-22T00:00:00+00:00",
            ),
            workspace=WorkspaceInfo(working_tree_clean=True),
            analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
        )
        return compose_facts(
            snapshot, run_analyzers(sample_repo), "main_2026-09-22_bbbbbbb"
        )

    a = build()
    b = build()
    ids_a = [f["fact_id"] for f in a.facts]
    ids_b = [f["fact_id"] for f in b.facts]
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a))
    assert a.to_dict() == b.to_dict()


def test_compute_fact_id_deterministic_and_position_free():
    attrs = {"path": "src/a.py", "type": "python", "size": 12}
    assert (
        compute_fact_id("file", "repository_file", "filesystem", attrs)
        == "fact:file:src/a.py"
    )
    assert compute_fact_id("file", "repository_file", "filesystem", attrs) == (
        compute_fact_id("file", "repository_file", "filesystem", dict(attrs))
    )

    entity_attrs = {
        "language": "python",
        "qualified_name": "src/a.py::main",
        "path": "src/a.py",
        "name": "main",
    }
    fid = compute_fact_id("entity", "function", "entities", entity_attrs)
    # no line → path::name only
    assert fid == "fact:entity:python:function:src/a.py::main"
    # with line → disambiguates redefined symbols
    entity_attrs_line = dict(entity_attrs)
    entity_attrs_line["line_start"] = 12
    assert (
        compute_fact_id("entity", "function", "entities", entity_attrs_line)
        == "fact:entity:python:function:src/a.py::main:12"
    )
    # independent of insertion / list position
    assert fid == compute_fact_id(
        "entity", "function", "entities", dict(reversed(list(entity_attrs.items())))
    )


def test_future_analyzer_facts_validate_without_schema_edits():
    """Synthetic fact_types/kinds from a new analyzer pass the open schema."""
    commit = "a" * 40
    terraform = make_fact(
        fact_type="terraform",
        kind="aws_s3_bucket",
        source="terraform",
        attributes={
            "path": "infra/main.tf",
            "resource": "aws_s3_bucket",
            "name": "logs",
            "provider": "aws",
            "extra_future_field": {"nested": True},
        },
        evidence=[
            make_evidence(
                path="infra/main.tf",
                artifact_type="file",
                commit=commit,
                line_start=12,
                symbol="aws_s3_bucket.logs",
            )
        ],
    )
    social = make_fact(
        fact_type="social-post",
        kind="linkedin",
        source="social-post-analyzer",
        attributes={"path": "content/post.md", "network": "linkedin"},
        evidence=[make_evidence(path="content/post.md", artifact_type="documentation", commit=commit)],
    )
    assert terraform["fact_id"] == "fact:terraform:infra/main.tf"
    assert social["fact_id"] == "fact:social-post:content/post.md"

    doc = _minimal_facts_doc(facts_list=[terraform, social])
    validate_facts_document(doc)  # must not raise — open fact_type/kind

    facts = RepositoryFacts.from_dict(doc)
    validate_facts_invariants(facts)
    assert facts.counts()["facts"] == 2


def test_facts_sorted_deterministically(sample_repo: Path):
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name="y", url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha="d" * 40,
            commit_short_sha="ddddddd",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    facts = compose_facts(snapshot, run_analyzers(sample_repo), "x")
    type_order = [
        "file",
        "directory",
        "entity",
        "import",
        "configuration",
        "dependency",
        "artifact",
        "documentation",
        "test",
    ]
    ranks = []
    for fact in facts.facts:
        t = fact["fact_type"]
        ranks.append(type_order.index(t) if t in type_order else len(type_order))
    assert ranks == sorted(ranks)
    # within each type, fact_ids are sorted
    from itertools import groupby

    for _, group in groupby(facts.facts, key=lambda f: f["fact_type"]):
        ids = [f["fact_id"] for f in group]
        assert ids == sorted(ids)


def test_duplicate_fact_ids_rejected_by_invariants():
    doc = _minimal_facts_doc(
        facts_list=[
            make_fact(
                fact_type="file",
                kind="repository_file",
                source="filesystem",
                attributes={"path": "a.py", "type": "python"},
                evidence=[make_evidence("a.py", "file", "a" * 40)],
            ),
            make_fact(
                fact_type="file",
                kind="repository_file",
                source="filesystem",
                attributes={"path": "a.py", "type": "python"},
                evidence=[make_evidence("a.py", "file", "a" * 40)],
            ),
        ]
    )
    facts = RepositoryFacts.from_dict(doc)
    with pytest.raises(FactsSchemaValidationError) as exc:
        validate_facts_invariants(facts)
    assert "duplicate fact_id" in str(exc.value)


# ---------------------------------------------------------------------------
# M2 v2 freeze cleanup — single canonical facts[] + repository metadata
# ---------------------------------------------------------------------------


def _compose_sample(sample_repo: Path, name="y"):
    from repo_intelligence.snapshot.models import (
        AnalyzerInfo,
        RepositoryInfo,
        RepositorySnapshot,
        SnapshotInfo,
        WorkspaceInfo,
    )

    snapshot = RepositorySnapshot(
        repository=RepositoryInfo(name=name, url="https://example.com/y.git"),
        snapshot=SnapshotInfo(
            ref="main",
            commit_sha="a" * 40,
            commit_short_sha="aaaaaaa",
            captured_at="2026-09-22T00:00:00+00:00",
        ),
        workspace=WorkspaceInfo(working_tree_clean=True),
        analyzer=AnalyzerInfo(name="repository-intelligence", version="0.1.0"),
    )
    return compose_facts(snapshot, run_analyzers(sample_repo), "main_2026-09-22_aaaaaaa")


def test_a_repository_metadata_identity_only(sample_repo: Path):
    """repository keeps name/url only — no files/directories inventory."""
    facts = _compose_sample(sample_repo, name="sample-repo")
    assert set(facts.repository) == {"name", "url"}
    assert facts.repository["name"] == "sample-repo"
    assert facts.repository["url"] == "https://example.com/y.git"
    doc = facts.to_dict()
    assert set(doc["repository"]) == {"name", "url"}
    validate_facts_document(doc)
    validate_facts_invariants(facts)


def test_b_file_facts_in_canonical_store(sample_repo: Path):
    """Representative file observations live in facts[] with evidence."""
    facts = _compose_sample(sample_repo)
    file_facts = facts.facts_of_type("file")
    assert file_facts
    by_path = {f["attributes"]["path"]: f for f in file_facts}
    assert "src/app/main.py" in by_path
    entry = by_path["src/app/main.py"]
    assert entry["kind"] == "repository_file"
    assert entry["source"] == "filesystem"
    assert entry["attributes"]["type"] == "python"
    assert isinstance(entry["attributes"].get("size"), int)
    assert entry["evidence"]
    ref = entry["evidence"][0]
    assert ref["path"] == "src/app/main.py"
    assert ref["artifact_type"] == "file"
    assert ref["commit"] == facts.snapshot.commit_sha
    # no forced line numbers on filesystem facts
    assert "line_start" not in ref


def test_c_directory_facts_in_canonical_store(sample_repo: Path):
    """Representative directory observations live in facts[] with evidence."""
    facts = _compose_sample(sample_repo)
    dir_facts = facts.facts_of_type("directory")
    assert dir_facts
    paths = {f["attributes"]["path"] for f in dir_facts}
    assert "src" in paths
    entry = next(f for f in dir_facts if f["attributes"]["path"] == "src")
    assert entry["kind"] == "repository_directory"
    assert entry["source"] == "filesystem"
    assert entry["evidence"]
    ref = entry["evidence"][0]
    assert ref["path"] == "src"
    assert ref["artifact_type"] == "directory"
    assert ref["commit"] == facts.snapshot.commit_sha
    assert "line_start" not in ref


def test_d_no_duplicate_canonical_inventory(sample_repo: Path):
    """Single source of truth: no repository.files / repository.directories."""
    facts = _compose_sample(sample_repo)
    doc = facts.to_dict()
    assert "files" not in doc["repository"]
    assert "directories" not in doc["repository"]
    assert set(doc["repository"]) == {"name", "url"}

    # counts of files/dirs only come from facts[]
    file_from_facts = [f for f in doc["facts"] if f["fact_type"] == "file"]
    dir_from_facts = [f for f in doc["facts"] if f["fact_type"] == "directory"]
    assert file_from_facts
    assert dir_from_facts

    # schema rejects reintroducing inventory on repository
    broken = json.loads(json.dumps(doc))
    broken["repository"]["files"] = []
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken)
    broken = json.loads(json.dumps(doc))
    broken["repository"]["directories"] = []
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_document(broken)

    # invariants also reject metadata-only violations
    bad = RepositoryFacts.from_dict(doc)
    bad.repository["files"] = []
    with pytest.raises(FactsSchemaValidationError):
        validate_facts_invariants(bad)


def test_e_counts_derived_from_facts_only(sample_repo: Path):
    """counts() is a reporting summary computed solely from facts[]."""
    facts = _compose_sample(sample_repo)
    counts = facts.counts()

    # derive expected values independently from facts[]
    by_type = {}
    by_source = {}
    entity_kinds = {}
    evidence_total = 0
    for fact in facts.facts:
        by_type[fact["fact_type"]] = by_type.get(fact["fact_type"], 0) + 1
        by_source[fact["source"]] = by_source.get(fact["source"], 0) + 1
        if fact["fact_type"] == "entity":
            entity_kinds[fact["kind"]] = entity_kinds.get(fact["kind"], 0) + 1
        evidence_total += len(fact["evidence"])

    assert counts["facts"] == len(facts.facts)
    assert counts["files"] == by_type.get("file", 0)
    assert counts["directories"] == by_type.get("directory", 0)
    assert counts["entities"] == by_type.get("entity", 0)
    assert counts["python_modules"] == entity_kinds.get("module", 0)
    assert counts["python_symbols"] == entity_kinds.get("class", 0) + entity_kinds.get(
        "function", 0
    )
    assert counts["imports"] == by_type.get("import", 0)
    assert counts["configuration"] == by_type.get("configuration", 0)
    assert counts["dependencies"] == by_type.get("dependency", 0)
    assert counts["agents"] == by_source.get("agents", 0)
    assert counts["mcp"] == by_source.get("mcp", 0)
    assert counts["documentation"] == by_type.get("documentation", 0)
    assert counts["tests"] == by_type.get("test", 0)
    assert counts["evidence"] == evidence_total

    # counts are not part of the JSON contract
    doc = facts.to_dict()
    assert "counts" not in doc
    assert set(doc) == {
        "schema_version",
        "snapshot",
        "repository",
        "facts",
        "analyzers",
    }

    # mutating counts' return value does not affect facts
    counts["files"] = 999
    assert facts.counts()["files"] == by_type.get("file", 0)


def test_f_generic_extensibility_social_and_terraform():
    """Future analyzer fact_types/kinds validate without schema edits."""
    commit = "a" * 40
    social = make_fact(
        fact_type="artifact",
        kind="post",
        source="social-post-analyzer",
        attributes={"source": "linkedin"},
        evidence=[
            make_evidence(path="content/post.md", artifact_type="documentation", commit=commit)
        ],
    )
    terraform = make_fact(
        fact_type="terraform",
        kind="aws_s3_bucket",
        source="terraform",
        attributes={"path": "infra/main.tf", "resource": "aws_s3_bucket"},
        evidence=[make_evidence(path="infra/main.tf", artifact_type="file", commit=commit)],
    )
    assert social["fact_id"].startswith("fact:artifact:")
    assert terraform["fact_id"] == "fact:terraform:infra/main.tf"

    doc = _minimal_facts_doc(facts_list=[social, terraform])
    validate_facts_document(doc)
    facts = RepositoryFacts.from_dict(doc)
    validate_facts_invariants(facts)
    assert facts.counts()["facts"] == 2


def test_filesystem_fact_ids_stable_when_kind_changed():
    """kind change must not break path-based fact_id stability for files."""
    attrs = {"path": "src/a.py", "type": "python", "size": 10}
    a = compute_fact_id("file", "repository_file", "filesystem", attrs)
    b = compute_fact_id("file", "repository_file", "filesystem", dict(attrs))
    assert a == b == "fact:file:src/a.py"
