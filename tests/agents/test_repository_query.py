"""Repository Query — the module and the CLI (offline, never the network).

Two things are under test:

1. the **subject resolution** — by default the query analyzes this
   project's pinned ADOP snapshot (``artifacts/main_2026-09-22_ef11dac/
   repository`` at ``ef11dac…``), never the analyzer's own source tree,
   and never without verifying the commit first;
2. the **entry point itself** — flags, sections, interactive session,
   exit codes.

Same discipline as ``tests/agents/test_architecture_visualization_demo.py``:
every run is driven over temporary artifact/trace/memory/context roots
and the CLI is loaded by path — the demonstration must be standalone
and never import ``tests/``.  Nothing here writes into the repository's
own ``runtime/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import enable_search
from repo_intelligence import repository_query as rq
from repo_intelligence.agent import trace_sink as trace_sink_module
from repo_intelligence.agents import visualization_artifacts as viz_module
from repo_intelligence.graph.exporters import graphviz_available
from repo_intelligence.context import models as context_models
from repo_intelligence.context_engine import engine as context_engine_module
from repo_intelligence.orchestration import VALIDATION_FAILED
from repo_intelligence.planning.models import InvestigationPlan, PlanStep

CLI_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_repository_query.py"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARCH_MCP_ROOT = FIXTURES / "arch_mcp"

QUESTION = (
    "How do the agents, tools, and MCP servers interact?"
)


def _load_cli(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # register before exec: the module resolves annotations through
    # sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli(CLI_PATH, "agentrun_repository_query")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every root a run can write to, pointed at ``tmp_path``."""
    monkeypatch.setattr(
        viz_module,
        "default_artifacts_root",
        lambda: tmp_path / "artifacts",
    )
    monkeypatch.setattr(
        trace_sink_module,
        "default_trace_root",
        lambda: tmp_path / "runtime" / "agent_runs",
    )
    monkeypatch.setattr(
        context_engine_module,
        "default_context_root",
        lambda: tmp_path / "runtime" / "context",
    )
    monkeypatch.setattr(
        rq,
        "default_query_db_path",
        lambda: tmp_path / "runtime" / "agent_memory" / "repository_query.db",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


@pytest.fixture
def relative(sandbox, monkeypatch):
    """``sandbox`` plus the display root, so printed paths are
    relative exactly as they are for a default project run."""
    monkeypatch.setattr(context_models, "PROJECT_ROOT", sandbox)
    return sandbox


def _scripted_input(lines):
    values = list(lines)

    def fake(prompt=""):
        if not values:
            raise EOFError()
        return values.pop(0)

    return fake


def _run(argv, monkeypatch, capsys, inputs=None):
    if inputs is not None:
        monkeypatch.setattr("builtins.input", _scripted_input(inputs))
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _cli_default(monkeypatch, capsys):
    """One full offline query with NO ``--repository`` — the default
    subject (the pinned ADOP snapshot)."""
    return _run([QUESTION], monkeypatch, capsys)


def _cli_run(root, monkeypatch, capsys, *extra):
    """One full offline query with an explicit ``--repository``."""
    return _run(
        ["--repository", str(root)] + list(extra) + [QUESTION],
        monkeypatch,
        capsys,
    )


def _fake_snapshot(tmp_path, monkeypatch, commit=None, metadata=True):
    """Build the M1 layout under *tmp_path* and make it the default
    subject: ``<tmp>/artifacts/<snapshot-id>/repository`` (+ the
    snapshot record beside it).  ``commit=None`` leaves no record."""
    root = tmp_path / "artifacts" / rq.SUBJECT_SNAPSHOT_ID / "repository"
    root.mkdir(parents=True)
    (root / "README.md").write_text(
        "# a subject repository\n", encoding="utf-8"
    )
    if metadata:
        record = root.parent / "snapshot"
        record.mkdir(parents=True, exist_ok=True)
        (record / "repository_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": "repository_snapshot.v1",
                    "repository": {
                        "name": "a-subject-repository",
                        "url": "https://example.invalid/subject.git",
                    },
                    "snapshot": {
                        "ref": "main",
                        "commit_sha": commit,
                        "commit_short_sha": (commit or "")[:7],
                        "captured_at": "2026-09-22T00:00:00+00:00",
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(rq, "default_subject_repository", lambda: root)
    return root


# -- §15 test 1: default subject resolution --------------------------------


def test_default_subject_is_the_pinned_adop_snapshot():
    expected = (
        rq.PROJECT_ROOT
        / "artifacts"
        / "main_2026-09-22_ef11dac"
        / "repository"
        / "sample-Agentic-Ai-Data-Operations"
    )
    assert rq.default_subject_repository() == expected
    # resolved from THIS project, never the analyzer's own source tree
    assert rq.default_subject_repository() != Path(rq.PROJECT_ROOT)
    assert (expected / ".mcp.json").is_file()
    assert (expected / "README.md").is_file()
    # and the target resolution agrees
    root, target, error = rq.resolve_read_target()
    assert error is None
    assert root == expected
    assert target == "README.md"


# -- §15 test 2: explicit override -----------------------------------------


def test_explicit_repository_overrides_the_default(relative, monkeypatch,
                                                   capsys):
    code, out, err = _cli_run(ARCH_MCP_ROOT, monkeypatch, capsys)

    assert code == 0
    assert "Repository: arch_mcp" in out
    assert "target: agents/collector.py" in out
    assert "main_2026-09-22_ef11dac" not in out


# -- §15 test 3: missing default snapshot ----------------------------------


def test_missing_default_snapshot_fails_without_running(
    relative, tmp_path, monkeypatch, capsys
):
    missing = (
        tmp_path / "artifacts" / rq.SUBJECT_SNAPSHOT_ID / "repository"
    )
    monkeypatch.setattr(
        rq, "default_subject_repository", lambda: missing
    )

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == rq.EXIT_INVESTIGATION_FAILED != 0
    assert (
        "ERROR: Subject repository snapshot verification failed." in out
    )
    assert rq.SUBJECT_COMMIT in out
    assert "directory does not exist" in out
    assert "No investigation was executed." in out
    # no investigation at all
    assert "Agent execution summary" not in out
    assert "Investigation ID:" not in out
    assert not (tmp_path / "runtime").exists()


# -- §15 test 4: wrong commit / unverifiable subject -----------------------


def test_wrong_commit_stops_before_any_investigation(
    relative, tmp_path, monkeypatch, capsys
):
    wrong = "0" * 40
    _fake_snapshot(tmp_path, monkeypatch, commit=wrong)

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == rq.EXIT_INVESTIGATION_FAILED != 0
    assert (
        "ERROR: Subject repository snapshot verification failed." in out
    )
    assert rq.SUBJECT_COMMIT in out
    assert wrong in out
    assert "commit mismatch" in out
    assert "No investigation was executed." in out
    assert "Agent execution summary" not in out
    assert not (tmp_path / "runtime").exists()


def test_unverifiable_subject_is_refused(
    relative, tmp_path, monkeypatch, capsys
):
    """No Git metadata and no M1 record → the commit cannot be proven,
    so nothing is analyzed."""
    _fake_snapshot(tmp_path, monkeypatch, metadata=False)

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == rq.EXIT_INVESTIGATION_FAILED != 0
    assert (
        "ERROR: Subject repository snapshot verification failed." in out
    )
    assert "commit cannot be verified" in out
    assert "No investigation was executed." in out
    assert "Agent execution summary" not in out
    assert not (tmp_path / "runtime").exists()


# -- §15 test 5: the correct commit proceeds --------------------------------


def test_correct_commit_proceeds_to_the_m17_workflow(
    sandbox, monkeypatch, capsys
):
    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    assert "Subject repository:" in out
    assert "artifacts/main_2026-09-22_ef11dac/repository" in out
    assert "Pinned commit:" in out
    assert rq.SUBJECT_COMMIT in out
    assert "Status:     COMPLETED" in out
    assert "Agent execution summary" in out
    for role in (
        "Architecture Investigator",
        "Flow / Dependency Investigator",
        "Architecture Visualization",
    ):
        assert role in out
    assert "collaboration: passed" in out
    assert "diagram:       passed" in out
    assert "exit code: 0" in out


# -- §15 tests 6 and 7: the subject that reaches the agents ----------------


def _spy_story(monkeypatch):
    """The ``(root, target)`` the investigation is handed — observed
    at the resolution step every run uses, since the offline
    investigation now discovers the rest of the story itself."""
    seen = []
    original = rq.resolve_read_target

    def spy(repository=None):
        resolved = original(repository)
        if resolved[2] is None:
            seen.append((resolved[0], resolved[1]))
        return resolved

    monkeypatch.setattr(rq, "resolve_read_target", spy)
    return seen


def test_subject_repository_reaches_the_agents(
    sandbox, monkeypatch, capsys
):
    seen = _spy_story(monkeypatch)

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    assert len(seen) == 1
    root, target = seen[0]
    assert root == rq.default_subject_repository()
    assert root == (
        rq.PROJECT_ROOT
        / "artifacts"
        / "main_2026-09-22_ef11dac"
        / "repository"
        / "sample-Agentic-Ai-Data-Operations"
    )
    assert target == "README.md"


def test_default_never_analyzes_the_analyzer_repository(
    sandbox, monkeypatch, capsys
):
    """Regression: living in repository-intelligence must not make the
    analyzer's own README the architecture source."""
    seen = _spy_story(monkeypatch)

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    root, target = seen[0]
    assert Path(root).resolve() != Path(rq.PROJECT_ROOT).resolve()
    assert (root / target).resolve() != (
        Path(rq.PROJECT_ROOT) / "README.md"
    ).resolve()
    assert str(rq.default_subject_repository()) in str(
        (root / target).resolve()
    )
    assert "main_2026-09-22_ef11dac/repository" in out


# -- §15 test 8: the final answer is displayed -----------------------------


def test_final_answer_is_displayed(sandbox, monkeypatch, capsys):
    expected = rq.run_repository_query(
        QUESTION,
        db_path=str(sandbox / "expected.db"),
        trace_root=str(sandbox / "expected_traces"),
    )
    assert expected.answer

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    assert expected.answer in out
    assert expected.conclusions[0] in out
    assert "Final answer" in out


# -- §15 test 9: SVG stays optional ----------------------------------------


def test_svg_is_optional_and_graphviz_is_not_required(
    sandbox, monkeypatch, capsys
):
    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    assert "architecture.svg" in out
    svg = (
        sandbox
        / "artifacts"
        / "architecture_visualization"
        / "architecture.svg"
    )
    claimed = "✓ architecture.svg" in out
    if claimed:
        # a tick is printed only when the file really exists
        assert svg.is_file()
    else:
        assert "○ architecture.svg — unavailable" in out
    assert not [
        name for name in sys.modules if name.startswith("graphviz")
    ]


# -- the documented output sections -----------------------------------------


def test_positional_query_renders_every_section(
    relative, monkeypatch, capsys
):
    code, out, err = _cli_run(ARCH_MCP_ROOT, monkeypatch, capsys)

    assert code == 0
    assert err == ""
    assert "exit code: 0" in out
    for heading in (
        "Final answer",
        "Agent execution summary",
        "Validation",
        "Artifacts",
        "Trace",
    ):
        assert "{}\n".format(heading) in out
    assert "{}\n{}".format("Final answer", "─" * 40) in out
    assert "Subject repository:" in out
    assert "Pinned commit:" in out
    assert "Question:" in out
    assert "Investigation ID:" in out
    assert "Runs:" in out
    assert "Status:     COMPLETED" in out
    assert "termination: SUCCESS" in out
    assert "collaboration: passed" in out
    assert "diagram:       passed" in out
    assert "S1" in out and "S2" in out and "S3" in out
    assert out.count("\u2713") >= 4
    assert "runtime/agent_runs/" in out


def test_success_produces_relative_paths_only(
    relative, tmp_path, monkeypatch, capsys
):
    """§19 test 7 / §17 — no absolute machine path reaches the
    output: the subject, the artifacts and the trace are all rendered
    project-relative, and the analyzer's own tree never appears."""
    _fake_snapshot(tmp_path, monkeypatch, commit=rq.SUBJECT_COMMIT)

    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    assert "artifacts/main_2026-09-22_ef11dac/repository" in out
    assert "artifacts/architecture_visualization/diagram_spec.json" in out
    assert "runtime/agent_runs/" in out
    assert str(relative) not in out
    assert str(FIXTURES) not in out
    assert str(rq.PROJECT_ROOT) not in out


def test_missing_svg_is_reported_never_claimed(
    relative, sandbox, monkeypatch, capsys
):
    code, out, err = _cli_run(ARCH_MCP_ROOT, monkeypatch, capsys)

    assert code == 0
    assert "architecture.svg" in out
    svg = (
        sandbox
        / "artifacts"
        / "architecture_visualization"
        / "architecture.svg"
    )
    if graphviz_available():
        assert "\u2713 architecture.svg" in out
        assert svg.is_file()
    else:
        assert "\u2713 architecture.svg" not in out
        assert "\u25cb architecture.svg \u2014 unavailable" in out
        assert "unavailable" in out
    assert "\u2713 diagram_spec.json" in out
    assert "\u2713 architecture.mmd" in out
    assert "\u2713 run_summary.md" in out


def test_module_run_returns_the_whole_result(
    sandbox, tmp_path, monkeypatch
):
    # M18: the unpinned explicit subject gets an isolated store and a
    # deterministic index so the real search_repository tool feeds the
    # module run's evidence package.
    enable_search(ARCH_MCP_ROOT, monkeypatch, tmp_path)
    result = rq.run_repository_query(
        QUESTION,
        repository=str(ARCH_MCP_ROOT),
        db_path=str(sandbox / "query.db"),
        trace_root=str(sandbox / "traces"),
    )

    assert rq.exit_code(result) == rq.EXIT_SUCCESS
    assert result.status == "COMPLETED"
    assert result.termination_reason == "SUCCESS"
    assert result.mode == "offline"
    assert result.answer
    assert result.conclusions
    assert result.collaboration_status == "passed"
    assert result.diagram_in_scope is True
    assert result.diagram_status == "passed"
    assert [step.label for step in result.steps] == [
        "architecture",
        "flow",
        "visualization",
    ]
    assert all(step.status == "completed" for step in result.steps)
    assert len(result.run_ids) == 3
    assert result.investigation_id
    assert result.failures == []
    assert result.artifacts_error is None
    assert any(".mcp.json" in line for line in result.evidence)
    assert len(result.trace_paths) == 3
    # an explicit, unpinned subject is identified but never pinned
    assert result.repository == "arch_mcp"
    assert result.pinned_commit == ""

    rows = {row.name: row for row in result.artifacts}
    assert rows["diagram_spec.json"].exists is True
    assert rows["architecture.mmd"].exists is True
    assert rows["run_summary.md"].exists is True
    if graphviz_available():
        assert rows["architecture.svg"].exists is True
        assert rows["architecture.svg"].note == ""
    else:
        assert rows["architecture.svg"].exists is False
        assert "NOT AVAILABLE" in rows["architecture.svg"].note


# -- invalid usage ----------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    (
        [""],
        ["   "],
        ["--query", ""],
        ["--query", "\t "],
    ),
)
def test_empty_query_is_a_usage_error(argv, monkeypatch, capsys):
    code, out, err = _run(argv, monkeypatch, capsys)

    assert code == rq.EXIT_USAGE
    assert err == "Error: repository query cannot be empty.\n"
    assert out == ""


def test_query_in_both_forms_is_a_usage_error(monkeypatch, capsys):
    code, out, err = _run(
        ["what is it", "--query", "and this"], monkeypatch, capsys
    )

    assert code == rq.EXIT_USAGE
    assert "not both" in err
    assert out == ""


def test_repository_flag_requires_a_directory(
    relative, monkeypatch, capsys
):
    code, out, err = _run(
        [QUESTION, "--repository", str(relative / "not-a-repo")],
        monkeypatch,
        capsys,
    )

    assert code == rq.EXIT_USAGE
    assert "is not a directory" in err
    assert out == ""


# -- a failed investigation -------------------------------------------------


def test_missing_target_fails_the_investigation(
    sandbox, monkeypatch, capsys
):
    empty = sandbox / "empty_repo"
    empty.mkdir()

    code, out, err = _cli_run(empty, monkeypatch, capsys)

    assert code == rq.EXIT_INVESTIGATION_FAILED
    assert err == ""
    assert "Status:     FAILED" in out
    assert "termination: NO_TARGET" in out
    assert "(no answer" in out
    assert "exit code: 1" in out


# -- the two scope flags ----------------------------------------------------


def test_no_diagram_answers_from_the_report(
    relative, monkeypatch, capsys
):
    code, out, err = _cli_run(
        ARCH_MCP_ROOT, monkeypatch, capsys, "--no-diagram"
    )

    assert code == 0
    assert "evidence-backed statement(s) established" in out
    assert "diagram:       skipped (--no-diagram)" in out
    assert "--no-diagram puts the diagram out of scope" in out
    assert "\u2713 architecture.svg" not in out
    assert "diagram_spec.json" not in out


def test_no_artifacts_writes_nothing_but_still_validates(
    sandbox, monkeypatch, capsys
):
    code, out, err = _cli_run(
        ARCH_MCP_ROOT, monkeypatch, capsys, "--no-artifacts"
    )

    assert code == 0
    assert "diagram:       passed" in out
    assert "validation still ran" in out
    assert not (sandbox / "artifacts").exists()


def test_verbose_prints_the_progress_lines(
    relative, monkeypatch, capsys
):
    code, out, err = _cli_run(
        ARCH_MCP_ROOT, monkeypatch, capsys, "--verbose"
    )

    assert code == 0
    assert "[1/3] S1" in out
    assert "[2/3] S2" in out
    assert "[3/3] S3" in out
    assert "[validation] collaboration passed; diagram passed" in out
    assert "[artifacts]" in out


# -- interactive ------------------------------------------------------------


def test_interactive_exit_leaves_without_a_query(
    sandbox, monkeypatch, capsys
):
    code, out, err = _run([], monkeypatch, capsys, inputs=["exit"])

    assert code == rq.EXIT_SUCCESS
    assert "interactive" in out
    assert "Final answer" not in out


def test_interactive_end_of_input_ends_the_session(
    sandbox, monkeypatch, capsys
):
    code, out, err = _run([], monkeypatch, capsys, inputs=[])

    assert code == rq.EXIT_SUCCESS
    assert "interactive" in out


def test_interactive_answers_then_quits(relative, monkeypatch, capsys):
    code, out, err = _run(
        ["--repository", str(ARCH_MCP_ROOT)],
        monkeypatch,
        capsys,
        inputs=[QUESTION, "quit"],
    )

    assert code == rq.EXIT_SUCCESS
    assert out.count("exit code: 0") == 1
    assert "Final answer" in out
    assert "Status:     COMPLETED" in out


def test_keyboard_interrupt_exits_130(sandbox, monkeypatch, capsys):
    def interrupt(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", interrupt)
    code = cli.main([])
    captured = capsys.readouterr()

    assert code == rq.EXIT_INTERRUPTED == 130
    assert "Interrupted." in captured.err


# -- the exit contract ------------------------------------------------------


def test_exit_code_contract():
    completed = rq.RepositoryQueryResult(status="COMPLETED")
    assert rq.exit_code(completed) == rq.EXIT_SUCCESS

    assert (
        rq.exit_code(rq.RepositoryQueryResult(status=VALIDATION_FAILED))
        == rq.EXIT_VALIDATION_FAILED
        == 3
    )
    assert (
        rq.exit_code(
            rq.RepositoryQueryResult(
                status="COMPLETED", diagram_status="failed"
            )
        )
        == 3
    )
    assert (
        rq.exit_code(
            rq.RepositoryQueryResult(
                status="COMPLETED",
                diagram_in_scope=False,
                diagram_status="failed",
            )
        )
        == rq.EXIT_SUCCESS
    )
    assert (
        rq.exit_code(rq.RepositoryQueryResult(status="INCOMPLETE"))
        == rq.EXIT_INVESTIGATION_FAILED
        == 1
    )
    assert (
        rq.exit_code(
            rq.RepositoryQueryResult(
                status="COMPLETED", artifacts_error="disk full"
            )
        )
        == 1
    )


# -- snapshot verification, paths, routing ----------------------------------


def test_snapshot_metadata_is_read_from_the_m1_record():
    metadata = rq.read_snapshot_metadata(
        rq.default_subject_repository()
    )
    assert metadata["commit"] == rq.SUBJECT_COMMIT
    assert metadata["name"] == "sample-Agentic-Ai-Data-Operations"

    actual, source, error = rq.verify_subject_snapshot(
        rq.default_subject_repository()
    )
    assert error is None
    assert actual == rq.SUBJECT_COMMIT
    assert source in ("git", "snapshot")


def test_display_path_is_project_relative():
    assert rq.display_path(rq.PROJECT_ROOT / "README.md") == "README.md"
    assert rq.display_path(None) == ""
    outside = "/definitely/not/under/the/project/root/x.json"
    assert rq.display_path(outside) == outside


def test_resolve_read_target_reports_missing_targets(tmp_path):
    root, target, error = rq.resolve_read_target(str(ARCH_MCP_ROOT))
    assert error is None
    assert root == ARCH_MCP_ROOT
    assert target == "agents/collector.py"

    empty = tmp_path / "empty"
    empty.mkdir()
    _, target, error = rq.resolve_read_target(str(empty))
    assert target is None
    assert "no readable target" in error

    _, target, error = rq.resolve_read_target(str(tmp_path / "gone"))
    assert target is None
    assert "is not a directory" in error


class _StubSpecialist:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.last_trace = None

    def run_result(self, question):
        self.calls += 1
        return "{}-{}".format(self.name, self.calls)


def test_router_honours_the_plan_step_ids():
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    assert router.run_result("step: S1 \u2014 document it") == "arch-1"
    assert router.run_result("step: S2 \u2014 trace it") == "flow-1"
    assert router.run_result("step: arch \u2014 again") == "arch-2"
    assert router.calls == ["S1", "S2", "arch"]
    assert router.labels["S1"] == "architecture"
    assert router.labels["S2"] == "flow"


def test_router_falls_back_to_the_goal_then_the_memo():
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)
    question = "step: X7 \u2014 list the components and relationships"

    router.run_result(question)
    router.run_result(question)

    assert architecture.calls == 2
    assert flow.calls == 0
    assert router.labels["X7"] == "architecture"
    assert router.calls == ["X7", "X7"]


def test_router_assigns_by_order_then_refuses_to_guess():
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    router.run_result("step: Q1 \u2014 do something vague")
    router.run_result("step: Q2 \u2014 do something else vague")

    assert architecture.calls == 1
    assert flow.calls == 1

    with pytest.raises(rq.RepositoryQueryRouterError) as excinfo:
        router.run_result("step: Q3 \u2014 still no useful goal")
    assert "unroutable" in str(excinfo.value)
    assert "Q3" in str(excinfo.value)


def test_router_routes_five_investigator_steps_by_goal():
    """A five-step plan routes entirely on goal content: two
    specialists never imply a two-investigator-step ceiling."""
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    router.run_result(
        "step: P1 \u2014 document the architecture components"
    )
    router.run_result(
        "step: P2 \u2014 trace the dependency path in this repository"
    )
    router.run_result(
        "step: P3 \u2014 explain how the tools and MCP servers "
        "interact with the agents"
    )
    router.run_result(
        "step: P4 \u2014 identify the modules and their "
        "responsibilities"
    )
    router.run_result(
        "step: P5 \u2014 describe the communication protocol between "
        "the agents"
    )

    assert router.labels == {
        "P1": "architecture",
        "P2": "flow",
        "P3": "flow",
        "P4": "architecture",
        "P5": "flow",
    }
    assert architecture.calls == 2
    assert flow.calls == 3


def test_router_lets_many_steps_share_one_specialist():
    """Routing is per goal, never per position: several steps may
    honestly go to the same specialist."""
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    router.run_result(
        "step: P1 \u2014 identify the components of this system"
    )
    router.run_result(
        "step: P2 \u2014 trace the dependency path through the "
        "repository"
    )
    router.run_result(
        "step: P3 \u2014 list the services and their roles"
    )

    assert router.labels == {
        "P1": "architecture",
        "P2": "flow",
        "P3": "architecture",
    }
    assert architecture.calls == 2
    assert flow.calls == 1


def test_router_routes_architecture_family_goals():
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    router.run_result(
        "step: A1 \u2014 document the architecture and its "
        "components"
    )
    router.run_result(
        "step: A2 \u2014 identify the modules and their "
        "relationships"
    )

    assert router.labels == {
        "A1": "architecture",
        "A2": "architecture",
    }
    assert architecture.calls == 2
    assert flow.calls == 0


def test_router_routes_tools_mcp_and_interaction_goals():
    """Tools, MCP servers, protocols, and interaction goals belong
    to the flow specialist (the family the failing live run's
    question uses)."""
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    router.run_result(
        "step: F1 \u2014 how do the agents invoke the MCP servers"
    )
    router.run_result(
        "step: F2 \u2014 which protocol do the tools speak when they "
        "interact"
    )
    router.run_result(
        "step: F3 \u2014 list the dependencies between the agents "
        "and their tools"
    )

    assert router.labels == {
        "F1": "flow",
        "F2": "flow",
        "F3": "flow",
    }
    assert flow.calls == 3
    assert architecture.calls == 0


def test_router_forwards_the_request_level_tool_budget():
    """M20 participant contract: the router is an orchestrator
    participant, so a request-level per-attempt budget must reach the
    routed specialist VERBATIM. This forwarding was missing — the
    executor's ``run_result(question, max_tool_calls=...)`` call
    raised ``TypeError`` at the router before any execution, failing
    every investigator task (zero tool calls, no trace)."""
    class _RecordingSpecialist:
        def __init__(self):
            self.calls = []
            self.last_trace = None

        def run_result(self, question, *, max_tool_calls=None):
            self.calls.append((question, max_tool_calls))
            return "recorded"

    architecture = _RecordingSpecialist()
    flow = _RecordingSpecialist()
    router = rq.QueryRouter(architecture, flow)

    outcome = router.run_result(
        "step: S1 \u2014 document it", max_tool_calls=12
    )

    assert outcome == "recorded"
    assert architecture.calls == [("step: S1 \u2014 document it", 12)]
    assert flow.calls == []

    # Budget unset -> the EXACT pre-M20 one-argument call, so legacy
    # participants that never accepted the override keep working.
    legacy = _StubSpecialist("legacy-arch")
    legacy_router = rq.QueryRouter(legacy, _StubSpecialist("flow"))
    assert legacy_router.run_result(
        "step: L1 \u2014 document it"
    ) == "legacy-arch-1"

    # A budget against a participant that rejects it fails LOUDLY at
    # the call (never a silently ignored budget) — the executor's
    # boundary policy, preserved through the router.
    with pytest.raises(TypeError):
        legacy_router.run_result(
            "step: L1 \u2014 document it", max_tool_calls=3
        )


def test_observed_agent_forwards_the_request_level_tool_budget():
    """``ObservedAgent`` wraps every specialist's Agent in the
    executor participant chain (``build_architect`` /
    ``build_flow_investigator``) — it must accept and forward the M20
    per-attempt budget unchanged, not drop it."""
    class _RecordingInner:
        def __init__(self):
            self.calls = []
            self.last_trace = None

        def run_result(self, question, *, max_tool_calls=None):
            self.calls.append((question, max_tool_calls))
            return "inner"

    inner = _RecordingInner()
    observed = rq.ObservedAgent(inner)

    assert observed.run_result("q", max_tool_calls=5) == "inner"
    assert inner.calls == [("q", 5)]

    # Budget unset -> exact one-argument call for a pre-M20 inner.
    class _LegacyInner:
        last_trace = None

        def run_result(self, question):
            return "legacy-ok"

    assert rq.ObservedAgent(_LegacyInner()).run_result("q") == (
        "legacy-ok"
    )


def test_router_keeps_a_step_stable_across_later_routing():
    """A step id keeps its specialist for the whole run, even after
    the order fallback is exhausted — a retry must never re-route."""
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)

    router.run_result("step: Q1 \u2014 do something vague")
    router.run_result("step: Q2 \u2014 do something else vague")
    router.run_result("step: Q1 \u2014 do something vague")

    assert router.calls == ["Q1", "Q2", "Q1"]
    assert router.labels["Q1"] == "architecture"
    assert router.labels["Q2"] == "flow"


def test_unroutable_step_fails_identically_on_every_attempt():
    """The refusal is deterministic and leaves no half-recorded
    decision behind: the bounded retry sees the same clear error and
    the step never claims a specialist label."""
    architecture = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = rq.QueryRouter(architecture, flow)
    question = "step: Q3 \u2014 still no useful goal"

    router.run_result("step: Q1 \u2014 do something vague")
    router.run_result("step: Q2 \u2014 do something else vague")

    with pytest.raises(rq.RepositoryQueryRouterError) as first:
        router.run_result(question)
    with pytest.raises(rq.RepositoryQueryRouterError) as second:
        router.run_result(question)

    assert "Q3" in str(first.value)
    assert "Q3" in str(second.value)
    assert "Q3" not in router.labels
    assert router.calls == ["Q1", "Q2"]


def test_empty_module_query_raises_the_documented_error():
    with pytest.raises(rq.RepositoryQueryError):
        rq.run_repository_query("   ")


def test_live_bindings_use_real_pinned_context_and_graph_tools(
    tmp_path, monkeypatch
):
    calls = []

    def context(arguments, *, output_root=None, snapshot_id=None):
        calls.append(("context", arguments, output_root, snapshot_id))
        return rq.ToolResult(
            success=True,
            tool="get_architecture_context",
            data={"text": "real context"},
        )

    def graph(arguments, *, output_root=None, snapshot_id=None):
        calls.append(("graph", arguments, output_root, snapshot_id))
        return rq.ToolResult(
            success=True,
            tool="query_engineering_graph",
            data={"nodes": ["real graph"]},
        )

    monkeypatch.setattr(rq, "get_architecture_context", context)
    monkeypatch.setattr(rq, "query_engineering_graph_tool", graph)

    output_root = tmp_path / "artifacts"
    bindings = rq.live_bindings(
        tmp_path,
        output_root=output_root,
        snapshot_id="pinned_snapshot",
    )
    context_result = bindings["get_architecture_context"]({})
    graph_result = bindings["query_engineering_graph"]({"start_node_id": None})

    assert context_result.data == {"text": "real context"}
    assert graph_result.data == {"nodes": ["real graph"]}
    assert calls == [
        ("context", {}, output_root, "pinned_snapshot"),
        ("graph", {"start_node_id": None}, output_root, "pinned_snapshot"),
    ]





def test_live_bindings_read_real_pinned_context_and_graph_artifacts():
    artifacts = rq.default_subject_repository().parents[2]
    # The subject's persisted M6 context and M11 graph are required for
    # the live binding integration test; skip only if a checkout omits
    # generated artifacts.
    context_path = (
        artifacts / rq.SUBJECT_SNAPSHOT_ID / "architecture_context"
        / "architecture_context.md"
    )
    graph_path = (
        artifacts / rq.SUBJECT_SNAPSHOT_ID / "engineering"
        / "engineering_graph.json"
    )
    if not context_path.is_file() or not graph_path.is_file():
        pytest.skip("pinned architecture context / graph artifacts are absent")

    bindings = rq.live_bindings(
        rq.default_subject_repository(),
        output_root=artifacts,
    )
    context = bindings["get_architecture_context"]({})
    graph = bindings["query_engineering_graph"]({})

    assert context.success is True
    assert context.data and context.data.get("context")
    assert graph.success is True
    assert graph.data and graph.data.get("kinds")
    assert graph.source is not None
    assert graph.source["snapshot_id"] == rq.SUBJECT_SNAPSHOT_ID


def test_live_bindings_refuse_to_borrow_context_from_another_repository(
    tmp_path, monkeypatch
):
    def unexpected(*args, **kwargs):
        raise AssertionError("must not read an unrelated snapshot")

    monkeypatch.setattr(rq, "get_architecture_context", unexpected)
    monkeypatch.setattr(rq, "query_engineering_graph_tool", unexpected)

    bindings = rq.live_bindings(
        tmp_path / "external-repository",
        output_root=tmp_path / "artifacts",
    )

    context = bindings["get_architecture_context"]({})
    graph = bindings["query_engineering_graph"]({})

    assert context.success is False
    assert context.error is not None
    assert context.error.code == "SNAPSHOT_CONTEXT_UNAVAILABLE"
    assert graph.success is False
    assert graph.error is not None
    assert graph.error.code == "SNAPSHOT_GRAPH_UNAVAILABLE"


def test_live_bindings_derive_snapshot_id_from_repository_layout(tmp_path):
    artifacts = tmp_path / "artifacts"
    repository = artifacts / "snapshot_123" / "repository" / "repo"
    repository.mkdir(parents=True)

    assert rq._snapshot_id_for_repository_root(repository, artifacts) == (
        "snapshot_123"
    )
    assert rq._snapshot_id_for_repository_root(
        tmp_path / "external-repository", artifacts
    ) is None


# -- the M17 execution budget at this composition root ---------------------


def test_specialists_receive_bounded_m17_budget_and_early_reassessment(
    sandbox, monkeypatch
):
    """The composition root keeps the 64-call hard ceiling but asks
    each investigator to reassess after 8 calls. One targeted follow-up
    then normally caps a healthy specialist run at 9 calls, leaving
    enough room for a multi-step plan plus validation."""
    assert rq.M17_MAX_TOOL_CALLS == 64
    assert rq.M17_EXECUTION_CONFIG.max_tool_calls == 64
    assert rq.M17_EXECUTION_CONFIG.soft_tool_call_target == 8
    assert rq.M17_EXECUTION_CONFIG.finalize_on_boundary is True

    captured = []
    original = rq.Agent

    def spy(*args, **kwargs):
        captured.append(kwargs.get("execution_config"))
        return original(*args, **kwargs)

    monkeypatch.setattr(rq, "Agent", spy)

    scope = "sample-Agentic-Ai-Data-Operations"
    manager = rq.build_memory_manager()
    memory_context = rq.build_memory_context(scope, "c" * 40)
    engine = rq.build_engine(scope)
    sink = trace_sink_module.JsonFileTraceSink()
    never_called = object()

    rq.build_architect(never_called, sink, manager, memory_context, engine)
    rq.build_flow_investigator(
        never_called, sink, manager, memory_context, engine
    )
    rq.build_visualizer(never_called, sink, manager, memory_context, engine)

    assert len(captured) == 3
    assert all(config is rq.M17_EXECUTION_CONFIG for config in captured)
    assert all(config.max_tool_calls == 64 for config in captured)
    assert all(config.soft_tool_call_target == 8 for config in captured)
    assert all(config.finalize_on_boundary for config in captured)


def test_orchestrator_receives_the_64_call_global_budget(
    sandbox, monkeypatch, capsys
):
    """The whole-run hard ceiling remains 64. With the composition
    root's 8-call reassessment target, up to five 9-call investigator
    runs plus one 9-call retry and two validation calls fit (56 total)."""
    assert rq.M17_ORCHESTRATION_CONFIG.max_total_tool_calls == 64
    assert 5 * (rq.M17_EXECUTION_CONFIG.soft_tool_call_target + 1) + (
        rq.M17_EXECUTION_CONFIG.soft_tool_call_target + 1
    ) + 2 < rq.M17_ORCHESTRATION_CONFIG.max_total_tool_calls

    seen = []
    original = rq.Orchestrator

    def spy(*args, **kwargs):
        seen.append(kwargs.get("config"))
        return original(*args, **kwargs)

    monkeypatch.setattr(rq, "Orchestrator", spy)
    code, out, err = _cli_default(monkeypatch, capsys)

    assert code == 0
    assert len(seen) == 1
    assert seen[0] is rq.M17_ORCHESTRATION_CONFIG


# -- the router contract at this composition root ---------------------------


def test_failed_step_is_not_reported_as_the_visualization_agent(
    sandbox, monkeypatch
):
    """The live-run failure shape, offline and deterministic: a
    five-step plan whose third investigator goal classifies nowhere.
    The run stops at P3 with a clear, retry-bounded router error —
    and the failed step is NEVER read back as though the
    visualization agent had been assigned to it."""
    enable_search(ARCH_MCP_ROOT, monkeypatch, sandbox)

    plan = InvestigationPlan(
        objective=QUESTION,
        steps=[
            PlanStep(
                id="P1",
                goal="Document the architecture components and "
                "their relationships.",
            ),
            PlanStep(
                id="P2",
                goal="Trace the dependency path through this "
                "repository.",
            ),
            PlanStep(
                id="P3",
                goal="Do a final review of everything.",
            ),
            PlanStep(
                id="P4",
                goal="Look at the repository once more.",
            ),
            PlanStep(
                id="P5",
                goal="Interpret the investigated findings as one "
                "diagram.",
            ),
        ],
    )

    def _five_step_plan(planner, question):
        rq.validate_plan(plan)
        return plan

    monkeypatch.setattr(rq, "_build_plan", _five_step_plan)

    p3_attempts = []
    original_router = rq.QueryRouter

    class _SpyRouter(original_router):
        def run_result(self, question):
            if "step: P3 " in question:
                p3_attempts.append(question)
            return super().run_result(question)

    monkeypatch.setattr(rq, "QueryRouter", _SpyRouter)

    result = rq.run_repository_query(
        QUESTION,
        repository=str(ARCH_MCP_ROOT),
        db_path=str(sandbox / "query.db"),
        trace_root=str(sandbox / "traces"),
    )

    assert rq.exit_code(result) == rq.EXIT_INVESTIGATION_FAILED
    assert result.status == "INCOMPLETE"
    # no collaboration verdict and no diagram: the run stopped first
    assert result.collaboration_status is None
    assert result.diagram_status is None

    by_id = {step.step_id: step for step in result.steps}
    assert by_id["P1"].label == "architecture"
    assert by_id["P2"].label == "flow"
    # the refused step: routed to nothing, never to the visualizer
    assert by_id["P3"].label == "unrouted"
    assert by_id["P3"].role == "INVESTIGATOR"
    assert by_id["P3"].status == "failed"
    # the refusal happened before any specialist ran
    assert by_id["P3"].run_id is None
    assert by_id["P3"].output_type == ""
    # the steps blocked behind it were never assigned at all
    assert by_id["P4"].label == ""
    assert by_id["P4"].status == "blocked"
    assert by_id["P5"].label == ""
    assert by_id["P5"].status == "blocked"

    # the orchestrator's bounded retry is unchanged: one retry only
    assert len(p3_attempts) == 2
