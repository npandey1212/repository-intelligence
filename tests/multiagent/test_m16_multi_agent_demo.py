"""M16 — multi-agent demonstration, offline (spec §15–§35).

Scripted doubles only — no network, no API key:

 * one workflow run tells the whole story: permissions → typed
   report → six-message handoff → report-constrained finding →
   eight deterministic checks → status/events/traces
 * ``main`` default mode is OFFLINE and succeeds without
  OPENAI_API_KEY; ``--live`` without a key fails with exit 6
 * fixture mode (``--repository``) binds the read-only tools to a
  foreign root through the dispatcher seam
 * the demo is a composition root — a source scan shows no second
  agent framework, no broker, no sqlite3 import in the demo itself
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

DEMO_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_m16_multi_agent_demo.py"
)
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "construction"


def _load_demo(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # register before exec: pydantic resolves annotations through
    # sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo(DEMO_PATH, "agentrun_m16_multi_agent_demo")


def _components(tmp_path, suffix=""):
    """Fresh offline composition over temporary memory/context/trace
    roots (never the repository's runtime/ during tests)."""
    manager = demo.build_memory_manager(str(tmp_path / ("m16{}.db".format(suffix))))
    memory_context = demo.build_memory_context(
        demo.REPOSITORY, demo.OFFLINE_COMMIT
    )
    engine = demo.build_engine(demo.REPOSITORY, str(tmp_path / "snapshots"))
    sink = demo.JsonFileTraceSink(str(tmp_path / "traces"))
    return manager, memory_context, engine, sink


def _run_demo(tmp_path, suffix="", stream=None):
    question = demo.DEFAULT_QUESTION
    _, evidence_path, _ = demo.demo_read_target(None)
    investigator = demo.build_offline_investigator(
        question, demo.REPOSITORY, demo.OFFLINE_COMMIT, evidence_path
    )
    analyst = demo.build_offline_analyst(
        question, demo.REPOSITORY, demo.OFFLINE_COMMIT, evidence_path
    )
    manager, memory_context, engine, sink = _components(tmp_path, suffix)
    return demo.run_demo(
        question,
        demo.REPOSITORY,
        demo.OFFLINE_COMMIT,
        investigator,
        analyst,
        sink,
        manager,
        memory_context,
        engine,
        trace_root=tmp_path / "traces",
        stream=stream or io.StringIO(),
    )


# -- module contract / composition root ------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "RULE",
        "SECTION",
        "EXIT_SUCCESS",
        "EXIT_DEMO_FAILED",
        "EXIT_EXECUTION_FAILED",
        "DEFAULT_QUESTION",
        "OFFLINE_COMMIT",
        "OFFLINE_INVESTIGATION_ID",
        "run_permissions_phase",
        "run_investigation_phase",
        "run_handoff_phase",
        "run_analysis_phase",
        "run_validation_phase",
        "run_status_phase",
        "run_demo",
        "main",
    ):
        assert hasattr(demo, name), name


def test_composition_root_no_second_framework_no_sqlite3():
    source = DEMO_PATH.read_text(encoding="utf-8")
    for banned in (
        "langgraph",
        "crewai",
        "autogen",
        "sqlite3",
        "import redis",
    ):
        assert banned not in source, banned
    assert "MultiAgentRunner" in source
    assert "restricted_dispatcher" in source


# -- scripted doubles -------------------------------------------------------


def test_offline_models_answer_only_what_was_scripted():
    investigator = demo.build_offline_investigator(
        "q", "r", "c" * 40, "README.md"
    )
    first = investigator.complete(None)
    second = investigator.complete(None)
    assert first.finish_reason == "tool_calls"
    assert second.structured_output["investigation_id"] == (
        demo.OFFLINE_INVESTIGATION_ID
    )
    with pytest.raises(AssertionError):
        investigator.complete(None)


# -- read target selection --------------------------------------------------


def test_read_target_default_mode_uses_project_files():
    absolute, relative, bound = demo.demo_read_target(None)
    assert absolute is not None and absolute.is_file()
    assert relative is not None
    assert bound is None  # pinned snapshot mode: no bindings


def test_read_target_fixture_mode_binds_foreign_root(tmp_path):
    root = tmp_path / "foreign"
    (root / "src").mkdir(parents=True)
    (root / "src" / "service.py").write_text(
        "# foreign service\n", encoding="utf-8"
    )
    absolute, relative, bound = demo.demo_read_target(str(root))
    assert absolute == root / "src" / "service.py"
    assert relative == "src/service.py"
    assert bound == str(root)
    bindings = demo.fixture_bindings(Path(bound))
    assert set(bindings) == {"read_file", "search_repository"}
    result = bindings["read_file"]({"path": "src/service.py"})
    assert result.success is True
    assert "foreign service" in (result.data.get("content") or "")


def test_read_target_missing_repository_returns_none(tmp_path):
    absolute, relative, bound = demo.demo_read_target(
        str(tmp_path / "does-not-exist")
    )
    assert (absolute, relative, bound) == (None, None, None)


# -- permissions phase ------------------------------------------------------


def test_permissions_phase_refuses_probe_and_reports_yes():
    stream = io.StringIO()
    outcome = demo.run_permissions_phase(stream=stream)
    assert outcome["ok"] is True
    assert "TOOL_NOT_FOUND (refused)" in stream.getvalue()


# -- whole story ------------------------------------------------------------


def test_run_demo_tells_the_whole_story_offline(tmp_path):
    outcome = _run_demo(tmp_path)
    assert outcome["ok"] is True, outcome
    state = outcome["state"]
    assert state.status == demo.COMPLETED
    assert [message.message_type for message in state.messages] == [
        "INVESTIGATION_REQUEST",
        "INVESTIGATION_RESULT",
        "ANALYSIS_REQUEST",
        "ANALYSIS_RESULT",
        "VALIDATION_REQUEST",
        "VALIDATION_RESULT",
    ]
    assert state.validation.status == "passed"
    assert len(state.validation.checks) == 8
    assert all(check.ok for check in state.validation.checks)
    assert state.tool_failures == []
    assert state.finding.investigation_id == state.report.investigation_id
    for name in (
        "permissions",
        "investigation",
        "handoff",
        "analysis",
        "validation",
        "status",
    ):
        assert outcome[name]["ok"] is True, name


def test_run_demo_is_deterministic_across_runs(tmp_path):
    first = _run_demo(tmp_path / "a", suffix="_a")
    second = _run_demo(tmp_path / "b", suffix="_b")
    assert first["ok"] and second["ok"]
    one, two = first["state"], second["state"]
    assert [m.message_type for m in one.messages] == [
        m.message_type for m in two.messages
    ]
    assert [c.rule for c in one.validation.checks] == [
        c.rule for c in two.validation.checks
    ]
    assert [
        e["operation"] for e in one.collaboration_events
    ] == [e["operation"] for e in two.collaboration_events]
    assert [
        e["sequence"] for e in one.collaboration_events
    ] == [e["sequence"] for e in two.collaboration_events]
    assert one.status == two.status == demo.COMPLETED


def test_run_demo_reports_analyst_isolation_in_state(tmp_path):
    outcome = _run_demo(tmp_path)
    # The Analyst ran with zero tools: its trace records no executed
    # tool results at all — every conclusion came from the report.
    state = outcome["state"]
    analyst_trace = outcome["analyst"].last_trace
    analyst_results = [
        result
        for iteration in analyst_trace.iterations
        for result in iteration.tool_results
    ]
    assert analyst_results == []
    assert state.validation.status == "passed"


# -- main -------------------------------------------------------------------


def test_main_offline_succeeds_without_api_key(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--db",
            str(tmp_path / "m16.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "exit code: 0" in captured.out
    assert "markdown section present" in captured.out
    assert "final status:   COMPLETED" in captured.out


def test_main_fixture_mode_succeeds(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--repository",
            str(FIXTURE_ROOT),
            "--db",
            str(tmp_path / "m16.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "target: src/service.py" in captured.out


def test_main_missing_repository_fails_cleanly(capsys, tmp_path):
    code = demo.main(
        ["--repository", str(tmp_path / "nope")]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_DEMO_FAILED
    assert "not a directory" in captured.err


def test_main_repository_without_target_file_fails(capsys, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code = demo.main(["--repository", str(empty)])
    captured = capsys.readouterr()
    assert code == demo.EXIT_DEMO_FAILED
    assert "no readable target file" in captured.err


def test_main_live_without_api_key_fails_with_six(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(["--live"])
    captured = capsys.readouterr()
    assert code == demo.EXIT_EXECUTION_FAILED
    assert "OPENAI_API_KEY" in captured.err
