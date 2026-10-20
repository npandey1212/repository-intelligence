"""Flow / Dependency Investigator demonstration tests (offline, never
the network).

Same discipline as ``tests/agents/test_architecture_investigator_demo.py``:
the phase functions and ``main`` are driven with scripted doubles over
temporary memory/context/trace roots (never the repository's
``runtime/``).
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

DEMO_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_flow_dependency_investigator_demo.py"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARCH_MCP_ROOT = FIXTURES / "arch_mcp"
ARCH_SIMPLE_ROOT = FIXTURES / "arch_simple"
FLOW_CHAIN_ROOT = FIXTURES / "flow_chain"


def _load_demo(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # register before exec: pydantic resolves annotations through
    # sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo(
    DEMO_PATH, "agentrun_flow_dependency_investigator_demo"
)


def _components(tmp_path, label: str, commit: str):
    manager = demo.build_memory_manager(str(tmp_path / "flow.db"))
    memory_context = demo.build_memory_context(label, commit)
    engine = demo.build_engine(label, str(tmp_path / "snapshots"))
    sink = demo.JsonFileTraceSink(str(tmp_path / "traces"))
    return manager, memory_context, engine, sink


def _run_demo(tmp_path, repository=None, stream=None):
    """Offline story over a fixture root; returns the outcome plus the
    participant handles the tests assert against."""
    root, target, _ = demo.demo_read_target(repository)
    assert target is not None
    label = root.name
    commit = demo.OFFLINE_COMMIT
    question = demo.DEFAULT_QUESTION
    story = demo.demo_story(root, target)
    direct = demo.build_offline_direct_model(
        question, label, commit, story
    )
    investigator = demo.build_offline_investigator_model(
        question, label, commit, story
    )
    analyst = demo.build_offline_analyst_model(
        question, label, commit, story
    )
    planner = demo.build_offline_planner_model(question)
    manager, memory_context, engine, sink = _components(
        tmp_path, label, commit
    )
    outcome = demo.run_demo(
        question,
        label,
        commit,
        story,
        direct,
        planner,
        investigator,
        analyst,
        sink,
        manager,
        memory_context,
        engine,
        demo.fixture_bindings(root),
        tmp_path / "traces",
        stream=stream if stream is not None else io.StringIO(),
    )
    return outcome, root, target, story


# -- module surface ---------------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "RULE",
        "SECTION",
        "EXIT_SUCCESS",
        "EXIT_DEMO_FAILED",
        "EXIT_EXECUTION_FAILED",
        "DEFAULT_QUESTION",
        "OFFLINE_COMMIT",
        "FLOW_DEPENDENCY_TOOLS",
        "demo_story",
        "run_scope_phase",
        "run_direct_phase",
        "run_evidence_phase",
        "run_view_phase",
        "run_orchestrated_phase",
        "run_trace_phase",
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
    assert "Orchestrator" in source
    assert "restricted_dispatcher" in source
    assert "FlowDependencyInvestigator" in source


# -- scripted doubles -------------------------------------------------------


def test_offline_models_answer_only_what_was_scripted():
    story = demo.demo_story(ARCH_MCP_ROOT, "agents/collector.py")
    direct = demo.build_offline_direct_model(
        "q", "r", "c" * 40, story
    )
    first = direct.complete(None)
    second = direct.complete(None)
    assert first.finish_reason == "tool_calls"
    assert second.structured_output["path_status"] == "FOUND"
    assert second.structured_output["source"] == "collector"
    assert second.structured_output["target"] == "records"
    with pytest.raises(AssertionError):
        direct.complete(None)


def test_offline_analyst_echoes_the_derived_investigation_id():
    story = demo.demo_story(ARCH_MCP_ROOT, "agents/collector.py")
    derived = demo.FlowDependencyReport(
        **demo.offline_step_endpoints_payload(
            "q", "r", "c" * 40, story
        )
    ).derived_investigation_id()
    analyst = demo.build_offline_analyst_model(
        "q", "r", "c" * 40, story
    )
    response = analyst.complete(None)
    assert response.structured_output["investigation_id"] == derived
    assert derived.startswith("inv-flow-")


# -- offline story ----------------------------------------------------------


def test_demo_story_mcp_root_reports_the_chain():
    story = demo.demo_story(ARCH_MCP_ROOT, "agents/collector.py")
    assert story["mcp"] is True
    assert story["source"] == "collector"
    assert story["tool"] == "fetch_record"
    assert story["server"] == "records"
    assert [
        (source, target, kind)
        for source, target, kind, _ in story["legs"]
    ] == [
        ("collector", "fetch_record", "IMPORTS"),
        ("fetch_record", "records", "EXPOSED_BY"),
    ]
    assert story["reads"] == ["agents/collector.py", ".mcp.json"]


def test_demo_story_without_mcp_is_honest_about_the_destination():
    story = demo.demo_story(
        FLOW_CHAIN_ROOT, "src/producer.py"
    )
    assert story["mcp"] is False
    assert story["destination"] is None
    assert story["legs"] == []
    assert story["reads"] == ["src/producer.py"]


# -- read target selection --------------------------------------------------


def test_read_target_default_mode_uses_the_mcp_fixture():
    root, target, bound = demo.demo_read_target(None)
    assert root == ARCH_MCP_ROOT
    assert target == "agents/collector.py"
    assert bound is not None  # bindings always point at the root


def test_read_target_fixture_mode_binds_foreign_root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "producer.py").write_text(
        "class Producer:\n    pass\n", encoding="utf-8"
    )
    root, target, bound = demo.demo_read_target(str(tmp_path))
    assert target == "src/producer.py"
    bindings = demo.fixture_bindings(root)
    result = bindings["read_file"]({"path": "src/producer.py"})
    assert result.success is True
    assert "class Producer" in str(result.data)


def test_read_target_missing_repository_returns_none(tmp_path):
    root, target, bound = demo.demo_read_target(
        str(tmp_path / "nope")
    )
    assert root is None
    assert target is None
    assert bound is not None


# -- phases over fixtures ---------------------------------------------------


def test_run_demo_tells_the_whole_story_offline(tmp_path):
    outcome, _, target, story = _run_demo(tmp_path)
    assert story["mcp"] is True
    assert target == "agents/collector.py"
    for name in (
        "scope",
        "direct",
        "evidence",
        "view",
        "orchestrated",
        "trace",
    ):
        assert outcome[name]["ok"] is True, name
    assert outcome["ok"] is True
    report = outcome["direct"]["report"]
    assert report.status == "complete"
    assert report.path_status == "FOUND"
    assert report.source == "collector"
    assert report.target == "records"
    assert [
        "{} {} {}".format(
            step.source, step.relationship_type, step.target
        )
        for step in report.path
    ] == [
        "collector IMPORTS fetch_record",
        "fetch_record EXPOSED_BY records",
    ]
    run = outcome["orchestrated"]["run"]
    assert run.validation.status == "passed"
    assert run.finding.investigation_id == (
        report.derived_investigation_id()
    )


def test_run_demo_keeps_every_intermediate_hop(tmp_path):
    outcome, _, _, story = _run_demo(tmp_path)
    statements = [
        finding.statement
        for finding in outcome["orchestrated"]["run"].report.findings
    ]
    assert "collector IMPORTS fetch_record [observed]" in statements
    assert "fetch_record EXPOSED_BY records [observed]" in statements
    assert not any(
        "collector EXPOSED_BY records" in statement
        for statement in statements
    )
    paths = {
        ref.path
        for ref in outcome["orchestrated"]["run"].report.evidence_refs
    }
    assert ".mcp.json" in paths
    assert "agents/collector.py" in paths


def test_run_demo_without_mcp_reports_unknown_destination(tmp_path):
    outcome, _, target, story = _run_demo(
        tmp_path, repository=str(FLOW_CHAIN_ROOT)
    )
    assert story["mcp"] is False
    assert target == "src/producer.py"
    assert outcome["ok"] is True
    report = outcome["direct"]["report"]
    assert report.path_status == "UNKNOWN"
    assert report.path == []
    assert report.target is None
    assert report.uncertainties
    run = outcome["orchestrated"]["run"]
    assert run.validation.status == "passed"


def test_run_demo_is_deterministic_across_runs(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first, _, _, _ = _run_demo(tmp_path / "one")
    second, _, _, _ = _run_demo(tmp_path / "two")
    for name in (
        "scope",
        "direct",
        "evidence",
        "view",
        "orchestrated",
        "trace",
    ):
        assert first[name]["ok"] == second[name]["ok"], name
    first_report = first["direct"]["report"]
    second_report = second["direct"]["report"]
    assert (
        first_report.derived_investigation_id()
        == second_report.derived_investigation_id()
    )
    assert (
        [step.target for step in first_report.path]
        == [step.target for step in second_report.path]
    )
    assert (
        [f.statement for f in first["orchestrated"]["run"].report.findings]
        == [f.statement for f in second["orchestrated"]["run"].report.findings]
    )


def test_scope_phase_checks_prompt_and_tools(tmp_path):
    manager, memory_context, engine, sink = _components(
        tmp_path, ARCH_MCP_ROOT.name, demo.OFFLINE_COMMIT
    )
    root, target, _ = demo.demo_read_target(None)
    story = demo.demo_story(root, target)
    specialist = demo.build_specialist(
        demo.build_offline_direct_model(
            demo.DEFAULT_QUESTION,
            ARCH_MCP_ROOT.name,
            demo.OFFLINE_COMMIT,
            story,
        ),
        sink,
        manager,
        memory_context,
        engine,
        bindings=demo.fixture_bindings(root),
    )
    stream = io.StringIO()
    outcome = demo.run_scope_phase(
        demo.DEFAULT_QUESTION,
        ARCH_MCP_ROOT.name,
        demo.OFFLINE_COMMIT,
        story,
        specialist,
        stream=stream,
    )
    assert outcome["ok"] is True
    prompt = demo.FLOW_DEPENDENCY_INVESTIGATOR_PROMPT_V1
    assert "orchestrat" not in prompt.lower()
    assert "planner" not in prompt.lower()
    assert specialist.tools == list(demo.FLOW_DEPENDENCY_TOOLS)
    assert (
        demo.FLOW_PROMPT_VERSION
        == "V{}".format(
            demo.FLOW_PROMPT_VERSION[1:]
        )
    )


def test_view_phase_prints_the_flow_chain(tmp_path):
    manager, memory_context, engine, sink = _components(
        tmp_path, ARCH_MCP_ROOT.name, demo.OFFLINE_COMMIT
    )
    root, target, _ = demo.demo_read_target(None)
    story = demo.demo_story(root, target)
    specialist = demo.build_specialist(
        demo.build_offline_direct_model(
            demo.DEFAULT_QUESTION,
            ARCH_MCP_ROOT.name,
            demo.OFFLINE_COMMIT,
            story,
        ),
        sink,
        manager,
        memory_context,
        engine,
        bindings=demo.fixture_bindings(root),
    )
    report = specialist.investigate(
        demo.DEFAULT_QUESTION,
        ARCH_MCP_ROOT.name,
        demo.OFFLINE_COMMIT,
    )
    stream = io.StringIO()
    outcome = demo.run_view_phase(
        demo.DEFAULT_QUESTION,
        ARCH_MCP_ROOT.name,
        demo.OFFLINE_COMMIT,
        report,
        [],
        specialist,
        stream=stream,
    )
    text = stream.getvalue()
    assert outcome["ok"] is True
    assert "FLOW INVESTIGATION" in text
    assert "collector" in text
    assert "fetch_record" in text
    assert "records" in text
    assert "EXPOSED_BY" in text


# -- main -------------------------------------------------------------------


def test_main_offline_succeeds_without_api_key(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--db",
            str(tmp_path / "flow.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "exit code: 0" in captured.out
    assert "orchestrated condition: yes" in captured.out
    assert "all conditions: yes" in captured.out


def test_main_fixture_mode_succeeds(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--repository",
            str(FLOW_CHAIN_ROOT),
            "--db",
            str(tmp_path / "flow.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "target: src/producer.py" in captured.out
    assert "mcp: no" in captured.out
    assert "path_status:      UNKNOWN" in captured.out


def test_main_missing_repository_fails_cleanly(capsys, tmp_path):
    code = demo.main(["--repository", str(tmp_path / "nope")])
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
