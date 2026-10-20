"""Architecture Visualization Agent demonstration tests (offline,
never the network).

Same discipline as ``tests/agents/test_flow_dependency_investigator_demo.py``:
the phase functions and ``main`` are driven with scripted doubles over
temporary memory/context/trace/artifact roots (never the repository's
``runtime/`` or ``artifacts/``).  The demo is loaded by path — the
demonstration must be standalone and never import ``tests/``.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

DEMO_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_architecture_visualization_demo.py"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ARCH_MCP_ROOT = FIXTURES / "arch_mcp"
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
    DEMO_PATH, "agentrun_architecture_visualization_demo"
)


def _components(tmp_path, label: str, commit: str):
    manager = demo.build_memory_manager(str(tmp_path / "viz.db"))
    memory_context = demo.build_memory_context(label, commit)
    engine = demo.build_engine(label, str(tmp_path / "snapshots"))
    sink = demo.JsonFileTraceSink(str(tmp_path / "traces"))
    return manager, memory_context, engine, sink


def _run_demo(tmp_path, repository=None, stream=None):
    """Offline story over a fixture root; returns the outcome plus the
    root/target/story the tests assert against."""
    root, target, _ = demo.demo_read_target(repository)
    assert target is not None
    label = root.name
    commit = demo.OFFLINE_COMMIT
    question = demo.DEFAULT_QUESTION
    story = demo.demo_story(root, target)
    models = demo.build_offline_models(question, label, commit, story)
    manager, memory_context, engine, sink = _components(
        tmp_path, label, commit
    )
    outcome = demo.run_demo(
        question,
        label,
        commit,
        story,
        models,
        sink,
        manager,
        memory_context,
        engine,
        demo.fixture_bindings(root),
        tmp_path / "traces",
        str(tmp_path / "artifacts"),
        stream=stream if stream is not None else io.StringIO(),
    )
    return outcome, root, target, story


def _direct_participants(tmp_path):
    """The three DIRECT specialists over one scripted model set."""
    root, target, _ = demo.demo_read_target(None)
    assert target is not None
    label = root.name
    story = demo.demo_story(root, target)
    question = demo.DEFAULT_QUESTION
    models = demo.build_offline_models(
        question, label, demo.OFFLINE_COMMIT, story
    )
    sub = tmp_path / "direct"
    sub.mkdir(exist_ok=True)
    manager, memory_context, engine, sink = _components(
        sub, label, demo.OFFLINE_COMMIT
    )
    bindings = demo.fixture_bindings(root)
    architect = demo.build_architect(
        models["arch"],
        sink,
        manager,
        memory_context,
        engine,
        bindings=bindings,
    )
    flow = demo.build_flow_investigator(
        models["flow"],
        sink,
        manager,
        memory_context,
        engine,
        bindings=bindings,
    )
    visualizer = demo.build_visualizer(
        models["viz"], sink, manager, memory_context, engine
    )
    return {
        "root": root,
        "story": story,
        "question": question,
        "label": label,
        "models": models,
        "architect": architect,
        "flow": flow,
        "visualizer": visualizer,
    }


# -- module surface --------------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "RULE",
        "SECTION",
        "EXIT_SUCCESS",
        "EXIT_DEMO_FAILED",
        "EXIT_EXECUTION_FAILED",
        "DEFAULT_QUESTION",
        "OFFLINE_COMMIT",
        "ARCHITECTURE_TOOLS",
        "FLOW_DEPENDENCY_TOOLS",
        "VISUALIZATION_TOOLS",
        "VISUALIZATION_PROMPT_VERSION",
        "SVG_UNAVAILABLE_NOTICE",
        "demo_read_target",
        "demo_story",
        "build_offline_models",
        "build_architect",
        "build_flow_investigator",
        "build_visualizer",
        "fixture_bindings",
        "RoutedInvestigator",
        "RoutedInvestigatorError",
        "offline_arch_payload",
        "offline_flow_payload",
        "offline_spec_payload",
        "offline_plan_payload",
        "run_scope_phase",
        "run_architecture_phase",
        "run_flow_phase",
        "run_visualization_phase",
        "run_validation_phase",
        "run_rendering_phase",
        "run_artifacts_phase",
        "run_m17_phase",
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
    assert "ArchitectureVisualizationAgent" in source
    assert "RoutedInvestigator" in source


def test_the_demo_is_standalone_and_never_imports_test_doubles():
    source = DEMO_PATH.read_text(encoding="utf-8")
    for banned in (
        "architecture_doubles",
        "flow_doubles",
        "viz_doubles",
        "from tests",
        "import tests",
    ):
        assert banned not in source, banned


# -- scripted models -------------------------------------------------------


def test_offline_models_answer_only_what_was_scripted():
    root, target, _ = demo.demo_read_target(None)
    story = demo.demo_story(root, target)
    models = demo.build_offline_models(
        "q", "r", "c" * 40, story
    )
    arch_first = models["arch"].complete(None)
    arch_second = models["arch"].complete(None)
    assert arch_first.finish_reason == "tool_calls"
    assert arch_second.structured_output["status"] == "complete"
    assert arch_second.structured_output["repository_scope"] == "r"
    with pytest.raises(AssertionError):
        models["arch"].complete(None)

    spec = models["viz"].complete(None).structured_output
    assert spec["diagram_type"] == "AGENT_TOOL_MCP"
    assert [node["id"] for node in spec["nodes"]] == [
        "collector",
        "fetch_record",
        "records",
    ]
    plan = models["planner"].complete(None).structured_output
    assert [step["id"] for step in plan["steps"]] == [
        "S1",
        "S2",
        "S3",
    ]


def test_offline_spec_without_mcp_is_a_single_node_diagram():
    root, target, _ = demo.demo_read_target(
        str(FLOW_CHAIN_ROOT)
    )
    story = demo.demo_story(root, target)
    models = demo.build_offline_models(
        "q", "r", "c" * 40, story
    )
    spec = models["viz"].complete(None).structured_output
    assert spec["diagram_type"] == "COMPONENT"
    assert [node["id"] for node in spec["nodes"]] == ["producer"]
    assert spec["edges"] == []
    assert spec["uncertainties"]


# -- offline story ---------------------------------------------------------


def test_demo_story_mcp_root_reports_the_chain():
    root, target, _ = demo.demo_read_target(None)
    story = demo.demo_story(root, target)
    assert target == "agents/collector.py"
    assert story["mcp"] is True
    assert story["source"] == "collector"
    assert story["tool"] == "fetch_record"
    assert story["server"] == "records"
    assert [
        (source, target_, kind)
        for source, target_, kind, _ in story["legs"]
    ] == [
        ("collector", "fetch_record", "IMPORTS"),
        ("fetch_record", "records", "EXPOSED_BY"),
    ]
    assert story["arch_reads"] == [
        "agents/collector.py",
        "tools/fetch_record.py",
        ".mcp.json",
    ]
    assert story["flow_reads"] == [
        "agents/collector.py",
        ".mcp.json",
    ]
    assert story["reads"] == [
        "agents/collector.py",
        "tools/fetch_record.py",
        ".mcp.json",
    ]


def test_demo_story_without_mcp_is_honest_about_the_destination():
    root, target, _ = demo.demo_read_target(str(FLOW_CHAIN_ROOT))
    story = demo.demo_story(root, target)
    assert story["mcp"] is False
    assert story["destination"] is None
    assert story["legs"] == []
    assert story["reads"] == ["src/producer.py"]
    assert story["arch_reads"] == ["src/producer.py"]
    assert story["flow_reads"] == ["src/producer.py"]


# -- read target selection -------------------------------------------------


def test_read_target_default_mode_uses_the_mcp_fixture():
    root, target, bound = demo.demo_read_target(None)
    assert root == ARCH_MCP_ROOT
    assert target == "agents/collector.py"
    assert bound is not None


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


# -- the router (demo-owned adapter) ---------------------------------------


class _StubSpecialist:
    def __init__(self, name):
        self.name = name
        self.last_trace = None

    def run_result(self, question):
        return "{}:{}".format(self.name, question)


def test_router_routes_by_the_plans_own_step_label():
    arch = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = demo.RoutedInvestigator(
        {"S1": arch, "S2": flow, "arch": arch, "flow": flow}
    )
    routed = router.run_result(
        "repository_scope: r\nstep: S1 \u2014 Document the "
        "architecture.\n"
    )
    assert routed.startswith("arch:")
    assert router.calls == ["S1"]
    assert router.last_trace is None


def test_router_falls_back_to_the_goal_for_foreign_step_ids():
    arch = _StubSpecialist("arch")
    flow = _StubSpecialist("flow")
    router = demo.RoutedInvestigator(
        {"arch": arch, "flow": flow}
    )
    routed = router.run_result(
        "step: P7 \u2014 Trace the path from agent to tool."
    )
    assert routed.startswith("flow:")
    assert router.calls == ["P7"]


def test_router_refuses_to_guess_without_a_route():
    router = demo.RoutedInvestigator({})
    with pytest.raises(demo.RoutedInvestigatorError):
        router.run_result("step: X9 \u2014 do something unknown.")


# -- phases over fixtures --------------------------------------------------


def test_run_demo_tells_the_whole_story_offline(tmp_path):
    stream = io.StringIO()
    outcome, _, target, story = _run_demo(tmp_path, stream=stream)
    assert story["mcp"] is True
    assert target == "agents/collector.py"
    for name in (
        "scope",
        "architecture",
        "flow",
        "visualization",
        "validation",
        "rendering",
        "artifacts",
        "m17",
        "trace",
    ):
        assert outcome[name]["ok"] is True, name
    assert outcome["ok"] is True

    arch_report = outcome["architecture"]["report"]
    assert arch_report.status == "complete"
    assert {
        component.id for component in arch_report.components
    } == {"collector", "fetch_record", "records"}
    assert len(arch_report.relationships) == 2

    flow_report = outcome["flow"]["report"]
    assert flow_report.status == "complete"
    assert flow_report.path_status == "FOUND"
    assert [
        "{} {} {}".format(
            step.source, step.relationship_type, step.target
        )
        for step in flow_report.path
    ] == [
        "collector IMPORTS fetch_record",
        "fetch_record EXPOSED_BY records",
    ]

    spec = outcome["visualization"]["spec"]
    assert spec.diagram_type == "AGENT_TOOL_MCP"
    assert [node.id for node in spec.nodes] == [
        "collector",
        "fetch_record",
        "records",
    ]
    assert [
        edge.relationship_type for edge in spec.edges
    ] == ["DECLARES", "EXPOSED_BY"]
    assert {ref.path for ref in spec.evidence_refs} == {
        "agents/collector.py",
        "tools/fetch_record.py",
        ".mcp.json",
    }

    validation = outcome["validation"]["result"]
    assert validation.status == "passed"
    assert all(check.ok for check in validation.checks)
    assert set(outcome["validation"]["scope"].node_ids) == {
        "collector",
        "fetch_record",
        "records",
    }

    assert "flowchart TD" in outcome["rendering"]["mermaid"]
    assert "DECLARES" in outcome["rendering"]["mermaid"]

    run = outcome["m17"]["run"]
    assert run.status == demo.COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert outcome["m17"]["roles"] == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    assert run.validation.status == "passed"
    assert run.finding.investigation_id == (
        run.report.investigation_id
    )
    assert outcome["m17"]["refreshed"] is True

    artifacts = outcome["artifact_set"]
    assert artifacts is not None
    assert Path(artifacts.spec).is_file()
    assert Path(artifacts.mermaid).is_file()
    assert Path(artifacts.run_summary).is_file()
    summary = Path(artifacts.run_summary).read_text(
        encoding="utf-8"
    )
    assert "## M17 Execution" in summary
    assert "Status: COMPLETED" in summary
    if not artifacts.svg_available:
        assert artifacts.svg is None
        assert demo.SVG_UNAVAILABLE_NOTICE in summary


def test_run_demo_prints_the_section_25_verification(tmp_path):
    stream = io.StringIO()
    outcome, _, _, _ = _run_demo(tmp_path, stream=stream)
    text = stream.getvalue()
    assert "ARTIFACTS (spec §25)" in text
    assert "diagram_spec.json" in text
    assert "exists: yes" in text
    assert "architecture.svg" in text
    assert "all conditions: yes" in text
    statuses = outcome["artifacts"]["statuses"]
    by_name = {status.name: status for status in statuses}
    assert by_name["diagram_spec.json"].exists is True
    assert by_name["architecture.mmd"].exists is True
    assert by_name["run_summary.md"].exists is True
    if not demo.graphviz_available():
        assert by_name["architecture.svg"].exists is False
        assert "SVG" in by_name["architecture.svg"].note


def test_run_demo_without_mcp_is_honest_end_to_end(tmp_path):
    outcome, _, target, story = _run_demo(
        tmp_path, repository=str(FLOW_CHAIN_ROOT)
    )
    assert story["mcp"] is False
    assert target == "src/producer.py"
    assert outcome["ok"] is True, {
        name: entry["ok"]
        for name, entry in outcome.items()
        if isinstance(entry, dict) and "ok" in entry
    }
    report = outcome["flow"]["report"]
    assert report.path_status == "UNKNOWN"
    assert report.path == []
    assert report.target is None
    assert report.uncertainties
    spec = outcome["visualization"]["spec"]
    assert spec.diagram_type == "COMPONENT"
    assert [node.id for node in spec.nodes] == ["producer"]
    assert spec.edges == []
    assert outcome["validation"]["result"].status == "passed"
    assert outcome["m17"]["run"].validation.status == "passed"


def test_run_demo_is_deterministic_across_runs(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    first, _, _, _ = _run_demo(tmp_path / "one")
    second, _, _, _ = _run_demo(tmp_path / "two")
    for name in (
        "scope",
        "architecture",
        "flow",
        "visualization",
        "validation",
        "rendering",
        "artifacts",
        "m17",
        "trace",
    ):
        assert first[name]["ok"] == second[name]["ok"], name
    # spec §19: one spec -> byte-identical Mermaid across runs
    assert Path(first["artifact_set"].spec).read_bytes() == (
        Path(second["artifact_set"].spec).read_bytes()
    )
    assert Path(first["artifact_set"].mermaid).read_bytes() == (
        Path(second["artifact_set"].mermaid).read_bytes()
    )
    first_spec = first["visualization"]["spec"]
    second_spec = second["visualization"]["spec"]
    assert first_spec.derived_investigation_id() == (
        second_spec.derived_investigation_id()
    )
    assert first["m17"]["run"].finding.investigation_id == (
        second["m17"]["run"].finding.investigation_id
    )


def test_direct_and_orchestrated_use_the_same_specialist(tmp_path):
    outcome, _, _, _ = _run_demo(tmp_path)
    visualizer = outcome["m17"]["visualizer"]
    assert isinstance(visualizer, demo.ArchitectureVisualizationAgent)
    assert visualizer.tools == []
    kit = _direct_participants(tmp_path)
    assert type(kit["visualizer"]) is type(visualizer)
    assert kit["visualizer"].tools == visualizer.tools
    assert type(kit["architect"]) is demo.ArchitectureInvestigator
    assert type(kit["flow"]) is demo.FlowDependencyInvestigator
    outputs = [
        type(result.output).__name__
        for result in outcome["m17"]["run"].step_results
    ]
    assert outputs == [
        "InvestigationReport",
        "InvestigationReport",
        "ArchitectureFinding",
    ]


# -- focused phases --------------------------------------------------------


def test_scope_phase_checks_prompt_and_tools(tmp_path):
    manager, memory_context, engine, sink = _components(
        tmp_path, ARCH_MCP_ROOT.name, demo.OFFLINE_COMMIT
    )
    root, target, _ = demo.demo_read_target(None)
    story = demo.demo_story(root, target)
    visualizer = demo.build_visualizer(
        demo.build_offline_models(
            demo.DEFAULT_QUESTION,
            ARCH_MCP_ROOT.name,
            demo.OFFLINE_COMMIT,
            story,
        )["viz"],
        sink,
        manager,
        memory_context,
        engine,
    )
    stream = io.StringIO()
    outcome = demo.run_scope_phase(
        demo.DEFAULT_QUESTION,
        ARCH_MCP_ROOT.name,
        demo.OFFLINE_COMMIT,
        story,
        visualizer,
        stream=stream,
    )
    assert outcome["ok"] is True
    text = stream.getvalue()
    assert "ARCHITECTURE_VISUALIZATION_PROMPT_V1" in text
    assert "visualizer=(none)" in text
    prompt = demo.ARCHITECTURE_VISUALIZATION_PROMPT_V1
    assert "orchestrat" not in prompt.lower()
    assert "planner" not in prompt.lower()
    assert visualizer.tools == list(demo.VISUALIZATION_TOOLS)
    assert demo.VISUALIZATION_PROMPT_VERSION.startswith("V")


def test_validation_phase_rejects_an_out_of_scope_spec(tmp_path):
    kit = _direct_participants(tmp_path)
    arch_report = kit["architect"].investigate(
        kit["question"], kit["label"], demo.OFFLINE_COMMIT
    )
    flow_report = kit["flow"].investigate(
        kit["question"], kit["label"], demo.OFFLINE_COMMIT
    )
    request = demo.VisualizationRequest(
        question=kit["question"],
        diagram_type="AGENT_TOOL_MCP",
        architecture_report=arch_report,
        flow_report=flow_report,
    )
    payload = demo.offline_spec_payload(
        kit["question"],
        kit["label"],
        demo.OFFLINE_COMMIT,
        kit["story"],
    )
    payload["nodes"] = list(payload["nodes"]) + [
        {
            "id": "notes_store",
            "label": "notes store",
            "node_type": "unknown",
            "classification": "unknown",
            "confidence": "unknown",
        }
    ]
    rejected = demo.DiagramSpec(**payload)
    stream = io.StringIO()
    outcome = demo.run_validation_phase(
        kit["visualizer"],
        rejected,
        request,
        kit["architect"],
        kit["flow"],
        stream=stream,
    )
    assert outcome["ok"] is False
    assert outcome["result"].status == "failed"
    text = stream.getvalue()
    assert "scope_validity" in text
    assert "notes_store" in text


def test_rendering_phase_is_deterministic_and_honest(tmp_path):
    kit = _direct_participants(tmp_path)
    arch_report = kit["architect"].investigate(
        kit["question"], kit["label"], demo.OFFLINE_COMMIT
    )
    flow_report = kit["flow"].investigate(
        kit["question"], kit["label"], demo.OFFLINE_COMMIT
    )
    visualization = demo.run_visualization_phase(
        kit["visualizer"],
        arch_report,
        flow_report,
        kit["question"],
        kit["label"],
        demo.OFFLINE_COMMIT,
        "AGENT_TOOL_MCP",
        stream=io.StringIO(),
    )
    assert visualization["ok"] is True
    stream = io.StringIO()
    outcome = demo.run_rendering_phase(
        visualization["spec"], stream=stream
    )
    assert outcome["ok"] is True
    text = stream.getvalue()
    assert "flowchart TD" in text
    assert "deterministic re-render: identical" in text
    if not demo.graphviz_available():
        assert demo.SVG_UNAVAILABLE_NOTICE in text


# -- main ------------------------------------------------------------------


def test_main_offline_succeeds_without_api_key(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--output",
            str(tmp_path / "artifacts"),
            "--db",
            str(tmp_path / "viz.db"),
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
    assert "ARTIFACTS (spec §25)" in captured.out
    assert (tmp_path / "artifacts" / "diagram_spec.json").is_file()
    assert (tmp_path / "artifacts" / "architecture.mmd").is_file()
    assert (tmp_path / "artifacts" / "run_summary.md").is_file()


def test_main_without_mcp_succeeds(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--repository",
            str(FLOW_CHAIN_ROOT),
            "--output",
            str(tmp_path / "artifacts"),
            "--db",
            str(tmp_path / "viz.db"),
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
    assert "diagram_type: COMPONENT" in captured.out


def test_main_default_output_reads_project_root_artifacts(
    monkeypatch, capsys, tmp_path
):
    """Regression: without --output the phase-9 summary check must
    read run_summary.md from the DEFAULT artifact directory (it used
    to gate the check on the --output flag itself, so the default run
    failed the trace condition with exit 1).  The default root is
    redirected into tmp so the suite never touches the repository's
    artifact set."""
    monkeypatch.setattr(
        "repo_intelligence.agents.visualization_artifacts."
        "default_artifacts_root",
        lambda: tmp_path / "artifacts",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--db",
            str(tmp_path / "viz.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "all conditions: yes" in captured.out
    assert "run_summary.md M17 status line: present (COMPLETED)" in (
        captured.out
    )
    assert "trace condition: yes" in captured.out
    default_dir = tmp_path / "artifacts" / "architecture_visualization"
    assert str(default_dir) in captured.out
    assert (default_dir / "diagram_spec.json").is_file()
    assert (default_dir / "architecture.mmd").is_file()
    assert (default_dir / "run_summary.md").is_file()


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
    monkeypatch, capsys
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(["--live"])
    captured = capsys.readouterr()
    assert code == demo.EXIT_EXECUTION_FAILED
    assert "OPENAI_API_KEY" in captured.err
