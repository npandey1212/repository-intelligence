"""M17 — orchestration demonstration tests (offline, never the network).

Same discipline as ``tests/multiagent/test_m16_multi_agent_demo.py``:
the phase functions and ``main`` are driven with scripted doubles over
temporary memory/context/trace roots (never the repository's
``runtime/`` during tests).
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

DEMO_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_m17_orchestration_demo.py"
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


demo = _load_demo(DEMO_PATH, "agentrun_m17_orchestration_demo")


def _components(tmp_path):
    """Fresh offline composition over temporary memory/context/trace
    roots (never the repository's runtime/ during tests)."""
    manager = demo.build_memory_manager(str(tmp_path / "m17.db"))
    memory_context = demo.build_memory_context(
        demo.REPOSITORY, demo.OFFLINE_COMMIT
    )
    engine = demo.build_engine(demo.REPOSITORY, str(tmp_path / "snapshots"))
    sink = demo.JsonFileTraceSink(str(tmp_path / "traces"))
    return manager, memory_context, engine, sink


def _run_demo(tmp_path, stream=None):
    """Offline story over a project (pinned) evidence file."""
    absolute, evidence_path, _ = demo.demo_read_target(None)
    assert evidence_path is not None
    planner = demo.build_offline_planner(
        demo.DEFAULT_QUESTION, demo.REPOSITORY, demo.OFFLINE_COMMIT
    )
    manager, memory_context, engine, sink = _components(tmp_path)

    def factory():
        return demo.build_agents(
            demo.build_offline_investigator(
                demo.DEFAULT_QUESTION,
                demo.REPOSITORY,
                demo.OFFLINE_COMMIT,
                evidence_path,
            ),
            demo.build_offline_analyst(
                demo.DEFAULT_QUESTION,
                demo.REPOSITORY,
                demo.OFFLINE_COMMIT,
                evidence_path,
            ),
            sink,
            manager,
            memory_context,
            engine,
        )

    return demo.run_demo(
        demo.DEFAULT_QUESTION,
        demo.REPOSITORY,
        demo.OFFLINE_COMMIT,
        planner,
        factory,
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
        "OFFLINE_PLAN_STEPS",
        "run_plan_phase",
        "run_roles_phase",
        "run_execution_phase",
        "run_handoff_phase",
        "run_validation_phase",
        "run_failure_phase",
        "run_status_phase",
        "run_demo",
        "main",
    ):
        assert hasattr(demo, name), name


def test_composition_root_no_second_framework():
    source = DEMO_PATH.read_text(encoding="utf-8")
    for banned in (
        "langgraph",
        "crewai",
        "autogen",
        "sqlite3",
        "import redis",
        "import pika",
    ):
        assert banned not in source, banned
    assert "Orchestrator" in source
    assert "assign_roles" in source
    assert "restricted_dispatcher" in source
    assert "validate_plan" in source


def test_no_f_strings_in_the_demo_module():
    source = DEMO_PATH.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.lstrip()
        assert not stripped.startswith("f\"") and not stripped.startswith(
            "f'"
        ), line
        assert ' f"' not in line and " f'" not in line, line


# -- scripted doubles -------------------------------------------------------


def test_offline_planner_answers_exactly_one_four_step_plan():
    planner = demo.build_offline_planner(
        "objective", demo.REPOSITORY, demo.OFFLINE_COMMIT
    )
    from repo_intelligence.planning.planner import plan_from_response

    plan = plan_from_response(planner.complete(None))
    assert [step.id for step in plan.steps] == ["P1", "P2", "P3", "P4"]
    assert plan.objective == "objective"
    with pytest.raises(AssertionError):
        planner.complete(None)


def test_offline_investigator_answers_one_scripted_run_per_step():
    investigator = demo.build_offline_investigator(
        "q", demo.REPOSITORY, "c" * 40, "README.md"
    )
    first = investigator.complete(None)
    second = investigator.complete(None)
    assert first.finish_reason == "tool_calls"
    assert second.structured_output["investigation_id"] == (
        "inv-m17-demo-001"
    )
    # three Investigator steps = three (tool call, report) pairs
    for _ in range(2):
        investigator.complete(None)
        investigator.complete(None)
    with pytest.raises(AssertionError):
        investigator.complete(None)


def test_offline_finding_echoes_the_first_investigation_id():
    analyst = demo.build_offline_analyst(
        "q", demo.REPOSITORY, "c" * 40, "README.md"
    )
    payload = analyst.complete(None).structured_output
    assert payload["investigation_id"] == demo.OFFLINE_INVESTIGATION_ID
    assert payload["evidence_refs"] == [{"path": "README.md"}]


# -- read target selection --------------------------------------------------


def test_read_target_default_mode_uses_project_files():
    absolute, relative, bound = demo.demo_read_target(None)
    assert absolute is not None and absolute.is_file()
    assert relative is not None
    assert bound is None


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


# -- phases -----------------------------------------------------------------


def test_plan_phase_gate_is_deterministic():
    stream = io.StringIO()
    planner = demo.build_offline_planner(
        demo.DEFAULT_QUESTION, demo.REPOSITORY, demo.OFFLINE_COMMIT
    )
    outcome = demo.run_plan_phase(
        planner, demo.DEFAULT_QUESTION, stream=stream
    )
    assert outcome["ok"] is True
    assert outcome["error"] is None
    assert [step.id for step in outcome["plan"].steps] == [
        "P1",
        "P2",
        "P3",
        "P4",
    ]
    text = stream.getvalue()
    assert "plan gate: SUCCESS" in text
    assert "P4" in text


def test_roles_phase_prints_a_reason_for_every_step():
    stream = io.StringIO()
    plan = demo.InvestigationPlan.parse_obj(
        demo.offline_plan_payload(
            "objective", demo.REPOSITORY, demo.OFFLINE_COMMIT
        )
    )
    outcome = demo.run_roles_phase(plan, stream=stream)
    assert outcome["ok"] is True
    text = stream.getvalue()
    for step_id in ("P1", "P2", "P3", "P4"):
        assert step_id in text
    assert "INVESTIGATOR" in text and "ANALYST" in text
    assert "evidence" in text.lower()


def test_failure_phase_shows_both_terminal_stories(tmp_path):
    absolute, evidence_path, _ = demo.demo_read_target(None)
    manager, memory_context, engine, sink = _components(tmp_path)
    task = demo.TaskPayload(
        question=demo.DEFAULT_QUESTION,
        repository_scope=demo.REPOSITORY,
        commit_scope=demo.OFFLINE_COMMIT,
    )
    plan_payload = demo.offline_plan_payload(
        demo.DEFAULT_QUESTION, demo.REPOSITORY, demo.OFFLINE_COMMIT
    )

    def factory():
        return demo.build_agents(
            demo.build_offline_investigator(
                demo.DEFAULT_QUESTION,
                demo.REPOSITORY,
                demo.OFFLINE_COMMIT,
                evidence_path,
            ),
            demo.build_offline_analyst(
                demo.DEFAULT_QUESTION,
                demo.REPOSITORY,
                demo.OFFLINE_COMMIT,
                evidence_path,
            ),
            sink,
            manager,
            memory_context,
            engine,
        )

    stream = io.StringIO()
    outcome = demo.run_failure_phase(
        task, plan_payload, factory, sink, stream=stream
    )
    assert outcome["ok"] is True, stream.getvalue()
    first = outcome["invalid_plan_run"]
    assert first.status == demo.PLAN_FAILED
    assert first.agent_runs == []
    second = outcome["bounded_run"]
    assert second.status == demo.BUDGET_EXCEEDED
    assert second.blocked_steps == ["P4"]
    assert second.report is not None
    assert second.finding is None
    text = stream.getvalue()
    assert "PLAN_FAILED" in text and "BUDGET_EXCEEDED" in text


# -- whole story ------------------------------------------------------------


def test_run_demo_tells_the_whole_story_offline(tmp_path):
    outcome = _run_demo(tmp_path)
    assert outcome["ok"] is True, outcome
    run = outcome["run"]
    assert run.status == demo.COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert [message.message_type for message in run.messages] == [
        "INVESTIGATION_REQUEST",
        "INVESTIGATION_RESULT",
        "ANALYSIS_REQUEST",
        "ANALYSIS_RESULT",
        "VALIDATION_REQUEST",
        "VALIDATION_RESULT",
    ]
    assert run.validation.status == "passed"
    assert len(run.validation.checks) == 8
    assert all(check.ok for check in run.validation.checks)
    assert run.finding.investigation_id == run.report.investigation_id
    # three Investigator steps merged into ONE evidence package
    assert len(run.report.findings) == 3
    assert len(run.report.evidence_refs) == 1
    assert [step.status for step in run.plan.steps] == ["completed"] * 4
    for name, phase in outcome["phases"].items():
        assert phase["ok"] is True, name


def test_run_demo_is_deterministic_across_runs(tmp_path):
    first = _run_demo(tmp_path / "a")
    second = _run_demo(tmp_path / "b")
    assert first["ok"] and second["ok"]
    one, two = first["run"], second["run"]
    assert one.status == two.status == demo.COMPLETED
    assert [m.message_type for m in one.messages] == [
        m.message_type for m in two.messages
    ]
    assert [c.rule for c in one.validation.checks] == [
        c.rule for c in two.validation.checks
    ]
    assert one.event_operations() == two.event_operations()
    assert [e["sequence"] for e in one.events] == [
        e["sequence"] for e in two.events
    ]


def test_run_demo_reports_bounds_and_mirrored_events(tmp_path):
    outcome = _run_demo(tmp_path)
    run = outcome["run"]
    assert len(run.agent_runs) == 4
    assert run.tool_calls_total == 3
    assert run.retries == 0
    trace = outcome["investigator"].last_trace
    operations = [e["operation"] for e in trace.orchestration_events]
    assert "orchestration.step.assign" in operations
    assert "orchestration.step.complete" in operations
    analyst_trace = outcome["analyst"].last_trace
    assert analyst_trace.orchestration_events


def test_run_demo_roles_match_the_assignment_table(tmp_path):
    outcome = _run_demo(tmp_path)
    assignments = outcome["phases"]["roles"]["assignments"]
    assert [item.role for item in assignments] == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    assert [result.agent_role for result in outcome["run"].step_results] == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]


# -- main -------------------------------------------------------------------


def test_main_offline_succeeds_without_api_key(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--db",
            str(tmp_path / "m17.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "exit code: 0" in captured.out
    assert "final status:   COMPLETED" in captured.out
    assert "orchestration markdown section present: yes" in captured.out
    assert "failure condition: yes" in captured.out


def test_main_fixture_mode_succeeds(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--repository",
            str(FIXTURE_ROOT),
            "--db",
            str(tmp_path / "m17.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "target: src/service.py" in captured.out
    assert "scope: construction@" in captured.out


def test_main_question_flag_changes_the_objective(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(
        [
            "--question",
            "How do the tools exchange typed messages?",
            "--db",
            str(tmp_path / "m17.db"),
            "--snapshots",
            str(tmp_path / "snapshots"),
            "--traces",
            str(tmp_path / "traces"),
        ]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS, captured.err
    assert "How do the tools exchange typed messages?" in captured.out


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
