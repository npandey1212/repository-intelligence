"""M19 — contract-driven coordinator demonstration tests (offline).

Same discipline as ``test_m17_demo.py``: the phase functions and
``main`` are driven with scripted doubles only — never the network,
never an API key, and never the repository's ``runtime/``.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

DEMO_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_m19_orchestrator_agents_demo.py"
)
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "construction"
)


def _load_demo(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # register before exec: pydantic resolves annotations through
    # sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo(DEMO_PATH, "agentrun_m19_orchestrator_agents_demo")


def _run_demo(stream=None, objective=None):
    absolute, evidence_path, _ = demo.demo_read_target(None)
    assert evidence_path is not None
    question = objective or demo.DEFAULT_QUESTION
    task = demo.build_task(question)
    return demo.run_demo(
        task,
        question,
        evidence_path,
        bindings=None,
        stream=stream or io.StringIO(),
    )


# -- module conventions -----------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "main",
        "run_demo",
        "run_contract_phase",
        "run_dispatch_phase",
        "run_coordinated_phase",
        "run_readiness_phase",
        "run_failure_phase",
        "run_recovery_phase",
        "run_assembly_phase",
        "demo_read_target",
        "fixture_bindings",
        "build_agents",
        "build_plan",
        "build_task",
    ):
        assert callable(getattr(demo, name)), name
    assert demo.EXIT_SUCCESS == 0
    assert demo.EXIT_DEMO_FAILED == 1
    assert demo.EXIT_EXECUTION_FAILED == 6


def test_no_f_strings_in_the_demo_module():
    source = DEMO_PATH.read_text()
    assert 'f"' not in source
    assert "f'" not in source


def test_composition_root_no_second_framework():
    source = DEMO_PATH.read_text()
    for forbidden in ("LangGraph", "CrewAI", "AutoGen"):
        assert forbidden not in source


def test_read_target_default_mode_uses_project_files():
    absolute, evidence_path, bound_root = demo.demo_read_target(None)
    assert absolute is not None and absolute.is_file()
    assert evidence_path in demo.DEFAULT_READ_CANDIDATES
    assert bound_root is None


def test_read_target_fixture_mode_binds_foreign_root(tmp_path):
    absolute, evidence_path, bound_root = demo.demo_read_target(
        str(FIXTURE_ROOT)
    )
    assert absolute is not None and absolute.is_file()
    assert evidence_path in demo.FIXTURE_READ_CANDIDATES
    assert bound_root == str(FIXTURE_ROOT)


def test_read_target_missing_repository_returns_none(tmp_path):
    absolute, evidence_path, bound_root = demo.demo_read_target(
        str(tmp_path / "nowhere")
    )
    assert absolute is None
    assert evidence_path is None


# -- the story --------------------------------------------------------------


def test_run_demo_tells_the_whole_story_offline():
    outcome = _run_demo()
    assert outcome["ok"] is True
    phases = outcome["phases"]
    assert list(phases) == [
        "contract",
        "dispatch",
        "coordinated",
        "readiness",
        "failure",
        "recovery",
        "assembly",
    ]
    assert all(phase["ok"] for phase in phases.values())
    run = outcome["run"]
    assert run.status == "COMPLETED"
    assert run.termination_reason == "SUCCESS"
    assert run.validation_status == "passed"
    assert [record.task_id for record in run.tasks] == [
        "P1",
        "P2",
        "P3",
    ]
    assert [record.status for record in run.tasks] == ["completed"] * 3
    assert [record.depends_on for record in run.tasks] == [
        [],
        ["P1"],
        ["P2"],
    ]


def test_run_demo_is_deterministic_across_runs():
    first = io.StringIO()
    second = io.StringIO()
    assert _run_demo(stream=first)["ok"] is True
    assert _run_demo(stream=second)["ok"] is True
    assert first.getvalue() == second.getvalue()


def test_run_demo_honours_a_custom_objective():
    question = "How is the memory subsystem scoped?"
    outcome = _run_demo(objective=question)
    assert outcome["ok"] is True
    assert outcome["run"].report.question == question
    assert outcome["run"].finding.question == question


def test_contract_phase_never_prints_prompt_text():
    stream = io.StringIO()
    task = demo.build_task(demo.DEFAULT_QUESTION)
    phase = demo.run_contract_phase(task, stream=stream)
    output = stream.getvalue()
    assert phase["ok"] is True
    assert "TASK-SCOPED PROMPT" not in output
    assert "  prompt" not in output
    assert "  question" not in output
    assert "chain: {'P1': [], 'P2': ['P1'], 'P3': ['P2']}" in output


def test_dispatch_phase_reports_measured_usage_not_tokens():
    stream = io.StringIO()
    task = demo.build_task(demo.DEFAULT_QUESTION)
    _, evidence_path, _ = demo.demo_read_target(None)
    phase = demo.run_dispatch_phase(task, evidence_path, None, stream=stream)
    assert phase["ok"] is True
    output = stream.getvalue()
    assert "1 tool call(s), 2 model iteration(s)" in output
    assert "not reported (stays None)" in output
    assert "typed         yes" in output


def test_failure_phase_shows_category_and_blocked_successors():
    stream = io.StringIO()
    task = demo.build_task(demo.DEFAULT_QUESTION)
    _, evidence_path, _ = demo.demo_read_target(None)
    phase = demo.run_failure_phase(
        task, demo.DEFAULT_QUESTION, evidence_path, None, stream=stream
    )
    assert phase["ok"] is True
    run = phase["run"]
    assert run.status == "INCOMPLETE"
    assert run.tasks[0].failure_category == "structured_output"
    assert run.tasks[1].status == "blocked"
    assert run.tasks[2].status == "blocked"
    assert run.retries == 1


def test_recovery_phase_retries_then_commits_success():
    stream = io.StringIO()
    task = demo.build_task(demo.DEFAULT_QUESTION)
    _, evidence_path, _ = demo.demo_read_target(None)
    phase = demo.run_recovery_phase(
        task, demo.DEFAULT_QUESTION, evidence_path, None, stream=stream
    )
    assert phase["ok"] is True
    run = phase["run"]
    assert run.status == "COMPLETED"
    assert run.termination_reason == "SUCCESS"
    assert run.retries == 1
    p1 = run.tasks[0]
    assert p1.attempts == 2
    assert p1.status == "completed"
    assert p1.failure_category is None
    # most recent attempt vs cumulative across both attempts
    assert p1.usage.model_iterations == 2
    assert p1.usage_total.model_iterations == 3
    assert p1.usage_total.tool_calls == 1
    assert [record.status for record in run.tasks] == ["completed"] * 3


def test_assembly_phase_merges_and_deduplicates():
    stream = io.StringIO()
    outcome = _run_demo()
    phase = demo.run_assembly_phase(outcome["run"], stream=stream)
    assert phase["ok"] is True
    report = outcome["run"].report
    assert len(report.findings) == 2
    assert len(report.evidence_refs) == 1
    assert report.investigation_id == demo.OFFLINE_INVESTIGATION_ID


# -- main -------------------------------------------------------------------


def test_main_offline_succeeds_without_api_key(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main([])
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS
    assert "condition: yes" in captured.out
    assert captured.err == ""


def test_main_fixture_mode_succeeds(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = demo.main(["--repository", str(FIXTURE_ROOT)])
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS
    assert "evidence file:" in captured.out


def test_main_question_flag_changes_the_objective(capsys):
    code = demo.main(
        ["--question", "How are MCP servers registered in this repo?"]
    )
    captured = capsys.readouterr()
    assert code == demo.EXIT_SUCCESS
    assert "condition: yes" in captured.out


def test_main_missing_repository_fails_cleanly(capsys, tmp_path):
    code = demo.main(["--repository", str(tmp_path / "nowhere")])
    captured = capsys.readouterr()
    assert code == demo.EXIT_EXECUTION_FAILED
    assert "no readable evidence file" in captured.err
