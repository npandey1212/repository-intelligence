"""Tests for the LangGraph-owned task lifecycle.

The dependency-free source contract runs in every environment. Runtime graph
integration tests activate automatically when the project dependency exists.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "repo_intelligence" / "orchestration" / "langgraph_runner.py"


def test_graph_adapter_does_not_delegate_to_monolithic_execute():
    source = MODULE.read_text(encoding="utf-8")
    assert "runtime.execute(" not in source
    assert "runtime.executor.execute(" in source
    assert "add_conditional_edges(\"commit_result\"" in source
    assert "add_conditional_edges(\"select_task\"" in source


def test_graph_has_dynamic_task_loop_and_existing_contract_gates():
    source = MODULE.read_text(encoding="utf-8")
    for required in (
        "ready_tasks(run.tasks)",
        "validate_task_result(request, result)",
        "runtime.pre_task_gate",
        "run.retries += 1",
        "assembler.assemble(run.step_results, task, run.finding)",
        "validate_collaboration(",
        "runtime.append_handoff",
        "max_tool_calls=runtime.config.max_task_tool_calls",
    ):
        assert required in source


def test_graph_module_compiles_when_langgraph_is_installed():
    pytest.importorskip("langgraph")
    import importlib.util

    spec = importlib.util.spec_from_file_location("langgraph_runner_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert callable(module.build_langgraph_orchestration_graph)
