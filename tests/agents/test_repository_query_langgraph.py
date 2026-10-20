"""Production entry-point orchestration routing (legacy vs LangGraph).

The seam under test is ``run_repository_query(..., orchestration=)``
in ``repository_query.py`` — the ONLY place the production workflow
may select the M21 LangGraph task loop.  Pinned here with
deterministic offline runs:

* default invocation stays on the legacy ``Orchestrator.execute``
  loop;
* ``orchestration="langgraph"`` reaches the real graph helper and
  NEVER the monolithic ``execute``;
* graph failures propagate (no silent legacy fallback) and a missing
  ``langgraph`` dependency is the structured
  ``ORCHESTRATION_UNAVAILABLE`` failure;
* the opted-in path completes end to end on the public result
  contract and is field-level equivalent to a legacy run offline.

Graph internals stay covered by
``tests/orchestration/test_langgraph_sequential.py``; the production
planner preflight stays covered by
``tests/agents/test_planner_request_budget.py`` (here we only pin
that the gate still fires BEFORE the graph is selected).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from conftest import enable_search
from repo_intelligence import repository_query as rq

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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli(CLI_PATH, "agentrun_repository_query_langgraph_tests")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every root a run can write to, pointed at ``tmp_path``."""
    from repo_intelligence.agent import trace_sink as trace_sink_module
    from repo_intelligence.agents import (
        visualization_artifacts as viz_module,
    )
    from repo_intelligence.context_engine import (
        engine as context_engine_module,
    )

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
        lambda: tmp_path / "runtime" / "agent_memory" / "query.db",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def _spy_execute(monkeypatch, explode=False):
    """Record (or forbid) legacy ``Orchestrator.execute`` calls."""
    calls = []
    original = rq.Orchestrator.execute

    def wrapper(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if explode:
            raise AssertionError(
                "Orchestrator.execute must not be called on this path"
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rq.Orchestrator, "execute", wrapper)
    return calls


def _spy_run_with_langgraph(monkeypatch, delegate=True):
    """Record graph-helper invocations; optionally delegate to the
    real helper (requires the optional ``langgraph`` install)."""
    calls = []
    real = rq._run_with_langgraph

    def wrapper(orchestrator, task, plan):
        calls.append(
            {"orchestrator": orchestrator, "task": task, "plan": plan}
        )
        if not delegate:
            raise AssertionError(
                "the graph helper must not be invoked on this path"
            )
        return real(orchestrator, task, plan)

    monkeypatch.setattr(rq, "_run_with_langgraph", wrapper)
    return calls


# -- routing ----------------------------------------------------------------


def test_unsupported_orchestration_value_is_rejected(sandbox):
    with pytest.raises(rq.RepositoryQueryError) as excinfo:
        rq.run_repository_query(
            QUESTION, str(ARCH_MCP_ROOT), orchestration="spark"
        )
    assert "orchestration must be" in str(excinfo.value)
    assert "spark" in str(excinfo.value)


def test_default_invocation_selects_the_legacy_loop(sandbox, monkeypatch):
    legacy_calls = _spy_execute(monkeypatch)
    graph_calls = _spy_run_with_langgraph(monkeypatch, delegate=False)

    result = rq.run_repository_query(
        QUESTION, str(ARCH_MCP_ROOT), diagram=False, artifacts=False
    )

    assert result.status == rq.COMPLETED
    assert len(legacy_calls) == 1
    assert graph_calls == []


def test_langgraph_opt_in_reaches_the_graph_and_never_execute(
    sandbox, monkeypatch
):
    pytest.importorskip("langgraph.graph", reason="optional graph extra")
    legacy_calls = _spy_execute(monkeypatch, explode=True)
    graph_calls = _spy_run_with_langgraph(monkeypatch)

    result = rq.run_repository_query(
        QUESTION,
        str(ARCH_MCP_ROOT),
        diagram=False,
        artifacts=False,
        orchestration=rq.ORCHESTRATION_LANGGRAPH,
    )

    assert result.status == rq.COMPLETED
    assert result.termination_reason == "SUCCESS"
    assert len(graph_calls) == 1
    # the exact public inputs: the production task and the validated
    # three-step plan (planner stays off-graph, pre-validated)
    task = graph_calls[0]["task"]
    plan = graph_calls[0]["plan"]
    assert task.question == QUESTION
    assert [step.id for step in plan.steps] == ["S1", "S2", "S3"]
    assert legacy_calls == []


def test_graph_failure_is_not_silently_fallen_back(sandbox, monkeypatch):
    legacy_calls = _spy_execute(monkeypatch, explode=True)

    def explode(orchestrator, task, plan):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(rq, "_run_with_langgraph", explode)

    with pytest.raises(RuntimeError, match="graph exploded"):
        rq.run_repository_query(
            QUESTION,
            str(ARCH_MCP_ROOT),
            diagram=False,
            artifacts=False,
            orchestration=rq.ORCHESTRATION_LANGGRAPH,
        )
    assert legacy_calls == []


def test_missing_langgraph_dependency_is_a_structured_failure(
    sandbox, monkeypatch
):
    legacy_calls = _spy_execute(monkeypatch, explode=True)

    def missing(orchestrator, task, plan):
        raise ImportError("No module named 'langgraph'")

    monkeypatch.setattr(rq, "_run_with_langgraph", missing)

    result = rq.run_repository_query(
        QUESTION,
        str(ARCH_MCP_ROOT),
        diagram=False,
        artifacts=False,
        orchestration=rq.ORCHESTRATION_LANGGRAPH,
    )

    assert result.status == "FAILED"
    assert result.termination_reason == rq.ORCHESTRATION_UNAVAILABLE
    assert rq.exit_code(result) == rq.EXIT_INVESTIGATION_FAILED
    assert result.failures
    assert "LangGraph" in result.failures[0]
    assert legacy_calls == []


def test_oversized_planner_request_is_rejected_before_the_graph(
    sandbox, monkeypatch
):
    """H1 regression on the opted-in path: the planner preflight
    still fires before any graph selection — the provider is never
    called and the graph helper is never invoked."""
    graph_calls = _spy_run_with_langgraph(monkeypatch, delegate=False)

    result = rq.run_repository_query(
        "x" * 400_000,
        str(ARCH_MCP_ROOT),
        diagram=False,
        artifacts=False,
        orchestration=rq.ORCHESTRATION_LANGGRAPH,
    )

    assert graph_calls == []
    assert result.status == rq.RUN_PLAN_FAILED
    assert result.termination_reason == "PLAN_FAILED"
    assert result.failures
    assert result.failures[0].startswith("RequestBudgetExceededError:")


# -- end to end + parity ----------------------------------------------------


def _run(question, orchestration, sandbox, monkeypatch):
    enable_search(ARCH_MCP_ROOT, monkeypatch, sandbox)
    return rq.run_repository_query(
        question,
        str(ARCH_MCP_ROOT),
        db_path=str(sandbox / "query-{}.db".format(orchestration)),
        trace_root=str(sandbox / "traces-{}".format(orchestration)),
        orchestration=orchestration,
    )


def test_langgraph_end_to_end_offline_completes(sandbox, monkeypatch):
    pytest.importorskip("langgraph.graph", reason="optional graph extra")
    result = _run(QUESTION, rq.ORCHESTRATION_LANGGRAPH, sandbox, monkeypatch)

    assert rq.exit_code(result) == rq.EXIT_SUCCESS
    assert result.status == rq.COMPLETED
    assert result.termination_reason == "SUCCESS"
    assert result.collaboration_status == "passed"
    assert result.diagram_status == "passed"
    assert [step.label for step in result.steps] == [
        "architecture",
        "flow",
        "visualization",
    ]
    assert [step.role for step in result.steps] == [
        "INVESTIGATOR",
        "INVESTIGATOR",
        "ANALYST",
    ]
    assert all(step.status == "completed" for step in result.steps)
    assert len(result.run_ids) == 3
    assert result.evidence
    assert result.conclusions
    assert result.answer


def test_safety_records_ride_every_model_call_of_the_graph_workflow(
    sandbox, monkeypatch
):
    """H1/H2 active DURING the actual graph workflow, not only at
    its planner boundary: every agent trace the offline end-to-end
    run writes carries a per-iteration request-budget record under
    the global default budget (initial request and every repeated
    request alike)."""
    pytest.importorskip("langgraph.graph", reason="optional graph extra")
    result = _run(QUESTION, rq.ORCHESTRATION_LANGGRAPH, sandbox, monkeypatch)

    assert rq.exit_code(result) == rq.EXIT_SUCCESS
    trace_files = sorted(
        (sandbox / "traces-{}".format(rq.ORCHESTRATION_LANGGRAPH)).glob(
            "*/trace.json"
        )
    )
    assert len(trace_files) == len(result.run_ids) == 3
    total_iterations = 0
    for path in trace_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        iterations = payload["iterations"]
        assert iterations, "every agent run made at least one call"
        for iteration in iterations:
            budget = iteration.get("request_budget")
            assert isinstance(budget, dict), "{}: {}".format(
                path.name, iteration.get("iteration")
            )
            assert budget["disposition"] in ("accepted", "reduced")
            assert budget["within_budget"] is True
            assert budget["budget"][
                "effective_input_budget_tokens"
            ] == 95_232
            total_iterations += 1
    # planner + 3 specialist runs all go through the gated paths;
    # several model calls happened across the workflow
    assert total_iterations >= 4


def test_legacy_and_langgraph_results_are_equivalent_offline(
    sandbox, monkeypatch
):
    """Parity on the public contract for the same deterministic input.

    The only intentional difference is run identity: each path mints
    fresh per-step run ids (uuid4), so the id sets are disjoint but
    the same length."""
    pytest.importorskip("langgraph.graph", reason="optional graph extra")
    legacy = _run(QUESTION, rq.ORCHESTRATION_LEGACY, sandbox, monkeypatch)
    graph = _run(QUESTION, rq.ORCHESTRATION_LANGGRAPH, sandbox, monkeypatch)

    for field in (
        "status",
        "termination_reason",
        "collaboration_status",
        "diagram_status",
        "mode",
        "repository",
        "target",
        "investigation_id",
        "answer",
        "conclusions",
        "evidence",
        "uncertainties",
        "failures",
        "artifacts_error",
    ):
        assert getattr(legacy, field) == getattr(graph, field), field

    def shape(result):
        return [
            (
                step.step_id,
                step.goal,
                step.role,
                step.status,
                step.label,
                step.output_type,
                step.run_id is not None,
            )
            for step in result.steps
        ]

    assert shape(legacy) == shape(graph)
    assert len(legacy.run_ids) == len(graph.run_ids) == 3
    assert set(legacy.run_ids).isdisjoint(set(graph.run_ids))


# -- CLI --------------------------------------------------------------------


def test_cli_defaults_to_legacy_orchestration(
    sandbox, monkeypatch, capsys
):
    seen = {}
    real = rq.run_repository_query

    def wrapper(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_repository_query", wrapper)

    code = cli.main(
        [QUESTION, "--repository", str(ARCH_MCP_ROOT), "--no-artifacts"]
    )
    capsys.readouterr()
    assert code == rq.EXIT_SUCCESS
    assert seen["orchestration"] == rq.ORCHESTRATION_LEGACY


def test_cli_forwards_langgraph_and_marks_the_banner(
    sandbox, monkeypatch, capsys
):
    pytest.importorskip("langgraph.graph", reason="optional graph extra")
    seen = {}
    real = rq.run_repository_query

    def wrapper(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_repository_query", wrapper)

    code = cli.main(
        [
            QUESTION,
            "--repository",
            str(ARCH_MCP_ROOT),
            "--no-artifacts",
            "--orchestration",
            "langgraph",
        ]
    )
    out = capsys.readouterr().out
    assert code == rq.EXIT_SUCCESS
    assert seen["orchestration"] == rq.ORCHESTRATION_LANGGRAPH
    assert "Orchestration: langgraph" in out


def test_cli_rejects_an_unknown_orchestration_value(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([QUESTION, "--orchestration", "spark"])
    assert excinfo.value.code == rq.EXIT_USAGE
