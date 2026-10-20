"""M15 — context-engineering demonstration, offline (§31–§41/§57, §62).

Scripted doubles only — no network, no API key:

 * inventory → snapshot → model step → tool step → snapshot #2 →
   memory interaction (fresh/stale/unknown) → budget demonstration →
   full agent run (claims validated + context events on the trace) →
   evaluation report — the whole §57 story in one function
 * ``main`` default mode is OFFLINE and succeeds without
   OPENAI_API_KEY; ``--live`` without a key fails with exit 6
 * persisted snapshots are JSON (``context_001.json`` /
   ``context_002.json``) whose bytes decode to the canonical
   ``to_dict()`` projection — no database anywhere
 * the demo is a composition root — a source scan shows no sqlite3
   import, no second agent loop, and no re-implemented policy
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

DEMO_PATH = Path(__file__).resolve().parents[2] / "agentrun_m15_context_demo.py"


def _load_demo(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # register before exec: Pydantic resolves the model's annotations
    # through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo(DEMO_PATH, "agentrun_m15_context_demo")

CURRENT_COMMIT = demo.OFFLINE_COMMIT


# -- helpers ----------------------------------------------------------------


def _manager(tmp_path, name="m15.db"):
    return demo.build_memory_manager(tmp_path / name)


def _seeded(tmp_path):
    manager = _manager(tmp_path)
    ids = demo.seed_demo_memories(manager, CURRENT_COMMIT)
    return manager, ids


def _task():
    return demo.build_task(demo.DEFAULT_QUESTION, CURRENT_COMMIT)


def _engine(tmp_path):
    return demo.build_engine(tmp_path / "snapshots")


def _inventory(tmp_path):
    manager, ids = _seeded(tmp_path)
    graph, label = demo.build_graph()
    outcome = demo.run_inventory_phase(
        _task(), manager, ids, graph, label, stream=io.StringIO()
    )
    return manager, ids, outcome


def _scripted_agent_model():
    """The offline story's model-step reply + the agent's four
    planned-run responses (plan, tool call, step result, claim
    payload) — consumed in order."""
    target = demo.demo_read_target()
    model = demo.build_offline_model(target)
    # discard the phase-3 model-step response; the agent phase needs
    # exactly the remaining four.
    model.complete(demo.LLMRequest(messages=[]))
    return model


def _payload_for(target):
    end = min(9, demo._count_lines(target))
    return {
        "finding": "README.md is this repository's entry-point "
        "documentation (offline scripted conclusion).",
        "evidence_paths": ["README.md"],
        "claims": [
            {
                "statement": "The repository root contains README.md, "
                "read lines 1-{} as evidence.".format(end),
                "evidence_refs": [
                    {
                        "path": "README.md",
                        "line_start": 1,
                        "line_end": end,
                    }
                ],
            }
        ],
    }


# -- module contract / composition root -------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "main",
        "build_agent",
        "build_engine",
        "build_memory_manager",
        "build_memory_context",
        "seed_demo_memories",
        "build_offline_model",
        "build_graph",
        "run_inventory_phase",
        "run_selection_phase",
        "run_model_step_phase",
        "run_tool_step_phase",
        "run_evolution_phase",
        "run_memory_phase",
        "run_budget_phase",
        "run_agent_phase",
        "run_evaluation_phase",
        "run_demo",
        "diff_snapshots",
        "print_plan",
        "InvestigationSummary",
        "EXIT_SUCCESS",
        "EXIT_DEMO_FAILED",
        "EXIT_EXECUTION_FAILED",
    ):
        assert hasattr(demo, name), name


def test_demo_is_a_composition_root_no_second_loop_no_sqlite():
    source = DEMO_PATH.read_text()
    assert "import sqlite3" not in source
    # exactly one Agent construction helper; the loop, the selection
    # policy, and the validator are never re-implemented here
    assert source.count("def build_agent") == 1
    assert "while True" not in source
    assert "_execute_loop" not in source
    assert "def select_context" not in source


def test_claim_contract_requires_claims():
    with pytest.raises(ValidationError):
        demo.InvestigationSummary(
            finding="no claims field", evidence_paths=["README.md"]
        )


def test_legacy_commit_is_deterministically_older():
    legacy = demo.legacy_commit_for(CURRENT_COMMIT)
    assert legacy != CURRENT_COMMIT
    assert legacy[:32] == CURRENT_COMMIT[:32]
    assert legacy == demo.OFFLINE_LEGACY_COMMIT


def test_offline_model_answers_only_what_was_scripted():
    model = demo.OfflineModel(demo._text("first"), demo._text("second"))
    first = model.complete(demo.LLMRequest(messages=[]))
    second = model.complete(demo.LLMRequest(messages=[]))
    assert first.message.content == "first"
    assert second.message.content == "second"
    assert len(model.requests) == 2
    try:
        model.complete(demo.LLMRequest(messages=[]))
    except AssertionError:
        pass
    else:
        raise AssertionError("exhausted model must fail loudly")


def test_seed_demo_memories_is_idempotent(tmp_path):
    manager = _manager(tmp_path)
    first = demo.seed_demo_memories(manager, CURRENT_COMMIT)
    second = demo.seed_demo_memories(manager, CURRENT_COMMIT)
    assert set(first) == {"fresh", "stale", "unknown"}
    assert first == second
    rows = manager.search(repository=demo.REPOSITORY)
    assert len(rows) == 3
    by_scope = {
        memory.scope.commit: memory for memory in rows
    }
    assert CURRENT_COMMIT in by_scope
    assert demo.legacy_commit_for(CURRENT_COMMIT) in by_scope
    assert None in by_scope


def test_graph_explicit_path_branch(tmp_path):
    artifact = tmp_path / "graph.json"
    artifact.write_text(json.dumps(demo.SYNTHETIC_GRAPH_DOCUMENT))
    graph, label = demo.build_graph(artifact)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert str(artifact) in label


# -- phase 1: inventory (§33) ------------------------------------------------


def test_inventory_phase_offers_every_source(tmp_path):
    _, ids, outcome = _inventory(tmp_path)
    assert outcome["ok"] is True
    assert outcome["by_source"] == {
        "task": 1,
        "repository": 1,
        "memory": 3,
        "graph": 2,
    }
    ids_seen = [c.candidate_id for c in outcome["candidates"]]
    assert ids_seen[:5] == [
        "cand_task",
        "cand_repo",
        "cand_mem_fresh",
        "cand_mem_stale",
        "cand_mem_unknown",
    ]
    assert "cand_graph_summary" in ids_seen
    assert "cand_graph_neighborhood" in ids_seen
    assert set(ids) == {"fresh", "stale", "unknown"}


def test_inventory_phase_is_deterministic_across_calls(tmp_path):
    _, _, first = _inventory(tmp_path)
    _, _, second = _inventory(tmp_path)

    def projection(candidates):
        rows = []
        for candidate in candidates:
            data = candidate.to_dict()
            data.pop("created_at", None)
            rows.append(data)
        return rows

    assert projection(first["candidates"]) == projection(second["candidates"])


def test_inventory_prints_the_candidate_table(tmp_path):
    buf = io.StringIO()
    manager, ids = _seeded(tmp_path)
    graph, label = demo.build_graph()
    demo.run_inventory_phase(_task(), manager, ids, graph, label, stream=buf)
    text = buf.getvalue()
    assert "CANDIDATE INVENTORY (§33)" in text
    assert "cand_mem_stale" in text
    assert "cand_graph_neighborhood" in text


# -- phase 2: selection snapshot (§34) --------------------------------------


def test_selection_phase_persists_auditable_json(tmp_path):
    manager, _, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    buf = io.StringIO()
    phase = demo.run_selection_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=buf,
    )
    assert phase["ok"] is True
    path = phase["path"]
    assert path.is_file()
    assert path.name == "context_001.json"
    document = json.loads(path.read_text())
    snapshot = phase["snapshot"]
    assert document == snapshot.to_dict()
    # context snapshots are JSON records — no database reference
    assert "sqlite" not in path.read_text().lower()
    # one decision per candidate, nothing dropped silently
    assert len(snapshot.selection_decisions) == len(
        outcome["candidates"]
    )
    text = buf.getvalue()
    assert "SELECTION SNAPSHOT #1" in text
    assert "STALE_MEMORY" in text
    assert "UNKNOWN_FRESHNESS" in text


def test_selection_phase_is_deterministic_across_engines(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    run_a = demo.run_selection_phase(
        _engine(tmp_path / "a"),
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=io.StringIO(),
    )
    run_b = demo.run_selection_phase(
        _engine(tmp_path / "b"),
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=io.StringIO(),
    )
    items_a = [
        (item.candidate_id, item.selection_reason)
        for item in run_a["snapshot"].package.selected_items
    ]
    items_b = [
        (item.candidate_id, item.selection_reason)
        for item in run_b["snapshot"].package.selected_items
    ]
    assert items_a == items_b
    decisions_a = [
        (d.candidate_id, d.selected, d.reason)
        for d in run_a["snapshot"].selection_decisions
    ]
    decisions_b = [
        (d.candidate_id, d.selected, d.reason)
        for d in run_b["snapshot"].selection_decisions
    ]
    assert decisions_a == decisions_b
    assert run_a["metrics"] == run_b["metrics"]


# -- phase 3: model step (§35) ----------------------------------------------


def test_model_step_receives_exactly_the_rendered_package(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    selection = demo.run_selection_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=io.StringIO(),
    )
    model = demo.OfflineModel(demo._text("scripted reply"))
    phase = demo.run_model_step_phase(
        model,
        selection["snapshot"].package,
        demo.DEFAULT_QUESTION,
        stream=io.StringIO(),
    )
    assert phase["ok"] is True
    request = phase["request"]
    content = request.messages[0].content
    # the request carries the task AND the rendered package verbatim
    assert demo.DEFAULT_QUESTION in content
    assert phase["rendered"] in content
    assert phase["response"].message.content == "scripted reply"
    assert model.requests and model.requests[0] is request


# -- phase 4: tool step (§36) ------------------------------------------------


def test_tool_step_reads_a_real_file_offline(tmp_path):
    target = demo.demo_read_target()
    assert target is not None
    phase = demo.run_tool_step_phase(
        demo.default_tool_dispatcher(), target, stream=io.StringIO()
    )
    assert phase["ok"] is True
    assert phase["result"].success is True
    data = phase["result"].data
    assert data["path"] == demo._relative(target)
    assert data["line_start"] == 1
    assert data["line_end"] >= 1


def test_tool_step_without_target_reports_failure(tmp_path):
    phase = demo.run_tool_step_phase(
        demo.default_tool_dispatcher(), None, stream=io.StringIO()
    )
    assert phase["ok"] is False
    assert phase["result"] is None


# -- phase 5: evolution (§37) ------------------------------------------------


def test_evolution_adds_tool_and_evidence_candidates(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    demo.run_selection_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=io.StringIO(),
    )
    tool_step = demo.run_tool_step_phase(
        demo.default_tool_dispatcher(),
        demo.demo_read_target(),
        stream=io.StringIO(),
    )
    phase = demo.run_evolution_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        tool_step["result"],
        demo.DEMO_RUN_ID,
        stream=io.StringIO(),
    )
    assert phase["ok"] is True
    assert phase["path"].name == "context_002.json"
    assert phase["metrics"]["considered"] == 9
    selected = {
        item.candidate_id for item in phase["snapshot"].package.selected_items
    }
    assert "cand_tool_read" in selected
    assert "cand_evidence_read" in selected

    snapshot1 = json.loads(
        (engine.snapshot_root / demo.DEMO_RUN_ID / "context_001.json")
        .read_text()
    )
    first_ids = set(
        row["candidate_id"]
        for row in snapshot1["selection_decisions"]
        if row["selected"]
    )
    diff = {
        "added": sorted(selected - first_ids),
        "removed": sorted(first_ids - selected),
    }
    assert diff["added"] == ["cand_evidence_read", "cand_tool_read"]
    assert diff["removed"] == []


def test_tool_candidates_are_stable_and_carry_evidence(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    tool_step = demo.run_tool_step_phase(
        demo.default_tool_dispatcher(),
        demo.demo_read_target(),
        stream=io.StringIO(),
    )
    candidates = demo.tool_candidates(
        outcome["task"], tool_step["result"]
    )
    ids = [candidate.candidate_id for candidate in candidates]
    assert ids == ["cand_tool_read", "cand_evidence_read"]
    evidence = next(
        c for c in candidates if c.source_type == "evidence"
    )
    assert evidence.evidence
    ref = evidence.evidence[0]
    assert ref.path == demo._relative(demo.demo_read_target())
    assert ref.line_start == 1


def test_diff_snapshots_reports_added_and_removed():
    class Entry:
        def __init__(self, candidate_id, selected):
            self.candidate_id = candidate_id
            self.selected = selected

    class Snap:
        def __init__(self, decisions):
            self.selection_decisions = decisions

    first = Snap([Entry("a", True), Entry("b", True)])
    second = Snap([Entry("b", True), Entry("c", True)])
    assert demo.diff_snapshots(first, second) == {
        "added": ["c"],
        "removed": ["a"],
    }


# -- phase 6: memory interaction (§38) --------------------------------------


def test_memory_phase_enforces_the_freshness_tristate(tmp_path):
    manager, ids, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    demo.run_selection_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=io.StringIO(),
    )
    snapshot = json.loads(
        (engine.snapshot_root / demo.DEMO_RUN_ID / "context_001.json")
        .read_text()
    )

    class SnapshotView:
        selection_decisions = [
            SimpleNamespace(**row) for row in snapshot["selection_decisions"]
        ]

    phase = demo.run_memory_phase(SnapshotView(), ids, stream=io.StringIO())
    assert phase["ok"] is True
    assert phase["outcomes"]["fresh"]["selected"] is True
    assert phase["outcomes"]["fresh"]["reason"] == "VERIFIED_MEMORY"
    assert phase["outcomes"]["stale"]["selected"] is False
    assert phase["outcomes"]["stale"]["reason"] == "STALE_MEMORY"
    assert phase["outcomes"]["unknown"]["selected"] is False
    assert phase["outcomes"]["unknown"]["reason"] == (
        "UNKNOWN_FRESHNESS"
    )


# -- phase 7: budget (§39) ---------------------------------------------------


def test_budget_phase_tight_budget_and_impossible_budget(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    phase = demo.run_budget_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        stream=io.StringIO(),
    )
    assert phase["ok"] is True
    tiny = phase["tiny"].package
    assert tiny.budget.estimated_characters <= (
        demo.TINY_BUDGET.max_characters
    )
    assert tiny.budget.selected_items <= demo.TINY_BUDGET.max_items
    assert tiny.budget.selected_items > 0
    assert phase["blocked"], "the tight budget must block candidates"
    assert phase["impossible_error"] == demo.CONTEXT_BUDGET_EXCEEDED


def test_budget_phase_never_reports_silent_overflow(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    buf = io.StringIO()
    demo.run_budget_phase(
        engine, outcome["task"], outcome["candidates"], stream=buf
    )
    text = buf.getvalue()
    assert "CONTEXT_BUDGET_EXCEEDED" in text
    assert "never a silent empty" in text


# -- phase 8: full agent run (§40) ------------------------------------------


def test_agent_phase_validates_claims_and_records_context_events(
    tmp_path,
):
    manager, ids = _seeded(tmp_path)
    engine = _engine(tmp_path)
    model = _scripted_agent_model()
    phase = demo.run_agent_phase(
        model,
        demo.default_tool_dispatcher(),
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
        engine,
        demo.DEFAULT_QUESTION,
        trace_sink=None,
        stream=io.StringIO(),
    )
    assert phase["ok"] is True
    trace = phase["trace"]
    operations = [
        event["operation"] for event in trace.context_events
    ]
    assert "context.select" in operations
    assert "context.render" in operations
    assert phase["agent"].last_context_error is None
    # M13 validation: every claim supported by an inspected file
    assert phase["result"].evidence_validation
    assert all(
        entry.status == demo.STATUS_SUPPORTED
        for entry in phase["result"].evidence_validation
    )
    # deterministic memory formation still happened (M14 contract)
    assert any(
        event.get("operation") == "memory.store"
        for event in trace.memory_events
    )


def test_agent_context_available_is_not_the_same_as_selected(tmp_path):
    manager, ids = _seeded(tmp_path)
    engine = _engine(tmp_path)
    model = _scripted_agent_model()
    phase = demo.run_agent_phase(
        model,
        demo.default_tool_dispatcher(),
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
        engine,
        demo.DEFAULT_QUESTION,
        trace_sink=None,
        stream=io.StringIO(),
    )
    trace = phase["trace"]
    retrieve = next(
        event
        for event in trace.memory_events
        if event.get("operation") == "memory.retrieve"
    )
    select = next(
        event
        for event in trace.context_events
        if event.get("operation") == "context.select"
    )
    # §56: memory AVAILABLE (3 retrieved) vs memory SELECTED (1)
    assert retrieve["count"] == 3
    assert select["candidate_count"] == 4
    reasons = select["reasons"]
    assert reasons.get("VERIFIED_MEMORY") == 1
    assert reasons.get("STALE_MEMORY") == 1
    assert reasons.get("UNKNOWN_FRESHNESS") == 1
    assert reasons.get("TASK_DIRECT") == 1
    assert select["budget"]["selected_items"] == 2


def test_agent_phase_reports_per_step_context_evolution(tmp_path):
    """§17/§18 — the REAL Agent execution shows two reasoning steps:
    boundary selection (001) and post-tool re-selection (002), each
    with its own persisted snapshot and rendered context printed."""
    manager, ids = _seeded(tmp_path)
    engine = _engine(tmp_path)
    model = _scripted_agent_model()
    buf = io.StringIO()
    phase = demo.run_agent_phase(
        model,
        demo.default_tool_dispatcher(),
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
        engine,
        demo.DEFAULT_QUESTION,
        trace_sink=None,
        stream=buf,
    )
    assert phase["ok"] is True
    assert phase["step_ids"] == ["001", "002"]
    assert len(phase["snapshots"]) == 2
    assert phase["metrics"]["context_selection_count"] == 2
    assert phase["metrics"]["context_evolution_count"] == 1
    # one persisted snapshot per reasoning boundary (§12)
    run_root = engine.snapshot_root / phase["trace"].run_id
    assert (run_root / "context_001.json").is_file()
    assert (run_root / "context_002.json").is_file()
    text = buf.getvalue()
    assert "=== STEP 1 ===" in text
    assert "=== STEP 2 ===" in text
    assert "agent context evolution:" in text
    assert "trigger initial" in text
    assert "trigger tool_result" in text


def test_agent_phase_plans_then_executes_one_linked_run(tmp_path):
    """§40 end-to-end + §7 ownership — the SAME run performs the
    complete flow: planner iteration → validated plan → planned
    ReAct execution with per-step context re-selection; every
    persisted snapshot carries this run's id."""
    manager, ids = _seeded(tmp_path)
    engine = _engine(tmp_path)
    model = _scripted_agent_model()
    buf = io.StringIO()
    phase = demo.run_agent_phase(
        model,
        demo.default_tool_dispatcher(),
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
        engine,
        demo.DEFAULT_QUESTION,
        trace_sink=None,
        stream=buf,
    )
    assert phase["ok"] is True
    trace = phase["trace"]

    # the planner ran FIRST — iteration 1 is the plan call: it
    # executed no repository tools of its own (M10 invariant) and
    # closed as a validated structured output
    planner = trace.iterations[0]
    assert planner.tool_results == []
    assert planner.response_type == "STRUCTURED_OUTPUT"
    assert (planner.structured_validation or {}).get("model") == (
        "InvestigationPlan"
    )
    # repository tool execution happens only in LATER iterations
    tool_iterations = [
        iteration
        for iteration in trace.iterations
        if iteration.tool_results
    ]
    assert tool_iterations
    assert trace.iterations.index(tool_iterations[0]) > 0

    # the validated plan is recorded on this trace and executed
    plan = trace.plan
    assert plan is not None
    assert plan["status"] == "completed"
    assert [step["id"] for step in plan["steps"]] == ["P1", "P2"]
    assert plan["steps"][0].get("result")  # P1 produced a result
    assert phase["plan"] == plan

    # context evolution happened INSIDE this planned run
    assert phase["step_ids"] == ["001", "002"]
    assert phase["metrics"]["context_selection_count"] == 2

    # §7 ownership: on-disk snapshots and in-memory snapshots all
    # carry the SAME run id as the trace
    run_root = engine.snapshot_root / trace.run_id
    for step_id in phase["step_ids"]:
        data = json.loads(
            (run_root / "context_{}.json".format(step_id)).read_text()
        )
        assert data["run_id"] == trace.run_id
    assert all(
        snapshot.run_id == trace.run_id
        for snapshot in phase["snapshots"]
    )

    text = buf.getvalue()
    assert "=== PLANNER ===" in text
    assert "plan generated: 2 step(s)" in text
    assert "plan validated: status completed" in text
    assert "snapshot ownership: 2 snapshot(s) belong to run " in text


def test_agent_phase_without_engine_records_no_context_events(tmp_path):
    manager, ids = _seeded(tmp_path)
    model = _scripted_agent_model()
    phase = demo.run_agent_phase(
        model,
        demo.default_tool_dispatcher(),
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
        None,  # no ContextEngine injected — legacy path
        demo.DEFAULT_QUESTION,
        trace_sink=None,
        stream=io.StringIO(),
    )
    assert phase["trace"].context_events == []
    # without context events the phase's demonstration condition fails
    assert phase["ok"] is False


# -- phase 9: evaluation (§41/§47) -------------------------------------------


def test_evaluation_phase_renders_deterministic_reports(tmp_path):
    _, _, outcome = _inventory(tmp_path)
    engine = _engine(tmp_path)
    selection = demo.run_selection_phase(
        engine,
        outcome["task"],
        outcome["candidates"],
        demo.DEMO_RUN_ID,
        demo.STEP_INITIAL,
        stream=io.StringIO(),
    )
    phase = demo.run_evaluation_phase(
        selection["snapshot"],
        selection["snapshot"],
        stream=io.StringIO(),
    )
    assert phase["ok"] is True
    assert phase["metrics1"]["isolation_ok"] is True
    assert phase["metrics1"]["stale_items_excluded"] == 1
    assert phase["metrics1"]["unknown_items_excluded"] == 1
    assert "Isolation: PASS" in phase["report1"]
    assert "STALE_MEMORY" in phase["report1"]


# -- full story: run_demo (§57) ----------------------------------------------


def test_run_demo_tells_the_whole_story_offline(tmp_path):
    manager, ids = _seeded(tmp_path)
    graph, label = demo.build_graph()
    model = demo.build_offline_model(demo.demo_read_target())
    buf = io.StringIO()
    outcome = demo.run_demo(
        model,
        demo.default_tool_dispatcher(),
        manager,
        _engine(tmp_path),
        ids,
        CURRENT_COMMIT,
        demo.DEFAULT_QUESTION,
        graph,
        label,
        demo.demo_read_target(),
        trace_sink=None,
        stream=buf,
    )
    assert outcome["ok"] is True
    text = buf.getvalue()
    for marker in (
        "CANDIDATE INVENTORY (§33)",
        "SELECTION SNAPSHOT #1",
        "MODEL STEP (§35)",
        "TOOL STEP (§36)",
        "CONTEXT EVOLUTION (§37)",
        "MEMORY INTERACTION (§38)",
        "BUDGET DEMONSTRATION (§39)",
        "PLANNER + FULL AGENT RUN (§40)",
        "=== PLANNER ===",
        "plan validated: status completed",
        "plan executed: P1 -> P2",
        "EVALUATION REPORT",
        "M15 demonstration summary (§41)",
        "=== STEP 1 ===",
        "=== STEP 2 ===",
        "agent context steps: 001 -> 002",
    ):
        assert marker in text, marker
    assert outcome["diff"]["added"] == [
        "cand_evidence_read",
        "cand_tool_read",
    ]
    assert outcome["diff"]["removed"] == []
    assert outcome["agent_phase"]["ok"] is True
    # both persisted artifacts exist and decode
    root = outcome["selection"]["path"].parent
    assert (root / "context_001.json").is_file()
    assert (root / "context_002.json").is_file()


def test_run_demo_is_deterministic_across_runs(tmp_path):
    def one(suffix):
        manager, ids = _seeded(tmp_path / suffix)
        graph, label = demo.build_graph()
        model = demo.build_offline_model(demo.demo_read_target())
        outcome = demo.run_demo(
            model,
            demo.default_tool_dispatcher(),
            manager,
            _engine(tmp_path / suffix),
            ids,
            CURRENT_COMMIT,
            demo.DEFAULT_QUESTION,
            graph,
            label,
            demo.demo_read_target(),
            trace_sink=None,
            stream=io.StringIO(),
        )
        return outcome

    first = one("run_a")
    second = one("run_b")
    assert first["ok"] is second["ok"] is True
    assert first["evaluation"]["metrics1"] == second["evaluation"][
        "metrics1"
    ]
    assert first["evaluation"]["metrics2"] == second["evaluation"][
        "metrics2"
    ]
    assert first["diff"] == second["diff"]
    assert (
        first["memory_phase"]["outcomes"]
        == second["memory_phase"]["outcomes"]
    )
    items_a = [
        (item.candidate_id, item.selection_reason)
        for item in first["selection"]["snapshot"].package.selected_items
    ]
    items_b = [
        (item.candidate_id, item.selection_reason)
        for item in second["selection"]["snapshot"].package.selected_items
    ]
    assert items_a == items_b


# -- main() ------------------------------------------------------------------


def test_main_offline_succeeds_without_api_key(monkeypatch, capsys,
                                                tmp_path):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = demo.main(
        [
            "--db", str(tmp_path / "m15.db"),
            "--snapshots", str(tmp_path / "ctx"),
            "--traces", str(tmp_path / "traces"),
        ]
    )

    assert code == demo.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "offline (deterministic, scripted model)" in out
    assert "exit code: 0" in out
    run_dir = tmp_path / "ctx" / demo.DEMO_RUN_ID
    for step in ("context_001.json", "context_002.json"):
        path = run_dir / step
        assert path.is_file(), step
        document = json.loads(path.read_text())
        assert document["run_id"] == demo.DEMO_RUN_ID
        assert document["selected_items"]
        assert document["selection_decisions"]


def test_main_writes_no_database_for_context(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    snapshots = tmp_path / "ctx"
    code = demo.main(
        [
            "--db", str(tmp_path / "m15.db"),
            "--snapshots", str(snapshots),
            "--traces", str(tmp_path / "traces"),
        ]
    )
    assert code == demo.EXIT_SUCCESS
    snapshots_written = list(snapshots.rglob("context_*.json"))
    # two runs persist snapshots: the demo's own composition-root
    # selection (phase 4/6/7, run dir "m15_demo") AND the real Agent
    # run's per-step selections (phase 8, one run dir per run) —
    # steps 001 (boundary) and 002 (after the tool batch) each.
    assert sorted(path.name for path in snapshots_written) == [
        "context_001.json",
        "context_001.json",
        "context_002.json",
        "context_002.json",
    ]
    run_dirs = {path.parent for path in snapshots_written}
    assert len(run_dirs) == 2
    for directory in run_dirs:
        assert sorted(p.name for p in directory.glob("*.json")) == [
            "context_001.json",
            "context_002.json",
        ]
    assert not list(snapshots.rglob("*.db"))
    assert not list(snapshots.rglob("*.sqlite*"))
    # the snapshots root holds ONLY json snapshots — never a database
    stray = [
        path
        for path in snapshots.rglob("*")
        if path.is_file() and path.suffix != ".json"
    ]
    assert stray == []


def test_main_live_without_api_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = demo.main(["--live"])

    assert code == demo.EXIT_EXECUTION_FAILED
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_bad_graph_path_fails_clearly(monkeypatch, capsys,
                                           tmp_path):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})

    code = demo.main(
        [
            "--db", str(tmp_path / "m15.db"),
            "--graph", str(tmp_path / "nope.json"),
        ]
    )

    assert code == demo.EXIT_EXECUTION_FAILED
    assert "Graph load failed" in capsys.readouterr().err
