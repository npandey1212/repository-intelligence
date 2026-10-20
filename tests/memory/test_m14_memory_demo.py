"""M14 — two-execution demonstration, offline (§35/§36/§37, §62/§65).

Scripted doubles only — no network, no API key:

 * phase store: planned investigation → validated claim → memory
   persisted + the labelled legacy (earlier-commit) memory for the
   stale demonstration
 * phase retrieve (fresh Agent, same database): search → memory
   enters the new execution → current investigation still runs →
   staleness printed → explicit supersede decision with an
   inspectable relationship
 * process boundary: ``_spawn_retrieve_phase`` really launches a new
   OS process (argv proven), and ``main`` in the default mode uses it
 * main() failure paths: missing API key / no prior memory → clear
   exit codes
 * the demo is a composition root — a source scan shows no second
   loop and no sqlite3 import
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolResult

DEMO_PATH = Path(__file__).resolve().parents[2] / "agentrun_m14_memory_demo.py"


def _load_demo(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    # register before exec: Pydantic resolves the model's annotations
    # through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo(DEMO_PATH, "agentrun_m14_memory_demo")

QUESTION = "Investigate the widget wiring."
CONTEXT_TEXT = "# Architecture Context\n\n- Contract: architecture_context.v1\n"
CONTEXT_RECORD = {
    "artifact": "artifacts/main_2026-09-22_ef11dac/architecture_context/"
    "architecture_context.md",
    "schema_version": "architecture_context.v1",
    "commit_sha": "e" * 40,
    "snapshot_id": "main_2026-09-22_ef11dac",
}
CURRENT_COMMIT = CONTEXT_RECORD["commit_sha"]

PAYLOAD = {
    "finding": "example wiring holds at the current commit.",
    "evidence_paths": ["agents/widget.md"],
    "claims": [
        {
            "statement": "widget registry declares widget:sync",
            "evidence_refs": [
                {"path": "agents/widget.md", "line_start": 4, "line_end": 9}
            ],
        }
    ],
}


class FakeLLMClient:
    def __init__(self, *responses) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = "test-model-1"

    def complete(self, request) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        return self._queue.pop(0)


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        return ToolResult(
            success=True,
            tool=tool_call.name,
            data={
                "path": "agents/widget.md",
                "line_start": 4,
                "line_end": 9,
                "content": "# widget registry\ntools:\n  widget:sync",
                "total_lines": 20,
            },
        )


def _plan(steps):
    payload = {"objective": QUESTION, "steps": steps}
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _tool_calls(*calls):
    return LLMResponse(
        message=Message(
            role="assistant", content=None, tool_calls=list(calls)
        ),
        finish_reason="tool_calls",
    )


def _text(content):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _structured(payload=PAYLOAD):
    return LLMResponse(
        message=Message(role="assistant", content=json.dumps(payload)),
        finish_reason="stop",
        structured_output=payload,
    )


def _call(call_id, path="agents/widget.md"):
    return ToolCall(
        id=call_id, name="read_file", arguments={"path": path}
    )


def _planned_script():
    steps = [
        {"id": "P1", "goal": "Inspect the widget wiring"},
        {"id": "P2", "goal": "Summarize the findings"},
    ]
    return (
        _plan(steps),
        _tool_calls(_call("c1")),
        _text("inspected the registry"),
        _structured(),
    )


def _manager(tmp_path, name="agent_memory.db"):
    manager = demo.build_memory_manager(tmp_path / name)
    return manager


def _agent(manager, responses=None):
    return demo.build_agent(
        FakeLLMClient(*(responses or _planned_script())),
        FakeDispatcher(),
        None,  # no sink: nothing is persisted outside the test
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
    )


def _store_phase(tmp_path, manager=None, stream=None):
    manager = manager or _manager(tmp_path)
    agent = _agent(manager)
    return (
        demo.run_store_phase(
            agent,
            manager,
            demo.DEFAULT_OBJECTIVE_STORE,
            CONTEXT_TEXT,
            CONTEXT_RECORD,
            CURRENT_COMMIT,
            stream=stream,
        ),
        manager,
    )


# ---------------------------------------------------------------------------
# module surface + composition-root discipline
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_the_demo_api():
    for name in (
        "main",
        "build_agent",
        "build_memory_manager",
        "build_memory_context",
        "run_store_phase",
        "run_retrieve_phase",
        "_spawn_retrieve_phase",
        "legacy_commit_for",
        "DEFAULT_OBJECTIVE_STORE",
        "DEFAULT_OBJECTIVE_RECHECK",
        "InvestigationSummary",
        "EXIT_SUCCESS",
        "EXIT_DEMO_FAILED",
        "EXIT_EXECUTION_FAILED",
    ):
        assert hasattr(demo, name), name


def test_demo_is_a_composition_root_no_second_loop_no_sqlite():
    source = DEMO_PATH.read_text()
    assert "import sqlite3" not in source
    # exactly one Agent construction helper; the loop/planner/validator
    # are never re-implemented here
    assert source.count("def build_agent") == 1
    assert "while True" not in source
    assert "_execute_loop" not in source


def test_legacy_commit_is_deterministically_older():
    legacy = demo.legacy_commit_for(CURRENT_COMMIT)
    assert legacy != CURRENT_COMMIT
    assert legacy[:32] == CURRENT_COMMIT[:32]
    assert demo.legacy_commit_for(CURRENT_COMMIT) == legacy


# ---------------------------------------------------------------------------
# phase 1 — store
# ---------------------------------------------------------------------------


def test_store_phase_persists_validated_memory_and_the_legacy_fixture(
    tmp_path,
):
    outcome, manager = _store_phase(tmp_path)
    assert outcome["ok"] is True
    trace = outcome["trace"]
    assert trace.plan is not None
    assert trace.plan["status"] == "completed"
    assert len(trace.plan.get("steps", [])) == 2
    # the agent's own memory + the labelled legacy fixture
    assert len(outcome["memory_ids"]) == 1
    legacy = outcome["legacy_memory"]
    assert legacy.scope.repository == demo.REPOSITORY
    assert legacy.scope.commit == demo.legacy_commit_for(CURRENT_COMMIT)
    assert legacy.statement == demo.LEGACY_STATEMENT
    assert legacy.subject == demo.SUBJECT
    stored = manager.search(repository=demo.REPOSITORY)
    assert len(stored) == 2


def test_store_phase_prints_the_full_recorded_chain(capsys, tmp_path):
    outcome, _ = _store_phase(tmp_path, stream=None)
    out = capsys.readouterr().out
    assert "M14 — Execution 1: TASK 1 → STORE" in out
    assert "P1 [completed]" in out or "P1" in out
    assert "Claims and evidence (current run)" in out
    assert "validation: SUPPORTED" in out
    assert "Memory operations (this run's trace)" in out
    assert "memory.retrieve" in out
    assert "memory.store" in out
    assert "Legacy memory (stale-memory fixture)" in out
    assert "simulated repository history" in out
    assert "only the SQLite file survives" in out


def test_store_phase_reports_failure_when_nothing_was_stored(
    tmp_path, capsys
):
    # a claim that cannot validate (search-only) → no memory formed
    search_hit = ToolResult(
        success=True,
        tool="search_repository",
        data={
            "query": "widget",
            "results": [
                {
                    "path": "agents/widget.md",
                    "line_start": 4,
                    "line_end": 9,
                    "excerpt": "widget",
                }
            ],
        },
    )
    manager = _manager(tmp_path)
    client = FakeLLMClient(
        _plan(
            [
                {"id": "P1", "goal": "Inspect"},
                {"id": "P2", "goal": "Summarize"},
            ]
        ),
        _tool_calls(_call("c1", path="agents/widget.md")),
        _text("inspected"),
        _structured(),
    )
    dispatcher = FakeDispatcher()
    dispatcher.dispatch = lambda tool_call: search_hit
    agent = demo.build_agent(
        client, dispatcher, None, manager, demo.build_memory_context(CURRENT_COMMIT)
    )
    outcome = demo.run_store_phase(
        agent,
        manager,
        demo.DEFAULT_OBJECTIVE_STORE,
        CONTEXT_TEXT,
        CONTEXT_RECORD,
        CURRENT_COMMIT,
    )
    assert outcome["memory_ids"] == []
    assert outcome["ok"] is False


# ---------------------------------------------------------------------------
# phase 2 — retrieve, verify, supersede
# ---------------------------------------------------------------------------


def test_retrieve_phase_finds_prior_memory_supersedes_stale(
    tmp_path, capsys
):
    _, manager = _store_phase(tmp_path)  # phase 1, THIS database
    # a genuinely new execution: fresh Agent, empty conversation
    agent = _agent(manager)
    outcome = demo.run_retrieve_phase(
        agent,
        manager,
        demo.DEFAULT_OBJECTIVE_RECHECK,
        CONTEXT_TEXT,
        CONTEXT_RECORD,
        CURRENT_COMMIT,
    )
    out = capsys.readouterr().out

    assert outcome["ok"] is True
    # prior = task-1 memory + legacy fixture
    assert len(outcome["prior"]) == 2
    assert len(outcome["fresh"]) == 1
    assert outcome["unknown"] == []
    assert [m.statement for m in outcome["stale"]] == [
        demo.LEGACY_STATEMENT
    ]
    # current evidence supported → the stale memory was superseded
    assert outcome["decisions"] == ["superseded"]

    legacy = outcome["stale"][0]
    flipped = manager.get(legacy.memory_id)
    assert flipped.status == "superseded"
    replacement = manager.get(flipped.superseded_by)
    assert replacement.supersedes == legacy.memory_id
    assert replacement.scope.commit == CURRENT_COMMIT
    assert replacement.source_run_id == outcome["trace"].run_id
    # the replacement cites the CURRENT run's evidence refs
    assert replacement.evidence[0].path == "agents/widget.md"

    # printed, explicit (§37)
    assert "M14 — Execution 2: NEW PROCESS" in out
    assert "Retrieved persistent memory" in out
    assert "STALE" in out
    assert "fresh — scope" in out  # tri-state verification printed
    assert "verification: fresh" in out
    assert "SUPERSEDED" in out
    assert "memory.retrieve" in out
    assert "memory.verify" in out
    # proven process boundary narrative
    assert "never existed in this process" in out


def test_retrieve_phase_still_investigates_current_evidence(
    tmp_path, capsys
):
    _, manager = _store_phase(tmp_path)
    dispatcher = FakeDispatcher()
    client = FakeLLMClient(*_planned_script())
    agent = demo.build_agent(
        client,
        dispatcher,
        None,
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
    )
    demo.run_retrieve_phase(
        agent,
        manager,
        demo.DEFAULT_OBJECTIVE_RECHECK,
        CONTEXT_TEXT,
        CONTEXT_RECORD,
        CURRENT_COMMIT,
    )
    # memory is a starting point, never a replacement (§44)
    assert dispatcher.calls, "tools must still run in execution 2"
    # the retrieved memory entered the planner request of THIS run
    planner_user = agent.last_trace.iterations[0].request["messages"][1][
        "content"
    ]
    assert demo.LEGACY_STATEMENT in planner_user
    assert "widget registry declares widget:sync" in planner_user
    # ...and the run produced fresh validated claims of its own
    statuses = [
        entry["status"]
        for entry in agent.last_trace.evidence_validation["results"]
    ]
    assert "SUPPORTED" in statuses


def test_retrieve_phase_without_prior_memory_reports_failure(tmp_path):
    manager = _manager(tmp_path)  # empty database
    agent = _agent(manager)
    outcome = demo.run_retrieve_phase(
        agent,
        manager,
        demo.DEFAULT_OBJECTIVE_RECHECK,
        CONTEXT_TEXT,
        CONTEXT_RECORD,
        CURRENT_COMMIT,
    )
    assert outcome["ok"] is False
    assert outcome["prior"] == ()


def test_retrieve_phase_retains_stale_memory_without_current_support(
    tmp_path, capsys
):
    _, manager = _store_phase(tmp_path)
    # current run answers, but its claim cites a file it never read →
    # UNSUPPORTED → nothing validated to replace the stale memory
    unsupported = {
        "finding": "a guess that no inspected file supports.",
        "evidence_paths": ["agents/never-read.md"],
        "claims": [
            {
                "statement": "never-read.md supposedly proves everything",
                "evidence_refs": [{"path": "agents/never-read.md"}],
            }
        ],
    }
    client = FakeLLMClient(
        _plan(
            [
                {"id": "P1", "goal": "Inspect"},
                {"id": "P2", "goal": "Summarize"},
            ]
        ),
        _tool_calls(_call("c1")),
        _text("inspected"),
        _structured(unsupported),
    )
    agent = demo.build_agent(
        client,
        FakeDispatcher(),
        None,
        manager,
        demo.build_memory_context(CURRENT_COMMIT),
    )
    outcome = demo.run_retrieve_phase(
        agent,
        manager,
        demo.DEFAULT_OBJECTIVE_RECHECK,
        CONTEXT_TEXT,
        CONTEXT_RECORD,
        CURRENT_COMMIT,
    )
    out = capsys.readouterr().out
    assert outcome["decisions"] == ["retained"]
    assert "RETAINED" in out
    # nothing was silently rewritten
    legacy = [m for m in outcome["prior"]
              if m.statement == demo.LEGACY_STATEMENT][0]
    assert manager.get(legacy.memory_id).status == "verified"


# ---------------------------------------------------------------------------
# process boundary (§36)
# ---------------------------------------------------------------------------


def test_spawn_retrieve_phase_launches_a_real_new_process(monkeypatch):
    captured = {}

    class Completed:
        returncode = 23

    def _fake_run(argv):
        captured["argv"] = argv
        return Completed()

    monkeypatch.setattr(demo.subprocess, "run", _fake_run)

    code = demo._spawn_retrieve_phase(
        model="gpt-4o-mini", db_path="/tmp/m.db", question="Q?"
    )

    assert code == 23
    argv = captured["argv"]
    assert argv[0] == sys.executable
    assert argv[1].endswith("agentrun_m14_memory_demo.py")
    assert argv[2:4] == ["--phase", "retrieve"]
    assert "--model" in argv and "gpt-4o-mini" in argv
    assert "--db" in argv and "/tmp/m.db" in argv
    assert "--question" in argv and "Q?" in argv


def test_main_default_mode_stores_then_spawns_the_second_process(
    monkeypatch, capsys, tmp_path
):
    fake, dispatcher, db = _wire_main(monkeypatch, tmp_path)
    spawned = {}
    monkeypatch.setattr(
        demo,
        "_spawn_retrieve_phase",
        lambda model=None, db_path=None, question=None: spawned.update(
            model=model, db_path=db_path, question=question
        )
        or demo.EXIT_SUCCESS,
    )

    code = demo.main(["--model", "gpt-4o-mini", "--db", db])

    assert code == demo.EXIT_SUCCESS
    # execution 1 really ran here before the spawn
    assert len(fake.requests) == 4
    assert dispatcher.calls
    # execution 2 is delegated to a NEW OS process
    assert spawned == {
        "model": "gpt-4o-mini",
        "db_path": db,
        "question": None,
    }
    # ...and execution 1 persisted before the child was launched
    from repo_intelligence.memory import MemoryManager

    assert len(MemoryManager(db).search(repository=demo.REPOSITORY)) == 2
    out = capsys.readouterr().out
    assert "Phase store runs in THIS process" in out
    assert "M14 — Execution 1: TASK 1 → STORE" in out
    assert "M14 — Execution 2" not in out


# ---------------------------------------------------------------------------
# main() wiring — offline paths and failure codes
# ---------------------------------------------------------------------------


def test_main_without_api_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = demo.main(["--phase", "store"])

    assert code == demo.EXIT_EXECUTION_FAILED
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def _wire_main(monkeypatch, tmp_path, responses=None, db_name="m.db"):
    monkeypatch.setattr(demo, "load_env", lambda *a, **k: {})
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key-for-this-test")
    monkeypatch.setattr(
        demo,
        "load_architecture_context",
        lambda *a, **k: (CONTEXT_TEXT, dict(CONTEXT_RECORD)),
    )
    script = responses or (_planned_script() + _planned_script())
    fake = FakeLLMClient(*script)
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(
        demo, "OpenAILLMClient", lambda model=None: fake
    )
    monkeypatch.setattr(
        demo, "default_tool_dispatcher", lambda: dispatcher
    )
    monkeypatch.setattr(demo, "JsonFileTraceSink", lambda: None)
    return fake, dispatcher, str(tmp_path / db_name)


def test_main_store_phase_runs_the_offline_path(
    monkeypatch, capsys, tmp_path
):
    _, _, db = _wire_main(monkeypatch, tmp_path)

    code = demo.main(["--phase", "store", "--db", db])

    out = capsys.readouterr().out
    assert code == demo.EXIT_SUCCESS
    assert "M14 — Execution 1: TASK 1 → STORE" in out
    assert "validation: SUPPORTED" in out
    assert "Legacy memory (stale-memory fixture)" in out
    assert "exit code: 0 (phase store)" in out
    # the database really holds both memories afterwards
    from repo_intelligence.memory import MemoryManager

    manager = MemoryManager(db)
    assert len(manager.search(repository=demo.REPOSITORY)) == 2


def test_main_retrieve_phase_succeeds_with_prior_memory(
    monkeypatch, capsys, tmp_path
):
    _, _, db = _wire_main(monkeypatch, tmp_path)
    assert demo.main(["--phase", "store", "--db", db]) == demo.EXIT_SUCCESS
    capsys.readouterr()

    code = demo.main(["--phase", "retrieve", "--db", db])

    out = capsys.readouterr().out
    assert code == demo.EXIT_SUCCESS
    assert "M14 — Execution 2: NEW PROCESS" in out
    assert "SUPERSEDED" in out
    assert "exit code: 0 (phase retrieve: 2 retrieved, 1 stale" in out


def test_main_retrieve_phase_fails_when_the_store_is_empty(
    monkeypatch, capsys, tmp_path
):
    _wire_main(monkeypatch, tmp_path, db_name="empty.db")

    code = demo.main(
        ["--phase", "retrieve", "--db", str(tmp_path / "empty.db")]
    )

    err = capsys.readouterr().err
    assert code == demo.EXIT_DEMO_FAILED
    assert "retrieved NO prior memory" in err


def test_retrieve_phase_uses_the_recheck_objective_by_default(
    monkeypatch, tmp_path
):
    fake, _, db = _wire_main(monkeypatch, tmp_path)
    demo.main(["--phase", "store", "--db", db])
    boundary = len(fake.requests)

    demo.main(["--phase", "retrieve", "--db", db])

    # the planner request of execution 2 carries Task 2's objective
    planner_user = fake.requests[boundary].messages[1].content
    assert demo.DEFAULT_OBJECTIVE_RECHECK in planner_user
