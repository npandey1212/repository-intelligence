"""Agent execution trace tests (observability MVP) — §24/§25/§26.

The trace must make one run inspectable as a sequence of observable
Agent → LLM → Tool → observation events, WITHOUT any hidden
chain-of-thought: no thought fields, no fabricated reasoning. Covers
the nine required trace tests (final answer / one tool call / multiple
tool calls / tool failure / LLM failure / execution budget / exact
arguments / no hidden reasoning / provenance), iteration ordering,
trace-failure policy, JSON-canonical serialization, Markdown
projection, and sink behavior.

Loop mechanics live in ``test_agent_loop.py``; the CLI's file sink in
``test_agent_cli.py``; dispatcher/tool mechanics in their own files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import repo_intelligence.artifacts as artifacts
from conftest import enable_search
from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
    InMemoryTraceSink,
    JsonFileTraceSink,
    default_trace_root,
    render_markdown,
)
from repo_intelligence.agent.trace import (
    BANNED_REASONING_FIELDS,
    TERMINATION_REASONS,
)
from repo_intelligence.llm import LLMError, LLMResponse, Message, ToolCall, Usage
from repo_intelligence.tool_dispatcher import default_tool_dispatcher
from repo_intelligence.tools import ToolError, ToolResult

QUESTION = "How does the data onboarding workflow work?"
ANSWER = "The workflow has four deterministic stages."

RUN_ID = "run_fixture_alpha"
COMMIT_SHA = "b" * 40

CONTEXT_TEXT = """# Architecture Context

- Contract: `architecture_context.v1`

## Repository

- Name: `sample-repo-alpha`
- Pinned commit: {}
- Snapshot: {}
""".format(
    COMMIT_SHA, RUN_ID
)


class FakeLLMClient:
    """Queued responses (or queued exceptions); records every request."""

    def __init__(self, *responses, model=None) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = model

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than queued responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    """Records dispatches; returns queued results in order (else a
    generic success for the called tool)."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


class RaisingDispatcher:
    """Simulates an application-level dispatch failure."""

    def __init__(self, exc) -> None:
        self._exc = exc

    def dispatch(self, tool_call):
        raise self._exc


class RaisingSink:
    """Simulates a trace persistence failure (§26)."""

    def __init__(self, exc) -> None:
        self._exc = exc

    def write(self, trace):
        raise self._exc


def _text_response(content=ANSWER, usage=None):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
        usage=usage,
    )


def _tool_calls_response(*calls, usage=None):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
        usage=usage,
    )


def _call(name, arguments=None, call_id="call_1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _run(fake, dispatcher=None, sink=None, question=QUESTION):
    agent = Agent(
        llm_client=fake, tool_dispatcher=dispatcher, trace_sink=sink
    )
    outcome = agent.run(question)
    return agent, outcome


def _walk_keys(value):
    """Yield every mapping key anywhere inside a JSON-ready value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, list):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


def _controlled_pinned_context(tmp_path: Path, monkeypatch):
    """Deterministic pinned-run context for the real repository tools
    (same layout contract as the loop/CLI tests — per-file helper)."""
    root = tmp_path / "artifacts"
    run_dir = root / RUN_ID
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "architecture_context").mkdir()
    (run_dir / "repository").mkdir()
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture_context" / "architecture_context.md").write_text(
        CONTEXT_TEXT, encoding="utf-8"
    )
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {
                    "name": "sample-repo-alpha",
                    "url": "https://example.invalid/sample.git",
                },
                "snapshot": {"ref": "main", "commit_sha": COMMIT_SHA},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "repository" / "README.md").write_text(
        "# Sample Project\n\nThe onboarding workflow lives here.\n",
        encoding="utf-8",
    )
    # M18: isolated store + deterministic index for search_repository.
    enable_search(run_dir / "repository", monkeypatch, tmp_path)
    monkeypatch.setattr(artifacts, "default_artifacts_root", lambda: root)
    return run_dir


# ---------------------------------------------------------------------------
# §24 Test 1 — final answer without tools
# ---------------------------------------------------------------------------


def test_final_answer_without_tools_traces_one_iteration():
    fake = FakeLLMClient(_text_response())
    agent, outcome = _run(fake, FakeDispatcher())

    assert outcome == ANSWER
    trace = agent.last_trace
    assert trace.summary.total_iterations == 1
    assert trace.summary.total_tool_calls == 0
    assert trace.summary.termination_reason == "FINAL_ANSWER"
    assert trace.summary.final_answer == ANSWER
    assert trace.summary.tools_used == []
    iteration = trace.iterations[0]
    assert iteration.iteration == 1
    assert iteration.response_type == "FINAL_ANSWER"
    assert iteration.decision == {"type": "FINAL_ANSWER"}
    assert iteration.final_answer == ANSWER
    # no repository was analyzed — no provenance is invented
    assert trace.repository is None


# ---------------------------------------------------------------------------
# §24 Test 2 — one tool call
# ---------------------------------------------------------------------------


def test_one_tool_call_traces_two_iterations():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "onboarding"})),
        _text_response(),
    )
    agent, outcome = _run(fake, FakeDispatcher())

    assert outcome == ANSWER
    trace = agent.last_trace
    assert trace.summary.total_iterations == 2
    assert trace.summary.total_tool_calls == 1
    assert trace.summary.termination_reason == "FINAL_ANSWER"

    first, second = trace.iterations
    assert first.response_type == "TOOL_CALL"
    assert first.decision == {"type": "TOOL_CALL", "tool": "search_repository"}
    assert first.tool_calls[0]["name"] == "search_repository"
    assert first.tool_results[0]["success"] is True
    assert first.observations[0] == {
        "type": "TOOL_RESULT",
        "tool": "search_repository",
        "success": True,
        "tool_call_id": "call_1",
    }
    assert second.response_type == "FINAL_ANSWER"
    assert second.decision == {"type": "FINAL_ANSWER"}


# ---------------------------------------------------------------------------
# §24 Test 3 — multiple tool calls (exact ordering) + §23 full flow
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_preserve_exact_order():
    fake = FakeLLMClient(
        _tool_calls_response(
            _call(
                "search_repository",
                {"query": "data onboarding"},
                call_id="call_s",
            )
        ),
        _tool_calls_response(
            _call(
                "read_file",
                {"path": "docs/workflows.md"},
                call_id="call_r",
            )
        ),
        _text_response(),
    )
    agent, outcome = _run(fake, FakeDispatcher())

    assert outcome == ANSWER
    trace = agent.last_trace
    assert trace.summary.total_iterations == 3
    assert trace.summary.total_tool_calls == 2
    assert [it.decision["type"] for it in trace.iterations] == [
        "TOOL_CALL",
        "TOOL_CALL",
        "FINAL_ANSWER",
    ]
    # canonical developer-facing flow: search → read → final answer
    assert trace.summary.tools_used == ["search_repository", "read_file"]
    assert [
        result["tool"] for result in trace.iterations[0].tool_results
    ] == ["search_repository"]
    assert [
        result["tool"] for result in trace.iterations[1].tool_results
    ] == ["read_file"]


# ---------------------------------------------------------------------------
# §24 Test 4 — tool failure appears in the trace
# ---------------------------------------------------------------------------


def test_failed_tool_result_is_recorded_in_trace():
    failed_read = ToolResult(
        success=False,
        tool="read_file",
        error=ToolError(code="PATH_NOT_FOUND", message="no such path"),
    )
    fake = FakeLLMClient(
        _tool_calls_response(_call("read_file", {"path": "missing.py"})),
        _text_response("I could not read that file."),
    )
    agent, outcome = _run(fake, FakeDispatcher(failed_read))

    assert outcome == "I could not read that file."
    trace = agent.last_trace
    assert trace.summary.termination_reason == "FINAL_ANSWER"
    iteration = trace.iterations[0]
    result = iteration.tool_results[0]
    assert result["success"] is False
    assert result["error"] == {
        "code": "PATH_NOT_FOUND",
        "message": "no such path",
    }
    assert iteration.observations[0]["success"] is False
    # a failed tool is not terminal — it became an observation
    assert len(fake.requests) == 2


# ---------------------------------------------------------------------------
# §24 Test 5 — LLM failure
# ---------------------------------------------------------------------------


def test_llm_failure_records_llm_error_termination():
    fake = FakeLLMClient(LLMError("provider unavailable"))
    agent = Agent(llm_client=fake, tool_dispatcher=FakeDispatcher())

    with pytest.raises(LLMError, match="provider unavailable"):
        agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.summary.termination_reason == "LLM_ERROR"
    assert trace.summary.termination_reason in TERMINATION_REASONS
    assert trace.completed_at is not None
    iteration = trace.iterations[0]
    assert iteration.response_type == "ERROR"
    assert iteration.decision == {"type": "ERROR"}
    assert "provider unavailable" in iteration.error
    assert iteration.request["message_count"] == 2


# ---------------------------------------------------------------------------
# §24 Test 6 — execution budget (M8.5)
# ---------------------------------------------------------------------------


def test_execution_budget_records_execution_limit_reached_termination():
    # One tool response per executed step, plus the extra request the
    # budget blocks (the raise happens BEFORE that call is executed).
    budget = AgentExecutionConfig().max_tool_calls
    responses = [
        _tool_calls_response(_call("search_repository", {"query": str(step)}))
        for step in range(budget + 1)
    ]
    fake = FakeLLMClient(*responses)
    agent = Agent(llm_client=fake, tool_dispatcher=FakeDispatcher())

    with pytest.raises(AgentError, match="execution budget exhausted"):
        agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert trace.summary.termination_reason in TERMINATION_REASONS
    # budget observability: configured, consumed, blocked
    assert trace.summary.max_tool_calls == budget
    assert trace.summary.tool_calls_executed == budget
    assert trace.summary.tool_calls_blocked == 1
    assert trace.summary.total_iterations == budget + 1
    assert trace.summary.total_tool_calls == budget + 1
    limiting = trace.iterations[-1]
    assert limiting.iteration == budget + 1
    assert limiting.response_type == "TOOL_CALL"
    assert limiting.decision["type"] == "LIMIT_REACHED"
    # the unexecuted request has no result/observation — but IS
    # recorded as blocked, explicitly distinct from a ToolResult
    assert limiting.tool_results == []
    assert limiting.observations == []
    assert len(limiting.blocked_tool_calls) == 1
    assert limiting.blocked_tool_calls[0]["tool"] == "search_repository"
    assert limiting.blocked_tool_calls[0]["reason"] == "EXECUTION_LIMIT_REACHED"


# ---------------------------------------------------------------------------
# §24 Test 7 — exact arguments
# ---------------------------------------------------------------------------


def test_trace_arguments_equal_the_actual_tool_call_arguments():
    arguments = {
        "query": "  data onboarding  ",  # whitespace must survive verbatim
        "max_results": 7,
        "case_sensitive": True,
    }
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", arguments)),
        _text_response(),
    )
    dispatcher = FakeDispatcher()
    agent, _ = _run(fake, dispatcher)

    trace = agent.last_trace
    recorded = trace.iterations[0].tool_calls[0]["arguments"]
    assert recorded == arguments
    assert recorded == dispatcher.calls[0].arguments  # exactly what ran
    assert recorded["query"] == "  data onboarding  "


# ---------------------------------------------------------------------------
# §24 Test 8 — no hidden reasoning field (+ §27 no secrets)
# ---------------------------------------------------------------------------


def test_trace_contains_no_hidden_reasoning_or_secret_fields():
    fake = FakeLLMClient(
        _tool_calls_response(
            _call("get_architecture_context", {}), usage=Usage(
                input_tokens=11, output_tokens=7, total_tokens=18
            )
        ),
        _text_response(),
    )
    agent, _ = _run(fake, FakeDispatcher())

    payload = agent.last_trace.to_dict()
    serialized = json.dumps(payload).lower()

    secret_banned = (
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "password",
        "secret",
        "headers",
    )
    for key in _walk_keys(payload):
        lowered = key.lower()
        for banned in BANNED_REASONING_FIELDS:
            assert banned not in lowered, key
        for banned in secret_banned:
            assert lowered != banned, key

    # not just keys — nothing in the whole payload may reference
    # chain-of-thought / fabricated reasoning
    for banned in BANNED_REASONING_FIELDS:
        assert banned not in serialized
    assert "thought" not in serialized
    assert "chain of thought" not in serialized

    # the human-readable summary stays observable (action, not mind)
    summary = agent.last_trace.iterations[0].reason_summary
    assert summary.startswith("Model requested ")


# ---------------------------------------------------------------------------
# §24 Test 9 — provenance (pinned snapshot)
# ---------------------------------------------------------------------------


def test_repository_provenance_matches_the_pinned_snapshot(
    tmp_path, monkeypatch
):
    _controlled_pinned_context(tmp_path, monkeypatch)
    fake = FakeLLMClient(
        _tool_calls_response(
            _call("search_repository", {"query": "onboarding workflow"})
        ),
        _text_response(),
    )
    agent, outcome = _run(fake, default_tool_dispatcher())

    assert outcome == ANSWER
    trace = agent.last_trace
    assert trace.repository is not None
    # consistent with the pinned snapshot actually used by the tool —
    # never a fabricated commit/run
    assert trace.repository["snapshot_id"] == RUN_ID
    assert trace.repository["commit_sha"] == COMMIT_SHA
    assert trace.repository["repository"] == "sample-repo-alpha"


# ---------------------------------------------------------------------------
# §6 / §25 — numbering and ordering
# ---------------------------------------------------------------------------


def test_iterations_are_numbered_from_one_and_ordered():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "a"})),
        _tool_calls_response(_call("read_file", {"path": "b"})),
        _text_response(),
    )
    agent, _ = _run(fake, FakeDispatcher())
    payload = agent.last_trace.to_dict()

    numbers = [item["iteration"] for item in payload["iterations"]]
    assert numbers == [1, 2, 3]
    # serialization order = execution order (never reordered)
    assert [item["decision"]["type"] for item in payload["iterations"]] == [
        "TOOL_CALL",
        "TOOL_CALL",
        "FINAL_ANSWER",
    ]
    # within a tool iteration: tool_call → tool_result → observation
    for item in payload["iterations"][:2]:
        assert list(item.keys())[:3] == [
            "iteration",
            "response_type",
            "decision",
        ]
        call_ids = [call["id"] for call in item["tool_calls"]]
        result_ids = [r["tool_call_id"] for r in item["tool_results"]]
        observation_ids = [o["tool_call_id"] for o in item["observations"]]
        assert call_ids == result_ids == observation_ids


def test_request_and_response_metadata_are_recorded():
    usage = Usage(input_tokens=11, output_tokens=7, total_tokens=18)
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {}), usage=usage),
        _text_response(),
        model="test-model-1",
    )
    agent, _ = _run(fake, FakeDispatcher())
    first, second = agent.last_trace.iterations

    assert first.request["message_count"] == 2
    assert first.request["tool_definition_count"] == 4
    assert first.request["model"] == "test-model-1"
    assert first.response["finish_reason"] == "tool_calls"
    assert first.response["tool_call_count"] == 1
    assert first.response["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    # full conversation history grows — but only its metadata is traced
    assert second.request["message_count"] == 4
    assert second.response["usage"] is None


# ---------------------------------------------------------------------------
# Termination reasons (§16) — the explicit small set
# ---------------------------------------------------------------------------


def test_missing_dispatcher_records_tool_dispatch_error():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {})),
    )
    agent = Agent(llm_client=fake)  # no dispatcher (M8.2 contract)

    with pytest.raises(AgentError, match="not supported"):
        agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.summary.termination_reason == "TOOL_DISPATCH_ERROR"
    iteration = trace.iterations[0]
    assert iteration.response_type == "TOOL_CALL"
    assert iteration.tool_calls[0]["name"] == "search_repository"
    assert iteration.tool_results == []  # recognized, not executed
    assert iteration.observations == []


def test_invalid_llm_response_records_invalid_llm_response():
    fake = FakeLLMClient(LLMResponse(message=Message(role="assistant")))
    agent = Agent(llm_client=fake, tool_dispatcher=FakeDispatcher())

    with pytest.raises(AgentError, match="no textual answer"):
        agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.summary.termination_reason == "INVALID_LLM_RESPONSE"
    iteration = trace.iterations[0]
    assert iteration.response_type == "INVALID_RESPONSE"
    assert iteration.response["has_text"] is False
    assert iteration.response["tool_call_count"] == 0


def test_dispatch_exception_records_tool_dispatch_error_and_reraises():
    fake = FakeLLMClient(
        _tool_calls_response(_call("read_file", {"path": "x"})),
    )
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=RaisingDispatcher(RuntimeError("registry offline")),
    )

    with pytest.raises(RuntimeError, match="registry offline"):
        agent.run(QUESTION)

    trace = agent.last_trace
    assert trace.summary.termination_reason == "TOOL_DISPATCH_ERROR"
    iteration = trace.iterations[0]
    assert "registry offline" in iteration.error
    assert iteration.tool_results == []


# ---------------------------------------------------------------------------
# §18 / §19 / §26 — automatic generation, sinks, failure policy
# ---------------------------------------------------------------------------


def test_trace_is_generated_automatically_without_a_sink():
    fake = FakeLLMClient(_text_response())
    agent, _ = _run(fake, FakeDispatcher())  # trace_sink=None

    assert agent.last_trace is not None
    assert agent.last_trace.summary.termination_reason == "FINAL_ANSWER"
    assert agent.last_trace_error is None


def test_in_memory_sink_receives_the_trace_of_every_run():
    sink = InMemoryTraceSink()
    fake = FakeLLMClient(_text_response())
    agent, _ = _run(fake, FakeDispatcher(), sink=sink)

    assert len(sink.traces) == 1
    assert sink.traces[0] is agent.last_trace


def test_sink_failure_does_not_corrupt_the_agent_result():
    sink = RaisingSink(OSError("disk full"))
    fake = FakeLLMClient(_text_response())
    agent, outcome = _run(fake, FakeDispatcher(), sink=sink)

    # execution unaffected, failure surfaced as error metadata only
    assert outcome == ANSWER
    assert "OSError" in agent.last_trace_error
    assert "disk full" in agent.last_trace_error
    assert "disk full" in agent.last_trace.sink_error
    assert agent.last_trace.summary.termination_reason == "FINAL_ANSWER"


def test_file_sink_writes_canonical_json_and_markdown(tmp_path):
    sink = JsonFileTraceSink(tmp_path)
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "x"})),
        _text_response(),
    )
    agent, _ = _run(fake, FakeDispatcher(), sink=sink)

    json_path = sink.json_path(agent.last_trace.run_id)
    md_path = sink.markdown_path(agent.last_trace.run_id)
    # layout: <root>/<run_id>/trace.{json,md} — no agent_trace/ segment
    assert json_path.parent == tmp_path / agent.last_trace.run_id
    assert json_path.name == "trace.json"
    assert md_path.name == "trace.md"
    assert "agent_trace" not in json_path.parts
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == agent.last_trace.to_dict()
    assert payload["summary"]["termination_reason"] == "FINAL_ANSWER"

    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(agent.last_trace)


# ---------------------------------------------------------------------------
# Runtime trace relocation (M8.7) — runtime/agent_runs, not artifacts/
# ---------------------------------------------------------------------------


def test_default_trace_root_is_runtime_agent_runs():
    root = default_trace_root()
    assert root == artifacts.PROJECT_ROOT / "runtime" / "agent_runs"
    assert root.name == "agent_runs"
    assert root.parent.name == "runtime"
    # execution telemetry never lives under the artifacts tree
    artifacts_root = artifacts.default_artifacts_root()
    assert artifacts_root not in root.parents
    assert not str(root).startswith(str(artifacts_root))


def test_default_file_sink_writes_trace_under_runtime_agent_runs(
    tmp_path, monkeypatch
):
    import repo_intelligence.agent.trace_sink as trace_sink_module

    monkeypatch.setattr(trace_sink_module, "PROJECT_ROOT", tmp_path)
    sink = JsonFileTraceSink()  # no root injected → default runtime tree
    assert sink.root == tmp_path / "runtime" / "agent_runs"

    fake = FakeLLMClient(_text_response())
    agent, _ = _run(fake, FakeDispatcher(), sink=sink)
    run_id = agent.last_trace.run_id

    json_path = sink.json_path(run_id)
    assert json_path == tmp_path / "runtime" / "agent_runs" / run_id / "trace.json"
    assert json_path.exists()
    assert (json_path.parent / "trace.md").exists()

    # canonical schema unchanged; runtime tree only, artifacts untouched
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == agent.last_trace.to_dict()
    assert "agent_trace" not in json_path.parts
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------------
# JSON canonical shape + Markdown projection (§20 / §22)
# ---------------------------------------------------------------------------


def test_fixed_trace_serializes_deterministically():
    from repo_intelligence.agent import AgentIteration, AgentTrace

    trace = AgentTrace(
        run_id="example-run",
        started_at="2026-01-01T00:00:00.000Z",
        user_question="How does the data onboarding workflow work?",
    )
    iteration = AgentIteration(
        iteration=1,
        request={
            "message_count": 2,
            "tool_definition_count": 3,
            "model": "gpt-4o-mini",
        },
    )
    iteration.response = {
        "finish_reason": "stop",
        "tool_call_count": 0,
        "has_text": True,
        "usage": None,
    }
    iteration.close(
        response_type="FINAL_ANSWER",
        decision={"type": "FINAL_ANSWER"},
        reason_summary="Model returned a final textual answer (42 characters).",
        final_answer=ANSWER,
    )
    trace.iterations.append(iteration)
    trace.terminate("FINAL_ANSWER", final_answer=ANSWER)
    trace.finalize(completed_at="2026-01-01T00:00:01.000Z")

    payload = trace.to_dict()
    assert payload == {
        "run_id": "example-run",
        "started_at": "2026-01-01T00:00:00.000Z",
        "completed_at": "2026-01-01T00:00:01.000Z",
        "agent": {"name": "repo_intelligence.agent", "version": "0.3.0"},
        "user_question": "How does the data onboarding workflow work?",
        "iterations": [
            {
                "iteration": 1,
                "response_type": "FINAL_ANSWER",
                "decision": {"type": "FINAL_ANSWER"},
                "request": {
                    "message_count": 2,
                    "tool_definition_count": 3,
                    "model": "gpt-4o-mini",
                },
                "response": {
                    "finish_reason": "stop",
                    "tool_call_count": 0,
                    "has_text": True,
                    "usage": None,
                },
                "tool_calls": [],
                "tool_results": [],
                "observations": [],
                "final_answer": ANSWER,
                "reason_summary": (
                    "Model returned a final textual answer (42 characters)."
                ),
            }
        ],
        "summary": {
            "total_iterations": 1,
            "total_tool_calls": 0,
            "tools_used": [],
            "termination_reason": "FINAL_ANSWER",
            "final_answer": ANSWER,
            "max_tool_calls": None,
            "tool_calls_executed": 0,
            "tool_calls_blocked": 0,
        },
    }
    # stable key order + round trip = canonical machine format
    assert list(payload.keys()) == [
        "run_id",
        "started_at",
        "completed_at",
        "agent",
        "user_question",
        "iterations",
        "summary",
    ]
    assert json.loads(trace.to_json()) == payload
    assert trace.to_json() == trace.to_json()


def test_markdown_projection_shows_the_complete_flow():
    fake = FakeLLMClient(
        _tool_calls_response(
            _call("search_repository", {"query": "data onboarding"})
        ),
        _tool_calls_response(_call("read_file", {"path": "docs/x.md"})),
        _text_response(),
    )
    agent, _ = _run(fake, FakeDispatcher())
    trace = agent.last_trace

    markdown = render_markdown(trace)
    assert markdown.startswith("# Agent Execution Trace")
    assert trace.user_question in markdown
    assert "## Iteration 1" in markdown
    assert "## Iteration 2" in markdown
    assert "## Iteration 3" in markdown
    assert "**Decision:** `TOOL_CALL`" in markdown
    assert "**Tool:** `search_repository`" in markdown
    assert '"query": "data onboarding"' in markdown
    assert "**Result:** SUCCESS" in markdown
    assert "**Observation:** search_repository result returned to the Agent." in markdown
    assert "**Decision:** `FINAL_ANSWER`" in markdown
    assert "* Total iterations: 3" in markdown
    assert "* Tool calls: 2" in markdown
    assert "Termination: `FINAL_ANSWER`" in markdown
    assert "thought" not in markdown.lower()


def test_large_tool_data_is_clipped_in_the_trace():
    big_text = "x" * 20000
    huge = ToolResult(
        success=True,
        tool="read_file",
        data={"content": big_text, "path": "big.py"},
    )
    fake = FakeLLMClient(
        _tool_calls_response(_call("read_file", {"path": "big.py"})),
        _text_response(),
    )
    agent, outcome = _run(fake, FakeDispatcher(huge))

    assert outcome == ANSWER
    recorded = agent.last_trace.iterations[0].tool_results[0]["data"]
    assert len(recorded["content"]) < len(big_text)
    assert "[trace clipped:" in recorded["content"]
    assert recorded["path"] == "big.py"  # untouched short values remain
