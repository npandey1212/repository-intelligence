"""Standalone Agent command (``python -m repo_intelligence.agent``) tests.

The concrete client is monkeypatched at the composition root, so the
full chain — ``__main__`` → ``Agent`` → injected client — runs without
any provider, key, or network.

The multi-step loop test additionally points the default artifacts root
at a controlled pinned-run fixture under ``tmp_path``, so the real
``search_repository`` resolves a deterministic snapshot context through
its normal production path instead of developer-machine artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import repo_intelligence.agent.__main__ as agent_main
import repo_intelligence.artifacts as artifacts
from conftest import enable_search
from repo_intelligence.agent import SYSTEM_PROMPT
from repo_intelligence.llm import (
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
)


class _FakeLLMClient:
    """Stands in for OpenAILLMClient at the composition root.

    ``responses`` (optional queue) lets a test script a multi-step
    loop: tool request first, final answer second. ``response`` remains
    the single canned fallback for one-shot answers.
    """

    response = None
    responses = None
    error = None
    last_request = None
    requests = None

    def __init__(self, model=None):
        self.model = model

    def complete(self, request):
        type(self).requests.append(request)
        type(self).last_request = request
        if type(self).error is not None:
            raise type(self).error
        if type(self).responses:
            return type(self).responses.pop(0)
        return type(self).response


class _MemoryTraceSink:
    """CLI test double for the file sink: keeps traces in memory so
    tests never write into the real artifacts/ directory."""

    def __init__(self) -> None:
        self.traces = []

    def write(self, trace) -> None:
        self.traces.append(trace)


@pytest.fixture(autouse=True)
def _fake_client(monkeypatch):
    _FakeLLMClient.response = None
    _FakeLLMClient.responses = None
    _FakeLLMClient.error = None
    _FakeLLMClient.last_request = None
    _FakeLLMClient.requests = []
    monkeypatch.setattr(agent_main, "OpenAILLMClient", _FakeLLMClient)
    # Default: in-memory trace sink (no files). Trace-file tests
    # override this fixture with a real JsonFileTraceSink(tmp_path).
    sink = _MemoryTraceSink()
    monkeypatch.setattr(agent_main, "_build_trace_sink", lambda: sink)
    yield _FakeLLMClient


@pytest.fixture()
def trace_sink(monkeypatch):
    """Overrides the autouse in-memory sink and exposes it for
    trace-focused CLI assertions."""
    sink = _MemoryTraceSink()
    monkeypatch.setattr(agent_main, "_build_trace_sink", lambda: sink)
    return sink


def _controlled_pinned_context(tmp_path: Path, monkeypatch) -> Path:
    """Minimal deterministic pinned-run context for ``search_repository``.

    Builds the run layout the real tool requires in order to resolve the
    pinned snapshot (facts + architecture run markers, snapshot artifact,
    and the pinned ``repository/`` checkout) under *tmp_path*, then points
    the default artifacts root at it. Only the root location is
    redirected — the production resolution path (discover the run →
    repository dir → search) and all snapshot validation are unchanged,
    so the CLI test neither depends on developer-machine artifacts nor
    bypasses the snapshot architecture. Cleanup is automatic (tmp_path).
    """
    root = tmp_path / "artifacts"
    run_dir = root / "run_fixture_alpha"
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "repository").mkdir(parents=True)
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {
                    "name": "sample-repo-alpha",
                    "url": "https://example.invalid/sample.git",
                },
                "snapshot": {"ref": "main", "commit_sha": "b" * 40},
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


def test_standalone_prints_question_and_answer(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(
            role="assistant",
            content="It builds deterministic repository intelligence artifacts.",
        ),
        finish_reason="stop",
    )
    exit_code = agent_main.main(["--question", "What is this repository for?"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Question:" in captured.out
    assert "What is this repository for?" in captured.out
    assert "Answer:" in captured.out
    assert "It builds deterministic repository intelligence artifacts." in captured.out
    assert captured.err == ""


def test_standalone_uses_default_question_and_agent_request(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="ok")
    )
    exit_code = agent_main.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    request = _FakeLLMClient.last_request
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.messages[1].content == agent_main.DEFAULT_QUESTION
    assert agent_main.DEFAULT_QUESTION in captured.out


def test_standalone_llm_failure_exits_six_on_stderr(capsys):
    _FakeLLMClient.error = LLMError("OPENAI_API_KEY is not configured.")
    exit_code = agent_main.main([])
    captured = capsys.readouterr()

    assert exit_code == 6
    assert "LLMError: OPENAI_API_KEY is not configured." in captured.err
    assert captured.out == ""


def test_standalone_invalid_arguments_become_an_observation(capsys):
    # {} is invalid for search_repository → the registered tool returns
    # its structured failure, which the Agent feeds back to the model as
    # an observation (M8.4): the loop continues, the CLI exits 0.
    _FakeLLMClient.responses = [
        LLMResponse(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="search_repository", arguments={})
                ],
            ),
            finish_reason="tool_calls",
        ),
        LLMResponse(
            message=Message(
                role="assistant",
                content="No valid search could be performed.",
            ),
            finish_reason="stop",
        ),
    ]
    exit_code = agent_main.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Tool steps:" in captured.out
    assert "1. search_repository -> success=False" in captured.out
    assert "Answer:" in captured.out
    assert "No valid search could be performed." in captured.out
    # the failure was observed by the model, not swallowed
    follow_up = _FakeLLMClient.requests[1]
    tool_messages = [m for m in follow_up.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert '"code": "INVALID_ARGUMENT"' in tool_messages[0].content


def test_standalone_multi_step_loop_prints_tool_steps_and_answer(
    tmp_path, monkeypatch, capsys, trace_sink
):
    # Deterministic test context: the pinned run lives under tmp_path,
    # so the CLI's real default dispatcher → real search tool resolves
    # it through the normal production path (default artifacts root →
    # discover run → repository dir). The loop then returns to the
    # model, which produces the final answer (M8.4).
    _controlled_pinned_context(tmp_path, monkeypatch)
    _FakeLLMClient.responses = [
        LLMResponse(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="search_repository",
                        arguments={"query": "onboarding workflow"},
                    )
                ],
            ),
            finish_reason="tool_calls",
        ),
        LLMResponse(
            message=Message(
                role="assistant",
                content="The onboarding workflow has four stages.",
            ),
            finish_reason="stop",
        ),
    ]
    exit_code = agent_main.main(
        ["--question", "What components participate in the onboarding workflow?"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Question:" in captured.out
    assert "What components participate in the onboarding workflow?" in captured.out
    assert "Tool steps:" in captured.out
    assert "1. search_repository -> success=True" in captured.out
    assert "Answer:" in captured.out
    assert "The onboarding workflow has four stages." in captured.out
    # superseded one-shot output must be gone
    assert "Tool execution (one shot)" not in captured.out
    assert "NOT EXECUTED" not in captured.out
    # genuine multi-step conversation: the ToolResult observation was
    # fed back to the model before the final answer
    assert len(_FakeLLMClient.requests) == 2
    follow_up = _FakeLLMClient.requests[1]
    assert [message.role for message in follow_up.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    tool_messages = [m for m in follow_up.messages if m.role == "tool"]
    assert '"query": "onboarding workflow"' in tool_messages[0].content
    assert '"success": true' in tool_messages[0].content
    assert '"snapshot_id": "run_fixture_alpha"' in tool_messages[0].content
    # automatic execution trace: one run, one tool step, final answer
    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert trace.summary.total_iterations == 2
    assert trace.summary.total_tool_calls == 1
    assert trace.summary.termination_reason == "FINAL_ANSWER"
    assert trace.repository["snapshot_id"] == "run_fixture_alpha"


def test_standalone_rejects_empty_question(capsys):
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="ok")
    )
    exit_code = agent_main.main(["--question", "   "])
    captured = capsys.readouterr()

    assert exit_code == 6
    assert isinstance(captured.err, str) and "AgentError" in captured.err
    assert _FakeLLMClient.last_request is None  # rejected before the LLM


# ---------------------------------------------------------------------------
# M8.5 bounded execution — --max-tool-calls
# ---------------------------------------------------------------------------


def test_cli_max_tool_calls_zero_blocks_execution(trace_sink, capsys):
    # Budget 0: the model's tool request is recognized but never
    # executed — the run terminates with EXECUTION_LIMIT_REACHED (6).
    _FakeLLMClient.response = LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="search_repository", arguments={"query": "x"})
            ],
        ),
        finish_reason="tool_calls",
    )
    exit_code = agent_main.main(["--max-tool-calls", "0"])
    captured = capsys.readouterr()

    assert exit_code == 6
    assert "AgentError" in captured.err
    assert "execution budget exhausted" in captured.err
    assert "Tool steps:" not in captured.out  # nothing executed
    assert len(trace_sink.traces) == 1
    summary = trace_sink.traces[0].summary
    assert summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    assert summary.max_tool_calls == 0
    assert summary.tool_calls_executed == 0
    assert summary.tool_calls_blocked == 1


def test_cli_max_tool_calls_allows_the_budgeted_execution(
    tmp_path, monkeypatch, capsys, trace_sink
):
    _controlled_pinned_context(tmp_path, monkeypatch)
    _FakeLLMClient.responses = [
        LLMResponse(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="search_repository",
                        arguments={"query": "onboarding workflow"},
                    )
                ],
            ),
            finish_reason="tool_calls",
        ),
        LLMResponse(
            message=Message(role="assistant", content="Four stages."),
            finish_reason="stop",
        ),
    ]
    exit_code = agent_main.main(["--max-tool-calls", "1"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "1. search_repository -> success=True" in captured.out
    assert "Answer:" in captured.out
    summary = trace_sink.traces[0].summary
    assert summary.max_tool_calls == 1
    assert summary.tool_calls_executed == 1
    assert summary.termination_reason == "FINAL_ANSWER"


def test_cli_max_tool_calls_rejects_a_negative_value(capsys):
    with pytest.raises(SystemExit) as excinfo:
        agent_main.main(["--max-tool-calls", "-1"])
    captured = capsys.readouterr()
    assert excinfo.value.code == 2  # argparse usage error
    assert "must be >= 0" in captured.err


# ---------------------------------------------------------------------------
# Execution trace (observability MVP) — composition-root behavior
# ---------------------------------------------------------------------------


def test_cli_failure_run_still_records_a_trace(trace_sink, capsys):
    # Every run produces a trace — including structured failures.
    _FakeLLMClient.error = LLMError("OPENAI_API_KEY is not configured.")
    exit_code = agent_main.main([])
    captured = capsys.readouterr()

    assert exit_code == 6
    assert "LLMError" in captured.err  # same CLI contract as before
    assert len(trace_sink.traces) == 1
    trace = trace_sink.traces[0]
    assert trace.summary.termination_reason == "LLM_ERROR"
    assert trace.iterations[0].response_type == "ERROR"
    assert "OPENAI_API_KEY" in trace.iterations[0].error


def test_cli_writes_json_and_markdown_trace_artifacts(
    tmp_path, monkeypatch, capsys
):
    from repo_intelligence.agent import JsonFileTraceSink

    # Real file sink under tmp_path (the trace root is redirected
    # for this test — production defaults to runtime/agent_runs/).
    sink = JsonFileTraceSink(tmp_path)
    monkeypatch.setattr(agent_main, "_build_trace_sink", lambda: sink)
    _FakeLLMClient.response = LLMResponse(
        message=Message(role="assistant", content="ok"),
        finish_reason="stop",
    )
    exit_code = agent_main.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    json_paths = list(tmp_path.glob("*/trace.json"))
    md_paths = list(tmp_path.glob("*/trace.md"))
    assert len(json_paths) == 1
    assert len(md_paths) == 1

    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert payload["user_question"] == agent_main.DEFAULT_QUESTION
    assert payload["summary"]["termination_reason"] == "FINAL_ANSWER"
    assert payload["summary"]["total_iterations"] == 1
    assert payload["summary"]["total_tool_calls"] == 0
    assert payload["iterations"][0]["decision"]["type"] == "FINAL_ANSWER"

    # the CLI points the developer at the trace
    assert "Trace: " in captured.out
    assert str(json_paths[0]) in captured.out

    markdown = md_paths[0].read_text(encoding="utf-8")
    assert markdown.startswith("# Agent Execution Trace")
    assert agent_main.DEFAULT_QUESTION in markdown
    assert "Termination: `FINAL_ANSWER`" in markdown
