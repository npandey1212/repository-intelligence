"""M8.6 bounded ReAct investigation tests — adaptive actions.

A LATER tool action must be selected BECAUSE OF an earlier
observation: the scripted offline model below returns each response
ONLY when the expected prior ToolResult is present in the request
(otherwise it raises AssertionError and the test fails), and derives
every action's arguments from that observed content. The mechanics
under test are the existing M8.4/M8.5 ones — the real ``Agent`` loop,
the real application dispatcher, the four read-only tools over a
pinned fixture of REAL project files, and the single
``AgentExecutionConfig.max_tool_calls`` tool-call limit (no second
budget, no new orchestration layer).

Required M8.6 cases:

    1. adaptive investigation — action 2 derives from observation 1
       (same script, different observation → different action)
    2. observation propagation — ToolResults appear in later requests
    3. evidence steps — at least two, preferably three, actions before
       the final answer
    4. evidence-backed final answer — cites the observations
    5. bounded investigation — the single tool-call limit truncates an
       investigation that would continue
    6. tool failure is an observation — the model adapts, no
       application retry
    7. the execution trace records investigation steps but no hidden
       reasoning (``thought`` / ``chain_of_thought`` / … fields)

The loop itself lives in ``test_agent_loop.py``; budget boundaries in
``test_agent_bounded.py``; trace serialization in
``test_agent_trace.py``. ``agentrun_react_test.py`` is the standalone
offline demo of this milestone; ``agentrun_test.py`` is retained as
the separate M8.4/M8.5 learning/debugging harness.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

import repo_intelligence.artifacts as artifacts
from conftest import enable_search
from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
)
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS
from repo_intelligence.llm import LLMRequest, LLMResponse, Message, ToolCall
from repo_intelligence.tool_dispatcher import default_tool_dispatcher

OBJECTIVE = (
    "How does this repository bound the Agent's investigation loop "
    "(the tool-call limit)?"
)

RUN_ID = "run_fixture_m86"
FIXTURE_COMMIT = "f" * 40

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COPIED_PATHS = ("README.md", "src/repo_intelligence/agent")

READ_ONLY_TOOLS = (
    "get_architecture_context",
    "query_engineering_graph",
    "read_file",
    "search_repository",
)

# The order this investigation actually executes the tools (the
# registry itself stays alphabetical).
INVESTIGATION_TOOLS = (
    "get_architecture_context",
    "search_repository",
    "read_file",
)

# Field names that must never appear (the user's explicit M8.6 list on
# top of the trace module's own ban list).
EXTRA_BANNED_FIELDS = ("reasoning_trace",)

# Two mechanism descriptions: SAME scripted model, different
# observation → different derived action (adaptation, not a
# pre-programmed sequence).
MECHANISM_A = """## Mechanism

- The bounded Agent loop lives in `src/repo_intelligence/agent/agent.py`.
- The single tool-call limit is `max_tool_calls`
  (`AgentExecutionConfig.max_tool_calls`, default 8).
"""

MECHANISM_B = """## Mechanism

- The system instruction constant lives in `src/repo_intelligence/agent/agent.py`.
- The exported name is `SYSTEM_PROMPT`.
"""


# ---------------------------------------------------------------------------
# Fixture: pinned run whose repository/ = real files copied from this project
# ---------------------------------------------------------------------------


def _context_text(mechanism_section: str) -> str:
    return (
        "# Architecture Context\n"
        "\n"
        "- Contract: `architecture_context.v1`\n"
        "\n"
        "## Repository\n"
        "\n"
        "- Name: `repository-intelligence`\n"
        "- Pinned commit: {}\n"
        "- Snapshot: {}\n"
        "\n"
        "{}"
    ).format(FIXTURE_COMMIT, RUN_ID, mechanism_section)


def _pinned_project_context(base: Path, monkeypatch, mechanism_section: str):
    """Build ``<base>/artifacts/<RUN_ID>`` with a ``repository/`` of
    REAL project files and point the default artifacts root at it.
    Only the root is redirected — production run discovery and tool
    resolution are unchanged."""
    root = base / "artifacts"
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
        _context_text(mechanism_section), encoding="utf-8"
    )
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {"name": "repository-intelligence"},
                "snapshot": {"ref": "main", "commit_sha": FIXTURE_COMMIT},
            }
        ),
        encoding="utf-8",
    )
    repo = run_dir / "repository"
    for relative in COPIED_PATHS:
        source = PROJECT_ROOT / relative
        destination = repo / relative
        if source.is_dir():
            shutil.copytree(
                str(source),
                str(destination),
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
    # M18: isolated store + deterministic index so the real
    # search_repository tool answers over this pinned checkout.
    enable_search(run_dir / "repository", monkeypatch, base)
    monkeypatch.setattr(artifacts, "default_artifacts_root", lambda: root)
    return run_dir


# ---------------------------------------------------------------------------
# Observation-derived arguments (values come from the observation)
# ---------------------------------------------------------------------------


def _mechanism_span(context: str, pattern: str, label: str) -> str:
    if "## Mechanism" not in context:
        raise AssertionError(
            "observation has no ## Mechanism section (cannot derive {})".format(
                label
            )
        )
    section = context.split("## Mechanism", 1)[1]
    matcher = re.compile("`(" + pattern + ")`")
    for span in matcher.findall(section):
        if re.fullmatch(pattern, span):
            return span
    raise AssertionError(
        "no {} code span in the observation's ## Mechanism section".format(label)
    )


def _mechanism_symbol(context: str) -> str:
    return _mechanism_span(context, r"[A-Za-z_][A-Za-z0-9_]*", "symbol")


def _source_file(context: str) -> str:
    return _mechanism_span(context, r"[^`\s]+\.py", "source file")


def _first_content_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "(empty excerpt)"


# ---------------------------------------------------------------------------
# Scripted offline model: conditional on observations, derived arguments
# ---------------------------------------------------------------------------


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _observations(request: LLMRequest):
    observed = []
    for message in request.messages:
        if message.role != "tool":
            continue
        try:
            observed.append(json.loads(message.content))
        except (TypeError, ValueError) as exc:
            raise AssertionError("observation is not JSON: {}".format(exc))
    return observed


def _expect_observations(*tool_names):
    """Expect exactly these prior observations, in order, each a
    successful ToolResult — otherwise this scripted model raises
    AssertionError (the run FAILS: the application must feed every
    ToolResult back)."""

    def expect(request):
        observed = _observations(request)
        if len(observed) != len(tool_names):
            raise AssertionError(
                "expected prior observations {} ({}); request carries {} — "
                "the Agent must feed ToolResults back".format(
                    list(tool_names), len(tool_names), len(observed)
                )
            )
        for expected, observation in zip(tool_names, observed):
            if observation.get("tool") != expected:
                raise AssertionError(
                    "expected observation from {!r}, got {!r}".format(
                        expected, observation.get("tool")
                    )
                )
            if observation.get("success") is not True:
                raise AssertionError(
                    "expected a successful observation from {!r}".format(expected)
                )
        return observed

    return expect


def _respond_orient(observed):
    # The default gac answer is now a compact heading index; the
    # scripted model narrows to the section its derivation needs —
    # exactly the documented index → section retrieval loop.
    return _tool_response(
        ToolCall(
            id="c_orient",
            name="get_architecture_context",
            arguments={"section": "Mechanism"},
        )
    )


def _context_observation(observed):
    for item in observed:
        if item.get("tool") == "get_architecture_context" and item.get(
            "success"
        ) is True:
            return item
    raise AssertionError("no successful architecture-context observation")


def _respond_search(observed):
    context = _context_observation(observed)["data"]["context"]
    token = _mechanism_symbol(context)
    return _tool_response(
        ToolCall(
            id="c_search",
            name="search_repository",
            arguments={"query": token},
        )
    )


def _respond_read(observed):
    results = observed[1]["data"]["results"]
    assert results, "observation 2 carries no search results"
    first = results[0]
    return _tool_response(
        ToolCall(
            id="c_read",
            name="read_file",
            arguments={
                "path": first["path"],
                "line_start": first["line_start"],
                "line_end": first["line_end"],
            },
        )
    )


def _respond_answer(observed):
    context = observed[0]["data"]["context"]
    symbol = _mechanism_symbol(context)
    source_file = _source_file(context)
    search = observed[1]["data"]
    first = search["results"][0]
    read = observed[2]["data"]
    snippet = _first_content_line(read.get("content") or "")
    answer = (
        "Evidence-backed answer (cited from the investigation observations):\n"
        "- Observation 1 (architecture context): mechanism `{symbol}` in "
        "`{source_file}`.\n"
        "- Observation 2 (search): {query!r} found {total} match(es); first "
        "candidate {hit_path}:{hit_start}-{hit_end}.\n"
        "- Observation 3 (read): {path}:{line_start}-{line_end} → {snippet!r}.\n"
        "Conclusion: {source_file} documents the tool-call limit `{symbol}`."
    ).format(
        symbol=symbol,
        source_file=source_file,
        query=search["query"],
        total=search["total_matches"],
        hit_path=first["path"],
        hit_start=first["line_start"],
        hit_end=first["line_end"],
        path=read["path"],
        line_start=read["line_start"],
        line_end=read["line_end"],
        snippet=snippet,
    )
    return _text_response(answer)


def _investigation_steps():
    """orient → search (query from observation 1) → read (path/lines
    from observation 2) → answer (cites observations 1-3)."""
    return [
        ("orient", _expect_observations(), _respond_orient),
        ("search", _expect_observations("get_architecture_context"), _respond_search),
        (
            "read",
            _expect_observations("get_architecture_context", "search_repository"),
            _respond_read,
        ),
        (
            "answer",
            _expect_observations(
                "get_architecture_context", "search_repository", "read_file"
            ),
            _respond_answer,
        ),
    ]


class AdaptiveLLM:
    """Scripted offline model: serves one step per call, each
    conditional on the prior observations its request carries."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.requests = []

    def complete(self, request):
        index = len(self.requests)
        if index >= len(self._steps):
            raise AssertionError(
                "scripted model invoked {} time(s); {} step(s) scripted".format(
                    index + 1, len(self._steps)
                )
            )
        _, expect, respond = self._steps[index]
        observed = expect(request)  # raises if an observation is missing
        self.requests.append(request)
        response = respond(observed)
        return response


class RecordingDispatcher:
    """Wraps the real dispatcher: records calls and executed results."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []
        self.results = []

    @property
    def tool_names(self):
        return tuple(self._inner.tool_names)

    def dispatch(self, tool_call):
        result = self._inner.dispatch(tool_call)
        self.calls.append(tool_call)
        self.results.append(result)
        return result


def _run_investigation(tmp_path, monkeypatch, mechanism=MECHANISM_A,
                       max_tool_calls=8):
    _pinned_project_context(tmp_path, monkeypatch, mechanism)
    fake = AdaptiveLLM(_investigation_steps())
    dispatcher = RecordingDispatcher(default_tool_dispatcher())
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=max_tool_calls),
    )
    answer = agent.run(OBJECTIVE)
    return agent, fake, dispatcher, answer


def _walk_keys(value):
    if isinstance(value, dict):
        keys = []
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
        return keys
    if isinstance(value, list):
        keys = []
        for item in value:
            keys.extend(_walk_keys(item))
        return keys
    return []


def _banned_fields():
    return tuple(BANNED_REASONING_FIELDS) + EXTRA_BANNED_FIELDS


# ---------------------------------------------------------------------------
# 1 — adaptive investigation: action 2 derives from observation 1
#     (same script, different observation → different action)
# ---------------------------------------------------------------------------


def test_second_action_derives_from_first_observation(tmp_path, monkeypatch):
    # Run A: the architecture context names `max_tool_calls`.
    _, fake_a, dispatcher_a, _ = _run_investigation(
        tmp_path / "run_a", monkeypatch, MECHANISM_A
    )
    # Run B: identical script, but a different observation.
    _, fake_b, dispatcher_b, _ = _run_investigation(
        tmp_path / "run_b", monkeypatch, MECHANISM_B
    )

    observed_a = _observations(fake_a.requests[1])[0]
    observed_b = _observations(fake_b.requests[1])[0]
    symbol_a = _mechanism_symbol(observed_a["data"]["context"])
    symbol_b = _mechanism_symbol(observed_b["data"]["context"])

    query_a = dispatcher_a.calls[1].arguments["query"]
    query_b = dispatcher_b.calls[1].arguments["query"]

    # The action was taken FROM the observation, not from a script
    # constant: it equals the symbol extracted from what the model saw…
    assert query_a == symbol_a == "max_tool_calls"
    assert query_b == symbol_b == "SYSTEM_PROMPT"
    # …and it CHANGES when the observation changes.
    assert query_a != query_b
    assert dispatcher_a.calls[1].name == "search_repository"
    assert dispatcher_b.calls[1].name == "search_repository"


# ---------------------------------------------------------------------------
# 2 — observation propagation: ToolResult verbatim in later requests
# ---------------------------------------------------------------------------


def test_observations_propagate_into_following_requests(tmp_path, monkeypatch):
    _, fake, dispatcher, _ = _run_investigation(tmp_path, monkeypatch)

    assert len(fake.requests) == 4
    # request 2 carries observation 1 only; request 3 carries 1+2;
    # request 4 (the answer turn) carries all three, in order.
    assert [len(_observations(request)) for request in fake.requests] == [0, 1, 2, 3]
    for request, expected_count in zip(fake.requests[1:], (1, 2, 3)):
        names = [item.get("tool") for item in _observations(request)]
        assert names == list(INVESTIGATION_TOOLS[:expected_count])

    # the carried content is the executed ToolResult, verbatim
    assert _observations(fake.requests[1])[0] == dispatcher.results[0].to_dict()
    assert _observations(fake.requests[2])[1] == dispatcher.results[1].to_dict()
    assert _observations(fake.requests[3])[2] == dispatcher.results[2].to_dict()


# ---------------------------------------------------------------------------
# 3 — evidence steps: three actions before the final answer
# ---------------------------------------------------------------------------


def test_three_evidence_actions_before_final_answer(tmp_path, monkeypatch):
    agent, fake, dispatcher, answer = _run_investigation(tmp_path, monkeypatch)

    assert isinstance(answer, str)
    # three real tool actions, then the answer (4 LLM calls total)
    assert [call.name for call in dispatcher.calls] == list(INVESTIGATION_TOOLS)
    assert len(dispatcher.calls) == 3
    assert len(fake.requests) == 4
    # read-only registry, unchanged
    assert dispatcher.tool_names == READ_ONLY_TOOLS
    # the trace saw one action/observation iteration per step
    trace = agent.last_trace
    assert trace is not None
    assert [iteration.response_type for iteration in trace.iterations] == [
        "TOOL_CALL",
        "TOOL_CALL",
        "TOOL_CALL",
        "FINAL_ANSWER",
    ]


# ---------------------------------------------------------------------------
# 4 — the final answer cites the gathered evidence
# ---------------------------------------------------------------------------


def test_final_answer_cites_the_gathered_evidence(tmp_path, monkeypatch):
    agent, fake, dispatcher, answer = _run_investigation(tmp_path, monkeypatch)

    observed = _observations(fake.requests[3])
    assert len(observed) == 3
    context = observed[0]["data"]["context"]
    search = observed[1]["data"]
    read = observed[2]["data"]
    snippet = _first_content_line(read.get("content") or "")

    # every citation comes from an observation, not from the script
    assert _mechanism_symbol(context) in answer
    assert str(search["total_matches"]) in answer
    assert read["path"] in answer
    assert str(read["line_start"]) in answer
    assert snippet in answer
    # the answer and the recorded trace agree
    assert agent.last_trace is not None
    assert agent.last_trace.final_answer == answer
    assert agent.last_trace.termination_reason == "FINAL_ANSWER"


# ---------------------------------------------------------------------------
# 5 — bounded investigation: the single tool-call limit truncates it
# ---------------------------------------------------------------------------


def test_investigation_is_bounded_by_max_tool_calls(tmp_path, monkeypatch):
    _pinned_project_context(tmp_path, monkeypatch, MECHANISM_A)
    fake = AdaptiveLLM(_investigation_steps())
    dispatcher = RecordingDispatcher(default_tool_dispatcher())
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=2),
    )

    # orient (1) and search (2) run; the derived read is the 3rd tool
    # call — requested AFTER the limit, therefore never executed.
    with pytest.raises(AgentError) as excinfo:
        agent.run(OBJECTIVE)

    assert excinfo.value.termination_reason == "EXECUTION_LIMIT_REACHED"
    # only two executions ever happened; no third dispatch, no
    # fabricated ToolResult for the blocked read
    assert [call.name for call in dispatcher.calls] == [
        "get_architecture_context",
        "search_repository",
    ]
    assert len(fake.requests) == 3  # model got one turn past the limit

    trace = agent.last_trace
    summary = trace.summary
    assert summary.max_tool_calls == 2
    assert summary.tool_calls_executed == 2
    assert summary.tool_calls_blocked == 1
    assert summary.termination_reason == "EXECUTION_LIMIT_REACHED"
    blocked = trace.iterations[-1].blocked_tool_calls
    assert len(blocked) == 1
    assert blocked[0]["tool"] == "read_file"
    assert blocked[0]["reason"] == "EXECUTION_LIMIT_REACHED"
    # the blocked call produced no observation anywhere in the history
    tool_names_seen = [
        item.get("tool")
        for request in fake.requests
        for item in _observations(request)
    ]
    assert "read_file" not in tool_names_seen


# ---------------------------------------------------------------------------
# 6 — a tool failure is an observation; the model adapts, no app retry
# ---------------------------------------------------------------------------


def _respond_missing_read(observed):
    return _tool_response(
        ToolCall(
            id="c_miss",
            name="read_file",
            arguments={"path": "src/does_not_exist.py"},
        )
    )


def _expect_read_failure(request):
    observed = _observations(request)
    if len(observed) != 1:
        raise AssertionError(
            "expected the failed read observation; request carries {} "
            "observation(s)".format(len(observed))
        )
    first = observed[0]
    if first.get("success") is not False:
        raise AssertionError("expected a failed ToolResult observation")
    if (first.get("error") or {}).get("code") != "FILE_NOT_FOUND":
        raise AssertionError(
            "expected FILE_NOT_FOUND, got {!r}".format(
                (first.get("error") or {}).get("code")
            )
        )
    return observed


def _expect_failure_then_context(request):
    observed = _observations(request)
    if len(observed) != 2:
        raise AssertionError(
            "expected failure + architecture context; request carries {} "
            "observation(s)".format(len(observed))
        )
    first, second = observed
    if first.get("success") is not False:
        raise AssertionError("the failure observation did not propagate")
    if second.get("tool") != "get_architecture_context" or second.get(
        "success"
    ) is not True:
        raise AssertionError("expected a successful architecture context")
    return observed


def _expect_failure_then_evidence(request):
    observed = _observations(request)
    if len(observed) != 3:
        raise AssertionError(
            "expected failure + context + search; request carries {} "
            "observation(s)".format(len(observed))
        )
    if observed[0].get("success") is not False:
        raise AssertionError("the failure observation did not propagate")
    for observation in observed[1:]:
        if observation.get("success") is not True:
            raise AssertionError("expected a successful observation")
    return observed


def _respond_answer_after_search(observed):
    context = observed[1]["data"]["context"]
    symbol = _mechanism_symbol(context)
    search = observed[2]["data"]
    first = search["results"][0]
    answer = (
        "Recovered from the failed read:\n"
        "- read_file failed with FILE_NOT_FOUND (observed, not retried).\n"
        "- The architecture context names `{symbol}`.\n"
        "- search_repository({symbol!r}) found {total} match(es); first "
        "{path}:{line_start}-{line_end}."
    ).format(
        symbol=symbol,
        total=search["total_matches"],
        path=first["path"],
        line_start=first["line_start"],
        line_end=first["line_end"],
    )
    return _text_response(answer)


def test_tool_failure_is_an_observation_and_the_next_action_adapts(
    tmp_path, monkeypatch
):
    _pinned_project_context(tmp_path, monkeypatch, MECHANISM_A)
    steps = [
        ("miss", _expect_observations(), _respond_missing_read),
        ("reorient", _expect_read_failure, _respond_orient),
        ("search", _expect_failure_then_context, _respond_search),
        ("answer", _expect_failure_then_evidence, _respond_answer_after_search),
    ]
    fake = AdaptiveLLM(steps)
    dispatcher = RecordingDispatcher(default_tool_dispatcher())
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=8),
    )

    answer = agent.run(OBJECTIVE)

    # the failure became an observation, then a DIFFERENT tool was
    # chosen — the application never retried the failing call itself
    assert [call.name for call in dispatcher.calls] == [
        "read_file",
        "get_architecture_context",
        "search_repository",
    ]
    assert [call.name for call in dispatcher.calls].count("read_file") == 1

    failure_observation = _observations(fake.requests[1])[0]
    assert failure_observation["success"] is False
    assert failure_observation["error"]["code"] == "FILE_NOT_FOUND"

    assert isinstance(answer, str)
    assert "FILE_NOT_FOUND" in answer
    assert agent.last_trace is not None
    assert agent.last_trace.termination_reason == "FINAL_ANSWER"
    assert agent.last_trace.summary.tool_calls_executed == 3


# ---------------------------------------------------------------------------
# 7 — the trace records investigation steps, never hidden reasoning
# ---------------------------------------------------------------------------


def test_trace_records_investigation_steps_without_hidden_reasoning(
    tmp_path, monkeypatch
):
    agent, fake, dispatcher, answer = _run_investigation(tmp_path, monkeypatch)

    trace = agent.last_trace
    assert trace is not None
    payload = trace.to_dict()
    serialized = json.dumps(payload)

    banned = _banned_fields()

    # keys only: observed repository content may legitimately mention
    # reasoning in prose — no FIELD may ever capture it
    for key in _walk_keys(payload):
        lowered = key.lower()
        for needle in banned:
            assert needle not in lowered, key

    # the model-facing requests carry no reasoning fields either
    for request in fake.requests:
        for message in request.messages:
            message_payload = {
                "role": message.role,
                "content": message.content,
                "tool_call_id": message.tool_call_id,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in message.tool_calls
                ],
            }
            for key in _walk_keys(message_payload):
                lowered = key.lower()
                for needle in banned:
                    assert needle not in lowered, key

    # but the investigation itself is fully recorded
    assert len(trace.iterations) == 4
    for iteration in trace.iterations:
        assert iteration.response_type in ("TOOL_CALL", "FINAL_ANSWER")
        assert iteration.decision.get("type") in ("TOOL_CALL", "FINAL_ANSWER")
        assert iteration.reason_summary
    recorded_actions = [
        call["name"]
        for iteration in trace.iterations
        for call in iteration.tool_calls
    ]
    assert recorded_actions == list(INVESTIGATION_TOOLS)
    assert trace.final_answer == answer
    assert trace.termination_reason == "FINAL_ANSWER"
    assert trace.summary.tool_calls_executed == 3
    assert trace.summary.tool_calls_blocked == 0
    # serialized once for the sink path: still no banned field names
    for needle in banned:
        assert '"{}"'.format(needle) not in serialized
