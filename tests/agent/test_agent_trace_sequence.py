"""M8.7.1 — observable agent interaction trace clarity (Tests 1–7).

One ``trace.md``/``trace.json`` must read top-to-bottom as the
ordered interaction stages of the (unchanged) loop:

    LLM REQUEST → LLM RESPONSE → ACTION → TOOL EXECUTION →
    TOOL RESULT → STATE UPDATE → CONTINUATION/TERMINATION →
    FINAL ANSWER → M8.8 EVIDENCE VALIDATION

Observability only — the loop, the budget, the dispatcher contract
and the M8.8 validation are untouched; these tests pin the stage
order, the continuation record, per-execution ``duration_ms``, and
that a budget-blocked call is never rendered as executed.

Per-file test doubles (no conftest in this repo).
"""

from __future__ import annotations

import json

from repo_intelligence.agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
    render_markdown,
)
from repo_intelligence.llm import LLMResponse, Message, ToolCall
from repo_intelligence.tools import ToolError, ToolResult

QUESTION = "How does the data onboarding workflow work?"
ANSWER = "The workflow has four deterministic stages."

PATH_READ = ".claude/agents/dag-agent.md"
STRUCTURED_FINAL = (
    '{"answer": "%s", "claims": [{"statement": '
    '"The repository defines a DAG Agent.", "evidence_refs": '
    '[{"path": "%s", "line_start": 1, "line_end": 20}]}]}'
) % (ANSWER, PATH_READ)

# The ordered stages an iteration must render, in order.
TOOL_ITERATION_STAGES = (
    "### LLM Request",
    "### LLM Response",
    "### Actions",
    "### Tool Executions",
    "### Tool Results",
    "### State Update",
    "### Continuation",
)


# -- per-file test doubles (no conftest in this repo) ----------------------

class FakeLLMClient:
    """Queued responses (or queued exceptions); records requests."""

    def __init__(self, *responses, model=None) -> None:
        self._queue = list(responses)
        self.requests = []
        self.model = model

    def complete(self, request):
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("LLM invoked more times than scripted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeDispatcher:
    """Records dispatches; canned results first, else generic success."""

    def __init__(self, *results) -> None:
        self._queue = list(results)
        self.calls = []

    def dispatch(self, tool_call):
        self.calls.append(tool_call)
        if self._queue:
            return self._queue.pop(0)
        return ToolResult(success=True, tool=tool_call.name, data={})


def _text_response(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_calls_response(*calls):
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _call(name, arguments=None, call_id="call_1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _read_result(path=PATH_READ, start=1, end=20):
    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": path,
            "line_start": start,
            "line_end": end,
            "content": "# {} (excerpt)".format(path),
        },
        source={"repository": "sample-repo", "commit_sha": "a" * 40},
    )


def _search_result(query="x", success=True):
    if success:
        return ToolResult(
            success=True, tool="search_repository", data={"query": query}
        )
    return ToolResult(
        success=False,
        tool="search_repository",
        error=ToolError(code="QUERY_FAILED", message="backend unavailable"),
    )


def _run(responses, results=None, config=None, question=QUESTION):
    """One scripted run through the real Agent (run_result)."""
    fake = FakeLLMClient(*responses)
    dispatcher = FakeDispatcher(*(results or ()))
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=config or AgentExecutionConfig(),
    )
    result = agent.run_result(question)
    return result, agent, dispatcher


def _assert_stages_in_order(markdown, stages, start=0):
    """Every stage appears, at increasing offsets from *start*."""
    previous = start
    for stage in stages:
        at = markdown.index(stage, previous)
        assert at > previous, "stage out of order: {}".format(stage)
        previous = at
    return previous


def _between(markdown, start, end=None):
    """The slice of *markdown* from *start* up to *end* (or the end)."""
    begin = markdown.index(start)
    if end is None:
        return markdown[begin:]
    finish = markdown.index(end, begin + 1)
    return markdown[begin:finish]


# ---------------------------------------------------------------------------
# Test 1 — a single tool iteration renders every stage, in order
# ---------------------------------------------------------------------------


def test_single_tool_iteration_renders_every_stage_in_order():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("search_repository", {"query": "x"})),
            _text_response(),
        ],
        results=[_search_result("x")],
    )
    assert result.answer == ANSWER
    trace = agent.last_trace
    markdown = render_markdown(trace)

    first = _between(markdown, "## Iteration 1", "## Iteration 2")
    last = _between(markdown, "## Iteration 2", "## Run Summary")
    _assert_stages_in_order(first, TOOL_ITERATION_STAGES)
    # final iteration: request -> response -> termination -> answer
    _assert_stages_in_order(
        last,
        (
            "### LLM Request",
            "### LLM Response",
            "### Termination",
            "### Answer",
        ),
    )
    # the state update names the next request; the continuation
    # records the observable decision to make it — with learner prose
    assert "Next LLM request: iteration 2 (4 messages)" in first
    assert "Decision: CONTINUE" in first
    assert "Next iteration: 2" in first
    assert (
        "Tool result will be returned to the LLM.\n"
        "Continue investigation."
    ) in first
    # the response stage names the tool-call response (with the id the
    # action/result stages reuse) and the tool-less final response's
    # exact text; the terminal control flow is labelled Termination
    assert "Type: `TOOL_CALL`" in first
    assert "Tool calls requested: `call_1`" in first
    assert (
        "Text: <none \u2014 response requested tool execution>" in first
    )
    assert "Decision: FINAL_ANSWER" in last
    assert "Termination: `FINAL_ANSWER`" in last
    assert "Final answer received from LLM." in last
    assert "Text:\n{}".format(ANSWER) in last
    assert "### Continuation" not in last

    payload = trace.to_dict()
    first_json, last_json = payload["iterations"]
    assert first_json["continuation"] == {
        "decision": "CONTINUE",
        "reason": "tool result added to conversation state",
        "next_iteration": 2,
    }
    assert last_json["continuation"] == {
        "decision": "FINAL_ANSWER",
        "reason": "LLM returned final answer",
        "termination_reason": "FINAL_ANSWER",
    }
    # the executed result carries its observed duration
    duration = first_json["tool_results"][0]["duration_ms"]
    assert isinstance(duration, int) and duration >= 0


# ---------------------------------------------------------------------------
# Test 2 — a final-answer-only run never fabricates an execution
# ---------------------------------------------------------------------------


def test_final_answer_run_has_no_fabricated_execution_stage():
    result, agent, _ = _run(responses=[_text_response()])
    assert result.answer == ANSWER
    trace = agent.last_trace
    markdown = render_markdown(trace)

    for stage in (
        "### Actions",
        "### Tool Executions",
        "### Tool Results",
        "### State Update",
    ):
        assert stage not in markdown
    assert "### Answer" in markdown
    assert ANSWER in markdown
    assert "Decision: FINAL_ANSWER" in markdown
    # a tool-less run still reads: response with its text, then an
    # explicit termination — never a missing/empty response stage
    assert "Type: `FINAL_ANSWER`" in markdown
    assert "Text:\n{}".format(ANSWER) in markdown
    assert "### Termination" in markdown
    assert "Final answer received from LLM." in markdown
    assert "### Continuation" not in markdown

    payload = trace.to_dict()
    assert payload["iterations"][0]["tool_calls"] == []
    assert payload["iterations"][0]["tool_results"] == []
    assert payload["summary"]["total_tool_calls"] == 0
    assert payload["summary"]["termination_reason"] == "FINAL_ANSWER"


# ---------------------------------------------------------------------------
# Test 3 — multi-tool responses correlate across every stage
# ---------------------------------------------------------------------------


def test_multi_tool_calls_correlate_across_all_stages():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(
                _call("search_repository", {"query": "a"}, "c1"),
                _call("read_file", {"path": PATH_READ}, "c2"),
                _call("search_repository", {"query": "b"}, "c3"),
            ),
            _text_response(),
        ],
        results=[
            _search_result("a"),
            _read_result(),
            _search_result("b", success=False),
        ],
    )
    trace = agent.last_trace
    markdown = render_markdown(trace)
    first = _between(markdown, "## Iteration 1", "## Iteration 2")

    # the response requested exactly these three calls, verbatim
    requested = trace.iterations[0].llm_response["tool_calls"]
    assert [call["id"] for call in requested] == ["c1", "c2", "c3"]
    # ... and the response stage renders those exact ids, so the same
    # tool_call_id visibly connects RESPONSE -> ACTION -> TOOL RESULT
    assert "Tool calls requested: `c1`, `c2`, `c3`" in first
    assert "(`c1`)" in first and "(`c2`)" in first and "(`c3`)" in first

    # ... executions, results and observations all pair by id ...
    results = trace.iterations[0].tool_results
    assert [entry["tool_call_id"] for entry in results] == ["c1", "c2", "c3"]
    observations = trace.iterations[0].observations
    assert [entry["tool_call_id"] for entry in observations] == [
        "c1",
        "c2",
        "c3",
    ]
    # ... and each stage renders every id with its own outcome
    assert "* `c1` (`search_repository`) \u2192 SUCCESS" in first
    assert "* `c2` (`read_file`) \u2192 SUCCESS" in first
    assert "* `c3` (`search_repository`) \u2192 FAILURE (QUERY_FAILED)" in first
    assert "**Result:** FAILURE (QUERY_FAILED)" in first
    _assert_stages_in_order(first, TOOL_ITERATION_STAGES)
    # Actions are listed before anything executed
    assert first.index("### Actions") < first.index("### Tool Executions")
    assert first.index("1. `search_repository`") < first.index(
        "### Tool Executions"
    )
    # every executed result carries its duration
    assert all(isinstance(entry["duration_ms"], int) for entry in results)


# ---------------------------------------------------------------------------
# Test 4 — a failed tool reads as executed-and-failed, not as absent
# ---------------------------------------------------------------------------


def test_failed_tool_reads_as_executed_then_failed_all_the_way_down():
    missing = "tool-registry/README.md"
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("read_file", {"path": missing}, "c1")),
            _text_response("Could not find the registry README."),
        ],
        results=[
            ToolResult(
                success=False,
                tool="read_file",
                error=ToolError(code="FILE_NOT_FOUND", message="no such file"),
            )
        ],
    )
    assert result.answer == "Could not find the registry README."
    trace = agent.last_trace
    markdown = render_markdown(trace)
    first = _between(markdown, "## Iteration 1", "## Iteration 2")

    # executed (a dispatch happened) AND failed — in every stage
    assert "* `c1` (`read_file`) \u2192 FAILURE (FILE_NOT_FOUND)" in first
    assert "**Result:** FAILURE (FILE_NOT_FOUND)" in first
    assert (
        "**Observation:** read_file failure (FILE_NOT_FOUND) "
        "returned to the Agent."
    ) in first
    _assert_stages_in_order(first, TOOL_ITERATION_STAGES)

    payload = trace.to_dict()
    entry = payload["iterations"][0]["tool_results"][0]
    assert entry["tool_call_id"] == "c1"
    assert entry["success"] is False
    assert entry["error"]["code"] == "FILE_NOT_FOUND"
    assert isinstance(entry["duration_ms"], int)
    # the loop continued after the failure (the observation went back)
    assert payload["iterations"][0]["continuation"]["decision"] == "CONTINUE"
    assert payload["iterations"][0]["continuation"]["next_iteration"] == 2


# ---------------------------------------------------------------------------
# Test 5 — a multi-step ReAct run reads as one ordered stage sequence
# ---------------------------------------------------------------------------


def test_multi_step_run_reads_as_one_ordered_stage_sequence():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("search_repository", {"query": "a"}, "c1")),
            _tool_calls_response(_call("read_file", {"path": "b"}, "c2")),
            _text_response(),
        ],
        results=[_search_result("a"), _read_result("b", 1, 2)],
    )
    assert result.answer == ANSWER
    trace = agent.last_trace
    payload = trace.to_dict()
    markdown = render_markdown(trace)

    # three iterations, each rendering its stages in order
    for number, next_number in ((1, 2), (2, 3)):
        segment = _between(
            markdown,
            "## Iteration {}".format(number),
            "## Iteration {}".format(next_number),
        )
        _assert_stages_in_order(segment, TOOL_ITERATION_STAGES)
    final_segment = _between(markdown, "## Iteration 3", "## Run Summary")
    _assert_stages_in_order(
        final_segment,
        (
            "### LLM Request",
            "### LLM Response",
            "### Termination",
            "### Answer",
        ),
    )
    # the whole story, stage by stage: two tool iterations that
    # continued, then a text response that terminated
    first_segment = _between(markdown, "## Iteration 1", "## Iteration 2")
    assert "Tool result will be returned to the LLM." in first_segment
    assert "Final answer received from LLM." in final_segment
    assert "Text:\n{}".format(ANSWER) in final_segment
    assert "### Continuation" not in final_segment
    # the final iteration requested nothing — no execution stages
    for stage in ("### Actions", "### Tool Executions", "### State Update"):
        assert stage not in final_segment

    # the continuation sequence tells the whole story, in JSON
    continuations = [item["continuation"] for item in payload["iterations"]]
    assert [entry["decision"] for entry in continuations] == [
        "CONTINUE",
        "CONTINUE",
        "FINAL_ANSWER",
    ]
    assert continuations[0]["next_iteration"] == 2
    assert continuations[1]["next_iteration"] == 3
    assert continuations[2]["termination_reason"] == "FINAL_ANSWER"
    # each State Update bridges to the request that actually followed
    second_segment = _between(markdown, "## Iteration 2", "## Iteration 3")
    assert "Next LLM request: iteration 2 (4 messages)" in first_segment
    assert "Next LLM request: iteration 3 (6 messages)" in second_segment
    assert "Next LLM request:" not in final_segment
    # still one loop, one question, one answer — no new engine
    assert payload["summary"]["total_iterations"] == 3
    assert payload["summary"]["termination_reason"] == "FINAL_ANSWER"


# ---------------------------------------------------------------------------
# Test 6 — a budget-blocked call is never rendered as executed
# ---------------------------------------------------------------------------


def test_budget_blocked_call_is_never_rendered_as_executed():
    fake = FakeLLMClient(
        _tool_calls_response(_call("search_repository", {"query": "x"}, "c9"))
    )
    dispatcher = FakeDispatcher()
    agent = Agent(
        llm_client=fake,
        tool_dispatcher=dispatcher,
        execution_config=AgentExecutionConfig(max_tool_calls=0),
    )
    try:
        agent.run(QUESTION)
    except AgentError as exc:
        assert exc.termination_reason == "EXECUTION_LIMIT_REACHED"
    trace = agent.last_trace
    markdown = render_markdown(trace)

    # requested and visible ...
    assert "### Actions" in markdown
    assert "### Tool Executions" in markdown
    # ... but the execution stage says NOT EXECUTED (max_tool_calls),
    # the result stage says NOT EXECUTED — and no duration, no SUCCESS
    assert "* `c9` (`search_repository`) \u2192 NOT EXECUTED (max_tool_calls reached)" in markdown
    assert "**Result:** NOT EXECUTED (execution budget exhausted)" in markdown
    assert "\u00b7 duration:" not in markdown
    assert "\u2192 SUCCESS" not in markdown
    # no dispatch ever happened, no fabricated ToolResult
    assert dispatcher.calls == []
    payload = trace.to_dict()
    iteration = payload["iterations"][0]
    assert iteration["tool_results"] == []
    assert len(iteration["blocked_tool_calls"]) == 1
    assert payload["summary"]["tool_calls_executed"] == 0
    assert payload["summary"]["tool_calls_blocked"] == 1
    # the continuation terminates the run for that exact reason
    assert iteration["continuation"] == {
        "decision": "TERMINATE",
        "reason": "max_tool_calls reached",
        "termination_reason": "EXECUTION_LIMIT_REACHED",
    }
    assert (
        "Next LLM request: none (run terminated: EXECUTION_LIMIT_REACHED)"
        in markdown
    )
    # the blocked run terminates with explicit learner prose
    assert "### Termination" in markdown
    assert "Tool-call limit reached." in markdown
    assert "### Continuation" not in markdown


# ---------------------------------------------------------------------------
# Test 7 — the M8.8 validation stage is unchanged semantically
# ---------------------------------------------------------------------------


def test_evidence_validation_stage_is_unchanged_semantically():
    result, agent, dispatcher = _run(
        responses=[
            _tool_calls_response(
                _call("read_file", {"path": PATH_READ}, "c_read")
            ),
            _text_response(STRUCTURED_FINAL),
        ],
        results=[_read_result()],
        config=AgentExecutionConfig(structured_claims=True),
    )
    # same validator, same statuses, same reason vocabulary
    assert [entry.status for entry in result.evidence_validation] == [
        "SUPPORTED"
    ]
    assert result.evidence_validation[0].reason.startswith(
        "matched gathered evidence"
    )
    payload = agent.last_trace.to_dict()
    recorded = payload["evidence_validation"]
    assert recorded["results"][0]["status"] == "SUPPORTED"
    assert recorded["claims"][0]["statement"] == (
        "The repository defines a DAG Agent."
    )
    # validation is post-run: the only dispatch was the investigation
    assert [call.id for call in dispatcher.calls] == ["c_read"]

    # ... and in the markdown it reads as the stage after the
    # iterations and before the run summary, never inside the loop
    markdown = render_markdown(agent.last_trace)
    answer_at = markdown.index("### Answer")
    claims_at = markdown.index("### Structured Claims")
    validation_at = markdown.index("### M8.8 Evidence Validation")
    summary_at = markdown.index("## Run Summary")
    assert answer_at < claims_at < validation_at < summary_at
    assert "`SUPPORTED` \u2014 The repository defines a DAG Agent." in markdown
    assert "evidence: `{}`".format("{}:1-20".format(PATH_READ)) in markdown


# ---------------------------------------------------------------------------
# Banned reasoning vocabulary never leaks into either projection
# ---------------------------------------------------------------------------


def test_stage_projections_carry_no_hidden_reasoning():
    result, agent, _ = _run(
        responses=[
            _tool_calls_response(_call("search_repository", {"query": "x"})),
            _text_response(),
        ],
        results=[_search_result("x")],
    )
    payload = agent.last_trace.to_dict()
    markdown = render_markdown(agent.last_trace)
    serialized = json.dumps(payload).lower()
    for banned in (
        "thought",
        "chain_of_thought",
        "private_reasoning",
        "hidden_reasoning",
        "internal_monologue",
        "reasoning",
        "scratchpad",
    ):
        assert banned not in serialized
        assert banned not in markdown.lower()
