"""Deterministic tests for the shared ReAct context-safety controls
(``repo_intelligence.react_safety``).

Covers the milestone's required properties at the shared policy,
execution, and provider boundaries — with fake clients and fake tools
only (no live API calls):

* every outgoing ReAct model request is checked, not just the first;
* enforcement uses the COMPLETE constructed request (messages +
  tool definitions + response-format schema);
* oversized requests cause ZERO provider calls;
* tool outputs are bounded BEFORE entering subsequent history;
* many individually acceptable observations still hit the boundary;
* per-request history restriction preserves evidence/provenance;
* repeated iterations cannot exceed configured execution limits;
* a budget rejection does not consume a provider-call allowance;
* an actual provider failure (reached provider) does consume it;
* two distinct dummy ReAct agents reuse the SAME abstractions.
"""

from __future__ import annotations

import pytest

from repo_intelligence.react_safety import (
    MAX_TOOL_RESPONSE_CHARS,
    BudgetedLLMClient,
    ReActLoopGuard,
    ReActSafetyConfig,
    ReActTermination,
    RequestBudgetExceededError,
    bound_observation,
    restrict_history,
)
from repo_intelligence.agent.request_budget import RequestBudgetPolicy
from repo_intelligence.llm.models import (
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    ToolCall,
    ToolDefinition,
    Usage,
)


# -- fakes -------------------------------------------------------------------


class FakeClient:
    """Scripted client: *turns* tool-call turns, then an answer."""

    def __init__(self, turns: int = 0, answer: str = "done"):
        self._turns = turns
        self._answer = answer
        self.seen = []
        self.model = "fake-model"

    def complete(self, request):
        self.seen.append(request)
        index = len(self.seen)
        if index <= self._turns:
            message = Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="c{}".format(index),
                    name="fake_tool",
                    arguments={"i": index},
                )],
            )
            return LLMResponse(
                message=message,
                finish_reason="tool_calls",
                usage=Usage(input_tokens=7, output_tokens=3,
                            total_tokens=10),
            )
        return LLMResponse(
            message=Message(role="assistant", content=self._answer),
            finish_reason="stop",
            usage=Usage(input_tokens=7, output_tokens=3, total_tokens=10),
        )


class RaisingClient:
    """Provider-side failure AFTER the request reached the provider."""

    def __init__(self):
        self.calls = 0
        self.model = "fake-model"

    def complete(self, request):
        self.calls += 1
        raise RuntimeError("provider-side failure")


def run_mini_react(client, question, config, payload_fn):
    """The generic mini loop: shared controls only, no agent names."""
    budgeted = BudgetedLLMClient(
        client, policy=config.request_budget_policy()
    )
    guard = ReActLoopGuard(config)
    messages = [
        Message(role="system", content="react contract"),
        Message(role="user", content=question),
    ]
    while True:
        if not guard.allow_iteration():
            break
        guard.begin_iteration()
        try:
            response = budgeted.complete(
                LLMRequest(messages=list(messages))
            )
        except RequestBudgetExceededError:
            guard.record_budget_rejection(budgeted.rejected[-1])
            break
        guard.record_model_response(
            response, budget_record=budgeted.records[-1]
        )
        messages.append(response.message)
        if response.finish_reason != "tool_calls":
            break
        for call in response.message.tool_calls:
            if not guard.allow_tool_call():
                break
            bounded, _record = bound_observation(
                payload_fn(call),
                config.max_observation_chars,
                label="fake_tool",
            )
            guard.record_tool_call()
            messages.append(Message(role="tool", content=bounded))
    guard.stop(guard.termination_reason or ReActTermination.COMPLETED)
    return budgeted, guard, messages


# -- policy boundary ---------------------------------------------------------


def test_config_validates_and_separates_local_from_global_budget():
    config = ReActSafetyConfig()
    # defaults are finite and safe
    assert config.max_iterations >= 1
    assert config.max_request_input_tokens is None
    # None → the PROJECT default policy (global H1 default untouched)
    assert config.request_budget_policy() == RequestBudgetPolicy()
    assert (
        config.request_budget_policy().effective_input_budget_tokens
        == 95_232
    )
    # a set value → a tighter LOCAL policy (same machinery, no global
    # default change)
    local = ReActSafetyConfig(max_request_input_tokens=8000)
    policy = local.request_budget_policy()
    assert policy.effective_input_budget_tokens == 8000
    assert policy.context_capacity_tokens is None
    # constructing local configs never mutates the global default
    assert RequestBudgetPolicy().effective_input_budget_tokens == 95_232

    for bad in (0, -1, True, "8"):
        with pytest.raises(ValueError):
            ReActSafetyConfig(max_iterations=bad)
    with pytest.raises(ValueError):
        ReActSafetyConfig(max_observation_chars=0)
    with pytest.raises(ValueError):
        ReActSafetyConfig(max_request_input_tokens=0)


# -- provider boundary: EVERY request checked --------------------------------


def test_every_outgoing_request_is_checked_not_just_the_first():
    config = ReActSafetyConfig(
        max_iterations=6, max_tool_calls=6, max_model_calls=8,
        max_observation_chars=4000, max_request_input_tokens=12000,
    )
    client = FakeClient(turns=3, answer="done")
    budgeted, guard, _messages = run_mini_react(
        client, "q", config, lambda call: "evidence\n" * 100
    )
    # one record per attempted complete() — initial AND every
    # subsequent iteration
    assert guard.model_calls >= 4
    assert len(budgeted.records) == guard.model_calls
    assert len(budgeted.sent) == guard.model_calls
    assert budgeted.rejected == []
    # the fake provider saw every allowed request (none skipped,
    # none silently dropped)
    assert len(client.seen) == guard.model_calls
    assert guard.termination_reason == ReActTermination.COMPLETED


def test_budget_uses_the_complete_constructed_request():
    config = ReActSafetyConfig(max_request_input_tokens=1000)
    inner = FakeClient(turns=0)
    budgeted = BudgetedLLMClient(
        inner, policy=config.request_budget_policy()
    )
    request = LLMRequest(
        messages=[
            Message(role="system", content="instructions"),
            Message(role="user", content="task"),
        ],
        tools=[ToolDefinition(
            name="t",
            description="D" * 400,
            input_schema={"type": "object"},
        )],
        structured_output=StructuredOutputSpec(
            name="Result", schema={"type": "object"}
        ),
    )
    budgeted.complete(request)
    record = budgeted.records[-1]
    totals = record["totals"]
    # complete-request measurement: messages AND tool definitions
    # AND structured-output schema all counted
    assert totals["messages_characters"] >= len("instructions") + 1
    assert record["tool_definitions"]["characters"] >= 400
    assert record["structured_output"]["characters"] > 0
    assert totals["total_characters"] == (
        totals["messages_characters"]
        + record["tool_definitions"]["characters"]
        + record["structured_output"]["characters"]
    )
    assert record["estimation"]["is_estimate"] is True


def test_oversized_request_rejected_with_zero_provider_calls():
    config = ReActSafetyConfig(max_request_input_tokens=1000)
    inner = FakeClient(turns=0)
    budgeted = BudgetedLLMClient(
        inner, policy=config.request_budget_policy()
    )
    oversized = LLMRequest(
        messages=[Message(role="system", content="S" * 9000),
                  Message(role="user", content="q")],
        tools=[ToolDefinition(
            name="big", description="D" * 6000,
            input_schema={"type": "object"},
        )],
        structured_output=StructuredOutputSpec(
            name="Result", schema={"type": "object"}
        ),
    )
    with pytest.raises(RequestBudgetExceededError) as caught:
        budgeted.complete(oversized)
    assert caught.value.record["within_budget"] is False
    assert inner.seen == [], "the provider must never be invoked"
    assert budgeted.calls_reached_provider == 0
    assert budgeted.sent == []
    assert len(budgeted.rejected) == 1
    # the rejection is recorded but never retried unchanged: there is
    # no retry loop in the client itself
    assert len(budgeted.records) == 1


def test_budget_rejection_does_not_consume_provider_allowance():
    config = ReActSafetyConfig(max_request_input_tokens=1000)
    inner = FakeClient(turns=0)
    budgeted = BudgetedLLMClient(
        inner, policy=config.request_budget_policy()
    )
    # rejection first (provider untouched)
    with pytest.raises(RequestBudgetExceededError):
        budgeted.complete(LLMRequest(
            messages=[Message(role="system", content="S" * 9000)]
        ))
    assert budgeted.calls_reached_provider == 0
    # a within-budget call afterwards is still permitted: the one-call
    # allowance (as maintained by the caller) was NOT consumed
    budgeted.complete(LLMRequest(
        messages=[Message(role="user", content="small")]
    ))
    assert budgeted.calls_reached_provider == 1
    assert len(inner.seen) == 1


def test_provider_failure_consumes_the_allowance():
    config = ReActSafetyConfig(max_request_input_tokens=8000)
    raiser = RaisingClient()
    budgeted = BudgetedLLMClient(
        raiser, policy=config.request_budget_policy()
    )
    with pytest.raises(RuntimeError):
        budgeted.complete(LLMRequest(
            messages=[Message(role="user", content="hi")]
        ))
    # the request DID reach the provider — the call counts
    assert raiser.calls == 1
    assert budgeted.calls_reached_provider == 1
    # and the failure itself is surfaced unchanged (never swallowed)
    # — no automatic retry happened (exactly one inner call)
    assert len(budgeted.sent) == 1


def test_accepted_request_is_sent_byte_identical():
    config = ReActSafetyConfig(max_request_input_tokens=8000)
    inner = FakeClient(turns=0)
    budgeted = BudgetedLLMClient(
        inner, policy=config.request_budget_policy()
    )
    request = LLMRequest(
        messages=[Message(role="user", content="hello")]
    )
    response = budgeted.complete(request)
    assert inner.seen[0] is budgeted.sent[0]
    assert budgeted.records[-1]["disposition"] == "accepted"
    assert response.finish_reason == "stop"


# -- observation bounding ----------------------------------------------------


def test_observation_is_bounded_before_entering_history():
    config = ReActSafetyConfig(max_observation_chars=2000)
    client = FakeClient(turns=3, answer="done")
    _budgeted, _guard, messages = run_mini_react(
        client, "q", config,
        lambda call: "evidence {}\n".format(call.arguments["i"])
        + "row\n" * 2000,
    )
    tool_messages = [m for m in messages if m.role == "tool"]
    assert tool_messages, "observations must enter history"
    for message in tool_messages:
        assert len(message.content) <= config.max_observation_chars, \
            "bounded BEFORE entering history"


def test_bound_observation_metadata_and_determinism():
    raw = "line one\nline two\n" + "x" * 10_000
    first, record = bound_observation(raw, 4000, label="tool_t")
    second, record2 = bound_observation(raw, 4000, label="tool_t")
    assert first == second, "deterministic"
    assert record == record2
    assert record["truncated"] is True
    assert record["original_chars"] == len(raw)
    assert record["returned_chars"] < len(raw)
    assert record["char_limit"] == 4000
    assert record["label"] == "tool_t"
    assert record["note"], "continuation hint must be present"
    assert len(first) <= 4000
    # a small observation is returned untouched (no false truncation)
    small, small_rec = bound_observation("tiny", 4000)
    assert small == "tiny"
    assert small_rec["truncated"] is False
    with pytest.raises(ValueError):
        bound_observation("x", 0)


def test_many_acceptable_observations_still_hit_the_boundary():
    """Each observation fits its own cap; the COMBINED request does
    not fit the input budget — the client boundary must intervene."""
    config = ReActSafetyConfig(
        max_iterations=8, max_tool_calls=8, max_model_calls=8,
        max_observation_chars=2000, max_request_input_tokens=1000,
    )
    client = FakeClient(turns=4, answer="done")
    budgeted, guard, messages = run_mini_react(
        client, "q", config,
        lambda call: "evidence {}\n".format(call.arguments["i"])
        + "row\n" * 900,
    )
    history_chars = sum(len(m.content or "") for m in messages)
    budget_chars = (
        config.request_budget_policy()
        .effective_input_budget_characters
    )
    # individually acceptable (each <= 2000 cap)...
    for message in messages:
        if message.role == "tool":
            assert len(message.content) <= 2000
    # ...but the combined history grew past the request budget...
    assert history_chars > budget_chars
    # ...so the boundary intervened: reduction and/or rejection, and
    # every request allowed to be SENT actually fit
    assert budgeted.rejected or any(
        r["disposition"] == "reduced" for r in budgeted.records
    )
    for record in budgeted.records:
        if record["disposition"] in ("accepted", "reduced"):
            assert record["within_budget"] is True
    assert guard.termination_reason in (
        ReActTermination.COMPLETED,
        ReActTermination.REQUEST_BUDGET_EXCEEDED,
    )


# -- per-request history restriction (provenance) ----------------------------


def test_restriction_preserves_evidence_and_provenance():
    transcript = [
        Message(role="system", content="SYS provenance contract"),
        Message(role="user", content="question"),
    ]
    for step in range(6):
        transcript.append(Message(
            role="assistant", content="",
            tool_calls=[ToolCall(
                id="s{}".format(step), name="t", arguments={"i": step},
            )],
        ))
        transcript.append(Message(
            role="tool",
            content="obs {} {}".format(step, "y" * 4000),
        ))
        transcript.append(Message(
            role="user", content="proceed to step {}".format(step)
        ))
    original_len = len(transcript)
    kept, record = restrict_history(
        transcript, max_chars=8000, keep_recent=1
    )
    # the caller's transcript is NEVER mutated (evidence stays put)
    assert len(transcript) == original_len
    # protected content survives
    assert any(m.role == "system" for m in kept)
    # tool call/result pairs stay WHOLE
    calls = [m for m in kept if m.role == "assistant" and m.tool_calls]
    results = [m for m in kept if m.role == "tool"]
    assert len(calls) == len(results)
    # the record is inspectable and honest
    assert record["dropped_units"] >= 1
    assert record["dropped_chars"] > 0
    assert record["within_limit"] is True
    assert record["original_chars"] > record["returned_chars"]
    # under the ceiling → no drop at all
    kept2, record2 = restrict_history(transcript, max_chars=10 ** 9)
    assert len(kept2) == original_len
    assert record2["dropped_units"] == 0
    assert record2["within_limit"] is True


# -- bounded execution -------------------------------------------------------


def test_iterations_cannot_exceed_configured_limits():
    config = ReActSafetyConfig(max_iterations=3, max_tool_calls=100)
    client = FakeClient(turns=50, answer="never reached")
    _budgeted, guard, _messages = run_mini_react(
        client, "q", config, lambda call: "obs"
    )
    assert guard.iterations <= config.max_iterations
    assert guard.termination_reason == (
        ReActTermination.ITERATION_LIMIT_REACHED
    )
    # every later action is refused once terminal
    assert guard.allow_iteration() is False
    assert guard.allow_tool_call() is False
    assert guard.allow_model_call() is False



def test_recording_last_allowed_tool_does_not_preempt_finalization():
    config = ReActSafetyConfig(max_tool_calls=2)
    guard = ReActLoopGuard(config)

    assert guard.allow_tool_call() is True
    guard.record_tool_call()
    assert guard.allow_tool_call() is True
    guard.record_tool_call()

    # The middleware must get a chance to run one no-tools synthesis turn.
    # The next attempted tool is denied and marks the budget termination.
    assert guard.tool_calls == config.max_tool_calls
    assert guard.termination_reason is None
    assert guard.allow_tool_call() is False
    assert guard.termination_reason == ReActTermination.TOOL_CALL_LIMIT_REACHED

def test_tool_calls_and_model_calls_cannot_exceed_limits():
    config = ReActSafetyConfig(
        max_iterations=50, max_tool_calls=2, max_model_calls=50
    )
    client = FakeClient(turns=50, answer="never")
    _budgeted, guard, _messages = run_mini_react(
        client, "q", config, lambda call: "obs"
    )
    assert guard.tool_calls <= config.max_tool_calls
    assert guard.termination_reason == (
        ReActTermination.TOOL_CALL_LIMIT_REACHED
    )

    model_config = ReActSafetyConfig(max_model_calls=2)
    guard2 = ReActLoopGuard(model_config)
    response = LLMResponse(
        message=Message(role="assistant", content="x"),
        finish_reason="stop",
    )
    assert guard2.allow_model_call() is True
    guard2.record_model_response(response)
    assert guard2.allow_model_call() is True
    guard2.record_model_response(response)
    assert guard2.allow_model_call() is False
    assert guard2.termination_reason == (
        ReActTermination.MODEL_CALL_LIMIT_REACHED
    )


def test_cumulative_token_allowance_uses_reported_then_estimated():
    config = ReActSafetyConfig(
        max_cumulative_input_tokens=100,
        max_model_calls=100,
    )
    guard = ReActLoopGuard(config)
    reported = LLMResponse(
        message=Message(role="assistant", content="x"),
        finish_reason="stop",
        usage=Usage(input_tokens=40, output_tokens=1, total_tokens=41),
    )
    guard.record_model_response(reported)
    assert guard.cumulative_reported_input_tokens == 40
    assert guard.cumulative_input_tokens == 40
    # no usage on the response → fall back to the PREFLIGHT ESTIMATE,
    # tracked separately so reported and estimated never blur
    no_usage = LLMResponse(
        message=Message(role="assistant", content="x"),
        finish_reason="stop",
    )
    record = {
        "totals": {"total_tokens_est": 70},
        "estimation": {"is_estimate": True},
    }
    guard.record_model_response(no_usage, budget_record=record)
    assert guard.cumulative_estimated_input_tokens == 70
    assert guard.cumulative_input_tokens == 110
    # ceiling reached → the next model call is refused
    assert guard.allow_model_call() is False
    assert guard.termination_reason == (
        ReActTermination.CUMULATIVE_BUDGET_EXCEEDED
    )
    snapshot = guard.snapshot()
    assert snapshot["cumulative_input_tokens"] == 110
    assert snapshot["termination_reason"] == (
        ReActTermination.CUMULATIVE_BUDGET_EXCEEDED
    )
    # a budget rejection terminates safely and is inspectable
    guard2 = ReActLoopGuard(config)
    guard2.record_budget_rejection({"totals": {"total_tokens_est": 5}})
    assert guard2.termination_reason == (
        ReActTermination.REQUEST_BUDGET_EXCEEDED
    )
    assert guard2.budget_exceeded_record == {
        "totals": {"total_tokens_est": 5}
    }
    assert guard2.allow_model_call() is False


# -- reuse across distinct agents --------------------------------------------


def test_two_distinct_dummy_agents_share_the_same_abstractions():
    """Two different loops (different questions, tools, payloads,
    clients) are protected by the IDENTICAL shared classes — the
    module contains no agent-name or task-name branching."""
    import inspect

    import repo_intelligence.react_safety as react_safety

    source = inspect.getsource(react_safety)
    for forbidden in ("investigator", "Investigator",
                      "architecture", "flow_dependency",
                      "task_id", "agent_name"):
        assert forbidden not in source, \
            "shared module must not reference {!r}".format(forbidden)

    config = ReActSafetyConfig(
        max_iterations=6, max_tool_calls=6, max_model_calls=8,
        max_observation_chars=4000, max_request_input_tokens=12000,
    )
    client_a = FakeClient(turns=2, answer="A done")
    budgeted_a, guard_a, _ = run_mini_react(
        client_a, "question A", config,
        lambda call: "alpha evidence\n" * 50,
    )
    client_b = FakeClient(turns=3, answer="B done")
    budgeted_b, guard_b, _ = run_mini_react(
        client_b, "question B", config,
        lambda call: "beta evidence " * 300,
    )
    assert type(budgeted_a) is type(budgeted_b) is BudgetedLLMClient
    assert type(guard_a) is type(guard_b) is ReActLoopGuard
    assert guard_a.termination_reason == guard_b.termination_reason
    assert guard_a.termination_reason == ReActTermination.COMPLETED
    # both got per-request enforcement on EVERY call
    assert len(budgeted_a.records) == guard_a.model_calls
    assert len(budgeted_b.records) == guard_b.model_calls
    # both ceilings come from the same policy builder
    assert (
        budgeted_a.records[0]["budget"]
        == budgeted_b.records[0]["budget"]
    )


def test_module_reuses_project_machinery_not_a_competing_budget():
    import inspect

    import repo_intelligence.react_safety as react_safety
    from repo_intelligence.tools import protocol

    source = inspect.getsource(react_safety)
    # the shared module DELEGATES to the existing estimator/machinery
    assert "enforce_request_budget_or_raise" in source
    assert "RequestBudgetPolicy" in source
    # no duplicate estimator constants or second tokenizer
    assert "CHARS_PER_TOKEN =" not in source
    assert "def estimate_tokens" not in source
    # and it reuses the tool-protocol response ceiling as the
    # observation default rather than inventing a competing size cap
    assert react_safety.MAX_TOOL_RESPONSE_CHARS is (
        protocol.MAX_TOOL_RESPONSE_CHARS
    )


def test_observation_default_matches_tool_protocol_ceiling():
    assert (
        ReActSafetyConfig().max_observation_chars
        == MAX_TOOL_RESPONSE_CHARS
    )
