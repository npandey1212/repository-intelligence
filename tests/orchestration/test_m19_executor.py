"""M19 executor boundary — dispatch, classification, trace + usage.

Every participant here is scripted (``m17_doubles``) — no network, no
API key, no timing dependence. The adapter must classify failures into
the one taxonomy (never raise them), attribute only THIS call's trace,
and reject boundary misuse explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from repo_intelligence.agent import AgentExecutionConfig
from repo_intelligence.llm import LLMError
from repo_intelligence.multiagent import (
    ArchitectureFinding,
    InvestigationReport,
)
from repo_intelligence.multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from repo_intelligence.orchestration import (
    ANALYST,
    INVESTIGATOR,
    FAILURE_AGENT_BUDGET,
    FAILURE_PROVIDER,
    TASK_COMPLETED,
    TASK_FAILED,
    AgentTaskExecutor,
    AgentTaskRequest,
    AgentTaskResult,
    ParticipantRegistry,
    sequential_dependencies,
)

from m17_doubles import (
    COMMIT,
    README_REF,
    REPOSITORY,
    build,
    call,
    finding_payload,
    investigate_read_then_report,
    report_payload,
    structured,
    tool_calls,
)


def _request(task_id="P1", role=INVESTIGATOR, prompt="do the step"):
    return AgentTaskRequest(
        run_id="run-1",
        task_id=task_id,
        attempt=1,
        role=role,
        assignment_reason="test",
        question="question",
        goal="goal",
        prompt=prompt,
        repository_scope=REPOSITORY,
        commit_scope=COMMIT,
        depends_on=tuple(
            sequential_dependencies([task_id]).get(task_id, [])
        ),
    )


def test_default_executor_is_wired_into_the_orchestrator():
    orch, _, _ = build()
    assert isinstance(orch.executor, AgentTaskExecutor)
    assert orch.executor.registry.roles == (ANALYST, INVESTIGATOR)


def test_investigator_dispatch_returns_typed_output_trace_and_usage():
    orch, investigator, _ = build()
    result = orch.executor.execute(_request())
    assert isinstance(result, AgentTaskResult)
    assert result.status == TASK_COMPLETED
    assert result.failure is None
    assert isinstance(result.output, InvestigationReport)
    assert result.output.repository_scope == REPOSITORY
    assert result.evidence_refs
    assert result.agent_run_id
    assert result.trace is investigator.last_trace
    assert result.usage.tool_calls == 1
    assert result.usage.model_iterations == 2
    assert result.started_at and result.completed_at


def test_analyst_dispatch_selects_the_analyst_participant():
    orch, investigator, analyst = build()
    result = orch.executor.execute(
        _request(task_id="P2", role=ANALYST, prompt="analyse")
    )
    assert result.status == TASK_COMPLETED
    assert isinstance(result.output, ArchitectureFinding)
    assert analyst.last_trace is not None
    assert investigator._llm_client.requests == []


def test_participant_exception_is_returned_as_classified_failure():
    orch, _, _ = build(inv_responses=[LLMError("rate limit exceeded")])
    result = orch.executor.execute(_request())
    assert result.status == TASK_FAILED
    assert result.failure is not None
    assert result.failure.category == FAILURE_PROVIDER
    assert result.failure.retryable is True
    assert result.failure.error_type == "LLMError"
    assert result.output is None


def test_agent_budget_exhaustion_is_classified_not_raised():
    orch, _, _ = build(
        inv_responses=[
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ],
        inv_kwargs={
            "execution_config": AgentExecutionConfig(max_tool_calls=0)
        },
    )
    result = orch.executor.execute(_request())
    assert result.status == TASK_FAILED
    assert result.failure.category == FAILURE_AGENT_BUDGET
    assert result.failure.retryable is False


def test_trace_attribution_is_per_call_never_a_previous_trace():
    orch, _, _ = build(
        inv_responses=(
            investigate_read_then_report()
            + investigate_read_then_report()
        )
    )
    first = orch.executor.execute(_request())
    second = orch.executor.execute(_request())
    assert first.trace is not None
    assert second.trace is not None
    assert first.trace is not second.trace
    assert first.agent_run_id != second.agent_run_id


def _executor_for(participant) -> AgentTaskExecutor:
    return AgentTaskExecutor(ParticipantRegistry({INVESTIGATOR: participant}))


def test_failed_call_with_no_fresh_trace_returns_none_trace():
    participant = SimpleNamespace(
        last_trace=SimpleNamespace(run_id="old"),
        run_result=lambda prompt: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
    )
    executor = _executor_for(participant)
    result = executor.execute(_request())
    assert result.status == TASK_FAILED
    assert result.trace is None
    assert result.agent_run_id is None
    assert result.failure.error_type == "RuntimeError"


def test_specialist_participant_runs_through_the_same_boundary():
    trace = SimpleNamespace(
        run_id="specialist-run",
        iterations=[
            # Real AgentTraceIteration shape: the requested call AND
            # its executed result (M20 counts EXECUTED calls — a
            # result entry exists only when the tool actually ran).
            SimpleNamespace(
                tool_calls=[{"name": "read_file"}],
                tool_results=[{"tool": "read_file", "success": True}],
                response={},
            )
        ],
    )
    calls = []

    class FakeSpecialist:
        def run_result(self, prompt):
            calls.append(prompt)
            self.last_trace = trace
            return SimpleNamespace(
                structured_output=InvestigationReport.model_validate(
                    report_payload()
                ),
                last_trace=trace,
            )

    specialist = FakeSpecialist()
    executor = _executor_for(specialist)
    result = executor.execute(_request(prompt="specialist prompt"))
    assert calls == ["specialist prompt"]
    assert isinstance(result.output, InvestigationReport)
    assert result.trace is trace
    assert result.agent_run_id == "specialist-run"
    assert result.usage.tool_calls == 1
    assert result.usage.model_iterations == 1


def test_executor_rejects_non_request_boundary_misuse():
    orch, _, _ = build()
    with pytest.raises(ValueError, match="AgentTaskRequest"):
        orch.executor.execute({"task_id": "P1"})


def test_registry_never_falls_back_to_an_unknown_role():
    registry = ParticipantRegistry({INVESTIGATOR: object()})
    with pytest.raises(ValueError, match="known roles"):
        registry.select("JUDGE")


def test_registry_rejects_invalid_registrations_explicitly():
    registry = ParticipantRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", object())
    with pytest.raises(ValueError, match="must not be None"):
        registry.register("INVESTIGATOR", None)
    registry.register("INVESTIGATOR", object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("INVESTIGATOR", object())


def test_executor_usage_tokens_stay_none_without_provider_metadata():
    orch, _, _ = build()
    result = orch.executor.execute(_request())
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None


def test_scripted_investigator_script_is_exercised_by_the_adapter():
    orch, investigator, _ = build(
        inv_responses=investigate_read_then_report()
    )
    orch.executor.execute(_request())
    assert len(investigator._llm_client.requests) == 2
