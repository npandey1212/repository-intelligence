"""M17 boundary/finalization — the workflow keeps going past P1.

The minimal lifecycle fix at the orchestration level: an investigator
that reaches its execution boundary returns a STRUCTURED report (from
the state it already collected) instead of failing the step, so
P2…validation run as usual. Also pinned here:

* a boundary report that DECLARES ``incomplete`` still validates and
  still reaches the analyst — an execution boundary is a LOCAL status,
  never ``BUDGET_EXCEEDED`` for the workflow;
* provider context exhaustion is NOT retried (replaying the same
  oversized request cannot help);
* an ordinary transient provider failure IS still retried by the
  existing bounded retry.

Everything is scripted and LLM-free (see ``m17_doubles``).
"""

from __future__ import annotations

from repo_intelligence.agent import AgentExecutionConfig
from repo_intelligence.llm import LLMError
from repo_intelligence.orchestration import (
    COMPLETED,
    INCOMPLETE,
)

from m17_doubles import (
    README_REF,
    REPOSITORY,
    build,
    call,
    report_payload,
    structured,
    task,
    tool_calls,
    two_step_plan,
)

# The M17 budget under test: one tool call, then the boundary.
BOUNDARY_CONFIG = AgentExecutionConfig(
    max_tool_calls=1,
    finalize_on_boundary=True,
)

CONTEXT_LENGTH_FAILURE = (
    "LLM request failed: Error code: 400 - {'error': {'message': "
    "\"This model's maximum context length is 128000 tokens. However, "
    "your messages resulted in 129592 tokens.\", "
    "'type': 'invalid_request_error', "
    "'code': 'context_length_exceeded'}}"
)


def _ops(run):
    return run.event_operations()


def _events(run, operation):
    return [event for event in run.events if event["operation"] == operation]


def _boundary_responses(payload=None):
    """One executed tool call, one blocked tool call, then the
    finalization response (the pass that closes the run)."""
    return [
        tool_calls(call("read_file", dict(README_REF))),
        tool_calls(call("read_file", dict(README_REF))),
        structured(payload if payload is not None else report_payload()),
    ]


# ---------------------------------------------------------------------------
# 1 — boundary → structured result → P2 and the rest of the workflow
# ---------------------------------------------------------------------------


def test_boundary_finalization_lets_the_workflow_continue_past_p1():
    orch, investigator, analyst = build(
        inv_responses=_boundary_responses(),
        inv_kwargs={"execution_config": BOUNDARY_CONFIG},
    )

    run = orch.execute(task(), two_step_plan())

    # P1 completed through the boundary; nothing was retried.
    assert run.status == COMPLETED
    assert run.termination_reason == "SUCCESS"
    assert run.retries == 0
    assert run.step_result("P1").status == "completed"
    assert run.step_result("P1").attempts == 1
    assert run.completed_steps == ["P1", "P2"]
    assert run.failed_steps == [] and run.blocked_steps == []

    # The typed handoff really crossed to the downstream agents.
    assert run.report is not None
    assert run.report.repository_scope == REPOSITORY
    assert [ref.path for ref in run.report.evidence_refs] == ["README.md"]
    assert run.finding is not None
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert analyst._llm_client.requests

    # The lifecycle is observable on the investigator's OWN trace.
    operations = [
        event["operation"]
        for event in investigator.last_trace.lifecycle_events
    ]
    assert operations == [
        "execution.boundary_reached",
        "finalization.started",
        "tools.disabled",
        "finalization.completed",
    ]
    completed = investigator.last_trace.lifecycle_events[-1]
    assert completed["termination_reason"] == "STRUCTURED_OUTPUT"
    assert completed["tool_calls_executed"] == 1
    assert completed["max_tool_calls"] == 1
    assert _ops(run)[-1] == "orchestration.complete"


# ---------------------------------------------------------------------------
# 2 — a boundary report that declares `incomplete` is not a failure
# ---------------------------------------------------------------------------


def test_boundary_report_declared_incomplete_still_reaches_the_analyst():
    orch, investigator, analyst = build(
        inv_responses=_boundary_responses(
            report_payload(status="incomplete")
        ),
        inv_kwargs={"execution_config": BOUNDARY_CONFIG},
    )

    run = orch.execute(task(), two_step_plan())

    # Local status stays local: the report's own `incomplete` is
    # reported as such, the evidence still validates, and the analyst
    # still ran — the workflow never becomes BUDGET_EXCEEDED.
    assert run.status == INCOMPLETE
    assert run.termination_reason == "DECLARED_INCOMPLETE"
    assert run.report is not None
    assert run.report.status == "incomplete"
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert run.finding is not None
    assert analyst._llm_client.requests
    assert run.step_result("P1").status == "completed"
    assert run.final is True
    assert _ops(run)[-1] == "orchestration.terminated"
    assert _events(run, "orchestration.budget.exceeded") == []
    assert investigator.last_trace.lifecycle_events


# ---------------------------------------------------------------------------
# 3 — context exhaustion is an execution boundary, never a retry
# ---------------------------------------------------------------------------


def test_context_length_failure_is_not_retried():
    orch, investigator, analyst = build(
        inv_responses=[
            LLMError(CONTEXT_LENGTH_FAILURE),
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ],
    )

    run = orch.execute(task(), two_step_plan())

    # No blind replay of the oversized request: one attempt, no retry
    # event, and an explicit non-success status for the step.
    assert run.retries == 0
    assert _events(run, "orchestration.retry") == []
    assert run.step_result("P1").attempts == 1
    assert run.status == INCOMPLETE
    assert run.failed_steps == ["P1"]
    assert run.blocked_steps == ["P2"]
    assert analyst._llm_client.requests == []
    assert _ops(run)[-1] == "orchestration.terminated"


# ---------------------------------------------------------------------------
# 4 — ordinary transient failures keep the existing retry semantics
# ---------------------------------------------------------------------------


def test_transient_llm_error_is_still_retried_by_the_existing_policy():
    orch, investigator, analyst = build(
        inv_responses=[
            LLMError("LLM request failed: rate limit exceeded"),
            tool_calls(call("read_file", dict(README_REF))),
            structured(report_payload()),
        ],
    )

    run = orch.execute(task(), two_step_plan())

    assert run.status == COMPLETED
    assert run.retries == 1
    assert run.step_result("P1").attempts == 2
    retries = _events(run, "orchestration.retry")
    assert len(retries) == 1
    assert retries[0]["step_id"] == "P1"
    assert investigator.last_trace.lifecycle_events == []
    assert run.validation is not None
    assert run.validation.status == "passed"
    assert analyst._llm_client.requests
