"""M19 trace attribution — step events mirror onto the RESULT's own
execution trace, never onto a participant's ``last_trace``.

``AgentTaskResult.trace`` is the boundary's attribution for one
execution: the orchestrator mirrors that attempt's step window onto
it and re-writes it through the sink. An injected executor's
participants may never run, may expose a stale ``last_trace``, or the
result may expose no usable trace at all — none of that may change or
crash the run: the run's event list stays authoritative, and a trace
whose result failed contract validation is never touched.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from repo_intelligence.agent.trace import AgentTrace
from repo_intelligence.agent.trace_sink import JsonFileTraceSink
from repo_intelligence.multiagent import (
    ArchitectureFinding,
    InvestigationReport,
)
from repo_intelligence.orchestration import (
    FAILURE_CONTRACT_VIOLATION,
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    VALIDATION_FAILED,
    AgentTaskRequest,
    AgentTaskResult,
    Orchestrator,
)

from m17_doubles import (
    build,
    finding_payload,
    report_payload,
    task,
    two_step_plan,
)


class _Scripted:
    """Fake boundary echoing the request's per-run ``run_id``."""

    def __init__(self, *results):
        self.requests = []
        self.results = list(results)

    def execute(self, request: AgentTaskRequest) -> AgentTaskResult:
        self.requests.append(request)
        if not self.results:
            raise AssertionError("executor invoked more times than scripted")
        return replace(self.results.pop(0), run_id=request.run_id)


def _ok(task_id, role, output, trace=None):
    return AgentTaskResult(
        run_id="run-1",
        task_id=task_id,
        attempt=1,
        role=role,
        status=TASK_COMPLETED,
        output=output,
        evidence_refs=tuple(getattr(output, "evidence_refs", ()) or ()),
        agent_run_id=getattr(trace, "run_id", None),
        trace=trace,
    )


def _report() -> InvestigationReport:
    return InvestigationReport.model_validate(report_payload())


def _finding() -> ArchitectureFinding:
    return ArchitectureFinding.model_validate(finding_payload())


def _orchestrator(executor, trace_sink=None):
    _, investigator, analyst = build()
    kwargs = {}
    if trace_sink is not None:
        kwargs["trace_sink"] = trace_sink
    orch = Orchestrator(investigator, analyst, executor=executor, **kwargs)
    return orch, investigator, analyst


# -- attribution: the result's trace, never the participant's ---------------


def test_step_events_mirror_onto_the_result_trace_not_the_participant():
    result_trace = AgentTrace.start("attempt-1")
    stale = AgentTrace.start("a previous run's trace")
    executor = _Scripted(
        _ok("P1", "INVESTIGATOR", _report(), trace=result_trace),
        _ok("P2", "ANALYST", _finding(), trace=AgentTrace.start("p2")),
    )
    orch, investigator, _ = _orchestrator(executor)
    # The participant never ran through this executor — its exposed
    # last_trace is stale and must stay untouched.
    investigator.last_trace = stale
    run = orch.execute(task(), two_step_plan())

    assert run.status == VALIDATION_FAILED  # scripted evidence only
    ops = [
        event["operation"]
        for event in result_trace.orchestration_events
    ]
    assert "orchestration.step.assign" in ops
    assert "orchestration.handoff.send" in ops
    assert "orchestration.agent.start" in ops
    assert "orchestration.agent.complete" in ops
    assert "orchestration.step.complete" in ops
    # run-level events stay off every trace (belong to no one task)
    assert "orchestration.complete" not in ops
    # the stale participant trace was never written to
    assert stale.orchestration_events == []
    assert investigator.last_trace is stale
    # the run's own event list stays authoritative and complete
    run_ops = run.event_operations()
    assert "orchestration.step.assign" in run_ops
    assert "orchestration.step.complete" in run_ops


def test_result_trace_is_written_through_the_sink(tmp_path):
    sink = JsonFileTraceSink(root=str(tmp_path / "traces"), markdown=True)
    result_trace = AgentTrace.start("attempt-1")
    executor = _Scripted(
        _ok("P1", "INVESTIGATOR", _report(), trace=result_trace),
        _ok("P2", "ANALYST", _finding(), trace=AgentTrace.start("p2")),
    )
    orch, _, _ = _orchestrator(executor, trace_sink=sink)
    orch.execute(task(), two_step_plan())

    trace_dir = tmp_path / "traces" / result_trace.run_id
    payload = json.loads(
        (trace_dir / "trace.json").read_text(encoding="utf-8")
    )
    assert payload.get("orchestration_events")
    markdown = (trace_dir / "trace.md").read_text(encoding="utf-8")
    assert "## Orchestration" in markdown


def test_contract_violation_never_touches_the_untrusted_trace():
    untrusted = AgentTrace.start("belongs to another execution")
    mismatched = replace(
        _ok("P1", "INVESTIGATOR", _report(), trace=untrusted),
        task_id="WRONG",
    )
    executor = _Scripted(mismatched)
    orch, _, _ = _orchestrator(executor)
    run = orch.execute(task(), two_step_plan())

    p1, p2 = run.tasks
    assert p1.status == TASK_FAILED
    assert p1.failure_category == FAILURE_CONTRACT_VIOLATION
    assert p2.status == TASK_BLOCKED
    assert untrusted.orchestration_events == []
    # the window still closed onto the authoritative run log
    run_ops = run.event_operations()
    assert "orchestration.step.assign" in run_ops
    assert "orchestration.step.failed" in run_ops


def test_result_without_a_usable_trace_closes_the_window_cleanly():
    # No record_orchestration_event on the object: mirroring is
    # skipped (observability only), the run itself is unaffected.
    bare = SimpleNamespace(iterations=[])
    executor = _Scripted(
        _ok("P1", "INVESTIGATOR", _report(), trace=bare),
        _ok("P2", "ANALYST", _finding(), trace=AgentTrace.start("p2")),
    )
    orch, _, _ = _orchestrator(executor)
    run = orch.execute(task(), two_step_plan())

    assert run.status == VALIDATION_FAILED
    p1, _ = run.tasks
    assert p1.status == TASK_COMPLETED
    run_ops = run.event_operations()
    assert "orchestration.step.complete" in run_ops
    assert not hasattr(bare, "orchestration_events")
