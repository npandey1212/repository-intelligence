"""M19.1 sequential dispatch — plan order, one task at a time.

The dispatch policy stays application-owned and strictly sequential:
the loop walks plan order, each dispatch is gated by ``ready_tasks``
(``scheduling.py``), and no second task starts before the first one
finished (request order, attempt log, and event sequences all agree).
Parallelism remains out of scope (docs/m19 §2/§12).
"""

from __future__ import annotations

from repo_intelligence.multiagent import InvestigationReport
from repo_intelligence.orchestration import (
    COMPLETED,
    TASK_COMPLETED,
    AgentTaskExecutor,
    Orchestrator,
)

from m17_doubles import build, task, two_step_plan


class _Recorder:
    """Wraps the default adapter, capturing every dispatched request
    (execution behavior is unchanged — pure observation)."""

    def __init__(self, inner):
        self._inner = inner
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return self._inner.execute(request)


def _first_sequence(run, operation, step_id):
    for event in run.events:
        if (
            event.get("operation") == operation
            and event.get("step_id") == step_id
        ):
            return event["sequence"]
    raise AssertionError(
        "no {} event for step {}".format(operation, step_id)
    )


def test_dispatch_walks_plan_order_one_task_at_a_time():
    orch, investigator, analyst = build()
    recorder = _Recorder(
        AgentTaskExecutor.for_agents(investigator, analyst)
    )
    dispatched = Orchestrator(
        investigator, analyst, executor=recorder
    )
    run = dispatched.execute(task(), two_step_plan())

    assert run.status == COMPLETED
    # requests are issued in plan order, never speculatively ahead:
    # P2's request exists only after P1 returned a result.
    assert [request.task_id for request in recorder.requests] == [
        "P1",
        "P2",
    ]
    assert [entry["step_id"] for entry in run.agent_runs] == ["P1", "P2"]

    # event sequences prove non-overlap: P1 finished before P2 started
    p1_start = _first_sequence(
        run, "orchestration.agent.start", "P1"
    )
    p1_done = _first_sequence(
        run, "orchestration.agent.complete", "P1"
    )
    p2_start = _first_sequence(
        run, "orchestration.agent.start", "P2"
    )
    p2_done = _first_sequence(
        run, "orchestration.agent.complete", "P2"
    )
    assert p1_start < p1_done < p2_start < p2_done

    p1, p2 = run.tasks
    assert p1.status == TASK_COMPLETED
    assert p2.status == TASK_COMPLETED
    # both records exist up front (created together, dispatched in
    # order) — but only one ever ran at a time
    assert p1.depends_on == []
    assert p2.depends_on == ["P1"]
    assert p1.completed_at <= p2.started_at
    assert run.current_step is None  # sealed terminal state


def test_successor_request_carries_only_its_dependency_results():
    orch, investigator, analyst = build()
    recorder = _Recorder(
        AgentTaskExecutor.for_agents(investigator, analyst)
    )
    dispatched = Orchestrator(
        investigator, analyst, executor=recorder
    )
    run = dispatched.execute(task(), two_step_plan())

    first, second = recorder.requests
    assert first.depends_on == ()
    assert first.dependency_results == ()
    assert second.depends_on == ("P1",)
    # the successor sees exactly the merged report assembled from the
    # completed predecessor — nothing else
    assert len(second.dependency_results) == 1
    assert isinstance(second.dependency_results[0], InvestigationReport)
    assert second.dependency_results[0] is run.report
