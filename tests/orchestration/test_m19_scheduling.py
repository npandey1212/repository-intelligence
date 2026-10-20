"""M19 scheduling — pure dependency derivation and readiness.

These are the ONLY places plan order becomes a dependency edge;
readiness itself is a pure function over task records, testable
without running a single participant.
"""

from __future__ import annotations

from repo_intelligence.orchestration import (
    TASK_BLOCKED,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_PENDING,
    TASK_READY,
    TASK_RUNNING,
    ready_tasks,
    sequential_dependencies,
)
from repo_intelligence.orchestration.models import TaskRecord


def _records(*specs):
    return [
        TaskRecord(
            task_id=task_id,
            role=role,
            status=status,
            depends_on=list(depends_on),
        )
        for task_id, role, status, depends_on in specs
    ]


def test_sequential_dependencies_is_the_plan_order_chain():
    assert sequential_dependencies([]) == {}
    assert sequential_dependencies(["P1"]) == {"P1": []}
    assert sequential_dependencies(["P1", "P2", "P3"]) == {
        "P1": [],
        "P2": ["P1"],
        "P3": ["P2"],
    }


def test_first_pending_task_is_ready_and_the_rest_are_not():
    tasks = _records(
        ("P1", "INVESTIGATOR", TASK_PENDING, ()),
        ("P2", "ANALYST", TASK_PENDING, ("P1",)),
    )
    ready = ready_tasks(tasks)
    assert [record.task_id for record in ready] == ["P1"]


def test_ready_tasks_follow_plan_order():
    tasks = _records(
        ("P1", "INVESTIGATOR", TASK_PENDING, ()),
        ("P2", "INVESTIGATOR", TASK_PENDING, ("P1",)),
        ("P3", "ANALYST", TASK_PENDING, ("P2",)),
    )
    assert [record.task_id for record in ready_tasks(tasks)] == ["P1"]


def test_dependency_completion_unlocks_the_successor():
    tasks = _records(
        ("P1", "INVESTIGATOR", TASK_PENDING, ()),
        ("P2", "ANALYST", TASK_PENDING, ("P1",)),
    )
    tasks[0].status = TASK_COMPLETED
    assert [record.task_id for record in ready_tasks(tasks)] == ["P2"]


def test_running_completed_failed_and_blocked_tasks_are_never_ready():
    tasks = _records(
        ("P1", "INVESTIGATOR", TASK_RUNNING, ()),
        ("P2", "INVESTIGATOR", TASK_COMPLETED, ("P1",)),
        ("P3", "ANALYST", TASK_FAILED, ("P2",)),
        ("P4", "ANALYST", TASK_BLOCKED, ("P2",)),
        ("P5", "INVESTIGATOR", TASK_READY, ()),
    )
    assert [record.task_id for record in ready_tasks(tasks)] == ["P5"]


def test_pending_task_with_failed_dependency_is_not_ready():
    tasks = _records(
        ("P1", "INVESTIGATOR", TASK_FAILED, ()),
        ("P2", "ANALYST", TASK_PENDING, ("P1",)),
    )
    assert ready_tasks(tasks) == []


def test_completed_ids_override_supports_pure_readiness_tests():
    tasks = _records(
        ("P1", "INVESTIGATOR", TASK_PENDING, ()),
        ("P2", "ANALYST", TASK_PENDING, ("P1",)),
    )
    ready = ready_tasks(tasks, completed_ids={"P1": True})
    assert [record.task_id for record in ready] == ["P2"]
    ready = ready_tasks(tasks, completed_ids={"P1": False})
    assert [record.task_id for record in ready] == ["P1"]
