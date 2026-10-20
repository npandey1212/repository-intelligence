"""M19 task scheduling — dependency readiness over task records.

The scheduler's ONE job today: say which tasks are READY. It is
deliberately separate from the execution policy (the Orchestrator's
sequential loop decides WHEN to dispatch), so readiness logic can be
tested independently and later drive concurrent dispatch without
redesigning task records or the agent boundary.

Dependency model:

* **Explicit** — every ``TaskRecord`` carries ``depends_on`` task ids;
  a task is ready only when EVERY dependency reached ``completed``
  (a failed/blocked dependency means the dependent task is never
  ready — it gets blocked by the coordinator's failure semantics).
* **Implicit sequential chain (the compatibility path)** —
  ``sequential_dependencies`` derives "step i depends on step i-1"
  from plan order, which reproduces today's list-order sequential
  execution exactly: readiness advances one plan-order step at a
  time. Nothing else assumes order.

Result ORDERING is always plan/task-list order, never completion
order — the property future parallel dispatch must preserve.

No parallelism lives here: no threads, no dispatch loop, no budget
reservation. A pending task with a failed dependency simply never
becomes ready; whether an already-running sibling is allowed to
finish is coordinator policy, documented as deferred (M19 design
doc), not implemented.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from .contracts import TASK_COMPLETED, TASK_PENDING, TASK_READY
from .models import TaskRecord

__all__ = [
    "ready_tasks",
    "sequential_dependencies",
]


def sequential_dependencies(
    task_ids: Sequence[str],
) -> Dict[str, List[str]]:
    """The implicit sequential chain: each task depends on its
    predecessor (first task has no dependencies).

    This is the compatibility mapping from "plan list order" to
    explicit dependency edges — the only place order becomes
    dependency, so readiness and plan order stay separable concepts.
    """
    dependencies: Dict[str, List[str]] = {}
    previous: Optional[str] = None
    for task_id in task_ids:
        dependencies[task_id] = [] if previous is None else [previous]
        previous = task_id
    return dependencies


def ready_tasks(
    tasks: Sequence[TaskRecord],
    completed_ids: Optional[Mapping[str, bool]] = None,
) -> List[TaskRecord]:
    """Tasks whose dependencies ALL completed, in plan/list order.

    *completion* defaults to each record's own status
    (``TASK_COMPLETED``); ``completed_ids`` replaces that source
    entirely so readiness can be tested purely without full records —
    a task marked completed THERE is not ready, and a dependency not
    marked there blocks. Tasks not in ``pending``/``ready`` are never
    returned (they already ran, are running, or are terminal), and a
    pending task whose dependency failed is not ready.
    """
    if completed_ids is None:
        done = {
            task.task_id
            for task in tasks
            if task.status == TASK_COMPLETED
        }
    else:
        done = {task_id for task_id, ok in completed_ids.items() if ok}
    ready: List[TaskRecord] = []
    for task in tasks:
        if task.status not in (TASK_PENDING, TASK_READY):
            continue
        if completed_ids is not None and completed_ids.get(task.task_id):
            continue
        if all(dependency in done for dependency in task.depends_on):
            ready.append(task)
    return ready
