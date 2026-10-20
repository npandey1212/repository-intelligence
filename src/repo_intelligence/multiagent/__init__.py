"""M16 Multi-Agent Foundations — the smallest two-specialist workflow.

Three participants, typed contracts, no new agent framework:

* **Investigator** — the existing ``Agent`` with all four read-only
  tools; output: typed ``InvestigationReport`` (an evidence package).
* **Analyst** — the existing ``Agent`` with NO tools; output: typed
  ``ArchitectureFinding`` constrained to the report's evidence.
* **Validator** — deterministic code (``validate_collaboration``):
  eight rules over recorded state, reusing the M13 evidence system.

Everything else here is plumbing the spec demands and nothing more:
typed messages (``contracts``), an explicit status machine
(``status``), per-role tool isolation (``permissions``), sequential
execution + traceable handoffs (``runner``).

Boundaries (spec §24): no LangGraph/CrewAI/AutoGen, no parallel
agents, no message broker, no M17 orchestration, no speculative tool
execution. The M17 orchestrator (across tasks/analyses) is a
different layer built ON these foundations.
"""

from __future__ import annotations

from .contracts import (
    CLASSIFICATIONS,
    MESSAGE_TYPES,
    AgentMessage,
    ArchitectureConclusion,
    ArchitectureFinding,
    ContractError,
    InvestigationFinding,
    InvestigationReport,
    ReportEvidenceRef,
    TaskPayload,
    ToolFailure,
    ValidationCheck,
    ValidationRequest,
    ValidationResult,
    make_message,
    message_from_dict,
    message_to_dict,
    report_json,
)
from .permissions import (
    ANALYST_TOOLS,
    INVESTIGATOR_TOOLS,
    VALIDATOR_TOOLS,
    restricted_dispatcher,
    tool_allowlist,
    tool_definitions_for,
)
from .runner import (
    DEFAULT_MAX_MESSAGES,
    MultiAgentRunner,
    MultiAgentTaskState,
    RunnerConfig,
)
from .status import (
    BUDGET_EXCEEDED,
    COMPLETED,
    FAILED,
    INCOMPLETE,
    PENDING,
    RUNNING,
    TASK_STATUSES,
    TIMEOUT,
    VALIDATION_FAILED,
    StatusTransitionError,
    advance,
    can_advance,
    is_terminal,
)
from .validation import RULE_ORDER, validate_collaboration

__all__ = [
    "ANALYST_TOOLS",
    "BUDGET_EXCEEDED",
    "CLASSIFICATIONS",
    "COMPLETED",
    "DEFAULT_MAX_MESSAGES",
    "FAILED",
    "INCOMPLETE",
    "INVESTIGATOR_TOOLS",
    "MESSAGE_TYPES",
    "PENDING",
    "RULE_ORDER",
    "RUNNING",
    "TASK_STATUSES",
    "TIMEOUT",
    "VALIDATION_FAILED",
    "VALIDATOR_TOOLS",
    "AgentMessage",
    "ArchitectureConclusion",
    "ArchitectureFinding",
    "ContractError",
    "InvestigationFinding",
    "InvestigationReport",
    "MultiAgentRunner",
    "MultiAgentTaskState",
    "ReportEvidenceRef",
    "RunnerConfig",
    "StatusTransitionError",
    "TaskPayload",
    "ToolFailure",
    "ValidationCheck",
    "ValidationRequest",
    "ValidationResult",
    "advance",
    "can_advance",
    "is_terminal",
    "make_message",
    "message_from_dict",
    "message_to_dict",
    "restricted_dispatcher",
    "report_json",
    "tool_allowlist",
    "tool_definitions_for",
    "validate_collaboration",
]
