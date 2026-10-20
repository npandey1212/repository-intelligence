"""Middleware-protected LangGraph ``create_agent`` POC.

A small, fully-tested demonstration that the project's generic ReAct
context-safety controls (``repo_intelligence.react_safety``) can be
adapted as REAL langchain middleware around a REAL ``create_agent``
investigator inside a LangGraph ``StateGraph`` workflow, while reusing
the existing planning/role contracts and the typed investigation/analysis
contracts, including explicit source-report provenance for P4.

This is a POC/extension — it is NOT on the production path (the
production path remains the legacy orchestrator with the opt-in
M21 graph task loop). Importing this package requires the optional
``langgraph``/``langchain`` extras.
"""

from __future__ import annotations

from .graph import (
    AGENT_RECURSION_LIMIT,
    CREATE_AGENT_INVESTIGATION_PROMPT_V1,
    CREATE_AGENT_ANALYST_PROMPT_V1,
    GRAPH_RECURSION_LIMIT,
    LangGraphPocState,
    POC_PLAN_STEP_IDS,
    POC_ROLE_OVERRIDES,
    PocInvestigator,
    PocAnalyst,
    build_create_agent_analyst,
    build_create_agent_investigator,
    build_poc_graph,
    run_poc,
    validate_poc_synthesis,
)
from .middleware import (
    ExecutionLimitMiddleware,
    ObservationBoundingMiddleware,
    ReActExecutionLimitExceeded,
    RequestBudgetMiddleware,
    llm_request_from_model_request,
)
from .tools import investigator_tool_definitions, to_langchain_tool

__all__ = [
    "AGENT_RECURSION_LIMIT",
    "CREATE_AGENT_INVESTIGATION_PROMPT_V1",
    "CREATE_AGENT_ANALYST_PROMPT_V1",
    "GRAPH_RECURSION_LIMIT",
    "ExecutionLimitMiddleware",
    "LangGraphPocState",
    "ObservationBoundingMiddleware",
    "POC_PLAN_STEP_IDS",
    "POC_ROLE_OVERRIDES",
    "PocInvestigator",
    "PocAnalyst",
    "ReActExecutionLimitExceeded",
    "RequestBudgetMiddleware",
    "build_create_agent_analyst",
    "build_create_agent_investigator",
    "build_poc_graph",
    "investigator_tool_definitions",
    "llm_request_from_model_request",
    "run_poc",
    "validate_poc_synthesis",
    "to_langchain_tool",
]
