from __future__ import annotations
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

import random 
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from pathlib import Path

# Allows running this script from the repository root without installing
# the package in editable mode.
#PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path("C:/self-study/ai/build/repository-intelligence/")
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.langgraph.planner import get_plan

from repo_intelligence.planning import (  # noqa: E402
    InvestigationPlan,
    PlanValidationError,
    build_planning_request,
    plan_from_response,
    plan_to_dict,
    validate_plan,
)
from repo_intelligence.planning.planner import PLANNING_SYSTEM_INSTRUCTION  # noqa: E402

class PlannerState(TypedDict):
    objective:str
    plan: InvestigationPlan


def planner_node(state: PlannerState) -> dict:
    plan = get_plan(state["objective"])
    if isinstance(plan, int):
        raise RuntimeError(f"Planner failed with code {plan}")
    return {"plan": plan}

# Build graph
builder = StateGraph(InvestigationPlan)
builder.add_node("planner_node", get_plan)
builder.add_edge(START, "planner_node")
builder.add_edge("planner_node", END)

# Compile graph
graph = builder.compile()
#display(Image(graph.get_graph().draw_mermaid_png()))
result = graph.invoke({
    "objective": "How do agents and MCP servers interact?"
})
print(result["plan"])