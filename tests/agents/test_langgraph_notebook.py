"""Notebook validation for notebooks/langgraph_end_to_end_workflow.ipynb.

Pins the notebook's structural contract (valid nbformat-4 JSON, every
code cell parses, no secrets or machine-specific paths) and checks
that the production symbols it references actually exist — a renamed
seam must break this test, not a learner's kernel.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from repo_intelligence import repository_query as rq

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "langgraph_end_to_end_workflow.ipynb"
)

REQUIRED_RQ_SYMBOLS = (
    "_run_with_langgraph",
    "_build_plan",
    "ORCHESTRATION_LANGGRAPH",
    "ORCHESTRATION_LEGACY",
    "ORCHESTRATION_MODES",
    "ORCHESTRATION_UNAVAILABLE",
    "run_repository_query",
    "build_offline_models",
    "build_architect",
    "build_flow_investigator",
    "build_visualizer",
    "fixture_bindings",
    "assign_roles",
)


def _notebook():
    assert NOTEBOOK.is_file(), "primary LangGraph notebook is missing"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _sources(nb):
    return ["".join(cell.get("source", [])) for cell in nb["cells"]]


def test_notebook_is_valid_nbformat4_json():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell.get("source"), list)
        assert "id" in cell
        if cell["cell_type"] == "code":
            assert "outputs" in cell
            assert "execution_count" in cell


def test_every_code_cell_parses_and_no_secrets_or_machine_paths():
    nb = _notebook()
    code_sources = [
        source
        for cell, source in zip(nb["cells"], _sources(nb))
        if cell["cell_type"] == "code"
    ]
    assert len(code_sources) >= 12  # the walkthrough is substantial
    for source in code_sources:
        ast.parse(source)  # no IPython magics: pure Python cells
        assert "sk-" not in source
        assert "OPENAI_API_KEY=" not in source.replace(" ", "")
        assert "/home/" not in source
        assert "C:/" not in source


def test_notebook_covers_every_required_section():
    text = "\n".join(_sources(_notebook()))
    for heading in (
        "Learning objectives",
        "Environment and setup",
        "Inspect the real components",
        "Build (compile) the actual graph",
        "Execute an end-to-end query",
        "Inspect state transitions and evidence",
        "Failure behavior",
        "Legacy versus LangGraph",
        "Safeguards",
        "Learning summary",
    ):
        assert heading in text, heading
    assert "```mermaid" in text  # the architecture diagram
    assert 'orchestration="langgraph"' in text


def test_referenced_production_symbols_exist():
    for name in REQUIRED_RQ_SYMBOLS:
        assert hasattr(rq, name), name
    assert rq.ORCHESTRATION_LEGACY == "legacy"
    assert rq.ORCHESTRATION_LANGGRAPH == "langgraph"
    assert rq.ORCHESTRATION_MODES == (
        rq.ORCHESTRATION_LEGACY,
        rq.ORCHESTRATION_LANGGRAPH,
    )

    pytest.importorskip("langgraph.graph", reason="optional graph extra")
    from repo_intelligence.orchestration import langgraph_runner as lg

    assert callable(lg.run_with_langgraph)
    assert callable(lg.build_langgraph_orchestration_graph)
    assert "task" in lg.LangGraphOrchestrationState.__annotations__
    assert "plan" in lg.LangGraphOrchestrationState.__annotations__

    doubles_path = (
        Path(__file__).resolve().parents[1]
        / "orchestration"
        / "m17_doubles.py"
    )
    assert doubles_path.is_file()
