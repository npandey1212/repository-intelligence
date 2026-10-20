"""M15 — renderer + provider neutrality (§28/§55).

 * rendered text = header + budget line + task + one block per item
   in presentation order, with provenance (source, reason, authority,
   freshness, scope, evidence) and the untouched open questions /
   constraints
 * an empty package renders an explicit "(no context items...)" line
   — never an empty string
 * a non-package argument → CONTEXT_RENDER_FAILURE (a render failure
   is explicit, never a silent empty context)
 * provider neutrality by SOURCE SCAN: nothing in the package
   imports an LLM client/SDK/model class or sqlite3 — the memory DB
   stays the ONLY SQLite surface (§43), and only ``adapters`` may
   touch the memory package (records, never the store)
 * NO hidden context: the renderer emits only what the package
   selected — no decision audit, no excluded content
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repo_intelligence.context_engine import (
    CONTEXT_HEADER,
    CONTEXT_RENDER_FAILURE,
    ContextBudget,
    ContextCandidate,
    ContextOperationError,
    TaskContext,
    render_context_text,
    select_context,
)
from repo_intelligence.evidence import EvidenceRef

REPO = "example-repository"
COMMIT = "a" * 40

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / (
    "src/repo_intelligence/context_engine"
)


def _package(step_id="001", **task_overrides):
    task_fields = dict(
        question="How is the widget registry wired?",
        repository=REPO,
        commit=COMMIT,
        open_questions=("who owns widget:sync?",),
        constraints=("do not modify tools/",),
    )
    task_fields.update(task_overrides)
    task = TaskContext(**task_fields)
    candidates = [
        ContextCandidate(
            candidate_id="c_ev",
            source_type="evidence",
            content="widget:sync is declared in agents/widget.md\n"
            "line 4 starts the block",
            source_id="agents/widget.md",
            repository_scope=REPO,
            commit_scope=COMMIT,
            evidence=(EvidenceRef(path="agents/widget.md", line_start=4,
                                  line_end=9),),
        ),
        ContextCandidate(
            candidate_id="c_mem",
            source_type="memory",
            content="registry layout remembered",
            source_id="mem_1",
            memory_id="mem_1",
            repository_scope=REPO,
            commit_scope=COMMIT,
            freshness="fresh",
        ),
        ContextCandidate(
            candidate_id="c_other",
            source_type="graph",
            content="must not appear",
            repository_scope="elsewhere",
        ),
    ]
    return select_context(task, candidates, ContextBudget(),
                          step_id).package


# ---------------------------------------------------------------------------
# rendered text
# ---------------------------------------------------------------------------


def test_render_contains_header_task_budget_and_scope():
    text = render_context_text(_package())
    assert text.startswith(CONTEXT_HEADER)
    assert "Task: How is the widget registry wired?" in text
    assert "step 001" in text
    assert "scope: {}@aaaaaaaaaaaa".format(REPO) in text
    assert "budget:" in text
    assert "/12000 chars" in text
    assert "/20 items" in text


def test_render_shows_every_item_with_provenance_and_reason():
    text = render_context_text(_package())
    assert "[ctxitem_001]" in text
    assert "CURRENT_EVIDENCE" in text
    assert "VERIFIED_MEMORY" in text
    assert "freshness fresh" in text
    assert "authority evidence" in text
    assert "source mem_1" in text
    # evidence rendered as path:lines
    assert "evidence: agents/widget.md:4-9" in text
    # multi-line content keeps its lines, indented under the item
    assert "  line 4 starts the block" in text


def test_render_presentation_order_is_task_then_sections():
    text = render_context_text(_package())
    assert text.index("CURRENT_EVIDENCE") < text.index("VERIFIED_MEMORY")
    assert text.index("[ctxitem_001]") < text.index("[ctxitem_002]")
    # excluded candidate content never enters the model context
    assert "must not appear" not in text


def test_render_carries_open_questions_and_constraints():
    text = render_context_text(_package())
    assert "Open questions:" in text
    assert "* who owns widget:sync?" in text
    assert "Constraints:" in text
    assert "* do not modify tools/" in text


def test_empty_package_renders_explicit_empty_not_blank():
    task = TaskContext(question="q?", repository=REPO, commit=COMMIT)
    package = select_context(
        task,
        [ContextCandidate(
            candidate_id="c1", source_type="evidence",
            content="x", repository_scope="elsewhere",
        )],
        ContextBudget(),
    ).package
    text = render_context_text(package)
    assert "(no context items selected for this step)" in text
    assert "Task: q?" in text
    assert text.strip()


def test_render_is_deterministic():
    first = render_context_text(_package())
    second = render_context_text(_package())
    # positional ids are stable; context_id/created_at differ only in
    # the metadata line — compare item blocks exactly
    first_items = [l for l in first.splitlines() if l.startswith("[")]
    second_items = [l for l in second.splitlines() if l.startswith("[")]
    assert first_items == second_items
    assert first.index("Task:") == second.index("Task:")


def test_render_failure_for_non_package():
    with pytest.raises(ContextOperationError) as excinfo:
        render_context_text("just a string")
    assert excinfo.value.code == CONTEXT_RENDER_FAILURE
    with pytest.raises(ContextOperationError) as excinfo:
        render_context_text({"context_id": "ctx_x"})
    assert excinfo.value.code == CONTEXT_RENDER_FAILURE


# ---------------------------------------------------------------------------
# provider neutrality + boundaries (source scans)
# ---------------------------------------------------------------------------


def _sources():
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE_ROOT.glob("*.py"))
    }


def test_package_imports_no_llm_client_sdk_or_model_class():
    for name, source in _sources().items():
        assert "import sqlite3" not in source, name
        assert "from ..llm" not in source, name
        assert "repo_intelligence.llm" not in source, name
        assert "openai" not in source.lower(), name
        assert "anthropic" not in source.lower(), name


def test_only_adapters_may_import_the_memory_package():
    for name, source in _sources().items():
        if name == "adapters.py":
            continue
        assert "from ..memory" not in source, name
        assert "repo_intelligence.memory" not in source, name
    # adapters consume RECORDS — never the sqlite store itself
    assert "import sqlite3" not in _sources()["adapters.py"]


def test_policy_and_engine_have_no_file_or_llm_logic():
    policy = (PACKAGE_ROOT / "policy.py").read_text(encoding="utf-8")
    assert "open(" not in policy
    assert "json" not in policy
    assert "import llm" not in policy
