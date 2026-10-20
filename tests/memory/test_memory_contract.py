"""M14 — memory contract + application-generated identity (§11/§12).

Contract enforcement is the Memory Manager/application's authority —
never the model's:

 * valid candidate accepted (returned unchanged)
 * non-candidate / empty statement / empty owner / invalid memory_type
   / invalid scope / invalid evidence rejected (MEMORY_CONTRACT_
   VIOLATION, structured code)
 * line-bound rules match the M13 EvidenceRef semantics
 * MemoryContext (the injected execution configuration) validates its
   own fields
 * identity: the application generates ``mem_<opaque>`` ids, ids are
   unique, and a candidate CANNOT carry an id (the model never
   controls identity)
 * the stored projection (``to_dict``) is the documented contract

Everything here is synthetic (``example-repository``) and LLM-free.
"""

from __future__ import annotations

import pytest

from repo_intelligence.evidence import EvidenceRef
from repo_intelligence.memory import (
    MEMORY_CONTRACT_VIOLATION,
    MEMORY_STATUSES,
    MEMORY_TYPES,
    MEMORY_TYPE_EPISODIC,
    MEMORY_TYPE_SEMANTIC,
    MemoryCandidate,
    MemoryContext,
    MemoryManager,
    MemoryOperationError,
    MemoryOwner,
    MemoryScope,
    candidate_from,
    new_memory_id,
    validate_candidate,
)

OWNER = MemoryOwner(type="agent", id="repo_intelligence.agent")
REPO = "example-repository"
COMMIT = "a" * 40


def _candidate(**overrides) -> MemoryCandidate:
    fields = dict(
        owner=OWNER,
        memory_type=MEMORY_TYPE_SEMANTIC,
        statement="example-repository exposes widget:sync",
        scope=MemoryScope(repository=REPO, commit=COMMIT),
        evidence=(EvidenceRef(path="agents/widget.md", line_start=4, line_end=9),),
        subject="widget",
        source_run_id="run-1",
    )
    fields.update(overrides)
    return MemoryCandidate(**fields)


# ---------------------------------------------------------------------------
# contract — accept / reject
# ---------------------------------------------------------------------------


def test_valid_candidate_is_accepted_unchanged():
    candidate = _candidate()
    assert validate_candidate(candidate) is candidate


def test_non_candidate_is_rejected_with_structured_code():
    with pytest.raises(MemoryOperationError) as excinfo:
        validate_candidate({"statement": "a mapping, not a candidate"})
    assert excinfo.value.code == MEMORY_CONTRACT_VIOLATION


@pytest.mark.parametrize(
    "overrides",
    [
        {"statement": ""},
        {"statement": "   "},
        {"statement": 42},
        {"memory_type": "working"},
        {"memory_type": ""},
        {"owner": None},
        {"owner": MemoryOwner(type="", id="x")},
        {"owner": MemoryOwner(type="agent", id="")},
        {"scope": {"repository": REPO}},
        {"scope": None},
        {"subject": ""},
        {"source_run_id": "  "},
        {"evidence": "not-a-sequence"},
        {"evidence": ("path-as-string",)},
        {"evidence": (EvidenceRef(path=""),)},
        {"evidence": (EvidenceRef(path="x", line_start=0),)},
        {"evidence": (EvidenceRef(path="x", line_start=5, line_end=2),)},
    ],
)
def test_contract_violations_are_rejected(overrides):
    with pytest.raises(MemoryOperationError) as excinfo:
        validate_candidate(_candidate(**overrides))
    assert excinfo.value.code == MEMORY_CONTRACT_VIOLATION


def test_scope_values_may_be_absent_but_not_blank():
    validate_candidate(_candidate(scope=MemoryScope()))
    with pytest.raises(MemoryOperationError) as excinfo:
        validate_candidate(_candidate(scope=MemoryScope(repository=" ")))
    assert excinfo.value.code == MEMORY_CONTRACT_VIOLATION


def test_valid_vocabulary_is_small_and_explicit():
    assert MEMORY_TYPES == (MEMORY_TYPE_SEMANTIC, MEMORY_TYPE_EPISODIC)
    assert MEMORY_STATUSES == ("verified", "superseded", "invalid")


# ---------------------------------------------------------------------------
# identity — application-generated, opaque, unique
# ---------------------------------------------------------------------------


def test_candidate_cannot_carry_a_memory_id():
    candidate = _candidate()
    assert not hasattr(candidate, "memory_id")


def test_manager_generates_opaque_unique_ids(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    first = manager.store(_candidate())
    second = manager.store(_candidate())
    for memory in (first, second):
        assert memory.memory_id.startswith("mem_")
        assert memory.memory_id[4:].isalnum()
        assert len(memory.memory_id) == 4 + 32
    assert first.memory_id != second.memory_id


def test_new_memory_id_is_never_content_derived():
    first = new_memory_id()
    second = new_memory_id()
    assert first != second
    assert "widget" not in first and REPO not in first


# ---------------------------------------------------------------------------
# MemoryContext — injected execution configuration
# ---------------------------------------------------------------------------


def test_memory_context_defaults_are_bounded_and_deterministic():
    context = MemoryContext()
    assert context.scope == MemoryScope()
    assert context.owner is None
    assert context.memory_type == MEMORY_TYPE_EPISODIC
    assert context.search_limit == 10
    assert context.subject is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory_type": "working"},
        {"search_limit": 0},
        {"search_limit": -1},
        {"search_limit": True},
        {"owner": "agent"},
        {"scope": None},
        {"subject": ""},
    ],
)
def test_memory_context_rejects_invalid_fields(kwargs):
    with pytest.raises(MemoryOperationError) as excinfo:
        MemoryContext(**kwargs)
    assert excinfo.value.code == MEMORY_CONTRACT_VIOLATION


# ---------------------------------------------------------------------------
# stored projection + mapping construction
# ---------------------------------------------------------------------------


def test_stored_memory_to_dict_is_the_documented_contract(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    stored = manager.store(_candidate())
    payload = stored.to_dict()
    assert payload["memory_id"] == stored.memory_id
    assert payload["owner"] == {"type": "agent", "id": OWNER.id}
    assert payload["memory_type"] == MEMORY_TYPE_SEMANTIC
    assert payload["content"]["statement"] == _candidate().statement
    assert payload["scope"] == {"repository": REPO, "commit": COMMIT}
    assert payload["evidence"] == [
        {"path": "agents/widget.md", "line_start": 4, "line_end": 9}
    ]
    assert payload["status"] == "verified"
    assert payload["subject"] == "widget"
    assert payload["source_run_id"] == "run-1"
    assert payload["supersedes"] is None
    assert payload["superseded_by"] is None
    assert payload["created_at"] <= payload["last_verified_at"]
    assert set(payload) == {
        "memory_id",
        "owner",
        "memory_type",
        "content",
        "scope",
        "evidence",
        "subject",
        "source_run_id",
        "status",
        "created_at",
        "last_verified_at",
        "supersedes",
        "superseded_by",
    }


def test_candidate_from_mapping_builds_and_validates():
    candidate = candidate_from(
        {
            "owner": {"type": "agent", "id": OWNER.id},
            "memory_type": "episodic",
            "content": {"statement": "investigation X pinned the graph"},
            "scope": {"repository": REPO, "commit": COMMIT},
            "evidence": [
                {"path": "a.py", "line_start": 1, "line_end": 3},
                {"path": "b.py"},
            ],
            "subject": "graph",
        }
    )
    assert candidate.memory_type == MEMORY_TYPE_EPISODIC
    assert len(candidate.evidence) == 2
    assert candidate.evidence[1].line_start is None


def test_candidate_from_mapping_enforces_the_same_contract():
    with pytest.raises(MemoryOperationError) as excinfo:
        candidate_from(
            {"owner": {"type": "agent", "id": OWNER.id}, "memory_type": "x"}
        )
    assert excinfo.value.code == MEMORY_CONTRACT_VIOLATION


def test_memory_has_no_hidden_reasoning_fields(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db")
    manager.initialize()
    stored = manager.store(_candidate())
    payload = stored.to_dict()
    for banned in ("reasoning", "reasoning_trace", "thoughts", "chain"):
        assert banned not in payload
