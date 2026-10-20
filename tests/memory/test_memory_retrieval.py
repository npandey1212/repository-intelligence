"""M14 — deterministic retrieval (§13/§24).

search() is the normal discovery mechanism: exact/controlled matching
over repository, commit, owner, memory_type, subject, status — with a
bound and a stable order. No embeddings, no ranking, no LLM anywhere
in this file; every fixture is synthetic ``example-repository`` data.
"""

from __future__ import annotations

import pytest

from repo_intelligence.memory import (
    MemoryCandidate,
    MemoryManager,
    MemoryOwner,
    MemoryScope,
)

AGENT_A = MemoryOwner(type="agent", id="repo_intelligence.agent")
AGENT_B = MemoryOwner(type="agent", id="example.second_agent")
REPO = "example-repository"
OTHER_REPO = "example-other-repository"
COMMIT = "a" * 40


def _candidate(
    statement,
    repository=REPO,
    commit=COMMIT,
    owner=AGENT_A,
    memory_type="semantic",
    subject=None,
):
    return MemoryCandidate(
        owner=owner,
        memory_type=memory_type,
        statement=statement,
        scope=MemoryScope(repository=repository, commit=commit),
        subject=subject,
    )


@pytest.fixture()
def manager(tmp_path):
    memory_manager = MemoryManager(tmp_path / "memory.db")
    memory_manager.initialize()
    return memory_manager


def _seed(manager):
    return {
        "semantic_widget": manager.store(
            _candidate(
                "widget registry is generated",
                subject="widget",
                memory_type="semantic",
            )
        ),
        "episodic_widget": manager.store(
            _candidate(
                "run X validated the widget registry",
                subject="widget",
                memory_type="episodic",
            )
        ),
        "semantic_other": manager.store(
            _candidate(
                "other module declared",
                subject="other",
                memory_type="semantic",
            )
        ),
        "other_repo": manager.store(
            _candidate(
                "foreign repository memory",
                repository=OTHER_REPO,
                subject="widget",
            )
        ),
        "agent_b": manager.store(
            _candidate(
                "owned by another agent",
                owner=AGENT_B,
                subject="widget",
            )
        ),
    }


# ---------------------------------------------------------------------------
# dimension filters
# ---------------------------------------------------------------------------


def test_repository_filter(manager):
    _seed(manager)
    found = manager.search(repository=REPO)
    assert len(found) == 4
    assert all(m.scope.repository == REPO for m in found)
    assert manager.search(repository=OTHER_REPO)[0].statement == (
        "foreign repository memory"
    )


def test_commit_filter_finds_the_exact_scope(manager):
    _seed(manager)
    fresh = manager.store(
        _candidate("newer state memory", commit="b" * 40)
    )
    assert manager.search(commit="b" * 40) == (fresh,)
    assert len(manager.search(commit=COMMIT)) == 5


def test_owner_filter(manager):
    _seed(manager)
    assert len(manager.search(owner=AGENT_A)) == 4
    only_b = manager.search(owner=AGENT_B)
    assert [m.statement for m in only_b] == ["owned by another agent"]


def test_memory_type_filter(manager):
    _seed(manager)
    semantic = manager.search(memory_type="semantic")
    assert {m.memory_type for m in semantic} == {"semantic"}
    assert len(semantic) == 4
    episodic = manager.search(memory_type="episodic")
    assert [m.statement for m in episodic] == [
        "run X validated the widget registry"
    ]


def test_subject_filter(manager):
    _seed(manager)
    widget = manager.search(subject="widget")
    assert len(widget) == 4
    assert all(m.subject == "widget" for m in widget)
    assert manager.search(subject="does-not-exist") == ()


def test_status_filter(manager):
    stored = manager.store(_candidate("verified memory"))
    manager.invalidate(stored.memory_id)
    verified = manager.search(status="verified")
    assert verified == ()
    invalid = manager.search(status="invalid")
    assert [m.memory_id for m in invalid] == [stored.memory_id]


def test_combined_filters_narrow_exactly(manager):
    _seed(manager)
    found = manager.search(
        repository=REPO,
        owner=AGENT_A,
        memory_type="semantic",
        subject="widget",
        status="verified",
        commit=COMMIT,
    )
    assert [m.statement for m in found] == ["widget registry is generated"]


def test_no_filters_returns_everything_bounded(manager):
    memories = _seed(manager)
    assert len(manager.search()) == len(memories)


# ---------------------------------------------------------------------------
# determinism + bounds
# ---------------------------------------------------------------------------


def test_search_is_deterministic_and_ordered_by_creation(manager):
    _seed(manager)
    first = manager.search()
    second = manager.search()
    assert [m.memory_id for m in first] == [m.memory_id for m in second]
    created = [m.created_at for m in first]
    assert created == sorted(created)


def test_search_limit_is_enforced(manager):
    for index in range(5):
        manager.store(_candidate("memory {}".format(index)))
    assert len(manager.search(limit=3)) == 3
    assert len(manager.search()) == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": -2},
        {"limit": True},
        {"memory_type": "working"},
        {"status": "pending"},
    ],
)
def test_invalid_search_arguments_are_value_errors(manager, kwargs):
    with pytest.raises(ValueError):
        manager.search(**kwargs)


def test_search_never_calls_an_llm():
    """The memory package must stay provider-neutral (§64): its
    source contains no provider import, SDK, or client reference."""
    from pathlib import Path

    import repo_intelligence.memory as memory_package

    package_dir = Path(memory_package.__file__).parent
    sources = sorted(package_dir.glob("*.py"))
    assert sources
    for source in sources:
        text = source.read_text().lower()
        assert "openai" not in text
        assert "llmclient" not in text
        assert "import llm" not in text
