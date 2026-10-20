"""M14 — the deterministic memory formation policy after M13
validation (§18/§42).

Formation is a NAMED, bounded, DETERMINISTIC policy over validation
results — not a side effect of validation, and never "every claim":

 * only SUPPORTED claims become candidates (UNSUPPORTED/UNKNOWN die
   with the execution)
 * order preserved, default bound ``MAX_MEMORY_CANDIDATES``
 * provenance carried forward: statement + evidence_refs + run id +
   scope + subject + owner + type
 * ``Claim`` (M13) stays the M13 concept; ``MemoryCandidate`` (M14)
   is built FROM it, never instead of it
 * same validation in → same candidates out: no LLM, no scoring,
   no learned policy — the formation policy is deterministic
"""

from __future__ import annotations

import pytest

from repo_intelligence.evidence import (
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    Claim,
    ClaimValidation,
    EvidenceRef,
)
from repo_intelligence.memory import (
    MAX_MEMORY_CANDIDATES,
    MEMORY_TYPE_EPISODIC,
    MemoryOwner,
    MemoryScope,
    select_memory_candidates,
)

OWNER = MemoryOwner(type="agent", id="repo_intelligence.agent")
SCOPE = MemoryScope(repository="example-repository", commit="a" * 40)


def _claim(statement, *refs):
    return Claim(
        statement=statement,
        evidence_refs=tuple(
            EvidenceRef(path=path, line_start=start, line_end=end)
            for path, start, end in refs
        ),
    )


def _result(status, statement, refs=()):
    claim = _claim(statement, *refs)
    return ClaimValidation(
        claim=claim,
        status=status,
        matched_evidence=(),
        reason="deterministic",
    )


SUPPORTED_REFS = (("agents/widget.md", 4, 9),)


def _select(validation, **overrides):
    kwargs = dict(owner=OWNER, scope=SCOPE, subject="widget", run_id="run-7")
    kwargs.update(overrides)
    return select_memory_candidates(validation, **kwargs)


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------


def test_only_supported_claims_become_candidates():
    candidates = _select(
        [
            _result(STATUS_SUPPORTED, "supported finding", SUPPORTED_REFS),
            _result(STATUS_UNSUPPORTED, "unsupported finding"),
            _result(STATUS_UNKNOWN, "unknown finding"),
        ]
    )
    assert [c.statement for c in candidates] == ["supported finding"]


def test_selection_is_deterministic_for_identical_input():
    """The formation policy is DETERMINISTIC (M14 §18): the same
    validation results always produce the same candidates — no model,
    no randomness, no judgment call is involved."""
    validations = (
        _result(STATUS_SUPPORTED, "first supported claim"),
        _result(STATUS_SUPPORTED, "second supported claim"),
        _result(STATUS_UNSUPPORTED, "unsupported guess"),
    )
    kwargs = dict(
        owner=OWNER, scope=SCOPE, subject="widget", run_id="run-7"
    )
    first = select_memory_candidates(validations, **kwargs)
    second = select_memory_candidates(validations, **kwargs)
    assert first == second
    assert [c.statement for c in first] == [
        "first supported claim",
        "second supported claim",
    ]


def test_empty_validation_yields_no_candidates():
    assert _select([]) == ()


def test_all_unsupported_yields_no_candidates():
    assert _select([_result(STATUS_UNSUPPORTED, "x")]) == ()


# ---------------------------------------------------------------------------
# provenance + bounds
# ---------------------------------------------------------------------------


def test_candidate_carries_full_provenance():
    (candidate,) = _select(
        [
            _result(
                STATUS_SUPPORTED,
                "widget registry is generated",
                SUPPORTED_REFS,
            )
        ]
    )
    assert candidate.owner == OWNER
    assert candidate.memory_type == MEMORY_TYPE_EPISODIC  # default
    assert candidate.statement == "widget registry is generated"
    assert candidate.scope == SCOPE
    assert candidate.subject == "widget"
    assert candidate.source_run_id == "run-7"
    # evidence REFERENCES only — the M13 refs travel verbatim
    assert candidate.evidence == (
        EvidenceRef(path="agents/widget.md", line_start=4, line_end=9),
    )
    assert not hasattr(candidate, "memory_id")


def test_memory_type_is_configurable_per_execution():
    (candidate,) = _select(
        [_result(STATUS_SUPPORTED, "fact", SUPPORTED_REFS)],
        memory_type="semantic",
    )
    assert candidate.memory_type == "semantic"


def test_selection_is_ordered_and_bounded_by_default_limit():
    validation = [
        _result(STATUS_SUPPORTED, "supported {}".format(index), SUPPORTED_REFS)
        for index in range(9)
    ]
    candidates = _select(validation)
    assert len(candidates) == MAX_MEMORY_CANDIDATES
    assert [c.statement for c in candidates] == [
        "supported {}".format(index)
        for index in range(MAX_MEMORY_CANDIDATES)
    ]


def test_custom_limit_bounds_selection():
    validation = [
        _result(STATUS_SUPPORTED, "supported {}".format(index))
        for index in range(4)
    ]
    assert len(_select(validation, limit=2)) == 2
    assert _select(validation, limit=0) == ()


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_limit_is_a_value_error(limit):
    with pytest.raises(ValueError):
        _select([_result(STATUS_SUPPORTED, "x")], limit=limit)
