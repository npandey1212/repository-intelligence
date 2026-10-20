"""Runtime metadata on the specialist's typed report (M17 final).

The live failure this file pins down: the model frequently returned a
report without ``question`` / ``repository_scope`` / ``commit_scope``,
so Pydantic rejected it and the run died with a validation error even
though the runtime KNEW all three values from the request it built.
They are execution state, so they come from the request itself — the
labeled assignment lines every question builder writes — filled in
before validation, never invented, never asked of the model:

1. direct mode      ``investigate(question, repository_scope,
                     commit_scope)`` — the TASK block is the source,
                     the three values round-trip into the report
2. orchestrated     the runner's own question (labels included) is
                     the source, through the same specialist path
3. never rewritten  a value the model DID provide stays the model's
4. substantive      the fill never rescues genuinely broken content:
                     a malformed entry still fails the run, recorded
                     as the explicit failure it is (with the fill
                     visible on the iteration, so nothing is hidden)

Everything is scripted and LLM-free: queued responses, fixture-bound
read-only tools, one question — no network, no API key.
"""

from __future__ import annotations

import pytest

from architecture_doubles import (
    COMMIT,
    QUESTION,
    REPOSITORY,
    build_investigator,
    report_payload,
    structured,
)

from repo_intelligence.llm import StructuredOutputValidationError

RUNNER_QUESTION = (
    "Investigate this plan step and produce your structured "
    "InvestigationReport.\n"
    "question: {question}\n"
    "step: S1 — identify the components\n"
    "repository_scope: {repository}\n"
    "commit_scope: {commit}\n"
    "Report findings backed only by evidence you inspected; carry "
    "open questions and uncertainties in the typed fields."
).format(question=QUESTION, repository=REPOSITORY, commit=COMMIT)

ALL_THREE = ["question", "repository_scope", "commit_scope"]


def _payload_without_metadata():
    payload = report_payload()
    del payload["question"]
    del payload["repository_scope"]
    del payload["commit_scope"]
    return payload


def _validation(specialist):
    return specialist.agent.last_trace.iterations[-1].structured_validation


# ---------------------------------------------------------------------------
# 1 — direct mode: the TASK block is the source of truth
# ---------------------------------------------------------------------------


def test_direct_run_fills_the_three_fields_from_the_request():
    specialist = build_investigator(
        structured(_payload_without_metadata())
    )

    report = specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    assert report.question == QUESTION
    assert report.repository_scope == REPOSITORY
    assert report.commit_scope == COMMIT
    validation = _validation(specialist)
    assert validation["status"] == "SUCCESS"
    assert validation["metadata_injected"] == ALL_THREE
    # the derived handoff id is built from exactly those values
    assert report.derived_investigation_id().startswith("inv-arch-")


# ---------------------------------------------------------------------------
# 2 — orchestrated mode: the runner's question is the source
# ---------------------------------------------------------------------------


def test_orchestrated_run_fills_the_fields_from_the_runner_question():
    specialist = build_investigator(
        structured(_payload_without_metadata())
    )

    result = specialist.run_result(RUNNER_QUESTION)

    report = result.structured_output
    # the projected M16 handoff envelope carries the same values
    assert report.question == QUESTION
    assert report.repository_scope == REPOSITORY
    assert report.commit_scope == COMMIT
    assert report.investigation_id.startswith("inv-arch-")
    validation = _validation(specialist)
    assert validation["metadata_injected"] == ALL_THREE


# ---------------------------------------------------------------------------
# 3 — a value the model provided is never rewritten
# ---------------------------------------------------------------------------


def test_a_metadata_value_the_model_provided_is_never_rewritten():
    payload = report_payload()
    payload["question"] = "A question the model stated itself"
    del payload["repository_scope"]
    del payload["commit_scope"]
    specialist = build_investigator(structured(payload))

    report = specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    # the model's own value stands; only the missing two are filled
    assert report.question == "A question the model stated itself"
    assert _validation(specialist)["metadata_injected"] == [
        "repository_scope",
        "commit_scope",
    ]


# ---------------------------------------------------------------------------
# 4 — substantive content is never rescued and never hidden
# ---------------------------------------------------------------------------


def test_a_substantive_failure_is_never_hidden_by_the_fill():
    payload = _payload_without_metadata()
    # a required field of a required entry is missing — substantive,
    # not metadata
    payload["components"] = [{"name": "component without an id"}]
    specialist = build_investigator(structured(payload))

    with pytest.raises(StructuredOutputValidationError):
        specialist.investigate(QUESTION, REPOSITORY, COMMIT)

    trace = specialist.agent.last_trace
    assert trace.termination_reason == (
        "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    )
    validation = _validation(specialist)
    assert validation["status"] == "FAILED"
    # the fill happened AND did not rescue the payload: the failure
    # below is the model's own content
    assert validation["metadata_injected"] == ALL_THREE
    assert "components" in validation["error"]
