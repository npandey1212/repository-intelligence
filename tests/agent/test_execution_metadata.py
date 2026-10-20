"""Deterministic execution metadata — the unit contract (M17 final).

``question`` / ``repository_scope`` / ``commit_scope`` say which
investigation a typed report belongs to: runtime state, never a model
conclusion. This file pins the two pure functions that make that
possible, without any run:

1. whitelist        exactly the three runtime fields — no summary,
                    component, finding, or evidence value ever
2. extraction       the request's own line-start ``label: value``
                    lines, first non-empty occurrence each; a request
                    without them (a plain question) yields nothing
3. injection        only fields the contract declares, only where the
                    payload left them missing or empty; a value the
                    model provided is never rewritten, a present
                    non-string value stays the model's own (the
                    contract's type errors must remain visible)
4. no-op            nothing to fill → the payload object itself comes
                    back unchanged, so an un-injected run's records
                    stay byte-for-byte what they were

No network, no API key, no LLM.
"""

from __future__ import annotations

from repo_intelligence.agent import (
    EXECUTION_METADATA_FIELDS,
    execution_metadata_from_request,
    inject_execution_metadata,
)
from repo_intelligence.agents.models.architecture_report import (
    ArchitectureComponent,
    ArchitectureReport,
)

QUESTION = "How do the modules interact?"
REPOSITORY = "example-repository"
COMMIT = "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"

REQUEST = (
    "ROLE\n"
    "\n"
    "You are an Architecture Investigator.\n"
    "\n"
    "TASK\n"
    "question: {question}\n"
    "repository_scope: {repository}\n"
    "commit_scope: {commit}".format(
        question=QUESTION,
        repository=REPOSITORY,
        commit=COMMIT,
    )
)


def _metadata(**overrides):
    metadata = {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }
    metadata.update(overrides)
    return metadata


# ---------------------------------------------------------------------------
# 1 — the whitelist
# ---------------------------------------------------------------------------


def test_the_whitelist_is_exactly_the_three_runtime_fields():
    assert EXECUTION_METADATA_FIELDS == (
        "question",
        "repository_scope",
        "commit_scope",
    )


# ---------------------------------------------------------------------------
# 2 — extraction
# ---------------------------------------------------------------------------


def test_extraction_reads_the_requests_own_labeled_lines():
    found = execution_metadata_from_request(REQUEST)

    assert found == {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
    }


def test_extraction_ignores_a_request_without_the_labels():
    plain = "How do the modules interact?"

    assert execution_metadata_from_request(plain) == {}
    assert execution_metadata_from_request("") == {}
    assert execution_metadata_from_request(None) == {}


def test_extraction_takes_line_start_labels_only_and_skips_empty():
    request = (
        "  question: indented, not the assignment line\n"
        "objective: a step objective is not a question\n"
        "question:\n"
        "question: {question}\n"
        "repository_scope:   \n"
        "repository_scope: {repository}\n"
        "commit_scope: {commit}\n"
        "later question: trailing text is not a label line".format(
            question=QUESTION, repository=REPOSITORY, commit=COMMIT
        )
    )

    found = execution_metadata_from_request(request)

    # first NON-EMPTY line-start occurrence of each field wins
    assert found["question"] == QUESTION
    assert found["repository_scope"] == REPOSITORY
    assert found["commit_scope"] == COMMIT


def test_extraction_stops_the_value_at_the_end_of_its_own_line():
    request = (
        "question: {question}\n"
        "repository_scope: {repository}\n"
        "commit_scope: {commit}\n"
        "Report findings backed only by evidence you inspected.".format(
            question=QUESTION, repository=REPOSITORY, commit=COMMIT
        )
    )

    found = execution_metadata_from_request(request)

    assert found["question"] == QUESTION
    assert found["commit_scope"] == COMMIT


# ---------------------------------------------------------------------------
# 3 — injection
# ---------------------------------------------------------------------------


def test_injection_fills_missing_whitelisted_fields_in_order():
    payload = {"summary": "observed interaction"}
    metadata = _metadata()

    filled, injected = inject_execution_metadata(
        payload, metadata, ArchitectureReport
    )

    assert injected == (
        "question",
        "repository_scope",
        "commit_scope",
    )
    assert filled["question"] == QUESTION
    assert filled["repository_scope"] == REPOSITORY
    assert filled["commit_scope"] == COMMIT
    # everything else — the model's own content — is untouched
    assert filled["summary"] == "observed interaction"


def test_injection_never_rewrites_a_value_the_model_provided():
    payload = {"question": "A question the model stated itself"}
    metadata = _metadata()

    filled, injected = inject_execution_metadata(
        payload, metadata, ArchitectureReport
    )

    assert injected == ("repository_scope", "commit_scope")
    assert filled["question"] == "A question the model stated itself"
    assert filled["repository_scope"] == REPOSITORY


def test_injection_only_sets_fields_the_contract_declares():
    metadata = _metadata()
    payload = {"name": "researcher"}

    filled, injected = inject_execution_metadata(
        payload, metadata, ArchitectureComponent
    )

    # ArchitectureComponent declares none of the three fields
    assert injected == ()
    assert filled is payload


def test_injection_keeps_a_present_non_string_value_for_the_contract():
    payload = {"question": 42}
    metadata = _metadata()

    filled, injected = inject_execution_metadata(
        payload, metadata, ArchitectureReport
    )

    # the model's own (wrong-shaped) data stays visible to validation
    assert injected == ("repository_scope", "commit_scope")
    assert filled["question"] == 42


def test_injection_without_anything_to_fill_returns_the_same_object():
    payload = {
        "question": QUESTION,
        "repository_scope": REPOSITORY,
        "commit_scope": COMMIT,
        "summary": "already complete",
    }

    filled, injected = inject_execution_metadata(
        payload, _metadata(), ArchitectureReport
    )

    assert filled is payload
    assert injected == ()


def test_injection_leaves_payloads_it_does_not_understand_alone():
    assert inject_execution_metadata(
        ["not", "a", "payload"], _metadata(), ArchitectureReport
    ) == (["not", "a", "payload"], ())
    assert inject_execution_metadata(
        {"summary": "x"}, {}, ArchitectureReport
    ) == ({"summary": "x"}, ())


# ---------------------------------------------------------------------------
# 4 — the request is the source of truth
# ---------------------------------------------------------------------------


def test_extracted_values_round_trip_into_the_payload():
    metadata = execution_metadata_from_request(REQUEST)

    filled, injected = inject_execution_metadata(
        {"summary": ""}, metadata, ArchitectureReport
    )

    typed = ArchitectureReport(**filled)
    assert injected == EXECUTION_METADATA_FIELDS
    assert typed.question == QUESTION
    assert typed.repository_scope == REPOSITORY
    assert typed.commit_scope == COMMIT
    # the derived handoff id is built from exactly those values
    assert typed.derived_investigation_id().startswith("inv-arch-")
