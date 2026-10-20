"""M8.9 — provider-neutral structured output (llm layer).

Covers the application-side validation contract (typed result or
explicit ``StructuredOutputValidationError`` — never silent defaults),
the deterministic envelope/schema level selection in front of it
(``normalize_structured_payload``, M17 finalization fix), and the
OpenAI adapter's translation of ``StructuredOutputSpec`` to its
``response_format`` parameter. Provider fakes only; no real API calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from repo_intelligence.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    StructuredOutputSpec,
    StructuredOutputValidationError,
    normalize_structured_payload,
    spec_for_model,
    validate_structured,
)
from repo_intelligence.llm.openai_client import DEFAULT_MODEL, OpenAILLMClient
from repo_intelligence.llm.structured import NORMALIZATION_MAX_DEPTH


# ---------------------------------------------------------------------------
# Per-file test doubles (no conftest in this repo)
# ---------------------------------------------------------------------------


class TicketSummary(BaseModel):
    """Generic example model — repo-agnostic on purpose."""

    title: str
    severity: int


class _FakeProvider:
    def __init__(self, response=None):
        self.calls = []
        self._response = response
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _raw_response(content=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _client(response=None):
    provider = _FakeProvider(response=response)
    return OpenAILLMClient(provider_client=provider), provider


# ---------------------------------------------------------------------------
# spec_for_model — Pydantic model → provider-neutral request spec
# ---------------------------------------------------------------------------


def test_spec_is_derived_from_the_pydantic_model():
    spec = spec_for_model(TicketSummary)
    assert isinstance(spec, StructuredOutputSpec)
    assert spec.name == "TicketSummary"
    assert spec.schema["type"] == "object"
    assert set(spec.schema["properties"]) == {"title", "severity"}
    assert sorted(spec.schema["required"]) == ["severity", "title"]


def test_spec_rejects_anything_that_is_not_a_model():
    with pytest.raises(TypeError):
        spec_for_model(dict)
    with pytest.raises(TypeError):
        spec_for_model(TicketSummary(title="x", severity=1))  # instance


# ---------------------------------------------------------------------------
# validate_structured — typed result or explicit failure
# ---------------------------------------------------------------------------


def test_valid_payload_yields_the_typed_result():
    result = validate_structured(
        TicketSummary, {"title": "login fails", "severity": 2}
    )
    assert isinstance(result, TicketSummary)
    assert result.title == "login fails"
    assert result.severity == 2


def test_missing_field_is_an_explicit_failure_without_a_default():
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        validate_structured(TicketSummary, {"severity": 2})
    message = str(excinfo.value)
    assert "severity" in message or "title" in message  # names the field
    assert excinfo.value.errors  # per-field errors are carried
    # nothing was defaulted or repaired: the failure is raised, not papered over
    assert not isinstance(excinfo.value, TicketSummary)


def test_wrong_type_is_an_explicit_failure():
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        validate_structured(
            TicketSummary, {"title": "login fails", "severity": "high"}
        )
    assert excinfo.value.errors
    error_types = [item.get("type") for item in excinfo.value.errors]
    assert error_types  # the per-field failure detail travels with it


def test_non_object_payload_is_an_explicit_failure():
    for payload in (None, "just text", 42, ["a", "list"]):
        with pytest.raises(StructuredOutputValidationError):
            validate_structured(TicketSummary, payload)


def test_extra_fields_follow_the_models_own_pydantic_config():
    # Pydantic v1's default model config (Extra.ignore) DROPS unknown
    # keys — this is the model's documented behavior, not silent repair
    # by this module: the declared fields still must all be valid.
    result = validate_structured(
        TicketSummary,
        {"title": "login fails", "severity": 2, "unmodeled": "ignored"},
    )
    assert result.title == "login fails"
    assert "unmodeled" not in result.dict()
    # A model that forbids extras rejects them itself — same explicit
    # failure path, driven entirely by the application's own config.
    from pydantic import Extra

    class StrictTicket(BaseModel):
        title: str

        class Config:
            extra = Extra.forbid

    with pytest.raises(StructuredOutputValidationError):
        validate_structured(StrictTicket, {"title": "x", "unmodeled": "y"})


# ---------------------------------------------------------------------------
# normalize_structured_payload — deterministic envelope/schema selection
# ---------------------------------------------------------------------------


def _ticket():
    return {"title": "login fails", "severity": 2}


def test_normalize_returns_a_payload_that_already_validates_unchanged():
    payload = _ticket()
    assert normalize_structured_payload(TicketSummary, payload) is payload


def test_normalize_descends_into_a_known_envelope_key():
    payload = {"structured_output": _ticket()}
    selected = normalize_structured_payload(TicketSummary, payload)
    assert selected is payload["structured_output"]
    typed = validate_structured(TicketSummary, payload)
    assert typed.title == "login fails"


def test_normalize_descends_into_an_echoed_schema_wrapper():
    # The model answered with the schema it was given, instance data
    # written under ``properties`` — the shape that failed live.
    payload = {
        "structured_output": {
            "title": "TicketSummary",
            "type": "object",
            "properties": _ticket(),
        }
    }
    selected = normalize_structured_payload(TicketSummary, payload)
    assert selected == _ticket()
    typed = validate_structured(TicketSummary, payload)
    assert isinstance(typed, TicketSummary)


def test_normalize_descends_into_a_single_key_wrapper():
    payload = {"report": _ticket()}
    assert normalize_structured_payload(
        TicketSummary, payload
    ) == _ticket()


def test_normalize_stops_at_the_depth_bound():
    deep = _ticket()
    for _ in range(NORMALIZATION_MAX_DEPTH + 1):
        deep = {"output": deep}
    # One level past the bound: the instance exists but is never
    # reached, so the original payload comes back untouched.
    assert normalize_structured_payload(TicketSummary, deep) is deep
    with pytest.raises(StructuredOutputValidationError):
        validate_structured(TicketSummary, deep)


def test_normalize_never_modifies_a_payload_it_cannot_validate():
    # The failure contract is unchanged: what the model sent is what
    # gets reported, with the same per-field errors as before.
    payload = {"severity": 2, "wrapper": {"title": "nested"}}
    selected = normalize_structured_payload(TicketSummary, payload)
    assert selected is payload
    with pytest.raises(StructuredOutputValidationError) as excinfo:
        validate_structured(TicketSummary, payload)
    assert excinfo.value.errors
    missing = [item.get("loc") for item in excinfo.value.errors]
    assert ("title",) in [tuple(loc or ()) for loc in missing]


def test_normalize_passes_non_object_payloads_through():
    for payload in (None, "text", 42, ["list"]):
        assert normalize_structured_payload(TicketSummary, payload) is payload


def test_normalize_skips_envelope_keys_that_are_model_fields():
    class Enveloped(BaseModel):
        output: dict
        severity: int

    payload = {
        "output": {"output": {"severity": 1}, "severity": 1},
        "severity": "bad",
    }
    # ``output`` is a FIELD of this model — it holds data, so it is
    # never descended into even though it is also an envelope key.
    assert normalize_structured_payload(Enveloped, payload) is payload


# ---------------------------------------------------------------------------
# OpenAI adapter — spec reaches the provider, payload comes back decoded
# ---------------------------------------------------------------------------


def test_spec_translates_to_provider_response_format():
    spec = spec_for_model(TicketSummary)
    client, provider = _client(response=_raw_response(content="{}"))
    client.complete(
        LLMRequest(messages=[Message(role="user", content="hi")],
                   structured_output=spec)
    )
    response_format = provider.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "TicketSummary"
    assert response_format["json_schema"]["schema"] == spec.schema


def test_no_response_format_without_a_spec():
    client, provider = _client(response=_raw_response(content="plain"))
    client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert "response_format" not in provider.calls[0]


def test_structured_response_content_is_decoded_to_a_payload():
    spec = spec_for_model(TicketSummary)
    client, _ = _client(
        response=_raw_response(content='{"title": "login fails", "severity": 2}')
    )
    response = client.complete(
        LLMRequest(messages=[Message(role="user", content="hi")],
                   structured_output=spec)
    )
    assert response.structured_output == {"title": "login fails", "severity": 2}
    assert response.message.content is not None  # raw text is never discarded


def test_non_json_content_under_a_spec_leaves_the_payload_unset():
    spec = spec_for_model(TicketSummary)
    client, _ = _client(response=_raw_response(content="I refuse to answer JSON."))
    response = client.complete(
        LLMRequest(messages=[Message(role="user", content="hi")],
                   structured_output=spec)
    )
    assert response.structured_output is None  # explicit validation comes later
    assert response.message.content == "I refuse to answer JSON."


def test_json_array_content_is_not_an_object_payload():
    spec = spec_for_model(TicketSummary)
    client, _ = _client(response=_raw_response(content='["not", "an", "object"]'))
    response = client.complete(
        LLMRequest(messages=[Message(role="user", content="hi")],
                   structured_output=spec)
    )
    assert response.structured_output is None


def test_plain_response_never_carries_a_payload():
    # Even with a spec in the request, decoding only happens for string
    # content; the response model's default stays provider-neutral None.
    client, _ = _client(response=_raw_response(content=None))
    response = client.complete(
        LLMRequest(messages=[Message(role="user", content="hi")],
                   structured_output=spec_for_model(TicketSummary))
    )
    assert response.structured_output is None
    default = LLMResponse(message=Message(role="assistant", content="hi"))
    assert default.structured_output is None
    assert DEFAULT_MODEL  # sanity: module constants untouched
