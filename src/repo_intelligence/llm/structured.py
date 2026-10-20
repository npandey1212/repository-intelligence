"""Application-side structured output contract (M8.9).

Three validation layers exist for a structured LLM response, and this
module is exactly the middle one — they never overlap:

    provider level      the provider is *asked* for JSON matching a
                        schema (translated from ``StructuredOutputSpec``
                        by the provider adapter, llm/openai_client.py)
        │
        ▼
    application level   THIS MODULE: decode already happened in the
                        adapter; the application validates the decoded
                        payload against its own Pydantic model and
                        receives a typed result — or an explicit
                        failure, never a silent default or repair.
                        One deterministic normalization step sits in
                        front of validation (M17 finalization fix):
                        when the model hands back its object one level
                        DEEPER than requested (an envelope or an
                        echoed schema wrapper), the application picks
                        the level that actually validates — and only
                        that level; anything else is passed through
                        unchanged so the failure below stays explicit
        │
        ▼
    domain level        domain-specific checks against application
                        data (outside this package's scope)

Provider neutrality: this module imports only Pydantic and the
provider-neutral models — never a provider SDK. Any ``LLMClient``
(fakes included) that returns ``LLMResponse.structured_output`` feeds
the exact same validation path.

Pydantic v2 API (``model_validate``/``model_json_schema``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple, Type

from pydantic import BaseModel, ValidationError

from .models import StructuredOutputSpec

__all__ = [
    "NORMALIZATION_MAX_DEPTH",
    "STRUCTURED_ENVELOPE_KEYS",
    "StructuredOutputValidationError",
    "normalize_structured_payload",
    "spec_for_model",
    "validate_structured",
]


class StructuredOutputValidationError(Exception):
    """Explicit structured-output validation failure (M8.9).

    Raised instead of injecting defaults, dropping the payload, or
    "repairing" the model's answer: a payload that does not match the
    requested schema is a failure the caller must see. ``errors``
    carries the individual field failures (Pydantic error dicts,
    JSON-ready) when the failure came from schema validation, and is
    empty for a payload that never arrived.
    """

    def __init__(
        self, message: str, errors: Iterable[Dict[str, Any]] = ()
    ) -> None:
        super().__init__(message)
        self.errors: Tuple[Dict[str, Any], ...] = tuple(errors)


def spec_for_model(model_type: Type[BaseModel]) -> StructuredOutputSpec:
    """Build the request spec for a Pydantic model (its JSON Schema).

    Raises ``TypeError`` for anything that is not a ``BaseModel``
    subclass — the contract is derived from the application's own type,
    never hand-assembled at the call site.
    """
    if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
        raise TypeError(
            "structured output model must be a pydantic BaseModel "
            "subclass, got {!r}".format(model_type)
        )
    return StructuredOutputSpec(
        name=model_type.__name__,
        schema=dict(model_type.model_json_schema()),
    )


# -- payload normalization (deterministic level selection, M17) -----------
#
# Observed live shape (the M17 finalization failure this fixes): the
# model answered one level deeper than the application expected —
#
#     {"structured_output": {"title": ..., "type": "object",
#                             "properties": {...}}}
#
# — so ``parse_obj`` saw a wrapper object and reported the model's own
# required fields as missing. Which level the model chose is provider
# behavior the application does not control; selecting the level that
# validates IS the application's job. The rules below are the whole
# mechanism — bounded, ordered, and provably non-destructive:
#
#   1. if the payload itself validates, it is returned untouched
#   2. otherwise a bounded breadth-first descent (never deeper than
#      ``NORMALIZATION_MAX_DEPTH`` levels below the payload) looks for
#      exactly one child that validates: a known envelope key, the
#      ``properties`` object of an echoed JSON-Schema wrapper, or a
#      single-key wrapper
#   3. if nothing validates, the ORIGINAL payload is returned and the
#      caller's own validation fails with its original per-field
#      errors — no field is ever invented, defaulted, coerced, or
#      fabricated, and a payload that never matched keeps failing
#      exactly as it did before this module learned to look deeper
#
# Descending is not "repair": the application is choosing WHICH object
# the model sent, not editing what the model wrote.
STRUCTURED_ENVELOPE_KEYS = (
    "structured_output",
    "output",
    "result",
    "payload",
    "data",
)
# Root is level 0; children are tried down to this level. Three covers
# envelope → schema wrapper → instance (and one more wrapper beyond)
# while keeping pathological nesting irrelevant to correctness.
NORMALIZATION_MAX_DEPTH = 3


def _payload_validates(model_type: Type[BaseModel], payload: Any) -> bool:
    """True only when *payload* fully validates against *model_type*."""
    try:
        model_type.model_validate(payload)
    except ValidationError:
        return False
    return True


def _schema_wrapper(node: Dict[str, Any], model_fields: Set[str]) -> bool:
    """True when *node* looks like an echoed JSON-Schema object.

    Detected structurally (``type``/``title``/``$schema`` plus a dict
    of ``properties``), never by matching a particular model's name,
    and never when the model itself declares a ``properties`` field.
    """
    if "properties" in model_fields:
        return False
    if not isinstance(node.get("properties"), dict):
        return False
    return bool(
        node.get("type") == "object"
        or isinstance(node.get("title"), str)
        or "$schema" in node
    )


def _candidate_children(
    node: Dict[str, Any], model_fields: Set[str]
) -> Iterator[Dict[str, Any]]:
    """Yield the child objects worth trying, in a fixed order.

    Envelope keys first (the shape that was actually observed), then
    the ``properties`` object of a schema wrapper, then a single-key
    wrapper. A key that is itself a field of the target model is never
    descended into — that key holds data, not another envelope.
    """
    for key in STRUCTURED_ENVELOPE_KEYS:
        if key in model_fields:
            continue
        value = node.get(key)
        if isinstance(value, dict):
            yield value
    if _schema_wrapper(node, model_fields):
        yield node["properties"]
    if len(node) == 1:
        key = next(iter(node))
        value = node[key]
        if key not in model_fields and isinstance(value, dict):
            yield value


def normalize_structured_payload(
    model_type: Type[BaseModel], payload: Any
) -> Any:
    """Select the payload level that validates against *model_type*.

    Deterministic and provider-neutral: returns *payload* unchanged
    when it already validates (the overwhelmingly common case), the
    one child object that validates when the model wrapped its answer
    (envelope / echoed schema / single-key wrapper, bounded by
    ``NORMALIZATION_MAX_DEPTH``), and the ORIGINAL payload when
    nothing validates — so the caller's explicit failure keeps naming
    the payload the model actually sent.

    Never edits a payload: no field is added, removed, renamed, or
    coerced here, and only whole objects are ever compared.
    """
    if not isinstance(payload, dict):
        return payload
    if not isinstance(model_type, type) or not issubclass(
        model_type, BaseModel
    ):
        return payload
    if _payload_validates(model_type, payload):
        return payload

    model_fields: Set[str] = set(model_type.model_fields)
    seen: Set[int] = {id(payload)}
    frontier: List[Tuple[Dict[str, Any], int]] = [(payload, 0)]
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= NORMALIZATION_MAX_DEPTH:
            continue
        for child in _candidate_children(node, model_fields):
            if id(child) in seen:
                continue
            seen.add(id(child))
            if _payload_validates(model_type, child):
                return child
            frontier.append((child, depth + 1))
    return payload


def validate_structured(
    model_type: Type[BaseModel], payload: Any
) -> BaseModel:
    """Validate *payload* against *model_type* → typed instance.

    Explicit failure contract (no silent behavior of any kind):
        payload is not a JSON object → ``StructuredOutputValidationError``
        object does not match the schema (missing field, wrong type,
        any other Pydantic error) → ``StructuredOutputValidationError``
        carrying the per-field errors

    Before validation the payload goes through
    ``normalize_structured_payload``: a model that answered one level
    deeper than requested is validated at the level it actually wrote,
    while a payload nothing validates is passed through untouched —
    so every failure below still reports the original payload, with
    its original per-field errors.

    Pydantic's model configuration is used exactly as defined by the
    application's model — this function never coerces, drops, or
    injects anything beyond what the model type itself declares.
    """
    if not isinstance(payload, dict):
        raise StructuredOutputValidationError(
            "structured payload must be a JSON object, got {}".format(
                type(payload).__name__
            )
        )
    candidate = normalize_structured_payload(model_type, payload)
    try:
        return model_type.model_validate(candidate)
    except ValidationError as exc:
        errors = tuple(dict(item) for item in exc.errors())
        raise StructuredOutputValidationError(
            "structured output does not match model {}: {}".format(
                model_type.__name__, str(exc)
            ),
            errors,
        ) from exc
