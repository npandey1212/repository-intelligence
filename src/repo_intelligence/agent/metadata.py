"""Deterministic execution metadata on the structured path (M17).

The three fields that anchor a typed report to its request —
``question``, ``repository_scope``, ``commit_scope`` — are EXECUTION
STATE: they say which investigation this report belongs to. They are
never model conclusions, so they never depend on whether the model
chose to echo them. This module is the one place that reads them from
the run's own request text (the application-built prompt every
specialist received) and fills them into the model's payload before
validation.

The rules, in full:

* exactly three whitelisted fields (``EXECUTION_METADATA_FIELDS``) —
  never a summary, component, finding, evidence reference, or any
  other substantive value;
* only fields the contract itself declares (a contract that does not
  know ``question`` never receives it);
* only where the payload left the field missing or empty — a value the
  model DID provide is never rewritten;
* line-start ``label: value`` occurrences, first non-empty one per
  field, single-line values (every question builder in this codebase
  composes them that way);
* nothing here validates: the unchanged Pydantic contract runs after
  injection, so substantive problems still fail exactly as before.

Source of truth is the request text, which the runtime composed from
the actual investigation state (direct-mode TASK block, or the
runner's question) — so an injected value always equals the
request/execution state by construction.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "EXECUTION_METADATA_FIELDS",
    "execution_metadata_from_request",
    "inject_execution_metadata",
]

# The complete whitelist. Order is stable so the injection record on
# an iteration is deterministic.
EXECUTION_METADATA_FIELDS: Tuple[str, ...] = (
    "question",
    "repository_scope",
    "commit_scope",
)

_METADATA_LINE = re.compile(
    r"^({labels}):[ \t]*(.*)$".format(
        labels="|".join(EXECUTION_METADATA_FIELDS)
    )
)


def execution_metadata_from_request(request: Any) -> Dict[str, str]:
    """Read the request's own ``label: value`` assignment lines.

    Returns only fields that appear at the start of a line with a
    non-empty value (first occurrence wins); a request without the
    labels — a plain question, another specialist's prompt — yields an
    empty dict and changes nothing downstream.
    """
    if not isinstance(request, str) or not request:
        return {}
    found: Dict[str, str] = {}
    for line in request.splitlines():
        match = _METADATA_LINE.match(line)
        if match is None:
            continue
        label = match.group(1)
        if label in found:
            continue
        value = match.group(2).strip()
        if value:
            found[label] = value
    return found


def inject_execution_metadata(
    payload: Any,
    metadata: Mapping[str, str],
    model: Any,
) -> Tuple[Any, Tuple[str, ...]]:
    """Fill missing whitelisted metadata into a parsed model payload.

    Returns ``(payload, injected)`` where ``injected`` lists exactly
    the fields that were set, in ``EXECUTION_METADATA_FIELDS`` order.
    When nothing was filled the payload object itself comes back
    unchanged (byte-for-byte pre-injection behavior), so an
    un-injected run's records stay identical to what they were.

    ``model`` gates the fill to the contract's own declared fields —
    metadata never lands on a payload whose contract does not know it.
    """
    if not isinstance(payload, dict) or not metadata:
        return payload, ()
    declared = getattr(model, "model_fields", None) or getattr(
        model, "__fields__", None
    )
    if not isinstance(declared, dict):
        return payload, ()
    filled: Tuple[str, ...] = ()
    updated: Any = None
    for label in EXECUTION_METADATA_FIELDS:
        value = metadata.get(label)
        if not value or label not in declared:
            continue
        current = payload.get(label)
        if isinstance(current, str) and current.strip():
            # The model provided it — never rewritten.
            continue
        if current is not None and not isinstance(current, str):
            # A present non-string value is the model's own (wrong)
            # data: the contract's type errors must stay visible.
            continue
        if updated is None:
            updated = dict(payload)
        updated[label] = value
        filled = filled + (label,)
    if updated is None:
        return payload, ()
    return updated, filled
