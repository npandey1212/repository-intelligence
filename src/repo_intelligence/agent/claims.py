"""Structured final-response contract (M8.8 integration).

The Agent's final response is normally free-form text. When a run
requests structured claims (``AgentExecutionConfig.structured_claims``)
the model is asked for ONE explicit JSON object — an output contract,
not prose extraction:

    {"answer": "human-readable answer",
     "claims": [{"statement": "...",
                 "evidence_refs": [{"path": "...",
                                    "line_start": 1, "line_end": 20}]}]}

``parse_structured_answer`` decodes that contract into
``(answer, Tuple[Claim, ...])``. Two hard rules:

* it NEVER invents evidence — a claim and its references are taken
  verbatim from the model's JSON (tolerating only malformed shapes by
  dropping them); whether the references actually resolve to gathered
  evidence is decided later, solely by the deterministic validator
  against this run's trace — the model is not the authority for
  "was this evidence gathered?";
* it NEVER fails the run — malformed JSON degrades to ``(raw text,
  no claims)``, and a payload whose ``answer`` field is unusable
  keeps the raw text as the answer while still returning its claims:
  the investigation's answer stands, and validation runs on whatever
  structured claims the response actually contained. The one
  tolerated slip is an early-closed object whose fields are still
  intact JSON (observed live: a stray ``}`` after ``answer``) — the
  parsing prefix and the verbatim ``claims`` array are then
  RECOVERED, never repaired or invented;

No hidden-reasoning fields exist in this contract (claims carry
``statement`` + ``evidence_refs`` only) and nothing here calls a
model, executes a tool, or reads the repository.

M13: a typed (M8.9) structured payload may carry the very same
``claims`` array inside its schema — ``claims_from_payload`` decodes
it after schema validation, so one claim contract serves both run
modes and the deterministic validator treats them identically.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..evidence import Claim, EvidenceRef

__all__ = [
    "STRUCTURED_ANSWER_INSTRUCTION",
    "claims_from_payload",
    "parse_structured_answer",
]

# Appended to SYSTEM_PROMPT only for structured-claims runs. It
# describes an output format — never a reasoning style (no chain-of-
# thought request, no hidden fields).
STRUCTURED_ANSWER_INSTRUCTION = (
    "When the investigation is complete, your FINAL response must be "
    "EXACTLY one JSON object and nothing else (no markdown fence, no "
    'prose before or after): {"answer": "...", "claims": [{"statement": '
    '"...", "evidence_refs": [{"path": "...", "line_start": 1, '
    '"line_end": 20}]}]}. The "answer" value is the human-readable '
    "answer to the user. Each claim states one finding of this "
    "investigation and cites the repository files you actually "
    "inspected during this run as evidence_refs: a repository-relative "
    "path, plus line_start AND line_end when you know the line range "
    "you read (omit both when you do not). Only claim what this run's "
    "observations support — claims must not exceed the inspected "
    "evidence, and represent uncertainty as uncertainty rather than "
    "inventing it away."
)


def _strip_code_fence(text: str) -> str:
    """Drop one surrounding ``` fence (models often add one anyway)."""
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    if newline < 0:
        return text
    body = text[newline + 1 :]
    end = body.rfind("```")
    if end < 0:
        return text
    return body[:end].strip()


def _scan_string(text: str, index: int) -> int:
    """Index just past the JSON string whose opening quote is at
    *index* (string-aware: ``\\"`` does not terminate it), or ``-1``."""
    i = index + 1
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == '"':
            return i + 1
        i += 1
    return -1


def _match_bracket(text: str, start: int) -> int:
    """Index of the bracket matching the one at *start*, or ``-1``.

    Tracks ``[]``/``{}`` depth while skipping string contents, so a
    bracket inside a string never confuses the match.
    """
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == '"':
            i = _scan_string(text, i)
            if i < 0:
                return -1
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _recovered_payload(text: str) -> Optional[Dict[str, Any]]:
    """Verbatim recovery of intact fields after a strict parse failed.

    Real-model runs (gpt-4o-mini) have produced one recurring slip: a
    stray ``}`` closes the top-level object early, leaving the payload
    split into an object prefix and its intact ``"claims"`` array as
    trailing data. Recovery is strictly read-only and deterministic:

    * the longest object PREFIX that parses (``raw_decode`` from the
      first ``{``) supplies fields such as ``answer``, verbatim;
    * the top-level ``"claims"`` key (found by a string-aware scan, so
      the word inside an answer string never counts) contributes its
      array only if the bracket-matched slice parses as JSON.

    Nothing is repaired, reordered, or invented — an array that does
    not parse (a stray bracket INSIDE the claims) stays dropped and
    the payload degrades exactly as before.
    """
    payload: Dict[str, Any] = {}
    start = text.find("{")
    if start >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text, start)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    if isinstance(payload.get("claims"), list):
        return payload
    # locate a top-level-ish `"claims": [` outside any string
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            end = _scan_string(text, index)
            if end < 0:
                break
            if text[index + 1 : end - 1] == "claims":
                cursor = end
                while cursor < len(text) and text[cursor] in " \t\r\n":
                    cursor += 1
                if cursor < len(text) and text[cursor] == ":":
                    cursor += 1
                    while cursor < len(text) and text[cursor] in " \t\r\n":
                        cursor += 1
                    if cursor < len(text) and text[cursor] == "[":
                        close = _match_bracket(text, cursor)
                        if close >= 0:
                            try:
                                claims = json.loads(text[cursor : close + 1])
                            except ValueError:
                                claims = None
                            if isinstance(claims, list):
                                payload["claims"] = claims
                                return payload
            index = end
            continue
        index += 1
    return payload or None


def _decode_payload(text: str) -> Optional[Dict[str, Any]]:
    """Decode one JSON object from *text*, or ``None``.

    Deterministic candidates only: the whole text, then the substring
    from the first ``{`` to the last ``}`` (a model that added a
    sentence around its JSON), then the verbatim field recovery for
    the observed early-close slip. Anything else → ``None``.
    """
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return _recovered_payload(text)


def _line_bound(value: Any) -> Optional[int]:
    """Accept only positive integers (never bools); anything else → ``None``.

    Sanitized here so the validator always sees ``int``/``None`` and
    can apply its own deterministic rules (a single surviving bound is
    insufficient metadata → ``UNKNOWN``; two valid bounds → coverage
    check). The parser never invents or repairs line numbers.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _ref_from(data: Mapping[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        path=data.get("path", ""),
        line_start=_line_bound(data.get("line_start")),
        line_end=_line_bound(data.get("line_end")),
    )


def _claims_from(raw: Any) -> Tuple[Claim, ...]:
    """Build claims from the JSON ``claims`` value, dropping entries
    whose shape is unusable (non-object, missing/non-string statement).
    References that are not objects are dropped; everything else is
    passed through verbatim for the validator to judge."""
    if not isinstance(raw, list):
        return ()
    claims: List[Claim] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            continue
        refs_raw = item.get("evidence_refs")
        if isinstance(refs_raw, list):
            refs = tuple(
                _ref_from(entry)
                for entry in refs_raw
                if isinstance(entry, Mapping)
            )
        else:
            refs = ()
        claims.append(Claim(statement=statement, evidence_refs=refs))
    return tuple(claims)


def claims_from_payload(payload: Any) -> Tuple[Claim, ...]:
    """Claims from an ALREADY-DECODED structured payload (M13).

    The M8.9 typed contract may carry the same ``claims`` array as the
    M8.8 text contract: once the application has validated the payload
    against its schema, the claims are taken verbatim under exactly
    the ``_claims_from`` rules (malformed entries dropped, nothing
    invented, never a run failure). A payload without a usable
    ``claims`` list yields ``()`` — a typed run whose contract
    declares no claims behaves exactly as before M13.
    """
    if not isinstance(payload, Mapping):
        return ()
    return _claims_from(payload.get("claims"))


def parse_structured_answer(content: str) -> Tuple[str, Tuple[Claim, ...]]:
    """Decode the explicit structured final response → ``(answer, claims)``.

    Tolerant by contract: malformed JSON or an unexpected shape never
    raises — the raw text stays the human-readable answer and the
    claim list is empty (the run produced no structured claims, which
    is observable downstream: an empty validation block), except for
    the one tolerated slip — an early-closed object with an intact
    ``claims`` array — whose fields are recovered verbatim (see
    ``_recovered_payload``). When the payload DOES contain a usable ``claims`` list it is returned even
    if the ``answer`` field is missing or not a string: the claims are
    what M8.8 validation exists to check, and dropping them here would
    silently skip validation. Claims are taken verbatim from the
    model's JSON; this parser neither invents evidence references nor
    decides whether they resolve.
    """
    text = _strip_code_fence(content.strip())
    payload = _decode_payload(text)
    if payload is None:
        return content, ()
    answer = payload.get("answer")
    claims = _claims_from(payload.get("claims"))
    if not isinstance(answer, str):
        return content, claims
    return answer, claims
