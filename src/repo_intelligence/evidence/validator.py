"""Deterministic evidence-traceability validation (M8.8, M13 eligibility).

One flow, no LLM, no network, no second agent:

    Claim
      ↓
    EvidenceRef (path + optional lines)
      ↓  was this gathered during the run?
    GatheredEvidence (from successful ToolResults in the trace)
      ↓
    SUPPORTED / UNSUPPORTED / UNKNOWN

Evidence source — reused, never rebuilt
---------------------------------------

``gather_evidence`` reads the execution data an Agent run already
produced (an ``AgentTrace`` or its ``to_dict()`` JSON mapping) and
derives the gathered-evidence set from **successful** tool results:

* a successful result whose ``data`` carries a repository ``path``
  (``read_file``) → one evidence item: path + reported line range,
  marked **inspected** (the content was actually read);
* a successful ``search_repository`` result (``data.results`` list of
  ``{path, line_start, line_end, ...}``) → one evidence item per
  match, marked **not inspected** — a discovery that names a
  candidate file, never an inspection of it;
* anything else (e.g. ``get_architecture_context``, which returns
  ``{"context": ...}`` with no file address) is not file-addressable
  and is skipped — it still lives in the trace, it just cannot be
  *referenced by path*;
* a **failed** result never becomes gathered evidence; it is recorded
  as a failed attempt (path + error code) purely so reasons can say
  ``FILE_NOT_FOUND`` instead of an unexplained miss.

Deterministic matching rules (explicit)
---------------------------------------

1. **Path normalization.** Claim-reference paths and gathered paths
   are normalized with the existing repository path-safety helper
   (``normalize_repository_relative_path``): backslashes → ``/``,
   empty/``.`` segments dropped, ``./x`` ≡ ``x``. Absolute paths
   (POSIX or drive-letter) and ``..`` escapes are rejected — a claim
   can never point outside the repository boundary, so arbitrary
   absolute filesystem paths cannot bypass it.

2. **Reference structure.** A line range requires *both* bounds
   (``line_start``/``line_end``), each ≥ 1, with
   ``line_start ≤ line_end``. Exactly one bound (or a non-positive
   bound) is insufficient metadata → the reference is not evaluable.

3. **Line coverage.** If the claim reference carries a line range,
   gathered evidence matches only when it *covers* that range:
   ``gathered.line_start ≤ claim.line_start`` and
   ``claim.line_end ≤ gathered.line_end``. Gathered ``1-50`` covers
   claim ``1-20``; gathered ``1-20`` does NOT cover claim ``40-50``.
   Gathered evidence without line bounds is treated as whole-file
   evidence (covers any range on that path). A path-only claim
   reference (no lines) matches any gathered evidence on that path.

4. **One reference is enough.** A claim with several references is
   SUPPORTED when at least one resolves to gathered evidence; the
   result records exactly which items matched.

Inspected-only eligibility (M13)
--------------------------------

``validate_claim(..., require_inspected=True)`` narrows the match
pool from ``EvidenceIndex.gathered`` to ``EvidenceIndex.inspected``:
a reference is SUPPORTED only when the run actually *read* the file
(e.g. ``read_file``). A path that merely appeared in search results,
graph metadata, the architecture context, or the planner context
never satisfies a reference — discovery is not inspection. The
unsupported reason then says exactly which of the three non-matches
occurred (failed attempt / discovered-but-not-inspected / never
seen), and every UNSUPPORTED result carries
``code = CLAIM_EVIDENCE_INVALID`` — the recorded structured failure
of the evidence contract (M13 §6). The default
(``require_inspected=False``) keeps the established gathered-
evidence semantics for direct callers.

Statuses
--------

``SUPPORTED``    ≥ 1 reference matched successfully gathered evidence
                 (inspected evidence, when ``require_inspected``).
``UNSUPPORTED``  references exist and are valid, but none matched —
                 recorded with ``code = CLAIM_EVIDENCE_INVALID``.
``UNKNOWN``      no references, or all references structurally
                 insufficient — the claim cannot be evaluated
                 deterministically (never silently "supported").

Scope: **traceability, not truth.** This module never inspects the
semantics of ``statement`` — it cannot and must not decide whether a
claim is *correct*, complete, or exhaustive. It does not modify the
Agent or its loop: validation is an independent provenance layer fed
by the run's trace. (Since the M8.8 integration, ``Agent.run_result``
calls it exactly once, AFTER the investigation — these functions
still only read the finished trace and never dispatch a tool.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..tools.search_repository import normalize_repository_relative_path
from .models import (
    CLAIM_EVIDENCE_INVALID,
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    Claim,
    ClaimValidation,
    EvidenceIndex,
    EvidenceRef,
    FailedEvidenceAttempt,
    GatheredEvidence,
)

__all__ = [
    "gather_evidence",
    "validate_claim",
    "validate_claims",
]

_SEARCH_RESULTS_KEY = "results"
_MAX_MATCH_ITEMS = 1000  # defensive cap; search results are already capped


# ---------------------------------------------------------------------------
# Evidence extraction (from execution data the run already produced)
# ---------------------------------------------------------------------------


def _as_mapping(item: Any) -> Mapping[str, Any]:
    """Accept a mapping or an object exposing ``to_dict()``."""
    if isinstance(item, Mapping):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError(
        "execution data items must be mappings or to_dict()-ables; "
        "got {!r}".format(type(item).__name__)
    )


def _iterations_of(execution: Any) -> Sequence[Mapping[str, Any]]:
    payload = _as_mapping(execution)
    iterations = payload.get("iterations")
    if iterations is None:
        raise ValueError(
            "execution data has no 'iterations' — not an AgentTrace or "
            "trace mapping"
        )
    return [_as_mapping(item) for item in iterations]


def _normalize_path(value: Any) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(value, str):
        return None, "path is not a string"
    return normalize_repository_relative_path(value)


def _error_code(error: Any) -> Optional[str]:
    if isinstance(error, Mapping):
        code = error.get("code")
        return code if isinstance(code, str) else None
    return None


def _failed_path(
    call: Mapping[str, Any], data: Any
) -> Optional[str]:
    """Best-effort path of a failed attempt (arguments, then data)."""
    arguments = call.get("arguments")
    if isinstance(arguments, Mapping):
        normalized, _ = _normalize_path(arguments.get("path"))
        if normalized is not None:
            return normalized
    if isinstance(data, Mapping):
        normalized, _ = _normalize_path(data.get("path"))
        if normalized is not None:
            return normalized
    return None


def _append_file_evidence(
    bucket: List[GatheredEvidence],
    data: Mapping[str, Any],
    tool: str,
    tool_call_id: Optional[str],
    source: Optional[Mapping[str, Any]],
    inspected: bool,
) -> None:
    normalized, _ = _normalize_path(data.get("path"))
    if normalized is None:
        return
    line_start = data.get("line_start")
    line_end = data.get("line_end")
    bucket.append(
        GatheredEvidence(
            path=normalized,
            line_start=line_start if isinstance(line_start, int) else None,
            line_end=line_end if isinstance(line_end, int) else None,
            tool=tool,
            tool_call_id=tool_call_id,
            source=dict(source) if isinstance(source, Mapping) else None,
            inspected=inspected,
        )
    )


def gather_evidence(execution: Any) -> EvidenceIndex:
    """Derive the run's file-addressable evidence from its execution data.

    *execution* may be an ``AgentTrace`` or the mapping produced by
    ``AgentTrace.to_dict()`` / ``runtime/agent_runs/<id>/trace.json``.
    Only successful tool results contribute to ``gathered``; failures
    are recorded separately (never positive evidence).
    """
    gathered: List[GatheredEvidence] = []
    failed: List[FailedEvidenceAttempt] = []
    for iteration in _iterations_of(execution):
        calls = {}
        for call in iteration.get("tool_calls") or []:
            if isinstance(call, Mapping) and call.get("id") is not None:
                calls[call.get("id")] = call
        for result in iteration.get("tool_results") or []:
            if not isinstance(result, Mapping):
                continue
            call = calls.get(result.get("tool_call_id")) or {}
            tool = result.get("tool") or call.get("name") or ""
            tool_call_id = result.get("tool_call_id")
            data = result.get("data")
            source = result.get("source")
            if not result.get("success"):
                failed.append(
                    FailedEvidenceAttempt(
                        path=_failed_path(call, data),
                        tool=tool,
                        tool_call_id=tool_call_id,
                        error_code=_error_code(result.get("error")),
                    )
                )
                continue
            if not isinstance(data, Mapping):
                continue
            if isinstance(data.get("path"), str):
                # read_file-style: one file (+ reported line range) —
                # content was returned, so the run INSPECTED it (M13)
                _append_file_evidence(
                    gathered, data, tool, tool_call_id, source,
                    inspected=True,
                )
            elif isinstance(data.get(_SEARCH_RESULTS_KEY), list):
                # search_repository-style: one item per match — a
                # discovery, not an inspection: the file was NAMED
                # here, its content was never read (M13 §4)
                for match in data.get(_SEARCH_RESULTS_KEY)[
                    :_MAX_MATCH_ITEMS
                ]:
                    if isinstance(match, Mapping):
                        _append_file_evidence(
                            gathered, match, tool, tool_call_id, source,
                            inspected=False,
                        )
            # else: not file-addressable (e.g. get_architecture_context)
    return EvidenceIndex(gathered=tuple(gathered), failed=tuple(failed))


# ---------------------------------------------------------------------------
# Claim validation
# ---------------------------------------------------------------------------


def _validated_ref(
    ref: EvidenceRef,
) -> Tuple[Optional[EvidenceRef], Optional[str]]:
    """Normalize one reference; return (valid_ref, error_message)."""
    normalized, error = _normalize_path(ref.path)
    if error is not None:
        return None, "{} ({})".format(ref.render(), error)
    if normalized is None:
        return None, "{} (empty path)".format(ref.render())
    start, end = ref.line_start, ref.line_end
    if start is None and end is None:
        return EvidenceRef(path=normalized), None
    if start is None or end is None:
        return None, "{} (line range needs both bounds)".format(ref.render())
    if start < 1 or end < 1:
        return None, "{} (line bounds must be >= 1)".format(ref.render())
    if end < start:
        return None, "{} (line_end must be >= line_start)".format(ref.render())
    return EvidenceRef(path=normalized, line_start=start, line_end=end), None


def _covers(gathered: GatheredEvidence, ref: EvidenceRef) -> bool:
    """Deterministic line coverage (rule 3 in the module docstring)."""
    if ref.line_start is None or ref.line_end is None:
        return True  # path-only reference: any gathered item on the path
    if gathered.line_start is None or gathered.line_end is None:
        return True  # whole-file observation: covers any range
    return (
        gathered.line_start <= ref.line_start
        and ref.line_end <= gathered.line_end
    )


def _match_key(item: GatheredEvidence) -> Tuple[Any, ...]:
    return (
        item.path,
        item.line_start,
        item.line_end,
        item.tool,
        item.tool_call_id,
    )


def _describe_matched(items: Sequence[GatheredEvidence]) -> str:
    return ", ".join(item.render() for item in items)


def _describe_unsupported(
    valid_refs: Sequence[EvidenceRef],
    evidence: EvidenceIndex,
    require_inspected: bool = False,
) -> str:
    failed_by_path: Dict[str, List[FailedEvidenceAttempt]] = {}
    for attempt in evidence.failed:
        if attempt.path is not None:
            failed_by_path.setdefault(attempt.path, []).append(attempt)
    details: List[str] = []
    for ref in valid_refs:
        attempts = failed_by_path.get(ref.path)
        if attempts:
            codes = sorted(
                {a.error_code for a in attempts if a.error_code}
            )
            details.append(
                "{} (attempted during the run and failed: {})".format(
                    ref.render(),
                    "/".join(codes) if codes else "tool failure",
                )
            )
            continue
        if require_inspected:
            # M13: distinguish the three non-matches precisely —
            # inspected but not covering the range, discovered but
            # never inspected, or never seen at all.
            path_items = [
                item
                for item in evidence.gathered
                if item.path == ref.path
            ]
            if any(item.inspected for item in path_items):
                details.append(
                    "{} (inspected during the run, but no observed "
                    "line range covers this reference)".format(
                        ref.render()
                    )
                )
                continue
            if path_items:
                details.append(
                    "{} (found during the run but never inspected — "
                    "search results are discoveries, not evidence)".format(
                        ref.render()
                    )
                )
                continue
        details.append("{} (never gathered)".format(ref.render()))
    return (
        "no gathered evidence matches: {}".format("; ".join(details))
    )


def validate_claim(
    claim: Claim,
    evidence: EvidenceIndex,
    require_inspected: bool = False,
) -> ClaimValidation:
    """Deterministically validate one claim against gathered evidence.

    Does NOT inspect the semantics of ``claim.statement`` — only
    whether its evidence references resolve to gathered, successful
    tool evidence from the run.

    ``require_inspected`` (M13) tightens the match to evidence the run
    actually *inspected* (``read_file``-style content reads):
    search discoveries, graph knowledge, and context never satisfy a
    reference. The default (``False``) keeps the established M8.8
    gathered-evidence semantics for direct callers; the Agent's
    post-run validation passes ``True``.
    """
    if not claim.evidence_refs:
        return ClaimValidation(
            claim=claim,
            status=STATUS_UNKNOWN,
            matched_evidence=(),
            reason=(
                "no evidence references supplied; traceability cannot "
                "be evaluated (not counted as supported)"
            ),
        )

    valid_refs: List[EvidenceRef] = []
    invalid_reasons: List[str] = []
    for ref in claim.evidence_refs:
        valid, error = _validated_ref(ref)
        if valid is not None:
            valid_refs.append(valid)
        else:
            invalid_reasons.append(error or ref.render())

    if not valid_refs:
        return ClaimValidation(
            claim=claim,
            status=STATUS_UNKNOWN,
            matched_evidence=(),
            reason="evidence metadata insufficient: {}".format(
                "; ".join(invalid_reasons)
            ),
        )

    pool = evidence.inspected if require_inspected else evidence.gathered
    matched: List[GatheredEvidence] = []
    seen = set()
    for ref in valid_refs:
        for item in pool:
            if item.path != ref.path:
                continue
            if not _covers(item, ref):
                continue
            key = _match_key(item)
            if key in seen:
                continue
            seen.add(key)
            matched.append(item)

    if matched:
        return ClaimValidation(
            claim=claim,
            status=STATUS_SUPPORTED,
            matched_evidence=tuple(matched),
            reason=(
                "matched gathered evidence: {}".format(
                    _describe_matched(matched)
                )
            ),
        )

    return ClaimValidation(
        claim=claim,
        status=STATUS_UNSUPPORTED,
        matched_evidence=(),
        reason=_describe_unsupported(
            valid_refs, evidence, require_inspected
        ),
        code=CLAIM_EVIDENCE_INVALID,
    )


def validate_claims(
    claims: Sequence[Claim],
    evidence: EvidenceIndex,
    require_inspected: bool = False,
) -> List[ClaimValidation]:
    """Validate each claim independently; result order = input order.

    ``require_inspected`` (M13) is forwarded to every claim — the
    eligibility rule is one deterministic switch, never per-claim.
    """
    return [
        validate_claim(claim, evidence, require_inspected)
        for claim in claims
    ]
