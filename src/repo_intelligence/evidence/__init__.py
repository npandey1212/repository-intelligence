"""Deterministic evidence traceability (M8.8, eligibility tightened M13).

Validate whether **structured claims** point to evidence actually
gathered during an Agent run — offline, no LLM, no critic, no second
agent. Traceability, not truth:

    Agent ──► Execution trace ──► Evidence Traceability Validator
                                        │
                              Claim + evidence_refs
                                        │
                          SUPPORTED / UNSUPPORTED / UNKNOWN

Public surface:

    EvidenceRef, Claim                 (input model — claims are given,
                                        never extracted from free text)
    GatheredEvidence, EvidenceIndex    (what the run actually observed,
                                        derived from successful tool
                                        results in the trace; the
                                        inspected subset is M13 claim
                                        eligibility)
    ClaimValidation, STATUS_*          (deterministic result)
    CLAIM_EVIDENCE_INVALID             (M13 structured failure code)
    gather_evidence                    (trace → EvidenceIndex)
    validate_claim / validate_claims   (claims + evidence → results)

Consumes the existing execution data only (``AgentTrace`` or its JSON
mapping, e.g. ``runtime/agent_runs/<run_id>/trace.json``). Nothing in
this package calls a model, mutates the Agent, or stores hidden
reasoning fields. Since the M8.8 integration, ``Agent.run_result``
calls ``gather_evidence`` + ``validate_claims`` exactly once after the
investigation loop — this package stays pure functions over the
finished trace. Since M13 that Agent call validates
**inspected-only** evidence (``require_inspected=True``): a search
hit is a discovery, not evidence.
"""

from .models import (
    CLAIM_EVIDENCE_INVALID,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    STATUS_UNKNOWN,
    VALIDATION_STATUSES,
    Claim,
    ClaimValidation,
    EvidenceIndex,
    EvidenceRef,
    FailedEvidenceAttempt,
    GatheredEvidence,
    ValidationStatus,
)
from .validator import gather_evidence, validate_claim, validate_claims

__all__ = [
    "CLAIM_EVIDENCE_INVALID",
    "STATUS_SUPPORTED",
    "STATUS_UNSUPPORTED",
    "STATUS_UNKNOWN",
    "VALIDATION_STATUSES",
    "Claim",
    "ClaimValidation",
    "EvidenceIndex",
    "EvidenceRef",
    "FailedEvidenceAttempt",
    "GatheredEvidence",
    "ValidationStatus",
    "gather_evidence",
    "validate_claim",
    "validate_claims",
]
