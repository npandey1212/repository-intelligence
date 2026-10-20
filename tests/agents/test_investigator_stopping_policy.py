"""Evidence sufficiency as the stopping policy (M17 cleanup).

The reliability gap had two halves. The prompt half is here: the
Architecture Investigator is told — in its own versioned instructions —
that EVIDENCE SUFFICIENCY is the primary stopping condition and the
tool-call limit is an emergency ceiling, not a target, so a healthy run
closes from evidence long before any budget. The code half lives in
``tests/agent/test_working_context.py`` (the working-context ceiling)
and in the existing 64-call boundary tests.

1. stopping policy in the prompt  the strategy section states the
                              policy, the reassessment checklist, the
                              duplicate-exploration ban, and the
                              one-targeted-follow-up rule — with the
                              repository-agnostic / non-product
                              vocabulary constraints intact
2. the investigator stops from evidence  one read, then the typed
                              report: termination is voluntary, the
                              budget never had to intervene, nothing
                              else was requested
3. M17 composition              ONE 64-call number (orchestration and
                              execution agree) and the working-context
                              ceiling inherited unchanged

Everything here is scripted and LLM-free — no network, no API key.
"""

from __future__ import annotations

from architecture_doubles import (
    build_investigator,
    read_then_report,
)

from repo_intelligence.agent import DEFAULT_MAX_CONTEXT_CHARACTERS
from repo_intelligence.agents import (
    ARCHITECTURE_INVESTIGATOR_PROMPT_V1,
    PROMPT_VERSION,
)
from repo_intelligence import repository_query as rq


def _strategy_section() -> str:
    prompt = ARCHITECTURE_INVESTIGATOR_PROMPT_V1
    start = prompt.index("INVESTIGATION STRATEGY")
    end = prompt.index("OUTPUT")
    return prompt[start:end]


# ---------------------------------------------------------------------------
# 1 — the policy is in the prompt
# ---------------------------------------------------------------------------


def test_prompt_carries_the_evidence_sufficiency_stopping_policy():
    section = _strategy_section()

    assert "Evidence sufficiency is the primary stopping condition." in section
    assert (
        "The\ntool-call limit is an emergency hard ceiling, not a target"
        in section
    )
    assert "When it is sufficient: stop using tools and" in section
    assert "finalize — never keep exploring because budget remains" in section
    assert "stop when the question is answered" in section
    assert "search is not a substitute for reasoning" in section
    assert "make ONE targeted tool call" in section
    assert (
        "Prefer 8 strong evidence-producing\ncalls over 40 exploratory "
        "calls" in section
    )


def test_prompt_bans_duplicate_exploration():
    section = _strategy_section()

    assert "reuse existing evidence" in section
    assert "do not search again for facts" in section
    assert "avoid duplicate exploration" in section
    assert "do not repeat the same repository" in section
    assert "form an evidence plan before extensive tool usage" in section


def test_prompt_stays_within_the_shared_language_contract():
    prompt = ARCHITECTURE_INVESTIGATOR_PROMPT_V1
    lowered = prompt.lower()

    for banned in ("orchestrat", "planner", "adop"):
        assert banned not in lowered
    assert prompt.startswith("ROLE")
    for heading in (
        "RESPONSIBILITIES",
        "RULES",
        "INVESTIGATION STRATEGY",
        "OUTPUT",
    ):
        assert heading in prompt
    # In-place extension of V1: the version log in
    # docs/architecture-investigator.md records the addition.
    assert PROMPT_VERSION == "V1"


# ---------------------------------------------------------------------------
# 2 — the run stops from evidence, budget untouched
# ---------------------------------------------------------------------------


def test_investigator_stops_from_evidence_without_consuming_the_budget():
    client_responses = read_then_report()
    specialist = build_investigator(*client_responses)

    result = specialist.investigate(
        "How do the agents, tools, and MCP servers interact?",
        "example-repository",
        "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0",
    )

    assert result is not None
    trace = specialist.last_trace
    summary = trace.summary

    # Voluntary termination: evidence was sufficient, the budget never
    # had to intervene, and no boundary/finalization machinery ran.
    assert trace.termination_reason == "STRUCTURED_OUTPUT"
    assert trace.stop_reason == "EVIDENCE_SUFFICIENT"
    assert summary.tool_calls_executed == 1
    assert summary.tool_calls_blocked == 0
    assert trace.lifecycle_events == []
    assert trace.context_restrictions == 0

    # The script was satisfied by ONE inspection: no exploratory tail.
    assert len(specialist.agent._llm_client.requests) == 2


# ---------------------------------------------------------------------------
# 3 — M17 composition: one 64-call number, ceiling inherited
# ---------------------------------------------------------------------------


def test_m17_keeps_one_64_call_budget_and_the_context_ceiling():
    assert rq.M17_MAX_TOOL_CALLS == 64
    assert rq.M17_EXECUTION_CONFIG.max_tool_calls == 64
    assert rq.M17_ORCHESTRATION_CONFIG.max_total_tool_calls == 64
    assert rq.M17_EXECUTION_CONFIG.finalize_on_boundary is True
    # The working-context ceiling rides along unchanged: no second
    # number, no M17-specific override.
    assert (
        rq.M17_EXECUTION_CONFIG.max_context_characters
        == DEFAULT_MAX_CONTEXT_CHARACTERS
    )
