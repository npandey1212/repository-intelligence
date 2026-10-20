"""Repository Query — the evidence-driven offline run and its modes.

What is under test here:

1. **the offline run really investigates** — a synthetic fixture
   (``tests/fixtures/query_evidence``) is read through the real
   read-only tools, and every component, relationship, path,
   uncertainty and answer line comes from that evidence: change the
   fixture and the answer changes with it;
2. **the answer is the merged report's**, never the diagram's —
   ``AGENT_TOOL_MCP diagram …`` (the finding's summary) is not the
   answer, and the diagram stays a projection of the same report;
3. **the run is bounded** — a fixed number of reads and tool calls,
   no matter how much the repository offers;
4. **the two modes are the caller's** — offline ignores a provider
   key that happens to be set, ``--live`` fails with its own two-line
   message when there is no provider, and neither mode ever falls
   back to the other.

Standalone by the same discipline as the neighbouring test module:
the CLI is loaded by path, every writable root is redirected to
``tmp_path``, and nothing here touches ``tests/``' internals.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from repo_intelligence import repository_query as rq
from repo_intelligence.agent import trace_sink as trace_sink_module
from repo_intelligence.agents import visualization_artifacts as viz_module
from repo_intelligence.context_engine import engine as context_engine_module
from repo_intelligence.offline_investigation import (
    MAX_READS,
    MAX_TOOL_CALLS,
)

CLI_PATH = Path(__file__).resolve().parents[2] / (
    "agentrun_repository_query.py"
)
SOURCES = Path(__file__).resolve().parents[2] / "src" / (
    "repo_intelligence"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPO_A = FIXTURES / "query_evidence"
REPO_B = FIXTURES / "query_evidence_alt"

QUESTION = (
    "How do the agents, tools, and MCP servers interact?"
)
LIVE_HELP = (
    "Use the configured live LLM provider instead of the "
    "deterministic offline model."
)
LIVE_FAILURE_FIRST = (
    "Live mode requested but the configured LLM provider is "
    "unavailable."
)
LIVE_FAILURE_SECOND = "Set the required configuration and retry."


def _load_cli(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli(CLI_PATH, "agentrun_repository_query_evidence_cli")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Every root a run can write to, pointed at ``tmp_path``."""
    monkeypatch.setattr(
        viz_module,
        "default_artifacts_root",
        lambda: tmp_path / "artifacts",
    )
    monkeypatch.setattr(
        trace_sink_module,
        "default_trace_root",
        lambda: tmp_path / "runtime" / "agent_runs",
    )
    monkeypatch.setattr(
        context_engine_module,
        "default_context_root",
        lambda: tmp_path / "runtime" / "context",
    )
    monkeypatch.setattr(
        rq,
        "default_query_db_path",
        lambda: tmp_path / "runtime" / "agent_memory" / "repository_query.db",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def _query(root, **kwargs):
    return rq.run_repository_query(
        QUESTION,
        repository=str(root),
        mode=kwargs.pop("mode", "offline"),
        artifacts=False,
        **kwargs
    )


# -- 1. the run investigates: every line is derived ------------------------


def test_offline_run_derives_components_and_relationships(
    sandbox, monkeypatch
):
    """The fixture's own agent, tool and MCP server — read from the
    fixture, stated as evidence-backed findings."""
    result = _query(REPO_A)

    assert result.status == "COMPLETED"
    assert result.mode == "offline"
    assert result.collaboration_status == "passed"
    assert result.diagram_status == "passed"
    assert result.failures == []

    statements = "\n".join(result.conclusions)
    assert "Component research-agent (agent)" in statements
    assert "Component weather-tool (tool)" in statements
    assert "Component custom-server-a (mcp_server)" in statements
    assert "research-agent DECLARES get-forecast" in statements
    assert "get-forecast EXPOSED_BY custom-server-a" in statements
    # every cited path is a file the investigation actually read
    for ref in (".mcp.json", "agents/research_agent.md"):
        assert ref in "\n".join(result.evidence)


def test_answer_reports_the_path_the_evidence_established(
    sandbox, monkeypatch
):
    result = _query(REPO_A)

    lines = result.answer.splitlines()
    assert lines[0].endswith(
        "evidence-backed statement(s) established by the investigation."
    )
    assert any(
        line.startswith("established path: research-agent -> DECLARES")
        for line in lines
    )
    assert "components:" in result.answer
    assert "relationships:" in result.answer


def test_uncertainty_is_stated_when_the_protocol_is_unknown(
    sandbox, monkeypatch
):
    """No inspected file names a transport: the answer says so rather
    than inventing one."""
    result = _query(REPO_A)

    assert result.uncertainties
    assert "wire-level protocol" in result.uncertainties[0]
    assert "The evidence does not establish:" in result.answer


def test_no_hard_coded_story_in_the_investigation_sources():
    """The subject's own names live in the repository, never in the
    investigation script."""
    text = ""
    for name in ("offline_investigation.py", "repository_query.py"):
        text += (SOURCES / name).read_text(encoding="utf-8").lower()
    assert "cloudtrail" not in text


# -- the answer is the report, not the diagram -----------------------------


def test_answer_is_not_the_diagram_summary(sandbox, monkeypatch):
    """``finding.summary`` is the diagram's own line; the answer is
    synthesized from the merged investigation report instead."""
    result = _query(REPO_A)

    assert "AGENT_TOOL_MCP diagram" not in result.answer
    assert "evidence-backed statement(s)" in result.answer
    assert "established path:" in result.answer


# -- 3. the run is bounded -------------------------------------------------


def test_investigation_reads_a_bounded_number_of_files(
    sandbox, monkeypatch
):
    captured = {}
    original = rq.build_offline_models

    def spy(question, repository, commit, entry_target):
        models = original(
            question, repository, commit, entry_target
        )
        captured.update(models)
        return models

    monkeypatch.setattr(rq, "build_offline_models", spy)
    result = _query(REPO_A)

    assert result.status == "COMPLETED"
    arch = captured["arch"]
    assert 1 <= len(arch.read_files) <= MAX_READS
    # one entry read + two discovery searches + the ranked reads
    calls = len(arch.attempted) + len(arch.searched)
    assert calls <= MAX_TOOL_CALLS


# -- 4. the two modes belong to the caller ---------------------------------


def test_offline_mode_ignores_a_provider_key(sandbox, monkeypatch):
    """A key in the environment does not flip the default: offline is
    offline until --live asks for more."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")

    result = _query(REPO_A, mode="offline")

    assert result.mode == "offline"
    assert result.status == "COMPLETED"


def test_live_mode_without_a_provider_fails_clearly(
    sandbox, monkeypatch
):
    result = _query(REPO_A, mode="live")

    assert result.mode == "live"
    assert result.status == "FAILED"
    assert (
        result.termination_reason
        == rq.LIVE_CONFIGURATION_UNAVAILABLE
    )
    assert result.answer == ""
    assert result.failures[0] == "{}\n{}".format(
        LIVE_FAILURE_FIRST, LIVE_FAILURE_SECOND
    )
    assert result.failures[1]


def test_unknown_mode_is_rejected(sandbox, monkeypatch):
    with pytest.raises(rq.RepositoryQueryError):
        rq.run_repository_query(
            QUESTION,
            repository=str(REPO_A),
            mode="maybe-live",
        )


# -- the CLI surface -------------------------------------------------------


def test_cli_help_documents_the_live_flag():
    # argparse wraps long help to the terminal width: compare the
    # sentence as the reader sees it, whitespace normalized.
    help_text = " ".join(cli.build_parser().format_help().split())
    assert "--live" in help_text
    assert LIVE_HELP in help_text


def test_cli_live_failure_prints_the_two_lines_and_exits_non_zero(
    sandbox, monkeypatch, capsys
):
    code = cli.main(
        ["--live", "--repository", str(REPO_A), QUESTION]
    )
    out = capsys.readouterr().out

    assert code != 0
    assert "ERROR: Live mode unavailable." in out
    assert LIVE_FAILURE_FIRST in out
    assert LIVE_FAILURE_SECOND in out
    assert "No investigation was executed." in out
    assert "Final answer" not in out


def test_cli_offline_still_reports_its_mode(
    sandbox, monkeypatch, capsys
):
    code = cli.main(["--repository", str(REPO_A), QUESTION])
    out = capsys.readouterr().out

    assert code == 0
    assert "Mode:       offline" in out
    assert "evidence-backed statement(s)" in out


# -- two repositories, two answers -----------------------------------------


def test_two_repositories_produce_two_different_answers(
    sandbox, monkeypatch
):
    """The answer follows the evidence: swap the fixture's server and
    the answer swaps with it — no fixed story survives."""
    answer_a = _query(REPO_A).answer
    answer_b = _query(REPO_B).answer

    assert "custom-server-a" in answer_a
    assert "custom-server-b" not in answer_a
    assert "custom-server-b" in answer_b
    assert "custom-server-a" not in answer_b
