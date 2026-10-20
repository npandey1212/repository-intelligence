"""M9 — evaluation sink tests (separate artifact, same run tree).

Evaluation persistence is a separable concern from evaluation itself:

 1  writes <root>/<run_id>/evaluation.json (canonical) + .md
 2  JSON is valid and round-trips through EvaluationResult.from_dict
 3  Markdown is a projection of the same JSON (no independent facts)
 4  custom root respected (deterministic temp dirs)
 5  empty run_id raises (an evaluation always belongs to a run)
 6  markdown=False writes JSON only
 7  evaluation files never overwrite the run's trace files
 8  rendered markdown carries no banned vocabulary and no score
"""

from __future__ import annotations

import json

import pytest

from repo_intelligence.evaluation import (
    STATUS_PASS,
    EvaluationCriteria,
    EvaluationCriterionResult,
    EvaluationResult,
    Evaluator,
    JsonFileEvaluationSink,
    ObjectiveCriterion,
    render_markdown,
)

ANSWER = "Configuration flows from config.yaml through loader.py."


def _evaluation(run_id="run-sink"):
    return EvaluationResult(
        status=STATUS_PASS,
        criteria=(
            EvaluationCriterionResult(
                "final_result", STATUS_PASS, "final answer present"
            ),
            EvaluationCriterionResult(
                "evidence_gathered", STATUS_PASS, "2 successful"
            ),
            EvaluationCriterionResult(
                "claim_traceability", STATUS_PASS, "1 of 1 supported"
            ),
            EvaluationCriterionResult(
                "execution_bound", STATUS_PASS, "executed 1 of 8"
            ),
            EvaluationCriterionResult(
                "objective_coverage", STATUS_PASS, "all 3 present"
            ),
        ),
        summary="5 PASS, 0 PARTIAL, 0 UNKNOWN, 0 FAIL",
        run_id=run_id,
    )


def _trace(run_id="run-sink"):
    return {
        "run_id": run_id,
        "iterations": [
            {
                "iteration": 1,
                "tool_results": [
                    {
                        "tool": "read_config",
                        "success": True,
                        "data": {"path": "config.yaml"},
                    }
                ],
            }
        ],
        "summary": {
            "max_tool_calls": 8,
            "tool_calls_executed": 1,
            "tool_calls_blocked": 0,
        },
    }


# ---------------------------------------------------------------------------
# 1 — artifact layout beside the trace
# ---------------------------------------------------------------------------


def test_writes_evaluation_json_and_markdown(tmp_path):
    sink = JsonFileEvaluationSink(root=tmp_path)
    evaluation = _evaluation()
    sink.write(evaluation)

    json_path = tmp_path / "run-sink" / "evaluation.json"
    md_path = tmp_path / "run-sink" / "evaluation.md"
    assert json_path.is_file()
    assert md_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_PASS
    assert payload["run_id"] == "run-sink"
    assert len(payload["criteria"]) == 5


# ---------------------------------------------------------------------------
# 2 — JSON round-trip
# ---------------------------------------------------------------------------


def test_written_json_round_trips(tmp_path):
    sink = JsonFileEvaluationSink(root=tmp_path)
    original = _evaluation()
    sink.write(original)

    payload = json.loads(
        (tmp_path / "run-sink" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    restored = EvaluationResult.from_dict(payload)
    assert restored == original
    assert restored.to_json() == original.to_json()


# ---------------------------------------------------------------------------
# 3 — markdown is a projection of the same JSON
# ---------------------------------------------------------------------------


def test_markdown_projects_the_json_verdict(tmp_path):
    sink = JsonFileEvaluationSink(root=tmp_path)
    evaluation = _evaluation()
    sink.write(evaluation)

    text = (tmp_path / "run-sink" / "evaluation.md").read_text(
        encoding="utf-8"
    )
    assert text == render_markdown(evaluation)
    assert "**Overall:** `{}`".format(evaluation.status) in text
    for line in text.splitlines():
        if line.startswith("**") and ":** `" in line:
            status = line.split("`")[1]
            assert status in {
                "PASS",
                "FAIL",
                "PARTIAL",
                "UNKNOWN",
            } or line.startswith("**Overall")


# ---------------------------------------------------------------------------
# 4 — custom root
# ---------------------------------------------------------------------------


def test_custom_root_respected(tmp_path):
    root = tmp_path / "elsewhere"
    sink = JsonFileEvaluationSink(root=root)
    assert sink.root == root
    sink.write(_evaluation(run_id="run-x"))
    assert (root / "run-x" / "evaluation.json").is_file()


def test_default_root_is_the_runtime_agent_runs_tree():
    from repo_intelligence.evaluation.sink import (
        JsonFileEvaluationSink as Sink,
    )

    sink = Sink()
    assert sink.root.name == "agent_runs"
    assert sink.root.parent.name == "runtime"


# ---------------------------------------------------------------------------
# 5 — run identity required
# ---------------------------------------------------------------------------


def test_empty_run_id_raises(tmp_path):
    sink = JsonFileEvaluationSink(root=tmp_path)
    with pytest.raises(ValueError) as excinfo:
        sink.write(_evaluation(run_id=""))
    assert "run_id" in str(excinfo.value)
    # nothing half-written
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 6 — markdown optional
# ---------------------------------------------------------------------------


def test_markdown_disabled_writes_json_only(tmp_path):
    sink = JsonFileEvaluationSink(root=tmp_path, markdown=False)
    sink.write(_evaluation())
    run_dir = tmp_path / "run-sink"
    assert (run_dir / "evaluation.json").is_file()
    assert not (run_dir / "evaluation.md").exists()


# ---------------------------------------------------------------------------
# 7 — evaluation never touches the run's trace
# ---------------------------------------------------------------------------


def test_evaluation_artifacts_never_overwrite_trace(tmp_path):
    run_dir = tmp_path / "run-sink"
    run_dir.mkdir(parents=True)
    trace_json = run_dir / "trace.json"
    trace_json.write_text('{"run_id": "run-sink"}', encoding="utf-8")
    trace_md = run_dir / "trace.md"
    trace_md.write_text("# Agent Execution Trace", encoding="utf-8")

    JsonFileEvaluationSink(root=tmp_path).write(_evaluation())

    assert trace_json.read_text(encoding="utf-8") == '{"run_id": "run-sink"}'
    assert trace_md.read_text(encoding="utf-8") == "# Agent Execution Trace"
    assert (run_dir / "evaluation.json").is_file()
    assert (run_dir / "evaluation.md").is_file()


# ---------------------------------------------------------------------------
# 8 — render: no score, no banned vocabulary
# ---------------------------------------------------------------------------


def test_render_markdown_has_no_score_and_no_banned_words():
    text = render_markdown(_evaluation())
    from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS

    lowered = text.lower()
    for word in BANNED_REASONING_FIELDS + ("reasoning", "reasoning_trace"):
        assert word not in lowered, word
    for score_word in ("score", "quality: ", "confidence"):
        assert score_word not in lowered, score_word


def test_render_markdown_explains_non_pass_criteria():
    evaluation = EvaluationResult(
        status=STATUS_PASS,
        criteria=(
            EvaluationCriterionResult(
                "final_result", STATUS_PASS, "answer present"
            ),
            EvaluationCriterionResult(
                "objective_coverage",
                "UNKNOWN",
                "no expected objective criteria defined",
            ),
        ),
        summary="1 PASS, 0 PARTIAL, 1 UNKNOWN, 0 FAIL",
        run_id="run-2",
    )
    text = render_markdown(evaluation)
    # PASS rows stay terse; non-PASS rows show the observable reason
    assert "**Objective coverage:** `UNKNOWN`" in text
    assert "no expected objective criteria" in text
    assert "answer present" not in text


# ---------------------------------------------------------------------------
# end-to-end: evaluate a real-shaped trace, then persist it
# ---------------------------------------------------------------------------


def test_evaluate_then_persist(tmp_path):
    evaluation = Evaluator.evaluate(
        None,  # no final result → the FAIL case is what we persist
        _trace(),
        EvaluationCriteria(
            objective="Explain how configuration is loaded.",
            objective_criteria=(ObjectiveCriterion("entry_point"),),
        ),
    )
    JsonFileEvaluationSink(root=tmp_path).write(evaluation)
    payload = json.loads(
        (tmp_path / "run-sink" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "FAIL"
    by_id = {c["criterion"]: c for c in payload["criteria"]}
    assert by_id["final_result"]["status"] == "FAIL"
    assert by_id["evidence_gathered"]["status"] == "PASS"
