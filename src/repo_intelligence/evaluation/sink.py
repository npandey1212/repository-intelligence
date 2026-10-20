"""Evaluation sinks — where an ``EvaluationResult`` is persisted (M9).

    Evaluator.evaluate(...) ──► EvaluationResult ──► EvaluationSink
                                                      ├── evaluation.json
                                                      └── evaluation.md

One implementation: ``JsonFileEvaluationSink``. Evaluation artifacts
co-locate with the run they describe (runtime telemetry tree, same
layout convention as traces):

    <root>/<run_id>/trace.json        what happened  (M8.x — untouched)
    <root>/<run_id>/evaluation.json   how it performed (M9)
    <root>/<run_id>/evaluation.md     rendered projection of the same JSON

JSON is canonical; Markdown is rendered from ``EvaluationResult.to_dict()``
— never an independent source of truth. ``root`` defaults to
``default_trace_root()`` (``<project-root>/runtime/agent_runs``), so a
run's trace and its evaluation sit side by side.

The sink does NOT evaluate: persistence is a separable concern from the
deterministic evaluation itself (``evaluator.py`` does no I/O).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from typing_extensions import Protocol, runtime_checkable

from ..agent.trace_sink import default_trace_root
from .evaluator import render_markdown
from .models import EvaluationResult

__all__ = [
    "EvaluationSink",
    "JsonFileEvaluationSink",
]


@runtime_checkable
class EvaluationSink(Protocol):
    """Receives one finished evaluation — one ``write`` call per run."""

    def write(self, evaluation: EvaluationResult) -> None:
        ...


class JsonFileEvaluationSink:
    """File sink: ``<root>/<run_id>/evaluation.{json,md}``."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        markdown: bool = True,
    ) -> None:
        self.root = Path(root) if root is not None else default_trace_root()
        self.markdown = markdown

    def evaluation_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def json_path(self, run_id: str) -> Path:
        return self.evaluation_dir(run_id) / "evaluation.json"

    def markdown_path(self, run_id: str) -> Path:
        return self.evaluation_dir(run_id) / "evaluation.md"

    def write(self, evaluation: EvaluationResult) -> None:
        if not evaluation.run_id:
            raise ValueError(
                "cannot write evaluation without a run_id — pass a "
                "result evaluated from a run trace"
            )
        directory = self.evaluation_dir(evaluation.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.json_path(evaluation.run_id).write_text(
            evaluation.to_json() + "\n", encoding="utf-8"
        )
        if self.markdown:
            self.markdown_path(evaluation.run_id).write_text(
                render_markdown(evaluation), encoding="utf-8"
            )
