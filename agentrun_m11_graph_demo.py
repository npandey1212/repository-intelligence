"""M11 — engineering graph persistence & query demonstration (real model).

One human-runnable script that shows the M11 learning goal happening
for a REAL investigation (spec §14):

    python agentrun_m11_graph_demo.py
    python agentrun_m11_graph_demo.py --model gpt-4o-mini
    python agentrun_m11_graph_demo.py --question "…"

Objective (spec §14 example, over the pinned snapshot's artifacts):

    "Identify the repository's MCP-related engineering artifacts and
     show their graph relationships."

What it does (each step reuses the existing implementation):

     1  verify the persisted graph artifact (M11.1)
        load_engineering_graph_document() → engineering_graph.json:
        schema version, provenance, counts — the canonical artifact,
        read as bytes, before any LLM is involved
     2  compose the existing Agent              (OpenAILLMClient +
                                                  default_tool_dispatcher +
                                                  JsonFileTraceSink; four
                                                  read-only tools, Tool 4
                                                  = query_engineering_graph)
     3  run ONE investigation                   agent.run(objective)
     4  read back the ONE AgentTrace and        stage-by-stage, chrono-
        classify the stages                     logical:
                                                  LLM reasoning
                                                  Graph retrieval
                                                  Source-file retrieval
                                                  Structured answer
     5  print the stage breakdown and the       graph/tool facts come
        final answer                            from the trace, never
                                                  re-derived
     6  exit code reflects the outcome          (mapping below)

This script does NOT implement graph construction, persistence,
projection, queries, or the agent loop. It is an entry point into the
existing implementation: it supplies the objective, checks the
artifact, and reads back what the trace recorded.

Stage semantics (read from the trace only):

* LLM reasoning          — every agent iteration (one LLM turn); its
                           response type and requested tool names
* Graph retrieval        — tool executions of
                           query_engineering_graph (Tool 4)
* Source-file retrieval  — tool executions of search_repository /
                           read_file (observed file content)
* Structured answer      — the final answer text recorded on the
                           terminating iteration (the model's prose,
                           never raw JSON)

Exit codes:
     0  run completed AND at least one engineering graph query was
        executed (the milestone actually demonstrated)
     6  configuration/execution failure (missing OPENAI_API_KEY,
        missing engineering_graph.json artifact, LLMError, AgentError)
        — or the run completed WITHOUT a single graph query: reported
        explicitly as such, never counted as success

Unexpected programming errors are NOT caught: they propagate with a
traceback.

Credentials/configuration: requires OPENAI_API_KEY and the persisted
M11 artifact (generate it with
``python -m repo_intelligence.graph.engineering_graph_persist``).
Without either the script fails CLEARLY with exit 6 and never falls
back to a fake model or the raw model projection. The offline test
suite never runs this script against the network
(tests/graph/test_m11_graph_demo.py wires it with doubles only).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentError,
    JsonFileTraceSink,
)
from repo_intelligence.context import (  # noqa: E402
    ArchitectureContextError,
)
from repo_intelligence.context.models import (  # noqa: E402
    discover_snapshot_id,
)
from repo_intelligence.artifacts import (  # noqa: E402
    engineering_graph_path,
    resolve_output_root,
)
from repo_intelligence.engineering import (  # noqa: E402
    EngineeringModelError,
)
from repo_intelligence.graph import (  # noqa: E402
    load_engineering_graph_document,
)
from repo_intelligence.llm import LLMError  # noqa: E402
from repo_intelligence.llm.config import load_env  # noqa: E402
from repo_intelligence.llm.openai_client import (  # noqa: E402
    DEFAULT_MODEL,
    OpenAILLMClient,
)
from repo_intelligence.tool_dispatcher import (  # noqa: E402
    default_tool_dispatcher,
)

RULE = "=" * 72
SECTION = "-" * 52

EXIT_SUCCESS = 0
EXIT_EXECUTION_FAILED = 6

GRAPH_TOOL = "query_engineering_graph"
SOURCE_FILE_TOOLS = ("search_repository", "read_file")

# M11 explicitly allows a repository-aware objective (spec §14): the
# question is answered by the LLM over the existing tools, not
# hard-coded to any repository structure or node id.
DEFAULT_OBJECTIVE = (
    "Identify the repository's MCP-related engineering artifacts and "
    "show their graph relationships. Start from the engineering graph: "
    "call query_engineering_graph with {} to see the node-kind summary, "
    "then run bounded traversals from the node ids it reports. "
    "Corroborate with search_repository or read_file only when a graph "
    "result points at a file worth quoting. Base every statement on "
    "tool results you actually received: name the node ids and "
    "relationship kinds you found, and say plainly when the graph "
    "shows no relationship for something."
)


# ---------------------------------------------------------------------------
# Artifact preflight (M11.1 persistence, read-only)
# ---------------------------------------------------------------------------


def check_graph_artifact(output_root: Optional[Path] = None) -> Tuple[
    Dict[str, Any], Path
]:
    """Load and report the persisted engineering graph artifact.

    Raises ArchitectureContextError when no run is discoverable and
    FileNotFoundError when the artifact itself is absent (both mapped
    to exit 6 by ``main``).
    """
    root = resolve_output_root(output_root)
    snapshot_id = discover_snapshot_id(root)
    path = engineering_graph_path(root, snapshot_id)
    if not path.is_file():
        raise FileNotFoundError(
            "{} not found — generate it with: python -m "
            "repo_intelligence.graph.engineering_graph_persist".format(path)
        )
    document = load_engineering_graph_document(path)
    return document, path


# ---------------------------------------------------------------------------
# Stage classification (trace projection — nothing is re-derived)
# ---------------------------------------------------------------------------


def classify_stages(trace: Any) -> Dict[str, Any]:
    """Project the ONE completed trace into the spec §14 stage list.

    Pure read of ``trace.iterations``: LLM reasoning (one entry per
    iteration), graph retrieval and source-file retrieval (executed
    tool results, by tool name), and the structured answer (the final
    answer recorded on the terminating iteration).
    """
    reasoning: List[Dict[str, Any]] = []
    graph_calls: List[Dict[str, Any]] = []
    source_calls: List[Dict[str, Any]] = []
    answer: Optional[str] = None
    for iteration in trace.iterations:
        requested = [
            str(tool_call.get("name"))
            for tool_call in iteration.tool_calls
        ]
        reasoning.append(
            {
                "iteration": iteration.iteration,
                "response_type": iteration.response_type,
                "requested_tools": requested,
                "reason_summary": iteration.reason_summary,
            }
        )
        for result in iteration.tool_results:
            name = result.get("tool")
            record = {
                "iteration": iteration.iteration,
                "tool": name,
                "success": bool(result.get("success")),
                "source": result.get("source"),
            }
            if name == GRAPH_TOOL:
                graph_calls.append(record)
            elif name in SOURCE_FILE_TOOLS:
                source_calls.append(record)
        if iteration.final_answer is not None:
            answer = iteration.final_answer
    return {
        "llm_reasoning": reasoning,
        "graph_retrieval": graph_calls,
        "source_file_retrieval": source_calls,
        "structured_answer": answer,
        "counts": {
            "iterations": len(reasoning),
            "graph_queries": len(graph_calls),
            "source_file_calls": len(source_calls),
            "answer_recorded": answer is not None,
        },
    }


def exit_code_for(stages: Dict[str, Any]) -> int:
    """0 only when the run completed AND queried the engineering
    graph at least once — the M11 demonstration itself."""
    counts = stages["counts"]
    if not counts["answer_recorded"] or counts["graph_queries"] == 0:
        return EXIT_EXECUTION_FAILED
    return EXIT_SUCCESS


# ---------------------------------------------------------------------------
# Console projections
# ---------------------------------------------------------------------------


def _relative(path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_graph_artifact(document: Dict[str, Any], path: Path) -> None:
    print("Persisted engineering graph (M11.1)", file=sys.stdout)
    print(SECTION, file=sys.stdout)
    print("artifact: {}".format(_relative(path)))
    print("schema:   {}".format(document.get("schema_version")))
    provenance = document.get("provenance") or {}
    print("source:   {}".format(provenance.get("source_artifact")))
    print("commit:   {}".format(
        (provenance.get("repository") or {}).get("commit_sha")
    ))
    print("snapshot: {}".format(provenance.get("snapshot_id")))
    counts = (document.get("metadata") or {}).get("counts") or {}
    print(
        "counts:   {} nodes, {} relationships, {} unresolved".format(
            counts.get("nodes"),
            counts.get("relationships"),
            counts.get("unresolved"),
        )
    )
    print(file=sys.stdout)


def print_stages(stages: Dict[str, Any], stream=None) -> None:
    """Chronological stage view + per-stage totals (trace only)."""
    stream = stream or sys.stdout
    print("Run stages (from the recorded trace)", file=stream)
    print(SECTION, file=stream)
    for entry in stages["llm_reasoning"]:
        tools = ", ".join(entry["requested_tools"]) or "-"
        graph_hits = sum(
            1
            for call in stages["graph_retrieval"]
            if call["iteration"] == entry["iteration"]
        )
        source_hits = sum(
            1
            for call in stages["source_file_retrieval"]
            if call["iteration"] == entry["iteration"]
        )
        print(
            "iter {} [{}] tools={} (graph {}, source-file {})".format(
                entry["iteration"],
                entry["response_type"],
                tools,
                graph_hits,
                source_hits,
            ),
            file=stream,
        )
    counts = stages["counts"]
    print(file=stream)
    print(
        "llm reasoning:         {} iteration(s)".format(
            counts["iterations"]
        ),
        file=stream,
    )
    print(
        "graph retrieval:       {} query_engineering_graph call(s)".format(
            counts["graph_queries"]
        ),
        file=stream,
    )
    for call in stages["graph_retrieval"]:
        artifact = (call.get("source") or {}).get("artifact") or "-"
        print(
            "    iter {} success={} artifact={}".format(
                call["iteration"],
                call["success"],
                _relative(Path(artifact)) if artifact != "-" else "-",
            ),
            file=stream,
        )
    print(
        "source-file retrieval: {} search_repository/read_file call(s)".format(
            counts["source_file_calls"]
        ),
        file=stream,
    )
    print(
        "structured answer:     {}".format(
            "recorded" if counts["answer_recorded"] else "MISSING"
        ),
        file=stream,
    )


def print_answer(answer: Optional[str], stream=None) -> None:
    stream = stream or sys.stdout
    print("Answer", file=stream)
    print(SECTION, file=stream)
    if not answer:
        print("no final answer recorded on the trace", file=stream)
        return
    print(answer.strip(), file=stream)


def _print_trace_paths(trace_sink, trace, stream) -> None:
    json_path = getattr(trace_sink, "json_path", None)
    if trace is None or not callable(json_path):
        return
    path = json_path(trace.run_id)
    if path.exists():
        print("trace: {}".format(_relative(path)), file=stream)


def _failed(message: str, detail: Optional[BaseException] = None) -> int:
    print("M11 demonstration failed.", file=sys.stderr)
    print(message, file=sys.stderr)
    if detail is not None:
        print(
            "{}: {}".format(type(detail).__name__, detail), file=sys.stderr
        )
    return EXIT_EXECUTION_FAILED


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m11_graph_demo.py",
        description=(
            "M11 end-to-end demonstration: one real-model investigation "
            "over the persisted engineering graph — artifact preflight, "
            "the existing Agent with Tool 4, then the recorded trace "
            "projected into the spec §14 stages (LLM reasoning, graph "
            "retrieval, source-file retrieval, structured answer)."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation objective (default: the MCP "
        "artifacts/graph-relationships question from spec §14)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="application context: artifact output root "
        "(default: <project>/artifacts)",
    )
    args = parser.parse_args(argv)
    objective = args.question or DEFAULT_OBJECTIVE
    output_root = Path(args.output_root) if args.output_root else None

    # configuration (explicit .env load at the composition root — same
    # convention as the M9/M10 demos)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M11 demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_m11_graph_demo.py",
            file=sys.stderr,
        )
        return EXIT_EXECUTION_FAILED

    # artifact preflight: the canonical persisted graph, read as bytes
    # before any LLM is involved (never generated by this script)
    try:
        document, graph_path = check_graph_artifact(output_root)
    except ArchitectureContextError as exc:
        return _failed(
            "Re-run M3/M4 to restore the artifact run context.", exc
        )
    except FileNotFoundError as exc:
        return _failed(str(exc))
    except EngineeringModelError as exc:
        return _failed("Persisted graph artifact is unreadable.", exc)

    try:
        llm_client = OpenAILLMClient(model=args.model or DEFAULT_MODEL)
    except LLMError as exc:
        return _failed("cannot construct the LLM client", exc)

    trace_sink = JsonFileTraceSink()
    agent = Agent(
        llm_client=llm_client,
        tool_dispatcher=default_tool_dispatcher(),
        trace_sink=trace_sink,
    )

    print(RULE)
    print("M11 — Engineering Graph Persistence & Query (demonstration)")
    print(RULE)
    print("Objective:")
    print(objective)
    print()
    print_graph_artifact(document, graph_path)
    print(
        "model: {}   tools: 4 (search_repository, read_file, "
        "get_architecture_context, query_engineering_graph)".format(
            args.model or DEFAULT_MODEL
        )
    )
    print()

    try:
        agent.run(objective)
    except AgentError as exc:
        return _failed("agent run failed", exc)
    except LLMError as exc:
        return _failed("agent run failed", exc)

    trace = agent.last_trace
    if trace is None:
        return _failed("run produced no trace")
    stages = classify_stages(trace)
    print_stages(stages)
    print()
    print_answer(stages["structured_answer"])
    print()
    print("Artifacts", file=sys.stdout)
    print(SECTION, file=sys.stdout)
    _print_trace_paths(trace_sink, trace, sys.stdout)
    print(
        "graph artifact: {}".format(_relative(graph_path)),
        file=sys.stdout,
    )
    summary = trace.summary
    if summary is not None:
        print(
            "run: {} iterations, {} tool calls, termination={}".format(
                summary.total_iterations,
                summary.total_tool_calls,
                summary.termination_reason,
            ),
            file=sys.stdout,
        )
    print(file=sys.stdout)

    code = exit_code_for(stages)
    if code == EXIT_SUCCESS:
        print(
            "exit code: 0 (run completed; engineering graph queried "
            "{} time(s))".format(stages["counts"]["graph_queries"]),
            file=sys.stdout,
        )
    else:
        if stages["counts"]["answer_recorded"]:
            print(
                "run completed but never called query_engineering_graph — "
                "the M11 demonstration requires at least one graph "
                "query; re-run the objective.",
                file=sys.stderr,
            )
        else:
            print(
                "run completed without a recorded final answer.",
                file=sys.stderr,
            )
        print("exit code: {}".format(code), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
