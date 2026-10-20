"""M12 — Agent → Tool → MCP Server → MCP Configuration demonstration.

One human-runnable script that shows the M12 knowledge slice happening
for a REAL investigation (spec §20):

    python agentrun_m12_knowledge_demo.py
    python agentrun_m12_knowledge_demo.py --model gpt-4o-mini
    python agentrun_m12_knowledge_demo.py --question "…"

Objective (spec §20 example, over the pinned snapshot's artifacts):

    "Explain how a specific repository agent is connected to an MCP
     server and identify the evidence supporting the connection."

What it does (each step reuses the existing implementation):

     1  verify the persisted graph artifact           load_engineering_
        AND the M12 slice — offline,                  graph_document() →
        before any LLM is involved                    project_engineering_
                                                      graph() → slice
                                                      counts, then ONE
                                                      bounded depth-3 path
                                                      query through
                                                      DECLARES → EXPOSED_BY
                                                      → CONFIGURED_BY
     2  print the structured slice result             finding /
        (relationships + evidence paths,              relationships /
         reviewer-answerable offline)                 evidence_paths,
                                                      straight from the
                                                      graph + its evidence
     3  compose the existing Agent                    OpenAILLMClient +
                                                      default_tool_dispatcher
                                                      + JsonFileTraceSink;
                                                      four read-only tools
     4  run ONE investigation                         agent.run(objective)
     5  read back the ONE AgentTrace and             stage-by-stage,
        classify the stages                           chronological:
                                                      LLM reasoning /
                                                      Graph retrieval /
                                                      Source-file retrieval /
                                                      Structured answer
     6  print the stage breakdown and the            graph/tool facts come
        final answer                                  from the trace, never
                                                      re-derived
     7  exit code reflects the outcome               (mapping below)

This script does NOT implement graph construction, persistence,
projection, queries, or the agent loop. It is an entry point into the
existing implementation: it supplies the objective, checks the slice,
and reads back what the trace recorded.

Slice semantics (offline, from the persisted graph only):

* finding            — one agent that DECLARES a tool, the tool
                       EXPOSED_BY a server, and the server
                       CONFIGURED_BY a configuration (complete
                       forward path, or the demonstration fails)
* relationships      — the three canonical edge records (kind,
                       source, target) walked by the bounded query
* evidence_paths     — the repository paths cited by those edges
                       (what a reviewer opens to verify each hop)

Exit codes:
     0  slice path found offline AND the run completed with a
        recorded answer AND at least one engineering graph query
        was executed (the milestone actually demonstrated)
     6  configuration/execution failure (missing OPENAI_API_KEY,
        missing engineering_graph.json artifact, artifact without
        the M12 slice, no complete forward path, LLMError,
        AgentError) — or the run completed WITHOUT a single graph
        query: reported explicitly as such, never counted as success

Unexpected programming errors are NOT caught: they propagate with a
traceback.

Credentials/configuration: requires OPENAI_API_KEY and the persisted
M12-updated artifact (generate it with
``python -m repo_intelligence.graph.engineering_graph_persist``).
Without either the script fails CLEARLY with exit 6 and never falls
back to a fake model. The offline test suite never runs this script
against the network (tests/graph/test_m12_knowledge_demo.py wires it
with doubles only).
"""

from __future__ import annotations

import argparse
import json
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
    find_nodes,
    load_engineering_graph_document,
    project_engineering_graph,
    query_engineering_graph,
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

# The M12 forward path, in order (spec §4 vocabulary).
PATH_KINDS = ["DECLARES", "EXPOSED_BY", "CONFIGURED_BY"]

# Agent artifact subkinds (M2 vocabulary — repository-agnostic).
AGENT_SUBKINDS = (
    "agent_definition",
    "agent_command",
    "agent_config",
    "agent_skill",
)

SLICE_COUNT_KEYS = (
    "nodes_tool",
    "nodes_mcp_server",
    "rel_declares",
    "rel_exposed_by",
    "rel_configured_by",
)

# Spec §20 example objective (repo-aware by design; the answer must be
# grounded in graph/source evidence the agent actually retrieved).
DEFAULT_OBJECTIVE = (
    "Explain how a specific repository agent — such as the ontology "
    "agent — is connected to an MCP server, and identify the evidence "
    "supporting the connection. Start from the engineering graph: call "
    "query_engineering_graph with {} to see the node-kind summary, then "
    "run a bounded depth-3 traversal outward from the agent node along "
    "DECLARES, EXPOSED_BY and CONFIGURED_BY. Name every node id and "
    "relationship kind you found, quote the evidence paths each "
    "relationship cites, and say plainly when the graph shows no "
    "relationship for something. Corroborate with search_repository or "
    "read_file only when a graph result points at a file worth quoting."
)


# ---------------------------------------------------------------------------
# Artifact preflight (persistence, read-only, before any LLM)
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


def slice_counts(graph) -> Dict[str, int]:
    """M12 slice counters read directly from the projected graph."""
    node_kinds: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        kind = str(data.get("kind") or "")
        node_kinds[kind] = node_kinds.get(kind, 0) + 1
    edge_kinds: Dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        kind = str(data.get("kind") or "")
        edge_kinds[kind] = edge_kinds.get(kind, 0) + 1
    return {
        "nodes_tool": node_kinds.get("tool", 0),
        "nodes_mcp_server": node_kinds.get("mcp_server", 0),
        "rel_declares": edge_kinds.get("DECLARES", 0),
        "rel_exposed_by": edge_kinds.get("EXPOSED_BY", 0),
        "rel_configured_by": edge_kinds.get("CONFIGURED_BY", 0),
    }


def slice_present(counts: Dict[str, int]) -> bool:
    """True when the artifact carries every M12 slice counter > 0."""
    return all(counts.get(key, 0) > 0 for key in SLICE_COUNT_KEYS)


# ---------------------------------------------------------------------------
# Offline forward-path walk (the existing bounded query, no LLM)
# ---------------------------------------------------------------------------


def _complete_chain(
    start: str, relationships: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """First complete DECLARES → EXPOSED_BY → CONFIGURED_BY chain.

    Pure read of one query result's relationship records (already
    deterministically ordered by the query layer).
    """
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rel in relationships:
        out.setdefault((rel.get("kind"), rel.get("source")), []).append(rel)
    for declares in out.get(("DECLARES", start), []):
        tool = declares.get("target")
        for exposes in out.get(("EXPOSED_BY", tool), []):
            server = exposes.get("target")
            for configures in out.get(("CONFIGURED_BY", server), []):
                return [declares, exposes, configures]
    return None


def find_slice_path(graph) -> Optional[Dict[str, Any]]:
    """Walk the first agent (sorted, deterministic) whose bounded
    depth-3 outward query reaches a configuration through all three
    path kinds — the offline slice demonstration."""
    agents = find_nodes(
        graph, kind=["artifact/{}".format(k) for k in AGENT_SUBKINDS]
    )
    for entry in agents:
        agent_id = entry["id"]
        result = query_engineering_graph(
            graph,
            start_node_id=agent_id,
            direction="out",
            relationship_kinds=PATH_KINDS,
            max_depth=3,
        )
        hops = _complete_chain(agent_id, result.relationships)
        if hops:
            return {
                "agent": agent_id,
                "hops": hops,
                "query": dict(result.query),
            }
    return None


def structured_slice_result(report: Dict[str, Any]) -> Dict[str, Any]:
    """spec §21 structured result: finding / relationships / evidence.

    Every field comes from the graph records the bounded query already
    returned — nothing is re-derived or invented here.
    """
    hops = report["hops"]
    evidence_paths: List[str] = []
    relationships: List[Dict[str, Any]] = []
    for rel in hops:
        relationships.append(
            {
                "kind": rel.get("kind"),
                "source": rel.get("source"),
                "target": rel.get("target"),
            }
        )
        for item in rel.get("evidence") or []:
            path = str(item.get("path") or "")
            line = item.get("line_start")
            label = "{}:{}".format(path, line) if line else path
            if label and label not in evidence_paths:
                evidence_paths.append(label)
    finding = "{} --DECLARES--> {} --EXPOSED_BY--> {} --CONFIGURED_BY--> {}".format(
        hops[0].get("source"),
        hops[0].get("target"),
        hops[1].get("target"),
        hops[2].get("target"),
    )
    return {
        "finding": finding,
        "relationships": relationships,
        "evidence_paths": evidence_paths,
    }


# ---------------------------------------------------------------------------
# Stage classification (trace projection — nothing is re-derived)
# ---------------------------------------------------------------------------


def classify_stages(trace: Any) -> Dict[str, Any]:
    """Project the ONE completed trace into the spec §14 stage list."""
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


def exit_code_for(
    stages: Dict[str, Any], slice_path: Optional[Dict[str, Any]]
) -> int:
    """0 only when the slice path was found offline AND the run
    completed AND queried the engineering graph at least once — the
    M12 demonstration itself."""
    counts = stages["counts"]
    if (
        slice_path is None
        or not counts["answer_recorded"]
        or counts["graph_queries"] == 0
    ):
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
    print("Persisted engineering graph (M11.1 persistence)", file=sys.stdout)
    print(SECTION, file=sys.stdout)
    print("artifact: {}".format(_relative(path)))
    print("schema:   {}".format(document.get("schema_version")))
    provenance = document.get("provenance") or {}
    print("source:   {}".format(provenance.get("source_artifact")))
    print(
        "commit:   {}".format(
            (provenance.get("repository") or {}).get("commit_sha")
        )
    )
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


def print_slice_result(
    counts: Dict[str, int], result: Dict[str, Any], stream=None
) -> None:
    """Structured slice result (spec §21): finding / relationships /
    evidence_paths — offline, reviewer-answerable."""
    stream = stream or sys.stdout
    print("Knowledge slice (offline preflight, no LLM)", file=stream)
    print(SECTION, file=stream)
    print(
        "nodes:    {} tool, {} mcp_server".format(
            counts.get("nodes_tool"), counts.get("nodes_mcp_server")
        ),
        file=stream,
    )
    print(
        "edges:    {} DECLARES, {} EXPOSED_BY, {} CONFIGURED_BY".format(
            counts.get("rel_declares"),
            counts.get("rel_exposed_by"),
            counts.get("rel_configured_by"),
        ),
        file=stream,
    )
    print("finding:  {}".format(result["finding"]), file=stream)
    print("relationships:", file=stream)
    for rel in result["relationships"]:
        print(
            "  {}  {}  ->  {}".format(
                rel["kind"], rel["source"], rel["target"]
            ),
            file=stream,
        )
    print("evidence_paths:", file=stream)
    for label in result["evidence_paths"]:
        print("  {}".format(label), file=stream)
    print(
        "structured_result:",
        file=stream,
    )
    print(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False),
        file=stream,
    )
    print(file=stream)


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
        "llm reasoning:         {} iteration(s)".format(counts["iterations"]),
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
    print("M12 demonstration failed.", file=sys.stderr)
    print(message, file=sys.stderr)
    if detail is not None:
        print(
            "{}: {}".format(type(detail).__name__, detail), file=sys.stderr
        )
    return EXIT_EXECUTION_FAILED


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python agentrun_m12_knowledge_demo.py",
        description=(
            "M12 end-to-end demonstration: offline knowledge-slice "
            "preflight (bounded Agent → Tool → MCP Server → MCP "
            "Configuration path over the persisted graph) followed by "
            "one real-model investigation whose trace is projected into "
            "the spec §14 stages."
        ),
    )
    parser.add_argument(
        "--question",
        default=None,
        help="override the investigation objective (default: the spec "
        "§20 agent-to-server connection question)",
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
    # convention as the M9/M10/M11 demos)
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "M12 demonstration requires OPENAI_API_KEY.\n"
            "Copy .env.example to .env, set OPENAI_API_KEY (never commit "
            ".env), and re-run:\n"
            "    python agentrun_m12_knowledge_demo.py",
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

    # offline slice walk: the existing bounded query over the persisted
    # projection — read-only, deterministic, no LLM
    graph = project_engineering_graph(document)
    counts = slice_counts(graph)
    if not slice_present(counts):
        return _failed(
            "The persisted graph has no M12 knowledge slice (expected "
            "tool/mcp_server nodes and DECLARES/EXPOSED_BY/CONFIGURED_BY "
            "edges). Regenerate facts, model and graph:\n"
            "    python -m repo_intelligence.cli facts --snapshot "
            "artifacts/<id>/snapshot/repository_snapshot.json\n"
            "    python -m repo_intelligence.cli model\n"
            "    python -m repo_intelligence.graph."
            "engineering_graph_persist"
        )
    slice_path = find_slice_path(graph)
    if slice_path is None:
        return _failed(
            "No complete Agent → Tool → MCP Server → MCP Configuration "
            "path exists in the persisted graph; nothing to demonstrate."
        )
    slice_result = structured_slice_result(slice_path)

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
    print(
        "M12 — Agent → Tool → MCP Server → MCP Configuration (demonstration)"
    )
    print(RULE)
    print("Objective:")
    print(objective)
    print()
    print_graph_artifact(document, graph_path)
    print_slice_result(counts, slice_result)
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

    code = exit_code_for(stages, slice_path)
    if code == EXIT_SUCCESS:
        print(
            "exit code: 0 (slice path found offline; run completed; "
            "engineering graph queried {} time(s))".format(
                stages["counts"]["graph_queries"]
            ),
            file=sys.stdout,
        )
    else:
        if stages["counts"]["answer_recorded"]:
            print(
                "run completed but never called query_engineering_graph — "
                "the M12 demonstration requires at least one graph "
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
