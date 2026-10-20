"""M18 — end-to-end repository retrieval flow diagnostic.

First-class M18 deliverable (spec §2-§10, §34-§39): one offline,
human-runnable script that exposes EACH stage's actual output for:

    Query
      |
Repository Retrieval  -->  Engineering Graph
      |                          |
      +-----------+--------------+
                  v
         candidate evidence
                  v
    agent evidence selection   (OFFLINE / deterministic by default;
                                --live-reasoning is optional LLM)
                  v
             read_file()
                  v
          verified evidence

Usage:

    python agentrun_repository_retrieval_flow_demo.py --offline
    python agentrun_repository_retrieval_flow_demo.py \\
        --query "How do the agents, tools, and MCP servers interact?" \\
        --offline
    python agentrun_repository_retrieval_flow_demo.py \\
        --output artifacts/retrieval_flow/latest/

Options (§36):  --query  --top-k  --output  --offline
Optional:       --live-reasoning  --verbose
plus test/application context: --output-root --snapshot-id
--embedding-model --store --no-build --llm-model.

Sections printed exactly per §35 (real values only — nothing faked):

    QUERY
    1. REPOSITORY RETRIEVAL
    2. ENGINEERING GRAPH
    3. COMBINED EVIDENCE
    4. AGENT EVIDENCE SELECTION
    5. READ_FILE VERIFICATION
    6. VERIFIED EVIDENCE
    SUMMARY

Index handling: the index is built EXPLICITLY as its own printed step
when missing (never as a query side effect; --no-build forbids it).
The flow never requires M17 (§38) and never calls OpenAI unless
--live-reasoning is passed (§10).

Exit codes (M13/M10 demo convention):
    0  flow completed; Summary verdict PASS
    1  flow completed; Summary verdict FAIL (a mandatory stage was
       empty or verification did not cover every selected item)
    6  configuration failure (missing run/snapshot/identity, missing
       index with --no-build, missing OPENAI_API_KEY with
       --live-reasoning, invalid arguments, LLM error)

With --output (or always, defaulting to
artifacts/retrieval_flow/<timestamp>/) the complete intermediate
result is written as flow_result.json + flow_summary.md — never
under runtime/ (§9).

Unexpected programming errors are NOT caught: they propagate with a
traceback rather than masquerading as a flow failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from repo_intelligence.artifacts import (  # noqa: E402
    default_artifacts_root,
    repository_dir,
    resolve_output_root,
)
from repo_intelligence.context.models import (  # noqa: E402
    ArchitectureContextError,
    discover_snapshot_id,
)
from repo_intelligence.retrieval import (  # noqa: E402
    active_index_identity,
    embedder_for,
    index_run,
    resolve_index_identity,
)
from repo_intelligence.retrieval.errors import (  # noqa: E402
    RAG_INDEX_NOT_FOUND,
    RetrievalError,
)
from repo_intelligence.tools.query_engineering_graph import (  # noqa: E402
    query_engineering_graph_tool,
)
from repo_intelligence.tools.read_file import read_file  # noqa: E402
from repo_intelligence.tools.search_repository import (  # noqa: E402
    search_repository,
)

DEFAULT_QUERY = "How do the agents, tools, and MCP servers interact?"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
GRAPH_START_NODES = 2
GRAPH_MAX_DEPTH = 1
BANNER = "=" * 60
DUMP_LIMIT = 2000
VERBOSE_LIMIT = 50000

__all__ = ["main", "run_flow"]


# ---------------------------------------------------------------------------
# Console helpers — labels on their own line, real values underneath
# ---------------------------------------------------------------------------

_VERBOSE = False


def _banner(title: str) -> None:
    print(BANNER)
    print(title)
    print(BANNER)


def _section(title: str) -> None:
    _banner(title)
    print("")


def _field(label: str, value: Any) -> None:
    print("{}:".format(label))
    print(value)
    print("")


def _dump(label: str, payload: Any) -> None:
    limit = VERBOSE_LIMIT if _VERBOSE else DUMP_LIMIT
    text = json.dumps(payload, indent=2, sort_keys=False)
    if len(text) > limit:
        text = text[:limit] + "\n...[truncated]"
    print("{}:\n{}".format(label, text))


def _lines_range(
    line_start: Optional[int], line_end: Optional[int]
) -> str:
    if line_start is None and line_end is None:
        return "all"
    if line_end is None:
        return "{}".format(line_start)
    if line_start is None:
        return "-{}".format(line_end)
    return "{}-{}".format(line_start, line_end)


def _numbered_lines(
    content: str, first_line: Optional[int]
) -> str:
    """Render content as ``42 | text`` rows (spec §7)."""
    start = first_line if isinstance(first_line, int) else 1
    rows = content.splitlines() or [""]
    return "\n".join(
        "{} | {}".format(start + offset, row)
        for offset, row in enumerate(rows)
    )


def _primary_source(item: Dict[str, Any]) -> str:
    origins = item.get("origins") or [item.get("origin")]
    if "repository-retrieval" in origins:
        return "RAG"
    return "GRAPH"


# ---------------------------------------------------------------------------
# Evidence handling
# ---------------------------------------------------------------------------


def _retrieval_candidates(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Typed candidates from the repository retrieval stage."""
    candidates: List[Dict[str, Any]] = []
    for rank, item in enumerate(results):
        path = item.get("path") or item.get("file_path")
        if not isinstance(path, str) or not path:
            continue
        candidates.append(
            {
                "origin": "repository-retrieval",
                "origins": ["repository-retrieval"],
                "rank": rank,
                "similarity_score": item.get("similarity_score"),
                "chunk_id": item.get("chunk_id"),
                "path": path,
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "excerpt": item.get("excerpt", ""),
            }
        )
    return candidates


def _graph_candidates(
    evidence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Typed candidates from engineering graph evidence refs."""
    candidates: List[Dict[str, Any]] = []
    for item in evidence:
        path = item.get("path")
        if not isinstance(path, str) or not path:
            continue
        candidates.append(
            {
                "origin": "engineering-graph",
                "origins": ["engineering-graph"],
                "from": item.get("origin"),
                "path": path,
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "classification": item.get("classification"),
            }
        )
    return candidates


def _merge_candidates(
    retrieval: List[Dict[str, Any]], graph: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Deduplicate on (path, line_start, line_end); keep all origins."""
    merged: List[Dict[str, Any]] = []
    index: Dict[Any, int] = {}
    for candidate in retrieval + graph:
        key = (
            candidate.get("path"),
            candidate.get("line_start"),
            candidate.get("line_end"),
        )
        slot = index.get(key)
        if slot is None:
            index[key] = len(merged)
            merged.append(candidate)
            continue
        existing = merged[slot]
        for origin in candidate.get("origins", []):
            if origin not in existing["origins"]:
                existing["origins"].append(origin)
        if (
            existing.get("similarity_score") is None
            and candidate.get("similarity_score") is not None
        ):
            existing["similarity_score"] = candidate["similarity_score"]
    return merged


def _select_deterministic(
    candidates: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """OFFLINE selection: rank each source deterministically and
    guarantee both sources a voice — ceil(top_k/2) retrieval slots
    (score desc, stable tie-break), the rest graph evidence — so the
    flow demonstrates complementary, not competing, evidence (§5).
    """
    retrieval_ranked = sorted(
        [c for c in candidates if c.get("origin") == "repository-retrieval"],
        key=lambda c: (
            -(c.get("similarity_score") or 0.0),
            c.get("path") or "",
            c.get("line_start") or 0,
        ),
    )
    graph_ranked = sorted(
        [c for c in candidates if c.get("origin") == "engineering-graph"],
        key=lambda c: (
            c.get("path") or "",
            c.get("line_start") or 0,
        ),
    )
    rag_quota = (top_k + 1) // 2
    graph_quota = top_k - rag_quota
    selected = retrieval_ranked[:rag_quota]
    selected.extend(graph_ranked[:graph_quota])
    if len(selected) < top_k:  # fill shortages from the other source
        have = {id(item) for item in selected}
        rest = [
            item
            for item in retrieval_ranked + graph_ranked
            if id(item) not in have
        ]
        selected.extend(rest[: top_k - len(selected)])
    for position, item in enumerate(selected, start=1):
        if item.get("similarity_score") is not None:
            item["selection_reason"] = (
                "OFFLINE deterministic rank #{position} "
                "(semantic score {score})".format(
                    position=position,
                    score=item["similarity_score"],
                )
            )
        else:
            item["selection_reason"] = (
                "OFFLINE deterministic rank #{position} "
                "(graph evidence)".format(position=position)
            )
    return selected


def _select_live(
    candidates: List[Dict[str, Any]],
    query: str,
    top_k: int,
    model: str,
) -> List[Dict[str, Any]]:
    """Optional LLM selection (requires OPENAI_API_KEY; exit 6 else)."""
    from repo_intelligence.llm import (  # local: only with live mode
        LLMError,
        LLMRequest,
        Message,
        StructuredOutputSpec,
    )
    from repo_intelligence.llm.config import load_env
    from repo_intelligence.llm.openai_client import OpenAILLMClient

    load_env()
    client = OpenAILLMClient(model=model,http_client=httpx.Client(verify=False))
    rows = [
        {
            "index": index,
            "path": item.get("path"),
            "line_start": item.get("line_start"),
            "line_end": item.get("line_end"),
            "source": _primary_source(item),
        }
        for index, item in enumerate(candidates)
    ]
    prompt = (
        "Question: {query}\n\n"
        "Candidate evidence items (JSON):\n{rows}\n\n"
        "Select the {top_k} items most useful to answer the question. "
        "Return selected as a list of objects with keys index "
        "(integer) and reason (short string).".format(
            query=query, rows=json.dumps(rows, indent=1), top_k=top_k
        )
    )
    spec = StructuredOutputSpec(
        name="evidence_selection",
        schema={
            "type": "object",
            "properties": {
                "selected": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["index", "reason"],
                    },
                }
            },
            "required": ["selected"],
        },
    )
    response = client.complete(
        LLMRequest(
            messages=[Message(role="user", content=prompt)],
            structured_output=spec,
        )
    )
    payload = response.structured_output
    if not isinstance(payload, dict) or not isinstance(
        payload.get("selected"), list
    ):
        raise LLMError(
            "the model did not return the evidence_selection payload."
        )
    selected: List[Dict[str, Any]] = []
    for entry in payload["selected"]:
        if not isinstance(entry, dict):
            continue
        raw_index = entry.get("index")
        if not isinstance(raw_index, int):
            continue
        if raw_index < 0 or raw_index >= len(candidates):
            continue
        item = dict(candidates[raw_index])
        item["selection_reason"] = "LIVE {model}: {reason}".format(
            model=model, reason=str(entry.get("reason", ""))[:200]
        )
        selected.append(item)
        if len(selected) >= top_k:
            break
    if not selected:
        raise LLMError("the model selected no usable evidence indices.")
    return selected


# ---------------------------------------------------------------------------
# Flow stages (return real stage data; printing is separate)
# ---------------------------------------------------------------------------


def _retrieval_stage(
    args: argparse.Namespace,
    output_root: Path,
    snapshot_id: str,
    checkout: Path,
    identity: Any,
) -> Dict[str, Any]:
    """Repository retrieval with an EXPLICIT index build when missing."""
    from dataclasses import replace

    embedder = None
    if args.embedding_model:
        embedder = embedder_for(args.embedding_model)
        identity = replace(identity, embedding_model=embedder.name)

    def _search() -> Any:
        return search_repository(
            {"query": args.query, "max_results": args.top_k},
            repository_root=checkout,
            output_root=output_root,
            store_root=args.store,
        )

    result = _search()
    built = None
    if not result.success and result.error is not None:
        if result.error.code != RAG_INDEX_NOT_FOUND:
            raise RetrievalError(result.error.code, result.error.message)
        if args.no_build:
            raise RetrievalError(
                result.error.code,
                "index missing and --no-build was requested: {}".format(
                    result.error.message
                ),
            )
        # Explicit build (printed as its own step); queries never
        # build implicitly.
        report = index_run(
            output_root,
            snapshot_id,
            store_root=args.store,
            embedder=embedder,
        )
        built = report.to_dict()
        result = _search()
    if not result.success or result.error is not None:
        message = (
            result.error.message if result.error is not None else "unknown"
        )
        code = (
            result.error.code if result.error is not None else "SEARCH_FAILED"
        )
        raise RetrievalError(code, message)

    data = dict(result.data or {})
    return {
        "data": data,
        "identity": identity.to_dict(),
        "source": result.source,
        "built": built,
        "candidates": _retrieval_candidates(list(data.get("results", []))),
    }


def _graph_stage(output_root: Path) -> Dict[str, Any]:
    """Deterministic graph traversal: kinds census + bounded walk."""
    summary = query_engineering_graph_tool({}, output_root=output_root)
    if not summary.success or summary.error is not None:
        code = summary.error.code if summary.error else "ENGINEERING_GRAPH"
        message = summary.error.message if summary.error else "unknown"
        print(
            "graph summary failed: {code} {message}".format(
                code=code, message=message
            )
        )
        return {"ok": False, "code": code, "message": message}
    kinds = list((summary.data or {}).get("kinds", []))
    counts = (summary.data or {}).get("counts", {})
    if _VERBOSE:
        _dump("graph kinds census", summary.data)

    start_ids: List[str] = []
    for kind in kinds:
        for node_id in kind.get("sample_node_ids", []):
            if node_id not in start_ids:
                start_ids.append(node_id)
        if len(start_ids) >= GRAPH_START_NODES:
            break
    start_ids = start_ids[:GRAPH_START_NODES]

    relationships: Dict[str, Dict[str, Any]] = {}
    evidence: List[Dict[str, Any]] = []
    traversals: List[Dict[str, Any]] = []
    for node_id in start_ids:
        walked = query_engineering_graph_tool(
            {
                "start_node_id": node_id,
                "direction": "both",
                "max_depth": GRAPH_MAX_DEPTH,
            },
            output_root=output_root,
        )
        if not walked.success or walked.error is not None:
            code = walked.error.code if walked.error else "ENGINEERING_GRAPH"
            print(
                "walk {nid} failed: {code}".format(nid=node_id, code=code)
            )
            continue
        payload = walked.data or {}
        if _VERBOSE:
            _dump(
                "graph walk {}".format(node_id),
                payload,
            )
        for rel in payload.get("relationships", []):
            relationships[str(rel.get("id"))] = rel
        for item in payload.get("evidence", []):
            evidence.append(item)
        traversals.append(
            {
                "start_node_id": node_id,
                "nodes_returned": len(payload.get("nodes", [])),
                "relationships_returned": len(
                    payload.get("relationships", [])
                ),
                "evidence_returned": len(payload.get("evidence", [])),
            }
        )
    rel_by_origin = {
        "relationship:{}".format(rel_id): rel
        for rel_id, rel in relationships.items()
    }
    return {
        "ok": True,
        "kinds": kinds,
        "counts": counts,
        "traversals": traversals,
        "relationships": list(relationships.values()),
        "rel_by_origin": rel_by_origin,
        "evidence": evidence,
        "candidates": _graph_candidates(evidence),
    }


def _verification_stage(
    selected: List[Dict[str, Any]], output_root: Path
) -> List[Dict[str, Any]]:
    """Re-read every selected candidate through the read_file tool."""
    verified: List[Dict[str, Any]] = []
    for item in selected:
        arguments: Dict[str, Any] = {"path": item.get("path")}
        if item.get("line_start") is not None:
            arguments["line_start"] = item["line_start"]
        if item.get("line_end") is not None:
            arguments["line_end"] = item["line_end"]
        result = read_file(arguments, output_root=output_root)
        entry: Dict[str, Any] = {
            "path": item.get("path"),
            "line_start": item.get("line_start"),
            "line_end": item.get("line_end"),
            "verified": bool(result.success),
            "selection_reason": item.get("selection_reason"),
            "origins": list(item.get("origins", [])),
            "from": item.get("from"),
            "chunk_id": item.get("chunk_id"),
            "similarity_score": item.get("similarity_score"),
        }
        if result.success and result.data is not None:
            entry["source"] = result.source
            content = str(result.data.get("content", ""))
            entry["content"] = content
            entry["content_chars"] = len(content)
            entry["read_line_start"] = result.data.get("line_start")
            entry["read_line_end"] = result.data.get("line_end")
        else:
            entry["error"] = (
                result.error.code
                if result.error is not None
                else "READ_FAILED"
            )
            entry["message"] = (
                result.error.message if result.error is not None else ""
            )
        verified.append(entry)
    return verified


# ---------------------------------------------------------------------------
# Printing (spec §3-§8, §35 — labels + real values only)
# ---------------------------------------------------------------------------


def _print_query(
    identity: Any,
    snapshot_id: str,
    query: str,
    top_k: int,
    mode: str,
) -> None:
    _section("QUERY")
    _field("Repository", identity.repository_name)
    _field("Commit", identity.commit_sha)
    _field("Snapshot", snapshot_id)
    _field("Question", query)
    _field("Reasoning mode", mode)
    _field("top-k", top_k)


def _print_retrieval(retrieval: Dict[str, Any]) -> None:
    _section("1. REPOSITORY RETRIEVAL")
    data = retrieval["data"]
    results = list(data.get("results", []))
    if retrieval.get("built") is not None:
        print("index build (explicit, never a query side effect):")
        print("")
        _dump("build report", retrieval["built"])
    _field("count", data.get("returned", len(results)))
    print("total_matches: {}".format(data.get("total_matches")))
    print("truncated: {}".format(data.get("truncated")))
    print("")
    for position, item in enumerate(results, start=1):
        print("[{}]".format(position))
        _field("file", item.get("path"))
        _field(
            "lines",
            _lines_range(item.get("line_start"), item.get("line_end")),
        )
        _field("score", item.get("similarity_score"))
        _field("chunk", item.get("chunk_id"))
        _field("excerpt", item.get("excerpt", ""))
        _field("repository", item.get("repository_name"))
        _field("commit", item.get("commit_sha"))
    if _VERBOSE:
        _dump("retrieval data", data)


def _print_graph(graph: Dict[str, Any]) -> None:
    _section("2. ENGINEERING GRAPH")
    relationships = list(graph.get("relationships", []))
    _field("count", len(relationships))
    if not graph.get("ok", False):
        print("error: {} {}".format(graph.get("code"), graph.get("message")))
        print("")
        return
    _field("counts", graph.get("counts", {}))
    for position, rel in enumerate(relationships, start=1):
        print("[{}]".format(position))
        _field("source", rel.get("source"))
        _field("relationship", rel.get("kind"))
        _field("target", rel.get("target"))
        _field("evidence", rel.get("evidence", []))


def _print_combined(
    retrieval: Dict[str, Any],
    graph: Dict[str, Any],
    combined: List[Dict[str, Any]],
    identity: Any,
) -> None:
    _section("3. COMBINED EVIDENCE")
    retrieval_count = len(retrieval.get("candidates", []))
    graph_count = len(graph.get("relationships", []))
    _field("Repository Retrieval", "{} candidates".format(retrieval_count))
    _field("Engineering Graph", "{} relationships".format(graph_count))
    _field("Combined evidence", "{} items".format(len(combined)))
    for position, item in enumerate(combined, start=1):
        print("[{}]".format(position))
        _field(
            "evidence_type",
            "RAG"
            if item.get("origin") == "repository-retrieval"
            else "GRAPH",
        )
        _field("source", item.get("chunk_id") or item.get("from"))
        _field(
            "location",
            "{}:{}".format(
                item.get("path"),
                _lines_range(
                    item.get("line_start"), item.get("line_end")
                ),
            ),
        )
        _field("repository", identity.repository_name)
        _field("commit", identity.commit_sha)
        _field(
            "provenance",
            "{repo}@{commit} ({snapshot})".format(
                repo=identity.repository_name,
                commit=identity.commit_sha,
                snapshot=identity.snapshot_id or "-",
            ),
        )


def _print_selection(selected: List[Dict[str, Any]]) -> None:
    _section("4. AGENT EVIDENCE SELECTION")
    print("Selected evidence for verification:")
    print("")
    for position, item in enumerate(selected, start=1):
        print("[{}]".format(position))
        _field("file", item.get("path"))
        _field(
            "lines",
            _lines_range(item.get("line_start"), item.get("line_end")),
        )
        _field("reason", item.get("selection_reason"))
        _field("source", _primary_source(item))


def _print_verification(
    verified: List[Dict[str, Any]], identity: Any
) -> None:
    _section("5. READ_FILE VERIFICATION")
    for position, entry in enumerate(verified, start=1):
        print("[{}]".format(position))
        print("Requested:")
        if entry.get("line_start") is None and entry.get(
            "line_end"
        ) is None:
            print(entry.get("path"))
        else:
            print(
                "{}:L{}-{}".format(
                    entry.get("path"),
                    entry.get("line_start"),
                    entry.get("line_end"),
                )
            )
        print("")
        if entry.get("verified"):
            print("Returned:")
            print(
                _numbered_lines(
                    str(entry.get("content", "")),
                    entry.get("read_line_start"),
                )
            )
            print("")
            _field("Repository", identity.repository_name)
            _field("Commit", identity.commit_sha)
            _field("Verification", "SUCCESS")
        else:
            print("Returned:")
            print(entry.get("error"))
            print(entry.get("message"))
            print("")
            _field("Verification", "FAILED")


def _verified_blocks(
    verified: List[Dict[str, Any]],
    graph: Dict[str, Any],
    identity: Any,
) -> List[Dict[str, Any]]:
    """Typed final evidence entries (CODE vs GRAPH_RELATIONSHIP)."""
    rel_by_origin = graph.get("rel_by_origin", {})
    blocks: List[Dict[str, Any]] = []
    for entry in verified:
        rel = None
        origin = entry.get("from")
        if (
            isinstance(origin, str)
            and origin.startswith("relationship:")
            and "repository-retrieval" not in entry.get("origins", [])
        ):
            rel = rel_by_origin.get(origin)
        if rel is not None:
            supporting = (rel.get("evidence") or [{}])[0]
            blocks.append(
                {
                    "type": "GRAPH_RELATIONSHIP",
                    "source": rel.get("source"),
                    "relationship": rel.get("kind"),
                    "target": rel.get("target"),
                    "supporting_source": supporting.get("path"),
                    "supporting_lines": _lines_range(
                        supporting.get("line_start"),
                        supporting.get("line_end"),
                    ),
                    "repository_name": identity.repository_name,
                    "commit_sha": identity.commit_sha,
                    "path": entry.get("path"),
                    "line_start": entry.get("line_start"),
                    "line_end": entry.get("line_end"),
                }
            )
        else:
            blocks.append(
                {
                    "type": "CODE",
                    "repository_name": identity.repository_name,
                    "commit_sha": identity.commit_sha,
                    "path": entry.get("path"),
                    "line_start": entry.get("line_start"),
                    "line_end": entry.get("line_end"),
                    "content": str(entry.get("content", "")),
                    "verified": bool(entry.get("verified")),
                }
            )
    return blocks


def _print_verified(blocks: List[Dict[str, Any]]) -> None:
    _section("6. VERIFIED EVIDENCE")
    for position, block in enumerate(blocks, start=1):
        print("Evidence {}".format(position))
        print("-" * 10)
        _field("Type", block.get("type"))
        if block.get("type") == "GRAPH_RELATIONSHIP":
            _field("Source", block.get("source"))
            _field("Relationship", block.get("relationship"))
            _field("Target", block.get("target"))
            _field(
                "Supporting source",
                "{}:{}".format(
                    block.get("supporting_source"),
                    block.get("supporting_lines"),
                ),
            )
        else:
            _field("Repository", block.get("repository_name"))
            _field("Commit", block.get("commit_sha"))
            _field("File", block.get("path"))
            _field(
                "Lines",
                _lines_range(
                    block.get("line_start"), block.get("line_end")
                ),
            )
            _field("Content", str(block.get("content", "")))
    print(BANNER)
    print("FLOW COMPLETE")
    print(BANNER)
    print("")


def _print_summary(checks: Dict[str, bool]) -> None:
    _section("SUMMARY")
    labels = [
        ("Repository retrieval", "retrieval"),
        ("Engineering graph", "graph"),
        ("Evidence selection", "selection"),
        ("Read-file verification", "verification"),
        ("Provenance", "provenance"),
    ]
    for label, key in labels:
        print(
            "{}: {}".format(label, "PASS" if checks[key] else "FAIL")
        )
    print("")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print("Overall: {}".format(verdict))


# ---------------------------------------------------------------------------
# Output artifacts (§9 — never under runtime/)
# ---------------------------------------------------------------------------


def _write_output(
    out_arg: Optional[str], payload: Dict[str, Any], lines: List[str]
) -> List[str]:
    """Write flow_result.json + flow_summary.md; return saved paths."""
    if out_arg:
        target = Path(out_arg)
        if target.suffix == ".json":
            json_path = target
            md_path = target.with_name(target.stem + ".md")
        else:
            json_path = target / "flow_result.json"
            md_path = target / "flow_summary.md"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        directory = default_artifacts_root() / "retrieval_flow" / stamp
        json_path = directory / "flow_result.json"
        md_path = directory / "flow_summary.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(json_path), str(md_path)]


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


def run_flow(args: argparse.Namespace) -> int:
    """Run every stage; return the process exit code."""
    global _VERBOSE
    _VERBOSE = bool(args.verbose)
    output_root = resolve_output_root(
        Path(args.output_root) if args.output_root else None
    )
    live = bool(args.live_reasoning)
    mode = "LIVE / {}".format(args.llm_model) if live else (
        "OFFLINE / deterministic"
    )
    try:
        snapshot_id = args.snapshot_id or discover_snapshot_id(output_root)
    except ArchitectureContextError as exc:
        print("no run snapshot under {}: {}".format(output_root, exc))
        return 6
    checkout = repository_dir(output_root, snapshot_id)

    # Identity first: the QUERY banner needs Repository/Commit.
    try:
        identity = resolve_index_identity(checkout, output_root)
        if not args.embedding_model:
            identity = active_index_identity(identity, args.store)
    except RetrievalError as exc:
        print(
            "identity resolution failed: {} {}".format(
                exc.code, exc.message
            )
        )
        return 6

    _print_query(identity, snapshot_id, args.query, args.top_k, mode)

    try:
        retrieval = _retrieval_stage(
            args, output_root, snapshot_id, checkout, identity
        )
    except RetrievalError as exc:
        print("retrieval failed: {} {}".format(exc.code, exc.message))
        return 6
    identity_data = retrieval["identity"]
    _print_retrieval(retrieval)

    graph = _graph_stage(output_root)
    _print_graph(graph)

    retrieval_candidates = retrieval["candidates"]
    graph_candidates = list(graph.get("candidates", []))
    combined = _merge_candidates(retrieval_candidates, graph_candidates)
    _print_combined(retrieval, graph, combined, identity)

    if live:
        from repo_intelligence.llm import LLMError

        try:
            selected = _select_live(
                combined, args.query, args.top_k, args.llm_model
            )
        except LLMError as exc:
            print("live reasoning unavailable: {}".format(exc))
            return 6
    else:
        selected = _select_deterministic(combined, args.top_k)
    _print_selection(selected)

    verified = _verification_stage(selected, output_root)
    _print_verification(verified, identity)

    blocks = _verified_blocks(verified, graph, identity)
    _print_verified(blocks)

    checks: Dict[str, bool] = {
        "retrieval": len(retrieval_candidates) > 0,
        "graph": len(graph.get("relationships", [])) > 0,
        "selection": len(selected) > 0 and len(combined) > 0,
        "verification": (
            len(verified) == len(selected)
            and all(entry["verified"] for entry in verified)
        ),
        "provenance": bool(
            identity_data.get("repository_name")
            and identity_data.get("commit_sha")
        ),
    }
    _print_summary(checks)
    verdict = "PASS" if all(checks.values()) else "FAIL"

    payload: Dict[str, Any] = {
        "query": args.query,
        "repository": {
            "name": identity_data.get("repository_name"),
            "commit_sha": identity_data.get("commit_sha"),
            "snapshot_id": identity_data.get("snapshot_id"),
        },
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval": {
            "count": retrieval["data"].get("returned", 0),
            "results": retrieval["data"].get("results", []),
            "total_matches": retrieval["data"].get("total_matches"),
            "truncated": retrieval["data"].get("truncated"),
            "index": retrieval["data"].get("retrieval"),
            "source": retrieval["source"],
            "explicit_build": retrieval["built"],
        },
        "graph": {
            "relationships": graph.get("relationships", []),
            "counts": graph.get("counts", {}),
            "kinds": graph.get("kinds", []),
            "traversals": graph.get("traversals", []),
            "evidence": graph.get("evidence", []),
        },
        "combined_evidence": combined,
        "selected_evidence": selected,
        "verified_evidence": blocks,
        "verification": verified,
        "validation": {
            "retrieval": checks["retrieval"],
            "graph": checks["graph"],
            "selection": checks["selection"],
            "verification": checks["verification"],
            "provenance": checks["provenance"],
        },
        "checks": checks,
        "verdict": verdict,
    }
    lines = [
        "# Repository retrieval flow — {}".format(verdict),
        "",
        "- mode: {}".format(mode),
        "- query: {}".format(args.query),
        "- repository: {} @ {}".format(
            identity_data.get("repository_name"),
            str(identity_data.get("commit_sha"))[:12],
        ),
        "- snapshot: {}".format(identity_data.get("snapshot_id")),
        "- retrieval results: {}".format(len(retrieval_candidates)),
        "- graph relationships: {}".format(
            len(graph.get("relationships", []))
        ),
        "- combined evidence: {}".format(len(combined)),
        "- selected: {}".format(len(selected)),
        "- verified: {}".format(
            sum(1 for entry in verified if entry["verified"])
        ),
        "",
        "## Validation",
        "",
    ]
    for key in sorted(checks):
        lines.append(
            "- {}: {}".format(key, "PASS" if checks[key] else "FAIL")
        )
    saved = _write_output(args.output, payload, lines)
    for path in saved:
        print("saved: {}".format(path))
    return 0 if verdict == "PASS" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentrun_repository_retrieval_flow_demo.py",
        description=(
            "M18 diagnostic: repository retrieval -> engineering graph "
            "-> combined evidence -> evidence selection -> read_file "
            "-> verified evidence (offline by default; §35 output)."
        ),
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no OpenAI call (default behavior; kept for spec parity)",
    )
    parser.add_argument(
        "--live-reasoning",
        action="store_true",
        help="LLM evidence selection (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also dump each stage's raw intermediate JSON",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="fail instead of building a missing index explicitly",
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.top_k < 1 or args.top_k > 50:
        print("--top-k must be between 1 and 50.")
        return 6
    if args.live_reasoning and args.offline:
        print("choose one of --offline / --live-reasoning.")
        return 6
    if args.live_reasoning:
        # Fail fast before any snapshot discovery: load .env first so
        # the key is judged from the final environment (a checkout
        # carrying .env would otherwise succeed all the way through
        # live selection), then report a missing key as configuration
        # (exit 6) rather than a mid-run failure.
        from repo_intelligence.llm.config import (
            get_openai_api_key,
            load_env,
        )

        load_env()
        if not get_openai_api_key():
            print(
                "live reasoning unavailable: OPENAI_API_KEY is not "
                "configured (set it in the environment or in .env)."
            )
            return 6
    return run_flow(args)


if __name__ == "__main__":
    sys.exit(main())
