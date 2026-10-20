"""M8.6 — bounded ReAct investigation test (offline, deterministic).

Proves that a LATER tool action is selected BECAUSE OF an earlier
observation — adaptive investigation through the EXISTING Agent loop:

    Decision → Action → Observation → Decision → Action → Observation
    → … → Final Answer

The script runs no orchestration of its own. It calls the real
``Agent.run(...)``, which owns the M8.4 multi-step loop under the
single M8.5 tool-call limit (``AgentExecutionConfig.max_tool_calls`` —
there is no second budget), through the real application dispatcher
and the four read-only tools.

The LLM is a scripted offline fake whose every response is
conditional on the expected prior observation actually being present
in the request — a missing observation raises AssertionError (this
script FAILS) — and whose action arguments are DERIVED from that
observed content:

    action 2 (search query)    ← mechanism symbol read from observation 1
    action 3 (read path/lines) ← first search result read from observation 2
    final answer               ← cites observations 1-3 (path, line
                                  range, query, match count, excerpt)

Evidence is real: the pinned fixture's ``repository/`` holds files
COPIED FROM THIS PROJECT (README.md, src/repo_intelligence/agent) and
its ``architecture_context.md`` describes the real mechanism — only
the artifacts root is redirected (a temp directory); production run
discovery and tool resolution are unchanged. No network, no API key,
no repository mutation, nothing written under artifacts/. The run's
execution trace (observable events only) is persisted through the
existing JsonFileTraceSink.

Checks (all required):

    1  Observation 1 carried into request 2 (ToolResult verbatim)
    2  Action 2 derived from observation 1 (search query)
    3  Observations 1+2 carried into request 3
    4  Action 3 derived from observation 2 (read path + line range)
    5  Three evidence actions before the final answer
    6  Final answer cites the gathered evidence
    7  Bounded by the single tool-call limit (max_tool_calls)
    8  Execution trace: observable events only (no hidden reasoning)

Usage (from the project root):

    python agentrun_react_test.py

Exit codes: 0 = all checks passed, 6 = failure.

See also: agentrun_test.py — retained as the separate M8.4/M8.5
learning/debugging harness (per-step capability checks, full cycle
printing, optional live run). This script answers a different
question and replaces neither.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import repo_intelligence.artifacts as artifacts  # noqa: E402
from repo_intelligence.agent import (  # noqa: E402
    Agent,
    AgentExecutionConfig,
    AgentTrace,
    JsonFileTraceSink,
)
from repo_intelligence.agent.trace import BANNED_REASONING_FIELDS  # noqa: E402
from repo_intelligence.llm import (  # noqa: E402
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
)
from repo_intelligence.tool_dispatcher import default_tool_dispatcher  # noqa: E402
from repo_intelligence.tools import ToolResult  # noqa: E402

EXIT_OK = 0
EXIT_FAIL = 6
TOTAL_CHECKS = 8
NAME_WIDTH = 46
RULE = "-" * 72

TOOL_CALL_LIMIT = 8

OBJECTIVE = (
    "How does this repository bound the Agent's investigation loop "
    "(the tool-call limit)?"
)

RUN_ID = "run_m86_react"
FIXTURE_COMMIT = "f" * 40

# Real project content copied into the fixture's pinned repository/ —
# the search/read evidence is genuine, never a fictional fact.
COPIED_PATHS = ("README.md", "src/repo_intelligence/agent")

# The mechanism description the first observation must carry. The
# extraction rules below read VALUES from it; nothing here hard-codes
# what the model must search for.
MECHANISM_SECTION = """## Mechanism

- The bounded Agent loop lives in `src/repo_intelligence/agent/agent.py`.
- The single tool-call limit is `max_tool_calls`
  (`AgentExecutionConfig.max_tool_calls`, default 8).
"""

# Field names that must never appear as trace/request keys (the user's
# explicit M8.6 list on top of the trace module's own ban list).
EXTRA_BANNED_FIELDS = ("reasoning_trace",)

READ_ONLY_TOOLS = (
    "get_architecture_context",
    "read_file",
    "search_repository",
)


# ---------------------------------------------------------------------------
# Fixture: a pinned run whose repository/ = real files from this project
# ---------------------------------------------------------------------------


def _context_text() -> str:
    return (
        "# Architecture Context\n"
        "\n"
        "- Contract: `architecture_context.v1`\n"
        "\n"
        "## Repository\n"
        "\n"
        "- Name: `repository-intelligence`\n"
        "- Pinned commit: {}\n"
        "- Snapshot: {}\n"
        "\n"
        "{}"
    ).format(FIXTURE_COMMIT, RUN_ID, MECHANISM_SECTION)


def _build_pinned_run(root: Path) -> Path:
    """Build ``<root>/<RUN_ID>`` (facts/architecture/context/snapshot +
    a ``repository/`` of real copied files). Only the layout is created
    here — every tool still resolves it through the production path."""
    run_dir = root / RUN_ID
    (run_dir / "facts").mkdir(parents=True)
    (run_dir / "architecture").mkdir(parents=True)
    (run_dir / "snapshot").mkdir(parents=True)
    (run_dir / "architecture_context").mkdir()
    (run_dir / "repository").mkdir()
    (run_dir / "facts" / "repository_facts.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture" / "architecture_model.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_dir / "architecture_context" / "architecture_context.md").write_text(
        _context_text(), encoding="utf-8"
    )
    # Provenance describes the fixture (no invented repository URL).
    (run_dir / "snapshot" / "repository_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "repository_snapshot.v1",
                "repository": {"name": "repository-intelligence"},
                "snapshot": {"ref": "main", "commit_sha": FIXTURE_COMMIT},
            }
        ),
        encoding="utf-8",
    )
    repo = run_dir / "repository"
    for relative in COPIED_PATHS:
        source = PROJECT_ROOT / relative
        destination = repo / relative
        if source.is_dir():
            shutil.copytree(
                str(source),
                str(destination),
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
    return run_dir


# ---------------------------------------------------------------------------
# Observation-derived arguments (values come from the observation)
# ---------------------------------------------------------------------------


def _mechanism_symbol(context: str) -> str:
    """The investigation token: first symbol-shaped code span in the
    observation's ``## Mechanism`` section."""
    return _mechanism_span(context, r"[A-Za-z_][A-Za-z0-9_]*", "symbol")


def _source_file(context: str) -> str:
    """The source file the observation points at (``*.py`` code span)."""
    return _mechanism_span(context, r"[^`\s]+\.py", "source file")


def _mechanism_span(context: str, pattern: str, label: str) -> str:
    if "## Mechanism" not in context:
        raise AssertionError(
            "observation has no ## Mechanism section (cannot derive the {})".format(
                label
            )
        )
    section = context.split("## Mechanism", 1)[1]
    matcher = re.compile("`(" + pattern + ")`")
    for span in matcher.findall(section):
        if re.fullmatch(pattern, span):
            return span
    raise AssertionError(
        "no {} code span in the observation's ## Mechanism section".format(label)
    )


def _first_content_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "(empty excerpt)"


# ---------------------------------------------------------------------------
# Scripted offline model: conditional on observations, derived arguments
# ---------------------------------------------------------------------------


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        finish_reason="tool_calls",
    )


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _observations(request: LLMRequest) -> List[Dict[str, Any]]:
    observed: List[Dict[str, Any]] = []
    for message in request.messages:
        if message.role != "tool":
            continue
        try:
            observed.append(json.loads(message.content))
        except (TypeError, ValueError) as exc:
            raise AssertionError("observation is not JSON: {}".format(exc))
    return observed


def _expect_observations(*tool_names: str) -> Callable[[LLMRequest], List[Dict[str, Any]]]:
    """Expect exactly these prior observations, in order, each a
    successful ToolResult — otherwise this scripted model FAILS the run
    (the application must feed every ToolResult back)."""

    def expect(request: LLMRequest) -> List[Dict[str, Any]]:
        observed = _observations(request)
        if len(observed) != len(tool_names):
            raise AssertionError(
                "expected prior observations {} ({}); request carries {} — "
                "the Agent must feed ToolResults back".format(
                    list(tool_names), len(tool_names), len(observed)
                )
            )
        for expected, observation in zip(tool_names, observed):
            if observation.get("tool") != expected:
                raise AssertionError(
                    "expected observation from {!r}, got {!r}".format(
                        expected, observation.get("tool")
                    )
                )
            if observation.get("success") is not True:
                raise AssertionError(
                    "expected a successful observation from {!r}, got a "
                    "failure — the script would have to adapt to an error "
                    "here".format(expected)
                )
        return observed

    return expect


def _respond_orient(observed: Sequence[Dict[str, Any]]) -> Tuple[LLMResponse, str]:
    call = ToolCall(
        id="call_m86_orient", name="get_architecture_context", arguments={}
    )
    return _tool_response(call), "start: orient on the architecture context"


def _respond_search(observed: Sequence[Dict[str, Any]]) -> Tuple[LLMResponse, str]:
    context = observed[0]["data"]["context"]
    token = _mechanism_symbol(context)
    call = ToolCall(
        id="call_m86_search",
        name="search_repository",
        arguments={"query": token},
    )
    note = "derived from observation 1: mechanism symbol `{}`".format(token)
    return _tool_response(call), note


def _respond_read(observed: Sequence[Dict[str, Any]]) -> Tuple[LLMResponse, str]:
    results = observed[1]["data"]["results"]
    if not results:
        raise AssertionError("observation 2 carries no search results")
    first = results[0]
    arguments = {
        "path": first["path"],
        "line_start": first["line_start"],
        "line_end": first["line_end"],
    }
    call = ToolCall(id="call_m86_read", name="read_file", arguments=arguments)
    note = "derived from observation 2: first search result {}:{}-{}".format(
        first["path"], first["line_start"], first["line_end"]
    )
    return _tool_response(call), note


def _respond_answer(observed: Sequence[Dict[str, Any]]) -> Tuple[LLMResponse, str]:
    context = observed[0]["data"]["context"]
    symbol = _mechanism_symbol(context)
    source_file = _source_file(context)
    search = observed[1]["data"]
    first = search["results"][0]
    read = observed[2]["data"]
    snippet = _first_content_line(read.get("content") or "")
    answer = (
        "Evidence-backed answer (cited from the investigation observations):\n"
        "- Observation 1 (architecture context): mechanism `{symbol}` in "
        "`{source_file}`.\n"
        "- Observation 2 (search): {query!r} over the pinned snapshot found "
        "{total} match(es); first candidate {hit_path}:{hit_start}-{hit_end}.\n"
        "- Observation 3 (read): {path}:{line_start}-{line_end} → {snippet!r}.\n"
        "Conclusion: {source_file} documents the tool-call limit `{symbol}`, "
        "verified above with repository evidence after three read-only "
        "actions."
    ).format(
        symbol=symbol,
        source_file=source_file,
        query=search["query"],
        total=search["total_matches"],
        hit_path=first["path"],
        hit_start=first["line_start"],
        hit_end=first["line_end"],
        path=read["path"],
        line_start=read["line_start"],
        line_end=read["line_end"],
        snippet=snippet,
    )
    note = "built from observations 1-3 ({} characters)".format(len(answer))
    return _text_response(answer), note


Step = Tuple[str, Callable[[LLMRequest], List[Dict[str, Any]]], Callable[[Sequence[Dict[str, Any]]], Tuple[LLMResponse, str]]]


def _scripted_steps() -> List[Step]:
    """The adaptive investigation: every step's action is derived from
    the observations its request carries (never pre-programmed)."""
    return [
        ("orient", _expect_observations(), _respond_orient),
        ("search", _expect_observations("get_architecture_context"), _respond_search),
        (
            "read",
            _expect_observations("get_architecture_context", "search_repository"),
            _respond_read,
        ),
        (
            "answer",
            _expect_observations(
                "get_architecture_context", "search_repository", "read_file"
            ),
            _respond_answer,
        ),
    ]


class _ScriptedModel:
    """Offline ``LLMClient``: serves one scripted step per call, in
    order, each conditional on its expected prior observations."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)
        self.requests: List[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        if index >= len(self._steps):
            raise AssertionError(
                "scripted model invoked {} time(s) but only {} step(s) are "
                "scripted".format(index + 1, len(self._steps))
            )
        label, expect, respond = self._steps[index]
        observed = expect(request)
        self.requests.append(request)
        response, note = respond(observed)
        print(
            "--- LLM call {} (observations carried: {}, step {})".format(
                index + 1, len(observed), label
            )
        )
        if response.tool_calls:
            for call in response.tool_calls:
                print(
                    "    action: {}({})".format(
                        call.name, json.dumps(call.arguments, sort_keys=True)
                    )
                )
        else:
            print(
                "    action: final answer ({} characters)".format(
                    len(response.message.content or "")
                )
            )
        print("    {}".format(note))
        return response


# ---------------------------------------------------------------------------
# Real dispatcher wrapper: prints each observation as it happens
# ---------------------------------------------------------------------------


def _observation_summary(result: ToolResult) -> str:
    if not result.success:
        if result.error is not None:
            return "failure {} — {}".format(
                result.error.code, result.error.message[:60]
            )
        return "failure"
    data = result.data or {}
    if result.tool == "search_repository":
        results = data.get("results") or []
        head = "{} match(es)".format(data.get("total_matches"))
        if results:
            first = results[0]
            head += ", first {}:{}-{}".format(
                first["path"], first["line_start"], first["line_end"]
            )
        return head
    if result.tool == "read_file":
        return "{}:{}-{} ({} chars)".format(
            data.get("path"),
            data.get("line_start"),
            data.get("line_end"),
            len(data.get("content") or ""),
        )
    if result.tool == "get_architecture_context":
        return "context ({} chars)".format(len(data.get("context") or ""))
    return "success"


class _RecordingDispatcher:
    """Wraps the real dispatcher: records and prints every dispatch
    (read-only registry, production resolution unchanged)."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: List[ToolCall] = []
        self.results: List[ToolResult] = []

    @property
    def tool_names(self) -> Tuple[str, ...]:
        return tuple(self._inner.tool_names)

    def dispatch(self, tool_call: ToolCall) -> ToolResult:
        result = self._inner.dispatch(tool_call)
        self.calls.append(tool_call)
        self.results.append(result)
        print(
            "--- observation {}: {} → {}".format(
                len(self.calls),
                tool_call.name,
                "success" if result.success else "failure: " + _observation_summary(result),
            )
        )
        if result.success:
            print("    {}".format(_observation_summary(result)))
        return result


# ---------------------------------------------------------------------------
# Execution-trace view (existing observability — observable events only)
# ---------------------------------------------------------------------------


def _print_investigation_trace(trace: AgentTrace) -> None:
    summary = trace.summary
    print(RULE)
    print("Investigation trace (observable events only — no hidden reasoning)")
    print("  run: {}".format(trace.run_id))
    if summary is not None:
        print(
            "  tool-call limit (max_tool_calls)={}  executed={}  blocked={}  "
            "iterations={}".format(
                summary.max_tool_calls,
                summary.tool_calls_executed,
                summary.tool_calls_blocked,
                summary.total_iterations,
            )
        )
    for iteration in trace.iterations:
        if iteration.response_type == "FINAL_ANSWER":
            print(
                "  [{}] FINAL    answer ({} characters)".format(
                    iteration.iteration, len(iteration.final_answer or "")
                )
            )
            continue
        actions = "; ".join(
            "{}({})".format(
                entry["name"], json.dumps(entry["arguments"], sort_keys=True)
            )
            for entry in iteration.tool_calls
        )
        print("  [{}] ACTION   {}".format(iteration.iteration, actions))
        for result in iteration.tool_results:
            data = result.get("data") or {}
            if result.get("tool") == "search_repository":
                hint = "total_matches={}".format(data.get("total_matches"))
            elif result.get("tool") == "read_file":
                hint = "{}:{}-{}".format(
                    data.get("path"), data.get("line_start"), data.get("line_end")
                )
            elif result.get("tool") == "get_architecture_context":
                hint = "{} chars".format(len(data.get("context") or ""))
            else:
                hint = ""
            print(
                "         → {} {}".format(
                    result.get("tool"),
                    "success ({})".format(hint) if hint and result.get("success")
                    else ("success" if result.get("success") else "failure"),
                )
            )
        for blocked in iteration.blocked_tool_calls:
            print(
                "         → {} BLOCKED (not executed: tool-call limit "
                "{}/{})".format(
                    blocked.get("tool"),
                    blocked.get("tool_calls_executed"),
                    blocked.get("max_tool_calls"),
                )
            )
    print("  termination: {}".format(trace.termination_reason))


def _walk_keys(value: Any) -> Sequence[str]:
    if isinstance(value, dict):
        keys: List[str] = []
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
        return keys
    if isinstance(value, list):
        keys = []
        for item in value:
            keys.extend(_walk_keys(item))
        return keys
    return []


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check(index: int, name: str, ok: bool, detail: str) -> bool:
    dots = "." * max(3, NAME_WIDTH - len(name))
    status = "PASS" if ok else "FAIL"
    line = "[{}/{}] {} {}  {}".format(index, TOTAL_CHECKS, name, dots, status)
    if detail:
        line += "  " + detail
    print(line)
    return ok


def _observations_of(fake: _ScriptedModel, index: int) -> List[Dict[str, Any]]:
    if len(fake.requests) <= index:
        return []
    return _observations(fake.requests[index])


def _evaluate(
    fake: _ScriptedModel,
    dispatcher: _RecordingDispatcher,
    agent: Agent,
    answer: str,
) -> int:
    passed = 0
    observed_1 = _observations_of(fake, 1)
    observed_2 = _observations_of(fake, 2)
    observed_3 = _observations_of(fake, 3)
    trace = agent.last_trace

    # 1 — observation 1 is carried into request 2, verbatim.
    first = observed_1[0] if observed_1 else {}
    ok = (
        len(observed_1) == 1
        and first.get("tool") == "get_architecture_context"
        and len(dispatcher.results) >= 1
        and first == dispatcher.results[0].to_dict()
    )
    passed += _check(
        1,
        "Observation 1 carried into request 2",
        ok,
        "ToolResult #1 present verbatim in request 2"
        if ok
        else "request 2 carries {} observation(s)".format(len(observed_1)),
    )

    # 2 — the search query was derived from observation 1's content.
    token_from_observation = ""
    if first.get("data", {}).get("context"):
        try:
            token_from_observation = _mechanism_symbol(first["data"]["context"])
        except AssertionError:
            token_from_observation = ""
    query = ""
    if len(dispatcher.calls) >= 2:
        query = dispatcher.calls[1].arguments.get("query", "")
    ok = bool(token_from_observation) and query == token_from_observation
    passed += _check(
        2,
        "Action 2 derived from observation 1",
        ok,
        "search query={!r} == symbol from observation".format(query)
        if ok
        else "query={!r} expected {!r}".format(query, token_from_observation),
    )

    # 3 — observations 1+2 are carried into request 3, in order.
    ok = (
        len(observed_2) == 2
        and observed_2[0].get("tool") == "get_architecture_context"
        and observed_2[1].get("tool") == "search_repository"
        and all(item.get("success") is True for item in observed_2)
    )
    passed += _check(
        3,
        "Observations 1+2 carried into request 3",
        ok,
        "arch context + search results observed by request 3"
        if ok
        else "request 3 carries {} observation(s)".format(len(observed_2)),
    )

    # 4 — the read path/line range came from observation 2's results.
    derived_ok = False
    if (
        len(observed_2) == 2
        and observed_2[1].get("success")
        and len(dispatcher.calls) >= 3
    ):
        results = (observed_2[1].get("data") or {}).get("results") or []
        if results:
            first_result = results[0]
            arguments = dispatcher.calls[2].arguments
            derived_ok = (
                dispatcher.calls[2].name == "read_file"
                and arguments.get("path") == first_result["path"]
                and arguments.get("line_start") == first_result["line_start"]
                and arguments.get("line_end") == first_result["line_end"]
            )
    passed += _check(
        4,
        "Action 3 derived from observation 2",
        derived_ok,
        "read path/line range taken from the first search result"
        if derived_ok
        else "read arguments do not match observation 2's first result",
    )

    # 5 — three evidence actions, then a final answer (4 LLM calls).
    names = [call.name for call in dispatcher.calls]
    ok = names == [
        "get_architecture_context",
        "search_repository",
        "read_file",
    ] and len(fake.requests) == 4
    passed += _check(
        5,
        "Three evidence actions before the answer",
        ok,
        "{} actions → final answer ({} LLM calls)".format(
            len(names), len(fake.requests)
        )
        if ok
        else "actions={} llm_calls={}".format(names, len(fake.requests)),
    )

    # 6 — the final answer cites the gathered evidence.
    cited_ok = False
    if observed_2 and observed_3:
        try:
            token = _mechanism_symbol(observed_3[0]["data"]["context"])
        except AssertionError:
            token = ""
        search_data = observed_2[1].get("data") or {}
        read_data = observed_3[2].get("data") if len(observed_3) > 2 else {}
        snippet = _first_content_line(read_data.get("content") or "")
        required = [
            token,
            str(search_data.get("total_matches")),
            str(read_data.get("path", "")),
            str(read_data.get("line_start", "")),
            snippet,
        ]
        cited_ok = (
            isinstance(answer, str)
            and bool(answer)
            and all(item and item in answer for item in required)
        )
    passed += _check(
        6,
        "Final answer cites the gathered evidence",
        cited_ok,
        "answer references symbol, match count, read path/lines, excerpt"
        if cited_ok
        else "answer missing observation-derived citations",
    )

    # 7 — the investigation stayed inside the single tool-call limit.
    bounded_ok = False
    bounded_detail = "no finalized trace"
    if trace is not None and trace.summary is not None:
        summary = trace.summary
        bounded_ok = (
            summary.max_tool_calls == TOOL_CALL_LIMIT
            and summary.tool_calls_executed == 3
            and summary.tool_calls_blocked == 0
            and summary.termination_reason == "FINAL_ANSWER"
            and len(dispatcher.calls) <= TOOL_CALL_LIMIT
        )
        bounded_detail = "{} of {} executed, 0 blocked, {}".format(
            summary.tool_calls_executed,
            summary.max_tool_calls,
            summary.termination_reason,
        )
    passed += _check(
        7,
        "Bounded by the single tool-call limit",
        bounded_ok,
        bounded_detail,
    )

    # 8 — the trace records investigation steps, never hidden reasoning.
    trace_ok = False
    trace_detail = "no trace"
    if trace is not None:
        payload = trace.to_dict()
        banned = tuple(BANNED_REASONING_FIELDS) + EXTRA_BANNED_FIELDS
        keys = [key.lower() for key in _walk_keys(payload)]
        offending = [
            key
            for key in keys
            for needle in banned
            if needle in key
        ]
        trace_ok = (
            not offending
            and trace.termination_reason == "FINAL_ANSWER"
            and trace.final_answer == answer
            and all(iteration.response_type for iteration in trace.iterations)
        )
        trace_detail = "{} keys checked, {} iterations, no banned fields".format(
            len(keys), len(trace.iterations)
        )
        if offending:
            trace_detail = "banned field(s): {}".format(sorted(set(offending)))
    passed += _check(
        8,
        "Trace: observable events only",
        trace_ok,
        trace_detail,
    )
    return passed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run() -> int:
    started = time.monotonic()
    temp_root = Path(tempfile.mkdtemp(prefix="m86-react-"))
    artifacts_root = temp_root / "artifacts"
    traces_root = temp_root / "traces"
    run_dir = _build_pinned_run(artifacts_root)

    original_root_factory = artifacts.default_artifacts_root
    artifacts.default_artifacts_root = lambda: artifacts_root
    try:
        print("M8.6 — bounded ReAct investigation test")
        print("========================================")
        print("mode: offline (scripted model, REAL Agent loop + real read-only tools)")
        print("objective: {}".format(OBJECTIVE))
        print("tool-call limit (max_tool_calls) = {}".format(TOOL_CALL_LIMIT))
        print("fixture pinned run: {}".format(run_dir))
        print(
            "  repository/ = real files copied from this project ({})".format(
                ", ".join(COPIED_PATHS)
            )
        )
        print()

        fake = _ScriptedModel(_scripted_steps())
        dispatcher = _RecordingDispatcher(default_tool_dispatcher())
        agent = Agent(
            llm_client=fake,
            tool_dispatcher=dispatcher,
            trace_sink=JsonFileTraceSink(traces_root),
            execution_config=AgentExecutionConfig(max_tool_calls=TOOL_CALL_LIMIT),
        )

        print(
            "Investigation (Decision → Action → Observation → … → Final Answer)"
        )
        try:
            answer = agent.run(OBJECTIVE)
        except AssertionError as exc:
            print()
            print("adaptation proof FAILED: {}".format(exc))
            print(RULE)
            print("Result: FAIL — the scripted model required an observation")
            print("         the request did not carry (see above).")
            return EXIT_FAIL
        except Exception as exc:  # noqa: BLE001 — report, never hide
            print()
            print("run FAILED: {}: {}".format(type(exc).__name__, exc))
            print(RULE)
            print("Result: FAIL — {}".format(exc))
            return EXIT_FAIL

        print()
        print("--- final answer")
        for line in answer.splitlines():
            print("    {}".format(line))
        print()

        _print_investigation_trace(agent.last_trace)
        if agent.last_trace is not None:
            print(
                "  trace: {}".format(
                    JsonFileTraceSink(traces_root).json_path(
                        agent.last_trace.run_id
                    )
                )
            )

        print(RULE)
        print("Checks")
        passed = _evaluate(fake, dispatcher, agent, answer)

        elapsed = time.monotonic() - started
        print(RULE)
        if passed >= TOTAL_CHECKS:
            print(
                "Result: PASS - {}/{} checks ({:.2f}s)".format(
                    passed, TOTAL_CHECKS, elapsed
                )
            )
            return EXIT_OK
        print(
            "Result: FAIL - {}/{} checks ({:.2f}s)".format(
                passed, TOTAL_CHECKS, elapsed
            )
        )
        return EXIT_FAIL
    finally:
        artifacts.default_artifacts_root = original_root_factory


def main(argv: Optional[List[str]] = None) -> int:
    if argv is not None and argv:
        print("usage: python agentrun_react_test.py", file=sys.stderr)
        return EXIT_FAIL
    try:
        return run()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
