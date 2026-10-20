"""Repository Query — one natural-language question, one evidence-
backed answer (the user-facing entry point over M1–M17).

This module is a *thin* composition root: it owns no agent runtime,
no planner, no orchestrator, no memory system, no context engine, no
validator, and no renderer.  Every stage it runs already exists:

    question
      → plan            (M10 planner — scripted offline by default)
      → M17 Orchestrator: Architecture Investigator → Flow /
        Dependency Investigator → Architecture Visualization Agent
        (ONE investigator slot routed by the plan's own step labels,
        the analyst slot holds the visualizer — unmodified machinery)
      → M16 deterministic collaboration verdict (on the run itself)
      → six-rule diagram validation (orchestrated scope + the
        INVESTIGATORS' traces as evidence authority)
      → the four visualization artifacts (a validated spec only)
      → one ``RepositoryQueryResult``: answer, evidence, validation
        status, run ids, trace location, artifact rows, failures.

What this module owns, and only this: composition of the
participants, the investigator router (glue the orchestrator's single
slot needs), the observable result type, and the offline story that
answers without a network.  ``agentrun_repository_query.py`` is the
CLI over this module; it renders and chooses exit codes.

Offline-first (spec): offline mode uses scripted models and the
fixture context/graph bindings while retaining repository-scoped text
reads, so its behavior stays deterministic. Live mode uses real model
clients AND the real read-only context/graph tools, all pinned to the
verified subject snapshot. Validators and artifact handling are shared.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import (
    Agent,
    AgentError,
    AgentExecutionConfig,
    JsonFileTraceSink,
)
from .agents import (
    ARCHITECTURE_TOOLS,
    FLOW_DEPENDENCY_TOOLS,
    ArchitectureInvestigator,
    ArchitectureInvestigatorError,
    ArchitectureReport,
    ArchitectureVisualizationAgent,
    ArchitectureVisualizationAgentError,
    DiagramScope,
    DiagramSpec,
    FlowDependencyInvestigator,
    FlowDependencyInvestigatorError,
    FlowDependencyReport,
    RunSummaryContext,
    VisualizationArtifactError,
    validate_diagram_spec,
    verify_artifacts,
    write_visualization_artifacts,
)
from .artifacts import (
    PRIMARY_SNAPSHOT_ID,
    PROJECT_ROOT,
    REPOSITORY_DIR_NAME,
    default_artifacts_root,
    repository_dir,
    snapshot_path,
)
from .context.models import display_path as _project_display_path
from .context_engine import ContextBudget, ContextEngine
from .evidence.validator import gather_evidence
from .llm import (
    LLMError,
    LLMResponse,
    StructuredOutputValidationError,
)
from .llm.config import load_env
from .llm.openai_client import DEFAULT_MODEL, OpenAILLMClient
from .memory import MemoryContext, MemoryManager, MemoryOwner, MemoryScope
from .multiagent import ArchitectureFinding, ContractError, TaskPayload
from .multiagent.permissions import (
    restricted_dispatcher,
    tool_definitions_for,
)
from .offline_investigation import (
    AnalystModel,
    InvestigatorModel,
    structured_response,
    synthesize_answer,
)
from .orchestration import (
    ANALYST,
    COMPLETED,
    INVESTIGATOR,
    PLAN_FAILED as RUN_PLAN_FAILED,
    VALIDATION_FAILED as RUN_VALIDATION_FAILED,
    OrchestrationConfig,
    Orchestrator,
)
from .planning.planner import build_planning_request, plan_from_response
from .planning.validation import PlanValidationError, validate_plan
from .repository.git import GitError, get_head_sha
from .storage import default_query_db_path as _storage_query_db_path
from .tools import ToolError, ToolResult
from .tools.get_architecture_context import get_architecture_context
from .tools.query_engineering_graph import query_engineering_graph_tool
from .tools.read_file import read_file
from .tools.search_repository import search_repository

__all__ = [
    "EXIT_INVESTIGATION_FAILED",
    "EXIT_INTERRUPTED",
    "EXIT_SUCCESS",
    "EXIT_USAGE",
    "EXIT_VALIDATION_FAILED",
    "LIVE_CONFIGURATION_UNAVAILABLE",
    "MAX_CONCLUSIONS",
    "MAX_EVIDENCE",
    "MAX_UNCERTAINTIES",
    "ObservedAgent",
    "QUERY_DB_PATH",
    "READ_CANDIDATES",
    "SNAPSHOT_VERIFICATION_FAILED",
    "SUBJECT_COMMIT",
    "SUBJECT_SNAPSHOT_ID",
    "ArtifactRow",
    "QueryRouter",
    "RepositoryQueryError",
    "RepositoryQueryResult",
    "RepositoryQueryRouterError",
    "StepInfo",
    "default_query_db_path",
    "default_subject_repository",
    "display_path",
    "exit_code",
    "fixture_bindings",
    "live_bindings",
    "offline_plan_payload",
    "read_snapshot_metadata",
    "resolve_read_target",
    "resolve_subject_commit",
    "run_repository_query",
    "verify_subject_snapshot",
]

# -- exit codes (the CLI's contract) ---------------------------------------

EXIT_SUCCESS = 0
EXIT_INVESTIGATION_FAILED = 1
EXIT_USAGE = 2
EXIT_VALIDATION_FAILED = 3
EXIT_INTERRUPTED = 130

# -- bounds ----------------------------------------------------------------

MAX_CONCLUSIONS = 10
MAX_EVIDENCE = 12
MAX_UNCERTAINTIES = 8

SUBJECT = "repository-query"

# The query's OWN memory database — centralized under
# ``db_store/memory/`` (M18 storage layout; db_store/ is git-ignored).
# ``QUERY_DB_PATH`` is the import-time snapshot, kept for existing
# references; runtime callers resolve through
# ``default_query_db_path()`` so the ``REPO_INTELLIGENCE_DB_STORE``
# seam applies at call time.
QUERY_DB_PATH = _storage_query_db_path()


def default_query_db_path() -> Path:
    """Runtime-resolved path of the repository-query memory database."""
    return _storage_query_db_path()

DEFAULT_BUDGET = ContextBudget(max_items=64, max_characters=6000)

# M17 global execution budget — the mandated 64 tool calls, passed
# explicitly at THIS composition root (the Agent's own default of 8
# still governs every other construction in the project).
#
# The same configuration opts the workflow's agents into the boundary /
# finalization lifecycle, AND into the bounded investigation lifecycle:
# crossing the 8-call soft target opens the decision phase (reassess
# → finalize, or exactly ONE targeted call → reassess → finalize). This
# normally caps a healthy investigator at 9 calls, so up to five
# investigator steps plus one retry and deterministic validation fit
# inside the unchanged 64-call whole-run ceiling. An execution boundary
# is a LOCAL status (``incomplete``), never a workflow failure.
M17_MAX_TOOL_CALLS = 64
M17_SOFT_TOOL_CALL_TARGET = 8
M17_EXECUTION_CONFIG = AgentExecutionConfig(
    max_tool_calls=M17_MAX_TOOL_CALLS,
    soft_tool_call_target=M17_SOFT_TOOL_CALL_TARGET,
    finalize_on_boundary=True,
    enforce_soft_target=True,
)
# The matching whole-run hard bound remains 64 (rather than the
# orchestrator's smaller default of 32). The soft target above, not a
# larger hard budget, keeps multi-step investigations within this bound.
# Every other construction keeps the shipped defaults.
M17_ORCHESTRATION_CONFIG = OrchestrationConfig(
    max_total_tool_calls=M17_MAX_TOOL_CALLS,
)

# Offline scope label: a visibly synthetic commit (only reached for a
# repository that carries neither Git metadata nor an M1 snapshot
# record; the pinned subject always reports its real commit).
OFFLINE_COMMIT = "c" * 40

# -- the pinned subject repository this project analyzes -------------------
#
# repository-intelligence analyzes ITS SUBJECT, never itself: M1's own
# checkout of the ADOP repository, stored under ``artifacts/`` by the
# existing snapshot layout (``artifacts.py`` already owns the id).
# No clone happens here — the snapshot is the authority.
SUBJECT_SNAPSHOT_ID = PRIMARY_SNAPSHOT_ID
SUBJECT_COMMIT = "ef11dac8618b6344d6a5e4add6ef51cd2811d7e0"

# ``RepositoryQueryResult.termination_reason`` for a subject snapshot
# that could not be verified — the CLI prints the standing ERROR block
# for it and no investigation is executed.
SNAPSHOT_VERIFICATION_FAILED = "SNAPSHOT_VERIFICATION_FAILED"

# ``RepositoryQueryResult.termination_reason`` for ``--live`` when the
# configured provider cannot be constructed (no key, no client
# package): an explicit request for live mode NEVER falls back to the
# offline model — it fails, loudly, with these two lines first.
LIVE_CONFIGURATION_UNAVAILABLE = "LIVE_CONFIGURATION_UNAVAILABLE"

LIVE_FAILURE_MESSAGE = (
    "Live mode requested but the configured LLM provider is "
    "unavailable.\nSet the required configuration and retry."
)

# Files a repository root is searched for — the first readable one is
# the investigation target and the entry point every read starts from.
# The pinned ADOP snapshot reaches README.md (its root carries no
# ``agents/`` or ``src/`` tree); the MCP configuration, the registries
# and the agent definitions are found by the investigation itself.
READ_CANDIDATES: Tuple[str, ...] = (
    "agents/collector.py",
    "tools/fetch_record.py",
    "agents/researcher.py",
    "src/producer.py",
    "src/api.py",
    "src/service.py",
    "README.md",
    "main.py",
    "src/main.py",
)

# The runner labels an investigator step inside its question as
# "step: <id> — <goal>" (em dash U+2014) on its own line; the goal
# rides along so the router can fall back to it for a live planner's
# ids.  MULTILINE because the label is not the whole question.
_STEP_LABEL = re.compile("^step: (\\S+) \u2014 (.*)$", re.MULTILINE)

# The investigators' own statement form ends in its classification
# ("... [observed]"); parse it back so evidence can be annotated with
# the classification of the statement that cites it — never re-typed.
_STATEMENT_CLASS = re.compile("\\s*\\[([a-z_]+)\\]\\s*$")


class RepositoryQueryError(Exception):
    """A query that could not be answered — explicit, never a guess."""


class RepositoryQueryRouterError(RepositoryQueryError):
    """A plan step with no routed specialist — explicit, never a
    guess."""


# -- result ----------------------------------------------------------------


@dataclass
class StepInfo:
    """One plan step as the CLI shows it — every field is an execution
    fact recorded by the orchestrator or the planner."""

    step_id: str = ""
    goal: str = ""
    role: str = ""
    status: str = ""
    label: str = ""
    run_id: Optional[str] = None
    output_type: str = ""


@dataclass
class ArtifactRow:
    """One persisted artifact, verified on disk after the run."""

    name: str = ""
    path: str = ""
    exists: bool = False
    bytes: int = 0
    note: str = ""


@dataclass
class RepositoryQueryResult:
    """Everything the CLI prints — observable execution facts only.

    ``status`` is the orchestration status when a run happened, or the
    pre-run failure status (``FAILED`` / ``PLAN_FAILED``) when the run
    never started; ``exit_code(result)`` maps it to the CLI contract.
    """

    question: str = ""
    answer: str = ""
    conclusions: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    status: str = "FAILED"
    termination_reason: Optional[str] = None
    mode: str = "offline"
    repository: str = ""
    target: Optional[str] = None
    # -- the subject under analysis (never the analyzer itself) --------
    subject_repository: str = ""
    pinned_commit: str = ""
    commit_source: str = ""
    investigation_id: Optional[str] = None
    steps: List[StepInfo] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
    collaboration_status: Optional[str] = None
    diagram_status: Optional[str] = None
    diagram_in_scope: bool = True
    artifacts: List[ArtifactRow] = field(default_factory=list)
    artifact_directory: Optional[str] = None
    trace_paths: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    artifacts_error: Optional[str] = None


def exit_code(result: RepositoryQueryResult) -> int:
    """The CLI's documented exit contract (0/1/3)."""
    if result.status == RUN_VALIDATION_FAILED:
        return EXIT_VALIDATION_FAILED
    if result.diagram_in_scope and result.diagram_status == "failed":
        return EXIT_VALIDATION_FAILED
    if result.status != COMPLETED:
        return EXIT_INVESTIGATION_FAILED
    if result.artifacts_error:
        return EXIT_INVESTIGATION_FAILED
    return EXIT_SUCCESS


def display_path(value: Any) -> str:
    """Render a path the way the CLI must — project-relative when the
    file lives under the project root, the given text otherwise (an
    artifact that was never produced carries a notice, not a path).
    Reuses the existing machine-independent renderer so a path never
    means two things in this project's output."""
    if value is None:
        return ""
    return _project_display_path(value)


def _ref_display(ref: Any) -> str:
    """``path`` / ``path:start`` / ``path:start-end`` — evidence
    references are already repository-relative."""
    start = getattr(ref, "line_start", None)
    end = getattr(ref, "line_end", None)
    if start is None:
        return str(ref.path)
    if end is None or end == start:
        return "{}:{}".format(ref.path, start)
    return "{}:{}-{}".format(ref.path, start, end)


def _statement_class(statement: str) -> str:
    match = _STATEMENT_CLASS.search(statement)
    return match.group(1) if match is not None else "unknown"


def _evidence_lines(
    refs: List[Any],
    statements: List[Tuple[str, List[Any]]],
    limit: int = MAX_EVIDENCE,
) -> List[str]:
    """Evidence references annotated with the classification of the
    statement(s) that cite them — one line per reference, capped."""
    classes: Dict[str, List[str]] = {}
    for statement, statement_refs in statements:
        classification = _statement_class(statement)
        for ref in statement_refs:
            seen = classes.setdefault(ref.path, [])
            if classification not in seen:
                seen.append(classification)
    lines: List[str] = []
    for ref in refs:
        if len(lines) >= limit:
            break
        line = "[{}] {}".format(
            ", ".join(classes.get(ref.path) or ["unknown"]),
            _ref_display(ref),
        )
        if line not in lines:
            lines.append(line)
    return lines


def _unique_statements(statements: List[str]) -> List[str]:
    """One entry per distinct statement.

    The merge keeps every investigator's record of a fact (two
    specialists may establish the same relationship independently),
    so the printed list says it once — the record itself is never
    rewritten.
    """
    seen: List[str] = []
    for statement in statements:
        if statement not in seen:
            seen.append(statement)
    return seen


# -- scripted planner (offline) --------------------------------------------


class OfflineModel:
    """Scripted LLM double: answers strictly from its queue, never
    touches the network, fails loudly if over-invoked."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._queue: List[LLMResponse] = list(responses)
        self.requests: List[Any] = []
        self.model = "offline-scripted"

    def complete(self, request: Any) -> LLMResponse:
        self.requests.append(request)
        if not self._queue:
            raise AssertionError("model invoked more times than scripted")
        return self._queue.pop(0)


def default_subject_repository() -> Path:
    """The pinned subject checkout M1 already persisted.

    ``<project-root>/artifacts/<snapshot-id>/repository`` — resolved
    from this module's location (``PROJECT_ROOT``), never from the
    process working directory, so it is machine- and platform-
    independent.  The ADOP repository is NOT cloned here: this is the
    authoritative local snapshot.
    """
    return repository_dir(default_artifacts_root(), PRIMARY_SNAPSHOT_ID)


def read_snapshot_metadata(root: Any) -> Dict[str, str]:
    """M1's own record beside a checkout — ``name`` / ``url`` /
    ``commit`` when the snapshot artifact is there, ``{}`` otherwise.

    The record lives at ``<snapshot-dir>/snapshot/repository_snapshot.json``
    (the existing ``snapshot_path`` layout), so a checkout that is not
    a Git working tree can still be identified and pinned.

    *root* may be the M18 named checkout (``<id>/repository/<name>``)
    or the legacy flat directory (``<id>/repository``) — the run
    directory is derived structurally, never from a hard-coded name.
    """
    root_path = Path(root)
    if root_path.parent.name == REPOSITORY_DIR_NAME:
        # named layout: .../<id>/repository/<name>
        run_dir = root_path.parent.parent
    else:
        # legacy flat layout (and the historical derivation):
        # .../<id>/repository
        run_dir = root_path.parent
    path = snapshot_path(run_dir.parent, run_dir.name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    repository = data.get("repository") or {}
    snapshot = data.get("snapshot") or {}
    metadata: Dict[str, str] = {}
    if isinstance(repository, dict) and repository.get("name"):
        metadata["name"] = str(repository["name"])
    if isinstance(repository, dict) and repository.get("url"):
        metadata["url"] = str(repository["url"])
    if isinstance(snapshot, dict) and snapshot.get("commit_sha"):
        metadata["commit"] = str(snapshot["commit_sha"])
    return metadata


def resolve_subject_commit(root: Any) -> Tuple[Optional[str], str]:
    """``(commit, source)`` for a checkout — Git when the working tree
    carries metadata, M1's snapshot record otherwise, ``("", "")`` when
    neither exists.  Never a guess: an absent commit stays absent.
    """
    path = Path(root)
    if (path / ".git").exists():
        try:
            return get_head_sha(path), "git"
        except (GitError, OSError):
            pass
    commit = read_snapshot_metadata(path).get("commit")
    if commit:
        return commit, "snapshot"
    return None, ""


def verify_subject_snapshot(
    root: Any, expected_commit: Optional[str] = None
) -> Tuple[Optional[str], str, Optional[str]]:
    """``(actual_commit, source, error)`` for the pinned subject — the
    M1 reproducibility guarantee stated before anything is analyzed.

    The directory name is never trusted: the commit comes from Git
    when the working tree has metadata and from M1's snapshot record
    when it does not.  ``error`` is ``None`` only when the checkout is
    provably at ``expected_commit`` (``SUBJECT_COMMIT`` unless given).
    """
    if expected_commit is None:
        expected_commit = SUBJECT_COMMIT
    path = Path(root)
    if not path.is_dir():
        return None, "", "directory does not exist"
    actual, source = resolve_subject_commit(path)
    if actual is None:
        return (
            None,
            "",
            "commit cannot be verified (no Git metadata and no M1 "
            "snapshot record beside the checkout)",
        )
    if expected_commit and actual != expected_commit:
        return (
            actual,
            source,
            "commit mismatch: expected {} but the checkout is at "
            "{}".format(expected_commit, actual),
        )
    return actual, source, None


def _is_pinned_root(root: Any) -> bool:
    """True when *root* IS the pinned subject checkout — the pin is
    verified for the default AND for an explicit path that points at
    the very same snapshot, so an override can never quietly escape
    the verification rules."""
    try:
        return Path(root).resolve() == default_subject_repository().resolve()
    except OSError:
        return False


def resolve_read_target(
    repository: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    """Resolve ``(root, target_relative_path, error)`` for one query.

    The default root is the pinned ADOP subject snapshot
    (``artifacts/<snapshot-id>/repository``) — the project analyzes its
    subject, never its own source tree.  ``--repository`` selects any
    other checkout.  ``error`` is ``None`` only when a target was
    actually found — no target means no investigation, said out loud.
    """
    if repository is not None:
        root = Path(repository).expanduser()
        if not root.is_dir():
            return None, None, "{} is not a directory".format(repository)
    else:
        root = default_subject_repository()
        if not root.is_dir():
            return None, None, "{} is not a directory".format(root)
    for name in READ_CANDIDATES:
        if (root / name).is_file():
            return root, name, None
    return (
        root,
        None,
        "no readable target file under {} (looked for: {})".format(
            root.name or root, ", ".join(READ_CANDIDATES)
        ),
    )




def offline_plan_payload(question: str) -> Dict[str, Any]:
    """Three steps: two INVESTIGATOR goals (architecture, flow) then
    one ANALYST goal (the role assigner reads the goal pattern)."""
    return {
        "objective": question,
        "steps": [
            {
                "id": "S1",
                "goal": "Document the architecture components and "
                "their relationships in this repository.",
            },
            {
                "id": "S2",
                "goal": "Trace the dependency path the question asks "
                "about through this repository.",
            },
            {
                "id": "S3",
                "goal": "Interpret the investigated findings as one "
                "diagram for the user's question.",
            },
        ],
    }


# -- model sets ------------------------------------------------------------


def build_offline_models(
    question: str,
    repository: str,
    commit: str,
    entry_target: str,
) -> Dict[str, Any]:
    """Every offline participant — no scripted answer anywhere.

    The two investigators are the evidence-driven script: turn by
    turn it reads the repository through the REAL ``read_file`` and
    ``search_repository`` tools and derives its report from what came
    back.  The analyst receives the merged report and projects it
    (the diagram is never the source of the answer); the planner
    keeps its one scripted plan.
    """
    return {
        "arch": InvestigatorModel(
            "architecture", question, repository, commit, entry_target
        ),
        "flow": InvestigatorModel(
            "flow", question, repository, commit, entry_target
        ),
        "viz": AnalystModel(question, repository, commit),
        "planner": OfflineModel(
            structured_response(offline_plan_payload(question))
        ),
    }


def build_live_models(model_id: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """A real model for every participant (the offline script does not
    apply — the specialists read the repository for real)."""
    return {
        key: OpenAILLMClient(model=model_id)
        for key in ("arch", "flow", "viz", "planner")
    }


# -- composition -----------------------------------------------------------


def fixture_bindings(root: Path) -> Dict[str, Any]:
    """Bind the read-only text tools to the investigation root via the
    documented ``tool(arguments)`` closure seam; graph/context tools
    answer with canned successes."""

    def read(arguments: Any) -> ToolResult:
        return read_file(arguments, repository_root=root)

    def search(arguments: Any) -> ToolResult:
        return search_repository(arguments, repository_root=root)

    def generic(name: str):
        def tool(arguments: Any) -> ToolResult:
            return ToolResult(
                success=True,
                tool=name,
                data={"path": "", "content": ""},
            )

        return tool

    return {
        "read_file": read,
        "search_repository": search,
        "get_architecture_context": generic("get_architecture_context"),
        "query_engineering_graph": generic("query_engineering_graph"),
    }


def _snapshot_id_for_repository_root(
    repository_root: Path, output_root: Path
) -> Optional[str]:
    """Resolve a snapshot ID only when the repository lives in its M1 layout.

    This prevents a query against an arbitrary explicit repository path
    from accidentally consuming the default subject repository's graph
    or architecture context.
    """
    try:
        relative = Path(repository_root).resolve().relative_to(
            Path(output_root).resolve()
        )
    except (OSError, ValueError):
        return None
    parts = relative.parts
    if (
        len(parts) >= 3
        and parts[1] == REPOSITORY_DIR_NAME
        and parts[0]
        and parts[0] not in {".", ".."}
    ):
        return parts[0]
    return None


def live_bindings(
    root: Path,
    *,
    output_root: Optional[Path] = None,
    snapshot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind real read-only tools to the verified repository snapshot.

    ``fixture_bindings`` remains the deterministic scripted-test seam.
    Live model runs use these actual tools instead of canned empty graph
    and context responses. Repository text tools are constrained to
    ``root``; context and graph artifacts are pinned by snapshot ID.
    """
    selected_output_root = (
        Path(output_root) if output_root is not None
        else default_artifacts_root()
    )
    selected_snapshot_id = snapshot_id or _snapshot_id_for_repository_root(
        root, selected_output_root
    )

    def read(arguments: Any) -> ToolResult:
        return read_file(arguments, repository_root=root)

    def search(arguments: Any) -> ToolResult:
        return search_repository(arguments, repository_root=root)

    def context(arguments: Any) -> ToolResult:
        if selected_snapshot_id is None:
            return ToolResult(
                success=False,
                tool="get_architecture_context",
                error=ToolError(
                    code="SNAPSHOT_CONTEXT_UNAVAILABLE",
                    message=(
                        "No architecture-context artifact is bound to this "
                        "repository root; refusing to use another snapshot."
                    ),
                ),
            )
        return get_architecture_context(
            arguments,
            output_root=selected_output_root,
            snapshot_id=selected_snapshot_id,
        )

    def graph(arguments: Any) -> ToolResult:
        if selected_snapshot_id is None:
            return ToolResult(
                success=False,
                tool="query_engineering_graph",
                error=ToolError(
                    code="SNAPSHOT_GRAPH_UNAVAILABLE",
                    message=(
                        "No engineering-graph artifact is bound to this "
                        "repository root; refusing to use another snapshot."
                    ),
                ),
            )
        return query_engineering_graph_tool(
            arguments,
            output_root=selected_output_root,
            snapshot_id=selected_snapshot_id,
        )

    return {
        "read_file": read,
        "search_repository": search,
        "get_architecture_context": context,
        "query_engineering_graph": graph,
    }


class ObservedAgent:
    """Delegating observer around one specialist's ``Agent``.

    The M16/M17 handoff projects each investigator's typed report onto
    the ``InvestigationReport`` envelope, so the ORIGINAL specialist
    report (``ArchitectureReport`` / ``FlowDependencyReport``) exists
    only inside this run — and ``run_summary.md``'s per-investigator
    sections are built from exactly those originals.  Capturing the
    pre-projection output here is the only way to keep that summary
    honest without touching the specialists or the orchestrator.
    Everything else delegates to the wrapped agent unchanged (the
    specialists themselves see an ordinary agent).
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self.report: Optional[Any] = None

    def run_result(
        self,
        question: str,
        *,
        max_tool_calls: Optional[int] = None,
    ) -> Any:
        # M20 participant contract: a request-level per-attempt budget
        # reaches the wrapped agent unchanged (forwarded only when
        # set, so pre-M20 one-argument doubles keep working). The
        # wrapper never reinterprets or drops the budget — a wrapped
        # participant that rejects the override fails loudly at the
        # call, exactly as the executor's boundary policy requires.
        if max_tool_calls is None:
            result = self._agent.run_result(question)
        else:
            result = self._agent.run_result(
                question, max_tool_calls=max_tool_calls
            )
        output = getattr(result, "structured_output", None)
        if isinstance(output, (ArchitectureReport, FlowDependencyReport)):
            self.report = output
        return result

    def __getattr__(self, name: str) -> Any:
        agent = self.__dict__.get("_agent")
        if agent is None:
            raise AttributeError(name)
        return getattr(agent, name)


def build_memory_manager(db_path: Optional[str] = None) -> MemoryManager:
    manager = MemoryManager(
        db_path if db_path is not None else str(default_query_db_path())
    )
    manager.initialize()
    return manager


def build_memory_context(
    repository: str, current_commit: str
) -> MemoryContext:
    return MemoryContext(
        scope=MemoryScope(repository=repository, commit=current_commit),
        subject=SUBJECT,
        owner=MemoryOwner(
            type="agent", id="repo_intelligence.repository_query"
        ),
    )


def build_engine(repository: str) -> ContextEngine:
    return ContextEngine(budget=DEFAULT_BUDGET, repository=repository)


def build_architect(
    llm: Any,
    trace_sink: Any,
    memory_manager: Any,
    memory_context: Any,
    context_engine: Any,
    bindings: Optional[Dict[str, Any]] = None,
) -> ArchitectureInvestigator:
    """The Architecture Investigator over the existing Agent runtime —
    with the observer that keeps its original report visible."""
    agent = ObservedAgent(
        Agent(
            llm,
            tool_dispatcher=restricted_dispatcher(
                "investigator", bindings=bindings
            ),
            trace_sink=trace_sink,
            structured_output_model=ArchitectureReport,
            tool_definitions=tool_definitions_for("investigator"),
            execution_config=M17_EXECUTION_CONFIG,
            memory_manager=memory_manager,
            memory_context=memory_context,
            context_engine=context_engine,
        )
    )
    return ArchitectureInvestigator(
        agent=agent, tools=ARCHITECTURE_TOOLS
    )


def build_flow_investigator(
    llm: Any,
    trace_sink: Any,
    memory_manager: Any,
    memory_context: Any,
    context_engine: Any,
    bindings: Optional[Dict[str, Any]] = None,
) -> FlowDependencyInvestigator:
    agent = ObservedAgent(
        Agent(
            llm,
            tool_dispatcher=restricted_dispatcher(
                "investigator", bindings=bindings
            ),
            trace_sink=trace_sink,
            structured_output_model=FlowDependencyReport,
            tool_definitions=tool_definitions_for("investigator"),
            execution_config=M17_EXECUTION_CONFIG,
            memory_manager=memory_manager,
            memory_context=memory_context,
            context_engine=context_engine,
        )
    )
    return FlowDependencyInvestigator(
        agent=agent, tools=FLOW_DEPENDENCY_TOOLS
    )


def build_visualizer(
    llm: Any,
    trace_sink: Any,
    memory_manager: Any,
    memory_context: Any,
    context_engine: Any,
) -> ArchitectureVisualizationAgent:
    """The visualizer over the existing Agent runtime with NO tools —
    the analyst-slot composition, carrying only the M17 execution
    budget/lifecycle config every specialist here shares."""
    agent = Agent(
        llm,
        tool_dispatcher=restricted_dispatcher("analyst"),
        trace_sink=trace_sink,
        structured_output_model=DiagramSpec,
        tool_definitions=tool_definitions_for("analyst"),
        execution_config=M17_EXECUTION_CONFIG,
        memory_manager=memory_manager,
        memory_context=memory_context,
        context_engine=context_engine,
    )
    return ArchitectureVisualizationAgent(agent=agent, tools=())


class QueryRouter:
    """One investigator slot, two specialists (application glue, not
    new orchestration infrastructure).

    Routing precedence, retry-stable:

    1. the plan's own step ids (``S1`` → architecture, ``S2`` → flow);
    2. the memo of steps already routed — a retry of a foreign step id
       always returns to the same specialist;
    3. the step goal's content (the fallback a live planner's ids
       need): short, repository-neutral keyword families —
       components/architecture terms route to the architecture
       specialist; dependencies, paths, tools, MCP servers,
       protocols, and interaction terms route to the flow
       specialist.  A goal that mentions both families is treated as
       architecture (the family checked first), never as a coin toss;
    4. order assignment — the n-th investigator step that reached the
       router without a classifiable goal goes to the n-th
       specialist, so a plan can still start when the planner's early
       goals are vague.

    A step none of these can route raises
    ``RepositoryQueryRouterError`` — naming the step and its goal —
    once the order fallback is exhausted: an unroutable step is an
    explicit failure, never a guess, and a two-specialist application
    never implies a two-investigator-step ceiling (any number of
    steps routes once their goals classify).  Neither the
    orchestrator nor the runner knows a router exists, so
    orchestration stays unmodified.
    """

    LABELS = ("architecture", "flow")

    # Repository-neutral goal keywords.  Substring stems, not whole
    # words, so plural and derived forms classify too ("architectur"
    # → architecture/architectural, "dependenc" → dependency/
    # dependencies, "interact" → interaction/interactions).
    _ARCH_GOALS = (
        "architectur",
        "component",
        "relationship",
        "module",
        "service",
        "role",
        "responsibilit",
    )
    _FLOW_GOALS = (
        "path",
        "trace",
        "reach",
        "dependenc",
        "tool",
        "mcp",
        "server",
        "protocol",
        "interact",
        "invok",
        "communicat",
    )

    def __init__(self, architecture: Any, flow: Any) -> None:
        self._static: Dict[str, Any] = {
            "S1": architecture,
            "S2": flow,
            "arch": architecture,
            "flow": flow,
        }
        self._specialists = {"architecture": architecture, "flow": flow}
        self._order = (architecture, flow)
        self._memo: Dict[str, str] = {}
        self.calls: List[str] = []
        self.labels: Dict[str, str] = {}
        self._active: Optional[Any] = None

    @property
    def last_trace(self) -> Any:
        return self._active.last_trace if self._active is not None else None

    @staticmethod
    def _by_goal(goal: str) -> Optional[str]:
        lower = goal.lower()
        if any(
            word in lower for word in QueryRouter._ARCH_GOALS
        ):
            return "architecture"
        if any(word in lower for word in QueryRouter._FLOW_GOALS):
            return "flow"
        return None

    def _resolve(self, step_id: Optional[str], goal: str) -> Any:
        if step_id is not None and step_id in self._static:
            return self._static[step_id]
        if step_id is not None and step_id in self._memo:
            return self._specialists[self._memo[step_id]]
        label = self._by_goal(goal)
        if label is None:
            index = len(self._memo)
            if index >= len(self._order):
                raise RepositoryQueryRouterError(
                    "step {!r} is unroutable: its goal {!r} matches "
                    "no specialist category and this query has "
                    "already routed {} investigator step(s)".format(
                        step_id, goal, len(self._memo)
                    )
                )
            label = self.LABELS[index]
        if step_id is not None:
            self._memo[step_id] = label
        return self._specialists[label]

    def run_result(
        self,
        question: str,
        *,
        max_tool_calls: Optional[int] = None,
    ) -> Any:
        match = _STEP_LABEL.search(question)
        step_id = match.group(1) if match is not None else None
        goal = match.group(2) if match is not None else ""
        specialist = self._resolve(step_id, goal)
        key = step_id or ""
        if key not in self.labels:
            for label in self.LABELS:
                if specialist is self._specialists[label]:
                    self.labels[key] = label
        self._active = specialist
        self.calls.append(key)
        # M20 participant contract: the router is an orchestrator
        # participant, so it must accept (and forward, never drop)
        # the request-level per-attempt budget. ``None`` keeps the
        # exact M19 one-argument call for pre-M20 participants.
        if max_tool_calls is None:
            return specialist.run_result(question)
        return specialist.run_result(
            question, max_tool_calls=max_tool_calls
        )


# -- failure assembly ------------------------------------------------------


def _failure(
    question: str,
    message: str,
    *,
    status: str = "FAILED",
    termination_reason: Optional[str] = None,
    mode: str = "offline",
    repository: str = "",
    target: Optional[str] = None,
    subject_repository: str = "",
    pinned_commit: str = "",
    commit_source: str = "",
    detail: Optional[str] = None,
) -> RepositoryQueryResult:
    """One pre-run (or pre-assembly) failure, rendered as the CLI
    prints it: ``message`` first, ``detail`` — the reason — second."""
    return RepositoryQueryResult(
        question=question,
        status=status,
        termination_reason=termination_reason or status,
        mode=mode,
        repository=repository,
        target=target,
        subject_repository=subject_repository,
        pinned_commit=pinned_commit,
        commit_source=commit_source,
        failures=[message] + ([detail] if detail else []),
    )


_PLAN_ERRORS = (
    RepositoryQueryError,
    PlanValidationError,
    StructuredOutputValidationError,
    LLMError,
    AssertionError,
)

_EXECUTION_ERRORS = (
    RepositoryQueryError,
    AgentError,
    LLMError,
    StructuredOutputValidationError,
    ContractError,
    ArchitectureInvestigatorError,
    FlowDependencyInvestigatorError,
    ArchitectureVisualizationAgentError,
    VisualizationArtifactError,
    PlanValidationError,
    OSError,
    ValueError,
)


# -- the query -------------------------------------------------------------


def _build_plan(planner: Any, question: str) -> Any:
    response = planner.complete(build_planning_request(question))
    plan = plan_from_response(response)
    validate_plan(plan)
    return plan


def _specialist_report(specialist: Any) -> Optional[Any]:
    """The observer's captured PRE-projection report, if this
    specialist ran under an observer."""
    agent = getattr(specialist, "agent", None)
    return getattr(agent, "report", None)


def run_repository_query(
    question: str,
    repository: Optional[str] = None,
    *,
    mode: str = "offline",
    diagram: bool = True,
    artifacts: bool = True,
    db_path: Optional[str] = None,
    trace_root: Optional[str] = None,
    artifact_directory: Optional[str] = None,
) -> RepositoryQueryResult:
    """Run one question end to end and return the observable result.

    ``mode`` is the CALLER's decision and the only one honored:
    ``"offline"`` (the default) runs the deterministic evidence-driven
    script and ignores any provider key that happens to be set;
    ``"live"`` (``--live``) requires the configured provider and fails
    — never falls back — when it cannot be constructed.

    ``diagram=False`` (``--no-diagram``) puts the diagram out of
    scope: no six-rule validation, no artifact writer, and the answer
    is rendered from the merged investigation report instead of the
    (unvalidated) diagram finding.  ``artifacts=False``
    (``--no-artifacts``) keeps the diagram in scope but writes
    nothing.  Both flags are honored by this module; the CLI only
    renders the result.
    """
    question_text = (question or "").strip()
    if not question_text:
        raise RepositoryQueryError("repository query cannot be empty")
    if mode not in ("offline", "live"):
        raise RepositoryQueryError(
            "mode must be 'offline' or 'live', not {!r}".format(mode)
        )

    # -- resolve the SUBJECT first, and verify the pin before any work --
    # Default (or an explicit path that IS the pinned checkout): the
    # ADOP snapshot must prove it is at the expected commit, otherwise
    # nothing at all runs.  Any other explicit --repository is the
    # caller's own subject: identified when possible, never pinned.
    subject_root = (
        Path(repository).expanduser()
        if repository is not None
        else default_subject_repository()
    )
    subject_display = display_path(subject_root)
    # ``verified`` is the commit this run can PROVE (what the banner
    # reports); ``commit`` is what the reports carry as commit_scope.
    verified = ""
    commit_source = ""

    if repository is None or _is_pinned_root(subject_root):
        actual, source, verification_error = verify_subject_snapshot(
            subject_root
        )
        if verification_error is not None:
            return _failure(
                question_text,
                verification_error,
                status="FAILED",
                termination_reason=SNAPSHOT_VERIFICATION_FAILED,
                repository=subject_root.name or str(subject_root),
                subject_repository=subject_display,
                pinned_commit=SUBJECT_COMMIT,
                commit_source=source,
            )
        verified = actual or ""
        commit_source = source

    root, target, resolve_error = resolve_read_target(repository)
    if resolve_error is not None:
        return _failure(
            question_text,
            resolve_error,
            status="FAILED",
            termination_reason="NO_TARGET",
            repository=(root.name if root is not None else "")
            or (repository or PROJECT_ROOT.name),
            target=target,
            subject_repository=subject_display,
            pinned_commit=verified,
            commit_source=commit_source,
        )

    subject_display = display_path(root)
    metadata = read_snapshot_metadata(root)
    # reported as the subject's own name (M1 recorded it), falling back
    # to the checkout directory — never the analyzer's project name.
    label = metadata.get("name") or root.name or str(root)
    if not verified:
        # an explicit, unpinned subject: identify it when the checkout
        # carries metadata, otherwise say nothing rather than guess.
        discovered, discovered_source = resolve_subject_commit(root)
        verified = discovered or ""
        commit_source = discovered_source
    commit = verified or OFFLINE_COMMIT

    # -- the mode is the CALLER's decision, never the environment's --
    # Offline ignores any provider key that happens to be set; live
    # (``--live``) never falls back to the offline script when the
    # provider cannot be constructed — it fails, saying exactly that.
    load_env()
    try:
        models = (
            build_live_models()
            if mode == "live"
            else build_offline_models(
                question_text, label, commit, target
            )
        )
    except Exception as exc:  # a model set never built → explicit fail
        if mode == "live":
            return _failure(
                question_text,
                LIVE_FAILURE_MESSAGE,
                status="FAILED",
                termination_reason=LIVE_CONFIGURATION_UNAVAILABLE,
                mode="live",
                repository=label,
                target=target,
                subject_repository=subject_display,
                pinned_commit=verified,
                commit_source=commit_source,
                detail=str(exc) or type(exc).__name__,
            )
        return _failure(
            question_text,
            "{}: {}".format(type(exc).__name__, exc),
            status="FAILED",
            termination_reason="MODEL_SET_FAILED",
            mode=mode,
            repository=label,
            target=target,
            subject_repository=subject_display,
            pinned_commit=verified,
            commit_source=commit_source,
        )

    try:
        plan = _build_plan(models["planner"], question_text)
    except _PLAN_ERRORS as exc:
        return _failure(
            question_text,
            "{}: {}".format(type(exc).__name__, exc),
            status=RUN_PLAN_FAILED,
            termination_reason="PLAN_FAILED",
            mode=mode,
            repository=label,
            target=target,
            subject_repository=subject_display,
            pinned_commit=verified,
            commit_source=commit_source,
        )

    try:
        manager = build_memory_manager(db_path)
        memory_context = build_memory_context(label, commit)
        engine = build_engine(label)
        sink = (
            JsonFileTraceSink(trace_root)
            if trace_root is not None
            else JsonFileTraceSink()
        )
        bindings = (
            live_bindings(
                root,
                output_root=default_artifacts_root(),
            )
            if mode == "live"
            else fixture_bindings(root)
        )

        architect = build_architect(
            models["arch"],
            sink,
            manager,
            memory_context,
            engine,
            bindings=bindings,
        )
        flow_investigator = build_flow_investigator(
            models["flow"],
            sink,
            manager,
            memory_context,
            engine,
            bindings=bindings,
        )
        visualizer = build_visualizer(
            models["viz"], sink, manager, memory_context, engine
        )
        router = QueryRouter(architect, flow_investigator)
        orchestrator = Orchestrator(
            router,
            visualizer,
            trace_sink=sink,
            config=M17_ORCHESTRATION_CONFIG,
        )
        run = orchestrator.execute(
            TaskPayload(
                question=question_text,
                repository_scope=label,
                commit_scope=commit,
            ),
            plan,
        )
    except _EXECUTION_ERRORS as exc:
        return _failure(
            question_text,
            "{}: {}".format(type(exc).__name__, exc),
            status="FAILED",
            termination_reason="EXECUTION_FAILED",
            mode=mode,
            repository=label,
            target=target,
            subject_repository=subject_display,
            pinned_commit=verified,
            commit_source=commit_source,
        )

    return _assemble(
        question=question_text,
        run=run,
        mode=mode,
        repository=label,
        target=target,
        subject_repository=subject_display,
        pinned_commit=verified,
        commit_source=commit_source,
        router=router,
        architect=architect,
        flow_investigator=flow_investigator,
        visualizer=visualizer,
        sink=sink,
        diagram=diagram,
        artifacts=artifacts,
        artifact_directory=artifact_directory,
    )


def _assemble(
    question: str,
    run: Any,
    mode: str,
    repository: str,
    target: Optional[str],
    subject_repository: str,
    pinned_commit: str,
    commit_source: str,
    router: QueryRouter,
    architect: Any,
    flow_investigator: Any,
    visualizer: ArchitectureVisualizationAgent,
    sink: Any,
    diagram: bool,
    artifacts: bool,
    artifact_directory: Optional[str],
) -> RepositoryQueryResult:
    """Turn one finished orchestration record into the printed
    result — every field read back from what actually happened."""
    result = RepositoryQueryResult(
        question=question,
        status=run.status,
        termination_reason=run.termination_reason,
        mode=mode,
        repository=repository,
        target=target,
        subject_repository=subject_repository,
        pinned_commit=pinned_commit,
        commit_source=commit_source,
        diagram_in_scope=diagram,
    )

    # -- steps, run ids, role chain ------------------------------------
    role_by_step: Dict[str, str] = {}
    for event in run.events:
        if event.get("operation") == "orchestration.step.assign":
            role_by_step[event["step_id"]] = event.get("agent_role", "")
    last_run_id: Dict[str, Optional[str]] = {}
    for entry in run.agent_runs:
        if entry.get("run_id"):
            last_run_id[entry["step_id"]] = entry["run_id"]
    for step in run.plan.steps:
        sid = step.id
        role = role_by_step.get(sid, "")
        result.steps.append(
            StepInfo(
                step_id=sid,
                goal=step.goal,
                role=role,
                status=step.status,
                label=_step_label(router, sid, role),
                run_id=last_run_id.get(sid),
                output_type=(
                    type(run.step_result(sid).output).__name__
                    if run.step_result(sid) is not None
                    and run.step_result(sid).output is not None
                    else ""
                ),
            )
        )
    result.run_ids = [
        step.run_id for step in result.steps if step.run_id
    ]

    trace_root = Path(sink.root)
    result.trace_paths = [
        display_path(trace_root / run_id) for run_id in result.run_ids
    ]

    report = run.report
    finding = run.finding
    result.investigation_id = (
        report.investigation_id
        if report is not None
        else (finding.investigation_id if finding is not None else None)
    )
    if run.validation is not None:
        result.collaboration_status = run.validation.status

    # -- the answer ------------------------------------------------------
    # The answer is ALWAYS read back from the merged investigation
    # report when one exists — synthesized from its own statements,
    # never from the diagram (which is only a projection of that same
    # report) and never from a summary authored elsewhere.  The
    # finding is the fallback for a run whose report never came
    # together.
    if report is not None:
        statements = list(report.findings)
        result.answer = synthesize_answer(report)
        result.conclusions = _unique_statements(
            [statement.statement for statement in statements]
        )[:MAX_CONCLUSIONS]
        result.evidence = _evidence_lines(
            list(report.evidence_refs),
            [
                (statement.statement, list(statement.evidence_refs))
                for statement in statements
            ],
        )
        result.uncertainties = (
            list(report.uncertainties) + list(report.open_questions)
        )[:MAX_UNCERTAINTIES]
    elif finding is not None and isinstance(
        finding, ArchitectureFinding
    ):
        result.answer = synthesize_answer(finding)
        result.conclusions = _unique_statements(
            [conclusion.statement for conclusion in finding.conclusions]
        )[:MAX_CONCLUSIONS]
        result.evidence = _evidence_lines(
            list(finding.evidence_refs),
            [
                (conclusion.statement, list(conclusion.evidence_refs))
                for conclusion in finding.conclusions
            ],
        )
        result.uncertainties = list(finding.open_questions)[
            :MAX_UNCERTAINTIES
        ]
    else:
        result.answer = ""

    # -- diagram validation (deterministic, no model) --------------------
    spec = visualizer.last_spec
    diagram_result = None
    if diagram and report is not None and spec is not None:
        scope = DiagramScope.from_investigation_report(report)
        merged_trace = {
            "iterations": [
                iteration.to_dict()
                for trace in (
                    architect.last_trace,
                    flow_investigator.last_trace,
                )
                if trace is not None
                for iteration in trace.iterations
            ]
        }
        diagram_result = validate_diagram_spec(
            spec, scope=scope, evidence=gather_evidence(merged_trace)
        )
        result.diagram_status = diagram_result.status
        for check in diagram_result.checks:
            if not check.ok:
                result.failures.append(
                    "diagram {}: {}".format(check.rule, check.detail)
                )

    # -- artifacts (a validated spec only) -------------------------------
    collaboration_ok = (
        run.validation is not None and run.validation.status == "passed"
    )
    diagram_ok = (
        diagram_result is not None
        and diagram_result.status == "passed"
    )
    if artifacts and diagram and spec is not None and diagram_result:
        if collaboration_ok and diagram_ok:
            context = RunSummaryContext(
                architecture_report=_specialist_report(architect),
                flow_report=_specialist_report(flow_investigator),
                evidence_notes=[],
                m17_agents=_m17_agents(run, router, wrote_renderer=True),
                m17_status=run.status,
                trace_location=display_path(trace_root),
            )
            try:
                written = write_visualization_artifacts(
                    spec,
                    diagram_result,
                    context=context,
                    directory=artifact_directory,
                )
                result.artifact_directory = display_path(written.directory)
                for status in verify_artifacts(written):
                    result.artifacts.append(
                        ArtifactRow(
                            name=status.name,
                            path=display_path(status.path),
                            exists=status.exists,
                            bytes=status.bytes,
                            note=status.note,
                        )
                    )
            except (VisualizationArtifactError, OSError) as exc:
                result.artifacts_error = "{}: {}".format(
                    type(exc).__name__, exc
                )
                result.failures.append(result.artifacts_error)
    return result


def _step_label(router: QueryRouter, step_id: str, role: str) -> str:
    """The specialist label read back from the router's own
    decisions.

    A step the router never routed is NEVER reported as the
    visualization agent: an assigned-but-unroutable investigator step
    is ``"unrouted"``, and a step that was never assigned at all
    (blocked behind an earlier failure) carries no label.  The happy
    path is unchanged — routed investigators keep their specialist
    label and the assigned analyst step is ``"visualization"``.
    """
    if step_id in router.labels:
        return router.labels[step_id]
    if role == ANALYST:
        return "visualization"
    if role == INVESTIGATOR:
        return "unrouted"
    return ""


def _m17_agents(
    run: Any, router: QueryRouter, wrote_renderer: bool
) -> List[str]:
    """The participant list ``run_summary.md`` records — read back
    from the run's own assignment events, never authored here."""
    agents: List[str] = []
    for event in run.events:
        if event.get("operation") != "orchestration.step.assign":
            continue
        step_id = event["step_id"]
        role = event.get("agent_role", "")
        label = _step_label(router, step_id, role)
        agents.append("{} ({})".format(role, label))
    agents.append("validator (deterministic M16 verdict)")
    if wrote_renderer:
        agents.append(
            "renderer (deterministic Mermaid/DOT/SVG stage)"
        )
    return agents
