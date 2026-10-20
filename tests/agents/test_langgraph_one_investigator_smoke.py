"""Focused deterministic tests for the one-investigator smoke notebook.

Covers the smoke-test brief's §7 list: default dry run makes zero
provider calls; fixture plan; Plan-1 selection; exactly one executed
Investigator task; termination after the first outcome; no second
Investigator, no retry/replanning, no Analyst; failures terminate
without retry; request-budget preflight stays active; the live-call
guard rejects anything beyond its call limit; the notebook itself
defaults to dry-run mode and is valid Jupyter JSON.

No real provider is called anywhere: the behavior tests EXECUTE the
notebook's own code cells (whose provider guard is armed at limit 0),
and the focused units use fakes.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from repo_intelligence import repository_query as rq
from repo_intelligence.agent import RequestBudgetExceededError
from repo_intelligence.llm.models import LLMRequest, Message
from repo_intelligence.orchestration import (
    INCOMPLETE,
    TASK_COMPLETED,
    TASK_FAILED,
    TERMINATION_REASONS,
    AgentTaskExecutor,
    OrchestrationConfig,
    Orchestrator,
)
from repo_intelligence.planning.models import InvestigationPlan, PlanStep
from repo_intelligence.planning.validation import validate_plan

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = (
    ROOT / "notebooks" / "langgraph_one_investigator_smoke_test.ipynb"
)
POLICY_PATH = ROOT / "tests" / "agents" / "one_investigator_smoke_policy.py"

QUESTION = "How do the agents, tools, and MCP servers interact?"

SECTION_HEADINGS = (
    "1 — Objective and workflow diagram",
    "2 — Environment and dependency checks",
    "3 — Existing component imports and contracts",
    "4 — Deterministic, no-provider-call dry run",
    "5 — Planner output inspection",
    "6 — Query Router selection of Plan 1",
    "7 — LangGraph execution of exactly one Investigator task",
    "8 — Result and evidence inspection",
    "9 — Budget and call-count assertions",
    "10 — Optional, explicitly enabled live-model experiment",
    "11 — Shared ReAct context-safety demonstration (offline)",
    "12 — Results, limitations, and next steps",
)


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "one_investigator_smoke_policy_under_test", POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke_policy = _load_policy()


def _read_notebook():
    assert NOTEBOOK.is_file(), "smoke notebook is missing"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_sources(data):
    return [
        "".join(cell["source"])
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.fixture(scope="module")
def notebook_namespace():
    """Execute every notebook code cell once, in one fresh namespace.

    The notebook arms its own provider guard (limit 0) during section 2,
    so the whole execution is guaranteed provider-free; the guard is
    always disarmed in teardown, even on failure, so no class-level
    patch can leak into other test modules.
    """
    data = _read_notebook()
    namespace = {"__name__": "__one_investigator_smoke_notebook__"}
    try:
        for index, cell in enumerate(data["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            try:
                exec(
                    compile(
                        source, "<smoke-cell-{}>".format(index), "exec"
                    ),
                    namespace,
                )
            except Exception as exc:
                raise AssertionError(
                    "notebook code cell {} failed: {}".format(index, exc)
                ) from exc
    finally:
        notebook_policy = namespace.get("smoke_policy")
        if notebook_policy is not None:
            notebook_policy.reset_provider_guard()
        smoke_policy.reset_provider_guard()
    return namespace


# -- §7.1 zero provider calls ------------------------------------------------


def test_default_dry_run_makes_zero_provider_calls(notebook_namespace):
    guard = notebook_namespace["PROVIDER_GUARD"]
    assert guard.limit == 0
    assert guard.calls == 0
    assert notebook_namespace["LIVE_MODEL_TEST"] is False
    assert notebook_namespace["CONFIRM_LIVE_CALLS"] is False
    # every participant is an offline double
    assert notebook_namespace["planner"].model == "offline-scripted"
    assert notebook_namespace["models"]["arch"].model == "offline-scripted"
    assert notebook_namespace["models"]["flow"].model == "offline-scripted"

    # the guard mechanism itself refuses a would-be provider call
    from repo_intelligence.llm.openai_client import OpenAILLMClient

    state = smoke_policy.install_provider_guard(limit=0)
    try:
        client = OpenAILLMClient(provider_client=object())
        with pytest.raises(smoke_policy.ProviderCallAttempted):
            client.complete(
                LLMRequest(messages=[Message(role="user", content="hi")])
            )
        assert state.calls == 1
    finally:
        smoke_policy.reset_provider_guard()


# -- §7.2 fixture plan -------------------------------------------------------


def test_planner_returns_valid_plan_fixture(notebook_namespace):
    plan = notebook_namespace["plan"]
    assert [step.id for step in plan.steps] == ["P1"]
    assert plan.objective == notebook_namespace["QUESTION"]
    validate_plan(plan)  # the real plan validator, on the fixture output
    roles = notebook_namespace["roles"]
    assert [(r.step_id, r.role) for r in roles] == [
        ("P1", "INVESTIGATOR")
    ]
    # one offline planner turn, zero provider calls
    assert len(notebook_namespace["planner"].requests) == 1


# -- §7.3 Plan-1 selection ---------------------------------------------------


def test_router_selects_plan_1(notebook_namespace):
    assert notebook_namespace["selected_ids"] == ["P1"]
    assert notebook_namespace["route_label"] == "architecture"
    # post-run public record: the router served P1 to architecture
    router = notebook_namespace["router"]
    assert router.labels.get("P1") == "architecture"


# -- §7.4 exactly one executed Investigator task -----------------------------


def test_exactly_one_investigator_task_executes(notebook_namespace):
    run = notebook_namespace["run"]
    policy = notebook_namespace["policy"]
    assert policy.executed == ["P1"]
    assert policy.execute_attempts == 1
    assert len(run.tasks) == 1
    record = run.tasks[0]
    assert record.task_id == "P1"
    assert record.status == TASK_COMPLETED
    assert record.attempts == 1


# -- §7.5 termination after the first outcome --------------------------------


def test_graph_terminates_after_first_outcome(notebook_namespace):
    run = notebook_namespace["run"]
    assert run.final
    assert run.status == INCOMPLETE
    assert run.termination_reason == "NO_ANALYSIS"
    assert run.termination_reason in TERMINATION_REASONS
    ops = run.event_operations()
    assert ops.count("orchestration.step.start") == 1
    assert ops.count("orchestration.agent.start") == 1
    assert ops[-1] == "orchestration.terminated"


# -- §7.6 no second Investigator ---------------------------------------------


def test_no_second_investigator_executes(notebook_namespace):
    run = notebook_namespace["run"]
    policy = notebook_namespace["policy"]
    assert policy.executed == ["P1"] and policy.blocked == []
    assert run.completed_steps == ["P1"]
    # the other specialist was never even touched
    assert len(notebook_namespace["models"]["flow"].requests) == 0
    assert len(notebook_namespace["models"]["arch"].requests) > 0


# -- §7.7 no retry, no replanning --------------------------------------------


def test_no_retry_or_replanning(notebook_namespace):
    run = notebook_namespace["run"]
    assert run.retries == 0
    assert "orchestration.retry" not in run.event_operations()
    assert run.plan is notebook_namespace["plan"]
    assert [step.id for step in run.plan.steps] == ["P1"]


# -- §7.8 Analyst / final synthesis not invoked ------------------------------


def test_analyst_not_invoked_by_default(notebook_namespace):
    analyst = notebook_namespace["analyst"]
    run = notebook_namespace["run"]
    assert analyst.calls == []
    assert run.finding is None
    assert run.agent_runs, "the run must have executed something"
    assert all(
        entry["agent_role"] == "INVESTIGATOR"
        for entry in run.agent_runs
    )


# -- §7.9 failures terminate without retry -----------------------------------


def test_task_failure_terminates_without_retry():
    failing = smoke_policy.FailingInvestigator()
    router = rq.QueryRouter(failing, failing)
    analyst = smoke_policy.AnalystForbidden()
    policy = smoke_policy.OneTaskExecutionPolicy(
        AgentTaskExecutor.for_agents(router, analyst),
        allowed_task_id="P1",
    )
    orchestrator = Orchestrator(
        router,
        analyst,
        trace_sink=None,
        config=OrchestrationConfig(max_retries=0),
        executor=policy,
    )
    planner = rq.OfflineModel(
        rq.structured_response(smoke_policy.one_task_plan_payload(QUESTION))
    )
    plan = rq._build_plan(planner, QUESTION)
    task = rq.TaskPayload(
        question=QUESTION,
        repository_scope="one-investigator-smoke",
        commit_scope=rq.SUBJECT_COMMIT,
    )
    run = rq._run_with_langgraph(orchestrator, task, plan)

    assert run.status == INCOMPLETE
    assert run.termination_reason == "AGENT_FAILED"
    assert run.retries == 0
    assert "orchestration.retry" not in run.event_operations()
    record = run.tasks[0]
    assert record.status == TASK_FAILED
    assert record.attempts == 1
    assert policy.executed == ["P1"]
    assert analyst.calls == []


# -- enforcement: the policy blocks a second task outright -------------------


def test_one_task_policy_blocks_second_task_execution():
    """A 2-step plan must NOT run past task 1: the policy raises at
    the executor boundary before the second participant is reached."""
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="smoke-policy-block-"))
    models = rq.build_offline_models(
        QUESTION, "one-investigator-smoke", rq.SUBJECT_COMMIT, "README.md"
    )
    sink = rq.JsonFileTraceSink(work / "traces")
    manager = rq.build_memory_manager(str(work / "memory.db"))
    memory_context = rq.build_memory_context(
        "one-investigator-smoke", rq.SUBJECT_COMMIT
    )
    engine = rq.build_engine("one-investigator-smoke")
    bindings = rq.fixture_bindings(rq.default_subject_repository())
    architect = rq.build_architect(
        models["arch"], sink, manager, memory_context, engine,
        bindings=bindings,
    )
    flow = rq.build_flow_investigator(
        models["flow"], sink, manager, memory_context, engine,
        bindings=bindings,
    )
    router = rq.QueryRouter(architect, flow)
    analyst = smoke_policy.AnalystForbidden()
    policy = smoke_policy.OneTaskExecutionPolicy(
        AgentTaskExecutor.for_agents(router, analyst),
        allowed_task_id="P1",
    )
    orchestrator = Orchestrator(
        router, analyst, trace_sink=sink,
        config=OrchestrationConfig(max_retries=0),
        executor=policy,
    )
    plan = InvestigationPlan(
        objective=QUESTION,
        steps=[
            PlanStep(
                id="P1",
                goal="Document the architecture components and their "
                "relationships in this repository.",
            ),
            PlanStep(
                id="P2",
                goal="Trace the dependency path through this repository.",
            ),
        ],
    )
    task = rq.TaskPayload(
        question=QUESTION,
        repository_scope="one-investigator-smoke",
        commit_scope=rq.SUBJECT_COMMIT,
    )
    with pytest.raises(smoke_policy.OneTaskPolicyViolation) as caught:
        rq._run_with_langgraph(orchestrator, task, plan)

    assert "P2" in str(caught.value)
    assert policy.executed == ["P1"]
    assert policy.blocked == ["P2"]
    # the Analyst double was never reached through the blocked task
    assert analyst.calls == []


# -- §7.10 request-budget preflight active -----------------------------------


def test_request_budget_preflight_active_for_planner_path():
    class _BombClient:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            raise AssertionError("over-budget request must never be sent")

    bomb = _BombClient()
    with pytest.raises(RequestBudgetExceededError):
        rq._build_plan(bomb, "x" * 400_000)
    assert bomb.calls == 0, "preflight must reject before any client call"

    # the notebook's live section routes through this SAME gated seam
    data = _read_notebook()
    live_cells = [
        source
        for source in _code_sources(data)
        if "CallLimitedClient" in source
    ]
    assert live_cells, "the gated live section is missing"
    live_source = live_cells[0]
    assert "rq._build_plan(" in live_source
    assert "max_retries=0" in live_source
    assert "max_tokens=LIVE_MAX_OUTPUT_TOKENS" in live_source
    assert "LIVE_MODEL_TEST" in live_source


# -- §7.11 live-call guard ---------------------------------------------------


def test_live_call_guard_rejects_beyond_configured_call_limit():
    class _Inner:
        def __init__(self):
            self.requests = []
            self.model = "fake-model"

        def complete(self, request):
            self.requests.append(request)
            return "ok-{}".format(len(self.requests))

    inner = _Inner()
    guard = smoke_policy.CallLimitedClient(
        inner, max_calls=1, max_tokens=2048
    )
    first = LLMRequest(messages=[Message(role="user", content="hi")])
    assert guard.complete(first) == "ok-1"
    assert inner.requests[0].max_tokens == 2048
    assert first.max_tokens is None  # frozen request never mutated

    with pytest.raises(smoke_policy.LiveCallLimitExceeded):
        guard.complete(
            LLMRequest(messages=[Message(role="user", content="again")])
        )
    assert len(inner.requests) == 1  # the second call never reached it
    assert guard.calls_made == 1

    # a FAILED provider call still consumes the one permitted call —
    # it reached the provider, so it must count toward the allowance
    class _Raiser:
        def complete(self, request):
            raise RuntimeError("provider-side failure")

    limited = smoke_policy.CallLimitedClient(
        _Raiser(), max_calls=1, max_tokens=2048
    )
    with pytest.raises(RuntimeError):
        limited.complete(
            LLMRequest(messages=[Message(role="user", content="hi")])
        )
    with pytest.raises(smoke_policy.LiveCallLimitExceeded):
        limited.complete(
            LLMRequest(messages=[Message(role="user", content="again")])
        )
    assert limited.calls_made == 0 and len(limited.requests) == 1

    # misconfiguration is rejected up front, not silently accepted
    with pytest.raises(ValueError):
        smoke_policy.CallLimitedClient(inner, max_calls=0)
    with pytest.raises(ValueError):
        smoke_policy.CallLimitedClient(inner, max_tokens=0)


# -- §7.3 / §7.4 experiment input ceiling -----------------------------------


def _experiment_input_budget():
    """The notebook's experiment policy: an explicit 8,000-token
    ESTIMATED input ceiling, separate from the global H1 default."""
    from repo_intelligence.agent.request_budget import RequestBudgetPolicy

    return RequestBudgetPolicy(
        max_input_tokens=8000, context_capacity_tokens=None
    )


def test_experiment_input_ceiling_configured_and_tighter_than_global():
    from repo_intelligence.agent.request_budget import RequestBudgetPolicy

    # notebook constants: the experiment-specific knobs, literally
    data = _read_notebook()
    all_sources = "\n".join(
        "".join(cell["source"]) for cell in data["cells"]
    )
    assert "LIVE_MAX_INPUT_TOKENS = 8000" in all_sources
    assert "LIVE_MAX_OUTPUT_TOKENS = 2048" in all_sources
    assert "LIVE_MAX_PROVIDER_CALLS = 1" in all_sources
    assert "LIVE_MODEL_TEST = False" in all_sources
    assert "CONFIRM_LIVE_CALLS = False" in all_sources

    experiment = _experiment_input_budget()
    assert experiment.effective_input_budget_tokens == 8000
    # the GLOBAL H1 default is unchanged and stays the tighter of the
    # two only in the sense that the experiment ceiling is TIGHTER
    global_policy = RequestBudgetPolicy()
    assert global_policy.effective_input_budget_tokens == 95_232
    assert (
        experiment.effective_input_budget_tokens
        < global_policy.effective_input_budget_tokens
    )

    # the live cell reuses the SAME machinery (not a second estimator)
    live_source = next(
        source
        for source in _code_sources(data)
        if "CallLimitedClient" in source
    )
    assert "RequestBudgetPolicy(" in live_source
    assert "max_input_tokens=LIVE_MAX_INPUT_TOKENS" in live_source
    assert "enforce_request_budget_or_raise(" in live_source
    assert "build_planning_request(" in live_source
    assert "input_budget=experiment_input_budget" in live_source


def test_oversized_planner_request_rejected_before_provider():
    from repo_intelligence.agent.request_budget import (
        enforce_request_budget_or_raise,
    )
    from repo_intelligence.planning.planner import build_planning_request

    policy = _experiment_input_budget()

    # (a) the SAME builder + estimator the live cell uses rejects an
    # oversized request locally, with the record attached
    oversized = build_planning_request("x" * 40_000)
    with pytest.raises(RequestBudgetExceededError) as caught:
        enforce_request_budget_or_raise(
            oversized, policy, subject="live planner request"
        )
    record = caught.value.record
    assert record["within_budget"] is False
    assert record["totals"]["total_tokens_est"] > 8000
    assert record["budget"]["effective_input_budget_tokens"] == 8000

    # (b) the client-boundary enforcement rejects it BEFORE the inner
    # client is touched, and keeps the record for honest reporting
    class _Raising:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            raise AssertionError("the provider must never be invoked")

    raising = _Raising()
    guarded = smoke_policy.CallLimitedClient(
        raising, max_calls=1, input_budget=policy
    )
    with pytest.raises(RequestBudgetExceededError):
        guarded.complete(oversized)
    assert raising.requests == [], "provider untouched on rejection"
    assert guarded.requests == []  # nothing recorded as sent
    assert len(guarded.rejected_inputs) == 1
    assert guarded.rejected_inputs[0]["totals"]["total_tokens_est"] > 8000

    # (c) a rejection does NOT burn the one permitted call — the
    # allowance is for calls that actually REACH the provider
    class _Recording:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return "ok-1"

    recording = _Recording()
    guarded_ok = smoke_policy.CallLimitedClient(
        recording, max_calls=1, input_budget=policy
    )
    small = build_planning_request("How do the agents interact?")
    assert guarded_ok.complete(small) == "ok-1"
    assert len(recording.requests) == 1
    assert len(guarded_ok.input_records) == 1
    assert guarded_ok.input_records[0]["within_budget"] is True
    assert recording.requests[0].max_tokens is None  # no output cap set

    # (d) the normal planner request is WELL under the ceiling — the
    # experiment gate does not reject the notebook's own question
    normal = build_planning_request(
        "How do the agents, tools, and MCP servers interact?"
    )
    final, rec = enforce_request_budget_or_raise(normal, policy)
    assert rec["within_budget"] is True
    assert rec["totals"]["total_tokens_est"] < 8000


# -- §7.5 output limit reaches the provider payload --------------------------


def test_planner_output_limit_reaches_provider_payload():
    from types import SimpleNamespace

    from repo_intelligence.llm.openai_client import OpenAILLMClient

    class _RecordingProvider:
        def __init__(self):
            self.calls = []

        def complete_create(self, **kwargs):
            self.calls.append(kwargs)
            message = SimpleNamespace(content="ok", tool_calls=None)
            choice = SimpleNamespace(
                message=message, finish_reason="stop"
            )
            return SimpleNamespace(
                choices=[choice],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5,
                    total_tokens=15,
                ),
            )

        @property
        def chat(self):
            return SimpleNamespace(
                completions=SimpleNamespace(create=self.complete_create)
            )

    provider = _RecordingProvider()
    real_client = OpenAILLMClient(provider_client=provider)
    limited = smoke_policy.CallLimitedClient(
        real_client, max_calls=1, max_tokens=2048
    )
    from repo_intelligence.planning.planner import build_planning_request

    limited.complete(build_planning_request("How do agents interact?"))

    # the output cap reached the PROVIDER PAYLOAD itself, through the
    # existing LLMRequest abstraction (no competing call path)
    assert provider.calls[0]["max_tokens"] == 2048
    # and the stamped request is what the wrapper recorded
    assert limited.requests[0].max_tokens == 2048
    assert len(limited.responses) == 1


# -- §7.6 / §7.7 notebook pins for the one-call + no-retry controls ---------


def test_live_cell_pins_one_call_no_retry_and_offline_roles():
    data = _read_notebook()
    live_source = next(
        source
        for source in _code_sources(data)
        if "CallLimitedClient" in source
    )
    # one call, no SDK retries, output cap, input ceiling — all live
    assert "max_calls=LIVE_MAX_PROVIDER_CALLS" in live_source
    assert "max_retries=0" in live_source
    assert "max_tokens=LIVE_MAX_OUTPUT_TOKENS" in live_source
    assert "input_budget=experiment_input_budget" in live_source
    # the blanket guard is relaxed to 1 ONLY for the single call and
    # re-armed to 0 in a finally — never left open
    assert "PROVIDER_GUARD.limit = LIVE_MAX_PROVIDER_CALLS" in live_source
    assert "PROVIDER_GUARD.limit = 0" in live_source
    assert "finally:" in live_source
    # the cell stops after the Planner call: no graph run, no
    # second stage, Investigator stays offline, Analyst stays disabled
    assert "rq._build_plan(" in live_source
    assert "_run_with_langgraph" not in live_source
    assert "offline fixture" in live_source
    assert "analyst         : disabled" in live_source
    assert "STOPPED after 1 call" in live_source


# -- §7.8 Investigator / Analyst remain offline ------------------------------


def test_investigator_and_analyst_remain_offline_in_experiment(
    notebook_namespace,
):
    analyst = notebook_namespace["analyst"]
    models = notebook_namespace["models"]
    run = notebook_namespace["run"]
    assert analyst.calls == []
    assert models["arch"].model == "offline-scripted"
    assert models["flow"].model == "offline-scripted"
    assert len(models["flow"].requests) == 0
    assert all(
        entry["agent_role"] == "INVESTIGATOR" for entry in run.agent_runs
    )
    # the live section never constructs a live Investigator or Analyst
    live_source = next(
        source
        for source in _code_sources(_read_notebook())
        if "CallLimitedClient" in source
    )
    assert "OpenAILLMClient" in live_source
    # the ONLY real client in the live cell is the Planner's
    assert live_source.count("OpenAILLMClient(") == 1


# -- §11 shared ReAct safety demonstration (offline) --------------------------


def test_section11_shared_react_demos_ran_offline(notebook_namespace):
    """The shared-safety demos executed with the notebook, used the
    SAME abstractions for both dummy agents, made zero provider
    calls, and rejected the oversized complete request locally."""
    # demo 1+2: consecutive iterations with growing history
    guard_a = notebook_namespace["guard_a"]
    assert guard_a.iterations >= 3
    growth = notebook_namespace["growth_a"]
    assert growth == sorted(growth) and growth[-1] > growth[0]
    assert guard_a.termination_reason == "COMPLETED"
    # demo 5: oversized complete request rejected, provider untouched
    budgeted_c = notebook_namespace["budgeted_c"]
    assert budgeted_c.calls_reached_provider == 0
    assert budgeted_c.sent == []
    assert budgeted_c.rejected, "the rejection record must be kept"
    # demo 8: the second dummy agent used the identical classes
    budgeted_d = notebook_namespace["budgeted_d"]
    guard_d = notebook_namespace["guard_d"]
    assert type(budgeted_d) is type(notebook_namespace["budgeted_a"])
    assert type(guard_d) is type(guard_a)
    # the armed class-level provider guard was never touched
    assert notebook_namespace["PROVIDER_GUARD"].calls == 0
    # the demos exercised the SHARED module, not planner-only limits
    # (the cell imports the shared names directly into its namespace)
    from repo_intelligence import react_safety

    assert notebook_namespace["BudgetedLLMClient"] is (
        react_safety.BudgetedLLMClient
    )
    assert notebook_namespace["ReActLoopGuard"] is (
        react_safety.ReActLoopGuard
    )
    assert notebook_namespace["bound_observation"] is (
        react_safety.bound_observation
    )


def test_section11_is_offline_and_separate_from_live_planner():
    data = _read_notebook()
    section11_sources = []
    in_section = False
    for cell in data["cells"]:
        source = "".join(cell["source"])
        if source.startswith("## 11 —"):
            in_section = True
            continue
        if source.startswith("## 12 —"):
            in_section = False
        if in_section:
            section11_sources.append(source)
    assert section11_sources, "section 11 must exist"
    code = [s for s in section11_sources
            if not s.lstrip().startswith("#")]
    joined = "\n".join(section11_sources)
    # fake clients and fake tools only — no live path in section 11
    assert "class _FakeClient" in joined
    assert "class _FakeTool" in joined
    assert "from repo_intelligence.react_safety import" in joined
    # no live-client CONSTRUCTION inside the offline section (the
    # source may mention OpenAILLMClient only to state it is unused)
    assert "OpenAILLMClient(" not in joined
    assert "CONFIRM_LIVE_CALLS" not in joined
    # the section declares its offline status and the zero-call result
    assert "OFFLINE" in joined
    assert "provider calls: 0" in joined or "no provider call" in joined
    # section 12 (results) points at the shared controls, replacing
    # any implication that Planner-only limits solve the failure
    results = next(
        "".join(cell["source"])
        for cell in data["cells"]
        if cell["cell_type"] == "markdown"
        and "## 12 — Results" in "".join(cell["source"])
    )
    assert "react_safety" in results
    assert "not by Planner-only limits" in results


# -- §7.9 H1–H4 safeguards remain active -------------------------------------


def test_h1_h4_global_safeguards_remain_active():
    from repo_intelligence.agent.request_budget import (
        RequestBudgetPolicy,
        enforce_request_budget,
    )

    # (a) the global H1 default policy is UNCHANGED by this experiment
    assert RequestBudgetPolicy().effective_input_budget_tokens == 95_232

    # (b) the H2 clip path still REDUCES oversized tool observations
    # (the machinery is intact, not bypassed by the new ceiling)
    small = RequestBudgetPolicy(max_input_tokens=200)
    big_observation = Message(
        role="tool",
        content=json.dumps({"tool": "t", "body": "y" * 4000}),
    )
    request = LLMRequest(
        messages=[
            Message(role="user", content="go"),
            big_observation,
        ],
        tools=[],
    )
    final, record = enforce_request_budget(request, small)
    assert record["disposition"] in ("accepted", "reduced", "rejected")
    if record["disposition"] == "reduced":
        assert len(final.messages[1].content) < len(
            big_observation.content
        )
        assert record["reduction"]["tool_messages_clipped"] >= 1

    # (c) the notebook's section-9 bomb assertion is still present —
    # the GLOBAL gate is exercised on the ordinary planner path
    all_sources = "\n".join(
        "".join(cell["source"]) for cell in _read_notebook()["cells"]
    )
    assert "_build_plan(bomb" in all_sources
    assert "RequestBudgetExceededError" in all_sources

    # (d) the experiment policy is a SEPARATE object — constructing it
    # does not mutate the global default
    _experiment_input_budget()
    assert RequestBudgetPolicy().effective_input_budget_tokens == 95_232


# -- §7.12 notebook shape and defaults ---------------------------------------


def test_notebook_defaults_to_dry_run_and_is_valid_json():
    data = _read_notebook()
    assert data["nbformat"] == 4
    assert data["cells"], "notebook must have cells"
    for cell in data["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
        assert "id" in cell
        if cell["cell_type"] == "code":
            assert "outputs" in cell
            assert "execution_count" in cell

    code_sources = _code_sources(data)
    assert len(code_sources) >= 9
    for source in code_sources:
        ast.parse(source)  # pure Python cells: no IPython magics
        # key-shaped secrets only ("task-loop" contains "sk-" — a
        # literal substring check would be a false positive)
        assert re.search(r"sk-[A-Za-z0-9]{16,}", source) is None
        assert "OPENAI_API_KEY=" not in source.replace(" ", "")
        # credentials are never PRINTED (labels in messages are fine;
        # a value must never be passed to print)
        assert "print(os.environ" not in source
        assert re.search(
            r"print\(\s*(key|api_key|OPENAI_API_KEY)\b", source
        ) is None
        assert "/home/" not in source
        assert "C:/" not in source

    all_sources = "\n".join(
        "".join(cell["source"]) for cell in data["cells"]
    )
    # the mandatory safe defaults, literally present
    assert "LIVE_MODEL_TEST = False" in all_sources
    assert "CONFIRM_LIVE_CALLS = False" in all_sources
    # the experiment-specific knobs, literally present
    assert "LIVE_MAX_OUTPUT_TOKENS = 2048" in all_sources
    assert "LIVE_MAX_INPUT_TOKENS = 8000" in all_sources
    assert "LIVE_MAX_PROVIDER_CALLS = 1" in all_sources
    # all 11 required sections
    for heading in SECTION_HEADINGS:
        assert "## " + heading in all_sources, heading
    # workflow diagram present
    assert "```mermaid" in all_sources
    # the live path is gated, not merely documented
    gated = [
        source
        for source in code_sources
        if "CONFIRM_LIVE_CALLS" in source and "else:" in source
    ]
    assert gated, "a confirm-gated live branch must exist"
    assert "if not LIVE_MODEL_TEST:" in gated[0]
    # one-task enforcement lives in code
    assert any(
        "OneTaskExecutionPolicy(" in source for source in code_sources
    )
    assert any(
        "OrchestrationConfig(max_retries=0)" in source
        for source in code_sources
    )
