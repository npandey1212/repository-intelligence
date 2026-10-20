"""M16 tool permission layers (spec §16/§23 tests 1–2).

Every role gets BOTH layers enforced: the definitions its model is
offered and the registry its calls execute through. Probes are
hermetic — fake bound callables record execution, and a probe name
that is never registered (including a name that EXISTS in the default
four-tool registry for the analyst) must come back as the existing
structured ``TOOL_NOT_FOUND`` refusal.
"""

from __future__ import annotations

import pytest

from repo_intelligence.agent.tool_definitions import AGENT_TOOL_DEFINITIONS
from repo_intelligence.llm import ToolCall
from repo_intelligence.multiagent.permissions import (
    ANALYST_TOOLS,
    INVESTIGATOR_TOOLS,
    VALIDATOR_TOOLS,
    restricted_dispatcher,
    tool_allowlist,
    tool_definitions_for,
)
from repo_intelligence.tools import ToolResult

FOUR_TOOLS = (
    "get_architecture_context",
    "query_engineering_graph",
    "read_file",
    "search_repository",
)

# Definitions travel in AGENT_TOOL_DEFINITIONS order; registries are
# sorted by the dispatcher.
FOUR_DEFINITION_ORDER = (
    "get_architecture_context",
    "search_repository",
    "read_file",
    "query_engineering_graph",
)


def _fake(name):
    calls = []

    def tool(arguments):
        calls.append(dict(arguments))
        return ToolResult(success=True, tool=name, data={"name": name})

    tool.executed = calls
    return tool


def _bound_all(names):
    return {name: _fake(name) for name in names}


class TestAllowlists:
    def test_investigator_may_use_every_read_only_tool(self):
        assert INVESTIGATOR_TOOLS == FOUR_TOOLS

    def test_analyst_and_validator_have_no_tools(self):
        assert ANALYST_TOOLS == ()
        assert VALIDATOR_TOOLS == ()

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError):
            tool_allowlist("planner")


class TestDefinitionLayer:
    def test_investigator_offered_all_four_definitions(self):
        names = [tool.name for tool in tool_definitions_for("investigator")]
        assert names == list(FOUR_DEFINITION_ORDER)

    def test_analyst_offered_no_definitions(self):
        assert tool_definitions_for("analyst") == []

    def test_validator_offered_no_definitions(self):
        assert tool_definitions_for("validator") == []

    def test_default_agent_definitions_unchanged(self):
        assert [tool.name for tool in AGENT_TOOL_DEFINITIONS] == list(
            FOUR_DEFINITION_ORDER
        )
        assert [
            tool.name for tool in tool_definitions_for("investigator", {})
        ] == []


class TestRegistryLayer:
    def test_investigator_registry_is_the_four_tools(self):
        dispatcher = restricted_dispatcher(
            "investigator", bindings=_bound_all(FOUR_TOOLS)
        )
        assert dispatcher.tool_names == FOUR_TOOLS

    def test_investigator_probe_executes_only_registered_tools(self):
        bindings = _bound_all(FOUR_TOOLS)
        dispatcher = restricted_dispatcher("investigator", bindings=bindings)
        result = dispatcher.dispatch(
            ToolCall(id="1", name="read_file", arguments={"path": "README.md"})
        )
        assert result.success is True
        assert bindings["read_file"].executed == [{"path": "README.md"}]
        probe = dispatcher.dispatch(
            ToolCall(id="2", name="write_file", arguments={"path": "x"})
        )
        assert probe.success is False
        assert probe.error is not None and probe.error.code == "TOOL_NOT_FOUND"
        assert "write_file" not in bindings["read_file"].executed

    def test_analyst_refuses_every_tool(self):
        dispatcher = restricted_dispatcher("analyst")
        assert dispatcher.tool_names == ()
        for name in ("read_file", "search_repository", "write_file"):
            result = dispatcher.dispatch(
                ToolCall(id="t", name=name, arguments={})
            )
            assert result.success is False
            assert result.error is not None
            assert result.error.code == "TOOL_NOT_FOUND"

    def test_validator_refuses_every_tool(self):
        dispatcher = restricted_dispatcher("validator")
        probe = dispatcher.dispatch(
            ToolCall(id="t", name="query_engineering_graph", arguments={})
        )
        assert probe.success is False
        assert probe.error is not None and probe.error.code == "TOOL_NOT_FOUND"

    def test_bindings_must_reference_registered_tools(self):
        with pytest.raises(ValueError):
            restricted_dispatcher("investigator", bindings={"rm": _fake("rm")})

    def test_bindings_override_only_allowed_tools(self):
        fake_read = _fake("read_file")
        dispatcher = restricted_dispatcher(
            "investigator", bindings={"read_file": fake_read}
        )
        assert dispatcher.tool_names == FOUR_TOOLS
        dispatcher.dispatch(ToolCall(id="1", name="read_file", arguments={}))
        assert fake_read.executed == [{}]
