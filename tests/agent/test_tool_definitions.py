"""M8.2 Agent tool-definition tests.

Proves the milestone boundary: the Agent *describes* the four existing
tools to the LLM (provider-neutral ``ToolDefinition``), recognizes
returned ``ToolCall``s, and never executes them. Offline — fake LLM
client only; no API key, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import jsonschema
import pytest

from repo_intelligence.agent import AGENT_TOOL_DEFINITIONS, Agent, AgentError
from repo_intelligence.agent.tool_definitions import (
    GET_ARCHITECTURE_CONTEXT_TOOL,
    QUERY_ENGINEERING_GRAPH_TOOL,
    READ_FILE_TOOL,
    SEARCH_REPOSITORY_TOOL,
)
from repo_intelligence.engineering.models import RELATIONSHIP_KINDS
from repo_intelligence.graph import MAX_QUERY_DEPTH, QUERY_DIRECTIONS
from repo_intelligence.llm import LLMResponse, Message, ToolCall, ToolDefinition
from repo_intelligence.tools import get_architecture_context as gac_module
from repo_intelligence.tools import query_engineering_graph as qeg_module
from repo_intelligence.tools import read_file as read_file_module
from repo_intelligence.tools import search_repository as search_repository_module

QUESTION = "What is the onboarding workflow?"
ANSWER = "The onboarding workflow is a four-step process."

AGENT_SRC = Path(__file__).resolve().parents[2] / "src" / "repo_intelligence" / "agent"
PROVIDER_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(openai|anthropic|bedrock|litellm|langchain|langgraph)\b",
    re.MULTILINE,
)
EXPECTED_TOOL_NAMES = [
    "get_architecture_context",
    "search_repository",
    "read_file",
    "query_engineering_graph",
]


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.response


def _text_response(content=ANSWER):
    return LLMResponse(
        message=Message(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_call_response(name="search_repository", arguments=None):
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name=name,
                    arguments=arguments or {"query": "onboarding workflow"},
                )
            ],
        ),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Test 1 — All tools are exposed (and nothing else)
# ---------------------------------------------------------------------------


def test_agent_request_exposes_exactly_the_four_tools():
    fake = FakeLLMClient(response=_text_response())
    Agent(llm_client=fake).run(QUESTION)
    assert [tool.name for tool in fake.requests[0].tools] == EXPECTED_TOOL_NAMES


def test_definition_list_contains_no_unrelated_tools():
    assert [tool.name for tool in AGENT_TOOL_DEFINITIONS] == EXPECTED_TOOL_NAMES


# ---------------------------------------------------------------------------
# Test 2 — Tool definitions are provider-neutral
# ---------------------------------------------------------------------------


def test_definitions_are_llm_abstraction_tool_definitions():
    for definition in AGENT_TOOL_DEFINITIONS:
        assert type(definition) is ToolDefinition  # repo_intelligence.llm


def test_agent_sources_import_no_provider_tool_types():
    for path in sorted(AGENT_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert PROVIDER_IMPORT_RE.findall(text) == [], path


# ---------------------------------------------------------------------------
# Test 3 — Schema correctness (against the actual tool contracts)
# ---------------------------------------------------------------------------


def test_get_architecture_context_schema():
    schema = GET_ARCHITECTURE_CONTEXT_TOOL.input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == set(gac_module._ALLOWED_ARGUMENTS)
    assert "required" not in schema  # no required arguments
    max_chars = schema["properties"]["max_chars"]
    assert max_chars["minimum"] == gac_module._MIN_CONTEXT_CHARS
    assert max_chars["maximum"] == gac_module.MAX_TOOL_RESPONSE_CHARS
    assert max_chars["default"] == gac_module.MAX_TOOL_RESPONSE_CHARS
    jsonschema.validate({}, schema)
    jsonschema.validate({"section": "Architecture"}, schema)
    jsonschema.validate({"section": "Technology", "max_chars": 4000}, schema)
    for invalid in (
        {"anything": 1},                       # unknown argument
        {"section": ""},                       # empty section
        {"section": 42},                       # wrong type
        {"max_chars": gac_module._MIN_CONTEXT_CHARS - 1},  # below minimum
        {"max_chars": gac_module.MAX_TOOL_RESPONSE_CHARS + 1},  # too big
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_get_architecture_context_matches_tool_contract(tmp_path):
    assert GET_ARCHITECTURE_CONTEXT_TOOL.name == gac_module.TOOL_NAME
    result = gac_module.get_architecture_context(
        arguments={"unexpected": 1}, output_root=tmp_path
    )
    assert not result.success
    assert result.error.code == "INVALID_ARGUMENTS"
    for instance in (
        {},
        None,
        {"section": "Architecture"},
        {"section": "Technology", "max_chars": 4000},
    ):
        normalized, error = gac_module._validate_arguments(instance)
        assert error is None, error  # tool accepts every schema-valid shape
    normalized, error = gac_module._validate_arguments({"section": "  "})
    assert normalized is None and error.error.code == "INVALID_ARGUMENTS"
    normalized, error = gac_module._validate_arguments({"max_chars": True})
    assert normalized is None and error.error.code == "INVALID_ARGUMENTS"


def test_search_repository_schema_matches_tool_contract():
    schema = SEARCH_REPOSITORY_TOOL.input_schema
    assert SEARCH_REPOSITORY_TOOL.name == search_repository_module.TOOL_NAME
    assert set(schema["properties"]) == set(search_repository_module._ALLOWED_ARGUMENTS)
    assert schema["required"] == ["query"]
    assert schema["additionalProperties"] is False
    max_results = schema["properties"]["max_results"]
    assert max_results["minimum"] == 1
    assert max_results["maximum"] == search_repository_module.MAX_RESULTS_LIMIT
    assert max_results["default"] == search_repository_module.DEFAULT_MAX_RESULTS


def test_search_repository_schema_examples_validate():
    schema = SEARCH_REPOSITORY_TOOL.input_schema
    jsonschema.validate(
        {"query": "onboarding workflow", "path": None, "max_results": 10}, schema
    )
    jsonschema.validate({"query": "onboarding workflow"}, schema)
    for invalid in (
        {"path": "docs"},                       # missing required query
        {"query": "q", "unexpected": 1},        # unknown argument
        {"query": "q", "max_results": 0},       # below minimum
        {"query": "q", "max_results": 51},      # above maximum
        {"query": 42},                          # wrong type
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_search_repository_schema_agrees_with_tool_validation():
    for instance in (
        {"query": "onboarding workflow"},
        {"query": "onboarding workflow", "path": None, "max_results": 10},
        {"query": "q", "path": "docs", "max_results": 1},
    ):
        normalized, error = search_repository_module._validate_arguments(instance)
        assert error is None, error  # tool accepts every schema-valid shape
    normalized, error = search_repository_module._validate_arguments({"path": "docs"})
    assert normalized is None and error.error.code == "INVALID_ARGUMENT"


def test_read_file_schema_matches_tool_contract():
    schema = READ_FILE_TOOL.input_schema
    assert READ_FILE_TOOL.name == read_file_module.TOOL_NAME
    assert set(schema["properties"]) == set(read_file_module._ALLOWED_ARGUMENTS)
    assert schema["required"] == ["path"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["path"]["minLength"] == 1
    assert schema["properties"]["line_start"]["minimum"] == 1
    assert schema["properties"]["line_end"]["minimum"] == 1


def test_read_file_schema_examples_validate():
    schema = READ_FILE_TOOL.input_schema
    jsonschema.validate(
        {"path": "docs/workflow.md", "line_start": 42, "line_end": 55}, schema
    )
    jsonschema.validate({"path": "docs/workflow.md"}, schema)
    for invalid in (
        {"line_start": 1},                      # missing required path
        {"path": "docs/workflow.md", "extra": 1},   # unknown argument
        {"path": "docs/workflow.md", "line_start": 0},  # below minimum
        {"path": "docs/workflow.md", "line_end": "55"},  # wrong type
        {"path": ""},                           # empty (non-empty contract)
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_read_file_schema_agrees_with_tool_validation():
    for instance in (
        {"path": "docs/workflow.md"},
        {"path": "docs/workflow.md", "line_start": 42, "line_end": 55},
    ):
        normalized, error = read_file_module._validate_arguments(instance)
        assert error is None, error  # tool accepts every schema-valid shape
    normalized, error = read_file_module._validate_arguments({"line_start": 1})
    assert normalized is None and error.error.code == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# Test 4 — Tool call is recognized
# ---------------------------------------------------------------------------


def test_tool_call_is_recognized_with_structured_details():
    fake = FakeLLMClient(
        response=_tool_call_response(
            name="search_repository", arguments={"query": "onboarding workflow"}
        )
    )
    with pytest.raises(AgentError) as excinfo:
        Agent(llm_client=fake).run(QUESTION)

    error = excinfo.value
    assert "recognized" in str(error)
    assert "search_repository" in str(error)
    assert "not supported" in str(error)  # M8.1 contract preserved
    assert len(error.tool_calls) == 1
    call = error.tool_calls[0]
    assert call.id == "call-1"
    assert call.name == "search_repository"
    assert call.arguments == {"query": "onboarding workflow"}


# ---------------------------------------------------------------------------
# Test 4b (M11) — query_engineering_graph schema vs tool contract
# ---------------------------------------------------------------------------


def test_query_engineering_graph_schema_matches_tool_contract():
    schema = QUERY_ENGINEERING_GRAPH_TOOL.input_schema
    assert QUERY_ENGINEERING_GRAPH_TOOL.name == qeg_module.TOOL_NAME
    assert set(schema["properties"]) == set(qeg_module._ALLOWED_ARGUMENTS)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "required" not in schema  # {} → node-kind summary (discovery)
    props = schema["properties"]
    assert props["direction"]["enum"] == list(QUERY_DIRECTIONS)
    assert (
        props["relationship_kinds"]["items"]["enum"] == list(RELATIONSHIP_KINDS)
    )
    assert props["max_depth"]["minimum"] == 0
    assert props["max_depth"]["maximum"] == MAX_QUERY_DEPTH
    assert props["start_node_id"]["type"] == ["string", "null"]
    assert props["relationship_kinds"]["type"] == ["array", "null"]
    assert props["node_kinds"]["type"] == ["array", "null"]


def test_query_engineering_graph_schema_examples_validate():
    schema = QUERY_ENGINEERING_GRAPH_TOOL.input_schema
    jsonschema.validate({}, schema)  # summary mode
    jsonschema.validate({"start_node_id": "artifact:file:pkg/mod.py"}, schema)
    jsonschema.validate({"start_node_id": None}, schema)
    jsonschema.validate(
        {
            "start_node_id": "artifact:file:pkg/mod.py",
            "direction": "both",
            "relationship_kinds": ["CONTAINS", "IMPORTS"],
            "node_kinds": ["artifact/file"],
            "max_depth": 3,
        },
        schema,
    )
    for invalid in (
        {"start_node_id": ""},                    # empty (non-empty contract)
        {"direction": "sideways"},                # unknown direction
        {"relationship_kinds": ["USES_TOOL"]},    # not in the vocabulary
        {"relationship_kinds": ["contains"]},     # case-sensitive
        {"node_kinds": [42]},                     # wrong item type
        {"max_depth": -1},                        # below minimum
        {"max_depth": 7},                         # above maximum
        {"start_node_id": "x", "unexpected": 1},  # unknown argument
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_query_engineering_graph_schema_agrees_with_tool_validation():
    for instance in (
        {},
        {"start_node_id": "artifact:file:pkg/mod.py"},
        {"start_node_id": None},
        {
            "start_node_id": "artifact:file:pkg/mod.py",
            "direction": "both",
            "relationship_kinds": ["CONTAINS"],
            "node_kinds": ["artifact/file"],
            "max_depth": 2,
        },
    ):
        assert qeg_module._validate_arguments(instance) is None
    failure = qeg_module._validate_arguments({"unexpected": 1})
    assert failure is not None and failure.error.code == "INVALID_ARGUMENT"


def test_relationship_kind_enum_is_pinned_to_the_tool_contract():
    # The agent description carries the enum as literals (the agent
    # package must not import the tool/engineering packages); tests
    # pin those literals to the actual vocabulary and query direction
    # constants so they cannot drift.
    props = QUERY_ENGINEERING_GRAPH_TOOL.input_schema["properties"]
    assert set(props["relationship_kinds"]["items"]["enum"]) == set(
        RELATIONSHIP_KINDS
    )
    assert list(props["direction"]["enum"]) == list(QUERY_DIRECTIONS)


# ---------------------------------------------------------------------------
# Test 5 — Tool is NOT executed (spy on all four application tools)
# ---------------------------------------------------------------------------


def test_application_tools_are_not_executed(monkeypatch):
    invoked = []
    monkeypatch.setattr(
        gac_module,
        "get_architecture_context",
        lambda *args, **kwargs: invoked.append("get_architecture_context"),
    )
    monkeypatch.setattr(
        search_repository_module,
        "search_repository",
        lambda *args, **kwargs: invoked.append("search_repository"),
    )
    monkeypatch.setattr(
        read_file_module,
        "read_file",
        lambda *args, **kwargs: invoked.append("read_file"),
    )
    monkeypatch.setattr(
        qeg_module,
        "query_engineering_graph_tool",
        lambda *args, **kwargs: invoked.append("query_engineering_graph"),
    )

    fake = FakeLLMClient(response=_tool_call_response())
    with pytest.raises(AgentError):
        Agent(llm_client=fake).run("Find the onboarding workflow.")

    assert invoked == []  # no application tool was ever invoked


# ---------------------------------------------------------------------------
# Test 6 — Existing simple-answer behavior remains intact
# ---------------------------------------------------------------------------


def test_plain_answer_behavior_remains_intact():
    fake = FakeLLMClient(response=_text_response())
    answer = Agent(llm_client=fake).run(QUESTION)

    assert answer == ANSWER
    request = fake.requests[0]
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[1].content == QUESTION
    assert [tool.name for tool in request.tools] == EXPECTED_TOOL_NAMES
