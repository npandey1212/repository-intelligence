"""LLMClient contract, dependency rule, and configuration tests.

Verifies that the abstraction boundary itself holds: any duck-typed
client satisfies the Protocol, the provider-neutral modules never touch
a provider SDK, and ``.env`` configuration works without extra deps.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from repo_intelligence.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
)
from repo_intelligence.llm.config import get_openai_api_key, load_env

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "repo_intelligence"
OPENAI_CLIENT = SRC_ROOT / "llm" / "openai_client.py"

# Any provider SDK import at the top of a source line.
# LangGraph and the langchain agent framework are deliberately NOT in
# this list: they are orchestration/agent frameworks, not provider
# SDKs. LangGraph is sanctioned by the orchestration-package guard
# (which names the ONE adapter file allowed to import it); langchain
# is sanctioned by the framework-allowlist test below (which confines
# it to the langgraph_poc package). The provider adapter
# ``langchain_openai`` IS a provider SDK and remains banned
# everywhere, including inside the POC. Provider SDKs remain banned
# everywhere, including inside graph modules.
PROVIDER_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(openai|anthropic|bedrock|litellm|langchain_openai)\b",
    re.MULTILINE,
)

# The agent framework itself (langchain / langchain_core) may be
# imported ONLY inside the langgraph_poc POC package — the one place
# the ``create_agent`` seam lives. Everywhere else the project's own
# provider-neutral ``llm`` contract remains the only model boundary.
LANGCHAIN_FRAMEWORK_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(langchain|langchain_core)\b",
    re.MULTILINE,
)
LANGGRAPH_POC_ROOT = SRC_ROOT / "langgraph_poc"


# ---------------------------------------------------------------------------
# LLMClient protocol
# ---------------------------------------------------------------------------


class _FakeClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(role="assistant", content="ok"),
            finish_reason="stop",
        )


def test_fake_client_satisfies_protocol_at_runtime():
    client = _FakeClient()
    assert isinstance(client, LLMClient)


def test_random_object_does_not_satisfy_protocol():
    assert not isinstance(object(), LLMClient)


def test_protocol_flow_is_provider_neutral():
    client = _FakeClient()
    response = client.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert isinstance(response, LLMResponse)
    assert response.message.content == "ok"


def test_llm_error_is_a_plain_exception():
    assert issubclass(LLMError, Exception)
    error = LLMError("provider said no")
    assert str(error) == "provider said no"


def test_public_package_surface_is_provider_neutral():
    import repo_intelligence.llm as llm_package

    assert set(llm_package.__all__) == {
        "LLMClient",
        "LLMError",
        "LLMRequest",
        "LLMResponse",
        "Message",
        "StructuredOutputSpec",
        "StructuredOutputValidationError",
        "ToolCall",
        "ToolDefinition",
        "Usage",
        "normalize_structured_payload",
        "spec_for_model",
        "validate_structured",
        "is_context_length_error",
    }
    assert not hasattr(llm_package, "OpenAILLMClient")


# ---------------------------------------------------------------------------
# Dependency rule: provider SDK only inside llm/openai_client.py
# ---------------------------------------------------------------------------


def test_provider_sdk_imported_only_by_openai_client():
    violations = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in PROVIDER_IMPORT_RE.finditer(text):
            if path != OPENAI_CLIENT:
                violations.append((str(path), match.group(0)))
    assert violations == []


def test_openai_client_is_the_single_provider_module():
    text = OPENAI_CLIENT.read_text(encoding="utf-8")
    matches = PROVIDER_IMPORT_RE.findall(text)
    assert matches == ["openai"]  # exactly one provider import, in one file


def test_langchain_framework_imports_confined_to_poc_package():
    """langchain/langchain_core may appear ONLY under
    ``langgraph_poc/`` — the one create_agent seam. The provider
    adapter (``langchain_openai``) stays banned everywhere via
    ``PROVIDER_IMPORT_RE``; the POC itself never imports it (the
    caller supplies the model instance)."""
    violations = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if LANGGRAPH_POC_ROOT in path.parents or path.parent == (
            LANGGRAPH_POC_ROOT
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for match in LANGCHAIN_FRAMEWORK_IMPORT_RE.finditer(text):
            violations.append((str(path), match.group(0)))
    assert violations == []


def test_poc_package_never_imports_a_provider_sdk():
    """The langgraph_poc seam takes an injected model — it must not
    import any provider SDK or the langchain provider adapter."""
    violations = []
    for path in sorted(LANGGRAPH_POC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in PROVIDER_IMPORT_RE.finditer(text):
            violations.append((str(path), match.group(0)))
    assert violations == []


# ---------------------------------------------------------------------------
# Configuration (.env loading, no third-party dependency)
# ---------------------------------------------------------------------------


def test_load_env_reads_key_value_pairs(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "OPENAI_API_KEY=sk-test-123\n"
        'QUOTED_VALUE="has spaces"\n'
        "NO_EQUALS_SIGN\n",
        encoding="utf-8",
    )
    loaded = load_env(env_file)
    assert loaded == {
        "OPENAI_API_KEY": "sk-test-123",
        "QUOTED_VALUE": "has spaces",
    }
    assert os.environ["OPENAI_API_KEY"] == "sk-test-123"


def test_load_env_does_not_override_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    assert load_env(env_file) == {}
    assert os.environ["OPENAI_API_KEY"] == "from-environment"


def test_load_env_missing_file_loads_nothing(tmp_path):
    assert load_env(tmp_path / "does-not-exist.env") == {}


def test_get_openai_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    assert get_openai_api_key() == "sk-abc"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert get_openai_api_key() is None
