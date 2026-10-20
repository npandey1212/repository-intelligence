"""REPO_INTELLIGENCE_INSECURE_SSL opt-in TLS bypass (net.py).

Corporate SSL-inspecting proxies fail certificate verification for
the HuggingFace model download (huggingface-hub over httpx, or the
legacy requests stack). The bypass must:

* stay off unless requested (process env, else project ``.env``);
* force ``verify=False`` on httpx clients and redirect requests'
  ``cert_verify`` to its no-verification branch;
* patch once per process (idempotent) and never double-wrap;
* run before sentence-transformers is imported (hub clients can be
  constructed at import time).
"""

from __future__ import annotations

import sys
import types

import pytest

from repo_intelligence import net
from repo_intelligence.retrieval.errors import RetrievalError

VAR = net.INSECURE_SSL_ENV_VAR


# ---------------------------------------------------------------------------
# flag resolution (process environment first, project .env second)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ('"1"', True),
        ("0", False),
        ("false", False),
        ("", False),
        ("maybe", False),
    ],
)
def test_flag_reads_process_environment(monkeypatch, value, expected):
    # always set (even falsy values): the process env must win over
    # any project .env on this machine
    monkeypatch.setenv(VAR, value)
    assert net.insecure_ssl_enabled() is expected


def test_flag_reads_project_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setattr(net, "PROJECT_ROOT", tmp_path)
    assert net.insecure_ssl_enabled() is False

    (tmp_path / ".env").write_text(
        "# proxy workaround\n{}=1\n".format(VAR), encoding="utf-8"
    )
    assert net.insecure_ssl_enabled() is True


def test_flag_in_env_file_ignores_comments_and_falsy(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setattr(net, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "# {}=1\nOTHER=2\n".format(VAR), encoding="utf-8"
    )
    assert net.insecure_ssl_enabled() is False
    (tmp_path / ".env").write_text(
        "{} = nope\n".format(VAR), encoding="utf-8"
    )
    assert net.insecure_ssl_enabled() is False


def test_process_environment_overrides_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(net, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "{}=1\n".format(VAR), encoding="utf-8"
    )
    monkeypatch.setenv(VAR, "0")
    assert net.insecure_ssl_enabled() is False


# ---------------------------------------------------------------------------
# application: off by default, on when requested, idempotent
# ---------------------------------------------------------------------------


def test_apply_is_a_noop_without_the_flag(monkeypatch, tmp_path):
    import httpx

    import requests.adapters as adapters

    monkeypatch.delenv(VAR, raising=False)
    monkeypatch.setattr(net, "PROJECT_ROOT", tmp_path)  # no .env
    monkeypatch.setattr(net, "_APPLIED", False)
    httpx_init = httpx.Client.__init__
    cert_verify = adapters.HTTPAdapter.cert_verify
    assert net.apply_insecure_ssl() is False
    assert httpx.Client.__init__ is httpx_init
    assert adapters.HTTPAdapter.cert_verify is cert_verify


def test_apply_forces_verify_false_on_httpx_clients(monkeypatch):
    class _FakeClient:
        def __init__(self, verify=True):
            self.verify = verify

    fake_httpx = types.SimpleNamespace(
        Client=_FakeClient, AsyncClient=_FakeClient
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setenv(VAR, "1")
    monkeypatch.setattr(net, "_APPLIED", False)

    assert net.apply_insecure_ssl() is True
    assert _FakeClient().verify is False
    assert _FakeClient(verify=True).verify is False


def test_apply_redirects_requests_cert_verify(monkeypatch):
    class _FakeAdapter:
        def cert_verify(self, conn, url, verify):  # original behavior
            raise AssertionError("original cert_verify must not run")

    fake_adapters = types.SimpleNamespace(HTTPAdapter=_FakeAdapter)
    fake_requests = types.SimpleNamespace(adapters=fake_adapters)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setitem(sys.modules, "requests.adapters", fake_adapters)
    monkeypatch.setenv(VAR, "1")
    monkeypatch.setattr(net, "_APPLIED", False)

    assert net.apply_insecure_ssl() is True
    conn = types.SimpleNamespace()
    _FakeAdapter().cert_verify(conn, "https://example.invalid", True)
    # the no-verification branch, even when verify was requested
    assert conn.cert_reqs == "CERT_NONE"
    assert conn.assert_hostname is False


def test_apply_is_idempotent(monkeypatch):
    class _FakeClient:
        def __init__(self, verify=True):
            self.verify = verify

    fake_httpx = types.SimpleNamespace(
        Client=_FakeClient, AsyncClient=_FakeClient
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setenv(VAR, "1")
    monkeypatch.setattr(net, "_APPLIED", False)

    assert net.apply_insecure_ssl() is True
    first = _FakeClient.__init__
    assert net.apply_insecure_ssl() is True
    assert _FakeClient.__init__ is first  # never double-wrapped


def test_apply_patches_the_installed_httpx_and_requests(monkeypatch):
    """End-to-end on the real installed stacks (restored on teardown)."""
    import httpx
    import requests.adapters as adapters

    monkeypatch.setenv(VAR, "1")
    monkeypatch.setattr(net, "_APPLIED", False)
    monkeypatch.setattr(
        httpx.Client, "__init__", httpx.Client.__init__
    )
    monkeypatch.setattr(
        httpx.AsyncClient, "__init__", httpx.AsyncClient.__init__
    )
    monkeypatch.setattr(
        adapters.HTTPAdapter,
        "cert_verify",
        adapters.HTTPAdapter.cert_verify,
    )

    assert net.apply_insecure_ssl() is True
    assert getattr(httpx.Client.__init__, "_repo_insecure_ssl", False)
    assert getattr(
        httpx.AsyncClient.__init__, "_repo_insecure_ssl", False
    )
    assert getattr(
        adapters.HTTPAdapter.cert_verify, "_repo_insecure_ssl", False
    )
    # still constructible: verify is forced, not rejected
    client = httpx.Client()
    assert client is not None
