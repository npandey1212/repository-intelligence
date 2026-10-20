"""Opt-in TLS verification bypass for SSL-inspecting networks.

Corporate proxies that re-sign TLS traffic
(``CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate``) block two outbound paths this project needs: the
HuggingFace model download behind the sentence-transformers
embedder, and the OpenAI API (the provider client already passes
``httpx.Client(verify=False)`` — see ``llm/openai_client.py``).

The bypass is strictly opt-in via ``REPO_INTELLIGENCE_INSECURE_SSL``
(process environment first, then the project ``.env``) and applies
to the HTTP stacks the downloads actually use:

* ``httpx`` — modern ``huggingface-hub`` and the ``openai`` SDK
  (both build ``httpx.Client``; ``verify`` is forced to ``False``).
* ``requests`` — legacy ``huggingface-hub`` / ``transformers``
  (``HTTPAdapter.cert_verify`` is redirected to the "no
  verification" branch).

Without the flag nothing is patched. Disable only on networks you
trust (or install your corporate root CA instead and keep
verification on).
"""

from __future__ import annotations

import os
from pathlib import Path

from .artifacts import PROJECT_ROOT

__all__ = [
    "INSECURE_SSL_ENV_VAR",
    "apply_insecure_ssl",
    "insecure_ssl_enabled",
]

INSECURE_SSL_ENV_VAR = "REPO_INTELLIGENCE_INSECURE_SSL"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Idempotency latch: one patch per process, applied at most once.
_APPLIED = False


def _is_truthy(value: str) -> bool:
    return value.strip().strip("\"'").lower() in _TRUTHY


def _flag_in_env_file() -> bool:
    """Read this module's flag from ``<project-root>/.env`` only.

    A dedicated single-key read (not the full llm ``load_env``) so
    the retrieval layer never injects unrelated variables such as
    ``OPENAI_API_KEY`` as a side effect.
    """
    env_path = Path(PROJECT_ROOT) / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == INSECURE_SSL_ENV_VAR:
            return _is_truthy(value)
    return False


def insecure_ssl_enabled() -> bool:
    """Whether the bypass is requested (env var, else project ``.env``)."""
    raw = os.environ.get(INSECURE_SSL_ENV_VAR)
    if raw is not None:
        return _is_truthy(raw)
    return _flag_in_env_file()


def _bypass_httpx() -> None:
    """Force ``verify=False`` on every httpx client construction."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with openai
        return

    def _make_insecure_init(original_init):
        # factory: bind the original per class (a shared loop
        # variable would be re-pointed by the next iteration)
        def _insecure_init(self, *args, **kwargs):
            kwargs["verify"] = False
            original_init(self, *args, **kwargs)

        _insecure_init._repo_insecure_ssl = True  # type: ignore
        return _insecure_init

    for cls_name in ("Client", "AsyncClient"):
        cls = getattr(httpx, cls_name, None)
        if cls is None:
            continue
        original_init = cls.__init__
        if getattr(original_init, "_repo_insecure_ssl", False):
            continue
        cls.__init__ = _make_insecure_init(original_init)


def _bypass_requests() -> None:
    """Redirect requests' cert verification to the no-verify branch."""
    try:
        import requests.adapters as adapters
    except ImportError:  # pragma: no cover - requests ships with hub
        return
    original = adapters.HTTPAdapter.cert_verify
    if getattr(original, "_repo_insecure_ssl", False):
        return

    def _insecure_cert_verify(self, conn, url, verify):
        # mirrors requests' own "verify=False" branch
        conn.assert_hostname = False
        conn.cert_reqs = "CERT_NONE"

    _insecure_cert_verify._repo_insecure_ssl = True  # type: ignore
    adapters.HTTPAdapter.cert_verify = _insecure_cert_verify


def apply_insecure_ssl() -> bool:
    """Disable TLS verification when the flag is set (idempotent).

    Returns True when the bypass is active. Call this immediately
    before building HTTP clients (third-party clients are
    constructed at import time by some libraries, so apply before
    importing them).
    """
    global _APPLIED
    if not insecure_ssl_enabled():
        return False
    if _APPLIED:
        return True
    _bypass_httpx()
    _bypass_requests()
    _APPLIED = True
    return True
