"""Minimal ``.env`` / environment configuration for the LLM layer.

No third-party configuration dependency: a small deterministic loader
that reads ``KEY=VALUE`` lines from a ``.env`` file into the process
environment (existing environment variables are never overridden).

Format supported::

    # comment
    OPENAI_API_KEY=sk-...

Quoted values (``KEY="value"``) are unquoted. Loading is explicit — the
composition root (``python -m repo_intelligence.llm`` or application
startup) calls ``load_env()``; provider clients then read configuration
from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from ..artifacts import PROJECT_ROOT

__all__ = ["default_env_path", "load_env", "get_openai_api_key"]

DEFAULT_ENV_FILENAME = ".env"


def default_env_path() -> Path:
    """Default config file: ``<project-root>/.env``."""
    return PROJECT_ROOT / DEFAULT_ENV_FILENAME


def load_env(path: Optional[Path] = None) -> Dict[str, str]:
    """Load ``KEY=VALUE`` pairs into ``os.environ``.

    Returns only the variables that were newly set (variables already
    present in the environment win). Missing files load nothing —
    configuration stays optional until it is needed.
    """
    env_path = Path(path) if path is not None else default_env_path()
    loaded: Dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
        loaded[key] = value
    return loaded


def get_openai_api_key() -> Optional[str]:
    """The configured API key, or ``None`` when unset (never logged)."""
    return os.environ.get("OPENAI_API_KEY")
