"""Git adapter.

All Git CLI interaction lives here. The rest of the application receives
plain metadata and never shells out to Git itself.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class GitError(Exception):
    """Base error for Git operations."""


class GitNotFoundError(GitError):
    """Git executable is not available on PATH."""


class CloneError(GitError):
    """Repository could not be cloned."""


class RefResolutionError(GitError):
    """The requested ref could not be resolved."""


@dataclass
class GitMetadata:
    """Metadata extracted from a checked-out Git workspace."""

    commit_sha: str
    commit_short_sha: str
    working_tree_clean: bool


def ensure_git_available() -> None:
    """Raise GitNotFoundError if the git executable is missing."""
    if shutil.which("git") is None:
        raise GitNotFoundError(
            "Git is required but was not found on PATH. "
            "Install Git and try again."
        )


def _run_git(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return stripped stdout.

    Raises GitError with stderr included on failure.
    """
    ensure_git_available()
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitNotFoundError(
            "Git is required but was not found on PATH."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown git error"
        raise GitError("git {} failed: {}".format(" ".join(args), detail))
    return result.stdout.strip()


def clone_repository(repository_url: str, destination: Path) -> Path:
    """Clone *repository_url* into *destination* (must not exist).

    Returns the destination path on success.
    """
    ensure_git_available()
    if destination.exists():
        raise CloneError(
            "Clone destination already exists: {}".format(destination)
        )

    try:
        result = subprocess.run(
            [
                "git",
                # byte-faithful pinned checkout: a user/OS
                # core.autocrlf=true would rewrite every line ending
                # on Windows and break exact read-back
                "-c",
                "core.autocrlf=false",
                "clone",
                "--quiet",
                repository_url,
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitNotFoundError(
            "Git is required but was not found on PATH."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown error"
        if "could not read" in detail or "not found" in detail.lower():
            raise CloneError(
                "Invalid or unreachable repository '{}': {}".format(
                    repository_url, detail
                )
            )
        raise CloneError(
            "Failed to clone '{}': {}".format(repository_url, detail)
        )

    return destination


def resolve_ref(repo_path: Path, ref: str) -> str:
    """Resolve *ref* to a full 40-character commit SHA.

    Tries the ref as given, then common remote-tracking / tag forms so that
    inputs like ``main`` or ``v1.0.0`` work after a default clone.
    """
    candidates = [
        ref,
        "refs/tags/{}".format(ref),
        "refs/remotes/origin/{}".format(ref),
        "origin/{}".format(ref),
    ]
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            sha = _run_git(
                ["rev-parse", "--verify", "--quiet", "{}^{{commit}}".format(candidate)],
                cwd=repo_path,
            )
        except GitError as exc:
            last_error = exc
            continue
        if sha:
            return sha.strip()

    raise RefResolutionError(
        "Cannot resolve ref '{}' in repository '{}'{}".format(
            ref,
            repo_path,
            " ({})".format(last_error) if last_error else "",
        )
    )


def checkout_ref(repo_path: Path, ref: str) -> str:
    """Resolve *ref*, check it out detached, and return the full SHA."""
    commit_sha = resolve_ref(repo_path, ref)
    _run_git(["checkout", "--quiet", "--detach", commit_sha], cwd=repo_path)

    actual = get_head_sha(repo_path)
    if actual != commit_sha:
        raise RefResolutionError(
            "Checkout mismatch: expected {} but HEAD is {}".format(
                commit_sha, actual
            )
        )
    return commit_sha


def get_head_sha(repo_path: Path) -> str:
    """Return the full commit SHA at HEAD."""
    return _run_git(["rev-parse", "HEAD"], cwd=repo_path)


def get_short_sha(repo_path: Path) -> str:
    """Return the short commit SHA at HEAD."""
    return _run_git(["rev-parse", "--short", "HEAD"], cwd=repo_path)


def is_detached_head(repo_path: Path) -> bool:
    """Return True when HEAD is detached (not pointing at a branch)."""
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path) == "HEAD"


def is_working_tree_clean(repo_path: Path) -> bool:
    """Return True when ``git status --porcelain`` reports no changes."""
    output = _run_git(["status", "--porcelain"], cwd=repo_path)
    return output == ""


def inspect_workspace(repo_path: Path) -> GitMetadata:
    """Collect Git metadata for the current checkout at *repo_path*."""
    commit_sha = get_head_sha(repo_path)
    commit_short_sha = get_short_sha(repo_path)
    clean = is_working_tree_clean(repo_path)
    return GitMetadata(
        commit_sha=commit_sha,
        commit_short_sha=commit_short_sha,
        working_tree_clean=clean,
    )
