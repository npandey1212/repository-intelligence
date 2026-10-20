"""Git adapter package."""

from .git import (
    CloneError,
    GitError,
    GitMetadata,
    GitNotFoundError,
    RefResolutionError,
    checkout_ref,
    clone_repository,
    ensure_git_available,
    inspect_workspace,
)

__all__ = [
    "CloneError",
    "GitError",
    "GitMetadata",
    "GitNotFoundError",
    "RefResolutionError",
    "checkout_ref",
    "clone_repository",
    "ensure_git_available",
    "inspect_workspace",
]
