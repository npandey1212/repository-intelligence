"""Chunking for repository retrieval (M18 §15).

Deterministic, dependency-free splitting of repository text files into
retrievable spans:

* **Python files** are chunked along structural boundaries when the
  file parses (``ast``): module preamble, class header, each method,
  each top-level function — line ranges derived from declaration
  starts (Python 3.7's AST has no ``end_lineno``, so a span runs until
  the next declaration and the file's final span runs to EOF).
* **Every other indexed language** uses plain line windows.
* Files whose AST fails to parse fall back to plain line windows —
  chunking never fails a whole index build for one odd file.

File selection reuses the shared searchable-file policy from
``facts.filesystem`` (extensions, size cap, binary sniff, ignored
directories) — one policy, no duplication. All walks are sorted, so
two builds of the same tree produce byte-identical chunk sequences.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from ..facts.filesystem import (
    MAX_FILE_BYTES,
    SEARCHABLE_SUFFIXES,
    classify_path,
    is_probably_binary,
    iter_repository_paths,
)
from .models import Chunk

__all__ = [
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_CHUNK_OVERLAP",
    "chunk_file",
    "chunk_repository",
    "iter_indexable_files",
    "read_indexable_text",
]

# Generic window policy (configurable per build; the defaults keep a
# chunk roughly one screenful of code while overlapping enough that a
# sentence straddling a boundary still matches).
DEFAULT_CHUNK_LINES = 40
DEFAULT_CHUNK_OVERLAP = 8


def iter_indexable_files(root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield ``(relative_path, resolved_path)`` in deterministic order.

    Only files under the shared searchable-file policy qualify;
    symlinks resolving outside *root* are skipped (never read).
    """
    root = Path(root)
    try:
        resolved_root = root.resolve()
    except OSError:
        resolved_root = root
    files, _directories = iter_repository_paths(root)
    for rel in files:
        if Path(rel).suffix.lower() not in SEARCHABLE_SUFFIXES:
            continue
        path = root / rel
        try:
            resolved = path.resolve()
            resolved.relative_to(resolved_root)
        except (OSError, ValueError, RuntimeError):
            continue
        yield rel, resolved


def read_indexable_text(path: Path) -> Optional[str]:
    """File text under the shared read policy, or ``None`` to skip.

    Skips: oversized files, binary sniff (NUL byte), unreadable paths.
    Text is decoded as UTF-8 with replacement — same as the tool layer
    has always done — so one odd byte never drops a file silently.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    if info.st_size > MAX_FILE_BYTES:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if is_probably_binary(raw):
        return None
    return raw.decode("utf-8", errors="replace")


def chunk_file(
    rel_path: str,
    text: str,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split *text* into chunks with deterministic line ranges."""
    if chunk_lines < 1:
        chunk_lines = 1
    if chunk_overlap < 0 or chunk_overlap >= chunk_lines:
        chunk_overlap = 0
    lines = text.splitlines()
    if not lines:
        return []
    language = classify_path(rel_path)
    segments: List[Tuple[int, int]] = []
    if language == "python":
        segments = _python_segments(lines)
    if not segments:
        segments = [(1, len(lines))]
    chunks: List[Chunk] = []
    for start, end in segments:
        for sub_start, sub_end in _window(
            lines, start, end, chunk_lines, chunk_overlap
        ):
            content = "\n".join(lines[sub_start - 1 : sub_end])
            if not content.strip():
                continue
            chunks.append(
                Chunk(
                    file_path=rel_path,
                    line_start=sub_start,
                    line_end=sub_end,
                    content=content,
                    language=language,
                )
            )
    return chunks


def chunk_repository(
    root: Path,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Tuple[int, List[Chunk]]:
    """Chunk every indexable file under *root*.

    Returns ``(files_indexed, chunks)``. Files that are skipped by the
    read policy, fail to read, or yield no non-blank content are not
    counted; every other indexed file contributes at least one chunk.
    """
    files_indexed = 0
    chunks: List[Chunk] = []
    for rel, resolved in iter_indexable_files(root):
        text = read_indexable_text(resolved)
        if text is None:
            continue
        produced = chunk_file(rel, text, chunk_lines, chunk_overlap)
        if not produced:
            continue
        files_indexed += 1
        chunks.extend(produced)
    return files_indexed, chunks


# ---------------------------------------------------------------------------
# structural (AST) segmentation for Python
# ---------------------------------------------------------------------------


def _node_start(node: ast.AST) -> int:
    """1-based first line of *node*, including decorators (py3.7-safe)."""
    starts = [getattr(node, "lineno", 1)]
    decorators = getattr(node, "decorator_list", None) or []
    for decorator in decorators:
        starts.append(getattr(decorator, "lineno", starts[0]))
    return min(starts)


def _python_segments(lines: List[str]) -> List[Tuple[int, int]]:
    """Partition a Python file into structural spans.

    Module-level statements before the first declaration form one
    preamble span; every top-level class/function then starts a span
    that runs up to the next declaration (or EOF), with classes split
    further so each method is its own span. Boundaries are declaration
    starts (Python 3.7's AST has no ``end_lineno``). Returns ``[]``
    when the file does not parse — the caller falls back to windows.
    """
    source = "\n".join(lines)
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    total = len(lines)
    declarations = [
        node
        for node in tree.body
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
    ]
    if not declarations:
        # statements only — let the caller use plain line windows
        return []
    declarations.sort(key=_node_start)
    segments: List[Tuple[int, int]] = []
    first_start = _node_start(declarations[0])
    if first_start > 1:
        segments.append((1, first_start - 1))
    for index, node in enumerate(declarations):
        start = _node_start(node)
        end = (
            _node_start(declarations[index + 1]) - 1
            if index + 1 < len(declarations)
            else total
        )
        if end < start:
            continue
        if isinstance(node, ast.ClassDef):
            segments.extend(_class_segments(node, start, end))
        else:
            segments.append((start, end))
    return segments


def _class_segments(
    node: ast.ClassDef, start: int, end: int
) -> List[Tuple[int, int]]:
    """Class header (with docstring/assignments) + one span per method."""
    members = sorted(
        [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ],
        key=_node_start,
    )
    if not members:
        return [(start, end)]
    segments: List[Tuple[int, int]] = []
    header_end = _node_start(members[0]) - 1
    if header_end >= start:
        segments.append((start, header_end))
    for index, member in enumerate(members):
        member_start = _node_start(member)
        member_end = (
            _node_start(members[index + 1]) - 1
            if index + 1 < len(members)
            else end
        )
        if member_end < member_start:
            continue
        segments.append((member_start, member_end))
    return segments


def _window(
    lines: List[str],
    start: int,
    end: int,
    chunk_lines: int,
    chunk_overlap: int,
) -> List[Tuple[int, int]]:
    """Line windows for one segment (identity when it already fits)."""
    if end - start + 1 <= chunk_lines:
        return [(start, end)]
    step = max(1, chunk_lines - chunk_overlap)
    windows: List[Tuple[int, int]] = []
    position = start
    while True:
        window_end = min(end, position + chunk_lines - 1)
        windows.append((position, window_end))
        if window_end >= end:
            break
        position += step
    return windows
