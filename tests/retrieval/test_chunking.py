"""Spec §33 A–C — chunking determinism, AST structure, generic windows.

Chunking is the deterministic half of the retrieval layer: same
input, same chunks, same ids, on every run. These tests pin that
contract plus the shared index-time file policy (binary, oversized,
ignored directories, symlinks leaving the checkout).
"""

from __future__ import annotations

from conftest import symlink_or_skip

from repo_intelligence.facts.filesystem import MAX_FILE_BYTES
from repo_intelligence.retrieval.chunking import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_CHUNK_OVERLAP,
    chunk_file,
    chunk_repository,
    classify_path,
    iter_indexable_files,
)

PY_SOURCE = """import os

def alpha():
    return 1

def beta():
    return 2

class Gamma:
    def run(self):
        return 3
"""

MD_SOURCE = "line1\nline2\nline3\nline4\nline5\n"


def test_chunk_file_is_deterministic():
    first = chunk_file("guide.md", MD_SOURCE)
    second = chunk_file("guide.md", MD_SOURCE)
    assert first == second
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_are_stable_sixteen_hex_chars():
    chunks = chunk_file("guide.md", MD_SOURCE)
    assert chunks
    for chunk in chunks:
        assert len(chunk.chunk_id) == 16
        assert int(chunk.chunk_id, 16) >= 0
    again = chunk_file("guide.md", MD_SOURCE)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_python_chunks_follow_ast_boundaries():
    chunks = chunk_file(
        "mod.py", PY_SOURCE, chunk_lines=3, chunk_overlap=0
    )
    starts = [c.line_start for c in chunks]
    # every def/class begins a chunk (the window never merges two
    # top-level definitions because the segment boundary is hard)
    assert 3 in starts  # def alpha
    assert 6 in starts  # def beta
    assert 9 in starts  # class Gamma
    for chunk in chunks:
        lines = PY_SOURCE.splitlines()[
            chunk.line_start - 1 : chunk.line_end
        ]
        assert chunk.content == "\n".join(lines)


def test_python_chunks_keep_imports_and_docstrings_intact():
    source = '"""Module doc."""\nimport os\n\n\ndef go():\n    return 1\n'
    chunks = chunk_file("m.py", source, chunk_lines=40)
    assert chunks[0].line_start == 1
    joined = "\n".join(c.content for c in chunks)
    assert '"""Module doc."""' in joined
    assert "import os" in joined


def test_generic_files_use_line_windows_with_overlap():
    chunks = chunk_file(
        "guide.md", MD_SOURCE, chunk_lines=2, chunk_overlap=1
    )
    assert [(c.line_start, c.line_end) for c in chunks] == [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    for chunk in chunks:
        expected = "\n".join(
            MD_SOURCE.splitlines()[chunk.line_start - 1 : chunk.line_end]
        )
        assert chunk.content == expected


def test_blank_only_text_yields_no_chunks():
    assert chunk_file("empty.md", "\n\n   \n") == []
    assert chunk_file("x.md", "") == []


def test_invalid_chunk_settings_fall_back_to_safe_values():
    chunks = chunk_file(
        "guide.md", MD_SOURCE, chunk_lines=0, chunk_overlap=99
    )
    assert chunks  # chunk_lines clamps to 1, overlap resets to 0


def test_language_classification_is_generic():
    assert classify_path("src/app.py") == "python"
    assert classify_path("docs/guide.md") == "markdown"
    assert classify_path("web/client.ts") == "typescript"
    assert classify_path("README") == "file"


def test_chunk_repository_counts_only_indexable_files(tmp_path):
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text(MD_SOURCE, encoding="utf-8")
    (root / "app.py").write_text(PY_SOURCE, encoding="utf-8")
    (root / "binary.txt").write_bytes(b"\x00\x01\x02")
    (root / "big.log").write_text(
        "x" * (MAX_FILE_BYTES + 1), encoding="utf-8"
    )
    files, chunks = chunk_repository(root)
    assert files == 2
    paths = {c.file_path for c in chunks}
    assert paths == {"docs/guide.md", "app.py"}


def test_chunk_repository_is_order_deterministic(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "b.md").write_text("bbb\n", encoding="utf-8")
    (root / "a").mkdir()
    (root / "a" / "a.md").write_text("aaa\n", encoding="utf-8")
    first = chunk_repository(root)
    second = chunk_repository(root)
    assert first == second


def test_iter_indexable_files_skips_noise(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text(
        "dep\n", encoding="utf-8"
    )
    (root / ".venv").mkdir()
    (root / ".venv" / "v.py").write_text("v\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG\r\n")
    rels = [rel for rel, _ in iter_indexable_files(root)]
    assert "src/app.py" in rels
    assert not any(
        rel.startswith(("node_modules/", ".venv/"))
        for rel in rels
    )
    assert "image.png" not in rels


def test_iter_indexable_files_skips_symlinks(tmp_path):
    """Symlinks leaving the checkout are never indexed — split from
    the noise test so hosts that cannot create symlinks (Windows
    without SeCreateSymbolicLinkPrivilege) still run the noise cases."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    symlink_or_skip(outside, root / "linked.md")
    rels = [rel for rel, _ in iter_indexable_files(root)]
    assert "src/app.py" in rels
    assert "linked.md" not in rels


def test_chunk_defaults_are_the_documented_values():
    assert DEFAULT_CHUNK_LINES == 40
    assert DEFAULT_CHUNK_OVERLAP == 8
    assert 0 <= DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_LINES
