"""Search notes by name (read-only)."""


def search_notes(root, query):
    hits = [name for name in _names(root) if query in name]
    return sorted(hits)


def _names(root):
    return [str(path.name) for path in root.iterdir()]
