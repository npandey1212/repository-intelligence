"""Fetch one note by name (read-only)."""


def fetch_note(root, name):
    path = root / name
    if not path.exists():
        return None
    return path.read_text()
