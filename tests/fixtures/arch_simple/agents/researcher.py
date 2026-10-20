"""Research agent: answers questions through its two registered tools."""

from tools.fetch_note import fetch_note
from tools.search_notes import search_notes

TOOLS = ("search_notes", "fetch_note")


class Researcher:
    """Reads notes and answers from them (read-only)."""

    def __init__(self, root):
        self.root = root

    def answer(self, question):
        hits = search_notes(self.root, question)
        if not hits:
            return None
        return fetch_note(self.root, hits[0])
