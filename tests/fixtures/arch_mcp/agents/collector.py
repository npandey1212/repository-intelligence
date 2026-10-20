"""Collector agent: declares one tool and reads records through it."""

from tools.fetch_record import fetch_record

TOOLS = ("fetch_record",)


class Collector:
    """Collects records for a query (read-only)."""

    def run(self, query):
        return fetch_record(query)
