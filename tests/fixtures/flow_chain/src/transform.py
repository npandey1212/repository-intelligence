"""Stage two: transform records, then publish them."""

from src.publisher import publish


def transform(payload):
    return publish({"value": payload})
