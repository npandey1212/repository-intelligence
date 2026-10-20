"""Stage one: produce records and hand them to the transformer."""

from src.transform import transform


def produce(payload):
    return transform(payload)
