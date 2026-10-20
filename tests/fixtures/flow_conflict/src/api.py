"""API entry point: routes through the bridge, not the renderer."""

from src.bridge import render


def handle(request):
    return render(request)
