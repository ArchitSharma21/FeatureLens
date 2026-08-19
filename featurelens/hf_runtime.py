from __future__ import annotations

try:
    import spaces  # type: ignore
except ImportError:  # Local development and unit tests.
    spaces = None


def gpu(duration: int = 60):
    if spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(duration=duration)

    def decorator(fn):
        return fn

    return decorator
