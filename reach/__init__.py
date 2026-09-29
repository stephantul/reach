"""A package for reading and manipulating word embeddings."""

from typing import TYPE_CHECKING, Any

from reach.reach import Reach, normalize

if TYPE_CHECKING:
    from reach.autoreach import AutoReach

__all__ = ["Reach", "normalize", "AutoReach"]


def __getattr__(name: str) -> Any:
    """Import AutoReach lazily."""
    if name == "AutoReach":
        from reach.autoreach import AutoReach

        return AutoReach
    raise AttributeError(f"module 'reach' has no attribute {name!r}")


__version__ = "5.0.0"
