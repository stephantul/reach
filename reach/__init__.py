"""A package for reading and manipulating word embeddings."""

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from reach.reach import Reach, normalize

if TYPE_CHECKING:
    from reach.autoreach import AutoReach as AutoReach

__all__ = ["Reach", "normalize"]
if find_spec("ahocorasick") is not None:
    __all__.append("AutoReach")


def __getattr__(name: str) -> Any:
    """Import AutoReach lazily."""
    if name == "AutoReach":
        from reach.autoreach import AutoReach

        return AutoReach
    raise AttributeError(f"module 'reach' has no attribute {name!r}")


__version__ = "5.0.0"
