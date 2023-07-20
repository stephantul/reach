"""A package for reading and manipulating word embeddings."""
from reach.reach import Reach, normalize

try:
    from reach.autoreach import AutoReach  # noqa

    __all__ = ["Reach", "normalize", "AutoReach"]
except ImportError:
    __all__ = ["Reach", "normalize"]

__version__ = "4.1.1"
