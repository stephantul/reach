"""A package for reading and manipulating word embeddings."""
from __future__ import absolute_import
from reach.reach import Reach, normalize

try:
    from reach.autoreach import AutoReach  # noqa

    __all__ = ["Reach", "normalize", "AutoReach"]
except ImportError:
    __all__ = ["Reach", "normalize"]
