from typing import Hashable, List

import numpy as np
import pytest

from reach import Reach


def test_init(words: List[Hashable], vectors: np.ndarray, instance: Reach) -> None:
    assert len(instance) == 6
    assert instance.size == 50
    assert np.allclose(instance.vectors, vectors)

    sorted_words, _ = zip(
        *sorted(instance.items.items(), key=lambda x: x[1]), strict=True
    )
    assert list(sorted_words) == words

    instance_2 = Reach(vectors.tolist(), words)
    assert np.allclose(instance_2.vectors, instance.vectors)


def test_init_mismatched_lengths(words: List[Hashable], vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Reach(vectors[:5], words)

    with pytest.raises(ValueError):
        Reach(vectors, words[:5])


def test_init_unordered_items(words: List[Hashable], vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        # Need to ignore type to trick mypy
        Reach(vectors, set(words))  # type: ignore


def test_init_name(words: List[Hashable], vectors: np.ndarray) -> None:
    instance = Reach(vectors, words, name="sensei")
    assert instance.name == "sensei"


def test_init_unk_index(words: List[Hashable], vectors: np.ndarray) -> None:
    instance = Reach(vectors, words, unk_index=1)
    assert instance.unk_index == 1
    assert list(instance.sorted_items) == words


def test_readonly_attributes(instance: Reach) -> None:
    with pytest.raises(AttributeError):
        instance.indices = [0, 1, 2]  # type: ignore

    with pytest.raises(AttributeError):
        instance.items = {"dog": 1}  # type: ignore


def test_init_vectors_no_norm(instance: Reach) -> None:
    assert not hasattr(instance, "_norm_vectors")
    # Initialize norm vectors
    instance.norm_vectors[0]
    assert hasattr(instance, "norm_vectors")
    assert instance.vectors is not instance.norm_vectors


def test_init_vectors_norm(words: List[Hashable], vectors: np.ndarray) -> None:
    r = Reach(Reach.normalize(vectors), words)
    assert not hasattr(r, "_norm_vectors")
    # Initialize norm vectors
    r.norm_vectors[0]
    assert hasattr(r, "norm_vectors")
    assert r.vectors is r.norm_vectors


def test_vectors_auto_norm(vectors: np.ndarray) -> None:
    result = Reach._normalize_or_copy(vectors)

    assert np.allclose(Reach.normalize(vectors), result)


def test_vectors_auto_norm_copy(vectors: np.ndarray) -> None:
    vectors = Reach.normalize(vectors)
    result = Reach._normalize_or_copy(vectors)

    assert vectors is result
