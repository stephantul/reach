from collections.abc import Hashable

import numpy as np
import pytest

from reach import Reach


def test_init(words: list[Hashable], vectors: np.ndarray, instance: Reach) -> None:
    assert len(instance) == 6
    assert instance.size == 50
    assert np.allclose(instance.vectors, vectors)

    sorted_words, _ = zip(
        *sorted(instance.items.items(), key=lambda x: x[1]), strict=True
    )
    assert list(sorted_words) == words

    instance_2 = Reach(vectors.tolist(), words)
    assert np.allclose(instance_2.vectors, instance.vectors)


def test_init_mismatched_lengths(words: list[Hashable], vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Reach(vectors[:5], words)

    with pytest.raises(ValueError):
        Reach(vectors, words[:5])


def test_init_unordered_items(words: list[Hashable], vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        # Need to ignore type to trick mypy
        Reach(vectors, set(words))  # type: ignore


def test_init_name(words: list[Hashable], vectors: np.ndarray) -> None:
    instance = Reach(vectors, words, name="sensei")
    assert instance.name == "sensei"


def test_init_unk_index(words: list[Hashable], vectors: np.ndarray) -> None:
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


def test_init_vectors_norm(words: list[Hashable], vectors: np.ndarray) -> None:
    r = Reach(Reach.normalize(vectors), words)
    assert not hasattr(r, "_norm_vectors")
    # Initialize norm vectors
    r.norm_vectors[0]
    assert hasattr(r, "norm_vectors")
    assert r.vectors is r.norm_vectors


@pytest.mark.parametrize("prenormalize", [True, False])
def test_vectors_read_only(
    words: list[Hashable], vectors: np.ndarray, prenormalize: bool
) -> None:
    if prenormalize:
        vectors = Reach.normalize(vectors)
    instance = Reach(vectors, words)

    with pytest.raises(ValueError, match="read-only"):
        instance.vectors[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        instance.norm_vectors[0] = 0


def test_vectors_input_stays_writeable(
    words: list[Hashable], vectors: np.ndarray
) -> None:
    Reach(vectors, words)

    assert vectors.flags.writeable


def test_vectors_setter_updates_norm_vectors(
    words: list[Hashable], vectors: np.ndarray
) -> None:
    instance = Reach(Reach.normalize(vectors), words)
    assert instance.vectors is instance.norm_vectors

    instance.vectors = vectors * 2
    assert instance.vectors is not instance.norm_vectors
    assert np.allclose(instance.norm_vectors, Reach.normalize(vectors))
    assert np.allclose(instance.vectors, vectors * 2)

    instance.vectors = Reach.normalize(vectors)
    assert instance.vectors is instance.norm_vectors


def test_vectors_auto_norm(vectors: np.ndarray) -> None:
    result = Reach._normalize_or_copy(vectors)

    assert np.allclose(Reach.normalize(vectors), result)


def test_vectors_auto_norm_copy(vectors: np.ndarray) -> None:
    vectors = Reach.normalize(vectors)
    result = Reach._normalize_or_copy(vectors)

    assert vectors is result


def test_init_duplicate_items(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Reach(vectors, ["a", "a", "b", "c", "d", "e"])


@pytest.mark.parametrize("unk_index", [-1, 6])
def test_init_unk_index_out_of_range(
    words: list[Hashable], vectors: np.ndarray, unk_index: int
) -> None:
    with pytest.raises(ValueError, match="out of range"):
        Reach(vectors, words, unk_index=unk_index)


def test_contains(instance: Reach) -> None:
    assert "donatello" in instance
    assert "shredder" not in instance


def test_intersect_remaps_unk(words: list[Hashable], vectors: np.ndarray) -> None:
    instance = Reach(vectors, words, name="sensei", unk_index=5)
    result = instance.intersect(["leonardo", "hideout"])

    assert result.unk_index == 1
    assert result.indices[result.unk_index] == "hideout"
    assert result.name == "sensei"
    assert np.allclose(result["hideout"], instance["hideout"])


def test_union(words: list[Hashable], vectors: np.ndarray) -> None:
    instance = Reach(vectors[:3], words[:3], name="sensei")
    other = Reach(vectors[2:], words[2:], unk_index=3)
    result = instance.union(other)

    assert list(result.sorted_items) == words
    assert result.name == "sensei"
    assert result.unk_index == 5
    assert np.allclose(result.vectors, vectors)


def test_union_keeps_own_unk(words: list[Hashable], vectors: np.ndarray) -> None:
    instance = Reach(vectors[:3], words[:3], unk_index=0)
    other = Reach(vectors[2:], words[2:], unk_index=3)

    assert instance.union(other).unk_index == 0


def test_normalize_int_with_zero_row() -> None:
    result = Reach.normalize(np.array([[3, 4], [0, 0]]))

    assert np.allclose(result, [[0.6, 0.8], [0.0, 0.0]])
