from collections.abc import Hashable

import numpy as np
import pytest

from reach import Reach


def test_vectorize_no_unk(instance: Reach) -> None:
    with pytest.raises(ValueError):
        instance.vectorize(("donatello", "abcd"), remove_oov=False)

    with pytest.raises(ValueError):
        instance.vectorize([])

    with pytest.raises(ValueError):
        instance.vectorize("")

    vec = instance.vectorize(("donatello", "abcd"), remove_oov=True)
    assert len(vec) == 1
    assert np.allclose(vec, instance["donatello"])


def test_vectorize_unk(unk_instance: Reach) -> None:
    assert unk_instance.indices[unk_instance.unk_index] == "<UNK>"  # type: ignore
    assert np.allclose(unk_instance.vectors[-1], np.zeros(unk_instance.size))


def test_bow_no_unk(instance: Reach) -> None:
    bow = instance.bow(["donatello", "leonardo", "michelangelo"])
    assert bow == [0, 1, 3]

    bow = instance.bow(["donatello", "leonardo", "rgieurghegh"], remove_oov=True)
    assert bow == [0, 1]

    with pytest.raises(ValueError):
        instance.bow(["donatello", "wroughwuorg"], remove_oov=False)

    with pytest.raises(ValueError):
        instance.bow("")


def test_bow_unk(unk_instance: Reach) -> None:
    bow = unk_instance.bow(["donatello", "leonardo", "rgieurghegh"])
    assert bow == [0, 1, unk_instance.unk_index]

    bow = unk_instance.bow(["donatello", "leonardo", "rgieurghegh"], remove_oov=True)
    assert bow == [0, 1]


def test_transform(instance: Reach) -> None:
    with pytest.raises(ValueError):
        instance.transform([["donatello", "raphael"], ["dog"], ["clown", "donatello"]])

    matrices = instance.transform([["donatello", "raphael"], ["donatello", "splinter"]])

    expected = [
        np.stack([instance["donatello"], instance["raphael"]]),
        np.stack([instance["donatello"], instance["splinter"]]),
    ]
    for matrix, exp_matrix in zip(matrices, expected, strict=True):
        assert np.allclose(matrix, exp_matrix)

    matrices = instance.transform(
        [["donatello", "raphael"], ["rqghqgr", "splinter"]], remove_oov=True
    )

    expected = [
        np.stack([instance["donatello"], instance["raphael"]]),
        np.stack([instance["splinter"]]),
    ]
    for matrix, exp_matrix in zip(matrices, expected, strict=True):
        assert np.allclose(matrix, exp_matrix)

    with pytest.raises(ValueError):
        instance.transform([[]])

    assert instance.transform([]) == []


def test_mean_pool(instance: Reach) -> None:
    with pytest.raises(ValueError):
        instance.mean_pool(["donatello", "dog"])

    vec = instance.mean_pool(["donatello", "dog"], safeguard=False)
    assert np.allclose(vec, np.zeros_like(vec))

    vec = instance.mean_pool(["donatello", "dog"], remove_oov=True)
    assert np.allclose(vec, instance["donatello"])


def test_mean_pool_unk(unk_instance: Reach) -> None:
    vec = unk_instance.mean_pool(["donatello", "dog"])
    assert np.allclose(vec, unk_instance["donatello"] / 2)

    vec = unk_instance.mean_pool(["donatello", "dog"], safeguard=True)
    assert np.allclose(vec, unk_instance["donatello"] / 2)

    vec = unk_instance.mean_pool(["donatello", "dog"], remove_oov=True)
    assert np.allclose(vec, unk_instance["donatello"])

    vec = unk_instance.mean_pool([], safeguard=False)
    assert np.allclose(vec, np.zeros_like(vec))

    with pytest.raises(ValueError):
        unk_instance.mean_pool_corpus([[], ["dog"], ["guogrwohu"]], safeguard=True)

    matrix = unk_instance.mean_pool_corpus(
        [[], ["dog"], ["guogrwohu"]], safeguard=False
    )
    assert np.allclose(matrix, np.zeros_like(matrix))


def test_mean_pool_safeguard_keeps_dtype(
    words: list[Hashable], vectors: np.ndarray
) -> None:
    instance = Reach(vectors.astype("float32"), words)

    vec = instance.mean_pool(["dog"], safeguard=False)
    assert vec.dtype == np.float32
