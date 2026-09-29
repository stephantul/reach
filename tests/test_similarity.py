from collections.abc import Hashable
from itertools import combinations

import numpy as np

from reach import Reach, normalize


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    x = x / norm_x
    y = y / norm_y

    return (x * y).sum()


def test_normalize_vector() -> None:
    x = np.arange(10)
    norm_x = Reach.normalize(x)
    norm_x_np = x / np.linalg.norm(x)

    assert np.allclose(norm_x, norm_x_np)
    assert np.allclose(norm_x, normalize(x))


def test_normalize_norm() -> None:
    x = np.arange(10)
    result = Reach.normalize(x)
    result_2 = Reach.normalize(x, np.linalg.norm(x))

    assert np.allclose(result, result_2)


def test_normalize_array() -> None:
    norms = []
    X = []
    for idx in range(10):
        x = np.full(shape=(10,), fill_value=idx, dtype="float32")
        X.append(x)
        norms.append(normalize(x))

    assert np.allclose(normalize(np.stack(X)), np.stack(norms))


def test_similarity(instance: Reach) -> None:
    sim = instance.similarity(["leonardo"], ["leonardo"])
    assert np.isclose(sim, 1.0)

    for w1, w2 in combinations(instance.items, r=2):
        sim = instance.similarity([w1], [w2])[0][0]
        assert np.isclose(sim, cosine(instance[w1], instance[w2]))


def test_correct_item_gets_deleted(words: list[Hashable], instance: Reach) -> None:
    for word, result in zip(words, instance.most_similar(words), strict=True):
        result_itemset = set(x[0] for x in result)
        assert set(words) - {word} == result_itemset


def test_ranking(instance: Reach) -> None:
    sim_matrix = instance.norm_vectors @ instance.norm_vectors.T
    argsorted_matrix = np.flip(np.argsort(sim_matrix, axis=1), axis=1)[:, 1:]

    for idx, w in enumerate(instance.items):
        similar_words: list[Hashable] = [
            x[0] for x in instance.most_similar([w], num=10)[0]
        ]
        indices = [instance.items[word] for word in similar_words]
        assert indices == argsorted_matrix[idx].tolist()


def test_item_similarity(words: list[Hashable], instance: Reach) -> None:
    sims = instance.norm_vectors @ instance.norm_vectors.T
    sims_2 = instance.similarity(words, words)
    assert np.allclose(sims, sims_2)


def test_batch_single(words: list[Hashable], instance: Reach) -> None:
    result = [[x[0] for x in sublist] for sublist in instance.most_similar(words)]
    other_result = [[x[0] for x in instance.most_similar([word])[0]] for word in words]

    assert result == other_result


def test_batch_single_threshold(words: list[Hashable], instance: Reach) -> None:
    result = [
        [x[0] for x in sublist] for sublist in instance.threshold(words, threshold=0.0)
    ]
    other_result = [
        [x[0] for x in instance.threshold([word], threshold=0.0)[0]] for word in words
    ]

    assert result == other_result


def test_threshold(instance: Reach) -> None:
    sim_matrix = instance.norm_vectors @ instance.norm_vectors.T
    sim_matrix[np.diag_indices_from(sim_matrix)] = -100

    threshold = 0.0
    for index, w in enumerate(instance.items):
        above_threshold_1: list[Hashable] = [
            x[0] for x in instance.threshold([w], threshold=threshold)[0]
        ]
        indices_1 = [instance.items[word] for word in above_threshold_1]
        sorted_items = sorted(
            enumerate(sim_matrix[index]), key=lambda x: x[1], reverse=True
        )
        assert indices_1 == [idx for idx, x in sorted_items if x > threshold]

    threshold = 0.9
    for w in instance.items:
        above_threshold_2: list[Hashable] = [
            x[0] for x in instance.threshold([w], threshold=threshold)[0]
        ]
        indices_2 = [instance.items[word] for word in above_threshold_2]
        assert indices_2 == []


def test_nearest_neighbor(
    words: list[Hashable], vectors: np.ndarray, instance: Reach
) -> None:
    for word, vector in zip(words, vectors, strict=True):
        nn1 = instance.nearest_neighbor(vector)[0][1:]
        nn2 = instance.most_similar([word])[0]
        assert nn1 == nn2


def test_nearest_neighbor_threshold(
    words: list[Hashable], vectors: np.ndarray, instance: Reach
) -> None:
    threshold = 0.0
    for word, vector in zip(words, vectors, strict=True):
        nn1 = instance.nearest_neighbor_threshold(vector, threshold=threshold)[0][1:]
        nn2 = instance.threshold([word], threshold=threshold)[0]
        assert nn1 == nn2


def test_neighbor_similarity(
    words: list[Hashable], vectors: np.ndarray, instance: Reach
) -> None:
    result = instance.norm_vectors[0] @ instance.norm_vectors[1:].T
    result2 = instance.vector_similarity(vectors[0], words[1:])

    assert np.allclose(result, result2)

    result = instance.norm_vectors[0] @ instance.norm_vectors[1].T
    result2 = instance.vector_similarity(vectors[0], [words[1]])

    assert result == result2
