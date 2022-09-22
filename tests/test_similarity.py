import logging
import unittest
from itertools import combinations
from typing import Tuple, List

import numpy as np
from reach import Reach, normalize


logger = logging.getLogger(__name__)


class TestSimilarity(unittest.TestCase):
    def data(self) -> Tuple[List[str], np.ndarray]:
        words = [
            "donatello",
            "leonardo",
            "raphael",
            "michelangelo",
            "splinter",
            "hideout",
        ]
        random_generator = np.random.RandomState(seed=44)
        vectors = random_generator.standard_normal((6, 50))

        return words, vectors

    @staticmethod
    def cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0.0
        x = x / norm_x
        y = y / norm_y

        return (x * y).sum()

    def test_normalize(self) -> None:
        x = np.arange(10)
        norm_x = Reach.normalize(x)
        norm_x_np = x / np.linalg.norm(x)

        self.assertTrue(np.allclose(norm_x, norm_x_np))
        self.assertTrue(np.allclose(norm_x, normalize(x)))

        norms = []
        X = []
        for idx in range(10):
            x = np.full(shape=(10,), fill_value=idx, dtype="float32")
            X.append(x)
            norms.append(normalize(x))

        norms = np.stack(norms)
        X = np.stack(X)

        self.assertTrue(np.allclose(normalize(X), norms))

    def test_similarity(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sim = instance.similarity("leonardo", "leonardo")
        self.assertTrue(np.isclose(sim, 1.0))

        for w1, w2 in combinations(instance.items, r=2):
            sim = instance.similarity(w1, w2)[0][0]
            self.assertTrue(np.isclose(sim, self.cosine(instance[w1], instance[w2])))

    def test_ranking(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sim_matrix = instance.norm_vectors @ instance.norm_vectors.T
        argsorted_matrix = np.flip(np.argsort(sim_matrix, axis=1), axis=1)[:, 1:]

        for idx, w in enumerate(instance.items):
            similar_words: List[str] = [
                x[0] for x in instance.most_similar(w, num=10)[0]
            ]
            indices = [instance.items[word] for word in similar_words]
            self.assertEqual(indices, argsorted_matrix[idx].tolist())
