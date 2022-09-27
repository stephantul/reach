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

    def test_correct_item_gets_deleted(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        vectors = np.ones_like(vectors)
        for word, result in zip(words, instance.most_similar(words)):
            result_itemset = set(x[0] for x in result)
            self.assertEqual(set(words) - {word}, result_itemset)

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

    def test_item_similarity(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sims = instance.norm_vectors @ instance.norm_vectors.T
        sims_2 = instance.similarity(words, words)
        self.assertTrue(np.allclose(sims, sims_2))

    def test_batch_single(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        # Test if batch is equal to single.
        result = [[x[0] for x in sublist] for sublist in instance.most_similar(words)]
        other_result = []
        for word in words:
            other_result.append([x[0] for x in instance.most_similar(word)[0]])

        self.assertEqual(result, other_result)

        result = [
            [x[0] for x in sublist]
            for sublist in instance.threshold(words, threshold=0.0)
        ]
        other_result = []
        for word in words:
            other_result.append(
                [x[0] for x in instance.threshold(word, threshold=0.0)[0]]
            )

        self.assertEqual(result, other_result)

    def test_threshold(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sim_matrix = instance.norm_vectors @ instance.norm_vectors.T
        sim_matrix[np.diag_indices_from(sim_matrix)] = -100

        threshold = 0.0
        for idx, w in enumerate(instance.items):
            above_threshold_1: List[str] = [
                x[0] for x in instance.threshold(w, threshold=threshold)[0]
            ]
            indices_1 = [instance.items[word] for word in above_threshold_1]
            sorted_items = sorted(
                enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True
            )
            self.assertEqual(
                indices_1, [idx for idx, x in sorted_items if x > threshold]
            )

        threshold = 0.9
        for idx, w in enumerate(instance.items):
            above_threshold_2: List[str] = [
                x[0] for x in instance.threshold(w, threshold=threshold)[0]
            ]
            indices_2 = [instance.items[word] for word in above_threshold_2]
            self.assertEqual(indices_2, [])

    def test_nearest_neighbor(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        for word, vector in zip(words, vectors):
            nn1 = instance.nearest_neighbor(vector)[0][1:]
            nn2 = instance.most_similar(word)[0]
            self.assertEqual(nn1, nn2)

        threshold = 0.0
        for word, vector in zip(words, vectors):
            nn1 = instance.nearest_neighbor_threshold(vector, threshold=threshold)[0][
                1:
            ]
            nn2 = instance.threshold(word, threshold=threshold)[0]
            self.assertEqual(nn1, nn2)
