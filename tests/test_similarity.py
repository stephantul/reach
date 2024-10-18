import logging
import unittest
from itertools import combinations
from typing import cast

import numpy as np
import numpy.typing as npt

from reach import Reach, normalize

logger = logging.getLogger(__name__)


class TestSimilarity(unittest.TestCase):
    def data(self) -> tuple[list[str], np.ndarray]:
        """Dummy data."""
        words: list[str] = [
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
    def cosine(x: np.ndarray, y: np.ndarray) -> float:
        """Compute the cosine."""
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0.0
        x = x / norm_x
        y = y / norm_y

        return (x * y).sum()

    def test_normalize_vector(self) -> None:
        """Test normalizing a vector."""
        x = np.arange(10)
        norm_x = Reach.normalize(x)
        norm_x_np = x / np.linalg.norm(x)

        self.assertTrue(np.allclose(norm_x, norm_x_np))
        self.assertTrue(np.allclose(norm_x, normalize(x)))

    def test_normalize_norm(self) -> None:
        """Test normalizing and comparing with norm."""
        x = np.arange(10)
        result = Reach.normalize(x)
        result_2 = Reach.normalize(x, cast(npt.NDArray, np.linalg.norm(x)))

        self.assertTrue(np.allclose(result, result_2))

    def test_normalize_array(self) -> None:
        """Test normalizing an array."""
        norms = []
        X = []
        for idx in range(10):
            x = np.full(shape=(10,), fill_value=idx, dtype="float32")
            X.append(x)
            norms.append(normalize(x))

        norms = np.stack(norms)

        self.assertTrue(np.allclose(normalize(np.stack(X)), norms))

    def test_similarity(self) -> None:
        """Test computing the similarity."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sim = instance.similarity(["leonardo"], ["leonardo"])
        self.assertTrue(np.isclose(sim, 1.0))

        for w1, w2 in combinations(instance.items, r=2):
            sim = instance.similarity([w1], [w2])[0][0]
            self.assertTrue(np.isclose(sim, self.cosine(instance[w1], instance[w2])))

    def test_correct_item_gets_deleted(self) -> None:
        """Test whether the correct items get deleted."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        vectors = np.ones_like(vectors)
        for word, result in zip(words, instance.most_similar(words)):
            result_itemset = set(x[0] for x in result)
            self.assertEqual(set(words) - {word}, result_itemset)

    def test_ranking(self) -> None:
        """Test whether the ranking is correct."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sim_matrix = instance.norm_vectors @ instance.norm_vectors.T
        argsorted_matrix = np.flip(np.argsort(sim_matrix, axis=1), axis=1)[:, 1:]

        for idx, w in enumerate(instance.items):
            similar_words: list[str] = [x[0] for x in instance.most_similar([w], num=10)[0]]
            indices = [instance.items[word] for word in similar_words]
            self.assertEqual(indices, argsorted_matrix[idx].tolist())

    def test_item_similarity(self) -> None:
        """Test the item similarity."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sims = instance.norm_vectors @ instance.norm_vectors.T
        sims_2 = instance.similarity(words, words)
        self.assertTrue(np.allclose(sims, sims_2))

    def test_batch_single(self) -> None:
        """Test batching for a single item."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        # Test if batch is equal to single.
        result = [[x[0] for x in sublist] for sublist in instance.most_similar(words)]
        other_result = []
        for word in words:
            other_result.append([x[0] for x in instance.most_similar([word])[0]])

        self.assertEqual(result, other_result)

    def test_batch_single_threshold(self) -> None:
        """Test thresholding for a single item."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        result = [[x[0] for x in sublist] for sublist in instance.threshold(words, threshold=0.0)]
        other_result = []
        for word in words:
            other_result.append([x[0] for x in instance.threshold([word], threshold=0.0)[0]])

        self.assertEqual(result, other_result)

    def test_threshold(self) -> None:
        """Test the thresholding in general."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        sim_matrix = instance.norm_vectors @ instance.norm_vectors.T
        sim_matrix[np.diag_indices_from(sim_matrix)] = -100

        threshold = 0.0
        for index, w in enumerate(instance.items):
            above_threshold_1: list[str] = [x[0] for x in instance.threshold([w], threshold=threshold)[0]]
            indices_1 = [instance.items[word] for word in above_threshold_1]
            sorted_items = sorted(enumerate(sim_matrix[index]), key=lambda x: x[1], reverse=True)
            self.assertEqual(indices_1, [idx for idx, x in sorted_items if x > threshold])

        threshold = 0.9
        for w in instance.items:
            above_threshold_2: list[str] = [x[0] for x in instance.threshold([w], threshold=threshold)[0]]
            indices_2 = [instance.items[word] for word in above_threshold_2]
            self.assertEqual(indices_2, [])

    def test_nearest_neighbor(self) -> None:
        """Test the nearest neighbor calculation."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        for word, vector in zip(words, vectors):
            nn1 = instance.nearest_neighbor(vector)[0][1:]
            nn2 = instance.most_similar([word])[0]
            self.assertEqual(nn1, nn2)

    def test_nearest_neighbor_threshold(self) -> None:
        """Test the nearest neighbor function."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        threshold = 0.0
        for word, vector in zip(words, vectors):
            nn1 = instance.nearest_neighbor_threshold(vector, threshold=threshold)[0][1:]
            nn2 = instance.threshold([word], threshold=threshold)[0]
            self.assertEqual(nn1, nn2)

    def test_neighbor_similarity(self) -> None:
        """Test neighborhood similarity."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        result = instance.norm_vectors[0] @ instance.norm_vectors[1:].T
        result2 = instance.vector_similarity(vectors[0], words[1:])

        self.assertTrue(np.allclose(result, result2))

        result = instance.norm_vectors[0] @ instance.norm_vectors[1].T
        result2 = instance.vector_similarity(vectors[0], [words[1]])

        self.assertEqual(result, result2)

    def test_indices_threshold(self) -> None:
        """Test the indices_threshold function."""
        words, vectors = self.data()
        instance = Reach(vectors, words)

        # Set a high threshold to test that no indices are returned
        threshold = 0.99
        for word, vector in zip(words, vectors):
            indices = list(instance.indices_threshold(np.array([vector]), threshold=threshold))[0]
            # Exclude self-similarity
            indices = indices[indices != instance.items[word]]
            self.assertEqual(indices.size, 0)

        # Set a low threshold to ensure some indices are returned
        threshold = 0.0
        for word, vector in zip(words, vectors):
            indices = list(instance.indices_threshold(np.array([vector]), threshold=threshold))[0]

            # Get the actual sorted indices
            similarities = instance.norm_vectors @ vector
            expected_indices = np.flatnonzero(similarities > threshold)
            indices_sorted = np.sort(indices)
            expected_indices_sorted = np.sort(expected_indices)

            # Assert that the filtered and sorted indices match
            self.assertTrue(np.array_equal(indices_sorted, expected_indices_sorted))
