import logging
import unittest
from typing import Hashable, List, Tuple

import numpy as np

from reach import Reach

logger = logging.getLogger(__name__)


class TestVectorize(unittest.TestCase):
    def data(self) -> Tuple[List[Hashable], np.ndarray]:
        words: List[Hashable] = [
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

    def test_vectorize_no_unk(self) -> None:
        words, vectors = self.data()
        reach = Reach(vectors, words)

        with self.assertRaises(ValueError):
            reach.vectorize(("donatello", "abcd"), remove_oov=False)

        with self.assertRaises(ValueError):
            reach.vectorize([])

        with self.assertRaises(ValueError):
            reach.vectorize("")

        vec = reach.vectorize(("donatello", "abcd"), remove_oov=True)
        self.assertEqual(len(vec), 1)
        self.assertTrue(np.allclose(vec, reach["donatello"]))

    def test_vectorize_unk(self) -> None:
        words, vectors = self.data()
        words.append("<UNK>")
        vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
        reach = Reach(vectors, words, unk_index=len(words) - 1)

        self.assertEqual(reach.indices[reach.unk_index], "<UNK>")  # type: ignore
        self.assertTrue(np.allclose(vectors[-1], np.zeros(reach.size)))

    def test_bow_no_unk(self) -> None:
        words, vectors = self.data()
        reach = Reach(vectors, words)

        bow = reach.bow(["donatello", "leonardo", "michelangelo"])
        self.assertEqual(bow, [0, 1, 3])

        bow = reach.bow(["donatello", "leonardo", "rgieurghegh"], remove_oov=True)
        self.assertEqual(bow, [0, 1])

        with self.assertRaises(ValueError):
            bow = reach.bow(["donatello", "wroughwuorg"], remove_oov=False)

        with self.assertRaises(ValueError):
            bow = reach.bow("")

    def test_bow_unk(self) -> None:
        words, vectors = self.data()
        words.append("<UNK>")
        vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
        reach = Reach(vectors, words, unk_index=len(words) - 1)

        bow = reach.bow(["donatello", "leonardo", "rgieurghegh"])
        self.assertEqual(bow, [0, 1, reach.unk_index])

        bow = reach.bow(["donatello", "leonardo", "rgieurghegh"], remove_oov=True)
        self.assertEqual(bow, [0, 1])

    def test_transform(self) -> None:
        words, vectors = self.data()
        reach = Reach(vectors, words)

        with self.assertRaises(ValueError):
            reach.transform([["donatello", "raphael"], ["dog"], ["clown", "donatello"]])

        matrices = reach.transform(
            [["donatello", "raphael"], ["donatello", "splinter"]]
        )

        expected = [
            np.stack([reach["donatello"], reach["raphael"]]),
            np.stack([reach["donatello"], reach["splinter"]]),
        ]
        for matrix, exp_matrix in zip(matrices, expected):
            self.assertTrue(np.allclose(matrix, exp_matrix))

        matrices = reach.transform(
            [["donatello", "raphael"], ["rqghqgr", "splinter"]], remove_oov=True
        )

        expected = [
            np.stack([reach["donatello"], reach["raphael"]]),
            np.stack([reach["splinter"]]),
        ]
        for matrix, exp_matrix in zip(matrices, expected):
            self.assertTrue(np.allclose(matrix, exp_matrix))

        with self.assertRaises(ValueError):
            reach.transform([[]])

        empty_result = reach.transform([])
        self.assertEqual(empty_result, [])

    def test_mean_pool(self) -> None:
        words, vectors = self.data()
        reach = Reach(vectors, words)

        with self.assertRaises(ValueError):
            reach.mean_pool(["donatello", "dog"])

        vec = reach.mean_pool(["donatello", "dog"], safeguard=False)
        self.assertTrue(np.allclose(vec, np.zeros_like(vec)))

        vec = reach.mean_pool(["donatello", "dog"], remove_oov=True)
        self.assertTrue(np.allclose(vec, reach["donatello"]))

    def test_mean_pool_unk(self) -> None:
        words, vectors = self.data()
        words.append("<UNK>")
        vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
        reach = Reach(vectors, words, unk_index=len(words) - 1)

        vec = reach.mean_pool(["donatello", "dog"])
        self.assertTrue(np.allclose(vec, reach["donatello"] / 2))

        vec = reach.mean_pool(["donatello", "dog"], safeguard=True)
        self.assertTrue(np.allclose(vec, reach["donatello"] / 2))

        vec = reach.mean_pool(["donatello", "dog"], remove_oov=True)
        self.assertTrue(np.allclose(vec, reach["donatello"]))

        vec = reach.mean_pool([], safeguard=False)
        self.assertTrue(np.allclose(vec, np.zeros_like(vec)))

        with self.assertRaises(ValueError):
            reach.mean_pool_corpus([[], ["dog"], ["guogrwohu"]], safeguard=True)

        matrix = reach.mean_pool_corpus([[], ["dog"], ["guogrwohu"]], safeguard=False)
        self.assertTrue(np.allclose(matrix, np.zeros_like(matrix)))
