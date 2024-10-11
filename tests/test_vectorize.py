import logging
import unittest

import numpy as np

from reach import Reach

logger = logging.getLogger(__name__)


class TestVectorize(unittest.TestCase):
    def data(self) -> tuple[list[str], np.ndarray]:
        """Data fixtures."""
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

    def test_vectorize_no_unk(self) -> None:
        """Test vectorize without an unk."""
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
        """Test vectorize with an unk."""
        words, vectors = self.data()
        words.append("<UNK>")
        vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
        reach = Reach(vectors, words)
        reach.unk_token = "<UNK>"

        self.assertIsNotNone(reach._unk_index)

        assert reach._unk_index is not None
        self.assertEqual(reach.indices[reach._unk_index], "<UNK>")
        self.assertTrue(np.allclose(vectors[-1], np.zeros(reach.size)))

    def test_bow_no_unk(self) -> None:
        """Test regular bow."""
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
        """Test bow with an unk."""
        words, vectors = self.data()
        words.append("<UNK>")
        vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
        reach = Reach(vectors, words)
        reach.unk_token = "<UNK>"

        bow = reach.bow(["donatello", "leonardo", "rgieurghegh"])
        self.assertEqual(bow, [0, 1, reach._unk_index])

        bow = reach.bow(["donatello", "leonardo", "rgieurghegh"], remove_oov=True)
        self.assertEqual(bow, [0, 1])

    def test_mean_pool(self) -> None:
        """Test regular mean pooling."""
        words, vectors = self.data()
        reach = Reach(vectors, words)

        with self.assertRaises(ValueError):
            reach.mean_pool(["donatello", "dog"])

        vec = reach.mean_pool(["donatello", "dog"], safeguard=False)
        self.assertTrue(np.allclose(vec, np.zeros_like(vec)))

        vec = reach.mean_pool(["donatello", "dog"], remove_oov=True)
        self.assertTrue(np.allclose(vec, reach["donatello"]))

    def test_mean_pool_unk(self) -> None:
        """Test mean pooling with an unk."""
        words, vectors = self.data()
        words.append("<UNK>")
        vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
        reach = Reach(vectors, words)
        reach.unk_token = "<UNK>"

        vec = reach.mean_pool(["donatello", "dog"])
        self.assertTrue(np.allclose(vec, reach["donatello"] / 2))

        vec = reach.mean_pool(["donatello", "dog"], safeguard=True)
        self.assertTrue(np.allclose(vec, reach["donatello"] / 2))

        vec = reach.mean_pool(["donatello", "dog"], remove_oov=True)
        self.assertTrue(np.allclose(vec, reach["donatello"]))

        vec = reach.mean_pool([], safeguard=False)
        self.assertTrue(np.allclose(vec, np.zeros_like(vec)))
