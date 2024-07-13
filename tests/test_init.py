import unittest

import numpy as np

from reach import Reach


class TestInit(unittest.TestCase):
    def data(self) -> tuple[list[str], np.ndarray]:
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

    def test_init(self) -> None:
        words, vectors = self.data()
        instance = Reach(vectors, words)

        self.assertEqual(len(instance), 6)
        self.assertEqual(instance.size, 50)
        self.assertTrue(np.allclose(instance.vectors, vectors))

        sorted_words, _ = zip(*sorted(instance.items.items(), key=lambda x: x[1]))
        self.assertEqual(list(sorted_words), words)

        with self.assertRaises(ValueError):
            Reach(vectors[:5], words)

        with self.assertRaises(ValueError):
            Reach(vectors, words[:5])

        with self.assertRaises(ValueError):
            # Need to ignore type to trick mypy
            Reach(vectors, set(words))  # type: ignore

        instance_2 = Reach(vectors.tolist(), words)
        self.assertTrue(np.allclose(instance_2.vectors, instance.vectors))

        instance = Reach(vectors, words, name="sensei")
        self.assertEqual(instance.name, "sensei")

        instance = Reach(vectors, words)
        self.assertEqual(list(instance.sorted_items), words)

        with self.assertRaises(AttributeError):
            instance.indices = [0, 1, 2]  # type: ignore

        with self.assertRaises(AttributeError):
            instance.items = {"dog": 1}  # type: ignore

    def test_init_vectors_no_norm(self) -> None:
        words, vectors = self.data()
        r = Reach(vectors, words)

        self.assertFalse(hasattr(r, "_norm_vectors"))
        # Initialize norm vectors
        r.norm_vectors[0]
        self.assertTrue(hasattr(r, "norm_vectors"))
        self.assertFalse(r.vectors is r.norm_vectors)

    def test_init_vectors_norm(self) -> None:
        words, vectors = self.data()
        vectors = Reach.normalize(vectors)

        r = Reach(vectors, words)
        self.assertFalse(hasattr(r, "_norm_vectors"))
        # Initialize norm vectors
        r.norm_vectors[0]
        self.assertTrue(hasattr(r, "norm_vectors"))
        self.assertTrue(r.vectors is r.norm_vectors)

    def test_vectors_auto_norm_no_copy(self) -> None:
        _, vectors = self.data()
        result = Reach._normalize_or_copy(vectors)

        self.assertTrue(np.allclose(Reach.normalize(vectors), result))

    def test_vectors_auto_norm_copy(self) -> None:
        _, vectors = self.data()
        vectors = Reach.normalize(vectors)
        result = Reach._normalize_or_copy(vectors)

        self.assertTrue(vectors is result)

    def test_vectors_auto_norm(self) -> None:
        _, vectors = self.data()
        result = Reach._normalize_or_copy(vectors)

        self.assertTrue(np.allclose(Reach.normalize(vectors), result))
