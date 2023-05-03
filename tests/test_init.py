import unittest
from typing import Hashable, List, Tuple

import numpy as np

from reach import Reach


class TestInit(unittest.TestCase):
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

        instance = Reach(vectors, words, unk_index=1)
        self.assertEqual(instance.unk_index, 1)
        self.assertEqual(list(instance.sorted_items), words)

        with self.assertRaises(AttributeError):
            instance.indices = [0, 1, 2]  # type: ignore

        with self.assertRaises(AttributeError):
            instance.items = {"dog": 1}  # type: ignore
