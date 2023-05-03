import unittest
from typing import Hashable, List, Tuple

import numpy as np

from reach import AutoReach, Reach


class TestAuto(unittest.TestCase):
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

    def test_load(self) -> None:
        words, vectors = self.data()
        instance = AutoReach(vectors, words)

        self.assertEqual(len(instance.automaton), len(words))

        normal_instance = Reach(vectors, words)

        self.assertEqual(instance.items, normal_instance.items)
        self.assertTrue(np.allclose(instance.vectors, normal_instance.vectors))

    def test_valid(self) -> None:
        words, vectors = self.data()
        instance = AutoReach(vectors, words)

        self.assertTrue(
            instance.is_valid_token("hideout", "the hideout was hidden", 10)
        )
        self.assertTrue(
            instance.is_valid_token("hideout", "the hideout, was hidden", 10)
        )
        self.assertTrue(
            instance.is_valid_token("hideout", "the ,hideout, was hidden", 11)
        )
        self.assertFalse(
            instance.is_valid_token("hideout", "the hideouts was hidden", 10)
        )

        # Punctuation tokens are always correct
        self.assertTrue(instance.is_valid_token(",", "the ,hideouts", 4))
        self.assertTrue(instance.is_valid_token(",", "the ,,,hideouts", 4))

        # Punctuation is allowed in tokens
        self.assertTrue(
            instance.is_valid_token("hide-out", "the hide-out was hidden", 11)
        )
        self.assertTrue(
            instance.is_valid_token("etc.", "we like this and that,etc....", 25)
        )

    def test_lower(self) -> None:
        words, vectors = self.data()
        instance = AutoReach(vectors, words, lowercase=False)
        self.assertFalse(instance.lowercase)

        instance = AutoReach(vectors, words, lowercase=True)
        self.assertTrue(instance.lowercase)

        instance = AutoReach(vectors, words, lowercase="auto")
        self.assertTrue(instance.lowercase)

        words[0] = words[0].title()  # type: ignore
        instance = AutoReach(vectors, words, lowercase="auto")
        self.assertFalse(instance.lowercase)

    def test_bow(self) -> None:
        words, vectors = self.data()
        instance = AutoReach(vectors, words)

        result = instance.bow(
            "leonardo, raphael, and the other turtles were in their hideout"
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result, [1, 2, 5])

    def test_vectorize(self) -> None:
        words, vectors = self.data()
        instance = AutoReach(vectors, words)

        result = instance.bow(
            "leonardo, raphael, and the other turtles were in their hideout"
        )

        vecs = instance.vectors[result]
        vecs2 = instance.vectorize(
            "leonardo, raphael, and the other turtles were in their hideout"
        )

        self.assertTrue(np.allclose(vecs, vecs2))
