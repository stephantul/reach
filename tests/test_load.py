from pathlib import Path
import unittest

import numpy as np

from reach import Reach
from tempfile import NamedTemporaryFile


class TestLoad(unittest.TestCase):
    def lines(
        self, header: bool = True, n: int = 6, dim: int = 5, sep: str = " "
    ) -> str:
        lines = []
        words = ["skateboard", "pizza", "splinter", "technodrome", "krang", "shredder"]
        if header:
            lines.append(f"{n}{sep}{dim}")
        for idx, word in enumerate(words):
            lines.append(f"{word}{sep}{sep.join([str(idx)] * dim)}")
        return "\n".join(lines)

    def test_truncation(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load(tempfile.name, truncate_embeddings=2)
            self.assertEqual(instance.size, 2)
            self.assertEqual(len(instance), 6)

            instance = Reach.load(tempfile.name, truncate_embeddings=100)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance), 6)

    def test_wordlist(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load(tempfile.name, wordlist=("shredder", "krang"))
            self.assertEqual(len(instance), 2)

            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name, wordlist=("doggo",))

    def test_duplicate(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            lines_split = lines.split("\n")
            lines_split[3] = lines_split[2]
            tempfile.write("\n".join(lines_split))
            tempfile.seek(0)

            with self.assertRaises(ValueError):
                Reach.load(tempfile.name, recover_from_errors=False)
            instance = Reach.load(tempfile.name, recover_from_errors=True)
            self.assertEqual(len(instance), 5)

    def test_unk(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load(tempfile.name, unk_word=None)
            self.assertEqual(instance.unk_index, None)

            instance = Reach.load(tempfile.name, unk_word="[UNK]")
            self.assertEqual(instance.unk_index, 0)
            self.assertEqual(instance.items["[UNK]"], instance.unk_index)

            instance = Reach.load(tempfile.name, unk_word="splinter")
            self.assertEqual(instance.unk_index, 2)
            self.assertEqual(instance.items["splinter"], instance.unk_index)

    def test_limit(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load(tempfile.name, num_to_load=2)
            self.assertEqual(len(instance), 2)

            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name, num_to_load=-1)

            instance = Reach.load(tempfile.name, num_to_load=10000)
            self.assertEqual(len(instance), 6)

    def test_sep(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(sep=",")
            tempfile.write(lines)
            tempfile.seek(0)
            Reach.load(tempfile.name, sep=",")

        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(False, sep=",")
            tempfile.write(lines)
            tempfile.seek(0)
            Reach.load(tempfile.name, sep=",")

    def test_corrupted_file(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(header=False)
            lines_split = lines.split("\n")
            lines_split[0] = " ".join(lines_split[0].split(" ")[:-1])
            tempfile.write("\n".join(lines_split))

            tempfile.seek(0)
            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name)

            instance = Reach.load(tempfile.name, recover_from_errors=True)
            self.assertEqual(instance.size, 4)
            self.assertEqual(len(instance.items), 1)
            self.assertEqual(instance.vectors.shape, (1, 4))

        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(header=False)
            lines_split = lines.split("\n")
            lines_split[1] = " ".join(lines_split[1].split(" ")[:-1])
            tempfile.write("\n".join(lines_split))

            tempfile.seek(0)
            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name)

            instance = Reach.load(tempfile.name, recover_from_errors=True)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 5)
            self.assertEqual(instance.vectors.shape, (5, 5))

        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(header=True)
            lines_split = lines.split("\n")
            lines_split[1] = " ".join(lines_split[1].split(" ")[:-1])
            tempfile.write("\n".join(lines_split))

            tempfile.seek(0)
            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name)

            instance = Reach.load(tempfile.name, recover_from_errors=True)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 5)
            self.assertEqual(instance.vectors.shape, (5, 5))

    def test_load_from_file_without_header(self) -> None:
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(header=False)
            tempfile.write(lines)
            tempfile.seek(0)

            instance = Reach.load(tempfile.name)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 6)
            self.assertEqual(instance.vectors.shape, (6, 5))

            for index, vector in enumerate(instance.vectors):
                self.assertTrue(np.all(vector == index))
            for item, index in instance.items.items():
                self.assertEqual(instance.indices[index], item)

            instance = Reach.load(tempfile.name, num_to_load=3)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 3)
            self.assertEqual(instance.vectors.shape, (3, 5))

            instance = Reach.load(tempfile.name)
            with open(tempfile.name) as f:
                instance_from_file = Reach.load(f)
            self.assertEqual(instance.size, instance_from_file.size)
            self.assertTrue(np.all(instance.vectors == instance_from_file.vectors))
            self.assertEqual(instance.name, instance_from_file.name)

            instance_from_path = Reach.load(Path(tempfile.name))
            self.assertEqual(instance.size, instance_from_path.size)
            self.assertTrue(np.all(instance.vectors == instance_from_path.vectors))
            self.assertEqual(instance.name, instance_from_path.name)

            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name, num_to_load=0)

            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name, num_to_load=-1)

    def test_load_from_file_with_header(self) -> None:

        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)

            instance = Reach.load(tempfile.name)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 6)
            self.assertEqual(instance.vectors.shape, (6, 5))

            for index, vector in enumerate(instance.vectors):
                self.assertTrue(np.all(vector == index))
            for item, index in instance.items.items():
                self.assertEqual(instance.indices[index], item)

            instance = Reach.load(tempfile.name, num_to_load=3)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 3)
            self.assertEqual(instance.vectors.shape, (3, 5))

            instance = Reach.load(tempfile.name)
            with open(tempfile.name) as f:
                instance_from_file = Reach.load(f)
            self.assertEqual(instance.size, instance_from_file.size)
            self.assertTrue(np.all(instance.vectors == instance_from_file.vectors))
            self.assertEqual(instance.name, instance_from_file.name)

            instance_from_path = Reach.load(Path(tempfile.name))
            self.assertEqual(instance.size, instance_from_path.size)
            self.assertTrue(np.all(instance.vectors == instance_from_path.vectors))
            self.assertEqual(instance.name, instance_from_path.name)

            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name, num_to_load=0)

            with self.assertRaises(ValueError):
                instance = Reach.load(tempfile.name, num_to_load=-1)

    def test_load_fast_format(self) -> None:

        with NamedTemporaryFile("w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)

            instance = Reach.load(tempfile.name)
            instance.save_fast_format(tempfile.name)
            instance_2 = Reach.load_fast_format(tempfile.name)

            self.assertEqual(instance.size, instance_2.size)
            self.assertEqual(len(instance), len(instance_2))
            self.assertTrue(np.allclose(instance.vectors, instance_2.vectors))
            self.assertEqual(instance.unk_index, instance_2.unk_index)
            self.assertEqual(instance.name, instance_2.name)
