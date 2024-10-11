import json
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np

from reach import Reach


class TestLoad(unittest.TestCase):
    def lines(self, header: bool = True, n: int = 6, dim: int = 5, sep: str = " ") -> str:
        """Lines fixture."""
        lines = []
        words = ["skateboard", "pizza", "splinter", "technodrome", "krang", "shredder"]
        if header:
            lines.append(f"{n}{sep}{dim}")
        for idx, word in enumerate(words):
            lines.append(f"{word}{sep}{sep.join([str(idx)] * dim)}")
        return "\n".join(lines)

    def test_truncation(self) -> None:
        """Test truncation."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load_word2vec_format(tempfile.name, truncate_embeddings=2)
            self.assertEqual(instance.size, 2)
            self.assertEqual(len(instance), 6)

            instance = Reach.load_word2vec_format(tempfile.name, truncate_embeddings=100)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance), 6)

    def test_wordlist(self) -> None:
        """Test that adding a wordlist works."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load_word2vec_format(tempfile.name, wordlist=("shredder", "krang"))
            self.assertEqual(len(instance), 2)

            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name, wordlist=("doggo",))

    def test_duplicate(self) -> None:
        """Test duplicates in a wordlist."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            lines_split = lines.split("\n")
            lines_split[3] = lines_split[2]
            tempfile.write("\n".join(lines_split))
            tempfile.seek(0)

            with self.assertRaises(ValueError):
                Reach.load_word2vec_format(tempfile.name, recover_from_errors=False)
            instance = Reach.load_word2vec_format(tempfile.name, recover_from_errors=True)
            self.assertEqual(len(instance), 5)

    def test_unk(self) -> None:
        """Test whether the unk token exists."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load_word2vec_format(tempfile.name, unk_token=None)
            self.assertEqual(instance._unk_index, None)

            desired_dtype = "float32"
            instance = Reach.load_word2vec_format(tempfile.name, unk_token="[UNK]", desired_dtype=desired_dtype)
            self.assertEqual(instance._unk_index, 6)
            self.assertEqual(instance.items["[UNK]"], instance._unk_index)
            self.assertEqual(instance.vectors.dtype, desired_dtype)

            instance = Reach.load_word2vec_format(tempfile.name, unk_token="splinter")
            self.assertEqual(instance._unk_index, 2)
            self.assertEqual(instance.items["splinter"], instance._unk_index)

    def test_limit(self) -> None:
        """Tests whether limit during loading works."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)
            instance = Reach.load_word2vec_format(tempfile.name, num_to_load=2)
            self.assertEqual(len(instance), 2)

            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name, num_to_load=-1)

            instance = Reach.load_word2vec_format(tempfile.name, num_to_load=10000)
            self.assertEqual(len(instance), 6)

    def test_sep(self) -> None:
        """Test different seps."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(sep=",")
            tempfile.write(lines)
            tempfile.seek(0)
            Reach.load_word2vec_format(tempfile.name, sep=",")

        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(False, sep=",")
            tempfile.write(lines)
            tempfile.seek(0)
            Reach.load_word2vec_format(tempfile.name, sep=",")

    def test_corrupted_file(self) -> None:
        """Test whether a corrupted file loads."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(header=False)
            lines_split = lines.split("\n")
            lines_split[0] = " ".join(lines_split[0].split(" ")[:-1])
            tempfile.write("\n".join(lines_split))

            tempfile.seek(0)
            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name)

            instance = Reach.load_word2vec_format(tempfile.name, recover_from_errors=True)
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
                instance = Reach.load_word2vec_format(tempfile.name)

            instance = Reach.load_word2vec_format(tempfile.name, recover_from_errors=True)
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
                instance = Reach.load_word2vec_format(tempfile.name)

            instance = Reach.load_word2vec_format(tempfile.name, recover_from_errors=True)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 5)
            self.assertEqual(instance.vectors.shape, (5, 5))

    def test_load_from_file_without_header(self) -> None:
        """Test whether we can load files without headers."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines(header=False)
            tempfile.write(lines)
            tempfile.seek(0)

            instance = Reach.load_word2vec_format(tempfile.name)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 6)
            self.assertEqual(instance.vectors.shape, (6, 5))

            for index, vector in enumerate(instance.vectors):
                self.assertTrue(np.all(vector == index))
            for item, index in instance.items.items():
                self.assertEqual(instance.indices[index], item)

            instance = Reach.load_word2vec_format(tempfile.name, num_to_load=3)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 3)
            self.assertEqual(instance.vectors.shape, (3, 5))

            instance = Reach.load_word2vec_format(tempfile.name)
            with open(tempfile.name) as f:
                instance_from_file = Reach.load_word2vec_format(f)
            self.assertEqual(instance.size, instance_from_file.size)
            self.assertTrue(np.all(instance.vectors == instance_from_file.vectors))
            self.assertEqual(instance.name, instance_from_file.name)

            instance_from_path = Reach.load_word2vec_format(Path(tempfile.name))
            self.assertEqual(instance.size, instance_from_path.size)
            self.assertTrue(np.all(instance.vectors == instance_from_path.vectors))
            self.assertEqual(instance.name, instance_from_path.name)

            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name, num_to_load=0)

            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name, num_to_load=-1)

    def test_load_from_file_with_header(self) -> None:
        """Test whether we can load without headers."""
        with NamedTemporaryFile(mode="w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)

            instance = Reach.load_word2vec_format(tempfile.name)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 6)
            self.assertEqual(instance.vectors.shape, (6, 5))

            for index, vector in enumerate(instance.vectors):
                self.assertTrue(np.all(vector == index))
            for item, index in instance.items.items():
                self.assertEqual(instance.indices[index], item)

            instance = Reach.load_word2vec_format(tempfile.name, num_to_load=3)
            self.assertEqual(instance.size, 5)
            self.assertEqual(len(instance.items), 3)
            self.assertEqual(instance.vectors.shape, (3, 5))

            instance = Reach.load_word2vec_format(tempfile.name)
            with open(tempfile.name) as f:
                instance_from_file = Reach.load_word2vec_format(f)
            self.assertEqual(instance.size, instance_from_file.size)
            self.assertTrue(np.all(instance.vectors == instance_from_file.vectors))
            self.assertEqual(instance.name, instance_from_file.name)

            instance_from_path = Reach.load_word2vec_format(Path(tempfile.name))
            self.assertEqual(instance.size, instance_from_path.size)
            self.assertTrue(np.all(instance.vectors == instance_from_path.vectors))
            self.assertEqual(instance.name, instance_from_path.name)

            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name, num_to_load=0)

            with self.assertRaises(ValueError):
                instance = Reach.load_word2vec_format(tempfile.name, num_to_load=-1)

    def test_save_load_fast_format(self) -> None:
        """Test the saving and loading of the fast format."""
        with TemporaryDirectory() as temp_folder:
            lines = self.lines()

            temp_folder_path = Path(temp_folder)

            temp_file_name = temp_folder_path / "test.vec"
            with open(temp_file_name, "w") as tempfile:
                tempfile.write(lines)
                tempfile.seek(0)

            instance = Reach.load_word2vec_format(temp_file_name)
            fast_format_file = temp_folder_path / "temp.reach"
            instance.save(fast_format_file)
            instance_2 = Reach.load(fast_format_file)

            self.assertEqual(instance.size, instance_2.size)
            self.assertEqual(len(instance), len(instance_2))
            self.assertTrue(np.allclose(instance.vectors, instance_2.vectors))
            self.assertEqual(instance._unk_index, instance_2._unk_index)
            self.assertEqual(instance.name, instance_2.name)

    def test_save_load(self) -> None:
        """Test regular save and load."""
        with NamedTemporaryFile("w+") as tempfile:
            lines = self.lines()
            tempfile.write(lines)
            tempfile.seek(0)

            instance = Reach.load_word2vec_format(tempfile.name)
            # We know for sure that this writeable.
            instance.save_word2vec_format(tempfile.name)
            instance_2 = Reach.load_word2vec_format(tempfile.name)

            self.assertEqual(instance.size, instance_2.size)
            self.assertEqual(len(instance), len(instance_2))
            self.assertTrue(np.allclose(instance.vectors, instance_2.vectors))
            self.assertEqual(instance._unk_index, instance_2._unk_index)
            self.assertEqual(instance.name, instance_2.name)
