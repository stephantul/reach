from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from reach import Reach


def test_truncation(embedding_file: Path) -> None:
    instance = Reach.load(embedding_file, truncate_embeddings=2)
    assert instance.size == 2
    assert len(instance) == 6

    instance = Reach.load(embedding_file, truncate_embeddings=100)
    assert instance.size == 5
    assert len(instance) == 6


def test_wordlist(embedding_file: Path) -> None:
    instance = Reach.load(embedding_file, wordlist=("shredder", "krang"))
    assert len(instance) == 2

    with pytest.raises(ValueError):
        Reach.load(embedding_file, wordlist=("doggo",))


def test_duplicate(
    embedding_lines: Callable[..., list[str]],
    write_embedding_file: Callable[..., Path],
) -> None:
    lines = embedding_lines()
    lines[3] = lines[2]
    path = write_embedding_file(lines)

    with pytest.raises(ValueError):
        Reach.load(path, recover_from_errors=False)
    instance = Reach.load(path, recover_from_errors=True)
    assert len(instance) == 5


def test_unk(embedding_file: Path) -> None:
    instance = Reach.load(embedding_file, unk_word=None)
    assert instance.unk_index is None

    desired_dtype = "float32"
    instance = Reach.load(embedding_file, unk_word="[UNK]", desired_dtype=desired_dtype)
    assert instance.unk_index == 0
    assert instance.items["[UNK]"] == instance.unk_index
    assert instance.vectors.dtype == desired_dtype

    instance = Reach.load(embedding_file, unk_word="splinter")
    assert instance.unk_index == 2
    assert instance.items["splinter"] == instance.unk_index


def test_limit(embedding_file: Path) -> None:
    instance = Reach.load(embedding_file, num_to_load=2)
    assert len(instance) == 2

    with pytest.raises(ValueError):
        Reach.load(embedding_file, num_to_load=-1)

    instance = Reach.load(embedding_file, num_to_load=10000)
    assert len(instance) == 6


@pytest.mark.parametrize("header", [True, False])
def test_sep(
    embedding_lines: Callable[..., list[str]],
    write_embedding_file: Callable[..., Path],
    header: bool,
) -> None:
    path = write_embedding_file(embedding_lines(header=header, sep=","))
    Reach.load(path, sep=",")


@pytest.mark.parametrize(
    "header,corrupted_line,expected_shape",
    [(False, 0, (1, 4)), (False, 1, (5, 5)), (True, 1, (5, 5))],
)
def test_corrupted_file(
    embedding_lines: Callable[..., list[str]],
    write_embedding_file: Callable[..., Path],
    header: bool,
    corrupted_line: int,
    expected_shape: tuple[int, int],
) -> None:
    lines = embedding_lines(header=header)
    lines[corrupted_line] = " ".join(lines[corrupted_line].split(" ")[:-1])
    path = write_embedding_file(lines)

    with pytest.raises(ValueError):
        Reach.load(path)

    instance = Reach.load(path, recover_from_errors=True)
    assert instance.size == expected_shape[1]
    assert len(instance.items) == expected_shape[0]
    assert instance.vectors.shape == expected_shape


@pytest.mark.parametrize("header", [True, False])
def test_load_from_file(
    embedding_lines: Callable[..., list[str]],
    write_embedding_file: Callable[..., Path],
    header: bool,
) -> None:
    path = write_embedding_file(embedding_lines(header=header))

    instance = Reach.load(str(path))
    assert instance.size == 5
    assert len(instance.items) == 6
    assert instance.vectors.shape == (6, 5)

    for index, vector in enumerate(instance.vectors):
        assert np.all(vector == index)
    for item, index in instance.items.items():
        assert instance.indices[index] == item

    instance = Reach.load(str(path), num_to_load=3)
    assert instance.size == 5
    assert len(instance.items) == 3
    assert instance.vectors.shape == (3, 5)

    instance = Reach.load(str(path))
    with open(path) as f:
        instance_from_file = Reach.load(f)
    assert instance.size == instance_from_file.size
    assert np.all(instance.vectors == instance_from_file.vectors)
    assert instance.name == instance_from_file.name

    instance_from_path = Reach.load(path)
    assert instance.size == instance_from_path.size
    assert np.all(instance.vectors == instance_from_path.vectors)
    assert instance.name == instance_from_path.name

    with pytest.raises(ValueError):
        Reach.load(path, num_to_load=0)

    with pytest.raises(ValueError):
        Reach.load(path, num_to_load=-1)


def test_save_load_fast_format(embedding_file: Path, tmp_path: Path) -> None:
    instance = Reach.load(embedding_file)
    fast_path = tmp_path / "fast"
    instance.save_fast_format(fast_path)
    instance_2 = Reach.load_fast_format(fast_path)

    assert instance.size == instance_2.size
    assert len(instance) == len(instance_2)
    assert np.allclose(instance.vectors, instance_2.vectors)
    assert instance.unk_index == instance_2.unk_index
    assert instance.name == instance_2.name


def test_save_load(embedding_file: Path, tmp_path: Path) -> None:
    instance = Reach.load(embedding_file)
    save_path = tmp_path / "saved" / embedding_file.name
    save_path.parent.mkdir()
    instance.save(save_path)
    instance_2 = Reach.load(save_path)

    assert instance.size == instance_2.size
    assert len(instance) == len(instance_2)
    assert np.allclose(instance.vectors, instance_2.vectors)
    assert instance.unk_index == instance_2.unk_index
    assert instance.name == instance_2.name
