from pathlib import Path
from typing import Callable, Hashable, List

import numpy as np
import pytest

from reach import Reach


@pytest.fixture
def words() -> List[Hashable]:
    return [
        "donatello",
        "leonardo",
        "raphael",
        "michelangelo",
        "splinter",
        "hideout",
    ]


@pytest.fixture
def vectors() -> np.ndarray:
    random_generator = np.random.RandomState(seed=44)
    return random_generator.standard_normal((6, 50))


@pytest.fixture
def instance(words: List[Hashable], vectors: np.ndarray) -> Reach:
    return Reach(vectors, words)


@pytest.fixture
def unk_instance(words: List[Hashable], vectors: np.ndarray) -> Reach:
    words = [*words, "<UNK>"]
    vectors = np.concatenate([vectors, np.zeros((1, vectors.shape[1]))])
    return Reach(vectors, words, unk_index=len(words) - 1)


@pytest.fixture
def embedding_lines() -> Callable[..., List[str]]:
    def _lines(
        header: bool = True, n: int = 6, dim: int = 5, sep: str = " "
    ) -> List[str]:
        lines = []
        words = ["skateboard", "pizza", "splinter", "technodrome", "krang", "shredder"]
        if header:
            lines.append(f"{n}{sep}{dim}")
        for idx, word in enumerate(words):
            lines.append(f"{word}{sep}{sep.join([str(idx)] * dim)}")
        return lines

    return _lines


@pytest.fixture
def write_embedding_file(tmp_path: Path) -> Callable[..., Path]:
    def _write(lines: List[str]) -> Path:
        path = tmp_path / "embeddings.txt"
        path.write_text("\n".join(lines))
        return path

    return _write


@pytest.fixture
def embedding_file(
    write_embedding_file: Callable[..., Path],
    embedding_lines: Callable[..., List[str]],
) -> Path:
    return write_embedding_file(embedding_lines())
