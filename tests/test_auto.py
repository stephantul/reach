from collections.abc import Hashable

import numpy as np
import pytest

pytest.importorskip("ahocorasick")

from reach import AutoReach, Reach  # noqa: E402


@pytest.fixture
def auto_instance(words: list[Hashable], vectors: np.ndarray) -> AutoReach:
    return AutoReach(vectors, words)


def test_load(
    words: list[Hashable], vectors: np.ndarray, auto_instance: AutoReach
) -> None:
    assert len(auto_instance.automaton) == len(words)

    normal_instance = Reach(vectors, words)

    assert auto_instance.items == normal_instance.items
    assert np.allclose(auto_instance.vectors, normal_instance.vectors)


@pytest.mark.parametrize(
    "token,text,index",
    [
        ("hideout", "the hideout was hidden", 10),
        ("hideout", "the hideout, was hidden", 10),
        ("hideout", "the ,hideout, was hidden", 11),
        # Punctuation tokens are always correct
        (",", "the ,hideouts", 4),
        (",", "the ,,,hideouts", 4),
        # Punctuation is allowed in tokens
        ("hide-out", "the hide-out was hidden", 11),
        ("etc.", "we like this and that,etc....", 25),
    ],
)
def test_valid(auto_instance: AutoReach, token: str, text: str, index: int) -> None:
    assert auto_instance.is_valid_token(token, text, index)


def test_invalid(auto_instance: AutoReach) -> None:
    assert not auto_instance.is_valid_token("hideout", "the hideouts was hidden", 10)


def test_lower(words: list[Hashable], vectors: np.ndarray) -> None:
    instance = AutoReach(vectors, words, lowercase=False)
    assert not instance.lowercase

    instance = AutoReach(vectors, words, lowercase=True)
    assert instance.lowercase

    instance = AutoReach(vectors, words, lowercase="auto")
    assert instance.lowercase

    words[0] = words[0].title()  # type: ignore
    instance = AutoReach(vectors, words, lowercase="auto")
    assert not instance.lowercase


def test_bow(auto_instance: AutoReach) -> None:
    result = auto_instance.bow(
        "leonardo, raphael, and the other turtles were in their hideout"
    )
    assert result == [1, 2, 5]


def test_vectorize(auto_instance: AutoReach) -> None:
    text = "leonardo, raphael, and the other turtles were in their hideout"
    vecs = auto_instance.vectors[auto_instance.bow(text)]
    vecs2 = auto_instance.vectorize(text)

    assert np.allclose(vecs, vecs2)


def test_bow_falls_back_to_shorter_match() -> None:
    instance = AutoReach(np.ones((2, 4)), ["new", "new york"])

    assert instance.bow("i like new yorker magazine") == [0]
    assert instance.bow("i like new york") == [1]


def test_intersect_keeps_type(auto_instance: AutoReach) -> None:
    assert type(auto_instance.intersect(["leonardo"])) is AutoReach


@pytest.mark.parametrize("lowercase", [True, False])
def test_intersect_union_keep_lowercase(
    words: list[Hashable], vectors: np.ndarray, lowercase: bool
) -> None:
    instance = AutoReach(vectors[:3], words[:3], lowercase=lowercase)
    other = AutoReach(vectors[2:], words[2:], lowercase=lowercase)

    assert instance.intersect(["leonardo"]).lowercase is lowercase
    assert instance.union(other).lowercase is lowercase


def test_autoreach_in_all() -> None:
    import reach

    assert "AutoReach" in reach.__all__
