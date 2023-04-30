import re
from string import punctuation
from typing import Hashable, List, Optional, Union

try:
    from ahocorasick import Automaton
except ImportError as exc:
    raise ImportError(
        "pyahocorasick is not installed. Please reinstall reach with `pip install"
        " reach[auto]`"
    ) from exc

from reach.reach import Matrix, Reach, Tokens

PUNCT = set(punctuation)
SPACE = set("\n \t")
ALLOWED = PUNCT | SPACE
PUNCT_REGEX = re.compile(r"\W+")


class AutoReach(Reach):
    def __init__(
        self,
        vectors: Matrix,
        items: List[Hashable],
        lowercase: Union[str, bool] = "auto",
        name: str = "",
        unk_index: Optional[int] = None,
    ) -> None:
        super().__init__(vectors, items, name, unk_index)
        self.automaton = Automaton()
        if not all(isinstance(item, str) for item in self.items):
            raise ValueError("All your items should be strings.")
        for item, index in self.items.items():
            self.automaton.add_word(item, (item, index))
        self.automaton.make_automaton()
        if lowercase == "auto":
            # NOTE: we use type ignore here because we know we have strings here.
            lowercase = all(
                [item == item.lower() for item in self.items]  # type: ignore
            )
        self._lowercase = bool(lowercase)

    @property
    def lowercase(self) -> bool:
        return self._lowercase

    def is_valid_token(self, token: str, tokens: str, end_index: int) -> bool:
        """Checks whether a token is valid in the current context."""
        if PUNCT_REGEX.fullmatch(token):
            return True

        if tokens[end_index + 1] not in ALLOWED:
            return False
        if tokens[end_index - (len(token))] not in ALLOWED:
            return False

        return True

    def bow(self, tokens: Tokens, remove_oov: bool = True) -> List[int]:
        """
        Create a bow representation from a string.

        Parameters
        ----------
        tokens : str.
            The string from which to extract in vocabulary tokens
        remove_oov : bool.
            Not used.

        Returns
        -------
        bow : list
            A BOW representation of the list of items.

        """
        if not isinstance(tokens, str):
            raise ValueError("You did not pass a string.")
        out = []
        tokens = f" {tokens} "
        if self.lowercase:
            tokens = tokens.lower()
        for end_index, (token, index) in self.automaton.iter_long(tokens):
            if self.is_valid_token(token, tokens, end_index):
                out.append(index)

        return out
