import re
from string import punctuation
from typing import Optional, List, Union

try:
    from ahocorasick import Automaton
except ImportError:
    raise ImportError(
        "pyahocorasick is not installed. Please reinstall reach with `pip install"
        " reach[auto]`"
    )

from reach.reach import Reach, Matrix

PUNCT = set(punctuation)
SPACE = set("\n \t")
ALLOWED = PUNCT | SPACE
PUNCT_REGEX = re.compile(r"\W+")


class AutoReach(Reach):
    def __init__(
        self,
        vectors: Matrix,
        items: List[str],
        lowercase: Union[str, bool] = "auto",
        name: str = "",
        unk_index: Optional[int] = None,
    ) -> None:
        super().__init__(vectors, items, name, unk_index)
        self.automaton = Automaton()
        for item, index in self.items.items():
            self.automaton.add_word(item, (item, index))
        self.automaton.make_automaton()
        if lowercase == "auto":
            lowercase = all([x == x.lower() for x in self.items])
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

    def bow(self, tokens: Union[List[str], str], remove_oov: bool = True) -> List[int]:
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
