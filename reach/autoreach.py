import re
from string import punctuation

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
    """
    A Reach variant that does not require tokenization.

    It uses the aho-corasick algorithm to build an automaton, which is then
    used to find candidates in strings. These candidates are then selected
    using a "word rule" (see is_valid_token). This rule is now used to languages
    that delimit words using spaces. If this is not the case, please subclass
    this and write rules that fit your language of choice.

    Parameters
    ----------
    vectors : numpy array
        The vector space.
    items : list
        A list of items. Length must be equal to the number of vectors, and
        aligned with the vectors.
    lowercase : bool or str
        This determines whether the string should be lowercased or not before
        searching it. If this is set to 'auto', the items in the vector space
        are used to determine whether this attribute should be true or false.
    name : string, optional, default ''
        A string giving the name of the current reach. Only useful if you
        have multiple spaces and want to keep track of them.
    unk_index : int or None, optional, default None
        The index of the UNK item. If this is None, any attempts at vectorizing
        OOV items will throw an error.

    Attributes
    ----------
    unk_index : int
        The integer index of your unknown glyph. This glyph will be inserted
        into your BoW space whenever an unknown item is encountered.
    name : string
        The name of the Reach instance.

    """

    def __init__(
        self,
        vectors: Matrix,
        items: list[str],
        lowercase: str | bool = "auto",
        name: str = "",
        unk_index: int | None = None,
    ) -> None:
        """Initialize a Reach instance with an array and list of strings."""
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
        """Whether to lowercase a string before searching it."""
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

    def bow(self, tokens: Tokens, remove_oov: bool = True) -> list[int]:
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
