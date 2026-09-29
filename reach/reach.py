"""A class for working with vector representations."""

from __future__ import annotations

import json
import logging
from collections.abc import Hashable, Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, TypeAlias

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from typing_extensions import Self

Dtype: TypeAlias = str | np.dtype
PathLike: TypeAlias = str | Path
Matrix: TypeAlias = np.ndarray | list[np.ndarray]
SimilarityItem: TypeAlias = list[tuple[Hashable, float]]
SimilarityResult: TypeAlias = list[SimilarityItem]
Tokens: TypeAlias = Iterable[Hashable]


logger = logging.getLogger(__name__)


class Reach:
    """
    Work with vector representations of items.

    Supports functions for calculating fast batched similarity
    between items or composite representations of items.

    Parameters
    ----------
    vectors
        The vector space.
    items
        A list of items. Length must be equal to the number of vectors, and
        aligned with the vectors.
    name
        A string giving the name of the current reach. Only useful if you
        have multiple spaces and want to keep track of them.
    unk_index
        The index of the UNK item. If this is None, any attempts at vectorizing
        OOV items will throw an error.

    Attributes
    ----------
    unk_index : int
        The integer index of your unknown glyph. This glyph will be inserted
        into your BoW space whenever an unknown item is encountered.
    name : string
        The name of the Reach instance.

    Raises
    ------
    ValueError
        If there are no items, if the number of items and vectors differ, if
        items is a set or dict, if items contains duplicates, or if unk_index
        is out of range.

    """

    def __init__(
        self,
        vectors: Matrix,
        items: list[Hashable],
        name: str = "",
        unk_index: int | None = None,
    ) -> None:
        if len(items) == 0:
            raise ValueError("A Reach instance needs at least one item.")
        if len(items) != len(vectors):
            raise ValueError(
                "Your vector space and list of items are not the same length: "
                f"{len(vectors)} != {len(items)}"
            )
        if isinstance(items, (dict, set)):
            raise ValueError(
                "Your item list is a set or dict, and might not "
                "retain order in the conversion to internal look"
                "-ups. Please convert it to list and check the "
                "order."
            )

        self._items: dict[Hashable, int] = {w: idx for idx, w in enumerate(items)}
        if len(self._items) != len(items):
            raise ValueError("Your item list contains duplicate items.")
        if unk_index is not None and not 0 <= unk_index < len(items):
            raise ValueError(
                f"unk_index {unk_index} is out of range for {len(items)} items."
            )
        self._indices: dict[int, Hashable] = {idx: w for w, idx in self.items.items()}
        self.vectors = np.asarray(vectors)
        self.unk_index = unk_index
        self.name = name

    def __len__(self) -> int:
        """The number of the items in the vector space."""
        return len(self.items)

    def __contains__(self, item: Hashable) -> bool:
        """Whether an item is in the vector space."""
        return item in self.items

    @property
    def items(self) -> dict[Hashable, int]:
        """A mapping from item ids to their indices."""
        return self._items

    @property
    def indices(self) -> dict[int, Hashable]:
        """A mapping from integers to item indices."""
        return self._indices

    @property
    def sorted_items(self) -> Tokens:
        """The items, sorted by index."""
        items: Tokens = [
            item for item, _ in sorted(self.items.items(), key=lambda x: x[1])
        ]
        return items

    @property
    def size(self) -> int:
        """The dimensionality of the vectors"""
        return self.vectors.shape[1]

    @property
    def vectors(self) -> np.ndarray:
        """
        The vectors themselves.

        This is a read-only view. To change the vectors, assign a new array to
        this attribute, which also updates norm_vectors.
        """
        return self._vectors

    @vectors.setter
    def vectors(self, x: Matrix) -> None:
        x = np.asarray(x).view()
        if not np.ndim(x) == 2:
            raise ValueError(f"Your array does not have 2 dimensions: {np.ndim(x)}")
        if not x.shape[0] == len(self.items):
            raise ValueError(
                f"Your array does not have the correct length, got {x.shape[0]},"
                f" expected {len(self.items)}"
            )
        x.flags.writeable = False
        self._vectors = x
        # Make sure norm vectors is updated.
        if hasattr(self, "_norm_vectors"):
            self._norm_vectors = self._normalize_or_copy(x)

    @property
    def norm_vectors(self) -> np.ndarray:
        """
        Vectors, but normalized to unit length.

        This is a read-only array. When all vectors are unit length, this
        attribute _is_ vectors, so no extra memory is used.
        """
        if not hasattr(self, "_norm_vectors"):
            self._norm_vectors = self._normalize_or_copy(self.vectors)
        return self._norm_vectors

    @staticmethod
    def _normalize_or_copy(vectors: np.ndarray) -> np.ndarray:
        """
        Return vectors itself if all vectors are unit length.

        Otherwise, the vectors are normalized, and a new read-only array is returned.
        """
        norms = np.linalg.norm(vectors, axis=1)
        all_unit_length = np.allclose(norms[norms != 0], 1)
        if all_unit_length:
            return vectors
        normalized = Reach.normalize(vectors, norms)
        normalized.flags.writeable = False
        return normalized

    @classmethod
    def load(
        cls,
        vector_file: PathLike | TextIO,
        wordlist: tuple[str, ...] | None = None,
        num_to_load: int | None = None,
        truncate_embeddings: int | None = None,
        unk_word: str | None = None,
        sep: str = " ",
        recover_from_errors: bool = False,
        desired_dtype: Dtype = "float32",
    ) -> Self:
        r"""
        Read a file in word2vec .txt format.

        The load function will raise a ValueError when trying to load items
        which do not conform to line lengths.

        Parameters
        ----------
        vector_file
            The path to the vector file, or an opened vector file.
        wordlist
            A list of words you want loaded from the vector file. If this is
            None (default), all words will be loaded.
        num_to_load
            The number of items to load from the file. Because loading can take
            some time, it is sometimes useful to onlyl load the first n items
            from a vector file for quick inspection.
        truncate_embeddings
            If this value is not None, the vectors in the vector space will
            be truncated to the number of dimensions indicated by this value.
        unk_word
            The object to treat as UNK in your vector space. If this is not
            in your items dictionary after loading, we add it with a zero
            vector.
        sep
            The separator used between the item and the vector values.
        recover_from_errors
            If this flag is True, the model will continue after encountering
            duplicates or other errors.
        desired_dtype
            The dtype of the loaded vectors.

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        if isinstance(vector_file, str | Path):
            vector_file = Path(vector_file)
            name = vector_file.name
            file_handle: TextIO = open(vector_file, encoding="utf-8")
            came_from_path = True
        else:
            name = Path(getattr(vector_file, "name", "")).name
            file_handle = vector_file
            came_from_path = False

        try:
            vectors, items = Reach._load(
                file_handle,
                wordlist,
                num_to_load,
                truncate_embeddings,
                sep,
                recover_from_errors,
                desired_dtype,
            )
        finally:
            if came_from_path:
                file_handle.close()

        if unk_word is not None:
            if unk_word not in items:
                unk_vec = np.zeros((1, vectors.shape[1]), dtype=desired_dtype)
                vectors = np.concatenate([unk_vec, vectors], 0)
                items = [unk_word] + items
                unk_index = 0
            else:
                unk_index = items.index(unk_word)
        else:
            unk_index = None

        # NOTE: we use type: ignore because we pass a list of strings, which is hashable
        return cls(
            vectors,
            items,  # type: ignore
            name=name,
            unk_index=unk_index,
        )

    @staticmethod
    def _load(
        file_handle: TextIO,
        wordlist: tuple[str, ...] | None,
        num_to_load: int | None,
        truncate_embeddings: int | None,
        sep: str,
        recover_from_errors: bool,
        desired_dtype: Dtype,
    ) -> tuple[np.ndarray, list[str]]:
        """Load a matrix and wordlist from an opened .vec file."""
        vectors = []
        addedwords = set()
        words = []

        if num_to_load is not None and num_to_load <= 0:
            raise ValueError(f"num_to_load should be > 0, is now {num_to_load}")
        if truncate_embeddings is not None and truncate_embeddings < 0:
            raise ValueError(
                f"truncate_embeddings should be >= 0, is now {truncate_embeddings}"
            )

        if wordlist is None:
            wordset = set()
        else:
            wordset = set(wordlist)

        logger.info(f"Loading {getattr(file_handle, 'name', 'file handle')}")
        raw_firstline = file_handle.readline()
        firstline = raw_firstline.rstrip(" \n")
        try:
            num, size = map(int, firstline.split(sep))
            logger.info(f"Vector space: {num} by {size}")
            lines: Iterable[str] = file_handle
            start = 1
        except ValueError:
            size = len(firstline.split(sep)) - 1
            logger.info(f"Vector space: {size} dim, # items unknown")
            lines = chain([raw_firstline], file_handle)
            start = 0

        if truncate_embeddings is None or truncate_embeddings == 0:
            truncate_embeddings = size

        for idx, line in enumerate(lines, start=start):
            word, rest = line.rstrip(" \n").split(sep, 1)

            if wordset and word not in wordset:
                continue

            if word in addedwords:
                e = f"Duplicate: {word} on line {idx + 1} was in the vector space twice"
                if recover_from_errors:
                    logger.warning(e)
                    continue
                raise ValueError(e)

            line_size = len(rest.split(sep))
            if line_size != size:
                e = (
                    f"Incorrect input at index {idx + 1}, size is {line_size},"
                    f" expected {size}."
                )
                if recover_from_errors:
                    logger.warning(e)
                    continue
                raise ValueError(e)

            try:
                vector = np.fromstring(rest, sep=sep)
                parsed = len(vector) == size
                if not parsed:
                    raise ValueError()
                words.append(word)
                addedwords.add(word)
                vectors.append(vector[:truncate_embeddings])
            except ValueError:
                e = f"Could not parse the vector at index {idx + 1}."
                if recover_from_errors:
                    logger.warning(e)
                    continue
                raise ValueError(e) from None

            if num_to_load is not None and len(addedwords) >= num_to_load:
                break

        logger.info("Loading finished")
        if wordset:
            diff = wordset - addedwords
            if diff:
                logger.info(
                    "Not all items from your wordlist were in your "
                    f"vector space: {diff}."
                )
            if len(addedwords) == 0:
                raise ValueError(
                    "No words were found because of no overlap "
                    "between your wordlist and the vector vocabulary"
                )
        if len(addedwords) == 0:
            raise ValueError("No words found. Reason unknown")

        return np.array(vectors, dtype=desired_dtype), words

    def __getitem__(self, item: Hashable) -> np.ndarray:
        """Get the vector for a single item."""
        return self.vectors[self.items[item]]

    def vectorize(
        self,
        tokens: Tokens,
        remove_oov: bool = False,
        norm: bool = False,
    ) -> np.ndarray:
        """
        Vectorize a sentence by replacing all items with their vectors.

        Parameters
        ----------
        tokens
            The tokens to vectorize.
        remove_oov
            Whether to remove OOV items. If False, OOV items are replaced by
            the UNK glyph. If this is True, the returned sequence might
            have a different length than the original sequence.
        norm
            Whether to return the unit vectors, or the regular vectors.

        Returns
        -------
        s : numpy array
            An M * N matrix, where every item has been replaced by
            its vector. OOV items are either removed, or replaced
            by the value of the UNK glyph.

        Raises
        ------
        ValueError
            If tokens is empty, or if all tokens are removed as OOV.

        """
        token_list = tokens if isinstance(tokens, str) else list(tokens)
        if not token_list:
            raise ValueError("You supplied an empty list.")
        index = self.bow(token_list, remove_oov=remove_oov)
        if not index:
            raise ValueError(
                f"You supplied a list with only OOV tokens: {token_list}, "
                "which then got removed. Set remove_oov to False,"
                " or filter your sentences to remove any in which"
                " all items are OOV."
            )
        if norm:
            return self.norm_vectors[index]
        else:
            return self.vectors[index]

    def mean_pool(
        self, tokens: Tokens, remove_oov: bool = False, safeguard: bool = True
    ) -> np.ndarray:
        """
        Mean pool a list of tokens.

        Parameters
        ----------
        tokens
            The list of items to vectorize and then mean pool.
        remove_oov
            Whether to remove OOV items from the input.
            If this is False, and an unknown item is encountered, then
            the <UNK> symbol will be inserted if it is set. If it is not set,
            then the function will throw a ValueError.
        safeguard
            There are a variety of reasons why we can't vectorize a list of tokens:
                - The list might be empty after removing OOV
                - We remove OOV but haven't set <UNK>
                - The list of tokens is empty
            If safeguard is False, we simply supply a zero vector instead of erroring.

        Returns
        -------
        vector: np.ndarray
            a vector of the correct size, which is the mean of all tokens
            in the sentence.

        Raises
        ------
        ValueError
            If the tokens cannot be vectorized and safeguard is True.

        """
        try:
            return self.vectorize(tokens, remove_oov, False).mean(0)
        except ValueError as exc:
            if safeguard:
                raise exc
            return np.zeros(self.size, dtype=self.vectors.dtype)

    def mean_pool_corpus(
        self, corpus: list[Tokens], remove_oov: bool = False, safeguard: bool = True
    ) -> np.ndarray:
        """
        Mean pool a list of list of tokens.

        Parameters
        ----------
        corpus
            The list of items to vectorize and then mean pool.
        remove_oov
            Whether to remove OOV items from the input.
            If this is False, and an unknown item is encountered, then
            the <UNK> symbol will be inserted if it is set. If it is not set,
            then the function will throw a ValueError.
        safeguard
            There are a variety of reasons why we can't vectorize a list of tokens:
            - The list might be empty after removing OOV
            - We remove OOV but haven't set <UNK>
            - The list of tokens is empty
            If safeguard is False, we simply supply a zero vector instead of erroring.

        Returns
        -------
        vector: np.ndarray
            a matrix with number of rows n, where n is the number of input lists, and
            columns s, which is the number of columns of a single vector.

        Raises
        ------
        ValueError
            If any list of tokens cannot be vectorized and safeguard is True.

        """
        out = []
        for index, tokens in enumerate(corpus):
            try:
                out.append(self.mean_pool(tokens, remove_oov, safeguard))
            except ValueError as exc:
                raise ValueError(f"Tokens at {index} errored out") from exc

        if not out:
            return np.zeros((0, self.size), dtype=self.vectors.dtype)
        return np.stack(out)

    def bow(self, tokens: Tokens, remove_oov: bool = False) -> list[int]:
        """
        Create a bow representation of a list of tokens.

        Parameters
        ----------
        tokens
            The list of items to change into a bag of words representation.
        remove_oov
            Whether to remove OOV items from the input.
            If this is True, the length of the returned BOW representation
            might not be the length of the original representation.

        Returns
        -------
        bow : list
            A BOW representation of the list of items.

        Raises
        ------
        ValueError
            If tokens is a string, or if an OOV item is encountered while
            remove_oov is False and unk_index is None.

        """
        if isinstance(tokens, str):
            raise ValueError("You passed a string instead of a list of tokens.")

        out = []
        for t in tokens:
            try:
                out.append(self.items[t])
            except KeyError as exc:
                if remove_oov:
                    continue
                if self.unk_index is None:
                    raise ValueError(
                        "You supplied OOV items but didn't "
                        "provide the index of the replacement "
                        "glyph. Either set remove_oov to True, "
                        "or set unk_index to the index of the "
                        "item which replaces any OOV items."
                    ) from exc
                out.append(self.unk_index)

        return out

    def transform(
        self, corpus: list[Tokens], remove_oov: bool = False, norm: bool = False
    ) -> list[np.ndarray]:
        """
        Transform a corpus by repeated calls to vectorize, defined above.

        Parameters
        ----------
        corpus
            Represents a corpus as a list of sentences, where a sentence
            is a list of tokens.
        remove_oov
            If True, removes OOV items from the input before vectorization.
        norm
            If True, this will return normalized vectors.

        Returns
        -------
        c : list
            A list of numpy arrays, where each array represents the transformed
            sentence in the original list. The list is guaranteed to be the
            same length as the input list, but the arrays in the list may be
            of different lengths, depending on whether remove_oov is True.

        """
        return [self.vectorize(s, remove_oov=remove_oov, norm=norm) for s in corpus]

    def most_similar(
        self,
        items: Tokens,
        num: int = 10,
        batch_size: int = 100,
        show_progressbar: bool = False,
    ) -> SimilarityResult:
        """
        Return the num most similar items to a given list of items.

        Parameters
        ----------
        items
            The items to get the most similar items to.
        num
            The number of most similar items to retrieve.
        batch_size
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase the speed.
        show_progressbar
            Whether to show a progressbar.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, SIMILARITY) tuples.

        Raises
        ------
        ValueError
            If num is smaller than 1.

        """
        if num < 1:
            raise ValueError(f"num should be >= 1, is now {num}")
        items = [items] if isinstance(items, str) else list(items)
        vectors = self.norm_vectors[[self.items[item] for item in items]]
        result = self._most_similar_batch(
            vectors, batch_size, num + 1, show_progressbar
        )

        out: SimilarityResult = []
        # Remove queried item from similarity list
        for query_item, item_result in zip(items, result, strict=True):
            without_query = [
                (item, similarity)
                for item, similarity in item_result
                if item != query_item
            ]
            out.append(without_query[:num])
        return out

    def threshold(
        self,
        items: Tokens,
        threshold: float = 0.5,
        batch_size: int = 100,
        show_progressbar: bool = False,
    ) -> SimilarityResult:
        """
        Return all items whose similarity is higher than threshold.

        Parameters
        ----------
        items
            The items to get the most similar items to.
        threshold
            The radius within which to retrieve items.
        batch_size
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase the speed.
        show_progressbar
            Whether to show a progressbar.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, SIMILARITY) tuples.

        """
        items = [items] if isinstance(items, str) else list(items)
        vectors = self.norm_vectors[[self.items[item] for item in items]]
        result = self._threshold_batch(vectors, batch_size, threshold, show_progressbar)

        out: SimilarityResult = []
        # Remove queried item from similarity list
        for query_item, item_result in zip(items, result, strict=True):
            without_query = [
                (item, similarity)
                for item, similarity in item_result
                if item != query_item
            ]
            out.append(without_query)
        return out

    def nearest_neighbor(
        self,
        vectors: np.ndarray,
        num: int = 10,
        batch_size: int = 100,
        show_progressbar: bool = False,
    ) -> SimilarityResult:
        """
        Find the nearest neighbors to some arbitrary vector.

        This function is meant to be used in composition operations. The
        most_similar function can only handle items that are in vocab, and
        looks up their vector through a dictionary. Compositions, e.g.
        "King - man + woman" are necessarily not in the vocabulary.

        Parameters
        ----------
        vectors
            The vectors to find the nearest neighbors to.
        num
            The number of most similar items to retrieve.
        batch_size
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase speed.
        show_progressbar
            Whether to show a progressbar.

        Returns
        -------
        sim : list of tuples.
            For each item in the input the num most similar items are returned
            in the form of (NAME, SIMILARITY) tuples.

        """
        vectors = np.asarray(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        return list(
            self._most_similar_batch(vectors, batch_size, num, show_progressbar)
        )

    def nearest_neighbor_threshold(
        self,
        vectors: np.ndarray,
        threshold: float = 0.5,
        batch_size: int = 100,
        show_progressbar: bool = False,
    ) -> SimilarityResult:
        """
        Find the nearest neighbors to some arbitrary vector.

        This function is meant to be used in composition operations. The
        most_similar function can only handle items that are in vocab, and
        looks up their vector through a dictionary. Compositions, e.g.
        "King - man + woman" are necessarily not in the vocabulary.

        Parameters
        ----------
        vectors
            The vectors to find the nearest neighbors to.
        threshold
            The threshold within to retrieve items.
        batch_size
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase speed.
        show_progressbar
            Whether to show a progressbar.

        Returns
        -------
        sim : list of tuples.
            For each item in the input the num most similar items are returned
            in the form of (NAME, SIMILARITY) tuples.

        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        return list(
            self._threshold_batch(vectors, batch_size, threshold, show_progressbar)
        )

    def _threshold_batch(
        self,
        vectors: np.ndarray,
        batch_size: int,
        threshold: float,
        show_progressbar: bool,
    ) -> Iterator[SimilarityItem]:
        """Batched cosine similarity."""
        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):
            batch = vectors[i : i + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            for sims in similarities:
                indices = np.flatnonzero(sims >= threshold)
                sorted_indices = indices[np.flip(np.argsort(sims[indices]))]
                yield [(self.indices[d], float(sims[d])) for d in sorted_indices]

    def _most_similar_batch(
        self,
        vectors: np.ndarray,
        batch_size: int,
        num: int,
        show_progressbar: bool,
    ) -> Iterator[SimilarityItem]:
        """Batched cosine similarity."""
        if num < 1:
            raise ValueError(f"num should be >= 1, is now {num}")

        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):
            batch = vectors[i : i + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            if num == 1:
                sorted_indices = np.argmax(similarities, 1, keepdims=True)
            elif num >= len(self):
                # If we want more than we have, just sort everything.
                sorted_indices = np.stack([np.arange(len(self))] * len(batch))
            else:
                sorted_indices = np.argpartition(-similarities, kth=num, axis=1)
                sorted_indices = sorted_indices[:, :num]
            for lidx, indices in enumerate(sorted_indices):
                sims_for_word = similarities[lidx, indices]
                word_index = np.flip(np.argsort(sims_for_word))
                yield [
                    (self.indices[indices[idx]], float(sims_for_word[idx]))
                    for idx in word_index
                ]

    @staticmethod
    def normalize(vectors: np.ndarray, norms: np.ndarray | None = None) -> np.ndarray:
        """
        Normalize a matrix of row vectors to unit length.

        Contains a shortcut if there are no zero vectors in the matrix.
        If there are zero vectors, we do some indexing tricks to avoid
        dividing by 0.

        Parameters
        ----------
        vectors
            The vectors to normalize.
        norms
            Precomputed norms.

        Returns
        -------
        vectors : np.array
            The input vectors, normalized to unit length.

        """
        vectors = np.asarray(vectors)
        if not np.issubdtype(vectors.dtype, np.floating):
            vectors = vectors.astype(np.float64)

        if np.ndim(vectors) == 1:
            norm = np.linalg.norm(vectors)
            if norm == 0:
                return np.zeros_like(vectors)
            return vectors / norm

        if norms is None:
            norm = np.linalg.norm(vectors, axis=1)
        else:
            norm = norms

        if np.any(norm == 0):
            vectors = np.copy(vectors)
            nonzero = norm > 0
            result = np.zeros_like(vectors)
            n = norm[nonzero]  # type: ignore
            p = vectors[nonzero]
            result[nonzero] = p / n[:, None]

            return result
        else:
            return vectors / norm[:, None]  # type: ignore

    def vector_similarity(self, vector: np.ndarray, items: Tokens) -> np.ndarray:
        """Compute the similarity between a vector and a set of items."""
        items = [items] if isinstance(items, str) else list(items)
        items_vec = self.norm_vectors[[self.items[item] for item in items]]
        return self._sim(vector, items_vec)

    @classmethod
    def _sim(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cosine similarity function. This assumes y is normalized."""
        sim = cls.normalize(x).dot(y.T)
        return sim

    def similarity(self, items_1: Tokens, items_2: Tokens) -> np.ndarray:
        """
        Compute the similarity between two collections of items.

        Parameters
        ----------
        items_1
            The first collection of items.
        items_2
            The second collection of item.

        Returns
        -------
        sim : array of floats
            An array of similarity scores between 1 and -1.

        """
        items_1 = [items_1] if isinstance(items_1, str) else list(items_1)
        items_2 = [items_2] if isinstance(items_2, str) else list(items_2)

        items_1_matrix = self.norm_vectors[[self.items[item] for item in items_1]]
        items_2_matrix = self.norm_vectors[[self.items[item] for item in items_2]]
        return self._sim(items_1_matrix, items_2_matrix)

    def _new(
        self, vectors: np.ndarray, items: list[Hashable], unk_index: int | None
    ) -> Self:
        """Create a new instance of the same class, with the same settings."""
        return type(self)(vectors, items, name=self.name, unk_index=unk_index)

    def intersect(self, itemlist: Tokens) -> Self:
        """
        Intersect a reach instance with a list of items.

        Parameters
        ----------
        itemlist
            A list of items to keep. Note that this itemlist need not include
            all words in the Reach instance. Any words which are in the
            itemlist, but not in the reach instance, are ignored.

        Returns
        -------
        r : Reach
            A new Reach instance containing only the intersecting items.

        Raises
        ------
        ValueError
            If none of the items are in the Reach instance.

        """
        # Remove duplicates and oov words.
        itemlist = list(set(self.items) & set(itemlist))
        if not itemlist:
            raise ValueError("None of the items are in the Reach instance.")
        # Get indices of intersection.
        indices = sorted([self.items[item] for item in itemlist])
        unk_index: int | None = None
        if self.unk_index is not None and self.unk_index in indices:
            unk_index = indices.index(self.unk_index)
        vectors = self.vectors[indices]
        itemlist = [self.indices[index] for index in indices]
        return self._new(vectors, itemlist, unk_index)

    def union(self, other: Reach, check: bool = True) -> Self:
        """
        Union a reach with another reach.

        If items are in both reach instances, the current instance gets precedence.
        The items of the current instance come first, followed by the new items of
        the other instance, and the name and unk_index of the current instance are
        kept. If the current instance has no unk_index, the one of other is used.

        Parameters
        ----------
        other
            Another Reach instance.
        check
            Whether to check if duplicates are the same vector.

        Returns
        -------
        r : Reach
            A new Reach instance containing the items of both instances.

        Raises
        ------
        ValueError
            If the vector sizes differ, or if check is True and a shared item
            has different vectors.

        """
        if self.size != other.size:
            raise ValueError(
                f"The size of the embedding spaces was not the same: {self.size} and"
                f" {other.size}"
            )
        if check:
            for item in self.items.keys() & other.items.keys():
                if not np.allclose(self[item], other[item]):
                    raise ValueError(f"Term {item} was not the same in both instances")
        new_items = [item for item in other.sorted_items if item not in self]
        union = [*self.sorted_items, *new_items]
        vectors = np.concatenate(
            [self.vectors, other.vectors[[other.items[item] for item in new_items]]]
        )
        if self.unk_index is not None:
            unk_index: int | None = self.unk_index
        elif other.unk_index is not None:
            unk_index = union.index(other.indices[other.unk_index])
        else:
            unk_index = None

        return self._new(vectors, union, unk_index)

    def save(self, path: PathLike, write_header: bool = True) -> None:
        """
        Save the current vector space in word2vec format.

        Parameters
        ----------
        path
            The path to save the vector file to.
        write_header
            Whether to write a word2vec-style header as the first line of the
            file

        Raises
        ------
        ValueError
            If an item contains a space or newline, as it could not be loaded again.

        """
        for item in self.items:
            if any(char in str(item) for char in " \n"):
                raise ValueError(f"Item {item!r} contains a space or newline.")
        with open(path, "w", encoding="utf-8") as f:
            if write_header:
                f.write(f"{self.vectors.shape[0]} {self.vectors.shape[1]}\n")

            for i in range(len(self.items)):
                w = self.indices[i]
                vec = self.vectors[i]
                vec_string = " ".join([str(x) for x in vec])
                f.write(f"{w} {vec_string}\n")

    def save_fast_format(self, filename: PathLike) -> None:
        """
        Save a reach instance in a fast format.

        The reach fast format stores the words and vectors of a Reach instance
        separately in a JSON and numpy format, respectively.

        Parameters
        ----------
        filename
            The prefix to add to the saved filename. Note that this is not the
            real filename under which these items are stored.
            The words and unk_index are stored under "{filename}_items.json",
            and the numpy matrix is saved under "{filename}_vectors.npy".

        """
        items, _ = zip(*sorted(self.items.items(), key=lambda x: x[1]), strict=True)
        items_dict = {"items": items, "unk_index": self.unk_index, "name": self.name}

        with open(f"{filename}_items.json", "w", encoding="utf-8") as file_handle:
            json.dump(items_dict, file_handle)
        with open(f"{filename}_vectors.npy", "wb") as file_handle:
            np.save(file_handle, self.vectors)

    @classmethod
    def load_fast_format(
        cls, filename: PathLike, desired_dtype: Dtype = "float32"
    ) -> Self:
        """
        Load a reach instance in fast format.

        As described above, the fast format stores the words and vectors of the
        Reach instance separately, and is drastically faster than loading from
        .txt files.

        Parameters
        ----------
        filename
            The filename prefix from which to load. Note that this is not a
            real filepath as such, but a shared prefix for both files.
            In order for this to work, both {filename}_items.json and
            {filename}_vectors.npy should be present.
        desired_dtype
            The dtype of the loaded vectors.

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        with open(f"{filename}_items.json", encoding="utf-8") as file_handle:
            items = json.load(file_handle)
        words, unk_index, name = items["items"], items["unk_index"], items["name"]

        with open(f"{filename}_vectors.npy", "rb") as file_handle:
            vectors = np.load(file_handle)
        vectors = vectors.astype(desired_dtype)
        return cls(vectors, words, unk_index=unk_index, name=name)


def normalize(vectors: np.ndarray, norms: np.ndarray | None = None) -> np.ndarray:
    """Normalize an array to unit length."""
    return Reach.normalize(vectors, norms)
