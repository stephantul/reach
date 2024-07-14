"""A class for working with vector representations."""

from __future__ import annotations

import json
import logging
from io import TextIOWrapper, open
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeAlias

import numpy as np
from numpy import typing as npt
from tqdm import tqdm

from reach.legacy.load import load_old_fast_format_data


Dtype: TypeAlias = str | np.dtype
File = Path | TextIOWrapper
PathLike = str | Path
Matrix: TypeAlias = npt.NDArray | list[npt.NDArray]
SimilarityItem = list[tuple[str, float]]
SimilarityResult = list[SimilarityItem]
Tokens = Iterable[str]


logger = logging.getLogger(__name__)


class Reach:
    """
    Work with vector representations of items.

    Supports functions for calculating fast batched similarity
    between items or composite representations of items.

    Parameters
    ----------
    vectors : numpy array
        The vector space.
    items : list
        A list of items. Length must be equal to the number of vectors, and
        aligned with the vectors.
    name : string, optional, default ''
        A string giving the name of the current reach. Only useful if you
        have multiple spaces and want to keep track of them.

    Attributes
    ----------
    name : string
        The name of the Reach instance.

    """

    def __init__(
        self,
        vectors: Matrix,
        items: list[str],
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Reach instance with an array and list of items."""
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

        self._items: dict[str, int] = {w: idx for idx, w in enumerate(items)}
        self._indices: dict[int, str] = {idx: w for w, idx in self.items.items()}
        self.vectors = np.asarray(vectors)
        self.name = name
        self._unk_token: str | None = None
        self._unk_index: int | None = None
        self.metadata = metadata or {}

    @property
    def unk_token(self) -> str | None:
        """The unknown token."""
        return self._unk_token

    @unk_token.setter
    def unk_token(self, token: str | None) -> None:
        if token is None:
            if self.unk_token is not None:
                logger.info(f"Setting unk token from {self.unk_token} to None.")
            self._unk_token = None
            self._unk_index = None
        else:
            if token not in self.items:
                self.insert([token])
            self._unk_token = token
            self._unk_index = self.items[token]

    def __len__(self) -> int:
        """The number of the items in the vector space."""
        return len(self.items)

    @property
    def items(self) -> dict[str, int]:
        """A mapping from item ids to their indices."""
        return self._items

    @property
    def indices(self) -> dict[int, str]:
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
    def vectors(self) -> npt.NDArray:
        """The vectors themselves"""
        return self._vectors

    @vectors.setter
    def vectors(self, x: Matrix) -> None:
        matrix = np.asarray(x)
        if not np.ndim(matrix) == 2:
            raise ValueError(
                f"Your array does not have 2 dimensions: {np.ndim(matrix)}"
            )
        if not matrix.shape[0] == len(self.items):
            raise ValueError(
                f"Your array does not have the correct length, got {matrix.shape[0]},"
                f" expected {len(self.items)}"
            )
        self._vectors = matrix
        # Make sure norm vectors is updated.
        if hasattr(self, "_norm_vectors"):
            self._norm_vectors = self._normalize_or_copy(matrix)

    @property
    def norm_vectors(self) -> npt.NDArray:
        """
        Vectors, but normalized to unit length.

        NOTE: when all vectors are unit length, this attribute _is_ vectors.
        """
        if not hasattr(self, "_norm_vectors"):
            self._norm_vectors = self._normalize_or_copy(self.vectors)
        return self._norm_vectors

    @staticmethod
    def _normalize_or_copy(vectors: npt.NDArray) -> npt.NDArray:
        """
        This function returns a copy of vectors if they are unit length.
        Otherwise, the vectors are normalized, and a new array is returned.
        """
        norms = np.linalg.norm(vectors, axis=1)
        all_unit_length = np.allclose(norms[norms != 0], 1)
        if all_unit_length:
            return vectors
        return Reach.normalize(vectors, norms)

    def insert(self, tokens: list[str], vectors: npt.NDArray | None = None) -> None:
        """
        Insert new items into the vector space.

        Parameters
        ----------
        tokens : list
            A list of items to insert into the vector space.
        vectors : numpy array, optional, default None
            The vectors to insert into the vector space. If this is None,
            the vectors will be set to zero.

        """
        if vectors is None:
            vectors = np.zeros((len(tokens), self.size), dtype=self.vectors.dtype)
        else:
            vectors = np.asarray(vectors, dtype=self.vectors.dtype)

        if len(tokens) != len(vectors):
            raise ValueError(
                f"Your tokens and vectors are not the same length: {len(tokens)} != {len(vectors)}"
            )

        for token in tokens:
            if token in self.items:
                raise ValueError(f"Token {token} is already in the vector space.")
            self.items[token] = len(self.items)
            self.indices[len(self.items) - 1] = token
        self.vectors = np.concatenate([self.vectors, vectors], 0)

    @classmethod
    def load(
        cls,
        vector_file: File | str,
        wordlist: tuple[str, ...] | None = None,
        num_to_load: int | None = None,
        truncate_embeddings: int | None = None,
        unk_token: str | None = None,
        sep: str = " ",
        recover_from_errors: bool = False,
        desired_dtype: Dtype = "float32",
        **kwargs: Any,
    ) -> Reach:
        r"""
        Read a file in word2vec .txt format.

        The load function will raise a ValueError when trying to load items
        which do not conform to line lengths.

        Parameters
        ----------
        vector_file : string, Path or file handle
            The path to the vector file, or an opened vector file.
        header : bool
            Whether the vector file has a header of the type
            (NUMBER OF ITEMS, SIZE OF VECTOR).
        wordlist : iterable, optional, default ()
            A list of words you want loaded from the vector file. If this is
            None (default), all words will be loaded.
        num_to_load : int, optional, default None
            The number of items to load from the file. Because loading can take
            some time, it is sometimes useful to onlyl load the first n items
            from a vector file for quick inspection.
        truncate_embeddings : int, optional, default None
            If this value is not None, the vectors in the vector space will
            be truncated to the number of dimensions indicated by this value.
        unk_token : str
            The string to treat as UNK in your vector space. If this is not
            in your items dictionary after loading, we add it with a zero
            vector.
        recover_from_errors : bool
            If this flag is True, the model will continue after encountering
            duplicates or other errors.

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        if isinstance(vector_file, TextIOWrapper):
            name = Path(vector_file.name).name
            file_handle = vector_file
            came_from_path = False
        else:
            if isinstance(vector_file, str):
                vector_file = Path(vector_file)
            name = vector_file.name
            file_handle = open(vector_file)
            came_from_path = True

        try:
            vectors, items = Reach._load(
                file_handle,
                wordlist,
                num_to_load,
                truncate_embeddings,
                sep,
                recover_from_errors,
                desired_dtype,
                **kwargs,
            )
        except ValueError as exc:
            raise exc
        finally:
            if came_from_path:
                file_handle.close()

        # NOTE: we use type: ignore because we pass a list of strings, which is hashable
        instance = cls(
            vectors,
            items,
            name=name,
        )

        if unk_token is not None:
            if unk_token not in items:
                logger.info(f"Adding unk token {unk_token} to the vocabulary.")
                instance.insert([unk_token])
            instance.unk_token = unk_token

        return instance

    @staticmethod
    def _load(
        file_handle: TextIOWrapper,
        wordlist: tuple[str, ...] | None,
        num_to_load: int | None,
        truncate_embeddings: int | None,
        sep: str,
        recover_from_errors: bool,
        desired_dtype: Dtype,
    ) -> tuple[npt.NDArray, list[str]]:
        """Load a matrix and wordlist from an opened .vec file."""
        vectors = []
        addedwords = set()
        words = []

        if num_to_load is not None and num_to_load <= 0:
            raise ValueError(f"num_to_load should be > 0, is now {num_to_load}")

        if wordlist is None:
            wordset = set()
        else:
            wordset = set(wordlist)

        logger.info(f"Loading {file_handle.name}")
        firstline = file_handle.readline().rstrip(" \n")
        try:
            num, size = map(int, firstline.split(sep))
            logger.info(f"Vector space: {num} by {size}")
            header = True
        except ValueError:
            size = len(firstline.split(sep)) - 1
            logger.info(f"Vector space: {size} dim, # items unknown")
            # If the first line is correctly parseable, set header to False.
            header = False
        file_handle.seek(0)

        if truncate_embeddings is None or truncate_embeddings == 0:
            truncate_embeddings = size

        for idx, line in enumerate(file_handle):
            if header and idx == 0:
                continue

            word, rest = line.rstrip(" \n").split(sep, 1)

            if wordset and word not in wordset:
                continue

            if word in addedwords:
                e = f"Duplicate: {word} on line {idx+1} was in the vector space twice"
                if recover_from_errors:
                    logger.warning(e)
                    continue
                raise ValueError(e)

            if len(rest.split(sep)) != size:
                e = (
                    f"Incorrect input at index {idx+1}, size is {len(rest.split())},"
                    f" expected {size}."
                )
                if recover_from_errors:
                    logger.warning(e)
                    continue
                raise ValueError(e)

            words.append(word)
            addedwords.add(word)
            vectors.append(np.fromstring(rest, sep=sep)[:truncate_embeddings])

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

    def __getitem__(self, item: str) -> npt.NDArray:
        """Get the vector for a single item."""
        return self.vectors[self.items[item]]

    def vectorize(
        self,
        tokens: Tokens,
        remove_oov: bool = False,
        norm: bool = False,
    ) -> npt.NDArray:
        """
        Vectorize a sentence by replacing all items with their vectors.

        Parameters
        ----------
        tokens : object or list of objects
            The tokens to vectorize.
        remove_oov : bool, optional, default False
            Whether to remove OOV items. If False, OOV items are replaced by
            the UNK glyph. If this is True, the returned sequence might
            have a different length than the original sequence.
        norm : bool, optional, default False
            Whether to return the unit vectors, or the regular vectors.

        Returns
        -------
        s : numpy array
            An M * N matrix, where every item has been replaced by
            its vector. OOV items are either removed, or replaced
            by the value of the UNK glyph.

        """
        if not tokens:
            raise ValueError("You supplied an empty list.")
        index = self.bow(tokens, remove_oov=remove_oov)
        if not index:
            raise ValueError(
                f"You supplied a list with only OOV tokens: {tokens}, "
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
    ) -> npt.NDArray:
        """
        Mean pool a list of tokens.

        Parameters
        ----------
        tokens : list.
            The list of items to vectorize and then mean pool.
        remove_oov : bool.
            Whether to remove OOV items from the input.
            If this is False, and an unknown item is encountered, then
            the <UNK> symbol will be inserted if it is set. If it is not set,
            then the function will throw a ValueError.
        safeguard : bool.
            There are a variety of reasons why we can't vectorize a list of tokens:
                - The list might be empty after removing OOV
                - We remove OOV but haven't set <UNK>
                - The list of tokens is empty
            If safeguard is False, we simply supply a zero vector instead of erroring.

        Returns
        -------
        vector: npt.NDArray
            a vector of the correct size, which is the mean of all tokens
            in the sentence.

        """
        try:
            return self.vectorize(tokens, remove_oov, False).mean(0)
        except ValueError as exc:
            if safeguard:
                raise exc
            return np.zeros(self.size)

    def mean_pool_corpus(
        self, corpus: list[Tokens], remove_oov: bool = False, safeguard: bool = True
    ) -> npt.NDArray:
        """
        Mean pool a list of list of tokens.

        Parameters
        ----------
        corpus : a list of list of tokens.
            The list of items to vectorize and then mean pool.
        remove_oov : bool.
            Whether to remove OOV items from the input.
            If this is False, and an unknown item is encountered, then
            the <UNK> symbol will be inserted if it is set. If it is not set,
            then the function will throw a ValueError.
        safeguard : bool.
            There are a variety of reasons why we can't vectorize a list of tokens:
            - The list might be empty after removing OOV
            - We remove OOV but haven't set <UNK>
            - The list of tokens is empty
            If safeguard is False, we simply supply a zero vector instead of erroring.

        Returns
        -------
        vector: npt.NDArray
            a matrix with number of rows n, where n is the number of input lists, and
            columns s, which is the number of columns of a single vector.

        """
        out = []
        for index, tokens in enumerate(corpus):
            try:
                out.append(self.mean_pool(tokens, remove_oov, safeguard))
            except ValueError as exc:
                raise ValueError(f"Tokens at {index} errored out") from exc

        return np.stack(out)

    def bow(self, tokens: Tokens, remove_oov: bool = False) -> list[int]:
        """
        Create a bow representation of a list of tokens.

        Parameters
        ----------
        tokens : list.
            The list of items to change into a bag of words representation.
        remove_oov : bool.
            Whether to remove OOV items from the input.
            If this is True, the length of the returned BOW representation
            might not be the length of the original representation.

        Returns
        -------
        bow : list
            A BOW representation of the list of items.

        """
        if isinstance(tokens, str):
            raise ValueError("You passed a string instead of a list of tokens.")

        out = []
        for token in tokens:
            try:
                out.append(self.items[token])
            except KeyError as exc:
                if remove_oov:
                    continue
                if self._unk_index is None:
                    raise ValueError(
                        "You supplied OOV items but didn't "
                        "provide the index of the replacement "
                        "glyph. Either set remove_oov to True, "
                        "or set unk_index to the index of the "
                        "item which replaces any OOV items."
                    ) from exc
                out.append(self._unk_index)

        return out

    def transform(
        self, corpus: list[Tokens], remove_oov: bool = False, norm: bool = False
    ) -> list[npt.NDArray]:
        """
        Transform a corpus by repeated calls to vectorize, defined above.

        Parameters
        ----------
        corpus : A list of list of strings.
            Represents a corpus as a list of sentences, where a sentence
            is a list of tokens.
        remove_oov : bool, optional, default False
            If True, removes OOV items from the input before vectorization.
        norm : bool, optional, default False
            If True, this will return normalized vectors.

        Returns
        -------
        c : list
            A list of numpy arrays, where each array represents the transformed
            sentence in the original list. The list is guaranteed to be the
            same length as the input list, but the arrays in the list may be
            of different lengths, depending on whether remove_oov is True.

        """
        return [
            self.vectorize(string, remove_oov=remove_oov, norm=norm)
            for string in corpus
        ]

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
        items : list of objects or a single object.
            The items to get the most similar items to.
        num : int, optional, default 10
            The number of most similar items to retrieve.
        batch_size : int, optional, default 100.
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase the speed.
        show_progressbar : bool, optional, default False
            Whether to show a progressbar.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, SIMILARITY) tuples.

        """
        if isinstance(items, str):
            items = [items]
        vectors = np.stack([self.norm_vectors[self.items[item]] for item in items])
        result = self._most_similar_batch(
            vectors, batch_size, num + 1, show_progressbar
        )

        out: SimilarityResult = []
        # Remove queried item from similarity list
        for query_item, item_result in zip(items, result):
            without_query = [
                (item, similarity)
                for item, similarity in item_result
                if item != query_item
            ]
            out.append(without_query)
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
        items : list of objects or a single object.
            The items to get the most similar items to.
        threshold : float, optional, default .5
            The radius within which to retrieve items.
        batch_size : int, optional, default 100.
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase the speed.
        show_progressbar : bool, optional, default False
            Whether to show a progressbar.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, SIMILARITY) tuples.

        """
        if isinstance(items, str):
            items = [items]

        vectors = np.stack([self.norm_vectors[self.items[item]] for item in items])
        result = self._threshold_batch(vectors, batch_size, threshold, show_progressbar)

        out: SimilarityResult = []
        # Remove queried item from similarity list
        for query_item, item_result in zip(items, result):
            without_query = [
                (item, similarity)
                for item, similarity in item_result
                if item != query_item
            ]
            out.append(without_query)
        return out

    def nearest_neighbor(
        self,
        vectors: npt.NDArray,
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
        vectors : list of arrays or numpy array
            The vectors to find the nearest neighbors to.
        num : int, optional, default 10
            The number of most similar items to retrieve.
        batch_size : int, optional, default 100.
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase speed.
        show_progressbar : bool, optional, default False
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
        vectors: npt.NDArray,
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
        vectors : list of arrays or numpy array
            The vectors to find the nearest neighbors to.
        threshold : float, optional, default .5
            The threshold within to retrieve items.
        batch_size : int, optional, default 100.
            The batch size to use. 100 is a good default option. Increasing
            the batch size may increase speed.
        show_progressbar : bool, optional, default False
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
        vectors: npt.NDArray,
        batch_size: int,
        threshold: float,
        show_progressbar: bool,
    ) -> Iterator[SimilarityItem]:
        """Batched cosine similarity."""
        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):
            batch = vectors[i : i + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            for _, sims in enumerate(similarities):
                indices = np.flatnonzero(sims >= threshold)
                sorted_indices = indices[np.flip(np.argsort(sims[indices]))]
                yield [(self.indices[d], sims[d]) for d in sorted_indices]

    def _most_similar_batch(
        self,
        vectors: npt.NDArray,
        batch_size: int,
        num: int,
        show_progressbar: bool,
    ) -> Iterator[SimilarityItem]:
        """Batched cosine similarity."""
        if num < 1:
            raise ValueError("num should be >= 1, is now {num}")

        for index in tqdm(
            range(0, len(vectors), batch_size), disable=not show_progressbar
        ):
            batch = vectors[index : index + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            if num == 1:
                sorted_indices = np.argmax(similarities, 1, keepdims=True)
            elif num >= len(self):
                # If we want more than we have, just sort everything.
                sorted_indices = np.stack([np.arange(len(self))] * len(vectors))
            else:
                sorted_indices = np.argpartition(-similarities, kth=num, axis=1)
                sorted_indices = sorted_indices[:, :num]
            for lidx, indices in enumerate(sorted_indices):
                sims_for_word = similarities[lidx, indices]
                word_index = np.flip(np.argsort(sims_for_word))
                yield [
                    (self.indices[indices[idx]], sims_for_word[idx])
                    for idx in word_index
                ]

    @staticmethod
    def normalize(
        vectors: npt.NDArray, norms: npt.NDArray | None = None
    ) -> npt.NDArray:
        """
        Normalize a matrix of row vectors to unit length.

        Contains a shortcut if there are no zero vectors in the matrix.
        If there are zero vectors, we do some indexing tricks to avoid
        dividing by 0.

        Parameters
        ----------
        vectors : np.array
            The vectors to normalize.
        norms: npt.NDArray
            Precomputed norms.

        Returns
        -------
        vectors : np.array
            The input vectors, normalized to unit length.

        """
        if np.ndim(vectors) == 1:
            norm_float = np.linalg.norm(vectors)
            if np.isclose(norm_float, 0):
                return np.zeros_like(vectors)
            return vectors / norm_float

        if norms is None:
            norm: npt.NDArray = np.linalg.norm(vectors, axis=1)
        else:
            norm = norms

        if np.any(np.isclose(norm, 0.0)):
            vectors = np.copy(vectors)
            nonzero = norm > 0.0
            result = np.zeros_like(vectors)
            masked_norm = norm[nonzero]
            masked_vectors = vectors[nonzero]
            result[nonzero] = masked_vectors / masked_norm[:, None]

            return result
        else:
            return vectors / norm[:, None]

    def vector_similarity(self, vector: npt.NDArray, items: Tokens) -> npt.NDArray:
        """Compute the similarity between a vector and a set of items."""
        if isinstance(items, str):
            items = [items]

        items_vec = np.stack([self.norm_vectors[self.items[item]] for item in items])
        return self._sim(vector, items_vec)

    @classmethod
    def _sim(cls, x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        """Cosine similarity function. This assumes y is normalized."""
        sim = cls.normalize(x).dot(y.T)
        return sim

    def similarity(self, items_1: Tokens, items_2: Tokens) -> npt.NDArray:
        """
        Compute the similarity between two collections of items.

        Parameters
        ----------
        items_1 : iterable of items
            The first collection of items.
        items_2 : iterable of items
            The second collection of item.

        Returns
        -------
        sim : array of floats
            An array of similarity scores between 1 and -1.

        """
        if isinstance(items_1, str):
            items_1 = [items_1]
        if isinstance(items_2, str):
            items_2 = [items_2]

        items_1_matrix = np.stack(
            [self.norm_vectors[self.items[item]] for item in items_1]
        )
        items_2_matrix = np.stack(
            [self.norm_vectors[self.items[item]] for item in items_2]
        )
        return self._sim(items_1_matrix, items_2_matrix)

    def intersect(self, itemlist: Tokens) -> Reach:
        """
        Intersect a reach instance with a list of items.

        Parameters
        ----------
        itemlist : list of hashables
            A list of items to keep. Note that this itemlist need not include
            all words in the Reach instance. Any words which are in the
            itemlist, but not in the reach instance, are ignored.

        """
        # Remove duplicates and oov words.
        itemlist = list(set(self.items) & set(itemlist))
        # Get indices of intersection.
        indices = sorted([self.items[item] for item in itemlist])
        # Index vectors
        vectors = self.vectors[indices]
        # Index words
        itemlist = [self.indices[index] for index in indices]
        instance = Reach(vectors, itemlist, name=self.name)
        instance.unk_token = self.unk_token

        return instance

    def union(self, other: Reach, check: bool = True) -> Reach:
        """
        Union a reach with another reach.
        If items are in both reach instances, the current instance gets precedence.

        Parameters
        ----------
        other : Reach
            Another Reach instance.
        check : bool
            Whether to check if duplicates are the same vector.

        """
        if self.size != other.size:
            raise ValueError(
                f"The size of the embedding spaces was not the same: {self.size} and"
                f" {other.size}"
            )
        union = list(set(self.items) | set(other.items))
        if check:
            intersection = set(self.items) & set(other.items)
            for item in intersection:
                if not np.allclose(self[item], other[item]):
                    raise ValueError(f"Term {item} was not the same in both instances")
        vectors = []
        for item in union:
            try:
                vectors.append(self[item])
            except KeyError:
                vectors.append(other[item])

        return Reach(np.stack(vectors), union, name=self.name)

    def save(self, path: str, write_header: bool = True) -> None:
        """
        Save the current vector space in word2vec format.

        Parameters
        ----------
        path : str
            The path to save the vector file to.
        write_header : bool, optional, default True
            Whether to write a word2vec-style header as the first line of the
            file

        """
        with open(path, "w") as f:
            if write_header:
                f.write(f"{self.vectors.shape[0]} {self.vectors.shape[1]}\n")

            for index in range(len(self.items)):
                w = self.indices[index]
                vec = self.vectors[index]
                vec_string = " ".join([str(x) for x in vec])
                f.write(f"{w} {vec_string}\n")

    def save_fast_format(
        self,
        path: PathLike,
        overwrite: bool = False,
    ) -> None:
        """
        Save a reach instance in a fast format.

        The reach fast format stores the words and vectors of a Reach instance
        separately in a JSON and numpy format, respectively.

        Parameters
        ----------
        filename : str
            The path to which to save the JSON file. The vectors are saved separately.
            The JSON contains a path to the numpy file.
        overwrite : bool, optional, default False
            Whether to overwrite the JSON and numpy files if they already exist.

        """
        path_object = Path(path)
        if path_object.exists() and not overwrite:
            raise ValueError(
                f"The file at {path} already exists. Set overwrite to True, or choose another path."
            )
        if path_object.is_dir():
            raise ValueError(f"Path {path} is a directory. Please provide a filename.")
        numpy_path = path_object.with_suffix(".npy")

        if numpy_path.exists() and not overwrite:
            raise ValueError(
                f"The file at {numpy_path} already exists. Set overwrite to True, or choose another path."
            )

        metadata = {
            "unk_token": self.unk_token,
            "name": self.name,
        }
        metadata.update(self.metadata)

        items = self.sorted_items
        items_dict = {
            "items": items,
            "metadata": metadata,
            "vectors_path": numpy_path.name,
        }

        with open(path_object, "w") as file_handle:
            json.dump(items_dict, file_handle)
        with open(numpy_path, "wb") as file_handle:
            np.save(file_handle, self.vectors)

    @classmethod
    def load_fast_format(
        cls, filename: PathLike, desired_dtype: Dtype = "float32"
    ) -> Reach:
        """
        Load a reach instance in fast format.

        As described above, the fast format stores the words and vectors of the
        Reach instance separately, and is drastically faster than loading from
        .txt files.

        Parameters
        ----------
        filename : str
            The filename from which to load. In order for this to work, both the base file and
            file.npy should be present.

        """
        filename_path = Path(filename)

        try:
            with open(filename) as file_handle:
                data: dict[str, Any] = json.load(file_handle)
            items: list[str] = data["items"]

            metadata: dict[str, Any] = data["metadata"]
            unk_token = metadata.pop("unk_token")
            name = metadata.pop("name")
            numpy_path = filename_path.parent / Path(data["vectors_path"])

            if not numpy_path.exists():
                raise ValueError(f"Could not find the vectors file at {numpy_path}")

            with open(numpy_path, "rb") as file_handle:
                vectors: npt.NDArray = np.load(file_handle)
        except FileNotFoundError as exc:
            logger.warning("Attempting to load from old format.")
            try:
                vectors, items, unk_token, name = load_old_fast_format_data(
                    filename_path
                )
                metadata = {}
            except FileNotFoundError:
                logger.warning("Loading from old format failed")
                # NOTE: reraise old exception.
                raise exc

        vectors = vectors.astype(desired_dtype)
        instance = cls(vectors, items, name=name, metadata=metadata)
        instance.unk_token = unk_token

        return instance


def normalize(vectors: npt.NDArray, norms: npt.NDArray | None = None) -> npt.NDArray:
    """Normalize an array to unit length."""
    return Reach.normalize(vectors, norms)
