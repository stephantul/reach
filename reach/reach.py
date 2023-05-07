"""A class for working with vector representations."""
from __future__ import annotations

import json
import logging
from io import TextIOWrapper, open
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from tqdm import tqdm

Dtype = Union[str, np.dtype]
File = Union[Path, TextIOWrapper]
PathLike = Union[str, Path]
Matrix = Union[np.ndarray, List[np.ndarray]]
SimilarityItem = List[Tuple[Hashable, float]]
SimilarityResult = List[SimilarityItem]
Tokens = Iterable[Hashable]


logger = logging.getLogger(__name__)


class Reach(object):
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
    unk_index : int or None, optional, default None
        The index of the UNK item. If this is None, any attempts at vectorizing
        OOV items will throw an error.

    Attributes
    ----------
    items : dict
        A mapping from items to ids.
    indices : dict
        A mapping from ids to items.
    vectors : numpy array
        The array representing the vector space.
    unk_index : int
        The integer index of your unknown glyph. This glyph will be inserted
        into your BoW space whenever an unknown item is encountered.
    norm_vectors : numpy array
        A normalized version of the vector space.
    size : int
        The dimensionality of the vector space.
    name : string
        The name of the Reach instance.

    """

    def __init__(
        self,
        vectors: Matrix,
        items: List[Hashable],
        name: str = "",
        unk_index: Optional[int] = None,
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

        self._items: Dict[Hashable, int] = {w: idx for idx, w in enumerate(items)}
        self._indices: Dict[int, Hashable] = {idx: w for w, idx in self.items.items()}
        self.vectors = np.asarray(vectors)
        self.unk_index = unk_index
        self.name = name

    def __len__(self) -> int:
        return len(self.items)

    @property
    def items(self) -> Dict[Hashable, int]:
        return self._items

    @property
    def indices(self) -> Dict[int, Hashable]:
        return self._indices

    @property
    def sorted_items(self) -> Tokens:
        items: Tokens = [
            item for item, _ in sorted(self.items.items(), key=lambda x: x[1])
        ]
        return items

    @property
    def size(self) -> int:
        return self.vectors.shape[1]

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @vectors.setter
    def vectors(self, x: Matrix) -> None:
        x = np.asarray(x)
        if not np.ndim(x) == 2:
            raise ValueError(f"Your array does not have 2 dimensions: {np.ndim(x)}")
        if not x.shape[0] == len(self.items):
            raise ValueError(
                f"Your array does not have the correct length, got {x.shape[0]},"
                f" expected {len(self.items)}"
            )
        self._vectors = x
        # Make sure norm vectors is updated.
        if hasattr(self, "_norm_vectors"):
            self._norm_vectors = self.normalize(x)

    @property
    def norm_vectors(self) -> np.ndarray:
        if not hasattr(self, "_norm_vectors"):
            self._norm_vectors = self.normalize(self.vectors)
        return self._norm_vectors

    @classmethod
    def load(
        cls,
        vector_file: Union[File, str],
        wordlist: Optional[Tuple[str, ...]] = None,
        num_to_load: Optional[int] = None,
        truncate_embeddings: Optional[int] = None,
        unk_word: Optional[str] = None,
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
        unk_word : object
            The object to treat as UNK in your vector space. If this is not
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
        file_handle: TextIOWrapper,
        wordlist: Optional[Tuple[str, ...]],
        num_to_load: Optional[int],
        truncate_embeddings: Optional[int],
        sep: str,
        recover_from_errors: bool,
        desired_dtype: Dtype,
    ) -> Tuple[np.ndarray, List[str]]:
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
    ) -> np.ndarray:
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
        vector: np.ndarray
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
        self, corpus: List[Tokens], remove_oov: bool = False, safeguard: bool = True
    ) -> np.ndarray:
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
        vector: np.ndarray
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

    def bow(self, tokens: Tokens, remove_oov: bool = False) -> List[int]:
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
        self, corpus: List[Tokens], remove_oov: bool = False, norm: bool = False
    ) -> List[np.ndarray]:
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

        vectors = np.stack([self.norm_vectors[self.items[x]] for x in items])
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
        vectors: np.ndarray,
        batch_size: int,
        threshold: float,
        show_progressbar: bool,
    ) -> Generator[SimilarityItem, None, None]:
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
        vectors: np.ndarray,
        batch_size: int,
        num: int,
        show_progressbar: bool,
    ) -> Generator[SimilarityItem, None, None]:
        """Batched cosine similarity."""
        if num < 1:
            raise ValueError("num should be >= 1, is now {num}")

        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):
            batch = vectors[i : i + batch_size]
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
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize a matrix of row vectors to unit length.

        Contains a shortcut if there are no zero vectors in the matrix.
        If there are zero vectors, we do some indexing tricks to avoid
        dividing by 0.

        Parameters
        ----------
        vectors : np.array
            The vectors to normalize.

        Returns
        -------
        vectors : np.array
            The input vectors, normalized to unit length.

        """
        if np.ndim(vectors) == 1:
            norm = np.linalg.norm(vectors)
            if norm == 0:
                return np.zeros_like(vectors)
            return vectors / norm

        vectors = np.copy(vectors)
        norm = np.linalg.norm(vectors, axis=1)

        if np.any(norm == 0):
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
        if isinstance(items, str):
            items = [items]

        items_vec = np.stack([self.norm_vectors[self.items[item]] for item in items])
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
        # Set unk_index to None if it is None or if it is not in indices
        unk_index = self.unk_index if self.unk_index in indices else None
        # Index vectors
        vectors = self.vectors[indices]
        # Index words
        itemlist = [self.indices[index] for index in indices]
        return Reach(vectors, itemlist, unk_index=unk_index)

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

        return Reach(np.stack(vectors), union)

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

            for i in range(len(self.items)):
                w = self.indices[i]
                vec = self.vectors[i]
                vec_string = " ".join([str(x) for x in vec])
                f.write(f"{w} {vec_string}\n")

    def save_fast_format(self, filename: str) -> None:
        """
        Save a reach instance in a fast format.

        The reach fast format stores the words and vectors of a Reach instance
        separately in a JSON and numpy format, respectively.

        Parameters
        ----------
        filename : str
            The prefix to add to the saved filename. Note that this is not the
            real filename under which these items are stored.
            The words and unk_index are stored under "{filename}_words.json",
            and the numpy matrix is saved under "{filename}_vectors.npy".

        """
        items, _ = zip(*sorted(self.items.items(), key=lambda x: x[1]))
        items_dict = {"items": items, "unk_index": self.unk_index, "name": self.name}

        with open(f"{filename}_items.json", "w") as file_handle:
            json.dump(items_dict, file_handle)
        with open(f"{filename}_vectors.npy", "wb") as file_handle:
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
            The filename prefix from which to load. Note that this is not a
            real filepath as such, but a shared prefix for both files.
            In order for this to work, both {filename}_words.json and
            {filename}_vectors.npy should be present.

        """
        with open(f"{filename}_items.json") as file_handle:
            items = json.load(file_handle)
        words, unk_index, name = items["items"], items["unk_index"], items["name"]

        with open(f"{filename}_vectors.npy", "rb") as file_handle:
            vectors = np.load(file_handle)
        vectors = vectors.astype(desired_dtype)
        return cls(vectors, words, unk_index=unk_index, name=name)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize an array to unit length."""
    return Reach.normalize(vectors)
