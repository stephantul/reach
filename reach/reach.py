"""A class for working with vector representations."""
import json
import logging
import os
from io import open

import numpy as np
from tqdm import tqdm


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

    def __init__(self, vectors, items, name="", unk_index=None):
        """Initialize a Reach instance with an array and list of items."""
        if len(items) != len(vectors):
            raise ValueError(
                "Your vector space and list of items are not "
                "the same length: "
                f"{len(vectors)} != {len(items)}"
            )
        if isinstance(items, (dict, set)):
            raise ValueError(
                "Your item list is a set or dict, and might not "
                "retain order in the conversion to internal look"
                "-ups. Please convert it to list and check the "
                "order."
            )

        self.items = {w: idx for idx, w in enumerate(items)}
        self.indices = {v: k for k, v in self.items.items()}

        self.vectors = np.asarray(vectors)
        self.unk_index = unk_index

        self.name = name

    @property
    def size(self):
        return self.vectors.shape[1]

    @property
    def vectors(self):
        return self._vectors

    @vectors.setter
    def vectors(self, x):
        x = np.array(x)
        assert np.ndim(x) == 2 and x.shape[0] == len(self.items)
        self._vectors = x
        self.norm_vectors = self.normalize(x)

    @staticmethod
    def load(
        pathtovector,
        wordlist=(),
        num_to_load=None,
        truncate_embeddings=None,
        unk_word=None,
        sep=" ",
        recover_from_errors=False,
        **kwargs,
    ):
        r"""
        Read a file in word2vec .txt format.

        The load function will raise a ValueError when trying to load items
        which do not conform to line lengths.

        Parameters
        ----------
        pathtovector : string
            The path to the vector file.
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
        vectors, items = Reach._load(
            pathtovector,
            wordlist,
            num_to_load,
            truncate_embeddings,
            sep,
            recover_from_errors,
        )
        if unk_word is not None:
            if unk_word not in set(items):
                unk_vec = np.zeros((1, vectors.shape[1]))
                vectors = np.concatenate([unk_vec, vectors], 0)
                items = [unk_word] + items
                unk_index = 0
            else:
                unk_index = items.index(unk_word)
        else:
            unk_index = None

        return Reach(
            vectors, items, name=os.path.split(pathtovector)[-1], unk_index=unk_index
        )

    @staticmethod
    def _load(
        pathtovector,
        wordlist,
        num_to_load,
        truncate_embeddings,
        sep,
        recover_from_errors,
    ):
        """Load a matrix and wordlist from a .vec file."""
        vectors = []
        addedwords = set()
        words = []

        try:
            wordlist = set(wordlist)
        except ValueError:
            wordlist = set()

        logger.info(f"Loading {pathtovector}")

        firstline = open(pathtovector).readline().strip()
        try:
            num, size = firstline.split(sep)
            num, size = int(num), int(size)
            logger.info(f"Vector space: {num} by {size}")
            header = True
        except ValueError:
            size = len(firstline.split(sep)) - 1
            logger.info(f"Vector space: {size} dim, # items unknown")
            word, rest = firstline.split(sep, 1)
            # If the first line is correctly parseable, set header to False.
            header = False

        if truncate_embeddings is None or truncate_embeddings == 0:
            truncate_embeddings = size

        for idx, line in enumerate(open(pathtovector, encoding="utf-8")):

            if header and idx == 0:
                continue

            word, rest = line.rstrip(" \n").split(sep, 1)

            if wordlist and word not in wordlist:
                continue

            if word in addedwords:
                e = f"Duplicate: {word} on line {idx+1} was in the vector space twice"
                if recover_from_errors:
                    print(e)
                    continue
                raise ValueError(e)

            if len(rest.split(sep)) != size:
                e = (
                    f"Incorrect input at index {idx+1}, size"
                    f"is {len(rest.split())}, expected {size}."
                )
                if recover_from_errors:
                    print(e)
                    continue
                raise ValueError(e)

            words.append(word)
            addedwords.add(word)
            vectors.append(np.fromstring(rest, sep=sep)[:truncate_embeddings])

            if num_to_load is not None and len(addedwords) >= num_to_load:
                break

        vectors = np.array(vectors).astype(np.float32)

        logger.info("Loading finished")
        if wordlist:
            diff = wordlist - addedwords
            if diff:
                logger.info(
                    "Not all items from your wordlist were in your "
                    f"vector space: {diff}."
                )

        return vectors, words

    def __getitem__(self, item):
        """Get the vector for a single item."""
        return self.vectors[self.items[item]]

    def vectorize(self, tokens, remove_oov=False, norm=False):
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
        index = np.asarray(index)
        if norm:
            return self.norm_vectors[index]
        else:
            return self.vectors[index]

    def mean_vector(self, tokens, remove_oov=False, norm=False):
        """
        Get the mean vector of a sentence.

        Parameters
        ----------
        tokens : object or list of objects
            The tokens to take the mean of.
        remove_oov : bool, optional, default False
            Whether to remove OOV items. If False, OOV items are replaced by
            the UNK glyph.
        norm : bool, optional, default False
            Whether to take the mean of the unit vectors, or the regular vectors.

        Returns
        -------
        mean : numpy array
            A vector with M dimensions, where M is the size of the vector space.

        """
        return self.vectorize(tokens, remove_oov, norm).mean(0)

    def bow(self, tokens, remove_oov=False):
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
        bow : generator
            A BOW representation of the list of items.

        """
        out = []
        for t in tokens:
            try:
                out.append(self.items[t])
            except KeyError:
                if remove_oov:
                    pass
                if self.unk_index is None:
                    raise ValueError(
                        "You supplied OOV items but didn't "
                        "provide the index of the replacement "
                        "glyph. Either set remove_oov to True, "
                        "or set unk_index to the index of the "
                        "item which replaces any OOV items."
                    )
                out.append(self.unk_index)

        return out

    def transform(self, corpus, remove_oov=False, norm=False):
        """
        Transform a corpus by repeated calls to vectorize, defined above.

        Parameters
        ----------
        corpus : A list of strings, list of list of strings.
            Represents a corpus as a list of sentences, where sentences
            can either be strings or lists of tokens.
        remove_oov : bool, optional, default False
            If True, removes OOV items from the input before vectorization.

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
        self, items, num=10, batch_size=100, show_progressbar=False, return_names=True
    ):
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
        return_names : bool, optional, default True
            Whether to return the item names, or just the distances.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is false,
            the returned list just contains distances.

        """
        # This line allows users to input single items.
        # We used to rely on string identities, but we now also allow
        # anything hashable as keys.
        # Might fail if a list of passed items is also in the vocabulary.
        # but I can't think of cases when this would happen, and what
        # user expectations are.
        try:
            if items in self.items:
                items = [items]
        except TypeError:
            pass
        x = np.stack([self.norm_vectors[self.items[x]] for x in items])

        result = self._batch(x, batch_size, num + 1, show_progressbar, return_names)

        # list call consumes the generator.
        return [x[1:] for x in result]

    def threshold(
        self,
        items,
        threshold=0.5,
        batch_size=100,
        show_progressbar=False,
        return_names=True,
    ):
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
        return_names : bool, optional, default True
            Whether to return the item names, or just the distances.

        Returns
        -------
        sim : array
            For each items in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is false,
            the returned list just contains distances.

        """
        # This line allows users to input single items.
        # We used to rely on string identities, but we now also allow
        # anything hashable as keys.
        # Might fail if a list of passed items is also in the vocabulary.
        # but I can't think of cases when this would happen, and what
        # user expectations are.
        try:
            if items in self.items:
                items = [items]
        except TypeError:
            pass
        x = np.stack([self.norm_vectors[self.items[x]] for x in items])

        result = self._threshold_batch(
            x, batch_size, threshold, show_progressbar, return_names
        )

        # list call consumes the generator.
        return [x[1:] for x in result]

    def nearest_neighbor(
        self, vectors, num=10, batch_size=100, show_progressbar=False, return_names=True
    ):
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
        return_names : bool, optional, default True
            Whether to return the item names, or just the distances.

        Returns
        -------
        sim : list of tuples.
            For each item in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is set to
            false, only the distances are returned.

        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        return list(
            self._batch(vectors, batch_size, num, show_progressbar, return_names)
        )

    def nearest_neighbor_threshold(
        self,
        vectors,
        threshold=0.5,
        batch_size=100,
        show_progressbar=False,
        return_names=True,
    ):
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
        return_names : bool, optional, default True
            Whether to return the item names, or just the distances.

        Returns
        -------
        sim : list of tuples.
            For each item in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is set to
            false, only the distances are returned.

        """
        vectors = np.array(vectors)
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        return list(
            self._threshold_batch(
                vectors, batch_size, threshold, show_progressbar, return_names
            )
        )

    def _threshold_batch(
        self, vectors, batch_size, threshold, show_progressbar, return_names
    ):
        """Batched cosine distance."""
        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):

            batch = vectors[i : i + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            for lidx, sims in enumerate(similarities):
                indices = np.flatnonzero(sims >= threshold)
                sorted_indices = indices[np.argsort(-sims[indices])]
                if return_names:
                    yield [(self.indices[d], sims[d]) for d in sorted_indices]
                else:
                    yield list(sims[sorted_indices])

    def _batch(self, vectors, batch_size, num, show_progressbar, return_names):
        """Batched cosine distance."""
        if num < 1:
            raise ValueError("num should be >= 1, is now {num}")

        for i in tqdm(range(0, len(vectors), batch_size), disable=not show_progressbar):

            batch = vectors[i : i + batch_size]
            similarities = self._sim(batch, self.norm_vectors)
            if num == 1:
                sorted_indices = np.argmax(similarities, 1)[:, None]
            else:
                sorted_indices = np.argpartition(-similarities, kth=num, axis=1)
                sorted_indices = sorted_indices[:, :num]
            for lidx, indices in enumerate(sorted_indices):
                sims = similarities[lidx, indices]
                if return_names:
                    dindex = np.argsort(-sims)
                    yield [(self.indices[indices[d]], sims[d]) for d in dindex]
                else:
                    yield list(-1 * np.sort(-sims))

    @staticmethod
    def normalize(vectors):
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
        vectors = np.copy(vectors)
        if np.ndim(vectors) == 1:
            norm = np.linalg.norm(vectors)
            if norm == 0:
                return np.zeros_like(vectors)
            return vectors / norm

        norm = np.linalg.norm(vectors, axis=1)

        if np.any(norm == 0):

            nonzero = norm > 0

            result = np.zeros_like(vectors)

            n = norm[nonzero]
            p = vectors[nonzero]
            result[nonzero] = p / n[:, None]

            return result
        else:
            return vectors / norm[:, None]

    def vector_similarity(self, vector, items):
        """Compute the similarity between a vector and a set of items."""
        items_vec = np.stack([self.norm_vectors[self.items[x]] for x in items])
        return self._sim(vector, items_vec)

    def _sim(self, x, y):
        """Distance function."""
        sim = self.normalize(x).dot(y.T)
        return np.clip(sim, a_min=0.0, a_max=1.0)

    def similarity(self, i1, i2):
        """
        Compute the similarity between two sets of items.

        Parameters
        ----------
        i1 : object
            The first set of items.
        i2 : object
            The second set of item.

        Returns
        -------
        sim : array of floats
            An array of similarity scores between 1 and 0.

        """
        try:
            if i1 in self.items:
                i1 = [i1]
        except TypeError:
            pass
        try:
            if i2 in self.items:
                i2 = [i2]
        except TypeError:
            pass
        i1_vec = np.stack([self.norm_vectors[self.items[x]] for x in i1])
        i2_vec = np.stack([self.norm_vectors[self.items[x]] for x in i2])
        return self._sim(i1_vec, i2_vec)

    def intersect(self, wordlist):
        """
        Intersect a reach instance with a wordlist.

        Parameters
        ----------
        wordlist : list of str
            A list of words to keep. Note that this wordlist need not include
            all words in the Reach instance. Any words which are in the
            wordlist, but not in the reach instance are ignored.

        """
        # Remove duplicates and oov words.
        wordlist = set(self.items) & set(wordlist)
        # Get indices of intersection.
        indices = np.sort([self.items[x] for x in wordlist])
        # Set unk_index to None if it is None or if it is not in indices
        unk_index = self.unk_index if self.unk_index in indices else None
        # Index vectors
        vectors = self.vectors[indices]
        # Index words
        wordlist = [self.indices[x] for x in indices]
        return Reach(vectors, wordlist, unk_index=unk_index)

    def union(self, other, check=True):
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
                f"The size of the embedding spaces was not the same: {self.size} and {other.size}"
            )
        union = sorted(set(self.items) | set(other.items))
        if check:
            intersection = set(self.items) & set(other.items)
            for x in intersection:
                if not np.allclose(self[x], other[x]):
                    raise ValueError(f"Term {x} was not the same in both instances")
        vectors = []
        for x in union:
            try:
                vectors.append(self[x])
            except KeyError:
                vectors.append(other[x])

        return Reach(np.stack(vectors), union)

    def save(self, path, write_header=True):
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

    def save_fast_format(self, filename):
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
        items = {"items": items, "unk_index": self.unk_index, "name": self.name}

        json.dump(items, open(f"{filename}_items.json", "w"))
        np.save(open(f"{filename}_vectors.npy", "wb"), self.vectors)

    @staticmethod
    def load_fast_format(filename):
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
        words, unk_index, name, vectors = Reach._load_fast(filename)
        return Reach(vectors, words, unk_index=unk_index, name=name)

    @staticmethod
    def _load_fast(filename):
        """Sub for fast loader."""
        it = json.load(open(f"{filename}_items.json"))
        words, unk_index, name = it["items"], it["unk_index"], it["name"]
        vectors = np.load(open(f"{filename}_vectors.npy", "rb"))
        return words, unk_index, name, vectors
