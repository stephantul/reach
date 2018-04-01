"""A class for working with vector representations."""
import logging
import json
import numpy as np
import os

from collections import Counter
from io import open
from os import path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Spreach(object):
    """
    Sparse variant of Reach.

    Usable when word embeddings have been transformed
    into a sparse space with e.g. https://github.com/mfaruqui/sparse-coding.

    Parameters
    ----------
    vecp : str
        System path to vector file location.

    header : bool, optional, default True
        Indicates if first line should be skipped.

    Attributes
    ----------
    vecd : dict
        Sparse dictionary representation of word embedding matrix where key
        is a vocab word and value a dictionary with indices and values.

    """

    def __init__(self, vecp, header=True):
        """Load file, pop header if necessary."""
        self.vecd = {}
        jsf = vecp.replace('.txt', '.json')
        if not path.isfile(jsf):
            self._load(open(vecp, encoding='utf-8'), header)
            json.dump(self.vecd, open(jsf, 'w'))
        else:
            self.vecd = json.load(open(jsf))

    def _load(self, vecf, header):
        """Load data to vec dictionary."""
        for i, line in enumerate(vecf):
            if not i and header:
                continue
            row = line.split()
            key = row.pop(0)
            self.vecd[key] = {str(i): float(k) for i, k in
                              enumerate(row) if float(k)}
        vecf.close()

    def transform(self, tokens):
        """Transform string or list of tokens to vectors.

        Parameters
        ----------
        tokens : str or list of strings
            Tokens that will be looked up in the vocab for their embedding.

        Returns
        -------
        c : dict
            Sparse vector summed across words.

        """
        if isinstance(tokens, str):
            tokens = tokens.split()
        c = Counter()
        for token in tokens:
            c += Counter(self.vecd.get(token, {'OOV': 0.01}))
        return dict(c)


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
    name : string, optional
        A string giving the name of the current reach. Only useful if you
        have multiple spaces and want to keep track of them.

    Attributes
    ----------
    items : dict
        A mapping from items to ids.
    indices : dict
        A mapping from ids to items.
    vectors : numpy array
        The array representing the vector space.
    norm_vectors : numpy array
        A normalized version of the vector space.
    unk_index : int
        The integer index of your unknown glyph. This glyph will be inserted
        into your BoW space whenever an unknown item is encountered.
    size : int
        The dimensionality of the vector space.
    name : string
        The name of the Reach instance.

    """

    def __init__(self, vectors, items, name="", unk_index=None):
        """Initialize a Reach instance with an array and list of items."""
        if len(items) != len(vectors):
            raise ValueError("Your vector space and list of items are not "
                             "the same length: "
                             "{} != {}".format(len(vectors), len(items)))
        if isinstance(items, dict) or isinstance(items, set):
            raise ValueError("Your item list is a set or dict, and might not "
                             "retain order in the conversion to internal look"
                             "-ups. Please convert it to list and check the "
                             "order.")

        self.items = {w: idx for idx, w in enumerate(items)}
        self.indices = {v: k for k, v in self.items.items()}

        self.vectors = np.asarray(vectors)
        self.norm_vectors = self.normalize(self.vectors)
        self.unk_index = unk_index

        self.size = self.vectors.shape[1]
        self.name = name

    @staticmethod
    def load(pathtovector,
             header=True,
             unk_index=None,
             wordlist=(),
             num_to_load=None,
             truncate_embeddings=None):
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
        unk_index : int, optional, default None
            The index of your unknown glyph. If this is set to None, your reach
            can't assing a BOW index to unknown items, and will throw an error
            whenever you try to do so.
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

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        vectors, items = Reach._load(pathtovector,
                                     header,
                                     wordlist,
                                     num_to_load,
                                     truncate_embeddings)
        return Reach(vectors,
                     items,
                     name=os.path.split(pathtovector)[-1],
                     unk_index=unk_index)

    @staticmethod
    def _load(pathtovector,
              header=True,
              wordlist=(),
              num_to_load=None,
              truncate_embeddings=None):
        """Load a matrix and wordlist from a .vec file."""
        vectors = []
        addedwords = set()
        words = []

        try:
            wordlist = set(wordlist)
        except ValueError:
            wordlist = set()

        logger.info("Loading {0}".format(pathtovector))

        firstline = open(pathtovector).readline()
        if header:
            num, size = firstline.split()
            num, size = int(num), int(size)
            logger.info("Vector space: {} by {}".format(num, size))
        else:
            size = len(firstline.split()) - 1
            logger.info("Vector space: {} dim, # items unknown".format(size))

        if truncate_embeddings is None or truncate_embeddings == 0:
            truncate_embeddings = size

        for idx, line in enumerate(open(pathtovector, encoding='utf-8')):

            if header and idx == 0:
                continue

            word, rest = line.rstrip(" \n").split(" ", 1)

            if wordlist and word not in wordlist:
                continue

            if word in addedwords:
                raise ValueError("Duplicate: {} was in the "
                                 "vector space twice".format(line[0]))

            if len(rest.split()) != size:
                raise ValueError("Incorrect input at index {}, size "
                                 "is {}, expected "
                                 "{}".format(idx, len(rest.split()), size))

            words.append(word)
            addedwords.add(word)
            vectors.append(np.fromstring(rest, sep=" ")[:truncate_embeddings])

            if num_to_load is not None and len(addedwords) >= num_to_load:
                break

        vectors = np.array(vectors).astype(np.float32)

        logger.info("Loading finished")
        if wordlist:
            diff = wordlist - addedwords
            if diff:
                logger.info("Not all items from your wordlist were in your "
                            "vector space: {}.".format(diff))

        return vectors, words

    def _vector(self, i, norm=False):
        """
        Return the vector of an item, or the zero vector if the item is OOV.

        Parameters
        ----------
        item : object
            The item for which to retrieve the vector.
        norm : bool, optional, default False
            If true, this function returns the normalized vector.
            If not, this returns the regular vector.

        Returns
        -------
        v : numpy array
            The vector of the item if the item is in vocab, the zero vector
            otherwise.

        """
        try:
            if norm:
                return self.norm_vectors[self.items[i]]
            return self.vectors[self.items[i]]
        except KeyError:
            if self.unk_index is not None:
                return self.vectors[self.unk_index]
            raise ValueError("'{}' is not present in the vector "
                             "space.".format(i))

    def __getitem__(self, item):
        """Get the vector for a single item."""
        return self.vectors[self.items[item]]

    def _zero(self):
        """Get a zero vector."""
        return np.zeros((self.size,))

    def vectorize(self, tokens, remove_oov=False, norm=False):
        """
        Vectorize a sentence by replacing all items with their vectors.

        Parameters
        ----------
        tokens : object or list of objects
            The tokens to vectorize.
        remove_oov : bool, optional, default False
            Whether to remove OOV items. If False, OOV items are replaced by
            the zero vector. If this is True, the returned sequence might
            have a different length than the original sequence.

        Returns
        -------
        s : numpy array
            An M * N matrix, where every item has been replaced by
            its vector. OOV items are either removed, replaced by
            zero vectors, or by the value of the UNK glyph.

        """
        if remove_oov:
            tokens = [t for t in tokens if t in self.items]
        if not tokens:
            return self._zero()[None, :]

        return np.stack([self._vector(t, norm=norm) for t in tokens])

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
        if remove_oov:
            tokens = [x for x in tokens if x in self.items]

        for t in tokens:
            try:
                yield self.items[t]
            except KeyError:
                if self.unk_index is None:
                    raise ValueError("You supplied OOV items but didn't supply"
                                     "the index of the replacement glyph. "
                                     "Either set remove_oov to True, or set "
                                     "unk_index to the index of the item "
                                     "which replaces any OOV items.")
                yield self.unk_index

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
        return [self.vectorize(s, remove_oov=remove_oov, norm=norm)
                for s in corpus]

    def most_similar(self,
                     items,
                     num=10,
                     batch_size=100,
                     show_progressbar=False,
                     return_names=True):
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
        sim : list of tuples.
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
        x = self.vectorize(items, norm=True, remove_oov=False)
        return [x[1:] for x in self._batch(x,
                                           batch_size,
                                           num+1,
                                           show_progressbar,
                                           return_names)]

    def _batch(self, vectors, batch_size, num, show_progressbar, return_names):
        """Batched cosine distance."""
        vectors = self.normalize(vectors)

        # Single transpose, makes things faster.
        normed_transpose = self.norm_vectors.T

        for i in tqdm(range(0, len(vectors), batch_size),
                      disable=not show_progressbar):

            distances = vectors[i: i+batch_size].dot(normed_transpose)
            for lidx, dist in enumerate(distances):
                sorted_indices = np.argsort(-dist)
                if return_names:
                    yield [(self.indices[idx], distances[lidx, idx])
                           for idx in sorted_indices[:num]]
                else:
                    yield [distances[lidx, idx]
                           for idx in sorted_indices[:num]]

    def nearest_neighbor(self,
                         vectors,
                         num=10,
                         batch_size=100,
                         show_progressbar=False,
                         return_names=True):
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
            in the form of (NAME, DISTANCE) tuples.

        """
        vectors = np.array(vectors)
        return list(self._batch(vectors,
                                batch_size,
                                num,
                                show_progressbar,
                                return_names))

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
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

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
        vector = self.normalize(vector)
        items = self.vectorize(items, norm=True, remove_oov=False)
        return self._similarity(vector, items)[0]

    def similarity(self, i1, i2):
        """
        Compute the similarity between two sets of items based.

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
        i1 = self.vectorize(i1, norm=True, remove_oov=False)
        i2 = self.vectorize(i2, norm=True, remove_oov=False)
        return self._similarity(i1, i2)

    def _similarity(self, v1, v2):
        """Return the similarity between two vectors."""
        return v1.dot(v2.T)

    def weighted_compose(self,
                         sentences,
                         f1,
                         f2,
                         weight_dict,
                         default_value=1.0,
                         remove_oov=True):
        """
        Compose items using a dictionary filled with weights.

        Useful for Tf-idf weighted composition.

        Parameters
        ----------
        sequences : nested list of items
            The sentences to compose over.
        f1 : function
            The first composition function.
        f2 : function
            The second composition function.
        weight_dict : dict
            A dictionary with mappings to assign to items.
            The values in this dictionary are first all made positive, and
            are then scaled between 0 and 1.
        default_value : float, optional, default 1.0
            The default value to assign to items not in the dictionary.
        remove_oov : bool, optional, default True

        Returns
        -------
        v : np.array
            Vector of a dimensionality equal to the number of dimensions of
            the items in this vector space.

        """
        min_val = np.abs(min(weight_dict.values()))
        max_val = max(weight_dict.values()) + min_val
        weight_dict = {k: (v + min_val) / max_val
                       for k, v in weight_dict.items()}

        weights = []
        for s in sentences:
            weights.append([weight_dict.get(w, default_value) for w in s])

        return self.compose(sentences,
                            f1,
                            f2,
                            weights=weights,
                            remove_oov=remove_oov)

    def compose(self, sequences, f1, f2, weights=(1,), remove_oov=False):
        """
        Complicated composition function.

        Parameters
        ----------
        sequences : nested list of items
            The sequences to compose over.
        f1 : function
            The first composition function.
        f2 : function
            The second composition function.
        weights : tuple
            The weight to assign to different parts of the composition.
        remove_oov : bool, optional, default True
            Whether to remove OOV items from the input.

        Returns
        -------
        v : np.array
            Vector of a dimensionality equal to the number of dimensions of
            the items in this vector space.

        """
        def _compose(vectors, function, weight):
            """Sub function for composition."""
            vectors *= weight[:, None]
            return function(vectors, axis=0)

        if len(weights) == 1:
            weights = [np.array([weights[0]] * len(x)) for x in sequences]
        elif any([len(x) != len(y) for x, y in zip(sequences, weights)]):
            raise ValueError("The number of items and number of weights "
                             "must match for each sequence.")
        else:
            weights = [np.array(w) for w in weights]

        composed = []

        for sent, weight in zip(sequences, weights):
            vec = self.vectorize(sent, remove_oov=remove_oov)
            composed.append(_compose(vec, f1, weight))

        return _compose(np.asarray(composed), f2, np.ones(len(composed)))

    def prune(self, wordlist):
        """
        Prune the current reach instance by removing items.

        Parameters
        ----------
        wordlist : list of str
            A list of words to keep. Note that this wordlist need not include
            all words in the Reach instance. Any words which are in the
            wordlist, but not in the reach instance are ignored.

        """
        # Remove duplicates
        wordlist = set(wordlist).intersection(set(self.items.keys()))
        indices = [self.items[w] for w in wordlist if w in self.items]
        if self.unk_index is not None and self.unk_index not in indices:
            raise ValueError("Your unknown item is not in your list of items. "
                             "Set it to None before pruning, or pass your "
                             "unknown item.")
        self.vectors = self.vectors[indices]
        self.norm_vectors = self.norm_vectors[indices]
        self.items = {w: idx for idx, w in enumerate(wordlist)}
        self.indices = {v: k for k, v in self.items.items()}
        if self.unk_index is not None:
            self.unk_index = self.items[wordlist[self.unk_index]]

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
        with open(path, 'w') as f:

            if write_header:
                f.write(u"{0} {1}\n".format(str(self.vectors.shape[0]),
                        str(self.vectors.shape[1])))

            for i in range(len(self.items)):

                w = self.indices[i]
                vec = self.vectors[i]

                f.write(u"{0} {1}\n".format(w,
                                            " ".join([str(x) for x in vec])))

    def save_fast_format(self, filepath):
        """Save a reach instance in a fast format, which includes ."""
        words, _ = zip(*sorted(self.items.items(), key=lambda x: x[1]))
        json.dump(words, open("{}_words.json".format(filepath), 'w'))
        np.save(open("{}_vectors.npy".format(filepath), 'wb'), self.vectors)

    @staticmethod
    def load_fast_format(filepath):
        """Load a reach instance in fast format."""
        words = json.load(open("{}_words.json".format(filepath)))
        vectors = np.load(open("{}_vectors.npy".format(filepath), 'rb'))
        return Reach(vectors, words)
