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
    Work with vector representations of words.

    Supports functions for calculating fast batched similarity
    between words or composite representations of words.

    Parameters
    ----------
    vectors : numpy array
        The vector space.
    words : list
        A list of words. Length must be equal to the number of vectors, and
        aligned with the vectors.
    name : string, optional
        A string giving the name of the current reach. Only useful if you
        have multiple spaces and want to keep track of them.

    Attributes
    ----------
    words : dict
        A mapping from words to ids.
    indices : dict
        A mapping from ids to words.
    vectors : numpy array
        The array representing the vector space.
    norm_vectors : numpy array
        A normalized version of the vector space.
    unk_index : int
        The integer index of your unknown glyph. This glyph will be inserted
        into your BoW space whenever an unknown word is encountered.
    size : int
        The dimensionality of the vector space.
    name : string
        The name of the Reach instance.

    """

    def __init__(self, vectors, words, name="", unk_index=None):
        """Initialize a Reach instance with an array and words."""
        if len(words) != len(vectors):
            raise ValueError("Your vector space and list of words are not "
                             "the same length: "
                             "{} != {}".format(len(vectors), len(words)))

        self.words = {w: idx for idx, w in enumerate(words)}
        self.indices = {v: k for k, v in self.words.items()}

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
            can't assing a BOW index to unknown words, and will throw an error
            whenever you try to do so.
        wordlist : iterable, optional, default ()
            A list of words you want loaded from the vector file. If this is
            None (default), all words will be loaded.
        num_to_load : int, optional, default None
            The number of words to load from the file. Because loading can take
            some time, it is sometimes useful to onlyl load the first n words
            from a vector file for quick inspection.
        truncate_embeddings : int, optional, default None
            If this value is not None, the vectors in the vector space will
            be truncated to the number of dimensions indicated by this value.

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        vectors, words = Reach._load(pathtovector,
                                     header,
                                     wordlist,
                                     num_to_load,
                                     truncate_embeddings)
        return Reach(vectors,
                     words,
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
            logger.info("Vector space: {} dim, # words unknown".format(size))

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
                logger.info("Not all words from your wordlist were in your "
                            "vector space: {}.".format(diff))

        return vectors, words

    def _vector(self, w, norm=False):
        """
        Return the vector of a word, or the zero vector if the word is OOV.

        Parameters
        ----------
        w : str
            The word for which to retrieve the vector.
        norm : bool, optional, default False
            If true, this function returns the normalized vector.
            If not, this returns the regular vector.

        Returns
        -------
        v : numpy array
            The vector of the word if the word is in vocab, the zero vector
            otherwise.

        """
        try:
            if norm:
                return self.norm_vectors[self.words[w]]
            return self.vectors[self.words[w]]
        except KeyError:
            if self.unk_index is not None:
                return self.vectors[self.unk_index]
            return self._zero()

    def __getitem__(self, word):
        """Get the vector for a single word."""
        return self.vectors[self.words[word]]

    def _zero(self):
        """Get a zero vector."""
        return np.zeros((self.size,))

    def vectorize(self, tokens, remove_oov=False):
        """
        Vectorize a sentence by replacing all words with their vectors.

        Parameters
        ----------
        tokens : string or list of string
            The tokens to vectorize.
        remove_oov : bool, optional, default False
            Whether to remove OOV words. If False, OOV words are replaced by
            the zero vector. If this is True, the returned sequence might
            have a different length than the original sequence.

        Returns
        -------
        s : numpy array
            An M * N matrix, where every word has been replaced by
            its vector. OOV words are either removed, replaced by
            zero vectors, or by the value of the UNK glyph.

        """
        if not tokens:
            return np.copy(self._zero())[None, :]
        if isinstance(tokens, str):
            tokens = tokens.split()

        if remove_oov:
            return np.stack([self._vector(t) for t in tokens
                            if t in self.words])
        else:
            return np.stack([self._vector(t) for t in tokens])

    def bow(self, tokens, remove_oov=False):
        """
        Create a bow representation of a list of tokens.

        Parameters
        ----------
        tokens : list.
            The list of words to change into a bag of words.
        remove_oov : bool.
            Whether to remove OOV words from the input.
            If this is True, the length of the returned BOW representation
            might not be the length of the original representation.

        Returns
        -------
        bow : generator
            A BOW representation of the list of words.

        """
        if remove_oov:
            tokens = [x for x in tokens if x in self.words]

        for t in tokens:
            try:
                yield self.words[t]
            except KeyError:
                if self.unk_index is None:
                    raise ValueError("You supplied OOV words but didn't supply"
                                     "the index of the replacement glyph. "
                                     "Either set remove_oov to True, or set "
                                     "unk_index to the index of the word "
                                     "which replaces any OOV words.")
                yield self.unk_index

    def transform(self, corpus, remove_oov=False):
        """
        Transform a corpus by repeated calls to vectorize, defined above.

        Parameters
        ----------
        corpus : A list of strings, list of list of strings.
            Represents a corpus as a list of sentences, where sentences
            can either be strings or lists of tokens.
        remove_oov : bool, optional, default False
            If True, removes OOV words from the input before vectorization.

        Returns
        -------
        c : list
            A list of numpy arrays, where each array represents the transformed
            sentence in the original list. The list is guaranteed to be the
            same length as the input list, but the arrays in the list may be
            of different lengths, depending on whether remove_oov is True.

        """
        return [self.vectorize(s, remove_oov=remove_oov) for s in corpus]

    def most_similar(self,
                     words,
                     num=10,
                     batch_size=100,
                     show_progressbar=False,
                     return_names=True):
        """
        Return the num most similar words to a given list of words.

        Parameters
        ----------
        words : list of strings or a single string.
            The words to get the most similar words to.
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
            For each word in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples. If return_names is false,
            the returned list just contains distances.

        """
        # This line allows users to input single items.
        # We used to rely on string identities, but we now also allow
        # anything hashable as keys.
        # Might fail if a list of passed items is also in the vocabulary.
        # but I can't think of cases when this would happen, and what
        # user expectations are.
        if words in self.words:
            words = [words]
        x = np.stack([self[w] for w in words])
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
        most_similar function can only handle words that are in vocab, and
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
            For each word in the input the num most similar items are returned
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
        """Normalize a matrix rowwise."""
        if np.ndim(vectors) == 1:
            vectors = vectors[None, :]

        norm = np.linalg.norm(vectors, axis=1)
        nonzero = norm > 0

        result = np.zeros_like(vectors)

        n = norm[nonzero]
        p = vectors[nonzero]
        result[nonzero] = p / n[:, None]
        return result

    def similarity(self, w1, w2):
        """
        Compute the similarity between two words based on cosine distance.

        First normalizes the vectors.

        Parameters
        ----------
        w1 : str
            The first word.
        w2 : str
            The second word.

        Returns
        -------
        sim : float
            A similarity score between 1 and 0.

        """
        vec = self.norm_vectors[self.words[w1]]
        return vec.dot(self.norm_vectors[self.words[w2]])

    def weighted_compose(self,
                         sentences,
                         f1,
                         f2,
                         weight_dict,
                         default_value=1.0,
                         remove_oov=True):
        """
        Compose words using a dictionary filled with weights.

        Useful for Tf-idf weighted composition.

        Parameters
        ----------
        sentences : nested list of items
            The sentences to compose over.
        f1 : function
            The first composition function.
        f2 : function
            The second composition function.
        weight_dict : dict
            A dictionary with mappings to assign to words.
            The values in this dictionary are first all made positive, and
            are then scaled between 0 and 1.
        default_value : float, optional, default 1.0
            The default value to assign to words not in the dictionary.
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

    def compose(self, sentences, f1, f2, weights=(1,), remove_oov=False):
        """
        Complicated composition function.

        Parameters
        ----------
        sentences : nested list of items
            The sentences to compose over.
        f1 : function
            The first composition function.
        f2 : function
            The second composition function.
        weight_dict : dict
            A dictionary with mappings to assign to words.
            The values in this dictionary are first all made positive, and
            are then scaled between 0 and 1.
        default_value : float, optional, default 1.0
            The default value to assign to words not in the dictionary.
        remove_oov : bool, optional, default True

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
            weights = [np.array([weights[0]] * len(x)) for x in sentences]
        elif any([len(x) != len(y) for x, y in zip(sentences, weights)]):
            raise ValueError("The number of words and number of weights "
                             "must match for each sentence.")
        else:
            weights = [np.array(w) for w in weights]

        composed = []

        for sent, weight in zip(sentences, weights):
            vec = self.vectorize(sent, remove_oov=remove_oov)
            composed.append(_compose(vec, f1, weight))

        return _compose(np.asarray(composed), f2, np.ones(len(composed)))

    def prune(self, wordlist):
        """
        Prune the current reach instance by removing words.

        Can be helpful if you don't have the vector space loaded on disk, or
        if you want to perform random sampling on the vector space in
        memory.

        Parameters
        ----------
        wordlist : list of str
            A list of words to keep.

        """
        # Remove duplicates
        wordlist = set(wordlist)
        indices = [self.words[w] for w in wordlist]
        if self.unk_index is not None and self.unk_index not in indices:
            raise ValueError("Your unknown word is not in your wordlist. "
                             "Set it to None before pruning, or pass your "
                             "unknown word.")
        self.vectors = self.vectors[indices]
        self.norm_vectors = self.norm_vectors[indices]
        self.words = {w: idx for idx, w in enumerate(wordlist)}
        self.indices = {v: k for k, v in self.words.items()}
        if self.unk_index is not None:
            self.unk_index = self.words[wordlist[self.unk_index]]

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

            for i in range(len(self.words)):

                w = self.indices[i]
                vec = self.vectors[i]

                f.write(u"{0} {1}\n".format(w,
                                            " ".join([str(x) for x in vec])))
