import logging
import json
import numpy as np
import os

from collections import Counter
from io import open
from os import path

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
    zero : numpy array
        A vector of zeros of the dimensionality of the vector space.
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

        self.vectors = vectors
        self.norm_vectors = self.normalize(vectors)
        self.unk_index = unk_index

        self.size = vectors.shape[1]

        self.zero = np.zeros((self.size,))
        self.name = name

    @staticmethod
    def load(pathtovector, header=True, unk_index=None):
        r"""
        Read a file in word2vec .txt format.

        The load function will not take into account lines which are longer
        than vectorsize + 1 when split on space.
        Can cause problems if tokens like \n are assigned separate vectors,
        or if the file includes words which have spaces.

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

        Returns
        -------
        r : Reach
            An initialized Reach instance.

        """
        firstline = open(pathtovector, encoding='utf-8').readline().strip()

        if header:
            numlines, size = firstline.split()
            size, numlines = int(size), int(numlines)
        else:
            size = len(firstline.split()[1:])
            numlines = sum([1 for x in open(pathtovector, encoding='utf-8')])

        vectors = np.zeros((numlines, size), dtype=np.float32)
        addedwords = set()
        words = []

        logger.info("Loading {0}".format(pathtovector))
        logger.info("Vocab: {0}, Dim: {1}".format(numlines, size))

        for idx, line in enumerate(open(pathtovector, encoding='utf-8')):

            if header and idx == 0:
                continue

            line = line.split()
            if len(line) != size + 1:
                logger.error("wrong input at idx: {0}, {1}"
                             "{2}".format(idx,
                                          line[:-size],
                                          len(line)))
                continue

            if line[0] in addedwords:
                raise ValueError("Duplicate: {} was in the "
                                 "vector space twice".format(line[0]))

            words.append(line[0])
            addedwords.add(line[0])
            vectors[len(words)-1] = list(map(np.float32, line[1:]))

        vectors = np.array(vectors).astype(np.float32)

        logger.info("Loading finished")

        return Reach(vectors,
                     words,
                     name=os.path.split(pathtovector)[-1],
                     unk_index=unk_index)

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
            logging.info("{0} was OOV".format(w))
            if self.unk_index:
                return self.vectors[self.unk_index]
            return self.zero

    def __getitem__(self, word):
        """Get the vector for a single word."""
        return self.vectors[self.words[word]]

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
            return np.copy(self.zero)[None, :]
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

    def most_similar(self, words, num=10, batch_size=100):
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

        Returns
        -------
        sim : list of tuples.
            For each word in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples.

        """
        if isinstance(words, str):
            words = [words]
        x = np.stack([self[w] for w in words])
        return [x[1:] for x in self._batch(x, batch_size, num+1)]

    def _batch(self, vectors, batch_size, num):
        """Batched cosine distance."""
        vectors = self.normalize(vectors)

        # Single transpose, makes things faster.
        normed_transpose = self.norm_vectors.T

        for i in range(0, len(vectors), batch_size):

            distances = vectors[i: i+batch_size].dot(normed_transpose)
            lines = np.argsort(-distances)
            for lidx, line in enumerate(lines):
                yield [(self.indices[idx], distances[lidx, idx])
                       for idx in line[1: num + 1]]

    def nearest_neighbor(self, vector, num=10):
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
            the batch size may increase the speed.

        Returns
        -------
        sim : list of tuples.
            For each word in the input the num most similar items are returned
            in the form of (NAME, DISTANCE) tuples.

        """
        vector = np.array(vector)
        return list(self._batch(vector, batch_size=1, num=10))

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
