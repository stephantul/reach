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
    Sparse variant of Reach. Usable when word embeddings have been transformed
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
    A class for working with pre-made vector representions of words.
    """

    def __init__(self, vectors, words, verbose=False, name=""):
        """
        A class for working with vector representations of words.

        :param vectors: a numpy array containing word representations
        :param words: a list of words, corresponding to the above word representations
        :param verbose: whether to use logging.INFO to show information about Reach.
        """

        # norm_vectors = [self.normalize(v) for v in vectors]

        self.words = {w: idx for idx, w in enumerate(words)}
        self.indices = {v: k for k, v in self.words.items()}

        self.vectors = vectors

        norm = np.sqrt(np.sum(np.square(self.vectors), axis=1))
        nonzero = norm > 0
        self.norm_vectors = np.zeros_like(self.vectors)
        self.norm_vectors[nonzero] = self.vectors[nonzero] / norm[nonzero, np.newaxis]

        self.size = vectors.shape[1]

        self._verbose = verbose
        self.zero = np.zeros((self.size,))
        self.name = name

    @staticmethod
    def load(pathtovector, header=True, verbose=False, unkword="UNK", padword="PAD"):
        """
        Reads a file in word2vec vector format.

        The load function will not take into account lines which are longer
        than vectorsize + 1 when split on space.
        Can cause problems if tokens like \n are assigned separate vectors,
        or the file includes vectors that include spaces.

        :param pathtovector: the path to the vector file.
        :param header: whether the vector file has a header of the type
        (NUMBER OF ITEMS, SIZE OF VECTOR).
        :param verbose: whether to make the resulting Reach verbose.
        :param unkword: the glyph to use as unknown word. Defaults to UNK
        :param padword: the glyph to use as padding word. Defaults to PAD
        """

        if unkword == padword:
            raise ValueError("The unkown and padding glyphs have the same form.")

        # open correctly.
        firstline = open(pathtovector, encoding='utf-8').readline().strip()

        if header:
            numlines, size = firstline.split()
            size, numlines = int(size), int(numlines)
        else:
            size = len(firstline.split()[1:])
            numlines = sum([1 for x in open(pathtovector, encoding='utf-8')])

        vectors = np.zeros((numlines+2, size), dtype=np.float32)
        words = [unkword, padword]

        logger.info("Loading {0}".format(pathtovector))
        logger.info("Vocab: {0}, Dim: {1}".format(numlines, size))

        for idx, line in enumerate(open(pathtovector, encoding='utf-8')):

            if header and idx == 0:
                continue

            line = line.split()
            if len(line) != size + 1:
                logger.error("wrong input at idx: {0}, {1}, {2}".format(idx,
                                                                        line[:-size],
                                                                        len(line)))
                continue

            words.append(line[0])
            vectors[len(words)-1] = list(map(np.float, line[1:]))

        if len(words) != len(set(words)):
            raise ValueError("The words contain duplicates, are your pad or unknown glyphs in the vocabulary of your vector space?")

        vectors = np.array(vectors).astype(np.float32)

        logger.info("Loading finished")

        return Reach(vectors, words, verbose, name=os.path.split(pathtovector)[-1])

    def vector(self, w):
        """
        Returns the vector of a word, or the zero vector if the word is OOV.

        :param w: the word for which to retrieve the vector.
        :return: the vector of the word if it is in vocab, the zero vector
        otherwise.
        """
        try:
            return self.vectors[self.words[w]]
        except KeyError:
            if self._verbose:
                logging.info("{0} was OOV".format(w))
            return self.zero

    def _calc_sim(self, vector):
        """
        Calculates the most similar words to a given vector.
        Useful for pre-computed means and sums of vectors.
        :param vector: the vector for which to return the most similar items.
        :return: a list of tuples, representing most similar items.
        """

        vector = self.normalize(vector)
        distances = np.dot(self.norm_vectors, vector)
        return [(self.indices[idx], distances[idx]) for idx in np.argsort(-distances)]

    def __getitem__(self, word):

        return self.vectors[self.words[word]]

    def bow(self, tokens, remove_oov=False):
        """
        Create a bow representation consistent with the vector space model
        defined by this class.

        :param tokens: a list of words -> ["I", "am", "a", "dog"]
        :param remove_oov: whether to remove OOV words from the input.
        If this is True, the length of the returned BOW representation might
        not be the length of the original representation.
        :return: a bow representation of the list of words.
            ex. ["0", "55", "23", "3456"]
        """
        temp = []

        if remove_oov:
            tokens = [x for x in tokens if x in self.words]

        for t in tokens:
            try:
                temp.append(self.words[t])
            except KeyError:
                temp.append(0)

        return temp

    def vectorize(self, tokens, remove_oov=False):
        """
        Vectorizes a sentence.

        :param tokens: a string or list of tokens
        :param remove_oov: whether to remove OOV words. If False, OOV words are replaced by the zero vector.
        :return: a vectorized sentence, where every word has been replaced by
        its vector. OOv words are either removed or replaced by zeros or a vector average.
        """
        if not tokens:
            return [self.zero]
        if isinstance(tokens, str):
            tokens = tokens.split()

        if remove_oov:
            return [self.vector(t) for t in tokens if t in self.words]
        else:
            return [self.vector(t) for t in tokens]

    def transform(self, corpus, remove_oov=False):
        """
        Transforms a corpus by repeated calls to vectorize, defined above.

        :param corpus: a list of list of tokens.
        :param remove_oov: removes OOV words from the input before
        vectorization.
        :return: the same list of lists, but with all words replaced by dense
        vectors.
        """
        return [self.vectorize(s, remove_oov=remove_oov) for s in corpus]

    def most_similar(self, w, num=10):
        """
        Returns the num most similar words to a given word.

        :param w: the word for which to return the most similar items.
        :param num: the number of most similar items to return.
        :return a list of most similar items.
        """
        return self._calc_sim(self[w])[1:num + 1]

    def _get_normed(self, word):
        """
        Get the normed version of a word vector.

        :param word: The word for which to retrieve the normed version.
        :return: A normed vector.
        """

        return self.norm_vectors[self.words[word]]

    def most_similar_batch(self, words, num=10, batch_size=100):
        """
        A batched version of the

        :param words: A list of words for which to return the n most similar items
        :param num: The number of most similar items to return.
        :param batch_size: The batch size to use.
        :return: A list of list of tuples, representing the most similar items
        """
        vectors = np.array([self._get_normed(w) for w in words])

        results = []

        # Single transpose, makes things faster.
        normed_transpose = self.norm_vectors.T

        for i in range(0, len(vectors), batch_size):

            distances = vectors[i: i+batch_size].dot(normed_transpose)
            lines = np.argsort(-distances)
            for lidx, line in enumerate(lines):
                results.append([(self.indices[idx], distances[lidx, idx]) for idx in line[1:num+1]])

        return results

    def nearest_neighbor(self, vector, num=10):
        """
        Finds the nearest neighbors to some vector.

        :param vector: The vector to find nearest neighbors to.
        :param num: The number of nearest neighbors to return.
        :return A list of tuples representing the words and their distances
        to the given vector.
        """

        vector = np.array(vector)
        return self._calc_sim(vector)[:num]

    @staticmethod
    def normalize(vector):
        """
        Normalizes a vector.

        :param vector: a numpy array or list to normalize.
        :return a normalized vector.
        """
        if not vector.any():
            return vector

        return vector / np.linalg.norm(vector)

    def similarity(self, w1, w2):
        """
        Computes the similarity between two words based on the cosine distance.
        First normalizes the vectors.

        :param w1: the first word.
        :param w2: the second word.
        :return: a similarity score between 1 and 0.
        """
        return self.norm_vectors[self.words[w1]].dot(self.norm_vectors[self.words[w2]])

    def save(self, path, write_header=True, unk_word="UNK", pad_word="PAD"):
        """
        Saves the current vector space in word2vec format.
        Note that UNK and PAD are not written to file, as these are technically
        not part of the vector space.

        :param path: The path to which to save the file.
        :param write_header: Whether to write a header.
        :param unk_word: the unknown glyph in the current vector space (if any)
        :param pad_word: the padding glyph in the current vector space (if any)
        :return: None
        """

        with open(path, 'w') as f:
            # header

            decrement = int(unk_word in self.words)
            decrement += int(pad_word in self.words)

            if write_header:
                f.write(u"{0} {1}\n".format(str(self.vectors.shape[0] - decrement), str(self.vectors.shape[1])))

            for i in range(len(self.words)):

                w = self.indices[i]

                if w in [unk_word, pad_word]:
                    continue

                vec = self.vectors[i]

                f.write(u"{0} {1}\n".format(w, " ".join([str(x) for x in vec])))


if __name__ == "__main__":

    pass
