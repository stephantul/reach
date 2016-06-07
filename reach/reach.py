import logging
import numpy as np

from io import open


class Reach(object):
    """
    A class for working with pre-made vector representions of words.

    Partially based on Gensim word2vec class.
    """

    def __init__(self, pathtovector, header=True, verbose=True):
        r"""
        A class for working with vector representations of tokens.

        It will not take into account lines which are longer than
        vectorsize + 1 when split on space.
        Can cause problems if tokens like \n are assigned separate vectors,
        or the file includes vectors that include spaces.

        :param pathtovector: the path to the vector file.
        :param header: whether the vector file has a header of the type
        (NUMBER OF ITEMS, SIZE OF VECTOR).
        """
        # open correctly.
        firstline = open(pathtovector, encoding='utf-8').readline().strip()

        if header:
            numlines, size = firstline.split()
            size, numlines = int(size), int(numlines)
        else:
            size = len(firstline.split()[1:])
            numlines = sum([1 for x in open(pathtovector, encoding='utf-8')])

        vectors = np.zeros((numlines+2, size), dtype=np.float32)
        words = {u"UNK": 0, u"PAD": 1}

        print("Vocab: {0}, Dim: {1}".format(size, numlines))

        for idx, line in enumerate(open(pathtovector, encoding='utf-8')):

            if header and idx == 0:
                continue

            line = line.split()
            if len(line) != size + 1:
                print("wrong input at idx: {0}, {1}, {2}".format(idx,
                                                                 line[:-size],
                                                                 len(line)))
                continue

            words[line[0]] = len(words)
            vectors[words[line[0]]] = list(map(np.float, line[1:]))

        self.size = size

        norm_vectors = [self.normalize(v) for v in vectors]

        self.vectors = np.array(vectors).astype(np.float32)
        self.norm_vectors = np.array(norm_vectors).astype(np.float32)

        self._words = words
        self._indices = {v: k for k, v in self._words.items()}
        self._verbose = verbose
        self._zero = np.zeros((self.size,))

    def vector(self, w):
        """
        Returns the vector of a word, or the zero vector if the word is OOV.

        :param w: the word for which to retrieve the vector.
        :return: the vector of the word if it is in vocab, the zero vector
        otherwise.
        """
        try:
            return self.vectors[self._words[w]]
        except KeyError:
            if self._verbose:
                logging.info("{0} was OOV".format(w))
            return self.vectors[0]

    def __getitem__(self, word):

        return self.vectors[self._words[word]]

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
            tokens = [x for x in tokens if x in self._words]

        for t in tokens:
            try:
                temp.append(self._words[t])
            except KeyError:
                temp.append(0)

        return temp

    def vectorize(self, tokens, remove_oov=False):
        """
        Vectorizes a sentence.

        :param tokens: a list of tokens
        :return: a vectorized sentence, where every word has been replaced by
        its vector, and OOV words are replaced
        by the zero vector.
        """
        if not tokens:
            return [self._zero]

        if remove_oov:
            return [self.vector(t) for t in tokens if t in self._words]
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
        return self._calc_sim(self[w])[1:num+1]

    def nearest_neighbor(self, vector, num=10):

        return self._calc_sim(vector)[:num]

    def _calc_sim(self, vector):
        """
        Calculates the most similar words to a given vector.
        Useful for pre-computed means and sums of vectors.

        :param vector: the vector for which to return the most similar items.
        :param num: the number of most similar items to return.
        :return: a list of tuples, representing most similar items.
        """
        vector = self.normalize(vector)
        distances = np.dot(self.norm_vectors, vector)

        return [(self._indices[idx], distances[idx]) for idx in np.argsort(-distances)]

    @staticmethod
    def normalize(vector):
        """
        Normalizes a vector.

        :param vector: a numpy array or list to normalize.
        :return a normalized vector.
        """
        if not vector.any():
            return vector

        return vector / np.sqrt(sum(np.power(vector, 2)))

    def similarity(self, w1, w2):
        """
        Computes the similarity between two words based on the cosine distance.
        First normalizes the vectors.

        :param w1: the first word.
        :param w2: the second word.
        :return: a similarity score between 1 and 0.
        """
        return self.norm_vectors[self._words[w1]].dot(self.norm_vectors[self._words[w2]])

if __name__ == "__main__":

    pass
