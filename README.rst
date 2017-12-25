reach
=====

A light-weight package for working with pre-trained word embeddings.
Useful for input into neural networks, or for doing compositional semantics.

``reach`` can read in word vectors in ``word2vec`` or ``glove`` format without
any preprocessing.

The assumption behind ``reach`` is a no-hassle approach to featurization. The
vectorization and bow approaches know how to deal with OOV words, removing
these problems from your code.

``reach`` also includes nearest neighbor calculation for arbitrary vectors,
and composition functions for lists of items.

Example
'''''''

.. code-block:: python

  import numpy as np

  from reach import Reach


  # Word2vec style: with header.
  r = Reach.load("path/to/embeddings", header=True)

  # Glove style: without header.
  r = Reach.load("path/to/embeddings", header=False)

  # Alternatively, if you have a matrix, you can directly
  # input it.

  # Stand-in for word embeddings
  mtr = np.zeros((2, 300))
  words = ["cat", "dog"]
  r = Reach(mtr, words)

  # Get vectors through indexing.
  # Throws a KeyError if a word is not present.
  vector = r['cat']

  # Compare two words.
  similarity = r.similarity('cat', 'dog')

  # Find most similar.
  similarities = r.most_similar('cat', 5)

  sentence = 'a dog is the best creature alive'.split()
  corpus = [sentence, sentence, sentence]

  # bow representation consistent with word vectors,
  # for input into neural network.
  bow = r.bow(sentence)

  # vectorized representation.
  vectorized = r.vectorize(sentence)

  # can remove OOV words automatically.
  vectorized = r.vectorize(sentence, remove_oov=True)

  # vectorize corpus.
  transformed = r.transform(corpus)
