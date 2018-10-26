reach
=====

A light-weight package for working with pre-trained word embeddings.
Useful for input into neural networks, or for doing compositional semantics.

``reach`` can read in word vectors in ``word2vec`` or ``glove`` format without
any preprocessing.

The assumption behind ``reach`` is a no-hassle approach to featurization. The
vectorization and bow approaches know how to deal with OOV words, removing
these problems from your code.

``reach`` also includes nearest neighbor calculation for arbitrary vectors.

Example
'''''''

.. code-block:: python

  import numpy as np

  from reach import Reach

  # Load from a .vec or .txt file
  # unk_word specifies which token is the "unknown" token.
  # If this is token is not in your vector space, it is added as an extra word
  # and a corresponding zero vector.
  # If it is in your embedding space, it is used.
  r = Reach.load("path/to/embeddings", unk_word="UNK")

  # Alternatively, if you have a matrix, you can directly
  # input it.

  # Stand-in for word embeddings
  mtr = np.random.rand(8, 300)
  words = ["UNK", "cat", "dog", "best", "creature", "alive", "span", "prose"]
  r = Reach(mtr, words, unk_index=0)

  # Get vectors through indexing.
  # Throws a KeyError if a word is not present.
  vector = r['cat']

  # Compare two words.
  similarity = r.similarity('cat', 'dog')

  # Find most similar.
  similarities = r.most_similar('cat', 2)

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
