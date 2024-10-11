# reach

[![Documentation Status](https://readthedocs.org/projects/reach/badge/?version=latest)](https://reach.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/reach.svg)](https://badge.fury.io/py/reach)
[![Downloads](https://pepy.tech/badge/reach)](https://pepy.tech/project/reach)

A light-weight package for working with pre-trained word embeddings.
Useful for input into neural networks, or for doing compositional semantics.

`reach` can read in word vectors in `word2vec` or `glove` format without
any preprocessing.

The assumption behind `reach` is a no-hassle approach to featurization. The
vectorization and bow approaches know how to deal with OOV words, removing
these problems from your code.

`reach` also includes nearest neighbor calculation for arbitrary vectors.

## Installation

If you just want `reach`:

```
pip install reach
```

## Example

```python
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
mtr = np.random.randn(8, 300)
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

# Can mean pool out of the box.
mean = r.mean_pool(sentence)
# Automatically take care of incorrect sentences
# these are set to the vector of the UNK word, or a vector of zeros.
corpus_mean = r.mean_pool_corpus([sentence, sentence, ["not_a_word"]], remove_oov=True, safeguard=False)

# vectorize corpus.
transformed = r.transform(corpus)

# Get nearest words to arbitrary vector
nearest = r.nearest_neighbor(np.random.randn(1, 300))

# Get every word within a certain threshold
thresholded = r.threshold("cat", threshold=.0)
```

## Loading and saving

`reach` has many options for saving and loading files, including custom separators, custom number of dimensions, loading a custom wordlist, custom number of words, and error recovery. One difference between `gensim` and `reach` is that `reach` loads both GloVe-style .vec files and regular word2vec files. Unlike `gensim`, `reach` does not support loading binary files.

### benchmark

On my machine (a 2022 M1 macbook pro), we get the following times for [`COW BIG`](https://github.com/clips/dutchembeddings), a file containing about 3 million rows and 320 dimensions.

| System | Time (7 loops)    |
|--------|-------------------|
| Gensim | 3min 57s ± 344 ms |
| reach  | 2min 14s ± 4.09 s |

## Fast format

`reach` has a special fast format, which is useful if you want to reload your word vectors often. The fast format can be created using the `save_fast_format` function, and loaded using the `load_fast_format` function. This is about equivalent to saving word vectors in `gensim`'s own format in terms of loading speed.

# License

MIT

# Author

Stéphan Tulkens
