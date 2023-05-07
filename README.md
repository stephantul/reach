# reach

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

# Installation

If you just want `reach`:

```
pip install reach
```

If you also want [`AutoReach`](#autoreach):

```
pip install reach[auto]
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

# autoreach

Reach also has a way of automatically inferring words from strings without using a pre-defined tokenizer, i.e., without splitting the string into words. This is useful because there might be mismatches between the tokenizer you happen to have on hand, and the word vectors you use. For example, if your vector space contains an embedding for the word `"it's"`, and your tokenizer splits this string into two tokens: `["it", "'s"]`, the embedding for `"it's"` will never be found.

autoreach solves this problem by only finding words from your pre-defined vocabulary in a string, this removing the need for any tokenization. We use the [aho-corasick algorithm](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm), which allows us to find substrings in linear time. The downside of using aho-corasick is that it also finds substrings of regular words. For example, the word `the` will be found as a substring of `these`. To circumvent this, we perform a regex-based clean-up step.

**Warning! The clean-up step involves checking for surrounding spaces and punctuation marks. Hence, if the language for which you use Reach does not actually use spaces and/or punctuation marks to designate word boundaries, the entire process might not work.**

### Example

```python
import numpy as np

from reach import AutoReach

words = ["dog", "walked", "home"]
vectors = np.random.randn(3, 32)

r = AutoReach(vectors, words)

sentence = "The dog, walked, home"
bow = r.bow(sentence)

found_words = [r.indices[index] for index in bow]
```

### benchmark

Because we no longer need to tokenize, `AutoReach` can be many times faster. In this benchmark, we compare to just splitting, and `nltk`'s `word_tokenize` function.

We will use the entirety of Mary Shelley's Frankenstein, which you can find [here](https://www.gutenberg.org/cache/epub/42324/pg42324.txt), and the glove.6b.50d vectors, which you can find [here](https://nlp.stanford.edu/data/glove.6B.zip).

```python
from pathlib import Path

from nltk import word_tokenize

from reach import AutoReach, Reach


txt = Path("pg42324.txt").read_text().lower()
normal_reach = Reach.load("glove.6B.100d.txt")
auto_reach = AutoReach.load("glove.6B.100d.txt")

# Ipython magic commands
%timeit normal_reach.vectorize(word_tokenize(txt), remove_oov=True)
# 345 ms ± 3.42 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit normal_reach.vectorize(txt.split(), remove_oov=True)
# 25.4 ms ± 132 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
%timeit auto_reach.vectorize(txt)
# 69.9 ms ± 237 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

```

As you can see, the tokenizer introduces significant overhead compared to just splitting, while using the aho-corasick algorithm to split is still reasonably fast.

# License

MIT

# Author

Stéphan Tulkens
