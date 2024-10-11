
<div align="center">
    <picture>
      <img width="25%" alt="Reach logo" src="assets/reach_logo.png">
    </picture>
  </a>
</div>

<div align="center">
  <h2>A small vector database for your RAG system</h2>
</div>

[![PyPI version](https://badge.fury.io/py/reach.svg)](https://badge.fury.io/py/reach)
[![Downloads](https://pepy.tech/badge/reach)](https://pepy.tech/project/reach)

# Table of contents

1. [Quickstart](#quickstart)
2. [What do I use it for?](#what-do-i-use-it-for)
3. [Example](#)

Reach is the lightest-weight vector store. Just put in some vectors, calculate query vectors, and off you go.

## Quickstart

```bash
pip install reach
```

Assume you've got some vectors and a model. We'll assume you have a nice [model2vec](https://github.com/MinishLab/model2vec) model.

```python
from model2vec import StaticModel
from reach import Reach

model = StaticModel.from_pretrained("minishlab/m2v_output_base")
texts = ["dog walked home", "cat walked home", "robot was in his lab"]
vectors = model.encode(texts)

r = Reach(vectors, texts)
r.most_similar(texts[0])

new_text = "robot went to his house"
similarities = r.nearest_neighbor(model.encode(new_text))

print(similarities)

# Store the vector space
r.save("tempo.json")
# Load it again
new_reach = Reach.load("tempo.json")

```

And that's it!

## What do I use it for?

Reach is an extremely simple but extremely fast vector store. No magic here, it just uses numpy really effectively to obtain impressive speeds. Reach will be fast enough for your RAG projects until 1M vectors, after which you may have to switch to something heavier.

Reach is designed to load really quickly from disk, see below, making it ideal for just-in-time projects, such as querying texts on the fly. No need to keep a heavy vector database running, just load your reach, do the computation, and then throw it away.

# Examples

Here's some examples and benchmarks.

## Retrieval

For your RAG system, you need fast retrieval. We got it!

```python
import numpy as np
from reach import Reach

dummy_words = list(map(str, range(100_000)))
dummy_vector = np.random.randn(100_000, 768)
r = Reach(dummy_vector, dummy_words)

# Query with a single vector
x = np.random.randn(768)
%timeit r.nearest_neighbor(x)
# 6.8 ms ± 286 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# Query reach with 10 vectors
x = np.random.randn(10, 768)
%timeit r.nearest_neighbor(x)
# 27.5 ms ± 187 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 2.7 ms per vector

# 100 vectors.
x = np.random.randn(100, 768)
%timeit r.nearest_neighbor(x)
# 143 ms ± 943 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 1.4 ms per vector
```

# Saving and loading

No need to keep a vector database in memory, or on some server. Just load and save your thing whenever you need it.

```python
import numpy as np
from reach import Reach

dummy_words = list(map(str, range(100_000)))
dummy_vector = np.random.randn(100_000, 768)
r = Reach(dummy_vector, dummy_words)

# Loading from disk
r.save("temp.json")
%timeit Reach.load("temp.json")
# 79.9 ms ± 1.22 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Installation

```
pip install reach
```

# License

MIT

# Author

Stéphan Tulkens
