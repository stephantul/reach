`reach <https://github.com/stephantul/reach>`_ documentation
===================================================================

Installation
-------------
A light-weight package for working with pre-trained word embeddings.
Useful for input into neural networks, or for doing compositional semantics.

reach can read in word vectors in word2vec or glove format without
any preprocessing.

The assumption behind reach is a no-hassle approach to featurization. The
vectorization and bow approaches know how to deal with OOV words, removing
these problems from your code.

reach also includes nearest neighbor calculation for arbitrary vectors.

Installation
------------

If you just want reach:


::

    $ pip install reach



If you also want AutoReach:


::

    $ pip install reach[auto]


Contents
--------

.. toctree::
   :maxdepth: 1

   source/api
