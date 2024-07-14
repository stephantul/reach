.. reach documentation master file, created by
   sphinx-quickstart on Sun Jul 14 20:50:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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


Contents
--------

.. toctree::
   :maxdepth: 1

   source/api
