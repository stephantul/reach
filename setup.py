# -*- coding: utf-8 -*-
"""Setup file."""

from setuptools import setup
from setuptools import find_packages


setup(name='reach',
      version='0.0.3',
      description='A light-weight package for working with pre-trained word embeddings',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/reach',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      install_requires=['numpy>=1.11.0'],
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',],
      keywords='word vectors natural language processing',
      zip_safe=True)
