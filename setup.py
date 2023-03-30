# -*- coding: utf-8 -*-
"""Setup file."""
from pathlib import Path
from setuptools import setup, find_packages


setup(
    name="reach",
    version="4.0.1",
    description="A light-weight package for working with pre-trained word embeddings",
    author="Stéphan Tulkens",
    author_email="stephantul@gmail.com",
    url="https://github.com/stephantul/reach",
    license="MIT",
    packages=find_packages(include=["reach"]),
    install_requires=["numpy", "tqdm"],
    extras_require={"auto": ["pyahocorasick"]},
    project_urls={
        "Source Code": "https://github.com/stephantul/reach",
        "Issue Tracker": "https://github.com/stephantul/reach/issues",
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="word vectors natural language processing",
    zip_safe=True,
    long_description=Path("README.md").read_text(),
    long_description_content_type='text/markdown',
)
