from setuptools import setup, Extension
import setuptools
import os
import sys

# get __version__, __author__, and __email__
exec(open("./fincoretails/metadata.py").read())

setup(
    name='fincoretails',
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/benmaier/fincoretails',
    license=__license__,
    description="Fit a bunch of distributions with finite core and heavy tails to data.",
    long_description='',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
                'numpy>=1.23',
                'scipy>=1.9',
                'sympy>=1.12',
    ],
    setup_requires=['pytest-runner'],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 ],
    project_urls={
        'Documentation': 'http://fincoretails.benmaier.org',
        'Contributing Statement': 'https://github.com/benmaier/fincoretails/blob/master/CONTRIBUTING.md',
        'Bug Reports': 'https://github.com/benmaier/fincoretails/issues',
        'Source': 'https://github.com/benmaier/fincoretails/',
        'PyPI': 'https://pypi.org/project/fincoretails/',
    },
    include_package_data=True,
    zip_safe=False,
)
