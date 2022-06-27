import sys
import warnings

import setuptools

from src.pt2keras import PROJECT_NAME, __version__

# Python supported version checks.
if sys.version_info[:2] < (3, 7):
    raise RuntimeError('Python version >= 3.7 required.')

if sys.version_info[:2] > (3, 9):
    warnings.warn('Python version >= 3.10 may not be supported.')

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name=PROJECT_NAME,
    version=__version__,
    author='Jay Lee',
    author_email='ljay189@gmail.com',
    description='A simple PyTorch to Keras Model converter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JWLee89/pt2keras',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.7',
)
