#!/usr/bin/env python
import numpy as np

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

with open('README.md', 'r') as f:
    long_description = f.read()

ext_modules = [Extension('sella.force_match',
                         ['sella/force_match.pyx']),
               Extension('sella.cython_routines',
                         ['sella/cython_routines.pyx']),
               ]

setup(name='Sella',
      version='0.0.1',
      author='Eric Hermes',
      author_email='ehermes@sandia.gov',
      long_description=long_description,
      long_description_type='text/markdown',
      packages=find_packages(),
      ext_modules=cythonize(ext_modules),
      include_dirs=[np.get_include()],
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 3.6'
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Physics'],
      python_requires='>=3.5',
      install_requires=['numpy', 'scipy>=1.1.0', 'cython'],
      )
