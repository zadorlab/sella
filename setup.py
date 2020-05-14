#!/usr/bin/env python
import sys
import os

import numpy as np

from setuptools import setup, Extension, find_packages

debug = '--debug' in sys.argv or '-g' in sys.argv

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

cy_suff = '.pyx' if use_cython else '.c'

cy_files = [
    ['force_match'],
    ['internal', 'int_eval'],
    ['internal', 'int_classes'],
    ['utilities', 'blas'],
    ['utilities', 'math']
]

macros = []
if debug:
    macros.append(('CYTHON_TRACE_NOGIL', '1'))

ext_modules = []
for cy_file in cy_files:
    ext_modules.append(Extension('.'.join(['sella', *cy_file]),
                                 [os.path.join('sella', *cy_file) + cy_suff],
                                 define_macros=macros))

if use_cython:
    compdir = dict(linetrace=debug, boundscheck=debug, language_level=3,
                   wraparound=False, cdivision=True)
    ext_modules = cythonize(ext_modules, compiler_directives=compdir)

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().strip().split()

setup(name='Sella',
      version='1.0.3',
      author='Eric Hermes',
      author_email='ehermes@sandia.gov',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Cython',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Physics'],
      python_requires='>=3.5',
      install_requires=install_requires,
      )
