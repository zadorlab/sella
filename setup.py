#!/usr/bin/env python
import numpy as np

from setuptools import setup, Extension, find_packages

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = dict()

if use_cython:
    ext_modules = [Extension('sella.force_match',
                             ['sella/force_match.pyx']),
                   Extension('sella.cython_routines',
                             ['sella/cython_routines.pyx']),
                   Extension('sella.internal_cython',
                             ['sella/internal_cython.pyx']),
                   ]
    cmdclass['build_ext'] = build_ext
else:
    ext_modules = [Extension('sella.force_match',
                             ['sella/force_match.c']),
                   Extension('sella.cython_routines',
                             ['sella/cython_routines.c']),
                   Extension('sella.internal_cython',
                             ['sella/internal_cython.c']),
                   ]

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().strip().split()

setup(name='Sella',
      version='0.1.1',
      author='Eric Hermes',
      author_email='ehermes@sandia.gov',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      classifiers=['Development Status :: 3 - Alpha',
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
