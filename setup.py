#!/usr/bin/env python
import sys
import os

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

debug = '--debug' in sys.argv or '-g' in sys.argv



class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())
        try:
            from Cython.Build import cythonize
        except ImportError:
            use_cython = False
        else:
            use_cython = True
        cy_suff = '.pyx' if use_cython else '.c'

        cy_files = [
            ['force_match'],
            ['utilities', 'blas'],
            ['utilities', 'math'],
        ]

        macros = []
        if debug:
            macros.append(('CYTHON_TRACE_NOGIL', '1'))

        ext_modules = []
        for cy_file in cy_files:
            ext_modules.append(Extension('.'.join(['sella', *cy_file]),
                                        [os.path.join('sella', *cy_file) + cy_suff],
                                        define_macros=macros,
                                        include_dirs=[np.get_include()]))

        if use_cython:
            compdir = dict(linetrace=debug, boundscheck=debug, language_level=3,
                        wraparound=False, cdivision=True)
            ext_modules = cythonize(ext_modules, compiler_directives=compdir)
        self.ext_modules = ext

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().strip().split()

setup(name='Sella',
      version='2.0.2',
      author='Eric Hermes',
      author_email='ehermes@sandia.gov',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Cython',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   'Topic :: Scientific/Engineering :: Physics'],
      python_requires='>=3.6',
      cmdclass={'build_ext':build_ext},
      setup_requires=['numpy'],
      install_requires=install_requires,
      )
