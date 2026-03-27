from setuptools import setup, Extension
import numpy as np

extensions = [
    Extension("sella.force_match", ["sella/force_match.pyx"], include_dirs=[np.get_include()]),
    Extension("sella.utilities.blas", ["sella/utilities/blas.pyx"], include_dirs=[np.get_include()]),
    Extension("sella.utilities.math", ["sella/utilities/math.pyx"], include_dirs=[np.get_include()]),
]

setup(ext_modules=extensions)
