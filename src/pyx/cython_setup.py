from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(
    name="One hot encoding",
    ext_modules=cythonize("src/pyx/one_hot.pyx"),
    include_dirs=[np.get_include()],
)
