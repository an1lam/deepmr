from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(
    name="Hello",
    ext_modules=cythonize("pyx/hello.pyx"),
    include_dirs=[np.get_include()],
)
