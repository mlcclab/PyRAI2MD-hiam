######################################################
#
# PyRAI2MD 2 setup file for fssh.pyx
#
# Author Jingbai Li
# Aug 30 2022
#
######################################################

import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('PyRAI2MD/Dynamics/Propagators/fssh.pyx', compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
    package_dir={'cython_fssh': ''},
)


