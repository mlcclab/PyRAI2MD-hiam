######################################################
#
# PyRAI2MD 2 setup file
#
# Author Jingbai Li
# Aug 30 2022
#
######################################################

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyRAI2MD",
    version="2.4",
    author="Jingbai Li",
    author_email="lijingbai@zspt.edu.cn",
    description="Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.5.0',
        'tensorflow>=2.3.0',
        'cython>=0.29.0',
        'scikit-learn'
    ],
    extras_require={
        "pyNNsMD": ["pyNNsMD>=2.0.0"],
        # "GCNNP": ["GCNNP>=0.1.0"],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"PyRAI2MD": ["*.pyx"]},
    entry_points={
        'console_scripts': [
            'pyrai2md=PyRAI2MD.pyrai2md:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["materials", "science", "machine", "learning", "deep", "dynamics", "molecular", "potential"],
)
