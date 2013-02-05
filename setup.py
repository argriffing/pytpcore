"""
This setup.py assumes that the C code has already been generated using Cython.
"""

from distutils.core import setup
from distutils.extension import Extension

my_ext = Extension(
        'pytpcore',
        ['pytpcore.c'],
        )

setup(
        name = 'pytpcore',
        version = '0.1',
        ext_modules = [my_ext],
        )

