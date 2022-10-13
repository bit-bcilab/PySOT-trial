from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name='eval_toolkit.utils.region',
        sources=[
            'eval_toolkit/utils/region.pyx',
            'eval_toolkit/utils/src/region.c',
        ],
        include_dirs=[
            'eval_toolkit/utils/src'
        ]
    )
]

setup(
    name='eval_toolkit',
    packages=['eval_toolkit'],
    ext_modules=cythonize(ext_modules)
)
