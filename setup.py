from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

cymodule = 'inference'
setup(
  name='inference-cython',
  ext_modules=[Extension(cymodule,
                         [cymodule + '.pyx'],
                        include_dirs = [numpy.get_include()],
                         extra_compile_args=["-ffast-math"])],
  cmdclass={'build_ext': build_ext},
)