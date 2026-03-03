from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Identify if CUDA is available for the C++ extension
# For now, we use a universal C++ extension which handles both CPU and GPU tensors via LibTorch.

setup(
    name='mamba_scan',
    ext_modules=[
        CppExtension(
            name='mamba_scan', 
            sources=['mamba_scan.cpp'],
            extra_compile_args={'cxx': ['/O2', '/std:c++17']} if os.name == 'nt' else {'cxx': ['-O3', '-std=c++17']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
