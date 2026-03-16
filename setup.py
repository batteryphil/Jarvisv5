"""
Setup for the custom mamba_scan CUDA extension.

Linux-optimized: Uses -O3 with fast-math and sm_86 for RTX 3060.
Build with:  python3 setup.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mamba_scan",
    ext_modules=[
        CUDAExtension(
            name="mamba_scan",
            sources=["mamba_scan.cpp", "mamba_scan_kernel.cu"],
            extra_compile_args={
                # Linux g++
                "cxx": ["-O3", "-fno-strict-aliasing"],
                # CUDA: RTX 3060 = sm_86
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_86",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
