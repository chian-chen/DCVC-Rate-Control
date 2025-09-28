# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import glob
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ------------------------------------------------------------------
# CUDA 編譯旗標
#   - compute_86 / sm_86 對應 RTX 3080
#   - 如需向下相容，可再加一行   '-gencode', 'arch=compute_80,code=sm_80',
# ------------------------------------------------------------------
extra_cuda_cflags = [
    '-O3', '--use_fast_math',
    '-gencode', 'arch=compute_86,code=sm_86',
]

# ------------------------------------------------------------------
# C++ / NVCC 旗標
# ------------------------------------------------------------------
cxx_flags  = ['-O3', '-std=c++17']
nvcc_flags = extra_cuda_cflags + ['-std=c++17']   # nvcc 也要 C++17

# Windows 平台改用 MSVC 最佳化旗標
if sys.platform == 'win32':
    cxx_flags = ['/O2', '/std:c++17']

# ------------------------------------------------------------------
# setup()
# ------------------------------------------------------------------
setup(
    name='inference_extensions_cuda',
    ext_modules=[
        CUDAExtension(
            name='inference_extensions_cuda',
            sources=glob.glob('*.cpp') + glob.glob('*.cu'),
            extra_compile_args={
                'cxx':  cxx_flags,
                'nvcc': nvcc_flags,
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
