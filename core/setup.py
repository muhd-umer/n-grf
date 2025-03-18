# setup.py

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root_dir = os.path.dirname(os.path.abspath(__file__))

cuda_sources = [
    os.path.join(root_dir, "core/cuda/rasterizer_impl.cu"),
    os.path.join(root_dir, "core/cuda/forward.cu"),
    os.path.join(root_dir, "core/cuda/backward.cu"),
    os.path.join(root_dir, "ext.cpp"),
]

setup(
    name="ard-r2f",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="core._C",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"
                    )
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
