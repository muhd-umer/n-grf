# render/setup.py

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root_dir = os.path.dirname(os.path.abspath(__file__))

cuda_sources = [
    os.path.join(root_dir, "_cuda_impl/render.cu"),
    os.path.join(root_dir, "_cuda_impl/render_backward.cu"),
    os.path.join(root_dir, "_cuda_impl/ext.cpp"),
]

c_flags = ["-O3", "-std=c++17"]
nvcc_flags = ["-O3", "-std=c++17"]


setup(
    name="render",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="_C",
            sources=cuda_sources,
            extra_compile_args={
                "cxx": c_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["setuptools>=75.8.0", "torch>=2.7.0", "numpy>=2.2.2"],
)
