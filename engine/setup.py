# engine/setup.py

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root_dir = os.path.dirname(os.path.abspath(__file__))

cuda_sources = []


setup(
    name="engine",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="_C",
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
    install_requires=["setuptools>=75.8.0", "torch>=2.6.0", "numpy>=2.2.2"],
)
