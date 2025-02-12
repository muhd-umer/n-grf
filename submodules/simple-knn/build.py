import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        cxx_compiler_flags = []
        if os.name == "nt":
            cxx_compiler_flags.append("/wd4624")

        extension = CUDAExtension(
            name="simple_knn._C",
            sources=["spatial.cu", "simple_knn.cu", "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags},
        )
        BuildExtension.build_extension(extension)
