import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        extension = CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/adam.cu",
                "rasterize_points.cu",
                "conv.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"
                    )
                ]
            },
        )
        BuildExtension.build_extension(extension)
