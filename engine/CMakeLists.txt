# CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	_cuda_impl/checks.cuh
	_cuda_impl/matrix.cuh
	_cuda_impl/projection.cu
	_cuda_impl/projection_backward.cu
	_cuda_impl/rasterize.cu
	_cuda_impl/rasterize_backward.cu
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/_cuda_impl)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
