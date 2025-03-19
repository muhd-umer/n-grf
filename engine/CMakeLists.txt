# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(ard-r2f LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# find required packages
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm)

# add CUDA sources
set(CUDA_SOURCES
    _cuda_impl/rasterize.cu
    _cuda_impl/forward.cu
    _cuda_impl/backward.cu
    _cuda_impl/auxiliary.h
    _cuda_impl/rasterize.h
    _cuda_impl/forward.h
    _cuda_impl/backward.h
)

# add library target
cuda_add_library(ard_r2f_cuda ${CUDA_SOURCES})

# link against PyTorch
target_link_libraries(ard_r2f_cuda ${TORCH_LIBRARIES})

# set compile flags
set_target_properties(ard_r2f_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

# add test executable
add_executable(test_rasterize tests/test_rasterize.py)