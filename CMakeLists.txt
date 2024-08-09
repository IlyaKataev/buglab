cmake_minimum_required(VERSION 3.28)
project(buglab_cuda_clion CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(buglab_cuda_clion main.cu)
set_target_properties(buglab_cuda_clion PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
