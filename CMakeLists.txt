cmake_minimum_required(VERSION 3.23)
project(idealNN CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(
    idealNN
        main.cu
)

set_target_properties(
    idealNN
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
)
