cmake_minimum_required(VERSION 3.23)
project(idealNN)
#project(idealNN CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(
    idealNN
        src/main.cpp
        src/layers/layer.cpp src/layers/layer.hpp src/sequential.cpp src/sequential.hpp)


set_target_properties(
    idealNN
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
)
