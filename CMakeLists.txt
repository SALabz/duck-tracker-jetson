cmake_minimum_required(VERSION 3.10)
project(duck_tracker LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 87)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    include
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
)

link_directories(
    /usr/local/cuda/lib64
    /usr/lib/aarch64-linux-gnu
)

add_executable(duck_tracker
    src/main.cpp
    src/pid_controller.cpp
    src/serial_control.cpp
    src/preprocess_kernel.cu
    src/centroid_kernel.cu
    src/preprocess_plugin.cpp
)

target_link_libraries(duck_tracker
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    stdc++fs
)
