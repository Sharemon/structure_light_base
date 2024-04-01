cmake_minimum_required(VERSION 3.0.0)
project(StructureLight VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
set(NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets)

set(CMAKE_CUDA_ARCHITECTURES 86)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE cpp_srcs src/*.cpp)
file(GLOB_RECURSE cu_srcs src/*.cu)
file(GLOB_RECURSE cpp_apps app/*.cpp)

add_executable(StructureLight ${cpp_srcs} ${cpp_apps} ${cu_srcs})
target_link_libraries(StructureLight ${OpenCV_LIBS})
