cmake_minimum_required(VERSION 3.0.0)
project(StructureLight VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE cpp_srcs src/*.cpp)
file(GLOB_RECURSE cpp_apps app/*.cpp)

add_executable(StructureLight ${cpp_srcs} ${cpp_apps})
target_link_libraries(StructureLight ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
