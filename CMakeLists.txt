cmake_minimum_required(VERSION 3.10)

project(distribution_map)

add_executable(distribution_map main.cpp)

find_package(OpenCV REQUIRED)
include_directories(distribution_map ${OpenCV_INCLUDE_DIRS})
target_link_libraries(distribution_map ${OpenCV_LIBS})

target_compile_options(distribution_map PRIVATE -O3 -ffast-math -flto -march=native -s)
set(CMAKE_BUILD_TYPE Release)