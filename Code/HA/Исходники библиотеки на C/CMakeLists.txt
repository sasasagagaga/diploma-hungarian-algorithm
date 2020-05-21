cmake_minimum_required(VERSION 3.9)

project(hungarian_algorithm)

set(CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(hungarian_algorithm SHARED hungarian_algorithm.c)
