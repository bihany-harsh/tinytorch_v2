cmake_minimum_required(VERSION 3.15)
project(tinytorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(SOURCES
    csrc/tensor_core/tensor.cpp
    csrc/python/module.cpp
    csrc/python/tensor_core_bindings.cpp
)

pybind11_add_module(_core ${SOURCES})

target_include_directories(_core PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tensor_core
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/python
)
