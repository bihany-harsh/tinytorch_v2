cmake_minimum_required(VERSION 3.16)

project(tinytorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build C++ tests" ON)

include_directories(${CMAKE_SOURCE_DIR}/csrc)

# collecting the files
file(GLOB_RECURSE TENSOR_CORE_SOURCES
    "csrc/tensor_core/*.cpp"
)

file(GLOB_RECURSE UTIL_SOURCES
    "csrc/utils/*.cpp"
)

add_library(tinytorch_core STATIC
    ${TENSOR_CORE_SOURCES}
    ${UTIL_SOURCES}
)

target_include_directories(tinytorch_core PUBLIC
    ${CMAKE_SOURCE_DIR}/csrc
)

target_compile_options(tinytorch_core PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTS)
    add_executable(test_tinytorch
        main.cpp
    )
    
    target_link_libraries(test_tinytorch PRIVATE tinytorch_core)
    
    set_target_properties(test_tinytorch PROPERTIES
        OUTPUT_NAME "main"
    )
endif()

message(STATUS "Build C++ tests: ${BUILD_TESTS}")
message(STATUS "=================================")