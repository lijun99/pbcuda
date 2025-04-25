cmake_minimum_required(VERSION 3.18)
project(pbcuda LANGUAGES CXX CUDA)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80 86 89 90)
endif()


# Add the pybind11 module
pybind11_add_module(pbcuda
    src/cuda_module.cu
    src/pybind_wrapper.cpp
)

# Include directories
target_include_directories(pbcuda PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Condition to check if using skbuild
if (SKBUILD)
    # Install package
    install(TARGETS pbcuda
            LIBRARY DESTINATION pbcuda)
else()
    # Install package
    set(PBCUDA_INSTALL_DIR ${Python_SITEARCH}/pbcuda)

    # Install compiled library
    install(TARGETS pbcuda
            LIBRARY DESTINATION ${PBCUDA_INSTALL_DIR})

    # Install __init__.py
    install(FILES ${CMAKE_SOURCE_DIR}/pbcuda/__init__.py
            DESTINATION ${PBCUDA_INSTALL_DIR})
    message(STATUS "not using skbuild, standard install.")
endif()
