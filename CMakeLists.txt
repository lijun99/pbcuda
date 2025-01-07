cmake_minimum_required(VERSION 3.18)

project(pbcuda LANGUAGES CXX CUDA)

# Use modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# python and pybind11 requirement
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
set(PYBIND11_FINDPYTHON ON) # Use new FindPython if available
find_package(pybind11 CONFIG REQUIRED)

# Set CUDA architecture targets
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86 90)
endif()

# Include directories
include_directories(include)

# Add pybind wrapper
pybind11_add_module(pbcuda
    src/cuda_module.cu
    src/pybind_wrapper.cpp
    )
# Set the installation path
set(PBCUDA_INSTALL_DIR ${Python_SITEARCH}/pbcuda)
# Specify the installation target
install(TARGETS pbcuda
        LIBRARY DESTINATION ${PBCUDA_INSTALL_DIR})
# Install the __init__.py file
install(FILES ${CMAKE_SOURCE_DIR}/pbcuda/__init__.py
        DESTINATION ${PBCUDA_INSTALL_DIR})