import os
from skbuild import setup
from setuptools import find_packages

# Set the CMake arguments
cmake_args = [
    f"-DPYTHON_EXECUTABLE={os.sys.executable}",
    "-DCMAKE_BUILD_TYPE=Release"  # Adjust as needed for debug or release
]

setup(
    name="pbcuda",
    version="0.1.2",
    description="A Python Interface to CUDA with pybind11",
    license="MIT",
    packages=find_packages(),  # Specify where to find the Python packages
    include_package_data=True,  # Include other files like .txt, .md, or .dat
    zip_safe=False,  # This is typically set to False for binary distributions
)
