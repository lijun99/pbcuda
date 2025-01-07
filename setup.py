import os
from skbuild import setup
from setuptools import find_packages

# Set the CMake arguments
cmake_args = [
    f"-DPYTHON_EXECUTABLE={os.sys.executable}",
    "-DCMAKE_BUILD_TYPE=Release"  # You can adjust this as needed
]

setup(
    name="pbcuda",
    version="0.1.1",
    description="A Python Interface to CUDA with pybind11",
    author="",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.6",
)