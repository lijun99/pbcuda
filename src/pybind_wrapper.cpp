#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cuda_module.h"
#include <iostream>

namespace py = pybind11;

template <typename T>
py::array_t<T> add_array_typed(const py::array_t<T>& a, const py::array_t<T>& b)
{
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    auto c = py::array_t<T>(a_info.shape);
    py::buffer_info c_info = c.request();

    // Call CUDA function
    cuda_add<T>(static_cast<T*>(c_info.ptr), static_cast<const T*>(a_info.ptr), static_cast<const T*>(b_info.ptr), c_info.size);

    return c;
}


py::array py_add_array(py::array a, py::array b) {
    if (!py::array::ensure(a) || !py::array::ensure(b)) {
        throw std::runtime_error("Invalid array inputs");
    }

    if (a.dtype().is(py::dtype::of<float>())) {
        return add_array_typed<float>(a.cast<py::array_t<float>>(), b.cast<py::array_t<float>>());
    } else if (a.dtype().is(py::dtype::of<double>())) {
        return add_array_typed<double>(a.cast<py::array_t<double>>(), b.cast<py::array_t<double>>());
    } else if (a.dtype().is(py::dtype::of<int32_t>())) {
        return add_array_typed<int32_t>(a.cast<py::array_t<int32_t>>(), b.cast<py::array_t<int32_t>>());
   } else if (a.dtype().is(py::dtype::of<int64_t>())) {
        return add_array_typed<int64_t>(a.cast<py::array_t<int64_t>>(), b.cast<py::array_t<int64_t>>());
    } else {
        throw std::runtime_error("Unsupported data type. Supported types: float32, float64, int32, int64.");
    }
}


PYBIND11_MODULE(pbcuda, m) {
    m.def("cuda_add", &py_add_array, "A function that adds two arrays using CUDA");
}
