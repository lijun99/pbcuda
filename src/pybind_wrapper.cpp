#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cuda_module.h"

namespace py = pybind11;

void py_cuda_add(py::array_t<int> a, py::array_t<int> b, py::array_t<int> c) {
    auto a_buf = a.unchecked<1>();
    auto b_buf = b.unchecked<1>();
    auto c_buf = c.mutable_unchecked<1>();

    int size = a_buf.size();
    std::vector<int> host_a(size);
    std::vector<int> host_b(size);
    std::vector<int> host_c(size);

    // Copy input arrays to host vectors
    for (int i = 0; i < size; i++) {
        host_a[i] = a_buf(i);
        host_b[i] = b_buf(i);
    }

    // Call CUDA function
    cuda_add(host_c.data(), host_a.data(), host_b.data(), size);

    // Copy result back to output array
    for (int i = 0; i < size; i++) {
        c_buf(i) = host_c[i];
    }
}

PYBIND11_MODULE(pbcuda, m) {
    m.def("cuda_add", &py_cuda_add, "A function that adds two arrays using CUDA");
}
