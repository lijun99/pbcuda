#include "cuda_module.h"
#include <iostream>

template <typename T>
__global__ void addKernel(T* c, const T* a, const T* b, const int size) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void cuda_add(T* c, const T* a, const T* b, int size) {
    T* d_a, * d_b, * d_c;
    size_t bytes = size * sizeof(T);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, size);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error after cudaSetDevice: " << cudaGetErrorString(error) << std::endl;
    }
    
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// **Explicit Instantiations: required for python module**
template void cuda_add<float>(float* c, const float* a, const float* b, int size);
template void cuda_add<double>(double* c, const double* a, const double* b, int size);
template void cuda_add<int>(int* c, const int* a, const int* b, int size);
template void cuda_add<int64_t>(int64_t* c, const int64_t* a, const int64_t* b, int size);