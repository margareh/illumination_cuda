#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "illuminationCUDAKernel.cuh"

void illuminationCUDA(at::Tensor elev_db, at::Tensor eph, at::Tensor grid,
                  at::Tensor illumin, int M, int N, int T) {
    // call to CUDA kernel
    illuminationCUDAKernel(elev_db.data_ptr<float>(), eph.data_ptr<float>(), grid.data_ptr<float>(), illumin.data_ptr<float>(),
                       M, N, T, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){        
    m.def("illuminationCUDA", &illuminationCUDA, "Perform horizon-based illumination using CUDA");
}