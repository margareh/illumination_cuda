#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <iostream>
#include "illuminationCUDAKernel.cuh"

void illuminationCUDA(at::Tensor eph, at::Tensor grid, at::Tensor hmap,
                  at::Tensor illumin, int N, int T, float max_range, float res, float min_elev, float elev_delta) {
    // call to CUDA kernel
    illuminationCUDAKernel(eph.data_ptr<float>(), grid.data_ptr<float>(), hmap.data_ptr<float>(), illumin.data_ptr<float>(),
                       N, T, max_range, res, min_elev, elev_delta, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){        
    m.def("illuminationCUDA", &illuminationCUDA, "Perform horizon-based illumination using CUDA");
}