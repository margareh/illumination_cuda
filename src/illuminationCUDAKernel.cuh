void illuminationCUDAKernel(float *eph, float *grid, float *hmap, float *illumin, int N, int T, float max_range, float res, float min_elev, float elev_delta, cudaStream_t stream);
