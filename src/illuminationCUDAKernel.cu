#include "gpu.cuh"
#include <iostream>
#include <cmath>

__device__ double KM_AU = 149597870.700;
__device__ float SUN_DISC_RADIUS_DEG = 0.53 / 2;


__device__ void spherical2cartesian(float range, float lat, float lon, float& x, float& y, float& z) {
    lat *= M_PI / 180;
    lon *= M_PI / 180;
    x = range * cos(lat) * cos(lon);
    y = range * cos(lat) * sin(lon);
    z = range * sin(lat);
}

__device__ void cartesian2spherical(float x, float y, float z, float& range, float& lat, float& lon) {

    range = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    lat = asin(z / range);
    
    if (abs(x) < 1e-7 && abs(y) < 1e-7) {
        lon = 0;
    } else {
        lon = acos(x / sqrt(pow(x,2) + pow(y,2)));
        if (y < 0) lon *= -1;
        lon += M_PI;
    }

    lat *= 180 / M_PI;
    lon *= 180 / M_PI;

}


__device__ void calc_elev_azim(float sun_lat, float sun_lon, float sun_range, float lat, float lon, float& elev, float& azim) {

    // Convert range to meters
    sun_range *= KM_AU;

    // Transform sun coords to local coords
    float x, y, z, xl, yl, zl, dx, dy, dz;
    spherical2cartesian(sun_range, sun_lat, sun_lon, x, y, z);
    spherical2cartesian(1737400, lat, lon, xl, yl, zl);
    x *= 1000;
    y *= 1000;
    z *= 1000;
    dx = x-xl;
    dy = y-yl;
    dz = z-zl;

    // Convert lat longs to radians
    lat *= M_PI / 180;
    lon *= M_PI / 180;

    float cos_th = cos(lat);
    float sin_th = sin(lat);
    float cos_phi = cos(lon);
    float sin_phi = sin(lon);

    float t, xs, ys, zs;
    t = cos_phi * dx + sin_phi * dy;
    xs = -sin_phi * dx + cos_phi * dy;
    ys = -(-sin_th * t + cos_th * dz);
    zs = cos_th * t + sin_th * dz;

    float range_s;
    cartesian2spherical(xs, ys, zs, range_s, elev, azim);

}


__global__ void illuminate_k(float *elev_db, float *eph, float *grid, float *illumin, int M, int N, int T) {
    
    // Get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // index for this thread
    if (i >= N * N) return;

    // Get specific ephemeris values and lat longs for this thread
    float lat, lon;
    lat = grid[2*i];
    lon = grid[2*i+1];

    // Loop through ephemeris data and compute illumination fraction for each
    float sun_lat, sun_lon, sun_range, elev, azim, azim_low, azim_high;
    float elev_low, elev_high, horizon_elev, sun_elev_low, sun_elev_high, h, chord_area_degsq;
    int low_ind, high_ind;
    float illumin_frac = 0;
    float sun_rad_degsq = pow(SUN_DISC_RADIUS_DEG, 2);
    float sun_area_degsq = M_PI * sun_rad_degsq;
    for (int t=0; t < T; t++){
    // for (int t = 722; t < 723; t++) {
        
        // Current ephemeris values
        sun_lat = eph[3*t];
        sun_lon = eph[3*t+1];
        sun_range = eph[3*t+2];

        // Compute elevation and azimuth of sun relative to this lat and long
        calc_elev_azim(sun_lat, sun_lon, sun_range, lat, lon, elev, azim);

        // Compare to horizon for this azimuth
        azim_low = floor(azim);
        azim_high = ceil(azim);
        if (azim_low < 0) azim_low += 360;
        if (azim_high < 0) azim_high += 360;
        if (azim_low > 359) azim_low -= 360;
        if (azim_high > 359) azim_high -= 360;

        low_ind = M * i + int(azim_low);
        high_ind = M * i + int(azim_high);
        elev_low = elev_db[low_ind];
        elev_high = elev_db[high_ind];
        horizon_elev = 0.5 * (elev_low + elev_high);

        sun_elev_low = elev - SUN_DISC_RADIUS_DEG;
        sun_elev_high = elev + SUN_DISC_RADIUS_DEG;

        // height to use in calculating chordal area
        if (sun_elev_low < horizon_elev && horizon_elev < elev) {
            // h represents height of horizon above lowest point on solar disk
            // chordal area will be dark portion
            h = horizon_elev - sun_elev_low;
        } else if (sun_elev_low < horizon_elev && horizon_elev > elev && horizon_elev < sun_elev_high) {
            // h represents height of highest point on solar disk above horizon
            // chordal area will be lit portion
            h = sun_elev_high - horizon_elev;
        } else {
            h = 0.0;
        }

        // illumination fraction
        if (horizon_elev < sun_elev_low) {
            // all lit
            chord_area_degsq = sun_area_degsq;
            illumin_frac += 1.0;
        } else if (horizon_elev > sun_elev_high) {
            // all dark
            chord_area_degsq = 0;
            illumin_frac += 0.0;
        } else if (sun_elev_low < horizon_elev && horizon_elev < elev) {
            // area of lower disk (dark portion)
            chord_area_degsq = sun_rad_degsq * acos(1 - (h / SUN_DISC_RADIUS_DEG)) - (SUN_DISC_RADIUS_DEG - h) * sqrt(sun_rad_degsq - pow(SUN_DISC_RADIUS_DEG - h, 2));
            illumin_frac += 1.0 - (chord_area_degsq / sun_area_degsq);
        } else {
            // area of upper disk (lit portion)
            chord_area_degsq = sun_rad_degsq * acos(1 - (h / SUN_DISC_RADIUS_DEG)) - (SUN_DISC_RADIUS_DEG - h) * sqrt(sun_rad_degsq - pow(SUN_DISC_RADIUS_DEG - h, 2));
            illumin_frac += chord_area_degsq / sun_area_degsq;
        }

    }

    // Store output
    illumin[i] = illumin_frac / T;
    // illumin[i] = elev;

}

void illuminationCUDAKernel(float *elev_db, float *eph, float *grid, float *illumin, int M, int N, int T, cudaStream_t stream) {

    // Create shared arrays
    float *d_elev_db, *d_eph, *d_grid, *d_illumin;
    cudaMalloc(&d_elev_db, M * N * N * sizeof(float));
    cudaMalloc(&d_eph, T * 3 * sizeof(float));
    cudaMalloc(&d_grid, N * N * 2 * sizeof(float));
    cudaMalloc(&d_illumin, N * N * sizeof(float));

    // Copy data over to shared arrays
    cudaMemcpy(d_elev_db, elev_db, M * N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eph, eph, T * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid, N * N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_illumin, illumin, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the kernel
    illuminate_k<<<GET_BLOCKS(N*N), CUDA_NUM_THREADS, 0, stream>>>(d_elev_db, d_eph, d_grid, d_illumin, M, N, T);

    // Read the results back into the ratios array
    cudaMemcpy(illumin, d_illumin, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Error handling
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err){
        std::cout << "CUDA kernel failed with error: " << cudaGetErrorString(err) << std::endl;
    }

    // Clear memory
    cudaFree(d_elev_db);
    cudaFree(d_eph);
    cudaFree(d_grid);
    cudaFree(d_illumin);

    d_elev_db=NULL;
    d_eph=NULL;
    d_grid=NULL;
    d_illumin=NULL;

}