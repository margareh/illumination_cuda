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

__device__ float raytrace(float *hmap, float *start_point, float *end_point, float res, int H, int W) {

	// Adapted from http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html

	/***** raytrace through ray (this is sloppy repetitive code but I'm lazy) *****/

	// Pull ending values for specific ray
	int x_max = (int)end_point[0];
	int y_max = (int)end_point[1];
	float z_max = end_point[2];

	// Pull pose values
	int pose_x = start_point[0];
	int pose_y = start_point[1];
	float pose_z = start_point[2];
	int z_start = int(floor(pose_z));
	
	// Setup
	int dx = abs(x_max - pose_x);
	int dy = abs(y_max - pose_y);
	float dz = fabs(z_max - pose_z);

	double dt_dx = 1.0 / dx;
	double dt_dy = 1.0 / dy;
	double dt_dz = 1.0 / dz;

	int n = 1;
	int x_inc, y_inc, z_inc;
	double t_next_x, t_next_y, t_next_z;

	// define initial variables based on cases
	// x
	if (dx == 0){
		x_inc = 0;
		t_next_x = 1000.0;
	} else if (x_max > pose_x) {
		x_inc = 1;
		n += x_max - pose_x;
		t_next_x = dt_dx;
	} else {
		x_inc = -1;
		n += pose_x - x_max;
		t_next_x = dt_dx;
	}

	// y
	if (dy == 0){
		y_inc = 0;
		t_next_y = 1000.0;
	} else if (y_max > pose_y) {
		y_inc = 1;
		n += y_max - pose_y;
		t_next_y = dt_dy;
	} else {
		y_inc = -1;
		n += pose_y - y_max;
		t_next_y = dt_dy;
	}

	// z
	if (dz == 0){
		z_inc = 0;
		t_next_z = 1000.0;
	} else if (z_max > z_start) {
		z_inc = 1;
		n += int(floor(z_max)) - z_start;
		t_next_z = (z_start + 1 - pose_z) * dt_dz;
	} else {
		z_inc = -1;
		n += z_start - int(floor(z_max));
		t_next_z = (pose_z - z_start) * dt_dz;
	}

	// loop through ray and update mask as necessary
	float z_curr = pose_z;
	int x_grid = pose_x;
	int y_grid = pose_y;
	int z_grid = z_start;
	float hmap_z, range;
	float x_out, y_out;
	double t = 0;
	float pose_x_m = res * pose_x;
	float pose_y_m = res * pose_y;

	for (; n > 0; --n){

		// check if current grid index is valid (return if not)
		if (x_grid >= W || x_grid < 0 || y_grid >= H || y_grid < 0) return -1.0;

		// Get current x, y, and z given t
		z_curr = pose_z + t * z_inc * dz;
		
		// check if current position is above ground (update scan and return if not)
		hmap_z = hmap[y_grid * W + x_grid];
		if (hmap_z >= z_curr && abs(t) > 0) {
			x_out = res * x_grid;
			y_out = res * y_grid;
			range = sqrt((x_out - pose_x_m) * (x_out - pose_x_m) + (y_out - pose_y_m) * (y_out - pose_y_m) + (z_curr - pose_z) * (z_curr - pose_z));
			return range;
		}

		// take a step along the ray
		if (t_next_x <= t_next_y && t_next_x <= t_next_z) {

			// x is min
			x_grid += x_inc;
			t = t_next_x;
			t_next_x += dt_dx;

		} else if(t_next_y <= t_next_x && t_next_y <= t_next_z) {

			// y is min
			y_grid += y_inc;
			t = t_next_y;
			t_next_y += dt_dy;

		} else {

			// z is min
			z_grid += z_inc;
			t = t_next_z;
			t_next_z += dt_dz;

		}
	}
	return -1.0;

}

__device__ void get_horizon_elev(int i, float azim, int H, int W, float max_range, float res, float min_elev, float elev_delta, float* hmap, float& horizon_elev){

    // Get indices
	int y_ind = int(floor(i / W)); // Y index
	int x_ind = i - W*y_ind; // X index

	// Get grid point indices for heightmap grid cell
	// These are offset by the boundary area
	int b = int(floor(max_range * 1000 / res)); // this will be max range in pixels (grid indices)
	int kb = i + W*b + 2*b*y_ind + 2*b*b + b;
	int xb_ind = x_ind+b; // Grid point index, x axis
	int yb_ind = y_ind+b; // Grid point index, y axis

	// Define start point and azimuthal angle
	float curr_height = hmap[kb]; // k*A+j but need to adjust k to account for boundary points
	float start_point[3] = {xb_ind, yb_ind, curr_height + 0.01};
	float curr_azim = azim * (M_PI / 180); // converted to rad

	// Max range in meters
	float max_range_m = max_range * 1000;

	// Start with min elevation and loop until we find no terrain
	float range = 0;
	float curr_elev = min_elev * (M_PI / 180); // converted to rad
	elev_delta *= (M_PI / 180); // converted to rad
	int iter=0;
	while (max_range_m - range > 0.00001) {

		// Increment elevation
		// we're technically skipping the first but that's fine
		// min_elev used later to mark points without intersections
		curr_elev += elev_delta;
	
		// Define end point based on current grid cell, azimuth, elevation
		float cos_elev = cos(curr_elev);
		float end_point[3] = { cos(curr_azim) * cos_elev / res, sin(curr_azim) * cos_elev / res, sin(curr_elev) };
		for (int c=0; c<3; c++){
			end_point[c] *= max_range_m;
			end_point[c] += start_point[c];
		}

		// Call raytrace
		range = raytrace(hmap, start_point, end_point, res, H+2*b, W+2*b);
		if (range < 0) {
			// error in raytracing (out of bounds or didn't find intersection)
			// elevation for this point will be set to minimum value
			// also setting range to max range to break out of loop
			range = max_range_m;
		}
		iter++;

	}

	// Store the results
	// need to subtract off change in elevation for last one that intersects with the terrain
	horizon_elev = curr_elev - elev_delta; // dimension order: y, x, azim


}


__global__ void illuminate_k(float *eph, float *grid, float *hmap, float *illumin, int N, int T, float max_range, float res, float min_elev, float elev_delta) {
    
    // Get indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // index for this thread
    if (i >= N * N) return;

    // Get specific lat longs for this thread
    float lat, lon;
    lat = grid[2*i];
    lon = grid[2*i+1];
    
    // Loop through ephemeris data and compute illumination fraction for each
    float sun_lat, sun_lon, sun_range, elev, azim;
    float horizon_elev, sun_elev_low, sun_elev_high, h, chord_area_degsq;
    float illumin_frac = 0;
    float sun_rad_degsq = pow(SUN_DISC_RADIUS_DEG, 2);
    float sun_area_degsq = M_PI * sun_rad_degsq;
    for (int t=0; t < T; t++){
        
        // Current ephemeris values
        sun_lat = eph[3*t];
        sun_lon = eph[3*t+1];
        sun_range = eph[3*t+2];

        // Compute elevation and azimuth of sun relative to this lat and long
        calc_elev_azim(sun_lat, sun_lon, sun_range, lat, lon, elev, azim);

        // Calculate horizon elevation for this azimuth
        get_horizon_elev(i, azim, N, N, max_range, res, min_elev, elev_delta, hmap, horizon_elev);

        // // Compare to horizon for this azimuth
        // azim_low = floor(azim);
        // azim_high = ceil(azim);
        // if (azim_low < 0) azim_low += 360;
        // if (azim_high < 0) azim_high += 360;
        // if (azim_low > 359) azim_low -= 360;
        // if (azim_high > 359) azim_high -= 360;

        // low_ind = M * i + int(azim_low);
        // high_ind = M * i + int(azim_high);
        // elev_low = elev_db[low_ind];
        // elev_high = elev_db[high_ind];
        // horizon_elev = 0.5 * (elev_low + elev_high);

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

void illuminationCUDAKernel(float *eph, float *grid, float *hmap, float *illumin, int N, int T, float max_range, float res, float min_elev, float elev_delta, cudaStream_t stream) {

    // Create shared arrays
    float *d_eph, *d_grid, *d_hmap, *d_illumin;
    cudaMalloc(&d_eph, T * 3 * sizeof(float));
    cudaMalloc(&d_grid, N * N * 2 * sizeof(float));
    cudaMalloc(&d_hmap, N * N * sizeof(float));
    cudaMalloc(&d_illumin, N * N * sizeof(float));

    // Copy data over to shared arrays
	// std::cout << "memcpy" << std::endl;
    cudaMemcpy(d_eph, eph, T * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, grid, N * N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hmap, hmap, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the kernel
	// std::cout << "CUDA KERNEL" << std::endl;
    illuminate_k<<<GET_BLOCKS(N*N), CUDA_NUM_THREADS, 0, stream>>>(d_eph, d_grid, d_hmap, d_illumin, N, T, max_range, res, min_elev, elev_delta);

    // Read the results back into the ratios array
    cudaMemcpy(illumin, d_illumin, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Error handling
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err){
        std::cout << "CUDA kernel failed with error: " << cudaGetErrorString(err) << std::endl;
    }

    // Clear memory
    cudaFree(d_eph);
    cudaFree(d_grid);
	cudaFree(d_hmap);
    cudaFree(d_illumin);

    d_eph=NULL;
    d_grid=NULL;
	d_hmap=NULL;
    d_illumin=NULL;

}