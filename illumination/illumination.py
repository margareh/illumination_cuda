# -*- coding: utf-8 -*-
"""
Performs illumination modeling over an ephemeris dataset and terrain model
"""

import numpy as np
import torch
from illuminationCUDA import illuminationCUDA

def illuminate_cuda(eph, grid, psr_threshold=0.001, max_range=0.2, res=1, min_elev=-89, elev_delta=0.25):
    """
    Python wrapper for illumination model using CUDA

    Inputs: ephemeris (elevation, azimuth, range), grid (lat long for surface)
    Outputs: illumination fraction map, PSR map
    """

    N = grid.shape[0]
    T = eph.shape[0]

    eph = torch.tensor(eph, dtype=torch.float32).flatten() # should be T * 3 (lat, lon, range)
    hmap = torch.tensor(grid[...,-1], dtype=torch.float32).flatten() # should be N * N (height)
    grid = torch.tensor(grid[...,0:2], dtype=torch.float32).flatten() # should be N * N * 2 (lat, lon) for each point on surface
    
    # print(eph[0:10])
    # print(grid[0:10])

    # output
    illumin = torch.zeros_like(hmap)
    # print(illumin.shape)

    # print(eph.shape)
    # print(grid.shape)
    # print(hmap.shape)
    # print(illumin.shape)
    # print(N)
    # print(T)

    # call to CUDA kernel wrapper
    # print("Call to kernel")
    illuminationCUDA(eph, grid, hmap, illumin, N, T, max_range, res, min_elev, elev_delta)

    # reshape output
    # result should be N x N
    illumin = illumin.unflatten(-1, (N, N)).cpu().numpy()

    # average illumination fraction over all time points to get average
    psr = (illumin < psr_threshold)

    return illumin, psr