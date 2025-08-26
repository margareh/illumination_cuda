# -*- coding: utf-8 -*-
"""
Performs illumination modeling over an ephemeris dataset and terrain model
"""

import numpy as np
import torch
from illuminationCUDA import illuminationCUDA

def illuminate_cuda(elev_db, eph, grid, psr_threshold=0.001):
    """
    Python wrapper for illumination model using CUDA

    Inputs: horizon elevation maps, ephemeris (elevation, azimuth, range), grid (lat long for surface)
    Outputs: illumination fraction map
    """

    M = elev_db.shape[0]
    N = elev_db.shape[1]
    T = eph.shape[0]

    elev_db = torch.tensor(elev_db).transpose(0,2).transpose(0,1).flatten().float() # should be M * N * N
    eph = torch.tensor(eph).flatten().float() # should be T * 3 (lat, lon, range)
    grid = torch.tensor(grid).flatten().float() # should be N * N * 2 (lat, lon) for each point on surface

    # print(eph[0:10])
    # print(grid[0:10])

    # output
    illumin = torch.zeros((N, N)).flatten().float()
    # print(illumin.shape)

    # print(M) # 360
    # print(N) # 100
    # print(T) # 7671
    # print(elev_db.shape) # 3600000 = 360 * 100 * 100
    # print(eph.shape) # 23013 = 3 * 7671
    # print(grid.shape) # 20000 = 100 * 100 * 2

    # call to CUDA kernel wrapper
    illuminationCUDA(elev_db, eph, grid, illumin, M, N, T)

    # reshape output
    # result should be N x N
    illumin = illumin.unflatten(-1, (N, N)).cpu().numpy()

    # average illumination fraction over all time points to get average
    psr = (illumin < psr_threshold)

    return illumin, psr