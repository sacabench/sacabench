/*******************************************************************************
 * Copyright (C) 2019 Hermann Foot <hermann.foot@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include<cuda.h>
#include "cuda_wrapper_interface.hpp"
#include "cuda_util.cuh"

#include "check_for_gpu_interface.hpp"
#include <iostream>

int cuda_GPU_available(){
    int deviceCount;
    int deviceNumber;
    size_t free_bytes;
    size_t total_bytes;
    cudaError_t e1 = cudaGetDeviceCount(&deviceCount);
    cudaError_t e2 = cudaGetDevice(&deviceNumber);   
    cudaError_t e3=cudaMemGetInfo(&free_bytes, &total_bytes);
    bool available = (e1 == cudaSuccess) && (e2 == cudaSuccess) && (e3 == cudaSuccess);
    return available;
}