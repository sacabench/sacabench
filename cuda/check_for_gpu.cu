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


bool cuda_device_available() {
    int deviceCount;
    int deviceNumber;
    size_t free_bytes;
    size_t total_bytes;
    cudaError_t e1 = cudaGetDeviceCount(&deviceCount);
    cudaError_t e2 = cudaGetDevice(&deviceNumber);   
    cudaError_t e3=cudaMemGetInfo(&free_bytes, &total_bytes);
    
    bool available = (e1 == cudaSuccess) && (e2 == cudaSuccess) && (e3 == cudaSuccess);

    if(!available) {
        std::cout<<"[No suitable GPU detected]"<<std::endl;
    }
    return available;
}

bool cuda_version_sufficient() {
    int cuda_version;
    cudaDriverGetVersion(&cuda_version);
    bool sufficient = cuda_version>=10000;
    if(!sufficient)
    {
        std::cout<<"[CUDA Version ("<<cuda_version<<") not sufficient]"<<std::endl;
    }
    return (sufficient);
}

//TODO: check compute comp
int cuda_GPU_available(){
    return (cuda_device_available()&&cuda_version_sufficient());
}