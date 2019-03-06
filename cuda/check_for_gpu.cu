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


bool cuda_device_memory_available() {

    size_t free_bytes;
    size_t total_bytes;
 
    cudaError_t e =cudaMemGetInfo(&free_bytes, &total_bytes);
    
    bool available = (e == cudaSuccess);

    if(!available) {
        std::cerr<<"[Couldn't reach GPU memory]"<<std::endl;
    }
    return available;
}

bool cuda_version_sufficient() {
    int cuda_version;
    cudaDriverGetVersion(&cuda_version);
    bool sufficient = cuda_version>=10000;
    if(!sufficient)
    {
        std::cerr<<"[CUDA Version ("<<cuda_version<<") not sufficient]"<<std::endl;
    }
    return (sufficient);
}

bool cuda_sufficient_card_available()
{
  int number_devices;
  cudaError_t e = cudaGetDeviceCount(&number_devices);
  if(e==cudaSuccess && number_devices>0) {
    for (int i = 0; i < number_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if(prop.major>=3) {
            return true;
        }
    }
    std::cerr<<"[CUDA compute capability not sufficient]"<<std::endl;
    return false;
  }
  else {
    std::cerr<<"[Couldn't find suitable GPU]"<<std::endl;
    return false;
  }
}

int cuda_GPU_available(){
    return (cuda_sufficient_card_available()&&cuda_device_memory_available()&&cuda_version_sufficient());
}