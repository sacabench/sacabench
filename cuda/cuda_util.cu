#include<iostream>
#include<cstdint>
#include<cstddef>

#include "cuda_util.cuh"

#define map_single_error_code(e) case e: out << #e; break;

static void map_error_code(cudaError e, std::ostream& out) {
    switch (e) {
    map_single_error_code(cudaErrorMissingConfiguration)
    map_single_error_code(cudaErrorMemoryAllocation)
    map_single_error_code(cudaErrorInitializationError)
    map_single_error_code(cudaErrorLaunchFailure)
    map_single_error_code(cudaErrorPriorLaunchFailure)
    map_single_error_code(cudaErrorLaunchTimeout)
    map_single_error_code(cudaErrorLaunchOutOfResources)
    map_single_error_code(cudaErrorInvalidDeviceFunction)
    map_single_error_code(cudaErrorInvalidConfiguration)
    map_single_error_code(cudaErrorInvalidDevice)
    map_single_error_code(cudaErrorInvalidValue)
    map_single_error_code(cudaErrorInvalidPitchValue)
    map_single_error_code(cudaErrorInvalidSymbol)
    map_single_error_code(cudaErrorMapBufferObjectFailed)
    map_single_error_code(cudaErrorUnmapBufferObjectFailed)
    map_single_error_code(cudaErrorInvalidHostPointer)
    map_single_error_code(cudaErrorInvalidDevicePointer)
    map_single_error_code(cudaErrorInvalidTexture)
    map_single_error_code(cudaErrorInvalidTextureBinding)
    map_single_error_code(cudaErrorInvalidChannelDescriptor)
    map_single_error_code(cudaErrorInvalidMemcpyDirection)
    map_single_error_code(cudaErrorAddressOfConstant)
    map_single_error_code(cudaErrorTextureFetchFailed)
    map_single_error_code(cudaErrorTextureNotBound)
    map_single_error_code(cudaErrorSynchronizationError)
    map_single_error_code(cudaErrorInvalidFilterSetting)
    map_single_error_code(cudaErrorInvalidNormSetting)
    map_single_error_code(cudaErrorMixedDeviceExecution)
    map_single_error_code(cudaErrorCudartUnloading)
    map_single_error_code(cudaErrorUnknown)
    map_single_error_code(cudaErrorNotYetImplemented)
    map_single_error_code(cudaErrorMemoryValueTooLarge)
    map_single_error_code(cudaErrorInvalidResourceHandle)
    map_single_error_code(cudaErrorNotReady)
    map_single_error_code(cudaErrorInsufficientDriver)
    map_single_error_code(cudaErrorSetOnActiveProcess)
    map_single_error_code(cudaErrorInvalidSurface)
    map_single_error_code(cudaErrorNoDevice)
    map_single_error_code(cudaErrorECCUncorrectable)
    map_single_error_code(cudaErrorSharedObjectSymbolNotFound)
    map_single_error_code(cudaErrorSharedObjectInitFailed)
    map_single_error_code(cudaErrorUnsupportedLimit)
    map_single_error_code(cudaErrorDuplicateVariableName)
    map_single_error_code(cudaErrorDuplicateTextureName)
    map_single_error_code(cudaErrorDuplicateSurfaceName)
    map_single_error_code(cudaErrorDevicesUnavailable)
    map_single_error_code(cudaErrorInvalidKernelImage)
    map_single_error_code(cudaErrorNoKernelImageForDevice)
    map_single_error_code(cudaErrorIncompatibleDriverContext)
    map_single_error_code(cudaErrorPeerAccessAlreadyEnabled)
    map_single_error_code(cudaErrorPeerAccessNotEnabled)
    map_single_error_code(cudaErrorDeviceAlreadyInUse)
    map_single_error_code(cudaErrorProfilerDisabled)
    map_single_error_code(cudaErrorProfilerNotInitialized)
    map_single_error_code(cudaErrorProfilerAlreadyStarted)
    map_single_error_code(cudaErrorProfilerAlreadyStopped)
    map_single_error_code(cudaErrorStartupFailure)
    map_single_error_code(cudaErrorApiFailureBase)
    default:
        out << "unknown code " << int(e);
    }
}

void cuda_check_internal(char const* file, int line, cudaError v, char const* reason) {
    if (v != cudaSuccess) {
        std::cerr << "CUDA ERROR at " << file << ":" << line << ": ";
        map_error_code(v, std::cerr);
        if (std::string(reason) != "") {
            std::cerr << " (" << reason << ")" << std::endl;
        }
        std::abort();
    }
}
