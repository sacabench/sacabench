add_library(cuda_wrapper
    cuda_wrapper.cu
    cuda_util.cu
    prefix_doubler.cu
    check_for_gpu.cu
)

target_include_directories(cuda_wrapper INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(cuda_wrapper PRIVATE cxx_std_14)
set_target_properties(cuda_wrapper
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
