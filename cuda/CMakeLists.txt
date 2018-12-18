add_executable(cuda_prefix_sum
    Prefix_Sum_Benchmark.cu
)

add_library(cuda_wrapper
    cuda_wrapper.cu
)

target_include_directories(cuda_wrapper INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

#target_compile_features(cuda_wrapper PUBLIC cxx_std_14)
