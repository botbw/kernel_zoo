#ifndef COMMON_CUH
#define COMMON_CUH

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      exit(code);
    }
  }
}

unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

__device__ unsigned int cdiv_d(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

size_t getSharedMemorySize() {
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);
  return deviceProp.sharedMemPerBlock;
}

#endif  // COMMON_CUH
