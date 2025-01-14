#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.cuh"

constexpr int TILE_SIZE = 64;

template <typename T>
__global__ void matmul_tiled_out(const T *m1, const T *m2, T *out,
                                 int m, int k, int n) {
  __shared__ T _m1[TILE_SIZE][TILE_SIZE], _m2[TILE_SIZE][TILE_SIZE];

  int tile_i = blockIdx.x * blockDim.x + threadIdx.x;
  int tile_j = blockIdx.y * blockDim.y + threadIdx.y;

  int local_i = threadIdx.x;
  int local_j = threadIdx.y;

  T sum = 0;
  for (int k_step = 0; k_step < cdiv_d(k, TILE_SIZE); k_step++) {
    // _m1[local_i][local_j] = m1[tile_i * TILE_SIZE + local_i][k_step * TILE_SIZE + local_j]
    // _m2[local_i][local_j] = m2[k_step * TILE_SIZE + local_i][tile_j * TILE_SIZE + local_j]
    _m1[local_i][local_j] = (tile_i * TILE_SIZE + local_i < m && k_step * TILE_SIZE + local_j < k)
                              ? m1[(tile_i * TILE_SIZE + local_i) * k + (k_step * TILE_SIZE + local_j)]
                              : 0;
    _m2[local_i][local_j] = ((k_step * TILE_SIZE + local_i) < k && tile_j * TILE_SIZE + local_j < n)
                              ? m2[(k_step * TILE_SIZE + local_i) * n + (tile_j * TILE_SIZE + local_j)]
                              : 0;
    __syncthreads();
    #pragma unroll
    for (int local_k = 0; local_k < TILE_SIZE; local_k++) {
      sum += _m1[local_i][local_k] * _m2[local_k][local_j];
    }
    __syncthreads();
  }

  if (tile_i * TILE_SIZE + local_i < m && tile_j * TILE_SIZE + local_j < n) {
    out[(tile_i * TILE_SIZE + local_i) * n + (tile_j * TILE_SIZE + local_j)] = sum;
  }
}

torch::Tensor matmul_tiled(const torch::Tensor &m1, const torch::Tensor &m2) {
  int m = m1.size(0);
  int k = m1.size(1);
  int n = m2.size(1);

  TORCH_CHECK(k == m2.size(0), "matmul sizes don't match");

  auto out = torch::empty({m, n}, m1.options());

  dim3 tShape(TILE_SIZE, TILE_SIZE);
  dim3 bShape(cdiv(m, tShape.x), cdiv(n, tShape.y));
  matmul_tiled_out<float><<<bShape, tShape>>>(m1.data_ptr<float>(),
                                       m2.data_ptr<float>(),
                                       out.data_ptr<float>(), m, k, n);

  return out;
}
