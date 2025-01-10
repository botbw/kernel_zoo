#include "common.cuh"

constexpr int TILE_SIZE = 32;

template <typename T>
__global__ void matmul_tiled_out(const T *m1, const T *m2, T *out,
                                 int m1_r, int m1_c, int m2_c) {
  __shared__ T _m1[TILE_SIZE][TILE_SIZE], _m2[TILE_SIZE][TILE_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int tile_i = threadIdx.x;
  int tile_j = threadIdx.y;

  T sum = 0;
  for (int k_step = 0; k_step < cdiv_d(m1_c, TILE_SIZE); k_step++) {
    // _m1[tile_i][tile_j] = m1[i][k_step * TILE_SIZE + tile_j]
    // _m2[tile_i][tile_j] = m2[k_step * TILE_SIZE + tile_i][j]
    _m1[tile_i][tile_j] = (i < m1_r && k_step * TILE_SIZE + tile_j < m1_c)
                              ? m1[i * m1_c + k_step * TILE_SIZE + tile_j]
                              : 0;
    _m2[tile_i][tile_j] = ((k_step * TILE_SIZE + tile_i) < m1_c && j < m2_c)
                              ? m2[(k_step * TILE_SIZE + tile_i) * m2_c + j]
                              : 0;
    __syncthreads();
    #pragma unroll
    for (int tile_k = 0; tile_k < TILE_SIZE; tile_k++) {
      sum += _m1[tile_i][tile_k] * _m2[tile_k][tile_j];
    }
    __syncthreads();
  }

  if (i < m1_r && j < m2_c) {
    out[i * m2_c + j] = sum;
  }
}

torch::Tensor matmul_tiled(const torch::Tensor &m1, const torch::Tensor &m2) {
  int m1_r = m1.size(0);
  int m1_c = m1.size(1);
  int m2_c = m2.size(1);

  TORCH_CHECK(m1_c == m2.size(0), "matmul sizes don't match");

  auto out = torch::empty({m1_r, m2_c}, m1.options());

  dim3 tShape(TILE_SIZE, TILE_SIZE);
  dim3 bShape(cdiv(m1_r, tShape.x), cdiv(m2_c, tShape.y));
  matmul_tiled_out<float><<<bShape, tShape>>>(m1.data_ptr<float>(),
                                       m2.data_ptr<float>(),
                                       out.data_ptr<float>(), m1_r, m1_c, m2_c);

  return out;
}
