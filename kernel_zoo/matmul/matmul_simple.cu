#include "common.cuh"

constexpr int THREAD_DIM = 16;

template<typename T>
__global__ void matmul_simple_out(const T *m1, const T *m2, T *out,
                                  int m1_r, int m1_c, int m2_c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m1_r && j < m2_c) {
    T sum = 0;
    for (int k = 0; k < m1_c; k++) {
      sum += m1[i * m1_c + k] * m2[k * m2_c + j];
    }
    out[i * m2_c + j] = sum;
  }
}

torch::Tensor matmul_simple(const torch::Tensor &m1, const torch::Tensor &m2) {
  const auto m1_r = m1.size(0);
  const auto m1_c = m1.size(1);
  const auto m2_c = m2.size(1);

  TORCH_CHECK(m1_c == m2.size(0), "matmul sizes don't match");

  auto out = torch::empty({m1_r, m2_c}, m1.options());
  dim3 tShape(THREAD_DIM, THREAD_DIM);
  dim3 bShape(cdiv(m1_r, tShape.x), cdiv(m2_c, tShape.y));
  matmul_simple_out<float><<<bShape, tShape>>>(
      m1.data_ptr<float>(), m2.data_ptr<float>(), out.data_ptr<float>(), m1_r,
      m1_c, m2_c);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
