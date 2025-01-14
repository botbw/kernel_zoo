#include "common.h"

template<typename T>
void matmul_cpu_out(const T *m1, const T *m2, T *out, int m1_r,
                    int m1_c, int m2_c) {
  for (int i = 0; i < m1_r; i++) {
    for (int j = 0; j < m2_c; j++) {
      T sum = 0.0f;
      for (int k = 0; k < m1_c; k++) {
        sum += m1[i * m1_c + k] * m2[k * m2_c + j];
      }
      out[i * m2_c + j] = sum;
    }
  }
}

torch::Tensor matmul_cpu(const torch::Tensor &m1, const torch::Tensor &m2) {
  const auto m1_r = m1.size(0);
  const auto m1_c = m1.size(1);
  const auto m2_c = m2.size(1);

  TORCH_CHECK(m1_c == m2.size(0), "matmul sizes don't match");

  auto out = torch::empty({m1_r, m2_c}, m1.options());
  matmul_cpu_out<float>(m1.data_ptr<float>(), m2.data_ptr<float>(),
                 out.data_ptr<float>(), m1_r, m1_c, m2_c);

  return out;
}