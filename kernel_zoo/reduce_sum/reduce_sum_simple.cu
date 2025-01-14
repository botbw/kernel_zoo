#include "common.cuh"

template <typename T>
__global__ void reduce_sum_simple_out(T *data, T *out) {
  int tid = threadIdx.x;
  int aid = tid * 2;

  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if ((tid % stride) == 0) {
      data[aid] += data[aid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *out = data[0];
  }
}

torch::Tensor reduce_sum_simple(const torch::Tensor &t) {
  auto len = t.numel();
  auto out = torch::zeros({}, t.options());

  reduce_sum_simple_out<float><<<1, cdiv(len, 2)>>>(
      t.data_ptr<float>(), out.data_ptr<float>());

  return out;
}