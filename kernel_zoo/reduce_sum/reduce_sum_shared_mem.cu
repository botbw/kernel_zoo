#include "common.cuh"

constexpr int BLOCK_SIZE = 1024;

template <typename T>
__global__ void reduce_sum_shared_mem_out(T *data, T *out) {
  __shared__ T shared_mem[BLOCK_SIZE];
  int tid = threadIdx.x;
  int aid = tid;

  shared_mem[aid] = data[aid] + data[aid + blockDim.x];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    if (aid < stride) {
        shared_mem[aid] += shared_mem[aid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out = shared_mem[0];
  }
}

torch::Tensor reduce_sum_shared_mem(const torch::Tensor &t) {
  auto len = t.numel();
  TORCH_CHECK(len <= BLOCK_SIZE, "tensor too big")

  auto out = torch::zeros({}, t.options());

  reduce_sum_shared_mem_out<float>
      <<<1, cdiv(len, 2)>>>(t.data_ptr<float>(), out.data_ptr<float>());

  return out;
}