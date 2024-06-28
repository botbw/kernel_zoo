#include <torch/extension.h>
torch::Tensor reduce_sum_shared_mem(const torch::Tensor &t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("reduce_sum_shared_mem", torch::wrap_pybind_function(reduce_sum_shared_mem), "reduce_sum_shared_mem");
}