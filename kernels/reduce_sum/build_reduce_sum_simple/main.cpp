#include <torch/extension.h>
torch::Tensor reduce_sum_simple(const torch::Tensor &t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("reduce_sum_simple", torch::wrap_pybind_function(reduce_sum_simple), "reduce_sum_simple");
}