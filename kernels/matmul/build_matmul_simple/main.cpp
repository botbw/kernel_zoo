#include <torch/extension.h>
torch::Tensor matmul_simple(const torch::Tensor &m1, const torch::Tensor &m2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul_simple", torch::wrap_pybind_function(matmul_simple), "matmul_simple");
}