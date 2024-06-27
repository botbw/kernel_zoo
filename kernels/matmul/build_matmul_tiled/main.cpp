#include <torch/extension.h>
torch::Tensor matmul_tiled(const torch::Tensor &m1, const torch::Tensor &m2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul_tiled", torch::wrap_pybind_function(matmul_tiled), "matmul_tiled");
}