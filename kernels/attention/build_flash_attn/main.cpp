#include <torch/extension.h>
torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("flash_attn", torch::wrap_pybind_function(flash_attn), "flash_attn");
}