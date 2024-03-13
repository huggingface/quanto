#include <torch/extension.h>

torch::Tensor udqmm(torch::Tensor &input, torch::Tensor &other, torch::Tensor& other_scale, int bits);
