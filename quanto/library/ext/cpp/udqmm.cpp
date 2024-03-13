#include "udqmm.h"
#include "unpack.h"

#include <iostream>
#include <torch/extension.h>

using namespace std;

torch::Tensor udqmm(torch::Tensor &input, torch::Tensor &weights, torch::Tensor& scale, int bits) {
    torch::Tensor unpacked_weights = unpack(weights, bits);
    torch::Tensor dq_output = unpacked_weights * scale;

    // Optionally ungroup 
    // TODO: deal with the case where group_axis != 0
    if (dq_output.size(0) != input.size(1)) {
        int64_t last_dim = torch::numel(dq_output) / input.size(1);
        std::vector<int64_t> shape = { input.size(1), last_dim };   

        torch::Tensor ungrouped_output = torch::reshape(dq_output, shape);
        return torch::mm(input, ungrouped_output);
    };

    return torch::mm(input, dq_output);
}