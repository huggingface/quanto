import torch


@torch.library.impl("quanto_py::udqmm", "default")
def udqmm(input: torch.Tensor, weights: torch.Tensor, scale: torch.Tensor, bits: int):
    unpacked_weights = torch.ops.quanto.unpack(weights, bits)
    scaled_weights = unpacked_weights * scale

    # TODO: future: pass `orig_shape`
    if scaled_weights.size(0) != input.size(1):
        last_dim = int(scaled_weights.numel() / input.size(1))
        original_shape = (input.size(1), last_dim)
        scaled_weights = scaled_weights.reshape(original_shape)

    return torch.ops.aten.mm(input, scaled_weights)
