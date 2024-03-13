import pytest
import torch
from helpers import random_tensor

from quanto.tensor.core import group, ungroup
from quanto.tensor.packed import pack_weights, unpack_weights


@pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
@pytest.mark.parametrize("output_features", [48, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dqmm(input_shape, output_features, dtype, device):
    input = random_tensor(input_shape, dtype=dtype).to(device)
    other = torch.randint(-127, 127, (input_shape[-1], output_features), dtype=torch.int8).to(device)
    other_scale = random_tensor((output_features,), dtype=dtype).to(device)
    output = torch.ops.quanto.dqmm(input, other, other_scale)
    expected = torch.ops.aten.mm(input, other * other_scale)
    assert torch.equal(expected, output)


@pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
@pytest.mark.parametrize("output_features", [48, 64])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_packed_udqmm(input_shape, output_features, dtype, device, bits):
    input = random_tensor(input_shape, dtype=dtype).to(device)

    qmax = 2**bits
    a = torch.randint(0, qmax, (input_shape[-1], output_features), dtype=torch.uint8).to(device)

    packed_a = pack_weights(a, bits)
    unpacked_weights = unpack_weights(packed_a, bits)

    other_scale = random_tensor((output_features,), dtype=dtype).to(device)
    output = torch.ops.quanto.udqmm(input, packed_a, other_scale, bits)
    expected = torch.ops.aten.mm(input, unpacked_weights * other_scale)

    assert torch.equal(expected, output)


@pytest.mark.parametrize("input_shape", [[10, 32], [32, 32]])
@pytest.mark.parametrize("output_features", [48, 64])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_grouped_udqmm(input_shape, output_features, dtype, device, bits):
    input = random_tensor(input_shape, dtype=dtype).to(device)
    qmax = 2**bits

    weights = torch.randint(0, qmax, (input_shape[-1], output_features), dtype=torch.uint8).to(device)
    grouped_weights = group(weights, axis=0, group_size=int(input_shape[-1] / 4))
    # grouped_weights = weights
    output_shape = grouped_weights.shape

    packed_weights = pack_weights(grouped_weights, bits)
    unpacked_weights = unpack_weights(packed_weights, bits)

    other_scale = random_tensor((1, output_shape[1]), dtype=dtype).to(device)

    output = torch.ops.quanto.udqmm(input, packed_weights, other_scale, bits)
    expected = torch.ops.aten.mm(input, ungroup(unpacked_weights * other_scale, axis=0, orig_shape=weights.shape))

    assert torch.equal(expected, output)
