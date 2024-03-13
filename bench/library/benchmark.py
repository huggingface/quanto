import argparse
import time
from contextlib import nullcontext

import numpy as np
import torch
from tqdm.auto import tqdm

from quanto.library import disable_extensions
from quanto.tensor.core import group
from quanto.tensor.packed import pack_weights


def get_udqmm_bench(input_dtype, device, bits):
    input = torch.rand([128, 128], dtype=input_dtype).to(device)
    weight = torch.randint(-127, 127, [128, 128], dtype=torch.int8).to(device)

    input_shape = weight.shape
    grouped_weights = group(weight, axis=0, group_size=int(input_shape[-1] / 4))
    scale = torch.ones((1, grouped_weights.shape[1]), dtype=input_dtype, device=device) * 0.5

    packed_weights = pack_weights(grouped_weights, bits)

    def bench_fn():
        return torch.ops.quanto.udqmm(input, packed_weights, scale, bits)

    return bench_fn


def get_dqmm_bench(input_dtype, device):
    input = torch.rand([1024, 1024], dtype=input_dtype).to(device)
    other = torch.randint(-127, 127, [1024, 1024], dtype=torch.int8).to(device)
    other_scale = torch.ones((1024,), dtype=input_dtype, device=device) * 0.5

    def bench_fn():
        return torch.ops.quanto.dqmm(input, other, other_scale)

    return bench_fn


def get_quantize_symmetric_bench(src_dtype, dst_dtype, per_axis, device):
    a = torch.rand([10240, 10240], dtype=src_dtype).to(device)
    scale = torch.ones((10240,)) * 0.5 if per_axis else torch.tensor(0.5)
    scale = scale.to(src_dtype).to(device)

    def bench_fn():
        return torch.ops.quanto.quantize_symmetric(a, scale, dst_dtype)

    return bench_fn


def get_unpack_bench(bits, device):
    qmax = 2**bits
    a = torch.randint(0, qmax, [10240, 10240], dtype=torch.uint8).to(device)

    def bench_fn():
        return torch.ops.quanto.unpack(a, bits)

    return bench_fn


def timing(get_bench_func, device, iterations=10):
    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cpu.synchronize()

    def timing_event(device):
        if device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        elif device.type == "mps":
            return torch.mps.Event(enable_timing=True)

        class CPUEvent:
            def __init__(self):
                self.time = None

            def record(self):
                self.time = time.time()

            def elapsed_time(self, other):
                assert self.time is not None
                assert other.time is not None
                return (other.time - self.time) * 1000

        return CPUEvent()

    synchronize(device)

    bench_func = get_bench_func(device)
    # Warmup to load library
    bench_func()
    latencies = np.empty((iterations, 2))
    for i in tqdm(range(iterations)):
        for j, context in enumerate([disable_extensions(), nullcontext()]):
            start_event = timing_event(device)
            end_event = timing_event(device)
            synchronize(device)
            start_event.record()
            with context:
                bench_func()
            end_event.record()
            synchronize(device)
            latencies[i, j] = start_event.elapsed_time(end_event)
    return np.mean(latencies[:, 0]), np.mean(latencies[:, 1])


GET_BENCH_FUNCTIONS = {
    "dqmm_w8a16": lambda device: get_dqmm_bench(torch.float16, device),
    "quantize_symmetric_fp32_int8_per_tensor": lambda device: get_quantize_symmetric_bench(
        torch.float32, torch.int8, False, device
    ),
    "unpack_2bit": lambda device: get_unpack_bench(2, device),
    "unpack_4bit": lambda device: get_unpack_bench(4, device),
    "udqmm_4bit": lambda device: get_udqmm_bench(torch.float16, device, 4),
}


def main():
    parser = argparse.ArgumentParser(description="Kernel benchmark")
    parser.add_argument("--kernel", type=str, default=None, help="The kernel to benchmark. None to test all of them")
    parser.add_argument("--device", type=str, default=None, help="The device to use for benchmark.")
    parser.add_argument("--it", type=int, default=10, help="The number of benchmark iterations")
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    all_kernels = GET_BENCH_FUNCTIONS.keys()
    kernels = all_kernels if args.kernel is None else [args.kernel]
    for kernel in kernels:
        get_bench_fn = GET_BENCH_FUNCTIONS[kernel]
        python_ms, ext_ms = timing(get_bench_fn, device, iterations=args.it)
        ratio = python_ms / ext_ms
        print(f"\n{kernel}[{device.type}]: python = {python_ms:.3f} ms, ext = {ext_ms:.3f} ms, ratio = {ratio:.1f}x")


if __name__ == "__main__":
    main()
