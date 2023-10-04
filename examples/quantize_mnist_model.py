import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AutoModel

from quanto.quantization import QLinear, QuantizedTensor, quantize
from quanto.quantization.calibrate import calibration


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).dequantize()
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def print_quantization_stats(model):
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            print(f"{name} quantization stats:")
            qweight = QuantizedTensor.quantize(m.weight)
            weight_mae = torch.nn.L1Loss()(qweight.dequantize(), m.weight)
            weight_stats = f"  weight mae = {weight_mae}"
            if m.bias is not None:
                bias_scale = m.in_scale * qweight._scale
                qbias = QuantizedTensor.quantize(m.bias, torch.int32, bias_scale)
                bias_mae = torch.nn.L1Loss()(qbias.dequantize(), m.bias)
                weight_stats += f", bias mae = {bias_mae}"
            print(weight_stats)
            print(f"  scale: in = {m.in_scale}, out = {m.out_scale}")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--model", type=str, default="dacorvo/mnist-mlp", help="The name of the trained Model.")
    parser.add_argument("--stats", action="store_true", default=False, help="Display quantization statistics")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_kwargs = {"batch_size": args.batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    dataset2 = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    # Test inference for reference
    print("Float model")
    test(model, device, test_loader)
    # Quantize model
    quantize(model)
    # Test inference (should be lower than float)
    print("Quantized model")
    test(model, device, test_loader)
    # Test inference with calibration (should be equivalent to float)
    print("Quantized calibrated model")
    with calibration():
        test(model, device, test_loader)
    if args.stats:
        print_quantization_stats(model)


if __name__ == "__main__":
    main()
