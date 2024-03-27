import argparse
import io
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from quanto import Calibration, freeze, qint4, qint8, qfloat8, qint2, quantize


def evaluate_model(model, tokenizer, dataset, device, batch_size):
    p = pipeline("sentiment-analysis", model, tokenizer=tokenizer, device=device)
    results = p(KeyDataset(dataset, "sentence"), batch_size=batch_size)
    start = time.time()
    pred_labels = [0 if result["label"] == "NEGATIVE" else 1 for result in results]
    end = time.time()
    accuracy = np.sum(np.equal(pred_labels, dataset["label"])) / len(pred_labels)
    print(f"{len(pred_labels)} sentences evaluated in {end - start:.2f} s. accuracy = {accuracy}")


def keyword_to_itype(k):
    return {"none": None, "int8": qint8, "int4": qint4, "int2": qint2, "float8": qfloat8}[k]


def main():
    parser = argparse.ArgumentParser(description="Transformers SST2 Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="The name of the trained Model.",
    )
    parser.add_argument("--samples", type=int, default=872, help="The number of sst2 samples to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size to use for evaluation.")
    parser.add_argument("--weights", type=str, default="int8", choices=["int4", "int8", "int2", "float8"])
    parser.add_argument("--activations", type=str, default="int8", choices=["none", "int8", "float8", "int4", "int2"])
    parser.add_argument("--device", type=str, default=None, help="The device to use for evaluation.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("sst2", split=f"validation[:{args.samples}]")

    print("Float model")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    weights = keyword_to_itype(args.weights)
    activations = keyword_to_itype(args.activations)
    quantize(model, weights=weights, activations=activations)
    if activations is not None:
        print("Calibrating ...")
        with Calibration():
            evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    freeze(model)
    print(f"Quantized model (w: {args.weights}, a: {args.activations})")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    b.seek(0)
    state_dict = torch.load(b)
    model_reloaded = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    quantize(model_reloaded, weights=weights, activations=activations)
    model_reloaded.load_state_dict(state_dict)
    print("Serialized quantized model")
    evaluate_model(model, tokenizer, dataset, device, args.batch_size)


if __name__ == "__main__":
    main()

