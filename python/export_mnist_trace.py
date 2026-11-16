"""Export a reproducible held-out MNIST SNN workload and PyTorch spike oracle.

The binary artifact stores trained float32 weights, exact binary input spikes,
and expected hidden/output spikes for every timestep of every selected image.
The companion C++ benchmark verifies predictions and all spike events before
collecting any gem5 kernel statistics.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from python.train_mnist import TwoLayerSNN

MAGIC = b"SNNMNST1"


def write_float32(file, values: torch.Tensor) -> None:
    """Serialize contiguous little-endian float32 tensors without transposing."""
    values = values.detach().cpu().contiguous().numpy()
    file.write(np.ascontiguousarray(values, dtype="<f4").tobytes())


def write_binary(file, values: torch.Tensor) -> None:
    """Serialize 0/1 spikes in time-major order as one byte per event."""
    values = values.detach().cpu().contiguous().numpy()
    file.write(np.ascontiguousarray(values, dtype=np.uint8).tobytes())


def generate_reference(model: TwoLayerSNN, spikes: torch.Tensor):
    """Evaluate trained network from fixed spikes, preserving per-image state."""
    hidden_count = model.fc1.out_features
    hidden_state = torch.zeros(hidden_count, dtype=torch.float32)
    output_state = torch.zeros(10, dtype=torch.float32)
    outputs_hidden, outputs_final = [], []
    counts = torch.zeros(10, dtype=torch.float32)

    # Exact ordering from TwoLayerSNN.forward; gradients are unnecessary.
    with torch.no_grad():
        for incoming in spikes:
            hidden_state = model.beta * hidden_state + model.fc1(incoming)
            s_hidden = (hidden_state >= model.threshold).to(torch.float32)
            hidden_state = hidden_state * (1.0 - s_hidden)

            output_state = model.beta * output_state + model.fc2(s_hidden)
            s_output = (output_state >= model.threshold).to(torch.float32)
            output_state = output_state * (1.0 - s_output)
            counts += s_output
            outputs_hidden.append(s_hidden.to(torch.uint8))
            outputs_final.append(s_output.to(torch.uint8))
    return torch.stack(outputs_hidden), torch.stack(outputs_final), int(counts.argmax())


def main() -> None:
    """Build a deterministic benchmark from the saved model and MNIST test set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/mnist_snn.pt")
    parser.add_argument("--data", default=None, help="MNIST directory (defaults to checkpoint setting)")
    parser.add_argument("--output", default="artifacts/mnist_trace.bin")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if not 1 <= args.samples <= 10000:
        parser.error("--samples must be between 1 and 10000")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    original = checkpoint["args"]
    steps = int(original["steps"])
    hidden = int(original["hidden"])
    beta = float(original["beta"])
    threshold = float(original["threshold"])
    model = TwoLayerSNN(hidden, beta, threshold, float(original.get("slope", 5.0)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(1)

    data_dir = args.data if args.data is not None else original.get("data", "data/mnist")
    test_set = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor(), download=True)
    indices = np.random.default_rng(args.seed).choice(len(test_set), size=args.samples, replace=False)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    correct = input_spike_count = hidden_spike_count = 0
    with target.open("wb") as file:
        # Header: magic + samples, steps, inputs, hidden, outputs + beta, threshold.
        file.write(struct.pack("<8sIIIIIff", MAGIC, args.samples, steps, 784, hidden, 10, beta, threshold))
        write_float32(file, model.fc1.weight)
        write_float32(file, model.fc1.bias)
        write_float32(file, model.fc2.weight)
        write_float32(file, model.fc2.bias)

        # Record the actual spike inputs once, never regenerate inside C++.
        for index in indices:
            pixels, label = test_set[int(index)]
            pixels = pixels.reshape(784).float()
            spikes = (torch.rand((steps, 784), generator=generator) < pixels).float()
            hidden_spikes, output_spikes, prediction = generate_reference(model, spikes)
            file.write(struct.pack("<BB", int(label), prediction))
            write_binary(file, spikes)
            write_binary(file, hidden_spikes)
            write_binary(file, output_spikes)
            correct += int(prediction == int(label))
            input_spike_count += int(spikes.sum().item())
            hidden_spike_count += int(hidden_spikes.sum().item())

    print(json.dumps({
        "output": str(target), "samples": args.samples, "timesteps": steps,
        "hidden": hidden, "reference_accuracy_on_subset": correct / args.samples,
        "realised_input_density": input_spike_count / (args.samples * steps * 784),
        "realised_hidden_density": hidden_spike_count / (args.samples * steps * hidden),
        "saved_checkpoint_test_accuracy": checkpoint.get("test_accuracy"),
        "note": "Reference predictions and full spike traces saved for strict C++ verification",
    }, indent=2))


if __name__ == "__main__":
    main()
