"""Optional MNIST baseline with surrogate-gradient LIF and rate-coded input.

Requires: pip install torch torchvision. Downloads MNIST when run. This script
is a starting baseline, not evidence of validated classification accuracy.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LIFStep(torch.autograd.Function):
    """Hard forward spikes with an explicit smooth surrogate backward rule."""

    @staticmethod
    def forward(ctx, voltage: torch.Tensor, threshold: float, slope: float) -> torch.Tensor:
        """Emit binary spikes and preserve voltage for the surrogate derivative."""
        ctx.save_for_backward(voltage)
        ctx.threshold = threshold
        ctx.slope = slope
        return (voltage >= threshold).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Use the derivative of sigmoid(slope * (voltage - threshold))."""
        (voltage,) = ctx.saved_tensors
        value = torch.sigmoid(ctx.slope * (voltage - ctx.threshold))
        return gradient * ctx.slope * value * (1.0 - value), None, None


class TwoLayerSNN(nn.Module):
    """784 -> hidden -> 10 feedforward SNN using zero-reset LIF neurons."""

    def __init__(self, hidden: int, beta: float, threshold: float, slope: float) -> None:
        """Create real-valued weights and record neuron dynamics parameters."""
        super().__init__()
        self.fc1 = nn.Linear(784, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 10, bias=True)
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    def forward(self, pixels: torch.Tensor, timesteps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rate-code pixels and integrate output spikes over T timesteps."""
        batch = pixels.shape[0]
        x = pixels.reshape(batch, -1)
        hidden = torch.zeros(batch, self.fc1.out_features, device=x.device)
        output = torch.zeros(batch, 10, device=x.device)
        counts = torch.zeros_like(output)
        hidden_spikes = 0.0
        for _ in range(timesteps):
            # Random inputs are Bernoulli(x): bright pixels spike more often.
            incoming = (torch.rand_like(x) < x).to(x.dtype)
            hidden = self.beta * hidden + self.fc1(incoming)
            s_hidden = LIFStep.apply(hidden, self.threshold, self.slope)
            hidden = hidden * (1.0 - s_hidden.detach())
            output = self.beta * output + self.fc2(s_hidden)
            s_output = LIFStep.apply(output, self.threshold, self.slope)
            output = output * (1.0 - s_output.detach())
            counts = counts + s_output
            hidden_spikes = hidden_spikes + s_hidden.detach().mean()
        return counts / timesteps, hidden_spikes / timesteps


def set_seeds(seed: int) -> None:
    """Fix RNG seeds; CPU numerical determinism depends on the installed BLAS."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: TwoLayerSNN, loader: DataLoader, steps: int) -> tuple[float, float]:
    """Measure test accuracy and mean hidden spike density without gradients."""
    model.eval()
    correct = samples = 0
    mean_density = 0.0
    with torch.no_grad():
        for pixels, labels in loader:
            rates, density = model(pixels, steps)
            correct += (rates.argmax(dim=1) == labels).sum().item()
            samples += labels.numel()
            mean_density += density.item() * labels.numel()
    return correct / samples, mean_density / samples


def main() -> None:
    """Train and evaluate a simple SNN, saving hyperparameters with its weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--output", default="artifacts/mnist_snn.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--slope", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1:
        parser.error("epochs and steps must be positive")
    set_seeds(args.seed)
    transform = transforms.ToTensor()
    train = datasets.MNIST(args.data, train=True, transform=transform, download=True)
    test = datasets.MNIST(args.data, train=False, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0)
    model = TwoLayerSNN(args.hidden, args.beta, args.threshold, args.slope)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for pixels, labels in train_loader:
            optim.zero_grad(set_to_none=True)
            rates, _ = model(pixels, args.steps)
            loss = nn.functional.cross_entropy(rates, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item() * labels.numel()
        print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / len(train)}))
    set_seeds(args.seed + 1000)  # Fixed independent evaluation RNG stream.
    accuracy, density = evaluate(model, test_loader, args.steps)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "test_accuracy": accuracy, "hidden_spike_density": density}, path)
    print(json.dumps({"test_accuracy": accuracy, "hidden_spike_density": density,
                      "saved_to": str(path)}))


if __name__ == "__main__":
    main()
