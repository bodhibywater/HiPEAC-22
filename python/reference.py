"""Numerically explicit NumPy reference for the project's LIF state transition.

This module is a correctness reference, not a performance benchmark. Grouping
uses the same *mean incoming weights and bias* approximation as C++.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LIFLayer:
    """One fully connected LIF layer with output-by-input weight orientation."""

    weights: np.ndarray
    bias: np.ndarray
    beta: float = 0.9
    threshold: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid configurations before reference calculations begin."""
        if self.weights.ndim != 2 or self.weights.size == 0:
            raise ValueError("weights must have nonempty [outputs, inputs] shape")
        if self.bias.shape != (self.weights.shape[0],):
            raise ValueError("bias shape mismatch")
        if not (0.0 <= self.beta <= 1.0) or not self.threshold > 0.0:
            raise ValueError("invalid LIF parameters")
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise ValueError("nonfinite weights or bias")


def step(layer: LIFLayer, spikes: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute one LIF timestep: decay + current + bias, threshold, hard reset."""
    spikes = np.asarray(spikes, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if spikes.shape != (layer.weights.shape[1],) or state.shape != (layer.weights.shape[0],):
        raise ValueError("input or state has incorrect shape")
    if not np.all((spikes == 0.0) | (spikes == 1.0)):
        raise ValueError("spikes must be binary")
    current = np.asarray(layer.weights, dtype=np.float32) @ spikes
    voltage = np.float32(layer.beta) * state + current + np.asarray(layer.bias, dtype=np.float32)
    output = (voltage >= np.float32(layer.threshold)).astype(np.uint8)
    next_state = np.where(output != 0, 0.0, voltage).astype(np.float32)
    return output, next_state


def group_adjacent_mean(layer: LIFLayer, group_size: int) -> LIFLayer:
    """Approximate adjacent neurons using average weights and bias per group.

    Shared voltage means that every member of a group emits the same spike.
    The approximation is exact for group_size=1 but not generally otherwise.
    """
    if group_size < 1:
        raise ValueError("group_size must be positive")
    new_weights = []
    new_bias = []
    for start in range(0, layer.weights.shape[0], group_size):
        stop = min(start + group_size, layer.weights.shape[0])
        new_weights.append(layer.weights[start:stop].mean(axis=0))
        new_bias.append(layer.bias[start:stop].mean())
    return LIFLayer(
        weights=np.asarray(new_weights, dtype=np.float32),
        bias=np.asarray(new_bias, dtype=np.float32),
        beta=layer.beta,
        threshold=layer.threshold,
    )


def grouped_step(
    layer: LIFLayer,
    spikes: np.ndarray,
    group_state: np.ndarray,
    group_size: int,
    original_outputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Update one state per group, broadcasting spikes back to original neurons."""
    group_spikes, next_state = step(layer, spikes, group_state)
    output = np.repeat(group_spikes, group_size)[:original_outputs]
    return output, next_state


def simulate(
    layer: LIFLayer, inputs_by_time: np.ndarray, group_size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a sequence; reset state at start, not between timesteps."""
    if inputs_by_time.ndim != 2:
        raise ValueError("time-major inputs must have shape [T, inputs]")
    grouped = group_adjacent_mean(layer, group_size)
    state = np.zeros(grouped.weights.shape[0], dtype=np.float32)
    outputs = []
    for input_spikes in inputs_by_time:
        out, state = grouped_step(
            grouped, input_spikes, state, group_size, layer.weights.shape[0]
        )
        outputs.append(out)
    return np.stack(outputs, axis=0), state
