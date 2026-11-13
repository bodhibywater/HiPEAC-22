"""Reference tests are independent of the C++ implementation."""

import numpy as np
import pytest

from python.reference import LIFLayer, group_adjacent_mean, simulate, step


def test_expected_spike_sequence_and_decay() -> None:
    """Check leak, threshold, and hard reset using hand-calculated values."""
    layer = LIFLayer(np.array([[1.0]], np.float32), np.zeros(1, np.float32), 0.5, 1.0)
    inputs = np.array([[1], [0], [1], [0], [1]], dtype=np.float32)
    spikes, state = simulate(layer, inputs)
    assert spikes[:, 0].tolist() == [1, 0, 1, 0, 1]
    np.testing.assert_allclose(state, [0.0])


def test_group_one_exactly_matches_raw() -> None:
    """The grouping implementation must reduce to the baseline at g=1."""
    rng = np.random.default_rng(20261113)
    weights = rng.normal(size=(7, 11)).astype(np.float32)
    bias = rng.normal(size=7).astype(np.float32) * 0.05
    layer = LIFLayer(weights, bias, beta=0.82, threshold=1.2)
    inputs = rng.binomial(1, 0.31, size=(33, 11)).astype(np.float32)
    out, state = simulate(layer, inputs)
    np.testing.assert_array_equal(out, simulate(layer, inputs, group_size=1)[0])
    np.testing.assert_allclose(state, simulate(layer, inputs, group_size=1)[1])
    np.testing.assert_array_equal(group_adjacent_mean(layer, 1).weights, layer.weights)


def test_grouped_partial_group_and_broadcast() -> None:
    """Partial groups have their actual member count in the mean."""
    w = np.array([[1.0], [3.0], [5.0]], np.float32)
    layer = LIFLayer(w, np.zeros(3, np.float32), threshold=3.0)
    agg = group_adjacent_mean(layer, 2)
    np.testing.assert_allclose(agg.weights, [[2.0], [5.0]])
    spikes, _ = simulate(layer, np.ones((1, 1), np.float32), group_size=2)
    assert spikes.tolist() == [[0, 0, 1]]


def test_nonbinary_rejected() -> None:
    """Non-spike activations are not silently accepted by the reference."""
    layer = LIFLayer(np.ones((1, 1), np.float32), np.zeros(1, np.float32))
    with pytest.raises(ValueError, match="binary"):
        step(layer, np.array([0.4], np.float32), np.zeros(1, np.float32))
