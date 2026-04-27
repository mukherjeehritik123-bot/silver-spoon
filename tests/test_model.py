"""Integration tests for the full model pipeline."""

import torch
import pytest

from observer_nn import ModelConfig, ObserverNeuralNetwork, init_weights, count_parameters

BATCH = 2


@pytest.fixture
def model_and_config():
    config = ModelConfig(
        input_dim=8,
        hidden_dim=32,
        n_qubits=4,
        n_qlayers=1,
        output_dim=3,
        n_harmonics=2,
        sample_rate=100,
    )
    model = ObserverNeuralNetwork(config)
    model.apply(init_weights)
    return model, config


def test_count_parameters(model_and_config):
    model, _ = model_and_config
    n = count_parameters(model)
    assert n > 0


def test_output_is_finite(model_and_config):
    model, config = model_and_config
    x = torch.randn(BATCH, config.beat_samples * 2, config.input_dim)
    logits, _ = model(x)
    assert torch.isfinite(logits).all()


def test_vortex_changes_representation(model_and_config):
    """The observer vortex must meaningfully transform h across harmonic cycles."""
    model, config = model_and_config
    x = torch.randn(BATCH, config.beat_samples * 2, config.input_dim)
    _, internals = model(x)
    # Get first beat's contexts — they should not all be identical
    contexts = internals["contexts"][0]
    assert len(contexts) == config.n_harmonics
    # At least two contexts should differ
    diffs = [
        (contexts[i] - contexts[i + 1]).abs().mean().item()
        for i in range(len(contexts) - 1)
    ]
    assert any(d > 1e-6 for d in diffs), "All harmonic contexts are identical"


def test_deterministic_eval(model_and_config):
    model, config = model_and_config
    x = torch.randn(BATCH, config.beat_samples * 2, config.input_dim)
    model.eval()
    with torch.no_grad():
        logits1, _ = model(x)
        logits2, _ = model(x)
    assert torch.allclose(logits1, logits2)


def test_partial_beat_padding(model_and_config):
    """Model should handle inputs whose length is not a multiple of beat_samples."""
    model, config = model_and_config
    # seq_len = 1.5 beats
    seq_len = int(config.beat_samples * 1.5)
    x = torch.randn(BATCH, seq_len, config.input_dim)
    logits, _ = model(x)
    assert logits.shape == (BATCH, config.output_dim)
