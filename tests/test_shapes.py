"""Shape correctness and gradient flow tests."""

import pytest
import torch

from observer_nn import ModelConfig, ObserverNeuralNetwork
from observer_nn.model import InnerNetwork, ObserverLayer, MultiDomainFusion, IonisedFeedback

BATCH = 4

@pytest.fixture
def small_config():
    return ModelConfig(
        input_dim=8,
        hidden_dim=32,
        n_qubits=4,
        n_qlayers=1,
        output_dim=3,
        n_harmonics=2,
        sample_rate=100,
        bpm=70.0,
    )


# ── MultiDomainFusion ────────────────────────────────────────────────────────

def test_multi_domain_fusion_shape(small_config):
    m = MultiDomainFusion(small_config)
    beat = torch.randn(BATCH, small_config.beat_samples, small_config.input_dim)
    out = m(beat)
    assert out.shape == (BATCH, small_config.hidden_dim)


# ── InnerNetwork ─────────────────────────────────────────────────────────────

def test_inner_decode_no_context(small_config):
    net = InnerNetwork(small_config)
    h = torch.randn(BATCH, small_config.hidden_dim)
    out = net.decode(h)
    assert out.shape == (BATCH, small_config.hidden_dim)


def test_inner_decode_with_context(small_config):
    net = InnerNetwork(small_config)
    h = torch.randn(BATCH, small_config.hidden_dim)
    ctx = torch.randn(BATCH, small_config.hidden_dim)
    out = net.decode(h, ctx, alpha=0.5)
    assert out.shape == (BATCH, small_config.hidden_dim)


# ── ObserverLayer ─────────────────────────────────────────────────────────────

def test_observer_output_shape(small_config):
    obs = ObserverLayer(small_config)
    h = torch.randn(BATCH, small_config.hidden_dim)
    ctx = obs(h, phase=0.0)
    assert ctx.shape == (BATCH, small_config.hidden_dim)


def test_observer_different_phases(small_config):
    obs = ObserverLayer(small_config)
    h = torch.randn(BATCH, small_config.hidden_dim)
    ctx0 = obs(h, phase=0.0)
    ctx1 = obs(h, phase=0.5)
    # Different phases should produce different outputs
    assert not torch.allclose(ctx0, ctx1)


# ── ObserverNeuralNetwork ─────────────────────────────────────────────────────

def test_full_forward_shape(small_config):
    model = ObserverNeuralNetwork(small_config)
    seq_len = small_config.beat_samples * 2
    x = torch.randn(BATCH, seq_len, small_config.input_dim)
    logits, internals = model(x)
    assert logits.shape == (BATCH, small_config.output_dim)


def test_internals_structure(small_config):
    model = ObserverNeuralNetwork(small_config)
    seq_len = small_config.beat_samples * 2
    x = torch.randn(BATCH, seq_len, small_config.input_dim)
    _, internals = model(x)
    assert "beat_states" in internals
    assert "contexts" in internals
    assert len(internals["beat_states"]) == 2  # 2 beats
    assert len(internals["contexts"]) == 2
    assert len(internals["contexts"][0]) == small_config.n_harmonics


def test_beat_state_shapes(small_config):
    model = ObserverNeuralNetwork(small_config)
    seq_len = small_config.beat_samples * 2
    x = torch.randn(BATCH, seq_len, small_config.input_dim)
    _, internals = model(x)
    for h in internals["beat_states"]:
        assert h.shape == (BATCH, small_config.hidden_dim)


# ── IonisedFeedback ───────────────────────────────────────────────────────────

def test_ionised_feedback_no_prev(small_config):
    fb = IonisedFeedback(small_config)
    h = torch.randn(BATCH, small_config.hidden_dim)
    out = fb(h, prev_logits=None)
    assert out.shape == (BATCH, small_config.hidden_dim)
    assert torch.allclose(out, h)  # no injection when prev_logits is None


def test_ionised_feedback_with_prev(small_config):
    fb = IonisedFeedback(small_config)
    h = torch.randn(BATCH, small_config.hidden_dim)
    prev = torch.randn(BATCH, small_config.output_dim)
    out = fb(h, prev_logits=prev)
    assert out.shape == (BATCH, small_config.hidden_dim)


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flows_through_observer(small_config):
    model = ObserverNeuralNetwork(small_config)
    seq_len = small_config.beat_samples * 2
    x = torch.randn(BATCH, seq_len, small_config.input_dim)
    logits, _ = model(x)
    logits.sum().backward()
    for name, p in model.observer.named_parameters():
        assert p.grad is not None, f"No gradient for observer.{name}"


def test_gradient_flows_through_domain_fusion(small_config):
    model = ObserverNeuralNetwork(small_config)
    seq_len = small_config.beat_samples * 2
    x = torch.randn(BATCH, seq_len, small_config.input_dim)
    logits, _ = model(x)
    logits.sum().backward()
    for name, p in model.domain_fusion.named_parameters():
        assert p.grad is not None, f"No gradient for domain_fusion.{name}"


def test_domain_weights_gradient(small_config):
    model = ObserverNeuralNetwork(small_config)
    seq_len = small_config.beat_samples * 2
    x = torch.randn(BATCH, seq_len, small_config.input_dim)
    logits, _ = model(x)
    logits.sum().backward()
    assert model.domain_fusion.domain_weights.grad is not None
