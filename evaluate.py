"""
Evaluation and inference script for ObserverNeuralNetwork.

Logs per-beat state changes and domain fusion weights for interpretability.
"""

import torch
import torch.nn.functional as F

from observer_nn import ModelConfig, ObserverNeuralNetwork


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.mean(0, keepdim=True), b.mean(0, keepdim=True)).item()


@torch.no_grad()
def evaluate(model: ObserverNeuralNetwork, dataloader, device: str = "cpu"):
    model.eval()
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, internals = model(batch_x)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    acc = correct / total
    print(f"Accuracy: {acc:.3f}  ({correct}/{total})")
    return acc


@torch.no_grad()
def inspect_vortex(model: ObserverNeuralNetwork, x: torch.Tensor):
    """Show how the observer vortex transforms the hidden state across harmonic cycles."""
    model.eval()
    logits, internals = model(x)

    beat_states = internals["beat_states"]
    contexts_per_beat = internals["contexts"]

    print(f"\nNumber of beats processed: {len(beat_states)}")
    for b_idx, (h_final, contexts) in enumerate(zip(beat_states, contexts_per_beat)):
        print(f"\n  Beat {b_idx + 1}:")
        print(f"    Final hidden state norm : {h_final.norm(dim=-1).mean().item():.4f}")
        for k, ctx in enumerate(contexts):
            alpha = (k + 1) / len(contexts)
            print(
                f"    Harmonic {k+1} (α={alpha:.2f}) "
                f"context norm={ctx.norm(dim=-1).mean().item():.4f}  "
                f"variance={ctx.var(dim=0).mean().item():.4f}"
            )

    # Domain weight distribution (how strongly each domain is pulled)
    w = torch.softmax(model.domain_fusion.domain_weights, dim=0)
    domains = ["time", "freq", "stat", "delta"]
    print("\nDomain vortex pull weights:")
    for name, weight in zip(domains, w.tolist()):
        bar = "#" * int(weight * 40)
        print(f"  {name:6s}: {weight:.3f}  {bar}")

    return logits


if __name__ == "__main__":
    from observer_nn import ModelConfig

    config = ModelConfig(
        input_dim=16,
        hidden_dim=64,
        n_qubits=4,
        n_qlayers=1,
        output_dim=5,
        n_harmonics=2,
        sample_rate=100,
    )
    model = ObserverNeuralNetwork(config)
    x = torch.randn(2, config.beat_samples * 2, config.input_dim)
    inspect_vortex(model, x)
