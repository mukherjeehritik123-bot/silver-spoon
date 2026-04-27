import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_shape(tensor: torch.Tensor, expected: tuple, name: str = "") -> None:
    actual = tuple(tensor.shape)
    assert actual == expected, (
        f"Shape mismatch{' for ' + name if name else ''}: "
        f"expected {expected}, got {actual}"
    )
