import math
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_dim: int = 32
    hidden_dim: int = 256
    n_qubits: int = 8
    n_qlayers: int = 2
    output_dim: int = 10
    inner_num_layers: int = 3
    dropout: float = 0.1
    sample_rate: int = 1000
    base_freq: float = 108.0
    bpm: float = 70.0
    n_harmonics: int = 4

    @property
    def beat_samples(self) -> int:
        return round(self.sample_rate * 60.0 / self.bpm)

    @property
    def harmonic_phases(self) -> list:
        return [
            2 * math.pi * k * self.base_freq / self.sample_rate
            for k in range(self.n_harmonics)
        ]

    @property
    def beat_input_dim(self) -> int:
        return self.beat_samples * self.input_dim
