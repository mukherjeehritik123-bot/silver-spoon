import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

from .config import ModelConfig


def _build_encoder_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.LayerNorm(out_dim),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
        nn.GELU(),
    )


class MultiDomainFusion(nn.Module):
    """
    Extracts representations from four signal domains for a single beat chunk
    and fuses them into one hidden vector. The vortex suction pulls from every
    domain simultaneously before the observer feedback loop begins.

    Domains:
      - time:  raw flattened beat (beat_samples x input_dim)
      - freq:  FFT magnitude mean-pooled across time -> (input_dim,)
      - stat:  mean and std across time -> (input_dim x 2,)
      - delta: first-order time differences flattened (beat_samples-1) x input_dim
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        h = config.hidden_dim
        d = config.dropout
        inp = config.input_dim
        bs = config.beat_samples

        self.enc_time  = _build_encoder_block(bs * inp, h, d)
        self.enc_freq  = _build_encoder_block(inp, h, d)
        self.enc_stat  = _build_encoder_block(inp * 2, h, d)
        self.enc_delta = _build_encoder_block((bs - 1) * inp, h, d)

        # Learned scalar weights determine each domain's pull strength in the vortex
        self.domain_weights = nn.Parameter(torch.ones(4))
        self.norm = nn.LayerNorm(h)

    def forward(self, beat: torch.Tensor) -> torch.Tensor:
        # beat: (batch, beat_samples, input_dim)
        batch, bs, inp = beat.shape

        # --- time domain ---
        h_time = self.enc_time(beat.reshape(batch, -1))

        # --- frequency domain (FFT magnitude, mean-pooled over frequency bins) ---
        freq_feat = torch.fft.rfft(beat, dim=1).abs().mean(dim=1)  # (batch, input_dim)
        h_freq = self.enc_freq(freq_feat)

        # --- statistical domain (mean + std across time) ---
        h_stat = self.enc_stat(
            torch.cat([beat.mean(dim=1), beat.std(dim=1)], dim=-1)
        )

        # --- gradient/delta domain (first differences) ---
        delta_feat = (beat[:, 1:, :] - beat[:, :-1, :]).reshape(batch, -1)
        h_delta = self.enc_delta(delta_feat)

        # Softmax-normalised domain weights control how hard each domain is pulled
        w = torch.softmax(self.domain_weights, dim=0)
        fused = w[0] * h_time + w[1] * h_freq + w[2] * h_stat + w[3] * h_delta
        return self.norm(fused)  # (batch, hidden_dim)


class IonisedFeedback(nn.Module):
    """
    Projects the model's own previous output (logits) back into the hidden
    space and injects it at the start of the next beat's vortex cycle.
    This 'ionised release' lets each beat's response re-enter the vortex
    as charged context, closing the outer feedback loop.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(config.output_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        nn.init.zeros_(self.gate.bias)  # start with zero gate (no injection for first beat)

    def forward(
        self, h: torch.Tensor, prev_logits: torch.Tensor | None
    ) -> torch.Tensor:
        if prev_logits is None:
            return h
        # Ionise: project logits to hidden space, gate the injection
        ion = self.proj(prev_logits.detach())  # detach to avoid unrolling the full history
        gate = torch.sigmoid(self.gate(h))
        return self.norm(h + gate * ion)


class InnerNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.gate_proj = nn.Linear(h, h)
        self.decoder = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
        )
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)

        # Gate bias initialised to +1 so observer context is used from the start
        nn.init.ones_(self.gate_proj.bias)

    def decode(
        self,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if context is not None:
            gate = torch.sigmoid(self.gate_proj(h))
            h = self.norm1(h + alpha * gate * context)
        out = self.decoder(h)
        return self.norm2(out)


class ObserverLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n = config.n_qubits
        h = config.hidden_dim
        n_qlayers = config.n_qlayers

        self.proj_in = nn.Linear(h, n)
        self.proj_out = nn.Linear(n, h)
        self.norm_out = nn.LayerNorm(h)

        # Quantum weights managed as nn.Parameters.
        # TorchLayer is avoided because its internal batching conflicts with
        # backprop execution in PennyLane 0.44 when batch != n_qubits.
        self.weights_ry = nn.Parameter(torch.randn(n_qlayers, n) * 0.1)
        self.weights_rz = nn.Parameter(torch.randn(n_qlayers, n) * 0.1)

        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(angles, weights_ry, weights_rz):
            # angles: (n_qubits,) for a single sample, carrying the harmonic phase offset
            for i in range(n):
                qml.RY(angles[i], wires=i)

            # Star geometry: qubit 0 is the centre, qubits 1..n-1 are the spokes.
            # Centre controls all outer qubits (hub broadcasts outward).
            for i in range(1, n):
                qml.CNOT(wires=[0, i])

            # Parameterised rotation layers
            for layer in range(n_qlayers):
                for i in range(n):
                    qml.RY(weights_ry[layer][i], wires=i)
                    qml.RZ(weights_rz[layer][i], wires=i)

            # Reverse star: outer qubits report back to centre (spokes collapse inward).
            for i in range(1, n):
                qml.CNOT(wires=[i, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        self._circuit = _circuit

    def forward(self, hidden: torch.Tensor, phase: float = 0.0) -> torch.Tensor:
        # Phase offset shifts qubit angles to the current harmonic of 108 Hz
        angles = torch.tanh(self.proj_in(hidden)) * math.pi + phase  # (batch, n_qubits)

        # Run the quantum circuit per sample; torch.stack preserves the grad graph.
        # Cast to float32 because default.qubit returns float64.
        measurements = torch.stack([
            torch.stack(self._circuit(angles[i], self.weights_ry, self.weights_rz))
            for i in range(angles.shape[0])
        ]).float()  # (batch, n_qubits)

        return self.norm_out(self.proj_out(measurements))  # (batch, hidden_dim)


class ObserverNeuralNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.domain_fusion  = MultiDomainFusion(config)
        self.ionised_fb     = IonisedFeedback(config)
        self.inner          = InnerNetwork(config)
        self.observer       = ObserverLayer(config)
        self.head           = nn.Linear(config.hidden_dim, config.output_dim)
        self.norm_out       = nn.LayerNorm(config.hidden_dim)

    def _iter_beats(self, x: torch.Tensor):
        beat = self.config.beat_samples
        seq_len = x.shape[1]
        start = 0
        while start < seq_len:
            chunk = x[:, start : start + beat, :]
            if chunk.shape[1] < beat:
                pad = beat - chunk.shape[1]
                chunk = F.pad(chunk, (0, 0, 0, pad))
            yield chunk  # (batch, beat_samples, input_dim)
            start += beat

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        # x: (batch, seq_len, input_dim)
        beat_states:  list[torch.Tensor]       = []
        all_contexts: list[list[torch.Tensor]] = []
        prev_logits:  torch.Tensor | None      = None

        for beat_chunk in self._iter_beats(x):
            # Step 1 — multi-domain vortex intake
            h = self.domain_fusion(beat_chunk)  # (batch, hidden_dim)

            # Step 2 — ionised release: previous beat's response re-enters the vortex
            h = self.ionised_fb(h, prev_logits)

            # Step 3 — quantum observer vortex (108 Hz harmonic cycles)
            contexts: list[torch.Tensor] = []
            for k, phase in enumerate(self.config.harmonic_phases):
                alpha = (k + 1) / self.config.n_harmonics
                ctx = self.observer(h, phase=phase)
                h = self.inner.decode(h, ctx, alpha=alpha)
                contexts.append(ctx)

            beat_states.append(h)
            all_contexts.append(contexts)

            # Step 4 — compute beat-level logits for ionised re-entry next iteration
            prev_logits = self.head(self.norm_out(h))

        h_final = torch.stack(beat_states, dim=1).mean(dim=1)  # (batch, hidden_dim)
        logits  = self.head(self.norm_out(h_final))

        return logits, {"beat_states": beat_states, "contexts": all_contexts}
