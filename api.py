"""
Live model server — send data in, get predictions back.
Run with:  python api.py
Then open: http://localhost:8000
"""

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from observer_nn import ModelConfig, ObserverNeuralNetwork, init_weights

# ── Boot the model ────────────────────────────────────────────────────────────

config = ModelConfig(
    input_dim=16,
    hidden_dim=64,
    n_qubits=4,
    n_qlayers=1,
    output_dim=5,
    n_harmonics=2,
    sample_rate=100,
)

LABELS = ["Class A", "Class B", "Class C", "Class D", "Class E"]

model = ObserverNeuralNetwork(config)
model.apply(init_weights)
model.eval()

print("Model ready.")

# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI()


class PredictRequest(BaseModel):
    values: list[float] | None = None  # optional: send your own numbers


@app.post("/predict")
def predict(req: PredictRequest):
    seq_len = config.beat_samples * 2

    if req.values:
        # Pad or trim to the right length
        vals = req.values[:seq_len * config.input_dim]
        vals += [0.0] * (seq_len * config.input_dim - len(vals))
        x = torch.tensor(vals, dtype=torch.float32)
        x = x.reshape(1, seq_len, config.input_dim)
    else:
        # Demo: generate random data
        x = torch.randn(1, seq_len, config.input_dim)

    with torch.no_grad():
        logits, internals = model(x)
        probs = torch.softmax(logits, dim=-1)[0]

    top_idx = probs.argmax().item()

    # Domain pull weights
    w = torch.softmax(model.domain_fusion.domain_weights, dim=0).tolist()

    return {
        "prediction": LABELS[top_idx],
        "confidence": f"{probs[top_idx].item() * 100:.1f}%",
        "all_scores": {
            LABELS[i]: f"{probs[i].item() * 100:.1f}%"
            for i in range(len(LABELS))
        },
        "vortex_domain_pull": {
            "time":  f"{w[0]*100:.1f}%",
            "freq":  f"{w[1]*100:.1f}%",
            "stat":  f"{w[2]*100:.1f}%",
            "delta": f"{w[3]*100:.1f}%",
        },
        "beats_processed": len(internals["beat_states"]),
        "harmonic_cycles": len(internals["contexts"][0]),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Observer Neural Network — Live</title>
  <style>
    body { font-family: monospace; background: #0d0d0d; color: #00ff88;
           max-width: 700px; margin: 60px auto; padding: 20px; }
    h1   { color: #00ffcc; letter-spacing: 2px; }
    button { background: #00ff88; color: #000; border: none; padding: 12px 28px;
             font-size: 16px; cursor: pointer; margin-top: 12px; font-family: monospace; }
    button:hover { background: #00ffcc; }
    pre  { background: #111; padding: 20px; border-left: 3px solid #00ff88;
           white-space: pre-wrap; word-break: break-word; }
    textarea { width: 100%; background: #111; color: #00ff88; border: 1px solid #00ff88;
               padding: 10px; font-family: monospace; font-size: 13px; }
    .label { color: #888; font-size: 12px; margin-top: 16px; display: block; }
  </style>
</head>
<body>
  <h1>⚛ Observer Neural Network</h1>
  <p>Quantum vortex · 108 Hz harmonics · 70 BPM · Star geometry</p>
  <hr style="border-color:#333">

  <span class="label">Optional: paste your own numbers separated by commas
  (leave blank for random demo data)</span>
  <textarea id="vals" rows="3" placeholder="e.g. 0.1, 0.5, -0.3, 0.8, ..."></textarea>

  <br>
  <button onclick="run()">▶ Run Model</button>
  <button onclick="document.getElementById('vals').value=''; run()">⟳ Random Data</button>

  <pre id="out">Press "Run Model" to get a prediction...</pre>

<script>
async function run() {
  document.getElementById('out').textContent = 'Processing through the vortex...';
  const raw = document.getElementById('vals').value.trim();
  const values = raw ? raw.split(',').map(Number).filter(n => !isNaN(n)) : null;

  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ values })
  });
  const data = await res.json();

  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
