import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort
import json

# ─── 1. Load all three traces ─────────────────────────────────────────────────
traces = []
workload_map = {"sequential": 0.0, "random": 1.0, "mixed": 2.0}
for name in ["sequential_trace.csv", "random_trace.csv", "mixed_trace.csv"]:
    t = pd.read_csv(f"../build/{name}")
    t["workload_feat"] = workload_map[name.split("_")[0]]
    traces.append(t)


df = pd.concat(traces, ignore_index=True)
print(f"Loaded {len(df)} total records across 3 workloads")
print(df.head())

# ─── 2. Feature engineering (per-workload to avoid boundary crossing) ─────────
processed = []
for wl_name, wl_val in workload_map.items():
    sub = df[df["workload_feat"] == wl_val].copy()
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    sub["frequency"] = sub.groupby("frame_id").cumcount() + 1
    sub["prev_timestamp"] = sub.groupby("frame_id")["timestamp"].shift(1)
    sub["time_since_last_access"] = (sub["timestamp"] - sub["prev_timestamp"]).fillna(
        999
    )
    sub["interval"] = sub["timestamp"] - sub["prev_timestamp"]
    sub["avg_interval"] = (
        sub.groupby("frame_id")["interval"]
        .transform(lambda x: x.expanding().mean())
        .fillna(0)
    )
    processed.append(sub)

df = pd.concat(processed, ignore_index=True)

# ─── 3. Generate reuse distance labels (Belady's approximation) ───────────────
print("\nGenerating reuse distance labels...")
frame_ids = df["frame_id"].values
labels = []
MAX_DIST = 500  # cap — if not accessed within 500 ops, treat as MAX_DIST

for i in range(len(df)):
    next_access = MAX_DIST
    for j in range(i + 1, min(i + MAX_DIST, len(df))):
        if frame_ids[j] == frame_ids[i]:
            next_access = j - i
            break
    labels.append(float(next_access))

df["label"] = labels
print(f"Reuse distance stats:\n{df['label'].describe()}")

# Debug: check label distribution per workload
for wl_name, wl_val in workload_map.items():
    mask = df["workload_feat"] == wl_val
    print(
        f"{wl_name}: {mask.sum()} rows, label mean = {df.loc[mask, 'label'].mean():.1f}"
    )

# ─── 4. Build feature matrix ──────────────────────────────────────────────────
FEATURES = [
    "frequency",
    "time_since_last_access",
    "avg_interval",
    "access_type",
    "workload_feat",
]
X = df[FEATURES].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

# Normalize features and save params for C++ inference
means = X.mean(axis=0)
stds = X.std(axis=0)
stds[stds == 0] = 1.0

scaler_params = {"means": means.tolist(), "stds": stds.tolist(), "features": FEATURES}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"\nScaler params saved to scaler_params.json")
print(f"  means: {means}")
print(f"  stds:  {stds}")

X_norm = (X - means) / stds

# Also normalize labels to help training
y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

# Train/val split (80/20)
split = int(0.8 * len(X_norm))
X_train, X_val = X_norm[:split], X_norm[split:]
y_train, y_val = y_norm[:split], y_norm[split:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
)


# ─── 5. Define model (regression — no sigmoid) ───────────────────────────────
class ReplacerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # raw reuse distance prediction (no sigmoid)
        )

    def forward(self, x):
        return self.net(x)


model = ReplacerNet()
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ─── 6. Train ─────────────────────────────────────────────────────────────────
EPOCHS = 40
print("\nTraining...")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
        val_mae = (val_pred * y_std - y_val_t * y_std).abs().mean().item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:2d} | loss: {total_loss/len(train_loader):.4f} "
            f"| val_loss: {val_loss:.4f} | val_mae: {val_mae:.1f} ops"
        )

print(f"\nBest val_loss: {best_val_loss:.4f}")

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# ─── 7. Export to ONNX ────────────────────────────────────────────────────────
onnx_path = "model.onnx"
dummy_input = torch.randn(1, 5)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["features"],
    output_names=["score"],
    dynamic_axes={"features": {0: "batch"}},
    opset_version=18,
)

model_onnx = onnx.load(onnx_path)
onnx.save(model_onnx, onnx_path, save_as_external_data=False)
print(f"\nExported self-contained model to {onnx_path}")

# ─── 8. Sanity check ──────────────────────────────────────────────────────────
sess = ort.InferenceSession(onnx_path)
test_raw = np.array([[10.0, 5.0, 8.0, 1.0, 2.0]], dtype=np.float32)
test_norm = ((test_raw - means) / stds).astype(np.float32)
result = sess.run(["score"], {"features": test_norm})
print(f"Test inference (normalized reuse distance): {result[0][0][0]:.4f}")
print("Higher value = evict this frame (longer until next access)")

print(
    """
Done!
Next steps:
  1. Update C++ MEANS/STDS to 5 values from scaler_params.json
  2. Update C++ Evict() to pick HIGHEST score (not lowest)
  3. Copy model.onnx to build/learned_replacer.onnx
  4. Run ./bin/buffer_benchmark
"""
)
