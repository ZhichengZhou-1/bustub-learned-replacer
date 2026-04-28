import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort
import json
import os

# ─── 1. Load trace ────────────────────────────────────────────────────────────
TRACE_PATH = "../build/mixed_trace.csv"
df = pd.read_csv(TRACE_PATH)
print(f"Loaded {len(df)} access records")
print(df.head())

# ─── 2. Feature engineering ───────────────────────────────────────────────────
# Fix 3: Replace raw recency timestamp with time_since_last_access
# This is meaningful (small = recently used) vs raw timestamp (grows forever)

WINDOW = 50  # look-ahead window for label generation

df = df.sort_values("timestamp").reset_index(drop=True)

# Feature: cumulative access count per frame
df["frequency"] = df.groupby("frame_id").cumcount() + 1

# Feature: time since last access (Fix 3 — replaces raw recency)
df["prev_timestamp"] = df.groupby("frame_id")["timestamp"].shift(1)
df["time_since_last_access"] = df["timestamp"] - df["prev_timestamp"]
df["time_since_last_access"] = df["time_since_last_access"].fillna(
    999
)  # large value for first access

# Feature: rolling average interval between accesses
df["interval"] = df["timestamp"] - df["prev_timestamp"]
df["avg_interval"] = df.groupby("frame_id")["interval"].transform(
    lambda x: x.expanding().mean()
)
df["avg_interval"] = df["avg_interval"].fillna(0)

# Feature: access type (already encoded as int)
# access_type: 0=Unknown, 1=Lookup, 2=Scan, 3=Index

# ─── 3. Generate labels ───────────────────────────────────────────────────────
print("\nGenerating labels...")
labels = []
frame_ids = df["frame_id"].values
for i in range(len(df)):
    window_end = min(i + WINDOW, len(df))
    window_frames = set(frame_ids[i + 1 : window_end])
    labels.append(1 if frame_ids[i] in window_frames else 0)

df["label"] = labels
print(f"Label distribution:\n{df['label'].value_counts()}")
pos_ratio = df["label"].mean()
neg_ratio = 1 - pos_ratio
print(f"Positive ratio: {pos_ratio:.3f}, Negative ratio: {neg_ratio:.3f}")

# ─── 4. Build feature matrix ──────────────────────────────────────────────────
FEATURES = ["frequency", "time_since_last_access", "avg_interval", "access_type"]
X = df[FEATURES].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

# Fix 1: Manual normalization — compute mean/std and save to JSON
# so C++ can apply the exact same normalization at inference time
means = X.mean(axis=0)
stds = X.std(axis=0)
stds[stds == 0] = 1.0  # avoid division by zero

scaler_params = {"means": means.tolist(), "stds": stds.tolist(), "features": FEATURES}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"\nScaler params saved to scaler_params.json")
print(f"  means: {means}")
print(f"  stds:  {stds}")

X_norm = (X - means) / stds

# Train/val split (80/20)
split = int(0.8 * len(X_norm))
X_train, X_val = X_norm[:split], X_norm[split:]
y_train, y_val = y[:split], y[split:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
)


# ─── 5. Define model ──────────────────────────────────────────────────────────
class ReplacerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = ReplacerNet()

# Fix 2: pos_weight to handle class imbalance
# if 65% are positive, weight negative class higher to force learning
pos_weight = torch.tensor([neg_ratio / pos_ratio])
print(f"\nUsing pos_weight: {pos_weight.item():.3f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Use Sigmoid separately since BCEWithLogitsLoss expects raw logits
class ReplacerNetLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # no sigmoid here — BCEWithLogitsLoss handles it
        )

    def forward(self, x):
        return self.net(x)


model = ReplacerNetLogits()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ─── 6. Train ─────────────────────────────────────────────────────────────────
EPOCHS = 40
print("\nTraining...")
best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc = (val_preds == y_val_t).float().mean().item()

        # Also compute per-class accuracy
        pos_mask = y_val_t == 1
        neg_mask = y_val_t == 0
        pos_acc = (
            (val_preds[pos_mask] == y_val_t[pos_mask]).float().mean().item()
            if pos_mask.any()
            else 0
        )
        neg_acc = (
            (val_preds[neg_mask] == y_val_t[neg_mask]).float().mean().item()
            if neg_mask.any()
            else 0
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:2d} | loss: {total_loss/len(train_loader):.4f} "
            f"| val_acc: {val_acc:.4f} | pos_acc: {pos_acc:.4f} | neg_acc: {neg_acc:.4f}"
        )

print(f"\nBest val_acc: {best_val_acc:.4f}")

# Load best model for export
model.load_state_dict(torch.load("best_model.pt"))
model.eval()


# ─── 7. Export to ONNX ────────────────────────────────────────────────────────
# Wrap with sigmoid for ONNX export so C++ gets 0-1 scores directly
class ReplacerNetWithSigmoid(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base(x))


export_model = ReplacerNetWithSigmoid(model)
export_model.eval()

onnx_path = "model.onnx"
dummy_input = torch.randn(1, 4)

torch.onnx.export(
    export_model,
    dummy_input,
    onnx_path,
    input_names=["features"],
    output_names=["score"],
    dynamic_axes={"features": {0: "batch"}},
    opset_version=18,
)

# Save as self-contained single file
model_onnx = onnx.load(onnx_path)
onnx.save(model_onnx, onnx_path, save_as_external_data=False)
print(f"\nExported self-contained model to {onnx_path}")

# ─── 8. Sanity check ──────────────────────────────────────────────────────────
sess = ort.InferenceSession(onnx_path)

# Test with a normalized input
test_raw = np.array([[100.0, 2.0, 5.0, 1.0]], dtype=np.float32)
test_norm = ((test_raw - means) / stds).astype(np.float32)
result = sess.run(["score"], {"features": test_norm})
print(f"Test inference score: {result[0][0][0]:.4f}  (should be between 0 and 1)")

print(
    """
Done!
Next steps:
  1. Copy model.onnx to build/learned_replacer.onnx
  2. Copy scaler_params.json to build/scaler_params.json
  3. Update C++ RunInference() to load and apply scaler_params.json
  4. Run ./bin/buffer_benchmark
"""
)
