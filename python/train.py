import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

# ─── 1. Load trace ────────────────────────────────────────────────────────────
TRACE_PATH = "../build/access_trace.csv"
df = pd.read_csv(TRACE_PATH)
print(f"Loaded {len(df)} access records")
print(df.head())

# ─── 2. Feature engineering ───────────────────────────────────────────────────
# For each access event, compute per-frame rolling features up to that point.
# We compute: frequency, recency, avg_interval, access_type
# Label: will this frame be accessed again within the next N steps? (1=yes, 0=no)

WINDOW = 50  # look-ahead window for label generation

df = df.sort_values("timestamp").reset_index(drop=True)

# Compute rolling features per frame
df["frequency"] = df.groupby("frame_id").cumcount() + 1
df["recency"] = df["timestamp"]
df["prev_timestamp"] = df.groupby("frame_id")["timestamp"].shift(1)
df["interval"] = df["timestamp"] - df["prev_timestamp"]
df["avg_interval"] = df.groupby("frame_id")["interval"].transform(
    lambda x: x.expanding().mean()
)
df["avg_interval"] = df["avg_interval"].fillna(0)

# Label: 1 if this frame appears again within next WINDOW accesses
future_accesses = set()
labels = []
for i, row in df.iterrows():
    window_frames = set(df.loc[i + 1 : i + WINDOW, "frame_id"].values)
    labels.append(1 if row["frame_id"] in window_frames else 0)

df["label"] = labels
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

# ─── 3. Build features and labels ─────────────────────────────────────────────
FEATURES = ["frequency", "recency", "avg_interval", "access_type"]
X = df[FEATURES].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler stats for reference
print(f"\nScaler means: {scaler.mean_}")
print(f"Scaler stds:  {scaler.scale_}")

# Train/val split (80/20)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
)


# ─── 4. Define model ──────────────────────────────────────────────────────────
class ReplacerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = ReplacerNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# ─── 5. Train ─────────────────────────────────────────────────────────────────
EPOCHS = 20
print("\nTraining...")
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

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_acc = ((val_pred > 0.5) == y_val_t).float().mean().item()

    print(
        f"Epoch {epoch+1:2d} | loss: {total_loss/len(train_loader):.4f} | val_acc: {val_acc:.4f}"
    )

# ─── 6. Export to ONNX ────────────────────────────────────────────────────────
model.eval()
dummy_input = torch.randn(1, 4)
onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["features"],
    output_names=["score"],
    dynamic_axes={"features": {0: "batch"}},
    opset_version=18,
)
print(f"\nExported model to {onnx_path}")

# ─── 7. Quick sanity check ────────────────────────────────────────────────────
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path)
test_input = np.array([[1.0, 100.0, 5.0, 1.0]], dtype=np.float32)
result = sess.run(["score"], {"features": test_input})
print(f"Test inference score: {result[0][0][0]:.4f}  (should be between 0 and 1)")
print("\nDone! Copy model.onnx to the build directory before running BusTub.")
