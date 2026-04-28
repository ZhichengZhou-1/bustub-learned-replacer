# Benchmark Results

## System Info

- Buffer pool size: 64 frames
- Total pages: 200
- Operations per workload: 10,000
- Machine: MacBook Pro M-series

---

## Baseline: LRU Fallback (no model)

| Workload        | Hit Rate | Hits  | Evictions |
| --------------- | -------- | ----- | --------- |
| Sequential Scan | 0%       | 0     | 9,936     |
| Random Access   | 31.58%   | 3,158 | 6,778     |
| Mixed (80/20)   | 84.34%   | 8,434 | 1,502     |

---

## V1: Learned Model (20 epochs, 4 features, BCELoss)

| Workload        | Hit Rate | Hits  | Evictions |
| --------------- | -------- | ----- | --------- |
| Sequential Scan | 26.46%   | 2,646 | 7,290     |
| Random Access   | 31.76%   | 3,176 | 6,760     |
| Mixed (80/20)   | 66.87%   | 6,687 | 3,249     |

### Known issues with V1

- val_acc stuck at 65.28% — model predicts majority class
- Scaler normalization not applied at C++ inference time
- Only 4 features — access history depth too shallow
- Mixed workload regression vs LRU — hot pages not being retained

---

## TODO: V2 improvements

- [ ] Apply StandardScaler normalization in C++ inference
- [ ] Add pos_weight to BCELoss to fix class imbalance
- [ ] Add more features: time_since_last_access, access_count_in_window
- [ ] Increase training data size
- [ ] Tune window size for label generation
- [ ] Compare against ARC baseline (not just LRU)

## V2: Mixed-workload trained model

- Trained only on mixed_trace.csv
- Fixed trace file naming (per-workload traces)

| Workload      | Hit Rate | Delta vs LRU |
| ------------- | -------- | ------------ |
| Sequential    | 26.48%   | +26.48% ✅   |
| Random        | 32.36%   | +0.78% ✅    |
| Mixed (80/20) | 84.26%   | -0.08% ≈     |

### Key finding

Training on representative workload data is critical.
Model trained on mixed trace matches LRU on mixed workload
while still outperforming on sequential scans.

## V3: Belady's approximation via reuse distance regression

- Trained on all 3 workload traces with workload_feat as 5th feature
- Per-workload feature engineering (no boundary crossing)
- HuberLoss regression instead of binary classification
- Evict frame with HIGHEST predicted reuse distance

| Workload      | Hit Rate | Delta vs LRU |
| ------------- | -------- | ------------ |
| Sequential    | 30.87%   | +30.87% ✅   |
| Random        | 31.73%   | +0.15% ≈     |
| Mixed (80/20) | 84.21%   | -0.13% ≈     |

Validation MAE: 14.6 ops (predicting reuse distance)

### Key finding

A single learned model trained on combined traces with workload type as
a feature can match LRU on workloads where LRU is near-optimal (random,
mixed) while dramatically beating LRU on workloads where it fails
(sequential scan).
