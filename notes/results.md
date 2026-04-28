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
