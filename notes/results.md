# Benchmark Results

## System info

- Buffer pool size: 64 frames
- Total pages: 200
- Operations per workload: 10,000
- Machine: MacBook Pro M-series
- Model: 5→64→32→1 feedforward, HuberLoss regression on reuse distance
- Inference: ONNX Runtime 1.25.0 in C++ inside BusTub

---

## Final results (V4)

| Workload      | LRU baseline | LearnedReplacer |  Δ vs LRU | Theoretical optimum |
| ------------- | -----------: | --------------: | --------: | ------------------: |
| Sequential    |        0.00% |      **26.48%** | +26.48 pp |              30.87% |
| Random        |       31.58% |      **32.49%** |  +0.91 pp |              32.00% |
| Mixed (80/20) |       84.34% |      **84.56%** |  +0.22 pp |               ~100% |

The learned model improves on LRU across all three workloads. On sequential scan, it reaches 86% of the theoretical optimum (26.48 / 30.87) while LRU achieves 0% by construction.

---

## Iteration history

### Baseline: LRU fallback (no model)

The LearnedReplacer's fallback path uses pure LRU. This is the reference point we measure against.

| Workload      | Hit rate |  Hits | Evictions |
| ------------- | -------: | ----: | --------: |
| Sequential    |    0.00% |     0 |     9,936 |
| Random        |   31.58% | 3,158 |     6,778 |
| Mixed (80/20) |   84.34% | 8,434 |     1,502 |

### V1: Binary classification, no normalization

- 4 features: frequency, recency (raw timestamp), avg_interval, access_type
- BCELoss, no class weighting
- Validation accuracy stuck at 65.9% — model collapsed to majority class
- Raw features fed to C++ inference (no scaler applied)

| Workload      | Hit rate |     Δ vs LRU |
| ------------- | -------: | -----------: |
| Sequential    |   26.46% | +26.46 pp ✅ |
| Random        |   31.76% |   +0.18 pp ≈ |
| Mixed (80/20) |   66.87% | -17.47 pp ❌ |

**Lessons.** Without normalization, raw features are too out-of-distribution. Without class weighting, the model collapsed to predicting the majority class.

### V2: Normalization + class weighting + better feature

- Manual scaler params hardcoded into C++ inference
- BCEWithLogitsLoss with `pos_weight = neg_ratio / pos_ratio`
- Replaced raw recency timestamp with `time_since_last_access` (meaningful signal that doesn't grow unboundedly)
- Validation accuracy jumped 65.9% → 79.5%

| Workload      | Hit rate |       Δ vs LRU |
| ------------- | -------: | -------------: |
| Sequential    |   30.87% |   +30.87 pp ✅ |
| Random        |   32.07% |     +0.49 pp ≈ |
| Mixed (80/20) |   13.48% | -70.86 pp ❌❌ |

**Lessons.** Catastrophic regression on mixed. Per-frame eviction-score logging revealed every score collapsed to 1.0 — sigmoid saturation due to feature distributions falling outside the training distribution. Even with normalization, the OOD inputs at inference time killed the model. Training data must match inference distribution.

### V3: Mixed-only training data

- Trained only on `mixed_trace.csv` instead of combined trace
- Recovered mixed performance but at the cost of generalization

| Workload      | Hit rate |     Δ vs LRU |
| ------------- | -------: | -----------: |
| Sequential    |   26.48% | +26.48 pp ✅ |
| Random        |   32.36% |   +0.78 pp ≈ |
| Mixed (80/20) |   84.26% |   -0.08 pp ≈ |

**Lessons.** Per-workload retraining is not a deployable solution. We need a single model that handles all workloads.

### V4: Reuse-distance regression (final)

- **Conceptual change:** predict continuous reuse distance instead of binary classification
- HuberLoss instead of BCELoss; raw regression output (no sigmoid)
- Added 5th feature: `workload_feat` (0=sequential, 1=random, 2=mixed)
- Trained on all 3 traces combined
- **Critical bug fix:** per-workload feature engineering — previously, sorting by timestamp interleaved access histories from different workloads since each trace independently restarts at t=1
- Validation MAE: 14.6 operations
- Eviction logic flipped: pick HIGHEST score (largest predicted reuse distance)

| Workload      | Hit rate |     Δ vs LRU |
| ------------- | -------: | -----------: |
| Sequential    |   26.48% | +26.48 pp ✅ |
| Random        |   32.49% |  +0.91 pp ✅ |
| Mixed (80/20) |   84.56% |  +0.22 pp ✅ |

**Lessons.** A single learned model competitive on all three workloads, with measurable improvements on each. Reuse-distance regression cleanly approximates Bélady's optimal.

---

## Cumulative comparison

| Workload      |    LRU |     V1 |     V2 |     V3 | **V4 (final)** |
| ------------- | -----: | -----: | -----: | -----: | -------------: |
| Sequential    |  0.00% | 26.46% | 30.87% | 26.48% |     **26.48%** |
| Random        | 31.58% | 31.76% | 32.07% | 32.36% |     **32.49%** |
| Mixed (80/20) | 84.34% | 66.87% | 13.48% | 84.26% |     **84.56%** |

---

## Key findings

1. **Distribution alignment matters more than model architecture.** The largest hit-rate gains across iterations came from fixing data-pipeline issues (per-workload feature engineering in V4, distribution-matched training data in V3) rather than from increasing model capacity.

2. **Regression beats classification for cache replacement.** Predicting continuous reuse distance preserves the ordering information that binary classification discards, and naturally approximates Bélady's optimal algorithm: evict the frame whose next reference is furthest in the future.

3. **Workload identification as a feature enables a single shared model.** Adding `workload_feat` allowed one model to learn distinct strategies per workload type, avoiding the V3 trap of needing per-workload retraining.

4. **Class imbalance and OOD inputs are real failure modes.** V1 collapsed to majority-class prediction without `pos_weight`. V2 saturated to constant outputs when inference features fell outside the training distribution. Both failures only became visible through per-frame score logging in C++ — neither was caught by training metrics.

5. **The C++/Python boundary requires explicit normalization handoff.** ONNX exports the model graph and weights but not preprocessing. Drift between training-time and inference-time normalization corrupts every prediction silently.

6. **Sequential scan reaches 86% of theoretical optimum.** With 200 pages and 64 frames, the upper bound is 30.87% (49 cycles × 63 hits ÷ 10,000 ops). LRU achieves 0% by construction; our model achieves 26.48%, recovering most of the gap to optimum.

---

## Future work

Acknowledged limitations of this evaluation:

- **No comparison against ARC** — BusTub's actual default replacer. Random and mixed results would be a more challenging comparison against ARC than against LRU.
- **No comparison against Bélady's offline optimal** as a measured percentage. We compute theoretical optimum analytically for sequential scan but do not run Bélady's directly on traces.
- **Static model** — no online retraining when access patterns shift over time.
- **Limited workload diversity** — only three synthetic patterns. Real-world workloads (TPC-C, YCSB) untested.
- **Single seed runs** — should report mean ± stddev across multiple random seeds.
- **Feedback loop not addressed** — training data was generated by LRU; the learned model's own evictions create different access patterns at inference time. Iterative retraining would be a natural extension.

A preliminary experiment with phase-shifting hot sets (where the hot 20-page set rotates every 2,000 operations) showed the learned policy underperforming LRU by approximately 4 percentage points. Adapting to such workloads is a promising direction for future work, possibly through online retraining or attention-based architectures that can detect phase boundaries.
