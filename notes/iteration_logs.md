# Iteration Log — Learned Buffer Replacement Project

This document tracks every iteration of the model and system, what changed, what we learned, and the metrics at each step. Used as the basis for the paper's methodology and results sections.

---

## Setup

- **Database:** CMU BusTub (forked at master), modified to add `LearnedReplacer` alongside the existing `ArcReplacer`
- **Pool size:** 64 frames
- **Total pages:** 200
- **Operations per workload:** 10,000
- **Workloads:**
  - Sequential scan (pages 0–199 in order, repeated)
  - Random access (uniform over all pages)
  - Mixed 80/20 (80% of accesses hit a hot set of 20 pages, 20% random over remaining 180)

---

## Baseline: LRU Fallback (no model)

The LearnedReplacer's fallback path uses pure LRU. This is the baseline we measure against.

| Workload      | Hit Rate | Hits  | Evictions |
| ------------- | -------- | ----- | --------- |
| Sequential    | 0%       | 0     | 9,936     |
| Random        | 31.58%   | 3,158 | 6,778     |
| Mixed (80/20) | 84.34%   | 8,434 | 1,502     |

**Why these numbers:** LRU is provably worst-case on sequential scans larger than cache (every access misses). Random hits ~64/200=32% (cache size / page count). Mixed is high because the 20-page hot set fits easily in 64 frames.

---

## V1 — Binary classification (will-be-accessed-soon)

**Goal:** simplest possible learned model. Predict whether a page will be accessed within the next 50 ops.

**Architecture:**

- 4 features: `frequency`, `recency` (raw timestamp), `avg_interval`, `access_type`
- Network: Linear(4→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1) → Sigmoid
- Loss: BCELoss (no class weighting)
- Trained on combined trace from all 3 workloads (30k records)
- Eviction: pick lowest score (least likely to be accessed soon)

**Training:**

- Validation accuracy: stuck at **65.9%** across all 20 epochs
- Model collapsed to majority class (always predicting 1)

**Inference issue:** scaler normalization not applied in C++. Raw features fed to model.

**Results:**

| Workload      | Hit Rate | Delta vs LRU   |
| ------------- | -------- | -------------- |
| Sequential    | 26.46%   | +26.46% ✅     |
| Random        | 31.76%   | +0.18% ≈       |
| Mixed (80/20) | 66.87%   | **-17.47%** ❌ |

**Lessons:**

- Without normalization, raw features are too out-of-distribution
- Class imbalance (~65% positive) caused model collapse
- Mixed workload regression — the model evicted hot pages

---

## V2 — Add normalization, fix class imbalance, better feature

**Changes:**

- **Fix 1:** Manually compute mean/std in Python, save to `scaler_params.json`, hardcode in C++ `RunInference()`
- **Fix 2:** Switch to `BCEWithLogitsLoss` with `pos_weight = neg_ratio/pos_ratio` to handle class imbalance
- **Fix 3:** Replace raw `recency` (monotonically growing timestamp) with `time_since_last_access` (meaningful: small = recent)
- Added dropout (0.2), weight decay (1e-4), LR scheduler

**Training:**

- Validation accuracy: jumped from 65.9% → **79.5%**
- Per-class: pos_acc=81%, neg_acc=75%

**Results:**

| Workload      | Hit Rate | Delta vs LRU     |
| ------------- | -------- | ---------------- |
| Sequential    | 30.87%   | +30.87% ✅       |
| Random        | 32.07%   | +0.49% ≈         |
| Mixed (80/20) | 13.48%   | **-70.86%** ❌❌ |

**Catastrophic regression on mixed.** Why?

- Debug output revealed all eviction scores = 1.0 — model saturated
- Feature distributions at inference (very large `avg_interval`, `frequency`) were far outside training distribution
- The training trace had different frame access patterns than what the model produced at inference time
- Even with normalization, OOD inputs caused sigmoid saturation

**Lesson:** training data distribution must match inference distribution. Mixing all 3 workloads dilutes per-workload signal.

---

## V3 — Train only on mixed workload

**Changes:**

- Train only on `mixed_trace.csv` instead of combined trace
- Everything else same as V2

**Training:**

- Val acc 79% on mixed-only data
- Scaler reflects mixed distribution

**Results:**

| Workload      | Hit Rate | Delta vs LRU |
| ------------- | -------- | ------------ |
| Sequential    | 26.48%   | +26.48% ✅   |
| Random        | 32.36%   | +0.78% ≈     |
| Mixed (80/20) | 84.26%   | -0.08% ≈     |

**Solid baseline result.** Model essentially recovers LRU on mixed and beats it on sequential.

**Lesson:** training must use representative data. But "training only on mixed" is workload-specific — not a general solution.

---

## V4 — Belady's approximation via reuse distance regression

**Conceptual change:** instead of binary classification ("will this be accessed soon?"), predict the **exact reuse distance** (how many ops until next access). Then evict the frame with the **largest** predicted reuse distance — this directly approximates Bélady's optimal algorithm.

**Changes:**

- Label: continuous reuse distance (capped at MAX_DIST=500)
- Loss: `HuberLoss` (robust to outliers, better than MSE for skewed distributions)
- Removed sigmoid from model — raw regression output
- Added 5th feature: `workload_feat` (0=sequential, 1=random, 2=mixed)
- Train on **all 3 traces combined** with workload as a feature
- Evict logic flipped: pick HIGHEST score (largest predicted reuse distance)

**Critical bug discovered during this iteration:**

- `df.sort_values("timestamp")` was mixing all 3 workloads' timestamps together
- Each trace starts timestamps at 1, so combining them caused features like `frequency` and `time_since_last_access` to be computed across workload boundaries
- **Fix:** process each workload separately, then concat. Per-workload feature engineering.

**Training results:**

- Validation MAE: **14.6 ops** in original reuse distance units (vs typical reuse distance of 30-100)
- val_loss: 0.062 (down from 0.30 in earlier versions)
- Per-workload label means: sequential=64, random=59, mixed=30 — meaningful separation

**Results:**

| Workload      | Hit Rate | Delta vs LRU |
| ------------- | -------- | ------------ |
| Sequential    | 30.87%   | +30.87% ✅   |
| Random        | 31.73%   | +0.15% ≈     |
| Mixed (80/20) | 84.21%   | -0.13% ≈     |

**Final result:**

- Single model competitive on ALL three workloads
- No regression on any workload
- Substantially beats LRU on sequential
- Matches LRU on random (which is at theoretical ceiling) and mixed (where LRU is near-optimal)

---

## Cumulative results summary

| Workload      | LRU    | V1     | V2     | V3     | V4 (final) |
| ------------- | ------ | ------ | ------ | ------ | ---------- |
| Sequential    | 0%     | 26.46% | 30.87% | 26.48% | **30.87%** |
| Random        | 31.58% | 31.76% | 32.07% | 32.36% | **31.73%** |
| Mixed (80/20) | 84.34% | 66.87% | 13.48% | 84.26% | **84.21%** |

---

## Key takeaways for the paper

1. **Distribution alignment matters more than model architecture.** The biggest hit-rate gains came from fixing data pipeline issues (per-workload feature engineering, training data representativeness), not from model sophistication.

2. **Regression beats classification for cache replacement.** Predicting continuous reuse distance gave a more useful signal than binary "hot/cold" classification, and naturally approximates Bélady's optimal.

3. **Workload identification as a feature is powerful.** Adding `workload_feat` allowed a single model to learn distinct strategies per workload type.

4. **Class imbalance and saturation are real failure modes.** Without `pos_weight` (V1) or with OOD features (V2), the model collapsed to constant predictions. Debug output of per-frame scores was essential to diagnose.

5. **The C++/Python boundary needs careful normalization handoff.** ONNX exports the model graph but not the preprocessing. The scaler params must be hardcoded or loaded explicitly in the inference path.

---

## Future work (acknowledged limitations)

- **No comparison against ARC** — the actual BusTub default. Random/mixed results would be closer to ARC than LRU.
- **No comparison against Belady's optimal** — we should measure as % of optimal hit rate.
- **Static model** — no online retraining when access patterns shift.
- **Limited workload diversity** — only 3 synthetic patterns. Real DB workloads (TPC-C, YCSB) untested.
- **Single seed runs** — should report mean ± stddev across multiple random seeds.
- **Feedback loop not addressed** — training data was generated by LRU; the model's own evictions create different patterns at inference.
