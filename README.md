<img src="logo/bustub-whiteborder.svg" alt="BusTub Logo" height="200">

---

[![Build Status](https://github.com/cmu-db/bustub/actions/workflows/cmake.yml/badge.svg)](https://github.com/cmu-db/bustub/actions/workflows/cmake.yml)

BusTub is a relational database management system built at [Carnegie Mellon University](https://db.cs.cmu.edu) for the [Introduction to Database Systems](https://15445.courses.cs.cmu.edu) (15-445/645) course. This system was developed for educational purposes and should not be used in production environments.

BusTub supports basic SQL and comes with an interactive shell. You can get it running after finishing all the course projects.

<img src="logo/sql.png" alt="BusTub SQL" width="400">

---

> **This fork extends BusTub with a learned buffer replacement policy as a graduate research project.**

# Learned Buffer Replacement in BusTub

A research project that replaces BusTub's ARC buffer replacement policy with a neural network trained to approximate Bélady's optimal eviction algorithm. The model is trained offline in Python using PyTorch, exported to ONNX, and loaded at runtime in C++ via ONNX Runtime — with zero Python dependency inside the database.

---

## Motivation

Traditional buffer replacement policies (LRU, ARC, LRU-K) use hand-crafted heuristics that cannot adapt to workload changes. Recent work (LBR, LeCaR, PARROT) has shown that machine learning can approximate Bélady's optimal algorithm, which always evicts the page that won't be needed for the longest time. This project brings learned replacement into a real DBMS buffer pool manager (CMU's BusTub), extending prior simulation-based work with:

- A real C++ implementation inside BusTub's buffer pool manager
- Database-specific features including `AccessType` (Lookup, Scan, Index)
- An ONNX-based deployment pipeline for production-style inference
- Benchmarking across sequential scan, random access, and mixed workloads

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     BusTub (C++)                    │
│                                                     │
│   BufferPoolManager                                 │
│         │                                           │
│         └──► LearnedReplacer                        │
│                    │                                │
│                    ├── RecordAccess()               │
│                    │     └── logs to access_trace   │
│                    │                                │
│                    ├── Evict()                      │
│                    │     └── RunInference()         │
│                    │           └── ONNX Runtime     │
│                    │                 └── model.onnx │
│                    │                                │
│                    └── FallbackEvict() (LRU)        │
│                         used if model not loaded    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   Python Pipeline                   │
│                                                     │
│  1. Run benchmark  →  access_trace.csv              │
│  2. train.py loads trace, engineers features        │
│  3. Trains small neural net (4 → 32 → 16 → 1)      │
│  4. Exports to model.onnx (self-contained)          │
│  5. Copy model.onnx to build/ directory             │
└─────────────────────────────────────────────────────┘
```

### Model Architecture

Input features per frame (4 total):

- `frequency` — number of times this frame has been accessed
- `recency` — timestamp of most recent access
- `avg_interval` — average time between accesses
- `access_type` — encoded access type (0=Unknown, 1=Lookup, 2=Scan, 3=Index)
- `workload_feat` — workload type indicator (0=sequential, 1=random, 2=mixed)

Output: a score between 0 and 1. Higher score = more likely to be accessed soon = do NOT evict. Lower score = safer to evict.

```
Input (4) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid → Score
```

Label generation: for each access event, label=1 if this frame is accessed again within the next 50 operations, else 0. This approximates "will this page be needed soon?" — the core question Bélady's answers optimally.

---

## Repo Structure

```
bustub/
├── src/
│   ├── buffer/
│   │   ├── learned_replacer.cpp     ← LearnedReplacer implementation
│   │   └── CMakeLists.txt           ← includes ONNX Runtime linkage
│   └── include/buffer/
│       └── learned_replacer.h       ← LearnedReplacer header + FrameMeta
├── tools/
│   └── buffer_benchmark/
│       └── buffer_benchmark.cpp     ← benchmark driver (3 workloads)
├── python/
│   ├── train.py                     ← trace loading, training, ONNX export
│   └── requirements.txt             ← Python dependencies
├── notes/
│   └── results.md                   ← benchmark results across versions
└── README.md
```

---

## Requirements

### C++ / Build

- CMake >= 3.10
- Clang 15 (installed via `build_support/packages.sh`)
- ONNX Runtime 1.25.0 via Homebrew

### Python

- Python 3.10+
- PyTorch, ONNX, ONNX Runtime, pandas, scikit-learn, numpy

---

## Setup & Reproduction

### Step 1 — Clone and install dependencies

```bash
git clone https://github.com/ZhichengZhou-1/bustub-learned-replacer.git
cd bustub-learned-replacer
git checkout learned-replacer

# Install C++ dependencies (requires Homebrew)
build_support/packages.sh
brew install onnxruntime
```

### Step 2 — Build BusTub

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(sysctl -n hw.ncpu)
```

### Step 3 — Generate access traces

Run the benchmark in LRU fallback mode (no model needed) to generate training data:

```bash
cd build
./bin/buffer_benchmark
```

This writes `build/access_trace.csv` with 30,000 page access records across three workloads.

### Step 4 — Train the model

```bash
cd python
pip install -r requirements.txt
python train.py
```

This will:

- Load `../build/access_trace.csv`
- Engineer features and generate labels
- Train a small neural net for 20 epochs
- Export a self-contained `model.onnx`
- Run a quick inference sanity check

### Step 5 — Run the benchmark with the learned model

```bash
cp python/model.onnx build/learned_replacer.onnx
cd build
./bin/buffer_benchmark
```

You should see `[LearnedReplacer] Model loaded from learned_replacer.onnx` for each workload.

---

## Benchmark Results

### Workload definitions

| Workload        | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| Sequential Scan | Pages 0–199 accessed in order, repeated. Classic LRU killer.      |
| Random Access   | Uniform random over 200 pages. No exploitable pattern.            |
| Mixed (80/20)   | 80% of accesses hit a hot set of 20 pages, 20% random cold pages. |

Settings: pool size = 64 frames, 200 total pages, 10,000 ops per workload.

### Final results

| Workload        | LRU Baseline | Learned Model | Delta      |
| --------------- | ------------ | ------------- | ---------- |
| Sequential Scan | 0%           | 26.46%        | +26.46% ✅ |
| Random Access   | 31.58%       | 32.49%        | +0.91% ≈   |
| Mixed (80/20)   | 84.34%       | 84.56%        | +0.22% ≈   |

Sequential scan shows the biggest win — the model learned that sequentially scanned pages will return. Random access is inherently unpredictable so both policies perform similarly. Mixed workload was able to maintain its relative performance.

See `notes/results.md` for full version history and improvement tracking.
