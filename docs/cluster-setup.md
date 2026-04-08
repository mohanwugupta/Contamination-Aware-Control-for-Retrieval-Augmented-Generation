# Cluster Setup — Pre-caching Guide

This document covers everything that must be done **on a login node** (with
internet access) before submitting any SLURM baseline jobs.

Compute nodes at Princeton (Della, Stellar, etc.) have **no outbound internet**.
All models and datasets must be downloaded in advance into the shared scratch
cache so jobs can load them with `HF_HUB_OFFLINE=1`.

---

## Pre-requisites

1. You have SSH access to a Princeton HPC login node.
2. The `rag_baseline` conda environment is already created and all packages are
   installed (see [Environment Setup](#1-environment-setup) below).
3. You have write access to `/scratch/gpfs/JORDANAT/mg9965/`.

---

## Quick Reference — Order of Steps

```
[login node, once]

Step 1  Create conda environment          (first time only)
Step 2  Download the Qwen LLM             (first time only)
Step 3  Pre-cache retrieval/reranking models  ← NEW (fixes bge offline error)
Step 4  Pre-cache benchmark datasets      (first time or after dataset changes)

[then submit jobs]

sbatch slurm/run_baselines.sh
```

---

## 1. Environment Setup

```bash
# Load Anaconda
module load anaconda3/2025.6

# Create the environment (first time only — takes ~10 min)
conda create -n rag_baseline python=3.11 -y
conda activate rag_baseline

# Install the package with all dependencies
pip install -e ".[dev]"
```

---

## 2. Download the Qwen LLM (first time only)

The main generator model is large (~65 GB). Download it once to scratch:

```bash
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache

huggingface-cli download Qwen/Qwen3-32B \
    --local-dir /scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen3-32B
```

> ⚠️  If `huggingface-cli` is not in PATH, run:
> `conda activate rag_baseline && pip install huggingface_hub[cli]`

---

## 3. Pre-cache retrieval and reranking models  ← run this to fix the bge error

The dense retriever uses `BAAI/bge-base-en-v1.5` and the reranker uses
`BAAI/bge-reranker-v2-m3`. Both are loaded via `sentence-transformers` and
must be present in the HF cache before jobs run offline.

```bash
bash slurm/precache_models.sh
```

**What it does:**
- Sets `HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache`
- Downloads weights via both `transformers` (AutoModel) and
  `sentence_transformers` (SentenceTransformer / CrossEncoder) so all cache
  entries are populated
- Prints `✅` confirmation for each model

**Models downloaded:**

| Model | Purpose | Size (approx) |
|-------|---------|--------------|
| `BAAI/bge-base-en-v1.5` | Dense passage/query embeddings | ~440 MB |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker | ~570 MB |

---

## 4. Pre-cache benchmark datasets

```bash
bash slurm/precache_datasets.sh
```

**Datasets downloaded and saved to disk:**

| Dataset | Split | Saved path |
|---------|-------|-----------|
| `google-research-datasets/nq_open` | validation | `…/datasets/nq_open_validation` |
| `yoonsanglee/AmbigDocs` (dev.json) | validation | `…/datasets/ambigdocs_validation` |
| `Salesforce/FaithEval-unanswerable-v1.0` | test | `…/datasets/faitheval_unanswerable_test` |
| `Salesforce/FaithEval-inconsistent-v1.0` | test | `…/datasets/faitheval_inconsistent_test` |
| `Salesforce/FaithEval-counterfactual-v1.0` | test | `…/datasets/faitheval_counterfactual_test` |
| `HanNight/RAMDocs` | test | `…/datasets/ramdocs_test` |

---

## 5. Verify pre-caching (optional smoke check)

Run this on the login node (no GPU required) to confirm the retrieval model
loads from cache before submitting a full job:

```bash
module load anaconda3/2025.6
conda activate rag_baseline

export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import os
os.environ["HF_HOME"] = "/scratch/gpfs/JORDANAT/mg9965/hf_cache"
from sentence_transformers import SentenceTransformer, CrossEncoder

print("Loading dense model …")
m = SentenceTransformer("BAAI/bge-base-en-v1.5",
                        cache_folder=os.environ["HF_HOME"])
v = m.encode(["test sentence"], normalize_embeddings=True)
print(f"  ✅ bge-base-en-v1.5 loaded — embedding dim: {v.shape[1]}")

print("Loading reranker …")
r = CrossEncoder("BAAI/bge-reranker-v2-m3",
                  cache_folder=os.environ["HF_HOME"])
score = r.predict([("query", "passage")])
print(f"  ✅ bge-reranker-v2-m3 loaded — test score: {float(score):.4f}")
PY
```

Both lines should print `✅` with no network calls.

---

## 6. Submit jobs

Once all pre-caching is done:

```bash
# Full baseline sweep (Baselines 0-D on AmbigDocs + LLM-only on NQ-Open)
sbatch slurm/run_baselines.sh

# Smoke test (5 examples, LLM-only on NQ-Open — quick cluster check)
sbatch slurm/smoke_test.sh
```

---

## Troubleshooting

### `LocalEntryNotFoundError` / `OSError: We couldn't connect to huggingface.co`

The model is not in the HF cache. Fix: re-run `bash slurm/precache_models.sh`
on a login node and verify you are writing to the same `HF_HOME` path used in
the SLURM scripts.

### `No sentence-transformers model found with name BAAI/bge-base-en-v1.5. Creating a new one with mean pooling.`

This warning is benign when the model **is** cached — sentence-transformers
falls back to an AutoModel load (which still works offline). If it is followed
by a traceback, the weights are missing; re-run `precache_models.sh`.

### Dataset not found on compute node

Re-run `bash slurm/precache_datasets.sh` on the login node.

### `HF_HUB_OFFLINE=1` disables all downloads

This is intentional. Compute nodes have no internet. All assets must be
pre-cached as described above.

---

## Cache Directory Map

All cached assets live under one root:

```
/scratch/gpfs/JORDANAT/mg9965/
├── hf_cache/
│   ├── models--BAAI--bge-base-en-v1.5/       ← dense model
│   ├── models--BAAI--bge-reranker-v2-m3/     ← reranker
│   ├── transformers/                          ← transformers cache
│   └── datasets/
│       ├── nq_open_validation/
│       ├── ambigdocs_validation/
│       ├── faitheval_unanswerable_test/
│       ├── faitheval_inconsistent_test/
│       ├── faitheval_counterfactual_test/
│       └── ramdocs_test/
└── models/
    └── Qwen--Qwen3-32B/                       ← LLM weights
```
