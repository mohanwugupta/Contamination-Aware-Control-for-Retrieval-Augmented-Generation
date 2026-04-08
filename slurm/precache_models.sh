#!/bin/bash
# =============================================================================
# Pre-cache HuggingFace retrieval & reranking models for offline cluster use
#
# Run this on a LOGIN NODE (with internet access) BEFORE submitting SLURM jobs.
# Models are downloaded into the shared HF cache on scratch so that compute
# nodes, which have no outbound internet, can load them in offline mode.
#
# Models downloaded:
#   • BAAI/bge-base-en-v1.5     — dense embedding model (sentence-transformers)
#   • BAAI/bge-reranker-v2-m3   — cross-encoder reranker (sentence-transformers)
#
# Usage:
#   bash slurm/precache_models.sh
#
# Pre-requisites:
#   • Run on a Princeton cluster login node (della, stellar, or similar)
#   • conda env rag_baseline must already be created
#   • sentence-transformers, transformers, huggingface_hub installed in that env
# =============================================================================

set -eo pipefail

# Hide all GPUs: this script only downloads files to disk — it must not try
# to allocate VRAM.  Login nodes may have a GPU visible, and both AutoModel
# and SentenceTransformer default to CUDA when available; loading the same
# ~440 MB model twice (once via transformers, once via sentence-transformers)
# exhausts GPU memory on shared login-node GPUs.
export CUDA_VISIBLE_DEVICES=""

echo "=========================================="
echo "Pre-caching retrieval / reranking models"
echo "=========================================="

# ---------------------------------------------------------------------------
# 0. Cache paths — must match SLURM scripts exactly
# ---------------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "HF_HOME:            $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# ---------------------------------------------------------------------------
# 1. Activate conda environment
# ---------------------------------------------------------------------------
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate rag_baseline

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ---------------------------------------------------------------------------
# 2. Download models via Python
#    sentence-transformers wraps transformers; we pre-download via both the
#    transformers AutoModel stack (which populates transformers cache) AND
#    sentence_transformers.SentenceTransformer / CrossEncoder (which also look
#    up the cache and may write extra config files).
# ---------------------------------------------------------------------------
python - <<'PY'
import os
import sys

HF_HOME = os.environ["HF_HOME"]

DENSE_MODEL     = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"

# ------------------------------------------------------------------ helpers --

def precache_via_transformers(model_id: str) -> None:
    """Download all transformers config / tokenizer / model weights."""
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    print(f"  [transformers] Downloading config …")
    AutoConfig.from_pretrained(model_id, cache_dir=HF_HOME)
    print(f"  [transformers] Downloading tokenizer …")
    AutoTokenizer.from_pretrained(model_id, cache_dir=HF_HOME)
    print(f"  [transformers] Downloading model weights …")
    # device_map="cpu" prevents AutoModel from moving weights to GPU
    AutoModel.from_pretrained(model_id, cache_dir=HF_HOME, device_map="cpu")


def precache_sentence_transformer(model_id: str) -> None:
    """Load via sentence-transformers to write its extra module config."""
    from sentence_transformers import SentenceTransformer
    print(f"  [sentence-transformers] Loading SentenceTransformer …")
    # device="cpu" prevents sentence-transformers from allocating VRAM
    SentenceTransformer(model_id, cache_folder=HF_HOME, device="cpu")


def precache_cross_encoder(model_id: str) -> None:
    """Load via sentence-transformers CrossEncoder to write its config."""
    from sentence_transformers import CrossEncoder
    print(f"  [sentence-transformers] Loading CrossEncoder …")
    # device="cpu" prevents the cross-encoder from allocating VRAM
    CrossEncoder(model_id, cache_folder=HF_HOME, device="cpu")


# ------------------------------------------------------------------ models --

print("=" * 58)
print(f"Dense embedding model: {DENSE_MODEL}")
print("=" * 58)
try:
    precache_via_transformers(DENSE_MODEL)
    precache_sentence_transformer(DENSE_MODEL)
    print(f"✅ {DENSE_MODEL} — cached successfully\n")
except Exception as exc:
    print(f"❌ {DENSE_MODEL} — FAILED: {exc}", file=sys.stderr)
    sys.exit(1)

print("=" * 58)
print(f"Reranker model: {RERANKER_MODEL}")
print("=" * 58)
try:
    precache_via_transformers(RERANKER_MODEL)
    precache_cross_encoder(RERANKER_MODEL)
    print(f"✅ {RERANKER_MODEL} — cached successfully\n")
except Exception as exc:
    print(f"❌ {RERANKER_MODEL} — FAILED: {exc}", file=sys.stderr)
    sys.exit(1)

# ----------------------------------------------------------------- summary --

print("=" * 58)
print("✅ All retrieval / reranking models pre-cached.")
print(f"   Cache root: {HF_HOME}")
print("")
print("You can now submit SLURM jobs with HF_HUB_OFFLINE=1.")
print("The compute nodes will load the models from disk without")
print("any outbound network requests.")
print("=" * 58)
PY

echo ""
echo "Done. Next steps:"
echo "  1. If you have not yet pre-cached the datasets, run:"
echo "       bash slurm/precache_datasets.sh"
echo "  2. Then submit the baseline job:"
echo "       sbatch slurm/run_baselines.sh"
