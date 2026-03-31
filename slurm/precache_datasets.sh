#!/bin/bash
# =============================================================================
# Pre-cache HuggingFace datasets for offline cluster use
#
# Run this on a LOGIN NODE (with internet access) BEFORE submitting SLURM jobs.
# Datasets will be cached so compute nodes can load them in offline mode.
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Pre-caching HuggingFace datasets"
echo "=========================================="

# Cache location — must match SLURM scripts
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "Cache dir: $HF_DATASETS_CACHE"
echo ""

# Load conda (adjust for your environment)
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate rag_baseline

echo "--- NQ-Open (Tier 0) ---"
python -c "
from datasets import load_dataset
print('Loading NQ-Open validation split...')
ds = load_dataset('google-research-datasets/nq_open', split='validation')
print(f'  Cached: {len(ds)} examples')
"

echo ""
echo "--- AmbigDocs (Tier 1) ---"
python -c "
from datasets import load_dataset
print('Loading AmbigDocs validation split...')
ds = load_dataset('yoonsanglee/AmbigDocs', split='validation')
print(f'  Cached: {len(ds)} examples')
"

echo ""
echo "--- FaithEval (Tier 2) ---"
python -c "
from datasets import load_dataset
for subtask in ['unanswerable', 'inconsistent', 'counterfactual']:
    hf_id = f'Salesforce/FaithEval-{subtask}-v1.0'
    print(f'Loading {hf_id}...')
    ds = load_dataset(hf_id, split='test')
    print(f'  Cached: {len(ds)} examples')
"

echo ""
echo "--- RAMDocs (Tier 3) ---"
python -c "
from datasets import load_dataset
print('Loading RAMDocs...')
ds = load_dataset('HanNight/RAMDocs', split='test')
print(f'  Cached: {len(ds)} examples')
"

echo ""
echo "✅ All datasets cached at: $HF_DATASETS_CACHE"
echo ""
echo "You can now submit SLURM jobs with offline mode enabled."
echo "  sbatch slurm/run_baselines.sh"
