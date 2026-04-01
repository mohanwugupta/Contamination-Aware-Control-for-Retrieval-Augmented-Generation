#!/bin/bash
# =============================================================================
# Pre-cache HuggingFace datasets for offline cluster use
#
# Run this on a LOGIN NODE (with internet access) BEFORE submitting SLURM jobs.
# Datasets will be cached so compute nodes can load them in offline mode.
#
# All datasets are saved with datasets.save_to_disk(...) so downstream code can
# uniformly use load_from_disk(...) on compute nodes.
# =============================================================================

set -eo pipefail

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

python - <<'PY'
import os
from huggingface_hub import hf_hub_download
from datasets import load_dataset

HF_HOME = os.environ["HF_HOME"]
HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]


def save_hf_dataset(hf_id: str, split: str, out_name: str):
    """
    Load a dataset split from the Hub and save it to disk in Arrow format.
    """
    out_dir = os.path.join(HF_DATASETS_CACHE, out_name)
    print(f"Loading {hf_id} [{split}]...")
    ds = load_dataset(hf_id, split=split)
    print(f"  Loaded: {len(ds)} examples")
    print(f"  Saving to: {out_dir}")
    ds.save_to_disk(out_dir)
    print("  Done.")


def save_json_dataset_from_hub(repo_id: str, filename: str, out_name: str):
    """
    Download a raw JSON file from a dataset repo, parse it locally, and save it
    to disk in Arrow format.

    This is used for AmbigDocs because the repo schema is broken for direct
    load_dataset(repo_id, split=...).
    """
    raw_dir = os.path.join(HF_DATASETS_CACHE, f"{out_name}_raw")
    out_dir = os.path.join(HF_DATASETS_CACHE, out_name)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading {repo_id}/{filename}...")
    json_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=HF_HOME,
        local_dir=raw_dir,
    )
    print(f"  Downloaded to: {json_path}")

    print("  Parsing raw JSON...")
    ds = load_dataset("json", data_files=json_path, split="train")
    print(f"  Loaded: {len(ds)} examples")

    print(f"  Saving to: {out_dir}")
    ds.save_to_disk(out_dir)
    print("  Done.")


print("--- NQ-Open (Tier 0) ---")
save_hf_dataset(
    hf_id="google-research-datasets/nq_open",
    split="validation",
    out_name="nq_open_validation",
)

print("\n--- AmbigDocs (Tier 1) ---")
# dev.json corresponds to the validation split
save_json_dataset_from_hub(
    repo_id="yoonsanglee/AmbigDocs",
    filename="dev.json",
    out_name="ambigdocs_validation",
)

print("\n--- FaithEval (Tier 2) ---")
for subtask in ["unanswerable", "inconsistent", "counterfactual"]:
    save_hf_dataset(
        hf_id=f"Salesforce/FaithEval-{subtask}-v1.0",
        split="test",
        out_name=f"faitheval_{subtask}_test",
    )

print("\n--- RAMDocs (Tier 3) ---")
save_hf_dataset(
    hf_id="HanNight/RAMDocs",
    split="test",
    out_name="ramdocs_test",
)

print("\n✅ All datasets cached and saved in load_from_disk format.")
print(f"Base directory: {HF_DATASETS_CACHE}")

print("\nSaved dataset directories:")
for name in [
    "nq_open_validation",
    "ambigdocs_validation",
    "faitheval_unanswerable_test",
    "faitheval_inconsistent_test",
    "faitheval_counterfactual_test",
    "ramdocs_test",
]:
    print(f"  {os.path.join(HF_DATASETS_CACHE, name)}")
PY

echo ""
echo "You can now submit SLURM jobs with offline mode enabled."
echo "On compute nodes, load datasets uniformly with:"
echo "  from datasets import load_from_disk"
echo "  ds = load_from_disk('/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets/<dataset_dir>')"
echo ""
echo "Example:"
echo "  ds = load_from_disk('/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets/ambigdocs_validation')"