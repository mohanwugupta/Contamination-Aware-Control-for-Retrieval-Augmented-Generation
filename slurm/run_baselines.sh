#!/bin/bash
#SBATCH --job-name=rag-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Single task (not MPI)
#SBATCH --cpus-per-task=8      # CPUs for tokenization, data loading, retrieval
#SBATCH --mem=64G             # Memory for Qwen2.5-32B + FAISS indices
#SBATCH --gres=gpu:2           # 2 GPUs for 32B model (~65GB VRAM with TP=2)
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=16:00:00        # 16 hours for full baseline sweep
#SBATCH --output=logs/rag_baselines_%j.out
#SBATCH --error=logs/rag_baselines_%j.err

# =============================================================================
# RAG Baseline Pipeline — SLURM Job Script
#
# Runs all baseline RAG configurations on the cluster.
# Architecture: vLLM server as background process + pipeline CLI as client.
#
# Mirrors the proven environment setup from God's Reach project.
# =============================================================================

set -eo pipefail  # Note: no -u — conda shell hook uses unbound internal vars (_CE_M)

echo "=========================================="
echo "RAG Baseline Pipeline"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo ""

# ------------------------------------------------------------------
# 0. Configuration (adjust these paths for your environment)
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/Contamination-Aware-Control-for-Retrieval-Augmented-Generation
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen3-32B
# Derive the served model name from the local path — passed to both
# --served-model-name (vLLM) and --generator-model (CLI) so they always match.
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
CONDA_ENV=rag_baseline
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=2      # 2 GPUs for 32B model
MAX_MODEL_LEN=32768         # Context window
GPU_MEMORY_UTILIZATION=0.9

# Baselines to run (comment out any you want to skip)
BASELINES=(
    "configs/baselines/llm_only.yaml"
    "configs/baselines/vanilla_rag.yaml"
    "configs/baselines/hybrid_rag.yaml"
    "configs/baselines/hybrid_rerank.yaml"
    "configs/baselines/reduced_context.yaml"
)

# Optional: limit examples for quick testing (set to "" for full run)
MAX_EXAMPLES=""
# MAX_EXAMPLES="--max-examples 50"

# ------------------------------------------------------------------
# 1. Environment setup (mirrors God's Reach pattern)
# ------------------------------------------------------------------
cd "$PROJECT_DIR"

# Load required modules
module load anaconda3/2025.6

# Activate Python environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi

# Make the package importable without `pip install -e .`
# (mirrors what pytest does via pyproject.toml `pythonpath = ["src"]`)
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# 2. Cache & offline configuration
#    (CRITICAL: compute nodes have no internet)
# ------------------------------------------------------------------

# HuggingFace model + dataset caches → scratch (avoid home quota)
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache

# vLLM and compilation caches → scratch (avoid home quota)
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/usage_stats
export TRITON_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/triton
export XDG_CACHE_HOME=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/xdg
export TIKTOKEN_CACHE_DIR=$HOME/.tiktoken_cache

# Create cache directories
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

# Force offline mode (compute nodes typically have no internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 3. GPU / Memory optimization
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# PyTorch CUDA Memory Optimization — reduces fragmentation / OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL Configuration for multi-GPU on single node
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 4. Validate prerequisites
# ------------------------------------------------------------------
echo "Environment:"
echo "  Working dir:  $(pwd)"
echo "  Python:       $(which python)"
echo "  Python ver:   $(python --version)"
echo "  Model path:   $MODEL_PATH"
echo "  HF cache:     $HF_HOME"
echo "  vLLM cache:   $VLLM_CACHE_DIR"
echo ""

# Check model exists
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Model found at: $MODEL_PATH"
    # Verify critical files (same check as God's Reach Qwen72BProvider)
    for f in config.json tokenizer.json tokenizer_config.json; do
        if [ ! -f "$MODEL_PATH/$f" ]; then
            echo "❌ ERROR: Missing model file: $MODEL_PATH/$f"
            echo "   Model directory is incomplete. Re-download the model."
            exit 1
        fi
    done
    echo "✅ Model files verified (config.json, tokenizer.json, tokenizer_config.json)"
else
    echo "❌ ERROR: Model not found at: $MODEL_PATH"
    echo "   Download it first on a login node:"
    echo "   huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir $MODEL_PATH"
    exit 1
fi

# Check HF datasets are pre-cached
echo ""
echo "Checking HF dataset cache..."
if [ -d "$HF_DATASETS_CACHE" ] && [ "$(ls -A $HF_DATASETS_CACHE 2>/dev/null)" ]; then
    echo "✅ HF datasets cache populated at: $HF_DATASETS_CACHE"
else
    echo "⚠️  WARNING: HF datasets cache appears empty."
    echo "   Pre-cache datasets on a login node:"
    echo "     python -c \"from datasets import load_dataset; load_dataset('google-research-datasets/nq_open', split='validation')\""
    echo "     python -c \"from datasets import load_dataset; load_dataset('yoonsanglee/AmbigDocs', split='validation')\""
    echo "   Datasets will fail to load in offline mode without cache."
fi

# Create output directories
mkdir -p logs outputs

# ------------------------------------------------------------------
# 5. Start vLLM server as background process
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Starting vLLM server..."
echo "=========================================="
echo "  Model:    $MODEL_PATH"
echo "  TP size:  $TENSOR_PARALLEL_SIZE"
echo "  Port:     $VLLM_PORT"
echo "  Max len:  $MAX_MODEL_LEN"
echo "  GPU mem:  $GPU_MEMORY_UTILIZATION"
echo ""

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 256 \
    --disable-custom-all-reduce \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Cleanup function — kill vLLM server on exit (success, failure, or signal)
cleanup() {
    echo ""
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        echo "✅ vLLM server stopped"
    else
        echo "vLLM server already exited"
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 6. Wait for vLLM server to be ready
# ------------------------------------------------------------------
echo ""
echo "Waiting for vLLM server to be ready..."

MAX_WAIT=600  # 10 minutes max (large model loading can take a while)
WAIT_INTERVAL=10
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if server process is still alive
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM server exited unexpectedly"
        echo "   Check logs/rag_baselines_${SLURM_JOB_ID}.err for details"
        exit 1
    fi

    # Health check
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM server ready after ${ELAPSED}s"
        break
    fi

    echo "   Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM server did not become ready within ${MAX_WAIT}s"
    exit 1
fi

# Verify model is loaded by listing models
echo ""
echo "Verifying model endpoint..."
curl -s "http://localhost:${VLLM_PORT}/v1/models" | python -m json.tool 2>/dev/null || echo "(model list check skipped)"
echo ""

# ------------------------------------------------------------------
# 7. Run baseline pipeline
# ------------------------------------------------------------------
echo "=========================================="
echo "Running baseline pipeline"
echo "=========================================="

TOTAL_BASELINES=${#BASELINES[@]}
COMPLETED=0
FAILED=0

for CONFIG in "${BASELINES[@]}"; do
    BASELINE_NAME=$(basename "$CONFIG" .yaml)
    echo ""
    echo "------------------------------------------"
    echo "[$((COMPLETED + FAILED + 1))/$TOTAL_BASELINES] Running: $BASELINE_NAME"
    echo "  Config: $CONFIG"
    echo "------------------------------------------"

    if python -m rag_baseline.cli \
            --config "$CONFIG" \
            --generator-model "$SERVED_MODEL_NAME" \
            --num-workers 64 \
            $MAX_EXAMPLES; then
        echo "✅ $BASELINE_NAME completed successfully"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "❌ $BASELINE_NAME failed"
        FAILED=$((FAILED + 1))
    fi
done

# ------------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "BATCH RESULTS"
echo "=========================================="
echo "  Total baselines: $TOTAL_BASELINES"
echo "  Completed:       $COMPLETED"
echo "  Failed:          $FAILED"
echo "  Outputs:         outputs/"
echo ""
echo "✅ Job completed at $(date)"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
