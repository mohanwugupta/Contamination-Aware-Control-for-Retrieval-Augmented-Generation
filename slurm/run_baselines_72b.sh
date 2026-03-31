#!/bin/bash
#SBATCH --job-name=rag-baselines-72b
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Single task (not MPI)
#SBATCH --cpus-per-task=8      # CPUs for tokenization, data loading, retrieval
#SBATCH --mem=256G             # More memory for 72B model
#SBATCH --gres=gpu:4           # 4 GPUs for 72B model (~140GB VRAM with TP=4)
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=6:00:00         # Longer time for larger model
#SBATCH --output=logs/rag_baselines_72b_%j.out
#SBATCH --error=logs/rag_baselines_72b_%j.err

# =============================================================================
# RAG Baseline Pipeline — Qwen2.5-72B-Instruct variant
#
# Same as run_baselines.sh but with 4 GPUs and 72B model.
# Mirrors God's Reach run_batch_qwen72b.sh environment setup.
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "RAG Baseline Pipeline (Qwen2.5-72B)"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo ""

# ------------------------------------------------------------------
# 0. Configuration — 72B-specific overrides
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/Contamination-Aware-Control-for-Retrieval-Augmented-Generation
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct
CONDA_ENV=rag_baseline
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=4      # 4 GPUs for 72B model
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.9

BASELINES=(
    "configs/baselines/llm_only.yaml"
    "configs/baselines/vanilla_rag.yaml"
    "configs/baselines/hybrid_rag.yaml"
    "configs/baselines/hybrid_rerank.yaml"
    "configs/baselines/reduced_context.yaml"
)

MAX_EXAMPLES=""
# MAX_EXAMPLES="--max-examples 50"

# ------------------------------------------------------------------
# 1. Environment setup (identical to God's Reach run_batch_qwen72b.sh)
# ------------------------------------------------------------------
cd "$PROJECT_DIR"

module load anaconda3/2025.6

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi

# ------------------------------------------------------------------
# 2. Cache & offline (identical to God's Reach)
# ------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/transformers
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/usage_stats
export TRITON_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/triton
export XDG_CACHE_HOME=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/xdg
export TIKTOKEN_CACHE_DIR=$HOME/.tiktoken_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 3. GPU / Memory optimization (identical to God's Reach)
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
echo "  TP size:      $TENSOR_PARALLEL_SIZE"
echo ""

if [ -d "$MODEL_PATH" ]; then
    echo "✅ Qwen2.5-72B model found at: $MODEL_PATH"
    for f in config.json tokenizer.json tokenizer_config.json; do
        if [ ! -f "$MODEL_PATH/$f" ]; then
            echo "❌ ERROR: Missing model file: $MODEL_PATH/$f"
            exit 1
        fi
    done
    echo "✅ Model files verified"
else
    echo "❌ ERROR: Model not found at: $MODEL_PATH"
    echo "   Download: huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir $MODEL_PATH"
    exit 1
fi

mkdir -p logs outputs

# ------------------------------------------------------------------
# 5. Start vLLM server
# ------------------------------------------------------------------
echo ""
echo "Starting vLLM server (Qwen2.5-72B, TP=$TENSOR_PARALLEL_SIZE)..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Qwen/Qwen2.5-72B-Instruct" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --disable-custom-all-reduce \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        echo "✅ vLLM server stopped"
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 6. Wait for server readiness (72B takes longer to load)
# ------------------------------------------------------------------
echo "Waiting for vLLM server (72B model loading may take 5-10 min)..."

MAX_WAIT=900  # 15 minutes for 72B
WAIT_INTERVAL=15
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM server exited unexpectedly"
        exit 1
    fi

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

echo ""
curl -s "http://localhost:${VLLM_PORT}/v1/models" | python -m json.tool 2>/dev/null || true
echo ""

# ------------------------------------------------------------------
# 7. Run baselines (override generator_model in config via env)
# ------------------------------------------------------------------
echo "=========================================="
echo "Running baseline pipeline (72B)"
echo "=========================================="

TOTAL=${#BASELINES[@]}
COMPLETED=0
FAILED=0

for CONFIG in "${BASELINES[@]}"; do
    BASELINE_NAME=$(basename "$CONFIG" .yaml)
    echo ""
    echo "[$((COMPLETED + FAILED + 1))/$TOTAL] Running: $BASELINE_NAME"

    if python -m rag_baseline.cli --config "$CONFIG" $MAX_EXAMPLES; then
        echo "✅ $BASELINE_NAME completed"
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
echo "BATCH RESULTS (Qwen2.5-72B)"
echo "=========================================="
echo "  Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED"
echo "  Outputs: outputs/"
echo ""
echo "✅ Job completed at $(date)"

[ $FAILED -gt 0 ] && exit 1
