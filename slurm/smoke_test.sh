#!/bin/bash
#SBATCH --job-name=rag-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=0:30:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# =============================================================================
# RAG Smoke Test — quick end-to-end check before a full run
#
# Runs 5 examples through each smoke config, covering all pipeline branches:
#
#   smoke_test.yaml          NQ-Open    LLM-only        no retrieval
#   smoke_test_ambigdocs     AmbigDocs  sparse (BM25)   no reranker
#   smoke_test_faitheval     FaithEval  dense (FAISS)   no reranker
#   smoke_test_ramdocs       RAMDocs    hybrid           + reranker
#
# Together these exercise every component: BM25, SentenceTransformer/FAISS,
# dense/sparse fusion, CrossEncoder reranker, and the LLM-only path.
# Expected runtime: ~10-15 min (mostly model loading).
#
# Steps:
#   1. Unit tests (no GPU)
#   2. CLI dry-run (no GPU)
#   3. Start vLLM server
#   4. Pipeline smoke test (5 examples × 4 configs)
#   5. Validate output files exist
# =============================================================================

set -eo pipefail  # Note: no -u — conda shell hook uses unbound internal vars (_CE_M)

echo "=========================================="
echo "RAG Smoke Test"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo ""

# ------------------------------------------------------------------
# 0. Configuration — adjust paths to match your cluster
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/Contamination-Aware-Control-for-Retrieval-Augmented-Generation
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen3-32B
# Derive the served model name from the local path — passed to both
# --served-model-name (vLLM) and --generator-model (CLI) so they always match.
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
CONDA_ENV=rag_baseline
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.9
SMOKE_EXAMPLES=5

# ------------------------------------------------------------------
# 1. Environment setup (identical to run_baselines.sh)
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

# Make the package importable without `pip install -e .`
# (mirrors what pytest does via pyproject.toml `pythonpath = ["src"]`)
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# 2. Cache & offline (identical to run_baselines.sh)
# ------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/hub
# Tells adapters where to find the Arrow dirs written by precache_datasets.sh
export HF_DATASETS_DISK_DIR=$HF_DATASETS_CACHE
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
# 3. GPU / memory optimization (identical to run_baselines.sh)
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 4. Validate prerequisites
# ------------------------------------------------------------------
echo "--- Environment ---"
echo "  Working dir: $(pwd)"
echo "  Python:      $(which python)"
echo "  Python ver:  $(python --version)"
echo "  Model path:  $MODEL_PATH"
echo ""

# Check model files exist (same checks as God's Reach Qwen72BProvider)
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "  Run on login node: huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir $MODEL_PATH"
    exit 1
fi

for f in config.json tokenizer.json tokenizer_config.json; do
    if [ ! -f "$MODEL_PATH/$f" ]; then
        echo "ERROR: Missing model file: $MODEL_PATH/$f"
        echo "  Model directory is incomplete — re-download the model."
        exit 1
    fi
done
echo "OK: Model files verified"

# Check NQ-Open dataset is pre-cached
NQ_CACHE="$HF_DATASETS_CACHE/google-research-datasets___nq_open"
if [ ! -d "$NQ_CACHE" ]; then
    echo "WARNING: NQ-Open dataset cache not found at $NQ_CACHE"
    echo "  Run slurm/precache_datasets.sh on the login node first."
    echo "  Temporarily disabling offline mode to allow download attempt..."
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
else
    echo "OK: NQ-Open dataset cache found"
fi

mkdir -p logs outputs/smoke_test

# ------------------------------------------------------------------
# 5. Step 1/4: Unit tests (no GPU required)
# ------------------------------------------------------------------
echo ""
echo "--- Step 1/4: Unit tests ---"

python -m pytest tests/ -m "not slow" -x -q --tb=short 2>&1
PYTEST_EXIT=$?

if [ $PYTEST_EXIT -ne 0 ]; then
    echo "ERROR: Unit tests failed — fix before submitting full run."
    exit 1
fi
echo "OK: All unit tests passed"

# ------------------------------------------------------------------
# 6. Step 2/4: CLI dry-run (no GPU) — validate all smoke configs
# ------------------------------------------------------------------
echo ""
echo "--- Step 2/4: CLI dry-run ---"

for SMOKE_CONFIG in \
        configs/smoke_test.yaml \
        configs/smoke_test_ambigdocs.yaml \
        configs/smoke_test_faitheval.yaml \
        configs/smoke_test_ramdocs.yaml; do
    python -m rag_baseline.cli \
        --config "$SMOKE_CONFIG" \
        --dry-run
done

echo "OK: CLI dry-run passed (all configs)"

# ------------------------------------------------------------------
# 7. Step 3/4: Start vLLM server
# ------------------------------------------------------------------
echo ""
echo "--- Step 3/4: Start vLLM server ---"
echo "  Model:   $MODEL_PATH"
echo "  TP size: $TENSOR_PARALLEL_SIZE"
echo "  Port:    $VLLM_PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
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
echo "vLLM server PID: $VLLM_PID"

cleanup() {
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        echo "OK: vLLM server stopped"
    fi
}
trap cleanup EXIT INT TERM

# Wait for server to be ready (32B takes ~3-5 min to load)
echo "Waiting for vLLM server (up to 10 min)..."
MAX_WAIT=600
WAIT_INTERVAL=10
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server exited unexpectedly"
        echo "  Check: logs/smoke_test_${SLURM_JOB_ID}.err"
        exit 1
    fi

    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "OK: vLLM server ready after ${ELAPSED}s"
        break
    fi

    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server did not become ready within ${MAX_WAIT}s"
    exit 1
fi

echo "Loaded models:"
curl -s "http://localhost:${VLLM_PORT}/v1/models" | python -m json.tool 2>/dev/null || true

# ------------------------------------------------------------------
# 8. Step 4/4: Pipeline smoke tests (all datasets)
# ------------------------------------------------------------------
echo ""
echo "--- Step 4/4: Pipeline smoke tests ($SMOKE_EXAMPLES examples each) ---"

# Coverage matrix — every pipeline branch is exercised:
#   smoke_test          → NQ-Open   / LLM-only             (no retrieval)
#   smoke_test_ambigdocs → AmbigDocs / sparse (BM25)        (no reranker)
#   smoke_test_faitheval → FaithEval / dense  (FAISS + bge) (no reranker)
#   smoke_test_ramdocs   → RAMDocs   / hybrid + reranker    (full pipeline)
SMOKE_CONFIGS=(
    "configs/smoke_test.yaml"               # NQ-Open   — LLM-only
    "configs/smoke_test_ambigdocs.yaml"     # AmbigDocs — sparse RAG, multi-answer
    "configs/smoke_test_faitheval.yaml"     # FaithEval — dense RAG, unknown-compatible
    "configs/smoke_test_ramdocs.yaml"       # RAMDocs   — hybrid RAG + reranker, multi-answer
)

SMOKE_FAILED=0

for SMOKE_CONFIG in "${SMOKE_CONFIGS[@]}"; do
    SMOKE_NAME=$(basename "$SMOKE_CONFIG" .yaml)
    echo ""
    echo "  Running: $SMOKE_NAME"

    if python -m rag_baseline.cli \
            --config "$SMOKE_CONFIG" \
            --generator-model "$SERVED_MODEL_NAME" \
            --max-examples "$SMOKE_EXAMPLES"; then
        echo "  OK: $SMOKE_NAME"
    else
        echo "  ERROR: $SMOKE_NAME failed (exit $?)"
        SMOKE_FAILED=$((SMOKE_FAILED + 1))
    fi
done

if [ "$SMOKE_FAILED" -ne 0 ]; then
    echo ""
    echo "ERROR: $SMOKE_FAILED pipeline smoke test(s) failed"
    exit 1
fi
echo ""
echo "OK: All pipeline smoke tests passed"

PIPELINE_EXIT=0

# ------------------------------------------------------------------
# 9. Validate output files were written for each smoke run
# ------------------------------------------------------------------
echo ""
echo "--- Output validation ---"

SMOKE_OUTPUT_DIRS=(
    "outputs/smoke_test"
    "outputs/smoke_test_ambigdocs"
    "outputs/smoke_test_faitheval"
    "outputs/smoke_test_ramdocs"
)

ALL_OK=true

for OUTPUT_DIR in "${SMOKE_OUTPUT_DIRS[@]}"; do
    for f in summary_metrics.json run_config.yaml; do
        if [ -f "$OUTPUT_DIR/$f" ]; then
            echo "OK: $OUTPUT_DIR/$f"
        else
            echo "MISSING: $OUTPUT_DIR/$f"
            ALL_OK=false
        fi
    done
done

echo ""
echo "NQ-Open metrics:"
cat "outputs/smoke_test/summary_metrics.json" 2>/dev/null || echo "(not found)"
echo ""
echo "AmbigDocs metrics:"
cat "outputs/smoke_test_ambigdocs/summary_metrics.json" 2>/dev/null || echo "(not found)"
echo ""
echo "FaithEval metrics:"
cat "outputs/smoke_test_faitheval/summary_metrics.json" 2>/dev/null || echo "(not found)"
echo ""
echo "RAMDocs metrics:"
cat "outputs/smoke_test_ramdocs/summary_metrics.json" 2>/dev/null || echo "(not found)"

# ------------------------------------------------------------------
# 10. Final result
# ------------------------------------------------------------------
echo ""
echo "=========================================="
if [ "$ALL_OK" = true ]; then
    echo "SMOKE TEST PASSED"
    echo ""
    echo "Next step: sbatch slurm/run_baselines.sh"
else
    echo "SMOKE TEST FAILED — check logs above"
    exit 1
fi
echo "Time: $(date)"
echo "=========================================="
