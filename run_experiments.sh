#!/bin/bash
# run_experiments.sh — Sweep all optimization configs for Wan2.2-TI2V-5B
#
# Usage: bash run_experiments.sh
#
# Runs inside the pod. Each experiment uses torchrun to launch the optimized
# inference script with different env var combinations. Results go to
# /tmp/wan2_experiments/<experiment_name>/
#
# Adjust TP_DEGREE and --nproc_per_node to match your NeuronCore count.
# For 4 cores: TP_DEGREE=4, nproc=4, SP_DEGREE=2
# For 8 cores: TP_DEGREE=8, nproc=8, SP_DEGREE=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/inference_neuron_ti2v_optimized.py"

# ── Hardware config — CHANGE THESE for your setup ──
NPROC=4          # NeuronCores available
TP=4             # Tensor parallelism degree
SP=2             # Sequence parallelism degree (2 when TP=4)
T5_RANK=2        # Which rank hosts T5

export WAN_DIR="${SCRIPT_DIR}"
export MODEL_PATH="${MODEL_PATH:-/tmp/Wan2.2-TI2V-5B}"
export TP_DEGREE=$TP
export SP_DEGREE=$SP
export T5_RANK=$T5_RANK
export VAE_TP_DEGREE=1
export FRAME_NUM=81
export NUM_RUNS=2
export SAVE_LATENT=1
export OUTPUT_DIR=/tmp/wan2_experiments
export NEURON_SKIP_EFA_AFFINITY=1

mkdir -p "$OUTPUT_DIR"

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "  EXPERIMENT: $name"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    export EXPERIMENT_NAME="$name"
    # Apply experiment-specific env vars
    for var in "$@"; do
        export "$var"
    done

    torchrun --nproc_per_node=$NPROC --master_port=29500 \
        "$SCRIPT" 2>&1 | tee "${OUTPUT_DIR}/${name}.log"

    # Reset all optimization flags to defaults
    export TEACACHE_ENABLED=0
    export CFG_RESCALE_ENABLED=0
    export SHARPEN_ENABLED=0
    export TEMPORAL_ENABLED=0
    export EARLY_EXIT_ENABLED=0
    export SCHEDULER_SHIFT=1
}

echo "Starting experiment sweep at $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Hardware: TP=$TP, SP=$SP, nproc=$NPROC"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINES — Step count sweep (no other optimizations)
# Establishes the quality/speed frontier from step reduction alone.
# ═══════════════════════════════════════════════════════════════════════

run_experiment "01_baseline_10step" \
    "NUM_STEPS=10"

run_experiment "02_baseline_8step" \
    "NUM_STEPS=8"

run_experiment "03_baseline_5step" \
    "NUM_STEPS=5"

run_experiment "04_baseline_4step" \
    "NUM_STEPS=4"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: SCHEDULER SHIFT — Does concentrating steps on early
# (noisy) phases help at low step counts?
# ═══════════════════════════════════════════════════════════════════════

run_experiment "05_5step_shift2" \
    "NUM_STEPS=5" "SCHEDULER_SHIFT=2"

run_experiment "06_5step_shift3" \
    "NUM_STEPS=5" "SCHEDULER_SHIFT=3"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: QUALITY RECOVERY — CFG rescaling + sharpening at 5 steps
# These are safe interventions that correct known artifacts.
# ═══════════════════════════════════════════════════════════════════════

run_experiment "07_5step_cfg_rescale" \
    "NUM_STEPS=5" "CFG_RESCALE_ENABLED=1" "CFG_RESCALE_FACTOR=0.7"

run_experiment "08_5step_sharpen" \
    "NUM_STEPS=5" "SHARPEN_ENABLED=1" "SHARPEN_FACTOR=0.1"

run_experiment "09_5step_cfg_and_sharpen" \
    "NUM_STEPS=5" \
    "CFG_RESCALE_ENABLED=1" "CFG_RESCALE_FACTOR=0.7" \
    "SHARPEN_ENABLED=1" "SHARPEN_FACTOR=0.1"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: TEMPORAL CONSISTENCY — Prevent flickering at 5 steps
# ═══════════════════════════════════════════════════════════════════════

run_experiment "10_5step_temporal" \
    "NUM_STEPS=5" "TEMPORAL_ENABLED=1" "TEMPORAL_ALPHA=0.15"

run_experiment "11_5step_all_quality" \
    "NUM_STEPS=5" \
    "CFG_RESCALE_ENABLED=1" "CFG_RESCALE_FACTOR=0.7" \
    "SHARPEN_ENABLED=1" "SHARPEN_FACTOR=0.1" \
    "TEMPORAL_ENABLED=1" "TEMPORAL_ALPHA=0.15"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: TEACACHE — Skip redundant DiT calls
# Combined with quality interventions from Phase 3-4.
# ═══════════════════════════════════════════════════════════════════════

run_experiment "12_5step_teacache_004" \
    "NUM_STEPS=5" "TEACACHE_ENABLED=1" "TEACACHE_THRESHOLD=0.04"

run_experiment "13_5step_teacache_008" \
    "NUM_STEPS=5" "TEACACHE_ENABLED=1" "TEACACHE_THRESHOLD=0.08"

run_experiment "14_5step_teacache_and_quality" \
    "NUM_STEPS=5" \
    "TEACACHE_ENABLED=1" "TEACACHE_THRESHOLD=0.04" \
    "CFG_RESCALE_ENABLED=1" "CFG_RESCALE_FACTOR=0.7" \
    "SHARPEN_ENABLED=1" "SHARPEN_FACTOR=0.1"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: EARLY EXIT — Skip last 1-2 steps via interpolation
# Most aggressive; try with and without quality recovery.
# ═══════════════════════════════════════════════════════════════════════

run_experiment "15_5step_early_exit_1" \
    "NUM_STEPS=5" "EARLY_EXIT_ENABLED=1" "EARLY_EXIT_SKIP_LAST_N=1"

run_experiment "16_5step_early_exit_1_with_quality" \
    "NUM_STEPS=5" \
    "EARLY_EXIT_ENABLED=1" "EARLY_EXIT_SKIP_LAST_N=1" \
    "CFG_RESCALE_ENABLED=1" "CFG_RESCALE_FACTOR=0.7" \
    "SHARPEN_ENABLED=1" "SHARPEN_FACTOR=0.1"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7: BEST COMBO — Everything together, tuned
# ═══════════════════════════════════════════════════════════════════════

run_experiment "17_best_combo" \
    "NUM_STEPS=5" \
    "TEACACHE_ENABLED=1" "TEACACHE_THRESHOLD=0.04" \
    "CFG_RESCALE_ENABLED=1" "CFG_RESCALE_FACTOR=0.7" \
    "SHARPEN_ENABLED=1" "SHARPEN_FACTOR=0.1" \
    "TEMPORAL_ENABLED=1" "TEMPORAL_ALPHA=0.10"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results: $OUTPUT_DIR"
echo "  Run: python3 measure_quality.py to compute quality metrics"
echo "═══════════════════════════════════════════════════════════════"
