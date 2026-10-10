#!/bin/bash
# Wan2.2-TI2V-5B (fast-dit branch) benchmark on a trn2 pod, without Kubernetes.
# Converted from the k8s job's command block. Everything goes under $WORK_DIR and a
# personal venv, so the pod's shared Python is never modified.
#
# Needs: the pod's python3 with a working torch-neuronx; git; internet for pip/HF.
#
# Usage:
#   bash run_wan_fastdit.sh                  # setup (first time) + run
#   SKIP_SETUP=1 bash run_wan_fastdit.sh     # run only
#   PROFILE=1 bash run_wan_fastdit.sh        # also capture a Neuron profile + NEFF report
#
# Override any of these from the environment:
WORK_DIR="${WORK_DIR:-/workspace/hs-wan}"
REPO_URL="${REPO_URL:-https://github.com/yahavb/wan2-ti2v-5b.git}"
REPO_BRANCH="${REPO_BRANCH:-fast-dit}"
REPO_DIR="${REPO_DIR:-$WORK_DIR/wan2-ti2v-5b-$REPO_BRANCH}"
VENV="${VENV:-$WORK_DIR/venv}"
MODEL_PATH="${MODEL_PATH:-$WORK_DIR/models/Wan2.2-TI2V-5B}"
TORCHVISION_REF="${TORCHVISION_REF:-v0.25.0}"   # should match the pod's torch version
SKIP_SETUP="${SKIP_SETUP:-0}"
PROFILE="${PROFILE:-0}"

# Run settings (the k8s job set these in its env: section; adjust to match it).
# This pod: 1 Neuron device, LNC=1 -> 8 NeuronCores -> TP_DEGREE=8.
export TP_DEGREE="${TP_DEGREE:-8}"
export SP_DEGREE="${SP_DEGREE:-1}"
export T5_RANK="${T5_RANK:-2}"
export VAE_TP_DEGREE="${VAE_TP_DEGREE:-1}"
export USE_NKI_KERNELS="${USE_NKI_KERNELS:-1}"
export USE_NKI_VAE="${USE_NKI_VAE:-0}"
export USE_NST_SELF_ATTN="${USE_NST_SELF_ATTN:-1}"
export NUM_STEPS="${NUM_STEPS:-10}"
export NUM_RUNS="${NUM_RUNS:-2}"
export FRAME_NUM="${FRAME_NUM:-81}"
[[ -n "${DIT_FAST:-}" ]] && export DIT_FAST
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-1}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR TORCH_NEURONX_LOG_LEVEL=ERROR
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---model-type=transformer}"
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0
export NEURON_COMPILE_CACHE_URL="${NEURON_COMPILE_CACHE_URL:-$WORK_DIR/neuron-cache}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf-cache}"

set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

say() { printf '\n==== %s ====\n' "$*"; }
mkdir -p "$WORK_DIR"

# ─── Personal venv on top of the pod's Neuron torch ─────────────────────────
if [[ ! -x "$VENV/bin/python" ]]; then
  say "Creating venv $VENV (sees the pod's torch-neuronx, installs extras privately)"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"

# ─── Code ───────────────────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/.git" ]]; then
  say "Cloning $REPO_URL ($REPO_BRANCH) -> $REPO_DIR"
  git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ─── Dependencies (into the venv only) ──────────────────────────────────────
if [[ "$SKIP_SETUP" != "1" ]]; then
  say "Installing Python dependencies into $VENV"
  # Keep the pod's torch / Neuron packages / numpy exactly as they are.
  "$PY" -m pip freeze | grep -iE "^(torch|torch-neuronx|neuronx|libneuronxla|nki|numpy)==" \
    > "$WORK_DIR/pins.txt" || true
  # flash_attn/torchaudio are CUDA-only; numpy is pinned <2 upstream, but the Neuron
  # SDK's own kernels need numpy 2.x (np.long), so keep the image's numpy.
  sed -i '/flash_attn/d;/torchaudio/d;/numpy/d;/^torch>=/d;/^torchvision/d' wan_requirements.txt
  "$PY" -m pip install -q -c "$WORK_DIR/pins.txt" -r wan_requirements.txt
  "$PY" -m pip install -q -c "$WORK_DIR/pins.txt" "setuptools<81" imageio imageio-ffmpeg einops huggingface_hub

  if ! TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -c "import torchvision" 2>/dev/null; then
    say "Building torchvision $TORCHVISION_REF against the pod's torch"
    "$PY" -m pip install "git+https://github.com/pytorch/vision.git@$TORCHVISION_REF" \
      --no-deps --no-cache-dir --no-build-isolation
  fi

  # Some Neuron images ship a python3 without the _lzma module; torchvision's
  # datasets/utils.py imports lzma at import time. Stub it (only in the venv's copy).
  "$PY" - <<'PY'
import importlib.util as u, pathlib, sys
spec = u.find_spec('torchvision')
if spec is None:
    sys.exit("torchvision not installed")
loc = pathlib.Path(spec.submodule_search_locations[0])
if not str(loc).startswith(sys.prefix):
    print(f'torchvision at {loc} is outside the venv; not patching it')
    sys.exit(0)
p = loc / 'datasets' / 'utils.py'
s = p.read_text()
if '_no_lzma' in s:
    print('torchvision datasets/utils.py already patched')
elif 'import lzma\n' in s:
    stub = ('try:\n'
            '    import lzma\n'
            'except ImportError:\n'
            '    import types as _t\n'
            '    def _no_lzma(*a, **k):\n'
            '        raise RuntimeError("lzma unavailable in this Python build")\n'
            '    lzma = _t.SimpleNamespace(open=_no_lzma, LZMAFile=_no_lzma)\n')
    p.write_text(s.replace('import lzma\n', stub, 1))
    print('Patched torchvision datasets/utils.py: lzma replaced with stub')
PY
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -c "import torchvision, einops, imageio; print('torchvision', torchvision.__version__)"
  # The venv must still use the pod's Neuron torch, not a copy pulled in by a dependency.
  if TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -c "import torch, sys; sys.exit(0 if torch.__file__.startswith(sys.prefix) else 1)"; then
    echo "ERROR: a separate torch got installed into $VENV and shadows the pod's Neuron torch."
    echo "Remove it with: $PY -m pip uninstall -y torch   (only inside the venv)"
    exit 1
  fi
fi

# ─── Model weights (downloaded once to the persistent disk) ─────────────────
if [[ ! -f "$MODEL_PATH/Wan2.2_VAE.pth" ]]; then
  say "Downloading Wan-AI/Wan2.2-TI2V-5B -> $MODEL_PATH"
  mkdir -p "$MODEL_PATH"
  "$PY" - "$MODEL_PATH" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', local_dir=sys.argv[1], max_workers=2)
PY
fi
echo "Model weights ready at $MODEL_PATH"

# ─── Run ────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$WORK_DIR/runs/fastdit_$TIMESTAMP"
mkdir -p "$RUN_DIR"
PROFILE_SRC="$RUN_DIR/neuron_profile_raw"
if [[ "$PROFILE" == "1" ]]; then
  # Costs ~2.4 GB HBM per NeuronCore pair; turn off if the model no longer fits.
  mkdir -p "$PROFILE_SRC"
  export NEURON_RT_INSPECT_ENABLE=1
  export NEURON_RT_INSPECT_OUTPUT_DIR="$PROFILE_SRC"
  export NEURON_RT_INSPECT_DEVICE_PROFILE=session
fi

export WAN_DIR="$REPO_DIR" MODEL_PATH
say "BENCHMARK: TP=$TP_DEGREE DIT_FAST=${DIT_FAST:-unset} steps=$NUM_STEPS runs=$NUM_RUNS profile=$PROFILE"
echo "Log: $RUN_DIR/run.log"
rm -f /tmp/wan2_ti2v_output.mp4
set +e
"$PY" -m torch.distributed.run --nproc_per_node="$TP_DEGREE" --master_port=29500 \
  "$REPO_DIR/inference_neuron_ti2v.py" 2>&1 | tee "$RUN_DIR/run.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

# ─── Collect outputs ────────────────────────────────────────────────────────
if [[ -f /tmp/wan2_ti2v_output.mp4 ]]; then
  cp /tmp/wan2_ti2v_output.mp4 "$RUN_DIR/"
  echo "Video: $RUN_DIR/wan2_ti2v_output.mp4"
else
  echo "WARNING: no video produced (see $RUN_DIR/run.log)"
fi

# ─── Neuron Explorer: profile summary + NEFF size report (PROFILE=1 only) ────
NTFF_DIR=""
if [[ "$PROFILE" == "1" ]]; then
  NTFF_DIR=$(find "$PROFILE_SRC" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | head -1)
fi
if [[ -n "$NTFF_DIR" ]]; then
  say "NEURON EXPLORER ANALYSIS"
  neuron-explorer view -d "$NTFF_DIR" --output-format summary-text --ignore-dma-trace 2>&1 \
    | tee "$RUN_DIR/neuron_explorer_summary.txt" || true
  neuron-explorer view -d "$NTFF_DIR" --output-format json \
    --output-file "$RUN_DIR/neuron_explorer_profile.json" --ignore-dma-trace 2>&1 \
    | tee "$RUN_DIR/neuron_explorer_view.log" || true

  say "NEFF COUNT AND SIZE DISTRIBUTION"
  echo "Total NEFFs: $(find "$NTFF_DIR" -name '*.neff' | wc -l)"
  find "$NTFF_DIR" -name '*.neff' -printf '%s\n' | sort -n | awk '
    { sizes[count++]=$1; sum+=$1 }
    END {
      if (count == 0) { print "No NEFFs found"; exit }
      printf "Total size: %.2f MB\n", sum/1024/1024
      printf "Min: %d bytes\n", sizes[0]
      printf "Max: %d bytes (%.2f MB)\n", sizes[count-1], sizes[count-1]/1024/1024
      printf "Median: %d bytes\n", sizes[int(count/2)]
      printf "Mean: %.0f bytes\n", sum/count
      for (i=0;i<count;i++) {
        if (sizes[i]<10000) s++; else if (sizes[i]<100000) m++; else if (sizes[i]<1000000) l++; else x++
      }
      printf "\nSize buckets:\n  <10KB  (tiny):   %d\n  10-100KB (small): %d\n  100KB-1MB (med):  %d\n  >1MB  (large):    %d\n", s, m, l, x
    }' | tee "$RUN_DIR/neff_sizes.txt"
  say "Top 15 LARGEST NEFFs"
  find "$NTFF_DIR" -name '*.neff' -printf '%s\t%p\n' | sort -n -r | head -15
  say "Top 15 SMALLEST NEFFs"
  find "$NTFF_DIR" -name '*.neff' -printf '%s\t%p\n' | sort -n | head -15
elif [[ "$PROFILE" == "1" ]]; then
  echo "WARNING: PROFILE=1 but no profile directory was written"
fi

say "Done (exit $RUN_STATUS). Outputs in $RUN_DIR"
ls -la "$RUN_DIR"
exit "$RUN_STATUS"
