"""Wan2.2-TI2V-5B OPTIMIZED inference with configurable speedup techniques.

All optimizations toggled via environment variables so you can sweep configs
without editing code. Drop-in replacement for inference_neuron_ti2v.py.

Optimizations:
  1. Step reduction          — NUM_STEPS (already in base)
  2. Scheduler shift tuning  — SCHEDULER_SHIFT (default 1)
  3. TeaCache step-caching   — TEACACHE_ENABLED, TEACACHE_THRESHOLD
  4. CFG rescaling           — CFG_RESCALE_ENABLED, CFG_RESCALE_FACTOR
  5. Latent sharpening       — SHARPEN_ENABLED, SHARPEN_FACTOR, SHARPEN_START_FRAC
  6. Temporal consistency    — TEMPORAL_ENABLED, TEMPORAL_ALPHA, TEMPORAL_START_FRAC
  7. Early-exit interpolation — EARLY_EXIT_ENABLED, EARLY_EXIT_SKIP_LAST_N

Also saves:
  - Latent tensor after denoising (for MSE comparison)
  - Per-step timing breakdown
  - Config metadata JSON alongside output video
"""
import os
import sys
import time
import math
import random
import logging
import gc
import json

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [rank %(process)d]: %(message)s',
    stream=sys.stdout,
    force=True,
)
for name in ['torch', 'transformers', 'torch_neuronx', 'torch_mlir']:
    logging.getLogger(name).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# ── Paths & parallelism ──
WAN_DIR = os.environ.get("WAN_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WAN_DIR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/Wan2.2-TI2V-5B")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "8"))
SP_DEGREE = int(os.environ.get("SP_DEGREE", "1"))
T5_RANK = int(os.environ.get("T5_RANK", "2"))
VAE_RANK = 0
VAE_TP_DEGREE = int(os.environ.get("VAE_TP_DEGREE", "1"))
VAE_TP_RANKS = list(range(VAE_TP_DEGREE))
USE_NST_SELF_ATTN = os.environ.get("USE_NST_SELF_ATTN", "0") == "1"

NEURON_DEVICE = torch.device("neuron")

# ── Optimization knobs (all configurable via env) ──
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
NUM_RUNS = int(os.environ.get("NUM_RUNS", "2"))
FRAME_NUM = int(os.environ.get("FRAME_NUM", "81"))
SCHEDULER_SHIFT = int(os.environ.get("SCHEDULER_SHIFT", "1"))

# TeaCache
TEACACHE_ENABLED = os.environ.get("TEACACHE_ENABLED", "0") == "1"
TEACACHE_THRESHOLD = float(os.environ.get("TEACACHE_THRESHOLD", "0.04"))

# CFG rescaling
CFG_RESCALE_ENABLED = os.environ.get("CFG_RESCALE_ENABLED", "0") == "1"
CFG_RESCALE_FACTOR = float(os.environ.get("CFG_RESCALE_FACTOR", "0.7"))

# Latent sharpening
SHARPEN_ENABLED = os.environ.get("SHARPEN_ENABLED", "0") == "1"
SHARPEN_FACTOR = float(os.environ.get("SHARPEN_FACTOR", "0.1"))
SHARPEN_START_FRAC = float(os.environ.get("SHARPEN_START_FRAC", "0.5"))

# Temporal consistency
TEMPORAL_ENABLED = os.environ.get("TEMPORAL_ENABLED", "0") == "1"
TEMPORAL_ALPHA = float(os.environ.get("TEMPORAL_ALPHA", "0.15"))
TEMPORAL_START_FRAC = float(os.environ.get("TEMPORAL_START_FRAC", "0.33"))

# Early-exit interpolation
EARLY_EXIT_ENABLED = os.environ.get("EARLY_EXIT_ENABLED", "0") == "1"
EARLY_EXIT_SKIP_LAST_N = int(os.environ.get("EARLY_EXIT_SKIP_LAST_N", "1"))

# Output
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/wan2_experiments")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
SAVE_LATENT = os.environ.get("SAVE_LATENT", "1") == "1"


def get_config_dict():
    """Return a dict of all optimization settings for logging/saving."""
    return {
        "num_steps": NUM_STEPS,
        "num_runs": NUM_RUNS,
        "frame_num": FRAME_NUM,
        "tp_degree": TP_DEGREE,
        "sp_degree": SP_DEGREE,
        "scheduler_shift": SCHEDULER_SHIFT,
        "teacache_enabled": TEACACHE_ENABLED,
        "teacache_threshold": TEACACHE_THRESHOLD,
        "cfg_rescale_enabled": CFG_RESCALE_ENABLED,
        "cfg_rescale_factor": CFG_RESCALE_FACTOR,
        "sharpen_enabled": SHARPEN_ENABLED,
        "sharpen_factor": SHARPEN_FACTOR,
        "sharpen_start_frac": SHARPEN_START_FRAC,
        "temporal_enabled": TEMPORAL_ENABLED,
        "temporal_alpha": TEMPORAL_ALPHA,
        "temporal_start_frac": TEMPORAL_START_FRAC,
        "early_exit_enabled": EARLY_EXIT_ENABLED,
        "early_exit_skip_last_n": EARLY_EXIT_SKIP_LAST_N,
        "experiment_name": EXPERIMENT_NAME,
    }


def setup_distributed():
    dist.init_process_group(backend="neuron")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


# ═══════════════════════════════════════════════════════════════════════
# LATENT-SPACE INTERVENTIONS
# These run on CPU between denoising steps (outside the compiled graph).
# ═══════════════════════════════════════════════════════════════════════

def apply_cfg_rescaling(noise_pred, noise_pred_cond, rescale_factor):
    """Rescale CFG output to match conditional prediction's variance.

    At low step counts, CFG can cause oversaturation. This rescales the
    guided prediction to keep its variance aligned with the conditional
    prediction, preventing color blowout and contrast artifacts.

    Args:
        noise_pred: The CFG-combined prediction (uncond + scale * (cond - uncond))
        noise_pred_cond: The conditional prediction alone
        rescale_factor: 0.0 = no rescaling, 1.0 = fully match cond variance
                        Sweet spot: 0.5–0.9
    """
    std_pred = noise_pred.std()
    std_cond = noise_pred_cond.std()
    if std_pred > 0:
        noise_pred = noise_pred * (
            rescale_factor * (std_cond / std_pred)
            + (1 - rescale_factor)
        )
    return noise_pred


def apply_latent_sharpening(latent, sharpen_factor):
    """Amplify high-frequency detail in the latent.

    With fewer steps, the solver takes bigger jumps in noise space and
    high-frequency detail (edges, textures, fine motion) gets smoothed.
    This extracts the high-frequency residual via blur subtraction and
    amplifies it — doing manually what extra steps would do automatically.

    Args:
        latent: [z_dim, T_latent, H_latent, W_latent]
        sharpen_factor: Amplification strength (0.05–0.2 safe range)
    """
    # 3D average pooling to get low-freq component
    blurred = torch.nn.functional.avg_pool3d(
        latent.unsqueeze(0),
        kernel_size=3, stride=1, padding=1
    ).squeeze(0)
    high_freq = latent - blurred
    return latent + sharpen_factor * high_freq


def apply_temporal_consistency(latent, alpha):
    """Dampen large frame-to-frame jumps in the latent.

    Video diffusion's biggest failure mode at low steps is temporal
    incoherence — flickering, objects shifting between frames. This
    constrains adjacent frames by dampening their differences.

    Args:
        latent: [z_dim, T_latent, H_latent, W_latent] where T_latent=21 for 81 frames
        alpha: Blending strength (0.05–0.25). Too high = over-smoothed motion.
    """
    for f in range(1, latent.shape[1]):
        diff = latent[:, f] - latent[:, f - 1]
        latent[:, f] = latent[:, f] - alpha * diff
    return latent


def apply_early_exit_interpolation(latent, latent_history):
    """Extrapolate the latent trajectory instead of running the DiT.

    Uses the velocity (difference between last two latents) to predict
    where the latent would end up, skipping the expensive DiT forward pass.

    Args:
        latent: Current latent
        latent_history: List of previous latents (need at least 2)
    Returns:
        Extrapolated latent
    """
    if len(latent_history) >= 2:
        velocity = latent_history[-1] - latent_history[-2]
        return latent + velocity * 0.5
    return latent


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    rank, world_size = setup_distributed()
    torch.set_grad_enabled(False)

    if SP_DEGREE > 1:
        from models.parallel_state import init_parallel_groups
        init_parallel_groups(sp_degree=SP_DEGREE, tp_degree=TP_DEGREE)

    opt_config = get_config_dict()

    if rank == 0:
        logger.info("=" * 70)
        logger.info("  Wan2.2-TI2V-5B  OPTIMIZED BENCHMARK")
        logger.info("=" * 70)
        logger.info(f"  Experiment:      {EXPERIMENT_NAME}")
        logger.info(f"  Steps:           {NUM_STEPS}")
        logger.info(f"  Scheduler shift: {SCHEDULER_SHIFT}")
        logger.info(f"  TP={TP_DEGREE}, SP={SP_DEGREE}")
        logger.info(f"  TeaCache:        {'ON (threshold={})'.format(TEACACHE_THRESHOLD) if TEACACHE_ENABLED else 'OFF'}")
        logger.info(f"  CFG rescale:     {'ON (factor={})'.format(CFG_RESCALE_FACTOR) if CFG_RESCALE_ENABLED else 'OFF'}")
        logger.info(f"  Sharpening:      {'ON (factor={}, start={})'.format(SHARPEN_FACTOR, SHARPEN_START_FRAC) if SHARPEN_ENABLED else 'OFF'}")
        logger.info(f"  Temporal:        {'ON (alpha={}, start={})'.format(TEMPORAL_ALPHA, TEMPORAL_START_FRAC) if TEMPORAL_ENABLED else 'OFF'}")
        logger.info(f"  Early exit:      {'ON (skip last {})'.format(EARLY_EXIT_SKIP_LAST_N) if EARLY_EXIT_ENABLED else 'OFF'}")
        logger.info("=" * 70)

    # ── Import Wan modules ──
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE
    from wan.utils.utils import best_output_size, masks_like
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from models.tp_utils import init_tp_group, shard_model_tp, get_tp_rank

    config = WAN_CONFIGS['ti2v-5B']

    # ── Load T5 ──
    if rank == T5_RANK:
        logger.info(f"Loading T5 on rank {T5_RANK}...")
        text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(MODEL_PATH, config.t5_checkpoint),
            tokenizer_path=os.path.join(MODEL_PATH, config.t5_tokenizer))
        text_encoder.model = text_encoder.model.to(NEURON_DEVICE)
        logger.info(f"T5 loaded on Neuron (rank {T5_RANK})")
    else:
        text_encoder = None

    # ── Load VAE ──
    if VAE_TP_DEGREE > 1 and rank in VAE_TP_RANKS:
        from models.vae_tp import create_vae_tp_group, shard_vae_model_tp
        if rank == VAE_TP_RANKS[0]:
            logger.info(f"Loading VAE with TP={VAE_TP_DEGREE}...")
        os.environ["USE_NKI_VAE"] = "0"
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16) if isinstance(s, torch.Tensor) else s for s in vae.scale]
        create_vae_tp_group(VAE_TP_RANKS)
        vae_tp_rank = VAE_TP_RANKS.index(rank)
        shard_vae_model_tp(vae.model, vae_tp_rank, VAE_TP_DEGREE)
    elif VAE_TP_DEGREE <= 1 and rank == VAE_RANK:
        logger.info(f"Loading VAE on rank {VAE_RANK}...")
        vae = Wan2_2_VAE(
            vae_pth=os.path.join(MODEL_PATH, config.vae_checkpoint),
            device=torch.device('cpu'))
        vae.model = vae.model.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        vae.scale = [s.to(device=NEURON_DEVICE, dtype=torch.bfloat16) if isinstance(s, torch.Tensor) else s for s in vae.scale]
    else:
        vae = None

    # ── Load DiT ──
    if rank == 0:
        logger.info(f"Loading WanModel from {MODEL_PATH}...")
    model = WanModel.from_pretrained(MODEL_PATH)
    model.eval().requires_grad_(False)
    init_tp_group(tp_degree=TP_DEGREE, sp_degree=SP_DEGREE)
    tp_rank = get_tp_rank()
    shard_model_tp(model, tp_rank, TP_DEGREE)

    if SP_DEGREE > 1:
        from models.sp_patch import patch_model_for_sp
        patch_model_for_sp(model)

    model = model.to(torch.bfloat16).to(NEURON_DEVICE)
    if rank == 0:
        logger.info("DiT moved to Neuron, compiling sub-modules...")

    model.patch_embedding = torch.compile(model.patch_embedding, backend='neuron', dynamic=False)
    model.text_embedding = torch.compile(model.text_embedding, backend='neuron', dynamic=False)
    model.head = torch.compile(model.head, backend='neuron', dynamic=False)
    for block in model.blocks:
        block.ffn = torch.compile(block.ffn, backend='neuron', dynamic=False)

    if rank == 0:
        logger.info(f"Compiled: patch_embed, text_embed, head, FFN x{len(model.blocks)}")

    dist.barrier()
    if rank == 0:
        logger.info("All models loaded!")

    # ── Encode prompt ──
    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, "
        "distant green hills, and a blue sky dotted with white clouds."
    )
    n_prompt = config.sample_neg_prompt

    ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
    ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)

    if rank == T5_RANK:
        context = text_encoder([prompt], NEURON_DEVICE)
        context_null = text_encoder([n_prompt], NEURON_DEVICE)
        ctx_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_tensor[0, :context[0].shape[0]] = context[0].to(torch.bfloat16)
        ctx_null_tensor = torch.zeros(1, 512, 4096, dtype=torch.bfloat16, device=NEURON_DEVICE)
        ctx_null_tensor[0, :context_null[0].shape[0]] = context_null[0].to(torch.bfloat16)

    dist.broadcast(ctx_tensor, src=T5_RANK)
    dist.broadcast(ctx_null_tensor, src=T5_RANK)
    context = [ctx_tensor]
    context_null = [ctx_null_tensor]

    if rank == T5_RANK:
        del text_encoder
        gc.collect()

    # ── Prepare image and noise ──
    import torchvision.transforms.functional as TF

    image_path = os.path.join(MODEL_PATH, "examples/i2v_input.JPG")
    img = Image.open(image_path).convert("RGB")

    vae_stride = config.vae_stride
    patch_size = config.patch_size
    max_area = 480 * 832

    ih, iw = img.height, img.width
    dh = patch_size[1] * vae_stride[1]
    dw = patch_size[2] * vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))

    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).unsqueeze(1)

    F = FRAME_NUM
    z_dim = config.vae_dim[0] if hasattr(config, 'vae_dim') else 48
    T_latent = (F - 1) // vae_stride[0] + 1
    H_latent = oh // vae_stride[1]
    W_latent = ow // vae_stride[2]
    seq_len = (
        T_latent
        * (oh // vae_stride[1])
        * (ow // vae_stride[2])
        // (patch_size[1] * patch_size[2])
    )

    seed = 42
    seed_g = torch.Generator(device=torch.device("cpu"))
    seed_g.manual_seed(seed)

    noise = torch.randn(
        z_dim, T_latent, H_latent, W_latent,
        dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

    # Encode image with VAE
    z_img_shape = (z_dim, 1, H_latent, W_latent)
    z_img_device = torch.zeros(z_img_shape, dtype=torch.bfloat16, device=NEURON_DEVICE)
    if rank == VAE_RANK:
        img_neuron = img_tensor.to(device=NEURON_DEVICE, dtype=torch.bfloat16)
        z = vae.encode([img_neuron])
        z_result = z[0].to(torch.bfloat16).contiguous()
        if z_result.device != NEURON_DEVICE:
            z_result = z_result.to(NEURON_DEVICE)
        z_img_device.copy_(z_result)
    dist.broadcast(z_img_device, src=VAE_RANK)

    noise = noise.to(torch.bfloat16).to(NEURON_DEVICE)
    mask1, mask2 = masks_like([noise.cpu()], zero=True)
    mask2_device = [m.to(torch.bfloat16).to(NEURON_DEVICE) for m in mask2]

    latent = noise.clone()
    latent = (1. - mask2_device[0]) * z_img_device.expand_as(noise) + mask2_device[0] * latent

    arg_c = {'context': [context[0][0]], 'seq_len': seq_len}
    arg_null = {'context': [context_null[0][0]], 'seq_len': seq_len}

    latent_orig = latent.clone()
    run_results = []

    # Create output directory
    exp_dir = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
    if rank == 0:
        os.makedirs(exp_dir, exist_ok=True)

    for run_idx in range(NUM_RUNS):
        run_label = f"Run {run_idx + 1}/{NUM_RUNS}"
        if rank == 0:
            logger.info(f"═══ {run_label} ({NUM_STEPS} steps) ═══")

        latent = latent_orig.clone()

        # ── Scheduler with configurable shift ──
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=SCHEDULER_SHIFT, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(NUM_STEPS, device=torch.device("cpu"), shift=config.sample_shift)
        timesteps = sample_scheduler.timesteps

        # ── TeaCache state ──
        prev_t_emb_norm = None
        cached_noise_pred = None
        cache_hits = 0

        # ── Early-exit state ──
        latent_history = []

        # ── Per-step timing ──
        step_times = []
        step_types = []  # 'full', 'cached', 'interpolated'

        run_start = time.time()
        denoise_start = time.time()

        for step_idx, t in enumerate(tqdm(timesteps, disable=(rank != 0), desc=run_label)):
            step_start = time.time()

            # ─── EARLY EXIT: skip last N steps via interpolation ───
            if (EARLY_EXIT_ENABLED
                    and step_idx >= len(timesteps) - EARLY_EXIT_SKIP_LAST_N
                    and len(latent_history) >= 2):
                latent_cpu = latent.cpu()
                latent_cpu = apply_early_exit_interpolation(latent_cpu, latent_history)
                latent = latent_cpu.to(NEURON_DEVICE)
                latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent
                step_times.append(time.time() - step_start)
                step_types.append('interpolated')
                if rank == 0:
                    logger.info(f"  Step {step_idx}: INTERPOLATED (early exit)")
                continue

            # ─── TEACACHE: check if we can skip this step ───
            if TEACACHE_ENABLED and step_idx > 0:
                # Compute timestep embedding norm on CPU for cache decision
                # (dynamic control flow can't be inside compiled graph)
                t_val = t.item()
                if prev_t_emb_norm is not None:
                    # Use timestep value as proxy for embedding distance
                    # (actual embedding extraction would require model internals)
                    delta = abs(t_val - prev_t_val) / max(abs(t_val), 1e-8)
                    if delta < TEACACHE_THRESHOLD and cached_noise_pred is not None:
                        # Reuse cached prediction — skip both DiT forward passes
                        noise_pred = cached_noise_pred
                        temp_x0 = sample_scheduler.step(
                            noise_pred.cpu().unsqueeze(0), t,
                            latent.cpu().unsqueeze(0),
                            return_dict=False, generator=seed_g)[0]
                        latent = temp_x0.squeeze(0).to(NEURON_DEVICE)
                        latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent
                        cache_hits += 1
                        step_times.append(time.time() - step_start)
                        step_types.append('cached')
                        if rank == 0:
                            logger.info(f"  Step {step_idx}: CACHED (delta={delta:.4f} < {TEACACHE_THRESHOLD})")
                        latent_history.append(latent.cpu().clone())
                        continue

            prev_t_val = t.item()
            prev_t_emb_norm = abs(t.item())

            # ─── FULL DiT FORWARD PASS ───
            latent_model_input = [latent]
            t_device = t.to(NEURON_DEVICE)
            temp_ts = (mask2_device[0][0][:, ::2, ::2] * t_device).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_device])
            timestep = temp_ts.unsqueeze(0)

            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]

            guide_scale = config.sample_guide_scale
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            # ─── POINT A: CFG rescaling (before scheduler step) ───
            if CFG_RESCALE_ENABLED:
                noise_pred = apply_cfg_rescaling(noise_pred, noise_pred_cond, CFG_RESCALE_FACTOR)

            cached_noise_pred = noise_pred  # save for TeaCache

            # Scheduler step (on CPU)
            temp_x0 = sample_scheduler.step(
                noise_pred.cpu().unsqueeze(0), t,
                latent.cpu().unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            # ─── POINT B: Latent-space interventions (on CPU, before re-conditioning) ───
            if SHARPEN_ENABLED and step_idx >= int(len(timesteps) * SHARPEN_START_FRAC):
                latent = apply_latent_sharpening(latent, SHARPEN_FACTOR)

            if TEMPORAL_ENABLED and step_idx >= int(len(timesteps) * TEMPORAL_START_FRAC):
                latent = apply_temporal_consistency(latent, TEMPORAL_ALPHA)

            # Move back to device
            latent = latent.to(NEURON_DEVICE)

            # ─── POINT C: Re-conditioning blend (lock in image frame) ───
            latent = (1. - mask2_device[0]) * z_img_device.expand_as(latent) + mask2_device[0] * latent

            latent_history.append(latent.cpu().clone())
            step_times.append(time.time() - step_start)
            step_types.append('full')

        denoise_time = time.time() - denoise_start

        if rank == 0:
            full_steps = step_types.count('full')
            cached_steps = step_types.count('cached')
            interp_steps = step_types.count('interpolated')
            logger.info(f"{run_label} denoising: {denoise_time:.1f}s "
                        f"({full_steps} full, {cached_steps} cached, {interp_steps} interpolated)")

        # ── Save latent for quality comparison ──
        if SAVE_LATENT and rank == 0:
            latent_path = os.path.join(exp_dir, f"latent_run{run_idx + 1}.pt")
            torch.save(latent.cpu(), latent_path)
            logger.info(f"Latent saved to {latent_path}")

        # ── Decode with VAE ──
        vae_start = time.time()
        output_path = os.path.join(exp_dir, f"video_run{run_idx + 1}.mp4")

        if VAE_TP_DEGREE > 1 and rank in VAE_TP_RANKS:
            x0 = [latent.to(torch.bfloat16)]
            videos = vae.decode(x0)
            if rank == VAE_RANK:
                video = videos[0]
                import imageio
                video_cpu = video.cpu().float()
                video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
                if video_np.dim() == 4 and video_np.shape[0] == 3:
                    video_np = video_np.permute(1, 2, 3, 0)
                frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
                imageio.mimwrite(output_path, frames, fps=24, codec='libx264')
                logger.info(f"Video saved to {output_path} ({len(frames)} frames)")
        elif rank == VAE_RANK:
            x0 = [latent.to(torch.bfloat16)]
            videos = vae.decode(x0)
            video = videos[0]
            import imageio
            video_cpu = video.cpu().float()
            video_np = ((video_cpu.clamp(-1, 1) * 0.5 + 0.5) * 255).byte()
            if video_np.dim() == 4 and video_np.shape[0] == 3:
                video_np = video_np.permute(1, 2, 3, 0)
            frames = [video_np[i].numpy() for i in range(video_np.shape[0])]
            imageio.mimwrite(output_path, frames, fps=24, codec='libx264')
            logger.info(f"Video saved to {output_path} ({len(frames)} frames)")

        dist.barrier()
        vae_time = time.time() - vae_start
        run_time = time.time() - run_start

        if rank == 0:
            logger.info(f"═══ {run_label} DONE: denoise={denoise_time:.1f}s, vae={vae_time:.1f}s, total={run_time:.1f}s ═══")
            result = {
                'run': run_idx + 1,
                'denoise': round(denoise_time, 2),
                'vae': round(vae_time, 2),
                'total': round(run_time, 2),
                'step_times': [round(t, 3) for t in step_times],
                'step_types': step_types,
                'cache_hits': cache_hits,
            }
            run_results.append(result)

    # ── Save experiment metadata ──
    if rank == 0 and run_results:
        warm = run_results[-1]
        logger.info("")
        logger.info("=" * 70)
        logger.info("  OPTIMIZED BENCHMARK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Experiment:      {EXPERIMENT_NAME}")
        logger.info(f"  Steps:           {NUM_STEPS}")
        logger.info(f"  Warm denoise:    {warm['denoise']:.1f}s ({warm['denoise']/NUM_STEPS:.1f}s/step)")
        logger.info(f"  Warm VAE:        {warm['vae']:.1f}s")
        logger.info(f"  Warm total:      {warm['total']:.1f}s")
        if TEACACHE_ENABLED:
            logger.info(f"  Cache hits:      {warm['cache_hits']}/{NUM_STEPS}")
        logger.info("=" * 70)

        metadata = {
            'config': opt_config,
            'results': run_results,
        }
        meta_path = os.path.join(exp_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {meta_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
