"""Measure quality vs speed trade-offs across Wan2.2 experiments.

Compares each experiment's output against the 10-step baseline using:
  1. LPIPS (perceptual similarity per frame)
  2. Latent MSE (how far the optimized latent drifted from baseline)
  3. Temporal consistency (frame-to-frame optical flow variance)

Usage:
    pip install lpips
    python3 measure_quality.py

    # Or compare specific experiments:
    python3 measure_quality.py --baseline 01_baseline_10step --compare 03_baseline_5step 09_5step_cfg_and_sharpen

Output:
    Prints a summary table and saves results to /tmp/wan2_experiments/quality_report.json
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np

EXPERIMENTS_DIR = os.environ.get("OUTPUT_DIR", "/tmp/wan2_experiments")


def load_video_frames(video_path):
    """Load video as tensor of shape [N, C, H, W] in [0, 1]."""
    try:
        from torchvision import io as tvio
        frames, _, _ = tvio.read_video(video_path, pts_unit='sec')
        # frames: [N, H, W, C] uint8
        return frames.float().permute(0, 3, 1, 2) / 255.0
    except Exception:
        # Fallback: use imageio
        import imageio
        reader = imageio.mimread(video_path, memsize=2**30)
        frames = np.stack(reader)  # [N, H, W, C]
        t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        return t


def compute_lpips(baseline_frames, optimized_frames):
    """Compute per-frame LPIPS between baseline and optimized videos.

    Returns dict with median, mean, max, p95, and per-frame scores.
    LPIPS < 0.15 = barely noticeable difference.
    """
    try:
        import lpips
    except ImportError:
        print("  [SKIP] lpips not installed. Run: pip install lpips")
        return None

    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    n = min(len(baseline_frames), len(optimized_frames))
    scores = []

    for i in range(n):
        b = baseline_frames[i].unsqueeze(0) * 2 - 1  # [0,1] -> [-1,1]
        o = optimized_frames[i].unsqueeze(0) * 2 - 1
        with torch.no_grad():
            score = loss_fn(b, o).item()
        scores.append(score)

    scores_sorted = sorted(scores)
    return {
        'median': scores_sorted[len(scores_sorted) // 2],
        'mean': sum(scores) / len(scores),
        'max': max(scores),
        'p95': scores_sorted[int(len(scores_sorted) * 0.95)],
        'per_frame': scores,
    }


def compute_latent_mse(baseline_latent_path, optimized_latent_path):
    """Compute MSE between baseline and optimized latent tensors.

    Lower = closer to baseline. Useful for measuring drift from
    interventions independent of the VAE decoder.
    """
    if not os.path.exists(baseline_latent_path) or not os.path.exists(optimized_latent_path):
        print(f"  [SKIP] Latent files not found")
        return None

    baseline = torch.load(baseline_latent_path, map_location='cpu', weights_only=True)
    optimized = torch.load(optimized_latent_path, map_location='cpu', weights_only=True)

    mse = torch.nn.functional.mse_loss(optimized.float(), baseline.float()).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        baseline.float().flatten().unsqueeze(0),
        optimized.float().flatten().unsqueeze(0)
    ).item()

    return {
        'mse': mse,
        'cosine_similarity': cosine_sim,
    }


def compute_temporal_consistency(frames):
    """Compute temporal consistency as frame-to-frame difference variance.

    Lower variance = more temporally consistent (less flickering).
    Uses pixel-space differences as a proxy for optical flow.
    """
    if len(frames) < 2:
        return None

    diffs = []
    for i in range(1, len(frames)):
        diff = (frames[i] - frames[i - 1]).abs().mean().item()
        diffs.append(diff)

    diffs = np.array(diffs)
    return {
        'mean_frame_diff': float(diffs.mean()),
        'std_frame_diff': float(diffs.std()),
        'max_frame_diff': float(diffs.max()),
        'variance': float(diffs.var()),
    }


def load_experiment_metadata(exp_dir):
    """Load metadata.json from an experiment directory."""
    meta_path = os.path.join(exp_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def find_experiments(base_dir):
    """Find all experiment directories that have output videos."""
    experiments = []
    for entry in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, entry)
        if os.path.isdir(exp_dir):
            # Look for warm run video (run 2)
            video = os.path.join(exp_dir, "video_run2.mp4")
            if not os.path.exists(video):
                video = os.path.join(exp_dir, "video_run1.mp4")
            if os.path.exists(video):
                experiments.append({
                    'name': entry,
                    'dir': exp_dir,
                    'video': video,
                    'latent': os.path.join(exp_dir, "latent_run2.pt")
                              if os.path.exists(os.path.join(exp_dir, "latent_run2.pt"))
                              else os.path.join(exp_dir, "latent_run1.pt"),
                })
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Measure quality across Wan2.2 experiments")
    parser.add_argument("--baseline", default="01_baseline_10step",
                        help="Name of baseline experiment directory")
    parser.add_argument("--compare", nargs="*", default=None,
                        help="Specific experiments to compare (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: quality_report.json in experiments dir)")
    args = parser.parse_args()

    base_dir = EXPERIMENTS_DIR
    if not os.path.exists(base_dir):
        print(f"Error: experiments directory not found: {base_dir}")
        print(f"Run experiments first with: bash run_experiments.sh")
        sys.exit(1)

    experiments = find_experiments(base_dir)
    if not experiments:
        print(f"No experiments found in {base_dir}")
        sys.exit(1)

    # Find baseline
    baseline = None
    for exp in experiments:
        if exp['name'] == args.baseline:
            baseline = exp
            break

    if baseline is None:
        print(f"Baseline '{args.baseline}' not found. Available experiments:")
        for exp in experiments:
            print(f"  - {exp['name']}")
        sys.exit(1)

    # Filter experiments to compare
    if args.compare:
        compare_exps = [e for e in experiments if e['name'] in args.compare]
    else:
        compare_exps = [e for e in experiments if e['name'] != args.baseline]

    print(f"Baseline: {baseline['name']}")
    print(f"Comparing: {len(compare_exps)} experiments")
    print()

    # Load baseline video and latent
    print(f"Loading baseline video: {baseline['video']}")
    baseline_frames = load_video_frames(baseline['video'])
    baseline_temporal = compute_temporal_consistency(baseline_frames)
    print(f"  Baseline: {len(baseline_frames)} frames, temporal variance={baseline_temporal['variance']:.6f}")
    print()

    results = []

    for exp in compare_exps:
        print(f"─── {exp['name']} ───")

        # Load metadata for timing info
        meta = load_experiment_metadata(exp['dir'])
        timing = {}
        if meta and 'results' in meta:
            warm = meta['results'][-1]  # last run = warm
            timing = {
                'denoise_s': warm['denoise'],
                'vae_s': warm['vae'],
                'total_s': warm['total'],
                'cache_hits': warm.get('cache_hits', 0),
                'num_steps': meta['config']['num_steps'],
            }
            print(f"  Timing: denoise={warm['denoise']:.1f}s, vae={warm['vae']:.1f}s, total={warm['total']:.1f}s")

        # Load optimized video
        opt_frames = load_video_frames(exp['video'])

        # LPIPS
        lpips_result = compute_lpips(baseline_frames, opt_frames)
        if lpips_result:
            quality_label = "GOOD" if lpips_result['median'] < 0.15 else "DEGRADED"
            print(f"  LPIPS:   median={lpips_result['median']:.4f}, "
                  f"max={lpips_result['max']:.4f}, p95={lpips_result['p95']:.4f} [{quality_label}]")

        # Latent MSE
        latent_result = compute_latent_mse(baseline['latent'], exp['latent'])
        if latent_result:
            print(f"  Latent:  MSE={latent_result['mse']:.6f}, "
                  f"cosine_sim={latent_result['cosine_similarity']:.4f}")

        # Temporal consistency
        temporal_result = compute_temporal_consistency(opt_frames)
        if temporal_result:
            delta_var = temporal_result['variance'] - baseline_temporal['variance']
            direction = "MORE flicker" if delta_var > 0 else "LESS flicker"
            print(f"  Temporal: variance={temporal_result['variance']:.6f} "
                  f"(delta={delta_var:+.6f}, {direction})")

        result = {
            'experiment': exp['name'],
            'config': meta['config'] if meta else {},
            'timing': timing,
            'lpips': {k: v for k, v in (lpips_result or {}).items() if k != 'per_frame'},
            'latent': latent_result,
            'temporal': temporal_result,
        }
        # Keep per-frame LPIPS in full results but not in summary
        if lpips_result:
            result['lpips_per_frame'] = lpips_result['per_frame']
        results.append(result)
        print()

    # ── Summary table ──
    print()
    print("=" * 100)
    print(f"{'Experiment':<35} {'Steps':>5} {'Denoise':>8} {'Total':>8} {'LPIPS':>8} {'Latent MSE':>11} {'Temporal':>10}")
    print("=" * 100)

    # Baseline row
    bl_timing = {}
    bl_meta = load_experiment_metadata(baseline['dir'])
    if bl_meta and 'results' in bl_meta:
        bl_warm = bl_meta['results'][-1]
        bl_timing = bl_warm
    print(f"{'* ' + baseline['name']:<35} "
          f"{bl_meta['config']['num_steps'] if bl_meta else '?':>5} "
          f"{bl_timing.get('denoise', 0):>7.1f}s "
          f"{bl_timing.get('total', 0):>7.1f}s "
          f"{'(ref)':>8} "
          f"{'(ref)':>11} "
          f"{baseline_temporal['variance']:>10.6f}")

    for r in results:
        lpips_str = f"{r['lpips']['median']:.4f}" if r.get('lpips') else "n/a"
        latent_str = f"{r['latent']['mse']:.6f}" if r.get('latent') else "n/a"
        temporal_str = f"{r['temporal']['variance']:.6f}" if r.get('temporal') else "n/a"
        steps = r['timing'].get('num_steps', '?')
        denoise = r['timing'].get('denoise_s', 0)
        total = r['timing'].get('total_s', 0)
        print(f"  {r['experiment']:<33} "
              f"{steps:>5} "
              f"{denoise:>7.1f}s "
              f"{total:>7.1f}s "
              f"{lpips_str:>8} "
              f"{latent_str:>11} "
              f"{temporal_str:>10}")

    print("=" * 100)
    print("LPIPS < 0.15 = barely visible difference from baseline")
    print()

    # ── Save report ──
    output_path = args.output or os.path.join(base_dir, "quality_report.json")
    report = {
        'baseline': baseline['name'],
        'baseline_temporal': baseline_temporal,
        'comparisons': results,
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report saved to {output_path}")


if __name__ == "__main__":
    main()
