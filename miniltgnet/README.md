# MiniLTGNet

MiniLTGNet is a lightweight, fully independent validation framework placed under `miniltgnet/`.
It does not modify or depend on training logic inside `ltg_net/`.

## Purpose

- Rapid local validation on limited GPU/CPU.
- Keep the same task structure: trajectory-guided field forecasting.
- Provide a cheaper baseline for ablation before large-scale runs.

## Quick Start

Run from repository root:

```bash
python -m miniltgnet.cli prepare --config miniltgnet/configs/lite.yaml
python -m miniltgnet.cli sanity --config miniltgnet/configs/lite_sanity.yaml --steps 2 --backward
python -m miniltgnet.cli train --config miniltgnet/configs/lite.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite.yaml --checkpoint outputs/miniltg/lite/best.pt --split test
```

One-click:

- Linux: `bash miniltgnet/scripts/run_lite.sh`
- PowerShell: `powershell -ExecutionPolicy Bypass -File miniltgnet/scripts/run_lite.ps1`

## Seed + Ablation Experiments

Run default ablations with multi-seed sweep:

```bash
python miniltgnet/scripts/run_seed_ablation.py
```

Skip repeated prepare step:

```bash
python miniltgnet/scripts/run_seed_ablation.py --skip_prepare
```

Or one-click wrappers:

- Linux: `bash miniltgnet/scripts/run_seed_ablation.sh`
- PowerShell: `powershell -ExecutionPolicy Bypass -File miniltgnet/scripts/run_seed_ablation.ps1`

Outputs:

- `outputs/miniltg/seed_ablation/summary.csv`
- `outputs/miniltg/seed_ablation/summary.json`

## Validation Benchmark (Before Large Model)

Run a stricter benchmark against lightweight baselines on the same split:

```bash
python miniltgnet/scripts/run_validation_benchmark.py \
  --config miniltgnet/configs/lite_plus.yaml \
  --checkpoint outputs/miniltg/lite_plus/best.pt \
  --split test \
  --methods model persistence linear climatology
```

One-click wrappers:

- Linux: `bash miniltgnet/scripts/run_validation_benchmark.sh`
- PowerShell: `powershell -ExecutionPolicy Bypass -File miniltgnet/scripts/run_validation_benchmark.ps1`

Generated files:

- `outputs/miniltg/validation/benchmark_summary.csv`
- `outputs/miniltg/validation/benchmark_leadtime.csv`
- `outputs/miniltg/validation/benchmark_summary.json`
- `outputs/miniltg/validation/benchmark_report.md`

## Track Cache Health Check

`miniltgnet.cli prepare` now validates trajectory cache quality.  
If it reports all-NaN tracks, regenerate with:

```bash
python -m ltg_net.cli prepare --config configs/prepare_tracks_miniltg.yaml
python -m miniltgnet.cli prepare --config miniltgnet/configs/lite_plus.yaml
```

Recommended gate before moving to large model:

- Model should beat `persistence` on `rmse` and `acc`.
- Model should be at least not worse on `extreme_f1`.
- Lead-time curves should remain better than baselines after lead 3+.

## One-Round Standardized Alignment

This round enforces the same protocol (same split, same lead steps, same metrics) for all methods.

```bash
python miniltgnet/scripts/run_standardized_alignment.py \
  --config miniltgnet/configs/lite_plus.yaml \
  --checkpoint outputs/miniltg/lite_plus/best.pt \
  --split test \
  --methods model persistence linear climatology \
  --leads 1 2 3 4 5 6 \
  --output_dir outputs/miniltg/standardized_round
```

One-click wrappers:

- Linux: `bash miniltgnet/scripts/run_standardized_alignment.sh`
- PowerShell: `powershell -ExecutionPolicy Bypass -File miniltgnet/scripts/run_standardized_alignment.ps1`

Outputs:

- `outputs/miniltg/standardized_round/alignment_summary.csv`
- `outputs/miniltg/standardized_round/alignment_lead.csv`
- `outputs/miniltg/standardized_round/alignment_var_lead.csv`
- `outputs/miniltg/standardized_round/alignment_results.json`
- `outputs/miniltg/standardized_round/alignment_report.md`

## Key Files

- `miniltgnet/configs/base.yaml`
- `miniltgnet/configs/lite.yaml`
- `miniltgnet/configs/lite_plus.yaml`
- `miniltgnet/configs/lite_plus_v2.yaml`
- `miniltgnet/configs/lite_sanity.yaml`
- `miniltgnet/configs/ablation_no_extreme.yaml`
- `miniltgnet/configs/ablation_low_traj.yaml`
- `miniltgnet/configs/ablation_no_ema.yaml`
- `miniltgnet/model.py`
- `miniltgnet/trainer.py`

## Spectral + Track Optimization (v2)

Use `lite_plus_v2` for:

- stronger trajectory recovery (`traj_max_step_deg`, MAE + velocity terms),
- high-frequency spectrum fidelity (`lambda_spec`, `spec_high_freq_power`).

```bash
python -m miniltgnet.cli prepare --config miniltgnet/configs/lite_plus_v2.yaml
python -m miniltgnet.cli train --config miniltgnet/configs/lite_plus_v2.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite_plus_v2.yaml --checkpoint outputs/miniltg/lite_plus_v2/best.pt --split test
```

## Track/Spectrum Two-Stage Repair

When `track_mae` or `spectral_distance` regresses in multi-objective training, use decoupled stages:

1) `lite_plus_track_fix.yaml` (track-focused),
2) `lite_plus_spec_finetune.yaml` (small spectrum finetune from stage-1 best checkpoint).

```bash
python -m miniltgnet.cli train --config miniltgnet/configs/lite_plus_track_fix.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite_plus_track_fix.yaml --checkpoint outputs/miniltg/lite_plus_track_fix/best.pt --split test

python -m miniltgnet.cli train --config miniltgnet/configs/lite_plus_spec_finetune.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite_plus_spec_finetune.yaml --checkpoint outputs/miniltg/lite_plus_spec_finetune/best.pt --split test
```

Note:

- `lite_plus_spec_finetune.yaml` uses `optimization.epochs_mode: additional`, so `epochs` means extra epochs after resume.

## Dual Weak-Spot Upgrade (Track + Spectrum)

`lite_plus_dual_opt.yaml` adds:

- trajectory refinement head coupled with predicted field (`traj_refiner`),
- high-frequency field refinement branch (`hf_refiner`),
- geodesic/heading trajectory loss,
- multi-band spectral + gradient + laplacian structural losses,
- trajectory/spectral lambda warmup to reduce regression risk on rmse/acc/extreme_f1.

```bash
python -m miniltgnet.cli train --config miniltgnet/configs/lite_plus_dual_opt.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite_plus_dual_opt.yaml --checkpoint outputs/miniltg/lite_plus_dual_opt/best.pt --split test
```

## Metric-Wise Checkpoint Stitching

Compose a single checkpoint from metric-best sources:

```bash
python miniltgnet/scripts/compose_metric_checkpoint.py \
  --base_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_overall.pt \
  --track_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_track_mae.pt \
  --spectral_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_spectral.pt \
  --acc_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_acc.pt \
  --traj_mode swap \
  --field_mode blend --field_alpha 0.25 \
  --hf_mode swap \
  --encoder_mode blend --encoder_alpha 0.10 \
  --use_ema_if_available \
  --output outputs/miniltg/lite_plus_spec_finetune/selected/metric_stitch.pt
```

## Automatic Multi-Try Dual-Metric Search

This script runs multiple stitching attempts automatically and picks the best one under no-regression constraints.

```bash
python miniltgnet/scripts/run_dual_metric_search.py \
  --config miniltgnet/configs/lite_plus_spec_finetune.yaml \
  --base_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_overall.pt \
  --track_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_track_mae.pt \
  --spectral_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_spectral.pt \
  --acc_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_acc.pt \
  --split test \
  --max_batches 0 \
  --use_ema_if_available \
  --output_dir outputs/miniltg/dual_metric_search
```

## Final Mini Version (Recommended)

Use this fixed config for stable reproducibility:

- `miniltgnet/configs/lite_final.yaml`
- checkpoint: `outputs/miniltg/dual_metric_search_dualopt/best_dual_metric.pt`

Final strict benchmark command:

```bash
python miniltgnet/scripts/run_strict_benchmark.py \
  --config miniltgnet/configs/lite_final.yaml \
  --checkpoint outputs/miniltg/dual_metric_search_dualopt/best_dual_metric.pt \
  --split test \
  --methods model persistence linear climatology \
  --leads 1 2 3 4 5 6 \
  --bootstrap 2000 \
  --ci_alpha 0.05 \
  --reference persistence \
  --strict_p_threshold 0.95 \
  --output_dir outputs/miniltg/strict_final_v1
```

Expected metric profile (from `strict_dual_opt_infer_v4`):

- `rmse`: `0.5266` (better than persistence `0.5938`)
- `acc`: `0.7902` (better than persistence `0.7648`)
- `extreme_f1`: `0.5321` (better than persistence `0.4776`)
- `track_mae`: `11.5492` (better than persistence `11.6413`)
- `spectral_distance`: `0.2091` (still weaker than persistence `0.1663`)
