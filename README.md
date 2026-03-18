# Lagrangian Trajectory Guided Generative Weather Forecasting for engineering-oriented prediction.

## 1. Project Positioning

LTG-Net is a physics-aware generative weather forecasting framework designed around a Lagrangian dynamical skeleton.  
It targets two coupled tasks:

- Field forecasting: multi-variable atmospheric fields (e.g., wind, geopotential, moisture, precipitation).
- Track forecasting: coherent weather-object trajectories (e.g., cyclone centers).

The core idea is:

1. Build a background dynamical representation from historical atmospheric states.
2. Predict future object trajectories as a dynamical skeleton.
3. Generate future fields conditioned on both background dynamics and predicted trajectories.

## 2. System Design and Architecture

LTG-Net has three primary modules:

1. Background Dynamical Encoder
2. Dynamical Skeleton Trajectory Head
3. Trajectory-Conditioned Field Generator

### 2.1 Background Dynamical Encoder

- Module: `SphereSpatiotemporalEncoder`
- Input: historical fields `x_hist`
- Outputs:
  - `z_bg_map`: spatial background representation
  - `z_bg_global`: global background representation

This module captures large-scale circulation control and multi-variable spatiotemporal context.

### 2.2 Trajectory Head (Skeleton Evolution)

- Modules:
  - `DeterministicTrajectoryPredictor`
  - `NeuralSDETrajectoryPredictor` (stochastic option)
- Input: `z_bg_global` + historical tracks `traj_hist`
- Output: future trajectories `traj_pred`

This branch explicitly models coherent-object evolution in a Lagrangian manner.

### 2.3 Trajectory-Conditioned Field Generator

- Modules:
  - `TrajectoryConditionedUNetGenerator`
  - `TrajectoryConditionedLatentDiffusion`
- Input: `x_hist`, `z_bg_map`, `traj_pred`
- Output: future fields `field_pred`

This branch restores and generates future atmospheric fields under trajectory guidance.

## 3. Forward Pipeline

```text
x_hist -> Background Encoder -> z_bg_map, z_bg_global
traj_hist + z_bg_global -> Trajectory Head -> traj_pred
x_hist + z_bg_map + traj_pred -> Generator -> field_pred
```

Optional components:

- `event_modulation`: event-adaptive modulation
- strict PDE terms in loss (when enabled)

## 4. Training Objective Design

LTG-Net uses a composite objective:

- `loss_field`: field reconstruction (L1/L2 with optional tail emphasis)
- `loss_traj`: trajectory supervision
- `loss_adv`: advection consistency
- `loss_phys`: physical consistency (including optional strict PDE terms)
- `loss_spec`: spectral consistency
- `loss_diff`: diffusion loss (for latent diffusion mode)

This objective set is designed to balance:

- numerical forecast quality,
- structural/topological consistency,
- physical interpretability.

## 5. Current Engineering Inference Strategy (Production Recommendation)

Current deployment follows a dual-checkpoint protocol:

- Field mode: use field-priority checkpoint for field/intensity-related outputs.
- Track mode: use track-priority checkpoint for trajectory/path outputs.

Important:

- The current closed-loop strategy is **field + track**.
- Fused checkpoint mode is **not required** for the current production path.

## 6. Physical Interpretability

LTG-Net provides explicit interpretability hooks at three levels:

1. Skeleton level: predicted trajectories are explicit, directly inspectable states.
2. Physics level: advection/PDE terms are decomposable and diagnosable.
3. Spectral level: multi-scale structure preservation can be measured via spectral metrics.

Compared with purely black-box pipelines, this improves traceability for model failure analysis.

## 7. Engineering Workflow

Recommended workflow:

1. Data prepare (`prepare`) for normalization and skeleton cache.
2. Sanity check (`sanity_check`) for forward/backward validation.
3. Recover stage training.
4. Finetune stage training.
5. Optional trajectory-head tuning for track MAE optimization.
6. Strict benchmark against persistence/linear/climatology baselines.

Common metrics:

- `rmse`
- `acc`
- `track_mae`
- `extreme_f1`
- `spectral_distance`

## 8. Scope and Roadmap

Suitable for:

- engineering weather forecast prototyping,
- trajectory-field coupled forecasting tasks,
- physically constrained generative forecasting research.

Planned improvements:

- larger data scale and broader temporal coverage,
- stronger cross-period generalization,
- robust multi-seed reproducibility,
- mode-aware gate refinement (field gate and track gate separated).

## 9. Key Code Entry Points

- Model integration: `ltg_net/models/ltg_net.py`
- Trajectory models: `ltg_net/models/trajectory.py`
- Field generators: `ltg_net/models/generator_unet.py`, `ltg_net/models/diffusion.py`
- Composite loss: `ltg_net/losses/composite.py`
- Training loop: `ltg_net/train/trainer.py`, `ltg_net/train/loops.py`
- Strict benchmark: `scripts/run_ltg_strict_benchmark.py`

## 10. Acknowledgments

This project uses ERA5 reanalysis data.

- Dataset: ERA5
- Provider: Copernicus Climate Change Service (C3S)
- Implementation and distribution infrastructure: ECMWF

Please follow ERA5/C3S data license and attribution requirements when publishing results.

## 11. Contact

- Contact: Yuzhi Wang (wangyzh267@mail2.sysu.edu.cn)
- Institution: College of Atmospheric Sciences, Sun Yat-sen University

# Details of project
## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Project Layout

- `configs/`: stage-wise training configs.
- `ltg_net/data/`: ERA5 loaders and skeleton extraction.
- `ltg_net/models/`: encoder, trajectory, generators, integrated LTG-Net.
- `ltg_net/losses/`: multi-objective constraints.
- `ltg_net/train/`: training loops and trainer.
- `ltg_net/eval/`: evaluation and diagnostics.
- `scripts/`: reproducible entry scripts.

## Quick Start

1. Prepare dataset index and normalization stats:

```bash
python scripts/download_era5.py --start 1990-01 --end 2021-12 --output data/era5.zarr --time-step-hours 6 --days-per-request 7 --overwrite
python scripts/prepare_era5.py --config configs/stage1_regional.yaml
```

Notes for ERA5 download:

- Install CDS API client and configure credential file `~/.cdsapirc` first.
- Accept ERA5 dataset license on Copernicus Climate Data Store website before first request.
- The downloader builds LTG-Net expected variables:
  - `z500, z850, u850, v850, t850, q850, msl, tp`
- To download only a subregion, add `--area NORTH WEST SOUTH EAST`.

2. Stage-1 deterministic training (regional prototype):

```bash
python scripts/train.py --config configs/stage1_regional.yaml
```

3. Stage-2 stochastic trajectory and event modulation:

```bash
python scripts/train.py --config configs/stage2_stochastic.yaml
```

4. Stage-3 global expansion:

```bash
python scripts/train.py --config configs/stage3_global.yaml
```

5. Evaluation:

```bash
python scripts/evaluate.py --config configs/stage3_global.yaml --checkpoint outputs/stage3/best.pt
```

## Redesigned Stage-2 Training

If stage-2 training is dominated by physics loss, use the redesigned objective.
Run sanity first (few batches only), then full training:

```bash
python -m ltg_net.cli train --config configs/stage2_stochastic_v2_sanity.yaml
python -m ltg_net.cli train --config configs/stage2_stochastic_v2.yaml
python -m ltg_net.cli evaluate --config configs/stage2_stochastic_v2.yaml --checkpoint outputs/stage2_v2/best.pt --split test
```

## One-Click PowerShell Pipeline

Run full pipeline from manual ERA5 NetCDF files:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1
```

Quick smoke run (reduced epochs):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1 -Quick -QuickEpochs 2
```

## Sanity Check (No Long Training)

Validate data -> model -> loss logic with only 1-2 mini-batches:

```powershell
python scripts/sanity_check.py --config configs/stage2_redesign_sanity.yaml --steps 2 --backward
```

Then start redesigned Stage2 training:

```powershell
python -m ltg_net.cli train --config configs/stage2_redesign.yaml
python -m ltg_net.cli evaluate --config configs/stage2_redesign.yaml --checkpoint outputs/stage2_redesign/best.pt --split test
```

## Stage-3 (Recommended Flow)

Sanity first:

```powershell
python scripts/sanity_check.py --config configs/stage3_redesign_sanity.yaml --steps 2 --backward
```

Then full Stage-3:

```powershell
python -m ltg_net.cli train --config configs/stage3_redesign.yaml
python -m ltg_net.cli evaluate --config configs/stage3_redesign.yaml --checkpoint outputs/stage3_redesign/best.pt --split test
```

Checkpoint selection note:

- When `loss.combine_mode: normalized_weighted`, prefer `evaluation.checkpoint_metric: loss_total_raw` so best-model
  selection is based on raw weighted losses instead of normalized values.

## Strict PDE + ERA5 Auto Pipeline (One Click)

This pipeline enables strict PDE constraints in loss and runs full stage-3 recover/finetune automatically:

- Strict PDE configs:
  - `configs/stage3_mini_align_strict_pde_recover.yaml`
  - `configs/stage3_mini_align_strict_pde_finetune.yaml`
- One-click script:
  - `scripts/run_stage3_strict_pde_auto.ps1`

Example (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stage3_strict_pde_auto.ps1 `
  -Start 1990-01 `
  -End 1991-12 `
  -DataOutput data/era5_strict_pde.nc `
  -OutputRoot outputs/stage3_strict_pde_auto `
  -OverwriteData
```

If you prefer passing CDS API directly (without `.cdsapirc`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stage3_strict_pde_auto.ps1 `
  -CdsUrl "https://cds.climate.copernicus.eu/api" `
  -CdsKey "UID:API_KEY" `
  -OverwriteData
```

## Strict PDE (No New Download, Train Directly on Existing Data)

If your local dataset is already prepared and download is slow, use:

- `configs/stage3_strict_pde_recover.yaml`
- `configs/stage3_strict_pde_finetune.yaml`

Run:

```powershell
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8
python -m ltg_net.cli prepare --config configs/stage3_strict_pde_recover.yaml
python scripts/sanity_check.py --config configs/stage3_strict_pde_recover.yaml --steps 2 --backward
python -m ltg_net.cli train --config configs/stage3_strict_pde_recover.yaml
python -m ltg_net.cli evaluate --config configs/stage3_strict_pde_recover.yaml --checkpoint outputs/stage3_strict_pde_recover/best.pt --split test

python -m ltg_net.cli train --config configs/stage3_strict_pde_finetune.yaml
python -m ltg_net.cli evaluate --config configs/stage3_strict_pde_finetune.yaml --checkpoint outputs/stage3_strict_pde_finetune/best.pt --split test
```

### Trajectory-Head-Only Fine-Tuning (Freeze Field Branch)

Use config:

- `configs/stage3_mini_align_strict_pde_traj_tune.yaml`

This run freezes non-trajectory parameters and optimizes trajectory loss only.

```powershell
python -m ltg_net.cli train --config configs/stage3_mini_align_strict_pde_traj_tune.yaml
python -m ltg_net.cli evaluate --config configs/stage3_mini_align_strict_pde_traj_tune.yaml --checkpoint outputs/stage3_mini_align_strict_pde_traj_tune/best.pt --split test
```

## Strict PDE with Existing Pressure + GEE Single-Level

If you already have pressure-level ERA5 and only need `msl/tp`, use the GEE-only pipeline.
This path is isolated from the CDS downloader and writes to separate `gee` cache/output files.

Scripts:

- `scripts/download_era5_single_gee_chunked.py`
- `scripts/run_stage3_strict_pde_existing_pressure_gee.ps1`

Example (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stage3_strict_pde_existing_pressure_gee.ps1 `
  -PressureFile data/era5_pressure_1940_2021_hourly.nc `
  -EeProject "your-ee-project-id" `
  -Start 1990-01 `
  -End 2021-12 `
  -DataOutput data/era5_strict_pde_gee_merged.nc `
  -OutputRoot outputs/stage3_strict_pde_existing_pressure_gee `
  -OverwriteMergedData
```

## Model Overhaul (Current Version)

The model now includes several stability-focused upgrades:

- Trajectory predictors support `model.trajectory.max_step_deg` to limit per-step jumps.
- UNet generator uses residual forecasting with trend anchor (`residual_forecast`, `trend_scale`, `residual_scale`).
- Latent diffusion supports residual mode (predicting future increments from `x_hist[-1]` baseline).
- Field loss supports tail emphasis with:
  - `loss.field_extreme_weight`
  - `loss.field_extreme_quantile`
  - `loss.field_extreme_max_elements`
- Transformer nested-tensor warning is reduced by disabling nested tensor in encoder construction.

## Early Stopping + Per-Epoch Test Metrics

Trainer now supports:

- Automatic early stopping.
- Per-epoch test metric evaluation.
- JSONL metric logs for plotting and diagnosis.

Config keys (under `evaluation`):

- `checkpoint_metric`, `checkpoint_mode`, `checkpoint_min_delta`
- `test_every`, `test_max_batches`, `test_metrics_file`
- `save_history`, `history_file`
- `early_stopping.enabled`, `early_stopping.monitor`, `early_stopping.mode`
- `early_stopping.patience`, `early_stopping.min_delta`, `early_stopping.warmup_epochs`

Runtime performance tuning (under `experiment`):

- `performance.allow_tf32`
- `performance.cudnn_benchmark`
- `performance.matmul_precision`
- `compile.enabled` (optional `torch.compile`)

Generated files in each output directory:

- `metrics_history.jsonl`: train/val(/test) records by epoch.
- `test_metrics.jsonl`: per-evaluation test metrics records.

## Optimized Stage-3 Full Run (Recover + Finetune)

Linux/AutoDL:

```bash
bash scripts/run_stage3_optimized.sh
```

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stage3_optimized.ps1
```

The scripts run:

1. Optional `prepare`
2. Optional sanity check
3. `stage3_recover` training + test evaluation
4. `stage3_finetune` training + test evaluation
5. Export summary CSV: `outputs/stage3_metrics_summary.csv`

## Full Research Pipeline (Stage1 + Stage2 + Stage3)

Linux:

```bash
bash scripts/run_research_full.sh
```

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_research_full.ps1
```


## MiniLTGNet (Local Lightweight Validation)

Lightweight independent framework is under `miniltgnet/` (does not alter `ltg_net/`).

Quick run:

```bash
python -m miniltgnet.cli prepare --config miniltgnet/configs/lite.yaml
python -m miniltgnet.cli sanity --config miniltgnet/configs/lite_sanity.yaml --steps 2 --backward
python -m miniltgnet.cli train --config miniltgnet/configs/lite.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite.yaml --checkpoint outputs/miniltg/lite/best.pt --split test
```

Seed + ablation sweep:

```bash
python miniltgnet/scripts/run_seed_ablation.py
```

Strict benchmark (CI + paired significance vs baseline):

```bash
python miniltgnet/scripts/run_strict_benchmark.py \
  --config miniltgnet/configs/lite_plus_spec_finetune.yaml \
  --checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_overall.pt \
  --split test \
  --methods model persistence linear climatology \
  --leads 1 2 3 4 5 6 \
  --bootstrap 2000 \
  --reference persistence \
  --strict_p_threshold 0.95 \
  --output_dir outputs/miniltg/strict_benchmark
```

Dual weak-spot upgrade (target `track_mae` + `spectral_distance`):

```bash
python -m miniltgnet.cli train --config miniltgnet/configs/lite_plus_dual_opt.yaml
python -m miniltgnet.cli evaluate --config miniltgnet/configs/lite_plus_dual_opt.yaml --checkpoint outputs/miniltg/lite_plus_dual_opt/best.pt --split test
```

Mini final recommended config (fixed):

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

Metric-wise parameter stitching:

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

Automatic multi-try search for dual-metric checkpoint:

```bash
python miniltgnet/scripts/run_dual_metric_search.py \
  --config miniltgnet/configs/lite_plus_spec_finetune.yaml \
  --base_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_overall.pt \
  --track_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_track_mae.pt \
  --spectral_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_spectral.pt \
  --acc_checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_acc.pt \
  --split test \
  --use_ema_if_available \
  --output_dir outputs/miniltg/dual_metric_search
```
