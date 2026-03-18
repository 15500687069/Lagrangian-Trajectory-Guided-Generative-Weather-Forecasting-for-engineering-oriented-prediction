# LTG-Net

LTG-Net is a research codebase for Lagrangian trajectory guided generative weather forecasting.
It implements:

- Background dynamical environment encoding.
- Explicit dynamical skeleton extraction and trajectory forecasting.
- Trajectory-conditioned field generation (UNet and latent diffusion variants).
- Lagrangian consistency, physical consistency, and spectral consistency constraints.
- Three-stage training pipeline from regional prototype to global expansion.

## Model Overview (Mainstream Large-Model Style)

### 1) Model Positioning

LTG-Net is a physics-aware generative weather forecasting model built around a Lagrangian dynamical skeleton.
Its goal is to jointly model:

- Large-scale background circulation dynamics.
- Coherent-object trajectory evolution (for example cyclone centers).
- Trajectory-conditioned multi-variable field generation.

The current engineering workflow uses a **dual-checkpoint deployment strategy**:

- `field mode`: use field-priority checkpoint for field/intensity-related prediction.
- `track mode`: use track-priority checkpoint for trajectory/path prediction.

Current production recommendation is **field + track closed-loop**, without fused checkpoint dependency.

### 2) Core Architecture

LTG-Net is organized into three major modules:

1. **Background Dynamical Encoder** (`SphereSpatiotemporalEncoder`)
2. **Dynamical Skeleton Trajectory Head** (`DeterministicTrajectoryPredictor` / `NeuralSDETrajectoryPredictor`)
3. **Trajectory-Conditioned Field Generator** (`TrajectoryConditionedUNetGenerator` / `TrajectoryConditionedLatentDiffusion`)

Forward path:

1. Encode historical multi-variable fields into background representations (`z_bg_map`, `z_bg_global`).
2. Predict future object trajectories from background state + historical tracks.
3. Generate future fields conditioned on both background and predicted trajectories.
4. Apply event modulation (optional) to strengthen key weather-event responses.

### 3) Training Objective System

LTG-Net adopts a multi-objective loss design, including:

- `loss_field`: field reconstruction (L1/L2 + optional extreme-tail emphasis).
- `loss_traj`: trajectory supervision.
- `loss_adv`: advection consistency.
- `loss_phys`: physical consistency (including optional strict PDE terms).
- `loss_spec`: spectral consistency.
- `loss_diff`: diffusion training loss (for latent diffusion generator).

This design balances forecast skill, structure coherence, and physical interpretability.

### 4) Physical Interpretability

Compared with purely data-driven black-box forecasting, LTG-Net keeps explicit physical hooks:

- Lagrangian trajectory constraints for coherent-object evolution.
- PDE-informed regularization (continuity, thermodynamic/moisture/geostrophic terms when enabled).
- Spectral constraints to preserve multi-scale structure.

This makes error diagnosis more traceable at trajectory, field, and physics-term levels.

### 5) Engineering Inference Protocol (Current)

To stabilize engineering performance under limited data scale:

- **Field outputs** are served by `field checkpoint`.
- **Trajectory outputs** are served by `track checkpoint`.
- Final product is a combined forecast package from the two modes (not parameter fusion).

This protocol is aligned with the observed tradeoff between field-optimal and track-optimal checkpoints.

### 6) Intended Use and Roadmap

Intended use:

- Regional-to-global weather forecasting prototype validation.
- Engineering forecasting scenarios where physical interpretability and controllability are required.

Roadmap:

- Expand data scale and temporal span.
- Strengthen cross-period generalization and multi-seed robustness.
- Improve strict benchmark performance against persistence-style baselines while preserving physics constraints.

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

## Tropical Depression Water-Pump Research Pipeline

This repository also includes an end-to-end research script for the South China Sea
non-monsoon tropical depression workflow:

- Download ERA5 pressure/single-level data.
- Download GPM DPR granules.
- Identify water pumps from 3D reflectivity.
- Compute OIDRA and local MSE variance.
- Build multi-snapshot sample library and summary tables.
- Reproduce core publication-style figures from the original notebook.

Config file:

```bash
configs/td_waterpump_research.yaml
```

Run:

```bash
python scripts/td_waterpump_research.py --config configs/td_waterpump_research.yaml
```

Dump default config template:

```bash
python scripts/td_waterpump_research.py --dump-default-config
```

Credential notes:

- ERA5: configure CDS API (`~/.cdsapirc`) or set env var `CDSAPI_KEY`.
- GPM: use Earthdata login (`.netrc`) or set env var `EARTHDATA_TOKEN`.

Key outputs:

- `sample_library.csv`: raw snapshot sample library.
- `sample_library_filtered.csv`: filtered library after QC/exclusion rules.
- `sample_exclusion_log.csv`: exclusion reasons by snapshot.
- `water_pump_catalog.csv`: per-pump spatial/structure records.
- `oidra_precip_summary.csv`: high/low OIDRA grouped means.
- `organization_precip_stats.json`: group test + correlation statistics.
- `mechanism_chain_summary.md`: auto-generated mechanism-chain wording.

How this supports the three-step research progression:

1. Sample library construction:
`sample_library.csv` stores snapshot-level pump count, pump positions, OIDRA, MSE variance,
precipitation intensity, and precipitation-type fractions.
2. Organization-precipitation comparison:
`sample_library_filtered.csv`, `oidra_precip_summary.csv`, and `organization_precip_stats.json`
support high/low OIDRA grouped comparison and statistical testing.
3. Mechanism-chain expression:
`mechanism_chain_summary.md` auto-generates an interpretable statement linking
environment uniformity, organization, and precipitation structure.

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

## LTG-Net Closed-Loop (Field + Track, No Fused)

Use `field ckpt` for atmospheric fields/intensity and `track ckpt` for path.
The default closed-loop scripts no longer run fused mode.

```bash
conda activate ltgnet
cd /d D:\ltg_net

# Strict benchmark (field + track only)
scripts\run_field_track_strict.bat

# Existing-data full loop: field tune -> evaluate -> strict benchmark (field + track only)
scripts\run_stage3_field_track_existing_data.bat
```
