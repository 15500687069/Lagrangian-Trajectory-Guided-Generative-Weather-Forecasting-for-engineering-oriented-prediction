# LTG-Net Model Overview

Lagrangian Trajectory Guided Generative Weather Forecasting for engineering-oriented prediction.

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

