@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM One-click dual-checkpoint evaluation suite for LTG-Net.
REM Usage:
REM   1) conda activate ltgnet
REM   2) cd /d D:\ltg_net
REM   3) scripts\run_dualmode_full_suite.bat

cd /d "%~dp0\.."
if errorlevel 1 goto :error

if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
if not defined CFG set "CFG=configs/stage3_mini_align_strict_pde_field_tune.yaml"
if not defined FIELD_CKPT set "FIELD_CKPT=outputs/stage3_mini_align_strict_pde_field_tune/best_field.pt"
if not defined TRACK_CKPT set "TRACK_CKPT=outputs/stage3_mini_align_strict_pde_traj_tune/best_track_mae.pt"
if not defined LEADS set "LEADS=1 2 3 4 5 6"
if not defined BOOT set "BOOT=2000"
if not defined ALPHA set "ALPHA=0.05"
if not defined REF set "REF=persistence"
if not defined PTH set "PTH=0.95"
if not defined GATE_PROFILE set "GATE_PROFILE=by_mode"
if not defined CALIB_SPLIT set "CALIB_SPLIT=val"
if not defined CALIB_MAX_BATCHES set "CALIB_MAX_BATCHES=0"
if not defined CALIB_MAX_CANDIDATES set "CALIB_MAX_CANDIDATES=120"
if not defined CALIB_ALPHA_FIELD_GRID set "CALIB_ALPHA_FIELD_GRID=1.0,0.95,0.9"
if not defined CALIB_ALPHA_TRAJ_GRID set "CALIB_ALPHA_TRAJ_GRID=1.0,0.9,0.8"
if not defined CALIB_BETA_HIGH_GRID set "CALIB_BETA_HIGH_GRID=0.0,0.12"
if not defined CALIB_K_RATIO_GRID set "CALIB_K_RATIO_GRID=0.55,0.65"
if not defined CALIB_PREFIX_LEADS_GRID set "CALIB_PREFIX_LEADS_GRID=0,2"
if not defined CALIB_ALPHA_FIELD_PREFIX_GRID set "CALIB_ALPHA_FIELD_PREFIX_GRID=1.0,0.6"
if not defined CALIB_ALPHA_TRAJ_PREFIX_GRID set "CALIB_ALPHA_TRAJ_PREFIX_GRID=1.0,0.7"

echo [INFO] Working dir: %CD%
echo [INFO] CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo [INFO] CFG=%CFG%
echo [INFO] FIELD_CKPT=%FIELD_CKPT%
echo [INFO] TRACK_CKPT=%TRACK_CKPT%
echo [INFO] GATE_PROFILE=%GATE_PROFILE%
echo [INFO] CALIB_SPLIT=%CALIB_SPLIT%
python -c "import torch;print('cuda_available=',torch.cuda.is_available());print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if errorlevel 1 goto :error
if not exist "%CFG%" (
  echo [ERROR] Config not found: %CFG%
  goto :error
)
if not exist "%FIELD_CKPT%" (
  echo [ERROR] Field checkpoint not found: %FIELD_CKPT%
  goto :error
)
if not exist "%TRACK_CKPT%" (
  echo [ERROR] Track checkpoint not found: %TRACK_CKPT%
  goto :error
)

echo.
echo [STEP] Strict benchmark - FIELD mode
python scripts/run_ltg_strict_benchmark.py --config %CFG% --inference_mode field --field_checkpoint %FIELD_CKPT% --split test --methods model persistence linear climatology --leads %LEADS% --bootstrap %BOOT% --ci_alpha %ALPHA% --reference %REF% --strict_p_threshold %PTH% --gate_profile %GATE_PROFILE% --calibrate_on_split %CALIB_SPLIT% --calib_max_batches %CALIB_MAX_BATCHES% --calib_max_candidates %CALIB_MAX_CANDIDATES% --calib_alpha_field_grid %CALIB_ALPHA_FIELD_GRID% --calib_alpha_traj_grid %CALIB_ALPHA_TRAJ_GRID% --calib_beta_high_grid %CALIB_BETA_HIGH_GRID% --calib_k_ratio_grid %CALIB_K_RATIO_GRID% --calib_prefix_leads_grid %CALIB_PREFIX_LEADS_GRID% --calib_alpha_field_prefix_grid %CALIB_ALPHA_FIELD_PREFIX_GRID% --calib_alpha_traj_prefix_grid %CALIB_ALPHA_TRAJ_PREFIX_GRID% --device cuda --output_dir outputs/ltg/strict_field
if errorlevel 1 goto :error

echo.
echo [STEP] Strict benchmark - TRACK mode
python scripts/run_ltg_strict_benchmark.py --config %CFG% --inference_mode track --track_checkpoint %TRACK_CKPT% --split test --methods model persistence linear climatology --leads %LEADS% --bootstrap %BOOT% --ci_alpha %ALPHA% --reference %REF% --strict_p_threshold %PTH% --gate_profile %GATE_PROFILE% --calibrate_on_split %CALIB_SPLIT% --calib_max_batches %CALIB_MAX_BATCHES% --calib_max_candidates %CALIB_MAX_CANDIDATES% --calib_alpha_field_grid %CALIB_ALPHA_FIELD_GRID% --calib_alpha_traj_grid %CALIB_ALPHA_TRAJ_GRID% --calib_beta_high_grid %CALIB_BETA_HIGH_GRID% --calib_k_ratio_grid %CALIB_K_RATIO_GRID% --calib_prefix_leads_grid %CALIB_PREFIX_LEADS_GRID% --calib_alpha_field_prefix_grid %CALIB_ALPHA_FIELD_PREFIX_GRID% --calib_alpha_traj_prefix_grid %CALIB_ALPHA_TRAJ_PREFIX_GRID% --device cuda --output_dir outputs/ltg/strict_track
if errorlevel 1 goto :error

echo.
echo [STEP] Multi-seed reproducibility (3 seeds, dual modes)
python scripts/run_ltg_multiseed.py --base_config %CFG% --seeds 3407 2025 1111 --inference_modes field track --field_checkpoint_name best_field.pt --track_checkpoint_name best_track_mae.pt --output_root outputs/ltg/multiseed_dualmode --split test --methods model persistence linear climatology --leads %LEADS% --bootstrap %BOOT% --ci_alpha %ALPHA% --reference %REF% --strict_p_threshold %PTH% --gate_profile %GATE_PROFILE% --calibrate_on_split %CALIB_SPLIT% --calib_max_batches %CALIB_MAX_BATCHES% --calib_max_candidates %CALIB_MAX_CANDIDATES% --calib_alpha_field_grid %CALIB_ALPHA_FIELD_GRID% --calib_alpha_traj_grid %CALIB_ALPHA_TRAJ_GRID% --calib_beta_high_grid %CALIB_BETA_HIGH_GRID% --calib_k_ratio_grid %CALIB_K_RATIO_GRID% --calib_prefix_leads_grid %CALIB_PREFIX_LEADS_GRID% --calib_alpha_field_prefix_grid %CALIB_ALPHA_FIELD_PREFIX_GRID% --calib_alpha_traj_prefix_grid %CALIB_ALPHA_TRAJ_PREFIX_GRID%
if errorlevel 1 goto :error

echo.
echo [STEP] Cross-period generalization (dual modes)
python scripts/run_ltg_cross_period_benchmark.py --base_config %CFG% --field_checkpoint %FIELD_CKPT% --track_checkpoint %TRACK_CKPT% --inference_modes field track --period train_win,1990-01-08,1990-01-15 --period val_win,1990-01-16,1990-01-23 --period test_win,1990-01-24,1990-01-31 --output_root outputs/ltg/cross_period_dualmode --methods model persistence linear climatology --leads %LEADS% --bootstrap %BOOT% --ci_alpha %ALPHA% --reference %REF% --strict_p_threshold %PTH% --gate_profile %GATE_PROFILE% --calibrate_on_split %CALIB_SPLIT% --calib_max_batches %CALIB_MAX_BATCHES% --calib_max_candidates %CALIB_MAX_CANDIDATES% --calib_alpha_field_grid %CALIB_ALPHA_FIELD_GRID% --calib_alpha_traj_grid %CALIB_ALPHA_TRAJ_GRID% --calib_beta_high_grid %CALIB_BETA_HIGH_GRID% --calib_k_ratio_grid %CALIB_K_RATIO_GRID% --calib_prefix_leads_grid %CALIB_PREFIX_LEADS_GRID% --calib_alpha_field_prefix_grid %CALIB_ALPHA_FIELD_PREFIX_GRID% --calib_alpha_traj_prefix_grid %CALIB_ALPHA_TRAJ_PREFIX_GRID%
if errorlevel 1 goto :error

echo.
echo [DONE] Full dual-mode suite completed.
echo [OUT] outputs/ltg/strict_field
echo [OUT] outputs/ltg/strict_track
echo [OUT] outputs/ltg/multiseed_dualmode
echo [OUT] outputs/ltg/cross_period_dualmode
exit /b 0

:error
echo.
echo [ERROR] Command failed. Stop at current step.
exit /b 1
