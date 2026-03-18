@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Field/Track strict benchmark with separated checkpoints (no fused mode).
REM
REM Usage (recommended):
REM   conda activate ltgnet
REM   cd /d D:\ltg_net
REM   set CUDA_VISIBLE_DEVICES=0
REM   set CFG=configs/stage3_mini_align_strict_pde_field_tune.yaml
REM   set FIELD_RUN_DIR=outputs/stage3_mini_align_strict_pde_field_tune
REM   set TRACK_RUN_DIR=outputs/stage3_mini_align_strict_pde_traj_tune
REM   scripts\run_field_track_strict.bat
REM
REM Optional manual override:
REM   set FIELD_CKPT=outputs\...\best_field.pt
REM   set TRACK_CKPT=outputs\...\best_track_mae.pt

cd /d "%~dp0\.."
if errorlevel 1 goto :error

if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
if not defined CFG set "CFG=configs/stage3_mini_align_strict_pde_traj_tune.yaml"
if not defined FIELD_RUN_DIR set "FIELD_RUN_DIR=outputs/stage3_mini_align_strict_pde_field_tune"
if not defined TRACK_RUN_DIR set "TRACK_RUN_DIR=outputs/stage3_mini_align_strict_pde_traj_tune"

if not defined FIELD_AUTO_NAME set "FIELD_AUTO_NAME=best_field.pt"
if not defined TRACK_AUTO_NAME set "TRACK_AUTO_NAME=best_track_mae.pt"

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
python -c "import torch;print('cuda_available=',torch.cuda.is_available());print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if errorlevel 1 goto :error

if not exist "%CFG%" (
  echo [ERROR] Config not found: %CFG%
  goto :error
)

if not defined FIELD_CKPT (
  echo [STEP] Auto-select field checkpoint from val/loss_field
  if not exist "%FIELD_RUN_DIR%" (
    echo [ERROR] FIELD_RUN_DIR not found: %FIELD_RUN_DIR%
    goto :error
  )
  python scripts/select_best_checkpoint.py --run-dir "%FIELD_RUN_DIR%" --jsonl metrics_history.jsonl --section val --metric loss_field --mode min --output-name "%FIELD_AUTO_NAME%" --summary-name best_field_from_val_loss_field_summary.json
  if errorlevel 1 goto :error
  set "FIELD_CKPT=%FIELD_RUN_DIR%\%FIELD_AUTO_NAME%"
)

if not defined TRACK_CKPT (
  echo [STEP] Auto-select track checkpoint from test/track_mae
  if not exist "%TRACK_RUN_DIR%" (
    echo [ERROR] TRACK_RUN_DIR not found: %TRACK_RUN_DIR%
    goto :error
  )
  python scripts/select_best_checkpoint.py --run-dir "%TRACK_RUN_DIR%" --jsonl test_metrics.jsonl --section test --metric track_mae --mode min --output-name "%TRACK_AUTO_NAME%" --summary-name best_track_mae_summary.json
  if errorlevel 1 goto :error
  set "TRACK_CKPT=%TRACK_RUN_DIR%\%TRACK_AUTO_NAME%"
)

echo [INFO] FIELD_CKPT=%FIELD_CKPT%
echo [INFO] TRACK_CKPT=%TRACK_CKPT%
echo [INFO] GATE_PROFILE=%GATE_PROFILE%
echo [INFO] CALIB_SPLIT=%CALIB_SPLIT%

if not exist "%FIELD_CKPT%" (
  echo [ERROR] Field checkpoint not found: %FIELD_CKPT%
  goto :error
)
if not exist "%TRACK_CKPT%" (
  echo [ERROR] Track checkpoint not found: %TRACK_CKPT%
  goto :error
)

echo.
echo [STEP] Strict benchmark - FIELD mode ^(pure field checkpoint^)
python scripts/run_ltg_strict_benchmark.py --config "%CFG%" --inference_mode field --field_checkpoint "%FIELD_CKPT%" --split test --methods model persistence linear climatology --leads %LEADS% --bootstrap %BOOT% --ci_alpha %ALPHA% --reference %REF% --strict_p_threshold %PTH% --gate_profile %GATE_PROFILE% --calibrate_on_split %CALIB_SPLIT% --calib_max_batches %CALIB_MAX_BATCHES% --calib_max_candidates %CALIB_MAX_CANDIDATES% --calib_alpha_field_grid %CALIB_ALPHA_FIELD_GRID% --calib_alpha_traj_grid %CALIB_ALPHA_TRAJ_GRID% --calib_beta_high_grid %CALIB_BETA_HIGH_GRID% --calib_k_ratio_grid %CALIB_K_RATIO_GRID% --calib_prefix_leads_grid %CALIB_PREFIX_LEADS_GRID% --calib_alpha_field_prefix_grid %CALIB_ALPHA_FIELD_PREFIX_GRID% --calib_alpha_traj_prefix_grid %CALIB_ALPHA_TRAJ_PREFIX_GRID% --device cuda --output_dir outputs/ltg/strict_field_split
if errorlevel 1 goto :error

echo.
echo [STEP] Strict benchmark - TRACK mode ^(pure track checkpoint^)
python scripts/run_ltg_strict_benchmark.py --config "%CFG%" --inference_mode track --track_checkpoint "%TRACK_CKPT%" --split test --methods model persistence linear climatology --leads %LEADS% --bootstrap %BOOT% --ci_alpha %ALPHA% --reference %REF% --strict_p_threshold %PTH% --gate_profile %GATE_PROFILE% --calibrate_on_split %CALIB_SPLIT% --calib_max_batches %CALIB_MAX_BATCHES% --calib_max_candidates %CALIB_MAX_CANDIDATES% --calib_alpha_field_grid %CALIB_ALPHA_FIELD_GRID% --calib_alpha_traj_grid %CALIB_ALPHA_TRAJ_GRID% --calib_beta_high_grid %CALIB_BETA_HIGH_GRID% --calib_k_ratio_grid %CALIB_K_RATIO_GRID% --calib_prefix_leads_grid %CALIB_PREFIX_LEADS_GRID% --calib_alpha_field_prefix_grid %CALIB_ALPHA_FIELD_PREFIX_GRID% --calib_alpha_traj_prefix_grid %CALIB_ALPHA_TRAJ_PREFIX_GRID% --device cuda --output_dir outputs/ltg/strict_track_split
if errorlevel 1 goto :error

echo.
echo [DONE] Field/Track strict benchmarks completed.
echo [OUT] outputs/ltg/strict_field_split
echo [OUT] outputs/ltg/strict_track_split
exit /b 0

:error
echo.
echo [ERROR] Pipeline failed.
exit /b 1
