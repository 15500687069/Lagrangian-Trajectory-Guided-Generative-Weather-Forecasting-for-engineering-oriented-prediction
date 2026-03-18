@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Existing-data optimization pipeline:
REM 1) Train field-priority checkpoint with gradual PDE constraints.
REM 2) Evaluate field checkpoint.
REM 3) Run field/track strict benchmarks (track from existing track-tune run).

cd /d "%~dp0\.."
if errorlevel 1 goto :error

if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0"
if not defined FIELD_CFG set "FIELD_CFG=configs/stage3_mini_align_strict_pde_field_tune.yaml"
if not defined FIELD_OUT_DIR set "FIELD_OUT_DIR=outputs/stage3_mini_align_strict_pde_field_tune"
if not defined TRAINED_FIELD_AUTO_NAME set "TRAINED_FIELD_AUTO_NAME=best_field.pt"
if not defined TRACK_RUN_DIR set "TRACK_RUN_DIR=outputs/stage3_mini_align_strict_pde_traj_tune"
if not defined TRACK_CKPT set "TRACK_CKPT=%TRACK_RUN_DIR%\best_track_mae.pt"

if not defined GATE_PROFILE set "GATE_PROFILE=by_mode"
if not defined CALIB_SPLIT set "CALIB_SPLIT=val"

echo [INFO] Working dir: %CD%
echo [INFO] CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo [INFO] FIELD_CFG=%FIELD_CFG%
if defined FIELD_CKPT (
  echo [INFO] FIELD_CKPT pre-set in env: %FIELD_CKPT%
)
echo [INFO] TRACK_CKPT=%TRACK_CKPT%
python -c "import torch;print('cuda_available=',torch.cuda.is_available());print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if errorlevel 1 goto :error

if not exist "%FIELD_CFG%" (
  echo [ERROR] Field config not found: %FIELD_CFG%
  goto :error
)

echo.
echo [STEP] Train field-priority stage
python -m ltg_net.cli train --config "%FIELD_CFG%"
if errorlevel 1 goto :error

REM Always use checkpoint produced by this field-tune run to avoid stale env override.
set "FIELD_CKPT=%FIELD_OUT_DIR%\%TRAINED_FIELD_AUTO_NAME%"
if not exist "%FIELD_CKPT%" (
  echo [WARN] best_field.pt not found, fallback to best.pt
  set "FIELD_CKPT=%FIELD_OUT_DIR%\best.pt"
)
if not exist "%FIELD_CKPT%" (
  echo [ERROR] Field checkpoint not found: %FIELD_CKPT%
  goto :error
)

echo.
echo [STEP] Evaluate field-priority checkpoint
echo [INFO] FIELD_CKPT=%FIELD_CKPT%
python -m ltg_net.cli evaluate --config "%FIELD_CFG%" --checkpoint "%FIELD_CKPT%" --split test
if errorlevel 1 goto :error

if not exist "%TRACK_CKPT%" (
  echo [STEP] Auto-select track checkpoint from %TRACK_RUN_DIR%
  if not exist "%TRACK_RUN_DIR%" (
    echo [ERROR] Track run dir not found: %TRACK_RUN_DIR%
    goto :error
  )
  python scripts/select_best_checkpoint.py --run-dir "%TRACK_RUN_DIR%" --jsonl test_metrics.jsonl --section test --metric track_mae --mode min --output-name best_track_mae.pt --summary-name best_track_mae_summary.json
  if errorlevel 1 goto :error
)
if not exist "%TRACK_CKPT%" (
  echo [ERROR] Track checkpoint not found: %TRACK_CKPT%
  goto :error
)

echo.
echo [STEP] Strict benchmarks: field/track
set "CFG=%FIELD_CFG%"
set "FIELD_RUN_DIR=%FIELD_OUT_DIR%"
set "FIELD_CKPT=%FIELD_CKPT%"
set "TRACK_RUN_DIR=%TRACK_RUN_DIR%"
set "TRACK_CKPT=%TRACK_CKPT%"
set "GATE_PROFILE=%GATE_PROFILE%"
set "CALIB_SPLIT=%CALIB_SPLIT%"
call scripts\run_field_track_strict.bat
if errorlevel 1 goto :error

echo.
echo [DONE] Existing-data field+track optimization pipeline completed.
exit /b 0

:error
echo.
echo [ERROR] Pipeline failed.
exit /b 1
