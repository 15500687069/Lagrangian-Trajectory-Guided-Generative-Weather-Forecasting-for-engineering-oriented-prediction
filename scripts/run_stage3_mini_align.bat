@echo off
setlocal

set RECOVER_CFG=configs\stage3_mini_align_recover.yaml
set FINETUNE_CFG=configs\stage3_mini_align_finetune.yaml
set RECOVER_BEST=outputs\stage3_mini_align_recover\best.pt
set FINETUNE_BEST=outputs\stage3_mini_align_finetune\best.pt

echo [1/6] Prepare stats and tracks (mini-mapped protocol)...
python -m ltg_net.cli prepare --config %RECOVER_CFG%
if errorlevel 1 goto :error

echo [2/6] Sanity check (2 steps + backward)...
python scripts\sanity_check.py --config %RECOVER_CFG% --steps 2 --backward
if errorlevel 1 goto :error

echo [3/6] Train recover stage...
python -m ltg_net.cli train --config %RECOVER_CFG%
if errorlevel 1 goto :error

if not exist %RECOVER_BEST% (
  echo Recover best checkpoint not found: %RECOVER_BEST%
  goto :error
)

echo [4/6] Evaluate recover best...
python -m ltg_net.cli evaluate --config %RECOVER_CFG% --checkpoint %RECOVER_BEST% --split test
if errorlevel 1 goto :error

echo [5/6] Train finetune stage...
python -m ltg_net.cli train --config %FINETUNE_CFG%
if errorlevel 1 goto :error

if not exist %FINETUNE_BEST% (
  echo Finetune best checkpoint not found: %FINETUNE_BEST%
  goto :error
)

echo [6/6] Evaluate finetune best...
python -m ltg_net.cli evaluate --config %FINETUNE_CFG% --checkpoint %FINETUNE_BEST% --split test
if errorlevel 1 goto :error

echo Done.
exit /b 0

:error
echo Failed.
exit /b 1

