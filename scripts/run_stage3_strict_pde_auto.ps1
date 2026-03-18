[CmdletBinding()]
param(
    [string]$Start = "1990-01",
    [string]$End = "1990-12",
    [string]$DataOutput = "data/era5_strict_pde.nc",
    [string]$CacheDir = "data/era5_download_cache",
    [int]$TimeStepHours = 6,
    [double]$PdeTimeStepHours = 6.0,
    [int]$DaysPerRequest = 7,
    [double[]]$Area = @(50.0, 100.0, 10.0, 170.0), # [N, W, S, E]
    [string]$CdsUrl = "",
    [string]$CdsKey = "",
    [string]$OutputRoot = "outputs/stage3_strict_pde_auto",
    [switch]$InstallDeps,
    [switch]$SkipDownload,
    [switch]$OverwriteData
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==== $Message ====" -ForegroundColor Cyan
}

function Invoke-Checked([string]$Exe, [string[]]$CommandArgs) {
    Write-Host "> $Exe $($CommandArgs -join ' ')" -ForegroundColor DarkGray
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Exe $($CommandArgs -join ' ')"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Step "Repository Root"
Write-Host "Working directory: $repoRoot"

Write-Step "Python Check"
Invoke-Checked "python" @("--version")

if ($InstallDeps) {
    Write-Step "Install Dependencies"
    Invoke-Checked "python" @("-m", "pip", "install", "-U", "pip", "setuptools", "wheel")
    Invoke-Checked "python" @("-m", "pip", "install", "-r", "requirements.txt")
    Invoke-Checked "python" @("-m", "pip", "install", "cdsapi", "xarray", "netcdf4", "tqdm")
}

if (-not $SkipDownload) {
    Write-Step "Download ERA5 + Build Training Dataset"
    $downloadArgs = @(
        "scripts/download_era5.py",
        "--start", $Start,
        "--end", $End,
        "--output", $DataOutput,
        "--cache-dir", $CacheDir,
        "--time-step-hours", "$TimeStepHours",
        "--days-per-request", "$DaysPerRequest",
        "--area", "$($Area[0])", "$($Area[1])", "$($Area[2])", "$($Area[3])",
        "--cleanup-raw"
    )
    if ($OverwriteData) {
        $downloadArgs += "--overwrite"
    }
    if ($CdsUrl -ne "") {
        $downloadArgs += @("--cds-url", $CdsUrl)
    }
    if ($CdsKey -ne "") {
        $downloadArgs += @("--cds-key", $CdsKey)
    }
    Invoke-Checked "python" $downloadArgs
}

if (-not (Test-Path -Path $DataOutput)) {
    throw "ERA5 dataset not found: $DataOutput"
}

$runtimeDir = Join-Path $OutputRoot "_runtime_configs"
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$recoverCfg = Join-Path $runtimeDir "stage3_strict_pde_recover.runtime.yaml"
$finetuneCfg = Join-Path $runtimeDir "stage3_strict_pde_finetune.runtime.yaml"
$recoverOut = Join-Path $OutputRoot "recover"
$finetuneOut = Join-Path $OutputRoot "finetune"

Write-Step "Generate Runtime Configs"
$pyCfg = @'
import sys
import yaml
from pathlib import Path
from ltg_net.config import load_config

recover_base = sys.argv[1]
finetune_base = sys.argv[2]
recover_out_cfg = Path(sys.argv[3])
finetune_out_cfg = Path(sys.argv[4])
era5_path = sys.argv[5]
recover_out_dir = sys.argv[6]
finetune_out_dir = sys.argv[7]
pde_dt_hours = float(sys.argv[8])

rcfg = load_config(recover_base)
rcfg["data"]["era5_path"] = era5_path
rcfg["experiment"]["output_dir"] = recover_out_dir
rcfg["experiment"]["name"] = "ltg_stage3_strict_pde_auto_recover"

fcfg = load_config(finetune_base)
fcfg["data"]["era5_path"] = era5_path
fcfg["experiment"]["output_dir"] = finetune_out_dir
fcfg["experiment"]["name"] = "ltg_stage3_strict_pde_auto_finetune"
fcfg["experiment"]["resume_checkpoint"] = str(Path(recover_out_dir) / "best.pt")

for _cfg in (rcfg, fcfg):
    _cfg["loss"]["physics"]["strict_pde"]["time_step_hours"] = pde_dt_hours

recover_out_cfg.parent.mkdir(parents=True, exist_ok=True)
with recover_out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(rcfg, f, allow_unicode=False, sort_keys=False)
with finetune_out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(fcfg, f, allow_unicode=False, sort_keys=False)

print(recover_out_cfg)
print(finetune_out_cfg)
'@
Invoke-Checked "python" @(
    "-c", $pyCfg,
    "configs/stage3_mini_align_strict_pde_recover.yaml",
    "configs/stage3_mini_align_strict_pde_finetune.yaml",
    $recoverCfg,
    $finetuneCfg,
    $DataOutput,
    $recoverOut,
    $finetuneOut,
    "$PdeTimeStepHours"
)

Write-Step "Prepare Stats + Tracks"
Invoke-Checked "python" @("-m", "ltg_net.cli", "prepare", "--config", $recoverCfg)

Write-Step "Sanity Check"
Invoke-Checked "python" @("scripts/sanity_check.py", "--config", $recoverCfg, "--steps", "2", "--backward")

Write-Step "Train Recover"
Invoke-Checked "python" @("-m", "ltg_net.cli", "train", "--config", $recoverCfg)

$recoverBest = Join-Path $recoverOut "best.pt"
if (-not (Test-Path -Path $recoverBest)) {
    throw "Recover checkpoint not found: $recoverBest"
}

Write-Step "Evaluate Recover"
Invoke-Checked "python" @("-m", "ltg_net.cli", "evaluate", "--config", $recoverCfg, "--checkpoint", $recoverBest, "--split", "test")

Write-Step "Train Finetune"
Invoke-Checked "python" @("-m", "ltg_net.cli", "train", "--config", $finetuneCfg)

$finetuneBest = Join-Path $finetuneOut "best.pt"
if (-not (Test-Path -Path $finetuneBest)) {
    throw "Finetune checkpoint not found: $finetuneBest"
}

Write-Step "Evaluate Finetune"
Invoke-Checked "python" @("-m", "ltg_net.cli", "evaluate", "--config", $finetuneCfg, "--checkpoint", $finetuneBest, "--split", "test")

Write-Step "Completed"
Write-Host "Strict PDE pipeline finished."
Write-Host "Recover best:  $recoverBest"
Write-Host "Finetune best: $finetuneBest"
