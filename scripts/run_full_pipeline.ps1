[CmdletBinding()]
param(
    [string]$PressureFile = "D:\download\554c0cbb811c1f085347e91efbde7309.nc",
    [string]$InstantFile = "D:\download\92752f5d6fbc2adb99c13e8aaae0e9b1\data_stream-oper_stepType-instant.nc",
    [string]$AccumFile = "D:\download\92752f5d6fbc2adb99c13e8aaae0e9b1\data_stream-oper_stepType-accum.nc",
    [string]$BuiltDataOutput = "data/era5_199001.nc",
    [switch]$InstallDeps,
    [switch]$SkipBuildData,
    [switch]$SkipPrepare,
    [switch]$SkipStage1,
    [switch]$SkipStage2,
    [switch]$SkipStage3,
    [switch]$Quick,
    [int]$QuickEpochs = 3
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

function Assert-File([string]$PathText, [string]$Label) {
    if (-not (Test-Path -Path $PathText)) {
        throw "$Label not found: $PathText"
    }
}

function New-QuickConfig([string]$SourceConfig, [string]$StageName, [string]$OutDir, [int]$Epochs) {
    $tmpDir = Join-Path $OutDir "_quick_configs"
    New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
    $target = Join-Path $tmpDir "$StageName.yaml"

    $py = @'
import sys
import yaml
from ltg_net.config import load_config

source_config = sys.argv[1]
target_config = sys.argv[2]
stage_name = sys.argv[3]
epochs = int(sys.argv[4])

cfg = load_config(source_config)
cfg["experiment"]["name"] = f'{cfg["experiment"]["name"]}_quick'
cfg["experiment"]["output_dir"] = f'outputs/quick/{stage_name}'
cfg["optimization"]["epochs"] = epochs
cfg["data"]["dataloader"]["num_workers"] = min(2, int(cfg["data"]["dataloader"].get("num_workers", 2)))

with open(target_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=False)

print(target_config)
'@

    $quickPath = $py | python - $SourceConfig $target $StageName $Epochs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed generating quick config from $SourceConfig"
    }
    return ($quickPath | Select-Object -Last 1).Trim()
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
    Invoke-Checked "python" @("-m", "pip", "install", "zarr")
    Invoke-Checked "python" @("-m", "pip", "install", "-e", ".", "--no-build-isolation")
}

if (-not $SkipBuildData) {
    Write-Step "Build ERA5 Dataset From Manual Files"
    Assert-File $PressureFile "Pressure file"
    Assert-File $InstantFile "Instant file"
    Assert-File $AccumFile "Accum file"

    Invoke-Checked "python" @(
        "scripts/build_era5_from_manual.py",
        "--pressure", $PressureFile,
        "--instant", $InstantFile,
        "--accum", $AccumFile,
        "--output", $BuiltDataOutput,
        "--overwrite"
    )
}

$stage1Config = "configs/stage1_regional.yaml"
$stage2Config = "configs/stage2_stochastic.yaml"
$stage3Config = "configs/stage3_global.yaml"

if ($Quick) {
    Write-Step "Generate Quick Configs"
    $stage1Config = New-QuickConfig -SourceConfig $stage1Config -StageName "stage1" -OutDir "outputs" -Epochs $QuickEpochs
    $stage2Config = New-QuickConfig -SourceConfig $stage2Config -StageName "stage2" -OutDir "outputs" -Epochs $QuickEpochs
    $stage3Config = New-QuickConfig -SourceConfig $stage3Config -StageName "stage3" -OutDir "outputs" -Epochs $QuickEpochs
    Write-Host "Quick configs:"
    Write-Host "  $stage1Config"
    Write-Host "  $stage2Config"
    Write-Host "  $stage3Config"
}

if (-not $SkipPrepare) {
    Write-Step "Prepare Normalization Stats and Skeleton Tracks"
    Invoke-Checked "python" @("-m", "ltg_net.cli", "prepare", "--config", $stage1Config)
}

function Run-Stage([string]$StageName, [string]$ConfigPath, [string]$OutputDir) {
    Write-Step "$StageName Train"
    Invoke-Checked "python" @("-m", "ltg_net.cli", "train", "--config", $ConfigPath)

    $bestCkpt = Join-Path $OutputDir "best.pt"
    if (-not (Test-Path $bestCkpt)) {
        throw "$StageName checkpoint not found: $bestCkpt"
    }

    Write-Step "$StageName Evaluate"
    Invoke-Checked "python" @(
        "-m", "ltg_net.cli", "evaluate",
        "--config", $ConfigPath,
        "--checkpoint", $bestCkpt,
        "--split", "test"
    )
}

if (-not $SkipStage1) {
    $out = if ($Quick) { "outputs/quick/stage1" } else { "outputs/stage1" }
    Run-Stage -StageName "Stage1" -ConfigPath $stage1Config -OutputDir $out
}

if (-not $SkipStage2) {
    $out = if ($Quick) { "outputs/quick/stage2" } else { "outputs/stage2" }
    Run-Stage -StageName "Stage2" -ConfigPath $stage2Config -OutputDir $out
}

if (-not $SkipStage3) {
    $out = if ($Quick) { "outputs/quick/stage3" } else { "outputs/stage3" }
    Run-Stage -StageName "Stage3" -ConfigPath $stage3Config -OutputDir $out
}

Write-Step "Completed"
Write-Host "Pipeline finished."
