[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PressureFile,
    [Parameter(Mandatory = $true)][string]$EeProject,
    [string]$Start = "1940-01",
    [string]$End = "2021-12",
    [string]$DataOutput = "data/era5_strict_pde_gee_merged.nc",
    [string]$InstantOutput = "data/era5_gee_single_instant_merged.nc",
    [string]$AccumOutput = "data/era5_gee_single_accum_merged.nc",
    [string]$CacheDir = "data/era5_gee_single_cache",
    [int]$TimeStepHours = 1,
    [double]$PdeTimeStepHours = 1.0,
    [int]$ChunkMonths = 2,
    [double]$ChunkSleepSeconds = 10.0,
    [double]$ScaleMeters = 27830.0,
    [double[]]$Area = @(50.0, 100.0, 10.0, 170.0), # [N, W, S, E]
    [string]$OutputRoot = "outputs/stage3_strict_pde_existing_pressure_gee",
    [switch]$EeAuthenticate,
    [switch]$EeHighVolume,
    [switch]$InstallDeps,
    [switch]$SkipSingleDownload,
    [switch]$OverwriteMergedData
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked([string]$Exe, [string[]]$CommandArgs) {
    Write-Host "> $Exe $($CommandArgs -join ' ')" -ForegroundColor DarkGray
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Exe $($CommandArgs -join ' ')"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Running strict-PDE pipeline with existing pressure file + GEE single-level..." -ForegroundColor Cyan
Write-Host "Pressure: $PressureFile"
Write-Host "Single-level range: $Start -> $End"

if ($InstallDeps) {
    Invoke-Checked "python" @("-m", "pip", "install", "-U", "pip", "setuptools", "wheel")
    Invoke-Checked "python" @("-m", "pip", "install", "earthengine-api", "xee", "xarray", "dask", "netcdf4", "pandas", "tqdm")
}

if (-not (Test-Path -Path $PressureFile)) {
    throw "Pressure file not found: $PressureFile"
}

if (-not $SkipSingleDownload) {
    $dlArgs = @(
        "scripts/download_era5_single_gee_chunked.py",
        "--start", $Start,
        "--end", $End,
        "--cache-dir", $CacheDir,
        "--time-step-hours", "$TimeStepHours",
        "--chunk-months", "$ChunkMonths",
        "--chunk-sleep-seconds", "$ChunkSleepSeconds",
        "--area", "$($Area[0])", "$($Area[1])", "$($Area[2])", "$($Area[3])",
        "--scale-meters", "$ScaleMeters",
        "--ee-project", $EeProject,
        "--output-instant", $InstantOutput,
        "--output-accum", $AccumOutput,
        "--pressure-file", $PressureFile,
        "--output-final", $DataOutput
    )
    if ($EeAuthenticate) {
        $dlArgs += "--ee-authenticate"
    }
    if ($EeHighVolume) {
        $dlArgs += "--ee-high-volume"
    }
    if ($OverwriteMergedData) {
        $dlArgs += "--overwrite-merged"
        $dlArgs += "--overwrite-final"
    }
    Invoke-Checked "python" $dlArgs
}

if (-not (Test-Path -Path $DataOutput)) {
    throw "Merged LTG dataset not found: $DataOutput"
}

$autoScript = Join-Path $PSScriptRoot "run_stage3_strict_pde_auto.ps1"
$autoParams = @{
    SkipDownload = $true
    DataOutput = $DataOutput
    OutputRoot = $OutputRoot
    PdeTimeStepHours = $PdeTimeStepHours
}
if ($InstallDeps) { $autoParams["InstallDeps"] = $true }

& $autoScript @autoParams
if ($LASTEXITCODE -ne 0) {
    throw "Training pipeline failed."
}

Write-Host "Completed." -ForegroundColor Green
