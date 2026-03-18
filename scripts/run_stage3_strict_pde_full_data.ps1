[CmdletBinding()]
param(
    [string]$CdsUrl = "https://cds.climate.copernicus.eu/api",
    [Parameter(Mandatory = $true)][string]$CdsKey,
    [string]$Start = "1990-01",
    [string]$End = "2021-12",
    [string]$DataOutput = "data/era5_strict_pde_full_1990_2021.nc",
    [string]$CacheDir = "data/era5_download_cache_full",
    [string]$OutputRoot = "outputs/stage3_strict_pde_full_1990_2021",
    [int]$TimeStepHours = 6,
    [double]$PdeTimeStepHours = 6.0,
    [int]$DaysPerRequest = 7,
    [double[]]$Area = @(50.0, 100.0, 10.0, 170.0), # [N, W, S, E]
    [switch]$InstallDeps,
    [switch]$SkipDownload,
    [switch]$OverwriteData
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Running strict-PDE full-data pipeline..." -ForegroundColor Cyan
Write-Host "Data range: $Start -> $End"
Write-Host "Area: N=$($Area[0]) W=$($Area[1]) S=$($Area[2]) E=$($Area[3])"
Write-Host "Output dataset: $DataOutput"
Write-Host "Output root: $OutputRoot"

$invokeParams = @{
    Start = $Start
    End = $End
    DataOutput = $DataOutput
    CacheDir = $CacheDir
    TimeStepHours = $TimeStepHours
    PdeTimeStepHours = $PdeTimeStepHours
    DaysPerRequest = $DaysPerRequest
    Area = $Area
    CdsUrl = $CdsUrl
    CdsKey = $CdsKey
    OutputRoot = $OutputRoot
}
if ($InstallDeps) { $invokeParams["InstallDeps"] = $true }
if ($SkipDownload) { $invokeParams["SkipDownload"] = $true }
if ($OverwriteData) { $invokeParams["OverwriteData"] = $true }

$autoScript = Join-Path $PSScriptRoot "run_stage3_strict_pde_auto.ps1"
& $autoScript @invokeParams
if ($LASTEXITCODE -ne 0) {
    throw "Full-data strict PDE pipeline failed."
}

Write-Host "Full-data strict PDE pipeline completed." -ForegroundColor Green
