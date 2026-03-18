[CmdletBinding()]
param(
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $repoRoot

$argsList = @("miniltgnet/scripts/run_seed_ablation.py")
if ($SkipPrepare) {
    $argsList += "--skip_prepare"
}

Write-Host ">>> python $($argsList -join ' ')" -ForegroundColor DarkGray
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Seed ablation run failed."
}
